// SPDX-License-Identifier: Apache-2.0
// The driver: validation, the workspace and its chunking by free device memory, the
// host-orchestrated mode loop with its host synchronizations S1-S5, G2's rung state machine, the
// ladder upload, the sifts with G1 and the guard, and the commit.
//
// The start of a chunk prescales every element by an exact power of two so that max|r| lies in
// [1, 2) (enqueue_mode_prescale); finish() scales the rows back (enqueue_mode_unscale)
// and the call scales the energy history. A guard (enqueue_mode_guard, after the thresholds
// and the orientation, and the commit) fails an element that meets a non-finite intermediate.
//
// One mode of a chunk of elements, in lockstep (an element that is done has on = 0 and zero
// windows, so every pass skips it). The host reads the per-element state back where the next
// launches depend on it (S1-S5):
//   step 0   mode_start; sigma_n
//   step 1-2 per pass: tau, flags, mode_rice                              S1: le[] (G2 eligibility)
//   step 3   G2 if eligible: tau floored by sigma_x, ff; the rung machine,
//            one g2_test per step, P in two buffers by step parity        S2: per step (the outcomes)
//            g2_accept; G3's coarse median
//   step 4   mode_window                                                  S3: le[] (who is active)
//   step 5   raw window, scale field, floor; G3's scaling
//   step 6   orientation (with Params::orientation, a copy of n and c into
//            the orientation rows), element build                         S4: le[] (d_min, d_max)
//   step 7   ladder upload (read-modify-write of le[]), weights, rung windows
//   step 8   sift 1 (envelope mean); the continuation weight
//   step 9   sifts 2..S: G1, the weight, the guard
//   step 10  commit (rows, r -= h, partials, decide)                      S5: any active
// Every launch is issued from the host (the launch structure depends on S1-S5).
#include "runtime/context.hpp"
#include "runtime/device.hpp"
#include "runtime/error.hpp"
#include "runtime/memory.hpp"
#include "decompose.hpp"
#include "field/convert.hpp"
#include "field/field_ops.hpp"
#include "field/vhgw.hpp"
#include "method/method.hpp"
#include "method/element.hpp"
#include "method/mode.hpp"
#include "method/stats.hpp"
#include "field/reduce.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace eda {

// ---------------------------------------------------------------------------------------------
// G2's rung state machine (host; mode.hpp), exposed for the C++ tests.

std::vector<int32_t> g2_candidate_rungs(int32_t w, int32_t w_prev, double jump) {
  // s_j = odd(s0 q^j) while s_j jump <= w, consecutive duplicates dropped (pow from the C library,
  // in double, as rung_ladder)
  const int32_t s0 = std::max(3, w_prev + 2);
  std::vector<int32_t> out;
  for (int32_t j = 0;; ++j) {
    const int32_t s = odd_window(static_cast<double>(s0) * std::pow(kRungRatio, static_cast<double>(j)));
    if (static_cast<double>(s) * jump > static_cast<double>(w)) break;
    if (out.empty() || s != out.back()) out.push_back(s);
    if (static_cast<int32_t>(out.size()) > kMaxG2Rungs) throw std::length_error("eda: too many G2 rungs");
  }
  return out;
}

int32_t RungMachine::next_rung() const {
  if (done) return 0;
  if (i < static_cast<int32_t>(rungs.size())) return rungs[static_cast<size_t>(i)];
  // only while pending: the confirmation rung of the last candidate, odd(s Q + 1)
  return odd_window(static_cast<double>(rungs.back()) * kRungRatio + 1.0);
}

void RungMachine::feed(const G2Result& o) {
  if (done) throw std::logic_error("eda: G2 machine fed after its decision");
  ++tests;
  if (pending) {
    if (o.outcome == kG2Accepted) {
      done = true;
      accepted = true;
      acc_index = i - 1;
      return;
    }
    if (o.outcome == kG2Stop) {
      done = true;
      return;
    }
    pending = false;
    ++i;
  } else {
    if (o.outcome == kG2Stop) {
      done = true;
      return;
    }
    if (o.outcome == kG2Accepted) {
      pending = true;
      cand = o;
      cand.s = rungs[static_cast<size_t>(i)];
    }
    ++i;
  }
  if (i >= static_cast<int32_t>(rungs.size()) && !pending) done = true;
}

RungMachine make_rung_machine(int32_t w, int32_t w_prev, double jump) {
  RungMachine m;
  m.rungs = g2_candidate_rungs(w, w_prev, jump);
  m.done = m.rungs.empty();
  return m;
}

namespace {

template <class T>
constexpr DType dtype_of() {
  return std::is_same_v<T, float> ? DType::F32 : DType::F64;
}
const char* dtype_name(DType d) { return d == DType::F32 ? "float32" : "float64"; }

// The largest mode count of a call: every window is at least max(3, w_prev + 2) and, from mode
// growth_from on, odd(growth w_prev); the loop stops when it exceeds n_max.
int32_t mode_bound(const MethodCtl& c) {
  int32_t w_prev = 1, count = 0;
  for (int32_t k = 0;; ++k) {
    int64_t w = std::max(3, w_prev + 2);
    if (c.growth > 1.0 && k >= c.growth_from) {
      w = std::max<int64_t>(w, odd_window(c.growth * static_cast<double>(w_prev)));
    }
    if (w > c.n_max) break;
    ++count;
    w_prev = static_cast<int32_t>(w);
  }
  return count;
}

// The working fields of a chunk.
template <class T>
struct ChunkFields {
  T* r;
  T* L;
  T* Lp;
  T* o[kMaxDim];
  T* t;
  T* X[12];
  uint8_t *flags, *ff, *P0, *P1, *Pacc, *A, *sel, *lo;
};

constexpr int kScratch = 12;
// int32 [B] arrays of a chunk: on, on_g2, on_floor, srung, slot, the window slots, the rung windows
// (kMaxRungs), the prescale exponents.
constexpr int kInts = 6 + kWinSlots + kMaxRungs;

template <class T>
struct MethodRun final : Run {
  DeviceBuffer<unsigned char> out;
  DeviceBuffer<unsigned char> orient;  // (B, rows_cap - 1, D + 1, count) with Params::orientation
  DeviceBuffer<int32_t> meta;
  Geom geom{};

  // n_imfs, then error, on the device.
  const int32_t* device_meta(Context& ctx) {
    if (meta.empty()) {
      cudaStream_t s = ctx.stream();
      meta = DeviceBuffer<int32_t>(static_cast<size_t>(2 * batch), s, ctx.device());
      EDA_CUDA(cudaMemcpyAsync(meta.data(), n_imfs.data(), sizeof(int32_t) * batch, cudaMemcpyHostToDevice, s));
      EDA_CUDA(cudaMemcpyAsync(meta.data() + batch, error.data(), sizeof(int32_t) * batch, cudaMemcpyHostToDevice, s));
    }
    return meta.data();
  }

  void collect_orientation(Context& ctx, void* dir, void* coh, bool dst_on_device, int32_t K) override {
    DeviceGuard guard(ctx.device());
    cudaStream_t s = ctx.stream();
    if (orient.empty()) throw std::logic_error("eda: the call kept no orientation");
    if (batch == 0 || K == 0) return;
    const int32_t* m = device_meta(ctx);
    const T* src = reinterpret_cast<const T*>(orient.data());
    const size_t nd = static_cast<size_t>(batch) * K * geom.dim * count, nc = static_cast<size_t>(batch) * K * count;
    if (dst_on_device) {
      enqueue_orientation_collect<T>(s, batch, geom.dim, count, m, m + batch, src, rows_cap - 1, static_cast<T*>(dir),
                                     static_cast<T*>(coh), K);
      return;
    }
    DeviceBuffer<T> tmp(nd + nc, s, ctx.device());
    enqueue_orientation_collect<T>(s, batch, geom.dim, count, m, m + batch, src, rows_cap - 1, tmp.data(),
                                   tmp.data() + nd, K);
    EDA_CUDA(cudaMemcpyAsync(dir, tmp.data(), nd * sizeof(T), cudaMemcpyDeviceToHost, s));
    EDA_CUDA(cudaMemcpyAsync(coh, tmp.data() + nd, nc * sizeof(T), cudaMemcpyDeviceToHost, s));
    ctx.synchronize();
  }

  void collect(Context& ctx, void* dst, bool dst_on_device) override {
    DeviceGuard guard(ctx.device());
    cudaStream_t s = ctx.stream();
    if (batch == 0 || rows == 0) return;
    const int32_t* m = device_meta(ctx);
    const T* src = reinterpret_cast<const T*>(out.data());
    if (dst_on_device) {
      enqueue_collect<T>(s, geom, m, m + batch, src, rows_cap, static_cast<T*>(dst), rows);
      return;
    }
    const size_t elems = static_cast<size_t>(batch) * rows * count;
    DeviceBuffer<T> tmp(elems, s, ctx.device());
    enqueue_collect<T>(s, geom, m, m + batch, src, rows_cap, tmp.data(), rows);
    EDA_CUDA(cudaMemcpyAsync(dst, tmp.data(), elems * sizeof(T), cudaMemcpyDeviceToHost, s));
    ctx.synchronize();
  }
};

// A slot of consecutive scratch fields X[first] .. X[first + count - 1].
template <class T>
struct ScratchSlot {
  int32_t first, count;
  ArenaSlot<T> slot;
};

// The workspace layout of a chunk of `geom.batch` elements. The scratch fields X0 .. X11 lie one
// after another in one slot; with `split` (EDA_SANITIZE, whose arena gives every slot its own
// allocation) each field is a slot of its own, so that memcheck sees an access that runs from one
// field into the next, except the orientation's work, the D (D + 1) / 2 fields from X4 that it
// reads as one block.
template <class T>
struct ChunkLayout {
  ArenaLayout lay;
  ArenaSlot<T> f[3 + kMaxDim + 1];  // r, L, L_prev, o (D >= 2), t
  std::vector<ScratchSlot<T>> scratch;
  ArenaSlot<uint8_t> u8[8];
  ArenaSlot<ElemRecord> le;
  ArenaSlot<ElemState> elems;
  ArenaSlot<CallCtl> ctl;
  ArenaSlot<double> dbl;            // 6 [B] arrays
  ArenaSlot<int64_t> count;
  ArenaSlot<int32_t> ints;          // kInts [B] arrays
  ArenaSlot<G2Result> g2;           // test results, accepted results
  ArenaSlot<RowPartial> rows_a, rows_b;
  ArenaSlot<unsigned char> sel_ws, ccl_ws;
  ArenaSlot<T> staged;
  ArenaSlot<unsigned char> orient;  // the orientation's group workspace (D >= 2)
  ArenaSlot<T> rungs;               // the operators' rung scratch: 2 group fields (group > 1)
  int32_t rung_group;               // ElemView::rung_group (1: no scratch)

  ChunkLayout(const Geom& g, bool stage_input, int32_t group, bool split)
      : rung_group(group > 1 ? group : 1) {
    const size_t field = static_cast<size_t>(g.batch) * static_cast<size_t>(g.field);
    const size_t B = static_cast<size_t>(g.batch);
    const int D = g.dim;
    for (int i = 0; i < 3 + kMaxDim + 1; ++i) {
      const bool o_slot = i >= 3 && i < 3 + kMaxDim;
      const bool used = !o_slot || (D >= 2 && i - 3 < D);
      f[i] = lay.add<T>(used ? field : 0);
    }
    const int32_t work = D >= 2 ? D * (D + 1) / 2 : 1;
    for (int32_t i = 0; i < kScratch;) {
      const int32_t count = !split ? kScratch : (i == 4 ? work : 1);
      scratch.push_back({i, count, lay.add<T>(static_cast<size_t>(count) * field)});
      i += count;
    }
    for (auto& s : u8) s = lay.add<uint8_t>(field);
    le = lay.add<ElemRecord>(B);
    elems = lay.add<ElemState>(B);
    ctl = lay.add<CallCtl>(1);
    dbl = lay.add<double>(6 * B);
    count = lay.add<int64_t>(B);
    ints = lay.add<int32_t>(static_cast<size_t>(kInts) * B);
    g2 = lay.add<G2Result>(2 * B);
    rows_a = lay.add<RowPartial>(B * static_cast<size_t>(g.rows));
    rows_b = lay.add<RowPartial>(B * static_cast<size_t>(g.rows));
    sel_ws = lay.add<unsigned char>(select_workspace_bytes(g));
    ccl_ws = lay.add<unsigned char>(ccl_workspace_bytes(g));
    staged = lay.add<T>(stage_input ? static_cast<size_t>(g.batch) * static_cast<size_t>(g.count) : 0);
    orient = lay.add<unsigned char>(D >= 2 ? orientation_workspace_bytes(g, dtype_of<T>()) : 0);
    rungs = lay.add<T>(rung_group > 1 ? 2 * static_cast<size_t>(rung_group) * field : 0);
  }
};

// The rows and the history of a call (B, cap, count) and (B, cap - 1).
template <class T>
struct RowStorage {
  DeviceBuffer<unsigned char> rows;
  DeviceBuffer<unsigned char> hist;
  int32_t cap = 0;
  int64_t B = 0;
  size_t entries() const { return static_cast<size_t>(B) * static_cast<size_t>(cap - 1); }
  int32_t* scale() const { return reinterpret_cast<int32_t*>(hist.data()); }
  int32_t* sifts() const { return scale() + entries(); }
  double* spacing() const { return reinterpret_cast<double*>(hist.data() + (2 * entries() * sizeof(int32_t) + 7) / 8 * 8); }
  double* energy() const { return spacing() + entries(); }
  static size_t hist_bytes(int64_t B, int32_t cap) {
    const size_t e = static_cast<size_t>(B) * static_cast<size_t>(cap - 1);
    return (2 * e * sizeof(int32_t) + 7) / 8 * 8 + 2 * e * sizeof(double);
  }
};

// ---------------------------------------------------------------------------------------------
// One chunk: the workspace and the mode loop.
template <class T>
class Chunk {
 public:
  Chunk(Context& ctx, const Geom& g, const MethodCtl& c, const ArenaBinding& ws,
        const ChunkLayout<T>& L, const DctPlans* orient_plans)
      : ctx_(ctx), g_(g), c_(c), s_(ctx.stream()), orient_plans_(orient_plans) {
    B_ = g.batch;
    const size_t Bz = static_cast<size_t>(B_);
    F_.r = ws.get(L.f[0]);
    F_.L = ws.get(L.f[1]);
    F_.Lp = ws.get(L.f[2]);
    for (int a = 0; a < kMaxDim; ++a) F_.o[a] = ws.get(L.f[3 + a]);
    F_.t = ws.get(L.f[3 + kMaxDim]);
    for (const ScratchSlot<T>& sc : L.scratch) {
      T* p = ws.get(sc.slot);
      for (int32_t j = 0; j < sc.count; ++j) {
        F_.X[sc.first + j] = p + static_cast<size_t>(j) * Bz * static_cast<size_t>(g.field);
      }
    }
    uint8_t** u8[8] = {&F_.flags, &F_.ff, &F_.P0, &F_.P1, &F_.Pacc, &F_.A, &F_.sel, &F_.lo};
    for (int i = 0; i < 8; ++i) *u8[i] = ws.get(L.u8[i]);
    double* dbl = ws.get(L.dbl);
    int32_t* ints = ws.get(L.ints);
    A_.le = ws.get(L.le);
    A_.elems = ws.get(L.elems);
    A_.tol = dbl;
    A_.sigma_n = dbl + Bz;
    A_.sigma_x = dbl + 2 * Bz;
    A_.floor2 = dbl + 3 * Bz;
    A_.wc = dbl + 4 * Bz;
    A_.e_med = dbl + 5 * Bz;
    A_.on = ints;
    A_.on_g2 = ints + Bz;
    A_.on_floor = ints + 2 * Bz;
    srung_ = ints + 3 * Bz;
    slot_ = ints + 4 * Bz;
    A_.win = ints + 5 * Bz;
    rung_win_ = ints + (5 + kWinSlots) * Bz;
    exp_ = ints + (5 + kWinSlots + kMaxRungs) * Bz;
    count_ = ws.get(L.count);
    g2res_ = ws.get(L.g2);
    g2acc_ = g2res_ + Bz;
    rows_a_ = ws.get(L.rows_a);
    rows_b_ = ws.get(L.rows_b);
    ctl_ = ws.get(L.ctl);
    sel_ws_ = ws.get(L.sel_ws);
    ccl_ws_ = ws.get(L.ccl_ws);
    staged_ = ws.get(L.staged);
    orient_ws_ = ws.get(L.orient);
    rung0_ = ws.get(L.rungs);
    if (rung0_ != nullptr) {
      rung_group_ = L.rung_group;
      rung1_ = rung0_ + static_cast<size_t>(rung_group_) * Bz * static_cast<size_t>(g.field);
    }
    if (g.dim >= 2 && (orient_plans_ == nullptr || orient_ws_ == nullptr)) {
      throw std::logic_error("eda: the orientation needs its plans and workspace");
    }
    // Deterministic contents everywhere (pads are zero, nothing reads unwritten memory).
    const size_t field = Bz * static_cast<size_t>(g.field);
    for (int i = 0; i < 3 + kMaxDim + 1; ++i) {
      T* p = ws.get(L.f[i]);
      if (p) EDA_CUDA(cudaMemsetAsync(p, 0, sizeof(T) * field, s_));
    }
    for (const ScratchSlot<T>& sc : L.scratch) {
      EDA_CUDA(cudaMemsetAsync(F_.X[sc.first], 0, sizeof(T) * field * static_cast<size_t>(sc.count), s_));
    }
    if (rung0_) EDA_CUDA(cudaMemsetAsync(rung0_, 0, sizeof(T) * field * 2 * static_cast<size_t>(rung_group_), s_));
    for (int i = 0; i < 8; ++i) EDA_CUDA(cudaMemsetAsync(ws.get(L.u8[i]), 0, field, s_));
    EDA_CUDA(cudaMemsetAsync(A_.le, 0, sizeof(ElemRecord) * Bz, s_));
    EDA_CUDA(cudaMemsetAsync(A_.elems, 0, sizeof(ElemState) * Bz, s_));
    EDA_CUDA(cudaMemsetAsync(dbl, 0, sizeof(double) * 6 * Bz, s_));
    EDA_CUDA(cudaMemsetAsync(ints, 0, sizeof(int32_t) * kInts * Bz, s_));
    EDA_CUDA(cudaMemsetAsync(g2res_, 0, sizeof(G2Result) * 2 * Bz, s_));
    const auto up64 = [](size_t v) { return (v + 63) / 64 * 64; };
    const size_t o_res = up64(Bz * sizeof(ElemRecord));
    const size_t o_acc = o_res + up64(Bz * sizeof(G2Result));
    const size_t o_srung = o_acc + up64(Bz * sizeof(G2Result));
    const size_t o_slot = o_srung + up64(Bz * sizeof(int32_t));
    const size_t o_ctl = o_slot + up64(Bz * sizeof(int32_t));
    pin_ = pinned_acquire(o_ctl + up64(sizeof(CallCtl)));
    char* base = static_cast<char*>(pin_.data());
    hle_ = reinterpret_cast<ElemRecord*>(base);
    hres_ = reinterpret_cast<G2Result*>(base + o_res);
    hacc_ = reinterpret_cast<G2Result*>(base + o_acc);
    hsrung_ = reinterpret_cast<int32_t*>(base + o_srung);
    hslot_ = reinterpret_cast<int32_t*>(base + o_slot);
    hctl_ = reinterpret_cast<CallCtl*>(base + o_ctl);
  }

  // Canonical copy of x (device pointer with strides), the prescale and the initial state.
  // orient: the chunk's orientation rows (B, cap - 1, D + 1, count), or nullptr.
  void start(const T* x, const InputStrides& xs, void* out_rows, int32_t cap, int32_t* scale, double* spacing,
             int32_t* sifts, double* energy, int32_t max_imf, double tau, bool check_finite, T* orient) {
    out_rows_ = static_cast<T*>(out_rows);
    orient_ = orient;
    cap_ = cap;
    CallCtl hc{};
    hc.out = out_rows;
    hc.rows_cap = cap;
    hc.max_imf = max_imf;
    hc.hist_scale = scale;
    hc.hist_spacing = spacing;
    hc.hist_sifts = sifts;
    hc.hist_energy = energy;
    hc.tau = static_cast<double>(static_cast<T>(tau));
    hc.check_finite = check_finite ? 1 : 0;
    EDA_CUDA(cudaMemcpyAsync(ctl_, &hc, sizeof hc, cudaMemcpyHostToDevice, s_));
    enqueue_canonical_in<T>(s_, g_, x, xs, F_.r);
    enqueue_row_partials<T>(s_, g_, F_.r, rows_a_, A_.elems);
    enqueue_reduce_partials(s_, g_, rows_a_, A_.elems);
    // r <- r 2^-k (exact), then amax and r_energy of the scaled field
    enqueue_mode_prescale<T>(s_, g_, A_.elems, exp_, F_.r);
    enqueue_row_partials<T>(s_, g_, F_.r, rows_a_, nullptr);
    enqueue_reduce_partials(s_, g_, rows_a_, A_.elems);
    hexp_.assign(static_cast<size_t>(B_), 0);
    enqueue_init_state<T>(s_, g_, A_.elems, ctl_);
    enqueue_mode_init(s_, B_, A_);
    enqueue_noise_sigma<T>(s_, g_, F_.r, true, A_.sigma_x, sel_ws_, A_.on);
  }

  // One mode of every active element; false when no element is active any more.
  bool mode() {
    const int D = g_.dim;
    const size_t field = static_cast<size_t>(B_) * static_cast<size_t>(g_.field);
    int32_t* win = A_.win;
    int32_t* Wd = win + kWinWd * B_;
    int32_t* Wl = win + kWinWl * B_;
    int32_t* Ww = win + kWinW * B_;
    T* tau = F_.X[0];
    T* tmp = F_.X[1];

    // step 0: mode start, sigma_n
    enqueue_mode_start(s_, B_, c_, A_);
    enqueue_noise_sigma<T>(s_, g_, F_.r, false, A_.sigma_n, sel_ws_, A_.on);
    // steps 1-2: the window passes
    for (int32_t pass = 0; pass < c_.sig_passes; ++pass) {
      enqueue_tau<T>(s_, g_, c_, F_.r, Wd, Wl, A_.sigma_n, A_.tol, nullptr, tau, tmp, A_.on, A_.le);
      enqueue_mode_guard(s_, B_, A_);
      enqueue_sig_flags<T>(s_, g_, F_.r, tau, F_.flags, A_.le, A_.on, tmp);
      enqueue_mode_rice(s_, B_, c_, g_, pass, A_);
    }
    // S1: which elements are active, and which run G2
    read_le();
    std::vector<RungMachine> mach(static_cast<size_t>(B_));
    bool any_active = false, any_g2 = false;
    for (int64_t b = 0; b < B_; ++b) {
      const ElemRecord& e = hle_[b];
      RungMachine& m = mach[static_cast<size_t>(b)];
      m.done = true;
      if (e.status != kRecordActive) continue;
      any_active = true;
      const int32_t s0 = std::max(3, e.w_prev + 2);
      if (c_.fine_jump * static_cast<double>(s0) <= static_cast<double>(e.w)) {
        m = make_rung_machine(e.w, e.w_prev, c_.fine_jump);
        any_g2 = any_g2 || !m.done;
      }
    }
    if (!any_active) {
      return false;
    }

    // step 3: G2
    const uint8_t* ff = F_.flags;
    if (any_g2) {
      if (c_.fine_theta > 0) {
        enqueue_tau<T>(s_, g_, c_, F_.r, Wd, Wl, A_.sigma_n, A_.tol, A_.floor2, tau, tmp, A_.on_g2, A_.le);
        enqueue_sig_flags<T>(s_, g_, F_.r, tau, F_.ff, A_.le, A_.on_g2, tmp);
        ff = F_.ff;
      }
      uint8_t* Pbuf[2] = {F_.P0, F_.P1};
      for (int32_t step = 0;; ++step) {
        bool any = false;
        for (int64_t b = 0; b < B_; ++b) {
          const RungMachine& m = mach[static_cast<size_t>(b)];
          hsrung_[b] = m.done ? 0 : m.next_rung();
          any = any || !m.done;
        }
        if (!any) break;
        if (step > kMaxG2Rungs + 1) throw std::logic_error("eda: the G2 rung loop did not end");
        EDA_CUDA(cudaMemcpyAsync(srung_, hsrung_, sizeof(int32_t) * B_, cudaMemcpyHostToDevice, s_));
        enqueue_g2_test<T>(s_, g_, c_, A_.le, srung_, ff, Pbuf[step & 1], F_.X[1], F_.X[2], g2res_, sel_ws_);
        EDA_CUDA(cudaMemcpyAsync(hres_, g2res_, sizeof(G2Result) * B_, cudaMemcpyDeviceToHost, s_));
        ctx_.synchronize();  // S2: the outcomes decide the next rung of every machine
        for (int64_t b = 0; b < B_; ++b) {
          RungMachine& m = mach[static_cast<size_t>(b)];
          if (!m.done) m.feed(hres_[b]);
        }
      }
      bool any_acc = false;
      for (int64_t b = 0; b < B_; ++b) {
        const RungMachine& m = mach[static_cast<size_t>(b)];
        std::memset(&hacc_[b], 0, sizeof(G2Result));
        hacc_[b].outcome = kG2None;
        hslot_[b] = 0;
        if (m.accepted) {
          hacc_[b] = m.cand;
          hacc_[b].outcome = kG2Accepted;
          hslot_[b] = m.acc_index & 1;
          any_acc = true;
        }
      }
      if (any_acc) {
        EDA_CUDA(cudaMemcpyAsync(g2acc_, hacc_, sizeof(G2Result) * B_, cudaMemcpyHostToDevice, s_));
        EDA_CUDA(cudaMemcpyAsync(slot_, hslot_, sizeof(int32_t) * B_, cudaMemcpyHostToDevice, s_));
        enqueue_mode_g2_accept(s_, g_, A_.le, g2acc_, slot_, F_.P0, F_.P1, F_.Pacc);
        enqueue_g3_coarse<T>(s_, g_, c_, A_.le, ff, F_.Pacc, A_.wc, F_.X[1], F_.X[2], F_.sel, sel_ws_);
      }
    }

    // step 4: the window (the guard first: G2's threshold)
    enqueue_mode_guard(s_, B_, A_);
    enqueue_mode_window(s_, B_, c_, A_);
    read_le();  // S3: the elements still active after the window rule
    any_active = false;
    for (int64_t b = 0; b < B_; ++b) {
      if (hle_[b].status != kRecordActive) {
        continue;
      }
      any_active = true;
    }
    if (!any_active) {
      return false;
    }

    // step 5: the scale field
    T* om = F_.X[0];
    enqueue_raw_window<T>(s_, g_, c_, F_.flags, Ww, om, F_.X[1], A_.on);
    enqueue_scale_field<T>(s_, g_, c_, A_.le, om, F_.A, F_.L, F_.X[1], F_.X[2],
                           reinterpret_cast<int32_t*>(F_.X[3]), ccl_ws_, A_.on);
    bool any_floor = false, any_fine = false;
    for (int64_t b = 0; b < B_; ++b) {
      if (hle_[b].status != kRecordActive) continue;
      any_floor = any_floor || hle_[b].k >= 1;
      any_fine = any_fine || hle_[b].fine;
    }
    if (any_floor) enqueue_local_floor<T>(s_, g_, c_, A_.le, F_.Lp, F_.A, F_.L, F_.X[1], A_.on_floor);
    if (any_fine) enqueue_mode_g3<T>(s_, g_, A_.le, F_.L, A_.on);

    // step 6: orientation and the element build
    FieldVec<T> n{};
    T* coh = F_.X[3];
    if (D >= 2) {
      for (int a = 0; a < D; ++a) n.p[a] = F_.X[a];
      // n: X0..X(D-1) (one field per component, component-major over the batch: the D component
      // slots of a FieldVec are separate fields here); work: X4..
      // the orientation: R1 in X0..X(D-1) (n's slots: n is written last), R2 and then T in X4..
      std::vector<uint8_t> run(static_cast<size_t>(B_));
      for (int64_t b = 0; b < B_; ++b) run[b] = hle_[b].status == kRecordActive ? 1 : 0;
      enqueue_orientation<T>(s_, g_, *orient_plans_, F_.r, Ww, n, coh, F_.X[4], n, orient_ws_, A_.on, A_.le,
                             run.data());
      enqueue_mode_guard(s_, B_, A_);
      // return_orientation: a copy of n and c at the row of this mode's IMF (every element that is
      // still active commits the mode)
      if (orient_) {
        FieldVec<const T> nv{};
        for (int a = 0; a < D; ++a) nv.p[a] = n.p[a];
        enqueue_orientation_store<T>(s_, g_, nv, coh, A_.le, A_.elems, orient_, cap_ - 1);
      }
    } else {
      EDA_CUDA(cudaMemsetAsync(coh, 0, sizeof(T) * field, s_));
    }
    FieldVec<const T> nc{};
    FieldVec<T> o{};
    for (int a = 0; a < D && D >= 2; ++a) {
      nc.p[a] = n.p[a];
      o.p[a] = F_.o[a];
    }
    enqueue_element_build<T>(s_, g_, c_, F_.L, nc, coh, o, A_.le, A_.on);
    // S4 and step 7: the ladders (the host builds them from d_min, d_max)
    read_le();
    int32_t n_rungs_max = 0;
    for (int64_t b = 0; b < B_; ++b) {
      ElemRecord& e = hle_[b];
      e.rung_pts = 0;
      e.rung_used = 0;
      if (e.status != kRecordActive) {
        e.n_rungs = 0;
        continue;
      }
      double d_min, d_max;
      std::memcpy(&d_min, &e.d_min_bits, sizeof d_min);
      std::memcpy(&d_max, &e.d_max_bits, sizeof d_max);
      const std::vector<int32_t> ladder = rung_ladder(d_min, d_max, e.w, c_.n_max);
      if (ladder.empty() || static_cast<int32_t>(ladder.size()) > kMaxRungs) {
        throw std::logic_error("eda: ladder size out of range");
      }
      e.n_rungs = static_cast<int32_t>(ladder.size());
      for (int32_t j = 0; j < kMaxRungs; ++j) e.rung_w[j] = j < e.n_rungs ? ladder[static_cast<size_t>(j)] : 0;
      n_rungs_max = std::max(n_rungs_max, e.n_rungs);
    }
    write_le();
    enqueue_element_weights<T>(s_, g_, c_, F_.L, coh, A_.le, F_.lo, F_.t, A_.on);
    enqueue_element_rung_windows(s_, static_cast<int32_t>(B_), A_.le, n_rungs_max, rung_win_);
    ElemView<T> ev{};
    ev.lo = F_.lo;
    ev.t = F_.t;
    for (int a = 0; a < D && D >= 2; ++a) ev.o.p[a] = F_.o[a];
    ev.rung_win = rung_win_;
    ev.le = A_.le;
    ev.n_rungs_max = n_rungs_max;
    ev.rung0 = rung0_;
    ev.rung1 = rung1_;
    ev.rung_group = rung_group_;

    // step 8: sift 1 and the continuation weight. h lives in the L_prev slot (the floor has read
    // L_prev).
    T* X[kScratch];
    for (int i = 0; i < kScratch; ++i) X[i] = F_.X[i];
    T* h = F_.Lp;
    T* r = F_.r;
    enqueue_envelope_mean<T>(s_, g_, ev, r, X[0], X[1], X[2], X[3], X[4]);
    enqueue_mode_sift<T>(s_, g_, A_.le, 0, r, r, X[0], nullptr, h, nullptr);
    enqueue_mode_accept<T>(s_, g_, A_.le, 0, h, h);
    T* a = X[8];
    if (c_.sifts > 1) {
      // R = max(Dil r - Ero r, 0); U = Dil^u R; g = ramp(R / U); a = clip(Avg^s g, 0, 1)
      T* R = X[5];
      enqueue_mode_range<T>(s_, g_, X[1], X[2], R, A_.on);
      T* U = R;
      T* pp[2] = {X[6], X[7]};
      for (int32_t i = 0; i < c_.u; ++i) {
        T* dst = pp[i & 1];
        enqueue_element_op<T>(s_, g_, ElemOp::kMax, ev, U, dst, nullptr, X[3], X[4]);
        U = dst;
      }
      T* gw = U == X[6] ? X[7] : X[6];
      enqueue_mode_ramp<T>(s_, g_, c_, A_.le, R, U, gw, A_.on);
      T* cur = gw;
      for (int32_t i = 0; i < c_.s; ++i) {
        T* dst = a;
        if (i + 1 < c_.s) dst = cur == X[5] ? X[6] : X[5];
        enqueue_element_op<T>(s_, g_, ElemOp::kBox, ev, cur, dst, nullptr, X[3], X[4]);
        cur = dst;
      }
      if (cur != a) EDA_CUDA(cudaMemcpyAsync(a, cur, sizeof(T) * field, cudaMemcpyDeviceToDevice, s_));
      enqueue_mode_clip01<T>(s_, g_, a, A_.on);
    }

    // step 9: sifts 2..S with G1
    for (int32_t j = 1; j < c_.sifts; ++j) {
      T* plain = X[0];
      T* up = X[1];
      T* lo = X[2];
      enqueue_envelope_mean<T>(s_, g_, ev, h, plain, up, lo, X[3], X[4]);
      enqueue_mode_half_diff<T>(s_, g_, up, lo, X[5], A_.on);
      T* e0 = X[6];
      enqueue_element_op<T>(s_, g_, ElemOp::kBox, ev, X[5], e0, nullptr, X[3], X[4]);
      // G1's ramp: kap from the W-cube max / min of e0, gk = clip(Box_W ramp(kap), 0, 1)
      enqueue_cube<T>(s_, g_, CubeOp::kMaxMin, e0, X[5], X[7], Ww);
      enqueue_mode_kappa_ramp<T>(s_, g_, c_, A_.le, X[5], X[7], X[5], A_.on);
      T* gk = X[7];
      enqueue_cube<T>(s_, g_, CubeOp::kBox, X[5], gk, nullptr, Ww);
      enqueue_mode_clip01<T>(s_, g_, gk, A_.on);
      // e = Box_W^smooth e0, floored by anorm_floor median(e)
      T* e = X[9];
      if (c_.anorm_smooth <= 0) {
        EDA_CUDA(cudaMemcpyAsync(e, e0, sizeof(T) * field, cudaMemcpyDeviceToDevice, s_));
      }
      for (int32_t i = 0; i < c_.anorm_smooth; ++i) enqueue_cube<T>(s_, g_, CubeOp::kBox, i ? e : e0, e, nullptr, Ww);
      Selection<T> sel{};
      sel.src = SelSource::kField;
      sel.field = e;
      sel.mask = nullptr;
      sel.bits = 0;
      sel.axes = 0;
      sel.center = nullptr;
      enqueue_median<T>(s_, g_, sel, A_.e_med, count_, sel_ws_, A_.on);
      enqueue_mode_floor<T>(s_, g_, c_, A_.le, A_.e_med, e, A_.on);
      // the normalized mean and the blend
      T* q = X[6];
      enqueue_mode_ratio<T>(s_, g_, h, e, q, A_.on);
      T* Mn = X[5];
      enqueue_envelope_mean<T>(s_, g_, ev, q, Mn, X[10], X[11], X[3], X[4]);
      enqueue_mode_blend<T>(s_, g_, gk, e, Mn, up, lo, plain, Mn, A_.on);
      T* mh = Mn;
      T* hn = X[6];
      enqueue_mode_sift<T>(s_, g_, A_.le, j, r, h, mh, a, hn, nullptr);
      enqueue_mode_accept<T>(s_, g_, A_.le, j, hn, h);
    }

    // step 10: commit; the next mode's L_prev is this L (the old L_prev slot held h and is free)
    enqueue_mode_commit<T>(s_, g_, ctl_, A_.elems, A_.le, h, r, rows_a_, rows_b_);
    std::swap(F_.L, F_.Lp);
    EDA_CUDA(cudaMemcpyAsync(hctl_, ctl_, sizeof(CallCtl), cudaMemcpyDeviceToHost, s_));
    ctx_.synchronize();  // S5: whether another mode runs
    if (hctl_->any_full) throw std::logic_error("eda: the output rows are full (mode bound violated)");
    return hctl_->any_active != 0;
  }

  // The residue row, the rows scaled back to x's magnitude and the per-element results; exps():
  // the prescale exponents.
  void finish(std::vector<ElemState>& out) {
    enqueue_write_residue<T>(s_, g_, A_.elems, ctl_, F_.r);
    enqueue_mode_unscale<T>(s_, g_, A_.elems, exp_, out_rows_, cap_);
    out.resize(static_cast<size_t>(B_));
    EDA_CUDA(cudaMemcpyAsync(out.data(), A_.elems, sizeof(ElemState) * B_, cudaMemcpyDeviceToHost, s_));
    EDA_CUDA(cudaMemcpyAsync(hexp_.data(), exp_, sizeof(int32_t) * B_, cudaMemcpyDeviceToHost, s_));
    ctx_.synchronize();
  }
  const std::vector<int32_t>& exps() const { return hexp_; }

 private:
  void read_le() {
    EDA_CUDA(cudaMemcpyAsync(hle_, A_.le, sizeof(ElemRecord) * B_, cudaMemcpyDeviceToHost, s_));
    ctx_.synchronize();
  }
  void write_le() {
    EDA_CUDA(cudaMemcpyAsync(A_.le, hle_, sizeof(ElemRecord) * B_, cudaMemcpyHostToDevice, s_));
  }

  Context& ctx_;
  Geom g_;
  MethodCtl c_;
  cudaStream_t s_;
  int64_t B_ = 0;
  ChunkFields<T> F_{};
  ModeArrays A_{};
  int32_t* srung_ = nullptr;
  int32_t* slot_ = nullptr;
  int32_t* rung_win_ = nullptr;
  int32_t* exp_ = nullptr;
  std::vector<int32_t> hexp_;
  T* out_rows_ = nullptr;
  T* orient_ = nullptr;
  int32_t cap_ = 0;
  int64_t* count_ = nullptr;
  G2Result* g2res_ = nullptr;
  G2Result* g2acc_ = nullptr;
  RowPartial* rows_a_ = nullptr;
  RowPartial* rows_b_ = nullptr;
  CallCtl* ctl_ = nullptr;
  void* sel_ws_ = nullptr;
  void* ccl_ws_ = nullptr;
  T* staged_ = nullptr;
  const DctPlans* orient_plans_ = nullptr;
  unsigned char* orient_ws_ = nullptr;
  T* rung0_ = nullptr;
  T* rung1_ = nullptr;
  int32_t rung_group_ = 1;
  PinnedBlock pin_;
  ElemRecord* hle_ = nullptr;
  G2Result* hres_ = nullptr;
  G2Result* hacc_ = nullptr;
  int32_t* hsrung_ = nullptr;
  int32_t* hslot_ = nullptr;
  CallCtl* hctl_ = nullptr;

 public:
  T* staged() const { return staged_; }
};

template <class T>
std::unique_ptr<Run> run_method_t(Context& ctx, const Input& in, const Params& p) {
  DeviceGuard guard(ctx.device());
  const Geom g = make_geom(in.dim, in.batch, in.n);
  const int64_t B = g.batch;
  MethodCtl c = method_ctl(g, dtype_of<T>());
  c.sifts = p.sifts;
  cudaStream_t stream = ctx.stream();
  const RuntimeConfig& cfg = ctx.config();

  // Output rows and history: the mode count is bounded (every window grows), so they never regrow.
  const int32_t bound = mode_bound(c);
  const int32_t modes_max = p.max_imf > 0 ? std::min(bound, p.max_imf) : bound;
  const int32_t cap = modes_max + 2;
  RowStorage<T> st;
  st.cap = cap;
  st.B = B;
  DeviceBuffer<unsigned char> orient;
  {
    const size_t row_bytes = static_cast<size_t>(B) * cap * static_cast<size_t>(g.count) * sizeof(T);
    const size_t hist_bytes = RowStorage<T>::hist_bytes(B, cap);
    // return_orientation: n and c of every mode, cap - 1 rows as the history
    const size_t orient_bytes =
        p.orientation ? static_cast<size_t>(B) * (cap - 1) * (g.dim + 1) * static_cast<size_t>(g.count) * sizeof(T) : 0;
    const int64_t avail = device_available_bytes(ctx.device());
    const auto oom = [&]() {
      return OutOfDeviceMemory("eda: the output rows do not fit the device memory: " + std::to_string(cap) +
                               " rows of " + std::to_string(g.count) + " " + dtype_name(dtype_of<T>()) +
                               " samples for " + std::to_string(B) + " element" + (B == 1 ? "" : "s") + " need " +
                               format_gib(static_cast<int64_t>(row_bytes)) +
                               (p.orientation ? " and their orientation " + format_gib(static_cast<int64_t>(orient_bytes))
                                              : std::string()) +
                               ", and " + format_gib(avail) +
                               " of device memory are free; set max_imf or decompose fewer elements per call");
    };
    if (static_cast<int64_t>(row_bytes + hist_bytes + orient_bytes) > avail) throw oom();
    try {
      st.rows = DeviceBuffer<unsigned char>(row_bytes, stream, ctx.device());
      st.hist = DeviceBuffer<unsigned char>(hist_bytes, stream, ctx.device());
      if (orient_bytes) orient = DeviceBuffer<unsigned char>(orient_bytes, stream, ctx.device());
    } catch (const CudaError& e) {
      if (e.code() != cudaErrorMemoryAllocation) throw;
      throw oom();
    }
    EDA_CUDA(cudaMemsetAsync(st.hist.data(), 0, hist_bytes, stream));
    if (cfg.sanitize) EDA_CUDA(cudaMemsetAsync(st.rows.data(), 0, row_bytes, stream));
  }

  // The orientation's plans (one group geometry per shape, cached by the context).
  const DctPlans* orient_plans = g.dim >= 2 ? &orientation_plans(ctx, g, dtype_of<T>()) : nullptr;

  // The chunk: the largest batch whose workspace fits the free device memory (halving). Then the
  // rung group of the operators' cubes (vhgw_rung_group): its scratch (at most half the L2 cache)
  // is added only when it fits beside the chunk's workspace, so it never changes the chunking; every
  // chunk takes the first one's group.
  const bool stage = !in.on_device || in.source != nullptr;  // host input, or converted device input
  int64_t Bc = B;
  if (p.max_chunk > 0) Bc = std::min(Bc, p.max_chunk);
  int32_t group = 1;
  {
    const int64_t avail = device_available_bytes(ctx.device());
    const int64_t margin = std::max<int64_t>(int64_t{256} << 20, avail / 20);
    for (;;) {
      const ChunkLayout<T> L(chunk_geom(g, Bc), stage, 1, cfg.sanitize);
      const int64_t need = static_cast<int64_t>(L.lay.bytes());
      if (need + margin <= avail) break;
      if (Bc == 1) {
        throw OutOfDeviceMemory("eda: one element needs a workspace of " + format_gib(need) + " and " +
                                format_gib(avail) + " of device memory are free");
      }
      Bc = (Bc + 1) / 2;
    }
    group = vhgw_rung_group<T>(chunk_geom(g, Bc), kMaxRungs);
    const ChunkLayout<T> grouped(chunk_geom(g, Bc), stage, group, cfg.sanitize);
    if (group > 1 && static_cast<int64_t>(grouped.lay.bytes()) + margin > avail) {
      group = 1;
    }
  }

  auto run = std::make_unique<MethodRun<T>>();
  run->geom = g;
  run->batch = B;
  run->count = g.count;
  run->rows_cap = cap;
  run->chunk = Bc;
  run->n_imfs.assign(static_cast<size_t>(B), 0);
  run->error.assign(static_cast<size_t>(B), 0);
  const double tau = p.tie_tolerance.value_or(default_tie_tolerance(dtype_of<T>()));
  std::vector<ElemState> host_elems;
  std::vector<int32_t> exps(static_cast<size_t>(B), 0);
  const size_t es = sizeof(T);
  for (int64_t b0 = 0; b0 < B; b0 += Bc) {
    const int64_t nb = std::min(Bc, B - b0);
    const Geom gc = chunk_geom(g, nb);
    const ChunkLayout<T> L(gc, stage, group, cfg.sanitize);
    const ArenaBinding ws = bind_arena(ctx, L.lay);
    Chunk<T> chunk(ctx, gc, c, ws, L, orient_plans);
    const T* x;
    InputStrides xs;
    if (in.source) {
      enqueue_convert_gather<T>(stream, *in.source, in.x, b0, gc, chunk.staged());
      x = chunk.staged();
      xs = c_strides(gc);
    } else if (stage) {
      EDA_CUDA(cudaMemcpyAsync(chunk.staged(), static_cast<const T*>(in.x) + b0 * g.count,
                               static_cast<size_t>(nb) * g.count * es, cudaMemcpyHostToDevice, stream));
      x = chunk.staged();
      xs = c_strides(gc);
    } else {
      x = static_cast<const T*>(in.x) + b0 * in.strides.s[0];
      xs = in.strides;
    }
    const size_t he = static_cast<size_t>(cap - 1);
    unsigned char* rows0 = st.rows.data() + static_cast<size_t>(b0) * cap * static_cast<size_t>(g.count) * es;
    T* orient0 = orient.empty() ? nullptr
                                : reinterpret_cast<T*>(orient.data()) +
                                      static_cast<size_t>(b0) * he * (g.dim + 1) * static_cast<size_t>(g.count);
    chunk.start(x, xs, rows0, cap, st.scale() + b0 * he, st.spacing() + b0 * he, st.sifts() + b0 * he,
                st.energy() + b0 * he, p.max_imf, tau, p.check_finite, orient0);
    while (chunk.mode()) {
    }
    chunk.finish(host_elems);
    std::copy(chunk.exps().begin(), chunk.exps().end(), exps.begin() + b0);
    for (int64_t b = 0; b < nb; ++b) {
      const ElemState& e = host_elems[static_cast<size_t>(b)];
      const size_t ib = static_cast<size_t>(b0 + b);
      if (e.status == kError) {
        run->error[ib] = e.error ? e.error : kErrInternal;
        run->n_imfs[ib] = 0;
      } else if (e.error == kErrNonFiniteResult) {  // a row overflowed when scaled back (unscale)
        run->error[ib] = kErrNonFiniteResult;
        run->n_imfs[ib] = 0;
      } else if (e.status != kDone || e.n_imfs + 1 > cap) {
        run->error[ib] = kErrInternal;
        run->n_imfs[ib] = 0;
      } else {
        run->n_imfs[ib] = e.n_imfs;
      }
    }
  }
  int32_t P = 0;
  bool any_error = false;
  for (int64_t b = 0; b < B; ++b) {
    if (run->error[b]) any_error = true;
    else P = std::max(P, run->n_imfs[b] + 1);
  }
  if (any_error) P = std::max(P, 1);
  run->rows = P;
  const size_t he = st.entries();
  run->hist_scale.resize(he);
  run->hist_sifts.resize(he);
  run->hist_spacing.resize(he);
  run->hist_energy.resize(he);
  if (he) {
    EDA_CUDA(cudaMemcpyAsync(run->hist_scale.data(), st.scale(), he * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    EDA_CUDA(cudaMemcpyAsync(run->hist_sifts.data(), st.sifts(), he * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    EDA_CUDA(cudaMemcpyAsync(run->hist_spacing.data(), st.spacing(), he * sizeof(double), cudaMemcpyDeviceToHost, stream));
    EDA_CUDA(cudaMemcpyAsync(run->hist_energy.data(), st.energy(), he * sizeof(double), cudaMemcpyDeviceToHost, stream));
  }
  ctx.synchronize();
  // sum IMF^2 in x's units: 4^k times the scaled sum (exact; +inf beyond the double range)
  for (int64_t b = 0; b < B && he; ++b) {
    const int32_t k = exps[static_cast<size_t>(b)];
    if (k == 0 || run->error[b]) continue;
    double* e = run->hist_energy.data() + static_cast<size_t>(b) * static_cast<size_t>(cap - 1);
    for (int32_t j = 0; j < run->n_imfs[b]; ++j) e[j] = std::ldexp(e[j], 2 * k);
  }
  run->out = std::move(st.rows);
  run->orient = std::move(orient);
  return run;
}

}  // namespace

InputStrides c_strides(const Geom& g) {
  InputStrides xs{};
  xs.s[g.dim] = 1;
  for (int a = g.dim - 1; a >= 1; --a) xs.s[a] = xs.s[a + 1] * g.n[a];
  xs.s[0] = g.count;
  return xs;
}

ArenaBinding bind_arena(Context& ctx, const ArenaLayout& lay) {
  try {
    return ctx.arena().bind(lay);
  } catch (const CudaError& e) {
    if (e.code() != cudaErrorMemoryAllocation) throw;
    throw OutOfDeviceMemory("eda: the workspace (" + format_gib(static_cast<int64_t>(lay.bytes())) +
                            ") does not fit the device memory (" + format_gib(device_available_bytes(ctx.device())) +
                            " free); decompose fewer elements per call");
  }
}

Geom chunk_geom(const Geom& g, int64_t batch) {
  int64_t n[kMaxDim] = {g.n[0], g.n[1], g.n[2]};
  return make_geom(g.dim, batch, n);
}

// ---------------------------------------------------------------------------------------------
std::string extent_message(int dim, int axis) {
  if (dim == 1) {
    return "eda: a 1D signal needs at least 2 samples, got 1 (one sample has no sample differences, "
           "so no noise level; it is its own residue)";
  }
  return "eda: every spatial axis of a " + std::to_string(dim) + "D field needs extent >= 2, but "
         "spatial axis " + std::to_string(axis) + " has extent 1; drop that axis (squeeze it, or leave it out of "
         "axes=) to decompose the field in " + std::to_string(dim - 1) + "D";
}

void validate_method(const Input& in, const Params& p) {
  if (in.dim < 1 || in.dim > kMaxDim) throw std::invalid_argument("eda: D must be 1, 2 or 3");
  if (in.dtype != DType::F32 && in.dtype != DType::F64) throw std::invalid_argument("eda: the input must be float32 or float64");
  const int32_t lim = max_axis_len(in.dtype);
  for (int a = 0; a < in.dim; ++a) {
    if (in.n[a] < 1) throw std::invalid_argument("eda: every spatial extent must be >= 1");
    if (in.n[a] > lim) {
      throw std::invalid_argument("eda: axis extent " + std::to_string(in.n[a]) + " exceeds the limit " +
                                  std::to_string(lim) + " for " + dtype_name(in.dtype));
    }
  }
  // 2 samples per axis: the noise level is a median of sample differences, which one sample does
  // not have
  for (int a = 0; a < in.dim; ++a) {
    if (in.n[a] < 2) throw std::invalid_argument(extent_message(in.dim, a));
  }
  if (in.batch < 1) throw std::invalid_argument("eda: the batch needs at least one element");
  if (p.tie_tolerance && (!(*p.tie_tolerance >= 0) || !std::isfinite(*p.tie_tolerance))) {
    throw std::invalid_argument("eda: tie_tolerance must be a finite value >= 0");
  }
  if (p.sifts < 1) throw std::invalid_argument("eda: sifts must be >= 1");
  if (p.max_chunk < 0) throw std::invalid_argument("eda: max_chunk must be >= 0");
  if (p.orientation && in.dim < 2) {
    throw std::invalid_argument("eda: return_orientation=True needs 2 or 3 spatial axes, got 1");
  }
}

std::unique_ptr<Run> run_method(Context& ctx, const Input& in, const Params& p) {
  validate_method(in, p);
  if (in.dtype == DType::F32) return run_method_t<float>(ctx, in, p);
  return run_method_t<double>(ctx, in, p);
}

}  // namespace eda
