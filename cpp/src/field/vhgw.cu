// SPDX-License-Identifier: Apache-2.0
// eda_vhgw_pass<T>: one separable pass of the cube filter (a running max, min or box sum of odd
// width along one axis), by the van Herk / Gil-Werman scheme.
//
// A CTA stages P whole lines of the pass axis in shared memory (two buffers per line: the stage
// input V and R, which holds the folds H and then the stage output; the shared memory and the grid
// depend on the geometry only, never on the window) and runs up to two stages on them: max (or
// min) and then the box sum, for the field U (max) and then L (min). The last stage of a pass
// stores its result (the box-scale epilogue multiplies it by fl_T(1 / prod w) first).
//
// Canonical arithmetic, per line of length n and window w (radius r = min((w-1)/2, n-1),
// width wa = 2r+1):
//   - the extended line e in [-r, n+r) reads the line through half-sample reflection;
//   - it is cut into segments of wa samples starting at o + k*wa, o = (n*(wa+1)/2) mod wa: the
//     unique offset whose cut set is symmetric under e -> n-1-e;
//   - the prefix fold G of a segment: chunks of 32 samples from its start, a sequential fold inside
//     each chunk, a sequential carry over the chunk totals, G_j = carry_q op local_j (chunk 0:
//     local_j). The suffix fold H is G of the reversed segment (chunks from its end), computed here
//     as G of the reversed line, whose segments are the mirror images;
//   - y(x) = op(H(x-r), G(x+r)); when x-r starts a segment (both ends in one segment) the sum takes
//     (H + G) * 0.5 and max/min take op(H, G) = H.
// Max/min are exact; every sum depends only on (line values, n, w), never on P, the CTA or the
// grid, and the mirrored line computes the mirrored values bitwise (G and H swap, op commutes).
//
// Work mapping. A stage runs two phases of warp tasks over (line, batch of segments): H, then G.
// A segment of up to 32 chunks takes ls = min(chunks, 32) lanes (32/ls segments per warp); a lane
// folds its chunk sequentially, the carries go sequentially through shuffles inside the segment's
// lanes, and each lane applies its carry to its own samples. A segment of more than 32 chunks takes
// a whole warp, 32 chunks at a time, with a running carry. The H phase stores the local folds in R
// and applies the carry in place; the G phase folds each chunk twice (its total for the carries,
// then the local folds again, the same operations in the same order) and replaces H(x-r) in R by
// y(x) at once. The stores then read R with a fixed mapping (idx = tid + k*256). Max/min stages
// fold order keys as integers (order_key; the same bits as
// max_c/min_c without NaN; a CTA with a NaN in its lines folds the values with max_c/min_c).
// Shared-memory lines use the swizzle swz<T> and a line stride of round_up(n, 32) + 1 elements:
// chunk walks and the panel transposes are then free of bank conflicts.
#include "runtime/error.hpp"
#include "field/common.cuh"
#include "field/vhgw.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace eda {
namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxLines = 64;
// Stage ops. kOpMax/kOpMin fold the values with max_c/min_c; kOpMaxK/kOpMinK fold their order keys
// (order_key) as integers, which gives the same bits for every input without NaN (a CTA whose lines
// hold a NaN takes kOpMax/kOpMin, whose result with NaN depends on the operand order).
constexpr int kOpMax = 0, kOpMin = 1, kOpSum = 2, kOpMaxK = 3, kOpMinK = 4;
constexpr int kLineBuffers = 2;  // V (stage input) and R (H folds, then the stage output)

template <class T>
constexpr int kPer = sizeof(T) == 4 ? 32 : 16;  // output samples per thread: P*n <= 256 * kPer

struct LineInfo {
  int64_t base;     // global offset of sample 0 (line stride: 1 on the contiguous axis, else step)
  int64_t b;        // element
  int32_t ok;       // valid line with an odd window >= 1
  int32_t r, wa, o;
  int32_t k_lo, nsegs, ls, spw, ng, tasks;  // phase A, per direction
  double fac;       // epilogue factor (exact in T)
};

__host__ __device__ inline int64_t line_stride_elems(int32_t n) { return (static_cast<int64_t>(n) + 31) / 32 * 32 + 1; }

template <class T>
size_t smem_bytes(int32_t n, int P) {
  const size_t lines = kLineBuffers * static_cast<size_t>(P) * static_cast<size_t>(line_stride_elems(n)) * sizeof(T);
  const size_t table = static_cast<size_t>(P) * sizeof(LineInfo) + static_cast<size_t>(2 * P + 2) * sizeof(int32_t);
  return (lines + 15) / 16 * 16 + table;
}

// The order key of v: its bits with the magnitude bits flipped when the sign bit is set, as a
// signed integer (int32 or int64), held in a T. The integer order of the keys is the order of
// max_c/min_c on every non-NaN value (-0 below +0); the map is its own inverse.
__device__ __forceinline__ float order_key(float v) {
  const int32_t b = __float_as_int(v);
  return __int_as_float(b ^ ((b >> 31) & 0x7fffffff));
}
__device__ __forceinline__ double order_key(double v) {
  const long long b = __double_as_longlong(v);
  return __longlong_as_double(b ^ ((b >> 63) & 0x7fffffffffffffffLL));
}
template <bool Max>
__device__ __forceinline__ float key_op(float a, float b) {
  const int32_t x = __float_as_int(a), y = __float_as_int(b);
  return __int_as_float(Max ? max(x, y) : min(x, y));
}
template <bool Max>
__device__ __forceinline__ double key_op(double a, double b) {
  const long long x = __double_as_longlong(a), y = __double_as_longlong(b);
  return __longlong_as_double(Max ? (x > y ? x : y) : (x < y ? x : y));
}

template <class T, int OP>
__device__ __forceinline__ T op_apply(T a, T b) {
  if constexpr (OP == kOpMax) return max_c(a, b);
  else if constexpr (OP == kOpMin) return min_c(a, b);
  else if constexpr (OP == kOpMaxK) return key_op<true>(a, b);
  else if constexpr (OP == kOpMinK) return key_op<false>(a, b);
  else return add_rn(a, b);
}
// V holds order keys in the max/min stages (decoded on load by kOpMax/kOpMin), values otherwise.
template <int OP>
constexpr bool kDecodeLoad = OP == kOpMax || OP == kOpMin;
template <int OP>
constexpr bool kKeyOp = OP == kOpMaxK || OP == kOpMinK;

// Phase A for one warp task of direction DIR (1 = H as G of the reversed line, 0 = G): the folds of
// up to 32/ls segments of line `L`. DIR = 1 writes H into R at the output positions (the local
// folds first, then the carry applied in place); DIR = 0 walks each chunk twice (the chunk total,
// then the local folds again: the same operations in the same order, so the same bits), reads H
// at the output positions and replaces it by the stage output y = op(H(x-r), G(x+r)) (phase D).
template <class T, int OP, int DIR>
__device__ __forceinline__ void fold_task(const T* __restrict__ V, T* __restrict__ R, const LineInfo& L, int sb,
                                          int32_t n) {
  const int lane = threadIdx.x & 31;
  const int ls = L.ls;
  const int sub = lane / ls;
  const int qi = lane - sub * ls;
  const int32_t k = L.k_lo + sb * L.spw + sub;
  const bool seg_ok = sub < L.spw && k < L.k_lo + L.nsegs;
  const int32_t r = L.r, wa = L.wa;
  const int32_t s = L.o + k * wa;
  const int32_t e_last = min(s + wa - 1, n - 1 + r);
  const auto load = [&](int32_t e) {
    int32_t i = reflect_index(e, n);
    if (DIR) i = n - 1 - i;
    const T v = V[swz<T>(i)];
    if constexpr (kDecodeLoad<OP>) return order_key(v);
    else return v;
  };
  const auto slot = [&](int32_t e) {
    int32_t x = e - r;
    if (DIR) x = n - 1 - x;
    return swz<T>(x);
  };
  T carry = T(0);
  bool carry_ok = false;
  const int base = (sub * ls) & 31;
  // Segments of one chunk (w_a <= 32) carry nothing: G-direction tasks skip the chunk totals.
  const bool chain = ls > 1 || L.ng > 1;
  for (int g = 0; g < L.ng; ++g) {
    const int32_t q = g * 32 + qi;
    const int32_t c0 = s + 32 * q;
    const int32_t c1 = min(c0 + 31, e_last);
    const bool have = seg_ok && c0 <= c1;
    T acc = T(0);
    if (have && (DIR == 1 || chain)) {
      acc = load(c0);
      if (DIR == 1 && c0 >= r) R[slot(c0)] = acc;
#pragma unroll 4
      for (int32_t e = c0 + 1; e <= c1; ++e) {
        acc = op_apply<T, OP>(acc, load(e));
        if (DIR == 1 && e >= r) R[slot(e)] = acc;
      }
    }
    T my_carry = T(0);
    bool my_ok = false;
    T run = carry;
    bool run_ok = carry_ok;
    for (int i = 0; chain && i < ls; ++i) {
      const T ti = __shfl_sync(0xffffffffu, acc, (base + i) & 31);
      if (i == qi) {
        my_carry = run;
        my_ok = run_ok;
      }
      run = run_ok ? op_apply<T, OP>(run, ti) : ti;
      run_ok = true;
    }
    carry = run;
    carry_ok = true;
    if (!have) continue;
    if constexpr (DIR == 1) {
      if (my_ok) {
#pragma unroll 4
        for (int32_t e = max(c0, r); e <= c1; ++e) {
          const int32_t sl = slot(e);
          R[sl] = op_apply<T, OP>(my_carry, R[sl]);
        }
      }
    } else {
      // Sample e sits at segment position e - s; y is aligned (both window ends in one segment)
      // when x+r = e is the last sample of a whole segment.
      const int32_t e_al = s + wa - 1;
      T loc = T(0);
#pragma unroll 4
      for (int32_t e = c0; e <= c1; ++e) {
        const T x = load(e);
        loc = e == c0 ? x : op_apply<T, OP>(loc, x);
        if (e >= r) {
          const T gv = my_ok ? op_apply<T, OP>(my_carry, loc) : loc;
          const int32_t sl = slot(e);
          const T h = R[sl];
          if constexpr (OP == kOpSum) {
            const T sum = add_rn(h, gv);
            R[sl] = e == e_al ? mul_rn(sum, T(0.5)) : sum;
          } else if constexpr (kKeyOp<OP>) {
            R[sl] = order_key(op_apply<T, OP>(h, gv));  // the stage output holds values
          } else {
            R[sl] = op_apply<T, OP>(h, gv);
          }
        }
      }
    }
  }
}

// The CTA-strided walk idx = tid, tid + kThreads, ... over (hi, lo) = (idx / nb, idx % nb), with
// two divisions per thread instead of two per step.
struct Walk {
  int32_t idx, hi, lo, nb, dhi, dlo;
  __device__ __forceinline__ Walk(int32_t start, int32_t nb_) : idx(start), nb(nb_) {
    hi = start / nb;
    lo = start - hi * nb;
    dhi = kThreads / nb;
    dlo = kThreads - dhi * nb;
  }
  __device__ __forceinline__ void next() {
    idx += kThreads;
    hi += dhi;
    lo += dlo;
    if (lo >= nb) {
      lo -= nb;
      ++hi;
    }
  }
};

template <class T>
struct Shared {
  T* V;             // stage input
  T* R;             // H folds, then the stage output
  LineInfo* info;
  int32_t* prefix;  // warp tasks of one direction before line p (prefix[P]: all)
  int64_t ls;       // line stride (elements)
};

template <class T, int OP, int DIR>
__device__ void phase_a(const Shared<T>& sh, int P, int32_t n) {
  const int warp = threadIdx.x >> 5;
  const int total = sh.prefix[P];
  for (int t = warp; t < total; t += kWarps) {
    int p = 0;
    while (sh.prefix[p + 1] <= t) ++p;
    fold_task<T, OP, DIR>(sh.V + p * sh.ls, sh.R + p * sh.ls, sh.info[p], t - sh.prefix[p], n);
  }
}

// One stage `op` on every line: V -> R (two phases).
template <class T, int OP>
__device__ void run_stage_op(const Shared<T>& sh, int P, int32_t n) {
  phase_a<T, OP, 1>(sh, P, n);
  __syncthreads();
  phase_a<T, OP, 0>(sh, P, n);
}
template <class T>
__device__ void run_stage(int op, bool keys, const Shared<T>& sh, int P, int32_t n) {
  if (op == kOpMax) {
    if (keys) run_stage_op<T, kOpMaxK>(sh, P, n);
    else run_stage_op<T, kOpMax>(sh, P, n);
  } else if (op == kOpMin) {
    if (keys) run_stage_op<T, kOpMinK>(sh, P, n);
    else run_stage_op<T, kOpMin>(sh, P, n);
  } else {
    run_stage_op<T, kOpSum>(sh, P, n);
  }
}

// A launch runs gridDim.y passes (rungs); blockIdx.y is the rung of the CTA.
template <class T>
__global__ void __launch_bounds__(kThreads, 3) eda_vhgw_pass(Geom g, VhgwPass pass, VhgwArgs<T> a, int P, int64_t panels) {
  // The offset of this CTA's rung in an array of per-rung blocks `stride` elements apart (applied
  // where the pointer is used: offset pointers kept live through the kernel cost registers).
  const auto rung_at = [](int64_t stride) { return static_cast<int64_t>(blockIdx.y) * stride; };
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const LineView lv = line_view(g, pass.axis);
  const int32_t n = lv.n;
  const bool contiguous = pass.axis == g.dim - 1;
  const int tid = threadIdx.x;
  Shared<T> sh;
  sh.ls = line_stride_elems(n);
  T* lines = reinterpret_cast<T*>(smem_raw);
  sh.V = lines;
  sh.R = lines + P * sh.ls;
  const size_t line_bytes = (kLineBuffers * static_cast<size_t>(P) * sh.ls * sizeof(T) + 15) / 16 * 16;
  sh.info = reinterpret_cast<LineInfo*>(smem_raw + line_bytes);
  sh.prefix = reinterpret_cast<int32_t*>(sh.info + P);
  __shared__ int any_ok;

  // Prologue: the lines of this CTA.
  const int64_t cta = blockIdx.x;
  if (tid == 0) any_ok = 0;
  __syncthreads();
  if (tid < P) {
    LineInfo L{};
    bool valid;
    if (contiguous) {
      const int64_t q = cta * P + tid;
      valid = q < g.batch * g.rows;
      L.base = q * g.pitch;
      L.b = valid ? q / g.rows : 0;
    } else {
      const int64_t o = cta / panels;
      const int64_t j = (cta % panels) * P + tid;
      valid = j < lv.inner && (j % g.pitch) < g.n[g.dim - 1];
      L.base = o * lv.outer_stride + j;
      L.b = o / lv.per_element;
    }
    bool ok = valid;
    int32_t w = 1;
    if (ok) {
      w = a.window[rung_at(g.batch) + L.b];
      if (w < 1 || (w & 1) == 0) ok = false;  // never produced by the drivers
    }
    L.ok = ok ? 1 : 0;
    if (ok) {
      const int32_t half = (w - 1) / 2;
      L.r = min(half, n - 1);
      L.wa = 2 * L.r + 1;
      L.o = static_cast<int32_t>((static_cast<int64_t>(n) * ((L.wa + 1) / 2)) % L.wa);
      L.k_lo = static_cast<int32_t>(floor_div(static_cast<int64_t>(L.r) - L.o, L.wa));
      const int32_t k_hi = static_cast<int32_t>(floor_div(static_cast<int64_t>(n) - 1 + L.r - L.o, L.wa));
      L.nsegs = k_hi - L.k_lo + 1;
      const int32_t chunks = (L.wa + 31) / 32;
      L.ls = min(chunks, 32);
      L.spw = 32 / L.ls;
      L.ng = (chunks + 31) / 32;
      L.tasks = (L.nsegs + L.spw - 1) / L.spw;
      double prod = 1.0;
      for (int ax = 0; ax < g.dim; ++ax) prod *= static_cast<double>(2 * min(half, g.n[ax] - 1) + 1);
      const double c = pass.epilogue == kEpiBoxScale ? 1.0 : 0.5;
      L.fac = static_cast<double>(static_cast<T>(c / prod));
      atomicOr(&any_ok, 1);
    }
    sh.info[tid] = L;
  }
  __syncthreads();
  if (!any_ok) return;
  if (tid == 0) {
    int32_t acc = 0;
    for (int p = 0; p < P; ++p) {
      sh.prefix[p] = acc;
      acc += sh.info[p].ok ? sh.info[p].tasks : 0;
    }
    sh.prefix[P] = acc;
  }
  // (synchronized below, before phase A)

  const int nfields = pass.op >= 0 ? 1 : 2;
  const int epilogue = pass.epilogue;
  const int32_t total_pos = P * n;  // <= kThreads * kPer<T> (vhgw_lines_per_cta)
  const int32_t ls = static_cast<int32_t>(sh.ls);
  int ops[2];
  int nops = 0;
  if (pass.maxmin) ops[nops++] = kOpMax;  // field 1 (L): kOpMin, set below
  if (pass.box) ops[nops++] = kOpSum;
  bool v_has_src = false;  // V still holds the staged lines of `src` (a single-stage pass of one field h)
  bool keys = false;       // the max/min stages fold order keys (no NaN in the staged lines)

  for (int f = 0; f < nfields; ++f) {
    const T* src = (pass.from_single ? a.in0 : (f == 0 ? a.in0 : a.in1)) + rung_at(a.in_stride);
    // Load the lines into V (a single-stage pass of one field h leaves V intact: field 1 reuses it).
    // Before a max/min stage V holds the order keys of the values.
    if (!v_has_src) {
      const bool enc = pass.maxmin;
      int nan = 0;
      const auto put = [&](int32_t at, T v) {
        if (enc) {
          nan |= v != v;
          v = order_key(v);
        }
        sh.V[at] = v;
      };
      if (contiguous) {
        for (Walk w(tid, n); w.idx < total_pos; w.next()) {
          const int32_t p = w.hi;
          const int32_t x = w.lo;
          if (sh.info[p].ok) put(p * ls + swz<T>(x), src[sh.info[p].base + x]);
        }
      } else {
        for (Walk w(tid, P); w.idx < total_pos; w.next()) {
          const int32_t i = w.hi;
          const int32_t p = w.lo;
          if (sh.info[p].ok) put(p * ls + swz<T>(i), src[sh.info[p].base + static_cast<int64_t>(i) * lv.step]);
        }
      }
      keys = __syncthreads_or(nan) == 0;
    } else {
      __syncthreads();
    }
    v_has_src = pass.from_single && nops == 1;
    if (pass.maxmin) ops[0] = pass.op >= 0 ? (pass.op == 1 ? kOpMin : kOpMax) : (f == 0 ? kOpMax : kOpMin);
    for (int st = 0; st < nops; ++st) {
      const int op = ops[st];
      const bool last = st == nops - 1;
      run_stage<T>(op, keys, sh, P, n);
      __syncthreads();
      if (!last) {
        // The stage output becomes the input of the next stage.
        T* t = sh.V;
        sh.V = sh.R;
        sh.R = t;
        continue;
      }
      // Store the stage output.
      T* dst = ((f == 0 || pass.op >= 0) ? a.out0 : a.out1) + rung_at(a.out_stride);
      if (contiguous) {
        for (Walk w(tid, n); w.idx < total_pos; w.next()) {
          const int32_t p = w.hi;
          const int32_t x = w.lo;
          const LineInfo& L = sh.info[p];
          if (L.ok) {
            T y = sh.R[p * ls + swz<T>(x)];
            if (epilogue == kEpiBoxScale) y = mul_rn(y, static_cast<T>(L.fac));
            dst[L.base + x] = y;
          }
        }
        // Pad lanes of the output rows stay zero (the padded layout's invariant, geom.hpp).
        const int32_t pad = static_cast<int32_t>(g.pitch) - n;
        if (pad > 0) {
          for (int32_t idx = tid; idx < P * pad; idx += kThreads) {
            const int32_t p = idx / pad;
            const int32_t x = n + (idx - p * pad);
            if (sh.info[p].ok) dst[sh.info[p].base + x] = T(0);
          }
        }
      } else {
        for (Walk w(tid, P); w.idx < total_pos; w.next()) {
          const int32_t i = w.hi;
          const int32_t p = w.lo;
          const LineInfo& L = sh.info[p];
          if (L.ok) {
            T y = sh.R[p * ls + swz<T>(i)];
            if (epilogue == kEpiBoxScale) y = mul_rn(y, static_cast<T>(L.fac));
            dst[L.base + static_cast<int64_t>(i) * lv.step] = y;
          }
        }
      }
      __syncthreads();
    }
  }
}

// Per device, read once: the SMs, the L2 cache and the dynamic shared memory a CTA may use (the
// kernel's attribute is raised to it then).
struct DeviceFacts {
  int sms = 0;
  int64_t l2_bytes = 0;
  size_t smem = 0;
};
template <class T>
DeviceFacts device_facts() {
  static std::mutex mutex;
  static std::map<int, DeviceFacts> cached;
  int device = 0;
  EDA_CUDA(cudaGetDevice(&device));
  std::lock_guard<std::mutex> lock(mutex);
  const auto it = cached.find(device);
  if (it != cached.end()) return it->second;
  int optin = 0, sms = 0, l2 = 0;
  EDA_CUDA(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  EDA_CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  EDA_CUDA(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device));
  cudaFuncAttributes attr{};
  EDA_CUDA(cudaFuncGetAttributes(&attr, eda_vhgw_pass<T>));
  DeviceFacts f;
  f.sms = sms;
  f.l2_bytes = l2;
  f.smem = static_cast<size_t>(optin) - attr.sharedSizeBytes;
  EDA_CUDA(cudaFuncSetAttribute(eda_vhgw_pass<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(f.smem)));
  cached.emplace(device, f);
  return f;
}

// The line shapes for lines of n samples on the current device, computed once
// per (device, n): `fit`, the most lines a CTA holds (output positions and shared memory); `full`,
// the most lines that keep the SM at the occupancy of one-line CTAs; `occ`, that occupancy (CTAs
// per SM).
struct LineShape {
  int fit = 1, full = 1, occ = 1;
};
template <class T>
int ctas_per_sm(int32_t n, int P) {
  int blocks = 0;
  EDA_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, eda_vhgw_pass<T>, kThreads, smem_bytes<T>(n, P)));
  return blocks;
}
template <class T>
LineShape line_shape(int32_t n) {
  const size_t cap = device_facts<T>().smem;
  static std::mutex mutex;
  static std::map<std::pair<int, int32_t>, LineShape> cached;
  int device = 0;
  EDA_CUDA(cudaGetDevice(&device));
  std::lock_guard<std::mutex> lock(mutex);
  const auto it = cached.find({device, n});
  if (it != cached.end()) return it->second;
  if (smem_bytes<T>(n, 1) > cap) throw std::logic_error("eda: a vHGW line does not fit the shared memory");
  LineShape s;
  s.fit = std::max(1, std::min(kMaxLines, (kThreads * kPer<T>) / std::max(n, 1)));
  while (s.fit > 1 && smem_bytes<T>(n, s.fit) > cap) --s.fit;
  s.occ = std::max(1, ctas_per_sm<T>(n, 1));
  s.full = s.fit;
  while (s.full > 1 && ctas_per_sm<T>(n, s.full) < s.occ) --s.full;
  cached.emplace(std::make_pair(device, n), s);
  return s;
}

// The CTAs of one pass along `axis` with P lines per CTA.
int64_t pass_ctas(const Geom& g, int axis, int P) {
  if (axis == g.dim - 1) return (g.batch * g.rows + P - 1) / P;
  const LineView lv = line_view(g, axis);
  return lv.outer * ((lv.inner + P - 1) / P);
}

// The fewest lines per CTA of a launch of k passes: 1 on the contiguous axis; on an outer axis the
// adjacent columns of one 32-byte sector (a CTA reads and writes P adjacent columns per row, and
// fewer use a fraction of every sector moved), fewer only where a sector would leave SMs idle.
template <class T>
int min_lines(const Geom& g, int axis, const LineShape& s, int64_t sms, int64_t k) {
  if (axis == g.dim - 1) return 1;
  int p = std::min(s.fit, static_cast<int>(32 / sizeof(T)));
  while (p > 1 && k * pass_ctas(g, axis, p) < sms) --p;
  return p;
}

}  // namespace

template <class T>
int vhgw_lines_per_cta(const Geom& g, int axis, int32_t requested, int32_t rungs) {
  const int32_t n = g.n[axis];
  if (n > vhgw_max_line<T>()) {
    throw std::invalid_argument("eda: axis " + std::to_string(axis) + " has " + std::to_string(n) +
                                " samples; the vHGW pass supports at most " + std::to_string(vhgw_max_line<T>()) +
                                (sizeof(T) == 4 ? " (float32)" : " (float64)"));
  }
  const LineShape s = line_shape<T>(n);
  if (requested > 0) return std::min(s.fit, static_cast<int>(requested));
  // Automatic (the choice never changes a result: every sum depends only on the line, n and w): the
  // fewest lines per CTA that fit the launch into one wave of resident CTAs, so that every CTA
  // starts at once and each is as short as it can be; at most `full`, so a launch larger than a
  // wave runs in waves of the fullest CTAs that keep the occupancy; at least min_lines.
  const int64_t sms = device_facts<T>().sms;
  const int64_t wave = sms * s.occ;
  const int64_t k = std::max<int32_t>(rungs, 1);
  int p = 1;
  while (p < s.full && k * pass_ctas(g, axis, p) > wave) ++p;
  return std::max(p, min_lines<T>(g, axis, s, sms, k));
}

template <class T>
int32_t vhgw_rung_group(const Geom& g, int32_t max_group) {
  // One rung fills the device when every axis's pass has a wave of CTAs at the fewest lines per CTA;
  // a group then only makes the CTAs longer.
  const DeviceFacts f = device_facts<T>();
  bool fills = true;
  for (int a = 0; a < g.dim; ++a) {
    const LineShape s = line_shape<T>(g.n[a]);
    fills = fills && pass_ctas(g, a, min_lines<T>(g, a, s, f.sms, 1)) >= static_cast<int64_t>(f.sms) * s.occ;
  }
  if (fills) return 1;
  const int64_t field_bytes = g.batch * g.field * static_cast<int64_t>(sizeof(T));
  const int64_t by_l2 = f.l2_bytes / (4 * std::max<int64_t>(field_bytes, 1));
  return static_cast<int32_t>(std::clamp<int64_t>(by_l2, 1, std::max(max_group, 1)));
}

template <class T>
void enqueue_vhgw(cudaStream_t s, const Geom& g, const VhgwPass& pass, const VhgwArgs<T>& a) {
  const int32_t rungs = std::max<int32_t>(a.rungs, 1);
  if (rungs > 65535) throw std::invalid_argument("eda: too many rungs for one vHGW launch");
  const int P = vhgw_lines_per_cta<T>(g, pass.axis, a.lines_per_cta, rungs);
  const LineView lv = line_view(g, pass.axis);
  const int64_t panels = pass.axis == g.dim - 1 ? 1 : (lv.inner + P - 1) / P;
  const int64_t ctas = pass_ctas(g, pass.axis, P);
  if (ctas > INT32_MAX) throw std::invalid_argument("eda: too many lines for one vHGW pass");
  const dim3 grid(static_cast<unsigned>(ctas), static_cast<unsigned>(rungs));
  eda_vhgw_pass<T><<<grid, kThreads, smem_bytes<T>(lv.n, P), s>>>(g, pass, a, P, panels);
  EDA_CHECK_LAUNCH("eda_vhgw_pass");
}

template int vhgw_lines_per_cta<float>(const Geom&, int, int32_t, int32_t);
template int vhgw_lines_per_cta<double>(const Geom&, int, int32_t, int32_t);
template int32_t vhgw_rung_group<float>(const Geom&, int32_t);
template int32_t vhgw_rung_group<double>(const Geom&, int32_t);
template void enqueue_vhgw<float>(cudaStream_t, const Geom&, const VhgwPass&, const VhgwArgs<float>&);
template void enqueue_vhgw<double>(cudaStream_t, const Geom&, const VhgwPass&, const VhgwArgs<double>&);

}  // namespace eda
