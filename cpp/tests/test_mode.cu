// SPDX-License-Identifier: Apache-2.0
// test_mode: the mode's control and sift kernels (method/mode.hpp, driver.cpp) against host twins
// written from their definitions (the window rule of the extrema counts, G2's rung loop, the
// window with G2's result, G3, the continuation weight, G1's pointwise parts, the sift with the
// guard, the commit), bitwise, float32 and float64, plus the validation of the method, the
// magnitude prescale with the unscale and the non-finite guard. The end-to-end equivalences (batch
// position, chunking, repeated runs, a new thread, a NaN element, a constant element, max_imf)
// are the separate executable test_e2e (they need the element and the statistics too).
#include "twin.hpp"
#include "field/state.hpp"
#include "decompose.hpp"
#include "method/method.hpp"
#include "method/element.hpp"
#include "method/mode.hpp"
#include "test_support.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <vector>

using host_twin::Rng;

namespace {

const bool kSmall = std::getenv("EDA_SANITIZE") && std::string(std::getenv("EDA_SANITIZE")) == "1";

template <class V>
struct Dev {
  V* p = nullptr;
  size_t n = 0;
  explicit Dev(size_t count) : n(count) {
    if (cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(V)) != cudaSuccess) std::abort();
    // zeroed: get() copies whole padded buffers, pads included, which the kernels never write
    // (initcheck reports the copy of never-written bytes as a host API error otherwise)
    if (cudaMemset(p, 0, std::max<size_t>(count, 1) * sizeof(V)) != cudaSuccess) std::abort();
  }
  Dev(const std::vector<V>& h) : Dev(h.size()) { put(h); }
  ~Dev() { cudaFree(p); }
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
  void put(const std::vector<V>& h) { cudaMemcpy(p, h.data(), h.size() * sizeof(V), cudaMemcpyHostToDevice); }
  std::vector<V> get() const {
    std::vector<V> h(n);
    cudaMemcpy(h.data(), p, n * sizeof(V), cudaMemcpyDeviceToHost);
    return h;
  }
};

bool same(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0 || (std::isnan(a) && std::isnan(b)); }
template <class T>
bool same_t(T a, T b) { return std::memcmp(&a, &b, sizeof a) == 0; }

int32_t odd(double v) { return 2 * static_cast<int32_t>(std::floor(std::max(v, 1.0) / 2.0)) + 1; }

// The per-element arrays of the driver on the device (test-owned).
struct Arrays {
  int64_t B;
  Dev<eda::ElemRecord> le;
  Dev<eda::ElemState> elems;
  Dev<double> dbl;
  Dev<int32_t> ints;
  explicit Arrays(int64_t b) : B(b), le(b), elems(b), dbl(6 * b), ints((3 + eda::kWinSlots) * b) {
    cudaMemset(le.p, 0, sizeof(eda::ElemRecord) * b);
    cudaMemset(elems.p, 0, sizeof(eda::ElemState) * b);
    cudaMemset(dbl.p, 0, sizeof(double) * 6 * b);
    cudaMemset(ints.p, 0, sizeof(int32_t) * (3 + eda::kWinSlots) * b);
  }
  eda::ModeArrays view() const {
    eda::ModeArrays A{};
    A.le = le.p;
    A.elems = elems.p;
    A.tol = dbl.p;
    A.sigma_n = dbl.p + B;
    A.sigma_x = dbl.p + 2 * B;
    A.floor2 = dbl.p + 3 * B;
    A.wc = dbl.p + 4 * B;
    A.e_med = dbl.p + 5 * B;
    A.on = ints.p;
    A.on_g2 = ints.p + B;
    A.on_floor = ints.p + 2 * B;
    A.win = ints.p + 3 * B;
    return A;
  }
  std::vector<double> dbl_h() const { return dbl.get(); }
  std::vector<int32_t> ints_h() const { return ints.get(); }
};

// ---------------------------------------------------------------------------------------------
// mode_init and mode_start

void test_start() {
  const int64_t B = 7;
  int64_t n[3] = {64, 48, 1};
  const eda::Geom g = eda::make_geom(2, B, n);
  const eda::MethodCtl c = eda::method_ctl(g, eda::DType::F64);
  Arrays A(B);
  std::vector<eda::ElemState> ne(B);
  std::memset(ne.data(), 0, sizeof(eda::ElemState) * B);
  for (int64_t b = 0; b < B; ++b) {
    ne[b].status = b == 2 ? eda::kError : eda::kActive;
    ne[b].error = b == 2 ? eda::kErrNonFinite : 0;
    ne[b].amax = 0.5 + b;
    ne[b].tol = std::ldexp(0.5 + b, -30);
  }
  A.elems.put(ne);
  std::vector<double> dbl(6 * B, 0.0);
  for (int64_t b = 0; b < B; ++b) dbl[2 * B + b] = 0.01 * (b + 1);  // sigma_x
  A.dbl.put(dbl);
  eda::enqueue_mode_init(nullptr, B, A.view());
  // the state of later modes: element 3 at mode 2 (w_prev 23), element 4 done, element 5 hit
  // max_imf (element state kDone), element 6 at mode 1 (w_prev 7)
  std::vector<eda::ElemRecord> le = A.le.get();
  CHECK(le[2].status == eda::kRecordError);
  CHECK(le[0].status == eda::kRecordActive && le[0].w_prev == 1 && le[0].k == 0 && std::isinf(le[0].wc));
  le[3].k = 2;
  le[3].w_prev = 23;
  le[4].status = eda::kRecordDone;
  le[6].k = 1;
  le[6].w_prev = 7;
  le[1].ext[0] = 99;
  le[1].sifting = 0;
  le[1].gmax_h_bits = 5;
  A.le.put(le);
  ne[5].status = eda::kDone;
  A.elems.put(ne);
  eda::enqueue_mode_start(nullptr, B, c, A.view());
  le = A.le.get();
  const std::vector<double> d = A.dbl_h();
  const std::vector<int32_t> in = A.ints_h();
  for (int64_t b = 0; b < B; ++b) {
    const bool run = b != 2 && b != 4 && b != 5;
    CHECK(in[b] == (run ? 1 : 0));
    CHECK(in[B + b] == 0);
    if (!run) {
      for (int k = 0; k < eda::kWinSlots; ++k) CHECK(in[3 * B + k * B + b] == 0);
      continue;
    }
    const eda::ElemRecord& e = le[b];
    CHECK(e.status == eda::kRecordActive);
    CHECK(same(e.tol, ne[b].tol) && same(d[b], ne[b].tol) && same(e.r_inf, ne[b].amax));
    CHECK(e.ext[0] == 0 && e.sifting == 1 && e.n_sifts == 0 && e.gmax_h_bits == 0 && e.fine == 0);
    CHECK(same(e.jump, 2.0) && same(e.g3, 1.0) && std::isinf(e.wc) && std::isinf(d[4 * B + b]));
    CHECK(same(e.sigma_x, dbl[2 * B + b]) && same(d[3 * B + b], 1.0 * dbl[2 * B + b]));
    const int32_t w0 = std::max(3, e.w_prev + 2);
    CHECK(in[3 * B + eda::kWinWd * B + b] == odd(2.0 * w0));
    CHECK(in[3 * B + eda::kWinWl * B + b] == odd(2.0 * w0));
    CHECK(in[3 * B + eda::kWinW * B + b] == 0);
    CHECK(in[2 * B + b] == (e.k >= 1 ? 1 : 0));
  }
  CHECK(le[4].status == eda::kRecordDone && le[5].status == eda::kRecordDone && le[2].status == eda::kRecordError);
}

// ---------------------------------------------------------------------------------------------
// The window passes against a host twin of the window rule

struct RiceTwin {
  bool stop;
  int32_t w;
  double spacing;
};
RiceTwin rice_twin(const uint64_t* ext, const eda::Geom& g, double kappa, int32_t prev) {
  const uint64_t N = static_cast<uint64_t>(g.count);
  bool enough = false;
  uint64_t s2 = 0;
  int32_t nmax = 1;
  for (int a = 0; a < g.dim; ++a) {
    enough = enough || ext[a] * static_cast<uint64_t>(g.n[a]) >= 3 * N;
    s2 += ext[a] * ext[a];
    nmax = std::max(nmax, g.n[a]);
  }
  if (!enough) return {true, 0, 0.0};
  const double s = static_cast<double>(N) / std::sqrt(static_cast<double>(s2));
  const double v = (2.0 * kappa) * s;
  if (!(v < 1073741824.0)) return {true, 0, 0.0};
  int32_t w = 2 * static_cast<int32_t>(std::floor(std::max(v, 1.0) / 2.0)) + 1;
  w = std::max({w, 3, prev + 2});
  if (w > nmax) return {true, 0, 0.0};
  return {false, w, s};
}

void test_rice_case(int dim, int64_t n0, int64_t n1, int64_t n2, uint64_t seed) {
  const int64_t B = 24;
  int64_t n[3] = {n0, n1, n2};
  const eda::Geom g = eda::make_geom(dim, B, n);
  const eda::MethodCtl c = eda::method_ctl(g, eda::DType::F32);
  Rng rng(seed);
  Arrays A(B);
  std::vector<eda::ElemState> ne(B);
  std::memset(ne.data(), 0, sizeof(eda::ElemState) * B);
  for (auto& e : ne) e.status = eda::kActive;
  A.elems.put(ne);
  eda::enqueue_mode_init(nullptr, B, A.view());
  std::vector<eda::ElemRecord> le = A.le.get();
  for (int64_t b = 0; b < B; ++b) {
    le[b].w_prev = b % 4 == 0 ? 1 : 2 * static_cast<int32_t>(rng.next() % 40) + 1;
    le[b].k = static_cast<int32_t>(b % 3);
  }
  A.le.put(le);
  std::vector<double> dbl(6 * B, 0.0);
  for (int64_t b = 0; b < B; ++b) dbl[B + b] = 0.125 * b;  // sigma_n
  A.dbl.put(dbl);
  eda::enqueue_mode_start(nullptr, B, c, A.view());
  std::vector<int32_t> prev(B);
  for (int64_t b = 0; b < B; ++b) prev[b] = le[b].w_prev;
  std::vector<bool> alive(B, true);
  for (int pass = 0; pass < 2; ++pass) {
    le = A.le.get();
    std::vector<std::vector<uint64_t>> ext(B, std::vector<uint64_t>(3, 0));
    for (int64_t b = 0; b < B; ++b) {
      for (int a = 0; a < dim; ++a) {
        const uint64_t lines = static_cast<uint64_t>(g.count / g.n[a]);
        // counts from far too few to dense; element 1: none; element 5: the 3-per-line edge
        uint64_t e = rng.next() % (lines * static_cast<uint64_t>(g.n[a]) / 2 + 1);
        if (b == 1) e = 0;
        if (b == 5) e = a == 0 ? 3 * lines : 0;
        if (b == 6) e = a == 0 ? 3 * lines - 1 : 0;
        ext[b][a] = e;
        le[b].ext[a] = e;
      }
    }
    A.le.put(le);
    eda::enqueue_mode_rice(nullptr, B, c, g, pass, A.view());
    const std::vector<eda::ElemRecord> out = A.le.get();
    const std::vector<int32_t> in = A.ints_h();
    const std::vector<eda::ElemState> ne2 = A.elems.get();
    for (int64_t b = 0; b < B; ++b) {
      if (!alive[b]) {
        CHECK(out[b].status == eda::kRecordDone);
        continue;
      }
      const RiceTwin t = rice_twin(ext[b].data(), g, 1.0, prev[b]);
      CHECK(out[b].ext[0] == 0 && out[b].ext[1] == 0 && out[b].ext[2] == 0);
      if (pass == 0) CHECK(same(out[b].sigma_n, dbl[B + b]));
      if (t.stop) {
        CHECK(out[b].status == eda::kRecordDone && ne2[b].status == eda::kDone && in[b] == 0);
        CHECK(out[b].w_pass[pass] == 0);
        for (int k = 0; k < eda::kWinSlots; ++k) CHECK(in[3 * B + k * B + b] == 0);
        alive[b] = false;
        continue;
      }
      CHECK(out[b].status == eda::kRecordActive && in[b] == 1);
      CHECK(out[b].w_pass[pass] == t.w && same(out[b].spacing, t.spacing));
      CHECK(in[3 * B + eda::kWinWd * B + b] == odd(2.0 * t.w));
      CHECK(in[3 * B + eda::kWinWl * B + b] == odd(2.0 * t.w));
      if (pass == 1) {
        CHECK(out[b].w == t.w);
        const bool elig = 2.0 * std::max(3, prev[b] + 2) <= t.w;
        CHECK(in[B + b] == (elig ? 1 : 0));
      }
    }
  }
}


void test_rice() {
  test_rice_case(1, 512, 1, 1, 11);
  test_rice_case(2, 64, 48, 1, 12);
  test_rice_case(2, 37, 5, 1, 13);
  test_rice_case(3, 20, 16, 12, 14);
  test_rice_case(3, 1, 9, 30, 15);  // an axis of extent 1
}

// ---------------------------------------------------------------------------------------------
// mode_window: G2's window, the growth floor and its stop, W, the G3 factor

void test_window() {
  const int64_t B = 64;
  int64_t n[3] = {300, 1, 1};
  const eda::Geom g = eda::make_geom(1, B, n);
  {
    const eda::MethodCtl c = eda::method_ctl(g, eda::DType::F64);
    Rng rng(78);
    Arrays A(B);
    std::vector<eda::ElemState> ne(B);
    std::memset(ne.data(), 0, sizeof(eda::ElemState) * B);
    for (auto& e : ne) e.status = eda::kActive;
    A.elems.put(ne);
    eda::enqueue_mode_init(nullptr, B, A.view());
    std::vector<eda::ElemRecord> le = A.le.get();
    std::vector<double> dbl(6 * B, 0.0);
    for (int64_t b = 0; b < B; ++b) {
      eda::ElemRecord& e = le[b];
      e.k = static_cast<int32_t>(b % 4);
      e.w_prev = e.k == 0 ? 1 : 2 * static_cast<int32_t>(rng.next() % 80) + 1;
      e.w = std::max(3, e.w_prev + 2) + 2 * static_cast<int32_t>(rng.next() % 60);
      e.fine = (b % 3) == 0;
      e.p_odd = 2 * static_cast<int32_t>(rng.next() % 30) + 3;
      e.jump = 2.0;
      // wc: finite (the G3 factor below, at and above 1), +inf; a typical case p_odd = 7, wc = 19.744
      double wc = 1.0 + rng.uniform() * 400.0;
      if (b % 5 == 0) wc = std::numeric_limits<double>::infinity();
      if (b == 3) {
        e.p_odd = 7;
        wc = 19.744;
        e.fine = 1;
        e.k = 0;
        e.w_prev = 1;
        e.w = 19;
      }
      dbl[4 * B + b] = wc;
    }
    A.le.put(le);
    A.dbl.put(dbl);
    eda::enqueue_mode_window(nullptr, B, c, A.view());
    const std::vector<eda::ElemRecord> out = A.le.get();
    const std::vector<int32_t> in = A.ints_h();
    for (int64_t b = 0; b < B; ++b) {
      const eda::ElemRecord& e = le[b];
      int32_t w = e.w;
      double jump = e.jump;
      if (e.fine) {
        w = std::max({e.p_odd, e.w_prev + 2, 3});
        jump = 1.0;
      }
      bool stop = false;
      if (e.k >= 2) {
        w = std::max(w, odd(2.0 * e.w_prev));
        stop = w > 300;
      }
      if (stop) {
        CHECK(out[b].status == eda::kRecordDone && in[b] == 0 && in[3 * B + eda::kWinW * B + b] == 0);
        continue;
      }
      CHECK(out[b].status == eda::kRecordActive);
      CHECK(out[b].w == w && out[b].W == odd(2.0 * w) && in[3 * B + eda::kWinW * B + b] == odd(2.0 * w));
      CHECK(same(out[b].jump, jump));
      double g3 = 1.0;
      if (e.fine) {
        const double wc = dbl[4 * B + b];
        g3 = std::isfinite(wc) ? std::min(1.0, 1.0 * std::sqrt(wc / e.p_odd) / 2.0) : 1.0;
        CHECK(same(out[b].wc, wc));
      }
      CHECK(same(out[b].g3, g3));
      CHECK(out[b].d_min_bits == 0x7ff0000000000000ULL && out[b].d_max_bits == 0);
      if (b == 3) CHECK(std::fabs(out[b].g3 - 0.8398) < 1e-3);  // sqrt(19.744 / 7) / 2 = 0.83973
    }
  }
}

// ---------------------------------------------------------------------------------------------
// G2: the rung list, the reused-test machine against a direct loop over the rungs, g2_accept

void test_g2_rungs() {
  struct Case {
    int32_t w, wp;
    std::vector<int32_t> r;
    int32_t confirm;
  };
  const std::vector<Case> cases = {
      {49, 1, {3, 5, 7, 9, 13, 17}, 25},
      {6, 1, {3}, 5},
      {257, 17, {19, 27, 39, 53, 77, 107}, 153},
      {8191, 1, {3, 5, 7, 9, 13, 17, 25, 33, 49, 67, 97, 135, 193, 271, 385, 543, 769, 1087, 1537, 2173, 3073}, 4347},
      {683, 215, {217, 307}, 435},
      {45, 13, {15, 21}, 31},
      {4095, 127, {129, 183, 259, 365, 517, 729, 1033, 1459}, 2065},
  };
  for (const Case& cs : cases) {
    const std::vector<int32_t> r = eda::g2_candidate_rungs(cs.w, cs.wp, 2.0);
    CHECK(r == cs.r);
    eda::RungMachine m = eda::make_rung_machine(cs.w, cs.wp, 2.0);
    // accept the last rung: the machine then asks for its confirmation rung
    for (size_t i = 0; i + 1 < r.size(); ++i) {
      eda::G2Result none{};
      none.outcome = eda::kG2None;
      m.feed(none);
    }
    eda::G2Result acc{};
    acc.outcome = eda::kG2Accepted;
    acc.p = 4.5;
    m.feed(acc);
    CHECK(!m.done && m.next_rung() == cs.confirm);
  }
  CHECK(eda::g2_candidate_rungs(5, 1, 2.0).empty());
}

// The direct loop over the candidate rungs (a candidate accepted only when the next rung confirms
// it, each rung's test with the population's own core) over an outcome table.
struct DirectLoop {
  bool accepted = false;
  int32_t s = 0;
  double p = 0.0;
};
DirectLoop direct_loop(const std::vector<int32_t>& rungs, const std::map<int32_t, eda::G2Result>& table) {
  for (size_t i = 0; i < rungs.size(); ++i) {
    const eda::G2Result t = table.at(rungs[i]);
    if (t.outcome == eda::kG2Stop) return {};
    if (t.outcome == eda::kG2None) continue;
    const int32_t s2 = i + 1 < rungs.size() ? rungs[i + 1] : odd(rungs[i] * eda::kRungRatio + 1.0);
    const eda::G2Result t2 = table.at(s2);
    if (t2.outcome != eda::kG2Accepted) continue;
    return {true, rungs[i], t.p};
  }
  return {};
}

void test_g2_machine() {
  Rng rng(4242);
  const int trials = kSmall ? 200 : 20000;
  int accepted = 0;
  for (int it = 0; it < trials; ++it) {
    const int32_t wp = it % 3 == 0 ? 1 : 2 * static_cast<int32_t>(rng.next() % 200) + 1;
    const int32_t s0 = std::max(3, wp + 2);
    const int32_t w = 2 * s0 + static_cast<int32_t>(rng.next() % (it % 7 == 0 ? 8000 : 400));
    const std::vector<int32_t> rungs = eda::g2_candidate_rungs(w, wp, 2.0);
    if (rungs.empty()) continue;
    const double pa = rng.uniform(), ps = rng.uniform() * 0.3;
    std::map<int32_t, eda::G2Result> table;
    std::vector<int32_t> all = rungs;
    all.push_back(odd(rungs.back() * eda::kRungRatio + 1.0));
    for (int32_t s : all) {
      eda::G2Result o{};
      const double u = rng.uniform();
      o.outcome = u < ps ? eda::kG2Stop : (u < ps + pa * (1 - ps) ? eda::kG2Accepted : eda::kG2None);
      o.p = s * (0.5 + rng.uniform());
      o.cov = rng.uniform();
      o.s = s;
      table[s] = o;
    }
    const DirectLoop ref = direct_loop(rungs, table);
    eda::RungMachine m = eda::make_rung_machine(w, wp, 2.0);
    std::set<int32_t> tested;
    bool once = true;
    bool lockstep = true;
    while (!m.done) {
      const int32_t s = m.next_rung();
      lockstep = lockstep && m.i == m.tests;  // the step index is the rung index
      if (!tested.insert(s).second) once = false;
      m.feed(table.at(s));
    }
    CHECK(once && lockstep);
    CHECK(m.tests <= static_cast<int32_t>(rungs.size()) + 1);
    CHECK(m.accepted == ref.accepted);
    if (m.accepted && ref.accepted) {
      ++accepted;
      CHECK(rungs[static_cast<size_t>(m.acc_index)] == ref.s && m.cand.s == ref.s && same(m.cand.p, ref.p));
    }
  }
  CHECK(accepted > 0);
}

void test_g2_accept() {
  const int64_t B = 4;
  int64_t n[3] = {9, 40, 1};
  const eda::Geom g = eda::make_geom(2, B, n);
  const size_t field = static_cast<size_t>(B * g.field);
  Rng rng(5);
  std::vector<uint8_t> p0(field), p1(field);
  for (size_t i = 0; i < field; ++i) {
    p0[i] = static_cast<uint8_t>(rng.next() & 1);
    p1[i] = static_cast<uint8_t>(2 + (rng.next() & 1));
  }
  Dev<uint8_t> d0(p0), d1(p1), dacc(std::vector<uint8_t>(field, 0xab));
  Dev<eda::ElemRecord> le(B);
  cudaMemset(le.p, 0, sizeof(eda::ElemRecord) * B);
  std::vector<eda::G2Result> res(B);
  std::memset(res.data(), 0, sizeof(eda::G2Result) * B);
  const int32_t outc[4] = {eda::kG2Accepted, eda::kG2None, eda::kG2Accepted, eda::kG2Stop};
  for (int64_t b = 0; b < B; ++b) {
    res[b].outcome = outc[b];
    res[b].s = 5 + 2 * static_cast<int32_t>(b);
    res[b].p = 6.25 + b;
    res[b].cov = 0.125 * b;
  }
  Dev<eda::G2Result> dres(res);
  Dev<int32_t> slot(std::vector<int32_t>{0, 1, 1, 0});
  eda::enqueue_mode_g2_accept(nullptr, g, le.p, dres.p, slot.p, d0.p, d1.p, dacc.p);
  const std::vector<uint8_t> acc = dacc.get();
  const std::vector<eda::ElemRecord> out = le.get();
  for (int64_t b = 0; b < B; ++b) {
    const bool a = outc[b] == eda::kG2Accepted;
    bool ok = true;
    for (int64_t v = 0; v < g.field; ++v) {
      const size_t i = static_cast<size_t>(b * g.field + v);
      const uint8_t want = a ? (b == 2 ? p1[i] : p0[i]) : 0xab;
      ok = ok && acc[i] == want;
    }
    CHECK(ok);
    CHECK(out[b].fine == (a ? 1 : 0));
    if (a) {
      CHECK(out[b].g2_rung == res[b].s && same(out[b].p, res[b].p) && same(out[b].cov, res[b].cov));
      CHECK(out[b].p_odd == odd(res[b].p));
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Field kernels: G3's scaling, the continuation weight, G1's parts

// A padded field of B elements with random values in [lo, hi) and NaN pads (never touched).
template <class T>
std::vector<T> rand_field(const eda::Geom& g, Rng& rng, double lo, double hi) {
  std::vector<T> f(static_cast<size_t>(g.batch * g.field), std::numeric_limits<T>::quiet_NaN());
  const int32_t nl = g.n[g.dim - 1];
  for (int64_t i = 0; i < g.batch * g.field; ++i) {
    if ((i % g.field) % g.pitch < nl) f[i] = static_cast<T>(lo + (hi - lo) * rng.uniform());
  }
  return f;
}
bool is_pad(const eda::Geom& g, int64_t i) { return (i % g.field) % g.pitch >= g.n[g.dim - 1]; }

template <class T, class F>
bool check_field(const eda::Geom& g, const std::vector<T>& got, const std::vector<T>& before,
                 const std::vector<int32_t>& on, F want) {
  bool ok = true;
  int bad = 0;
  for (int64_t i = 0; i < g.batch * g.field; ++i) {
    const int64_t b = i / g.field;
    T exp = (is_pad(g, i) || !on[b]) ? before[i] : want(i);
    if (!same_t(got[i], exp) && !(std::isnan(got[i]) && std::isnan(exp))) {
      if (bad++ < 3) std::fprintf(stderr, "  field mismatch at %lld: %.17g vs %.17g\n", static_cast<long long>(i),
                                  static_cast<double>(got[i]), static_cast<double>(exp));
      ok = false;
    }
  }
  return ok;
}

template <class T>
T npmax(T a, T b) { return a >= b ? a : b; }
template <class T>
T npmin(T a, T b) { return a <= b ? a : b; }
template <class T>
T npclip(T v, T lo, T hi) { return npmin(npmax(v, lo), hi); }

template <class T>
void test_fields_t(int dim) {
  const int64_t B = 3;
  int64_t n[3] = {dim == 1 ? 300 : 13, dim >= 2 ? 37 : 1, dim == 3 ? 6 : 1};
  const eda::Geom g = eda::make_geom(dim, B, n);
  const size_t field = static_cast<size_t>(B * g.field);
  const eda::DType dt = sizeof(T) == 4 ? eda::DType::F32 : eda::DType::F64;
  const eda::MethodCtl c = eda::method_ctl(g, dt);
  Rng rng(900 + dim + sizeof(T));
  const std::vector<int32_t> onv = {1, 0, 1};
  Dev<int32_t> on(onv);
  // le: tol per element (the second element does not run)
  std::vector<eda::ElemRecord> leh(B);
  std::memset(leh.data(), 0, sizeof(eda::ElemRecord) * B);
  const double tols[3] = {0.25, 0.0, std::ldexp(1.0, -14)};
  for (int64_t b = 0; b < B; ++b) {
    leh[b].status = onv[b] ? eda::kRecordActive : eda::kRecordDone;
    leh[b].tol = tols[b];
    leh[b].fine = b != 1;
    leh[b].g3 = b == 0 ? 0.8398 : 1.0;
  }
  Dev<eda::ElemRecord> le(leh);

  // G3's scaling: L = max(g3 L, 3)
  {
    const std::vector<T> L0 = rand_field<T>(g, rng, 1.0, 20.0);
    Dev<T> L(L0);
    eda::enqueue_mode_g3<T>(nullptr, g, le.p, L.p, on.p);
    CHECK(check_field<T>(g, L.get(), L0, onv, [&](int64_t i) {
      const int64_t b = i / g.field;
      const T v = static_cast<T>(static_cast<T>(leh[b].g3) * L0[i]);
      return v >= T(3) ? v : T(3);
    }));
  }
  // range, ramp, clip01
  const std::vector<T> u0 = rand_field<T>(g, rng, -1.0, 2.0);
  const std::vector<T> l0 = rand_field<T>(g, rng, -2.0, 1.0);
  Dev<T> u(u0), l(l0), R(rand_field<T>(g, rng, 0, 1)), U(field), gw(rand_field<T>(g, rng, 0, 1));
  const std::vector<T> Rb = R.get();
  eda::enqueue_mode_range<T>(nullptr, g, u.p, l.p, R.p, on.p);
  const std::vector<T> Rv = R.get();
  CHECK(check_field<T>(g, Rv, Rb, onv, [&](int64_t i) { return npmax(static_cast<T>(u0[i] - l0[i]), T(0)); }));
  std::vector<T> U0 = rand_field<T>(g, rng, 0.0, 3.0);
  for (size_t i = 0; i < field; i += 7) U0[i] = is_pad(g, static_cast<int64_t>(i)) ? U0[i] : T(0.25);  // == tol
  for (size_t i = 3; i < field; i += 11) U0[i] = is_pad(g, static_cast<int64_t>(i)) ? U0[i] : T(0);
  U.put(U0);
  const std::vector<T> gb = gw.get();
  eda::enqueue_mode_ramp<T>(nullptr, g, c, le.p, R.p, U.p, gw.p, on.p);
  const T t0 = static_cast<T>(0.2);
  const T den = static_cast<T>(0.5 - 0.2);
  CHECK(check_field<T>(g, gw.get(), gb, onv, [&](int64_t i) {
    const int64_t b = i / g.field;
    T rho = T(0);
    if (static_cast<double>(U0[i]) > leh[b].tol) rho = npclip(static_cast<T>(Rv[i] / U0[i]), T(0), T(1));
    return npclip(static_cast<T>(static_cast<T>(rho - t0) / den), T(0), T(1));
  }));
  {
    const std::vector<T> a0 = rand_field<T>(g, rng, -0.5, 1.5);
    Dev<T> a(a0);
    eda::enqueue_mode_clip01<T>(nullptr, g, a.p, on.p);
    CHECK(check_field<T>(g, a.get(), a0, onv, [&](int64_t i) { return npclip(a0[i], T(0), T(1)); }));
  }
  // G1: half_diff, floor, ratio, kappa_ramp, blend
  {
    Dev<T> hd(rand_field<T>(g, rng, 0, 1));
    const std::vector<T> hb = hd.get();
    eda::enqueue_mode_half_diff<T>(nullptr, g, u.p, l.p, hd.p, on.p);
    CHECK(check_field<T>(g, hd.get(), hb, onv,
                         [&](int64_t i) { return static_cast<T>(T(0.5) * static_cast<T>(u0[i] - l0[i])); }));
  }
  {
    std::vector<T> e0 = rand_field<T>(g, rng, 0.0, 1.0);
    for (size_t i = 0; i < field; i += 5) e0[i] = is_pad(g, static_cast<int64_t>(i)) ? e0[i] : T(0);
    Dev<T> e(e0);
    const std::vector<double> med = {0.4, 7.0, 1e-3};
    Dev<double> dmed(med);
    eda::enqueue_mode_floor<T>(nullptr, g, c, le.p, dmed.p, e.p, on.p);
    CHECK(check_field<T>(g, e.get(), e0, onv, [&](int64_t i) {
      const int64_t b = i / g.field;
      const double thr = std::max({0.1 * med[b], leh[b].tol, c.tiny});
      return static_cast<double>(e0[i]) >= thr ? e0[i] : static_cast<T>(thr);
    }));
    const std::vector<T> ef = e.get();
    const std::vector<T> h0 = rand_field<T>(g, rng, -1.0, 1.0);
    Dev<T> h(h0), q(rand_field<T>(g, rng, 0, 1));
    const std::vector<T> qb = q.get();
    eda::enqueue_mode_ratio<T>(nullptr, g, h.p, e.p, q.p, on.p);
    CHECK(check_field<T>(g, q.get(), qb, onv, [&](int64_t i) { return static_cast<T>(h0[i] / ef[i]); }));
  }
  {
    std::vector<T> emax = rand_field<T>(g, rng, 0.0, 2.0), emin = rand_field<T>(g, rng, 0.0, 1.0);
    for (size_t i = 0; i < field; i += 9) {
      if (!is_pad(g, static_cast<int64_t>(i))) emax[i] = emin[i] = T(0);  // the tol floor
    }
    Dev<T> dmax(emax), dmin(emin), out(rand_field<T>(g, rng, 0, 1));
    const std::vector<T> ob = out.get();
    eda::enqueue_mode_kappa_ramp<T>(nullptr, g, c, le.p, dmax.p, dmin.p, out.p, on.p);
    const auto want = [&](int64_t i) {
      const int64_t b = i / g.field;
      const T fl = static_cast<T>(std::max(leh[b].tol, c.tiny));
      const T dd = npmax(static_cast<T>(emax[i] + emin[i]), fl);
      const T kap = static_cast<T>(static_cast<T>(emax[i] - emin[i]) / dd);
      return npclip(static_cast<T>(static_cast<T>(kap - static_cast<T>(0.3)) / static_cast<T>(0.6 - 0.3)), T(0), T(1));
    };
    CHECK(check_field<T>(g, out.get(), ob, onv, want));
    // in place over emax
    eda::enqueue_mode_kappa_ramp<T>(nullptr, g, c, le.p, dmax.p, dmin.p, dmax.p, on.p);
    CHECK(check_field<T>(g, dmax.get(), emax, onv, want));
  }
  {
    const std::vector<T> gk = rand_field<T>(g, rng, 0.0, 1.0), e = rand_field<T>(g, rng, 0.1, 2.0);
    const std::vector<T> Mn = rand_field<T>(g, rng, -2.0, 2.0), pl = rand_field<T>(g, rng, -1.0, 1.0);
    Dev<T> dg(gk), de(e), dM(Mn), dp(pl), mh(rand_field<T>(g, rng, 0, 1));
    const std::vector<T> mb = mh.get();
    eda::enqueue_mode_blend<T>(nullptr, g, dg.p, de.p, dM.p, u.p, l.p, dp.p, mh.p, on.p);
    const auto want = [&](int64_t i) {
      const T mA = npclip(static_cast<T>(e[i] * Mn[i]), l0[i], u0[i]);
      return static_cast<T>(static_cast<T>(gk[i] * mA) + static_cast<T>(static_cast<T>(T(1) - gk[i]) * pl[i]));
    };
    CHECK(check_field<T>(g, mh.get(), mb, onv, want));
    eda::enqueue_mode_blend<T>(nullptr, g, dg.p, de.p, dM.p, u.p, l.p, dp.p, dM.p, on.p);  // in place
    CHECK(check_field<T>(g, dM.get(), Mn, onv, want));
  }
}

// ---------------------------------------------------------------------------------------------
// The sift with the guard: accepted, rejected at sift 2, rejected at sift 3

template <class T>
void test_sift_t() {
  const int64_t B = 4;
  int64_t n[3] = {70, 1, 1};
  const eda::Geom g = eda::make_geom(1, B, n);
  const size_t field = static_cast<size_t>(B * g.field);
  {
    Rng rng(32 + sizeof(T));
    const std::vector<T> r0 = rand_field<T>(g, rng, -1.0, 1.0);
    std::vector<eda::ElemRecord> leh(B);
    std::memset(leh.data(), 0, sizeof(eda::ElemRecord) * B);
    for (int64_t b = 0; b < B; ++b) {
      leh[b].status = b == 3 ? eda::kRecordDone : eda::kRecordActive;
      leh[b].sifting = 1;
      double m = 0;
      for (int64_t v = 0; v < g.n[0]; ++v) m = std::max(m, std::fabs(static_cast<double>(r0[b * g.field + v])));
      leh[b].r_inf = m;
    }
    Dev<eda::ElemRecord> le(leh);
    Dev<T> r(r0), h(rand_field<T>(g, rng, 0, 1)), hn(rand_field<T>(g, rng, 0, 1)), mhc(field);
    const std::vector<T> hstart = h.get();
    const std::vector<T> a0 = rand_field<T>(g, rng, 0.0, 1.0);
    Dev<T> a(a0);
    std::vector<T> ht = hstart;  // twin of h
    std::vector<int32_t> nsift(B, 0), sifting(B, 1);
    for (int32_t j = 0; j < 3; ++j) {
      // mh: small, except element 1 at sift 2 (j = 1) and element 2 at sift 3 (j = 2): the guard
      std::vector<T> mh0 = rand_field<T>(g, rng, -0.05, 0.05);
      // element 0: mh = 0 (|hn| = r_inf exactly: accepted); elements 1, 2: 0.25 r until their
      // rejecting sift
      for (int64_t v = 0; v < g.n[0]; ++v) {
        mh0[0 * g.field + v] = T(0);
        for (int64_t b = 1; b <= 2; ++b) mh0[b * g.field + v] = static_cast<T>(T(0.25) * r0[b * g.field + v]);
        if (j == 1) mh0[1 * g.field + v] = static_cast<T>(-3.0 * (v % 2 ? 1 : -1));
        if (j == 2) mh0[2 * g.field + v] = static_cast<T>(2.5);
      }
      Dev<T> mh(mh0);
      const T* hin = j == 0 ? r.p : h.p;
      T* hout = j == 0 ? h.p : hn.p;
      eda::enqueue_mode_sift<T>(nullptr, g, le.p, j, r.p, hin, mh.p, j ? a.p : nullptr, hout, mhc.p);
      eda::enqueue_mode_accept<T>(nullptr, g, le.p, j, hout, h.p);
      const std::vector<T> mo = mhc.get();
      for (int64_t b = 0; b < B; ++b) {
        if (leh[b].status != eda::kRecordActive || !sifting[b]) continue;
        std::vector<T> cand(static_cast<size_t>(g.field));
        double mres = 0, mh_ = 0;
        bool mh_ok = true;
        for (int64_t v = 0; v < g.n[0]; ++v) {
          const size_t i = static_cast<size_t>(b * g.field + v);
          T m = mh0[i];
          if (j >= 1) m = static_cast<T>(a0[i] * m);
          mh_ok = mh_ok && same_t(mo[i], m);
          const T base = j == 0 ? r0[i] : ht[i];
          cand[v] = static_cast<T>(base - m);
          mres = std::max(mres, std::fabs(static_cast<double>(static_cast<T>(r0[i] - cand[v]))));
          mh_ = std::max(mh_, std::fabs(static_cast<double>(cand[v])));
        }
        CHECK(mh_ok);
        const bool acc = j == 0 || (mres <= leh[b].r_inf && mh_ <= leh[b].r_inf);
        if (acc) {
          for (int64_t v = 0; v < g.n[0]; ++v) ht[b * g.field + v] = cand[v];
          ++nsift[b];
        } else {
          sifting[b] = 0;
        }
      }
    }
    const std::vector<T> hg = h.get();
    const std::vector<eda::ElemRecord> out = le.get();
    for (int64_t b = 0; b < B; ++b) {
      bool ok = true;
      for (int64_t v = 0; v < g.field; ++v) {
        const size_t i = static_cast<size_t>(b * g.field + v);
        const T want = (v < g.n[0] && leh[b].status == eda::kRecordActive) ? ht[i] : hstart[i];
        ok = ok && (same_t(hg[i], want) || (std::isnan(hg[i]) && std::isnan(want)));
      }
      CHECK(ok);
      if (b == 3) {
        CHECK(out[b].n_sifts == 0);
        continue;
      }
      CHECK(out[b].n_sifts == nsift[b] && out[b].sifting == sifting[b]);
      CHECK(out[b].gmax_res_bits == 0 && out[b].gmax_h_bits == 0);
    }
    CHECK(nsift[0] == 3 && nsift[1] == 1 && nsift[2] == 2);
  }
}

// ---------------------------------------------------------------------------------------------
// The commit: rows, r - h, the history, max_imf

template <class T>
void test_commit_t() {
  const int64_t B = 3;
  int64_t n[3] = {8, 40, 1};
  const eda::Geom g = eda::make_geom(2, B, n);
  const size_t field = static_cast<size_t>(B * g.field);
  const int32_t cap = 4;
  Rng rng(808 + sizeof(T));
  const std::vector<T> r0 = rand_field<T>(g, rng, -1.0, 1.0), h0 = rand_field<T>(g, rng, -0.5, 0.5);
  std::vector<T> r0z = r0, h0z = h0;
  for (size_t i = 0; i < field; ++i) {
    if (is_pad(g, static_cast<int64_t>(i))) r0z[i] = h0z[i] = T(0);
  }
  Dev<T> r(r0z), h(h0z);
  Dev<T> rows(static_cast<size_t>(B) * cap * g.count);
  cudaMemset(rows.p, 0, sizeof(T) * rows.n);
  const size_t he = static_cast<size_t>(B) * (cap - 1);
  Dev<int32_t> hs(he), hsi(he);
  Dev<double> hsp(he), hen(he);
  cudaMemset(hs.p, 0, he * 4);
  cudaMemset(hsi.p, 0, he * 4);
  cudaMemset(hsp.p, 0, he * 8);
  cudaMemset(hen.p, 0, he * 8);
  eda::CallCtl hc{};
  hc.out = rows.p;
  hc.rows_cap = cap;
  hc.max_imf = 2;
  hc.hist_scale = hs.p;
  hc.hist_spacing = hsp.p;
  hc.hist_sifts = hsi.p;
  hc.hist_energy = hen.p;
  hc.tau = static_cast<double>(static_cast<T>(std::ldexp(1.0, sizeof(T) == 4 ? -14 : -30)));
  Dev<eda::CallCtl> ctl(std::vector<eda::CallCtl>{hc});
  std::vector<eda::ElemState> ne(B);
  std::memset(ne.data(), 0, sizeof(eda::ElemState) * B);
  std::vector<eda::ElemRecord> leh(B);
  std::memset(leh.data(), 0, sizeof(eda::ElemRecord) * B);
  for (int64_t b = 0; b < B; ++b) {
    ne[b].status = b == 2 ? eda::kDone : eda::kActive;
    ne[b].n_imfs = b == 1 ? 1 : 0;
    leh[b].status = b == 2 ? eda::kRecordDone : eda::kRecordActive;
    leh[b].w = 5 + 2 * static_cast<int32_t>(b);
    leh[b].k = ne[b].n_imfs;
    leh[b].w_prev = 3;
    leh[b].n_sifts = 1 + static_cast<int32_t>(b);
    leh[b].spacing = 1.5 + b;
  }
  Dev<eda::ElemState> elems(ne);
  Dev<eda::ElemRecord> le(leh);
  Dev<eda::RowPartial> ra(static_cast<size_t>(B * g.rows)), rb(static_cast<size_t>(B * g.rows));
  eda::enqueue_mode_commit<T>(nullptr, g, ctl.p, elems.p, le.p, h.p, r.p, ra.p, rb.p);
  const std::vector<T> rg = r.get(), rowsg = rows.get();
  const std::vector<eda::ElemState> neg = elems.get();
  const std::vector<eda::ElemRecord> leg = le.get();
  const std::vector<int32_t> hsg = hs.get(), hsig = hsi.get();
  const std::vector<double> hspg = hsp.get(), heng = hen.get();
  host_twin::Shape sh;
  sh.dim = 2;
  sh.n[0] = 8;
  sh.n[1] = 40;
  const int32_t nl = g.n[1];
  for (int64_t b = 0; b < B; ++b) {
    const bool act = b != 2;
    std::vector<T> hu(static_cast<size_t>(g.count)), ru(static_cast<size_t>(g.count));
    bool ok = true;
    for (int64_t q = 0; q < g.rows; ++q) {
      for (int32_t t = 0; t < nl; ++t) {
        const size_t i = static_cast<size_t>(b * g.field + q * g.pitch + t);
        const T want = act ? static_cast<T>(r0z[i] - h0z[i]) : r0z[i];
        ok = ok && same_t(rg[i], want);
        hu[q * nl + t] = h0z[i];
        ru[q * nl + t] = want;
        const T row = rowsg[(static_cast<size_t>(b) * cap + ne[b].n_imfs) * g.count + q * nl + t];
        ok = ok && same_t(row, act ? h0z[i] : T(0));
      }
    }
    CHECK(ok);
    if (!act) {
      CHECK(neg[b].n_imfs == 0 && leg[b].k == 0);
      continue;
    }
    const size_t hi = static_cast<size_t>(b) * (cap - 1) + ne[b].n_imfs;
    CHECK(neg[b].n_imfs == ne[b].n_imfs + 1);
    CHECK(hsg[hi] == leh[b].w && hsig[hi] == leh[b].n_sifts && same(hspg[hi], leh[b].spacing));
    CHECK(same(heng[hi], host_twin::field_energy<T>(sh, hu)));
    CHECK(same(neg[b].amax, host_twin::amax_of<T>(ru)));
    CHECK(leg[b].w_prev == leh[b].w && leg[b].k == leh[b].k + 1);
    // element 1 reached max_imf = 2: the element state and the method's record are done
    CHECK(leg[b].status == (b == 1 ? eda::kRecordDone : eda::kRecordActive));
    CHECK(neg[b].status == (b == 1 ? eda::kDone : eda::kActive));
  }
}


// ---------------------------------------------------------------------------------------------
// The magnitude prescale, the unscale and the non-finite guard (method/mode.hpp): exact powers of
// two (host twin: ldexp in double, one rounding to T), the exponents, the overflow flag, and the
// guard's failure of a flagged element; the flags of tau and the orientation on non-finite input.

// v 2^k with one rounding (float: exact in double first).
template <class T>
T twin_pow2(T v, int k) {
  return static_cast<T>(std::ldexp(static_cast<double>(v), k));
}

template <class T>
void test_scale_t() {
  const bool f32 = sizeof(T) == 4;
  const int64_t B = 5;
  int64_t n[3] = {7, 37, 1};
  const eda::Geom g = eda::make_geom(2, B, n);
  Rng rng(4242 + sizeof(T));
  // element magnitudes 2^e: 0 (a zero field), in [1, 2) (k = 0), large, small (subnormal samples in
  // T after the scale back), and a mid one
  const int e_big = f32 ? 120 : 1000, e_small = f32 ? -130 : -1030;
  const int mag[B] = {0, 0, e_big, e_small, 7};
  std::vector<T> r = rand_field<T>(g, rng, -1.0, 1.0);
  std::vector<eda::ElemState> ne(B);
  std::memset(ne.data(), 0, sizeof(eda::ElemState) * B);
  std::vector<int32_t> want_k(B, 0);
  for (int64_t b = 0; b < B; ++b) {
    double amax = 0.0;
    for (int64_t v = 0; v < g.field; ++v) {
      const int64_t i = b * g.field + v;
      if (is_pad(g, i)) continue;
      // |r| < 1.75 2^mag, and one sample -1.5 2^mag (exact): max|r| in [2^mag, 2^(mag + 1))
      if (b == 0) r[i] = T(0);
      else r[i] = twin_pow2<T>(v == 0 ? static_cast<T>(-1.5) : static_cast<T>(1.75) * r[i], mag[b]);
      amax = std::max(amax, std::fabs(static_cast<double>(r[i])));
    }
    ne[b].amax = amax;
    ne[b].status = eda::kActive;
    want_k[b] = amax > 0 ? std::ilogb(amax) : 0;
    CHECK(want_k[b] == mag[b]);
  }
  Dev<T> dr(r);
  Dev<eda::ElemState> de(ne);
  Dev<int32_t> dexp(static_cast<size_t>(B));
  eda::enqueue_mode_prescale<T>(nullptr, g, de.p, dexp.p, dr.p);
  const std::vector<int32_t> k = dexp.get();
  CHECK(k == want_k);
  const std::vector<T> rs = dr.get();
  std::vector<int32_t> all(B, 1);
  CHECK(check_field<T>(g, rs, r, all, [&](int64_t i) { return twin_pow2<T>(r[i], -want_k[i / g.field]); }));
  for (int64_t b = 1; b < B; ++b) {  // max|r| in [1, 2) after the scale, exactly
    double m = 0.0;
    for (int64_t v = 0; v < g.field; ++v) {
      if (!is_pad(g, b * g.field + v)) m = std::max(m, std::fabs(static_cast<double>(rs[b * g.field + v])));
    }
    CHECK(m >= 1.0 && m < 2.0 && same(m, std::ldexp(ne[b].amax, -want_k[b])));
  }
  // unscale: rows 0..n_imfs of the elements that did not fail; element 4 overflows (k past the
  // largest exponent of T), element 1 failed (status kError: untouched)
  const int32_t cap = 3;
  std::vector<T> rows(static_cast<size_t>(B) * cap * g.count);
  for (auto& v : rows) v = static_cast<T>(-1.9 + 3.8 * rng.uniform());
  std::vector<int32_t> ks = {0, 5, f32 ? -140 : -1070, f32 ? 100 : 1000, f32 ? 128 : 1024};
  for (int64_t b = 0; b < B; ++b) {
    ne[b].n_imfs = b == 2 ? 0 : 1;
    ne[b].status = b == 1 ? eda::kError : eda::kDone;
    ne[b].error = b == 1 ? eda::kErrNonFinite : 0;
  }
  Dev<T> drows(rows);
  de.put(ne);
  dexp.put(ks);
  eda::enqueue_mode_unscale<T>(nullptr, g, de.p, dexp.p, drows.p, cap);
  const std::vector<T> got = drows.get();
  const std::vector<eda::ElemState> ne2 = de.get();
  bool ok = true;
  for (int64_t b = 0; b < B; ++b) {
    for (int32_t q = 0; q < cap; ++q) {
      for (int64_t i = 0; i < g.count; ++i) {
        const size_t at = (static_cast<size_t>(b) * cap + q) * g.count + i;
        const bool scaled = b != 1 && q <= ne[b].n_imfs;
        const T want = scaled ? twin_pow2<T>(rows[at], ks[b]) : rows[at];
        ok = ok && same_t(got[at], want);
      }
    }
    CHECK(ne2[b].status == ne[b].status);
    CHECK(ne2[b].error == (b == 4 ? eda::kErrNonFiniteResult : ne[b].error));
  }
  CHECK(ok);
}

void test_guard() {
  const int64_t B = 4;
  Arrays A(B);
  std::vector<eda::ElemState> ne(B);
  std::memset(ne.data(), 0, sizeof(eda::ElemState) * B);
  for (auto& e : ne) e.status = eda::kActive;
  A.elems.put(ne);
  eda::enqueue_mode_init(nullptr, B, A.view());
  std::vector<eda::ElemRecord> le = A.le.get();
  le[1].nonfinite = 1;               // active and flagged: fails
  le[2].nonfinite = 1;               // flagged but done: left alone
  le[2].status = eda::kRecordDone;
  A.le.put(le);
  std::vector<int32_t> ints = A.ints_h();
  for (int64_t b = 0; b < B; ++b) {
    for (int s = 0; s < eda::kWinSlots; ++s) ints[(3 + s) * B + b] = 9;
  }
  A.ints.put(ints);
  eda::enqueue_mode_guard(nullptr, B, A.view());
  le = A.le.get();
  const std::vector<eda::ElemState> got = A.elems.get();
  ints = A.ints_h();
  for (int64_t b = 0; b < B; ++b) {
    const bool fail = b == 1;
    CHECK(le[b].status == (fail ? eda::kRecordError : (b == 2 ? eda::kRecordDone : eda::kRecordActive)));
    CHECK(got[b].status == (fail ? eda::kError : eda::kActive));
    CHECK(got[b].error == (fail ? eda::kErrNonFiniteResult : 0));
    CHECK(ints[b] == (fail ? 0 : 1));  // on
    for (int s = 0; s < eda::kWinSlots; ++s) CHECK(ints[(3 + s) * B + b] == (fail ? 0 : 9));
  }
}

// tau and the eigen pass flag an element whose statistics are not finite: a T field near the
// top of its range (det^2 and g_a g_b overflow: the case the driver's prescale avoids), and an inf
// sample; a unit-magnitude element is not flagged.
template <class T>
void test_nonfinite_flags_t() {
  const bool f32 = sizeof(T) == 4;
  const int64_t B = 3;
  int64_t n[3] = {12, 33, 1};
  const eda::Geom g = eda::make_geom(2, B, n);
  Rng rng(77 + sizeof(T));
  std::vector<T> r = rand_field<T>(g, rng, -1.0, 1.0);
  const double big = f32 ? 1e30 : 1e300;
  for (int64_t v = 0; v < g.field; ++v) {
    if (is_pad(g, v)) {
      r[v] = r[g.field + v] = r[2 * g.field + v] = T(0);
      continue;
    }
    r[g.field + v] = static_cast<T>(big * static_cast<double>(r[g.field + v]));
  }
  r[2 * g.field + g.pitch + 3] = std::numeric_limits<T>::infinity();
  const eda::MethodCtl c = eda::method_ctl(g, f32 ? eda::DType::F32 : eda::DType::F64);
  Dev<T> dr(r), tau(static_cast<size_t>(B * g.field)), tmp(static_cast<size_t>(B * g.field));
  Dev<int32_t> W(std::vector<int32_t>(B, 5));
  Dev<double> sig(std::vector<double>(B, 0.01)), tol(std::vector<double>(B, 1e-6));
  std::vector<eda::ElemRecord> le0(B);
  std::memset(le0.data(), 0, sizeof(eda::ElemRecord) * B);
  {
    Dev<eda::ElemRecord> le(le0);
    eda::enqueue_tau<T>(nullptr, g, c, dr.p, W.p, W.p, sig.p, tol.p, nullptr, tau.p, tmp.p, nullptr, le.p);
    const std::vector<eda::ElemRecord> got = le.get();
    CHECK(got[0].nonfinite == 0 && got[1].nonfinite == 1 && got[2].nonfinite == 1);
  }
  {
    // the eigen pass flags a non-finite tensor (the orientation's last stage): an inf (an
    // overflow of the energy tensor) and a NaN; a finite one is not flagged
    Dev<eda::ElemRecord> le(le0);
    const size_t K = 3;  // (0,0), (0,1), (1,1)
    std::vector<T> jt(K * static_cast<size_t>(B * g.field), T(1));
    jt[(0 * B + 1) * g.field + g.pitch + 3] = std::numeric_limits<T>::infinity();
    jt[(2 * B + 2) * g.field + g.pitch + 4] = std::numeric_limits<T>::quiet_NaN();
    Dev<T> work(jt);
    Dev<T> n0(static_cast<size_t>(B * g.field)), n1(static_cast<size_t>(B * g.field));
    Dev<T> coh(static_cast<size_t>(B * g.field));
    eda::FieldVec<T> nv{};
    nv.p[0] = n0.p;
    nv.p[1] = n1.p;
    eda::enqueue_element_eigen<T>(nullptr, g, work.p, W.p, nv, coh.p, nullptr, le.p);
    const std::vector<eda::ElemRecord> got = le.get();
    CHECK(got[0].nonfinite == 0 && got[1].nonfinite == 1 && got[2].nonfinite == 1);
  }
}

// ---------------------------------------------------------------------------------------------
// validate_method (through eda::validate)

void test_validate() {
  std::vector<double> x(64, 0.0);
  eda::Input in;
  in.dtype = eda::DType::F64;
  in.x = x.data();
  in.dim = 1;
  in.batch = 1;
  in.n[0] = 64;
  eda::Params p;
  eda::validate(in, p);  // the defaults pass
  {
    eda::Input bad = in;
    bad.dim = 4;
    CHECK(eda_test::throws_with<std::invalid_argument>([&] { eda::validate(bad, p); }, "D must be 1, 2 or 3"));
    bad = in;
    bad.n[0] = 0;
    CHECK(eda_test::throws_with<std::invalid_argument>([&] { eda::validate(bad, p); }, "extent must be >= 1"));
    bad = in;
    bad.n[0] = 4097;
    CHECK(eda_test::throws_with<std::invalid_argument>([&] { eda::validate(bad, p); }, "exceeds the limit 4096"));
    bad.dtype = eda::DType::F32;
    eda::validate(bad, p);
    bad.n[0] = 8193;
    CHECK(eda_test::throws_with<std::invalid_argument>([&] { eda::validate(bad, p); }, "exceeds the limit 8192"));
    bad = in;
    bad.batch = 0;
    CHECK(eda_test::throws_with<std::invalid_argument>([&] { eda::validate(bad, p); }, "at least one element"));
    // an axis of extent 1 (no sample differences along it, so no noise level)
    bad = in;
    bad.n[0] = 1;
    CHECK(eda_test::throws_with<std::invalid_argument>([&] { eda::validate(bad, p); }, "at least 2 samples"));
    bad = in;
    bad.n[0] = 2;
    eda::validate(bad, p);
    bad.dim = 2;
    bad.n[0] = 1;
    bad.n[1] = 16;
    CHECK(eda_test::throws_with<std::invalid_argument>([&] { eda::validate(bad, p); },
                                                       "spatial axis 0 has extent 1"));
    bad.dim = 3;
    bad.n[0] = 8;
    bad.n[1] = 8;
    bad.n[2] = 1;
    CHECK(eda_test::throws_with<std::invalid_argument>([&] { eda::validate(bad, p); },
                                                       "spatial axis 2 has extent 1"));
    bad.n[2] = 2;
    eda::validate(bad, p);
  }
  {
    eda::Params q = p;
    q.tie_tolerance = -1.0;
    CHECK(eda_test::throws_with<std::invalid_argument>([&] { eda::validate(in, q); }, "tie_tolerance"));
    q = p;
    q.max_chunk = -1;
    CHECK(eda_test::throws_with<std::invalid_argument>([&] { eda::validate(in, q); }, "max_chunk"));
  }
}

}  // namespace

int main() {
  test_validate();
  test_start();
  test_rice();
  test_window();
  test_g2_rungs();
  test_g2_machine();
  test_g2_accept();
  for (int d = 1; d <= 3; ++d) {
    test_fields_t<float>(d);
    test_fields_t<double>(d);
  }
  test_sift_t<float>();
  test_sift_t<double>();
  test_commit_t<float>();
  test_commit_t<double>();
  test_scale_t<float>();
  test_scale_t<double>();
  test_guard();
  test_nonfinite_flags_t<float>();
  test_nonfinite_flags_t<double>();
  const cudaError_t e = cudaDeviceSynchronize();
  CHECK(e == cudaSuccess);
  if (e != cudaSuccess) std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));
  return eda_test::finish("test_mode");
}
