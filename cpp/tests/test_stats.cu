// SPDX-License-Identifier: Apache-2.0
// test_stats: the statistics and fields (method/stats.hpp) against host twins written from their
// definitions (no shared code with the kernels beyond field/common.cuh's scalar helpers and the
// vHGW host twin of twin.hpp):
//   - selections (median, quantile) bitwise against std::sort on random and adversarial value sets
//     (ties, +-0, negatives, one value, even and odd counts, empty sets, masks, pooled differences,
//     centers), and NumPy's quantile values tabulated below;
//   - noise levels (pooled and per axis) bitwise;
//   - the threshold field bitwise against a twin over the vHGW host twin box;
//   - significant extrema against two twins: SciPy's peak finding (its local-maxima scan and the
//     prominence walk) and monotone stacks (O(n) per line): the probed line, plateaus (also at the
//     line ends and the 32-sample block edges), ties, monotone and constant lines, decisive samples
//     around the block edges, equal peaks, float ties at tau (also inside a block summary),
//     subnormals, infinities, NaN lines, a line of 2^20 samples, every axis of 3-D fields, the
//     split of a call into one launch per element;
//   - the raw window bitwise; components against a breadth-first twin; the scale field and the floor
//     (masks exact, L within the device / host libm difference); G2's rung test and G3's coarse
//     median bitwise on crafted populations, with the outcomes none, accepted and stop all covered.
// Under the sanitizers (EDA_SANITIZE=1) the inputs are small.
#include "twin.hpp"
#include "method/method.hpp"
#include "method/stats.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace host_twin;

namespace {

bool g_small = false;
const double kInf = std::numeric_limits<double>::infinity();

Shape shape(int dim, int64_t a, int64_t b = 1, int64_t c = 1) {
  Shape s;
  s.dim = dim;
  s.n[0] = a;
  s.n[1] = b;
  s.n[2] = c;
  return s;
}
eda::Geom geom(const Shape& sh, int64_t B) { return eda::make_geom(sh.dim, B, sh.n); }

// padded in-element index of the unpadded linear index u
int64_t padded_of(const eda::Geom& g, int64_t u) {
  int64_t c[3] = {0, 0, 0};
  for (int a = g.dim - 1; a >= 0; --a) {
    c[a] = u % g.n[a];
    u /= g.n[a];
  }
  int64_t v = 0;
  for (int a = 0; a < g.dim; ++a) v += c[a] * (a == g.dim - 1 ? 1 : g.stride[a]);
  return v;
}

template <class V>
struct Dev {
  V* p = nullptr;
  size_t n = 0;
  explicit Dev(size_t count) : n(count) {
    CHECK(cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(V)) == cudaSuccess);
    // zeroed: get() copies whole padded buffers, pads included, which the kernels never write
    // (initcheck reports the copy of never-written bytes as a host API error otherwise)
    CHECK(cudaMemset(p, 0, std::max<size_t>(count, 1) * sizeof(V)) == cudaSuccess);
  }
  Dev(const std::vector<V>& h) : Dev(h.size()) { put(h); }
  ~Dev() { cudaFree(p); }
  Dev(const Dev&) = delete;
  void put(const std::vector<V>& h) { cudaMemcpy(p, h.data(), h.size() * sizeof(V), cudaMemcpyHostToDevice); }
  std::vector<V> get() const {
    std::vector<V> h(n);
    cudaMemcpy(h.data(), p, n * sizeof(V), cudaMemcpyDeviceToHost);
    return h;
  }
};

// B unpadded element fields (B * count) -> the padded layout (pads 0), and back.
template <class V>
std::vector<V> pad(const eda::Geom& g, const std::vector<V>& x) {
  std::vector<V> p(static_cast<size_t>(g.batch * g.field), V(0));
  for (int64_t b = 0; b < g.batch; ++b) {
    for (int64_t u = 0; u < g.count; ++u) p[b * g.field + padded_of(g, u)] = x[b * g.count + u];
  }
  return p;
}
template <class V>
std::vector<V> unpad(const eda::Geom& g, const std::vector<V>& p) {
  std::vector<V> x(static_cast<size_t>(g.batch * g.count));
  for (int64_t b = 0; b < g.batch; ++b) {
    for (int64_t u = 0; u < g.count; ++u) x[b * g.count + u] = p[b * g.field + padded_of(g, u)];
  }
  return x;
}
template <class V>
bool pads_zero(const eda::Geom& g, const std::vector<V>& p) {
  for (int64_t i = 0; i < g.batch * g.field; ++i) {
    if ((i % g.field) % g.pitch >= g.n[g.dim - 1] && !(p[i] == V(0))) return false;
  }
  return true;
}

template <class T>
std::vector<T> elem(const std::vector<T>& x, int64_t b, int64_t cnt) {
  return std::vector<T>(x.begin() + b * cnt, x.begin() + (b + 1) * cnt);
}

// ---- host twins of the cube filter ---------------------------------------------------------------
template <class T>
std::vector<T> box(const Shape& sh, std::vector<T> f, int32_t w) {
  for (int a = 0; a < sh.dim; ++a) vhgw_axis<T>(sh, a, w, kSum, f);
  const T fac = box_factor<T>(sh, w, 1.0);
  for (T& v : f) v = mul_rn(v, fac);
  return f;
}
template <class T>
std::vector<T> cube_mm(const Shape& sh, std::vector<T> f, int32_t w, Op op) {
  for (int a = sh.dim - 1; a >= 0; --a) vhgw_axis<T>(sh, a, w, op, f);
  return f;
}

// ---- selections -------------------------------------------------------------------------------------
double canon(double v) { return v == 0.0 ? 0.0 : v; }
double twin_median(std::vector<double> v) {
  if (v.empty()) return std::nan("");
  for (double& x : v) x = canon(x);
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[(n - 1) / 2] : eda::median_pair(v[(n - 1) / 2], v[n / 2]);
}
double twin_quantile(std::vector<double> v, double q) {
  if (v.empty()) return std::nan("");
  for (double& x : v) x = canon(x);
  std::sort(v.begin(), v.end());
  // NumPy 2.5 (method "linear"): virtual index (n - 1) q; >= n - 1: the last value twice, gamma =
  // v - (-1); < 0: the first; lerp: a + (b - a) gamma, or b - (b - a)(1 - gamma) for gamma >= 0.5
  const int64_t n = static_cast<int64_t>(v.size());
  const double vi = static_cast<double>(n - 1) * q;
  int64_t i = 0, j = 0;
  double gm = 0.0;
  if (vi >= static_cast<double>(n - 1)) {
    i = j = n - 1;
    gm = vi + 1.0;
  } else if (vi < 0.0) {
    gm = vi;
  } else {
    i = static_cast<int64_t>(std::floor(vi));
    j = i + 1;
    gm = vi - std::floor(vi);
  }
  const double a = v[i], b = v[j], diff = b - a;
  return gm >= 0.5 ? b - diff * (1.0 - gm) : a + diff * gm;
}
bool same_double(double a, double b) {
  if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
  return std::memcmp(&a, &b, 8) == 0;
}

// the value set of one element (unpadded data): kField with mask, or kDiff over the axes
template <class T>
std::vector<double> values_of(const Shape& sh, const std::vector<T>& f, const std::vector<uint8_t>* m, uint8_t bits,
                              bool diff, uint8_t axes, const double* center) {
  std::vector<double> v;
  if (!diff) {
    for (size_t i = 0; i < f.size(); ++i) {
      if (m && ((*m)[i] & bits) == 0) continue;
      v.push_back(static_cast<double>(f[i]));
    }
  } else {
    for (int a = 0; a < sh.dim; ++a) {
      if (!(axes & (1u << a))) continue;
      const int64_t st = sh.stride(a);
      for (int64_t u = 0; u < sh.count(); ++u) {
        if ((u / st) % sh.n[a] + 1 >= sh.n[a]) continue;
        v.push_back(static_cast<double>(sub_rn(f[u + st], f[u])));
      }
    }
  }
  if (center) {
    for (double& x : v) x = std::fabs(x - *center);
  }
  return v;
}

template <class T>
void check_select(const Shape& sh, int64_t B, int kind, bool use_mask, bool diff, uint8_t axes, bool centered,
                  uint64_t seed) {
  const eda::Geom g = geom(sh, B);
  const int64_t cnt = sh.count();
  std::vector<T> x;
  std::vector<uint8_t> m;
  Rng rng(seed);
  for (int64_t b = 0; b < B; ++b) {
    std::vector<T> f = make_field<T>(sh, kind, seed + b);
    if (kind == 4) {  // adversarial: signed zeros, negatives, huge, tiny, infinities
      // (no infinities in a difference set: inf - inf is NaN, which never reaches a selection)
      const double big = diff ? 1e30 : 1e300;  // 1e300 is inf in float32
      const double pool[] = {0.0, -0.0, -1.5, 1.5, big, -big, 4.9e-324, -2.5, 3.0, 1.0, kInf, -kInf};
      for (auto& v : f) v = static_cast<T>(pool[rng.next() % (diff ? 10 : 12)]);
    }
    x.insert(x.end(), f.begin(), f.end());
    for (int64_t i = 0; i < cnt; ++i) {
      // element 1: empty mask; element 2: one value; the others random
      uint8_t bit = static_cast<uint8_t>(rng.next() % 3 == 0 ? 1 : 2);
      if (b == 1) bit = 2;
      if (b == 2) bit = i == cnt / 2 ? 1 : 2;
      m.push_back(bit);
    }
  }
  std::vector<double> centers(B);
  for (int64_t b = 0; b < B; ++b) centers[b] = b % 2 ? -0.25 : 0.5;
  Dev<T> dx(pad(g, x));
  Dev<uint8_t> dm(pad(g, m));
  Dev<double> dc(centers);
  Dev<char> ws(eda::select_workspace_bytes(g));
  Dev<double> out(B);
  Dev<int64_t> count(B);
  eda::Selection<T> sel{};
  sel.src = diff ? eda::SelSource::kDiff : eda::SelSource::kField;
  sel.field = dx.p;
  sel.mask = use_mask ? dm.p : nullptr;
  sel.bits = 1;
  sel.axes = axes;
  sel.center = centered ? dc.p : nullptr;
  const double qs[] = {0.25, 0.0, 1.0, 0.5, 0.9};
  for (int mode = 0; mode < 6; ++mode) {
    const bool med = mode == 0;
    const double q = med ? 0.5 : qs[mode - 1];
    if (med) {
      eda::enqueue_median<T>(nullptr, g, sel, out.p, count.p, ws.p, nullptr);
    } else {
      eda::enqueue_quantile<T>(nullptr, g, sel, q, out.p, count.p, ws.p, nullptr);
    }
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    const std::vector<double> got = out.get();
    const std::vector<int64_t> gc = count.get();
    for (int64_t b = 0; b < B; ++b) {
      const std::vector<T> f = elem(x, b, cnt);
      const std::vector<uint8_t> mb = elem(m, b, cnt);
      const std::vector<double> v = values_of(sh, f, use_mask ? &mb : nullptr, 1, diff, axes,
                                              centered ? &centers[b] : nullptr);
      const double want = med ? twin_median(v) : twin_quantile(v, q);
      const bool ok = same_double(got[b], want) && gc[b] == static_cast<int64_t>(v.size());
      CHECK(ok);
      if (!ok) {
        std::fprintf(stderr, "  select dim %d kind %d mask %d diff %d b %lld mode %d: %.17g vs %.17g (n %lld/%zu)\n",
                     sh.dim, kind, use_mask, diff, static_cast<long long>(b), mode, got[b], want,
                     static_cast<long long>(gc[b]), v.size());
      }
    }
  }
}

template <class T>
void selections() {
  for (int kind : {0, 1, 3, 4}) {
    check_select<T>(shape(1, 97), 4, kind, false, false, 1, false, 11 + kind);
    check_select<T>(shape(1, 97), 4, kind, true, false, 1, false, 12 + kind);
    check_select<T>(shape(2, 13, 40), 3, kind, true, false, 3, kind == 1, 13 + kind);
    check_select<T>(shape(2, 13, 40), 3, kind, false, true, 3, false, 14 + kind);
    check_select<T>(shape(3, 5, 7, 33), 3, kind, false, true, 5, true, 15 + kind);
    check_select<T>(shape(3, 5, 7, 33), 3, kind, true, false, 7, false, 16 + kind);
  }
  if (!g_small) {
    check_select<T>(shape(2, 300, 301), 2, 0, false, false, 3, false, 21);
    check_select<T>(shape(2, 300, 301), 2, 1, false, true, 3, true, 22);
    check_select<T>(shape(1, 1), 2, 0, false, true, 1, false, 23);  // no differences: NaN, count 0
  }
  check_select<T>(shape(1, 2), 3, 0, false, false, 1, false, 24);
}

// NumPy 2.5 values (numpy.quantile(v, q) with v = [i * i + 0.5 for i in range(n)])
void quantile_table() {
  struct Row {
    int n;
    double v[7];
  };
  const double qs[7] = {0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0};
  const Row rows[] = {{1, {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}},
                      {2, {0.5, 0.6, 0.75, 1.0, 1.25, 1.4, 1.5}},
                      {3, {0.5, 0.7, 1.0, 1.5, 3.0, 3.9000000000000004, 4.5}},
                      {4, {0.5, 0.8, 1.25, 3.0, 5.75, 8.0, 9.5}},
                      {5, {0.5, 0.9, 1.5, 4.5, 9.5, 13.700000000000001, 16.5}},
                      {7, {0.5, 1.1, 3.0, 9.5, 21.0, 29.900000000000006, 36.5}},
                      {10, {0.5, 1.4, 5.75, 21.0, 46.25, 66.19999999999999, 81.5}}};
  const eda::Geom g = eda::make_geom(1, 1, std::vector<int64_t>{16}.data());
  for (const Row& r : rows) {
    std::vector<double> v;
    for (int i = 0; i < r.n; ++i) v.push_back(i * i + 0.5);
    for (int j = 0; j < 7; ++j) CHECK(same_double(twin_quantile(v, qs[j]), r.v[j]));
    // the device on the same set (a 16-sample field, masked to the first n samples, shuffled)
    std::vector<double> f(16, 99.0);
    std::vector<uint8_t> m(16, 0);
    for (int i = 0; i < r.n; ++i) {
      f[(i * 7) % 16] = v[i];
      m[(i * 7) % 16] = 1;
    }
    Dev<double> df(pad(g, f));
    Dev<uint8_t> dm(pad(g, m));
    Dev<char> ws(eda::select_workspace_bytes(g));
    Dev<double> out(1);
    eda::Selection<double> sel{};
    sel.src = eda::SelSource::kField;
    sel.field = df.p;
    sel.mask = dm.p;
    sel.bits = 1;
    for (int j = 0; j < 7; ++j) {
      eda::enqueue_quantile<double>(nullptr, g, sel, qs[j], out.p, nullptr, ws.p, nullptr);
      CHECK(same_double(out.get()[0], r.v[j]));
    }
  }
  std::vector<double> v = {1.0, 2.0, 10.0, 11.0};
  CHECK(twin_quantile(v, 0.25) == 1.75);  // a hand-checked value: index 0.75 between 1 and 2
}

// ---- noise levels -------------------------------------------------------------------------------------
template <class T>
double twin_sigma(const Shape& sh, const std::vector<T>& f, uint8_t axes) {
  const std::vector<double> d = values_of(sh, f, nullptr, 0, true, axes, nullptr);
  const double c = twin_median(d);
  const double mad = twin_median(values_of(sh, f, nullptr, 0, true, axes, &c));
  return 1.4826 * mad / std::sqrt(2.0);
}

template <class T>
void noise_levels(const Shape& sh, int64_t B, uint64_t seed) {
  const eda::Geom g = geom(sh, B);
  std::vector<T> x;
  for (int64_t b = 0; b < B; ++b) {
    const std::vector<T> f = make_field<T>(sh, static_cast<int>(b % 3), seed + b);
    x.insert(x.end(), f.begin(), f.end());
  }
  Dev<T> dx(pad(g, x));
  Dev<char> ws(eda::select_workspace_bytes(g));
  Dev<double> out(B);
  for (int per = 0; per < 2; ++per) {
    eda::enqueue_noise_sigma<T>(nullptr, g, dx.p, per != 0, out.p, ws.p, nullptr);
    const std::vector<double> got = out.get();
    for (int64_t b = 0; b < B; ++b) {
      const std::vector<T> f = elem(x, b, sh.count());
      double want;
      if (!per) {
        want = twin_sigma(sh, f, static_cast<uint8_t>((1u << sh.dim) - 1));
      } else {
        want = 0.0;
        for (int a = 0; a < sh.dim; ++a) {
          if (sh.n[a] < 2) continue;
          const double v = twin_sigma(sh, f, static_cast<uint8_t>(1u << a));
          if (v > want) want = v;
        }
      }
      const bool ok = same_double(got[b], want);
      CHECK(ok);
      if (!ok) std::fprintf(stderr, "  noise dim %d per %d b %lld: %.17g vs %.17g\n", sh.dim, per,
                            static_cast<long long>(b), got[b], want);
    }
  }
}

// ---- tau ------------------------------------------------------------------------------------------------
template <class T>
std::vector<T> twin_tau(const Shape& sh, const std::vector<T>& r, int32_t Wd, int32_t Wl, double sigma, double tol,
                        const double* floor2) {
  const std::vector<T> bx = box(sh, r, Wd);
  std::vector<T> e(r.size());
  for (size_t i = 0; i < r.size(); ++i) {
    const T det = sub_rn(r[i], bx[i]);
    e[i] = mul_rn(det, det);
  }
  e = box(sh, e, Wl);
  const double f = 1.0 * sigma;
  const double fl = f > 0.0 ? f : 0.0;
  std::vector<T> out(r.size());
  for (size_t i = 0; i < r.size(); ++i) {
    const T lev = std::sqrt(e[i] > T(0) ? e[i] : T(0));
    double t = 0.3 * std::max(static_cast<double>(lev), fl);
    t = std::max(t, tol);
    if (floor2) t = std::max(t, *floor2);
    out[i] = static_cast<T>(t);
  }
  return out;
}

template <class T>
void tau_field(const Shape& sh, const std::vector<int32_t>& ws_, uint64_t seed) {
  const int64_t B = static_cast<int64_t>(ws_.size());
  const eda::Geom g = geom(sh, B);
  const eda::MethodCtl c = eda::method_ctl(g, dtype_of<T>());
  std::vector<T> x;
  std::vector<int32_t> Wd(B), Wl(B);
  std::vector<double> sig(B), tol(B), f2(B);
  for (int64_t b = 0; b < B; ++b) {
    const std::vector<T> f = make_field<T>(sh, static_cast<int>(b % 3), seed + b);
    x.insert(x.end(), f.begin(), f.end());
    Wd[b] = ws_[b] ? eda::odd_window(2.0 * ws_[b]) : 0;
    Wl[b] = ws_[b] ? eda::odd_window(2.0 * ws_[b]) : 0;
    sig[b] = b == 1 ? 0.0 : 0.05 * (b + 1);
    tol[b] = b == 2 ? 0.2 : 1e-9;
    f2[b] = b == 0 ? 0.25 : 0.0;
  }
  Dev<T> dx(pad(g, x)), tau(B * g.field), tmp(B * g.field);
  cudaMemset(tau.p, 0xff, tau.n * sizeof(T));
  Dev<int32_t> dWd(Wd), dWl(Wl);
  Dev<double> dsig(sig), dtol(tol), df2(f2);
  for (int fl2 = 0; fl2 < 2; ++fl2) {
    eda::enqueue_tau<T>(nullptr, g, c, dx.p, dWd.p, dWl.p, dsig.p, dtol.p, fl2 ? df2.p : nullptr, tau.p, tmp.p,
                        nullptr);
    const std::vector<T> gp = tau.get();
    const std::vector<T> got = unpad(g, gp);
    for (int64_t b = 0; b < B; ++b) {
      const std::vector<T> gb = elem(got, b, sh.count());
      if (!ws_[b]) {
        CHECK(std::isnan(static_cast<double>(gb[0])));  // skipped: the sentinel stays
        continue;
      }
      const std::vector<T> want =
          twin_tau(sh, elem(x, b, sh.count()), Wd[b], Wl[b], sig[b], tol[b], fl2 ? &f2[b] : nullptr);
      const bool ok = same_bits(gb.data(), want.data(), want.size());
      CHECK(ok);
      if (!ok) std::fprintf(stderr, "  tau dim %d b %lld w %d\n", sh.dim, static_cast<long long>(b), ws_[b]);
    }
  }
}

// ---- significant extrema: twins of SciPy's peak finding ------------------------------------------------
// the local-maxima scan of scipy.signal.find_peaks (plateaus at their middle sample)
std::vector<int64_t> local_maxima(const std::vector<double>& x) {
  std::vector<int64_t> mids;
  const int64_t n = static_cast<int64_t>(x.size());
  int64_t i = 1;
  const int64_t i_max = n - 1;
  while (i < i_max) {
    if (x[i - 1] < x[i]) {
      int64_t ahead = i + 1;
      while (ahead < i_max && x[ahead] == x[i]) ++ahead;
      if (x[ahead] < x[i]) {
        mids.push_back((i + ahead - 1) / 2);
        i = ahead;
      }
    }
    ++i;
  }
  return mids;
}
// the topographic prominence of scipy.signal.peak_prominences without a window
double prominence(const std::vector<double>& x, int64_t peak) {
  const int64_t n = static_cast<int64_t>(x.size());
  int64_t i = peak;
  double left_min = x[peak];
  while (0 <= i && x[i] <= x[peak]) {
    if (x[i] < left_min) left_min = x[i];
    --i;
  }
  i = peak;
  double right_min = x[peak];
  while (i <= n - 1 && x[i] <= x[peak]) {
    if (x[i] < right_min) right_min = x[i];
    ++i;
  }
  return x[peak] - std::max(left_min, right_min);
}
// the second twin, O(n) per line: at every sample the minimum of x since the last strictly greater
// sample (inclusive), by a monotone stack from the left (left = true) or the right; a NaN is never
// popped and pops nothing, so it ends a side as it ends SciPy's walk
std::vector<double> side_minima(const std::vector<double>& x, bool left) {
  const int64_t n = static_cast<int64_t>(x.size());
  std::vector<double> m(static_cast<size_t>(n));
  std::vector<std::pair<int64_t, double>> st;  // (index, minimum since the entry below)
  for (int64_t k = 0; k < n; ++k) {
    const int64_t i = left ? k : n - 1 - k;
    double mi = x[i];
    while (!st.empty() && x[st.back().first] <= x[i]) {
      mi = std::min(mi, st.back().second);
      st.pop_back();
    }
    m[i] = mi;
    st.push_back({i, mi});
  }
  return m;
}

// flags and counts of one unpadded element: SciPy's prominence walk (scipy), or the stacks with
// fl(v - min) >= tau on both sides (rounding is monotone, so the two agree)
template <class T>
void twin_flags(const Shape& sh, const std::vector<T>& r, const std::vector<T>& tau, std::vector<uint8_t>& fl,
                uint64_t* ext, bool scipy) {
  fl.assign(r.size(), 0);
  for (int a = 0; a < sh.dim; ++a) {
    ext[a] = 0;
    const int64_t n = sh.n[a], st = sh.stride(a), outer = sh.count() / (n * st);
    for (int64_t o = 0; o < outer; ++o) {
      for (int64_t j = 0; j < st; ++j) {
        const int64_t base = o * n * st + j;
        for (double sgn : {1.0, -1.0}) {
          std::vector<double> l(n);
          for (int64_t i = 0; i < n; ++i) l[i] = sgn * static_cast<double>(r[base + i * st]);
          std::vector<double> lm, rm;
          if (!scipy) {
            lm = side_minima(l, true);
            rm = side_minima(l, false);
          }
          for (int64_t p : local_maxima(l)) {
            const double t = static_cast<double>(tau[base + p * st]);
            const bool sig = scipy ? prominence(l, p) >= t : (l[p] - lm[p] >= t && l[p] - rm[p] >= t);
            if (sig) {
              fl[base + p * st] |= static_cast<uint8_t>(1u << a);
              ++ext[a];
            }
          }
        }
      }
    }
  }
}

// The kernels against the twins (the SciPy walk only with scipy; it is quadratic on long monotone
// stretches), element 1 skipped when B > 2 (its flags, counts and scratch untouched), with one
// launch per axis and, for B > 1, one element per launch.
template <class T>
void check_flags(const Shape& sh, const std::vector<std::vector<T>>& rs, const std::vector<std::vector<T>>& taus,
                 const char* what, bool scipy = true) {
  const int64_t B = static_cast<int64_t>(rs.size());
  const eda::Geom g = geom(sh, B);
  std::vector<T> x, t;
  for (int64_t b = 0; b < B; ++b) {
    x.insert(x.end(), rs[b].begin(), rs[b].end());
    t.insert(t.end(), taus[b].begin(), taus[b].end());
  }
  Dev<T> dx(pad(g, x)), dt(pad(g, t));
  std::vector<int32_t> on(B, 1);
  if (B > 2) on[1] = 0;  // a skipped element
  Dev<int32_t> don(on);
  std::vector<std::vector<uint8_t>> want(B);
  std::vector<std::vector<uint64_t>> wext(B, std::vector<uint64_t>(3, 0));
  for (int64_t b = 0; b < B; ++b) {
    if (!on[b]) continue;
    twin_flags(sh, rs[b], taus[b], want[b], wext[b].data(), false);
    if (scipy) {
      std::vector<uint8_t> w2;
      uint64_t e2[3] = {0, 0, 0};
      twin_flags(sh, rs[b], taus[b], w2, e2, true);
      const bool same = w2 == want[b] && std::equal(e2, e2 + 3, wext[b].begin());
      CHECK(same);
      if (!same) std::fprintf(stderr, "  flag twins differ: %s dim %d b %lld\n", what, sh.dim, static_cast<long long>(b));
    }
  }
  for (const int64_t ids : {eda::kSigFlagsLaunchIds, int64_t{1}}) {
    if (ids == 1 && B == 1) continue;
    Dev<uint8_t> fl(B * g.field);
    cudaMemset(fl.p, 0xff, fl.n);
    Dev<T> scr(B * g.field);
    cudaMemset(scr.p, 0xff, scr.n * sizeof(T));
    std::vector<eda::ElemRecord> le(B);
    std::memset(le.data(), 0, le.size() * sizeof(eda::ElemRecord));
    Dev<eda::ElemRecord> dle(le);
    eda::enqueue_sig_flags<T>(nullptr, g, dx.p, dt.p, fl.p, dle.p, don.p, scr.p, ids);
    const std::vector<uint8_t> fp = fl.get();
    const std::vector<uint8_t> got = unpad(g, fp);
    const std::vector<eda::ElemRecord> lh = dle.get();
    const std::vector<T> sp = scr.get();
    for (int64_t b = 0; b < B; ++b) {
      if (!on[b]) {
        bool untouched = true;
        for (int64_t i = 0; i < g.field; ++i) untouched = untouched && fp[b * g.field + i] == 0xff;
        const unsigned char* sb = reinterpret_cast<const unsigned char*>(sp.data() + b * g.field);
        for (size_t i = 0; i < static_cast<size_t>(g.field) * sizeof(T); ++i) untouched = untouched && sb[i] == 0xff;
        CHECK(untouched && lh[b].ext[0] == 0 && lh[b].ext[1] == 0 && lh[b].ext[2] == 0);
        continue;
      }
      const bool ok = std::equal(want[b].begin(), want[b].end(), got.begin() + b * sh.count());
      bool ok_ext = true;
      for (int a = 0; a < 3; ++a) ok_ext = ok_ext && lh[b].ext[a] == (a < sh.dim ? wext[b][a] : 0);
      CHECK(ok && ok_ext);
      if (!(ok && ok_ext)) {
        std::fprintf(stderr, "  flags %s dim %d b %lld launch_ids %lld\n", what, sh.dim, static_cast<long long>(b),
                     static_cast<long long>(ids));
      }
      bool pz = true;
      for (int64_t i = 0; i < g.field; ++i) {
        if (i % g.pitch >= g.n[g.dim - 1]) pz = pz && fp[b * g.field + i] == 0;
      }
      CHECK(pz);
    }
  }
}

// lines of one length as the elements of one call; element 1 (skipped) repeats element 0
template <class T>
void check_lines(const std::vector<std::vector<double>>& ls, const std::vector<std::vector<double>>& ts,
                 const char* what, bool scipy = true) {
  std::vector<std::vector<T>> rs, us;
  for (size_t k = 0; k < ls.size(); ++k) {
    std::vector<T> r(ls[k].size()), u(ts[k].size());
    for (size_t i = 0; i < r.size(); ++i) r[i] = static_cast<T>(ls[k][i]);
    for (size_t i = 0; i < u.size(); ++i) u[i] = static_cast<T>(ts[k][i]);
    rs.push_back(r);
    us.push_back(u);
    if (k == 0 && ls.size() > 1) {
      rs.push_back(r);
      us.push_back(u);
    }
  }
  check_flags<T>(shape(1, static_cast<int64_t>(ls[0].size())), rs, us, what, scipy);
}

float next_float(float v, int k) {
  for (; k > 0; --k) v = std::nextafter(v, std::numeric_limits<float>::infinity());
  for (; k < 0; ++k) v = std::nextafter(v, -std::numeric_limits<float>::infinity());
  return v;
}

template <class T>
void flags_all() {
  const double kNan = std::numeric_limits<double>::quiet_NaN();
  // the probed line: peaks 2 and 7 (prominences 2 and 5), the border plateau 4, 4 none; minima 4 and 9
  {
    const Shape sh = shape(1, 12);
    const double l[] = {0, 1, 3, 3, 1, 2, 5, 5, 5, 0, 4, 4};
    std::vector<T> r(12);
    for (int i = 0; i < 12; ++i) r[i] = static_cast<T>(l[i]);
    std::vector<double> ld(l, l + 12);
    const std::vector<int64_t> pk = local_maxima(ld);
    CHECK(pk.size() == 2 && pk[0] == 2 && pk[1] == 7);
    CHECK(prominence(ld, 2) == 2.0 && prominence(ld, 7) == 5.0);
    for (double tv : {0.0, 2.0, 2.5, 5.0, 5.5}) {
      check_flags<T>(sh, {r, r}, {std::vector<T>(12, static_cast<T>(tv)), std::vector<T>(12, static_cast<T>(tv / 2))},
                     "probe");
    }
    const eda::Geom g = geom(sh, 1);
    Dev<T> dx(pad(g, r)), dt(pad(g, std::vector<T>(12, T(2))));
    Dev<uint8_t> fl(g.field);
    Dev<T> scr(g.field);
    std::vector<eda::ElemRecord> le(1);
    std::memset(le.data(), 0, sizeof(eda::ElemRecord));
    Dev<eda::ElemRecord> dle(le);
    eda::enqueue_sig_flags<T>(nullptr, g, dx.p, dt.p, fl.p, dle.p, nullptr, scr.p);
    const std::vector<uint8_t> f = unpad(g, fl.get());
    // tau 2: maxima 2 (prom 2) and 7 (prom 5); minima (peaks of -l): 4 (prom 2) and 9 (prom 4)
    CHECK(f[2] == 1 && f[7] == 1 && f[9] == 1 && f[4] == 1 && f[1] == 0 && f[10] == 0 && f[11] == 0);
    CHECK(dle.get()[0].ext[0] == 4);
    bool refused = false;
    try {
      eda::enqueue_sig_flags<T>(nullptr, g, dx.p, dt.p, fl.p, dle.p, nullptr, dx.p);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    CHECK(refused);  // the scratch must not be r
  }
  // hand-made lines: plateaus (odd, even, at the ends), ties, monotone, constant, short lines
  {
    const std::vector<std::vector<double>> lines = {
        {1, 2, 2, 1, 0, 3, 3, 3, 3, 0, 1},  // even plateau 1, 2 -> peak 1; 5..8 -> 6
        {5, 5, 5, 1, 2, 2, 2, 7, 7},        // plateaus at both ends: no peak there
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, // monotone
        {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3},  // constant
        {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},  // equal neighbours alternate
        {0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5},  // a rising sawtooth: long left walks
        {-0.0, 0.0, -1, 0, 1, 1, 0, -0.0, 0, 2, 1}};
    for (const auto& l : lines) {
      const int64_t n = static_cast<int64_t>(l.size());
      std::vector<T> r(n);
      for (int64_t i = 0; i < n; ++i) r[i] = static_cast<T>(l[i]);
      for (double tv : {0.0, 0.5, 1.0, 1.5, 3.0}) {
        check_flags<T>(shape(1, n), {r}, {std::vector<T>(n, static_cast<T>(tv))}, "lines");
      }
    }
    for (int64_t n : {1, 2, 3}) {
      std::vector<T> r(n);
      for (int64_t i = 0; i < n; ++i) r[i] = static_cast<T>(i == 1 ? 1 : 0);
      check_flags<T>(shape(1, n), {r}, {std::vector<T>(n, T(0))}, "short");
    }
  }
  // the decisive sample of a side (deep: -1, or higher: 2) at offsets around the 32-sample block
  // edges from peaks at every position class of a block, between plateaus of 0 (tau 1.5)
  {
    const int64_t n = 300;
    std::vector<std::vector<double>> ls, ts;
    const int64_t offs[] = {1, 2, 31, 32, 33, 63, 64, 65};
    for (int64_t p : {96, 97, 111, 126, 127, 128}) {
      for (int64_t dl : offs) {
        for (int64_t dr : offs) {
          for (int kinds = 0; kinds < 4; ++kinds) {
            std::vector<double> l(n, 0.0);
            l[p] = 1.0;
            l[p - dl] = (kinds & 1) ? 2.0 : -1.0;
            l[p + dr] = (kinds & 2) ? 2.0 : -1.0;
            ls.push_back(l);
            ts.push_back(std::vector<double>(n, 1.5));
          }
        }
      }
    }
    check_lines<T>(ls, ts, "block edges");
  }
  // plateau peaks starting and ending at block edges, one spanning three blocks; maxima and minima
  {
    const int64_t n = 160;
    std::vector<std::vector<double>> ls, ts;
    for (int64_t s0 : {1, 5, 31, 32, 33}) {
      for (int64_t e0 : {31, 32, 33, 63, 64, 95, 96, 97, 99}) {
        if (e0 < s0) continue;
        for (double sg : {1.0, -1.0}) {
          for (double tv : {0.5, 1.0, 1.5}) {
            std::vector<double> l(n, 0.0);
            for (int64_t i = s0; i <= e0; ++i) l[i] = sg;
            l[e0 + 20] = -0.5 * sg;
            ls.push_back(l);
            ts.push_back(std::vector<double>(n, tv));
          }
        }
      }
    }
    check_lines<T>(ls, ts, "plateau edges");
  }
  // equal peaks: walks across blocks whose max equals the peak (3), with the decisive minimum
  // exactly v - tau at two far samples (the line ends included)
  {
    const int64_t n = 600;
    std::vector<std::vector<double>> ls, ts;
    for (int64_t far : {0, 100, 250}) {
      for (double tv : {5.0, 6.0, 6.5, 7.0}) {
        std::vector<double> l(n);
        for (int64_t i = 0; i < n; ++i) {
          l[i] = std::max(-2.0, std::round(3.0 * std::sin(6.283185307179586 * static_cast<double>(i) / 23.0)));
        }
        l[far] = 3.0 - tv;
        l[n - 1 - far] = 3.0 - tv;
        ls.push_back(l);
        ts.push_back(std::vector<double>(n, tv));
      }
    }
    check_lines<T>(ls, ts, "equal peaks");
  }
  // float ties at tau: groups [H, u, v, u] with tau(v) = fl32(v - u) and its two neighbours, v - u
  // inexact in float32, so that the double difference decides
  {
    Rng rng(91);
    const int64_t groups = 256, n = 4 * groups + 1;
    std::vector<std::vector<double>> ls, ts;
    for (int k = -1; k <= 1; ++k) {
      std::vector<double> l(n, 0.0), t(n, 0.0);
      for (int64_t g = 0; g < groups; ++g) {
        const float vv = static_cast<float>(std::ldexp(1.0 + rng.uniform(), static_cast<int>(rng.uniform() * 40) - 20));
        const double frac = std::ldexp(1.0 + rng.uniform(), -static_cast<int>(rng.uniform() * 40));
        const float uu = static_cast<float>(static_cast<double>(vv) - frac * static_cast<double>(vv));
        const float d = static_cast<float>(static_cast<double>(vv) - static_cast<double>(uu));
        l[4 * g] = 1e30;
        t[4 * g] = 1e38;
        l[4 * g + 1] = uu;
        t[4 * g + 1] = rng.uniform();
        l[4 * g + 2] = vv;
        t[4 * g + 2] = next_float(d, k);
        l[4 * g + 3] = uu;
        t[4 * g + 3] = rng.uniform();
      }
      l[n - 1] = 1e30;
      ls.push_back(l);
      ts.push_back(t);
    }
    check_lines<T>(ls, ts, "float ties");
  }
  // float ties decided by a block summary: [w.., u, w x K, v, w x K, u, w..] with w just below v,
  // so that the walk passes every w and the block holding u (a block past v's own) decides with
  // tau(v) = fl32(v - u) and its neighbours; maxima and minima
  {
    Rng rng(92);
    std::vector<std::vector<double>> ls, ts;
    for (int64_t K : {40, 64, 100, 200}) {
      const int64_t n = 2 * K + 16;
      for (int rep = 0; rep < 16; ++rep) {
        for (int k = -1; k <= 1; ++k) {
          for (double sg : {1.0, -1.0}) {
            const float vv = static_cast<float>(1.0 + rng.uniform());
            const int e = static_cast<int>(rng.uniform() * 46) - 25;  // |u| from 2^-25 to 2^20
            const float uu = static_cast<float>(-std::ldexp(1.0 + rng.uniform(), e));
            const float ww = std::nextafter(vv, -std::numeric_limits<float>::infinity());
            const float d = static_cast<float>(static_cast<double>(vv) - static_cast<double>(uu));
            std::vector<double> l(n, sg * ww), t(n, 0.0);
            for (auto& v : t) v = rng.uniform() * 0.6;
            l[5] = sg * uu;
            l[6 + K] = sg * vv;
            l[7 + 2 * K] = sg * uu;
            t[6 + K] = next_float(d, k);
            ls.push_back(l);
            ts.push_back(t);
          }
        }
      }
      check_lines<T>(ls, ts, "block float ties");
      ls.clear();
      ts.clear();
    }
  }
  // subnormal differences and thresholds (flush-to-zero off), signed zeros
  {
    Rng rng(93);
    const double dm = static_cast<double>(std::numeric_limits<T>::denorm_min());
    const double vals[] = {0.0, -0.0, dm, -dm, 2 * dm, -2 * dm, 3 * dm, 1.0};
    std::vector<std::vector<double>> ls, ts;
    for (double tv : {dm, 2 * dm, 0.0, -0.0}) {
      for (int rep = 0; rep < 4; ++rep) {
        std::vector<double> l(200);
        for (auto& v : l) v = vals[static_cast<int>(rng.uniform() * 8)];
        ls.push_back(l);
        ts.push_back(std::vector<double>(200, tv));
      }
    }
    check_lines<T>(ls, ts, "subnormal");
  }
  // infinities: noise with +-inf samples and runs of +inf, an inf peak; thresholds up to +inf
  {
    Rng rng(94);
    const double inf = kInf;
    std::vector<std::vector<double>> ls, ts;
    for (double tv : {0.5, 1e30, inf}) {
      for (int rep = 0; rep < 4; ++rep) {
        std::vector<double> l(300);
        for (auto& v : l) v = rng.uniform() < 0.05 ? (rng.uniform() < 0.5 ? inf : -inf) : rng.uniform() * 2 - 1;
        if (rep == 1) for (int i = 100; i < 140; ++i) l[i] = inf;
        if (rep == 2) {
          l[40] = 0;
          l[41] = inf;
          l[42] = 0;
        }
        ls.push_back(l);
        ts.push_back(std::vector<double>(300, tv));
      }
    }
    check_lines<T>(ls, ts, "inf");
  }
  // NaN lines: a NaN ends a side like a higher sample (SciPy's rule)
  {
    Rng rng(95);
    std::vector<std::vector<double>> ls, ts;
    for (double pr : {0.002, 0.05}) {
      for (int rep = 0; rep < 6; ++rep) {
        std::vector<double> l(400);
        for (int64_t i = 0; i < 400; ++i) l[i] = rep < 3 ? rng.uniform() * 2 - 1 : std::round(3 * std::sin(0.3 * i));
        for (auto& v : l) if (rng.uniform() < pr) v = kNan;
        ls.push_back(l);
        ts.push_back(std::vector<double>(400, rep % 3 == 0 ? 0.3 : rep % 3 == 1 ? 1.0 : 1e-3));
      }
    }
    check_lines<T>(ls, ts, "nan");
  }
  // hard lines: a rising zigzag whose insignificant peaks walk to the line start, with plateaus,
  // next to easy lines; along the contiguous and the outer axes
  {
    const int64_t n = g_small ? 1500 : 3000;
    const Shape sh1 = shape(1, n);
    std::vector<T> zig(n), noise = make_field<T>(sh1, 0, 77);
    for (int64_t i = 0; i < n; ++i) zig[i] = static_cast<T>(i * 1e-3 + 3e-3 * (i % 2) + (i % 97 == 0 ? 0.5 : 0.0));
    for (int64_t i = 10; i < 20; ++i) zig[i] = zig[9];  // a plateau
    Rng rng(78);
    std::vector<T> tmix(n);
    for (auto& v : tmix) v = static_cast<T>(rng.uniform() < 0.98 ? 10.0 : 0.001);
    check_flags<T>(sh1, {zig, noise, zig}, {std::vector<T>(n, T(1)), std::vector<T>(n, T(0.3)), tmix}, "hard 1d");
    const Shape sh2 = shape(2, n / 10, 33);
    std::vector<T> z2(static_cast<size_t>(sh2.count()));
    for (int64_t i = 0; i < sh2.n[0]; ++i) {
      for (int64_t j = 0; j < 33; ++j) z2[i * 33 + j] = static_cast<T>(i * 1e-3 + 3e-3 * ((i + j) % 2) + 1e-4 * j);
    }
    check_flags<T>(sh2, {z2, z2}, {std::vector<T>(z2.size(), T(1)), std::vector<T>(z2.size(), T(0.002))}, "hard 2d");
  }
  // one long line (no line limit): noise, a zigzag and plateaus, by the stack twin
  {
    const int64_t n = g_small ? (int64_t{1} << 16) : (int64_t{1} << 20);
    const Shape sh = shape(1, n);
    std::vector<T> zig(n), q = make_field<T>(sh, 1, 79);
    for (int64_t i = 0; i < n; ++i) zig[i] = static_cast<T>((i % 65536) * 1e-3 + 3e-3 * (i % 2));
    check_flags<T>(sh, {make_field<T>(sh, 0, 80), zig, q},
                   {std::vector<T>(n, T(0.3)), std::vector<T>(n, T(1)), std::vector<T>(n, T(1))}, "long", false);
  }
  // random fields (noise, quantized ties, waves, constant) with random per-voxel thresholds; every
  // axis longer than 32 in 37 x 45 x 70
  const Shape shapes[] = {shape(1, g_small ? 300 : 3000), shape(2, 37, 45), shape(3, 9, 11, 14), shape(2, 1, 50),
                          shape(3, 7, 1, 5), shape(3, 37, 45, 70)};
  uint64_t seed = 500;
  for (const Shape& sh : shapes) {
    if (g_small && sh.count() > 20000) continue;
    std::vector<std::vector<T>> rs, ts;
    for (int b = 0; b < 4; ++b) {
      rs.push_back(make_field<T>(sh, b, seed++));
      Rng rng(seed++);
      std::vector<T> t(static_cast<size_t>(sh.count()));
      for (auto& v : t) v = static_cast<T>(b == 1 ? std::floor(rng.uniform() * 3.0) : rng.uniform() * 0.6);
      ts.push_back(t);
    }
    check_flags<T>(sh, rs, ts, "random");
  }
}

// ---- raw window -----------------------------------------------------------------------------------------
template <class T>
std::vector<T> twin_raw(const Shape& sh, const std::vector<uint8_t>& fl, int32_t W, double tiny) {
  std::vector<T> acc(fl.size(), T(0));
  for (int a = 0; a < sh.dim; ++a) {
    std::vector<T> f(fl.size());
    for (size_t i = 0; i < fl.size(); ++i) f[i] = ((fl[i] >> a) & 1u) ? T(1) : T(0);
    f = box(sh, f, W);
    for (size_t i = 0; i < fl.size(); ++i) acc[i] = a == 0 ? mul_rn(f[i], f[i]) : add_rn(acc[i], mul_rn(f[i], f[i]));
  }
  std::vector<T> om(fl.size());
  const T tn = static_cast<T>(tiny);
  for (size_t i = 0; i < fl.size(); ++i) {
    const T d = std::sqrt(acc[i]);
    om[i] = d > T(0) ? T(2) / (d > tn ? d : tn) : std::numeric_limits<T>::infinity();
  }
  return om;
}

std::vector<uint8_t> random_flags(const Shape& sh, double p, uint64_t seed, int dim) {
  Rng rng(seed);
  std::vector<uint8_t> f(static_cast<size_t>(sh.count()));
  for (auto& v : f) {
    uint8_t m = 0;
    for (int a = 0; a < dim; ++a) {
      if (rng.uniform() < p) m |= static_cast<uint8_t>(1u << a);
    }
    v = m;
  }
  return f;
}

template <class T>
void raw_window(const Shape& sh, const std::vector<int32_t>& W, uint64_t seed) {
  const int64_t B = static_cast<int64_t>(W.size());
  const eda::Geom g = geom(sh, B);
  const eda::MethodCtl c = eda::method_ctl(g, dtype_of<T>());
  std::vector<uint8_t> fl;
  for (int64_t b = 0; b < B; ++b) {
    const std::vector<uint8_t> f = random_flags(sh, b == 0 ? 0.0 : 0.02 * b, seed + b, sh.dim);
    fl.insert(fl.end(), f.begin(), f.end());
  }
  Dev<uint8_t> df(pad(g, fl));
  Dev<int32_t> dw(W);
  Dev<T> om(B * g.field), tmp(B * g.field);
  cudaMemset(om.p, 0xff, om.n * sizeof(T));
  eda::enqueue_raw_window<T>(nullptr, g, c, df.p, dw.p, om.p, tmp.p, nullptr);
  const std::vector<T> gp = om.get();
  CHECK(pads_zero(g, gp) || std::find(W.begin(), W.end(), 0) != W.end());
  const std::vector<T> got = unpad(g, gp);
  for (int64_t b = 0; b < B; ++b) {
    const std::vector<T> gb = elem(got, b, sh.count());
    if (!W[b]) {
      CHECK(std::isnan(static_cast<double>(gb[0])));
      continue;
    }
    const std::vector<T> want = twin_raw<T>(sh, elem(fl, b, sh.count()), W[b], c.tiny);
    const bool ok = same_bits(gb.data(), want.data(), want.size());
    CHECK(ok);
    if (!ok) std::fprintf(stderr, "  raw dim %d b %lld W %d\n", sh.dim, static_cast<long long>(b), W[b]);
  }
}

// ---- components ------------------------------------------------------------------------------------------
std::vector<int64_t> twin_ccl(const Shape& sh, const std::vector<uint8_t>& m, uint8_t bits) {
  const int64_t n = sh.count();
  std::vector<int64_t> lab(n, -1);
  for (int64_t s = 0; s < n; ++s) {
    if (!(m[s] & bits) || lab[s] >= 0) continue;
    std::deque<int64_t> q{s};
    lab[s] = s;  // s is the smallest member: the scan runs in index order
    while (!q.empty()) {
      const int64_t v = q.front();
      q.pop_front();
      for (int a = 0; a < sh.dim; ++a) {
        const int64_t st = sh.stride(a), c = (v / st) % sh.n[a];
        for (int d : {-1, 1}) {
          if ((d < 0 && c == 0) || (d > 0 && c + 1 >= sh.n[a])) continue;
          const int64_t u = v + d * st;
          if ((m[u] & bits) && lab[u] < 0) {
            lab[u] = s;
            q.push_back(u);
          }
        }
      }
    }
  }
  return lab;
}

void check_ccl(const Shape& sh, const std::vector<std::vector<uint8_t>>& ms, uint8_t bits, const char* what) {
  const int64_t B = static_cast<int64_t>(ms.size());
  const eda::Geom g = geom(sh, B);
  std::vector<uint8_t> m;
  for (const auto& x : ms) m.insert(m.end(), x.begin(), x.end());
  Dev<uint8_t> dm(pad(g, m));
  Dev<int32_t> lab(B * g.field);
  Dev<char> ws(eda::ccl_workspace_bytes(g));
  eda::enqueue_ccl(nullptr, g, dm.p, bits, lab.p, ws.p, nullptr);
  const std::vector<int32_t> got = unpad(g, lab.get());
  for (int64_t b = 0; b < B; ++b) {
    const std::vector<int64_t> want = twin_ccl(sh, ms[b], bits);
    bool ok = true;
    for (int64_t u = 0; u < sh.count(); ++u) {
      const int32_t gv = got[b * sh.count() + u];
      const int64_t wv = want[u] < 0 ? -1 : padded_of(g, want[u]);
      ok = ok && gv == wv;
    }
    CHECK(ok);
    if (!ok) std::fprintf(stderr, "  ccl %s dim %d b %lld\n", what, sh.dim, static_cast<long long>(b));
  }
}

std::vector<uint8_t> spiral(int64_t n) {
  std::vector<uint8_t> m(static_cast<size_t>(n * n), 0);
  int64_t lo = 0, hi = n - 1;
  while (lo <= hi) {  // a square spiral wall of width 1 with gaps of 1
    for (int64_t i = lo; i <= hi; ++i) m[lo * n + i] = 1;
    for (int64_t i = lo; i <= hi; ++i) m[i * n + hi] = 1;
    for (int64_t i = lo; i <= hi; ++i) m[hi * n + i] = 1;
    for (int64_t i = lo + 2; i <= hi; ++i) m[i * n + lo] = 1;
    if (lo + 2 <= hi) m[(lo + 2) * n + lo + 1] = 1;
    lo += 2;
    hi -= 2;
  }
  return m;
}

void ccl_all() {
  for (const Shape& sh : {shape(1, 1000), shape(2, 41, 67), shape(3, 13, 9, 35), shape(2, 1, 40), shape(3, 1, 6, 1)}) {
    std::vector<std::vector<uint8_t>> ms;
    for (double p : {0.3, 0.55, 0.7}) ms.push_back(random_flags(sh, p, static_cast<uint64_t>(p * 100) + sh.dim, 2));
    ms.push_back(std::vector<uint8_t>(static_cast<size_t>(sh.count()), 3));  // full
    ms.push_back(std::vector<uint8_t>(static_cast<size_t>(sh.count()), 0));  // empty
    std::vector<uint8_t> one(static_cast<size_t>(sh.count()), 0);
    one[sh.count() / 2] = 1;  // a single voxel
    ms.push_back(one);
    check_ccl(sh, ms, 1, "random");
    check_ccl(sh, ms, 2, "random bit 1");
  }
  const int64_t n = g_small ? 33 : 129;
  check_ccl(shape(2, n, n), {spiral(n), spiral(n)}, 1, "spiral");
  // components touching the border, a diagonal pair (two labels)
  std::vector<uint8_t> d(36, 0);
  d[0] = d[7] = d[5] = d[11] = d[30] = d[35] = 1;
  check_ccl(shape(2, 6, 6), {d}, 1, "border");
}

// ---- the scale field and the floor ------------------------------------------------------------------------
template <class T>
T host_log(T v) {
  return std::log(v);
}
template <class T>
T host_exp(T v) {
  return std::exp(v);
}

template <class T>
void twin_scale(const Shape& sh, const std::vector<T>& om, int32_t w, double jump, std::vector<uint8_t>& A,
                std::vector<T>& L) {
  const int32_t W = eda::odd_window(2.0 * w);
  const size_t n = om.size();
  std::vector<T> fine(n);
  std::vector<uint8_t> aw(n);
  for (size_t i = 0; i < n; ++i) {
    const double o = static_cast<double>(om[i]);
    aw[i] = o > 1.2 * w;
    fine[i] = o < (1.2 * w) / jump ? T(1) : T(0);
  }
  const std::vector<T> near = cube_mm(sh, fine, eda::odd_window(2.0 * 1.0 * W), kMax);
  const std::vector<int64_t> lab = twin_ccl(sh, aw, 1);
  std::vector<uint8_t> keep(n, 0);
  for (size_t i = 0; i < n; ++i) {
    if (aw[i] && near[i] > T(0.5)) keep[lab[i]] = 1;
  }
  A.assign(n, 0);
  std::vector<T> lw(n);
  for (size_t i = 0; i < n; ++i) {
    const double o = static_cast<double>(om[i]);
    A[i] = o > 2.0 * w && lab[i] >= 0 && keep[lab[i]];
    double v = 3.0;
    if (!A[i]) {
      const double ow = o > 2.0 * w ? static_cast<double>(w) : o;
      v = std::min(std::max(ow, w / 1.5), 2.0 * w);
    }
    lw[i] = host_log<T>(static_cast<T>(v));
  }
  lw = box(sh, lw, W);
  L.resize(n);
  for (size_t i = 0; i < n; ++i) L[i] = host_exp<T>(lw[i]);
}

// om fields with absent regions (+inf, large values), fine pockets and a slow drift
template <class T>
std::vector<T> make_om(const Shape& sh, int32_t w, uint64_t seed, int kind) {
  Rng rng(seed);
  std::vector<T> om(static_cast<size_t>(sh.count()));
  for (int64_t u = 0; u < sh.count(); ++u) {
    int64_t rem = u;
    double c[3] = {0, 0, 0};
    for (int a = sh.dim - 1; a >= 0; --a) {
      c[a] = static_cast<double>(rem % sh.n[a]) / static_cast<double>(sh.n[a]);
      rem /= sh.n[a];
    }
    double v;
    if (kind == 0) {  // blocks: absent (inf / 5 w), weak (1.5 w), fine pockets (0.3 w), normal
      const int blk = static_cast<int>(std::floor(c[0] * 4) + 4 * std::floor(c[1] * 3) + 12 * std::floor(c[2] * 2));
      const double lv[] = {kInf, 5.0, 1.5, 0.3, 1.0, 2.5, 0.4, kInf, 1.3, 0.9, 3.0, 0.2};
      v = lv[blk % 12] * w * (1.0 + 0.2 * rng.uniform());
    } else {  // random
      v = rng.uniform() < 0.1 ? kInf : w * 4.0 * rng.uniform();
    }
    om[u] = static_cast<T>(v);
  }
  return om;
}

template <class T>
bool close_rel(const std::vector<T>& a, const std::vector<T>& b, double rtol) {
  for (size_t i = 0; i < a.size(); ++i) {
    const double x = static_cast<double>(a[i]), y = static_cast<double>(b[i]);
    if (!(std::fabs(x - y) <= rtol * std::fabs(y))) return false;
  }
  return true;
}

template <class T>
void scale_and_floor(const Shape& sh, const std::vector<int32_t>& w, uint64_t seed) {
  const int64_t B = static_cast<int64_t>(w.size());
  const eda::Geom g = geom(sh, B);
  const eda::MethodCtl c = eda::method_ctl(g, dtype_of<T>());
  const double rtol = sizeof(T) == 4 ? 2e-6 : 1e-13;  // device vs host log / exp
  std::vector<T> om, Lprev;
  std::vector<eda::ElemRecord> le(B);
  std::memset(le.data(), 0, le.size() * sizeof(eda::ElemRecord));
  std::vector<int32_t> on(B, 1);
  for (int64_t b = 0; b < B; ++b) {
    const std::vector<T> o = make_om<T>(sh, w[b], seed + b, static_cast<int>(b % 2));
    om.insert(om.end(), o.begin(), o.end());
    Rng rng(seed + 100 + b);
    for (int64_t i = 0; i < sh.count(); ++i) Lprev.push_back(static_cast<T>(3.0 + 2.0 * w[b] * rng.uniform()));
    le[b].w = w[b];
    le[b].W = eda::odd_window(2.0 * w[b]);
    le[b].jump = b % 3 == 2 ? 1.0 : 2.0;
    le[b].k = static_cast<int32_t>(b % 4);
  }
  if (B > 1) on[B - 1] = 0;
  Dev<T> dom(pad(g, om)), L(B * g.field), t0(B * g.field), t1(B * g.field), dlp(pad(g, Lprev));
  cudaMemset(L.p, 0xff, L.n * sizeof(T));
  Dev<uint8_t> A(B * g.field);
  Dev<int32_t> lab(B * g.field), don(on);
  Dev<char> ws(eda::ccl_workspace_bytes(g));
  Dev<eda::ElemRecord> dle(le);
  eda::enqueue_scale_field<T>(nullptr, g, c, dle.p, dom.p, A.p, L.p, t0.p, t1.p, lab.p, ws.p, don.p);
  const std::vector<uint8_t> Ag = unpad(g, A.get());
  const std::vector<T> Lp_ = L.get();
  const std::vector<T> Lg = unpad(g, Lp_);
  std::vector<std::vector<T>> Lw(B);
  std::vector<std::vector<uint8_t>> Aw(B);
  for (int64_t b = 0; b < B; ++b) {
    if (!on[b]) {
      CHECK(std::isnan(static_cast<double>(Lg[b * sh.count()])));
      continue;
    }
    twin_scale<T>(sh, elem(om, b, sh.count()), w[b], le[b].jump, Aw[b], Lw[b]);
    const bool okA = std::equal(Aw[b].begin(), Aw[b].end(), Ag.begin() + b * sh.count());
    const bool okL = close_rel(elem(Lg, b, sh.count()), Lw[b], rtol);
    CHECK(okA && okL);
    if (!(okA && okL)) std::fprintf(stderr, "  scale dim %d b %lld A %d L %d\n", sh.dim, static_cast<long long>(b),
                                    okA, okL);
  }
  // the floor on the device's L and A (k >= 1 only; k = 0 untouched)
  eda::enqueue_local_floor<T>(nullptr, g, c, dle.p, dlp.p, A.p, L.p, t0.p, don.p);
  const std::vector<T> Lf = unpad(g, L.get());
  for (int64_t b = 0; b < B; ++b) {
    if (!on[b]) continue;
    const std::vector<T> L0 = elem(Lg, b, sh.count());
    if (le[b].k < 1) {
      CHECK(same_bits(elem(Lf, b, sh.count()).data(), L0.data(), L0.size()));
      continue;
    }
    std::vector<T> chi(L0.size());
    const std::vector<uint8_t> Ab = elem(Ag, b, sh.count());
    for (size_t i = 0; i < chi.size(); ++i) chi[i] = Ab[i] ? T(1) : T(0);
    chi = box(sh, chi, le[b].W);
    std::vector<T> want(L0.size());
    const std::vector<T> lp = elem(Lprev, b, sh.count());
    for (size_t i = 0; i < want.size(); ++i) {
      T fl = le[b].k >= 2 ? mul_rn(T(2), lp[i]) : add_rn(lp[i], T(2));
      fl = std::min(fl, static_cast<T>(2.0 * w[b]));
      const T e = add_rn(mul_rn(sub_rn(T(1), chi[i]), host_log<T>(fl)), mul_rn(chi[i], static_cast<T>(std::log(3.0))));
      const T f = host_exp<T>(e);
      want[i] = L0[i] >= f ? L0[i] : f;
    }
    const bool ok = close_rel(elem(Lf, b, sh.count()), want, rtol);
    CHECK(ok);
    if (!ok) std::fprintf(stderr, "  floor dim %d b %lld\n", sh.dim, static_cast<long long>(b));
  }
}

// ---- G2's rung test and G3 ------------------------------------------------------------------------------
struct G2Twin {
  int32_t outcome;
  double p, cov;
  int64_t n_p;
  std::vector<uint8_t> P;
};

template <class T>
G2Twin twin_g2(const Shape& sh, const std::vector<uint8_t>& ff, int32_t w, int32_t s) {
  G2Twin t{};
  const int32_t Wb = eda::odd_window(2.0 * s);
  const std::vector<T> om = twin_raw<T>(sh, ff, Wb, sizeof(T) == 4 ? 1.1754943508222875e-38 : 1e-300);
  const size_t n = om.size();
  t.P.assign(n, 0);
  std::vector<T> pf(n);
  std::vector<double> sel;
  for (size_t i = 0; i < n; ++i) {
    t.P[i] = static_cast<double>(om[i]) < 1.5 * s;
    pf[i] = t.P[i] ? T(1) : T(0);
    if (t.P[i]) sel.push_back(static_cast<double>(om[i]));
  }
  t.n_p = static_cast<int64_t>(sel.size());
  t.cov = static_cast<double>(t.n_p) / static_cast<double>(n);
  t.p = twin_quantile(sel, 0.25);
  t.outcome = eda::kG2None;
  if (sel.empty()) return t;
  const std::vector<T> core = cube_mm(sh, pf, eda::odd_window(2.0 * s), kMin);
  if (!std::any_of(core.begin(), core.end(), [](T v) { return v > T(0.5); })) return t;
  if (2.0 * t.p > w) {
    t.outcome = eda::kG2Stop;
    return t;
  }
  const std::vector<T> dil = cube_mm(sh, pf, eda::odd_window(2.0 * 1.0 * Wb), kMax);
  for (size_t i = 0; i < n; ++i) {
    if (static_cast<double>(om[i]) > 2.0 * t.p && dil[i] > T(0.5)) t.outcome = eda::kG2Accepted;
  }
  return t;
}

// a fine population (flags every `fine` samples along every axis) in a box, a coarse one elsewhere
std::vector<uint8_t> population(const Shape& sh, int fine, int coarse, double lo, double hi, uint64_t seed) {
  Rng rng(seed);
  std::vector<uint8_t> f(static_cast<size_t>(sh.count()), 0);
  for (int64_t u = 0; u < sh.count(); ++u) {
    int64_t rem = u;
    int64_t c[3] = {0, 0, 0};
    for (int a = sh.dim - 1; a >= 0; --a) {
      c[a] = rem % sh.n[a];
      rem /= sh.n[a];
    }
    bool inside = true;
    for (int a = 0; a < sh.dim; ++a) {
      const double x = static_cast<double>(c[a]) / static_cast<double>(sh.n[a]);
      inside = inside && x >= lo && x < hi;
    }
    for (int a = 0; a < sh.dim; ++a) {
      const int per = inside ? fine : coarse;
      if ((c[a] + static_cast<int64_t>(rng.next() % 2)) % per == 0) f[u] |= static_cast<uint8_t>(1u << a);
    }
  }
  return f;
}

template <class T>
void g2_and_g3(const Shape& sh, int coverage[3]) {
  struct Case {
    int fine, coarse;
    double lo, hi;
    int32_t w, s;
  };
  const Case cases[] = {{2, 40, 0.2, 0.5, 41, 3}, {2, 40, 0.2, 0.5, 41, 5}, {3, 30, 0.1, 0.6, 61, 7},
                        {2, 9, 0.3, 0.7, 13, 3},  {4, 50, 0.0, 0.3, 21, 9}, {2, 3, 0.0, 1.0, 9, 3},
                        {6, 7, 0.2, 0.9, 7, 3},   {2, 40, 0.2, 0.5, 41, 13}, {1, 40, 0.0, 1.0, 3, 3}};
  const int64_t B = static_cast<int64_t>(sizeof cases / sizeof cases[0]) + 1;  // + one skipped element
  const eda::Geom g = geom(sh, B);
  const eda::MethodCtl c = eda::method_ctl(g, dtype_of<T>());
  std::vector<uint8_t> ff;
  std::vector<eda::ElemRecord> le(B);
  std::memset(le.data(), 0, le.size() * sizeof(eda::ElemRecord));
  std::vector<int32_t> srung(B, 0);
  for (int64_t b = 0; b < B; ++b) {
    const Case& k = cases[b < B - 1 ? b : 0];
    const std::vector<uint8_t> f = population(sh, k.fine, k.coarse, k.lo, k.hi, 900 + b);
    ff.insert(ff.end(), f.begin(), f.end());
    le[b].w = k.w;
    le[b].status = eda::kRecordActive;
    srung[b] = b < B - 1 ? k.s : 0;
  }
  Dev<uint8_t> dff(pad(g, ff)), P(B * g.field), sel(B * g.field);
  cudaMemset(P.p, 0xee, P.n);
  Dev<T> t0(B * g.field), t1(B * g.field);
  Dev<eda::ElemRecord> dle(le);
  Dev<int32_t> ds(srung);
  std::vector<eda::G2Result> init(B);
  std::memset(init.data(), 0x5a, init.size() * sizeof(eda::G2Result));
  Dev<eda::G2Result> res(init);
  Dev<char> ws(eda::select_workspace_bytes(g));
  eda::enqueue_g2_test<T>(nullptr, g, c, dle.p, ds.p, dff.p, P.p, t0.p, t1.p, res.p, ws.p);
  const std::vector<eda::G2Result> rh = res.get();
  const std::vector<uint8_t> Pp = P.get();
  const std::vector<uint8_t> Pg = unpad(g, Pp);
  for (int64_t b = 0; b < B; ++b) {
    if (!srung[b]) {
      CHECK(std::memcmp(&rh[b], &init[b], sizeof(eda::G2Result)) == 0);
      bool untouched = true;
      for (int64_t i = 0; i < g.field; ++i) untouched = untouched && Pp[b * g.field + i] == 0xee;
      CHECK(untouched);
      continue;
    }
    const G2Twin t = twin_g2<T>(sh, elem(ff, b, sh.count()), le[b].w, srung[b]);
    const bool okP = std::equal(t.P.begin(), t.P.end(), Pg.begin() + b * sh.count());
    const bool ok = rh[b].outcome == t.outcome && same_double(rh[b].p, t.p) && same_double(rh[b].cov, t.cov) &&
                    rh[b].n_p == t.n_p && rh[b].s == srung[b] && okP;
    CHECK(ok);
    if (!ok) std::fprintf(stderr, "  g2 dim %d b %lld: outcome %d/%d p %.17g/%.17g n_p %lld/%lld P %d\n", sh.dim,
                          static_cast<long long>(b), rh[b].outcome, t.outcome, rh[b].p, t.p,
                          static_cast<long long>(rh[b].n_p), static_cast<long long>(t.n_p), okP);
    if (t.outcome >= 0 && t.outcome < 3) ++coverage[t.outcome];
  }
  // G3 on the populations just found: elements with fine = 1, p_odd = odd(p)
  std::vector<eda::ElemRecord> l3 = le;
  for (int64_t b = 0; b < B; ++b) {
    l3[b].fine = (srung[b] && rh[b].n_p > 0) ? 1 : 0;
    l3[b].p_odd = l3[b].fine ? eda::odd_window(rh[b].p) : 0;
  }
  Dev<eda::ElemRecord> dl3(l3);
  std::vector<double> wc0(B, -7.0);
  Dev<double> wc(wc0);
  eda::enqueue_g3_coarse<T>(nullptr, g, c, dl3.p, dff.p, P.p, wc.p, t0.p, t1.p, sel.p, ws.p);
  const std::vector<double> wg = wc.get();
  for (int64_t b = 0; b < B; ++b) {
    if (!l3[b].fine) {
      CHECK(wg[b] == -7.0);
      continue;
    }
    const std::vector<uint8_t> ffb = elem(ff, b, sh.count());
    const std::vector<T> omc = twin_raw<T>(sh, ffb, eda::odd_window(2.0 * l3[b].w), c.tiny);
    std::vector<T> pf(omc.size());
    for (size_t i = 0; i < pf.size(); ++i) pf[i] = Pg[b * sh.count() + i] ? T(1) : T(0);
    const std::vector<T> dil = cube_mm(sh, pf, eda::odd_window(2.0 * 1.0 * eda::odd_window(2.0 * l3[b].p_odd)), kMax);
    std::vector<double> v;
    for (size_t i = 0; i < omc.size(); ++i) {
      if (dil[i] < T(0.5) && std::isfinite(static_cast<double>(omc[i]))) v.push_back(static_cast<double>(omc[i]));
    }
    const double want = v.empty() ? kInf : twin_median(v);
    CHECK(same_double(wg[b], want));
  }
}

template <class T>
void all() {
  selections<T>();
  noise_levels<T>(shape(1, 513), 3, 31);
  noise_levels<T>(shape(2, 31, 47), 3, 32);
  noise_levels<T>(shape(3, 9, 1, 21), 2, 33);
  noise_levels<T>(shape(3, 11, 12, 13), 2, 34);
  tau_field<T>(shape(1, g_small ? 200 : 1500), {7, 0, 33, 101}, 41);
  tau_field<T>(shape(2, 37, 45), {3, 0, 11}, 42);
  tau_field<T>(shape(3, 11, 9, 14), {5, 9, 0, 3}, 43);
  flags_all<T>();
  raw_window<T>(shape(1, 700), {7, 0, 45, 1401}, 51);
  raw_window<T>(shape(2, 37, 45), {9, 0, 31}, 52);
  raw_window<T>(shape(3, 11, 9, 14), {5, 13}, 53);
  scale_and_floor<T>(shape(1, 600), {9, 21, 33, 15}, 61);
  scale_and_floor<T>(shape(2, 48, 52), {5, 9, 7, 3}, 62);
  scale_and_floor<T>(shape(3, 14, 12, 16), {3, 5, 7}, 63);
  int cov[3] = {0, 0, 0};
  g2_and_g3<T>(shape(1, 512), cov);
  g2_and_g3<T>(shape(2, 64, 64), cov);
  if (!g_small) g2_and_g3<T>(shape(3, 24, 24, 24), cov);
  CHECK(cov[eda::kG2None] > 0 && cov[eda::kG2Accepted] > 0 && cov[eda::kG2Stop] > 0);
  if (!(cov[0] && cov[1] && cov[2])) std::fprintf(stderr, "  g2 coverage %d %d %d\n", cov[0], cov[1], cov[2]);
}

}  // namespace

int main() {
  const char* sz = std::getenv("EDA_SANITIZE");
  g_small = sz != nullptr && sz[0] == '1';
  try {
    eda::Context& ctx = eda::Context::current(0);
    (void)ctx;
    quantile_table();
    ccl_all();
    all<float>();
    all<double>();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "test_stats: %s\n", error.what());
    eda::Context::shutdown();
    return 1;
  }
  eda::Context::shutdown();
  return eda_test::finish("test_stats");
}
