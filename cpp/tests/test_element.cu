// SPDX-License-Identifier: Apache-2.0
// test_element: the element (method/element.hpp) against the host twins of twin_element.hpp,
// float32 and float64, D = 1, 2, 3, batches:
//   - the eigen solver, fed with the boxed structure-tensor products of the host twin (gradient
//     products boxed by the vHGW twin), against a long-double Jacobi reference (|dc| <= 1e-12,
//     |n . n_ref| >= 1 - 1e-12 where c > 1e-6, float64); the W = 0 skip;
//   - rung_ladder against a table computed in double precision (rungs_table.inc), and its
//     length_error;
//   - the build (o, d_min, d_max), the hat weights (lo, t bitwise against a twin fed with the
//     device's logs), rung_used / rung_pts and the rung windows;
//   - the operators max / max+min / box and the envelope mean bitwise against the twin of
//     the cube + gather arithmetic (coherent and isotropic regions, points that leave the domain,
//     one rung, weights exactly 0 and 1, different ladders in one batch, a skipped element left
//     untouched), one rung at a time and with rung scratch for groups of 2, 3 and kMaxRungs rungs,
//     aliasing rejected;
//   - the flip of an axis: 1D bitwise for every op; D >= 2 with exact (dyadic) point coordinates
//     max / min bitwise and box within 2 ulp; with general orientations, the measured deviation.
#include "twin.hpp"
#include "twin_element.hpp"
#include "runtime/env.hpp"
#include "method/method.hpp"
#include "method/element.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

using host_twin::Shape;

namespace {

bool g_small = false;
int64_t g_exact01 = 0;  // voxels with a hat weight exactly 0 or 1 (on a rung)

template <class T>
struct Buf {
  T* p = nullptr;
  size_t n = 0;
  // zeroed: get() copies whole padded buffers, pads included, which the kernels never write
  // (initcheck reports the copy of never-written bytes as a host API error otherwise)
  explicit Buf(size_t n_) : n(n_) {
    CHECK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)) == cudaSuccess);
    CHECK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T)) == cudaSuccess);
  }
  ~Buf() { cudaFree(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  void put(const std::vector<T>& h) { CHECK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess); }
  std::vector<T> get() const {
    std::vector<T> h(n);
    CHECK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess);
    return h;
  }
  void bytes(int v) { CHECK(cudaMemset(p, v, n * sizeof(T)) == cudaSuccess); }
};

Shape shape(int dim, int64_t a, int64_t b = 1, int64_t c = 1) {
  Shape s;
  s.dim = dim;
  s.n[0] = a;
  s.n[1] = b;
  s.n[2] = c;
  return s;
}

// Host fields are unpadded, element-major with the components inside: h[(b * nc + c) * count + i];
// device fields component-major: d[(c * B + b) * field + padded(i)].
template <class T>
std::vector<T> to_dev(const eda::Geom& g, const std::vector<T>& h, int nc, T pad = T(0)) {
  std::vector<T> d(static_cast<size_t>(nc * g.batch * g.field), pad);
  const int64_t nl = g.n[g.dim - 1];
  for (int64_t b = 0; b < g.batch; ++b) {
    for (int c = 0; c < nc; ++c) {
      for (int64_t i = 0; i < g.count; ++i) {
        const int64_t row = i / nl, col = i - row * nl;
        d[(c * g.batch + b) * g.field + row * g.pitch + col] = h[(b * nc + c) * g.count + i];
      }
    }
  }
  return d;
}
template <class T>
std::vector<T> from_dev(const eda::Geom& g, const std::vector<T>& d, int nc) {
  std::vector<T> h(static_cast<size_t>(nc * g.batch * g.count));
  const int64_t nl = g.n[g.dim - 1];
  for (int64_t b = 0; b < g.batch; ++b) {
    for (int c = 0; c < nc; ++c) {
      for (int64_t i = 0; i < g.count; ++i) {
        const int64_t row = i / nl, col = i - row * nl;
        h[(b * nc + c) * g.count + i] = d[(c * g.batch + b) * g.field + row * g.pitch + col];
      }
    }
  }
  return h;
}
template <class T>
std::vector<T> slice(const std::vector<T>& h, int64_t at, int64_t n) {
  return std::vector<T>(h.begin() + at, h.begin() + at + n);
}
template <class T>
bool pads_hold(const eda::Geom& g, const std::vector<T>& d, int nc, uint64_t pattern_bits) {
  const int64_t nl = g.n[g.dim - 1];
  for (int64_t q = 0; q < nc * g.batch * g.rows; ++q) {
    for (int64_t col = nl; col < g.pitch; ++col) {
      uint64_t u = 0;
      std::memcpy(&u, &d[q * g.pitch + col], sizeof(T));
      if (u != pattern_bits) return false;
    }
  }
  return true;
}

template <class T>
eda::FieldVec<T> vec(T* base, const eda::Geom& g, int comps) {
  eda::FieldVec<T> v{};
  for (int a = 0; a < comps; ++a) v.p[a] = base + static_cast<int64_t>(a) * g.batch * g.field;
  return v;
}
template <class T>
eda::FieldVec<const T> cvec(eda::FieldVec<T> v) {
  eda::FieldVec<const T> c{};
  for (int a = 0; a < 3; ++a) c.p[a] = v.p[a];
  return c;
}

template <class T>
constexpr eda::DType dtype() {
  return sizeof(T) == 4 ? eda::DType::F32 : eda::DType::F64;
}
template <class T>
uint64_t nan_bits() {
  uint64_t u = 0;
  std::memset(&u, 0xff, sizeof(T));
  return u;
}
template <class T>
bool is_nan_pattern(T v) {
  uint64_t u = 0;
  std::memcpy(&u, &v, sizeof(T));
  return u == nan_bits<T>();
}

// ---------------------------------------------------------------------------------------------
// Device logs (the function the kernels use: CUDA's log in double), for the twins of the weights.
__global__ void test_device_log(const double* in, double* out, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = log(in[i]);
}
std::vector<double> device_log(const std::vector<double>& v) {
  Buf<double> a(v.size()), b(v.size());
  a.put(v);
  test_device_log<<<static_cast<unsigned>((v.size() + 255) / 256), 256>>>(a.p, b.p, static_cast<int64_t>(v.size()));
  return b.get();
}

// ---------------------------------------------------------------------------------------------
// Eigen pass, fed with a structure tensor J built by the host twin (any symmetric positive
// semi-definite field exercises it; the method feeds it the monogenic energy tensor)

template <class T>
void orientation_case(const Shape& sh, const std::vector<int32_t>& W, int kind, uint64_t seed, double* worst_c,
                      double* worst_n) {
  const int64_t B = static_cast<int64_t>(W.size());
  const int D = sh.dim;
  const eda::Geom g = eda::make_geom(D, B, sh.n);
  const int64_t N = sh.count();
  const int K = D * (D + 1) / 2;
  std::vector<T> J;
  for (int64_t b = 0; b < B; ++b) {
    const std::vector<T> f = host_twin::make_field<T>(sh, (kind + static_cast<int>(b)) % 3, seed + 11 * b);
    const std::vector<std::vector<T>> Jt = element_twin::structure_tensor(sh, f, W[b] > 0 ? W[b] : 1);
    for (int k = 0; k < K; ++k) J.insert(J.end(), Jt[k].begin(), Jt[k].end());
  }
  Buf<T> dn(static_cast<size_t>(D * B * g.field)), dc(static_cast<size_t>(B * g.field)),
      work(static_cast<size_t>(K * B * g.field));
  Buf<int32_t> dW(W.size());
  work.put(to_dev(g, J, K));
  dW.put(W);
  dn.bytes(0xff);
  dc.bytes(0xff);
  eda::enqueue_element_eigen<T>(nullptr, g, work.p, dW.p, vec(dn.p, g, D), dc.p, nullptr, nullptr);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const std::vector<T> n_dev = dn.get(), cd = dc.get();
  CHECK(pads_hold(g, n_dev, D, nan_bits<T>()));  // pads of the outputs never written
  CHECK(pads_hold(g, cd, 1, nan_bits<T>()));
  const std::vector<T> n = from_dev(g, n_dev, D), c = from_dev(g, cd, 1);
  for (int64_t b = 0; b < B; ++b) {
    if (W[b] == 0) {
      bool untouched = true;
      for (int64_t i = 0; i < N; ++i) untouched = untouched && is_nan_pattern(c[b * N + i]);
      for (int64_t i = 0; i < D * N; ++i) untouched = untouched && is_nan_pattern(n[b * D * N + i]);
      CHECK(untouched);
      continue;
    }
    // eigen against the long-double reference of the same J
    const double tol_c = sizeof(T) == 8 ? 1e-12 : 1e-12 + 6e-8;
    const double tol_n = sizeof(T) == 8 ? 1e-12 : 1e-6;
    const double c_min = sizeof(T) == 8 ? 1e-6 : 1e-3;
    double wc = 0.0, wn = 0.0;
    for (int64_t i = 0; i < N; ++i) {
      long double A[3][3] = {};
      int k = 0;
      for (int a = 0; a < D; ++a) {
        for (int q = a; q < D; ++q, ++k) A[a][q] = A[q][a] = static_cast<long double>(J[(b * K + k) * N + i]);
      }
      long double cr = 0, nr[3] = {0, 0, 0};
      element_twin::eigen_ref(D, A, &cr, nr);
      const double cg = static_cast<double>(c[b * N + i]);
      wc = std::max(wc, static_cast<double>(std::fabs(static_cast<long double>(cg) - cr)));
      long double dot = 0, nn = 0;
      for (int a = 0; a < D; ++a) {
        const long double v = static_cast<long double>(n[(b * D + a) * N + i]);
        dot += v * nr[a];
        nn += v * v;
      }
      CHECK(std::fabs(static_cast<double>(nn) - 1.0) <= (sizeof(T) == 8 ? 1e-14 : 1e-6));
      if (cr > c_min) wn = std::max(wn, static_cast<double>(1.0L - std::fabs(dot)));
    }
    CHECK(wc <= tol_c);
    CHECK(wn <= tol_n);
    if (wc > tol_c || wn > tol_n) std::fprintf(stderr, "  eigen: dim %d W %d |dc| %.3g 1-|n.n| %.3g\n", D, W[b], wc, wn);
    *worst_c = std::max(*worst_c, wc);
    *worst_n = std::max(*worst_n, wn);
  }
}

template <class T>
void orientation_all() {
  double wc = 0.0, wn = 0.0;
  if (g_small) {
    orientation_case<T>(shape(2, 9, 13), {5, 0, 1}, 2, 101, &wc, &wn);
    orientation_case<T>(shape(3, 5, 6, 7), {3, 1}, 2, 102, &wc, &wn);
  } else {
    orientation_case<T>(shape(2, 23, 37), {5, 0, 1, 13}, 2, 101, &wc, &wn);
    orientation_case<T>(shape(2, 1, 40), {5, 3}, 0, 103, &wc, &wn);
    orientation_case<T>(shape(2, 2, 35), {3, 7}, 1, 104, &wc, &wn);
    orientation_case<T>(shape(3, 9, 11, 14), {5, 1, 0}, 2, 102, &wc, &wn);
    orientation_case<T>(shape(3, 2, 5, 33), {3, 9}, 0, 105, &wc, &wn);
    orientation_case<T>(shape(3, 16, 16, 16), {7, 3}, 1, 106, &wc, &wn);
    orientation_case<T>(shape(3, 1, 12, 9), {5}, 2, 107, &wc, &wn);
  }
  std::printf("  eigen %s: max |dc| %.3g, max 1 - |n . n_ref| %.3g\n", sizeof(T) == 8 ? "f64" : "f32", wc, wn);
}

// ---------------------------------------------------------------------------------------------
// Ladder

struct RungRow {
  double d_min, d_max;
  int32_t w, n_max;
  std::vector<int32_t> rungs;
};

void rungs_table() {
  const RungRow rows[] = {
#include "rungs_table.inc"
  };
  int n = 0;
  for (const RungRow& r : rows) {
    const std::vector<int32_t> got = eda::rung_ladder(r.d_min, r.d_max, r.w, r.n_max);
    const bool ok = got == r.rungs;
    CHECK(ok);
    if (!ok) std::fprintf(stderr, "  rung_ladder(%a, %a, %d, %d) differs\n", r.d_min, r.d_max, r.w, r.n_max);
    ++n;
  }
  CHECK(n >= 50);
  CHECK(eda_test::throws_with<std::length_error>([] { (void)eda::rung_ladder(1.0, 3.0 * std::pow(2.0, 20.0), 3, 1 << 22); },
                                                 "at most 32"));
  CHECK(eda_test::throws_with<std::invalid_argument>(
      [] { (void)eda::rung_ladder(1.0, std::numeric_limits<double>::infinity(), 3, 10); }, "finite"));
}

// ---------------------------------------------------------------------------------------------
// The element and its operators

template <class T>
struct ElemCase {
  Shape sh;
  int64_t B = 1;
  std::vector<int32_t> w;
  std::vector<int> skip;         // 1: the element is finished (status kRecordDone) before the windows
  std::vector<T> L, c, n, h;     // host, unpadded, element-major (n: D components)
};

// Smooth positive field in [lo, hi] (and a flat region of `flat` where requested).
template <class T>
std::vector<T> ranged(const Shape& sh, uint64_t seed, double lo, double hi) {
  const std::vector<double> f = host_twin::make_field<double>(sh, 2, seed);
  double mn = 1e300, mx = -1e300;
  for (double v : f) {
    mn = std::min(mn, v);
    mx = std::max(mx, v);
  }
  std::vector<T> out(f.size());
  for (size_t i = 0; i < f.size(); ++i) out[i] = static_cast<T>(lo + (hi - lo) * (f[i] - mn) / (mx - mn + 1e-300));
  return out;
}

// Elements of the batch: kind 0 coherent smooth (c in [0.5, 1], n from a smooth angle), 1
// isotropic (c = 0, L = w exactly in half of the field: weights exactly 0 / 1), 2 mixed with exact
// c = 0 and c = 1 regions and large L (points far outside the domain), 3 one rung (L = w, c = 0), 4
// a ladder rung without voxels (w = 11, d = 7.5: rungs {5, 7}, t = 1 everywhere).
template <class T>
ElemCase<T> make_case(const Shape& sh, const std::vector<int32_t>& w, const std::vector<int>& kinds,
                      const std::vector<int>& skip, uint64_t seed) {
  ElemCase<T> ec;
  ec.sh = sh;
  ec.B = static_cast<int64_t>(w.size());
  ec.w = w;
  ec.skip = skip;
  const int D = sh.dim;
  const int64_t N = sh.count();
  for (int64_t b = 0; b < ec.B; ++b) {
    const int kind = kinds[b];
    const double wb = w[b];
    std::vector<T> L = ranged<T>(sh, seed + 3 * b, 3.0, 2.0 * wb);
    std::vector<T> c = ranged<T>(sh, seed + 3 * b + 1, kind == 0 ? 0.5 : 0.0, 1.0);
    const std::vector<double> ang = host_twin::make_field<double>(sh, 2, seed + 3 * b + 2);
    std::vector<T> n(static_cast<size_t>(D * N));
    for (int64_t i = 0; i < N; ++i) {
      int32_t x[3];
      element_twin::coords(sh, i, x);
      if (kind == 1) {
        c[i] = T(0);
        if (x[0] < sh.n[0] / 2) L[i] = static_cast<T>(wb);
      } else if (kind == 2) {
        if (i % 7 == 0) c[i] = T(0);
        if (i % 5 == 0) c[i] = T(1);
        if (x[D - 1] < 3) L[i] = static_cast<T>(2.0 * wb);  // e = 0.75 w at c = 1: points beyond the border
      } else if (kind == 3) {
        c[i] = T(0);
        L[i] = static_cast<T>(wb);
      } else if (kind == 4) {  // w = 11, d = 7.5: the ladder {5, 7}, every voxel clipped onto rung 1
        c[i] = T(0);
        L[i] = T(7.5);
      }
      // unit normal: spherical angles from a smooth field (axis-aligned in a quarter of kind 2)
      const double th = 2.0 * ang[i], ph = 1.3 * ang[i] + 0.4;
      double v[3] = {std::cos(th), std::sin(th), 0.0};
      if (D == 3) v[0] = std::cos(th) * std::sin(ph), v[1] = std::sin(th) * std::sin(ph), v[2] = std::cos(ph);
      if (kind == 2 && x[0] % 4 == 0) {
        v[0] = 1.0, v[1] = 0.0, v[2] = 0.0;
      }
      double nr = 0.0;
      for (int a = 0; a < D; ++a) nr += v[a] * v[a];
      for (int a = 0; a < D; ++a) n[a * N + i] = static_cast<T>(v[a] / std::sqrt(nr));
    }
    if (D == 1) std::fill(c.begin(), c.end(), T(0));
    ec.L.insert(ec.L.end(), L.begin(), L.end());
    ec.c.insert(ec.c.end(), c.begin(), c.end());
    ec.n.insert(ec.n.end(), n.begin(), n.end());
    const std::vector<T> h = host_twin::make_field<T>(sh, b % 2 == 0 ? 2 : 0, seed + 100 + b);
    ec.h.insert(ec.h.end(), h.begin(), h.end());
  }
  return ec;
}

template <class T>
struct DevElem {
  eda::Geom g;
  int D;
  Buf<T> L, c, n, o, t;
  Buf<uint8_t> lo;
  Buf<eda::ElemRecord> le;
  Buf<int32_t> win;
  std::vector<eda::ElemRecord> hle;
  int32_t n_rungs_max = 0;
  eda::MethodCtl ctl;
  explicit DevElem(const eda::Geom& g_)
      : g(g_),
        D(g_.dim),
        L(static_cast<size_t>(g_.batch * g_.field)),
        c(static_cast<size_t>(g_.batch * g_.field)),
        n(static_cast<size_t>(g_.dim * g_.batch * g_.field)),
        o(static_cast<size_t>(g_.dim * g_.batch * g_.field)),
        t(static_cast<size_t>(g_.batch * g_.field)),
        lo(static_cast<size_t>(g_.batch * g_.field)),
        le(static_cast<size_t>(g_.batch)),
        win(static_cast<size_t>(g_.batch * eda::kMaxRungs)) {}
  eda::ElemView<T> view() const {
    eda::ElemView<T> v{};
    v.lo = lo.p;
    v.t = t.p;
    if (D >= 2) v.o = cvec(vec(o.p, g, D));
    v.rung_win = win.p;
    v.le = le.p;
    v.n_rungs_max = n_rungs_max;
    return v;
  }
};

double bits_double(uint64_t u) {
  double d;
  std::memcpy(&d, &u, 8);
  return d;
}

// Builds the element on the device and checks build and weights against the twins; returns the
// host elements for the operator twins.
template <class T>
std::vector<element_twin::ElemHost<T>> build_and_check(const ElemCase<T>& ec, DevElem<T>& de) {
  const eda::Geom& g = de.g;
  const int D = g.dim;
  const int64_t N = g.count, B = g.batch;
  de.ctl = eda::method_ctl(g, dtype<T>());
  de.L.put(to_dev(g, ec.L, 1));
  de.c.put(to_dev(g, ec.c, 1));
  de.n.put(to_dev(g, ec.n, D));
  de.o.bytes(0xff);
  de.t.bytes(0xff);
  de.lo.bytes(0xff);
  de.hle.assign(static_cast<size_t>(B), eda::ElemRecord{});
  for (int64_t b = 0; b < B; ++b) {
    std::memset(&de.hle[b], 0, sizeof(eda::ElemRecord));
    de.hle[b].status = eda::kRecordActive;
    de.hle[b].w = ec.w[b];
    de.hle[b].d_min_bits = 0x7ff0000000000000ULL;
  }
  de.le.put(de.hle);
  eda::enqueue_element_build<T>(nullptr, g, de.ctl, de.L.p, cvec(vec(de.n.p, g, D >= 2 ? D : 0)),
                                D >= 2 ? de.c.p : nullptr, vec(de.o.p, g, D >= 2 ? D : 0), de.le.p, nullptr);
  de.hle = de.le.get();
  std::vector<element_twin::ElemHost<T>> els(static_cast<size_t>(B));
  std::vector<T> dd(static_cast<size_t>(B * N)), ee(static_cast<size_t>(B * N));
  for (int64_t b = 0; b < B; ++b) {
    double dmin = std::numeric_limits<double>::infinity(), dmax = 0.0;
    for (int64_t i = 0; i < N; ++i) {
      element_twin::element_de<T>(ec.L[b * N + i], D == 1 ? T(0) : ec.c[b * N + i], de.ctl.q, de.ctl.span, &dd[b * N + i],
                                  &ee[b * N + i]);
      dmin = std::min(dmin, static_cast<double>(dd[b * N + i]));
      dmax = std::max(dmax, static_cast<double>(dd[b * N + i]));
    }
    CHECK(bits_double(de.hle[b].d_min_bits) == dmin);
    CHECK(bits_double(de.hle[b].d_max_bits) == dmax);
    const std::vector<int32_t> rungs = eda::rung_ladder(dmin, dmax, ec.w[b], de.ctl.n_max);
    eda::ElemRecord& e = de.hle[b];
    e.n_rungs = static_cast<int32_t>(rungs.size());
    for (int j = 0; j < eda::kMaxRungs; ++j) e.rung_w[j] = j < e.n_rungs ? rungs[j] : 0;
    e.rung_pts = e.rung_used = 0u;
    de.n_rungs_max = std::max(de.n_rungs_max, e.n_rungs);
    els[b].K = e.n_rungs;
    els[b].rung_w = rungs;
  }
  // o = e n bitwise
  if (D >= 2) {
    const std::vector<T> o = from_dev(g, de.o.get(), D);
    bool ok = true;
    for (int64_t b = 0; b < B; ++b) {
      for (int a = 0; a < D; ++a) {
        for (int64_t i = 0; i < N; ++i) ok = ok && o[(b * D + a) * N + i] == ee[b * N + i] * ec.n[(b * D + a) * N + i];
      }
    }
    CHECK(ok);
  }
  de.le.put(de.hle);
  eda::enqueue_element_weights<T>(nullptr, g, de.ctl, de.L.p, D >= 2 ? de.c.p : nullptr, de.le.p, de.lo.p, de.t.p,
                                  nullptr);
  de.hle = de.le.get();
  const std::vector<uint8_t> lo_dev = de.lo.get();
  const std::vector<T> t_dev = de.t.get();
  CHECK(pads_hold(g, lo_dev, 1, 0xffULL));
  CHECK(pads_hold(g, t_dev, 1, nan_bits<T>()));
  const std::vector<uint8_t> lo = from_dev(g, lo_dev, 1);
  const std::vector<T> t = from_dev(g, t_dev, 1);
  // twin of the weights, fed with the device's logs
  std::vector<double> clipped(static_cast<size_t>(B * N));
  std::vector<double> rw;
  for (int64_t b = 0; b < B; ++b) {
    const element_twin::ElemHost<T>& el = els[b];
    for (int j = 0; j < el.K; ++j) rw.push_back(static_cast<double>(el.rung_w[j]));
    for (int64_t i = 0; i < N; ++i) {
      double v = static_cast<double>(dd[b * N + i]);
      v = v > el.rung_w[0] ? v : el.rung_w[0];
      v = v < el.rung_w[el.K - 1] ? v : el.rung_w[el.K - 1];
      clipped[b * N + i] = v;
    }
  }
  const std::vector<double> lg = device_log(clipped), lrw = device_log(rw);
  size_t at = 0;
  int64_t exact01 = 0;
  for (int64_t b = 0; b < B; ++b) {
    element_twin::ElemHost<T>& el = els[b];
    const eda::ElemRecord& e = de.hle[b];
    std::vector<double> lw(lrw.begin() + at, lrw.begin() + at + el.K);
    at += static_cast<size_t>(el.K);
    bool lw_ok = true;
    for (int j = 0; j < el.K; ++j) lw_ok = lw_ok && e.rung_lw[j] == lw[j];
    CHECK(lw_ok);
    el.lo.resize(static_cast<size_t>(N));
    el.t.resize(static_cast<size_t>(N));
    bool ok = true;
    uint32_t used = 0, pts = 0;
    for (int64_t i = 0; i < N; ++i) {
      int idx = 0;
      T tt = T(0);
      if (el.K > 1) {
        const double l = lg[b * N + i];
        int cnt = 0;
        for (int j = 0; j < el.K; ++j) cnt += lw[j] <= l ? 1 : 0;
        idx = std::min(std::max(cnt - 1, 0), el.K - 2);
        tt = static_cast<T>((l - lw[idx]) / (lw[idx + 1] - lw[idx]));
      }
      ok = ok && lo[b * N + i] == idx && std::memcmp(&t[b * N + i], &tt, sizeof(T)) == 0;
      el.lo[i] = static_cast<uint8_t>(idx);
      el.t[i] = tt;
      if (tt == T(0) || tt == T(1)) ++exact01;
      uint32_t bits = 0;
      if (T(1) - tt > T(0)) bits |= 1u << idx;
      if (tt > T(0)) bits |= 1u << (idx + 1);
      used |= bits;
      if (ee[b * N + i] > T(0.25)) pts |= bits;
    }
    CHECK(ok);
    CHECK(e.rung_used == used);
    CHECK(e.rung_pts == pts);
    el.used = used;
    el.pts = pts;
    el.active = !ec.skip[b];
    if (D >= 2) {
      el.o.assign(static_cast<size_t>(D), std::vector<T>(static_cast<size_t>(N)));
      for (int a = 0; a < D; ++a) {
        for (int64_t i = 0; i < N; ++i) el.o[a][i] = ee[b * N + i] * ec.n[(b * D + a) * N + i];
      }
    }
  }
  g_exact01 += exact01;
  // finished elements, then the rung windows
  for (int64_t b = 0; b < B; ++b) {
    if (ec.skip[b]) de.hle[b].status = eda::kRecordDone;
  }
  de.le.put(de.hle);
  de.win.bytes(0x7f);
  eda::enqueue_element_rung_windows(nullptr, static_cast<int32_t>(B), de.le.p, de.n_rungs_max, de.win.p);
  const std::vector<int32_t> win = de.win.get();
  bool wok = true;
  for (int j = 0; j < de.n_rungs_max; ++j) {
    for (int64_t b = 0; b < B; ++b) {
      const element_twin::ElemHost<T>& el = els[b];
      const int32_t want = (!ec.skip[b] && j < el.K && ((el.used >> j) & 1u)) ? el.rung_w[j] : 0;
      wok = wok && win[j * B + b] == want;
    }
  }
  CHECK(wok);
  return els;
}

template <class T>
void ops_case(const ElemCase<T>& ec, const char* name) {
  const Shape& sh = ec.sh;
  const eda::Geom g = eda::make_geom(sh.dim, ec.B, sh.n);
  const int64_t N = g.count, B = g.batch;
  DevElem<T> de(g);
  const std::vector<element_twin::ElemHost<T>> els = build_and_check<T>(ec, de);
  if (std::getenv("ELEMENT_TEST_VERBOSE")) {
    for (int64_t b = 0; b < B; ++b) {
      std::fprintf(stderr, "  %s b %lld: K %d used %#x pts %#x\n", name, static_cast<long long>(b), els[b].K,
                   els[b].used, els[b].pts);
    }
  }
  const size_t F = static_cast<size_t>(B * g.field);
  Buf<T> h(F), o0(F), o1(F), o2(F), t0(F), t1(F);
  h.put(to_dev(g, ec.h, 1));
  std::vector<T> nan_out(static_cast<size_t>(B * N));
  std::memset(nan_out.data(), 0xff, nan_out.size() * sizeof(T));  // the sentinel of untouched outputs
  for (int32_t group : {0, 2, 3, eda::kMaxRungs}) {
    Buf<T> rs(2 * static_cast<size_t>(group) * F);
    rs.bytes(0xff);
    eda::ElemView<T> ev = de.view();
    if (group > 0) {
      ev.rung0 = rs.p;
      ev.rung1 = rs.p + static_cast<size_t>(group) * F;
      ev.rung_group = group;
    }
    const eda::ElemOp ops[] = {eda::ElemOp::kMax, eda::ElemOp::kMaxMin, eda::ElemOp::kBox};
    for (eda::ElemOp op : ops) {
      o0.bytes(0xff);
      o1.bytes(0xff);
      const bool two = op == eda::ElemOp::kMaxMin;
      eda::enqueue_element_op<T>(nullptr, g, op, ev, h.p, o0.p, two ? o1.p : nullptr, t0.p, two ? t1.p : nullptr);
      const std::vector<T> got0 = from_dev(g, o0.get(), 1), got1 = from_dev(g, o1.get(), 1);
      for (int64_t b = 0; b < B; ++b) {
        const std::vector<T> hb = slice(ec.h, b * N, N), init = slice(nan_out, b * N, N);
        const int top = op == eda::ElemOp::kBox ? 3 : 0;
        const std::vector<T> want0 = element_twin::apply<T>(sh, els[b], hb, top, init);
        const bool ok0 = host_twin::same_bits(&got0[b * N], want0.data(), static_cast<size_t>(N));
        CHECK(ok0);
        if (!ok0) {
          int64_t bad = 0;
          for (int64_t i = 0; i < N; ++i) bad += std::memcmp(&got0[b * N + i], &want0[i], sizeof(T)) != 0;
          std::fprintf(stderr, "  %s op %d group %d b %lld: %lld of %lld voxels differ\n", name, static_cast<int>(op),
                       group, static_cast<long long>(b), static_cast<long long>(bad), static_cast<long long>(N));
        }
        if (two) {
          const std::vector<T> want1 = element_twin::apply<T>(sh, els[b], hb, 1, init);
          CHECK(host_twin::same_bits(&got1[b * N], want1.data(), static_cast<size_t>(N)));
        }
      }
    }
    // the envelope mean
    o0.bytes(0xff);
    o1.bytes(0xff);
    o2.bytes(0xff);
    eda::enqueue_envelope_mean<T>(nullptr, g, ev, h.p, o0.p, o1.p, o2.p, t0.p, t1.p);
    const std::vector<T> m = from_dev(g, o0.get(), 1), u = from_dev(g, o1.get(), 1), l = from_dev(g, o2.get(), 1);
    for (int64_t b = 0; b < B; ++b) {
      const std::vector<T> hb = slice(ec.h, b * N, N), init = slice(nan_out, b * N, N);
      const std::vector<T> uu = element_twin::apply<T>(sh, els[b], hb, 0, init);
      const std::vector<T> ll = element_twin::apply<T>(sh, els[b], hb, 1, init);
      std::vector<T> mid(static_cast<size_t>(N));
      for (int64_t i = 0; i < N; ++i) mid[i] = T(0.5) * (uu[i] + ll[i]);
      const std::vector<T> mm = els[b].active ? element_twin::apply<T>(sh, els[b], mid, 3, init) : init;
      CHECK(host_twin::same_bits(&u[b * N], uu.data(), static_cast<size_t>(N)));
      CHECK(host_twin::same_bits(&l[b * N], ll.data(), static_cast<size_t>(N)));
      CHECK(host_twin::same_bits(&m[b * N], mm.data(), static_cast<size_t>(N)));
    }
    // aliasing is rejected
    CHECK(eda_test::throws_with<std::invalid_argument>(
        [&] { eda::enqueue_element_op<T>(nullptr, g, eda::ElemOp::kMax, ev, h.p, h.p, nullptr, t0.p, nullptr); },
        "alias"));
    CHECK(eda_test::throws_with<std::invalid_argument>(
        [&] { eda::enqueue_element_op<T>(nullptr, g, eda::ElemOp::kBox, ev, h.p, o0.p, nullptr, o0.p, nullptr); },
        "alias"));
    CHECK(eda_test::throws_with<std::invalid_argument>(
        [&] { eda::enqueue_element_op<T>(nullptr, g, eda::ElemOp::kMaxMin, ev, h.p, o0.p, o0.p, t0.p, t1.p); },
        "alias"));
    CHECK(eda_test::throws_with<std::invalid_argument>(
        [&] { eda::enqueue_envelope_mean<T>(nullptr, g, ev, h.p, o0.p, o1.p, o1.p, t0.p, t1.p); }, "distinct"));
    if (group > 0 && de.n_rungs_max > 0) {
      // the rung scratch must not hold the input or an output
      CHECK(eda_test::throws_with<std::invalid_argument>(
          [&] {
            eda::enqueue_element_op<T>(nullptr, g, eda::ElemOp::kMax, ev, ev.rung0 + F, o0.p, nullptr, t0.p, nullptr);
          },
          "rung scratch"));
      CHECK(eda_test::throws_with<std::invalid_argument>(
          [&] {
            eda::enqueue_element_op<T>(nullptr, g, eda::ElemOp::kBox, ev, h.p, ev.rung0 + F, nullptr, t0.p, nullptr);
          },
          "rung scratch"));
    }
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
  }
}

template <class T>
void ops_all() {
  if (g_small) {
    ops_case<T>(make_case<T>(shape(1, 70), {7, 5}, {0, 3}, {0, 0}, 300), "1d");
    ops_case<T>(make_case<T>(shape(2, 11, 14), {5, 3}, {2, 0}, {0, 1}, 310), "2d");
    ops_case<T>(make_case<T>(shape(3, 5, 6, 7), {3}, {2}, {0}, 320), "3d");
    return;
  }
  ops_case<T>(make_case<T>(shape(1, 300), {7, 21, 5, 9}, {0, 1, 3, 2}, {0, 0, 0, 1}, 300), "1d");
  ops_case<T>(make_case<T>(shape(2, 23, 37), {5, 9, 13, 7}, {0, 1, 2, 3}, {0, 0, 0, 1}, 310), "2d");
  ops_case<T>(make_case<T>(shape(2, 1, 40), {5}, {2}, {0}, 311), "2d-flat-axis");
  ops_case<T>(make_case<T>(shape(2, 7, 9), {11, 5}, {4, 2}, {0, 0}, 313), "2d-unused-rung");
  ops_case<T>(make_case<T>(shape(2, 64, 3), {9, 3}, {2, 0}, {0, 0}, 312), "2d-narrow");
  ops_case<T>(make_case<T>(shape(3, 9, 11, 14), {5, 7, 3}, {2, 0, 1}, {0, 1, 0}, 320), "3d");
  ops_case<T>(make_case<T>(shape(3, 2, 16, 5), {5}, {2}, {0}, 321), "3d-thin");
  CHECK(g_exact01 > 0);  // some voxels sit exactly on a rung (weights 0 and 1)
}

// Flip of an axis. 1D: every op bitwise. D >= 2 with o = (3, 0, 0) (points at x +- 1.5, x +- 3: exact
// coordinates and weights 0.5): max / min bitwise, box within 2 ulp of max |out|; with the general
// orientations of make_case, the measured deviation is printed and bounded.
template <class T>
void ops_flip(const Shape& sh, int axis, bool dyadic, double* worst) {
  const int D = sh.dim;
  const int64_t N = sh.count();
  ElemCase<T> base = make_case<T>(sh, {9}, {0}, {0}, 400);
  if (dyadic) {
    std::fill(base.L.begin(), base.L.end(), T(8));
    std::fill(base.c.begin(), base.c.end(), D == 1 ? T(0) : T(1));
    for (int a = 0; a < D; ++a) std::fill(base.n.begin() + a * N, base.n.begin() + (a + 1) * N, a == 0 ? T(1) : T(0));
  }
  ElemCase<T> ec = base;
  ec.B = 2;
  ec.w = {9, 9};
  ec.skip = {0, 0};
  const auto flipped = [&](const std::vector<T>& v) { return host_twin::flip<T>(sh, v, axis); };
  const std::vector<T> Lf = flipped(base.L), cf = flipped(base.c), hf = flipped(base.h);
  ec.L.insert(ec.L.end(), Lf.begin(), Lf.end());
  ec.c.insert(ec.c.end(), cf.begin(), cf.end());
  ec.h.insert(ec.h.end(), hf.begin(), hf.end());
  for (int a = 0; a < D; ++a) {
    std::vector<T> na = flipped(slice(base.n, a * N, N));
    if (a == axis) {
      for (T& v : na) v = -v;
    }
    ec.n.insert(ec.n.end(), na.begin(), na.end());
  }
  const eda::Geom g = eda::make_geom(D, 2, sh.n);
  DevElem<T> de(g);
  (void)build_and_check<T>(ec, de);
  const size_t F = static_cast<size_t>(2 * g.field);
  Buf<T> h(F), o0(F), o1(F), t0(F), t1(F);
  h.put(to_dev(g, ec.h, 1));
  // kMaxMin: the min (out1) is checked
  const eda::ElemOp ops[] = {eda::ElemOp::kMax, eda::ElemOp::kMaxMin, eda::ElemOp::kBox};
  for (eda::ElemOp op : ops) {
    const bool two = op == eda::ElemOp::kMaxMin;
    eda::enqueue_element_op<T>(nullptr, g, op, de.view(), h.p, o0.p, two ? o1.p : nullptr, t0.p,
                               two ? t1.p : nullptr);
    const std::vector<T> out = from_dev(g, two ? o1.get() : o0.get(), 1);
    const std::vector<T> a0 = slice(out, 0, N), a1 = flipped(slice(out, N, N));
    double amax = 0.0, dmax = 0.0;
    for (int64_t i = 0; i < N; ++i) {
      amax = std::max(amax, std::fabs(static_cast<double>(a0[i])));
      dmax = std::max(dmax, std::fabs(static_cast<double>(a0[i]) - static_cast<double>(a1[i])));
    }
    const bool bitwise = host_twin::same_bits(a0.data(), a1.data(), static_cast<size_t>(N));
    const double ulp = amax * std::numeric_limits<T>::epsilon();
    if (D == 1 || (dyadic && op != eda::ElemOp::kBox)) {
      CHECK(bitwise);
    } else if (dyadic) {
      CHECK(dmax <= 2.0 * ulp);
    } else {
      // general orientations: the point coordinates x + u and the mirrored n - 1 - x - u round
      // differently (and the corner sums run in another order)
      CHECK(dmax <= (sizeof(T) == 8 ? 1e-12 : 1e-5) * amax);
      *worst = std::max(*worst, dmax / std::max(amax, 1e-300));
    }
  }
}

template <class T>
void flips_all() {
  double worst = 0.0;
  ops_flip<T>(shape(1, g_small ? 60 : 200), 0, false, &worst);
  ops_flip<T>(shape(2, g_small ? 12 : 24, g_small ? 13 : 31), 0, true, &worst);
  ops_flip<T>(shape(2, g_small ? 12 : 24, g_small ? 13 : 31), 1, true, &worst);
  if (!g_small) {
    ops_flip<T>(shape(3, 10, 11, 12), 0, true, &worst);
    ops_flip<T>(shape(3, 10, 11, 12), 2, true, &worst);
    ops_flip<T>(shape(2, 24, 31), 0, false, &worst);
    ops_flip<T>(shape(3, 10, 11, 12), 1, false, &worst);
  }
  std::printf("  op flips %s: general orientations max |diff| / max |out| = %.3g\n", sizeof(T) == 8 ? "f64" : "f32",
              worst);
}

}  // namespace

int main() {
  try {
    eda::Context& ctx = eda::Context::current(0);
    (void)ctx;
    g_small = eda::sanitize_mode();
    rungs_table();
    orientation_all<double>();
    orientation_all<float>();
    ops_all<double>();
    ops_all<float>();
    flips_all<double>();
    flips_all<float>();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "test_element: %s\n", error.what());
    eda::Context::shutdown();
    return 1;
  }
  eda::Context::shutdown();
  return eda_test::finish("test_element");
}
