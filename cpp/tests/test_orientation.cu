// SPDX-License-Identifier: Apache-2.0
// test_orientation: the orientation (method/orientation.cu with the DCT plans of field/dct.hpp and
// the eigen pass) against the long-double twin of twin_orientation.hpp, float32 and float64, 2-D and
// 3-D, batches of one and of several orientation groups (orientation_group). Notation: u the unit
// roundoff of T, nu = ||r - mean||_2 of the element, L = sum_a L_a, L_a = ceil(log2 n_a) (an axis
// with a prime factor above 127 runs cuFFT's Bluestein path: L_a = 3 ceil(log2(2 n_a - 1)) + 2,
// three transforms of the padded length and the chirp products).
//   1. R1 normwise, per axis: ||R1_dev - R1||_2 <= eps_R nu, eps_R = u (16 L + 13 D + 9): one
//      rounding of r - mean, 8 u per radix-2 stage of the forward and of the inverse transform
//      (Higham, Accuracy and Stability, Thm 24.2, with twiddles within 2 u), 6 u per butterfly
//      stage of the post-twiddle, the band and its rounding (2 u, 5 u in float64), the pre-twiddle
//      (7 D + 3 u). cuFFT's algorithm is not documented, so the transform term is a model, measured
//      at under 3 % of eps_R. Power: the twin's ||R1_a||_2 >= 100 eps_R nu on every element, so a
//      wrong R1 with an error above 1 % of it fails.
//   2. R1 pointwise (measured): |R1_dev(x) - R1(x)| <= kInfU u nu; measured at most 0.65 u on the
//      cases below (0.04 to 0.65 u by size); the margin of 12 covers sizes whose cuFFT kernels
//      round differently from the measured ones.
//   3. T pointwise with delta = kInfU u nu for R1 and R2 (R2 runs the same transforms with a
//      multiplier of magnitude <= 1; T overwrites it): the first-order error of the D + 1 products
//      plus the D + 1 roundings of the products and their sum in T.
//   4. c where s = mu1 + mu2 > 4 tau', tau' = tau + e_J, tau = ||T_dev - T||_F at the voxel, e_J =
//      64 2^-53 ||T_dev||_F (the double Jacobi): |c_dev - c| <= 2 tau' (mu1 + |mu2|) / (s (s - 2
//      tau')) + 4 u (Weyl: each eigenvalue moves at most tau'; 4 u: c's double arithmetic and its
//      rounding to T).
//   5. n where gap = mu1 - mu2 > 2 tau' (Davis-Kahan): sin(n_dev, n) <= tau' / (gap - tau') +
//      2 u sqrt(D), the sine as ||n_dev - (n_dev . n) n|| / ||n_dev||. At least 90 % of the voxels
//      of the smooth and plane-wave fields must meet the conditions of 4 and 5.
//   6. Exact: constant fields give R1 = T = 0 (by value: the unpermute's sign may make -0), c = +0
//      and n = e_0 bitwise. The constants are float32 (any value: every partial sum of the mean is
//      exact in double, so r - mean = 0) or float64 with few significand bits (0, 0.75, -32).
//      Elements with W = 0 or on = 0, and the groups whose host_run entries are all 0 (with on
//      agreeing, as the driver passes it), leave every output untouched; with on = nullptr such
//      a group's R1 stays untouched (its transforms are skipped); the batch of these checks ends
//      in a group of one element. Pads are never written. An element's outputs are bitwise the
//      same alone and at the first and last position of each group of a two-group batch whose
//      other elements differ, and after the plan cache evicted its plans: on 16x16 and 8^3, and on
//      16x51 and 12x23x19, sizes whose real-to-complex cuFFT plans compute neighbouring batch
//      entries together (the orientation's complex transforms never do).
// The monogenic signal (enqueue_monogenic, D = 1, 2, 3), against the twin's monogenic:
//   7. R1, D_a f and D_b R_a pointwise within kInfU u nu (every multiplier has magnitude <= 1);
//      n, c as in 4 and 5 with tau from those bounds (T itself is overwritten by D_b R_a); n has
//      the canonical sign (its first component of magnitude above the dtype's default tie
//      tolerance is positive).
//   8. amplitude, phase and frequency against the twin's values for the device's n, where A >
//      4 sqrt(D) delta (delta = kInfU u nu), to first order in the component errors: |dA| <=
//      sqrt(D) delta; |dphase| <= |dq| / A with |dq| <= sqrt(D) delta; |dfreq| <= (|f| D delta +
//      (|q| + |D_n f|) sqrt(D) delta + |freq| dA^2) / (A^2 - dA^2), dA^2 = 2 A sqrt(D) delta + D
//      delta^2; each plus the double arithmetic ((D + 4) 2^-53, the atan2 within 2 ulp) and the
//      rounding to T (u). At least 90 % of the voxels of the smooth and plane-wave fields qualify.
//   9. Exact: constant fields give zero components, n = e_0, c = +0, A = |v|, phase 0 or pi and
//      frequency 0 bitwise; an element prescaled by 2^-3 (exp 3) has 8 times the amplitude and the
//      same other bits; an element's outputs are the same bits alone and at the first and last
//      positions of the groups of a two-group batch, also written at an offset b0 of the planes
//      (the planes of the other elements untouched).
// `test_orientation --twin` (ctest test_orientation_twin, label cpu) checks the twin alone, with no
// CUDA call: the transform identities, the Riesz components and the derivatives of single DCT basis
// fields against their closed form (which fixes the twin's frequency and sign conventions), the
// monogenic outputs of 1-D basis fields, and the eigen solver against closed forms.
#include "twin.hpp"
#include "twin_orientation.hpp"
#include "runtime/env.hpp"
#include "field/dct.hpp"
#include "method/method.hpp"
#include "method/element.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using host_twin::Shape;
using orient_twin::LD;

namespace {

constexpr double kInfU = 8.0;  // the measured pointwise bound of R1 and R2, in u nu

Shape shape(int dim, int64_t a, int64_t b, int64_t c = 1) {
  Shape s;
  s.dim = dim;
  s.n[0] = a;
  s.n[1] = b;
  s.n[2] = c;
  return s;
}

template <class T>
struct Buf {
  T* p = nullptr;
  size_t n = 0;
  explicit Buf(size_t n_, int byte = 0) : n(n_) {
    CHECK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)) == cudaSuccess);
    CHECK(cudaMemset(p, byte, std::max<size_t>(n, 1) * sizeof(T)) == cudaSuccess);
  }
  ~Buf() { cudaFree(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  void put(const std::vector<T>& h) {
    CHECK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess);
  }
  std::vector<T> get() const {
    std::vector<T> h(n);
    CHECK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess);
    return h;
  }
};

template <class T>
bool is_ff(T v) {
  unsigned char b[sizeof(T)];
  std::memcpy(b, &v, sizeof(T));
  for (unsigned char c : b) {
    if (c != 0xff) return false;
  }
  return true;
}

// The device outputs of one call, in the device layouts (component-major, padded).
template <class T>
struct Out {
  eda::Geom g{};
  std::vector<T> r1, t, n, c;
  std::vector<eda::ElemRecord> le;
  // component k of element b at sample i (unpadded index)
  int64_t at(int k, int64_t b, int64_t i) const {
    const int64_t nl = g.n[g.dim - 1];
    return (k * g.batch + b) * g.field + (i / nl) * g.pitch + i % nl;
  }
};

// The gates of one call: W (default all 1), on and host_run (empty: nullptr).
struct Gate {
  std::vector<int32_t> W, on;
  std::vector<uint8_t> host_run;
};

// One enqueue_orientation call on x (B fields of the shape, unpadded, element-major) with windows
// w; every output starts as 0xff bytes, r1 distinct from n.
template <class T>
Out<T> device(const Shape& sh, const std::vector<T>& x, const std::vector<int32_t>& w, const Gate& gate = {}) {
  const int D = sh.dim, K = D * (D + 1) / 2;
  const int64_t B = static_cast<int64_t>(w.size()), N = sh.count();
  eda::Context& ctx = eda::Context::current(0);
  Out<T> o;
  o.g = eda::make_geom(D, B, sh.n);
  const eda::Geom& g = o.g;
  const int64_t F = B * g.field;
  std::vector<T> xp(static_cast<size_t>(F), T(0));
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t i = 0; i < N; ++i) xp[o.at(0, b, i)] = x[b * N + i];
  }
  const eda::DType dt = sizeof(T) == 4 ? eda::DType::F32 : eda::DType::F64;
  Buf<T> dr(F), dr1(D * F, 0xff), dn(D * F, 0xff), dc(F, 0xff), dw(K * F, 0xff);
  Buf<unsigned char> ws(eda::orientation_workspace_bytes(g, dt));
  Buf<int32_t> dW(B), don(B);
  Buf<eda::ElemRecord> dle(B);
  std::vector<eda::ElemRecord> le(B);
  std::memset(le.data(), 0, B * sizeof(eda::ElemRecord));
  for (int64_t b = 0; b < B; ++b) le[b].w = w[b];
  dle.put(le);
  dW.put(gate.W.empty() ? std::vector<int32_t>(B, 1) : gate.W);
  if (!gate.on.empty()) don.put(gate.on);
  dr.put(xp);
  const eda::DctPlans& plans = eda::orientation_plans(ctx, g, dt);
  // the uploads and fills above run on the legacy stream; the context's stream does not wait for them
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  eda::FieldVec<T> n{}, r1{};
  for (int a = 0; a < D; ++a) {
    n.p[a] = dn.p + a * F;
    r1.p[a] = dr1.p + a * F;
  }
  eda::enqueue_orientation<T>(ctx.stream(), g, plans, dr.p, dW.p, n, dc.p, dw.p, r1, ws.p,
                              gate.on.empty() ? nullptr : don.p, dle.p,
                              gate.host_run.empty() ? nullptr : gate.host_run.data());
  ctx.synchronize();
  CHECK(cudaGetLastError() == cudaSuccess);
  o.r1 = dr1.get();
  o.t = dw.get();
  o.n = dn.get();
  o.c = dc.get();
  o.le = dle.get();
  return o;
}

// Every pad sample of every output still holds 0xff bytes.
template <class T>
bool pads_untouched(const Out<T>& o) {
  const eda::Geom& g = o.g;
  const int D = g.dim, K = D * (D + 1) / 2;
  const int64_t nl = g.n[D - 1];
  auto pads = [&](const std::vector<T>& v, int comps) {
    for (int64_t q = 0; q < comps * g.batch * g.rows; ++q) {
      for (int64_t col = nl; col < g.pitch; ++col) {
        if (!is_ff(v[q * g.pitch + col])) return false;
      }
    }
    return true;
  };
  return pads(o.r1, D) && pads(o.t, K) && pads(o.n, D) && pads(o.c, 1);
}

// Every output sample of element b holds 0xff bytes (r1_only: R1 alone).
template <class T>
bool untouched(const Out<T>& o, int64_t b, bool r1_only = false) {
  const int D = o.g.dim, K = D * (D + 1) / 2;
  bool ok = true;
  for (int64_t i = 0; i < o.g.count; ++i) {
    for (int a = 0; a < D; ++a) ok = ok && is_ff(o.r1[o.at(a, b, i)]);
    if (r1_only) continue;
    for (int a = 0; a < D; ++a) ok = ok && is_ff(o.n[o.at(a, b, i)]);
    for (int k = 0; k < K; ++k) ok = ok && is_ff(o.t[o.at(k, b, i)]);
    ok = ok && is_ff(o.c[o.at(0, b, i)]);
  }
  return ok;
}

// Element b of p and element q of o hold the same bits in every output.
template <class T>
bool same_element(const Out<T>& p, int64_t b, const Out<T>& o, int64_t q) {
  const int D = o.g.dim, K = D * (D + 1) / 2;
  auto eq = [](T x, T y) { return std::memcmp(&x, &y, sizeof(T)) == 0; };
  bool ok = true;
  for (int64_t i = 0; i < o.g.count; ++i) {
    for (int a = 0; a < D; ++a) {
      ok = ok && eq(p.r1[p.at(a, b, i)], o.r1[o.at(a, q, i)]) && eq(p.n[p.at(a, b, i)], o.n[o.at(a, q, i)]);
    }
    for (int k = 0; k < K; ++k) ok = ok && eq(p.t[p.at(k, b, i)], o.t[o.at(k, q, i)]);
    ok = ok && eq(p.c[p.at(0, b, i)], o.c[o.at(0, q, i)]);
  }
  return ok;
}

// ---------------------------------------------------------------------------------------------
// Fields

enum Kind { kNoise = 0, kSmooth = 1, kPlane = 2 };

struct Elem {
  int32_t w;
  Kind kind;
  double offset;
};

// noise: uniform in [-1, 1); smooth: three waves plus 0.05 noise (host_twin::make_field kind 2);
// plane: cos(2 pi f . x + 0.3) with |f| = 1 / w along (1, 0.62, 0.37) (no axis direction).
template <class T>
std::vector<T> field(const Shape& sh, const Elem& e, uint64_t seed) {
  std::vector<T> f;
  if (e.kind == kPlane) {
    const double d[3] = {1.0, 0.62, 0.37};
    double nd = 0;
    for (int a = 0; a < sh.dim; ++a) nd += d[a] * d[a];
    nd = std::sqrt(nd);
    f.resize(sh.count());
    for (int64_t i = 0; i < sh.count(); ++i) {
      int32_t x[3];
      orient_twin::coords(sh, i, x);
      double ph = 0.3;
      for (int a = 0; a < sh.dim; ++a) ph += 2.0 * M_PI * d[a] / (nd * e.w) * x[a];
      f[i] = static_cast<T>(std::cos(ph));
    }
  } else {
    f = host_twin::make_field<T>(sh, e.kind == kNoise ? 0 : 2, seed);
  }
  if (e.offset != 0.0) {
    for (T& v : f) v = static_cast<T>(static_cast<double>(v) + e.offset);
  }
  return f;
}

template <class T>
std::vector<T> batch_fields(const Shape& sh, const std::vector<Elem>& el, uint64_t seed) {
  std::vector<T> x;
  for (size_t b = 0; b < el.size(); ++b) {
    const std::vector<T> f = field<T>(sh, el[b], seed + 7 * b);
    x.insert(x.end(), f.begin(), f.end());
  }
  return x;
}

std::vector<int32_t> windows(const std::vector<Elem>& el) {
  std::vector<int32_t> w;
  for (const Elem& e : el) w.push_back(e.w);
  return w;
}

// ---------------------------------------------------------------------------------------------
// Bounds

int64_t largest_prime_factor(int64_t n) {
  int64_t p = 1;
  for (int64_t f = 2; f * f <= n; ++f) {
    while (n % f == 0) {
      p = f;
      n /= f;
    }
  }
  return n > 1 ? std::max(p, n) : p;
}

int ceil_log2(int64_t n) {
  int l = 0;
  while ((int64_t{1} << l) < n) ++l;
  return l;
}

// eps_R / u of the shape.
double eps_r_units(const Shape& sh) {
  int L = 0;
  for (int a = 0; a < sh.dim; ++a) {
    L += largest_prime_factor(sh.n[a]) > 127 ? 3 * ceil_log2(2 * sh.n[a] - 1) + 2 : ceil_log2(sh.n[a]);
  }
  return 16.0 * L + 13.0 * sh.dim + 9.0;
}

// The worst ratios of one case (measured / bound), printed per case.
struct Worst {
  double r1_u = 0, r1_eps = 0, r1_inf = 0, power = std::numeric_limits<double>::infinity();
  double t = 0, c = 0, n = 0;
  int64_t voxels = 0, checked_c = 0, checked_n = 0;
};

// The twin checks 1-5 of element b.
template <class T>
void check_element(const orient_twin::Plans& P, const Out<T>& o, int64_t b, const std::vector<T>& x, const Elem& e,
                   Worst& wst) {
  const Shape& sh = P.sh;
  const int D = sh.dim;
  const int64_t N = sh.count();
  const double u = sizeof(T) == 4 ? 0x1p-24 : 0x1p-53;
  std::vector<LD> r(N);
  for (int64_t i = 0; i < N; ++i) r[i] = static_cast<LD>(x[b * N + i]);
  const orient_twin::Riesz R = orient_twin::riesz(P, r, e.w);
  const LD nu = R.nu;
  const LD eps_r = eps_r_units(sh) * u;
  CHECK(nu > 0);
  CHECK(o.le[b].nonfinite == 0);
  bool ok_r1 = true, ok_power = true, ok_inf = true;
  for (int a = 0; a < D; ++a) {
    LD e2 = 0, einf = 0, n2 = 0;
    for (int64_t i = 0; i < N; ++i) {
      const LD d = static_cast<LD>(o.r1[o.at(a, b, i)]) - R.R1[a][i];
      e2 += d * d;
      einf = std::max(einf, std::fabs(d));
      n2 += R.R1[a][i] * R.R1[a][i];
    }
    const LD e2n = std::sqrt(e2), nrm = std::sqrt(n2);
    ok_r1 = ok_r1 && e2n <= eps_r * nu;
    ok_power = ok_power && nrm >= 100 * eps_r * nu;
    ok_inf = ok_inf && einf <= kInfU * u * nu;
    wst.r1_u = std::max(wst.r1_u, static_cast<double>(e2n / (u * nu)));
    wst.r1_eps = std::max(wst.r1_eps, static_cast<double>(e2n / (eps_r * nu)));
    wst.r1_inf = std::max(wst.r1_inf, static_cast<double>(einf / (kInfU * u * nu)));
    wst.power = std::min(wst.power, static_cast<double>(nrm / (eps_r * nu)));
  }
  CHECK(ok_r1);
  CHECK(ok_power);
  CHECK(ok_inf);
  const LD delta = kInfU * u * nu;
  const LD tiny = 4.0L * (D + 1) * std::numeric_limits<T>::denorm_min();
  const double sqd = std::sqrt(static_cast<double>(D));
  bool ok_t = true, ok_c = true, ok_n = true;
  int64_t cc = 0, cn = 0;
  for (int64_t i = 0; i < N; ++i) {
    LD A[3][3];
    orient_twin::tensor_at(R, D, i, A);
    LD tau2 = 0, tf2 = 0;
    int k = 0;
    for (int a = 0; a < D; ++a) {
      for (int c = a; c < D; ++c, ++k) {
        const LD dv = static_cast<LD>(o.t[o.at(k, b, i)]);
        const LD d = dv - A[a][c];
        tau2 += (a == c ? 1 : 2) * d * d;
        tf2 += (a == c ? 1 : 2) * dv * dv;
        // the products p q of T_ac: R1_a R1_c and h(a, q) h(q, c)
        LD s1 = std::fabs(R.R1[a][i]) + std::fabs(R.R1[c][i]), s2 = std::fabs(R.R1[a][i] * R.R1[c][i]);
        for (int q = 0; q < D; ++q) {
          const LD p1 = orient_twin::h_at(R, D, a, q, i), p2 = orient_twin::h_at(R, D, q, c, i);
          s1 += std::fabs(p1) + std::fabs(p2);
          s2 += std::fabs(p1 * p2);
        }
        const LD first = delta * s1 + (D + 1) * delta * delta;
        const LD bound = first + (D + 1) * u * (1 + 0x1p-10L) * (s2 + first) + tiny;
        ok_t = ok_t && std::fabs(d) <= bound;
        wst.t = std::max(wst.t, static_cast<double>(std::fabs(d) / bound));
      }
    }
    LD mu[3], V[3][3];
    orient_twin::eig(D, A, mu, V);
    const LD tau = std::sqrt(tau2) + 64.0L * 0x1p-53L * std::sqrt(tf2);
    const LD s = mu[0] + mu[1], gap = mu[0] - mu[1];
    if (s > 4 * tau) {
      const LD bc = 2 * tau * (mu[0] + std::fabs(mu[1])) / (s * (s - 2 * tau)) + 4 * u;
      const LD err = std::fabs(static_cast<LD>(o.c[o.at(0, b, i)]) - orient_twin::coherence(mu[0], mu[1]));
      ok_c = ok_c && err <= bc;
      wst.c = std::max(wst.c, static_cast<double>(err / bc));
      ++cc;
    }
    if (gap > 2 * tau) {
      LD dot = 0, nn = 0, pr = 0;
      for (int a = 0; a < D; ++a) {
        const LD v = static_cast<LD>(o.n[o.at(a, b, i)]);
        dot += v * V[a][0];
        nn += v * v;
      }
      for (int a = 0; a < D; ++a) {
        const LD d = static_cast<LD>(o.n[o.at(a, b, i)]) - dot * V[a][0];
        pr += d * d;
      }
      const LD sn = std::sqrt(pr / nn);
      const LD bn = tau / (gap - tau) + 2 * u * sqd;
      ok_n = ok_n && sn <= bn;
      wst.n = std::max(wst.n, static_cast<double>(sn / bn));
      ++cn;
    }
  }
  CHECK(ok_t);
  CHECK(ok_c);
  CHECK(ok_n);
  if (e.kind != kNoise) {
    CHECK(cc >= 0.9 * N);
    CHECK(cn >= 0.9 * N);
  }
  wst.voxels += N;
  wst.checked_c += cc;
  wst.checked_n += cn;
}

const char* tname(int bytes) { return bytes == 4 ? "f32" : "f64"; }

std::string shape_text(const Shape& sh) {
  std::string s = "(";
  for (int a = 0; a < sh.dim; ++a) s += std::to_string(sh.n[a]) + (a + 1 < sh.dim ? "," : ")");
  return s;
}

// One case against the twin.
template <class T>
void twin_case(const Shape& sh, const std::vector<Elem>& el, uint64_t seed) {
  const std::vector<T> x = batch_fields<T>(sh, el, seed);
  const Out<T> o = device<T>(sh, x, windows(el));
  CHECK(pads_untouched(o));
  const orient_twin::Plans P(sh);
  Worst w;
  for (size_t b = 0; b < el.size(); ++b) check_element<T>(P, o, static_cast<int64_t>(b), x, el[b], w);
  std::printf(
      "  %s %s B %zu (group %ld, eps_R %.0f u): R1 2-norm %.3g u = %.3g eps_R (power >= %.3g eps_R), R1 max %.3g of "
      "%g u; T %.3g of bound; c %.3g (%ld of %ld voxels), n %.3g (%ld)\n",
      tname(sizeof(T)), shape_text(sh).c_str(), el.size(), static_cast<long>(eda::orientation_group(o.g)),
      eps_r_units(sh), w.r1_u, w.r1_eps, w.power, w.r1_inf, kInfU, w.t, w.c, static_cast<long>(w.checked_c),
      static_cast<long>(w.voxels), w.n, static_cast<long>(w.checked_n));
}

// B elements with windows and kinds rotating; float32 elements of the second group get an offset
// of 10^4, so that a group-1 element whose mean were taken from another element would round far
// beyond the bound.
std::vector<Elem> rotating(int64_t B, int64_t group, bool f32) {
  const int32_t ws[] = {3, 5, 7, 9, 4, 6};
  std::vector<Elem> el;
  for (int64_t b = 0; b < B; ++b) {
    const Kind k = static_cast<Kind>(b % 3);
    double off = b % 4 == 3 ? 1000.0 : 0.0;
    if (f32 && b >= group) off = 1e4;
    el.push_back({ws[b % 6], k, off});
  }
  return el;
}

template <class T>
void twin_all(bool small) {
  const bool f32 = sizeof(T) == 4;
  if (small) {
    twin_case<T>(shape(2, 9, 13), {{3, kNoise, 0}, {5, kSmooth, 0}, {9, kPlane, 0}}, 11);
    twin_case<T>(shape(3, 5, 6, 7), {{3, kNoise, 0}, {5, kPlane, 0}}, 12);
    if (!f32) twin_case<T>(shape(2, 16, 16), rotating(66, 64, f32), 13);
    return;
  }
  twin_case<T>(shape(2, 9, 13), {{3, kNoise, 0}, {5, kSmooth, 0}, {9, kPlane, 0}}, 11);
  twin_case<T>(shape(2, 2, 35), {{3, kNoise, 0}}, 14);
  twin_case<T>(shape(2, 24, 31), {{5, kPlane, 0}, {17, kSmooth, 1000.0}}, 15);
  twin_case<T>(shape(2, 16, 16), rotating(70, 64, f32), 13);
  twin_case<T>(shape(2, 256, 320), {{5, kSmooth, 0}, {9, kPlane, 0}, {17, kNoise, 0}, {3, kSmooth, 1000.0}}, 16);
  twin_case<T>(shape(2, 131, 12), {{5, kNoise, 0}, {9, kPlane, 0}}, 17);  // 131 prime: Bluestein
  twin_case<T>(shape(3, 5, 6, 7), {{3, kNoise, 0}, {5, kPlane, 0}}, 12);
  twin_case<T>(shape(3, 9, 11, 14), {{3, kSmooth, 0}, {7, kSmooth, 1000.0}}, 18);
  twin_case<T>(shape(3, 2, 5, 33), {{3, kNoise, 0}}, 19);
  twin_case<T>(shape(3, 8, 8, 8), rotating(70, 64, f32), 20);
  twin_case<T>(shape(3, 16, 20, 18), {{5, kPlane, 0}, {9, kSmooth, 0}}, 21);
}

// ---------------------------------------------------------------------------------------------
// Exact checks

template <class T>
void constants(const Shape& sh) {
  // float32: any constant; float64: few significand bits (an exact mean)
  const std::vector<double> vals = sizeof(T) == 4 ? std::vector<double>{0.0, 0.75, -32.0, 1.1, 1e4 / 3}
                                                  : std::vector<double>{0.0, 0.75, -32.0};
  const int D = sh.dim, K = D * (D + 1) / 2;
  const int64_t N = sh.count();
  std::vector<T> x;
  for (double v : vals) x.insert(x.end(), N, static_cast<T>(v));
  const Out<T> o = device<T>(sh, x, std::vector<int32_t>(vals.size(), 5));
  CHECK(pads_untouched(o));
  bool ok = true;
  const T one = T(1), zero = T(0);
  for (int64_t b = 0; b < static_cast<int64_t>(vals.size()); ++b) {
    for (int64_t i = 0; i < N; ++i) {
      for (int a = 0; a < D; ++a) {
        ok = ok && o.r1[o.at(a, b, i)] == T(0);
        ok = ok && std::memcmp(&o.n[o.at(a, b, i)], a == 0 ? &one : &zero, sizeof(T)) == 0;
      }
      for (int k = 0; k < K; ++k) ok = ok && o.t[o.at(k, b, i)] == T(0);
      ok = ok && std::memcmp(&o.c[o.at(0, b, i)], &zero, sizeof(T)) == 0;
    }
    ok = ok && o.le[b].nonfinite == 0;
  }
  CHECK(ok);
}

// W = 0, on = 0 and host_run skips on a two-group batch of B elements (group 64; B = 65: the
// second group holds one element).
template <class T>
void skips(const Shape& sh, int64_t B) {
  const std::vector<Elem> el = rotating(B, 64, false);
  const std::vector<T> x = batch_fields<T>(sh, el, 31);
  const std::vector<int32_t> w = windows(el);
  const Out<T> ref = device<T>(sh, x, w);
  CHECK(pads_untouched(ref));
  Gate gate;
  gate.W.assign(B, 1);
  gate.on.assign(B, 1);
  gate.W[5] = 0;
  gate.on[B - 3] = 0;
  const Out<T> o = device<T>(sh, x, w, gate);
  CHECK(pads_untouched(o));
  bool ok = true;
  for (int64_t b = 0; b < B; ++b) ok = ok && (b == 5 || b == B - 3 ? untouched(o, b) : same_element(o, b, ref, b));
  CHECK(ok);
  // host_run: group 0 off, with on agreeing (the driver's call), then with on = nullptr
  Gate hr;
  hr.host_run.assign(B, 0);
  hr.on.assign(B, 0);
  for (int64_t b = 64; b < B; ++b) hr.host_run[b] = hr.on[b] = 1;
  const Out<T> h = device<T>(sh, x, w, hr);
  CHECK(pads_untouched(h));
  ok = true;
  for (int64_t b = 0; b < B; ++b) ok = ok && (b < 64 ? untouched(h, b) : same_element(h, b, ref, b));
  CHECK(ok);
  hr.on.clear();
  const Out<T> h2 = device<T>(sh, x, w, hr);
  ok = true;
  for (int64_t b = 0; b < B; ++b) ok = ok && (b < 64 ? untouched(h2, b, true) : same_element(h2, b, ref, b));
  CHECK(ok);
}

// An element alone and at positions 0, 63, 64 and B - 1 of a B batch (two groups of 64; 49 for
// 12x23x19) whose other elements differ: bitwise the same outputs. Then the plan cache evicts the
// shape's plans and the element alone again gives the same bits.
template <class T>
void positions(const Shape& sh, int64_t B) {
  const Elem e{5, kSmooth, 1000.0};
  const std::vector<T> xe = field<T>(sh, e, 41);
  const Out<T> alone = device<T>(sh, xe, {e.w});
  std::vector<Elem> el = rotating(B, 64, false);
  const int64_t pos[] = {0, 63, 64, B - 1};
  for (int64_t p : pos) el[p] = e;
  std::vector<T> x = batch_fields<T>(sh, el, 43);
  const int64_t N = sh.count();
  for (int64_t p : pos) std::copy(xe.begin(), xe.end(), x.begin() + p * N);
  const Out<T> o = device<T>(sh, x, windows(el));
  bool ok = true;
  for (int64_t p : pos) ok = ok && same_element(o, p, alone, 0);
  CHECK(ok);
  // five other group geometries (the cache keeps four)
  eda::Context& ctx = eda::Context::current(0);
  for (int64_t k = 0; k < 5; ++k) {
    int64_t n[3] = {sh.n[0] + 1 + k, sh.n[1], sh.n[2]};
    (void)eda::orientation_plans(ctx, eda::make_geom(sh.dim, 1, n), sizeof(T) == 4 ? eda::DType::F32 : eda::DType::F64);
  }
  const Out<T> again = device<T>(sh, xe, {e.w});
  CHECK(same_element(again, 0, alone, 0));
}

template <class T>
void exact_all(bool small) {
  constants<T>(shape(2, 9, 13));
  constants<T>(shape(3, 5, 6, 7));
  if (small && sizeof(T) == 4) return;
  const int64_t B = small ? 66 : 70;
  skips<T>(shape(2, 16, 16), 65);
  positions<T>(shape(2, 16, 16), B);
  positions<T>(shape(2, 16, 51), B);
  if (small) return;
  skips<T>(shape(3, 8, 8, 8), 65);
  positions<T>(shape(3, 8, 8, 8), B);
  positions<T>(shape(3, 12, 23, 19), B);
}

// ---------------------------------------------------------------------------------------------
// The monogenic signal

// The device outputs of one enqueue_monogenic call: the component fields (layouts of Out) and the
// output planes (out_batch elements, the chunk at b0).
template <class T>
struct MonogenicOut {
  eda::Geom g{};
  int64_t out_batch = 0, b0 = 0;
  std::vector<T> r1, df, dr, n, c, out;
  int64_t at(int k, int64_t b, int64_t i) const {
    const int64_t nl = g.n[g.dim - 1];
    return (k * g.batch + b) * g.field + (i / nl) * g.pitch + i % nl;
  }
  // output plane value: o = 0 amplitude, 1 phase, 2 frequency, 3 + a direction a, 3 + D coherence
  T plane(int o, int64_t b, int64_t i) const {
    const int D = g.dim;
    const int64_t P = out_batch * g.count, e = b0 + b;
    if (o < 3) return out[o * P + e * g.count + i];
    if (o < 3 + D) return out[3 * P + (e * D + (o - 3)) * g.count + i];
    return out[(3 + D) * P + e * g.count + i];
  }
};

template <class T>
MonogenicOut<T> monogenic_device(const Shape& sh, const std::vector<T>& x, int64_t B, const std::vector<int32_t>& exps = {},
                       int64_t out_batch = 0, int64_t b0 = 0) {
  const int D = sh.dim, K = D * (D + 1) / 2;
  eda::Context& ctx = eda::Context::current(0);
  MonogenicOut<T> o;
  o.g = eda::make_geom(D, B, sh.n);
  o.out_batch = out_batch ? out_batch : B;
  o.b0 = b0;
  const eda::Geom& g = o.g;
  const int64_t F = B * g.field, N = sh.count();
  std::vector<T> xp(static_cast<size_t>(F), T(0));
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t i = 0; i < N; ++i) xp[o.at(0, b, i)] = x[b * N + i];
  }
  const eda::DType dt = sizeof(T) == 4 ? eda::DType::F32 : eda::DType::F64;
  const int rows = eda::monogenic_rows(D);
  Buf<T> df(F), r1(D * F, 0xff), dd(D * F, 0xff), dn(D * F, 0xff), dc(F, 0xff), dw(K * F, 0xff);
  Buf<T> dout(static_cast<size_t>(rows) * o.out_batch * N, 0xff);
  Buf<unsigned char> ws(eda::orientation_workspace_bytes(g, dt));
  Buf<int32_t> dexp(B);
  df.put(xp);
  if (!exps.empty()) dexp.put(exps);
  const eda::DctPlans& plans = eda::orientation_plans(ctx, g, dt);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  eda::MonogenicFields<T> mf{};
  for (int a = 0; a < D; ++a) {
    mf.r1.p[a] = r1.p + a * F;
    mf.df.p[a] = dd.p + a * F;
    mf.n.p[a] = D >= 2 ? dn.p + a * F : nullptr;
  }
  mf.work = dw.p;
  mf.c = D >= 2 ? dc.p : nullptr;
  const double tiny = sizeof(T) == 4 ? static_cast<double>(std::numeric_limits<float>::min()) : 1e-300;
  eda::enqueue_monogenic<T>(ctx.stream(), g, plans, df.p, dexp.p, mf, ws.p, tiny, dout.p, o.out_batch, b0);
  ctx.synchronize();
  CHECK(cudaGetLastError() == cudaSuccess);
  o.r1 = r1.get();
  o.df = dd.get();
  o.dr = dw.get();
  o.n = dn.get();
  o.c = dc.get();
  o.out = dout.get();
  return o;
}

constexpr LD kTwo53 = 0x1p-53L;

// The worst ratios of one monogenic case (measured / bound).
struct MonogenicWorst {
  double comp = 0, amp = 0, phase = 0, freq = 0, c = 0, n = 0;
  int64_t voxels = 0, checked = 0, checked_c = 0, checked_n = 0;
};

// The twin checks 7 and 8 of element b.
template <class T>
void monogenic_check_element(const orient_twin::Plans& P, const MonogenicOut<T>& o, int64_t b, const std::vector<T>& x,
                        Kind kind, MonogenicWorst& wst) {
  const Shape& sh = P.sh;
  const int D = sh.dim, K = D * (D + 1) / 2;
  const int64_t N = sh.count();
  const LD u = sizeof(T) == 4 ? 0x1p-24L : 0x1p-53L;
  const LD zero = sizeof(T) == 4 ? 0x1p-14L : 0x1p-30L;
  std::vector<LD> r(N);
  for (int64_t i = 0; i < N; ++i) r[i] = static_cast<LD>(x[b * N + i]);
  const orient_twin::Monogenic M = orient_twin::monogenic(P, r);
  const LD nu = M.R.nu;
  CHECK(nu > 0);
  const LD delta = kInfU * u * nu, sd = std::sqrt(static_cast<LD>(D));
  bool ok_comp = true, ok_t = true, ok_c = true, ok_n = true, ok_sign = true, ok_a = true, ok_p = true, ok_f = true;
  int64_t cc = 0, cn = 0, cm = 0;
  const auto comp = [&](const std::vector<T>& dev, int k, const std::vector<LD>& tw, int64_t i) {
    const LD d = std::fabs(static_cast<LD>(dev[o.at(k, b, i)]) - tw[i]);
    wst.comp = std::max(wst.comp, static_cast<double>(d / delta));
    return d <= delta;
  };
  for (int64_t i = 0; i < N; ++i) {
    for (int a = 0; a < D; ++a) ok_comp = ok_comp && comp(o.r1, a, M.R.R1[a], i) && comp(o.df, a, M.Df[a], i);
    for (int k = 0; k < (D == 1 ? 1 : K); ++k) ok_comp = ok_comp && comp(o.dr, k, M.DR[k], i);
    LD n[3] = {1, 0, 0};
    for (int a = 0; a < D && D >= 2; ++a) n[a] = static_cast<LD>(o.plane(3 + a, b, i));
    if (D >= 2) {
      // the canonical sign, and n, c against the twin's eigen-decomposition (tau from the bounds)
      int s = 0;
      for (int a = 0; a < D; ++a) s = s == 0 && std::fabs(n[a]) > zero ? (n[a] > 0 ? 1 : -1) : s;
      ok_sign = ok_sign && s > 0;
      LD A[3][3];
      orient_twin::tensor_at(M.R, D, i, A);
      LD tau2 = 0, tf2 = 0;
      for (int a = 0; a < D; ++a) {
        for (int c = a; c < D; ++c) {
          LD s1 = std::fabs(M.R.R1[a][i]) + std::fabs(M.R.R1[c][i]), s2 = std::fabs(M.R.R1[a][i] * M.R.R1[c][i]);
          for (int q = 0; q < D; ++q) {
            const LD p1 = orient_twin::h_at(M.R, D, a, q, i), p2 = orient_twin::h_at(M.R, D, q, c, i);
            s1 += std::fabs(p1) + std::fabs(p2);
            s2 += std::fabs(p1 * p2);
          }
          const LD first = delta * s1 + (D + 1) * delta * delta;
          const LD bound = first + (D + 1) * u * (1 + 0x1p-10L) * (s2 + first);
          tau2 += (a == c ? 1 : 2) * bound * bound;
          tf2 += (a == c ? 1 : 2) * (std::fabs(A[a][c]) + bound) * (std::fabs(A[a][c]) + bound);
        }
      }
      LD mu[3], V[3][3];
      orient_twin::eig(D, A, mu, V);
      const LD tau = std::sqrt(tau2) + 64.0L * kTwo53 * std::sqrt(tf2);
      const LD sum = mu[0] + mu[1], gap = mu[0] - mu[1];
      if (sum > 4 * tau) {
        const LD bc = 2 * tau * (mu[0] + std::fabs(mu[1])) / (sum * (sum - 2 * tau)) + 4 * u;
        const LD err = std::fabs(static_cast<LD>(o.plane(3 + D, b, i)) - orient_twin::coherence(mu[0], mu[1]));
        ok_c = ok_c && err <= bc;
        wst.c = std::max(wst.c, static_cast<double>(err / bc));
        ++cc;
      }
      if (gap > 2 * tau) {
        LD dot = 0, nn = 0, pr = 0;
        for (int a = 0; a < D; ++a) {
          dot += n[a] * V[a][0];
          nn += n[a] * n[a];
        }
        for (int a = 0; a < D; ++a) pr += (n[a] - dot * V[a][0]) * (n[a] - dot * V[a][0]);
        const LD sn = std::sqrt(pr / nn), bn = tau / (gap - tau) + 2 * u * sd;
        ok_n = ok_n && sn <= bn;
        wst.n = std::max(wst.n, static_cast<double>(sn / bn));
        ++cn;
      }
      // the device's n and c are what the eigen pass wrote (up to the canonical sign)
      for (int a = 0; a < D; ++a) {
        const T nf = o.n[o.at(a, b, i)], np = o.plane(3 + a, b, i);
        ok_t = ok_t && (np == nf || np == -nf);
      }
      ok_t = ok_t && o.plane(3 + D, b, i) == o.c[o.at(0, b, i)];
    }
    // the outputs for the device's n
    const LD f = static_cast<LD>(x[b * N + i]);
    const orient_twin::MonogenicAt t = orient_twin::monogenic_at(M, D, i, f, n);
    if (!(t.A > 4 * sd * delta)) continue;
    ++cm;
    LD sabs = 0;
    for (int a = 0; a < D; ++a) sabs += std::fabs(n[a] * M.R.R1[a][i]);
    const LD dA = sd * delta + (D + 4) * kTwo53 * t.A + u * t.A;
    const LD eA = std::fabs(static_cast<LD>(o.plane(0, b, i)) - t.A);
    ok_a = ok_a && eA <= dA;
    wst.amp = std::max(wst.amp, static_cast<double>(eA / dA));
    const LD dq = sd * delta + D * kTwo53 * sabs;
    const LD dP = dq / t.A + (4 * kTwo53 + u) * orient_twin::kPi;
    LD eP = std::fabs(static_cast<LD>(o.plane(1, b, i)) - t.phase);
    eP = std::min(eP, std::fabs(eP - 2 * orient_twin::kPi));
    ok_p = ok_p && eP <= dP;
    wst.phase = std::max(wst.phase, static_cast<double>(eP / dP));
    const LD dA2 = 2 * t.A * sd * delta + D * delta * delta;
    const LD num = std::fabs(f) * D * delta + (std::fabs(t.q) + std::fabs(t.dnf)) * sd * delta;
    const LD dF = (num + std::fabs(t.freq) * dA2) / (t.a2 - dA2) +
                  (2 * D + 8) * kTwo53 * (std::fabs(f * t.dnq) + std::fabs(t.q * t.dnf)) / t.a2 +
                  u * std::fabs(t.freq);
    const LD eF = std::fabs(static_cast<LD>(o.plane(2, b, i)) - t.freq);
    ok_f = ok_f && eF <= dF;
    wst.freq = std::max(wst.freq, static_cast<double>(eF / dF));
  }
  CHECK(ok_comp);
  CHECK(ok_t);
  CHECK(ok_sign);
  CHECK(ok_c);
  CHECK(ok_n);
  CHECK(ok_a);
  CHECK(ok_p);
  CHECK(ok_f);
  if (kind != kNoise) {
    CHECK(cm >= 0.9 * N);
    if (D >= 2) {
      CHECK(cc >= 0.9 * N);
      CHECK(cn >= 0.9 * N);
    }
  }
  wst.voxels += N;
  wst.checked += cm;
  wst.checked_c += cc;
  wst.checked_n += cn;
}

// One monogenic case against the twin (the windows of el set the plane waves' periods).
template <class T>
void monogenic_case(const Shape& sh, const std::vector<Elem>& el, uint64_t seed) {
  const std::vector<T> x = batch_fields<T>(sh, el, seed);
  const MonogenicOut<T> o = monogenic_device<T>(sh, x, static_cast<int64_t>(el.size()));
  const orient_twin::Plans P(sh);
  MonogenicWorst w;
  for (size_t b = 0; b < el.size(); ++b) monogenic_check_element<T>(P, o, static_cast<int64_t>(b), x, el[b].kind, w);
  std::printf(
      "  monogenic %s %s B %zu: components max %.3g of %g u nu; A %.3g, phase %.3g, frequency %.3g of their bounds "
      "(%ld of %ld voxels); c %.3g (%ld), n %.3g (%ld)\n",
      tname(sizeof(T)), shape_text(sh).c_str(), el.size(), w.comp, kInfU, w.amp, w.phase, w.freq,
      static_cast<long>(w.checked), static_cast<long>(w.voxels), w.c, static_cast<long>(w.checked_c), w.n,
      static_cast<long>(w.checked_n));
}

template <class T>
bool same_monogenic(const MonogenicOut<T>& p, int64_t b, const MonogenicOut<T>& o, int64_t q) {
  auto eq = [](T x, T y) { return std::memcmp(&x, &y, sizeof(T)) == 0; };
  bool ok = true;
  for (int k = 0; k < eda::monogenic_rows(o.g.dim); ++k) {
    for (int64_t i = 0; i < o.g.count; ++i) ok = ok && eq(p.plane(k, b, i), o.plane(k, q, i));
  }
  return ok;
}

// Check 9.
template <class T>
void monogenic_exact(const Shape& sh, int64_t B) {
  const int D = sh.dim;
  const int64_t N = sh.count();
  // constants
  const std::vector<double> vals = sizeof(T) == 4 ? std::vector<double>{0.0, 0.75, -32.0, 1.1}
                                                  : std::vector<double>{0.0, 0.75, -32.0};
  std::vector<T> x;
  for (double v : vals) x.insert(x.end(), N, static_cast<T>(v));
  const MonogenicOut<T> c = monogenic_device<T>(sh, x, static_cast<int64_t>(vals.size()));
  bool ok = true;
  const T one = T(1), zero = T(0), pi = static_cast<T>(3.141592653589793);
  for (int64_t b = 0; b < static_cast<int64_t>(vals.size()); ++b) {
    const T v = static_cast<T>(vals[b]);
    for (int64_t i = 0; i < N; ++i) {
      for (int a = 0; a < D; ++a) ok = ok && c.r1[c.at(a, b, i)] == T(0) && c.df[c.at(a, b, i)] == T(0);
      ok = ok && c.plane(0, b, i) == (v < 0 ? -v : v) && c.plane(2, b, i) == T(0);
      ok = ok && c.plane(1, b, i) == (v < 0 ? pi : zero);
      for (int a = 0; a < D && D >= 2; ++a) {
        const T na = c.plane(3 + a, b, i);
        ok = ok && std::memcmp(&na, a == 0 ? &one : &zero, sizeof(T)) == 0;
      }
      if (D >= 2) {
        const T cv = c.plane(3 + D, b, i);
        ok = ok && std::memcmp(&cv, &zero, sizeof(T)) == 0;
      }
    }
  }
  CHECK(ok);
  // the prescale exponent: element 1 is element 0 with exp 3
  const Elem e{5, kSmooth, 0.0};
  const std::vector<T> xe = field<T>(sh, e, 51);
  std::vector<T> x2 = xe;
  x2.insert(x2.end(), xe.begin(), xe.end());
  const MonogenicOut<T> s = monogenic_device<T>(sh, x2, 2, {0, 3});
  ok = true;
  for (int64_t i = 0; i < N; ++i) {
    ok = ok && s.plane(0, 1, i) == T(8) * s.plane(0, 0, i);
    for (int k = 1; k < eda::monogenic_rows(D); ++k) {
      const T a0 = s.plane(k, 0, i), a1 = s.plane(k, 1, i);
      ok = ok && std::memcmp(&a0, &a1, sizeof(T)) == 0;
    }
  }
  CHECK(ok);
  // positions: alone, and at 0, 63, 64, B - 1 of a two-group batch written at b0 = 2 of B + 3
  // elements' planes
  const MonogenicOut<T> alone = monogenic_device<T>(sh, xe, 1);
  std::vector<Elem> el = rotating(B, 64, false);
  const int64_t pos[] = {0, 63, 64, B - 1};
  std::vector<T> xb = batch_fields<T>(sh, el, 53);
  for (int64_t p : pos) std::copy(xe.begin(), xe.end(), xb.begin() + p * N);
  const MonogenicOut<T> o = monogenic_device<T>(sh, xb, B, {}, B + 3, 2);
  ok = true;
  for (int64_t p : pos) ok = ok && same_monogenic(o, p, alone, 0);
  CHECK(ok);
  // the planes of elements 0, 1 and B + 2 of the call are untouched
  ok = true;
  for (int k = 0; k < eda::monogenic_rows(D); ++k) {
    for (int64_t e2 : {int64_t{-2}, int64_t{-1}, B}) {
      for (int64_t i = 0; i < N; ++i) ok = ok && is_ff(o.plane(k, e2, i));
    }
  }
  CHECK(ok);
}

template <class T>
void monogenic_all(bool small) {
  monogenic_case<T>(shape(1, 37, 1), {{3, kNoise, 0}, {5, kSmooth, 0}, {9, kPlane, 0}}, 61);
  monogenic_case<T>(shape(2, 9, 13), {{3, kNoise, 0}, {5, kSmooth, 0}, {9, kPlane, 0}}, 62);
  monogenic_case<T>(shape(3, 5, 6, 7), {{3, kNoise, 0}, {5, kPlane, 0}}, 63);
  monogenic_exact<T>(shape(1, 64, 1), small ? 66 : 70);
  monogenic_exact<T>(shape(2, 16, 16), small ? 66 : 70);
  if (small) return;
  monogenic_case<T>(shape(1, 256, 1), {{7, kSmooth, 1000.0}, {17, kPlane, 0}}, 64);
  monogenic_case<T>(shape(2, 24, 31), {{5, kPlane, 0}, {17, kSmooth, 1000.0}}, 65);
  monogenic_case<T>(shape(3, 9, 11, 14), {{3, kSmooth, 0}, {7, kPlane, 1000.0}}, 66);
  monogenic_exact<T>(shape(3, 8, 8, 8), 70);
}

// ---------------------------------------------------------------------------------------------
// The twin alone (no CUDA)

LD rel_diff(const std::vector<LD>& a, const std::vector<LD>& b) {
  LD d = 0, m = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    d = std::max(d, std::fabs(a[i] - b[i]));
    m = std::max(m, std::fabs(b[i]));
  }
  return m > 0 ? d / m : d;
}

LD norm2(const std::vector<LD>& v) {
  LD s = 0;
  for (LD x : v) s += x * x;
  return s;
}

std::vector<LD> random_ld(int64_t n, uint64_t seed) {
  host_twin::Rng rng(seed);
  std::vector<LD> f(n);
  for (LD& v : f) v = static_cast<LD>(rng.uniform() * 2.0 - 1.0);
  return f;
}

constexpr LD kTol = 0x1p-50L;

void twin_identities(const Shape& sh, int32_t w) {
  const int D = sh.dim;
  const int64_t N = sh.count();
  const orient_twin::Plans P(sh);
  const std::vector<LD> f = random_ld(N, 5 + D);
  // the inverse inverts, and the forward transform is orthonormal
  std::vector<LD> g = f;
  P.dct(g);
  CHECK(std::fabs(norm2(g) - norm2(f)) <= kTol * norm2(f));
  P.inverse(g, 0);
  CHECK(rel_diff(g, f) <= kTol);
  // the separable transform is the N-D sum (full arguments, no tables)
  std::vector<LD> X = f, direct(N, 0);
  P.dct(X);
  for (int64_t kk = 0; kk < N; ++kk) {
    int32_t k[3];
    orient_twin::coords(sh, kk, k);
    LD acc = 0;
    for (int64_t i = 0; i < N; ++i) {
      int32_t x[3];
      orient_twin::coords(sh, i, x);
      LD t = f[i];
      for (int a = 0; a < D; ++a) {
        t *= std::sqrt((k[a] ? 2.0L : 1.0L) / sh.n[a]) * std::cos(orient_twin::kPi * k[a] * (2 * x[a] + 1) / (2 * sh.n[a]));
      }
      acc += t;
    }
    direct[kk] = acc;
  }
  CHECK(rel_diff(X, direct) <= kTol);
  // the Riesz identities: sum_a ||R1_a||^2 = ||X_b||^2, sum_a R2_aa = -IDCT(X_b),
  // sum_{a,b} ||R2_ab||^2 = ||X_b||^2
  const orient_twin::Riesz R = orient_twin::riesz(P, f, w);
  const LD xb = norm2(R.Xb);
  CHECK(xb > 0);
  LD s1 = 0, s2 = 0;
  std::vector<LD> tr(N, 0), ib = R.Xb;
  P.inverse(ib, 0);
  for (int a = 0; a < D; ++a) {
    s1 += norm2(R.R1[a]);
    for (int b = 0; b < D; ++b) s2 += norm2(R.R2[orient_twin::pair_index(D, a, b)]);
    for (int64_t i = 0; i < N; ++i) tr[i] -= orient_twin::h_at(R, D, a, a, i);
  }
  CHECK(std::fabs(s1 - xb) <= kTol * xb);
  CHECK(std::fabs(s2 - xb) <= kTol * xb);
  CHECK(rel_diff(tr, ib) <= kTol);
}

// A single DCT basis field prod_a cos(theta_a): the Riesz components in closed form, from the
// multipliers -i u_a / |u| on the plane waves of its extension.
void twin_anchor(const Shape& sh, const int32_t* k, int32_t w) {
  const int D = sh.dim;
  const int64_t N = sh.count();
  LD ua[3], r2 = 0;
  for (int a = 0; a < D; ++a) {
    ua[a] = static_cast<LD>(k[a]) / (2 * sh.n[a]);
    r2 += ua[a] * ua[a];
  }
  const LD rad = std::sqrt(r2), G = orient_twin::band(rad, w);
  CHECK(G > 0.5L);  // the anchor sits inside the band
  std::vector<LD> f(N);
  std::vector<LD> cs(N * 3), sn(N * 3);
  for (int64_t i = 0; i < N; ++i) {
    int32_t x[3];
    orient_twin::coords(sh, i, x);
    f[i] = 1;
    for (int a = 0; a < D; ++a) {
      const LD th = orient_twin::kPi * k[a] * (2 * x[a] + 1) / (2 * sh.n[a]);
      cs[a * N + i] = std::cos(th);
      sn[a * N + i] = std::sin(th);
      f[i] *= cs[a * N + i];
    }
  }
  const orient_twin::Plans P(sh);
  const orient_twin::Riesz R = orient_twin::riesz(P, f, w);
  LD worst = 0;
  for (int64_t i = 0; i < N; ++i) {
    for (int a = 0; a < D; ++a) {
      LD e1 = G * ua[a] / rad * sn[a * N + i];
      for (int b = 0; b < D; ++b) {
        if (b != a) e1 *= cs[b * N + i];
      }
      worst = std::max(worst, std::fabs(R.R1[a][i] - e1));
      for (int b = a; b < D; ++b) {
        LD e2 = a == b ? -G * (ua[a] / rad) * (ua[a] / rad) : G * ua[a] * ua[b] / r2;
        for (int q = 0; q < D; ++q) e2 *= (a != b && (q == a || q == b)) ? sn[q * N + i] : cs[q * N + i];
        worst = std::max(worst, std::fabs(orient_twin::h_at(R, D, a, b, i) - e2));
      }
    }
  }
  CHECK(worst <= kTol);
  if (worst > kTol) std::fprintf(stderr, "  anchor %s: max |twin - closed form| %.3Lg\n", shape_text(sh).c_str(), worst);
  // the monogenic components: no band, and the derivatives D_a f and D_b R_a
  const orient_twin::Monogenic M = orient_twin::monogenic(P, f);
  worst = 0;
  for (int64_t i = 0; i < N; ++i) {
    for (int a = 0; a < D; ++a) {
      LD r1 = ua[a] / rad * sn[a * N + i], d1 = -ua[a] * sn[a * N + i];
      for (int b = 0; b < D; ++b) {
        if (b != a) {
          r1 *= cs[b * N + i];
          d1 *= cs[b * N + i];
        }
      }
      worst = std::max({worst, std::fabs(M.R.R1[a][i] - r1), std::fabs(M.Df[a][i] - d1)});
      for (int b = a; b < D; ++b) {
        LD e = a == b ? ua[a] * ua[a] / rad : -ua[a] * ua[b] / rad;
        for (int q = 0; q < D; ++q) e *= (a != b && (q == a || q == b)) ? sn[q * N + i] : cs[q * N + i];
        worst = std::max(worst, std::fabs(M.DR[orient_twin::pair_index(D, a, b)][i] - e));
      }
    }
  }
  CHECK(worst <= kTol);
  if (worst > kTol) std::fprintf(stderr, "  monogenic anchor %s: max |twin - closed form| %.3Lg\n", shape_text(sh).c_str(), worst);
}

// A 1-D DCT basis field cos(theta), theta = pi k (2x + 1) / (2n), the plane wave of frequency
// u = k / (2n): amplitude 1, phase theta (wrapped) and frequency u at every sample.
void twin_monogenic_anchor_1d(int64_t n, int32_t k) {
  const Shape sh = shape(1, n, 1);
  std::vector<LD> f(n), th(n);
  for (int64_t x = 0; x < n; ++x) {
    th[x] = orient_twin::kPi * static_cast<LD>((k * (2 * x + 1)) % (4 * n)) / static_cast<LD>(2 * n);
    f[x] = std::cos(th[x]);
  }
  const orient_twin::Plans P(sh);
  const orient_twin::Monogenic M = orient_twin::monogenic(P, f);
  const LD one = 1, u = static_cast<LD>(k) / static_cast<LD>(2 * n);
  LD worst = 0;
  for (int64_t x = 0; x < n; ++x) {
    const orient_twin::MonogenicAt t = orient_twin::monogenic_at(M, 1, x, f[x], &one);
    LD dp = std::fabs(t.phase - th[x]);
    dp = std::min(dp, std::fabs(dp - 2 * orient_twin::kPi));
    worst = std::max({worst, std::fabs(t.A - 1), dp, std::fabs(t.freq - u)});
  }
  CHECK(worst <= kTol);
  if (worst > kTol) std::fprintf(stderr, "  monogenic anchor n %ld k %d: %.3Lg\n", static_cast<long>(n), k, worst);
}

// The eigen solver against closed forms: 2x2 eigenvalues and the eigenvector angle, the
// trigonometric eigenvalues of a symmetric 3x3, and A v = mu v.
void twin_eigen() {
  host_twin::Rng rng(77);
  LD worst = 0;
  for (int t = 0; t < 200; ++t) {
    const int D = 2 + t % 2;
    LD A[3][3] = {};
    for (int a = 0; a < D; ++a) {
      for (int c = a; c < D; ++c) A[a][c] = A[c][a] = static_cast<LD>(rng.uniform() * 2 - 1);
    }
    LD mu[3], V[3][3];
    orient_twin::eig(D, A, mu, V);
    LD ref[3];
    LD nrm = 0;
    for (int a = 0; a < D; ++a) {
      for (int c = 0; c < D; ++c) nrm += A[a][c] * A[a][c];
    }
    nrm = std::sqrt(nrm);
    if (D == 2) {
      const LD m = (A[0][0] + A[1][1]) / 2, h = std::hypot((A[0][0] - A[1][1]) / 2, A[0][1]);
      ref[0] = m + h;
      ref[1] = m - h;
      const LD phi = std::atan2(2 * A[0][1], A[0][0] - A[1][1]) / 2;
      worst = std::max(worst, std::fabs(std::fabs(V[0][0] * std::cos(phi) + V[1][0] * std::sin(phi)) - 1) / nrm);
    } else {
      const LD p1 = A[0][1] * A[0][1] + A[0][2] * A[0][2] + A[1][2] * A[1][2];
      const LD q = (A[0][0] + A[1][1] + A[2][2]) / 3;
      const LD p2 = (A[0][0] - q) * (A[0][0] - q) + (A[1][1] - q) * (A[1][1] - q) + (A[2][2] - q) * (A[2][2] - q) + 2 * p1;
      const LD p = std::sqrt(p2 / 6);
      LD Bm[3][3];
      for (int a = 0; a < 3; ++a) {
        for (int c = 0; c < 3; ++c) Bm[a][c] = (A[a][c] - (a == c ? q : 0)) / p;
      }
      const LD det = Bm[0][0] * (Bm[1][1] * Bm[2][2] - Bm[1][2] * Bm[2][1]) -
                     Bm[0][1] * (Bm[1][0] * Bm[2][2] - Bm[1][2] * Bm[2][0]) +
                     Bm[0][2] * (Bm[1][0] * Bm[2][1] - Bm[1][1] * Bm[2][0]);
      const LD phi = std::acos(std::clamp(det / 2, -1.0L, 1.0L)) / 3;
      ref[0] = q + 2 * p * std::cos(phi);
      ref[2] = q + 2 * p * std::cos(phi + 2 * orient_twin::kPi / 3);
      ref[1] = 3 * q - ref[0] - ref[2];
    }
    for (int j = 0; j < D; ++j) {
      worst = std::max(worst, std::fabs(mu[j] - ref[j]) / nrm);
      for (int a = 0; a < D; ++a) {
        LD av = 0;
        for (int c = 0; c < D; ++c) av += A[a][c] * V[c][j];
        worst = std::max(worst, std::fabs(av - mu[j] * V[a][j]) / nrm);
      }
    }
  }
  CHECK(worst <= kTol);
  std::printf("  eigen solver: max deviation from the closed forms %.3Lg of ||A||\n", worst);
}

void twin_self_test() {
  twin_identities(shape(2, 9, 13), 4);
  twin_identities(shape(3, 3, 4, 5), 3);
  twin_identities(shape(3, 5, 6, 7), 3);
  const int32_t k2[][2] = {{2, 3}, {0, 5}, {7, 0}, {8, 12}};
  for (const auto& k : k2) {
    const Shape sh = shape(2, 9, 13);
    LD r2 = 0;
    for (int a = 0; a < 2; ++a) r2 += (k[a] / (2.0L * sh.n[a])) * (k[a] / (2.0L * sh.n[a]));
    twin_anchor(sh, k, static_cast<int32_t>(std::lround(1 / std::sqrt(static_cast<double>(r2)))));
  }
  const int32_t k3[][3] = {{1, 2, 3}, {0, 0, 4}, {4, 5, 6}, {2, 0, 1}};
  for (const auto& k : k3) {
    const Shape sh = shape(3, 5, 6, 7);
    LD r2 = 0;
    for (int a = 0; a < 3; ++a) r2 += (k[a] / (2.0L * sh.n[a])) * (k[a] / (2.0L * sh.n[a]));
    twin_anchor(sh, k, static_cast<int32_t>(std::lround(1 / std::sqrt(static_cast<double>(r2)))));
  }
  for (const int32_t k : {1, 4, 12}) {
    const Shape sh = shape(1, 13, 1);
    twin_anchor(sh, &k, static_cast<int32_t>(std::lround(26.0 / k)));
    twin_monogenic_anchor_1d(13, k);
  }
  twin_monogenic_anchor_1d(64, 63);
  twin_eigen();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--twin") {
    twin_self_test();
    return eda_test::finish("test_orientation_twin");
  }
  try {
    const bool small = eda::sanitize_mode();
    twin_all<double>(small);
    twin_all<float>(small);
    exact_all<double>(small);
    exact_all<float>(small);
    monogenic_all<double>(small);
    monogenic_all<float>(small);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "test_orientation: %s\n", error.what());
    eda::Context::shutdown();
    return 1;
  }
  eda::Context::shutdown();
  return eda_test::finish("test_orientation");
}
