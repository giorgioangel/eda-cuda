// SPDX-License-Identifier: Apache-2.0
// The monogenic orientation and the monogenic signal (method/element.hpp).
//
// The band and its Riesz transforms are filters of the half-sample symmetric extension of r (2 n_a
// samples per axis), whose DFT is the DCT-II of r up to phase factors. They are therefore computed
// in the DCT domain on the N-D Makhoul DCT (field/dct.hpp), at half the transform size:
//   X  = DCT-II(r - mean(r)) (orthonormal), X_b = G(rad) X   G the one-octave log-Gabor at f0 = 1 / w
//   (G(0) = 0 discards the DC: the mean only keeps a large offset's DC rounding out of the band)
//   R1_a  = (-1)^(x_a)       IDCT(rev_{a}   [(u_a / rad) X_b])
//   R2_aa =                  IDCT(          [-(u_a / rad)^2 X_b])
//   R2_ab = (-1)^(x_a + x_b) IDCT(rev_{a,b} [(u_a / rad) (u_b / rad) X_b])   (a < b)
// with u_a = k_a / (2 n_a), rev_A[Y](k) = Y(k~), k~_a = n_a - k_a for a in A (0 where k_a = 0),
// then T = R1 R1^T + R2 R2^T and its eigen-decomposition (eda_element_eigen). The odd Riesz
// components are odd in the extension along their axes: the reversal rev turns the odd-symmetric
// spectrum into one the even DCT can invert, and the sign (-1)^(x_a) undoes it in space.
// The inverse transforms run two components at a time (one in the real part, one in the
// imaginary part of the complex transform, field/dct.hpp), in the order R1_0, R1_1, .., then R2.
//
// The monogenic signal (enqueue_monogenic) takes the same components of the whole spectrum (no
// band, X_b = X) and two more kinds, the derivatives in cycles per sample, D_a = d/dx_a / (2 pi):
//   D_a f   = (-1)^(x_a)       IDCT(rev_{a}   [-u_a X])
//   D_b R_a = (-1)^(x_a + x_b) IDCT(rev_{a,b} [-u_a (u_b / rad) X])   (a < b; symmetric in a, b)
//   D_a R_a =                  IDCT(          [u_a (u_a / rad) X])
// For the plane wave f = cos(2 pi k.x + t0) these are R_a f = (k_a / |k|) sin, R_a R_b f =
// -(k_a k_b / |k|^2) cos, D_b f = -k_b sin and D_b R_a f = (k_a k_b / |k|) cos (cpp/tests'
// twin_orientation.hpp checks them on DCT basis fields).
//
// An element's result depends on its own data only, bit for bit, wherever it sits in the batch and
// in whichever chunk: every kernel reads and writes the element's own fields, the transforms are
// complex ones (a batched complex transform computes each batch entry from that entry alone), and
// they run per group of orientation_group(g) elements with the plan of the group geometry, so that
// the batch and the chunking never change the cuFFT plan an element sees (a plan's internal
// algorithm, and so its rounding, may depend on the batch size).
//
// Determinism: no atomics; the twiddles come from DctPlans (host libm); the frequencies, the band
// and the multipliers are double round-to-nearest intrinsics in a fixed order. The pre-twiddle uses
// plain operators like the DCT's post-twiddle (field/dct.cuh): fixed bits for a given build.
#include "decompose.hpp"
#include "runtime/context.hpp"
#include "runtime/error.hpp"
#include "runtime/memory.hpp"
#include "field/common.cuh"
#include "field/dct.cuh"
#include "field/dct.hpp"
#include "method/method.hpp"
#include "method/element.cuh"
#include "method/element.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace eda {
namespace {

constexpr int kThreads = 256;
// Samples per cuFFT group (a power of two; the group holds floor(kGroupSamples / count) elements,
// 1 to kGroupMax): enough elements per transform to fill the GPU on small fields, a bounded
// workspace on large ones.
constexpr int64_t kGroupSamples = int64_t{1} << 18;
constexpr int64_t kGroupMax = 64;
// The band's centre period in units of the mode window (f0 = 1 / (kBandPeriod w)), and its width:
// 2 sigma^2 with sigma = log(2) / (2 sqrt(2 log 2)) = 0.29435250562886867, the log-Gabor filter
// whose full width at half maximum is one octave. Both in double, in this order of operations.
constexpr double kBandPeriod = 1.0;
// pi rounded to double.
constexpr double kPi = 3.141592653589793;

double band_two_sigma2() {
  // the volatile keeps the compiler from folding the libm calls at compile time
  volatile double two = 2.0;
  const double l2 = std::log(static_cast<double>(two));
  const double s = l2 / (2.0 * std::sqrt(2.0 * l2));
  return 2.0 * (s * s);
}

unsigned grid_for(int64_t work, int per_cta) {
  const int64_t g = (work + per_cta - 1) / per_cta;
  return static_cast<unsigned>(std::clamp<int64_t>(g, 1, int64_t{1} << 20));
}

// W nullptr: every element of the batch runs (the monogenic signal).
__device__ __forceinline__ bool orient_runs(const int32_t* W, const int32_t* on, int64_t b, int64_t B) {
  return b < B && (W == nullptr || W[b] > 0) && (on == nullptr || on[b] != 0);
}

// One component of the Riesz transforms: kind 0 = R1_a, 1 = R2_aa, 2 = R2_ab (a < b), 3 = D_a,
// 4 = D_b R_a (a <= b); rev: bit a set for the odd axes (the spectral reversal and the sign
// (-1)^(x_a) of the header).
struct RieszComp {
  int32_t kind, a, b, rev;
};

// The frequencies of DCT index k in cycles per sample, u_a = k_a (1 / (2 n_a)) (the DFT frequency of
// the 2 n_a-sample extension), and rad = sqrt((0 + u_0^2) + u_1^2 (+ u_2^2)), the sum from 0 in
// axis order.
template <int D>
__device__ __forceinline__ double dct_freq(const DctShape& d, const int32_t* k, double* u) {
  double r2 = 0.0;
#pragma unroll
  for (int a = 0; a < D; ++a) {
    u[a] = __dmul_rn(static_cast<double>(k[a]), __ddiv_rn(1.0, __dmul_rn(2.0, static_cast<double>(d.n[a]))));
    r2 = __dadd_rn(r2, __dmul_rn(u[a], u[a]));
  }
  return __dsqrt_rn(r2);
}

// The component's multiplier at the DCT index k, in double: u_a / rad, -(u_a / rad)^2,
// (u_a / rad) (u_b / rad), -u_a, u_a (u_a / rad) or -u_a (u_b / rad); 0 at the DC, where the Riesz
// transforms are undefined.
template <int D>
__device__ __forceinline__ double riesz_mult(const DctShape& d, const RieszComp& c, const int32_t* k) {
  double u[kMaxDim];
  const double rad = dct_freq<D>(d, k, u);
  if (!(rad > 0.0)) return 0.0;
  const double inv = __ddiv_rn(1.0, rad);
  const double ua = __dmul_rn(u[c.a], inv);
  if (c.kind == 0) return ua;
  if (c.kind == 1) return __dmul_rn(-ua, ua);
  if (c.kind == 3) return -u[c.a];
  if (c.kind == 4) return c.a == c.b ? __dmul_rn(u[c.a], ua) : -__dmul_rn(u[c.a], __dmul_rn(u[c.b], inv));
  return __dmul_rn(ua, __dmul_rn(u[c.b], inv));
}

// mean[gb] = the mean of the group element gb's samples (0 where it does not run or lies beyond the
// batch), in double: one CTA per element, a fixed per-thread order and a fixed tree, so the value
// depends on the element's samples only (never on the group, the chunk or the batch position).
// The band discards the DC (G(0) = 0), so subtracting the mean before the forward DCT changes the
// result by rounding only, and keeps the rounding of a large offset's DC term out of the float32
// band.
template <class T, int D>
__global__ void __launch_bounds__(kThreads) eda_orientation_mean(DctShape d, int64_t B, int64_t b0,
                                                                 const int32_t* __restrict__ W,
                                                                 const int32_t* __restrict__ on,
                                                                 const T* __restrict__ r,
                                                                 double* __restrict__ mean) {
  __shared__ double sh[kThreads];
  const int64_t gb = blockIdx.x;
  if (gb >= d.batch) return;  // uniform per CTA
  const int64_t b = b0 + gb;
  const bool run = orient_runs(W, on, b, B);  // uniform per CTA
  const int32_t nl = d.n[D - 1];
  const int64_t total = d.rows * nl;
  double acc = 0.0;
  if (run) {
    const T* rb = r + b * d.field;
    for (int64_t i = threadIdx.x; i < total; i += kThreads) {
      const int64_t q = i / nl;
      acc = __dadd_rn(acc, static_cast<double>(rb[q * d.pitch + (i - q * nl)]));
    }
  }
  sh[threadIdx.x] = acc;
  __syncthreads();
#pragma unroll
  for (int st = kThreads / 2; st > 0; st >>= 1) {
    if (threadIdx.x < st) sh[threadIdx.x] = __dadd_rn(sh[threadIdx.x], sh[threadIdx.x + st]);
    __syncthreads();
  }
  if (threadIdx.x == 0) mean[gb] = run ? __ddiv_rn(sh[0], static_cast<double>(total)) : 0.0;
}

// V = Makhoul's permutation of r - mean, imaginary part 0, for the group's elements b0 .. b0 + k - 1
// (zeros where the element does not run or lies beyond the batch; the difference in double, rounded
// to T). d: the group's shape (batch k); r: the batch's fields; mean: the group's means
// (eda_orientation_mean). Every value of V is written.
template <class T, int D>
__global__ void __launch_bounds__(kThreads) eda_orientation_permute(DctShape d, int64_t B, int64_t b0,
                                                                    const int32_t* __restrict__ W,
                                                                    const int32_t* __restrict__ on,
                                                                    const T* __restrict__ r,
                                                                    const double* __restrict__ mean,
                                                                    cplx_t<T>* __restrict__ V) {
  const int32_t nl = d.n[D - 1];
  const int64_t total = d.batch * d.rows * nl;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t q = idx / nl;
    const int32_t m = static_cast<int32_t>(idx - q * nl);
    const int64_t gb = q / d.rows;
    const int64_t row = q - gb * d.rows;
    const int64_t b = b0 + gb;
    T v = T(0);
    if (orient_runs(W, on, b, B)) {
      int32_t j[kMaxDim];
      dct_split_row<D>(d, row, j);
#pragma unroll
      for (int a = 0; a < D - 1; ++a) j[a] = dct_src(j[a], d.n[a]);
      const T x = r[b * d.field + dct_row<D>(d, j) * d.pitch + dct_src(m, nl)];
      v = static_cast<T>(__dsub_rn(static_cast<double>(x), mean[gb]));
    }
    V[gb * d.cfield + row * nl + m] = cplx_t<T>{v, T(0)};
  }
}

// X = G(rad) X in place (the band of the log-Gabor at f0 = 1 / (kBandPeriod w), w = le[b].w):
// G = exp(-(log(rad / f0))^2 / (2 sigma^2)) where rad > 0, else 0; the product in double, then
// rounded once to T.
template <class T, int D>
__global__ void __launch_bounds__(kThreads) eda_orientation_band(DctShape d, int64_t B, int64_t b0,
                                                                 const int32_t* __restrict__ W,
                                                                 const int32_t* __restrict__ on,
                                                                 const ElemRecord* __restrict__ le,
                                                                 double two_sigma2, T* __restrict__ X) {
  const int32_t nl = d.n[D - 1];
  const int64_t total = d.batch * d.rows * nl;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t q = idx / nl;
    const int64_t gb = q / d.rows;
    const int64_t row = q - gb * d.rows;
    const int64_t b = b0 + gb;
    if (!orient_runs(W, on, b, B)) continue;
    int32_t k[kMaxDim];
    dct_split_row<D>(d, row, k);
    k[D - 1] = static_cast<int32_t>(idx - q * nl);
    double u[kMaxDim];
    const double rad = dct_freq<D>(d, k, u);
    double G = 0.0;
    if (rad > 0.0) {
      const double f0 = __ddiv_rn(1.0, __dmul_rn(kBandPeriod, static_cast<double>(le[b].w)));
      const double l = log(__ddiv_rn(rad, f0));
      G = exp(__ddiv_rn(-__dmul_rn(l, l), two_sigma2));
    }
    T* p = X + gb * d.field + row * d.pitch + k[D - 1];
    *p = static_cast<T>(__dmul_rn(static_cast<double>(*p), G));
  }
}

// The inverse-DCT pre-twiddle of one orbit (the inverse map of field/dct.cuh) into U, the spectrum
// values of the 2^D orbit members, whose source value at the orbit member j is m(j~) X_b(j~):
// j~_a = n_a - j_a on the component's odd axes (0, i.e. no value, where j_a = 0: the extension's
// Nyquist index, where the DCT-II is exactly 0), j_a otherwise. The 1/N of the inverse is folded
// into the scale of every value.
template <class T, int D>
__device__ __forceinline__ void riesz_pre_orbit(const DctShape& d, const DctTw<T>& tw, const RieszComp& comp,
                                                const int32_t* k, const T* __restrict__ X, T inv_n, bool run,
                                                cplx_t<T>* U) {
  using C = cplx_t<T>;
  constexpr int NB = 1 << D;
#pragma unroll
  for (int u = 0; u < NB; ++u) {
    bool valid = run;
    T iscale = inv_n;
    int32_t j[kMaxDim];
#pragma unroll
    for (int a = 0; a < D; ++a) {
      const bool hi = (u >> a) & 1;
      if (hi && k[a] == 0) valid = false;
      j[a] = hi ? d.n[a] - k[a] : k[a];
      iscale *= tw.isc[d.tw_off[a] + j[a]];
    }
    U[u] = C{T(0), T(0)};
    if (valid) {
      int32_t s[kMaxDim];
      bool src = true;
#pragma unroll
      for (int a = 0; a < D; ++a) {
        if ((comp.rev >> a) & 1) {
          if (j[a] == 0) src = false;
          s[a] = d.n[a] - j[a];
        } else {
          s[a] = j[a];
        }
      }
      if (src) {
        const double m = riesz_mult<D>(d, comp, s);
        const T v = static_cast<T>(__dmul_rn(m, static_cast<double>(X[dct_row<D>(d, s) * d.pitch + s[D - 1]])));
        U[u] = C{v * iscale, T(0)};
      }
    }
  }
#pragma unroll
  for (int a = 0; a < D; ++a) {
    const int32_t off = d.tw_off[a];
    const T c = tw.cs[off + k[a]], sn = tw.sn[off + k[a]];
#pragma unroll
    for (int u = 0; u < NB; ++u) {
      if ((u >> a) & 1) continue;
      const C x0 = U[u], x1 = U[u | (1 << a)];
      const T mx = x0.x + x1.y, my = x0.y - x1.x;  // x0 - i x1
      const T px = x0.x - x1.y, py = x0.y + x1.x;  // x0 + i x1
      U[u] = C{c * mx - sn * my, c * my + sn * mx};           // conj(phi) (x0 - i x1)
      U[u | (1 << a)] = C{c * px + sn * py, c * py - sn * px};  // phi (x0 + i x1)
    }
  }
}

// V = U1 + i U2 of the components c1 and c2 (two = 0: c1 alone, V = U1) over the whole spectrum:
// every orbit member that is a distinct index is written (zeros where the element does not run).
template <class T, int D>
__global__ void __launch_bounds__(kThreads) eda_orientation_pre(DctShape d, DctTw<T> tw, RieszComp c1, RieszComp c2,
                                                                int32_t two, int64_t B, int64_t b0,
                                                                const int32_t* __restrict__ W,
                                                                const int32_t* __restrict__ on,
                                                                const T* __restrict__ X,
                                                                cplx_t<T>* __restrict__ V) {
  using C = cplx_t<T>;
  constexpr int NB = 1 << D;
  const int64_t total = d.batch * d.orbits;
  int64_t count = 1;
  for (int a = 0; a < D; ++a) count *= d.n[a];
  const T inv_n = static_cast<T>(1.0 / static_cast<double>(count));
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t gb = idx / d.orbits;
    int32_t k[kMaxDim];
    dct_orbit<D>(d, idx - gb * d.orbits, k);
    const bool run = orient_runs(W, on, b0 + gb, B);
    C U1[NB], U2[NB];
    riesz_pre_orbit<T, D>(d, tw, c1, k, X + gb * d.field, inv_n, run, U1);
    if (two) riesz_pre_orbit<T, D>(d, tw, c2, k, X + gb * d.field, inv_n, run, U2);
    C* Vb = V + gb * d.cfield;
#pragma unroll
    for (int u = 0; u < NB; ++u) {
      if (dct_member<D>(d, k, u)) Vb[dct_coff<D>(d, k, u)] = two ? C{U1[u].x - U2[u].y, U1[u].y + U2[u].x} : U1[u];
    }
  }
}

// out1 = (-1)^(sum_{a in rev1} x_a) Re unpermute(V) and out2 = (-1)^(sum_{a in rev2} x_a) Im
// unpermute(V) (out2 nullptr: none) for the group's running elements (out1, out2: the batch's
// fields of the two components).
template <class T, int D>
__global__ void __launch_bounds__(kThreads) eda_orientation_unpermute(DctShape d, int32_t rev1, int32_t rev2,
                                                                      int64_t B, int64_t b0,
                                                                      const int32_t* __restrict__ W,
                                                                      const int32_t* __restrict__ on,
                                                                      const cplx_t<T>* __restrict__ V,
                                                                      T* __restrict__ out1, T* __restrict__ out2) {
  const int32_t nl = d.n[D - 1];
  const int64_t total = d.batch * d.rows * nl;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t q = idx / nl;
    const int32_t x = static_cast<int32_t>(idx - q * nl);
    const int64_t gb = q / d.rows;
    const int64_t row = q - gb * d.rows;
    const int64_t b = b0 + gb;
    if (!orient_runs(W, on, b, B)) continue;
    int32_t jo[kMaxDim], js[kMaxDim];
    dct_split_row<D>(d, row, jo);
    jo[D - 1] = x;
    int p1 = 0, p2 = 0;
#pragma unroll
    for (int a = 0; a < D; ++a) {
      if ((rev1 >> a) & 1) p1 ^= jo[a] & 1;
      if ((rev2 >> a) & 1) p2 ^= jo[a] & 1;
    }
#pragma unroll
    for (int a = 0; a < D - 1; ++a) js[a] = dct_dst(jo[a], d.n[a]);
    const cplx_t<T> v = V[gb * d.cfield + dct_row<D>(d, js) * nl + dct_dst(x, nl)];
    const int64_t o = b * d.field + row * d.pitch + x;
    out1[o] = p1 ? -v.x : v.x;
    if (out2 != nullptr) out2[o] = p2 ? -v.y : v.y;
  }
}

// T = R1 R1^T + R2 R2^T per voxel (v = R1_a R1_b, then v = v + h(a, k) h(k, b) for k = 0..D-1, h the
// symmetric R2), written over the R2 fields (work, the eigen kernel's order).
template <int D>
__host__ __device__ constexpr int tensor_pair(int a, int b) {
  // (0,0) 0, (0,1) 1, (0,2) 2, (1,1) 3, (1,2) 4, (2,2) 5 in 3D; (0,0) 0, (0,1) 1, (1,1) 2 in 2D
  return D == 2 ? (a == 0 ? b : 2) : (a == 0 ? b : (a == 1 ? 2 + b : 5));
}

template <class T, int D>
__global__ void __launch_bounds__(kThreads) eda_orientation_tensor(Geom g, const int32_t* __restrict__ W,
                                                                   const int32_t* __restrict__ on,
                                                                   FieldVec<const T> r1, T* __restrict__ work) {
  constexpr int K = D * (D + 1) / 2;
  const int64_t total = g.batch * g.field;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t b = i / g.field;
    if (!orient_runs(W, on, b, g.batch)) continue;
    if ((i - b * g.field) % g.pitch >= g.n[D - 1]) continue;  // pad samples stay as they are
    T R1[D], R2[K];
#pragma unroll
    for (int a = 0; a < D; ++a) R1[a] = r1.p[a][i];
#pragma unroll
    for (int p = 0; p < K; ++p) R2[p] = work[p * total + i];
    T out[K];
#pragma unroll
    for (int a = 0; a < D; ++a) {
#pragma unroll
      for (int c = a; c < D; ++c) {
        T v = mul_rn(R1[a], R1[c]);
#pragma unroll
        for (int k = 0; k < D; ++k) {
          const T hak = R2[tensor_pair<D>(a < k ? a : k, a < k ? k : a)];
          const T hkc = R2[tensor_pair<D>(k < c ? k : c, k < c ? c : k)];
          v = add_rn(v, mul_rn(hak, hkc));
        }
        out[tensor_pair<D>(a, c)] = v;
      }
    }
#pragma unroll
    for (int p = 0; p < K; ++p) work[p * total + i] = out[p];
  }
}

// The canonical sign of a unit direction: negated when its first component, in axis order, whose
// magnitude exceeds `zero` is negative (a direction and its negation describe the same
// orientation). zero is the default tie tolerance of the dtype (2^-14 float32, 2^-30 float64): a
// component of a pattern that does not vary along its axis is rounding noise, whose sign is
// arbitrary, so it does not decide.
template <class T, int D>
__device__ __forceinline__ void canonical_sign(T (&v)[D], T zero) {
  int s = 0;
#pragma unroll
  for (int a = 0; a < D; ++a) s = s == 0 && fabs(v[a]) > zero ? (v[a] > T(0) ? 1 : -1) : s;
#pragma unroll
  for (int a = 0; a < D; ++a) v[a] = s < 0 ? -v[a] : v[a];
}

// The monogenic signal per voxel into the call's output planes (enqueue_monogenic). In double, in
// this order: n canonical (D >= 2; n = 1 in 1D), q = sum_a n_a R1_a, A^2 = f^2 + sum_a R1_a^2 (the
// sums from f^2 or 0 in axis order), D_n f = sum_b n_b D_b f, D_n q = sum_{a <= b} m_ab D_b R_a with
// m_aa = n_a n_a and m_ab = 2 (n_a n_b) (pair order); where A^2 > tiny: phase = atan2(q, f) (a
// phase that rounds to -pi in T is pi) and frequency = (f D_n q - q D_n f) / A^2, else both 0; amplitude = sqrt(A^2) 2^exp[b], rounded to T
// once after the exact product. The fields hold r 2^-exp[b] (the prescale), so the phase, the
// frequency, n and c do not depend on a power-of-two gain.
template <class T>
struct MonogenicIn {
  const T* f;
  FieldVec<const T> r1, df, n;
  const T* dr;
  const T* c;
  const int32_t* exp;
};

template <class T, int D>
__global__ void __launch_bounds__(kThreads) eda_orientation_monogenic(Geom g, MonogenicIn<T> in, double tiny, T zero,
                                                                      T* __restrict__ out, int64_t out_batch,
                                                                      int64_t b0) {
  const int64_t total = g.batch * g.field;
  const int32_t nl = g.n[D - 1];
  const int64_t plane = out_batch * g.count;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t b = i / g.field;
    const int64_t v = i - b * g.field;
    const int64_t row = v / g.pitch;
    const int64_t col = v - row * g.pitch;
    if (col >= nl) continue;
    T nt[D];
#pragma unroll
    for (int a = 0; a < D; ++a) nt[a] = D == 1 ? T(1) : in.n.p[a][i];
    canonical_sign<T, D>(nt, zero);
    double n[D];
#pragma unroll
    for (int a = 0; a < D; ++a) n[a] = static_cast<double>(nt[a]);
    const double f = static_cast<double>(in.f[i]);
    double q = 0.0, a2 = __dmul_rn(f, f), dnf = 0.0, dnq = 0.0;
#pragma unroll
    for (int a = 0; a < D; ++a) {
      const double r = static_cast<double>(in.r1.p[a][i]);
      q = __dadd_rn(q, __dmul_rn(n[a], r));
      a2 = __dadd_rn(a2, __dmul_rn(r, r));
      dnf = __dadd_rn(dnf, __dmul_rn(n[a], static_cast<double>(in.df.p[a][i])));
    }
    int p = 0;
#pragma unroll
    for (int a = 0; a < D; ++a) {
#pragma unroll
      for (int c = a; c < D; ++c, ++p) {
        const double m = a == c ? __dmul_rn(n[a], n[a]) : __dmul_rn(2.0, __dmul_rn(n[a], n[c]));
        dnq = __dadd_rn(dnq, __dmul_rn(m, static_cast<double>(in.dr[p * total + i])));
      }
    }
    double phase = 0.0, freq = 0.0;
    if (a2 > tiny) {
      phase = atan2(q, f);
      freq = __ddiv_rn(__dsub_rn(__dmul_rn(f, dnq), __dmul_rn(q, dnf)), a2);
    }
    const int64_t at = (b0 + b) * g.count + row * nl + col;
    out[at] = static_cast<T>(ldexp(__dsqrt_rn(a2), in.exp[b]));
    // (-pi, pi] as the dtype holds it: an angle that rounds to -pi is pi
    const T pt = static_cast<T>(phase);
    out[plane + at] = pt == -static_cast<T>(kPi) ? static_cast<T>(kPi) : pt;
    out[2 * plane + at] = static_cast<T>(freq);
    if constexpr (D >= 2) {
      const int64_t e = (b0 + b) * D * g.count + row * nl + col;
#pragma unroll
      for (int a = 0; a < D; ++a) out[3 * plane + e + a * g.count] = nt[a];
      out[(3 + D) * plane + at] = in.c[i];
    }
  }
}

// Row elems[b].n_imfs of out (B, rows, D + 1, count) <- n with the canonical sign (components 0 ..
// D - 1) and c (component D), unpadded, for the elements whose mode is active.
template <class T, int D>
__global__ void __launch_bounds__(kThreads) eda_orientation_store(Geom g, FieldVec<const T> n,
                                                                  const T* __restrict__ c,
                                                                  const ElemRecord* __restrict__ le,
                                                                  const ElemState* __restrict__ elems, T zero,
                                                                  T* __restrict__ out, int32_t rows) {
  const int64_t total = g.batch * g.field;
  const int32_t nl = g.n[D - 1];
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t b = i / g.field;
    if (le[b].status != kRecordActive) continue;
    const int64_t v = i - b * g.field;
    const int64_t row = v / g.pitch;
    const int64_t col = v - row * g.pitch;
    if (col >= nl) continue;
    T nt[D];
#pragma unroll
    for (int a = 0; a < D; ++a) nt[a] = n.p[a][i];
    canonical_sign<T, D>(nt, zero);
    T* o = out + ((b * rows + elems[b].n_imfs) * (D + 1)) * g.count + row * nl + col;
#pragma unroll
    for (int a = 0; a < D; ++a) o[a * g.count] = nt[a];
    o[D * g.count] = c[i];
  }
}

// dir (B, K, D, count) and coh (B, K, count) <- the rows 0 .. n_imfs[b] - 1 of src (B, rows, D + 1,
// count); 0 above them and for elements with error[b] != 0.
template <class T>
__global__ void __launch_bounds__(kThreads) eda_orientation_collect(int64_t B, int32_t D, int64_t count,
                                                                    const int32_t* __restrict__ n_imfs,
                                                                    const int32_t* __restrict__ error,
                                                                    const T* __restrict__ src, int32_t rows,
                                                                    T* __restrict__ dir, T* __restrict__ coh,
                                                                    int32_t K) {
  const int64_t total = B * K * (D + 1) * count;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t i = idx % count;
    const int64_t q = idx / count;
    const int32_t a = static_cast<int32_t>(q % (D + 1));
    const int64_t bk = q / (D + 1);
    const int64_t k = bk % K;
    const int64_t b = bk / K;
    const T v = error[b] == 0 && k < n_imfs[b] ? src[((b * rows + k) * (D + 1) + a) * count + i] : T(0);
    if (a < D) {
      dir[(bk * D + a) * count + i] = v;
    } else {
      coh[bk * count + i] = v;
    }
  }
}

template <class F>
void dispatch_dim(int dim, F&& f) {
  switch (dim) {
    case 1: f(std::integral_constant<int, 1>{}); break;
    case 2: f(std::integral_constant<int, 2>{}); break;
    case 3: f(std::integral_constant<int, 3>{}); break;
    default: throw std::invalid_argument("eda: orientation: D must be 1, 2 or 3");
  }
}

template <class F>
void dispatch_dim23(int dim, F&& f) {
  switch (dim) {
    case 2: f(std::integral_constant<int, 2>{}); break;
    case 3: f(std::integral_constant<int, 3>{}); break;
    default: throw std::invalid_argument("eda: orientation: D must be 2 or 3");
  }
}

Geom group_geom(const Geom& g) {
  int64_t n[kMaxDim] = {g.n[0], g.n[1], g.n[2]};
  return make_geom(g.dim, orientation_group(g), n);
}

// The group workspace inside ws: X (k padded real fields), V (k dense complex spectra), M (k
// doubles: the elements' means), 256-byte aligned.
struct OrientationWs {
  size_t X, V, M, bytes;
};
OrientationWs orientation_ws(const Geom& g, size_t es) {
  const DctShape d = dct_shape(group_geom(g));
  const auto up = [](size_t v) { return (v + 255) / 256 * 256; };
  OrientationWs w{};
  const size_t f = static_cast<size_t>(d.batch) * static_cast<size_t>(d.field) * es;
  w.X = 0;
  w.V = up(f);
  w.M = w.V + up(static_cast<size_t>(d.batch) * static_cast<size_t>(d.cfield) * 2 * es);
  w.bytes = w.M + up(static_cast<size_t>(d.batch) * sizeof(double));
  return w;
}

// The per-context plan cache: the plans of the group geometries of the most recent calls. Plans are
// slow to create and hold device memory, so a few are kept.
struct OrientationPlanCache final : ArenaAttachment {
  static constexpr size_t kPlans = 4;
  std::vector<std::pair<std::vector<int64_t>, std::unique_ptr<DctPlans>>> plans;  // most recent first
};

// R1_a, R2_ab (a <= b) and the derivatives D_a, D_b R_a (a <= b) (the header's kinds).
RieszComp comp_first(int kind, int a) { return RieszComp{kind, a, a, 1 << a}; }
RieszComp comp_pair(int kind, int a, int b) {
  return RieszComp{kind == 1 && a != b ? 2 : kind, a, b, a == b ? 0 : ((1 << a) | (1 << b))};
}

// Appends the components of one kind (0, 3: per axis; 1, 4: per pair a <= b in the eigen order)
// with their output fields (per axis: vec.p[a]; per pair: block + p field).
template <class T>
void add_comps(std::vector<RieszComp>& comp, std::vector<T*>& out, int dim, int kind, FieldVec<T> vec, T* block,
               int64_t field) {
  int p = 0;
  for (int a = 0; a < dim; ++a) {
    if (kind == 0 || kind == 3) {
      comp.push_back(comp_first(kind, a));
      out.push_back(vec.p[a]);
      continue;
    }
    for (int b = a; b < dim; ++b, ++p) {
      comp.push_back(comp_pair(kind, a, b));
      out.push_back(block + static_cast<int64_t>(p) * field);
    }
  }
}

}  // namespace

int64_t orientation_group(const Geom& g) {
  const int64_t count = g.count > 0 ? g.count : 1;
  return std::clamp<int64_t>(kGroupSamples / count, 1, kGroupMax);
}

size_t orientation_workspace_bytes(const Geom& g, DType dtype) {
  return orientation_ws(g, dtype == DType::F32 ? sizeof(float) : sizeof(double)).bytes;
}

const DctPlans& orientation_plans(Context& ctx, const Geom& g, DType dtype) {
  auto& cache = ctx.arena().attachment<OrientationPlanCache>([] { return std::make_unique<OrientationPlanCache>(); });
  const Geom gg = group_geom(g);
  const bool f64 = dtype == DType::F64;
  const std::vector<int64_t> key = {f64 ? 1 : 0, gg.dim, gg.n[0], gg.n[1], gg.n[2], gg.batch};
  for (size_t i = 0; i < cache.plans.size(); ++i) {
    if (cache.plans[i].first == key) {
      if (i) {
        std::rotate(cache.plans.begin(), cache.plans.begin() + static_cast<std::ptrdiff_t>(i),
                    cache.plans.begin() + static_cast<std::ptrdiff_t>(i) + 1);
      }
      return *cache.plans.front().second;
    }
  }
  if (cache.plans.size() >= OrientationPlanCache::kPlans) {
    // queued work may still use the plan being evicted
    ctx.synchronize();
    cache.plans.pop_back();
  }
  std::unique_ptr<DctPlans> p;
  try {
    p = std::make_unique<DctPlans>(gg, f64, ctx.stream());
  } catch (const OutOfDeviceMemory&) {
    // cuFFT allocates its memory outside the library's pools: give their unused memory back to the
    // driver (once the stream's pending frees are done) and try once more
    ctx.synchronize();
    trim_device_pools(ctx.device());
    p = std::make_unique<DctPlans>(gg, f64, ctx.stream());
  }
  cache.plans.emplace(cache.plans.begin(), key, std::move(p));
  return *cache.plans.front().second;
}

namespace {

// One pass over the groups of orientation_group(g) elements: the mean, the forward DCT, the band
// (band: the orientation's, at le[b].w; else none) and the inverse transforms of comp[i] into
// out[i], two at a time, in the order of comp.
template <class T>
void riesz_pass(cudaStream_t s, const Geom& g, const DctPlans& plans, const T* r, const int32_t* W, const int32_t* on,
                const ElemRecord* le, bool band, const std::vector<RieszComp>& comp, const std::vector<T*>& out,
                void* ws, const uint8_t* host_run) {
  const DctShape& d = plans.shape();
  const int64_t kg = orientation_group(g);
  if (d.batch != kg || d.dim != g.dim || d.n[0] != g.n[0] || d.n[1] != g.n[1] || d.n[2] != g.n[2] ||
      d.field != g.field) {
    throw std::invalid_argument("eda: orientation: the plans belong to another geometry");
  }
  if ((sizeof(T) == sizeof(double)) != plans.f64()) throw std::invalid_argument("eda: orientation: plan dtype mismatch");
  const OrientationWs mw = orientation_ws(g, sizeof(T));
  unsigned char* base = static_cast<unsigned char*>(ws);
  T* X = reinterpret_cast<T*>(base + mw.X);
  auto* V = reinterpret_cast<cplx_t<T>*>(base + mw.V);
  double* M = reinterpret_cast<double*>(base + mw.M);
  const DctTw<T> tw{static_cast<const T*>(plans.tw_cos()), static_cast<const T*>(plans.tw_sin()),
                    static_cast<const T*>(plans.tw_scale()), static_cast<const T*>(plans.tw_iscale())};
  const int ncomp = static_cast<int>(comp.size());
  const double two_sigma2 = band_two_sigma2();
  const int64_t B = g.batch;
  dispatch_dim(g.dim, [&](auto Dc) {
    constexpr int D = Dc();
    for (int64_t b0 = 0; b0 < B; b0 += kg) {
      // host_run skips only the transforms of a group none of whose elements runs; the tensor and
      // eigen passes (orientation_finish) still run on every element that W and on select
      if (host_run) {
        bool any = false;
        for (int64_t b = b0; b < std::min(B, b0 + kg); ++b) any = any || host_run[b] != 0;
        if (!any) continue;
      }
      const unsigned gr = grid_for(d.batch * d.rows * d.n[D - 1], kThreads * 4);
      eda_orientation_mean<T, D><<<static_cast<unsigned>(d.batch), kThreads, 0, s>>>(d, B, b0, W, on, r, M);
      EDA_CHECK_LAUNCH("eda_orientation_mean");
      eda_orientation_permute<T, D><<<gr, kThreads, 0, s>>>(d, B, b0, W, on, r, M, V);
      EDA_CHECK_LAUNCH("eda_orientation_permute");
      plans.exec_forward(s, V);
      enqueue_dct_post<T>(s, plans, V, X);
      if (band) {
        eda_orientation_band<T, D><<<gr, kThreads, 0, s>>>(d, B, b0, W, on, le, two_sigma2, X);
        EDA_CHECK_LAUNCH("eda_orientation_band");
      }
      for (int ci = 0; ci < ncomp; ci += 2) {
        const bool two = ci + 1 < ncomp;
        const RieszComp c1 = comp[ci], c2 = two ? comp[ci + 1] : c1;
        eda_orientation_pre<T, D><<<grid_for(d.batch * d.orbits, kThreads), kThreads, 0, s>>>(
            d, tw, c1, c2, two ? 1 : 0, B, b0, W, on, X, V);
        EDA_CHECK_LAUNCH("eda_orientation_pre");
        plans.exec_inverse(s, V);
        eda_orientation_unpermute<T, D><<<gr, kThreads, 0, s>>>(d, c1.rev, two ? c2.rev : 0, B, b0, W, on, V,
                                                                out[ci], two ? out[ci + 1] : nullptr);
        EDA_CHECK_LAUNCH("eda_orientation_unpermute");
      }
    }
  });
}

// The two halves of the orientation: the Riesz components into r1 / work, then T in place over
// work and the eigen pass into n, c.
template <class T>
void riesz_components(cudaStream_t s, const Geom& g, const DctPlans& plans, const T* r, const int32_t* W, FieldVec<T> r1,
                      T* work, void* ws, const int32_t* on, const ElemRecord* le, const uint8_t* host_run) {
  if (g.dim == 1) return;
  if (r == nullptr || W == nullptr || work == nullptr || ws == nullptr || le == nullptr) {
    throw std::invalid_argument("eda: orientation: null argument");
  }
  for (int a = 0; a < g.dim; ++a) {
    if (r1.p[a] == nullptr) throw std::invalid_argument("eda: orientation: null R1 component");
  }
  const int64_t field = static_cast<int64_t>(g.batch) * g.field;
  std::vector<RieszComp> comp;
  std::vector<T*> out;
  add_comps<T>(comp, out, g.dim, 0, r1, nullptr, field);
  add_comps<T>(comp, out, g.dim, 1, FieldVec<T>{}, work, field);
  riesz_pass<T>(s, g, plans, r, W, on, le, true, comp, out, ws, host_run);
}

template <class T>
void orientation_finish(cudaStream_t s, const Geom& g, const int32_t* W, FieldVec<T> r1, T* work, FieldVec<T> n, T* c,
                        const int32_t* on, ElemRecord* le) {
  if (g.dim == 1) return;
  if (work == nullptr || c == nullptr) throw std::invalid_argument("eda: orientation: null argument");
  for (int a = 0; a < g.dim; ++a) {
    if (r1.p[a] == nullptr || n.p[a] == nullptr) throw std::invalid_argument("eda: orientation: null vector component");
  }
  FieldVec<const T> rc{};
  for (int a = 0; a < kMaxDim; ++a) rc.p[a] = r1.p[a];
  dispatch_dim23(g.dim, [&](auto Dc) {
    constexpr int D = Dc();
    eda_orientation_tensor<T, D><<<grid_for(g.batch * g.field, kThreads * 4), kThreads, 0, s>>>(g, W, on, rc,
                                                                                                work);
    EDA_CHECK_LAUNCH("eda_orientation_tensor");
  });
  enqueue_element_eigen<T>(s, g, work, W, n, c, on, le);
}

}  // namespace

template <class T>
void enqueue_orientation(cudaStream_t s, const Geom& g, const DctPlans& plans, const T* r,
                         const int32_t* W, FieldVec<T> n, T* c, T* work, FieldVec<T> r1, void* ws,
                         const int32_t* on, ElemRecord* le, const uint8_t* host_run) {
  if (g.dim == 1) return;
  riesz_components<T>(s, g, plans, r, W, r1, work, ws, on, le, host_run);
  orientation_finish<T>(s, g, W, r1, work, n, c, on, le);
}

int32_t monogenic_rows(int dim) { return dim == 1 ? 3 : 4 + dim; }

namespace {
// The `zero` of canonical_sign: the dtype's default tie tolerance.
template <class T>
T sign_zero() {
  return static_cast<T>(default_tie_tolerance(sizeof(T) == 4 ? DType::F32 : DType::F64));
}
}  // namespace

template <class T>
void enqueue_monogenic(cudaStream_t s, const Geom& g, const DctPlans& plans, const T* f, const int32_t* exp,
                       const MonogenicFields<T>& w, void* ws, double tiny, T* out, int64_t out_batch, int64_t b0) {
  const int D = g.dim;
  bool ok = f != nullptr && exp != nullptr && w.work != nullptr && ws != nullptr && out != nullptr;
  for (int a = 0; a < D; ++a) ok = ok && w.r1.p[a] != nullptr && w.df.p[a] != nullptr && (D == 1 || w.n.p[a] != nullptr);
  if (!ok || (D >= 2 && w.c == nullptr)) throw std::invalid_argument("eda: monogenic: null argument");
  if (b0 < 0 || b0 + g.batch > out_batch) throw std::invalid_argument("eda: monogenic: the chunk lies outside the output");
  const int64_t field = static_cast<int64_t>(g.batch) * g.field;
  // pass 1: R1, R2 (D >= 2: then T over work and n, c; the work fields are free again) and D_a f;
  // in 1D the pass also takes D_0 R_0 into work, and there is no orientation
  std::vector<RieszComp> comp;
  std::vector<T*> out1;
  add_comps<T>(comp, out1, D, 0, w.r1, nullptr, field);
  if (D >= 2) add_comps<T>(comp, out1, D, 1, FieldVec<T>{}, w.work, field);
  add_comps<T>(comp, out1, D, 3, w.df, nullptr, field);
  if (D == 1) add_comps<T>(comp, out1, D, 4, FieldVec<T>{}, w.work, field);
  riesz_pass<T>(s, g, plans, f, nullptr, nullptr, nullptr, false, comp, out1, ws, nullptr);
  if (D >= 2) {
    orientation_finish<T>(s, g, nullptr, w.r1, w.work, w.n, w.c, nullptr, nullptr);
    // pass 2: D_b R_a over work
    comp.clear();
    out1.clear();
    add_comps<T>(comp, out1, D, 4, FieldVec<T>{}, w.work, field);
    riesz_pass<T>(s, g, plans, f, nullptr, nullptr, nullptr, false, comp, out1, ws, nullptr);
  }
  MonogenicIn<T> in{};
  in.f = f;
  for (int a = 0; a < D; ++a) {
    in.r1.p[a] = w.r1.p[a];
    in.df.p[a] = w.df.p[a];
    in.n.p[a] = w.n.p[a];
  }
  in.dr = w.work;
  in.c = w.c;
  in.exp = exp;
  dispatch_dim(D, [&](auto Dc) {
    eda_orientation_monogenic<T, Dc()><<<grid_for(field, kThreads * 4), kThreads, 0, s>>>(g, in, tiny, sign_zero<T>(),
                                                                                         out, out_batch, b0);
  });
  EDA_CHECK_LAUNCH("eda_orientation_monogenic");
}

template <class T>
void enqueue_orientation_store(cudaStream_t s, const Geom& g, FieldVec<const T> n, const T* c, const ElemRecord* le,
                               const ElemState* elems, T* out, int32_t rows) {
  if (c == nullptr || le == nullptr || elems == nullptr || out == nullptr || rows < 1) {
    throw std::invalid_argument("eda: orientation store: null argument");
  }
  dispatch_dim23(g.dim, [&](auto Dc) {
    eda_orientation_store<T, Dc()><<<grid_for(g.batch * g.field, kThreads * 4), kThreads, 0, s>>>(
        g, n, c, le, elems, sign_zero<T>(), out, rows);
  });
  EDA_CHECK_LAUNCH("eda_orientation_store");
}

template <class T>
void enqueue_orientation_collect(cudaStream_t s, int64_t batch, int dim, int64_t count, const int32_t* n_imfs,
                                 const int32_t* error, const T* src, int32_t rows, T* dir, T* coh, int32_t K) {
  if (batch < 1 || K < 1 || count < 1) return;
  if (K > rows) throw std::invalid_argument("eda: orientation collect: more modes than rows");
  eda_orientation_collect<T><<<grid_for(batch * K * (dim + 1) * count, kThreads * 4), kThreads, 0, s>>>(
      batch, dim, count, n_imfs, error, src, rows, dir, coh, K);
  EDA_CHECK_LAUNCH("eda_orientation_collect");
}

#define EDA_ORIENTATION_INST(T)                                                                                  \
  template void enqueue_orientation<T>(cudaStream_t, const Geom&, const DctPlans&, const T*,          \
                                                 const int32_t*, FieldVec<T>, T*, T*, FieldVec<T>, void*,           \
                                                 const int32_t*, ElemRecord*, const uint8_t*);                      \
  template void enqueue_monogenic<T>(cudaStream_t, const Geom&, const DctPlans&, const T*, const int32_t*,       \
                                     const MonogenicFields<T>&, void*, double, T*, int64_t, int64_t);            \
  template void enqueue_orientation_store<T>(cudaStream_t, const Geom&, FieldVec<const T>, const T*,            \
                                             const ElemRecord*, const ElemState*, T*, int32_t);                  \
  template void enqueue_orientation_collect<T>(cudaStream_t, int64_t, int, int64_t, const int32_t*,            \
                                               const int32_t*, const T*, int32_t, T*, T*, int32_t);
EDA_ORIENTATION_INST(float)
EDA_ORIENTATION_INST(double)
#undef EDA_ORIENTATION_INST

}  // namespace eda
