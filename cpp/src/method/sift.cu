// SPDX-License-Identifier: Apache-2.0
// The pointwise parts of the continuation weight and of G1, and the sift update with the sup-norm
// guard (method/mode.hpp).
//
// Every kernel runs over a 2D grid: blockIdx.y strides over the elements (an element that does not
// run costs one test per CTA), blockIdx.x and the threads over the element's padded field, pads
// skipped. Arithmetic in T by the round-to-nearest intrinsics in a fixed order; max / min / clip as
// (a >= b ? a : b), (a <= b ? a : b), min(max(a, lo), hi), which keep the first operand on ties.
#include "runtime/error.hpp"
#include "field/common.cuh"
#include "method/method.hpp"
#include "method/mode.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace eda {
namespace {

constexpr int kThreads = 256;
constexpr int64_t kMaxGridY = 65535;

template <class T>
__device__ __forceinline__ T div_t(T a, T b) {
  if constexpr (std::is_same_v<T, float>) return __fdiv_rn(a, b); else return __ddiv_rn(a, b);
}
template <class T>
__device__ __forceinline__ T np_max(T a, T b) { return a >= b ? a : b; }
template <class T>
__device__ __forceinline__ T np_min(T a, T b) { return a <= b ? a : b; }
template <class T>
__device__ __forceinline__ T clip(T v, T lo, T hi) { return np_min(np_max(v, lo), hi); }

// The non-pad padded index v of the element's field: true when the sample is a domain sample.
__device__ __forceinline__ bool in_domain(const Geom& g, int64_t v, int32_t nl) { return v % g.pitch < nl; }

dim3 grid2d(const Geom& g) {
  const int64_t x = std::clamp<int64_t>((g.field + kThreads * 4 - 1) / (kThreads * 4), 1, 4096);
  const int64_t y = std::clamp<int64_t>(g.batch, 1, kMaxGridY);
  return dim3(static_cast<unsigned>(x), static_cast<unsigned>(y), 1);
}

#define EDA_FOR_VOXELS(g, on)                                                                     \
  const int32_t nl_ = (g).n[(g).dim - 1];                                                          \
  for (int64_t b = blockIdx.y; b < (g).batch; b += gridDim.y)                                      \
    if (!(on) || (on)[b])                                                                          \
      for (int64_t v = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; v < (g).field; \
           v += static_cast<int64_t>(gridDim.x) * blockDim.x)                                      \
        if (in_domain((g), v, nl_))

// ---------------------------------------------------------------------------------------------
template <class T>
__global__ void __launch_bounds__(kThreads) eda_sift_range(Geom g, const T* u, const T* l, T* R,
                                                           const int32_t* on) {
  EDA_FOR_VOXELS(g, on) {
    const int64_t i = b * g.field + v;
    R[i] = np_max(sub_rn(u[i], l[i]), T(0));
  }
}

template <class T>
__global__ void __launch_bounds__(kThreads) eda_sift_ramp(Geom g, MethodCtl c, const ElemRecord* le, const T* R,
                                                          const T* U, T* gw, const int32_t* on) {
  const T t0 = static_cast<T>(c.t0);
  const double dd = __dsub_rn(c.t1, c.t0);
  const T den = static_cast<T>(dd >= 1e-12 ? dd : 1e-12);
  EDA_FOR_VOXELS(g, on) {
    const int64_t i = b * g.field + v;
    const double tol = le[b].tol;
    const T Uv = U[i];
    T rho = T(0);
    if (static_cast<double>(Uv) > tol) rho = clip(div_t(R[i], Uv), T(0), T(1));
    gw[i] = clip(div_t(sub_rn(rho, t0), den), T(0), T(1));
  }
}

template <class T>
__global__ void __launch_bounds__(kThreads) eda_sift_clip01(Geom g, T* a, const int32_t* on) {
  EDA_FOR_VOXELS(g, on) {
    const int64_t i = b * g.field + v;
    a[i] = clip(a[i], T(0), T(1));
  }
}

template <class T>
__global__ void __launch_bounds__(kThreads) eda_sift_half_diff(Geom g, const T* u, const T* l, T* out,
                                                               const int32_t* on) {
  EDA_FOR_VOXELS(g, on) {
    const int64_t i = b * g.field + v;
    out[i] = mul_rn(T(0.5), sub_rn(u[i], l[i]));
  }
}

template <class T>
__global__ void __launch_bounds__(kThreads) eda_sift_floor(Geom g, MethodCtl c, const ElemRecord* le,
                                                           const double* e_med, T* e, const int32_t* on) {
  EDA_FOR_VOXELS(g, on) {
    const int64_t i = b * g.field + v;
    // max(anorm_floor median(e), tol, tiny) in double, then max(e, thr) in T
    double thr = __dmul_rn(c.anorm_floor, e_med[b]);
    const double tol = le[b].tol;
    thr = thr >= tol ? thr : tol;
    thr = thr >= c.tiny ? thr : c.tiny;
    const T ev = e[i];
    e[i] = static_cast<double>(ev) >= thr ? ev : static_cast<T>(thr);
  }
}

template <class T>
__global__ void __launch_bounds__(kThreads) eda_sift_ratio(Geom g, const T* h, const T* e, T* out,
                                                           const int32_t* on) {
  EDA_FOR_VOXELS(g, on) {
    const int64_t i = b * g.field + v;
    out[i] = div_t(h[i], e[i]);
  }
}

template <class T>
__global__ void __launch_bounds__(kThreads) eda_sift_kappa_ramp(Geom g, MethodCtl c, const ElemRecord* le,
                                                                const T* emax, const T* emin, T* out,
                                                                const int32_t* on) {
  const T k0 = static_cast<T>(c.ramp_k0);
  const double dd = __dsub_rn(c.ramp_k1, c.ramp_k0);
  const T kden = static_cast<T>(dd >= 1e-12 ? dd : 1e-12);
  EDA_FOR_VOXELS(g, on) {
    const int64_t i = b * g.field + v;
    const double tol = le[b].tol;
    const T fl = static_cast<T>(tol >= c.tiny ? tol : c.tiny);
    const T hi = emax[i], lo = emin[i];
    const T den = np_max(add_rn(hi, lo), fl);
    const T kap = div_t(sub_rn(hi, lo), den);
    out[i] = clip(div_t(sub_rn(kap, k0), kden), T(0), T(1));
  }
}

template <class T>
__global__ void __launch_bounds__(kThreads) eda_sift_blend(Geom g, const T* gk, const T* e, const T* Mn,
                                                           const T* u, const T* l, const T* plain, T* mh,
                                                           const int32_t* on) {
  EDA_FOR_VOXELS(g, on) {
    const int64_t i = b * g.field + v;
    const T w = gk[i];
    const T mA = clip(mul_rn(e[i], Mn[i]), l[i], u[i]);
    const T p = plain[i];
    mh[i] = add_rn(mul_rn(w, mA), mul_rn(sub_rn(T(1), w), p));
  }
}

__device__ __forceinline__ uint64_t warp_max_u64(uint64_t v) {
  for (int o = 16; o > 0; o >>= 1) {
    const uint64_t u = __shfl_xor_sync(0xffffffffu, v, o);
    v = u > v ? u : v;
  }
  return v;
}

template <class T>
__global__ void __launch_bounds__(kThreads) eda_sift_step(Geom g, int32_t guard_on, ElemRecord* le, int32_t j,
                                                          const T* r, const T* h, const T* mh, const T* a, T* hn,
                                                          T* mh_out) {
  const int32_t nl = g.n[g.dim - 1];
  for (int64_t b = blockIdx.y; b < g.batch; b += gridDim.y) {
    const ElemRecord& e = le[b];
    if (e.status != kRecordActive || !e.sifting) continue;  // uniform over the CTA
    uint64_t m_res = 0, m_h = 0;
    for (int64_t v = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; v < g.field;
         v += static_cast<int64_t>(gridDim.x) * blockDim.x) {
      if (!in_domain(g, v, nl)) continue;
      const int64_t i = b * g.field + v;
      T m = mh[i];
      if (j >= 1 && a) m = mul_rn(a[i], m);
      const T x = sub_rn(h[i], m);
      hn[i] = x;
      if (mh_out) mh_out[i] = m;
      if (guard_on) {
        const uint64_t br = abs_bits(sub_rn(r[i], x));
        const uint64_t bh = abs_bits(x);
        m_res = br > m_res ? br : m_res;
        m_h = bh > m_h ? bh : m_h;
      }
    }
    if (guard_on) {
      m_res = warp_max_u64(m_res);
      m_h = warp_max_u64(m_h);
      if ((threadIdx.x & 31) == 0) {
        if (m_res) atomicMax(reinterpret_cast<unsigned long long*>(&le[b].gmax_res_bits),
                             static_cast<unsigned long long>(m_res));
        if (m_h) atomicMax(reinterpret_cast<unsigned long long*>(&le[b].gmax_h_bits),
                           static_cast<unsigned long long>(m_h));
      }
    }
  }
}

__device__ __forceinline__ bool sift_accepted(const ElemRecord& e, int32_t guard_on) {
  if (!guard_on) return true;
  // rejected when max|r - hn| > r_inf or max|hn| > r_inf (a NaN maximum compares false and does
  // not reject; the commit then fails the element on its non-finite max|r|)
  return !(bits_to_double(e.gmax_res_bits) > e.r_inf) && !(bits_to_double(e.gmax_h_bits) > e.r_inf);
}

template <class T>
__global__ void __launch_bounds__(kThreads) eda_sift_accept_field(Geom g, int32_t guard_on, const ElemRecord* le,
                                                                  const T* hn, T* h) {
  const int32_t nl = g.n[g.dim - 1];
  for (int64_t b = blockIdx.y; b < g.batch; b += gridDim.y) {
    const ElemRecord& e = le[b];
    if (e.status != kRecordActive || !e.sifting || !sift_accepted(e, guard_on)) continue;
    for (int64_t v = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; v < g.field;
         v += static_cast<int64_t>(gridDim.x) * blockDim.x) {
      if (!in_domain(g, v, nl)) continue;
      const int64_t i = b * g.field + v;
      h[i] = hn[i];
    }
  }
}

__global__ void __launch_bounds__(kThreads) eda_sift_accept_state(int64_t B, int32_t guard_on, ElemRecord* le) {
  const int64_t b = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (b >= B) return;
  ElemRecord& e = le[b];
  if (e.status != kRecordActive || !e.sifting) return;
  if (sift_accepted(e, guard_on)) {
    e.n_sifts += 1;
  } else {
    e.sifting = 0;
  }
  e.gmax_res_bits = 0;
  e.gmax_h_bits = 0;
}

#undef EDA_FOR_VOXELS

// ---------------------------------------------------------------------------------------------
unsigned grid1d(int64_t n) {
  return static_cast<unsigned>(std::clamp<int64_t>((n + kThreads - 1) / kThreads, 1, int64_t{1} << 20));
}

}  // namespace

// ---------------------------------------------------------------------------------------------
template <class T>
void enqueue_mode_range(cudaStream_t s, const Geom& g, const T* u, const T* l, T* R, const int32_t* on) {
  eda_sift_range<T><<<grid2d(g), kThreads, 0, s>>>(g, u, l, R, on);
  EDA_CHECK_LAUNCH("eda_sift_range");
}
template <class T>
void enqueue_mode_ramp(cudaStream_t s, const Geom& g, const MethodCtl& c, const ElemRecord* le, const T* R,
                       const T* U, T* gw, const int32_t* on) {
  if (gw == R || gw == U) throw std::invalid_argument("eda: ramp output must not alias R or U");
  eda_sift_ramp<T><<<grid2d(g), kThreads, 0, s>>>(g, c, le, R, U, gw, on);
  EDA_CHECK_LAUNCH("eda_sift_ramp");
}
template <class T>
void enqueue_mode_clip01(cudaStream_t s, const Geom& g, T* a, const int32_t* on) {
  eda_sift_clip01<T><<<grid2d(g), kThreads, 0, s>>>(g, a, on);
  EDA_CHECK_LAUNCH("eda_sift_clip01");
}
template <class T>
void enqueue_mode_half_diff(cudaStream_t s, const Geom& g, const T* u, const T* l, T* out, const int32_t* on) {
  eda_sift_half_diff<T><<<grid2d(g), kThreads, 0, s>>>(g, u, l, out, on);
  EDA_CHECK_LAUNCH("eda_sift_half_diff");
}
template <class T>
void enqueue_mode_floor(cudaStream_t s, const Geom& g, const MethodCtl& c, const ElemRecord* le,
                        const double* e_med, T* e, const int32_t* on) {
  eda_sift_floor<T><<<grid2d(g), kThreads, 0, s>>>(g, c, le, e_med, e, on);
  EDA_CHECK_LAUNCH("eda_sift_floor");
}
template <class T>
void enqueue_mode_ratio(cudaStream_t s, const Geom& g, const T* h, const T* e, T* out, const int32_t* on) {
  eda_sift_ratio<T><<<grid2d(g), kThreads, 0, s>>>(g, h, e, out, on);
  EDA_CHECK_LAUNCH("eda_sift_ratio");
}
template <class T>
void enqueue_mode_kappa_ramp(cudaStream_t s, const Geom& g, const MethodCtl& c, const ElemRecord* le,
                             const T* emax, const T* emin, T* out, const int32_t* on) {
  eda_sift_kappa_ramp<T><<<grid2d(g), kThreads, 0, s>>>(g, c, le, emax, emin, out, on);
  EDA_CHECK_LAUNCH("eda_sift_kappa_ramp");
}
template <class T>
void enqueue_mode_blend(cudaStream_t s, const Geom& g, const T* gk, const T* e, const T* Mn, const T* u,
                        const T* l, const T* plain, T* mh, const int32_t* on) {
  eda_sift_blend<T><<<grid2d(g), kThreads, 0, s>>>(g, gk, e, Mn, u, l, plain, mh, on);
  EDA_CHECK_LAUNCH("eda_sift_blend");
}
template <class T>
void enqueue_mode_sift(cudaStream_t s, const Geom& g, ElemRecord* le, int32_t j, const T* r, const T* h,
                       const T* mh, const T* a, T* hn, T* mh_out) {
  if (j < 0) throw std::invalid_argument("eda: sift index must be >= 0");
  if (j >= 1 && hn == h) throw std::invalid_argument("eda: sift j >= 1 needs hn distinct from h");
  const int32_t guard_on = j >= 1 ? 1 : 0;
  eda_sift_step<T><<<grid2d(g), kThreads, 0, s>>>(g, guard_on, le, j, r, h, mh, a, hn, mh_out);
  EDA_CHECK_LAUNCH("eda_sift_step");
}
template <class T>
void enqueue_mode_accept(cudaStream_t s, const Geom& g, ElemRecord* le, int32_t j, const T* hn, T* h) {
  const int32_t guard_on = j >= 1 ? 1 : 0;
  if (hn != h) {
    eda_sift_accept_field<T><<<grid2d(g), kThreads, 0, s>>>(g, guard_on, le, hn, h);
    EDA_CHECK_LAUNCH("eda_sift_accept_field");
  }
  eda_sift_accept_state<<<grid1d(g.batch), kThreads, 0, s>>>(g.batch, guard_on, le);
  EDA_CHECK_LAUNCH("eda_sift_accept_state");
}

#define EDA_SIFT_INST(T)                                                                                       \
  template void enqueue_mode_range<T>(cudaStream_t, const Geom&, const T*, const T*, T*, const int32_t*);   \
  template void enqueue_mode_ramp<T>(cudaStream_t, const Geom&, const MethodCtl&, const ElemRecord*, const T*,  \
                                          const T*, T*, const int32_t*);                                           \
  template void enqueue_mode_clip01<T>(cudaStream_t, const Geom&, T*, const int32_t*);                      \
  template void enqueue_mode_half_diff<T>(cudaStream_t, const Geom&, const T*, const T*, T*,                \
                                               const int32_t*);                                                    \
  template void enqueue_mode_floor<T>(cudaStream_t, const Geom&, const MethodCtl&, const ElemRecord*,           \
                                           const double*, T*, const int32_t*);                                     \
  template void enqueue_mode_ratio<T>(cudaStream_t, const Geom&, const T*, const T*, T*, const int32_t*);   \
  template void enqueue_mode_kappa_ramp<T>(cudaStream_t, const Geom&, const MethodCtl&, const ElemRecord*,      \
                                                const T*, const T*, T*, const int32_t*);                           \
  template void enqueue_mode_blend<T>(cudaStream_t, const Geom&, const T*, const T*, const T*, const T*,    \
                                           const T*, const T*, T*, const int32_t*);                                \
  template void enqueue_mode_sift<T>(cudaStream_t, const Geom&, ElemRecord*, int32_t, const T*, const T*,  \
                                          const T*, const T*, T*, T*);                                             \
  template void enqueue_mode_accept<T>(cudaStream_t, const Geom&, ElemRecord*, int32_t, const T*, T*);
EDA_SIFT_INST(float)
EDA_SIFT_INST(double)
#undef EDA_SIFT_INST

}  // namespace eda
