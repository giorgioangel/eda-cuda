// SPDX-License-Identifier: Apache-2.0
// The orientation's eigen pass, the element build (d, e, o and the per-element extremes of d), the
// ladder (host), the hat weights with the rung flags, and the rung windows of the operators
// (method/element.hpp). The operators are in element_op.cuh.
//
// Determinism: no float atomics. The extremes of d are integer atomicMin / atomicMax on the bits
// of positive doubles, the rung flags integer atomicOr: every result is independent of the order
// of the atomics and of the grid.
#include "runtime/error.hpp"
#include "field/common.cuh"
#include "method/method.hpp"
#include "method/element.cuh"
#include "method/element.hpp"

#include <cmath>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace eda {
namespace {

using element_detail::element_de;
using element_detail::kThreads;
using element_detail::voxel_coords;

// Sweeps of the cyclic Jacobi method on the 3x3 monogenic energy tensor T (fixed: no data-dependent loop).
// Measured against a long-double reference (cpp/tests/test_element): 2 sweeps leave 1.4e-3 in c,
// 3 sweeps 1.4e-12, 4 reach the double-precision limit (4e-16); 6 keep two sweeps of margin.
constexpr int kJacobiSweeps = 6;

// Blocks per element of the voxel kernels (each block strides over its element).
int blocks_per_element(const Geom& g) {
  const int64_t need = (g.field + kThreads - 1) / kThreads;
  return static_cast<int>(need < 1024 ? (need < 1 ? 1 : need) : 1024);
}

dim3 voxel_grid(const Geom& g, int bpe) {
  return dim3(static_cast<unsigned>(static_cast<int64_t>(bpe) * g.batch));
}

__device__ __forceinline__ bool runs(const int32_t* on, int64_t b) { return on == nullptr || on[b] != 0; }

// ---------------------------------------------------------------------------------------------
// Orientation: the per-voxel eigen-decomposition of the energy tensor.

// One Jacobi rotation of the symmetric 3x3 (or 2x2: r < 0) matrix A annihilating A[p][q], with the
// eigenvector matrix V (rows k, columns the eigenvectors). Numerically stable form (Rutishauser):
// theta = (A_qq - A_pp) / (2 A_pq), t = sign(theta) / (|theta| + sqrt(theta^2 + 1)). Every
// operation is sign-equivariant (copysign keeps the sign of a zero theta), so that negating
// the rows and columns of an axis negates the rows of V bitwise: a mirrored field gets the
// mirrored orientation exactly.
__device__ __forceinline__ void jacobi_rotate(double (&A)[3][3], double (&V)[3][3], int p, int q, int r, int dim) {
  const double apq = A[p][q];
  if (apq == 0.0) return;
  const double theta = div_rn(sub_rn(A[q][q], A[p][p]), mul_rn(2.0, apq));
  const double at = fabs(theta);
  const double t = div_rn(copysign(1.0, theta), add_rn(at, sqrt_rn(add_rn(mul_rn(theta, theta), 1.0))));
  const double c = div_rn(1.0, sqrt_rn(add_rn(mul_rn(t, t), 1.0)));
  const double s = mul_rn(t, c);
  const double tau = div_rn(s, add_rn(1.0, c));
  const double tapq = mul_rn(t, apq);
  A[p][p] = sub_rn(A[p][p], tapq);
  A[q][q] = add_rn(A[q][q], tapq);
  A[p][q] = A[q][p] = 0.0;
  if (r >= 0) {
    const double gg = A[r][p], hh = A[r][q];
    const double np_ = sub_rn(gg, mul_rn(s, add_rn(hh, mul_rn(gg, tau))));
    const double nq_ = add_rn(hh, mul_rn(s, sub_rn(gg, mul_rn(hh, tau))));
    A[r][p] = A[p][r] = np_;
    A[r][q] = A[q][r] = nq_;
  }
  for (int k = 0; k < dim; ++k) {
    const double gg = V[k][p], hh = V[k][q];
    V[k][p] = sub_rn(gg, mul_rn(s, add_rn(hh, mul_rn(gg, tau))));
    V[k][q] = add_rn(hh, mul_rn(s, sub_rn(gg, mul_rn(hh, tau))));
  }
}

template <class T, int D>
__global__ void __launch_bounds__(kThreads) eda_element_eigen(Geom g, const T* __restrict__ work,
                                                              const int32_t* __restrict__ W,
                                                              const int32_t* __restrict__ on, FieldVec<T> n,
                                                              T* __restrict__ coh, ElemRecord* __restrict__ le,
                                                              int bpe) {
  const int64_t b = blockIdx.x / bpe;
  if (b >= g.batch || (W != nullptr && W[b] <= 0) || !runs(on, b)) return;
  for (int64_t p = static_cast<int64_t>(blockIdx.x % bpe) * kThreads + threadIdx.x; p < g.field;
       p += static_cast<int64_t>(bpe) * kThreads) {
    int32_t x[kMaxDim];
    if (!voxel_coords(g, p, x)) continue;
    double A[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    double V[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    int k = 0;
    bool finite = true;
#pragma unroll
    for (int a = 0; a < D; ++a) {
#pragma unroll
      for (int c = a; c < D; ++c, ++k) {
        const double v = static_cast<double>(__ldg(work + (k * g.batch + b) * g.field + p));
        finite = finite && isfinite(v);
        A[a][c] = A[c][a] = v;
      }
    }
    // a non-finite J = T = R1 R1^T + R2 R2^T (orientation.cu: an overflow of the Riesz products
    // or of their sums, or an inf / NaN residual) would end as c = 0 or NaN: the element is
    // flagged for the driver's guard instead
    if (!finite && le != nullptr) le[b].nonfinite = 1;
    if constexpr (D == 2) {
      jacobi_rotate(A, V, 0, 1, -1, 2);
    } else {
      // scale by a power of two (exact): theta^2 and the products stay in range
      double amax = 0.0;
      for (int a = 0; a < 3; ++a) {
        for (int c = a; c < 3; ++c) amax = fmax(amax, fabs(A[a][c]));
      }
      if (amax > 0.0 && amax < INFINITY) {
        const double sc = scalbn(1.0, -ilogb(amax));
        for (int a = 0; a < 3; ++a) {
          for (int c = 0; c < 3; ++c) A[a][c] = mul_rn(A[a][c], sc);
        }
      }
#pragma unroll 1
      for (int sweep = 0; sweep < kJacobiSweeps; ++sweep) {
        jacobi_rotate(A, V, 0, 1, 2, 3);
        jacobi_rotate(A, V, 0, 2, 1, 3);
        jacobi_rotate(A, V, 1, 2, 0, 3);
      }
    }
    // the largest and second largest eigenvalues (ties: the lower index)
    double mu[D];
#pragma unroll
    for (int a = 0; a < D; ++a) mu[a] = A[a][a];
    int top = 0;
#pragma unroll
    for (int a = 1; a < D; ++a) top = mu[a] > mu[top] ? a : top;
    // second: the largest of the others (ties: the lower index)
    double mu2 = -INFINITY;
#pragma unroll
    for (int a = 0; a < D; ++a) mu2 = (a != top && mu[a] > mu2) ? mu[a] : mu2;
    double mu1 = mu[0];
    double nv[D];
#pragma unroll
    for (int a = 0; a < D; ++a) {
      if (a == top) mu1 = mu[a];
    }
#pragma unroll
    for (int a = 0; a < D; ++a) {
      double v = V[a][0];
#pragma unroll
      for (int c = 1; c < D; ++c) v = c == top ? V[a][c] : v;
      nv[a] = v;
    }
    const double ssum = add_rn(mu1, mu2);
    double cv = ssum > 0.0 ? div_rn(sub_rn(mu1, mu2), ssum) : 0.0;
    cv = cv < 0.0 ? 0.0 : (cv > 1.0 ? 1.0 : cv);
    const int64_t at = b * g.field + p;
    coh[at] = static_cast<T>(cv);
#pragma unroll
    for (int a = 0; a < D; ++a) n.p[a][at] = static_cast<T>(nv[a]);
  }
}

// ---------------------------------------------------------------------------------------------
// Element build and weights.

template <class T>
struct ElemArgs {
  T q, span_m1;
};

template <class T>
__device__ __forceinline__ T coh_at(const T* coh, int dim, int64_t at) {
  return (dim == 1 || coh == nullptr) ? T(0) : coh[at];
}

__device__ __forceinline__ uint64_t dbits(double v) { return static_cast<uint64_t>(__double_as_longlong(v)); }

template <class T>
__global__ void __launch_bounds__(kThreads) eda_element_build(Geom g, ElemArgs<T> ea, const T* __restrict__ L,
                                                              FieldVec<const T> n, const T* __restrict__ coh,
                                                              FieldVec<T> o, ElemRecord* __restrict__ le,
                                                              const int32_t* __restrict__ on, int bpe) {
  __shared__ unsigned long long s_min, s_max;
  const int64_t b = blockIdx.x / bpe;
  if (b >= g.batch || !runs(on, b)) return;
  if (threadIdx.x == 0) {
    s_min = 0x7ff0000000000000ULL;  // +inf
    s_max = 0ULL;
  }
  __syncthreads();
  const int D = g.dim;
  unsigned long long tmin = 0x7ff0000000000000ULL, tmax = 0ULL;
  for (int64_t p = static_cast<int64_t>(blockIdx.x % bpe) * kThreads + threadIdx.x; p < g.field;
       p += static_cast<int64_t>(bpe) * kThreads) {
    int32_t x[kMaxDim];
    if (!voxel_coords(g, p, x)) continue;
    const int64_t at = b * g.field + p;
    const element_detail::ElemDE<T> de = element_de<T>(L[at], coh_at(coh, D, at), ea.q, ea.span_m1);
    if (D >= 2) {
      for (int a = 0; a < D; ++a) o.p[a][at] = mul_rn(de.e, n.p[a][at]);
    }
    const unsigned long long bits = dbits(static_cast<double>(de.d));
    tmin = bits < tmin ? bits : tmin;
    tmax = bits > tmax ? bits : tmax;
  }
  atomicMin(&s_min, tmin);
  atomicMax(&s_max, tmax);
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicMin(reinterpret_cast<unsigned long long*>(&le[b].d_min_bits), s_min);
    atomicMax(reinterpret_cast<unsigned long long*>(&le[b].d_max_bits), s_max);
  }
}

// rung_lw[j] = log(rung_w[j]) with the device log, the same function as the voxels' log d: a d
// clipped to a rung then gets a weight of exactly 0 or 1.
__global__ void eda_element_rung_logs(int64_t batch, ElemRecord* __restrict__ le, const int32_t* __restrict__ on) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t b = i / kMaxRungs;
  const int j = static_cast<int>(i % kMaxRungs);
  if (b >= batch || !runs(on, b)) return;
  if (j < le[b].n_rungs) le[b].rung_lw[j] = log(static_cast<double>(le[b].rung_w[j]));
}

template <class T>
__global__ void __launch_bounds__(kThreads) eda_element_weights(Geom g, ElemArgs<T> ea, const T* __restrict__ L,
                                                                const T* __restrict__ coh,
                                                                ElemRecord* __restrict__ le, uint8_t* __restrict__ lo,
                                                                T* __restrict__ t, const int32_t* __restrict__ on,
                                                                int bpe) {
  __shared__ double s_lw[kMaxRungs];
  __shared__ unsigned s_used, s_pts;
  const int64_t b = blockIdx.x / bpe;
  if (b >= g.batch || !runs(on, b)) return;
  const int K = le[b].n_rungs;
  if (threadIdx.x < kMaxRungs) s_lw[threadIdx.x] = threadIdx.x < static_cast<unsigned>(K) ? le[b].rung_lw[threadIdx.x] : 0.0;
  if (threadIdx.x == 0) {
    s_used = 0u;
    s_pts = 0u;
  }
  __syncthreads();
  const int D = g.dim;
  const double w_lo = static_cast<double>(le[b].rung_w[0]);
  const double w_hi = static_cast<double>(le[b].rung_w[K > 0 ? K - 1 : 0]);
  unsigned used = 0u, pts = 0u;
  for (int64_t p = static_cast<int64_t>(blockIdx.x % bpe) * kThreads + threadIdx.x; p < g.field;
       p += static_cast<int64_t>(bpe) * kThreads) {
    int32_t x[kMaxDim];
    if (!voxel_coords(g, p, x)) continue;
    const int64_t at = b * g.field + p;
    const element_detail::ElemDE<T> de = element_de<T>(L[at], coh_at(coh, D, at), ea.q, ea.span_m1);
    int idx = 0;
    T tt = T(0);
    if (K > 1) {
      double dc = static_cast<double>(de.d);
      dc = dc > w_lo ? dc : w_lo;  // clip(d, w_0, w_K) = min(max(d, lo), hi)
      dc = dc < w_hi ? dc : w_hi;
      const double lg = log(dc);
      int cnt = 0;  // the number of rungs with lw <= lg (a right-sided binary search, done linearly)
      for (int j = 0; j < K; ++j) cnt += s_lw[j] <= lg ? 1 : 0;
      idx = cnt - 1;
      idx = idx < 0 ? 0 : (idx > K - 2 ? K - 2 : idx);
      const double tv = div_rn(sub_rn(lg, s_lw[idx]), sub_rn(s_lw[idx + 1], s_lw[idx]));
      tt = static_cast<T>(tv);
    }
    lo[at] = static_cast<uint8_t>(idx);
    t[at] = tt;
    const T wl = sub_rn(T(1), tt);
    unsigned bits = 0u;
    if (wl > T(0)) bits |= 1u << idx;
    if (tt > T(0)) bits |= 1u << (idx + 1);
    used |= bits;
    if (de.e > T(0.25)) pts |= bits;
  }
  // warp OR, then one shared and one global atomic per block (integer: order-independent)
  for (int off = 16; off > 0; off >>= 1) {
    used |= __shfl_xor_sync(0xffffffffu, used, off);
    pts |= __shfl_xor_sync(0xffffffffu, pts, off);
  }
  if ((threadIdx.x & 31) == 0) {
    if (used) atomicOr(&s_used, used);
    if (pts) atomicOr(&s_pts, pts);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (s_used) atomicOr(&le[b].rung_used, s_used);
    if (s_pts) atomicOr(&le[b].rung_pts, s_pts);
  }
}

__global__ void eda_element_rung_win(int32_t batch, const ElemRecord* __restrict__ le, int32_t n_rungs_max,
                                     int32_t* __restrict__ rung_win) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<int64_t>(batch) * n_rungs_max) return;
  const int j = static_cast<int>(i / batch);
  const int64_t b = i - static_cast<int64_t>(j) * batch;
  const ElemRecord& e = le[b];
  const bool use = e.status == kRecordActive && j < e.n_rungs && j < kMaxRungs && ((e.rung_used >> j) & 1u) != 0u;
  rung_win[i] = use ? e.rung_w[j] : 0;
}

template <class T>
ElemArgs<T> elem_args(const MethodCtl& c) {
  ElemArgs<T> ea;
  ea.q = static_cast<T>(c.q);
  ea.span_m1 = static_cast<T>(c.span - 1.0);  // (span - 1.0) in double, then rounded once to T
  return ea;
}

}  // namespace

template <class T>
void enqueue_element_eigen(cudaStream_t s, const Geom& g, const T* work, const int32_t* W, FieldVec<T> n, T* c,
                           const int32_t* on, ElemRecord* le) {
  if (g.dim == 1) return;
  const int bpe = blocks_per_element(g);
  if (g.dim == 2) {
    eda_element_eigen<T, 2><<<voxel_grid(g, bpe), kThreads, 0, s>>>(g, work, W, on, n, c, le, bpe);
  } else {
    eda_element_eigen<T, 3><<<voxel_grid(g, bpe), kThreads, 0, s>>>(g, work, W, on, n, c, le, bpe);
  }
  EDA_CHECK_LAUNCH("eda_element_eigen");
}

template <class T>
void enqueue_element_build(cudaStream_t s, const Geom& g, const MethodCtl& c, const T* L, FieldVec<const T> n,
                           const T* coh, FieldVec<T> o, ElemRecord* le, const int32_t* on) {
  if (L == nullptr || le == nullptr) throw std::invalid_argument("eda: element build: null field");
  if (g.dim >= 2) {
    if (coh == nullptr) throw std::invalid_argument("eda: element build: null coherence");
    for (int a = 0; a < g.dim; ++a) {
      if (n.p[a] == nullptr || o.p[a] == nullptr) throw std::invalid_argument("eda: element build: null vector component");
    }
  }
  const int bpe = blocks_per_element(g);
  eda_element_build<T><<<voxel_grid(g, bpe), kThreads, 0, s>>>(g, elem_args<T>(c), L, n, coh, o, le, on, bpe);
  EDA_CHECK_LAUNCH("eda_element_build");
}

std::vector<int32_t> rung_ladder(double d_min, double d_max, int32_t w, int32_t n_max) {
  if (!(std::isfinite(d_min) && std::isfinite(d_max)) || w < 1 || n_max < 1) {
    throw std::invalid_argument("eda: rung_ladder: d_min, d_max must be finite, w and n_max >= 1");
  }
  // In double with the C library's log, pow, floor and ceil, in this order of operations; the
  // volatile keeps the compiler from folding log(q) at compile time, where it could round
  // differently from the library.
  volatile double qv = kRungRatio;
  const double q = qv;
  const double lq = std::log(q);
  const double wd = static_cast<double>(w);
  const double jl = std::floor(std::log((d_min > 1.0 ? d_min : 1.0) / wd) / lq + 1e-9);
  const double jh = std::ceil(std::log((d_max > 1.0 ? d_max : 1.0) / wd) / lq - 1e-9);
  if (!(jh - jl <= 4096.0)) throw std::length_error("eda: rung_ladder: the ladder spans too many rungs");
  const int64_t cap = 2 * static_cast<int64_t>(n_max) + 1;
  std::set<int64_t> sides;
  for (int64_t j = static_cast<int64_t>(jl); j <= static_cast<int64_t>(jh); ++j) {
    const double v = wd * std::pow(q, static_cast<double>(j));
    // odd(v) >= v - 1: beyond cap + 1 the rung is the cap (also keeps odd_window's int32 in range)
    const int64_t side = v >= static_cast<double>(cap) + 1.0 ? cap : std::min<int64_t>(odd_window(v), cap);
    sides.insert(side);
  }
  if (sides.size() > static_cast<size_t>(kMaxRungs)) {
    throw std::length_error("eda: rung_ladder: the ladder has " + std::to_string(sides.size()) + " rungs (at most " +
                            std::to_string(kMaxRungs) + ")");
  }
  return std::vector<int32_t>(sides.begin(), sides.end());
}

template <class T>
void enqueue_element_weights(cudaStream_t s, const Geom& g, const MethodCtl& c, const T* L, const T* coh,
                             ElemRecord* le, uint8_t* lo, T* t, const int32_t* on) {
  if (L == nullptr || le == nullptr || lo == nullptr || t == nullptr) {
    throw std::invalid_argument("eda: element weights: null field");
  }
  if (g.dim >= 2 && coh == nullptr) throw std::invalid_argument("eda: element weights: null coherence");
  const int64_t nl = g.batch * kMaxRungs;
  eda_element_rung_logs<<<static_cast<unsigned>((nl + 127) / 128), 128, 0, s>>>(g.batch, le, on);
  EDA_CHECK_LAUNCH("eda_element_rung_logs");
  const int bpe = blocks_per_element(g);
  eda_element_weights<T><<<voxel_grid(g, bpe), kThreads, 0, s>>>(g, elem_args<T>(c), L, coh, le, lo, t, on, bpe);
  EDA_CHECK_LAUNCH("eda_element_weights");
}

void enqueue_element_rung_windows(cudaStream_t s, int32_t batch, const ElemRecord* le, int32_t n_rungs_max,
                                  int32_t* rung_win) {
  if (n_rungs_max < 0 || n_rungs_max > kMaxRungs) {
    throw std::invalid_argument("eda: rung windows: n_rungs_max outside 0..kMaxRungs");
  }
  if (batch < 1 || n_rungs_max == 0) return;
  if (le == nullptr || rung_win == nullptr) throw std::invalid_argument("eda: rung windows: null array");
  const int64_t n = static_cast<int64_t>(batch) * n_rungs_max;
  eda_element_rung_win<<<static_cast<unsigned>((n + 127) / 128), 128, 0, s>>>(batch, le, n_rungs_max, rung_win);
  EDA_CHECK_LAUNCH("eda_element_rung_win");
}

#define EDA_ELEMENT_INST(T)                                                                                     \
  template void enqueue_element_eigen<T>(cudaStream_t, const Geom&, const T*, const int32_t*, FieldVec<T>,  \
                                              T*, const int32_t*, ElemRecord*);                                     \
  template void enqueue_element_build<T>(cudaStream_t, const Geom&, const MethodCtl&, const T*,              \
                                              FieldVec<const T>, const T*, FieldVec<T>, ElemRecord*, const int32_t*); \
  template void enqueue_element_weights<T>(cudaStream_t, const Geom&, const MethodCtl&, const T*, const T*,  \
                                                ElemRecord*, uint8_t*, T*, const int32_t*);
EDA_ELEMENT_INST(float)
EDA_ELEMENT_INST(double)
#undef EDA_ELEMENT_INST

}  // namespace eda
