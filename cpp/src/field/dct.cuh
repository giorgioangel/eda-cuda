// SPDX-License-Identifier: Apache-2.0
// Device helpers of the N-D Makhoul DCT (dct.hpp): the per-axis permutation and the orbit
// twiddles. Derivation (per axis, k and n - k, U the DFT of the permuted field, phi = e^{-i pi k / 2n},
// c_k the orthonormal scale):
//   forward  X_k = c_k / 2 (phi U_k + conj(phi) U_{-k}),  X_{n-k} = c_{n-k} / 2 i (phi U_k - conj(phi) U_{-k})
//   inverse  U_k = conj(phi) (X_k - i X_{n-k}),           U_{-k} = phi (X_k + i X_{n-k})   (X_n := 0)
// The N-D maps are the tensor products over the axes; the forward reads the half k_{D-1} <= n/2 of
// the spectrum and takes U(-q) = conj(U(q)) for the other half. The twiddle products are plain
// operators, which nvcc may contract into fused multiply-adds: the bits are fixed for a given
// compiler and architecture, and nothing compares them with a host computation.
#pragma once

#include "field/dct.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace eda {

template <class T>
using cplx_t = std::conditional_t<std::is_same_v<T, float>, float2, double2>;

// u[m] = v[dct_src(m)]; v[x] = u[dct_dst(x)].
__host__ __device__ __forceinline__ int32_t dct_src(int32_t m, int32_t n) {
  return m < (n + 1) / 2 ? 2 * m : 2 * (n - 1 - m) + 1;
}
__host__ __device__ __forceinline__ int32_t dct_dst(int32_t x, int32_t n) {
  return (x & 1) ? n - 1 - (x >> 1) : (x >> 1);
}

template <class T>
struct DctTw {
  const T* cs;
  const T* sn;
  const T* sc;
  const T* isc;  // 1 / sc
};

// The padded row of the D-1 leading indices (j_0..j_{D-2}).
template <int D>
__device__ __forceinline__ int64_t dct_row(const DctShape& d, const int32_t* j) {
  int64_t row = 0;
#pragma unroll
  for (int a = 0; a < D - 1; ++a) row = row * d.n[a] + j[a];
  return row;
}

// The leading indices of padded row q.
template <int D>
__device__ __forceinline__ void dct_split_row(const DctShape& d, int64_t q, int32_t* j) {
#pragma unroll
  for (int a = D - 2; a >= 0; --a) {
    j[a] = static_cast<int32_t>(q % d.n[a]);
    q /= d.n[a];
  }
}

// The half index of orbit thread `o` of one element: k_a in [0, n_a/2] (a < D-1), k_l in [0, nc).
template <int D>
__device__ __forceinline__ void dct_orbit(const DctShape& d, int64_t o, int32_t* k) {
  k[D - 1] = static_cast<int32_t>(o % d.nc);
  o /= d.nc;
#pragma unroll
  for (int a = D - 2; a >= 0; --a) {
    const int32_t h = d.n[a] / 2 + 1;
    k[a] = static_cast<int32_t>(o % h);
    o /= h;
  }
}

// Offset in the dense complex spectrum of one element for the signs `s` (bit a: -k_a).
template <int D>
__device__ __forceinline__ int64_t dct_coff(const DctShape& d, const int32_t* k, int s) {
  int64_t off = 0;
#pragma unroll
  for (int a = 0; a < D; ++a) {
    const int32_t q = ((s >> a) & 1) ? (k[a] ? d.n[a] - k[a] : 0) : k[a];
    off = off * d.n[a] + q;
  }
  return off;
}

// Whether the orbit member `u` (bit a: index n_a - k_a) is a distinct, valid index.
template <int D>
__device__ __forceinline__ bool dct_member(const DctShape& d, const int32_t* k, int u) {
#pragma unroll
  for (int a = 0; a < D; ++a) {
    if ((u >> a) & 1) {
      if (k[a] == 0 || 2 * k[a] == d.n[a]) return false;
    }
  }
  return true;
}

// Real padded offset of orbit member `u` of one element (members with k_a = 0 and bit a set are
// invalid and must not be passed).
template <int D>
__device__ __forceinline__ int64_t dct_roff(const DctShape& d, const int32_t* k, int u) {
  int32_t j[kMaxDim];
#pragma unroll
  for (int a = 0; a < D; ++a) j[a] = ((u >> a) & 1) ? d.n[a] - k[a] : k[a];
  return dct_row<D>(d, j) * d.pitch + j[D - 1];
}

// Forward twiddle of one orbit: V (complex spectrum of the element, of which it reads the half
// k_{D-1} <= n/2) -> X (real padded).
template <class T, int D>
__device__ __forceinline__ void dct_post_orbit(const DctShape& d, const DctTw<T>& tw, const int32_t* k,
                                               const cplx_t<T>* __restrict__ V, T* __restrict__ X) {
  using C = cplx_t<T>;
  constexpr int NB = 1 << D;
  constexpr int NW = 1 << (D - 1);
  C W[NW];
#pragma unroll
  for (int s = 0; s < NW; ++s) W[s] = V[dct_coff<D>(d, k, s)];
  C U[NB];
#pragma unroll
  for (int u = 0; u < NB; ++u) {
    const int s = u & (NW - 1);
    if ((u >> (D - 1)) & 1) {
      const C w = W[(~s) & (NW - 1)];
      U[u] = C{w.x, -w.y};
    } else {
      U[u] = W[s];
    }
  }
#pragma unroll
  for (int a = 0; a < D; ++a) {
    const int32_t ka = k[a];
    const int32_t off = d.tw_off[a];
    const T c = tw.cs[off + ka], sn = tw.sn[off + ka];
    const T c0 = T(0.5) * tw.sc[off + ka], c1 = T(0.5) * tw.sc[off + d.n[a] - ka];
#pragma unroll
    for (int u = 0; u < NB; ++u) {
      if ((u >> a) & 1) continue;
      const C A = U[u], B = U[u | (1 << a)];
      const T pax = c * A.x + sn * A.y, pay = c * A.y - sn * A.x;  // phi A
      const T qbx = c * B.x - sn * B.y, qby = c * B.y + sn * B.x;  // conj(phi) B
      U[u] = C{c0 * (pax + qbx), c0 * (pay + qby)};
      U[u | (1 << a)] = C{-c1 * (pay - qby), c1 * (pax - qbx)};
    }
  }
#pragma unroll
  for (int u = 0; u < NB; ++u) {
    if (dct_member<D>(d, k, u)) X[dct_roff<D>(d, k, u)] = U[u].x;
  }
}

}  // namespace eda
