// SPDX-License-Identifier: Apache-2.0
// Device helpers shared by the N-D kernels: canonical arithmetic, reflection, the shared-memory
// swizzle of staged lines and the canonical reductions (reduce.hpp).
//
// Arithmetic. On the device every floating-point operation that feeds a result is an IEEE
// round-to-nearest intrinsic (__fadd_rn, __dmul_rn, ...), which nvcc never contracts or reorders.
// Host twins use the plain operators in translation units compiled with -ffp-contract=off, so both
// sides produce the same bits. max_c / min_c are commutative bitwise: for +0 and -0 the max is +0
// and the min is -0 whatever the operand order (needed for bitwise flip equivariance).
#pragma once

#include "field/geom.hpp"
#include "field/reduce.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace eda {

template <class T>
__host__ __device__ __forceinline__ T add_rn(T a, T b) {
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same_v<T, float>) return __fadd_rn(a, b); else return __dadd_rn(a, b);
#else
  return a + b;
#endif
}
template <class T>
__host__ __device__ __forceinline__ T sub_rn(T a, T b) {
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same_v<T, float>) return __fsub_rn(a, b); else return __dsub_rn(a, b);
#else
  return a - b;
#endif
}
template <class T>
__host__ __device__ __forceinline__ T mul_rn(T a, T b) {
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same_v<T, float>) return __fmul_rn(a, b); else return __dmul_rn(a, b);
#else
  return a * b;
#endif
}
__host__ __device__ __forceinline__ double div_rn(double a, double b) {
#if defined(__CUDA_ARCH__)
  return __ddiv_rn(a, b);
#else
  return a / b;
#endif
}
__host__ __device__ __forceinline__ double sqrt_rn(double a) {
#if defined(__CUDA_ARCH__)
  return __dsqrt_rn(a);
#else
  return std::sqrt(a);
#endif
}

template <class T>
__host__ __device__ __forceinline__ bool sign_bit(T v) {
#if defined(__CUDA_ARCH__)
  return signbit(v);
#else
  return std::signbit(v);
#endif
}
// Commutative max / min (bitwise, ±0 included). NaN operands are not ordered (callers never pass
// them: elements with non-finite data do not sift).
template <class T>
__host__ __device__ __forceinline__ T max_c(T a, T b) {
  return (a > b || (a == b && !sign_bit(a))) ? a : b;
}
template <class T>
__host__ __device__ __forceinline__ T min_c(T a, T b) {
  return (a < b || (a == b && sign_bit(a))) ? a : b;
}

// Floor division and modulo of int64 by a positive divisor.
__host__ __device__ __forceinline__ int64_t floor_div(int64_t a, int64_t b) {
  const int64_t q = a / b;
  return (a % b != 0 && a < 0) ? q - 1 : q;
}
__host__ __device__ __forceinline__ int64_t floor_mod(int64_t a, int64_t b) {
  const int64_t m = a % b;
  return m < 0 ? m + b : m;
}

// Half-sample symmetric reflection of e into [0, n), valid for e in [-n, 2n).
__host__ __device__ __forceinline__ int32_t reflect_index(int32_t e, int32_t n) {
  return e < 0 ? -e - 1 : (e >= n ? 2 * n - 1 - e : e);
}

// The bit pattern of the double |v| (its integer order is the order of |v| for non-NaN v).
template <class T>
__host__ __device__ __forceinline__ uint64_t abs_bits(T v) {
  const double d = static_cast<double>(v);
#if defined(__CUDA_ARCH__)
  return static_cast<uint64_t>(__double_as_longlong(d)) & 0x7fffffffffffffffULL;
#else
  uint64_t u;
  std::memcpy(&u, &d, sizeof u);
  return u & 0x7fffffffffffffffULL;
#endif
}
__host__ __device__ __forceinline__ double bits_to_double(uint64_t u) {
#if defined(__CUDA_ARCH__)
  return __longlong_as_double(static_cast<long long>(u));
#else
  double d;
  std::memcpy(&d, &u, sizeof d);
  return d;
#endif
}

// Shared-memory swizzle of a staged line: element i lives at i ^ ((i >> 5) & M), M = 31 (float)
// or 15 (double). Lanes that walk consecutive 32-sample chunks then hit distinct banks.
template <class T>
__host__ __device__ __forceinline__ int32_t swz(int32_t i) {
  constexpr int32_t kMask = sizeof(T) == 4 ? 31 : 15;
  return i ^ ((i >> 5) & kMask);
}

// fl_T(tau * amax), returned as a double (exact in T).
template <class T>
__host__ __device__ __forceinline__ double tie_tolerance(double tau, double amax) {
  return static_cast<double>(mul_rn(static_cast<T>(tau), static_cast<T>(amax)));
}

// ---------------------------------------------------------------------------------------------
// Canonical row partial of one row, by one warp. `at(e)` returns sample e < n as T.
// Energies: Σ v² in double over 32-sample chunks in sequential order, then the pairwise tree over
// the chunk sums in chunk order, padded with +0 to a power of two (s_i += s_{i+δ}, δ = 1, 2, 4, ...).
// The tree is fixed by n alone, so the sum does not depend on the launch, and its rounding error
// grows with log n rather than n. amax: the integer max of abs_bits (exact). Lane 0 holds the
// result.
inline constexpr int kRowSlots = 8;  // up to 8 * 32 chunks = 8192 samples per row

#if defined(__CUDACC__)
template <class T, class At>
__device__ __forceinline__ RowPartial warp_row_partial(At at, int32_t n) {
  const int lane = threadIdx.x & 31;
  const int32_t nchunks = (n + 31) >> 5;
  double s[kRowSlots];
  uint64_t amax = 0;
#pragma unroll
  for (int k = 0; k < kRowSlots; ++k) {
    const int32_t c = k * 32 + lane;
    double acc = 0.0;
    if (c < nchunks) {
      const int32_t e0 = c * 32;
      const int32_t e1 = min(e0 + 32, n);
      for (int32_t e = e0; e < e1; ++e) {
        const T v = at(e);
        const double d = static_cast<double>(v);
        acc = __dadd_rn(acc, __dmul_rn(d, d));
        const uint64_t ab = abs_bits(v);
        amax = ab > amax ? ab : amax;
      }
    }
    s[k] = acc;
  }
  // Levels δ = 1..16: inside each slot of 32 chunks (lanes).
#pragma unroll
  for (int k = 0; k < kRowSlots; ++k) {
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const double o = __shfl_down_sync(0xffffffffu, s[k], d);
      if ((lane & (2 * d - 1)) == 0) s[k] = __dadd_rn(s[k], o);
    }
  }
  // Levels δ = 32, 64, 128: across slots (lane 0).
#pragma unroll
  for (int d = 1; d < kRowSlots; d <<= 1) {
#pragma unroll
    for (int k = 0; k + d < kRowSlots; k += 2 * d) s[k] = __dadd_rn(s[k], s[k + d]);
  }
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) {
    const uint64_t o = __shfl_down_sync(0xffffffffu, amax, d);
    amax = o > amax ? o : amax;
  }
  return RowPartial{s[0], amax};
}

// Per-element total over `nrows` row partials by one CTA (blockDim.x = kReduceThreads): leaves of
// 1024 rows reduced by the pairwise tree, the leaf results combined by thread 0 as the same tree
// (a binary-counter stack, collapsed right to left at the end: the zero padding). Thread 0 gets
// the result. `get(i)` returns row partial i < nrows.
inline constexpr int kReduceThreads = 256;
inline constexpr int kReduceLeaf = 1024;

struct ElemTotal {
  double sum_sq;
  uint64_t amax_bits;
};

template <class Get>
__device__ ElemTotal block_reduce_rows(Get get, int64_t nrows, double* leaf /* smem [kReduceLeaf] */,
                                       uint64_t* amax_smem /* smem [kReduceThreads / 32] */) {
  const int tid = threadIdx.x;
  double stack[40];
  int level[40];
  int top = 0;
  uint64_t amax = 0;
  for (int64_t start = 0; start < nrows; start += kReduceLeaf) {
#pragma unroll
    for (int k = 0; k < kReduceLeaf / kReduceThreads; ++k) {
      const int i = tid + k * kReduceThreads;
      double v = 0.0;
      if (start + i < nrows) {
        const RowPartial p = get(start + i);
        v = p.sum_sq;
        amax = p.amax_bits > amax ? p.amax_bits : amax;
      }
      leaf[i] = v;
    }
    __syncthreads();
    for (int d = 1; d < kReduceLeaf; d <<= 1) {
      for (int i = tid * 2 * d; i < kReduceLeaf; i += kReduceThreads * 2 * d) leaf[i] = __dadd_rn(leaf[i], leaf[i + d]);
      __syncthreads();
    }
    if (tid == 0) {
      double v = leaf[0];
      int l = 0;
      while (top > 0 && level[top - 1] == l) {
        v = __dadd_rn(stack[top - 1], v);
        --top;
        ++l;
      }
      stack[top] = v;
      level[top] = l;
      ++top;
    }
    __syncthreads();
  }
  for (int d = 16; d > 0; d >>= 1) {
    const uint64_t o = __shfl_down_sync(0xffffffffu, amax, d);
    amax = o > amax ? o : amax;
  }
  if ((tid & 31) == 0) amax_smem[tid >> 5] = amax;
  __syncthreads();
  ElemTotal out{0.0, 0};
  if (tid == 0) {
    double v = 0.0;
    if (top > 0) {
      v = stack[top - 1];
      for (int i = top - 2; i >= 0; --i) v = __dadd_rn(stack[i], v);
    }
    uint64_t m = 0;
    for (int w = 0; w < kReduceThreads / 32; ++w) m = amax_smem[w] > m ? amax_smem[w] : m;
    out = ElemTotal{v, m};
  }
  __syncthreads();
  return out;
}
#endif  // __CUDACC__

}  // namespace eda
