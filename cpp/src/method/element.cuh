// SPDX-License-Identifier: Apache-2.0
// Device helpers shared by the translation units of the element (element.cu: the eigen pass,
// build, weights, rung windows; element_op.cuh: the gather operators).
//
// Arithmetic: every floating-point operation that feeds a result is a round-to-nearest intrinsic
// (add_rn, mul_rn, ... of field/common.cuh) in a fixed order; nvcc never contracts them, so the
// host twins of the tests reproduce the bits.
#pragma once

#include "field/common.cuh"
#include "field/geom.hpp"
#include "method/method.hpp"
#include "method/element.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace eda::element_detail {

inline constexpr int kThreads = 256;
// Segment points per side of the operators (m): 2 m + 1 = 5 points along each segment.
inline constexpr int kM = 2;

template <class T>
__host__ __device__ __forceinline__ T div_t(T a, T b) {
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same_v<T, float>) return __fdiv_rn(a, b); else return __ddiv_rn(a, b);
#else
  return a / b;
#endif
}

// The voxel coordinates of padded offset p inside one element (x[a] for a < D); false for a pad.
__host__ __device__ __forceinline__ bool voxel_coords(const Geom& g, int64_t p, int32_t* x) {
  const int64_t row = p / g.pitch;
  const int64_t col = p - row * g.pitch;
  if (col >= g.n[g.dim - 1]) return false;
  if (g.dim == 1) {
    x[0] = static_cast<int32_t>(col);
  } else if (g.dim == 2) {
    x[0] = static_cast<int32_t>(row);
    x[1] = static_cast<int32_t>(col);
  } else {
    x[0] = static_cast<int32_t>(row / g.n[1]);
    x[1] = static_cast<int32_t>(row - static_cast<int64_t>(x[0]) * g.n[1]);
    x[2] = static_cast<int32_t>(col);
  }
  return true;
}

// d = L ((1 - c) + c / q), e = 0.5 max(L (1 + (span - 1) c) - d, 0) in T: the cube side shrinks
// with the coherence c, and the segment half length e takes up what the side gave away.
template <class T>
struct ElemDE {
  T d, e;
};
template <class T>
__host__ __device__ __forceinline__ ElemDE<T> element_de(T L, T c, T q, T span_m1) {
  const T one = T(1);
  const T f = add_rn(sub_rn(one, c), div_t(c, q));
  ElemDE<T> r;
  r.d = mul_rn(L, f);
  const T len = mul_rn(L, add_rn(one, mul_rn(span_m1, c)));
  const T diff = sub_rn(len, r.d);
  r.e = mul_rn(T(0.5), diff > T(0) ? diff : T(0));
  return r;
}

// Half-sample symmetric periodic reflection of any integer index into [0, n).
__host__ __device__ __forceinline__ int64_t reflect_any(int64_t i, int64_t n) {
  const int64_t m = floor_mod(i, 2 * n);
  return m < n ? m : 2 * n - 1 - m;
}

}  // namespace eda::element_detail
