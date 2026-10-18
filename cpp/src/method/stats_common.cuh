// SPDX-License-Identifier: Apache-2.0
// Device helpers shared by the stats*.cu translation units (the padded layout of
// method/method.hpp, order keys of doubles, T-generic libm, launch helpers, the layout of the
// selection workspace). Not an interface: only the stats*.cu files include it.
#pragma once

#include "runtime/error.hpp"
#include "field/common.cuh"
#include "field/geom.hpp"
#include "method/method.hpp"
#include "method/stats.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace eda::stats_detail {

inline constexpr int kThreads = 256;

inline unsigned blocks_for(int64_t n, int threads = kThreads) {
  int64_t b = (n + threads - 1) / threads;
  if (b < 1) b = 1;
  if (b > 0x7fffffffLL) throw std::length_error("eda: stats: launch grid too large");
  return static_cast<unsigned>(b);
}

// A pad sample of the padded layout (innermost coordinate >= n[D-1]).
__host__ __device__ __forceinline__ bool pad_at(const Geom& g, int64_t v) { return v % g.pitch >= g.n[g.dim - 1]; }
// Coordinate along axis a of the padded in-element index v.
__host__ __device__ __forceinline__ int32_t coord_of(const Geom& g, int64_t v, int a) {
  if (a == g.dim - 1) return static_cast<int32_t>(v % g.pitch);
  return static_cast<int32_t>((v / g.stride[a]) % g.n[a]);
}
__device__ __forceinline__ bool runs(const int32_t* on, int64_t b) { return on == nullptr || on[b] != 0; }

// Order key of a double: the unsigned order of keys is the order of the values (-0 maps to +0, so
// equal values have equal keys; NaN never occurs).
__host__ __device__ __forceinline__ uint64_t order_key(double d) {
#if defined(__CUDA_ARCH__)
  uint64_t u = static_cast<uint64_t>(__double_as_longlong(d));
#else
  uint64_t u;
  std::memcpy(&u, &d, sizeof u);
#endif
  if ((u << 1) == 0) u = 0;  // -0 -> +0
  return (u >> 63) ? ~u : (u | 0x8000000000000000ULL);
}
__host__ __device__ __forceinline__ double key_value(uint64_t k) {
  const uint64_t u = (k >> 63) ? (k & 0x7fffffffffffffffULL) : ~k;
  return bits_to_double(u);
}

// T-generic libm and IEEE round-to-nearest operations (no contraction).
template <class T>
__device__ __forceinline__ T log_t(T v) {
  if constexpr (std::is_same_v<T, float>) return logf(v); else return log(v);
}
template <class T>
__device__ __forceinline__ T exp_t(T v) {
  if constexpr (std::is_same_v<T, float>) return expf(v); else return exp(v);
}
template <class T>
__device__ __forceinline__ T sqrt_t(T v) {
  if constexpr (std::is_same_v<T, float>) return __fsqrt_rn(v); else return __dsqrt_rn(v);
}
template <class T>
__device__ __forceinline__ T div_t(T a, T b) {
  if constexpr (std::is_same_v<T, float>) return __fdiv_rn(a, b); else return __ddiv_rn(a, b);
}
template <class T>
__device__ __forceinline__ T inf_t() {
  if constexpr (std::is_same_v<T, float>) {
    return __int_as_float(0x7f800000);
  } else {
    return __longlong_as_double(0x7ff0000000000000LL);
  }
}

// NumPy's quantile rule (method "linear": its virtual index, gamma and lerp) with round-to-nearest
// intrinsics. NumPy 2.5 computes the virtual index as (n - 1) q, not as n q + (1 - q) - 1 (which
// rounds differently, e.g. n = 3, q = 0.1 gives 0.20000000000000018 instead of 0.2); an index
// >= n - 1 takes the last value twice with gamma = v + 1 (NumPy subtracts the replaced index -1),
// an index < 0 the first. The lerp takes the form that is exact at its end (b - diff (1 - gamma)
// for gamma >= 0.5). Plain double expressions would contract into fused multiply-adds on the
// device.
__device__ __forceinline__ void quantile_index_rn(int64_t n, double q, int64_t* i, int64_t* j, double* gamma) {
  const double v = __dmul_rn(static_cast<double>(n - 1), q);
  if (v >= static_cast<double>(n - 1)) {
    *i = *j = n - 1;
    *gamma = __dadd_rn(v, 1.0);
  } else if (v < 0.0) {
    *i = *j = 0;
    *gamma = v;
  } else {
    const double f = floor(v);
    *i = static_cast<int64_t>(f);
    *j = *i + 1;
    *gamma = __dsub_rn(v, f);
  }
}
__device__ __forceinline__ double quantile_linear_rn(double a, double b, double gamma) {
  const double diff = __dsub_rn(b, a);
  return gamma >= 0.5 ? __dsub_rn(b, __dmul_rn(diff, __dsub_rn(1.0, gamma))) : __dadd_rn(a, __dmul_rn(diff, gamma));
}

// ---------------------------------------------------------------------------------------------
// The selection workspace (select_workspace_bytes): per element a SelState, two histograms of
// 2^11 64-bit counts (the two order statistics of a median / quantile), then the scratch of the
// callers that run a selection themselves (noise levels, G2, G3).
inline constexpr int kSelBits = 11;
inline constexpr int kSelBins = 1 << kSelBits;
inline constexpr int kSelPasses = 6;  // 11 + 11 + 11 + 11 + 11 + 9 bits

struct alignas(16) SelState {
  uint64_t prefix[2];  // the known high bits of the two keys (lower bits 0)
  uint64_t rank[2];    // the remaining ranks inside the current prefix bucket
  int64_t count;       // the size of the value set
  int64_t i_lo;        // quantile: the lower index
  double gamma;        // quantile: the interpolation weight
  int32_t valid;       // count > 0
  int32_t same;        // prefix[0] == prefix[1] (one histogram serves both)
};

struct SelExtra {     // per element scratch of the callers (each a [B] array)
  double* center;     // noise levels: median(d)
  double* mad;        // noise levels: the MAD
  double* acc;        // noise levels: the running max over axes
  double* p;          // G2: the quantile p
  int64_t* n_p;       // G2: |P|; G3: the selection size
  int32_t* win;       // G2 / G3: four window arrays [4 B]
  int32_t* any;       // G2: two any-flags [2 B]
  int32_t* on;        // G2 / G3: the running elements [B]
};

inline size_t align_up(size_t v) { return (v + 255) / 256 * 256; }

inline size_t sel_core_bytes(int64_t B) {
  return align_up(static_cast<size_t>(B) * sizeof(SelState)) +
         align_up(static_cast<size_t>(B) * 2 * kSelBins * sizeof(uint64_t));
}
inline size_t sel_extra_bytes(int64_t B) {
  return 5 * align_up(static_cast<size_t>(B) * 8) + align_up(static_cast<size_t>(B) * 4 * 4) +
         align_up(static_cast<size_t>(B) * 2 * 4) + align_up(static_cast<size_t>(B) * 4);
}
inline SelState* sel_state(void* ws) { return static_cast<SelState*>(ws); }
inline uint64_t* sel_hist(void* ws, int64_t B) {
  return reinterpret_cast<uint64_t*>(static_cast<char*>(ws) + align_up(static_cast<size_t>(B) * sizeof(SelState)));
}
inline SelExtra sel_extra(void* ws, int64_t B) {
  char* p = static_cast<char*>(ws) + sel_core_bytes(B);
  const size_t d = align_up(static_cast<size_t>(B) * 8);
  SelExtra e;
  e.center = reinterpret_cast<double*>(p);
  e.mad = reinterpret_cast<double*>(p + d);
  e.acc = reinterpret_cast<double*>(p + 2 * d);
  e.p = reinterpret_cast<double*>(p + 3 * d);
  e.n_p = reinterpret_cast<int64_t*>(p + 4 * d);
  char* q = p + 5 * d;
  e.win = reinterpret_cast<int32_t*>(q);
  q += align_up(static_cast<size_t>(B) * 4 * 4);
  e.any = reinterpret_cast<int32_t*>(q);
  q += align_up(static_cast<size_t>(B) * 2 * 4);
  e.on = reinterpret_cast<int32_t*>(q);
  return e;
}

// The CCL / scale-field workspace (ccl_workspace_bytes): a byte per padded sample (the
// components' "meets near" marks, indexed by the label) and two window arrays [B].
inline size_t ccl_ws_bytes(const Geom& g) {
  return align_up(static_cast<size_t>(g.batch * g.field)) + 2 * align_up(static_cast<size_t>(g.batch) * 4);
}
inline uint8_t* ccl_marks(void* ws) { return static_cast<uint8_t*>(ws); }
inline int32_t* ccl_win(void* ws, const Geom& g, int k) {
  return reinterpret_cast<int32_t*>(static_cast<char*>(ws) + align_up(static_cast<size_t>(g.batch * g.field)) +
                                    k * align_up(static_cast<size_t>(g.batch) * 4));
}

}  // namespace eda::stats_detail
