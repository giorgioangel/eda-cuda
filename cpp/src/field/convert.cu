// SPDX-License-Identifier: Apache-2.0
// The gather and cast of a CUDA input into the compute dtype (convert.hpp). The casts round as
// NumPy's astype does on x86-64: integers and float64 -> float32 to nearest even (float64 ->
// float32 overflows to inf and keeps subnormals: the build never flushes to zero), the narrower
// floats widen exactly, bool gives 0 or 1.
#include "field/convert.hpp"

#include "runtime/error.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace eda {
namespace {

constexpr int kThreads = 256;

template <class T>
struct Cast;

template <>
struct Cast<float> {
  __device__ static float of(bool v) { return v ? 1.0f : 0.0f; }
  __device__ static float of(long long v) { return __ll2float_rn(v); }
  __device__ static float of(unsigned long long v) { return __ull2float_rn(v); }
  __device__ static float of(__half v) { return __half2float(v); }
  __device__ static float of(__nv_bfloat16 v) { return __bfloat162float(v); }
  __device__ static float of(float v) { return v; }
  __device__ static float of(double v) { return __double2float_rn(v); }
};

template <>
struct Cast<double> {
  __device__ static double of(bool v) { return v ? 1.0 : 0.0; }
  __device__ static double of(long long v) { return __ll2double_rn(v); }
  __device__ static double of(unsigned long long v) { return __ull2double_rn(v); }
  __device__ static double of(__half v) { return static_cast<double>(__half2float(v)); }
  __device__ static double of(__nv_bfloat16 v) { return static_cast<double>(__bfloat162float(v)); }
  __device__ static double of(float v) { return static_cast<double>(v); }
  __device__ static double of(double v) { return v; }
};

// Storage of a bool element: one byte, any nonzero value is true.
struct BoolByte {
  uint8_t v;
};

// The stored value, widened exactly to an argument type of Cast::of.
template <class S>
__device__ auto load(const S* p) {
  if constexpr (std::is_same_v<S, BoolByte>) {
    return p->v != 0;
  } else if constexpr (std::is_integral_v<S> && std::is_unsigned_v<S>) {
    return static_cast<unsigned long long>(*p);
  } else if constexpr (std::is_integral_v<S>) {
    return static_cast<long long>(*p);
  } else {
    return *p;
  }
}

template <class S, class T>
__global__ void __launch_bounds__(kThreads) eda_convert_gather(const S* __restrict__ x, const __grid_constant__ SourceLayout src,
                                                               int64_t b0, Geom g, T* __restrict__ dst) {
  const int64_t total = g.batch * g.count;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t q = idx / g.count;
    int64_t i = idx - q * g.count;
    int64_t off = 0;
    for (int a = g.dim - 1; a >= 0; --a) {
      const int64_t k = i % g.n[a];
      i /= g.n[a];
      off += k * src.stride[a];
    }
    q += b0;
    for (int k = src.batch_axes - 1; k >= 0; --k) {
      const int64_t j = q % src.batch_shape[k];
      q /= src.batch_shape[k];
      off += j * src.batch_stride[k];
    }
    dst[idx] = Cast<T>::of(load<S>(x + off));
  }
}

template <class S, class T>
void launch(cudaStream_t s, const SourceLayout& src, const void* x, int64_t b0, const Geom& g, T* dst) {
  const int64_t work = g.batch * g.count;
  const unsigned grid = static_cast<unsigned>(std::clamp<int64_t>((work + kThreads * 4 - 1) / (kThreads * 4), 1, int64_t{1} << 20));
  eda_convert_gather<S, T><<<grid, kThreads, 0, s>>>(static_cast<const S*>(x), src, b0, g, dst);
  EDA_CHECK_LAUNCH("eda_convert_gather");
}

}  // namespace

std::optional<SourceType> source_type(std::string_view name) {
  for (int t = 0; t <= static_cast<int>(SourceType::F64); ++t) {
    if (name == source_type_name(static_cast<SourceType>(t))) return static_cast<SourceType>(t);
  }
  return std::nullopt;
}

const char* source_type_name(SourceType type) {
  switch (type) {
    case SourceType::Bool: return "bool";
    case SourceType::I8: return "int8";
    case SourceType::I16: return "int16";
    case SourceType::I32: return "int32";
    case SourceType::I64: return "int64";
    case SourceType::U8: return "uint8";
    case SourceType::U16: return "uint16";
    case SourceType::U32: return "uint32";
    case SourceType::U64: return "uint64";
    case SourceType::F16: return "float16";
    case SourceType::BF16: return "bfloat16";
    case SourceType::F32: return "float32";
    case SourceType::F64: return "float64";
  }
  return "?";
}

size_t source_item_size(SourceType type) {
  switch (type) {
    case SourceType::Bool:
    case SourceType::I8:
    case SourceType::U8: return 1;
    case SourceType::I16:
    case SourceType::U16:
    case SourceType::F16:
    case SourceType::BF16: return 2;
    case SourceType::I32:
    case SourceType::U32:
    case SourceType::F32: return 4;
    case SourceType::I64:
    case SourceType::U64:
    case SourceType::F64: return 8;
  }
  return 0;
}

template <class T>
void enqueue_convert_gather(cudaStream_t s, const SourceLayout& src, const void* x, int64_t b0, const Geom& g, T* dst) {
  if (src.batch_axes < 0 || src.batch_axes > kMaxBatchAxes) throw std::invalid_argument("eda: too many batch axes");
  switch (src.type) {
    case SourceType::Bool: return launch<BoolByte, T>(s, src, x, b0, g, dst);
    case SourceType::I8: return launch<int8_t, T>(s, src, x, b0, g, dst);
    case SourceType::I16: return launch<int16_t, T>(s, src, x, b0, g, dst);
    case SourceType::I32: return launch<int32_t, T>(s, src, x, b0, g, dst);
    case SourceType::I64: return launch<int64_t, T>(s, src, x, b0, g, dst);
    case SourceType::U8: return launch<uint8_t, T>(s, src, x, b0, g, dst);
    case SourceType::U16: return launch<uint16_t, T>(s, src, x, b0, g, dst);
    case SourceType::U32: return launch<uint32_t, T>(s, src, x, b0, g, dst);
    case SourceType::U64: return launch<uint64_t, T>(s, src, x, b0, g, dst);
    case SourceType::F16: return launch<__half, T>(s, src, x, b0, g, dst);
    case SourceType::BF16: return launch<__nv_bfloat16, T>(s, src, x, b0, g, dst);
    case SourceType::F32: return launch<float, T>(s, src, x, b0, g, dst);
    case SourceType::F64: return launch<double, T>(s, src, x, b0, g, dst);
  }
  throw std::invalid_argument("eda: unknown source type");
}

template void enqueue_convert_gather<float>(cudaStream_t, const SourceLayout&, const void*, int64_t, const Geom&, float*);
template void enqueue_convert_gather<double>(cudaStream_t, const SourceLayout&, const void*, int64_t, const Geom&, double*);

}  // namespace eda
