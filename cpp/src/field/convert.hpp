// SPDX-License-Identifier: Apache-2.0
// The conversion of a CUDA input that cannot be read in place: a strided gather of the elements of
// one chunk and a cast to the compute dtype, into the C-contiguous (B, *spatial) layout that host
// input has after its copy to the device (convert.cu, kernel eda_convert_gather).
#pragma once

#include "field/geom.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace eda {

// The element types a CUDA input may have.
enum class SourceType : uint8_t { Bool, I8, I16, I32, I64, U8, U16, U32, U64, F16, BF16, F32, F64 };

// "bool", "int8".."int64", "uint8".."uint64", "float16", "bfloat16", "float32", "float64".
std::optional<SourceType> source_type(std::string_view name);
const char* source_type_name(SourceType type);
size_t source_item_size(SourceType type);

inline constexpr int kMaxBatchAxes = 64;

// Element (b, i_0..i_{D-1}) of the input is x[sum_k j_k * batch_stride[k] + sum_a i_a * stride[a]],
// with j the multi-index of b over batch_shape in C order; strides in elements of the source type,
// any sign.
struct SourceLayout {
  SourceType type = SourceType::F64;
  int32_t batch_axes = 0;  // 0..kMaxBatchAxes
  int64_t batch_shape[kMaxBatchAxes] = {};
  int64_t batch_stride[kMaxBatchAxes] = {};
  int64_t stride[kMaxDim] = {};
};

// dst (C-contiguous (g.batch, g.count), unpadded) <- the elements b0 .. b0 + g.batch - 1 of x, each
// cast to T with round to nearest even (bool to 0 or 1; float16, bfloat16 and float32 to float64
// exactly).
template <class T>
void enqueue_convert_gather(cudaStream_t s, const SourceLayout& src, const void* x, int64_t b0, const Geom& g, T* dst);

}  // namespace eda
