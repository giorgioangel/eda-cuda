// SPDX-License-Identifier: Apache-2.0
// eda::decompose: validation and the dispatch to the method's driver (method/method.hpp).
#include "decompose.hpp"

#include "method/method.hpp"
#include "field/vhgw.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace eda {

int32_t max_axis_len(DType dtype) { return dtype == DType::F32 ? vhgw_max_line<float>() : vhgw_max_line<double>(); }

double default_tie_tolerance(DType dtype) { return dtype == DType::F32 ? std::ldexp(1.0, -14) : std::ldexp(1.0, -30); }

void validate(const Input& in, const Params& p) {
  validate_method(in, p);
}

void describe_device_input(const CudaArray& a, DType dtype, Input& in, SourceLayout& layout) {
  if (a.dim < 1 || a.dim > kMaxDim) throw std::invalid_argument("eda: 1 to 3 spatial axes");
  if (a.batch_shape.size() != a.batch_strides.size()) throw std::invalid_argument("eda: batch shape and strides differ in length");
  if (a.batch_shape.size() > static_cast<size_t>(kMaxBatchAxes)) {
    throw std::invalid_argument("eda: a CUDA array may have at most " + std::to_string(kMaxBatchAxes) + " batch axes");
  }
  int64_t batch = 1, count = 1;
  for (const int64_t e : a.batch_shape) {
    if (e < 0) throw std::invalid_argument("eda: negative extent");
    batch *= e;
  }
  for (int d = 0; d < a.dim; ++d) {
    if (a.n[d] < 0) throw std::invalid_argument("eda: negative extent");
    count *= a.n[d];
  }
  const size_t item = source_item_size(a.type);
  if (batch * count > 0) {
    if (!a.data) throw std::invalid_argument("eda: the CUDA array has no data pointer");
    if (reinterpret_cast<uintptr_t>(a.data) % item != 0) {
      throw std::invalid_argument("eda: the CUDA array's data pointer is not aligned to its item size (" +
                                  std::to_string(item) + " bytes)");
    }
  }
  // the batch axes without the extent-1 ones, neighbours merged
  std::vector<int64_t> shape, stride;
  for (size_t k = 0; k < a.batch_shape.size(); ++k) {
    if (a.batch_shape[k] == 1) continue;
    if (!shape.empty() && stride.back() == a.batch_strides[k] * a.batch_shape[k]) {
      shape.back() *= a.batch_shape[k];
      stride.back() = a.batch_strides[k];
    } else {
      shape.push_back(a.batch_shape[k]);
      stride.push_back(a.batch_strides[k]);
    }
  }
  in = Input{};
  in.dtype = dtype;
  in.x = a.data;
  in.on_device = true;
  in.dim = a.dim;
  in.batch = batch;
  for (int d = 0; d < a.dim; ++d) in.n[d] = a.n[d];
  const bool same_type = a.type == (dtype == DType::F32 ? SourceType::F32 : SourceType::F64);
  if (same_type && shape.size() <= 1) {
    in.strides.s[0] = shape.empty() ? 0 : stride[0];
    for (int d = 0; d < a.dim; ++d) in.strides.s[d + 1] = a.strides[d];
    return;
  }
  layout = SourceLayout{};
  layout.type = a.type;
  layout.batch_axes = static_cast<int32_t>(shape.size());
  for (size_t k = 0; k < shape.size(); ++k) {
    layout.batch_shape[k] = shape[k];
    layout.batch_stride[k] = stride[k];
  }
  for (int d = 0; d < a.dim; ++d) layout.stride[d] = a.strides[d];
  in.source = &layout;
}

std::unique_ptr<Run> decompose(Context& ctx, const Input& in, const Params& p) {
  validate(in, p);
  return run_method(ctx, in, p);
}

}  // namespace eda
