// SPDX-License-Identifier: Apache-2.0
// make_geom: the padded working-field geometry of a call. Every padded index fits int32 (field <=
// 2^31), which the kernels' int32 labels and offsets rely on.
#include "field/geom.hpp"

#include <stdexcept>
#include <string>

namespace eda {

Geom make_geom(int dim, int64_t batch, const int64_t* n) {
  if (dim < 1 || dim > kMaxDim) {
    throw std::invalid_argument("eda: the number of spatial axes must be 1, 2 or 3, got " + std::to_string(dim));
  }
  if (batch < 1) throw std::invalid_argument("eda: the batch must hold at least one element");
  constexpr int64_t kMaxField = int64_t{1} << 31;
  Geom g{};
  g.dim = dim;
  g.batch = batch;
  for (int a = 0; a < kMaxDim; ++a) g.n[a] = 1;
  int64_t count = 1;
  for (int a = 0; a < dim; ++a) {
    if (n[a] < 1) throw std::invalid_argument("eda: every spatial extent must be >= 1, got " + std::to_string(n[a]));
    if (n[a] > kMaxField) throw std::invalid_argument("eda: a spatial extent exceeds 2^31");
    g.n[a] = static_cast<int32_t>(n[a]);
    count *= n[a];
    if (count > kMaxField) throw std::invalid_argument("eda: an element has more than 2^31 samples");
  }
  g.count = count;
  g.pitch = (static_cast<int64_t>(g.n[dim - 1]) + kPitchAlign - 1) / kPitchAlign * kPitchAlign;
  g.stride[dim - 1] = 1;
  for (int a = dim; a < kMaxDim; ++a) g.stride[a] = 1;
  if (dim >= 2) g.stride[dim - 2] = g.pitch;
  for (int a = dim - 3; a >= 0; --a) g.stride[a] = static_cast<int64_t>(g.n[a + 1]) * g.stride[a + 1];
  g.field = dim == 1 ? g.pitch : static_cast<int64_t>(g.n[0]) * g.stride[0];
  if (g.field > kMaxField) {
    throw std::invalid_argument("eda: an element's padded field (" + std::to_string(g.field) +
                                " samples, the last axis padded to a multiple of 32) exceeds 2^31");
  }
  g.rows = g.field / g.pitch;
  return g;
}

}  // namespace eda
