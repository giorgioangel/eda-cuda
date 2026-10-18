// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace eda {

inline constexpr int kMaxDim = 3;       // every kernel handles D = 1, 2 or 3 spatial axes
inline constexpr int kPitchAlign = 32;  // innermost extent padded to a multiple of 32 elements

// One call's working field: C-contiguous (B, n_0..n_{D-1}), innermost extent padded to `pitch` so
// that every row starts on a 32-element boundary (aligned, coalesced row access). Pad elements are
// zero, so a pad never changes a sum or a maximum of |v|.
struct Geom {
  int32_t dim;               // D in 1..3
  int32_t n[kMaxDim];        // extents; n[a] = 1 for a >= dim
  int64_t batch;             // B >= 1
  int64_t pitch;             // round_up(n[dim-1], 32)
  int64_t stride[kMaxDim];   // padded element strides inside one element; stride[dim-1] = 1
  int64_t field;             // padded elements per element = n[0] * stride[0]
  int64_t count;             // samples per element = prod n (unpadded)
  int64_t rows;              // contiguous-axis lines per element = field / pitch
};

// Throws std::invalid_argument for dim outside 1..3, an extent < 1 or field > 2^31.
Geom make_geom(int dim, int64_t batch, const int64_t* n);

// Lines along axis a: sample i of line (o, j) is at o*outer_stride + i*step + j, o < outer,
// i < n, j < inner. Contiguous axis: inner = 1, step = 1, outer_stride = pitch. Other axes:
// step = inner = stride[a] (pad columns included; they are zero lines).
struct LineView {
  int64_t outer;          // B * prod_{b<a} n_b
  int32_t n;              // n_a
  int64_t step;
  int64_t inner;
  int64_t outer_stride;   // n_a * stride[a], or pitch for the contiguous axis
  int64_t per_element;    // outer positions per element: element(o) = o / per_element
};
// Inline: device code is built without relocatable device code, so every translation unit that
// calls it needs the definition.
__host__ __device__ inline LineView line_view(const Geom& g, int axis) {
  LineView v{};
  int64_t before = 1;
  for (int b = 0; b < axis; ++b) before *= g.n[b];
  v.outer = g.batch * before;
  v.n = g.n[axis];
  v.per_element = before;
  if (axis == g.dim - 1) {
    v.step = 1;
    v.inner = 1;
    v.outer_stride = g.pitch;
  } else {
    v.step = g.stride[axis];
    v.inner = g.stride[axis];
    v.outer_stride = static_cast<int64_t>(g.n[axis]) * g.stride[axis];
  }
  return v;
}

}  // namespace eda
