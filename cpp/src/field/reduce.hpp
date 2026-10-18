// SPDX-License-Identifier: Apache-2.0
// Canonical reductions: the energy and the maximum of |v| of every element in one fixed order of
// operations, so that the totals depend only on the element's samples, never on the batch, the
// chunking or the grid.
#pragma once
#include "field/state.hpp"
#include "field/geom.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace eda {
// The partial of one contiguous row: sum v^2 in double (the order of common.cuh's warp_row_partial)
// and the bit pattern of max |v|.
struct RowPartial { double sum_sq; uint64_t amax_bits; };
// Every element b: elems[b].r_energy = the pairwise tree over rows[b*g.rows ...].sum_sq in row
// order and elems[b].amax = the double whose bit pattern is the largest amax_bits (amax_bits holds
// the bit pattern of the double |v|, so the integer maximum is the maximum of |v|).
void enqueue_reduce_partials(cudaStream_t s, const Geom& g, const RowPartial* rows, ElemState* elems);
}  // namespace eda
