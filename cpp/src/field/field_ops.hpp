// SPDX-License-Identifier: Apache-2.0
// Host entry points of the field kernels (field_ops.cu): the input copy, the canonical
// reductions, the element state, the end-of-mode decision and the output rows. Used by the driver
// and the C++ tests. Every kernel is named eda_field_ops_*.
#pragma once

#include "field/state.hpp"
#include "field/geom.hpp"
#include "field/reduce.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace eda {

// Element strides of the input (B, n_0..n_{D-1}): s[0] batch, s[1 + a] axis a (any sign).
struct InputStrides {
  int64_t s[kMaxDim + 1];
};

// r <- the input gathered into the padded working field, pads written as 0.
template <class T>
void enqueue_canonical_in(cudaStream_t s, const Geom& g, const T* x, const InputStrides& xs, T* r);
// Canonical row partials (reduce.hpp) of every row of f; with elems, adds each element's count of
// non-finite samples to elems[b].nonfinite (integer atomics, so the count is order independent;
// the caller zeroes it).
template <class T>
void enqueue_row_partials(cudaStream_t s, const Geom& g, const T* f, RowPartial* rows, ElemState* elems);
// Initial element state after enqueue_reduce_partials over the input's row partials: tol from
// ctl->tau, status kActive (kError / kErrNonFinite for non-finite input with ctl->check_finite),
// n_imfs = 0, window = 0.
template <class T>
void enqueue_init_state(cudaStream_t s, const Geom& g, ElemState* elems, const CallCtl* ctl);
// End of a mode: the history row (scale, energy), n_imfs, amax/tol/r_energy of the new
// residual, the status, and the loop condition (last CTA: ctl->any_active, ctl->any_full).
template <class T>
void enqueue_decide(cudaStream_t s, const Geom& g, ElemState* elems, CallCtl* ctl, const RowPartial* imf_rows,
                    const RowPartial* r_rows);
// Row n_imfs of every element that is not in error <- r (unpadded).
template <class T>
void enqueue_write_residue(cudaStream_t s, const Geom& g, const ElemState* elems, const CallCtl* ctl, const T* r);
// dst (B, P, count) <- rows 0..n_imfs[b] of src (B, rows_cap, count), zeros above, NaN rows for
// elements with error[b] != 0.
template <class T>
void enqueue_collect(cudaStream_t s, const Geom& g, const int32_t* n_imfs, const int32_t* error, const T* src,
                     int32_t rows_cap, T* dst, int32_t P);

}  // namespace eda
