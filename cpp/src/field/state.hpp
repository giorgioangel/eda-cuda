// SPDX-License-Identifier: Apache-2.0
// Device-resident per-element state and per-call control of a decomposition: the state the field
// kernels (field_ops.hpp) keep for every batch element, and the block of call-wide settings and
// output pointers they read.
#pragma once
#include "field/geom.hpp"

#include <cstdint>

namespace eda {
enum ElemStatus : int32_t { kActive = 0, kDone = 1, kFull = 2, kError = 3 };
// kErrNonFinite: the input holds inf or NaN and check_finite is on. kErrNonFiniteResult: a
// non-finite intermediate or output row although the input passed (or skipped) the check_finite
// test: an overflow of the dtype's range, or an inf / NaN input with check_finite off.
// kErrInternal: an element ended in a state the driver does not expect (a bug).
enum ElemError : int32_t { kErrNone = 0, kErrNonFinite = 1, kErrNonFiniteResult = 2, kErrInternal = 255 };

struct alignas(16) ElemState {  // one per batch element, device-resident
  int32_t status;               // ElemStatus
  int32_t n_imfs;               // IMF rows written; the residue goes to row n_imfs
  int32_t window;               // the window of the current mode (odd); 0 before the first
  int32_t error;                // ElemError
  double amax;                  // max |r| (exact: a maximum needs no rounding)
  double tol;                   // fl_T(tau * amax): the tie tolerance of r, exact in T
  double r_energy;              // sum r^2 in the fixed reduction order of reduce.hpp
  uint64_t nonfinite;           // count of inf / NaN samples of the input copy
};

struct CallCtl {                // one per call, device-resident; kernels read pointers here
  void* out;                    // (B, rows_cap, count) unpadded, T
  int32_t rows_cap;
  int32_t max_imf;              // <= 0: no limit
  int32_t any_active;           // some element is still active (written by eda_field_ops_decide)
  int32_t any_full;             // some element ran out of output rows (a bug: the rows are bounded)
  uint32_t ticket;              // counts the finished CTAs of eda_field_ops_decide (last-CTA election)
  int32_t* hist_scale;          // (B, rows_cap-1): the mode's window
  double* hist_spacing;         // (B, rows_cap-1): the extrema spacing of the mode
  int32_t* hist_sifts;          // (B, rows_cap-1): the mode's sifts
  double* hist_energy;          // (B, rows_cap-1): sum IMF^2
  double tau;                   // relative tie tolerance, already rounded to T
  int32_t check_finite;         // non-finite input makes the element fail (kErrNonFinite)
  int32_t acc_active;           // eda_field_ops_decide accumulators, reset by its last CTA
  int32_t acc_full;
};
}  // namespace eda
