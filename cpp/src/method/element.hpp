// SPDX-License-Identifier: Apache-2.0
// The orientation, the variable structuring element B(x) of one mode and its operators Dil_B (with
// Ero_B) and Avg_B, and the envelope mean. Implementations: orientation.cu, element.cu and
// element_op.cuh; the kernels are named eda_orientation_*, eda_element_* and eda_element_op_*.
//
// The element of a mode (fixed within the mode), per voxel, from the scale field L, the coherence
// c and the orientation n:
//   d = L ((1 - c) + c / q)                   cube side, T
//   e = 0.5 max(L (1 + (span - 1) c) - d, 0)  segment half length, T
//   o = e n                                   segment half vector (D >= 2), T, D components
//   ladder: the odd rungs odd(w q^j) (q = sqrt 2) for j in [j_lo, j_hi] (rung_ladder), each capped
//           at 2 n_max + 1, deduplicated, ascending. The hat weights in log d select at most two
//           rungs per voxel: lo = the lower rung index, t = the weight of rung lo + 1 (rung lo has
//           1 - t; a weight of exactly 0 does not put the voxel on that rung). A filter of every
//           voxel's own side would cost a pass per distinct side; the ladder needs one cube filter
//           per rung for the whole field;
//   rung_pts bit j: some voxel with a non-zero weight on rung j has e > 0.25; then EVERY voxel of
//           rung j reads its cube at the 2 m + 1 points x + (i / m) o, i = -m..m (else at x only).
// Operators, per rung j in ascending order (out starts at 0, out += weight * acc):
//   f_j = cube(h, op, w_j) (enqueue_cube), acc = op over the points of f_j read by D-linear
//   interpolation with half-sample reflection (as scipy.ndimage.map_coordinates with order 1 and
//   mode "reflect");
//   op max / min: acc = max / min of the 2 m + 1 values; box: acc = (f(x) + sum of the 2 m point
//   values in the order +1/m, -1/m, +2/m, -2/m) / (2 m + 1) (a division, left to right sums).
//   A voxel's first positive weight STORES weight * acc, its second ADDS weight * acc (equal to
//   0 + a_lo acc_lo + a_hi acc_hi up to the sign of a zero); products and sums are separate
//   roundings (the _rn intrinsics: no contraction into a fused multiply-add). Point coordinates
//   are formed in double for both dtypes, so a float32 field reads the same interpolation weights
//   as far as float32 can hold them.
#pragma once

#include "runtime/context.hpp"
#include "field/dct.hpp"
#include "field/geom.hpp"
#include "field/state.hpp"
#include "method/method.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace eda {

// The element of the current mode, as the operators read it (device pointers).
template <class T>
struct ElemView {
  const uint8_t* lo;           // (B, field): lower rung index of every voxel
  const T* t;                  // (B, field): weight of rung lo + 1
  FieldVec<const T> o;         // segment half vector (D >= 2; all nullptr in 1D)
  const int32_t* rung_win;     // [n_rungs_max * B]: rung j of element b at rung_win[j * B + b] (0: none;
                               // enqueue_element_rung_windows)
  const ElemRecord* le;        // [B]: rung_pts (and n_rungs)
  int32_t n_rungs_max;         // the largest ladder of the call's active elements (host)
  // rung_group > 1: the operators run the cubes of up to rung_group rungs together
  // (enqueue_cube_rungs) into rung0 (and rung1, the min of max+min), rung_group fields each,
  // rung j of a group at + j B field; else one rung at a time through tmp0 (tmp1).
  T* rung0;
  T* rung1;
  int32_t rung_group;
};

// The monogenic orientation (D >= 2): the band X_b = G X of the orthonormal N-D DCT-II X of r (G
// the one-octave log-Gabor filter of centre period w = le[b].w), the first- and second-order Riesz
// components R1_a, R2_ab (a <= b) of the band in the DCT domain (the inverse DCT of the remapped,
// multiplied band, times (-1)^(sum of the odd axes' coordinates); orientation.cu derives it), the
// energy tensor T = R1 R1^T + R2 R2^T, then its eigen-decomposition into the dominant direction n
// and the coherence c. The DCT is the transform of the half-sample symmetric extension, so the
// field's borders reflect instead of wrapping around. Elements with W[b] <= 0 or on[b] == 0 are
// skipped (W is only a run flag here). An element's outputs depend on its own data only, bit for
// bit, whatever the other elements of the batch hold: the transforms are complex ones, which
// compute every batch entry alone (field/dct.hpp), in groups of orientation_group(g) elements with
// the plan of orientation_plans (fixed by the shape, so the batch and its chunking never change the
// plan an element sees); ws: the group workspace of orientation_workspace_bytes bytes. r1: D
// fields (may alias n: n is written last); work: D (D + 1) / 2 fields, R2 in the order (0,0),
// (0,1), (0,2), (1,1), (1,2), (2,2) (2D: (0,0), (0,1), (1,1)), replaced by T. host_run [B]
// (nullptr: every group): the transforms of a group whose elements are all 0 there are skipped,
// and its R1 and R2 stay as they were; the tensor and eigen passes still run on every element
// that W and on select, so host_run must not leave out such an element (the driver passes the
// active elements, which on selects).
int64_t orientation_group(const Geom& g);
size_t orientation_workspace_bytes(const Geom& g, DType dtype);
// The cached plan of the group geometry (dtype, D, n, orientation_group(g)) of the context (D = 1
// too, for the monogenic signal). When cuFFT cannot allocate the plan's memory, the device's pools
// give their unused memory back to the driver and the plan is made once more.
const DctPlans& orientation_plans(Context& ctx, const Geom& g, DType dtype);
// The Riesz components into r1 / work, then T in place over work and the eigen pass into n, c.
template <class T>
void enqueue_orientation(cudaStream_t s, const Geom& g, const DctPlans& plans, const T* r, const int32_t* W,
                         FieldVec<T> n, T* c, T* work, FieldVec<T> r1, void* ws, const int32_t* on, ElemRecord* le,
                         const uint8_t* host_run = nullptr);
// The eigen pass alone (element.cu): n, c of the D x D symmetric tensor in work (the order
// above). The last stage of enqueue_orientation; the tests drive it directly. W nullptr: every
// element runs; le nullptr: no non-finite flag.
template <class T>
void enqueue_element_eigen(cudaStream_t s, const Geom& g, const T* work, const int32_t* W, FieldVec<T> n, T* c,
                           const int32_t* on, ElemRecord* le);

// The monogenic signal of every element of a chunk (eda.monogenic, method/monogenic.cpp), on the
// orientation's transforms without the band: the first- and second-order Riesz components R1, R2
// and the derivatives in cycles per sample D_a f and D_b R_a (D_a = d/dx_a / (2 pi); orientation.cu
// derives them), then, for D >= 2, n and c from T = R1 R1^T + R2 R2^T by the eigen pass, n with the
// canonical sign (its first component, in axis order, of magnitude above the dtype's default tie
// tolerance 2^-14 / 2^-30 positive: the others are rounding noise). Per sample, with q = n . R1:
//   amplitude A = sqrt(f^2 + |R1|^2), phase = atan2(q, f), frequency = (f D_n q - q D_n f) / A^2
// (D_n f = sum_b n_b D_b f, D_n q = sum_ab n_a n_b D_b R_a), the phase and the frequency 0 where
// A^2 <= tiny; n = 1 in 1D. f: the chunk's fields times 2^-exp[b] (the prescale; the amplitude is
// scaled back). The outputs go to the call's planes in out, each C-contiguous over the call's
// out_batch elements, the chunk's element b at b0 + b: amplitude, phase, frequency (out_batch,
// count), direction (out_batch, D, count) and coherence (out_batch, count) for D >= 2
// (monogenic_rows(D) planes of out_batch count values). The fields of w are the chunk's (padded,
// D components each, work D (D + 1) / 2 fields, 1 in 1D); ws and the plans as for the orientation.
template <class T>
struct MonogenicFields {
  FieldVec<T> r1, df, n;
  T* work;
  T* c;
};
int32_t monogenic_rows(int dim);
template <class T>
void enqueue_monogenic(cudaStream_t s, const Geom& g, const DctPlans& plans, const T* f, const int32_t* exp,
                       const MonogenicFields<T>& w, void* ws, double tiny, T* out, int64_t out_batch, int64_t b0);

// return_orientation (D >= 2): the driver stores the orientation of each mode into out (B, rows,
// D + 1, count) of the chunk, at row elems[b].n_imfs (the row of the mode's IMF), for the elements
// whose mode is active (le[b].status): n with the canonical sign, then c; the collect copies the
// rows 0 .. n_imfs[b] - 1 into dir (B, K, D, count) and coh (B, K, count) and writes 0 above them
// and for elements with error[b] != 0.
template <class T>
void enqueue_orientation_store(cudaStream_t s, const Geom& g, FieldVec<const T> n, const T* c, const ElemRecord* le,
                               const ElemState* elems, T* out, int32_t rows);
template <class T>
void enqueue_orientation_collect(cudaStream_t s, int64_t batch, int dim, int64_t count, const int32_t* n_imfs,
                                 const int32_t* error, const T* src, int32_t rows, T* dir, T* coh, int32_t K);

// Element build, part 1: o = e n (D >= 2) and the per-element min / max of d into
// le[b].d_min_bits / d_max_bits (integer atomics on the bits of positive doubles, whose integer
// order is their order: the result does not depend on the order of the atomics; the caller sets
// them to +inf / 0 bits first). n, o: nullptr components in 1D.
template <class T>
void enqueue_element_build(cudaStream_t s, const Geom& g, const MethodCtl& c, const T* L, FieldVec<const T> n,
                           const T* coh, FieldVec<T> o, ElemRecord* le, const int32_t* on);

// The ladder of one element (host): j_lo = floor(log(max(d_min, 1) / w) / log q + 1e-9),
// j_hi = ceil(log(max(d_max, 1) / w) / log q - 1e-9), rungs {min(odd(w q^j), 2 n_max + 1)},
// ascending. The 1e-9 margins keep a d that lies on a rung up to rounding from adding the rung
// below or above. Throws std::length_error when the ladder exceeds kMaxRungs.
std::vector<int32_t> rung_ladder(double d_min, double d_max, int32_t w, int32_t n_max);

// Element build, part 2 (after the host uploaded the ladders into le[b].n_rungs / rung_w and
// cleared rung_pts and rung_used): lo, t of every voxel (the hat weights in log d: lo =
// clip(#{j : lw[j] <= log d'} - 1, 0, K - 2) with d' = clip(d, w_0, w_K), t = (log d' - lw[lo]) /
// (lw[lo + 1] - lw[lo]); one rung: lo = 0, t = 0) and le[b].rung_pts, le[b].rung_used (integer
// atomicOr). d and e are recomputed from L and c exactly as in part 1. Also fills le[b].rung_lw
// with the device log, the same function as the voxels' log d.
template <class T>
void enqueue_element_weights(cudaStream_t s, const Geom& g, const MethodCtl& c, const T* L, const T* coh,
                             ElemRecord* le, uint8_t* lo, T* t, const int32_t* on);

// The per-rung window arrays the operators pass to enqueue_cube: rung_win[j * B + b] =
// le[b].rung_w[j] where bit j of le[b].rung_used is set and le[b].status == kRecordActive, else 0
// (an unused rung, a rung beyond the element's ladder or a finished element is skipped), for
// j < n_rungs_max. Built on the device: no host round trip between the weights pass and the
// operators.
void enqueue_element_rung_windows(cudaStream_t s, int32_t batch, const ElemRecord* le, int32_t n_rungs_max,
                                  int32_t* rung_win);

// The operators. kMax: out0 = Dil_B in; kMaxMin: out0 = Dil_B in, out1 = Ero_B in (one cube
// max/min per rung); kBox: out0 = Avg_B in. tmp0 (and tmp1 for kMaxMin) hold the rung's cube, or
// the view's rung scratch the cubes of a rung group (the results are the same bits). in must not
// alias an output, a temporary or the rung scratch. Elements with no rung (rung_win 0 for every j)
// are skipped and their outputs untouched.
enum class ElemOp : int8_t { kMax = 0, kMaxMin = 2, kBox = 3 };
template <class T>
void enqueue_element_op(cudaStream_t s, const Geom& g, ElemOp op, const ElemView<T>& ev, const T* in, T* out0,
                        T* out1, T* tmp0, T* tmp1);

// The envelope mean: u = Dil_B h, l = Ero_B h, mid = 0.5 (u + l) (the sum, then the product by
// 0.5), m = Avg_B mid. u and l are outputs (the continuation weight and G1 read them); mid uses
// tmp0. Every pointer distinct.
template <class T>
void enqueue_envelope_mean(cudaStream_t s, const Geom& g, const ElemView<T>& ev, const T* h, T* m, T* u, T* l,
                           T* tmp0, T* tmp1);

}  // namespace eda
