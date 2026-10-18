// SPDX-License-Identifier: Apache-2.0
// The mode's control kernels (the two-pass window rule, the G2 bookkeeping, the growth floor, G3),
// the pointwise parts of the continuation weight and of G1, the sifts with the sup-norm guard and
// the mode commit. The driver (driver.cpp) composes them with the element (method/element.hpp) and
// the statistics (method/stats.hpp). Implementations: mode.cu (control, commit; kernels
// eda_mode_*) and sift.cu (weight, G1, sifts; kernels eda_sift_*). The driver and the C++ test
// test_mode are their only users.
//
// Arithmetic: every floating-point operation that feeds a result is a round-to-nearest intrinsic in
// a fixed order (no contraction into fused multiply-adds); fields are T, per-element statistics and
// thresholds double; a T value compared with a double threshold is converted exactly, so the
// comparison is exact too. Elementwise kernels touch the non-pad samples of the elements that run
// and nothing else (pads and skipped elements are never read or written), and may run in place
// where noted.
#pragma once

#include "field/state.hpp"
#include "field/geom.hpp"
#include "method/method.hpp"
#include "method/stats.hpp"
#include "field/reduce.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace eda {

// The per-element window array `win` of the driver: int32 [kWinSlots * B], slot k of element b
// at win[k * B + b], 0 for an element that does not run (the cube filter skips it):
//   kWinWd, kWinWl: the boxes of the next threshold field (odd(rho w_t), odd(max(beta, rho)
//     w_t) for its window w_t: w0 = max(3, w_prev + 2) after mode_start, w1 after pass 0 of
//     mode_rice, w2 = w after the last pass (G2's flags ff));
//   kWinW: W = odd(beta w) of the final window (mode_window), read by the raw window, the
//     orientation, G1 and the scale field's callers.
enum WinSlot : int32_t { kWinWd = 0, kWinWl = 1, kWinW = 2, kWinSlots = 3 };

// The per-element device arrays the driver keeps beside le[]; the statistics and the element take
// some of them as plain [B] arrays. Every pointer is [B] except win ([kWinSlots * B]).
struct ModeArrays {
  ElemRecord* le;
  ElemState* elems;
  double* tol;        // fl_T(tau max|r|) of the mode (mode_start, from elems)
  double* sigma_n;    // noise level of r, pooled (mirrored into le by the first window pass)
  double* sigma_x;    // noise level of the input, the largest per axis (once per call; mirrored by mode_start)
  double* floor2;     // fine_theta sigma_x: G2's threshold floor (mode_start)
  double* wc;         // G3's coarse scale (enqueue_g3_coarse; +inf from mode_start)
  double* e_med;      // G1: median(e) of the current sift
  int32_t* on;        // 1: the element runs this mode (le.status == kRecordActive)
  int32_t* on_g2;     // 1: G2 is eligible this mode (the last window pass)
  int32_t* on_floor;  // 1: active and k >= 1 (the per-voxel floor of the scale field)
  int32_t* win;       // the window slots (WinSlot)
};

// The method's initial state after enqueue_init_state: every ElemRecord zeroed, then w_prev = 1,
// k = 0, status kRecordActive (kRecordError where the element's state already failed), on = 1 for
// the active ones (the sigma_x selection runs on them), wc = +inf.
void enqueue_mode_init(cudaStream_t s, int64_t batch, const ModeArrays& A);

// Per-mode start of every element: an element whose field state is no longer active (max_imf
// reached, an error) leaves the method's loop (status kRecordDone / kRecordError). For the active
// ones: tol, r_inf from elems (after enqueue_reduce_partials or enqueue_decide), tol[b], w_pass = 0,
// ext = 0, fine = 0, jump = absent_jump, g3 = 1, wc = +inf (le and wc[b]), the G2 / G3 values 0,
// sifting = 1, n_sifts = 0, the guard maxima 0, le.sigma_x = sigma_x[b], floor2[b] = fine_theta
// sigma_x[b]; win's kWinWd / kWinWl slots for the first threshold window w0 = max(3, w_prev + 2),
// kWinW = 0; on = 1, on_floor = (k >= 1), on_g2 = 0. Elements that do not run get on = on_g2 =
// on_floor = 0 and zero windows.
void enqueue_mode_start(cudaStream_t s, int64_t batch, const MethodCtl& c, const ModeArrays& A);

// The window rule of a pass: the mean spacing of the significant extrema sets the scale (as in
// Rice's relation between the density of extrema and the dominant frequency). From le.ext (then
// zeroed for the next count), w = odd(2 kappa N / sqrt(sum ext^2)) floored at max(3, w_prev + 2);
// none when no axis has 3 extrema per line on average or w > max n_a: the element is done
// (le.status kRecordDone, elems.status kDone, on = 0, every window slot 0, w_pass[pass] = 0).
// Else w_pass[pass] = w, spacing = N / sqrt(sum ext^2), the next threshold windows Wd = odd(rho w),
// Wl = odd(max(beta, rho) w), and after the last pass (pass == sig_passes - 1) le.w = w and
// on_g2 = fine_jump max(3, w_prev + 2) <= w. Pass 0 also mirrors sigma_n[b] into le.sigma_n.
// The floors make the windows grow strictly from mode to mode, which bounds the number of modes.
void enqueue_mode_rice(cudaStream_t s, int64_t batch, const MethodCtl& c, const Geom& g, int32_t pass,
                       const ModeArrays& A);

// After G2 (the host-driven rung loop) and G3's coarse median: where G2 opened the mode (le.fine)
// w = max(p_odd, w_prev + 2, 3) and jump = 1; then the growth floor (growth > 1 and
// k >= growth_from: w = max(w, odd(growth w_prev)); the element is done when w > n_max); W =
// odd(beta w) into le.W and win's kWinW slot; with le.fine: le.wc = wc[b], le.g3 =
// min(1, split sqrt(wc / p_odd) / 2) when wc is finite (1 else); d_min_bits = +inf, d_max_bits = 0
// for the element build.
void enqueue_mode_window(cudaStream_t s, int64_t batch, const MethodCtl& c, const ModeArrays& A);

// G2's result (the host decides per element): where res[b].outcome == kG2Accepted, le.fine = 1,
// le.g2_rung = res.s, le.p = res.p, le.p_odd = odd_window(res.p), le.cov = res.cov, and the
// accepted rung's population is copied into P_acc (the whole padded field, from P_src0 or P_src1 by
// slot[b]: the buffer the rung's test wrote); other elements are untouched.
void enqueue_mode_g2_accept(cudaStream_t s, const Geom& g, ElemRecord* le, const G2Result* res, const int32_t* slot,
                            const uint8_t* P_src0, const uint8_t* P_src1, uint8_t* P_acc);

// G3's scaling in a mode G2 opened: L = max(g3 L, 3). The max applies for g3 = 1 too, which lifts
// L to 3 where the clamp w / c_lo left it below: a mode at the fine population's scale keeps
// elements of at least 3 samples. Elements without fine are untouched.
template <class T>
void enqueue_mode_g3(cudaStream_t s, const Geom& g, const ElemRecord* le, T* L, const int32_t* on);

// The continuation weight a, pointwise parts. It fades the sift out where the envelope range is
// small against its neighbourhood (no oscillation of this mode to continue):
//   range: R = max(u - l, 0) (u, l: Dil_B r, Ero_B r of sift 1);
//   ramp:  rho = where(U > tol, clip(R / U, 0, 1), 0), g = clip((rho - t0) / max(t1 - t0, 1e-12), 0, 1)
//          (U: the driver's u Dil_B passes of R; tol = le.tol);
//   clip01: a = clip(a, 0, 1) after the s Avg_B passes (in place).
template <class T>
void enqueue_mode_range(cudaStream_t s, const Geom& g, const T* u, const T* l, T* R, const int32_t* on);
template <class T>
void enqueue_mode_ramp(cudaStream_t s, const Geom& g, const MethodCtl& c, const ElemRecord* le, const T* R,
                       const T* U, T* gw, const int32_t* on);
template <class T>
void enqueue_mode_clip01(cudaStream_t s, const Geom& g, T* a, const int32_t* on);

// G1, the amplitude normalization of sifts j >= 1: the mean is taken of h / e, the signal divided
// by its smoothed envelope amplitude e, and scaled back by e, where the envelope's contrast (kap)
// says the amplitude varies; elsewhere the plain mean is kept. Pointwise parts, in this order of
// operations:
//   half_diff:  out = 0.5 (u - l)                 (the e0 numerator, then Avg_B by the driver)
//   floor:      e = max(e, max(anorm_floor e_med[b], tol, tiny)) (in place)
//   ratio:      out = h / e
//   kappa_ramp: out = clip((kap - k0) / max(k1 - k0, 1e-12), 0, 1), kap = (emax - emin) /
//               max(emax + emin, max(tol, tiny)) (out may alias emax or emin)
//   blend:      mh = gk clip(e Mn, l, u) + (1 - gk) plain (mh may alias Mn or plain)
template <class T>
void enqueue_mode_half_diff(cudaStream_t s, const Geom& g, const T* u, const T* l, T* out, const int32_t* on);
template <class T>
void enqueue_mode_floor(cudaStream_t s, const Geom& g, const MethodCtl& c, const ElemRecord* le, const double* e_med,
                        T* e, const int32_t* on);
template <class T>
void enqueue_mode_ratio(cudaStream_t s, const Geom& g, const T* h, const T* e, T* out, const int32_t* on);
template <class T>
void enqueue_mode_kappa_ramp(cudaStream_t s, const Geom& g, const MethodCtl& c, const ElemRecord* le, const T* emax,
                             const T* emin, T* out, const int32_t* on);
template <class T>
void enqueue_mode_blend(cudaStream_t s, const Geom& g, const T* gk, const T* e, const T* Mn, const T* u, const T* l,
                        const T* plain, T* mh, const int32_t* on);

// One sift's update with the guard, for the elements that are active and still sifting:
// mh' = a mh for j >= 1 with a given (else mh), hn = h - mh'; for j >= 1, the maxima max |r - hn|
// and max |hn| into le.gmax_res_bits / gmax_h_bits (integer atomics on the bits of the
// non-negative doubles, whose integer order is their order; zeroed by mode_start and by every
// accept). mh_out (tests, nullptr in the driver) receives mh'. hn may alias h only for j = 0.
// enqueue_mode_accept then copies hn into h where the sift is accepted (j = 0, or both maxima
// <= le.r_inf: a sift must not produce a mode or a remainder larger than the residual it started
// from; a no-op where hn == h) and updates the element: accepted: n_sifts += 1; rejected:
// sifting = 0 (h unchanged: the sifts of the mode end); the maxima are zeroed.
template <class T>
void enqueue_mode_sift(cudaStream_t s, const Geom& g, ElemRecord* le, int32_t j, const T* r, const T* h, const T* mh,
                       const T* a, T* hn, T* mh_out);
template <class T>
void enqueue_mode_accept(cudaStream_t s, const Geom& g, ElemRecord* le, int32_t j, const T* hn, T* h);

// The mode commit of every element that completed the mode (le.status == kRecordActive): the IMF
// row n_imfs of ctl->out <- h (unpadded), r <- r - h, elems.window = le.w; the row partials of the
// IMF (imf_rows) and of the new r (r_rows); enqueue_decide (the history row: scale = w, energy =
// sum IMF^2; n_imfs, amax, tol of the new r; max_imf; the loop condition ctl->any_active); then
// the method's history (hist_spacing = le.spacing, hist_sifts = le.n_sifts), w_prev = w, k += 1,
// and le.status kRecordDone where the field state is no longer active; a non-finite max|r| fails
// the element (kErrNonFiniteResult, as enqueue_mode_guard).
template <class T>
void enqueue_mode_commit(cudaStream_t s, const Geom& g, CallCtl* ctl, ElemState* elems, ElemRecord* le, const T* h,
                         T* r, RowPartial* imf_rows, RowPartial* r_rows);

// ---------------------------------------------------------------------------------------------
// Magnitude and the non-finite guard.
//
// prescale: after the input copy and its partials (elems[b].amax = max|x|, exact), the exact
// power-of-two prescale r <- r 2^-k_b with k_b = ilogb(amax) (0 when amax is 0 or not finite), so
// that max|r| lies in [1, 2); k_b into exp[b]. A product by a power of two is exact (or, below the
// normal range, one correct rounding of the exact value), so the decisions do not depend on the
// field's magnitude and a gain by 2^j is bitwise equivariant over the dtype's range (the squared
// statistics, the threshold's det^2 and the orientation tensor's products, stay inside the
// range). The caller recomputes the partials (amax, r_energy) of the scaled field.
//
// unscale: after the residue row, rows 0..n_imfs of every element that did not fail (status
// kError) and has k_b != 0 <- row 2^k_b (one correct rounding of the exact value); a non-finite
// result (an overflow of the dtype's range) sets elems[b].error = kErrNonFiniteResult (the status
// is left alone: the driver maps the error).
//
// guard: every element still active with le.nonfinite set (the threshold, the orientation) leaves
// the loop with status kRecordError, elems.status kError, elems.error kErrNonFiniteResult. The
// commit applies the same rule to a non-finite max|r| after the mode (elems.amax).
template <class T>
void enqueue_mode_prescale(cudaStream_t s, const Geom& g, const ElemState* elems, int32_t* exp, T* r);
template <class T>
void enqueue_mode_unscale(cudaStream_t s, const Geom& g, ElemState* elems, const int32_t* exp, T* rows,
                          int32_t rows_cap);
void enqueue_mode_guard(cudaStream_t s, int64_t batch, const ModeArrays& A);

// ---------------------------------------------------------------------------------------------
// G2's rung loop on the host (driver.cpp).
//
// g2_candidate_rungs: the candidate rungs s_j = odd(s0 sqrt(2)^j), s0 = max(3, w_prev + 2), while
// s_j jump <= w, consecutive duplicates dropped.
//
// RungMachine: the loop over the candidates with confirmation, as a machine that tests every rung
// once (test(s) depends on s only): next_rung() is the rung to test (rungs[i], or, while a
// candidate of the last rung is pending, its confirmation rung odd(rungs.back() sqrt(2) + 1));
// feed() takes its outcome. Pending: accepted -> G2 accepts the candidate rungs[i - 1]
// (acc_index); stop -> G2 fails; none -> not pending, i += 1. Not pending: stop -> fails; none ->
// i += 1; accepted -> pending (cand = the outcome), i += 1. Done when decided or when the rungs run
// out without a pending candidate. A candidate counts only when the next rung confirms it, so a
// population seen at one scale only is not taken. Every undecided machine has i == the number of
// tests made, so a batch steps in lockstep and step t writes its populations into the P buffer
// t mod 2 (the accepted candidate's buffer is acc_index mod 2).
std::vector<int32_t> g2_candidate_rungs(int32_t w, int32_t w_prev, double jump);
struct RungMachine {
  std::vector<int32_t> rungs;
  int32_t i = 0;
  bool pending = false;
  bool done = false;
  bool accepted = false;
  int32_t acc_index = -1;
  int32_t tests = 0;
  G2Result cand{};
  int32_t next_rung() const;  // 0 when done
  void feed(const G2Result& outcome);
};
RungMachine make_rung_machine(int32_t w, int32_t w_prev, double jump);

}  // namespace eda
