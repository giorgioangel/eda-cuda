// SPDX-License-Identifier: Apache-2.0
// The statistics and fields that decide a mode's scale: exact selections (median, quantile), the
// noise levels, the local threshold field, significant line extrema, the raw local window, the
// scale field with the absence rule (face-connected components), the per-voxel floor, G2's rung
// test and G3's coarse scale. Implementations: stats.cu, stats_fields.cu, stats_scale.cu and
// stats_rungs.cu; the kernels are named eda_stats_*.
//
// Determinism: selections are exact order statistics by radix passes over order keys with integer
// histograms; components are canonical (the minimum padded index of the component); every "any"
// and count is an integer reduction. No result depends on the order of atomics.
#pragma once

#include "field/geom.hpp"
#include "method/method.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace eda {

// ---------------------------------------------------------------------------------------------
// Selections. The value set of element b is one of:
//   kField: field[v] over the non-pad voxels v with (mask == nullptr || (mask[v] & bits) != 0);
//   kDiff:  r[v + e_a] - r[v] (the forward difference along axis a, computed in T) over every
//           axis a with bit a of axes set, pooled (the pooling order does not matter to order
//           statistics);
// optionally replaced by |value - center[b]| computed in double (the MAD's second pass).
// Values are compared as doubles (a float32 value converts exactly).
enum class SelSource : int8_t { kField = 0, kDiff = 1 };
template <class T>
struct Selection {
  SelSource src;
  const T* field;              // kField: the field; kDiff: r
  const uint8_t* mask;         // kField only; nullptr: every voxel
  uint8_t bits;                // mask bits tested
  uint8_t axes;                // kDiff: bit a pools the differences along axis a
  const double* center;        // [B] or nullptr
};
size_t select_workspace_bytes(const Geom& g);

// The median of every element's value set into out[b], as numpy.median (the mean of the two middle
// values for an even count; NaN for an empty set; count[b] the size).
template <class T>
void enqueue_median(cudaStream_t s, const Geom& g, const Selection<T>& sel, double* out, int64_t* count,
                    void* ws, const int32_t* on);
// The quantile q of every element's value set with linear interpolation between order statistics,
// exactly as numpy.quantile(values, q) with method "linear" (its index and interpolation rule; NaN
// for an empty set).
template <class T>
void enqueue_quantile(cudaStream_t s, const Geom& g, const Selection<T>& sel, double q, double* out,
                      int64_t* count, void* ws, const int32_t* on);

// Noise levels: pooled = 1.4826 MAD(diff over every axis) / sqrt 2; per_axis = max over axes with
// n_a >= 2 of the same per axis (0 when none). MAD(d) = median(|d - median(d)|); 1.4826 MAD is the
// standard deviation of Gaussian noise, and the sqrt 2 undoes the variance doubling of a
// difference of two samples. The product, then the quotient, in double.
template <class T>
void enqueue_noise_sigma(cudaStream_t s, const Geom& g, const T* r, bool per_axis, double* out, void* ws,
                         const int32_t* on);

// ---------------------------------------------------------------------------------------------
// Fields

// The local threshold of significant extrema: the detail det = r - cube(r, box, Wd), its local RMS
// level lev = sqrt(max(cube(det^2, box, Wl), 0)), and
//   tau = max(theta max(lev, k_noise sigma_n[b]), tol[b], floor2[b]) (floor2: nullptr or [B], G2's
//   theta_f sigma_x). Wd = odd(rho w), Wl = odd(max(beta, rho) w) per element. tmp: one field.
//   le (nullptr: none): le[b].nonfinite = 1 where cube(det^2) or sigma_n[b] is not finite (the
//   driver's guard fails such an element; the driver prescales r so that this needs an inf / NaN).
template <class T>
void enqueue_tau(cudaStream_t s, const Geom& g, const MethodCtl& c, const T* r, const int32_t* Wd,
                 const int32_t* Wl, const double* sigma_n, const double* tol, const double* floor2, T* tau,
                 T* tmp, const int32_t* on, ElemRecord* le = nullptr);

// Significant line extrema: along every line of axis a, the peaks of r and of -r as
// scipy.signal.find_peaks defines them (a strict rise, a run of exactly equal values, a strict
// fall; never the line's end samples; a plateau's peak at its sample (left + right) // 2), with
// topographic prominence (as scipy.signal.peak_prominences with no window: peak - max(left base,
// right base), the bases the minima up to the first strictly higher sample on each side or the
// line end; a NaN ends a side like a higher sample) >= tau at the peak. Sets bit a of flags
// (cleared first for the elements that run) and adds the count to le[b].ext[a] (integer atomics;
// the caller zeroes it). tmp: one field of T, scratch: overwritten in the elements that run, the
// other elements' part unchanged; neither r nor tau. A launch of an axis covers whole elements, as
// many as fit in launch_ids samples of the axis and at least one (the default is the kernels'
// 32-bit id range; smaller values only split the work, for the tests).
inline constexpr int64_t kSigFlagsLaunchIds = int64_t{1} << 31;
template <class T>
void enqueue_sig_flags(cudaStream_t s, const Geom& g, const T* r, const T* tau, uint8_t* flags, ElemRecord* le,
                       const int32_t* on, T* tmp, int64_t launch_ids = kSigFlagsLaunchIds);

// The raw local window, the local scale implied by the density of extrema: with f_a the 0/1 field
// of bit a of flags, rho_a = cube(f_a, box, win[b]) (the local extrema density along axis a),
// dens = sqrt(sum_a rho_a^2) (in T, sum in axis order), om = 2 kappa / max(dens, tiny) where
// dens > 0, +inf elsewhere (no extrema: no local scale). tmp: one field.
template <class T>
void enqueue_raw_window(cudaStream_t s, const Geom& g, const MethodCtl& c, const uint8_t* flags,
                        const int32_t* win, T* om, T* tmp, const int32_t* on);

// The scale field before the floor. The absence rule: a region without extrema of this mode's
// scale (A0, raw window above reset w) takes the small scale `absent` only where its weak region
// (a component of Aw) reaches fine structure (near); elsewhere the window is reset to w:
//   A0 = om > reset w; Aw = om > lo w; fine = om < lo w / jump; near = cube(fine, max, odd(2 reach W)) > 0.5;
//   A = A0 and (the face-connected component of Aw holding the voxel meets near);
//   om_w = where(om > reset w, w, om); lw = log(where(A, absent, clip(om_w, w / c_lo, c_hi w)));
//   L = exp(cube(lw, box, W)) (the box mean in log: a geometric mean of the local scales).
// w, W and jump from le. Writes A (0/1, bit 0) and L. tmp0, tmp1: fields of T; labels: int32
// field; ccl_ws: ccl_workspace_bytes.
template <class T>
void enqueue_scale_field(cudaStream_t s, const Geom& g, const MethodCtl& c, const ElemRecord* le, const T* om,
                         uint8_t* A, T* L, T* tmp0, T* tmp1, int32_t* labels, void* ccl_ws, const int32_t* on);
size_t ccl_workspace_bytes(const Geom& g);

// The per-voxel floor, for modes k >= 1, so that the scale grows from mode to mode at every voxel:
//   fl = growth L_prev (k >= growth_from) or L_prev + 2; fl = min(fl, c_hi w);
//   chi = cube(A, box, W); fl = exp((1 - chi) log fl + chi log absent); L = max(L, fl)
// (on the absence set the floor blends towards the absence scale, in log).
template <class T>
void enqueue_local_floor(cudaStream_t s, const Geom& g, const MethodCtl& c, const ElemRecord* le,
                         const T* L_prev, const uint8_t* A, T* L, T* tmp, const int32_t* on);

// Face-connected components of {(mask & bits) != 0} (the default structure of
// scipy.ndimage.label; the domain border does not connect): labels[v] = the smallest padded linear
// index of v's component, -1 outside. Exposed for the unit tests; enqueue_scale_field uses it.
void enqueue_ccl(cudaStream_t s, const Geom& g, const uint8_t* mask, uint8_t bits, int32_t* labels,
                 void* ccl_ws, const int32_t* on);

// ---------------------------------------------------------------------------------------------
// G2, the fine-population test, and G3, the coarse scale around an accepted population

// One rung test of every element with srung[b] > 0 (the candidate scale s, odd), on G2's flags ff:
//   Wb = odd(beta_f s); om = raw window of ff at Wb; P = om < lo s (bit 0 of P);
//   none when P is empty, or when cube(P, min, odd(core s)) has no voxel > 0.5;
//   p = quantile(om over P, pop) ; stop when jump p > w;
//   none unless some voxel has om > reset p and cube(P, max, odd(2 reach Wb)) > 0.5;
//   else accepted with (p, cov = mean(P)).
// A population needs a core (it is not a scatter of single voxels) and must border a coarser
// region (a jump in scale). out[b] receives the outcome. tmp0, tmp1: fields of T (om, the
// morphology); ws: the selection.
enum G2Outcome : int32_t { kG2None = 0, kG2Accepted = 1, kG2Stop = 2 };
struct G2Result {
  int32_t outcome;
  int32_t s;
  int64_t n_p;                 // |P|
  double p, cov;
};
template <class T>
void enqueue_g2_test(cudaStream_t s, const Geom& g, const MethodCtl& c, const ElemRecord* le,
                     const int32_t* srung, const uint8_t* ff, uint8_t* P, T* tmp0, T* tmp1, G2Result* out,
                     void* ws);

// G3's coarse scale of every element with le[b].fine: dc = raw window density of ff at odd(beta w)
// (om_c = 2 kappa / dc where dc > 0), outP = cube(P, max, odd(2 reach odd(beta_f p_odd))) < 0.5,
// wc[b] = median(om_c over outP and dc > 0), +inf when that set is empty.
template <class T>
void enqueue_g3_coarse(cudaStream_t s, const Geom& g, const MethodCtl& c, const ElemRecord* le, const uint8_t* ff,
                       const uint8_t* P, double* wc, T* tmp0, T* tmp1, uint8_t* sel, void* ws);

}  // namespace eda
