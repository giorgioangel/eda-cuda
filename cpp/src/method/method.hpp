// SPDX-License-Identifier: Apache-2.0
// The decomposition method, for D = 1, 2, 3 in float32 and float64: the shared types, constants
// and helpers of its three parts.
//
// The method splits a field into modes (IMFs, fine to coarse) and a residue. Each mode of the
// residual r takes these steps (the driver, driver.cpp, runs them for a chunk of elements in
// lockstep):
//   1. the mode window w from the density of significant extrema (a local threshold field, the
//      extrema along every axis, the window rule; mode.hpp, stats.hpp);
//   2. G2, the fine-population test: when a population of finer extrema exists below w, the mode
//      opens at that population's scale instead; G3 then estimates the coarse scale around it;
//   3. the scale field L (a per-voxel element length from the local extrema density, with the
//      absence rule for regions without extrema and a floor from the previous mode);
//   4. the orientation (D >= 2): the dominant direction n and its coherence c of the monogenic
//      signal of r in the band of w (orientation.cu);
//   5. the variable structuring element of every voxel, a cube of side d stretched into a segment
//      of half vector o along n, and its operators Dil_B, Ero_B, Avg_B (element.hpp);
//   6. the sifts: the envelope mean, the continuation weight and, from the second sift on, the
//      amplitude normalization G1, with a sup-norm guard (mode.hpp);
//   7. the commit: the IMF row, r -= IMF, the history and the loop condition.
// Parts: method/element.hpp (orientation, element, operators), method/stats.hpp (selections,
// noise levels, threshold, extrema, scale fields, G2's rung test, G3's median) and method/mode.hpp
// (per-element control, the sift kernels, the commit). Implemented with this header (common.cpp):
// the per-call constants (method_ctl) and the separable cube filter (enqueue_cube).
//
// Layout conventions:
//   - a field is a padded working field of the call's Geom (B elements of g.field samples, the
//     innermost axis padded to g.pitch with zeros); kernels never read or write pad samples except
//     as zeros, and reductions skip them;
//   - a D-component field (n, o) is component-major: component a of element b starts at
//     p + (a * B + b) * g.field (FieldVec holds the D component pointers);
//   - per-element scalars are device arrays [B] (int32_t windows, double statistics); a window 0
//     skips the element in every pass (the cube filter's convention);
//   - `on` [B] (int32, non-zero: run) selects the elements a stage runs on; nullptr: all.
//
// Arithmetic: every floating-point operation that feeds a result is a round-to-nearest intrinsic
// (field/common.cuh) in a fixed order, so that a result depends only on the element's own samples
// (never on the batch, the chunking or the launch shape) and the host twins of the tests reproduce
// it bit for bit. Decisions are integer reductions or exact order statistics.
#pragma once

#include "runtime/context.hpp"
#include "decompose.hpp"
#include "field/geom.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace eda {

// ---------------------------------------------------------------------------------------------
// Options and constants

// Rungs of one element's ladder in one mode: j runs over [floor(-2 log2 w), 2] (d >= L / 4 >=
// 3 / 4 is floored at 1, d <= 2 w), so w = 8191 gives at most 29 values before the deduplication
// of the small odd sides; rung_ladder throws std::length_error beyond. The bit masks of
// ElemRecord (rung_pts, rung_used) hold 32 rungs.
inline constexpr int kMaxRungs = 32;
// G2 candidate rungs of one element and mode (odd(s0 sqrt(2)^j) while 2 s <= w: at most 21 for
// w <= 8191), plus the confirmation rung of the last one.
inline constexpr int kMaxG2Rungs = 32;
// sqrt(2) as a double (the rung ratio of the element's ladder and of G2's candidates).
inline constexpr double kRungRatio = 1.4142135623730951;

// Status of an element in the method's loop (ElemRecord::status). The values match ElemStatus
// where they mean the same.
enum RecordStatus : int32_t { kRecordActive = 0, kRecordDone = 1, kRecordError = 3 };

// Per-call constants, passed by value to kernels. method_ctl builds them after a memset, so that
// the padding bytes are deterministic too; common.cpp documents every value.
struct MethodCtl {
  int32_t n_max;                // max n_a
  int64_t count;                // samples per element (unpadded)
  int32_t sifts, u, s, sig_passes, growth_from, anorm_smooth;
  double kappa, beta, c_lo, c_hi, reset, q, span, growth;
  double sig_theta, sig_rho, sig_noise;
  double absent, absent_jump, absent_reach, absent_lo;
  double t0, t1;
  double anorm_floor, ramp_k0, ramp_k1;
  double fine_beta, fine_theta, fine_jump, fine_lo, fine_pop, fine_core, split;
  double tiny;                  // the floor of divisions in T: 1e-300 (float64), FLT_MIN (float32)
};
MethodCtl method_ctl(const Geom& g, DType dtype);

// Per-element state of the method (device, one per batch element). Each field is written by the
// stage named in its comment and read by the later ones. The field kernels' ElemState (amax, tol,
// r_energy, n_imfs, error) is kept alongside for the input copy, the row partials and the output
// rows.
struct alignas(16) ElemRecord {
  int32_t status;               // RecordStatus (mode control)
  int32_t k;                    // 0-based index of the current mode
  int32_t w_prev;               // previous mode's window, 1 before the first
  int32_t w;                    // the mode window after the window passes, G2 and the growth floor
  int32_t w_pass[2];            // the window after pass 1 and pass 2 (0: the element stopped); read by the kernel tests
  int32_t W;                    // odd(beta w): the box of the scale field and of G1
  int32_t fine;                 // 1: G2 opened this mode
  uint64_t ext[kMaxDim];        // significant extrema per axis of the last flags pass (integer adds)
  double tol;                   // tie tolerance of r: fl_T(tau max|r|)
  double r_inf;                 // max |r| at the start of the mode (the sift guard's bound)
  double sigma_n;               // noise level of r, pooled over the axes; read by the kernel tests
  double sigma_x;               // noise level of the input, the largest per axis, fixed per call; read by the kernel tests
  double jump;                  // the absence rule's jump: absent_jump, 1 in a mode opened by G2
  double spacing;               // N / sqrt(sum ext^2) of the final window pass
  // G2 / G3 (the mode control, from the statistics' outputs)
  int32_t p_odd;                // odd(p) of the accepted rung
  int32_t g2_rung;              // the accepted candidate rung s (0: none); read by the kernel tests
  double p, cov, wc;            // population scale, coverage, coarse scale (inf: none); read by the kernel tests
  double g3;                    // the G3 factor (1: none)
  // the element (element.hpp)
  int32_t n_rungs;              // rungs of the ladder (1..kMaxRungs; uploaded by the host from rung_ladder)
  uint32_t rung_pts;            // bit j: rung j carries the segment points (some voxel with a non-zero
                                // weight on it has e > 1/4) (enqueue_element_weights)
  uint32_t rung_used;           // bit j: some voxel has a non-zero weight on rung j (same pass)
  int32_t rung_w[kMaxRungs];    // odd sides, ascending (uploaded by the host from rung_ladder)
  double rung_lw[kMaxRungs];    // log(rung_w) with the device log (enqueue_element_weights). It is
                                // the same function as the voxels' log d, so a d clipped to a rung
                                // has t exactly 0 or 1; the host never writes it
  uint64_t d_min_bits, d_max_bits;  // bits of the positive doubles min d, max d (integer atomics)
  // the sifts
  int32_t sifting;              // 1 until the guard rejects a sift
  int32_t n_sifts;              // completed sifts of the mode
  uint64_t gmax_res_bits;       // guard: bits of max |r - hn| (non-negative doubles order as integers)
  uint64_t gmax_h_bits;         // guard: bits of max |hn|
  int32_t nonfinite;            // 1: a stage met a non-finite intermediate (the threshold's box of
                                // det^2, the orientation tensor); the next enqueue_mode_guard fails
                                // the element with kErrNonFiniteResult
};

// A D-component field: p[a] is component a (nullptr for a >= D).
template <class T>
struct FieldVec {
  T* p[kMaxDim];
};

// ---------------------------------------------------------------------------------------------
// Shared helpers (host and device inline)

// The odd window of a scale v: 2 floor(max(v, 1) / 2) + 1, so every window is odd and at least 1.
__host__ __device__ inline int32_t odd_window(double v) {
  const double u = v > 1.0 ? v : 1.0;
  return 2 * static_cast<int32_t>(floor(u / 2.0)) + 1;
}
// The median of a set whose two middle order statistics are a <= b (b = a for an odd count): their
// mean (a + b) / 2 in double, the sum first, as numpy.median computes it.
__host__ __device__ inline double median_pair(double a, double b) { return (a + b) / 2.0; }

// The separable cube filter: along every axis a, the 1D filter of side 2 min((w - 1) / 2, n_a - 1)
// + 1 with half-sample reflection, by the vHGW passes (field/vhgw.hpp: exact max/min, canonical box
// sums times fl_T(1 / prod of the clamped sides)). The box sums run axis 0 first, in a fixed order
// because sums round; max and min are exact, so their axis order does not matter. win [B] holds one
// odd side per element (0: the element is skipped and its output untouched). in may equal out0
// (in place). kMaxMin writes the max into out0 and the min into out1 (both required). The box mean
// differs from a running-sum filter (scipy.ndimage.uniform_filter1d) by rounding only.
enum class CubeOp : int8_t { kMax = 0, kMin = 1, kMaxMin = 2, kBox = 3 };
template <class T>
void enqueue_cube(cudaStream_t s, const Geom& g, CubeOp op, const T* in, T* out0, T* out1, const int32_t* win,
                  int32_t lines_per_cta = 0);
// The cubes of `rungs` windows of one input in one launch per pass (field/vhgw.hpp): rung j filters
// in with the windows win + j B into out0 + j stride (and out1 + j stride); each rung's outputs are
// bitwise those of enqueue_cube with its windows. With rungs > 1 the rung blocks (B field elements
// each, stride >= B field apart) must not overlap each other, the other output's or in.
template <class T>
void enqueue_cube_rungs(cudaStream_t s, const Geom& g, CubeOp op, const T* in, T* out0, T* out1, const int32_t* win,
                        int32_t rungs, int64_t stride, int32_t lines_per_cta = 0);

// Whether the element ranges [p, p + n) and [q, q + m) share no element (nullptr: no range).
template <class T>
bool ranges_apart(const T* p, int64_t n, const T* q, int64_t m) noexcept {
  const uintptr_t a = reinterpret_cast<uintptr_t>(p), b = reinterpret_cast<uintptr_t>(q);
  return p == nullptr || q == nullptr || a + static_cast<uintptr_t>(n) * sizeof(T) <= b ||
         b + static_cast<uintptr_t>(m) * sizeof(T) <= a;
}

// ---------------------------------------------------------------------------------------------
// The driver (driver.cpp) and its validation (decompose.cpp dispatches here).
// validate_method raises std::invalid_argument (ValueError) for, among others, a spatial axis of
// extent 1 (extent_message): such an axis has no sample differences, so no noise level.
void validate_method(const Input& in, const Params& p);
std::string extent_message(int dim, int axis);
std::unique_ptr<Run> run_method(Context& ctx, const Input& in, const Params& p);
// Shared with eda.monogenic: the C-contiguous strides of a chunk's staged input, the binding of a
// chunk's workspace (OutOfDeviceMemory when it does not fit) and the geometry of a chunk of
// `batch` elements.
InputStrides c_strides(const Geom& g);
ArenaBinding bind_arena(Context& ctx, const ArenaLayout& lay);
Geom chunk_geom(const Geom& g, int64_t batch);

// ---------------------------------------------------------------------------------------------
// eda.monogenic (monogenic.cpp): the monogenic signal of every element of `in` (method/element.hpp,
// enqueue_monogenic), chunked by the free device memory, into out: device memory of
// monogenic_bytes(in) bytes, written on ctx.stream(), the planes amplitude, phase, frequency
// (B, count), then for D >= 2 direction (B, D, count) and coherence (B, count), in the compute
// dtype. With check_finite, nonfinite[b] = 1 for every element that holds inf or NaN (0 else; all
// 0 without it); such an element's outputs are not defined. max_chunk > 0 caps the elements per
// chunk (tests; 0: by free memory); the return value is the elements per chunk of the call.
// validate_monogenic takes the decomposition's rules for the input (validate_method).
void validate_monogenic(const Input& in);
size_t monogenic_bytes(const Input& in);
int64_t run_monogenic(Context& ctx, const Input& in, bool check_finite, int64_t max_chunk, void* out,
                      std::vector<int32_t>& nonfinite);

}  // namespace eda
