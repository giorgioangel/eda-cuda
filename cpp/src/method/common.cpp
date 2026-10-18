// SPDX-License-Identifier: Apache-2.0
// The shared pieces of method/method.hpp: the per-call constants and the separable cube filter over
// the vHGW passes.
#include "method/method.hpp"
#include "field/vhgw.hpp"

#include <cfloat>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace eda {

MethodCtl method_ctl(const Geom& g, DType dtype) {
  MethodCtl c;
  std::memset(&c, 0, sizeof c);  // deterministic padding bytes: the struct is copied bytewise into kernels
  c.n_max = 1;
  for (int a = 0; a < g.dim; ++a) c.n_max = g.n[a] > c.n_max ? g.n[a] : c.n_max;
  c.count = g.count;
  // the mode window and the scale field
  c.sifts = 3;            // sifts per mode (the call's Params::sifts replaces it)
  c.kappa = 1.0;          // window rule: w = odd(2 kappa N / sqrt(sum ext^2)), raw window 2 kappa / density
  c.beta = 2.0;           // scale-field box W = odd(beta w)
  c.c_lo = 1.5;           // the scale field is clamped to [w / c_lo, c_hi w]
  c.c_hi = 2.0;
  c.reset = 2.0;          // a raw window above reset w counts as no local scale (absence candidate)
  c.q = 4.0;              // element aspect: d = L ((1 - c) + c / q)
  c.span = 1.0;           // segment length factor: L (1 + (span - 1) c)
  c.growth = 2.0;         // from mode growth_from on, w >= odd(growth w_prev) and L >= growth L_prev
  c.growth_from = 2;      // 0-based mode index of the first floored mode
  // the threshold of significant extrema
  c.sig_theta = 0.3;      // relative significance: tau = theta max(level, k_noise sigma_n)
  c.sig_rho = 2.0;        // detail box odd(rho w)
  c.sig_noise = 1.0;      // k_noise
  c.sig_passes = 2;       // window passes per mode (the second measures with the first's window)
  // the absence rule
  c.absent = 3.0;         // the scale on the absence set
  c.absent_jump = 2.0;    // fine regions: raw window < lo w / jump (jump 1 in a mode opened by G2)
  c.absent_reach = 1.0;   // reach of the fine regions: cube odd(2 reach W)
  c.absent_lo = 1.2;      // weak set: raw window > lo w (its components carry the absence)
  // the continuation weight
  c.t0 = 0.2;             // ramp of the range ratio rho from t0 to t1
  c.t1 = 0.5;
  c.u = 2;                // Dil_B passes of the range
  c.s = 1;                // Avg_B passes of the weight
  // G1, the amplitude normalization (sifts 2..S)
  c.anorm_floor = 0.1;    // e >= anorm_floor median(e)
  c.anorm_smooth = 1;     // extra Box_W passes on e
  c.ramp_k0 = 0.3;        // ramp of the envelope contrast from k0 to k1
  c.ramp_k1 = 0.6;
  // G2, the fine-population test, and G3, the coarse scale
  c.fine_beta = 2.0;      // measuring box odd(beta_f s)
  c.fine_theta = 1.0;     // G2's extrema threshold floor theta_f sigma_x
  c.fine_jump = 2.0;      // G2 runs when jump s0 <= w and stops when jump p > w
  c.fine_lo = 1.5;        // presence: raw window < lo s
  c.fine_pop = 0.25;      // population scale: this quantile of the raw window over the population
  c.fine_core = 2.0;      // the population eroded by the cube odd(core s) must not be empty
  c.split = 1.0;          // G3 factor min(1, split sqrt(w_c / p) / 2)
  // The floor of divisions and thresholds: 1e-300 underflows in float32, so float32 uses its
  // smallest positive normal value.
  c.tiny = dtype == DType::F32 ? static_cast<double>(FLT_MIN) : 1e-300;
  return c;
}

template <class T>
void enqueue_cube_rungs(cudaStream_t s, const Geom& g, CubeOp op, const T* in, T* out0, T* out1, const int32_t* win,
                        int32_t rungs, int64_t stride, int32_t lines_per_cta) {
  if (in == nullptr || out0 == nullptr || win == nullptr) throw std::invalid_argument("eda: cube: null field");
  if (op == CubeOp::kMaxMin && (out1 == nullptr || out1 == out0 || out1 == in)) {
    throw std::invalid_argument("eda: cube: kMaxMin needs a distinct second output");
  }
  if (rungs < 1) throw std::invalid_argument("eda: cube: rungs must be >= 1");
  if (rungs > 1) {
    // the rung blocks [out + j stride, + B field) are disjoint from each other, from out1's and from in
    const int64_t field = g.batch * g.field;
    const int64_t span = (rungs - 1) * stride + field;
    bool ok = stride >= field && ranges_apart(in, field, out0, span);
    if (op == CubeOp::kMaxMin) ok = ok && ranges_apart(in, field, out1, span) && ranges_apart(out0, span, out1, span);
    if (!ok) throw std::invalid_argument("eda: cube: the rung outputs must not overlap each other or the input");
  }
  const int D = g.dim;
  VhgwArgs<T> base{};
  base.window = win;  // a window < 1 (0) leaves the element's lines untouched (eda_vhgw_pass)
  base.lines_per_cta = lines_per_cta;
  base.rungs = rungs;
  base.out_stride = stride;
  if (op == CubeOp::kBox) {
    // axis 0 first (sums round, so the axis order is fixed); the last pass scales by
    // fl_T(1 / prod of the clamped sides)
    for (int a = 0; a < D; ++a) {
      VhgwPass p{};
      p.axis = a;
      p.from_single = true;
      p.box = true;
      p.op = 2;
      p.epilogue = a == D - 1 ? kEpiBoxScale : kEpiStore;
      VhgwArgs<T> args = base;
      args.in0 = a == 0 ? in : out0;
      args.in_stride = a == 0 ? 0 : stride;
      args.out0 = out0;
      enqueue_vhgw<T>(s, g, p, args);
    }
    return;
  }
  // max / min are exact order statistics: any axis order gives the same bits; the contiguous axis
  // first. kMaxMin stages the input once and then carries the max and the min fields separately.
  for (int a = D - 1; a >= 0; --a) {
    VhgwPass p{};
    p.axis = a;
    p.maxmin = true;
    p.epilogue = kEpiStore;
    VhgwArgs<T> args = base;
    const bool first = a == D - 1;
    args.in_stride = first ? 0 : stride;
    if (op == CubeOp::kMaxMin) {
      p.from_single = first;
      p.op = -1;
      args.in0 = first ? in : out0;
      args.in1 = first ? nullptr : out1;
      args.out0 = out0;
      args.out1 = out1;
    } else {
      p.from_single = true;
      p.op = op == CubeOp::kMax ? 0 : 1;
      args.in0 = first ? in : out0;
      args.out0 = out0;
    }
    enqueue_vhgw<T>(s, g, p, args);
  }
}

template <class T>
void enqueue_cube(cudaStream_t s, const Geom& g, CubeOp op, const T* in, T* out0, T* out1, const int32_t* win,
                  int32_t lines_per_cta) {
  enqueue_cube_rungs<T>(s, g, op, in, out0, out1, win, 1, 0, lines_per_cta);
}

template void enqueue_cube<float>(cudaStream_t, const Geom&, CubeOp, const float*, float*, float*, const int32_t*,
                                  int32_t);
template void enqueue_cube<double>(cudaStream_t, const Geom&, CubeOp, const double*, double*, double*,
                                   const int32_t*, int32_t);
template void enqueue_cube_rungs<float>(cudaStream_t, const Geom&, CubeOp, const float*, float*, float*,
                                        const int32_t*, int32_t, int64_t, int32_t);
template void enqueue_cube_rungs<double>(cudaStream_t, const Geom&, CubeOp, const double*, double*, double*,
                                         const int32_t*, int32_t, int64_t, int32_t);

}  // namespace eda
