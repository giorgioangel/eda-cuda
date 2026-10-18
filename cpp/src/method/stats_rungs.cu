// SPDX-License-Identifier: Apache-2.0
// G2's rung test (the fine-population test of one candidate scale, with the population's own
// core) and G3's coarse scale (the median raw window outside the fine population's reach) of
// method/stats.hpp. Every decision is an integer any-reduction or an exact selection, so it does
// not depend on the order of atomics; nothing leaves the device.
#include "method/stats_common.cuh"

#include <stdexcept>

namespace eda {
namespace stats_detail {
namespace {

// Per element: the windows Wb = odd(beta_f s), Wcore = odd(core s), Rq = odd(2 reach Wb), the
// any-flags cleared; 0 windows for the elements that do not run (srung 0).
__global__ void eda_stats_rungs_g2_prep(int64_t B, MethodCtl c, const int32_t* srung, int32_t* win, int32_t* any) {
  const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= B) return;
  const int32_t s = srung[b];
  int32_t wb = 0, wc = 0, wr = 0;
  if (s > 0) {
    const double sd = static_cast<double>(s);
    wb = odd_window(__dmul_rn(c.fine_beta, sd));
    wc = odd_window(__dmul_rn(c.fine_core, sd));
    wr = odd_window(__dmul_rn(__dmul_rn(2.0, c.absent_reach), static_cast<double>(wb)));
  }
  win[b] = wb;
  win[B + b] = wc;
  win[2 * B + b] = wr;
  any[b] = 0;
  any[B + b] = 0;
}

// P = om < lo s (bit 0) and tmp1 = float(P).
template <class T>
__global__ void eda_stats_rungs_g2_presence(Geom g, double lo, const int32_t* srung, const T* om, uint8_t* P, T* tmp1) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= g.batch * g.field) return;
  const int64_t b = i / g.field;
  const int32_t s = srung[b];
  if (s <= 0) return;
  const bool in = !pad_at(g, i % g.field) && static_cast<double>(om[i]) < __dmul_rn(lo, static_cast<double>(s));
  P[i] = in ? 1 : 0;
  tmp1[i] = in ? T(1) : T(0);
}

// The core: any eroded value > 0.5 (any[b]); tmp1 = float(P) again for the dilation.
template <class T>
__global__ void eda_stats_rungs_g2_core(Geom g, const int32_t* srung, const uint8_t* P, T* tmp1, int32_t* any) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= g.batch * g.field) return;
  const int64_t b = i / g.field;
  if (srung[b] <= 0) return;
  const bool pad = pad_at(g, i % g.field);
  if (!pad && tmp1[i] > T(0.5)) atomicOr(any + b, 1);
  tmp1[i] = (!pad && (P[i] & 1u)) ? T(1) : T(0);
}

// The jump: some voxel with om > reset p and Dil_Rq P > 0.5 (any[B + b]).
template <class T>
__global__ void eda_stats_rungs_g2_jump(Geom g, double reset, const int32_t* srung, const T* om, const T* dil,
                                        const double* p, int32_t* any) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= g.batch * g.field) return;
  const int64_t b = i / g.field;
  if (srung[b] <= 0 || pad_at(g, i % g.field)) return;
  if (static_cast<double>(om[i]) > __dmul_rn(reset, p[b]) && dil[i] > T(0.5)) atomicOr(any + g.batch + b, 1);
}

__global__ void eda_stats_rungs_g2_final(int64_t B, MethodCtl c, const ElemRecord* le, const int32_t* srung,
                                         const double* p, const int64_t* n_p, const int32_t* any, G2Result* out) {
  const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= B || srung[b] <= 0) return;
  G2Result r{};
  r.s = srung[b];
  r.n_p = n_p[b];
  r.p = p[b];
  r.cov = __ddiv_rn(static_cast<double>(n_p[b]), static_cast<double>(c.count));  // mean(P): the covered share
  if (r.n_p == 0 || !any[b]) {
    r.outcome = kG2None;  // no presence; no core
  } else if (__dmul_rn(c.fine_jump, r.p) > static_cast<double>(le[b].w)) {
    r.outcome = kG2Stop;  // the population's own scale is too coarse for a jump below w
  } else {
    r.outcome = any[B + b] ? kG2Accepted : kG2None;
  }
  out[b] = r;
}

// ---- G3 ---------------------------------------------------------------------------------------------
__global__ void eda_stats_rungs_g3_prep(int64_t B, MethodCtl c, const ElemRecord* le, int32_t* on, int32_t* win) {
  const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= B) return;
  const bool run = le[b].fine != 0 && le[b].status == kRecordActive;
  on[b] = run ? 1 : 0;
  win[b] = run ? odd_window(__dmul_rn(c.beta, static_cast<double>(le[b].w))) : 0;  // Wc = odd(beta w)
  const int32_t wf = odd_window(__dmul_rn(c.fine_beta, static_cast<double>(le[b].p_odd)));
  win[B + b] = run ? odd_window(__dmul_rn(__dmul_rn(2.0, c.absent_reach), static_cast<double>(wf))) : 0;
}

template <class T>
__global__ void eda_stats_rungs_g3_expand(Geom g, const int32_t* on, const uint8_t* P, T* tmp1) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= g.batch * g.field) return;
  if (!on[i / g.field]) return;
  tmp1[i] = (!pad_at(g, i % g.field) && (P[i] & 1u)) ? T(1) : T(0);
}

template <class T>
__global__ void eda_stats_rungs_g3_select(Geom g, const int32_t* on, const T* omc, const T* dil, uint8_t* sel) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= g.batch * g.field) return;
  if (!on[i / g.field]) return;
  // outP = Dil P < 0.5 and dc > 0 (om_c finite exactly where dc > 0)
  sel[i] = (!pad_at(g, i % g.field) && dil[i] < T(0.5) && omc[i] < inf_t<T>()) ? 1 : 0;
}

__global__ void eda_stats_rungs_g3_final(int64_t B, const int32_t* on, const int64_t* cnt, double* wc) {
  const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= B || !on[b]) return;
  if (cnt[b] == 0) wc[b] = __longlong_as_double(0x7ff0000000000000LL);  // +inf: no estimate
}

}  // namespace
}  // namespace stats_detail

template <class T>
void enqueue_g2_test(cudaStream_t s, const Geom& g, const MethodCtl& c, const ElemRecord* le,
                     const int32_t* srung, const uint8_t* ff, uint8_t* P, T* tmp0, T* tmp1, G2Result* out,
                     void* ws) {
  using namespace stats_detail;
  if (!le || !srung || !ff || !P || !tmp0 || !tmp1 || !out || !ws || tmp0 == tmp1) {
    throw std::invalid_argument("eda: g2_test: bad pointers");
  }
  const int64_t B = g.batch;
  const SelExtra E = sel_extra(ws, B);
  const unsigned nb = blocks_for(B * g.field);
  int32_t* w_b = E.win;
  int32_t* w_core = E.win + B;
  int32_t* w_rq = E.win + 2 * B;
  eda_stats_rungs_g2_prep<<<blocks_for(B), kThreads, 0, s>>>(B, c, srung, E.win, E.any);
  EDA_CHECK_LAUNCH("eda_stats_rungs_g2_prep");
  enqueue_raw_window<T>(s, g, c, ff, w_b, tmp0, tmp1, srung);  // om at Wb
  eda_stats_rungs_g2_presence<T><<<nb, kThreads, 0, s>>>(g, c.fine_lo, srung, tmp0, P, tmp1);
  EDA_CHECK_LAUNCH("eda_stats_rungs_g2_presence");
  enqueue_cube<T>(s, g, CubeOp::kMin, tmp1, tmp1, nullptr, w_core);
  eda_stats_rungs_g2_core<T><<<nb, kThreads, 0, s>>>(g, srung, P, tmp1, E.any);
  EDA_CHECK_LAUNCH("eda_stats_rungs_g2_core");
  Selection<T> sel{};
  sel.src = SelSource::kField;
  sel.field = tmp0;
  sel.mask = P;
  sel.bits = 1;
  enqueue_quantile<T>(s, g, sel, c.fine_pop, E.p, E.n_p, ws, srung);
  enqueue_cube<T>(s, g, CubeOp::kMax, tmp1, tmp1, nullptr, w_rq);
  eda_stats_rungs_g2_jump<T><<<nb, kThreads, 0, s>>>(g, c.reset, srung, tmp0, tmp1, E.p, E.any);
  EDA_CHECK_LAUNCH("eda_stats_rungs_g2_jump");
  eda_stats_rungs_g2_final<<<blocks_for(B), kThreads, 0, s>>>(B, c, le, srung, E.p, E.n_p, E.any, out);
  EDA_CHECK_LAUNCH("eda_stats_rungs_g2_final");
}

template <class T>
void enqueue_g3_coarse(cudaStream_t s, const Geom& g, const MethodCtl& c, const ElemRecord* le, const uint8_t* ff,
                       const uint8_t* P, double* wc, T* tmp0, T* tmp1, uint8_t* sel, void* ws) {
  using namespace stats_detail;
  if (!le || !ff || !P || !wc || !tmp0 || !tmp1 || !sel || !ws || tmp0 == tmp1) {
    throw std::invalid_argument("eda: g3_coarse: bad pointers");
  }
  const int64_t B = g.batch;
  const SelExtra E = sel_extra(ws, B);
  const unsigned nb = blocks_for(B * g.field);
  eda_stats_rungs_g3_prep<<<blocks_for(B), kThreads, 0, s>>>(B, c, le, E.on, E.win);
  EDA_CHECK_LAUNCH("eda_stats_rungs_g3_prep");
  enqueue_raw_window<T>(s, g, c, ff, E.win, tmp0, tmp1, E.on);  // om_c = 2 kappa / dc at Wc
  eda_stats_rungs_g3_expand<T><<<nb, kThreads, 0, s>>>(g, E.on, P, tmp1);
  EDA_CHECK_LAUNCH("eda_stats_rungs_g3_expand");
  enqueue_cube<T>(s, g, CubeOp::kMax, tmp1, tmp1, nullptr, E.win + B);
  eda_stats_rungs_g3_select<T><<<nb, kThreads, 0, s>>>(g, E.on, tmp0, tmp1, sel);
  EDA_CHECK_LAUNCH("eda_stats_rungs_g3_select");
  Selection<T> ss{};
  ss.src = SelSource::kField;
  ss.field = tmp0;
  ss.mask = sel;
  ss.bits = 1;
  enqueue_median<T>(s, g, ss, wc, E.n_p, ws, E.on);
  eda_stats_rungs_g3_final<<<blocks_for(B), kThreads, 0, s>>>(B, E.on, E.n_p, wc);
  EDA_CHECK_LAUNCH("eda_stats_rungs_g3_final");
}

#define EDA_STATS_RUNGS_INST(T)                                                                                    \
  template void enqueue_g2_test<T>(cudaStream_t, const Geom&, const MethodCtl&, const ElemRecord*,              \
                                        const int32_t*, const uint8_t*, uint8_t*, T*, T*, G2Result*, void*);   \
  template void enqueue_g3_coarse<T>(cudaStream_t, const Geom&, const MethodCtl&, const ElemRecord*,            \
                                          const uint8_t*, const uint8_t*, double*, T*, T*, uint8_t*, void*);
EDA_STATS_RUNGS_INST(float)
EDA_STATS_RUNGS_INST(double)
#undef EDA_STATS_RUNGS_INST

}  // namespace eda
