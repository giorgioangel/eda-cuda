// SPDX-License-Identifier: Apache-2.0
// The mode's control kernels (init, mode start, the window passes, the window with G2's result,
// the growth floor and G3's factor, G2's accept, G3's scaling), the magnitude prescale and the
// mode commit (method/mode.hpp). The control kernels run one thread per element.
#include "runtime/error.hpp"
#include "field/common.cuh"
#include "field/field_ops.hpp"
#include "method/method.hpp"
#include "method/mode.hpp"
#include "field/reduce.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace eda {
namespace {

constexpr int kThreads = 256;

unsigned grid_for(int64_t work, int per_cta) {
  const int64_t g = (work + per_cta - 1) / per_cta;
  return static_cast<unsigned>(std::clamp<int64_t>(g, 1, int64_t{1} << 20));
}

__device__ __forceinline__ double d_inf() { return __longlong_as_double(0x7ff0000000000000LL); }
__device__ __forceinline__ double dmax(double a, double b) { return a >= b ? a : b; }

// The window slots of element b.
__device__ __forceinline__ void set_windows(int32_t* win, int64_t B, int64_t b, int32_t wd, int32_t wl, int32_t W) {
  win[kWinWd * B + b] = wd;
  win[kWinWl * B + b] = wl;
  win[kWinW * B + b] = W;
}

// The element leaves the method's loop (no IMF this mode).
__device__ __forceinline__ void finish(const ModeArrays& A, int64_t B, int64_t b, int32_t status) {
  A.le[b].status = status;
  if (status == kRecordDone && A.elems[b].status == kActive) A.elems[b].status = kDone;
  A.on[b] = 0;
  A.on_g2[b] = 0;
  A.on_floor[b] = 0;
  set_windows(A.win, B, b, 0, 0, 0);
}

// The element fails on a non-finite intermediate (enqueue_mode_guard, the commit).
__device__ __forceinline__ void fail_nonfinite(const ModeArrays& A, int64_t B, int64_t b) {
  finish(A, B, b, kRecordError);
  A.elems[b].status = kError;
  A.elems[b].error = kErrNonFiniteResult;
}

// The prescale exponent of an element: max|x| in [2^k, 2^(k + 1)); 0 for 0 or a non-finite max.
__device__ __forceinline__ int32_t scale_exp(double amax) {
  return (amax > 0.0 && isfinite(amax)) ? static_cast<int32_t>(ilogb(amax)) : 0;
}

// v 2^k: exact, or one correct rounding where the result leaves the normal range (float: the
// product is exact in double, the conversion rounds once; double: ldexp is exact over its range).
__device__ __forceinline__ float times_pow2(float v, int32_t k) {
  return static_cast<float>(ldexp(static_cast<double>(v), k));
}
__device__ __forceinline__ double times_pow2(double v, int32_t k) { return ldexp(v, k); }

// odd(rho w) and odd(max(beta, rho) w) of a threshold window w: the detail box W_d and the level
// box W_l of the threshold field.
__device__ __forceinline__ void tau_windows(const MethodCtl& c, int32_t w, int32_t* wd, int32_t* wl) {
  const double wv = static_cast<double>(w);
  *wd = odd_window(__dmul_rn(c.sig_rho, wv));
  *wl = odd_window(__dmul_rn(dmax(c.beta, c.sig_rho), wv));
}

// ---------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(kThreads) eda_mode_init(int64_t B, ModeArrays A) {
  const int64_t b = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (b >= B) return;
  ElemRecord e;
  memset(&e, 0, sizeof e);
  e.w_prev = 1;
  e.k = 0;
  e.jump = 1.0;
  e.g3 = 1.0;
  e.wc = d_inf();
  const bool ok = A.elems[b].status == kActive;
  e.status = ok ? kRecordActive : kRecordError;
  A.le[b] = e;
  A.on[b] = ok ? 1 : 0;
  A.on_g2[b] = 0;
  A.on_floor[b] = 0;
  A.wc[b] = d_inf();
  set_windows(A.win, B, b, 0, 0, 0);
}

__global__ void __launch_bounds__(kThreads) eda_mode_start(int64_t B, MethodCtl c, ModeArrays A) {
  const int64_t b = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (b >= B) return;
  ElemRecord& e = A.le[b];
  if (e.status == kRecordActive && A.elems[b].status != kActive) {
    e.status = A.elems[b].status == kError ? kRecordError : kRecordDone;
  }
  if (e.status != kRecordActive) {
    finish(A, B, b, e.status);
    return;
  }
  const ElemState& ne = A.elems[b];
  e.tol = ne.tol;
  e.r_inf = ne.amax;
  A.tol[b] = ne.tol;
  e.w_pass[0] = e.w_pass[1] = 0;
  e.W = 0;
  for (int a = 0; a < kMaxDim; ++a) e.ext[a] = 0;
  e.fine = 0;
  e.jump = c.absent_jump;
  e.p_odd = 0;
  e.g2_rung = 0;
  e.p = 0.0;
  e.cov = 0.0;
  e.wc = d_inf();
  A.wc[b] = d_inf();
  e.g3 = 1.0;
  e.sifting = 1;
  e.n_sifts = 0;
  e.gmax_res_bits = 0;
  e.gmax_h_bits = 0;
  const double sx = A.sigma_x[b];
  e.sigma_x = sx;
  A.floor2[b] = __dmul_rn(c.fine_theta, sx);
  const int32_t w0 = max(3, e.w_prev + 2);
  int32_t wd, wl;
  tau_windows(c, w0, &wd, &wl);
  set_windows(A.win, B, b, wd, wl, 0);
  A.on[b] = 1;
  A.on_g2[b] = 0;
  A.on_floor[b] = e.k >= 1 ? 1 : 0;
}

__global__ void __launch_bounds__(kThreads) eda_mode_rice(int64_t B, MethodCtl c, Geom g, int32_t pass,
                                                          ModeArrays A) {
  const int64_t b = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (b >= B) return;
  ElemRecord& e = A.le[b];
  if (e.status != kRecordActive) return;
  if (pass == 0) e.sigma_n = A.sigma_n[b];
  // Enough extrema: some axis has at least 3 per line on average (ext_a n_a >= 3 N, in exact
  // integers); then the spacing N / sqrt(sum ext^2), the rule and its floors.
  const uint64_t N = static_cast<uint64_t>(g.count);
  uint64_t S2 = 0;
  bool enough = false;
  for (int a = 0; a < g.dim; ++a) {
    const uint64_t ea = e.ext[a];
    S2 += ea * ea;
    if (ea * static_cast<uint64_t>(g.n[a]) >= 3 * N) enough = true;
  }
  for (int a = 0; a < kMaxDim; ++a) e.ext[a] = 0;
  if (!enough) {
    e.w_pass[pass] = 0;
    finish(A, B, b, kRecordDone);
    return;
  }
  const double sp = __ddiv_rn(static_cast<double>(N), __dsqrt_rn(static_cast<double>(S2)));
  const double v = __dmul_rn(__dmul_rn(2.0, c.kappa), sp);
  // w > max n_a <= 8192 for any v of 2^30 or more (and int32 holds the floor below)
  if (!(v < 1073741824.0)) {
    e.w_pass[pass] = 0;
    finish(A, B, b, kRecordDone);
    return;
  }
  int32_t w = 2 * static_cast<int32_t>(floor(dmax(v, 1.0) / 2.0)) + 1;
  w = max(w, 3);
  w = max(w, e.w_prev + 2);
  if (w > c.n_max) {
    e.w_pass[pass] = 0;
    finish(A, B, b, kRecordDone);
    return;
  }
  e.w_pass[pass] = w;
  e.spacing = sp;
  int32_t wd, wl;
  tau_windows(c, w, &wd, &wl);
  set_windows(A.win, B, b, wd, wl, 0);
  if (pass == c.sig_passes - 1) {
    e.w = w;
    const int32_t s0 = max(3, e.w_prev + 2);
    A.on_g2[b] = (__dmul_rn(c.fine_jump, static_cast<double>(s0)) <= static_cast<double>(w)) ? 1 : 0;
  }
}

__global__ void __launch_bounds__(kThreads) eda_mode_window(int64_t B, MethodCtl c, ModeArrays A) {
  const int64_t b = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (b >= B) return;
  ElemRecord& e = A.le[b];
  if (e.status != kRecordActive) return;
  int32_t w = e.w;
  if (e.fine) {
    w = max(max(e.p_odd, e.w_prev + 2), 3);
    e.jump = 1.0;
  }
  if (c.growth > 1.0 && e.k >= c.growth_from) {
    w = max(w, odd_window(__dmul_rn(c.growth, static_cast<double>(e.w_prev))));
    if (w > c.n_max) {
      finish(A, B, b, kRecordDone);
      return;
    }
  }
  e.w = w;
  const int32_t W = odd_window(__dmul_rn(c.beta, static_cast<double>(w)));
  e.W = W;
  A.win[kWinW * B + b] = W;
  if (e.fine) {
    const double wc = A.wc[b];
    e.wc = wc;
    double f = 1.0;
    if (isfinite(wc)) {
      const double q = __dsqrt_rn(__ddiv_rn(wc, static_cast<double>(e.p_odd)));
      f = __ddiv_rn(__dmul_rn(c.split, q), 2.0);
      f = f < 1.0 ? f : 1.0;
    }
    e.g3 = f;
  }
  e.d_min_bits = 0x7ff0000000000000ULL;
  e.d_max_bits = 0;
}

__global__ void __launch_bounds__(kThreads) eda_mode_g2_accept(Geom g, ElemRecord* le, const G2Result* res,
                                                               const int32_t* slot, const uint8_t* P0,
                                                               const uint8_t* P1, uint8_t* Pacc) {
  const int64_t total = g.batch * g.field;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t b = i / g.field;
    const G2Result r = res[b];
    if (r.outcome != kG2Accepted) continue;
    const uint8_t* src = slot[b] ? P1 : P0;
    Pacc[i] = src[i];
    if (i == b * g.field) {
      ElemRecord& e = le[b];
      e.fine = 1;
      e.g2_rung = r.s;
      e.p = r.p;
      e.p_odd = odd_window(r.p);
      e.cov = r.cov;
    }
  }
}

template <class T>
__global__ void __launch_bounds__(kThreads) eda_mode_g3(Geom g, const ElemRecord* le, T* L, const int32_t* on) {
  const int64_t total = g.batch * g.field;
  const int32_t nl = g.n[g.dim - 1];
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t b = i / g.field;
    if (on && !on[b]) continue;
    const ElemRecord& e = le[b];
    if (e.status != kRecordActive || !e.fine) continue;
    if ((i - b * g.field) % g.pitch >= nl) continue;
    const T v = mul_rn(static_cast<T>(e.g3), L[i]);
    L[i] = v >= T(3) ? v : T(3);
  }
}

__global__ void __launch_bounds__(kThreads) eda_mode_guard(int64_t B, ModeArrays A) {
  const int64_t b = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (b >= B) return;
  const ElemRecord& e = A.le[b];
  if (e.status == kRecordActive && e.nonfinite) fail_nonfinite(A, B, b);
}

template <class T>
__global__ void __launch_bounds__(kThreads) eda_mode_prescale(Geom g, const ElemState* elems, int32_t* exp,
                                                              T* r) {
  const int64_t total = g.batch * g.field;
  const int32_t nl = g.n[g.dim - 1];
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t b = i / g.field;
    const int64_t v = i - b * g.field;
    const int32_t k = scale_exp(elems[b].amax);
    if (v == 0) exp[b] = k;
    if (k == 0 || v % g.pitch >= nl) continue;
    r[i] = times_pow2(r[i], -k);
  }
}

// Reads elems[b].status and n_imfs, writes only elems[b].error.
template <class T>
__global__ void __launch_bounds__(kThreads) eda_mode_unscale(Geom g, ElemState* elems, const int32_t* exp,
                                                             T* rows, int32_t rows_cap) {
  const int64_t per = static_cast<int64_t>(rows_cap) * g.count;
  const int64_t total = g.batch * per;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t b = idx / per;
    const int32_t k = exp[b];
    if (k == 0 || elems[b].status == kError) continue;
    if ((idx - b * per) / g.count > elems[b].n_imfs) continue;
    const T v = times_pow2(rows[idx], k);
    rows[idx] = v;
    if (!isfinite(v)) elems[b].error = kErrNonFiniteResult;
  }
}

// The commit's field part: the IMF row, r - h, the element state's window.
template <class T>
__global__ void __launch_bounds__(kThreads) eda_mode_commit_rows(Geom g, const CallCtl* ctl, ElemState* elems,
                                                                 const ElemRecord* le, const T* h, T* r) {
  const int64_t total = g.batch * g.field;
  const int32_t nl = g.n[g.dim - 1];
  T* out = static_cast<T*>(ctl->out);
  const int64_t cap = ctl->rows_cap;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t b = i / g.field;
    if (le[b].status != kRecordActive) continue;
    const int64_t v = i - b * g.field;
    const int64_t q = v / g.pitch;
    const int64_t t = v - q * g.pitch;
    if (v == 0) elems[b].window = le[b].w;
    if (t >= nl) continue;
    const int32_t k = elems[b].n_imfs;
    const T hv = h[i];
    out[(b * cap + k) * g.count + q * nl + t] = hv;
    r[i] = sub_rn(r[i], hv);
  }
}

// After enqueue_decide: the method's history and the next mode's state.
__global__ void __launch_bounds__(kThreads) eda_mode_commit_post(int64_t B, const CallCtl* ctl, ElemState* elems,
                                                                 ElemRecord* le) {
  const int64_t b = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (b >= B) return;
  ElemRecord& e = le[b];
  if (e.status != kRecordActive) return;
  const ElemState& ne = elems[b];
  const int64_t h = b * (ctl->rows_cap - 1) + (ne.n_imfs - 1);
  ctl->hist_spacing[h] = e.spacing;
  ctl->hist_sifts[h] = e.n_sifts;
  e.w_prev = e.w;
  e.k += 1;
  if (!isfinite(ne.amax)) {  // a non-finite IMF or residual (the guard's rule)
    e.status = kRecordError;
    elems[b].status = kError;
    elems[b].error = kErrNonFiniteResult;
    return;
  }
  if (ne.status != kActive) e.status = ne.status == kError ? kRecordError : kRecordDone;
}

void check_batch(int64_t B) {
  if (B < 0 || B > (int64_t{1} << 31)) throw std::invalid_argument("eda: batch out of range");
}

}  // namespace

// ---------------------------------------------------------------------------------------------
void enqueue_mode_init(cudaStream_t s, int64_t batch, const ModeArrays& A) {
  check_batch(batch);
  eda_mode_init<<<grid_for(batch, kThreads), kThreads, 0, s>>>(batch, A);
  EDA_CHECK_LAUNCH("eda_mode_init");
}

void enqueue_mode_start(cudaStream_t s, int64_t batch, const MethodCtl& c, const ModeArrays& A) {
  check_batch(batch);
  eda_mode_start<<<grid_for(batch, kThreads), kThreads, 0, s>>>(batch, c, A);
  EDA_CHECK_LAUNCH("eda_mode_start");
}

void enqueue_mode_rice(cudaStream_t s, int64_t batch, const MethodCtl& c, const Geom& g, int32_t pass,
                       const ModeArrays& A) {
  check_batch(batch);
  if (pass < 0 || pass >= 2 || pass >= c.sig_passes) throw std::invalid_argument("eda: rice pass out of range");
  eda_mode_rice<<<grid_for(batch, kThreads), kThreads, 0, s>>>(batch, c, g, pass, A);
  EDA_CHECK_LAUNCH("eda_mode_rice");
}

void enqueue_mode_window(cudaStream_t s, int64_t batch, const MethodCtl& c, const ModeArrays& A) {
  check_batch(batch);
  eda_mode_window<<<grid_for(batch, kThreads), kThreads, 0, s>>>(batch, c, A);
  EDA_CHECK_LAUNCH("eda_mode_window");
}

void enqueue_mode_g2_accept(cudaStream_t s, const Geom& g, ElemRecord* le, const G2Result* res,
                            const int32_t* slot, const uint8_t* P_src0, const uint8_t* P_src1, uint8_t* P_acc) {
  if (P_acc == P_src0 || P_acc == P_src1) throw std::invalid_argument("eda: g2_accept needs a distinct P_acc");
  eda_mode_g2_accept<<<grid_for(g.batch * g.field, kThreads * 4), kThreads, 0, s>>>(g, le, res, slot, P_src0,
                                                                                    P_src1, P_acc);
  EDA_CHECK_LAUNCH("eda_mode_g2_accept");
}

template <class T>
void enqueue_mode_g3(cudaStream_t s, const Geom& g, const ElemRecord* le, T* L, const int32_t* on) {
  eda_mode_g3<T><<<grid_for(g.batch * g.field, kThreads * 4), kThreads, 0, s>>>(g, le, L, on);
  EDA_CHECK_LAUNCH("eda_mode_g3");
}

void enqueue_mode_guard(cudaStream_t s, int64_t batch, const ModeArrays& A) {
  check_batch(batch);
  eda_mode_guard<<<grid_for(batch, kThreads), kThreads, 0, s>>>(batch, A);
  EDA_CHECK_LAUNCH("eda_mode_guard");
}

template <class T>
void enqueue_mode_prescale(cudaStream_t s, const Geom& g, const ElemState* elems, int32_t* exp, T* r) {
  if (!elems || !exp || !r) throw std::invalid_argument("eda: prescale: null pointer");
  eda_mode_prescale<T><<<grid_for(g.batch * g.field, kThreads * 4), kThreads, 0, s>>>(g, elems, exp, r);
  EDA_CHECK_LAUNCH("eda_mode_prescale");
}

template <class T>
void enqueue_mode_unscale(cudaStream_t s, const Geom& g, ElemState* elems, const int32_t* exp, T* rows,
                          int32_t rows_cap) {
  if (!elems || !exp || !rows || rows_cap < 1) throw std::invalid_argument("eda: unscale: bad arguments");
  const int64_t total = g.batch * static_cast<int64_t>(rows_cap) * g.count;
  eda_mode_unscale<T><<<grid_for(total, kThreads * 4), kThreads, 0, s>>>(g, elems, exp, rows, rows_cap);
  EDA_CHECK_LAUNCH("eda_mode_unscale");
}

template <class T>
void enqueue_mode_commit(cudaStream_t s, const Geom& g, CallCtl* ctl, ElemState* elems, ElemRecord* le, const T* h,
                         T* r, RowPartial* imf_rows, RowPartial* r_rows) {
  if (h == r) throw std::invalid_argument("eda: commit needs h distinct from r");
  eda_mode_commit_rows<T><<<grid_for(g.batch * g.field, kThreads * 4), kThreads, 0, s>>>(g, ctl, elems, le, h, r);
  EDA_CHECK_LAUNCH("eda_mode_commit_rows");
  enqueue_row_partials<T>(s, g, h, imf_rows, nullptr);
  enqueue_row_partials<T>(s, g, r, r_rows, nullptr);
  enqueue_decide<T>(s, g, elems, ctl, imf_rows, r_rows);
  eda_mode_commit_post<<<grid_for(g.batch, kThreads), kThreads, 0, s>>>(g.batch, ctl, elems, le);
  EDA_CHECK_LAUNCH("eda_mode_commit_post");
}

#define EDA_MODE_INST(T)                                                                                     \
  template void enqueue_mode_g3<T>(cudaStream_t, const Geom&, const ElemRecord*, T*, const int32_t*);       \
  template void enqueue_mode_commit<T>(cudaStream_t, const Geom&, CallCtl*, ElemState*, ElemRecord*, const T*,   \
                                            T*, RowPartial*, RowPartial*);                                       \
  template void enqueue_mode_prescale<T>(cudaStream_t, const Geom&, const ElemState*, int32_t*, T*);         \
  template void enqueue_mode_unscale<T>(cudaStream_t, const Geom&, ElemState*, const int32_t*, T*, int32_t);
EDA_MODE_INST(float)
EDA_MODE_INST(double)
#undef EDA_MODE_INST

}  // namespace eda
