// SPDX-License-Identifier: Apache-2.0
// The local threshold field, the significant line extrema and the raw local window of
// method/stats.hpp.
//
// Significant extrema. Rounding is monotone, so a peak v with threshold tau is significant iff on
// each side some sample u before the first strictly higher one (or the line end) has fl(v - u) >=
// tau (in double), and for any set U of such samples that holds iff fl(v - min U) >= tau. Per axis
// two kernels: eda_stats_fields_flags_blocks writes the min and max of every 32-sample block of every
// line (lines longer than 32) into the scratch, and eda_stats_fields_flags runs one thread per
// sample: the thread of a run start finds the run's end and, at a peak, walks each side over the
// samples of the peak's own block and then over whole blocks, entering a block only when it holds
// the side's end (a higher sample or a NaN). At most 63 + n / 32 steps per side. Comparisons in T
// equal those of the doubles (a float converts exactly); see deep() for the float difference. A NaN
// ends the walk on its side, like a higher sample, as in SciPy's peak_prominences.
#include "method/stats_common.cuh"

#include <algorithm>
#include <stdexcept>

namespace eda {
namespace stats_detail {
namespace {

// ---- tau ------------------------------------------------------------------------------------------
template <class T>
__global__ void eda_stats_fields_tau_det(Geom g, const T* r, const T* box, T* e, const int32_t* Wd,
                                         const int32_t* on) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= g.batch * g.field) return;
  const int64_t b = i / g.field;
  if (!runs(on, b) || Wd[b] < 1) return;
  if (pad_at(g, i % g.field)) {
    e[i] = T(0);
    return;
  }
  const T det = sub_rn(r[i], box[i]);  // det = r - Box r
  e[i] = mul_rn(det, det);             // det * det
}

template <class T>
__global__ void eda_stats_fields_tau_final(Geom g, double theta, double k_noise, T* tau, const int32_t* Wd,
                                           const double* sigma_n, const double* tol, const double* floor2,
                                           const int32_t* on, ElemRecord* le) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= g.batch * g.field) return;
  const int64_t b = i / g.field;
  if (!runs(on, b) || Wd[b] < 1) return;
  if (pad_at(g, i % g.field)) {
    tau[i] = T(0);
    return;
  }
  const T be = tau[i];
  // a non-finite box of det^2 (an overflow, or an inf / NaN residual) would give tau = +inf or 0
  // here, and so no or every extremum: the element is flagged for the driver's guard instead
  if (le != nullptr && !(isfinite(be) && isfinite(sigma_n[b]))) le[b].nonfinite = 1;
  const T lev = sqrt_t<T>(be > T(0) ? be : T(0));  // sqrt(max(Box e, 0))
  // floor = max(0, k_noise sigma_n), f only when f > 0: a -0 or NaN product gives +0
  const double f = __dmul_rn(k_noise, sigma_n[b]);
  const double fl = f > 0.0 ? f : 0.0;
  const double ld = static_cast<double>(lev);
  double t = __dmul_rn(theta, ld > fl ? ld : fl);  // theta max(lev, floor)
  const double tl = tol[b];
  t = t > tl ? t : tl;
  if (floor2 != nullptr) {
    const double f2 = floor2[b];
    t = t > f2 ? t : f2;
  }
  tau[i] = static_cast<T>(t);
}

// ---- significant extrema ---------------------------------------------------------------------------
inline constexpr int kBlk = 32;           // samples per block summary
inline constexpr int kFlagThreads = 128;  // threads per CTA of the per-sample kernel

// x / d by multiply-high (the round-up method): with p = 31 + ceil(log2 d) and m = ceil(2^p / d) <
// 2^32, the error e = m d - 2^p < d gives x e < 2^p for every x < 2^31, so floor(x m / 2^p) =
// floor(x / d). Every id of a flags launch is below 2^31 (enqueue_sig_flags splits the batch).
struct FastDiv {
  uint32_t d, mul, shr;
};
inline FastDiv fast_div(int64_t d64) {
  if (d64 < 1 || d64 > (int64_t{1} << 31)) throw std::length_error("eda: sig_flags: divisor out of range");
  const uint32_t d = static_cast<uint32_t>(d64);
  FastDiv f{d, 0, 0};
  if (d != 1) {
    int lg = 0;
    while ((uint64_t{1} << lg) < d) ++lg;
    const int p = 31 + lg;
    f.mul = static_cast<uint32_t>(((uint64_t{1} << p) + d - 1) / d);
    f.shr = static_cast<uint32_t>(p - 32);
  }
  return f;
}
__device__ __forceinline__ uint32_t fdiv(uint32_t x, const FastDiv& f) {
  return f.d == 1 ? x : (__umulhi(x, f.mul) >> f.shr);
}

// The lines of one axis of one launch, computed on the host: sample s of line (o, j) at o *
// outer_stride + s * step + j, id (o * n + s) * inner + j. The kernels take no Geom (a Geom indexed
// at run time would be copied to local memory in every thread).
struct FlagLines {
  int64_t outer_stride, step;
  int64_t field;          // padded samples per element: element b's scratch starts at b * field
  FastDiv inner, n, per_element, nb, pitch;
  int32_t axis;
  int32_t outer_axis;     // 1: axis < D - 1, whose pad lines (j % pitch >= n[D - 1]) are skipped
  int32_t n_last;         // n[D - 1]
  uint32_t total;         // outer * n * inner ids
  uint32_t entries;       // outer * nb * inner block summaries
  uint32_t elem_entries;  // per_element * nb * inner: one element's minima, its maxima follow
};
FlagLines flag_lines(const Geom& g, int axis) {
  const LineView lv = line_view(g, axis);
  const int64_t nb = (lv.n + kBlk - 1) / kBlk;
  FlagLines L{};
  L.outer_stride = lv.outer_stride;
  L.step = lv.step;
  L.field = g.field;
  L.inner = fast_div(lv.inner);
  L.n = fast_div(lv.n);
  L.per_element = fast_div(lv.per_element);
  L.nb = fast_div(nb);
  L.pitch = fast_div(g.pitch);
  L.axis = axis;
  L.outer_axis = axis != g.dim - 1;
  L.n_last = g.n[g.dim - 1];
  const int64_t total = lv.outer * lv.n * lv.inner;
  if (total > (int64_t{1} << 31)) throw std::length_error("eda: sig_flags: a launch above 2^31 ids");
  L.total = static_cast<uint32_t>(total);
  L.entries = static_cast<uint32_t>(lv.outer * nb * lv.inner);
  L.elem_entries = static_cast<uint32_t>(lv.per_element * nb * lv.inner);
  return L;
}
__device__ __forceinline__ bool pad_line(const FlagLines& L, uint32_t j) {
  return L.outer_axis && j - fdiv(j, L.pitch) * L.pitch.d >= static_cast<uint32_t>(L.n_last);
}

// Integer count of a warp's peaks into le[b].ext[axis] (every lane: its element b and its count):
// one atomic when the warp's counted peaks are of one element.
__device__ __forceinline__ void add_counts(ElemRecord* le, int32_t axis, uint32_t b, uint32_t cnt) {
  const unsigned full = 0xffffffffu;
  const uint32_t b0 = __shfl_sync(full, b, 0);
  const bool uniform = __all_sync(full, b == b0 || cnt == 0);
  if (uniform) {
    unsigned long long c = cnt;
    for (int d = 16; d > 0; d >>= 1) c += __shfl_down_sync(full, c, d);
    if ((threadIdx.x & 31) == 0 && c) atomicAdd(reinterpret_cast<unsigned long long*>(&le[b0].ext[axis]), c);
  } else if (cnt) {
    atomicAdd(reinterpret_cast<unsigned long long*>(&le[b].ext[axis]), static_cast<unsigned long long>(cnt));
  }
}

// The min and max of every 32-sample block of every line (one thread per block; a NaN makes both
// NaN), into element b's part of the scratch.
template <class T>
__global__ void __launch_bounds__(kThreads) eda_stats_fields_flags_blocks(FlagLines L, const T* __restrict__ r,
                                                                          T* __restrict__ sum, const int32_t* on) {
  const uint32_t id = blockIdx.x * kThreads + threadIdx.x;
  if (id >= L.entries) return;
  const uint32_t t = fdiv(id, L.inner);
  const uint32_t j = id - t * L.inner.d;
  const uint32_t o = fdiv(t, L.nb);
  const int32_t c = static_cast<int32_t>(t - o * L.nb.d);
  const uint32_t b = fdiv(o, L.per_element);
  if (!runs(on, b) || pad_line(L, j)) return;
  const T* x = r + static_cast<int64_t>(o) * L.outer_stride + j;
  const int32_t n = static_cast<int32_t>(L.n.d);
  const int32_t i0 = c * kBlk;
  const int32_t i1 = min(i0 + kBlk, n);
  T mn = x[static_cast<int64_t>(i0) * L.step];
  T mx = mn;
  for (int32_t i = i0 + 1; i < i1; ++i) {
    const T v = x[static_cast<int64_t>(i) * L.step];
    mn = (v < mn || v != v) ? v : mn;  // once NaN, stays NaN
    mx = (v > mx || v != v) ? v : mx;
  }
  T* dst = sum + static_cast<int64_t>(b) * L.field + (id - b * L.elem_entries);
  dst[0] = mn;
  dst[L.elem_entries] = mx;
}

// fl64(a - b) >= tau, the side test (a >= b, or an infinity or NaN among them). For float the float
// difference d = fl32(a - b) decides unless d == tau: D = a - b is exact, rounding is monotone and
// tau is a float (tau = fl32(tau) = fl64(tau)), so
//   d > tau implies D > tau, so fl64(D) >= tau;
//   d < tau implies tau > 0 and D <= (pred32(tau) + tau) / 2 < (pred64(tau) + tau) / 2 (the double
//     grid is finer, subnormals included), so fl64(D) < tau; for tau = +inf, D <= 2 FLT_MAX is
//     finite;
//   unordered (tau NaN, or a = b = +-inf): false, as in double.
// Only a tie d == tau takes the double difference. A subnormal d must not flush to zero: the build
// uses -ftz=false (CMake refuses -ftz=true). deep64 is not inlined, so that the double operations
// run only on a tie (a real branch, not predicated).
__device__ __noinline__ bool deep64(float a, float b, float tau) {
  return __dsub_rn(static_cast<double>(a), static_cast<double>(b)) >= static_cast<double>(tau);
}
template <class T>
__device__ __forceinline__ bool deep(T a, T b, T tau) {
  if constexpr (std::is_same_v<T, float>) {
    const float d = __fsub_rn(a, b);
    if (d != tau) return d > tau;
    return deep64(a, b, tau);
  } else {
    return __dsub_rn(a, b) >= tau;
  }
}

// One sample u of a side of the peak v (maxima; minima mirror it): 2 ends the side not deep (a
// higher sample, or a NaN), 1 ends it deep (fl(v - u) >= tau), 0 goes on.
template <bool Max, class T>
__device__ __forceinline__ int side_step(T u, T v, T tau) {
  if (Max ? !(u <= v) : !(u >= v)) return 2;
  return (Max ? deep<T>(v, u, tau) : deep<T>(u, v, tau)) ? 1 : 0;
}

// Whether the side of p in direction Dir is deep: the samples of p's own block one by one, then
// block by block (bmin, bmax: the line's summaries, block c at c * sst). A block whose max exceeds
// v (or is NaN) holds the end of the side: its samples one by one. Otherwise every sample of it
// goes on or is deep, and some is deep iff fl(v - bmin) >= tau (rounding is monotone); else the
// block is skipped. The line end is not deep.
template <bool Max, int Dir, class T>
__device__ __forceinline__ bool side_deep(const T* __restrict__ x, int64_t st, int32_t n, int32_t p, T v, T tau,
                                          const T* __restrict__ bmin, const T* __restrict__ bmax, int64_t sst) {
  const int32_t cp = p / kBlk;
  const int32_t lim = Dir < 0 ? cp * kBlk : min(cp * kBlk + kBlk - 1, n - 1);
  for (int32_t k = p + Dir; Dir < 0 ? k >= lim : k <= lim; k += Dir) {
    const int s = side_step<Max>(x[static_cast<int64_t>(k) * st], v, tau);
    if (s) return s == 1;
  }
  const int32_t nb = (n + kBlk - 1) / kBlk;
  for (int32_t c = cp + Dir; Dir < 0 ? c >= 0 : c < nb; c += Dir) {
    const T hi = Max ? bmax[c * sst] : bmin[c * sst];
    if (Max ? !(hi <= v) : !(hi >= v)) {
      const int32_t k0 = Dir < 0 ? c * kBlk + kBlk - 1 : c * kBlk;
      const int32_t k1 = Dir < 0 ? c * kBlk : min(c * kBlk + kBlk - 1, n - 1);
      for (int32_t k = k0; Dir < 0 ? k >= k1 : k <= k1; k += Dir) {
        const int s = side_step<Max>(x[static_cast<int64_t>(k) * st], v, tau);
        if (s) return s == 1;
      }
      return false;  // not reached: the block holds a higher sample or a NaN
    }
    const T lo = Max ? bmin[c * sst] : bmax[c * sst];
    if (Max ? deep<T>(v, lo, tau) : deep<T>(lo, v, tau)) return true;
  }
  return false;
}

// The last sample of the run of samples equal to cur that starts at s: the rest of s's block one by
// one, then whole blocks of samples equal to cur at once, then the first other block one by one.
template <class T>
__device__ __forceinline__ int32_t run_end(const T* __restrict__ x, int64_t st, int32_t n, int32_t s, T cur,
                                           const T* __restrict__ bmin, const T* __restrict__ bmax, int64_t sst) {
  const int32_t lim = min((s / kBlk) * kBlk + kBlk - 1, n - 1);
  int32_t k = s + 1;
  while (k <= lim && x[static_cast<int64_t>(k) * st] == cur) ++k;
  if (k <= lim || k >= n) return k - 1;
  const int32_t nb = (n + kBlk - 1) / kBlk;
  for (int32_t c = k / kBlk; c < nb; ++c) {
    if (bmin[c * sst] == cur && bmax[c * sst] == cur) continue;
    k = c * kBlk;
    while (x[static_cast<int64_t>(k) * st] == cur) ++k;
    return k - 1;
  }
  return n - 1;
}

// One thread per sample (o, s, j), j fastest. Only the thread of a run start (x[s-1] < x[s] or
// x[s-1] > x[s]; a NaN on either side is neither) owns a peak: the run [s, e] of samples equal to
// x[s] is a peak of r (of -r) when both neighbours are lower (higher), at p = (s + e) / 2 with v =
// x[s]; it is flagged when both sides are deep (left first). Lines of n <= 32 samples are one block
// and need no summaries.
template <class T>
__global__ void __launch_bounds__(kFlagThreads) eda_stats_fields_flags(FlagLines L, const T* __restrict__ r,
                                                                       const T* __restrict__ tau,
                                                                       uint8_t* __restrict__ flags, ElemRecord* le,
                                                                       const int32_t* on, const T* __restrict__ sum) {
  const int32_t n = static_cast<int32_t>(L.n.d);
  const uint32_t total = L.total;
  const uint32_t id = blockIdx.x * kFlagThreads + threadIdx.x;
  const uint32_t idc = id < total ? id : total - 1;
  const uint32_t t = fdiv(idc, L.inner);
  const uint32_t j = idc - t * L.inner.d;
  const uint32_t o = fdiv(t, L.n);
  const int32_t s = static_cast<int32_t>(t - o * L.n.d);
  const uint32_t b = fdiv(o, L.per_element);
  bool flagged = false;
  if (id < total && s >= 1 && s <= n - 2 && runs(on, b) && !pad_line(L, j)) {
    const int64_t base = static_cast<int64_t>(o) * L.outer_stride + j;
    const T* x = r + base;
    const int64_t st = L.step;
    const T cur = x[static_cast<int64_t>(s) * st];
    const T prev = x[static_cast<int64_t>(s - 1) * st];
    if (prev < cur || prev > cur) {
      // block c of this line at bmin[c * inner], in element b's part of the scratch
      const T* bmin = sum + static_cast<int64_t>(b) * L.field +
                      static_cast<int64_t>(o - b * L.per_element.d) * L.nb.d * L.inner.d + j;
      const T* bmax = bmin + L.elem_entries;
      const int64_t sst = L.inner.d;
      const int32_t e = run_end(x, st, n, s, cur, bmin, bmax, sst);
      if (e <= n - 2) {
        const T nxt = x[static_cast<int64_t>(e + 1) * st];
        const bool is_max = prev < cur && nxt < cur;
        const bool is_min = prev > cur && nxt > cur;
        if (is_max || is_min) {
          const int32_t p = (s + e) / 2;
          const T tp = tau[base + static_cast<int64_t>(p) * st];
          const bool sig = is_max ? side_deep<true, -1>(x, st, n, p, cur, tp, bmin, bmax, sst) &&
                                        side_deep<true, 1>(x, st, n, p, cur, tp, bmin, bmax, sst)
                                  : side_deep<false, -1>(x, st, n, p, cur, tp, bmin, bmax, sst) &&
                                        side_deep<false, 1>(x, st, n, p, cur, tp, bmin, bmax, sst);
          if (sig) {
            flags[base + static_cast<int64_t>(p) * st] |= static_cast<uint8_t>(1u << L.axis);
            flagged = true;
          }
        }
      }
    }
  }
  // the count: one atomic per CTA when the CTA's samples are of one element, else per warp
  const uint32_t first = blockIdx.x * kFlagThreads;
  const uint32_t last = min(first + kFlagThreads, total) - 1;
  const uint32_t b_first = fdiv(fdiv(fdiv(first, L.inner), L.n), L.per_element);
  const uint32_t b_last = fdiv(fdiv(fdiv(last, L.inner), L.n), L.per_element);
  if (b_first == b_last) {
    const int c = __syncthreads_count(flagged);
    if (threadIdx.x == 0 && c) {
      atomicAdd(reinterpret_cast<unsigned long long*>(&le[b_first].ext[L.axis]), static_cast<unsigned long long>(c));
    }
  } else {
    add_counts(le, L.axis, b, flagged ? 1u : 0u);  // b is valid on every lane (idc)
  }
}

__global__ void eda_stats_fields_clear_u8(Geom g, uint8_t* m, const int32_t* on) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= g.batch * g.field) return;
  if (!runs(on, i / g.field)) return;
  m[i] = 0;
}

// ---- raw window ----------------------------------------------------------------------------------
// op 0: tmp = f_0; op 1: om = tmp^2 (axis 0) or om += tmp^2, then tmp = f_{a+1}; op 2: the last axis
// accumulated and om = 2 kappa / max(sqrt(dens2), tiny) where dens2 > 0, +inf elsewhere.
template <class T>
__global__ void eda_stats_fields_raw(Geom g, const uint8_t* flags, const int32_t* win, T* om, T* tmp, int32_t axis,
                                     int32_t op, T two_kappa, T tiny, const int32_t* on) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= g.batch * g.field) return;
  const int64_t b = i / g.field;
  if (!runs(on, b) || win[b] < 1) return;
  if (pad_at(g, i % g.field)) {
    tmp[i] = T(0);
    om[i] = T(0);
    return;
  }
  if (op == 0) {
    tmp[i] = (flags[i] & 1u) ? T(1) : T(0);
    return;
  }
  const T rho = tmp[i];
  const T sq = mul_rn(rho, rho);
  const T acc = axis == 0 ? sq : add_rn(om[i], sq);
  if (op == 1) {
    om[i] = acc;
    tmp[i] = ((flags[i] >> (axis + 1)) & 1u) ? T(1) : T(0);
    return;
  }
  const T dens = sqrt_t<T>(acc);
  om[i] = dens > T(0) ? div_t<T>(two_kappa, dens > tiny ? dens : tiny) : inf_t<T>();
}

}  // namespace
}  // namespace stats_detail

template <class T>
void enqueue_tau(cudaStream_t s, const Geom& g, const MethodCtl& c, const T* r, const int32_t* Wd,
                 const int32_t* Wl, const double* sigma_n, const double* tol, const double* floor2, T* tau,
                 T* tmp, const int32_t* on, ElemRecord* le) {
  using namespace stats_detail;
  if (!r || !Wd || !Wl || !sigma_n || !tol || !tau || !tmp) throw std::invalid_argument("eda: tau: null pointer");
  if (tau == r || tmp == r || tmp == tau) throw std::invalid_argument("eda: tau: aliased fields");
  const int64_t total = g.batch * g.field;
  enqueue_cube<T>(s, g, CubeOp::kBox, r, tmp, nullptr, Wd);
  eda_stats_fields_tau_det<T><<<blocks_for(total), kThreads, 0, s>>>(g, r, tmp, tau, Wd, on);
  EDA_CHECK_LAUNCH("eda_stats_fields_tau_det");
  enqueue_cube<T>(s, g, CubeOp::kBox, tau, tau, nullptr, Wl);
  eda_stats_fields_tau_final<T><<<blocks_for(total), kThreads, 0, s>>>(g, c.sig_theta, c.sig_noise, tau, Wd, sigma_n,
                                                                       tol, floor2, on, le);
  EDA_CHECK_LAUNCH("eda_stats_fields_tau_final");
}

template <class T>
void enqueue_sig_flags(cudaStream_t s, const Geom& g, const T* r, const T* tau, uint8_t* flags, ElemRecord* le,
                       const int32_t* on, T* tmp, int64_t launch_ids) {
  using namespace stats_detail;
  if (!r || !tau || !flags || !le || !tmp) throw std::invalid_argument("eda: sig_flags: null pointer");
  if (tmp == r || tmp == tau) throw std::invalid_argument("eda: sig_flags: aliased fields");
  if (launch_ids < 1) throw std::invalid_argument("eda: sig_flags: launch_ids must be >= 1");
  eda_stats_fields_clear_u8<<<blocks_for(g.batch * g.field), kThreads, 0, s>>>(g, flags, on);
  EDA_CHECK_LAUNCH("eda_stats_fields_clear_u8");
  const int64_t max_ids = std::min<int64_t>(launch_ids, int64_t{1} << 31);
  for (int a = 0; a < g.dim; ++a) {
    const LineView lv = line_view(g, a);
    if (lv.n < 3) continue;  // no peaks
    // a launch covers whole elements, at most max_ids ids (one element has <= field <= 2^31)
    const int64_t per = lv.per_element * lv.n * lv.inner;
    const int64_t group = std::max<int64_t>(1, max_ids / per);
    for (int64_t b0 = 0; b0 < g.batch; b0 += group) {
      Geom gs = g;
      gs.batch = std::min<int64_t>(group, g.batch - b0);
      const FlagLines L = flag_lines(gs, a);
      const int64_t off = b0 * g.field;
      const int32_t* ons = on ? on + b0 : nullptr;
      if (lv.n > kBlk) {
        eda_stats_fields_flags_blocks<T><<<blocks_for(L.entries), kThreads, 0, s>>>(L, r + off, tmp + off, ons);
        EDA_CHECK_LAUNCH("eda_stats_fields_flags_blocks");
      }
      eda_stats_fields_flags<T><<<blocks_for(L.total, kFlagThreads), kFlagThreads, 0, s>>>(
          L, r + off, tau + off, flags + off, le + b0, ons, tmp + off);
      EDA_CHECK_LAUNCH("eda_stats_fields_flags");
    }
  }
}

template <class T>
void enqueue_raw_window(cudaStream_t s, const Geom& g, const MethodCtl& c, const uint8_t* flags,
                        const int32_t* win, T* om, T* tmp, const int32_t* on) {
  using namespace stats_detail;
  if (!flags || !win || !om || !tmp || om == tmp) throw std::invalid_argument("eda: raw_window: bad pointers");
  const unsigned nb = blocks_for(g.batch * g.field);
  const T two_kappa = static_cast<T>(2.0 * c.kappa);
  const T tiny = static_cast<T>(c.tiny);
  eda_stats_fields_raw<T><<<nb, kThreads, 0, s>>>(g, flags, win, om, tmp, 0, 0, two_kappa, tiny, on);
  EDA_CHECK_LAUNCH("eda_stats_fields_raw");
  for (int a = 0; a < g.dim; ++a) {
    enqueue_cube<T>(s, g, CubeOp::kBox, tmp, tmp, nullptr, win);
    eda_stats_fields_raw<T><<<nb, kThreads, 0, s>>>(g, flags, win, om, tmp, a, a == g.dim - 1 ? 2 : 1, two_kappa, tiny,
                                                    on);
    EDA_CHECK_LAUNCH("eda_stats_fields_raw");
  }
}

#define EDA_STATS_FIELDS_INST(T)                                                                                \
  template void enqueue_tau<T>(cudaStream_t, const Geom&, const MethodCtl&, const T*, const int32_t*,         \
                                    const int32_t*, const double*, const double*, const double*, T*, T*,           \
                                    const int32_t*, ElemRecord*);                                                    \
  template void enqueue_sig_flags<T>(cudaStream_t, const Geom&, const T*, const T*, uint8_t*, ElemRecord*,    \
                                          const int32_t*, T*, int64_t);                                            \
  template void enqueue_raw_window<T>(cudaStream_t, const Geom&, const MethodCtl&, const uint8_t*,            \
                                           const int32_t*, T*, T*, const int32_t*);
EDA_STATS_FIELDS_INST(float)
EDA_STATS_FIELDS_INST(double)
#undef EDA_STATS_FIELDS_INST

}  // namespace eda
