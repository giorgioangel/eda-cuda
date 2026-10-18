// SPDX-License-Identifier: Apache-2.0
// Exact selections (the median and the linear quantile as NumPy defines them) and the noise levels
// of method/stats.hpp.
//
// A selection finds the order statistics of rank r0 <= r1 of every element's value set by six
// most-significant-digit radix passes (11, 11, 11, 11, 11 and 9 bits) over the 64-bit order keys of
// the values as doubles: per pass a histogram kernel counts the digit of every value whose higher
// digits equal the rank's known prefix (shared-memory uint32 bins, flushed into 64-bit global bins
// by integer atomics), and a pick kernel (one CTA per element) finds the bin that holds the rank.
// Integer counts are order independent, so the result never depends on the order of atomics. The
// 11-bit digits keep the two shared histograms at 2 x 2^11 uint32 bins (16 KiB), and six passes
// cover the 64-bit key. The median is the mean of the two middle values (median_pair), the
// quantile NumPy's linear rule (quantile_index_rn / quantile_linear_rn of stats_common.cuh: the
// (n - 1) q index, round-to-nearest intrinsics); an empty set gives NaN and count 0.
#include "method/stats_common.cuh"

#include <cmath>
#include <stdexcept>

namespace eda {
namespace stats_detail {
namespace {

enum SelMode : int32_t { kSelMedian = 0, kSelQuantile = 1 };

template <class T>
struct SelArgs {
  Selection<T> sel;
  int32_t naxes;
  int32_t ax[kMaxDim];
  int64_t items;  // candidate items per element
};

// Value t of element b's candidate list; false when t is not a member of the value set.
template <class T>
__device__ __forceinline__ bool sel_value(const Geom& g, const SelArgs<T>& a, int64_t b, int64_t t, double* out) {
  double val;
  if (a.sel.src == SelSource::kField) {
    if (pad_at(g, t)) return false;
    const int64_t i = b * g.field + t;
    if (a.sel.mask != nullptr && (a.sel.mask[i] & a.sel.bits) == 0) return false;
    val = static_cast<double>(a.sel.field[i]);
  } else {
    const int64_t k = t / g.field;
    const int64_t v = t - k * g.field;
    const int axis = a.ax[k];
    if (pad_at(g, v) || coord_of(g, v, axis) >= g.n[axis] - 1) return false;
    const int64_t i = b * g.field + v;
    const int64_t st = axis == g.dim - 1 ? 1 : g.stride[axis];
    val = static_cast<double>(sub_rn(a.sel.field[i + st], a.sel.field[i]));  // the difference in T
  }
  if (a.sel.center != nullptr) val = fabs(__dsub_rn(val, a.sel.center[b]));
  *out = val;
  return true;
}

__global__ void eda_stats_sel_init(int64_t B, SelState* st, uint64_t* hist, const int32_t* on) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= B * 2 * kSelBins) return;
  const int64_t b = i / (2 * kSelBins);
  if (!runs(on, b)) return;
  hist[i] = 0;
  if (i % (2 * kSelBins) == 0) {
    SelState z{};
    st[b] = z;
  }
}

template <class T>
__global__ void __launch_bounds__(kThreads) eda_stats_sel_hist(Geom g, SelArgs<T> a, const SelState* st,
                                                               uint64_t* hist, int32_t nblk, int32_t shift,
                                                               int32_t width, int32_t first, const int32_t* on) {
  __shared__ uint32_t h[2][kSelBins];
  const int64_t b = blockIdx.x / nblk;
  const int64_t chunk = blockIdx.x % nblk;
  if (!runs(on, b)) return;
  const SelState S = st[b];
  if (!first && !S.valid) return;
  const int nr = (first || S.same) ? 1 : 2;
  for (int i = threadIdx.x; i < 2 * kSelBins; i += blockDim.x) h[i / kSelBins][i % kSelBins] = 0u;
  __syncthreads();
  const int hi = shift + width;  // < 64 when !first
  const uint64_t dmask = (1ULL << width) - 1ULL;
  const int64_t stride = static_cast<int64_t>(nblk) * blockDim.x;
  for (int64_t t = chunk * blockDim.x + threadIdx.x; t < a.items; t += stride) {
    double v;
    if (!sel_value(g, a, b, t, &v)) continue;
    const uint64_t k = order_key(v);
    const uint32_t bin = static_cast<uint32_t>((k >> shift) & dmask);
    if (first) {
      atomicAdd(&h[0][bin], 1u);
    } else {
      if ((k >> hi) == (S.prefix[0] >> hi)) atomicAdd(&h[0][bin], 1u);
      if (nr == 2 && (k >> hi) == (S.prefix[1] >> hi)) atomicAdd(&h[1][bin], 1u);
    }
  }
  __syncthreads();
  const int nb = 1 << width;
  for (int i = threadIdx.x; i < nr * nb; i += blockDim.x) {
    const int r = i / nb, bin = i % nb;
    const uint32_t c = h[r][bin];
    if (c) atomicAdd(reinterpret_cast<unsigned long long*>(&hist[(b * 2 + r) * kSelBins + bin]),
                     static_cast<unsigned long long>(c));
  }
}

// One CTA (kThreads) per element: the bin of each rank, the prefix and the remaining rank; zeroes
// the histograms for the next pass.
__global__ void __launch_bounds__(kThreads) eda_stats_sel_pick(SelState* st, uint64_t* hist, int32_t shift,
                                                               int32_t width, int32_t first, int32_t mode,
                                                               double q, const int32_t* on) {
  __shared__ uint64_t part[kThreads];
  __shared__ uint64_t base[kThreads];
  __shared__ SelState S;
  __shared__ int32_t found_bin;
  __shared__ uint64_t found_rank;
  const int64_t b = blockIdx.x;
  if (!runs(on, b)) return;
  const int tid = threadIdx.x;
  if (tid == 0) S = st[b];
  __syncthreads();
  if (!first && !S.valid) return;  // uniform
  const int nb = 1 << width;
  const int per = nb / kThreads;  // 8 (11 bits) or 2 (9 bits)
  uint64_t* h0 = hist + (b * 2) * kSelBins;
  if (first) {
    uint64_t sum = 0;
    for (int j = 0; j < per; ++j) sum += h0[tid * per + j];
    part[tid] = sum;
    __syncthreads();
    if (tid == 0) {
      uint64_t n = 0;
      for (int t = 0; t < kThreads; ++t) n += part[t];
      S.count = static_cast<int64_t>(n);
      S.valid = n > 0;
      S.prefix[0] = S.prefix[1] = 0;
      S.same = 1;
      if (n > 0) {
        if (mode == kSelMedian) {
          S.rank[0] = (n - 1) / 2;
          S.rank[1] = n / 2;
        } else {
          int64_t i = 0, j = 0;
          double gm = 0.0;
          quantile_index_rn(static_cast<int64_t>(n), q, &i, &j, &gm);
          S.i_lo = i;
          S.gamma = gm;
          S.rank[0] = static_cast<uint64_t>(i);
          S.rank[1] = static_cast<uint64_t>(j);
        }
      }
    }
    __syncthreads();
    if (!S.valid) {
      if (tid == 0) st[b] = S;
      return;  // uniform; an empty set's histogram is all zero already
    }
  }
  for (int r = 0; r < 2; ++r) {
    const uint64_t* hr = hist + (b * 2 + (S.same ? 0 : r)) * kSelBins;
    uint64_t sum = 0;
    for (int j = 0; j < per; ++j) sum += hr[tid * per + j];
    part[tid] = sum;
    __syncthreads();
    if (tid == 0) {
      uint64_t acc = 0;
      for (int t = 0; t < kThreads; ++t) {
        base[t] = acc;
        acc += part[t];
      }
    }
    __syncthreads();
    const uint64_t rank = S.rank[r];
    if (rank >= base[tid] && rank < base[tid] + part[tid]) {
      uint64_t acc = base[tid];
      for (int j = 0; j < per; ++j) {
        const uint64_t c = hr[tid * per + j];
        if (rank < acc + c) {
          found_bin = tid * per + j;
          found_rank = rank - acc;
          break;
        }
        acc += c;
      }
    }
    __syncthreads();
    if (tid == 0) {
      S.prefix[r] |= static_cast<uint64_t>(found_bin) << shift;
      S.rank[r] = found_rank;
    }
    __syncthreads();
  }
  // zero both histograms for the next pass
  for (int i = tid; i < 2 * kSelBins; i += kThreads) h0[i] = 0;
  if (tid == 0) {
    S.same = S.prefix[0] == S.prefix[1];
    st[b] = S;
  }
}

__global__ void eda_stats_sel_final(int64_t B, const SelState* st, int32_t mode, double* out, int64_t* count,
                                    const int32_t* on) {
  const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= B || !runs(on, b)) return;
  const SelState S = st[b];
  if (count != nullptr) count[b] = S.valid ? S.count : 0;
  if (!S.valid) {
    out[b] = __longlong_as_double(0x7ff8000000000000LL);  // NaN
    return;
  }
  const double a = key_value(S.prefix[0]);
  const double c = key_value(S.prefix[1]);
  if (mode == kSelMedian) {
    out[b] = S.count % 2 ? a : median_pair(a, c);
  } else {
    out[b] = quantile_linear_rn(a, c, S.gamma);
  }
}

template <class T>
void select_impl(cudaStream_t s, const Geom& g, const Selection<T>& sel, int32_t mode, double q, double* out,
                 int64_t* count, void* ws, const int32_t* on) {
  if (sel.field == nullptr || out == nullptr || ws == nullptr) throw std::invalid_argument("eda: select: null pointer");
  SelArgs<T> a{};
  a.sel = sel;
  if (sel.src == SelSource::kField) {
    a.naxes = 1;
    a.items = g.field;
  } else {
    if (sel.mask != nullptr) throw std::invalid_argument("eda: select: kDiff takes no mask");
    for (int ax = 0; ax < g.dim; ++ax) {
      if (sel.axes & (1u << ax)) a.ax[a.naxes++] = ax;
    }
    a.items = static_cast<int64_t>(a.naxes) * g.field;
  }
  const int64_t B = g.batch;
  SelState* st = sel_state(ws);
  uint64_t* hist = sel_hist(ws, B);
  eda_stats_sel_init<<<blocks_for(B * 2 * kSelBins), kThreads, 0, s>>>(B, st, hist, on);
  EDA_CHECK_LAUNCH("eda_stats_sel_init");
  int64_t nblk = (a.items + int64_t{kThreads} * 64 - 1) / (int64_t{kThreads} * 64);
  nblk = nblk < 1 ? 1 : (nblk > 512 ? 512 : nblk);
  if (nblk * B > 0x7fffffffLL) throw std::length_error("eda: select: launch grid too large");
  if (a.items > 0) {
    for (int p = 0; p < kSelPasses; ++p) {
      const int top = 64 - kSelBits * p;
      const int shift = top - kSelBits > 0 ? top - kSelBits : 0;
      const int width = top - shift;
      eda_stats_sel_hist<T><<<static_cast<unsigned>(nblk * B), kThreads, 0, s>>>(
          g, a, st, hist, static_cast<int32_t>(nblk), shift, width, p == 0, on);
      EDA_CHECK_LAUNCH("eda_stats_sel_hist");
      eda_stats_sel_pick<<<static_cast<unsigned>(B), kThreads, 0, s>>>(st, hist, shift, width, p == 0, mode, q,
                                                                       on);
      EDA_CHECK_LAUNCH("eda_stats_sel_pick");
    }
  }
  eda_stats_sel_final<<<blocks_for(B), kThreads, 0, s>>>(B, st, mode, out, count, on);
  EDA_CHECK_LAUNCH("eda_stats_sel_final");
}

// out[b] = (1.4826 mad) / sqrt 2 (pooled), or the running max over axes of the same (per axis).
__global__ void eda_stats_noise_level(int64_t B, const double* mad, double* acc, double* out, double sqrt2,
                                      int32_t op, const int32_t* on) {
  const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= B || !runs(on, b)) return;
  if (op == 0) {  // pooled
    out[b] = __ddiv_rn(__dmul_rn(1.4826, mad[b]), sqrt2);
  } else if (op == 1) {  // per axis: init
    acc[b] = 0.0;
  } else if (op == 2) {  // per axis: the running max keeps acc unless v > acc (the first axis wins ties)
    const double v = __ddiv_rn(__dmul_rn(1.4826, mad[b]), sqrt2);
    if (v > acc[b]) acc[b] = v;
  } else {
    out[b] = acc[b];
  }
}

}  // namespace
}  // namespace stats_detail

size_t select_workspace_bytes(const Geom& g) {
  return stats_detail::sel_core_bytes(g.batch) + stats_detail::sel_extra_bytes(g.batch);
}

template <class T>
void enqueue_median(cudaStream_t s, const Geom& g, const Selection<T>& sel, double* out, int64_t* count,
                    void* ws, const int32_t* on) {
  stats_detail::select_impl<T>(s, g, sel, stats_detail::kSelMedian, 0.5, out, count, ws, on);
}

template <class T>
void enqueue_quantile(cudaStream_t s, const Geom& g, const Selection<T>& sel, double q, double* out,
                      int64_t* count, void* ws, const int32_t* on) {
  if (!(q >= 0.0 && q <= 1.0)) throw std::invalid_argument("eda: quantile: q must lie in [0, 1]");
  stats_detail::select_impl<T>(s, g, sel, stats_detail::kSelQuantile, q, out, count, ws, on);
}

template <class T>
void enqueue_noise_sigma(cudaStream_t s, const Geom& g, const T* r, bool per_axis, double* out, void* ws,
                         const int32_t* on) {
  using namespace stats_detail;
  if (r == nullptr || out == nullptr || ws == nullptr) throw std::invalid_argument("eda: noise_sigma: null pointer");
  const int64_t B = g.batch;
  const SelExtra E = sel_extra(ws, B);
  const double sqrt2 = std::sqrt(2.0);  // correctly rounded (IEEE sqrt)
  Selection<T> sel{};
  sel.src = SelSource::kDiff;
  sel.field = r;
  const unsigned nb = blocks_for(B);
  if (!per_axis) {
    sel.axes = static_cast<uint8_t>((1u << g.dim) - 1u);
    sel.center = nullptr;
    enqueue_median<T>(s, g, sel, E.center, nullptr, ws, on);
    sel.center = E.center;
    enqueue_median<T>(s, g, sel, E.mad, nullptr, ws, on);
    eda_stats_noise_level<<<nb, kThreads, 0, s>>>(B, E.mad, E.acc, out, sqrt2, 0, on);
    EDA_CHECK_LAUNCH("eda_stats_noise_level");
    return;
  }
  eda_stats_noise_level<<<nb, kThreads, 0, s>>>(B, E.mad, E.acc, out, sqrt2, 1, on);
  EDA_CHECK_LAUNCH("eda_stats_noise_level");
  for (int a = 0; a < g.dim; ++a) {
    if (g.n[a] < 2) continue;  // no differences along this axis: it has no noise level
    sel.axes = static_cast<uint8_t>(1u << a);
    sel.center = nullptr;
    enqueue_median<T>(s, g, sel, E.center, nullptr, ws, on);
    sel.center = E.center;
    enqueue_median<T>(s, g, sel, E.mad, nullptr, ws, on);
    eda_stats_noise_level<<<nb, kThreads, 0, s>>>(B, E.mad, E.acc, out, sqrt2, 2, on);
    EDA_CHECK_LAUNCH("eda_stats_noise_level");
  }
  eda_stats_noise_level<<<nb, kThreads, 0, s>>>(B, E.mad, E.acc, out, sqrt2, 3, on);
  EDA_CHECK_LAUNCH("eda_stats_noise_level");
}

#define EDA_STATS_INST(T)                                                                                   \
  template void enqueue_median<T>(cudaStream_t, const Geom&, const Selection<T>&, double*, int64_t*, void*,   \
                                       const int32_t*);                                                            \
  template void enqueue_quantile<T>(cudaStream_t, const Geom&, const Selection<T>&, double, double*,          \
                                         int64_t*, void*, const int32_t*);                                         \
  template void enqueue_noise_sigma<T>(cudaStream_t, const Geom&, const T*, bool, double*, void*,           \
                                            const int32_t*);
EDA_STATS_INST(float)
EDA_STATS_INST(double)
#undef EDA_STATS_INST

}  // namespace eda
