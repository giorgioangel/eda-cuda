// SPDX-License-Identifier: Apache-2.0
// The field kernels: the copy of the input into the padded working field, the row partials and
// their per-element totals, the element state, the end-of-mode decision with the loop condition,
// the residue row and the collect of the output rows.
#include "runtime/error.hpp"
#include "field/common.cuh"
#include "field/field_ops.hpp"
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

// ---------------------------------------------------------------------------------------------
template <class T>
__global__ void __launch_bounds__(kThreads) eda_field_ops_canonical_in(const T* __restrict__ x, InputStrides xs, Geom g,
                                                                       T* __restrict__ r) {
  const int64_t total = g.batch * g.field;
  const int32_t nl = g.n[g.dim - 1];
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t q = idx / g.pitch;
    const int32_t t = static_cast<int32_t>(idx - q * g.pitch);
    T v = T(0);
    if (t < nl) {
      const int64_t b = q / g.rows;
      int64_t rem = q - b * g.rows;
      int64_t off = b * xs.s[0] + static_cast<int64_t>(t) * xs.s[g.dim];
      for (int a = g.dim - 2; a >= 0; --a) {
        const int64_t i = rem % g.n[a];
        rem /= g.n[a];
        off += i * xs.s[a + 1];
      }
      v = x[off];
    }
    r[idx] = v;
  }
}

template <class T>
__global__ void __launch_bounds__(kThreads) eda_field_ops_row_partials(const T* __restrict__ f, Geom g, RowPartial* rows,
                                                                       ElemState* elems) {
  const int lane = threadIdx.x & 31;
  const int64_t total = g.batch * g.rows;
  const int32_t n = g.n[g.dim - 1];
  const int64_t warps = static_cast<int64_t>(gridDim.x) * (kThreads / 32);
  for (int64_t q = blockIdx.x * static_cast<int64_t>(kThreads / 32) + (threadIdx.x >> 5); q < total; q += warps) {
    const T* row = f + q * g.pitch;
    const RowPartial p = warp_row_partial<T>([&](int32_t e) { return row[e]; }, n);
    if (lane == 0) rows[q] = p;
    if (elems) {
      unsigned bad = 0;
      for (int32_t e = lane; e < n; e += 32) bad += isfinite(static_cast<double>(row[e])) ? 0u : 1u;
      for (int d = 16; d > 0; d >>= 1) bad += __shfl_down_sync(0xffffffffu, bad, d);
      if (lane == 0 && bad) atomicAdd(reinterpret_cast<unsigned long long*>(&elems[q / g.rows].nonfinite),
                                      static_cast<unsigned long long>(bad));
    }
  }
}

__global__ void __launch_bounds__(kReduceThreads) eda_field_ops_reduce_partials(Geom g, const RowPartial* rows, ElemState* elems) {
  __shared__ double leaf[kReduceLeaf];
  __shared__ uint64_t amax_smem[kReduceThreads / 32];
  for (int64_t b = blockIdx.x; b < g.batch; b += gridDim.x) {
    const RowPartial* base = rows + b * g.rows;
    const ElemTotal t = block_reduce_rows([&](int64_t i) { return base[i]; }, g.rows, leaf, amax_smem);
    if (threadIdx.x == 0) {
      elems[b].amax = bits_to_double(t.amax_bits);
      elems[b].r_energy = t.sum_sq;  // written by this reduction, kept so the kernels stay unchanged
    }
  }
}

template <class T>
__global__ void __launch_bounds__(kThreads) eda_field_ops_init_state(Geom g, ElemState* elems, const CallCtl* ctl) {
  const int64_t b = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (b >= g.batch) return;
  ElemState& e = elems[b];
  e.n_imfs = 0;
  e.window = 0;
  e.tol = tie_tolerance<T>(ctl->tau, e.amax);
  if (e.nonfinite != 0 && ctl->check_finite) {
    e.status = kError;
    e.error = kErrNonFinite;
  } else {
    e.status = kActive;
    e.error = kErrNone;
  }
}

template <class T>
__global__ void __launch_bounds__(kReduceThreads) eda_field_ops_decide(Geom g, ElemState* elems, CallCtl* ctl,
                                                                       const RowPartial* imf_rows, const RowPartial* r_rows) {
  __shared__ double leaf[kReduceLeaf];
  __shared__ uint64_t amax_smem[kReduceThreads / 32];
  const int64_t b = blockIdx.x;
  if (b >= g.batch) return;  // the grid is the batch: never taken
  ElemState& e = elems[b];
  const bool active = e.status == kActive;  // uniform over the CTA
  if (active) {
    const RowPartial* ib = imf_rows + b * g.rows;
    const RowPartial* rb = r_rows + b * g.rows;
    const ElemTotal ti = block_reduce_rows([&](int64_t i) { return ib[i]; }, g.rows, leaf, amax_smem);
    const ElemTotal tr = block_reduce_rows([&](int64_t i) { return rb[i]; }, g.rows, leaf, amax_smem);
    if (threadIdx.x == 0) {
      const int32_t k = e.n_imfs;
      const int64_t h = b * (ctl->rows_cap - 1) + k;
      ctl->hist_scale[h] = e.window;
      ctl->hist_energy[h] = ti.sum_sq;
      e.n_imfs = k + 1;
      e.amax = bits_to_double(tr.amax_bits);
      e.r_energy = tr.sum_sq;  // written by this reduction, kept so the kernels stay unchanged
      e.tol = tie_tolerance<T>(ctl->tau, e.amax);
      if (ctl->max_imf > 0 && e.n_imfs >= ctl->max_imf) {
        e.status = kDone;
      } else if (e.n_imfs + 1 >= ctl->rows_cap) {
        e.status = kFull;
      }
    }
  }
  if (threadIdx.x == 0) {
    // Every element reports its state into the accumulators (integer atomics: order independent).
    // The last CTA to take a ticket sees every report after the fence, publishes the loop
    // condition and resets the accumulators and the ticket, so the next mode needs no memset and
    // no host synchronization.
    const int32_t status = e.status;
    if (status == kActive) atomicOr(&ctl->acc_active, 1);
    if (status == kFull) atomicOr(&ctl->acc_full, 1);
    __threadfence();
    const unsigned ticket = atomicAdd(&ctl->ticket, 1u);
    if (ticket == gridDim.x - 1) {
      __threadfence();
      const int32_t any_active = atomicAdd(&ctl->acc_active, 0);
      const int32_t any_full = atomicAdd(&ctl->acc_full, 0);
      ctl->any_active = any_active;
      ctl->any_full = any_full;
      ctl->acc_active = 0;
      ctl->acc_full = 0;
      ctl->ticket = 0;
      __threadfence();
    }
  }
}

template <class T>
__global__ void __launch_bounds__(kThreads) eda_field_ops_write_residue(Geom g, const ElemState* elems, const CallCtl* ctl,
                                                                        const T* __restrict__ r) {
  const int32_t nl = g.n[g.dim - 1];
  const int64_t total = g.batch * g.rows * nl;
  if (total == 0) return;
  T* out = static_cast<T*>(ctl->out);
  const int64_t rows_cap = ctl->rows_cap;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t q = idx / nl;
    const int32_t x = static_cast<int32_t>(idx - q * nl);
    const int64_t b = q / g.rows;
    const ElemState& e = elems[b];
    if (e.status == kError) continue;
    const int64_t row = q - b * g.rows;
    out[(b * rows_cap + e.n_imfs) * g.count + row * nl + x] = r[q * g.pitch + x];
  }
}

template <class T>
__global__ void __launch_bounds__(kThreads) eda_field_ops_collect(Geom g, const int32_t* n_imfs, const int32_t* error,
                                                                  const T* __restrict__ src, int32_t rows_cap, T* __restrict__ dst,
                                                                  int32_t P) {
  const int64_t total = g.batch * P * g.count;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t i = idx % g.count;
    const int64_t bk = idx / g.count;
    const int64_t k = bk % P;
    const int64_t b = bk / P;
    T v;
    if (error[b] != 0) {
      v = static_cast<T>(__longlong_as_double(0x7ff8000000000000LL));
    } else if (k <= n_imfs[b]) {
      v = src[(b * rows_cap + k) * g.count + i];
    } else {
      v = T(0);
    }
    dst[idx] = v;
  }
}

}  // namespace

// ---------------------------------------------------------------------------------------------
template <class T>
void enqueue_canonical_in(cudaStream_t s, const Geom& g, const T* x, const InputStrides& xs, T* r) {
  eda_field_ops_canonical_in<T><<<grid_for(g.batch * g.field, kThreads * 4), kThreads, 0, s>>>(x, xs, g, r);
  EDA_CHECK_LAUNCH("eda_field_ops_canonical_in");
}

template <class T>
void enqueue_row_partials(cudaStream_t s, const Geom& g, const T* f, RowPartial* rows, ElemState* elems) {
  eda_field_ops_row_partials<T><<<grid_for(g.batch * g.rows, kThreads / 32), kThreads, 0, s>>>(f, g, rows, elems);
  EDA_CHECK_LAUNCH("eda_field_ops_row_partials");
}

void enqueue_reduce_partials(cudaStream_t s, const Geom& g, const RowPartial* rows, ElemState* elems) {
  eda_field_ops_reduce_partials<<<grid_for(g.batch, 1), kReduceThreads, 0, s>>>(g, rows, elems);
  EDA_CHECK_LAUNCH("eda_field_ops_reduce_partials");
}

template <class T>
void enqueue_init_state(cudaStream_t s, const Geom& g, ElemState* elems, const CallCtl* ctl) {
  eda_field_ops_init_state<T><<<grid_for(g.batch, kThreads), kThreads, 0, s>>>(g, elems, ctl);
  EDA_CHECK_LAUNCH("eda_field_ops_init_state");
}

template <class T>
void enqueue_decide(cudaStream_t s, const Geom& g, ElemState* elems, CallCtl* ctl, const RowPartial* imf_rows,
                    const RowPartial* r_rows) {
  if (g.batch > INT32_MAX) throw std::invalid_argument("eda: batch too large");
  eda_field_ops_decide<T><<<static_cast<unsigned>(g.batch), kReduceThreads, 0, s>>>(g, elems, ctl, imf_rows, r_rows);
  EDA_CHECK_LAUNCH("eda_field_ops_decide");
}

template <class T>
void enqueue_write_residue(cudaStream_t s, const Geom& g, const ElemState* elems, const CallCtl* ctl, const T* r) {
  eda_field_ops_write_residue<T><<<grid_for(g.batch * g.rows * g.n[g.dim - 1], kThreads * 4), kThreads, 0, s>>>(g, elems, ctl, r);
  EDA_CHECK_LAUNCH("eda_field_ops_write_residue");
}

template <class T>
void enqueue_collect(cudaStream_t s, const Geom& g, const int32_t* n_imfs, const int32_t* error, const T* src,
                     int32_t rows_cap, T* dst, int32_t P) {
  eda_field_ops_collect<T><<<grid_for(g.batch * P * g.count, kThreads * 4), kThreads, 0, s>>>(g, n_imfs, error, src, rows_cap, dst, P);
  EDA_CHECK_LAUNCH("eda_field_ops_collect");
}

#define EDA_FIELD_OPS_INST(T)                                                                                  \
  template void enqueue_canonical_in<T>(cudaStream_t, const Geom&, const T*, const InputStrides&, T*);             \
  template void enqueue_row_partials<T>(cudaStream_t, const Geom&, const T*, RowPartial*, ElemState*);             \
  template void enqueue_init_state<T>(cudaStream_t, const Geom&, ElemState*, const CallCtl*);                        \
  template void enqueue_decide<T>(cudaStream_t, const Geom&, ElemState*, CallCtl*, const RowPartial*,                \
                                  const RowPartial*);                                                             \
  template void enqueue_write_residue<T>(cudaStream_t, const Geom&, const ElemState*, const CallCtl*, const T*);     \
  template void enqueue_collect<T>(cudaStream_t, const Geom&, const int32_t*, const int32_t*, const T*, int32_t, \
                                   T*, int32_t);
EDA_FIELD_OPS_INST(float)
EDA_FIELD_OPS_INST(double)
#undef EDA_FIELD_OPS_INST

}  // namespace eda
