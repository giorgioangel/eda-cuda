// SPDX-License-Identifier: Apache-2.0
// The operators of the variable element (Dil_B, Ero_B, Avg_B) and the envelope mean
// (method/element.hpp). Included by element_op_f32.cu and element_op_f64.cu (one translation unit
// per dtype).
//
// Per rung j (ascending, j < n_rungs_max): the cube of the rung (enqueue_cube with the rung's
// windows rung_win + j B: 0 skips the element; with the view's rung scratch the cubes of a group of
// rungs run first, in one launch per pass, and then the group's gathers in rung order), then the
// gather kernel, one thread per voxel:
//   - a voxel is on rung j when its weight there is > 0 in T: rung lo with 1 - t, rung lo + 1
//     with t (the stored t of enqueue_element_weights);
//   - acc = f(x); where bit j of rung_pts is set (D >= 2) also the 2 m points
//     x + (i / m) o, x - (i / m) o for i = 1..m (order +1/m, -1/m, +2/m, -2/m), each read by D-linear
//     interpolation of the half-sample reflected periodic extension of f: per axis i0 = floor(y),
//     t = y - i0 in double (rounded to T for the weights, 1 - t in T), the 2^D corners summed
//     lexicographically (last axis fastest) from 0, each term ((f w_0) w_1) w_2;
//   - max / min (max_c / min_c), or the left-to-right sum, then for box with points a division by
//     2 m + 1;
//   - the voxel's first rung with a positive weight stores weight * acc, the second adds it.
// Reads go through __ldg (the inputs are read-only during a kernel); every operation is a
// round-to-nearest intrinsic, so the gather matches its host twin bit for bit.
#pragma once

#include "runtime/error.hpp"
#include "field/common.cuh"
#include "method/method.hpp"
#include "method/element.cuh"
#include "method/element.hpp"

#include <cstdint>
#include <stdexcept>

namespace eda::element_detail {

// Gather ops (the cube op of the rung: max, max+min, box).
inline constexpr int kGMax = 0, kGMaxMin = 2, kGBox = 3;

template <class T, int D>
struct Interp {
  int64_t off0[D], off1[D];  // reflected corner offsets (times the axis stride)
  T w0[D], w1[D];            // 1 - t and t per axis
};

template <class T, int D>
__device__ __forceinline__ Interp<T, D> interp_setup(const Geom& g, const double* y) {
  Interp<T, D> ip;
#pragma unroll
  for (int a = 0; a < D; ++a) {
    const double fl = floor(y[a]);
    const double tt = __dsub_rn(y[a], fl);
    const int64_t i0 = static_cast<int64_t>(fl);
    const T t1 = static_cast<T>(tt);
    ip.w1[a] = t1;
    ip.w0[a] = sub_rn(T(1), t1);
    ip.off0[a] = reflect_any(i0, g.n[a]) * g.stride[a];
    ip.off1[a] = reflect_any(i0 + 1, g.n[a]) * g.stride[a];
  }
  return ip;
}

template <class T, int D>
__device__ __forceinline__ T interp_read(const Interp<T, D>& ip, const T* __restrict__ f) {
  T acc = T(0);
#pragma unroll
  for (int corner = 0; corner < (1 << D); ++corner) {
    int64_t off = 0;
#pragma unroll
    for (int a = 0; a < D; ++a) off += ((corner >> (D - 1 - a)) & 1) ? ip.off1[a] : ip.off0[a];
    T term = __ldg(f + off);
#pragma unroll
    for (int a = 0; a < D; ++a) term = mul_rn(term, ((corner >> (D - 1 - a)) & 1) ? ip.w1[a] : ip.w0[a]);
    acc = add_rn(acc, term);
  }
  return acc;
}

template <class T, int OP>
__device__ __forceinline__ T combine(T acc, T v) {
  if constexpr (OP == kGMax) return max_c(acc, v);
  else return add_rn(acc, v);
}

// One rung of an operator. f0 (f1: the min cube of kGMaxMin) the rung's cube field, out0 (out1).
// bpe CTAs per element stride over the element's padded field; the rung test is uniform per CTA.
template <class T, int D, int OP>
__global__ void __launch_bounds__(kThreads) eda_element_op_gather(Geom g, int j, const int32_t* __restrict__ rung_win,
                                                                  const ElemRecord* __restrict__ le,
                                                                  const uint8_t* __restrict__ lo,
                                                                  const T* __restrict__ t, FieldVec<const T> o,
                                                                  const T* __restrict__ f0, const T* __restrict__ f1,
                                                                  T* __restrict__ out0, T* __restrict__ out1,
                                                                  int bpe) {
  const int64_t b = blockIdx.x / bpe;
  if (b >= g.batch || rung_win[static_cast<int64_t>(j) * g.batch + b] == 0) return;
  const bool pts = D >= 2 && ((le[b].rung_pts >> j) & 1u) != 0u;
  const int64_t base = b * g.field;
  const T* e0 = f0 + base;
  const T* e1 = OP == kGMaxMin ? f1 + base : nullptr;
  for (int64_t p = static_cast<int64_t>(blockIdx.x % bpe) * kThreads + threadIdx.x; p < g.field;
       p += static_cast<int64_t>(bpe) * kThreads) {
    int32_t x[kMaxDim];
    if (!voxel_coords(g, p, x)) continue;
    const int64_t at = base + p;
    const int l = __ldg(lo + at);
    const T tt = __ldg(t + at);
    const T wl = sub_rn(T(1), tt);
    T wgt;
    bool add;
    if (l == j && wl > T(0)) {
      wgt = wl;
      add = false;
    } else if (l + 1 == j && tt > T(0)) {
      wgt = tt;
      add = wl > T(0);
    } else {
      continue;
    }
    T acc0 = __ldg(e0 + p);
    T acc1 = OP == kGMaxMin ? __ldg(e1 + p) : T(0);
    if (pts) {
      double ov[D];
#pragma unroll
      for (int a = 0; a < D; ++a) ov[a] = static_cast<double>(__ldg(o.p[a] + at));
#pragma unroll
      for (int i = 1; i <= kM; ++i) {
        const double s = static_cast<double>(i) / static_cast<double>(kM);
#pragma unroll
        for (int sg = 0; sg < 2; ++sg) {
          double y[D];
#pragma unroll
          for (int a = 0; a < D; ++a) {
            const double off = __dmul_rn(s, ov[a]);
            const double xa = static_cast<double>(x[a]);
            y[a] = sg == 0 ? __dadd_rn(xa, off) : __dsub_rn(xa, off);
          }
          const Interp<T, D> ip = interp_setup<T, D>(g, y);
          if constexpr (OP == kGMaxMin) {
            acc0 = max_c(acc0, interp_read<T, D>(ip, e0));
            acc1 = min_c(acc1, interp_read<T, D>(ip, e1));
          } else {
            acc0 = combine<T, OP>(acc0, interp_read<T, D>(ip, e0));
          }
        }
      }
      if constexpr (OP == kGBox) acc0 = div_t(acc0, T(2 * kM + 1));
    }
    const T v0 = mul_rn(wgt, acc0);
    out0[at] = add ? add_rn(out0[at], v0) : v0;
    if constexpr (OP == kGMaxMin) {
      const T v1 = mul_rn(wgt, acc1);
      out1[at] = add ? add_rn(out1[at], v1) : v1;
    }
  }
}

// mid = 0.5 (u + l) on the data voxels of the elements with a rung window.
template <class T>
__global__ void __launch_bounds__(kThreads) eda_element_op_mid(Geom g, const int32_t* __restrict__ rung_win,
                                                               int32_t n_rungs_max, const T* __restrict__ u,
                                                               const T* __restrict__ l, T* __restrict__ mid, int bpe) {
  const int64_t b = blockIdx.x / bpe;
  if (b >= g.batch) return;
  bool any = false;
  for (int j = 0; j < n_rungs_max; ++j) any = any || rung_win[static_cast<int64_t>(j) * g.batch + b] != 0;
  if (!any) return;
  const int64_t base = b * g.field;
  for (int64_t p = static_cast<int64_t>(blockIdx.x % bpe) * kThreads + threadIdx.x; p < g.field;
       p += static_cast<int64_t>(bpe) * kThreads) {
    int32_t x[kMaxDim];
    if (!voxel_coords(g, p, x)) continue;
    const int64_t at = base + p;
    mid[at] = mul_rn(T(0.5), add_rn(__ldg(u + at), __ldg(l + at)));
  }
}

// CTAs per element: one per 256 samples, at most 1024 (each CTA strides over its element).
inline int op_blocks_per_element(const Geom& g) {
  const int64_t need = (g.field + kThreads - 1) / kThreads;
  return static_cast<int>(need < 1024 ? (need < 1 ? 1 : need) : 1024);
}

template <class T, int D, int OP>
void launch_gather(cudaStream_t s, const Geom& g, int j, const ElemView<T>& ev, const T* f0, const T* f1,
                   T* out0, T* out1) {
  const int bpe = op_blocks_per_element(g);
  const dim3 grid(static_cast<unsigned>(static_cast<int64_t>(bpe) * g.batch));
  eda_element_op_gather<T, D, OP><<<grid, kThreads, 0, s>>>(g, j, ev.rung_win, ev.le, ev.lo, ev.t, ev.o, f0, f1,
                                                            out0, out1, bpe);
  EDA_CHECK_LAUNCH("eda_element_op_gather");
}

template <class T, int OP>
void launch_gather_d(cudaStream_t s, const Geom& g, int j, const ElemView<T>& ev, const T* f0, const T* f1,
                     T* out0, T* out1) {
  if (g.dim == 1) launch_gather<T, 1, OP>(s, g, j, ev, f0, f1, out0, out1);
  else if (g.dim == 2) launch_gather<T, 2, OP>(s, g, j, ev, f0, f1, out0, out1);
  else launch_gather<T, 3, OP>(s, g, j, ev, f0, f1, out0, out1);
}

template <class T>
void op_impl(cudaStream_t s, const Geom& g, ElemOp op, const ElemView<T>& ev, const T* in, T* out0, T* out1,
             T* tmp0, T* tmp1) {
  const bool two = op == ElemOp::kMaxMin;
  if (in == nullptr || out0 == nullptr || tmp0 == nullptr || (two && (out1 == nullptr || tmp1 == nullptr))) {
    throw std::invalid_argument("eda: element op: null field");
  }
  const void* ptrs[5] = {in, out0, tmp0, two ? out1 : nullptr, two ? tmp1 : nullptr};
  for (int a = 0; a < 5; ++a) {
    for (int c = a + 1; c < 5; ++c) {
      if (ptrs[a] != nullptr && ptrs[a] == ptrs[c]) {
        throw std::invalid_argument("eda: element op: the input, outputs and temporaries must not alias");
      }
    }
  }
  if (ev.n_rungs_max < 0 || ev.n_rungs_max > kMaxRungs) {
    throw std::invalid_argument("eda: element op: n_rungs_max outside 0..kMaxRungs");
  }
  if (ev.n_rungs_max == 0) return;
  if (ev.lo == nullptr || ev.t == nullptr || ev.rung_win == nullptr || ev.le == nullptr) {
    throw std::invalid_argument("eda: element op: incomplete element view");
  }
  for (int a = 0; a < g.dim && g.dim >= 2; ++a) {
    if (ev.o.p[a] == nullptr) throw std::invalid_argument("eda: element op: null segment component");
  }
  const CubeOp cop = op == ElemOp::kMax      ? CubeOp::kMax
                         : op == ElemOp::kMaxMin ? CubeOp::kMaxMin
                                                     : CubeOp::kBox;
  const auto gather = [&](int j, const T* f0, const T* f1) {
    switch (op) {
      case ElemOp::kMax: launch_gather_d<T, kGMax>(s, g, j, ev, f0, nullptr, out0, nullptr); break;
      case ElemOp::kMaxMin: launch_gather_d<T, kGMaxMin>(s, g, j, ev, f0, f1, out0, out1); break;
      case ElemOp::kBox: launch_gather_d<T, kGBox>(s, g, j, ev, f0, nullptr, out0, nullptr); break;
    }
  };
  const int64_t fe = g.batch * g.field;
  if (ev.rung_group > 1) {
    const int64_t span = static_cast<int64_t>(ev.rung_group) * fe;
    bool ok = ev.rung0 != nullptr && (!two || (ev.rung1 != nullptr && ranges_apart(ev.rung0, span, ev.rung1, span)));
    // Named arrays: nvcc 12.9 emits a braced list in a range-for as a comma expression.
    const T* const rungs[] = {ev.rung0, two ? ev.rung1 : nullptr};
    const T* const fields[] = {in, out0, out1};
    for (const T* r : rungs) {
      for (const T* f : fields) {
        ok = ok && ranges_apart(r, span, f, fe);
      }
    }
    if (!ok) throw std::invalid_argument("eda: element op: the rung scratch must not alias the input or the outputs");
    for (int j0 = 0; j0 < ev.n_rungs_max; j0 += ev.rung_group) {
      const int k = ev.n_rungs_max - j0 < ev.rung_group ? ev.n_rungs_max - j0 : ev.rung_group;
      enqueue_cube_rungs<T>(s, g, cop, in, ev.rung0, two ? ev.rung1 : nullptr, ev.rung_win + j0 * g.batch, k, fe);
      for (int j = j0; j < j0 + k; ++j) {
        const int64_t at = static_cast<int64_t>(j - j0) * fe;
        gather(j, ev.rung0 + at, two ? ev.rung1 + at : nullptr);
      }
    }
    return;
  }
  for (int j = 0; j < ev.n_rungs_max; ++j) {
    const int32_t* win = ev.rung_win + static_cast<int64_t>(j) * g.batch;
    enqueue_cube<T>(s, g, cop, in, tmp0, two ? tmp1 : nullptr, win);
    gather(j, tmp0, tmp1);
  }
}

template <class T>
void mean_impl(cudaStream_t s, const Geom& g, const ElemView<T>& ev, const T* h, T* m, T* u, T* l, T* tmp0,
               T* tmp1) {
  const void* ptrs[6] = {h, m, u, l, tmp0, tmp1};
  for (int a = 0; a < 6; ++a) {
    if (ptrs[a] == nullptr) throw std::invalid_argument("eda: envelope mean: null field");
    for (int c = a + 1; c < 6; ++c) {
      if (ptrs[a] == ptrs[c]) throw std::invalid_argument("eda: envelope mean: every pointer must be distinct");
    }
  }
  op_impl<T>(s, g, ElemOp::kMaxMin, ev, h, u, l, tmp0, tmp1);
  if (ev.n_rungs_max == 0) return;
  const int bpe = op_blocks_per_element(g);
  const dim3 grid(static_cast<unsigned>(static_cast<int64_t>(bpe) * g.batch));
  eda_element_op_mid<T><<<grid, kThreads, 0, s>>>(g, ev.rung_win, ev.n_rungs_max, u, l, tmp0, bpe);
  EDA_CHECK_LAUNCH("eda_element_op_mid");
  op_impl<T>(s, g, ElemOp::kBox, ev, tmp0, m, nullptr, tmp1, nullptr);
}

}  // namespace eda::element_detail

namespace eda {

template <class T>
void enqueue_element_op(cudaStream_t s, const Geom& g, ElemOp op, const ElemView<T>& ev, const T* in,
                        T* out0, T* out1, T* tmp0, T* tmp1) {
  element_detail::op_impl<T>(s, g, op, ev, in, out0, out1, tmp0, tmp1);
}

template <class T>
void enqueue_envelope_mean(cudaStream_t s, const Geom& g, const ElemView<T>& ev, const T* h, T* m, T* u,
                           T* l, T* tmp0, T* tmp1) {
  element_detail::mean_impl<T>(s, g, ev, h, m, u, l, tmp0, tmp1);
}

}  // namespace eda
