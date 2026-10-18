// SPDX-License-Identifier: Apache-2.0
// The scale field with the absence rule, its face-connected components and the per-voxel floor of
// method/stats.hpp.
//
// Components: union-find over the padded in-element indices. Every voxel of the mask links with its
// +1 neighbour along every axis (the border does not connect); a link always hangs the larger root
// under the smaller one by atomicMin, so every parent is smaller than its child and the root of a
// component is its smallest padded index, whatever the order of the atomics (Playne and Hawick's
// lock-free union). A final pass writes every voxel's root (path compression).
#include "runtime/memory.hpp"
#include "method/stats_common.cuh"

#include <cmath>
#include <stdexcept>

namespace eda {
namespace stats_detail {
namespace {

// ---- components ----------------------------------------------------------------------------------
__device__ __forceinline__ int32_t find_root(const int32_t* L, int32_t x) {
  const volatile int32_t* V = L;
  int32_t p = V[x];
  while (p != x) {
    x = p;
    p = V[x];
  }
  return x;
}

__device__ __forceinline__ void unite(int32_t* L, int32_t a, int32_t c) {
  for (;;) {
    a = find_root(L, a);
    c = find_root(L, c);
    if (a == c) return;
    if (a > c) {
      const int32_t t = a;
      a = c;
      c = t;
    }
    const int32_t old = atomicMin(L + c, a);  // hang the larger root c under a
    if (old == c) return;
    c = old;  // c had been linked meanwhile: unite a with its new parent
  }
}

__global__ void eda_stats_scale_ccl_init(Geom g, const uint8_t* mask, uint8_t bits, int32_t* labels,
                                         const int32_t* on) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= g.batch * g.field) return;
  const int64_t b = i / g.field;
  if (!runs(on, b)) return;
  const int64_t v = i - b * g.field;
  labels[i] = (!pad_at(g, v) && (mask[i] & bits) != 0) ? static_cast<int32_t>(v) : -1;
}

__global__ void eda_stats_scale_ccl_merge(Geom g, int32_t* labels, const int32_t* on) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= g.batch * g.field) return;
  const int64_t b = i / g.field;
  if (!runs(on, b)) return;
  int32_t* L = labels + b * g.field;
  const int32_t v = static_cast<int32_t>(i - b * g.field);
  if (L[v] < 0) return;
  for (int a = 0; a < g.dim; ++a) {
    if (coord_of(g, v, a) + 1 >= g.n[a]) continue;
    const int32_t u = v + static_cast<int32_t>(a == g.dim - 1 ? 1 : g.stride[a]);
    if (L[u] < 0) continue;  // membership never changes during the merge
    unite(L, v, u);
  }
}

__global__ void eda_stats_scale_ccl_compress(Geom g, int32_t* labels, const int32_t* on) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= g.batch * g.field) return;
  const int64_t b = i / g.field;
  if (!runs(on, b)) return;
  int32_t* L = labels + b * g.field;
  const int32_t v = static_cast<int32_t>(i - b * g.field);
  if (L[v] < 0) return;
  const int32_t root = find_root(L, v);
  reinterpret_cast<volatile int32_t*>(L)[v] = root;
}

// ---- scale field ----------------------------------------------------------------------------------
__global__ void eda_stats_scale_sf_windows(int64_t B, double reach, const ElemRecord* le, int32_t* w_near, int32_t* w_box,
                                           const int32_t* on) {
  const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= B) return;
  const bool run = runs(on, b) && le[b].W >= 1;
  const int32_t W = le[b].W;
  w_near[b] = run ? odd_window(__dmul_rn(__dmul_rn(2.0, reach), static_cast<double>(W))) : 0;  // odd(2 reach W)
  w_box[b] = run ? W : 0;
}

// A bits: 1 << 1 the weak set Aw = om > lo w, 1 << 2 A0 = om > reset w; tmp0 = fine = om < lo w / jump;
// marks cleared.
template <class T>
__global__ void eda_stats_scale_sf_masks(Geom g, double reset, double lo, const ElemRecord* le, const T* om, uint8_t* A,
                                         T* fine, uint8_t* marks, const int32_t* w_box) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= g.batch * g.field) return;
  const int64_t b = i / g.field;
  if (w_box[b] < 1) return;
  marks[i] = 0;
  if (pad_at(g, i % g.field)) {
    A[i] = 0;
    fine[i] = T(0);
    return;
  }
  const double w = static_cast<double>(le[b].w);
  const double o = static_cast<double>(om[i]);
  const bool a0 = o > __dmul_rn(reset, w);
  const bool aw = o > __dmul_rn(lo, w);
  const bool fn = o < __ddiv_rn(__dmul_rn(lo, w), le[b].jump);  // (lo w) / jump
  A[i] = static_cast<uint8_t>((a0 ? 4u : 0u) | (aw ? 2u : 0u));
  fine[i] = fn ? T(1) : T(0);
}

template <class T>
__global__ void eda_stats_scale_sf_mark(Geom g, const uint8_t* A, const T* near, const int32_t* labels, uint8_t* marks,
                                        const int32_t* w_box) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= g.batch * g.field) return;
  const int64_t b = i / g.field;
  if (w_box[b] < 1) return;
  if ((A[i] & 2u) && near[i] > T(0.5)) marks[b * g.field + labels[i]] = 1;  // same value from every writer
}

// A = A0 and the voxel's component of Aw meets near; lw = log(where(A, absent, clip(om_w, w / c_lo, c_hi w)))
template <class T>
__global__ void eda_stats_scale_sf_logw(Geom g, MethodCtl c, const ElemRecord* le, const T* om, uint8_t* A,
                                        const int32_t* labels, const uint8_t* marks, T* lw, const int32_t* w_box) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= g.batch * g.field) return;
  const int64_t b = i / g.field;
  if (w_box[b] < 1) return;
  if (pad_at(g, i % g.field)) {
    lw[i] = T(0);
    return;
  }
  const uint8_t m = A[i];
  const int32_t lab = labels[i];
  const bool a = (m & 4u) && lab >= 0 && marks[b * g.field + lab];
  A[i] = a ? 1 : 0;
  const double w = static_cast<double>(le[b].w);
  double v;
  if (a) {
    v = c.absent;
  } else {
    const double o = static_cast<double>(om[i]);
    const double ow = o > __dmul_rn(c.reset, w) ? w : o;  // om_w
    const double lo = __ddiv_rn(w, c.c_lo);
    const double hi = __dmul_rn(c.c_hi, w);
    v = ow > lo ? ow : lo;  // clip: min(max(v, lo), hi)
    v = v < hi ? v : hi;
  }
  lw[i] = log_t<T>(static_cast<T>(v));
}

template <class T>
__global__ void eda_stats_scale_sf_exp(Geom g, T* L, const int32_t* w_box) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= g.batch * g.field) return;
  if (w_box[i / g.field] < 1 || pad_at(g, i % g.field)) return;
  L[i] = exp_t<T>(L[i]);
}

// ---- floor ------------------------------------------------------------------------------------------
__global__ void eda_stats_scale_floor_windows(int64_t B, const ElemRecord* le, int32_t* w_box, const int32_t* on) {
  const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= B) return;
  w_box[b] = (runs(on, b) && le[b].k >= 1 && le[b].W >= 1) ? le[b].W : 0;
}

template <class T>
__global__ void eda_stats_scale_floor_chi(Geom g, const uint8_t* A, T* chi, const int32_t* w_box) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= g.batch * g.field) return;
  if (w_box[i / g.field] < 1) return;
  chi[i] = (!pad_at(g, i % g.field) && (A[i] & 1u)) ? T(1) : T(0);
}

template <class T>
__global__ void eda_stats_scale_floor_apply(Geom g, MethodCtl c, double log_absent, const ElemRecord* le, const T* L_prev,
                                            const T* chi, T* L, const int32_t* w_box) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= g.batch * g.field) return;
  const int64_t b = i / g.field;
  if (w_box[b] < 1 || pad_at(g, i % g.field)) return;
  const T lp = L_prev[i];
  const bool grow = c.growth > 1.0 && le[b].k >= c.growth_from;
  T fl = grow ? mul_rn(static_cast<T>(c.growth), lp) : add_rn(lp, T(2));
  const T cap = static_cast<T>(__dmul_rn(c.c_hi, static_cast<double>(le[b].w)));
  fl = fl < cap ? fl : cap;  // min(fl, c_hi w)
  const T x = chi[i];
  const T e = add_rn(mul_rn(sub_rn(T(1), x), log_t<T>(fl)), mul_rn(x, static_cast<T>(log_absent)));
  const T f = exp_t<T>(e);
  const T l = L[i];
  L[i] = l >= f ? l : f;  // max(L, fl)
}

}  // namespace
}  // namespace stats_detail

size_t ccl_workspace_bytes(const Geom& g) { return stats_detail::ccl_ws_bytes(g); }

void enqueue_ccl(cudaStream_t s, const Geom& g, const uint8_t* mask, uint8_t bits, int32_t* labels,
                 void* ccl_ws, const int32_t* on) {
  using namespace stats_detail;
  (void)ccl_ws;  // the components need no workspace beyond labels
  if (!mask || !labels) throw std::invalid_argument("eda: ccl: null pointer");
  const unsigned nb = blocks_for(g.batch * g.field);
  eda_stats_scale_ccl_init<<<nb, kThreads, 0, s>>>(g, mask, bits, labels, on);
  EDA_CHECK_LAUNCH("eda_stats_scale_ccl_init");
  eda_stats_scale_ccl_merge<<<nb, kThreads, 0, s>>>(g, labels, on);
  EDA_CHECK_LAUNCH("eda_stats_scale_ccl_merge");
  eda_stats_scale_ccl_compress<<<nb, kThreads, 0, s>>>(g, labels, on);
  EDA_CHECK_LAUNCH("eda_stats_scale_ccl_compress");
}

template <class T>
void enqueue_scale_field(cudaStream_t s, const Geom& g, const MethodCtl& c, const ElemRecord* le, const T* om,
                         uint8_t* A, T* L, T* tmp0, T* tmp1, int32_t* labels, void* ccl_ws, const int32_t* on) {
  using namespace stats_detail;
  (void)tmp1;
  if (!le || !om || !A || !L || !tmp0 || !labels || !ccl_ws) throw std::invalid_argument("eda: scale_field: null");
  if (L == om || tmp0 == om || tmp0 == L) throw std::invalid_argument("eda: scale_field: aliased fields");
  const unsigned nb = blocks_for(g.batch * g.field);
  int32_t* w_near = ccl_win(ccl_ws, g, 0);
  int32_t* w_box = ccl_win(ccl_ws, g, 1);
  uint8_t* marks = ccl_marks(ccl_ws);
  eda_stats_scale_sf_windows<<<blocks_for(g.batch), kThreads, 0, s>>>(g.batch, c.absent_reach, le, w_near, w_box, on);
  EDA_CHECK_LAUNCH("eda_stats_scale_sf_windows");
  eda_stats_scale_sf_masks<T><<<nb, kThreads, 0, s>>>(g, c.reset, c.absent_lo, le, om, A, tmp0, marks, w_box);
  EDA_CHECK_LAUNCH("eda_stats_scale_sf_masks");
  enqueue_cube<T>(s, g, CubeOp::kMax, tmp0, tmp0, nullptr, w_near);  // near = Dil fine
  enqueue_ccl(s, g, A, 2u, labels, ccl_ws, w_box);
  eda_stats_scale_sf_mark<T><<<nb, kThreads, 0, s>>>(g, A, tmp0, labels, marks, w_box);
  EDA_CHECK_LAUNCH("eda_stats_scale_sf_mark");
  eda_stats_scale_sf_logw<T><<<nb, kThreads, 0, s>>>(g, c, le, om, A, labels, marks, L, w_box);
  EDA_CHECK_LAUNCH("eda_stats_scale_sf_logw");
  enqueue_cube<T>(s, g, CubeOp::kBox, L, L, nullptr, w_box);
  eda_stats_scale_sf_exp<T><<<nb, kThreads, 0, s>>>(g, L, w_box);
  EDA_CHECK_LAUNCH("eda_stats_scale_sf_exp");
}

template <class T>
void enqueue_local_floor(cudaStream_t s, const Geom& g, const MethodCtl& c, const ElemRecord* le, const T* L_prev,
                         const uint8_t* A, T* L, T* tmp, const int32_t* on) {
  using namespace stats_detail;
  if (!le || !L_prev || !A || !L || !tmp) throw std::invalid_argument("eda: local_floor: null pointer");
  if (tmp == L || tmp == L_prev) throw std::invalid_argument("eda: local_floor: aliased fields");
  // The signature has no workspace for the per-element window array of the box: B int32,
  // stream-ordered from the device pool (no host synchronisation).
  int device = 0;
  EDA_CUDA(cudaGetDevice(&device));
  DeviceBuffer<int32_t> w_box(static_cast<size_t>(g.batch), s, device);
  const unsigned nb = blocks_for(g.batch * g.field);
  eda_stats_scale_floor_windows<<<blocks_for(g.batch), kThreads, 0, s>>>(g.batch, le, w_box.data(), on);
  EDA_CHECK_LAUNCH("eda_stats_scale_floor_windows");
  eda_stats_scale_floor_chi<T><<<nb, kThreads, 0, s>>>(g, A, tmp, w_box.data());
  EDA_CHECK_LAUNCH("eda_stats_scale_floor_chi");
  enqueue_cube<T>(s, g, CubeOp::kBox, tmp, tmp, nullptr, w_box.data());
  const double log_absent = std::log(c.absent);  // in double with the C library, rounded to T in the kernel
  eda_stats_scale_floor_apply<T><<<nb, kThreads, 0, s>>>(g, c, log_absent, le, L_prev, tmp, L, w_box.data());
  EDA_CHECK_LAUNCH("eda_stats_scale_floor_apply");
}

#define EDA_STATS_SCALE_INST(T)                                                                                 \
  template void enqueue_scale_field<T>(cudaStream_t, const Geom&, const MethodCtl&, const ElemRecord*,          \
                                            const T*, uint8_t*, T*, T*, T*, int32_t*, void*, const int32_t*);      \
  template void enqueue_local_floor<T>(cudaStream_t, const Geom&, const MethodCtl&, const ElemRecord*,          \
                                            const T*, const uint8_t*, T*, T*, const int32_t*);
EDA_STATS_SCALE_INST(float)
EDA_STATS_SCALE_INST(double)
#undef EDA_STATS_SCALE_INST

}  // namespace eda
