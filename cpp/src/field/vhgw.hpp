// SPDX-License-Identifier: Apache-2.0
// The van Herk / Gil-Werman max/min/box pass with canonical arithmetic: the separable building
// block of the cube filter (method/common.cpp, enqueue_cube). Every output depends only on the
// line's values, its length and the window, never on the launch shape (vhgw.cu).
#pragma once
#include "field/geom.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace eda {

// What the last stage of a pass writes.
enum VhgwEpilogue : int8_t {
  kEpiStore = 0,      // the stage result into out0 (max, box) and out1 (min)
  kEpiBoxScale = 1,   // out0 = box_sum * fl_T(1 / prod w)
};

struct VhgwPass {
  int32_t axis;
  bool from_single;   // input is one field h (U = max, L = min); else two fields U, L
  bool maxmin;        // max (min) along `axis`
  bool box;           // then box-sum along the same staged line
  int8_t op;          // -1: max into out0 and min into out1; else one field: 0 max, 1 min, 2 box
  int8_t epilogue;    // VhgwEpilogue
};
template <class T>
struct VhgwArgs {
  const T* in0; const T* in1;   // h, or U and L
  T* out0; T* out1;             // U, L (out0 only for the one-field ops)
  const int32_t* window;        // [B] odd windows (a window < 1 skips the element)
  int32_t lines_per_cta;        // 0 = auto; tests force values (results are bitwise identical)
  // rungs > 1: that many independent passes in one launch; pass j reads window + j B and
  // in0 / in1 + j in_stride and writes out0 / out1 + j out_stride (0 or 1: one pass)
  int32_t rungs;
  int64_t in_stride, out_stride;
};
template <class T>
void enqueue_vhgw(cudaStream_t s, const Geom& g, const VhgwPass& pass, const VhgwArgs<T>& a);

// The lines per CTA of a launch of `rungs` passes along `axis` (a forced value is clamped to what
// fits), and the longest line a pass supports.
template <class T>
int vhgw_lines_per_cta(const Geom& g, int axis, int32_t requested, int32_t rungs = 1);
template <class T>
constexpr int32_t vhgw_max_line() { return sizeof(T) == 4 ? 8192 : 4096; }

// How many rungs of a cube (one pass per axis and rung, two output fields per rung) to launch
// together on geometry g: 1 when one rung's passes already give the device a wave of resident
// CTAs on every axis; else as many as keep the group's output fields within half the L2 cache,
// at most max_group.
template <class T>
int32_t vhgw_rung_group(const Geom& g, int32_t max_group);
}  // namespace eda
