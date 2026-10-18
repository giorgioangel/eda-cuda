// SPDX-License-Identifier: Apache-2.0
// test_cube: the method's cube filter (method/method.hpp, enqueue_cube over the vHGW passes)
// bitwise against the vHGW host twin (twin.hpp):
// max, min, max+min and box means, 1D/2D/3D, float32 and float64, windows clamped by short axes,
// in-place passes, per-element windows and the window-0 skip (the element's output untouched), and
// forced lines per CTA (the passes' results do not depend on it), and segments of more than 32
// chunks (cube widths above 1024, the warp-carry path). The cubes of k rungs in one
// launch per pass (enqueue_cube_rungs, k = 1, 2, 5, 32) bitwise against k one-rung cubes, every
// byte of every rung block, with windows that differ per rung, zero windows and a NaN line. Plus
// the scalar helpers of method/method.hpp (odd_window, the median of two middle values).
#include "twin.hpp"
#include "field/field_ops.hpp"
#include "method/method.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace host_twin;

namespace {

Shape shape(int dim, int64_t a, int64_t b = 1, int64_t c = 1) {
  Shape s;
  s.dim = dim;
  s.n[0] = a;
  s.n[1] = b;
  s.n[2] = c;
  return s;
}

template <class T>
std::vector<T> twin(const Shape& sh, eda::CubeOp op, const std::vector<T>& f, int32_t w, bool second) {
  std::vector<T> out = f;
  if (op == eda::CubeOp::kBox) {
    for (int a = 0; a < sh.dim; ++a) vhgw_axis<T>(sh, a, w, kSum, out);
    const T fac = box_factor<T>(sh, w, 1.0);
    for (T& v : out) v = mul_rn(v, fac);
    return out;
  }
  const Op o = op == eda::CubeOp::kMax ? kMax : (op == eda::CubeOp::kMin ? kMin : (second ? kMin : kMax));
  for (int a = sh.dim - 1; a >= 0; --a) vhgw_axis<T>(sh, a, w, o, out);
  return out;
}

template <class T>
void check_case(const Shape& sh, eda::CubeOp op, const std::vector<int32_t>& wins, bool in_place,
                int32_t lines_per_cta = 0) {
  const int64_t B = static_cast<int64_t>(wins.size());
  const eda::Geom g = eda::make_geom(sh.dim, B, sh.n);
  cudaStream_t s = nullptr;
  std::vector<T> x;
  for (int64_t b = 0; b < B; ++b) {
    const std::vector<T> f = make_field<T>(sh, static_cast<int>(b % 3), 1234 + 7 * b);
    x.insert(x.end(), f.begin(), f.end());
  }
  T *dx = nullptr, *in = nullptr, *o0 = nullptr, *o1 = nullptr;
  int32_t* dw = nullptr;
  const size_t field_bytes = static_cast<size_t>(B * g.field) * sizeof(T);
  cudaMalloc(&dx, x.size() * sizeof(T));
  cudaMalloc(&in, field_bytes);
  cudaMalloc(&o0, field_bytes);
  cudaMalloc(&o1, field_bytes);
  cudaMalloc(&dw, wins.size() * sizeof(int32_t));
  cudaMemcpy(dx, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(dw, wins.data(), wins.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemset(o0, 0xff, field_bytes);  // NaN sentinels: a skipped element keeps them
  cudaMemset(o1, 0xff, field_bytes);
  eda::InputStrides xs{};
  xs.s[sh.dim] = 1;
  for (int a = sh.dim - 1; a >= 1; --a) xs.s[a] = xs.s[a + 1] * g.n[a];
  xs.s[0] = g.count;
  eda::enqueue_canonical_in<T>(s, g, dx, xs, in);
  if (in_place && op != eda::CubeOp::kMaxMin) {
    cudaMemcpy(o0, in, field_bytes, cudaMemcpyDeviceToDevice);
    eda::enqueue_cube<T>(s, g, op, o0, o0, nullptr, dw, lines_per_cta);
  } else {
    eda::enqueue_cube<T>(s, g, op, in, o0, op == eda::CubeOp::kMaxMin ? o1 : nullptr, dw, lines_per_cta);
  }
  const int32_t nl = g.n[sh.dim - 1];
  std::vector<T> y0(x.size()), y1(x.size());
  cudaMemcpy2D(y0.data(), nl * sizeof(T), o0, g.pitch * sizeof(T), nl * sizeof(T), static_cast<size_t>(B * g.rows),
               cudaMemcpyDeviceToHost);
  cudaMemcpy2D(y1.data(), nl * sizeof(T), o1, g.pitch * sizeof(T), nl * sizeof(T), static_cast<size_t>(B * g.rows),
               cudaMemcpyDeviceToHost);
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  const size_t cnt = static_cast<size_t>(sh.count());
  for (int64_t b = 0; b < B; ++b) {
    const T* got0 = y0.data() + b * cnt;
    const T* got1 = y1.data() + b * cnt;
    if (wins[b] == 0) {
      bool untouched = true;
      for (size_t i = 0; i < cnt; ++i) untouched = untouched && std::isnan(static_cast<double>(got0[i]));
      CHECK(untouched);
      continue;
    }
    const std::vector<T> f(x.begin() + b * cnt, x.begin() + (b + 1) * cnt);
    const std::vector<T> want0 = twin<T>(sh, op, f, wins[b], false);
    const bool ok0 = same_bits(got0, want0.data(), cnt);
    CHECK(ok0);
    if (!ok0) std::fprintf(stderr, "  op %d dim %d b %lld w %d lines %d\n", static_cast<int>(op), sh.dim,
                           static_cast<long long>(b), wins[b], lines_per_cta);
    if (op == eda::CubeOp::kMaxMin) {
      const std::vector<T> want1 = twin<T>(sh, op, f, wins[b], true);
      CHECK(same_bits(got1, want1.data(), cnt));
    }
  }
  cudaFree(dx);
  cudaFree(in);
  cudaFree(o0);
  cudaFree(o1);
  cudaFree(dw);
}

template <class T>
std::vector<unsigned char> bytes_of(const T* d, size_t count) {
  std::vector<unsigned char> h(count * sizeof(T));
  CHECK(cudaMemcpy(h.data(), d, h.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
  return h;
}

// The cubes of k rungs of one input in one launch per pass against k one-rung cubes, byte for byte
// over each rung block (pads and skipped elements included); the gaps between the blocks stay
// untouched.
template <class T>
void check_rungs(const Shape& sh, eda::CubeOp op, int64_t B, int32_t k, int32_t lines_per_cta = 0) {
  const eda::Geom g = eda::make_geom(sh.dim, B, sh.n);
  const bool two = op == eda::CubeOp::kMaxMin;
  const int64_t fe = B * g.field;
  const int64_t stride = fe + 96;
  std::vector<T> x;
  for (int64_t b = 0; b < B; ++b) {
    const std::vector<T> f = make_field<T>(sh, static_cast<int>(b % 3), 4321 + 5 * b);
    x.insert(x.end(), f.begin(), f.end());
  }
  x[static_cast<size_t>(sh.count() / 2)] = std::numeric_limits<T>::quiet_NaN();  // a NaN line in element 0
  // windows per rung and element: odd sides from 1 to beyond 2 n, and zeros (the element skipped)
  std::vector<int32_t> win(static_cast<size_t>(k * B));
  for (int32_t j = 0; j < k; ++j) {
    for (int64_t b = 0; b < B; ++b) {
      const int64_t m = (7 * j + 3 * b) % 11;
      win[j * B + b] = (j + b) % 4 == 3 ? 0 : (m == 10 ? 301 : static_cast<int32_t>(2 * m + 1));
    }
  }
  T *dx = nullptr, *in = nullptr, *r0 = nullptr, *r1 = nullptr, *o0 = nullptr, *o1 = nullptr;
  int32_t* dw = nullptr;
  const size_t span = static_cast<size_t>(k * stride);
  cudaMalloc(&dx, x.size() * sizeof(T));
  cudaMalloc(&in, fe * sizeof(T));
  cudaMalloc(&r0, span * sizeof(T));
  cudaMalloc(&r1, span * sizeof(T));
  cudaMalloc(&o0, fe * sizeof(T));
  cudaMalloc(&o1, fe * sizeof(T));
  cudaMalloc(&dw, win.size() * sizeof(int32_t));
  cudaMemcpy(dx, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(dw, win.data(), win.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemset(r0, 0xff, span * sizeof(T));
  cudaMemset(r1, 0xff, span * sizeof(T));
  eda::InputStrides xs{};
  xs.s[sh.dim] = 1;
  for (int a = sh.dim - 1; a >= 1; --a) xs.s[a] = xs.s[a + 1] * g.n[a];
  xs.s[0] = g.count;
  cudaMemset(in, 0, fe * sizeof(T));
  eda::enqueue_canonical_in<T>(nullptr, g, dx, xs, in);
  eda::enqueue_cube_rungs<T>(nullptr, g, op, in, r0, two ? r1 : nullptr, dw, k, stride, lines_per_cta);
  const std::vector<unsigned char> got0 = bytes_of(r0, span), got1 = bytes_of(r1, span);
  bool same = true, gaps = true;
  for (int32_t j = 0; j < k; ++j) {
    cudaMemset(o0, 0xff, fe * sizeof(T));
    cudaMemset(o1, 0xff, fe * sizeof(T));
    eda::enqueue_cube<T>(nullptr, g, op, in, o0, two ? o1 : nullptr, dw + j * B);
    const std::vector<unsigned char> want0 = bytes_of(o0, fe), want1 = bytes_of(o1, fe);
    const size_t at = static_cast<size_t>(j * stride) * sizeof(T), len = static_cast<size_t>(fe) * sizeof(T);
    same = same && std::memcmp(got0.data() + at, want0.data(), len) == 0;
    if (two) same = same && std::memcmp(got1.data() + at, want1.data(), len) == 0;
    for (size_t i = at + len; i < at + static_cast<size_t>(stride) * sizeof(T); ++i) {
      gaps = gaps && got0[i] == 0xff && got1[i] == 0xff;
    }
  }
  if (!two) {
    for (unsigned char c : got1) gaps = gaps && c == 0xff;  // the one-field ops leave out1 alone
  }
  CHECK(same);
  CHECK(gaps);
  if (!same || !gaps) {
    std::fprintf(stderr, "  rungs: op %d dim %d B %lld k %d lines %d\n", static_cast<int>(op), sh.dim,
                 static_cast<long long>(B), k, lines_per_cta);
  }
  if (k > 1) {
    CHECK(eda_test::throws_with<std::invalid_argument>(
        [&] { eda::enqueue_cube_rungs<T>(nullptr, g, op, in, r0, two ? r1 : nullptr, dw, k, fe - 1); }, "overlap"));
    CHECK(eda_test::throws_with<std::invalid_argument>(
        [&] { eda::enqueue_cube_rungs<T>(nullptr, g, op, r0 + stride, r0, two ? r1 : nullptr, dw, k, stride); },
        "overlap"));
  }
  CHECK(eda_test::throws_with<std::invalid_argument>(
      [&] { eda::enqueue_cube_rungs<T>(nullptr, g, op, in, r0, two ? r1 : nullptr, dw, 0, stride); }, "rungs"));
  CHECK(cudaDeviceSynchronize() == cudaSuccess);
  cudaFree(dx);
  cudaFree(in);
  cudaFree(r0);
  cudaFree(r1);
  cudaFree(o0);
  cudaFree(o1);
  cudaFree(dw);
}

template <class T>
void all() {
  const eda::CubeOp ops[] = {eda::CubeOp::kMax, eda::CubeOp::kMin, eda::CubeOp::kMaxMin,
                                eda::CubeOp::kBox};
  for (eda::CubeOp op : ops) {
    check_case<T>(shape(1, 300), op, {1, 7, 0, 601}, false);
    check_case<T>(shape(2, 37, 45), op, {9, 0, 3, 91}, false);
    check_case<T>(shape(2, 5, 64), op, {13, 21}, true);
    check_case<T>(shape(3, 11, 9, 14), op, {5, 0, 27}, false);
    check_case<T>(shape(3, 4, 16, 6), op, {9, 3}, true);
    // Long segments: a cube width wa above 1024 runs the vHGW pass's warp-carry path (more than 32
    // chunks per segment). Widths 1023 and 1025 sit on both sides of the switch; 2999 is the whole
    // line (window beyond 2 n); the 2-D case puts the long segments on the outer axis.
    check_case<T>(shape(1, 1500), op, {1023, 1025, 1201, 2999}, false);
    check_case<T>(shape(1, 1500), op, {1025, 2999}, true);
    check_case<T>(shape(2, 1100, 5), op, {1025, 2201, 0}, false);
    for (int32_t lines : {1, 2, 3, 4, 7, 8, 16, 64}) {
      check_case<T>(shape(1, 300), op, {1, 7, 0, 601}, false, lines);
      check_case<T>(shape(2, 37, 45), op, {9, 0, 3, 91}, false, lines);
      check_case<T>(shape(3, 11, 9, 14), op, {5, 0, 27}, false, lines);
    }
    for (int32_t lines : {1, 3}) check_case<T>(shape(1, 1500), op, {1025, 2999}, false, lines);
    for (int32_t k : {1, 2, 5, 32}) {
      check_rungs<T>(shape(2, 37, 45), op, 3, k);
      if (k > 5) continue;
      check_rungs<T>(shape(1, 300), op, 3, k);
      check_rungs<T>(shape(3, 11, 9, 14), op, 2, k);
      check_rungs<T>(shape(3, 40, 5, 36), op, 2, k);
    }
    for (int32_t lines : {1, 4, 8, 64}) check_rungs<T>(shape(3, 11, 9, 14), op, 2, 5, lines);
  }
}

void helpers() {
  CHECK(eda::odd_window(0.2) == 1);
  CHECK(eda::odd_window(3.0) == 3);
  CHECK(eda::odd_window(4.0) == 5);
  CHECK(eda::odd_window(4.9) == 5);
  CHECK(eda::odd_window(6.0) == 7);
  CHECK(eda::median_pair(1.0, 2.0) == 1.5);
}

}  // namespace

int main() {
  try {
    eda::Context& ctx = eda::Context::current(0);
    (void)ctx;
    helpers();
    all<float>();
    all<double>();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "test_cube: %s\n", error.what());
    eda::Context::shutdown();
    return 1;
  }
  eda::Context::shutdown();
  return eda_test::finish("test_cube");
}
