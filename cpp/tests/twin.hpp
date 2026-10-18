// SPDX-License-Identifier: Apache-2.0
// Host twins of the field kernels (the vHGW passes and the canonical reductions), written directly
// from their definitions with the same canonical arithmetic (no shared code with the kernels beyond
// the scalar helpers of field/common.cuh), so that GPU results compare bitwise. Plus small helpers
// of the C++ tests: deterministic random fields, flips, run wrappers.
#pragma once

#include "runtime/context.hpp"
#include "field/common.cuh"
#include "decompose.hpp"
#include "field/vhgw.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace host_twin {

using eda::add_rn;
using eda::floor_div;
using eda::floor_mod;
using eda::max_c;
using eda::min_c;
using eda::mul_rn;
using eda::reflect_index;
using eda::sub_rn;

enum Op { kMax = 0, kMin = 1, kSum = 2 };

template <class T>
T apply(Op op, T a, T b) {
  return op == kMax ? max_c(a, b) : (op == kMin ? min_c(a, b) : add_rn(a, b));
}

// One field (unpadded, C order) of shape n[0..D-1].
struct Shape {
  int dim = 1;
  int64_t n[3] = {1, 1, 1};
  int64_t count() const { return n[0] * (dim > 1 ? n[1] : 1) * (dim > 2 ? n[2] : 1); }
  int64_t stride(int a) const {
    int64_t s = 1;
    for (int b = a + 1; b < dim; ++b) s *= n[b];
    return s;
  }
};

// Applies f to every line along axis a: f(read(i), write(i, v), length).
template <class T>
void for_lines(const Shape& sh, int axis, const std::function<void(std::vector<T>&)>& f, std::vector<T>& data) {
  const int64_t n = sh.n[axis];
  const int64_t st = sh.stride(axis);
  const int64_t outer = sh.count() / (n * st);
  std::vector<T> line(static_cast<size_t>(n));
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t j = 0; j < st; ++j) {
      const int64_t base = o * n * st + j;
      for (int64_t i = 0; i < n; ++i) line[i] = data[base + i * st];
      f(line);
      for (int64_t i = 0; i < n; ++i) data[base + i * st] = line[i];
    }
  }
}

// ---- vHGW twin, O(n * w): the segment folds of field/vhgw.cu, sample by sample ----------------
template <class T>
struct Vhgw1d {
  const std::vector<T>& v;
  int32_t n, r, wa, o;
  Op op;
  Vhgw1d(const std::vector<T>& line, int32_t w, Op op_) : v(line), op(op_) {
    n = static_cast<int32_t>(line.size());
    r = std::min((w - 1) / 2, n - 1);
    wa = 2 * r + 1;
    o = static_cast<int32_t>((static_cast<int64_t>(n) * ((wa + 1) / 2)) % wa);
  }
  T ext(int32_t e, bool rev) const {
    int32_t i = reflect_index(e, n);
    if (rev) i = n - 1 - i;
    return v[i];
  }
  // Sequential fold of ext over [a, b].
  T fold(int32_t a, int32_t b, bool rev) const {
    T acc = ext(a, rev);
    for (int32_t e = a + 1; e <= b; ++e) acc = apply(op, acc, ext(e, rev));
    return acc;
  }
  // G at e of the line (rev: of the reversed line).
  T G(int32_t e, bool rev) const {
    const int64_t k = floor_div(static_cast<int64_t>(e) - o, wa);
    const int32_t s = static_cast<int32_t>(o + k * wa);
    const int32_t q = (e - s) / 32;
    const T local = fold(s + 32 * q, e, rev);
    if (q == 0) return local;
    T carry = fold(s, s + 31, rev);
    for (int32_t j = 1; j < q; ++j) carry = apply(op, carry, fold(s + 32 * j, s + 32 * j + 31, rev));
    return apply(op, carry, local);
  }
  T H(int32_t e) const { return G(n - 1 - e, true); }
  T y(int32_t x) const {
    const T h = H(x - r);
    const T g = G(x + r, false);
    if (op == kSum) {
      const bool aligned = floor_mod(static_cast<int64_t>(x) - r - o, wa) == 0;
      const T s = add_rn(h, g);
      return aligned ? mul_rn(s, T(0.5)) : s;
    }
    return apply(op, h, g);
  }
};

template <class T>
void vhgw_axis(const Shape& sh, int axis, int32_t w, Op op, std::vector<T>& data) {
  for_lines<T>(sh, axis, [&](std::vector<T>& line) {
    const std::vector<T> in = line;
    const Vhgw1d<T> f(in, w, op);
    for (int32_t x = 0; x < static_cast<int32_t>(line.size()); ++x) line[x] = f.y(x);
  }, data);
}

template <class T>
T box_factor(const Shape& sh, int32_t w, double c) {
  double prod = 1.0;
  for (int a = 0; a < sh.dim; ++a) prod *= static_cast<double>(2 * std::min<int64_t>((w - 1) / 2, sh.n[a] - 1) + 1);
  return static_cast<T>(c / prod);
}

// ---- reductions (field/reduce.hpp, field/common.cuh) ---------------------------------------
inline double pairwise(std::vector<double> s) {
  size_t m = 1;
  while (m < s.size()) m <<= 1;
  s.resize(std::max<size_t>(m, 1), 0.0);
  for (size_t d = 1; d < s.size(); d <<= 1) {
    for (size_t i = 0; i + d < s.size(); i += 2 * d) s[i] = s[i] + s[i + d];
  }
  return s[0];
}

template <class T>
double row_energy(const T* row, int64_t n) {
  std::vector<double> chunks;
  for (int64_t c = 0; c < n; c += 32) {
    double acc = 0.0;
    for (int64_t e = c; e < std::min(c + 32, n); ++e) {
      const double d = static_cast<double>(row[e]);
      acc = acc + d * d;
    }
    chunks.push_back(acc);
  }
  return pairwise(chunks);
}

template <class T>
double field_energy(const Shape& sh, const std::vector<T>& f) {
  const int64_t nl = sh.n[sh.dim - 1];
  std::vector<double> rows;
  for (int64_t q = 0; q < sh.count() / nl; ++q) rows.push_back(row_energy(f.data() + q * nl, nl));
  return pairwise(rows);
}

template <class T>
double amax_of(const std::vector<T>& f) {
  double m = 0.0;
  for (T v : f) m = std::max(m, std::fabs(static_cast<double>(v)));
  return m;
}

// ---- test helpers -------------------------------------------------------------------------------
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ULL + 1) {}
  uint64_t next() {
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s * 2685821657736338717ULL;
  }
  double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
};

// Field kinds: 0 noise, 1 quantized noise (many ties), 2 smooth waves plus noise, 3 constant.
template <class T>
std::vector<T> make_field(const Shape& sh, int kind, uint64_t seed) {
  Rng rng(seed);
  std::vector<T> f(static_cast<size_t>(sh.count()));
  for (int64_t i = 0; i < sh.count(); ++i) {
    int64_t rem = i;
    double c[3] = {0, 0, 0};
    for (int a = sh.dim - 1; a >= 0; --a) {
      c[a] = static_cast<double>(rem % sh.n[a]);
      rem /= sh.n[a];
    }
    double v;
    switch (kind) {
      case 0: v = rng.uniform() * 2.0 - 1.0; break;
      case 1: v = std::floor(rng.uniform() * 5.0) - 2.0; break;
      case 2:
        v = std::sin(0.37 * c[0] + 0.11 * c[1]) + 0.5 * std::cos(0.9 * c[1] - 0.23 * c[2]) +
            0.25 * std::sin(1.7 * c[0] + 2.3 * c[2] + 0.4 * c[1]) + 0.05 * (rng.uniform() - 0.5);
        break;
      default: v = 1.25; break;
    }
    f[i] = static_cast<T>(v);
  }
  return f;
}

// Flips axis a of a field.
template <class T>
std::vector<T> flip(const Shape& sh, const std::vector<T>& f, int axis) {
  std::vector<T> out = f;
  for_lines<T>(sh, axis, [](std::vector<T>& line) { std::reverse(line.begin(), line.end()); }, out);
  return out;
}

template <class T>
bool same_bits(const T* a, const T* b, size_t n) {
  return std::memcmp(a, b, n * sizeof(T)) == 0;
}

template <class T>
constexpr eda::DType dtype_of() {
  return sizeof(T) == 4 ? eda::DType::F32 : eda::DType::F64;
}

// The rows of an eda::decompose call on host input: [B][P][count].
template <class T>
struct GpuRun {
  std::unique_ptr<eda::Run> run;
  std::vector<T> rows;
};

template <class T>
GpuRun<T> run_host(eda::Context& ctx, const Shape& sh, int64_t B, const std::vector<T>& x,
                   const eda::Params& p) {
  eda::Input in;
  in.dtype = dtype_of<T>();
  in.x = x.data();
  in.dim = sh.dim;
  in.batch = B;
  for (int a = 0; a < 3; ++a) in.n[a] = sh.n[a];
  GpuRun<T> g;
  g.run = eda::decompose(ctx, in, p);
  g.rows.resize(static_cast<size_t>(B * g.run->rows * sh.count()));
  if (!g.rows.empty()) g.run->collect(ctx, g.rows.data(), false);
  return g;
}

// Runs `body` in a fresh thread, with that thread's own new context of `device`; an exception
// thrown by `body` is rethrown in the caller.
inline void in_new_thread(int device, const std::function<void(eda::Context&)>& body) {
  std::exception_ptr error;
  std::thread t([&] {
    try {
      eda::Context& ctx = eda::Context::current(device, eda::RuntimeConfig::from_environment());
      body(ctx);
    } catch (...) {
      error = std::current_exception();
    }
  });
  t.join();
  if (error) std::rethrow_exception(error);
}

}  // namespace host_twin
