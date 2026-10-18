// SPDX-License-Identifier: Apache-2.0
// test_input: device input (describe_device_input in decompose.hpp, the conversion of
// field/convert.hpp) against the same values passed as host input, bitwise. Read in place: a
// padded batch stride, spatial axes stored in reverse order, negative strides, a zero batch stride
// and forced chunks of 1 and 2 elements. Converted: every source type, mergeable and non-mergeable
// batch axes, a forced chunk; the chunking equals the host input's. The cast table on edge values
// (rounding of 64-bit integers, float64 overflow and subnormals, float16 and bfloat16, bool bytes),
// the in-place / conversion decision and the argument errors. Device output (collect into device
// memory) throughout.
#include "runtime/context.hpp"
#include "runtime/memory.hpp"
#include "twin.hpp"
#include "decompose.hpp"
#include "field/convert.hpp"
#include "test_support.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using host_twin::Shape;

namespace {

const bool kSmall = std::getenv("EDA_SANITIZE") && std::string(std::getenv("EDA_SANITIZE")) == "1";

Shape shape(int dim) {
  Shape s;
  s.dim = dim;
  if (dim == 1) {
    s.n[0] = kSmall ? 128 : 384;
  } else if (dim == 2) {
    s.n[0] = kSmall ? 16 : 40;
    s.n[1] = kSmall ? 14 : 36;
  } else {
    s.n[0] = kSmall ? 9 : 16;
    s.n[1] = kSmall ? 8 : 14;
    s.n[2] = kSmall ? 7 : 12;
  }
  return s;
}

void check_cuda(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e));
}

// Device bytes (cudaMalloc), filled from the host.
class DeviceBytes {
 public:
  explicit DeviceBytes(const void* host, size_t bytes) : bytes_(bytes) {
    check_cuda(cudaMalloc(&ptr_, bytes > 0 ? bytes : 1));
    if (bytes) check_cuda(cudaMemcpy(ptr_, host, bytes, cudaMemcpyHostToDevice));
  }
  ~DeviceBytes() { (void)cudaFree(ptr_); }
  DeviceBytes(const DeviceBytes&) = delete;
  DeviceBytes& operator=(const DeviceBytes&) = delete;
  char* data() const { return static_cast<char*>(ptr_); }

 private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
};

template <class T>
struct Result {
  std::vector<T> rows;
  std::vector<int32_t> n_imfs, error, scales;
  int32_t P = 0;
  int64_t chunk = 0;
  bool converted = false;
};

template <class T>
Result<T> host_run(eda::Context& ctx, const Shape& sh, int64_t B, const std::vector<T>& x, const eda::Params& p) {
  host_twin::GpuRun<T> g = host_twin::run_host<T>(ctx, sh, B, x, p);
  Result<T> r;
  r.rows = std::move(g.rows);
  r.n_imfs = g.run->n_imfs;
  r.error = g.run->error;
  r.scales = g.run->hist_scale;
  r.P = g.run->rows;
  r.chunk = g.run->chunk;
  return r;
}

// The call on a CUDA array; the rows are collected into device memory, then copied back.
template <class T>
Result<T> device_run(eda::Context& ctx, const eda::CudaArray& a, const eda::Params& p) {
  eda::Input in;
  eda::SourceLayout layout;
  eda::describe_device_input(a, host_twin::dtype_of<T>(), in, layout);
  std::unique_ptr<eda::Run> run = eda::decompose(ctx, in, p);
  Result<T> r;
  r.converted = in.source != nullptr;
  r.n_imfs = run->n_imfs;
  r.error = run->error;
  r.scales = run->hist_scale;
  r.P = run->rows;
  r.chunk = run->chunk;
  const size_t n = static_cast<size_t>(run->batch) * run->rows * run->count;
  void* out = eda::output_alloc_async(n * sizeof(T), ctx.stream(), ctx.device());
  run->collect(ctx, out, true);
  ctx.synchronize();
  r.rows.resize(n);
  check_cuda(cudaMemcpy(r.rows.data(), out, n * sizeof(T), cudaMemcpyDeviceToHost));
  eda::defer_device_free(out, ctx.device());
  return r;
}

template <class T>
bool same(const Result<T>& a, const Result<T>& b) {
  return a.P == b.P && a.n_imfs == b.n_imfs && a.error == b.error && a.scales == b.scales &&
         a.rows.size() == b.rows.size() && host_twin::same_bits(a.rows.data(), b.rows.data(), a.rows.size());
}

// B elements of kinds 2, 0, 2, 1, ... (waves, noise, quantized noise), C order.
template <class T>
std::vector<T> batch_values(const Shape& sh, int64_t B, uint64_t seed) {
  std::vector<T> x;
  for (int64_t b = 0; b < B; ++b) {
    const int kind = b % 3 == 1 ? 0 : (b % 4 == 3 ? 1 : 2);
    const std::vector<T> f = host_twin::make_field<T>(sh, kind, seed + 17 * b);
    x.insert(x.end(), f.begin(), f.end());
  }
  return x;
}

// ---- read in place ------------------------------------------------------------------------
template <class T>
void check_in_place(int dim) {
  eda::Context& ctx = eda::Context::current(0);
  const Shape sh = shape(dim);
  const int64_t N = sh.count();
  const int64_t B = 4;
  const std::vector<T> x = batch_values<T>(sh, B, 300 + dim);
  const eda::Params p;
  const Result<T> want = host_run<T>(ctx, sh, B, x, p);

  // (1) a padded batch stride, C-ordered spatial axes
  {
    const int64_t pitch = N + 37;
    std::vector<T> mem(static_cast<size_t>(B * pitch), T(-7));
    for (int64_t b = 0; b < B; ++b) std::copy(x.begin() + b * N, x.begin() + (b + 1) * N, mem.begin() + b * pitch);
    DeviceBytes d(mem.data(), mem.size() * sizeof(T));
    eda::CudaArray a;
    a.data = d.data();
    a.type = sizeof(T) == 4 ? eda::SourceType::F32 : eda::SourceType::F64;
    a.batch_shape = {B};
    a.batch_strides = {pitch};
    a.dim = dim;
    for (int k = 0; k < dim; ++k) {
      a.n[k] = sh.n[k];
      a.strides[k] = sh.stride(k);
    }
    const Result<T> got = device_run<T>(ctx, a, p);
    CHECK(!got.converted && same(got, want));
    for (int64_t lim : {int64_t{1}, int64_t{2}}) {
      eda::Params pc = p;
      pc.max_chunk = lim;
      const Result<T> ch = device_run<T>(ctx, a, pc);
      CHECK(ch.chunk == lim && same(ch, want));
    }
  }
  // (2) the spatial axes stored in reverse order inside each element, the elements in reverse
  // order (a negative batch stride) and axis 0 reversed (a negative stride)
  {
    std::vector<T> mem(static_cast<size_t>(B * N));
    int64_t st[3] = {1, 1, 1};
    for (int k = 1; k < dim; ++k) st[k] = st[k - 1] * sh.n[k - 1];  // axis 0 fastest
    for (int64_t b = 0; b < B; ++b) {
      for (int64_t i = 0; i < N; ++i) {
        int64_t rem = i, off = 0;
        for (int k = dim - 1; k >= 0; --k) {
          int64_t c = rem % sh.n[k];
          rem /= sh.n[k];
          if (k == 0) c = sh.n[0] - 1 - c;
          off += c * st[k];
        }
        mem[static_cast<size_t>((B - 1 - b) * N + off)] = x[static_cast<size_t>(b * N + i)];
      }
    }
    DeviceBytes d(mem.data(), mem.size() * sizeof(T));
    eda::CudaArray a;
    a.type = sizeof(T) == 4 ? eda::SourceType::F32 : eda::SourceType::F64;
    a.data = d.data() + static_cast<size_t>((B - 1) * N + (sh.n[0] - 1) * st[0]) * sizeof(T);
    a.batch_shape = {B};
    a.batch_strides = {-N};
    a.dim = dim;
    for (int k = 0; k < dim; ++k) {
      a.n[k] = sh.n[k];
      a.strides[k] = k == 0 ? -st[0] : st[k];
    }
    const Result<T> got = device_run<T>(ctx, a, p);
    CHECK(!got.converted && same(got, want));
  }
  // (3) a zero batch stride: three views of element 0
  {
    DeviceBytes d(x.data(), static_cast<size_t>(N) * sizeof(T));
    eda::CudaArray a;
    a.type = sizeof(T) == 4 ? eda::SourceType::F32 : eda::SourceType::F64;
    a.data = d.data();
    a.batch_shape = {3};
    a.batch_strides = {0};
    a.dim = dim;
    for (int k = 0; k < dim; ++k) {
      a.n[k] = sh.n[k];
      a.strides[k] = sh.stride(k);
    }
    std::vector<T> x3;
    for (int r = 0; r < 3; ++r) x3.insert(x3.end(), x.begin(), x.begin() + N);
    const Result<T> got = device_run<T>(ctx, a, p);
    CHECK(!got.converted && same(got, host_run<T>(ctx, sh, 3, x3, p)));
  }
}

// ---- conversion ------------------------------------------------------------------------------
// Values of source type S (stored as bytes) and their host cast to T, as NumPy's astype does it.
template <class T>
struct Converted {
  std::vector<unsigned char> bytes;  // C order (B, count) in S
  std::vector<T> cast;
};

template <class T>
Converted<T> source_values(eda::SourceType type, const Shape& sh, int64_t B, uint64_t seed) {
  const std::vector<double> base = batch_values<double>(sh, B, seed);
  Converted<T> c;
  const size_t n = base.size();
  const size_t item = eda::source_item_size(type);
  c.bytes.resize(n * item);
  c.cast.resize(n);
  auto put = [&](size_t i, auto v) {
    std::memcpy(c.bytes.data() + i * item, &v, sizeof v);
  };
  for (size_t i = 0; i < n; ++i) {
    const double v = base[i];
    switch (type) {
      case eda::SourceType::Bool: {
        const uint8_t b = v > 0.1 ? 1 : 0;
        put(i, b);
        c.cast[i] = static_cast<T>(b);
        break;
      }
      case eda::SourceType::I8: { const int8_t q = static_cast<int8_t>(std::lround(v * 60)); put(i, q); c.cast[i] = static_cast<T>(q); break; }
      case eda::SourceType::U8: { const uint8_t q = static_cast<uint8_t>(std::lround(v * 60 + 128)); put(i, q); c.cast[i] = static_cast<T>(q); break; }
      case eda::SourceType::I16: { const int16_t q = static_cast<int16_t>(std::lround(v * 9000)); put(i, q); c.cast[i] = static_cast<T>(q); break; }
      case eda::SourceType::U16: { const uint16_t q = static_cast<uint16_t>(std::lround(v * 9000 + 30000)); put(i, q); c.cast[i] = static_cast<T>(q); break; }
      // 32- and 64-bit integers beyond the float mantissa: the casts round
      case eda::SourceType::I32: { const int32_t q = static_cast<int32_t>(std::llround(v * 6.0e8)); put(i, q); c.cast[i] = static_cast<T>(q); break; }
      case eda::SourceType::U32: { const uint32_t q = static_cast<uint32_t>(std::llround(v * 0.9e9 + 2.1e9)); put(i, q); c.cast[i] = static_cast<T>(q); break; }
      case eda::SourceType::I64: { const int64_t q = std::llround(v * 3.0e18); put(i, q); c.cast[i] = static_cast<T>(q); break; }
      case eda::SourceType::U64: {
        const uint64_t q = static_cast<uint64_t>(std::llround(v * 2.0e18 + 4.5e18)) * 2u + (i & 1u);
        put(i, q);
        c.cast[i] = static_cast<T>(q);
        break;
      }
      case eda::SourceType::F16: { const __half h = __double2half(v); put(i, h); c.cast[i] = static_cast<T>(__half2float(h)); break; }
      case eda::SourceType::BF16: { const __nv_bfloat16 h = __double2bfloat16(v); put(i, h); c.cast[i] = static_cast<T>(__bfloat162float(h)); break; }
      case eda::SourceType::F32: { const float f = static_cast<float>(v * 1.5); put(i, f); c.cast[i] = static_cast<T>(f); break; }
      case eda::SourceType::F64: { const double f = v * 1.5 + 1e-9 * static_cast<double>(i % 7); put(i, f); c.cast[i] = static_cast<T>(f); break; }
    }
  }
  return c;
}

template <class T>
void check_conversion(int dim, eda::SourceType type, bool forced_chunk) {
  eda::Context& ctx = eda::Context::current(0);
  const Shape sh = shape(dim);
  const int64_t N = sh.count();
  const int64_t B0 = 2, B1 = 3, B = B0 * B1;
  const Converted<T> c = source_values<T>(type, sh, B, 900 + dim + static_cast<int>(type));
  eda::Params p;
  if (forced_chunk) p.max_chunk = 2;
  const Result<T> want = host_run<T>(ctx, sh, B, c.cast, p);
  const size_t item = eda::source_item_size(type);
  // (1) batch axes (2, 3) that do not merge: element (i, j) at i * (3N + 11) + j * N
  {
    const int64_t row = B1 * N + 11;
    std::vector<unsigned char> mem(static_cast<size_t>(B0 * row) * item, 0xAB);
    for (int64_t b = 0; b < B; ++b) {
      const int64_t i = b / B1, j = b % B1;
      std::memcpy(mem.data() + static_cast<size_t>(i * row + j * N) * item, c.bytes.data() + static_cast<size_t>(b * N) * item,
                  static_cast<size_t>(N) * item);
    }
    DeviceBytes d(mem.data(), mem.size());
    eda::CudaArray a;
    a.type = type;
    a.data = d.data();
    a.batch_shape = {B0, B1};
    a.batch_strides = {row, N};
    a.dim = dim;
    for (int k = 0; k < dim; ++k) {
      a.n[k] = sh.n[k];
      a.strides[k] = sh.stride(k);
    }
    const Result<T> got = device_run<T>(ctx, a, p);
    CHECK(got.converted && got.chunk == want.chunk && same(got, want));
    if (!got.converted || got.chunk != want.chunk || !same(got, want)) {
      std::fprintf(stderr, "  conversion from %s, D = %d, %s\n", eda::source_type_name(type), dim, sizeof(T) == 4 ? "float32" : "float64");
    }
  }
  // (2) contiguous (the batch axes merge): converted unless the type is the compute dtype
  if (!kSmall) {
    DeviceBytes d(c.bytes.data(), c.bytes.size());
    eda::CudaArray a;
    a.type = type;
    a.data = d.data();
    a.batch_shape = {B0, B1};
    a.batch_strides = {B1 * N, N};
    a.dim = dim;
    for (int k = 0; k < dim; ++k) {
      a.n[k] = sh.n[k];
      a.strides[k] = sh.stride(k);
    }
    const bool own = type == (sizeof(T) == 4 ? eda::SourceType::F32 : eda::SourceType::F64);
    const Result<T> got = device_run<T>(ctx, a, p);
    CHECK(got.converted == !own && same(got, want));
    if (!own) CHECK(got.chunk == want.chunk);
  }
}

// ---- the cast table on edge values -----------------------------------------------------------
template <class T, class S>
void check_cast(eda::SourceType type, const std::vector<S>& values, const std::vector<T>& expected) {
  eda::Context& ctx = eda::Context::current(0);
  const int64_t n = static_cast<int64_t>(values.size());
  DeviceBytes d(values.data(), values.size() * sizeof(S));
  eda::SourceLayout src;
  src.type = type;
  src.batch_axes = 1;
  src.batch_shape[0] = n;
  src.batch_stride[0] = 1;
  src.stride[0] = 1;
  eda::Geom g{};
  g.dim = 1;
  g.n[0] = 1;
  g.n[1] = g.n[2] = 1;
  g.batch = n;
  g.count = 1;
  T* out = nullptr;
  check_cuda(cudaMalloc(&out, values.size() * sizeof(T)));
  eda::enqueue_convert_gather<T>(ctx.stream(), src, d.data(), 0, g, out);
  ctx.synchronize();
  std::vector<T> got(values.size());
  check_cuda(cudaMemcpy(got.data(), out, got.size() * sizeof(T), cudaMemcpyDeviceToHost));
  check_cuda(cudaFree(out));
  for (size_t i = 0; i < got.size(); ++i) {
    const bool ok = std::memcmp(&got[i], &expected[i], sizeof(T)) == 0;
    CHECK(ok);
    if (!ok) {
      std::fprintf(stderr, "  cast %s -> %s, value %zu: got %.17g, want %.17g\n", eda::source_type_name(type),
                   sizeof(T) == 4 ? "float32" : "float64", i, static_cast<double>(got[i]), static_cast<double>(expected[i]));
    }
  }
}

template <class T>
void check_cast_table() {
  {
    const std::vector<int64_t> v = {0, 1, -1, (int64_t{1} << 53) + 1, (int64_t{1} << 24) + 1, std::numeric_limits<int64_t>::max(),
                                    std::numeric_limits<int64_t>::min(), (int64_t{1} << 62) + 3, -((int64_t{1} << 60) + 5),
                                    // float32 midpoints after a float64 rounding: one rounding only
                                    (int64_t{1} << 62) + (int64_t{1} << 38) + 1, -((int64_t{1} << 62) + (int64_t{1} << 38) + 1)};
    std::vector<T> e;
    for (const int64_t x : v) e.push_back(static_cast<T>(x));
    check_cast<T>(eda::SourceType::I64, v, e);
  }
  {
    const std::vector<uint64_t> v = {0, 1, (uint64_t{1} << 53) + 1, std::numeric_limits<uint64_t>::max(), (uint64_t{1} << 63) + 1,
                                     (uint64_t{1} << 63) + (uint64_t{1} << 40) + 1, (uint64_t{1} << 63) + (uint64_t{1} << 39) + 1};
    std::vector<T> e;
    for (const uint64_t x : v) e.push_back(static_cast<T>(x));
    check_cast<T>(eda::SourceType::U64, v, e);
  }
  {
    const std::vector<int32_t> v = {0, -1, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max(), (1 << 24) + 1, -((1 << 25) + 3)};
    std::vector<T> e;
    for (const int32_t x : v) e.push_back(static_cast<T>(x));
    check_cast<T>(eda::SourceType::I32, v, e);
  }
  {
    const std::vector<uint32_t> v = {0, std::numeric_limits<uint32_t>::max(), (1u << 31) + 1, (1u << 24) + 3};
    std::vector<T> e;
    for (const uint32_t x : v) e.push_back(static_cast<T>(x));
    check_cast<T>(eda::SourceType::U32, v, e);
  }
  {
    const std::vector<double> v = {0.0, -0.0, 1e39, -1e39, 1e-40, -3e-45, 1e-46, 0.1, 3.4028235677973366e38,
                                   std::numeric_limits<double>::infinity(), 1.0 + std::ldexp(1.0, -24),
                                   1.0 + 3 * std::ldexp(1.0, -24), std::numeric_limits<double>::denorm_min()};
    std::vector<T> e;
    for (const double x : v) e.push_back(static_cast<T>(x));
    check_cast<T>(eda::SourceType::F64, v, e);
  }
  {
    const std::vector<float> v = {0.0f, -0.0f, 1e-40f, 3.0e38f, std::numeric_limits<float>::infinity(), 0.1f};
    std::vector<T> e;
    for (const float x : v) e.push_back(static_cast<T>(x));
    check_cast<T>(eda::SourceType::F32, v, e);
  }
  {
    std::vector<__half> v;
    std::vector<T> e;
    for (const unsigned short bits : {0x0000, 0x8000, 0x0001, 0x03ff, 0x3c00, 0x7bff, 0x7c00, 0xfc00, 0x3555}) {
      __half_raw raw;
      raw.x = bits;
      const __half h(raw);
      v.push_back(h);
      e.push_back(static_cast<T>(__half2float(h)));
    }
    check_cast<T>(eda::SourceType::F16, v, e);
  }
  {
    std::vector<__nv_bfloat16> v;
    std::vector<T> e;
    for (const unsigned short bits : {0x0000, 0x8000, 0x0001, 0x007f, 0x3f80, 0x7f7f, 0x7f80, 0xbe2a}) {
      __nv_bfloat16_raw raw;
      raw.x = bits;
      const __nv_bfloat16 h(raw);
      v.push_back(h);
      e.push_back(static_cast<T>(__bfloat162float(h)));
    }
    check_cast<T>(eda::SourceType::BF16, v, e);
  }
  {
    const std::vector<uint8_t> v = {0, 1, 2, 255};
    const std::vector<T> e = {T(0), T(1), T(1), T(1)};
    check_cast<T>(eda::SourceType::Bool, v, e);
  }
  {
    const std::vector<int8_t> v = {-128, 127, 0, -1};
    std::vector<T> e;
    for (const int8_t x : v) e.push_back(static_cast<T>(x));
    check_cast<T>(eda::SourceType::I8, v, e);
  }
  {
    const std::vector<uint16_t> v = {0, 65535, 32768};
    std::vector<T> e;
    for (const uint16_t x : v) e.push_back(static_cast<T>(x));
    check_cast<T>(eda::SourceType::U16, v, e);
  }
}

// ---- the decision and the errors -----------------------------------------------------------------
void check_describe() {
  alignas(8) static double cell[4] = {0, 0, 0, 0};
  auto base = [&](std::vector<int64_t> shape, std::vector<int64_t> strides) {
    eda::CudaArray a;
    a.data = cell;
    a.type = eda::SourceType::F64;
    a.batch_shape = std::move(shape);
    a.batch_strides = std::move(strides);
    a.dim = 1;
    a.n[0] = 8;
    a.strides[0] = 1;
    return a;
  };
  eda::Input in;
  eda::SourceLayout layout;
  // extent-1 axes drop out; the rest merge into one stride
  eda::describe_device_input(base({1, 4, 1}, {999, 7, 3}), eda::DType::F64, in, layout);
  CHECK(in.source == nullptr && in.batch == 4 && in.strides.s[0] == 7 && in.strides.s[1] == 1);
  eda::describe_device_input(base({2, 3, 5}, {150, 50, 10}), eda::DType::F64, in, layout);
  CHECK(in.source == nullptr && in.batch == 30 && in.strides.s[0] == 10);
  eda::describe_device_input(base({}, {}), eda::DType::F64, in, layout);
  CHECK(in.source == nullptr && in.batch == 1 && in.strides.s[0] == 0);
  // not mergeable, or another dtype: converted
  eda::describe_device_input(base({2, 3}, {40, 10}), eda::DType::F64, in, layout);
  CHECK(in.source == &layout && in.batch == 6 && layout.batch_axes == 2 && layout.batch_stride[0] == 40);
  eda::describe_device_input(base({4}, {8}), eda::DType::F32, in, layout);
  CHECK(in.source == &layout && layout.batch_axes == 1 && layout.type == eda::SourceType::F64);
  // an empty batch needs no pointer
  {
    eda::CudaArray a = base({0, 3}, {3, 1});
    a.data = nullptr;
    eda::describe_device_input(a, eda::DType::F64, in, layout);
    CHECK(in.batch == 0);
  }
  CHECK(eda_test::throws_with<std::invalid_argument>(
      [&] {
        eda::CudaArray a = base({2}, {8});
        a.data = reinterpret_cast<const char*>(cell) + 4;
        eda::describe_device_input(a, eda::DType::F64, in, layout);
      },
      "not aligned"));
  CHECK(eda_test::throws_with<std::invalid_argument>(
      [&] {
        eda::CudaArray a = base({2}, {8});
        a.data = nullptr;
        eda::describe_device_input(a, eda::DType::F64, in, layout);
      },
      "no data pointer"));
  CHECK(eda_test::throws_with<std::invalid_argument>(
      [&] {
        eda::describe_device_input(base(std::vector<int64_t>(65, 1), std::vector<int64_t>(65, 1)), eda::DType::F64, in, layout);
      },
      "at most 64 batch axes"));
  CHECK(eda_test::throws_with<std::invalid_argument>(
      [&] { eda::describe_device_input(base({2, 3}, {1}), eda::DType::F64, in, layout); }, "differ in length"));
  // 64 batch axes are accepted (63 of extent 1)
  {
    std::vector<int64_t> s(64, 1), st(64, 0);
    s[10] = 3;
    st[10] = 8;
    s[40] = 2;
    st[40] = 1;
    eda::describe_device_input(base(s, st), eda::DType::F64, in, layout);
    CHECK(in.batch == 6 && in.source == &layout && layout.batch_axes == 2);
  }
}

}  // namespace

// For the slower sanitizer tools: `test_input casts` runs the decision and the cast table only (the
// conversion kernel alone); `test_input quick` adds one call read in place and one converted call.
int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  try {
    check_describe();
    check_cast_table<float>();
    check_cast_table<double>();
    if (mode == "quick") {
      check_in_place<double>(1);
      check_conversion<float>(1, eda::SourceType::I16, false);
    }
    if (mode == "casts" || mode == "quick") {
      eda::Context::shutdown();
      return eda_test::finish("test_input");
    }
    for (int dim = 1; dim <= 3; ++dim) {
      check_in_place<double>(dim);
      if (!kSmall || dim == 2) check_in_place<float>(dim);
    }
    const eda::SourceType all[] = {eda::SourceType::Bool, eda::SourceType::I8,  eda::SourceType::I16, eda::SourceType::I32,
                                   eda::SourceType::I64,  eda::SourceType::U8,  eda::SourceType::U16, eda::SourceType::U32,
                                   eda::SourceType::U64,  eda::SourceType::F16, eda::SourceType::BF16, eda::SourceType::F32,
                                   eda::SourceType::F64};
    for (const eda::SourceType t : all) {
      check_conversion<float>(1, t, false);
      check_conversion<double>(1, t, false);
    }
    for (int dim = 2; dim <= 3; ++dim) {
      check_conversion<float>(dim, eda::SourceType::F64, false);
      check_conversion<double>(dim, eda::SourceType::I16, dim == 3);
      if (!kSmall) check_conversion<float>(dim, eda::SourceType::F16, true);
    }
    check_conversion<double>(1, eda::SourceType::U64, true);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "test_input: %s\n", error.what());
    (void)eda::drain_deferred_frees();
    eda::Context::shutdown();
    return 1;
  }
  eda::Context::shutdown();  // drains the device outputs, releases the pools (leak check at exit)
  return eda_test::finish("test_input");
}
