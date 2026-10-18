// SPDX-License-Identifier: Apache-2.0
// The host API of a decomposition: eda::decompose validates, plans and runs one call on the
// context stream and returns a Run, which holds the rows on the device and collects them into the
// caller's output.
#pragma once

#include "runtime/context.hpp"
#include "field/convert.hpp"
#include "field/geom.hpp"
#include "field/field_ops.hpp"

#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace eda {

enum class DType : uint8_t { F32 = 0, F64 = 1 };

// The output rows or the workspace of a call do not fit the device memory (Python: MemoryError).
class OutOfDeviceMemory : public std::bad_alloc {
 public:
  explicit OutOfDeviceMemory(std::string what) : what_(std::move(what)) {}
  const char* what() const noexcept override { return what_.c_str(); }

 private:
  std::string what_;
};

struct Input {
  DType dtype = DType::F64;  // the compute dtype
  const void* x = nullptr;
  bool on_device = false;   // device: ordered on ctx.stream() (the caller fenced it); host: C-contiguous
  int dim = 0;
  int64_t batch = 0;
  int64_t n[kMaxDim] = {1, 1, 1};
  InputStrides strides{};   // in elements; device input read in place only (host input is C-contiguous)
  // Device input that is converted: each chunk is gathered and cast from x into the workspace
  // slot that host input is copied into (field/convert.hpp), so the call runs exactly as for
  // the same values passed as host input. nullptr: x is read in place (or is host input).
  const SourceLayout* source = nullptr;
};

// A CUDA array described as the (B, *spatial) view of the call: the batch axes in C order of the
// elements (any number, extents >= 0), then the spatial axes; strides in elements of `type`, any
// sign.
struct CudaArray {
  const void* data = nullptr;
  SourceType type = SourceType::F64;
  std::vector<int64_t> batch_shape, batch_strides;
  int dim = 0;
  int64_t n[kMaxDim] = {1, 1, 1};
  int64_t strides[kMaxDim] = {0, 0, 0};
};

// Fills `in` (x, on_device, dim, batch, n and, for in-place reads, strides) for `a` decomposed in
// `dtype`. The array is read in place when its type is `dtype` and its batch axes merge into one
// stride (extent-1 axes dropped; neighbours merge when stride[k] == stride[k + 1] * shape[k + 1]);
// otherwise `layout` receives the gather and in.source points to it. Throws std::invalid_argument
// for more than kMaxBatchAxes batch axes, mismatched lists, negative extents, a null pointer of a
// non-empty array or a pointer not aligned to the item size.
void describe_device_input(const CudaArray& a, DType dtype, Input& in, SourceLayout& layout);

struct Params {
  int32_t max_imf = -1;                // <= 0: no limit
  std::optional<double> tie_tolerance; // relative; nullopt: 2^-14 (float32) / 2^-30 (float64)
  int32_t sifts = 3;                   // sifts per mode, >= 1
  bool check_finite = true;
  int64_t max_chunk = 0;               // > 0: at most this many elements per chunk (tests); 0: by free memory
  bool orientation = false;            // keep the orientation (n, c) of every mode (D >= 2; Run::collect_orientation)
};

// The longest axis per dtype (the vHGW pass stages a whole line in shared memory), and the default
// relative tie tolerance (2^-14 for float32, 2^-30 for float64).
int32_t max_axis_len(DType dtype);
double default_tie_tolerance(DType dtype);
// Validates params against the method and the geometry (std::invalid_argument).
void validate(const Input& in, const Params& p);

class Run {
 public:
  virtual ~Run() = default;
  int32_t rows = 0;                    // P = max n_imfs + 1 over the elements without an error
  int64_t batch = 0;
  int64_t count = 0;                   // samples per element
  int32_t rows_cap = 0;
  int64_t chunk = 0;                   // elements per chunk of the call
  std::vector<int32_t> n_imfs, error;  // [batch]
  // History, [batch][rows_cap - 1]: windows, spacing, sifts, energy of each mode.
  std::vector<int32_t> hist_scale, hist_sifts;
  std::vector<double> hist_spacing, hist_energy;
  // Rows [0, P) of every element into out (batch, P, count), C-contiguous: device output is
  // enqueued on ctx.stream(); host output is written synchronously.
  virtual void collect(Context& ctx, void* out, bool out_on_device) = 0;
  // With Params::orientation: the direction (batch, K, D, count) and the coherence (batch, K,
  // count) of the modes 0 .. K - 1 (K at most the largest n_imfs), C-contiguous; 0 beyond an
  // element's n_imfs and for every mode of a failed element. Device or host output as collect.
  virtual void collect_orientation(Context& ctx, void* dir, void* coh, bool out_on_device, int32_t K) = 0;
};

std::unique_ptr<Run> decompose(Context& ctx, const Input& in, const Params& p);

}  // namespace eda
