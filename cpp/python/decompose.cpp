// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Giorgio Angelotti
// The decomposition entry points of eda._core and their process lifecycle.
//
//   decompose_host(x, ...)    x: C-contiguous host (B, *spatial) float32/float64 -> (rows, info)
//   decompose_device(ptr, ...) a CUDA array of any supported dtype and layout -> (owner, ptr, info)
//   dlpack_info(capsule), pointer_device(ptr), dlpack_export(...): DLPack and CUDA array interface
//
// rows is (B, P, *spatial) in the compute dtype; info is a dict of n_imfs and error ([B] int32),
// windows, sifts ([B, K] int32), spacing, energy ([B, K] float64) and chunk (elements per chunk),
// K the largest n_imfs. With orientation, info also holds the direction (B, K, D, *spatial) and
// the coherence (B, K, *spatial): NumPy arrays for host input, and for device input
// "orientation" = (owner, ptr) of one buffer holding both, C-contiguous, the coherence after the
// direction.
//
//   monogenic_host(x, ...)     x as decompose_host -> (out, info)
//   monogenic_device(ptr, ...) as decompose_device -> (owner, ptr, info)
//
// out is the flat buffer of the planes of eda::run_monogenic (method/method.hpp); info is a dict of
// nonfinite ([B] int32) and chunk (elements per chunk). max_chunk, the last argument of every
// decompose_* and monogenic_* call, caps the elements per chunk for the tests (0: sized by the
// free memory).
//
// Every call validates its arguments before any GPU work, takes the
// environment snapshot with the GIL held and runs the GPU work, the frees of dropped results
// included (drain_deferred_frees), with the GIL released, on the calling thread's context. A
// device result is complete on return; its memory is released through the deferred-free queue
// when the last reference to the returned owner capsule is gone. A call that fails for lack of
// memory first gives the device memory the library does not use back to the driver
// (releasing_on_oom).
//
// Interpreter exit: an atexit callback refuses new calls, waits until no thread is inside a call
// (a daemon thread may still be), then releases the contexts, pools and pinned blocks. In a child
// made by os.fork() after CUDA was used, every call raises and nothing is released (the parent
// owns it).
#include "lifecycle.hpp"
#include "runtime/context.hpp"
#include "runtime/device.hpp"
#include "runtime/env.hpp"
#include "runtime/error.hpp"
#include "runtime/memory.hpp"
#include "decompose.hpp"
#include "method/method.hpp"

#include <cuda_runtime_api.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sys/mman.h>

namespace nb = nanobind;

namespace {

// Process state, read and changed only with the GIL held.
int g_calls = 0;           // calls in progress (a call counts from its first line to its return)
bool g_exiting = false;    // the exit hook has run: no call may start
bool g_gpu_used = false;   // some call reached its GPU work: the exit hook releases the library
bool g_cuda_used = false;  // CUDA was initialized through the library: a forked child cannot use it

constexpr const char* kForkedMessage =
    "eda: CUDA cannot be used in a process forked after eda used CUDA; start worker processes "
    "with the spawn method (multiprocessing.get_context(\"spawn\"))";

void refuse_in_forked_child() {
  if (eda::forked_child()) throw std::runtime_error(kForkedMessage);
}

// atexit (main thread, GIL held, before finalization starts; CPython has joined the non-daemon
// threads). A daemon thread inside a call finishes it while the hook waits with the GIL released;
// its later calls are refused. The wait has no limit: a call ends in finite time. The release
// runs with the GIL released (it synchronizes the devices, which may wait for host callbacks of
// other libraries that need the GIL).
void release_at_exit() {
  if (eda::forked_child()) return;
  g_exiting = true;
  while (g_calls > 0) {
    nb::gil_scoped_release release;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (g_gpu_used) {
    nb::gil_scoped_release release;
    eda::Context::shutdown();
  }
}

// os.register_at_fork(after_in_child=...): the forking thread holds the GIL, so it is not inside a
// call; the other threads do not exist in the child.
void after_fork_in_child() {
  g_calls = 0;
  if (g_cuda_used) eda::mark_forked_child();
}

// Frees the deferred device buffers when the GPU work of a call ends, however it ends.
struct DrainOnExit {
  DrainOnExit() = default;
  ~DrainOnExit() { (void)eda::drain_deferred_frees(); }
  DrainOnExit(const DrainOnExit&) = delete;
  DrainOnExit& operator=(const DrainOnExit&) = delete;
};

template <class T>
nb::object numpy_of(std::vector<T> v, std::vector<size_t> shape) {
  auto* heap = new std::vector<T>(std::move(v));
  nb::capsule owner(heap, [](void* ptr) noexcept { delete static_cast<std::vector<T>*>(ptr); });
  return nb::ndarray<nb::numpy, T>(heap->data(), shape.size(), shape.data(), owner).cast();
}

// An uninitialized host buffer for the output rows, owned until release(); the device copy writes
// every byte of it before it is handed out. From 8 MiB up it is aligned to 2 MiB and advised for
// transparent huge pages, so that the copy faults few pages. Always at least one byte, so that the
// pointer is never null.
class HostRows {
 public:
  explicit HostRows(size_t bytes) {
    constexpr size_t kHuge = size_t{2} << 20;
    if (bytes >= 4 * kHuge) {
      const size_t rounded = (bytes + kHuge - 1) / kHuge * kHuge;
      if (posix_memalign(&data_, kHuge, rounded) != 0) data_ = nullptr;
      if (data_) (void)madvise(data_, rounded, MADV_HUGEPAGE);
    } else {
      data_ = std::malloc(bytes > 0 ? bytes : 1);
    }
    if (!data_) throw std::bad_alloc();
  }
  ~HostRows() { std::free(data_); }
  HostRows(const HostRows&) = delete;
  HostRows& operator=(const HostRows&) = delete;
  void* data() const noexcept { return data_; }

  // A NumPy array of `shape` that owns the buffer from here on.
  template <class T>
  nb::object to_numpy(const std::vector<size_t>& shape) {
    nb::capsule owner(data_, [](void* ptr) noexcept { std::free(ptr); });
    void* data = data_;
    data_ = nullptr;
    return nb::ndarray<nb::numpy, T>(static_cast<T*>(data), shape.size(), shape.data(), owner).cast();
  }

 private:
  void* data_ = nullptr;
};

// The host-side results of a run, taken before the run is destroyed.
struct Summary {
  int64_t batch = 0;
  int32_t rows = 1;  // P
  int32_t rows_cap = 0;
  int64_t chunk = 0;
  std::vector<int32_t> n_imfs, error, hist_scale, hist_sifts;
  std::vector<double> hist_spacing, hist_energy;
};

Summary summarize(eda::Run& run) {
  Summary s;
  s.batch = run.batch;
  s.rows = run.rows;
  s.rows_cap = run.rows_cap;
  s.chunk = run.chunk;
  s.n_imfs = std::move(run.n_imfs);
  s.error = std::move(run.error);
  s.hist_scale = std::move(run.hist_scale);
  s.hist_sifts = std::move(run.hist_sifts);
  s.hist_spacing = std::move(run.hist_spacing);
  s.hist_energy = std::move(run.hist_energy);
  return s;
}

// Info arrays [B, K] per mode (K = the largest n_imfs); the entries of an element in error are 0.
nb::dict info_dict(const Summary& s) {
  const size_t B = static_cast<size_t>(s.batch);
  int32_t K = 0;
  for (size_t b = 0; b < B; ++b) K = std::max(K, s.n_imfs[b]);
  const size_t cap1 = static_cast<size_t>(std::max(s.rows_cap - 1, 0));
  const size_t Ku = static_cast<size_t>(K);
  std::vector<int32_t> windows(B * Ku, 0), sifts(B * Ku, 0);
  std::vector<double> spacing(B * Ku, 0.0), energy(B * Ku, 0.0);
  for (size_t b = 0; b < B; ++b) {
    const int32_t k_b = s.error[b] ? 0 : s.n_imfs[b];
    for (int32_t k = 0; k < k_b; ++k) {
      windows[b * Ku + k] = s.hist_scale[b * cap1 + k];
      sifts[b * Ku + k] = s.hist_sifts[b * cap1 + k];
      spacing[b * Ku + k] = s.hist_spacing[b * cap1 + k];
      energy[b * Ku + k] = s.hist_energy[b * cap1 + k];
    }
  }
  nb::dict info;
  info["n_imfs"] = numpy_of(s.n_imfs, {B});
  info["error"] = numpy_of(s.error, {B});
  info["windows"] = numpy_of(std::move(windows), {B, Ku});
  info["sifts"] = numpy_of(std::move(sifts), {B, Ku});
  info["spacing"] = numpy_of(std::move(spacing), {B, Ku});
  info["energy"] = numpy_of(std::move(energy), {B, Ku});
  info["chunk"] = s.chunk;
  return info;
}

eda::Params params_of(int32_t max_imf, std::optional<double> tie_tolerance, int32_t sifts, bool check_finite,
                      int64_t max_chunk, bool orientation) {
  eda::Params p;
  p.max_imf = max_imf;
  p.tie_tolerance = tie_tolerance;
  p.sifts = sifts;
  p.check_finite = check_finite;
  p.max_chunk = max_chunk;
  p.orientation = orientation;
  return p;
}

// The largest n_imfs of a run (the K of the information and of the orientation).
int32_t largest_imfs(const eda::Run& run) {
  int32_t K = 0;
  for (const int32_t n : run.n_imfs) K = std::max(K, n);
  return K;
}

// Validates the call before any GPU work; an empty batch is validated as one element.
void validate_call(const eda::Input& in, const eda::Params& p) {
  eda::Input one = in;
  if (one.batch == 0) one.batch = 1;
  eda::validate(one, p);
}

std::vector<size_t> rows_shape(const eda::Input& in, int32_t P) {
  std::vector<size_t> shape = {static_cast<size_t>(in.batch), static_cast<size_t>(P)};
  for (int a = 0; a < in.dim; ++a) shape.push_back(static_cast<size_t>(in.n[a]));
  return shape;
}

size_t item_size(eda::DType dtype) { return dtype == eda::DType::F32 ? sizeof(float) : sizeof(double); }

// The bytes of the output rows (B, P, *spatial) of a run.
size_t output_bytes(const eda::Run& run, eda::DType dtype) {
  return static_cast<size_t>(run.batch) * static_cast<size_t>(run.rows) * static_cast<size_t>(run.count) *
         item_size(dtype);
}

// Runs `step`, which allocates the output of a finished run (the device output, or the device copy
// of a host output). When the device is out of memory, the context's workspace, which the run no
// longer needs, is freed (the next call allocates it again) and the step runs once more; a second
// failure raises OutOfDeviceMemory.
template <class Step>
void without_workspace_on_oom(eda::Context& ctx, size_t bytes, Step&& step) {
  try {
    step();
    return;
  } catch (const eda::CudaError& e) {
    if (e.code() != cudaErrorMemoryAllocation) throw;
  }
  ctx.arena().free_all();
  try {
    step();
  } catch (const eda::CudaError& e) {
    if (e.code() != cudaErrorMemoryAllocation) throw;
    throw eda::OutOfDeviceMemory("eda: the output (" + eda::format_gib(static_cast<int64_t>(bytes)) +
                                 ") does not fit the device memory (" +
                                 eda::format_gib(eda::device_available_bytes(ctx.device())) + " free)");
  }
}

// Runs the GPU work of a call. When it fails for lack of memory (MemoryError), its buffers are gone
// by the time the handler runs, and the context gives the device memory it does not use back to
// the driver (Context::release_unused_memory) before the error propagates.
template <class Work>
void releasing_on_oom(eda::Context& ctx, Work&& work) {
  try {
    work();
  } catch (const std::bad_alloc&) {
    ctx.release_unused_memory();
    throw;
  } catch (const eda::CudaError& e) {
    if (e.code() == cudaErrorMemoryAllocation) ctx.release_unused_memory();
    throw;
  }
}

// --------------------------------------------------------------------------------------------
using HostInput = nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>;

// The host input (B, *spatial) of float32 or float64 as an Input (who: the entry point's name).
eda::Input host_input(const HostInput& x, const char* who) {
  eda::Input in;
  if (x.dtype() == nb::dtype<float>()) {
    in.dtype = eda::DType::F32;
  } else if (x.dtype() == nb::dtype<double>()) {
    in.dtype = eda::DType::F64;
  } else {
    throw nb::type_error((std::string("eda: ") + who + ": expected a float32 or float64 array").c_str());
  }
  if (x.ndim() < 2 || x.ndim() > static_cast<size_t>(eda::kMaxDim) + 1) {
    throw nb::value_error(
        (std::string("eda: ") + who + ": x must have the shape (B, *spatial) with 1 to 3 spatial axes").c_str());
  }
  in.dim = static_cast<int>(x.ndim()) - 1;
  in.batch = static_cast<int64_t>(x.shape(0));
  for (int a = 0; a < in.dim; ++a) in.n[a] = static_cast<int64_t>(x.shape(a + 1));
  in.x = x.data();
  in.on_device = false;
  return in;
}

// A NumPy array of `shape` in the compute dtype that owns the buffer from here on.
nb::object host_array(HostRows& buffer, const std::vector<size_t>& shape, eda::DType dtype) {
  return dtype == eda::DType::F32 ? buffer.to_numpy<float>(shape) : buffer.to_numpy<double>(shape);
}

nb::tuple decompose_host(const HostInput& x, int32_t max_imf, std::optional<double> tie_tolerance, int32_t sifts,
                         bool check_finite, bool orientation, int64_t max_chunk) {
  Call call;
  const eda::Input in = host_input(x, "decompose_host");
  const eda::Params p = params_of(max_imf, tie_tolerance, sifts, check_finite, max_chunk, orientation);
  validate_call(in, p);
  const eda::RuntimeConfig config = eda::RuntimeConfig::from_environment();

  Summary s;
  // the orientation's two host buffers and their K
  std::optional<HostRows> dir, coh;
  int32_t K = 0;
  const auto add_orientation = [&](nb::dict info) {
    if (!orientation) return info;
    const size_t B = static_cast<size_t>(in.batch), Ku = static_cast<size_t>(K);
    std::vector<size_t> dshape = {B, Ku, static_cast<size_t>(in.dim)}, cshape = {B, Ku};
    for (int a = 0; a < in.dim; ++a) {
      dshape.push_back(static_cast<size_t>(in.n[a]));
      cshape.push_back(static_cast<size_t>(in.n[a]));
    }
    if (!dir) dir.emplace(0);
    if (!coh) coh.emplace(0);
    info["direction"] = host_array(*dir, dshape, in.dtype);
    info["coherence"] = host_array(*coh, cshape, in.dtype);
    return info;
  };
  if (in.batch == 0) {  // nothing to compute: P = 1, no CUDA call
    HostRows empty(0);
    nb::object rows = in.dtype == eda::DType::F32 ? empty.to_numpy<float>(rows_shape(in, 1))
                                                  : empty.to_numpy<double>(rows_shape(in, 1));
    return nb::make_tuple(rows, add_orientation(info_dict(s)));
  }
  g_gpu_used = g_cuda_used = true;
  std::optional<HostRows> buffer;
  {
    nb::gil_scoped_release release;
    DrainOnExit drain;  // after the run is gone
    const int device = eda::current_device(config.visible_devices);
    eda::Context& ctx = eda::Context::current(device, config);
    eda::DeviceGuard guard(device);
    releasing_on_oom(ctx, [&] {
      std::unique_ptr<eda::Run> run = eda::decompose(ctx, in, p);
      const size_t bytes = output_bytes(*run, in.dtype);
      buffer.emplace(bytes);
      without_workspace_on_oom(ctx, bytes, [&] { run->collect(ctx, buffer->data(), false); });
      if (orientation) {
        K = largest_imfs(*run);
        const size_t nc = static_cast<size_t>(run->batch) * K * static_cast<size_t>(run->count) * item_size(in.dtype);
        dir.emplace(nc * static_cast<size_t>(in.dim));
        coh.emplace(nc);
        without_workspace_on_oom(ctx, nc * (in.dim + 1),
                                 [&] { run->collect_orientation(ctx, dir->data(), coh->data(), false, K); });
      }
      s = summarize(*run);
    });
  }
  const std::vector<size_t> shape = rows_shape(in, s.rows);
  nb::object rows = in.dtype == eda::DType::F32 ? buffer->to_numpy<float>(shape) : buffer->to_numpy<double>(shape);
  return nb::make_tuple(rows, add_orientation(info_dict(s)));
}

// --------------------------------------------------------------------------------------------
// A device result: its owner capsule queues the memory for release (no CUDA call, safe at any
// time, also during a consumer's graph capture and at interpreter teardown).
struct OutputMemory {
  void* ptr;
  int device;
};

void release_output(void* payload) noexcept {
  auto* memory = static_cast<OutputMemory*>(payload);
  eda::defer_device_free(memory->ptr, memory->device);
  delete memory;
}

nb::object output_owner(void* ptr, int device) {
  OutputMemory* memory = nullptr;
  try {
    memory = new OutputMemory{ptr, device};
    return nb::capsule(memory, release_output);
  } catch (...) {
    delete memory;
    eda::defer_device_free(ptr, device);
    throw;
  }
}

eda::DType compute_dtype(const std::string& target) {
  if (target == "float32") return eda::DType::F32;
  if (target == "float64") return eda::DType::F64;
  throw nb::type_error(("eda: the compute dtype must be float32 or float64, got " + target).c_str());
}

// The device input of a call as an Input (layout receives the gather of a converted input).
eda::Input device_input(uintptr_t ptr, const std::string& dtype, const std::string& target,
                        const std::vector<int64_t>& batch_shape, const std::vector<int64_t>& batch_strides,
                        const std::vector<int64_t>& shape, const std::vector<int64_t>& strides,
                        const std::vector<int64_t>& fences, const char* who, eda::SourceLayout& layout) {
  const std::optional<eda::SourceType> type = eda::source_type(dtype);
  if (!type) {
    throw nb::type_error(("eda: a CUDA array of dtype " + dtype +
                          " cannot be decomposed (supported: bool, int8 to int64, uint8 to uint64, float16, "
                          "bfloat16, float32, float64)")
                             .c_str());
  }
  const eda::DType compute = compute_dtype(target);
  if (shape.empty() || shape.size() > static_cast<size_t>(eda::kMaxDim) || shape.size() != strides.size()) {
    throw nb::value_error((std::string("eda: ") + who + ": 1 to 3 spatial axes, with one stride each").c_str());
  }
  for (const int64_t h : fences) {
    if (h < 0) throw nb::value_error(("eda: stream handle " + std::to_string(h) + " is negative").c_str());
  }
  eda::CudaArray a;
  a.data = reinterpret_cast<const void*>(ptr);
  a.type = *type;
  a.batch_shape = batch_shape;
  a.batch_strides = batch_strides;
  a.dim = static_cast<int>(shape.size());
  for (int d = 0; d < a.dim; ++d) {
    a.n[d] = shape[static_cast<size_t>(d)];
    a.strides[d] = strides[static_cast<size_t>(d)];
  }
  eda::Input in;
  eda::describe_device_input(a, compute, in, layout);
  return in;
}

nb::tuple decompose_device(uintptr_t ptr, int device, const std::string& dtype, const std::string& target,
                           const std::vector<int64_t>& batch_shape, const std::vector<int64_t>& batch_strides,
                           const std::vector<int64_t>& shape, const std::vector<int64_t>& strides,
                           const std::vector<int64_t>& fences, int32_t max_imf, std::optional<double> tie_tolerance,
                           int32_t sifts, bool check_finite, bool orientation, int64_t max_chunk) {
  Call call;
  eda::SourceLayout layout;
  const eda::Input in = device_input(ptr, dtype, target, batch_shape, batch_strides, shape, strides, fences,
                                     "decompose_device", layout);
  const eda::Params p = params_of(max_imf, tie_tolerance, sifts, check_finite, max_chunk, orientation);
  validate_call(in, p);
  const eda::RuntimeConfig config = eda::RuntimeConfig::from_environment();

  Summary s;
  if (in.batch == 0) {
    nb::dict info = info_dict(s);
    if (orientation) info["orientation"] = nb::make_tuple(nb::none(), uintptr_t{0});
    return nb::make_tuple(nb::none(), uintptr_t{0}, info);
  }
  g_gpu_used = g_cuda_used = true;
  void* out = nullptr;
  void* orient = nullptr;  // the orientation's buffer (nullptr: none, or no mode)
  {
    nb::gil_scoped_release release;
    DrainOnExit drain;  // after the run is gone
    eda::Context& ctx = eda::Context::current(device, config);
    eda::DeviceGuard guard(device);
    for (const int64_t h : fences) ctx.fence_on(static_cast<uintptr_t>(h));
    releasing_on_oom(ctx, [&] {
      std::unique_ptr<eda::Run> run = eda::decompose(ctx, in, p);
      const size_t bytes = output_bytes(*run, in.dtype);
      without_workspace_on_oom(ctx, bytes, [&] { out = eda::output_alloc_async(bytes, ctx.stream(), device); });
      try {
        run->collect(ctx, out, true);
        const int32_t K = orientation ? largest_imfs(*run) : 0;
        if (K > 0) {
          const size_t nc = static_cast<size_t>(run->batch) * K * static_cast<size_t>(run->count);
          const size_t ob = nc * (in.dim + 1) * item_size(in.dtype);
          without_workspace_on_oom(ctx, ob, [&] { orient = eda::output_alloc_async(ob, ctx.stream(), device); });
          run->collect_orientation(ctx, orient, static_cast<char*>(orient) + nc * in.dim * item_size(in.dtype), true,
                                   K);
        }
        ctx.synchronize();
      } catch (...) {
        if (orient) eda::device_free_async(orient, ctx.stream(), device);
        eda::device_free_async(out, ctx.stream(), device);
        throw;
      }
      s = summarize(*run);
    });
  }
  // both buffers have their owner before anything else can throw
  nb::object owner = output_owner(out, device);
  nb::object orient_owner = orient ? output_owner(orient, device) : nb::none();
  nb::dict info = info_dict(s);
  if (orientation) info["orientation"] = nb::make_tuple(orient_owner, reinterpret_cast<uintptr_t>(orient));
  return nb::make_tuple(owner, reinterpret_cast<uintptr_t>(out), info);
}

// --------------------------------------------------------------------------------------------
// eda.monogenic: the output planes into one device buffer (eda::run_monogenic).

// The information of a monogenic call: nonfinite ([B] int32) and chunk (elements per chunk).
nb::dict monogenic_info(std::vector<int32_t> nonfinite, int64_t batch, int64_t chunk) {
  nonfinite.resize(static_cast<size_t>(batch), 0);
  nb::dict info;
  info["nonfinite"] = numpy_of(std::move(nonfinite), {static_cast<size_t>(batch)});
  info["chunk"] = chunk;
  return info;
}

nb::tuple monogenic_host(const HostInput& x, bool check_finite, int64_t max_chunk) {
  Call call;
  const eda::Input in = host_input(x, "monogenic_host");
  validate_call(in, eda::Params{});
  const eda::RuntimeConfig config = eda::RuntimeConfig::from_environment();
  const size_t bytes = eda::monogenic_bytes(in);
  const size_t elements = bytes / item_size(in.dtype);
  HostRows buffer(bytes);
  std::vector<int32_t> nonfinite;
  int64_t chunk = 0;
  if (in.batch > 0) {
    g_gpu_used = g_cuda_used = true;
    nb::gil_scoped_release release;
    DrainOnExit drain;
    const int device = eda::current_device(config.visible_devices);
    eda::Context& ctx = eda::Context::current(device, config);
    eda::DeviceGuard guard(device);
    releasing_on_oom(ctx, [&] {
      std::optional<eda::DeviceBuffer<unsigned char>> out;
      without_workspace_on_oom(ctx, bytes, [&] { out.emplace(bytes, ctx.stream(), device); });
      chunk = eda::run_monogenic(ctx, in, check_finite, max_chunk, out->data(), nonfinite);
      EDA_CUDA(cudaMemcpyAsync(buffer.data(), out->data(), bytes, cudaMemcpyDeviceToHost, ctx.stream()));
      ctx.synchronize();
    });
  }
  return nb::make_tuple(host_array(buffer, {elements}, in.dtype),
                        monogenic_info(std::move(nonfinite), in.batch, chunk));
}

nb::tuple monogenic_device(uintptr_t ptr, int device, const std::string& dtype, const std::string& target,
                           const std::vector<int64_t>& batch_shape, const std::vector<int64_t>& batch_strides,
                           const std::vector<int64_t>& shape, const std::vector<int64_t>& strides,
                           const std::vector<int64_t>& fences, bool check_finite, int64_t max_chunk) {
  Call call;
  eda::SourceLayout layout;
  const eda::Input in = device_input(ptr, dtype, target, batch_shape, batch_strides, shape, strides, fences,
                                     "monogenic_device", layout);
  validate_call(in, eda::Params{});
  const eda::RuntimeConfig config = eda::RuntimeConfig::from_environment();
  std::vector<int32_t> nonfinite;
  if (in.batch == 0) return nb::make_tuple(nb::none(), uintptr_t{0}, monogenic_info(std::move(nonfinite), 0, 0));
  g_gpu_used = g_cuda_used = true;
  const size_t bytes = eda::monogenic_bytes(in);
  void* out = nullptr;
  int64_t chunk = 0;
  {
    nb::gil_scoped_release release;
    DrainOnExit drain;
    eda::Context& ctx = eda::Context::current(device, config);
    eda::DeviceGuard guard(device);
    for (const int64_t h : fences) ctx.fence_on(static_cast<uintptr_t>(h));
    releasing_on_oom(ctx, [&] {
      without_workspace_on_oom(ctx, bytes, [&] { out = eda::output_alloc_async(bytes, ctx.stream(), device); });
      try {
        chunk = eda::run_monogenic(ctx, in, check_finite, max_chunk, out, nonfinite);
        ctx.synchronize();
      } catch (...) {
        eda::device_free_async(out, ctx.stream(), device);
        throw;
      }
    });
  }
  nb::object owner = output_owner(out, device);
  return nb::make_tuple(owner, reinterpret_cast<uintptr_t>(out), monogenic_info(std::move(nonfinite), in.batch, chunk));
}

// --------------------------------------------------------------------------------------------
std::string dlpack_dtype_name(const nb::dlpack::dtype& t) {
  using Code = nb::dlpack::dtype_code;
  const std::string bits = std::to_string(t.bits);
  if (t.lanes == 1) {
    switch (static_cast<Code>(t.code)) {
      case Code::Bool:
        if (t.bits == 8) return "bool";
        break;
      case Code::Int:
        if (t.bits == 8 || t.bits == 16 || t.bits == 32 || t.bits == 64) return "int" + bits;
        break;
      case Code::UInt:
        if (t.bits == 8 || t.bits == 16 || t.bits == 32 || t.bits == 64) return "uint" + bits;
        break;
      case Code::Float:
        if (t.bits == 16 || t.bits == 32 || t.bits == 64) return "float" + bits;
        break;
      case Code::Bfloat:
        if (t.bits == 16) return "bfloat16";
        break;
      case Code::Complex:
        if (t.bits == 64 || t.bits == 128) return "complex" + bits;
        break;
      default:
        break;
    }
  }
  return "dlpack(code=" + std::to_string(t.code) + ", bits=" + bits + ", lanes=" + std::to_string(t.lanes) + ")";
}

using Imported = nb::ndarray<nb::ro>;

// (owner, ptr, device_type, device_id, dtype, shape, strides): the capsule is consumed; owner keeps
// the producer's memory until it is dropped. ptr includes the byte offset; strides in elements.
nb::tuple dlpack_info(nb::handle capsule) {
  Imported array;
  if (!nb::try_cast<Imported>(capsule, array, false)) {
    throw nb::type_error("eda: expected an unused DLPack capsule (\"dltensor\" or \"dltensor_versioned\")");
  }
  nb::list shape, strides;
  for (size_t i = 0; i < array.ndim(); ++i) {
    shape.append(static_cast<int64_t>(array.shape(i)));
    strides.append(array.stride(i));
  }
  auto* keep = new Imported(array);
  nb::capsule owner(keep, [](void* p) noexcept { delete static_cast<Imported*>(p); });
  return nb::make_tuple(owner, reinterpret_cast<uintptr_t>(array.data()), static_cast<int>(array.device_type()),
                        static_cast<int>(array.device_id()), dlpack_dtype_name(array.dtype()), nb::tuple(shape),
                        nb::tuple(strides));
}

// The device of a CUDA array interface pointer: device or managed memory; 0 gives the calling
// thread's current device (an empty array has no pointer).
int pointer_device(uintptr_t ptr) {
  Call call;
  note_cuda_used();
  std::optional<std::string> visible;
  if (const char* v = std::getenv("CUDA_VISIBLE_DEVICES")) visible = std::string(v);
  cudaPointerAttributes attributes{};
  {
    nb::gil_scoped_release release;
    if (eda::device_count() == 0) eda::require_supported_device(0, visible);  // throws the no-device message
    if (ptr == 0) return eda::current_device(visible);
    EDA_CUDA(cudaPointerGetAttributes(&attributes, reinterpret_cast<const void*>(ptr)));
  }
  if (attributes.type != cudaMemoryTypeDevice && attributes.type != cudaMemoryTypeManaged) {
    char text[96];
    std::snprintf(text, sizeof text, "eda: the data pointer 0x%llx is not CUDA device or managed memory",
                  static_cast<unsigned long long>(ptr));
    throw nb::value_error(text);
  }
  return attributes.device;
}

// An object with __dlpack__ and __dlpack_device__ over device memory that `owner` keeps alive.
nb::object dlpack_export(nb::handle owner, uintptr_t ptr, const std::vector<size_t>& shape,
                         const std::vector<int64_t>& strides, const std::string& dtype, int device) {
  if (shape.size() != strides.size()) throw nb::value_error("eda: dlpack_export: shape and strides differ in length");
  nb::dlpack::dtype dt;
  if (dtype == "float32") {
    dt = nb::dtype<float>();
  } else if (dtype == "float64") {
    dt = nb::dtype<double>();
  } else {
    throw nb::type_error(("eda: dlpack_export: float32 or float64, got " + dtype).c_str());
  }
  return nb::ndarray<nb::array_api>(reinterpret_cast<void*>(ptr), shape.size(), shape.data(), owner, strides.data(), dt,
                                    nb::device::cuda::value, device)
      .cast();
}

}  // namespace

Call::Call() {
  refuse_in_forked_child();
  if (g_exiting) throw std::runtime_error("eda: the interpreter is shutting down; no new call can start");
  ++g_calls;
}

Call::~Call() { --g_calls; }

void note_cuda_used() {
  refuse_in_forked_child();
  g_cuda_used = true;
}

void bind_decompose(nb::module_& m) {
  nb::module_::import_("atexit").attr("register")(nb::cpp_function(&release_at_exit));
  nb::dict fork_hooks;
  fork_hooks["after_in_child"] = nb::cpp_function(&after_fork_in_child);
  nb::module_::import_("os").attr("register_at_fork")(**fork_hooks);

  m.def("decompose_host", &decompose_host, nb::arg("x").noconvert(), nb::arg("max_imf"), nb::arg("tie_tolerance").none(),
        nb::arg("sifts"), nb::arg("check_finite"), nb::arg("orientation"), nb::arg("max_chunk"),
        "Decompose the C-contiguous host array x (B, *spatial) of float32 or float64 on the calling thread's "
        "current device: (rows (B, P, *spatial), info); with orientation, info holds direction and coherence.");
  m.def("decompose_device", &decompose_device, nb::arg("ptr"), nb::arg("device"), nb::arg("dtype"), nb::arg("target"),
        nb::arg("batch_shape"), nb::arg("batch_strides"), nb::arg("shape"), nb::arg("strides"), nb::arg("fences"),
        nb::arg("max_imf"), nb::arg("tie_tolerance").none(), nb::arg("sifts"), nb::arg("check_finite"),
        nb::arg("orientation"), nb::arg("max_chunk"),
        "Decompose a CUDA array on its device after waiting for the fence streams: (owner, ptr, info), the rows "
        "C-contiguous (B, P, *spatial) in target, complete on return; with orientation, info['orientation'] is "
        "(owner, ptr) of the direction (B, K, D, *spatial) followed by the coherence (B, K, *spatial).");
  m.def("monogenic_host", &monogenic_host, nb::arg("x").noconvert(), nb::arg("check_finite"), nb::arg("max_chunk"),
        "The monogenic signal of the C-contiguous host array x (B, *spatial) of float32 or float64: (out, "
        "info), info holding nonfinite and chunk, out the flat planes amplitude, phase, frequency (B, *spatial), "
        "then for 2 or 3 spatial axes direction (B, D, *spatial) and coherence (B, *spatial).");
  m.def("monogenic_device", &monogenic_device, nb::arg("ptr"), nb::arg("device"), nb::arg("dtype"), nb::arg("target"),
        nb::arg("batch_shape"), nb::arg("batch_strides"), nb::arg("shape"), nb::arg("strides"), nb::arg("fences"),
        nb::arg("check_finite"), nb::arg("max_chunk"),
        "The monogenic signal of a CUDA array after waiting for the fence streams: (owner, ptr, info), the "
        "planes of monogenic_host in one device buffer, complete on return.");
  m.def("dlpack_info", &dlpack_info, nb::arg("capsule"),
        "Import a DLPack capsule: (owner, ptr, device_type, device_id, dtype, shape, strides in elements).");
  m.def("pointer_device", &pointer_device, nb::arg("ptr"), "The CUDA device of a device or managed pointer.");
  m.def("dlpack_export", &dlpack_export, nb::arg("owner").none(), nb::arg("ptr"), nb::arg("shape"), nb::arg("strides"),
        nb::arg("dtype"), nb::arg("device"),
        "A DLPack exporter of float32/float64 device memory kept alive by owner (strides in elements).");
}
