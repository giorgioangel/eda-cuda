// SPDX-License-Identifier: Apache-2.0
// Per-(thread, device) execution context: a non-blocking stream and the workspace arena. A thread
// creates its context on first use; the context is destroyed when that thread exits or by
// shutdown().
#pragma once

#include "runtime/env.hpp"
#include "runtime/memory.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace eda {

class Context {
 public:
  // The calling thread's context for `device`, created on first use: startup device check, stream
  // and arena creation. A new context takes its settings (sanitize mode, which is also the arena's
  // debug mode, and poisoning) from `config`; an existing one keeps the settings it was created
  // with. The first form reads the environment (RuntimeConfig::from_environment) only when it
  // creates a context; code that runs while other threads may modify the environment (the Python
  // bindings, with the GIL released) passes a snapshot taken beforehand instead.
  static Context& current(int device);
  static Context& current(int device, const RuntimeConfig& config);

  // Destroys every context of every thread (synchronizing their streams first), drains the
  // deferred frees (memory.hpp) and destroys the memory pools. The devices are not reset: what
  // libraries such as cuFFT keep for the life of the process is released when they unload, and
  // memcheck's leak check at exit then sees only the library's own allocations. Only valid while
  // no other thread uses the library (process or test-session teardown).
  static void shutdown() noexcept;

  ~Context();
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  int device() const noexcept { return device_; }
  // The settings this context was created with.
  const RuntimeConfig& config() const noexcept { return config_; }
  cudaStream_t stream() const noexcept { return stream_; }
  Arena& arena() noexcept { return arena_; }

  // Makes this context's stream wait for the work already queued on a producer stream, through an
  // event (no host synchronization). `handle` follows the CUDA array interface and DLPack:
  // 0 or 1 = legacy default stream, 2 = per-thread default stream, any other value = cudaStream_t.
  void fence_on(uintptr_t handle);
  void fence_on_stream(cudaStream_t producer);

  // Blocks until the context stream is idle.
  void synchronize();

  // After a call failed for lack of device memory, once its buffers are destroyed: waits for the
  // stream, frees the workspace and the cached cuFFT plans (Arena::release; the next call makes
  // them again) and trims the device's pools, so that the memory the library does not use goes
  // back to the driver. Memory still allocated (results, other contexts) stays.
  void release_unused_memory() noexcept;

 private:
  Context(int device, const RuntimeConfig& config);
  static cudaStream_t create_stream(int device);
  static Context& current_impl(int device, const RuntimeConfig* config);
  static cudaStream_t stream_from_handle(uintptr_t handle) noexcept;

  int device_;
  RuntimeConfig config_;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t fence_event_ = nullptr;
  Arena arena_;
};

}  // namespace eda
