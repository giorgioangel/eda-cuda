// SPDX-License-Identifier: Apache-2.0
#include "runtime/context.hpp"

#include "runtime/device.hpp"
#include "runtime/env.hpp"
#include "runtime/error.hpp"

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace eda {
namespace {

using Key = std::pair<std::thread::id, int>;

struct Registry {
  std::mutex mutex;
  std::map<Key, std::unique_ptr<Context>> contexts;
  std::atomic<uint64_t> epoch{1};  // bumped by shutdown(): thread caches revalidate
};

// Leaked on purpose: thread-exit cleanup may run during process teardown.
Registry& registry() {
  static Registry* instance = new Registry;
  return *instance;
}

// Per-thread lookup cache; its destructor (thread exit) destroys the thread's contexts, except in
// a forked child, where they belong to the parent (memory.hpp, mark_forked_child).
struct ThreadContexts {
  uint64_t epoch = 0;
  std::map<int, Context*> by_device;
  ~ThreadContexts() {
    if (forked_child()) return;
    std::vector<std::unique_ptr<Context>> mine;
    {
      Registry& r = registry();
      std::lock_guard<std::mutex> lock(r.mutex);
      const std::thread::id self = std::this_thread::get_id();
      for (auto it = r.contexts.begin(); it != r.contexts.end();) {
        if (it->first.first == self) {
          mine.push_back(std::move(it->second));
          it = r.contexts.erase(it);
        } else {
          ++it;
        }
      }
    }
    mine.clear();  // destroyed outside the registry lock
  }
};
thread_local ThreadContexts t_contexts;

}  // namespace

cudaStream_t Context::create_stream(int device) {
  DeviceGuard guard(device);
  cudaStream_t stream = nullptr;
  EDA_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  return stream;
}

Context::Context(int device, const RuntimeConfig& config)
    : device_(device),
      config_(config),
      stream_(create_stream(device)),
      arena_(device, stream_, config.sanitize, config.arena_poison) {
  try {
    DeviceGuard guard(device_);
    EDA_CUDA(cudaEventCreateWithFlags(&fence_event_, cudaEventDisableTiming));
  } catch (...) {
    (void)cudaStreamDestroy(stream_);
    stream_ = nullptr;
    throw;
  }
}

Context::~Context() {
  int previous = -1;
  const bool switched = cudaGetDevice(&previous) == cudaSuccess && previous != device_ &&
                        cudaSetDevice(device_) == cudaSuccess;
  if (stream_) (void)cudaStreamSynchronize(stream_);
  arena_.release();
  if (stream_) (void)cudaStreamSynchronize(stream_);
  if (fence_event_) (void)cudaEventDestroy(fence_event_);
  if (stream_) (void)cudaStreamDestroy(stream_);
  (void)cudaGetLastError();
  if (switched) (void)cudaSetDevice(previous);
}

Context& Context::current(int device) { return current_impl(device, nullptr); }

Context& Context::current(int device, const RuntimeConfig& config) { return current_impl(device, &config); }

Context& Context::current_impl(int device, const RuntimeConfig* config) {
  Registry& r = registry();
  ThreadContexts& cache = t_contexts;
  const uint64_t epoch = r.epoch.load(std::memory_order_acquire);
  if (cache.epoch == epoch) {
    const auto it = cache.by_device.find(device);
    if (it != cache.by_device.end()) return *it->second;
  } else {
    cache.by_device.clear();
    cache.epoch = epoch;
  }
  // Without a snapshot from the caller, the environment is read here (context creation only).
  std::optional<RuntimeConfig> own;
  if (!config) config = &own.emplace(RuntimeConfig::from_environment());
  require_supported_device(device, config->visible_devices);
  const Key key{std::this_thread::get_id(), device};
  {
    std::lock_guard<std::mutex> lock(r.mutex);
    const auto it = r.contexts.find(key);
    if (it != r.contexts.end()) {
      cache.by_device[device] = it->second.get();
      return *it->second;
    }
  }
  std::unique_ptr<Context> created(new Context(device, *config));
  Context& out = *created;
  {
    std::lock_guard<std::mutex> lock(r.mutex);
    r.contexts.emplace(key, std::move(created));
  }
  cache.by_device[device] = &out;
  return out;
}

void Context::shutdown() noexcept {
  Registry& r = registry();
  std::map<Key, std::unique_ptr<Context>> doomed;
  {
    std::lock_guard<std::mutex> lock(r.mutex);
    doomed.swap(r.contexts);
    r.epoch.fetch_add(1, std::memory_order_acq_rel);
  }
  doomed.clear();
  // Exported memory released by consumers goes back before the pools are destroyed.
  (void)drain_deferred_frees();
  release_device_pools();
}

void Context::synchronize() {
  DeviceGuard guard(device_);
  EDA_CUDA(cudaStreamSynchronize(stream_));
}

void Context::release_unused_memory() noexcept {
  (void)cudaStreamSynchronize(stream_);
  arena_.release();
  (void)cudaStreamSynchronize(stream_);  // the frees are stream-ordered: complete them before the trim
  (void)cudaGetLastError();
  trim_device_pools(device_);
}

cudaStream_t Context::stream_from_handle(uintptr_t handle) noexcept {
  if (handle == 0 || handle == 1) return cudaStreamLegacy;
  if (handle == 2) return cudaStreamPerThread;
  return reinterpret_cast<cudaStream_t>(handle);
}

void Context::fence_on(uintptr_t handle) { fence_on_stream(stream_from_handle(handle)); }

void Context::fence_on_stream(cudaStream_t producer) {
  if (producer == stream_) return;
  DeviceGuard guard(device_);
  EDA_CUDA(cudaEventRecord(fence_event_, producer));
  EDA_CUDA(cudaStreamWaitEvent(stream_, fence_event_, 0));
}

}  // namespace eda
