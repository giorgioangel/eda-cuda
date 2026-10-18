// SPDX-License-Identifier: Apache-2.0
#include "runtime/memory.hpp"

#include "runtime/device.hpp"
#include "runtime/error.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <mutex>

namespace eda {
namespace {

constexpr size_t kMinArenaBytes = size_t{1} << 16;

size_t round_up(size_t value, size_t multiple) { return (value + multiple - 1) / multiple * multiple; }

std::mutex g_pools_mutex;
std::map<int, cudaMemPool_t> g_pools;         // workspace pools (memory kept cached)
std::map<int, cudaMemPool_t> g_output_pools;  // output pools (memory released at every drain)

std::atomic<bool> g_forked_child{false};

// A pool of `device` with the given release threshold; the caller holds g_pools_mutex.
cudaMemPool_t create_pool(int device, uint64_t release_threshold) {
  cudaMemPoolProps props{};
  props.allocType = cudaMemAllocationTypePinned;
  props.handleTypes = cudaMemHandleTypeNone;
  props.location.type = cudaMemLocationTypeDevice;
  props.location.id = device;
  cudaMemPool_t pool = nullptr;
  EDA_CUDA(cudaMemPoolCreate(&pool, &props));
  const cudaError_t status = cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &release_threshold);
  if (status != cudaSuccess) {
    (void)cudaMemPoolDestroy(pool);
    throw_cuda_error(status, "cudaMemPoolSetAttribute(cudaMemPoolAttrReleaseThreshold)", __FILE__, __LINE__);
  }
  return pool;
}

cudaMemPool_t pool_of(std::map<int, cudaMemPool_t>& pools, int device, uint64_t release_threshold) {
  std::lock_guard<std::mutex> lock(g_pools_mutex);
  const auto it = pools.find(device);
  if (it != pools.end()) return it->second;
  cudaMemPool_t pool = create_pool(device, release_threshold);
  pools.emplace(device, pool);
  return pool;
}

// Gives the output pool's freed memory of `device` back to the driver (device current).
void trim_output_pool(int device) noexcept {
  std::lock_guard<std::mutex> lock(g_pools_mutex);
  const auto it = g_output_pools.find(device);
  if (it != g_output_pools.end() && cudaMemPoolTrimTo(it->second, 0) != cudaSuccess) (void)cudaGetLastError();
}

struct DeferredFrees {
  std::mutex mutex;
  std::vector<std::pair<void*, int>> queue;  // (pointer, device)
};
// Leaked on purpose: deleters may run during interpreter teardown, after static destructors.
DeferredFrees& deferred() {
  static DeferredFrees* instance = new DeferredFrees;
  return *instance;
}

bool is_capture_error(cudaError_t code) {
  return code == cudaErrorStreamCaptureUnsupported || code == cudaErrorStreamCaptureImplicit ||
         code == cudaErrorStreamCaptureInvalidated || code == cudaErrorStreamCaptureWrongThread ||
         code == cudaErrorStreamCaptureUnjoined || code == cudaErrorStreamCaptureIsolation ||
         code == cudaErrorStreamCaptureMerge || code == cudaErrorStreamCaptureUnmatched;
}

// The pinned host pool (memory.hpp, PinnedBlock). Leaked on purpose: blocks may be returned during
// process teardown, after static destructors.
constexpr size_t kPinnedGranule = size_t{2} << 20;
struct PinnedPool {
  std::mutex mutex;
  std::multimap<size_t, void*> idle;  // bytes -> block
};
PinnedPool& pinned_pool() {
  static PinnedPool* instance = new PinnedPool;
  return *instance;
}
// Frees the idle blocks for which `doomed(bytes)` holds; the caller holds the pool's mutex.
template <class Pred>
void free_idle_pinned(PinnedPool& pool, Pred doomed) noexcept {
  for (auto it = pool.idle.begin(); it != pool.idle.end();) {
    if (!doomed(it->first)) {
      ++it;
      continue;
    }
    (void)cudaFreeHost(it->second);
    it = pool.idle.erase(it);
  }
  (void)cudaGetLastError();
}

}  // namespace

PinnedBlock pinned_acquire(size_t bytes) {
  PinnedPool& pool = pinned_pool();
  PinnedBlock out;
  if (bytes == 0) return out;
  if (bytes > std::numeric_limits<size_t>::max() - kPinnedGranule) throw std::length_error("eda: pinned block size overflow");
  const size_t rounded = round_up(bytes, kPinnedGranule);
  std::lock_guard<std::mutex> lock(pool.mutex);
  const auto it = pool.idle.lower_bound(rounded);
  if (it != pool.idle.end()) {
    out.data_ = it->second;
    out.bytes_ = it->first;
    pool.idle.erase(it);
    return out;
  }
  // Only inside library calls (which no capture may overlap): the idle blocks below the request are
  // what contexts outgrew.
  free_idle_pinned(pool, [rounded](size_t b) { return b < rounded; });
  void* ptr = nullptr;
  const cudaError_t status = cudaHostAlloc(&ptr, rounded, cudaHostAllocPortable);
  if (status != cudaSuccess) {
    (void)cudaGetLastError();
    throw_cuda_error(status, "cudaHostAlloc(" + std::to_string(rounded) + " bytes)", __FILE__, __LINE__);
  }
  out.data_ = ptr;
  out.bytes_ = rounded;
  return out;
}

void PinnedBlock::reset() noexcept {
  if (!data_) return;
  if (forked_child()) {  // the block belongs to the parent process
    data_ = nullptr;
    bytes_ = 0;
    return;
  }
  PinnedPool& pool = pinned_pool();
  try {
    std::lock_guard<std::mutex> lock(pool.mutex);
    pool.idle.emplace(bytes_, data_);
  } catch (...) {
    // Out of host memory while returning the block: it leaks (no CUDA call is safe here).
  }
  data_ = nullptr;
  bytes_ = 0;
}

void mark_forked_child() noexcept { g_forked_child.store(true, std::memory_order_relaxed); }
bool forked_child() noexcept { return g_forked_child.load(std::memory_order_relaxed); }

void defer_device_free(void* ptr, int device) noexcept {
  if (!ptr || forked_child()) return;
  DeferredFrees& d = deferred();
  try {
    std::lock_guard<std::mutex> lock(d.mutex);
    d.queue.emplace_back(ptr, device);
  } catch (...) {
    // Out of host memory while queueing: the device memory leaks (nothing safe can be done here).
  }
}

int64_t drain_deferred_frees() noexcept {
  if (forked_child()) return 0;
  DeferredFrees& d = deferred();
  std::vector<std::pair<void*, int>> work;
  {
    std::lock_guard<std::mutex> lock(d.mutex);
    if (d.queue.empty()) return 0;
    work.swap(d.queue);
  }
  std::map<int, std::vector<void*>> by_device;
  for (const auto& [ptr, device] : work) by_device[device].push_back(ptr);

  int previous = -1;
  const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
  std::vector<std::pair<void*, int>> keep;
  int64_t freed = 0;
  for (const auto& [device, pointers] : by_device) {
    cudaError_t status = cudaSetDevice(device);
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
      (void)cudaGetLastError();
      if (is_capture_error(status)) {
        for (void* ptr : pointers) keep.emplace_back(ptr, device);
        continue;
      }
      // A sticky device error (the context is unusable): the memory cannot be freed any more.
      continue;
    }
    // Stream-ordered frees (the pool's own release path; the device is idle), completed before the
    // trim so that the output pool can give the memory back.
    for (void* ptr : pointers) {
      if (cudaFreeAsync(ptr, cudaStreamPerThread) == cudaSuccess) {
        ++freed;
      } else {
        (void)cudaGetLastError();
      }
    }
    if (cudaStreamSynchronize(cudaStreamPerThread) != cudaSuccess) (void)cudaGetLastError();
    trim_output_pool(device);
  }
  if (have_previous) (void)cudaSetDevice(previous);
  (void)cudaGetLastError();
  try {
    std::lock_guard<std::mutex> lock(d.mutex);
    d.queue.insert(d.queue.end(), keep.begin(), keep.end());
  } catch (...) {
  }
  return freed;
}

cudaMemPool_t device_pool(int device) { return pool_of(g_pools, device, std::numeric_limits<uint64_t>::max()); }

void release_device_pools() noexcept {
  {
    PinnedPool& pool = pinned_pool();
    std::lock_guard<std::mutex> lock(pool.mutex);
    free_idle_pinned(pool, [](size_t) { return true; });
  }
  std::lock_guard<std::mutex> lock(g_pools_mutex);
  for (auto* pools : {&g_pools, &g_output_pools}) {
    for (const auto& [device, pool] : *pools) {
      (void)cudaMemPoolTrimTo(pool, 0);
      (void)cudaMemPoolDestroy(pool);
    }
    pools->clear();
  }
  (void)cudaGetLastError();
}

void trim_device_pools(int device) noexcept {
  std::lock_guard<std::mutex> lock(g_pools_mutex);
  for (auto* pools : {&g_pools, &g_output_pools}) {
    const auto it = pools->find(device);
    if (it != pools->end() && cudaMemPoolTrimTo(it->second, 0) != cudaSuccess) (void)cudaGetLastError();
  }
}

void* device_alloc_async(size_t bytes, cudaStream_t stream, int device) {
  DeviceGuard guard(device);
  void* ptr = nullptr;
  const cudaError_t status = cudaMallocFromPoolAsync(&ptr, bytes, device_pool(device), stream);
  if (status != cudaSuccess) {
    throw_cuda_error(status, "cudaMallocFromPoolAsync(" + std::to_string(bytes) + " bytes)", __FILE__, __LINE__);
  }
  return ptr;
}

void* output_alloc_async(size_t bytes, cudaStream_t stream, int device) {
  DeviceGuard guard(device);
  cudaMemPool_t pool = pool_of(g_output_pools, device, 0);
  void* ptr = nullptr;
  cudaError_t status = cudaMallocFromPoolAsync(&ptr, bytes, pool, stream);
  if (status == cudaErrorMemoryAllocation) {
    // The memory may be held unused by the workspace pool or by dropped results: give it back to
    // the driver and try once more.
    (void)cudaGetLastError();
    (void)drain_deferred_frees();
    if (cudaStreamSynchronize(stream) == cudaSuccess) (void)cudaMemPoolTrimTo(device_pool(device), 0);
    (void)cudaGetLastError();
    status = cudaMallocFromPoolAsync(&ptr, bytes, pool, stream);
  }
  if (status != cudaSuccess) {
    throw_cuda_error(status, "cudaMallocFromPoolAsync(" + std::to_string(bytes) + " bytes, output)", __FILE__, __LINE__);
  }
  return ptr;
}

void device_free_async(void* ptr, cudaStream_t stream, int device) noexcept {
  if (!ptr) return;
  int previous = -1;
  const bool switched = cudaGetDevice(&previous) == cudaSuccess && previous != device &&
                        cudaSetDevice(device) == cudaSuccess;
  if (cudaFreeAsync(ptr, stream) != cudaSuccess) (void)cudaGetLastError();
  if (switched) (void)cudaSetDevice(previous);
}

int64_t device_available_bytes(int device) {
  DeviceGuard guard(device);
  size_t free_bytes = 0, total_bytes = 0;
  EDA_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
  unsigned long long reserved = 0, used = 0;  // cudaMemPoolGetAttribute writes a cuuint64_t
  cudaMemPool_t pool = device_pool(device);
  EDA_CUDA(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved));
  EDA_CUDA(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used));
  return static_cast<int64_t>(free_bytes) + static_cast<int64_t>(reserved > used ? reserved - used : 0);
}

std::string format_gib(int64_t bytes) {
  char text[32];
  std::snprintf(text, sizeof text, "%.2f GiB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
  return text;
}

size_t ArenaLayout::bytes() const {
  size_t total = 0;
  for (size_t b : bytes_) total += round_up(b, kArenaAlignment);
  return total;
}

Arena::Arena(int device, cudaStream_t stream, bool debug, bool poison)
    : device_(device), stream_(stream), debug_(debug), poison_(debug && poison) {}

Arena::~Arena() { release(); }

void Arena::free_all() noexcept {
  for (void* slot : debug_slots_) device_free_async(slot, stream_, device_);
  debug_slots_.clear();
  device_free_async(base_, stream_, device_);
  base_ = nullptr;
  capacity_ = 0;
}

void Arena::release() noexcept {
  while (!attachments_.empty()) attachments_.pop_back();  // newest first, before the memory goes
  free_all();
}

ArenaBinding Arena::bind(const ArenaLayout& layout) {
  ArenaBinding binding;
  binding.pointers_.assign(layout.slots(), nullptr);
  if (debug_) {
    // Fresh allocations on every bind: nothing written by an earlier call can be read back.
    free_all();
    for (size_t i = 0; i < layout.slots(); ++i) {
      const size_t bytes = layout.slot_bytes(i);
      if (!bytes) continue;
      void* ptr = device_alloc_async(bytes, stream_, device_);
      debug_slots_.push_back(ptr);
      binding.pointers_[i] = ptr;
      if (poison_) {
        DeviceGuard guard(device_);
        EDA_CUDA(cudaMemsetAsync(ptr, 0xFF, bytes, stream_));
      }
      capacity_ += bytes;
    }
    return binding;
  }
  const size_t needed = layout.bytes();
  if (needed > capacity_) {
    const size_t grown = round_up(std::max({needed, capacity_ * 2, kMinArenaBytes}), kArenaAlignment);
    free_all();
    base_ = device_alloc_async(grown, stream_, device_);
    capacity_ = grown;
  }
  size_t offset = 0;
  for (size_t i = 0; i < layout.slots(); ++i) {
    const size_t bytes = layout.slot_bytes(i);
    if (bytes) binding.pointers_[i] = static_cast<char*>(base_) + offset;
    offset += round_up(bytes, kArenaAlignment);
  }
  return binding;
}

}  // namespace eda
