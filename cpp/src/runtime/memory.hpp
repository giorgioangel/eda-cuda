// SPDX-License-Identifier: Apache-2.0
// Device memory: an owning stream-ordered buffer and the per-context workspace arena.
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

namespace eda {

// The library's own stream-ordered memory pool of `device`, created on first use. Its release
// threshold keeps freed memory cached, so steady-state allocations neither map new memory nor
// synchronize the device (the device's default pool releases memory at every synchronization,
// and re-mapping it can block on unrelated work). The process-wide default pool is left untouched.
cudaMemPool_t device_pool(int device);
// Trims and destroys the pools (Context::shutdown). Allocations still
// alive keep their pool alive until they are freed. Also frees the idle pinned host blocks
// (PinnedBlock below).
void release_device_pools() noexcept;
// Gives the unused memory of both pools of `device` (workspace and output) back to the driver
// (cudaMemPoolTrimTo 0). Memory whose stream-ordered free has not completed yet stays, so the
// caller synchronizes the streams of those frees first. Live allocations are untouched.
void trim_device_pools(int device) noexcept;

// Stream-ordered allocation from device_pool(device) on `stream` (cudaMallocFromPoolAsync) and
// release (cudaFreeAsync). device_free_async never throws (destructors use it).
void* device_alloc_async(size_t bytes, cudaStream_t stream, int device);
void device_free_async(void* ptr, cudaStream_t stream, int device) noexcept;

// Stream-ordered allocation of a result that is handed to other libraries (the device output of
// the Python bindings), from a second pool of `device` that keeps no freed memory: every drain
// that frees such results (drain_deferred_frees) trims it, so a dropped result goes back to the
// driver, where other libraries can allocate it, instead of staying cached for the library. Its
// release is defer_device_free (or device_free_async before it is handed out). When the device is
// out of memory, it drains the deferred frees, trims the workspace pool and tries once more; then
// it throws CudaError (cudaErrorMemoryAllocation).
void* output_alloc_async(size_t bytes, cudaStream_t stream, int device);

// Device memory a call may still allocate on `device`: the free device memory (cudaMemGetInfo)
// plus what the library's pool holds unused (freed allocations it keeps cached, which the next
// allocations reuse). The driver compares its large allocations with it, so that a request beyond
// the device raises OutOfDeviceMemory naming the call instead of failing in the allocator.
int64_t device_available_bytes(int device);
// "1.50 GiB" (two decimals), for memory messages.
std::string format_gib(int64_t bytes);

// Deferred release of device memory whose last user is outside the library, for example a result
// handed to a framework through DLPack. Such a release can happen at any time: DLPack deleters run
// with the GIL held, possibly while the consumer captures a CUDA graph in global capture mode,
// where cudaDeviceSynchronize and cudaFree are illegal and invalidate the capture. The consumer may
// also still have work queued on its own streams that reads the memory, and cudaFree of
// stream-ordered memory does not wait for it.
//
// defer_device_free() therefore makes no CUDA call: it only queues the pointer, so a release is
// safe at any time, including inside a capture. Library calls drain the queue with
// drain_deferred_frees() at their end, after their own work (at the start, the device
// synchronization would act as an accidental fence on the caller's producer streams and hide a
// missing fence), with the GIL released: it synchronizes each device that has queued pointers (all
// consumer work is then complete) and frees them. A device synchronization conflicts with any
// stream capture in progress on the device, in any thread and any capture mode (with CUDA 13.4 it
// invalidates a PyTorch graph capture), so like every library call that touches the GPU, the drain
// must not overlap a capture; if the synchronization reports a capture error, the pointers stay
// queued for the next drain. Context::shutdown drains too. The device outputs of the Python
// bindings are freed through the same queue.
void defer_device_free(void* ptr, int device) noexcept;
// Returns the number of pointers freed by this call. Never throws.
int64_t drain_deferred_frees() noexcept;

// A child process made by fork() inherits the parent's CUDA handles and the library's mutexes
// (possibly locked by a thread that does not exist in the child), and CUDA cannot be used after a
// fork. mark_forked_child() (called in the child right after the fork, by the Python bindings)
// turns every release path that runs without the caller's control into a no-op that takes no lock
// and makes no CUDA call: defer_device_free, drain_deferred_frees, PinnedBlock::reset and the
// destruction of a thread's contexts at its exit. The memory belongs to the parent.
void mark_forked_child() noexcept;
bool forked_child() noexcept;

// Pinned (page-locked) host memory: the host-side control blocks of the driver (driver.cpp).
//
// The blocks come from a process-lifetime pool and go back to it, never to cudaFreeHost, when
// their owner is destroyed: cudaFreeHost synchronizes the whole device, and owners die at times
// the library does not choose (a context is destroyed when its thread exits), possibly while
// another thread captures a CUDA graph, which a device synchronization invalidates. So:
//   - pinned_acquire() reuses the smallest idle block that is large enough, else allocates one
//     (cudaHostAlloc, portable). When it allocates, it first frees the idle blocks smaller than the
//     request (they are what a context outgrew). It runs only inside library calls, which already
//     must not overlap a capture (drain_deferred_frees synchronizes the device at their end);
//   - a PinnedBlock's destructor returns the block to the idle list and makes no CUDA call, so it
//     is safe at any time, including at thread exit and inside a capture;
//   - release_device_pools() (Context::shutdown, at process or test-session teardown) frees the
//     idle blocks.
// Blocks are rounded up to 2 MiB.
class PinnedBlock {
 public:
  PinnedBlock() = default;
  ~PinnedBlock() { reset(); }
  PinnedBlock(const PinnedBlock&) = delete;
  PinnedBlock& operator=(const PinnedBlock&) = delete;
  PinnedBlock(PinnedBlock&& other) noexcept { swap(other); }
  PinnedBlock& operator=(PinnedBlock&& other) noexcept {
    if (this != &other) {
      reset();
      swap(other);
    }
    return *this;
  }
  void* data() const noexcept { return data_; }
  // Returns the block to the pool (no CUDA call).
  void reset() noexcept;

 private:
  friend PinnedBlock pinned_acquire(size_t bytes);
  void swap(PinnedBlock& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(bytes_, other.bytes_);
  }
  void* data_ = nullptr;
  size_t bytes_ = 0;
};
PinnedBlock pinned_acquire(size_t bytes);

// Owning, move-only device buffer of `count` elements, allocated and freed stream-ordered on one
// stream. It must not outlive that stream (in practice: its Context).
template <class T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(size_t count, cudaStream_t stream, int device) : count_(count), stream_(stream), device_(device) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) throw std::length_error("eda: DeviceBuffer size overflow");
    if (count) data_ = static_cast<T*>(device_alloc_async(count * sizeof(T), stream, device));
  }
  ~DeviceBuffer() { reset(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept { swap(other); }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      swap(other);
    }
    return *this;
  }

  T* data() const noexcept { return data_; }
  bool empty() const noexcept { return count_ == 0; }

  // Frees the memory (stream-ordered) and leaves the buffer empty.
  void reset() noexcept {
    if (data_) device_free_async(data_, stream_, device_);
    data_ = nullptr;
    count_ = 0;
  }
  // Gives up ownership; the caller frees the pointer (stream-ordered, on the buffer's stream).
  T* release() noexcept {
    T* out = data_;
    data_ = nullptr;
    count_ = 0;
    return out;
  }

 private:
  void swap(DeviceBuffer& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(count_, other.count_);
    std::swap(stream_, other.stream_);
    std::swap(device_, other.device_);
  }
  T* data_ = nullptr;
  size_t count_ = 0;
  cudaStream_t stream_ = nullptr;
  int device_ = -1;
};

inline constexpr size_t kArenaAlignment = 256;

template <class T>
struct ArenaSlot {
  size_t index = std::numeric_limits<size_t>::max();
};

// The workspace a call needs, declared slot by slot before binding it to arena memory.
class ArenaLayout {
 public:
  template <class T>
  ArenaSlot<T> add(size_t count) {
    static_assert(alignof(T) <= kArenaAlignment);
    if (count > std::numeric_limits<size_t>::max() / sizeof(T) - kArenaAlignment) {
      throw std::length_error("eda: arena slot size overflow");
    }
    bytes_.push_back(count * sizeof(T));
    return ArenaSlot<T>{bytes_.size() - 1};
  }
  size_t slots() const noexcept { return bytes_.size(); }
  size_t slot_bytes(size_t index) const { return bytes_.at(index); }
  // Packed size of all slots (every slot starts on a 256-byte boundary).
  size_t bytes() const;

 private:
  std::vector<size_t> bytes_;
};

// Slot pointers of a bound layout; valid until the arena's next bind() or release().
class ArenaBinding {
 public:
  template <class T>
  T* get(ArenaSlot<T> slot) const {
    return static_cast<T*>(pointers_.at(slot.index));
  }

 private:
  friend class Arena;
  std::vector<void*> pointers_;
};

// An object owned by an arena (Arena::attachment): per-context state kept across calls, such as
// cuFFT plans and the small device blocks their kernels read. Attachments are
// destroyed, newest first, by Arena::release() and the arena's destructor, before the arena frees
// its memory; their destructors run while the context stream still exists.
class ArenaAttachment {
 public:
  virtual ~ArenaAttachment() = default;
};

// Per-context workspace. Normal mode: one device allocation; a layout that does not fit grows it
// to max(needed, 2 x capacity). Debug mode (EDA_SANITIZE): every slot is
// its own allocation, allocated afresh on every bind and (unless EDA_ARENA_POISON=0) filled with
// 0xFF bytes, so that memcheck sees overruns between slots and stale reads produce NaN or, without
// the fill, show up in initcheck. Zero-sized slots bind to nullptr. A bind invalidates the
// pointers of every earlier binding.
class Arena {
 public:
  Arena(int device, cudaStream_t stream, bool debug, bool poison);
  ~Arena();
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  ArenaBinding bind(const ArenaLayout& layout);
  // Frees the workspace (stream-ordered) and keeps the attachments; the next bind allocates it
  // again.
  void free_all() noexcept;
  // Destroys the attachments, then frees the workspace.
  void release() noexcept;

  // The attachment of type T (derived from ArenaAttachment), made by `make()` (which returns a
  // std::unique_ptr<T>) on first use. One per type; it lives until release().
  template <class T, class Make>
  T& attachment(Make&& make) {
    const std::type_index key(typeid(T));
    for (auto& [type, object] : attachments_) {
      if (type == key) return static_cast<T&>(*object);
    }
    std::unique_ptr<T> created = make();
    T& out = *created;
    attachments_.emplace_back(key, std::move(created));
    return out;
  }


 private:
  int device_;
  cudaStream_t stream_;
  bool debug_;
  bool poison_;
  void* base_ = nullptr;             // normal mode
  size_t capacity_ = 0;              // normal mode: bytes of base_; debug mode: bytes currently bound
  std::vector<void*> debug_slots_;   // debug mode
  std::vector<std::pair<std::type_index, std::unique_ptr<ArenaAttachment>>> attachments_;
};

}  // namespace eda
