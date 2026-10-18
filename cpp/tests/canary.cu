// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Giorgio Angelotti
// Canaries for tools/sanitize.sh: one planted fault per compute-sanitizer tool, to prove that the
// tool reports it before its verdict on the real tests is trusted. Without a sanitizer every mode
// exits 0 except "synccheck", whose invalid warp mask can fault the kernel.
// Built as the executable eda_canary (mode as argument) and as the module libeda_canary.so
// (eda_canary(mode), loaded with ctypes; the build hides symbols by default); neither is a ctest.
#include <cuda_runtime.h>

#include <cstring>

namespace {

// memcheck: a write one int past an allocation of 64 ints
__global__ void write_past_end(int* p, int n) {
  if (threadIdx.x == 0) p[n] = 1;
}

// memcheck (pool): a write one byte past the first of two 2048-byte allocations of a memory pool,
// which places them back to back; memcheck reports it only with --padding
__global__ void write_past_byte(char* p, int n) {
  if (threadIdx.x == 0) p[n] = 1;
}

int pool_overrun() {
  cudaMemPool_t pool = nullptr;
  cudaMemPoolProps props{};
  props.allocType = cudaMemAllocationTypePinned;
  props.location.type = cudaMemLocationTypeDevice;
  if (cudaGetDevice(&props.location.id) != cudaSuccess || cudaMemPoolCreate(&pool, &props) != cudaSuccess) return 2;
  void *first = nullptr, *second = nullptr;
  if (cudaMallocFromPoolAsync(&first, 2048, pool, nullptr) != cudaSuccess ||
      cudaMallocFromPoolAsync(&second, 2048, pool, nullptr) != cudaSuccess)
    return 2;
  write_past_byte<<<1, 32>>>(static_cast<char*>(first), 2048);
  const cudaError_t status = cudaDeviceSynchronize();
  cudaFreeAsync(first, nullptr);
  cudaFreeAsync(second, nullptr);
  cudaDeviceSynchronize();
  cudaMemPoolDestroy(pool);
  return status == cudaSuccess ? 0 : 1;
}

// initcheck: a read of device memory that was never written
__global__ void read_unwritten(const int* in, int* out) { out[threadIdx.x] = in[threadIdx.x] + 1; }

// racecheck: a warp reads the shared slots of the other warp with no barrier between
__global__ void shared_race(int* out) {
  volatile __shared__ int slot[64];
  slot[threadIdx.x] = static_cast<int>(threadIdx.x);
  out[threadIdx.x] = slot[(threadIdx.x + 32) % 64];
}

// synccheck: a warp barrier whose mask leaves out the calling lanes
__global__ void bad_mask(int* out) {
  if (threadIdx.x < 16) __syncwarp(0xffff0000u);
  out[threadIdx.x] = 1;
}

}  // namespace

extern "C" __attribute__((visibility("default"))) int eda_canary(const char* mode) {
  int *a = nullptr, *b = nullptr;
  if (cudaMalloc(&a, 64 * sizeof(int)) != cudaSuccess || cudaMalloc(&b, 64 * sizeof(int)) != cudaSuccess) return 2;
  cudaMemset(b, 0, 64 * sizeof(int));
  if (!std::strcmp(mode, "pool")) {
    cudaFree(a);
    cudaFree(b);
    return pool_overrun();
  }
  if (!std::strcmp(mode, "memcheck")) {
    write_past_end<<<1, 32>>>(a, 64);
  } else if (!std::strcmp(mode, "leak")) {
    int* never_freed = nullptr;
    cudaMalloc(&never_freed, 256);
  } else if (!std::strcmp(mode, "initcheck")) {
    read_unwritten<<<1, 32>>>(a, b);
  } else if (!std::strcmp(mode, "racecheck")) {
    shared_race<<<1, 64>>>(b);
  } else if (!std::strcmp(mode, "synccheck")) {
    bad_mask<<<1, 32>>>(b);
  } else {
    return 2;
  }
  const cudaError_t status = cudaDeviceSynchronize();
  cudaFree(a);
  cudaFree(b);
  return status == cudaSuccess ? 0 : 1;
}

#ifndef EDA_CANARY_MODULE
int main(int argc, char** argv) { return argc == 2 ? eda_canary(argv[1]) : 2; }
#endif
