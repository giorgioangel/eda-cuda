// SPDX-License-Identifier: Apache-2.0
// Device selection, properties and the startup support check.
#pragma once

#include <cstddef>
#include <optional>
#include <string>

namespace eda {

// Makes `device` current for its lifetime and restores the previously current device.
class DeviceGuard {
 public:
  explicit DeviceGuard(int device);
  ~DeviceGuard();
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

 private:
  int previous_ = -1;
  bool changed_ = false;
};

struct DeviceInfo {
  int device = 0;
  std::string name;
  std::string pci_bus_id;
  int cc_major = 0;
  int cc_minor = 0;
  int sm_count = 0;
  int max_threads_per_sm = 0;
  int max_blocks_per_sm = 0;
  int registers_per_sm = 0;
  size_t smem_per_block = 0;        // default limit per block
  size_t smem_per_block_optin = 0;  // with cudaFuncAttributeMaxDynamicSharedMemorySize
  size_t smem_per_sm = 0;
  size_t l2_bytes = 0;
  size_t persisting_l2_max_bytes = 0;
  size_t total_memory = 0;
  bool kernel_timeout = false;  // display watchdog active: every launch must stay short
  bool memory_pools = false;
  bool integrated = false;
  int driver_version = 0;   // e.g. 13040 (0: no driver)
  int runtime_version = 0;  // the statically linked cudart
};

// Number of visible CUDA devices; 0 when there is no driver or no visible device. Throws CudaError
// with an actionable message for a broken installation (for example a driver older than cudart).
int device_count();

// Properties of `device` (std::invalid_argument for an ordinal that is not visible).
DeviceInfo device_info(int device);

// Startup check, run once per device before its first use: compute capability >= 8.0, a driver
// that supports this build's runtime, stream-ordered memory pools, and GPU code in this build for
// the device. Throws CudaError with an actionable message.
// visible_devices (CUDA_VISIBLE_DEVICES, for the message) comes from a RuntimeConfig snapshot:
// the check reads no environment.
void require_supported_device(int device, const std::optional<std::string>& visible_devices);

// "13.4" for 13040.
std::string cuda_version_string(int version);

// The oldest driver this build supports (EDA_REQUIRED_DRIVER): CUDA 13.0 for a CUDA 13 build, 12.3
// for a CUDA 12 build. The statically linked runtime runs on any driver of its major version.
int required_driver_version();

// The calling thread's current device (cudaGetDevice). Without a usable device it throws the
// startup check's CudaError ("found no CUDA device ...", driver messages) rather than the bare
// runtime error.
int current_device(const std::optional<std::string>& visible_devices);

}  // namespace eda
