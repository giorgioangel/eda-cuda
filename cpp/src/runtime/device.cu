// SPDX-License-Identifier: Apache-2.0
#include "runtime/device.hpp"
#include "runtime/error.hpp"

#include "eda_build_config.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>

namespace eda {
namespace {

// Compiled with the library's architecture list: if the driver cannot load this kernel for a
// device, it cannot load any other kernel of the build either.
__global__ void eda_device_probe() {}

std::mutex g_checked_mutex;
std::set<int> g_checked;  // devices that passed require_supported_device()

std::string device_label(int device, const std::string& name) {
  return "CUDA device " + std::to_string(device) + (name.empty() ? std::string() : " (" + name + ")");
}

bool is_image_error(cudaError_t code) {
  switch (code) {
    case cudaErrorInvalidDeviceFunction:
    case cudaErrorInvalidKernelImage:
    case cudaErrorNoKernelImageForDevice:
    case cudaErrorInvalidPtx:
    case cudaErrorUnsupportedPtxVersion:
    case cudaErrorJitCompilerNotFound:
    case cudaErrorJitCompilationDisabled:
      return true;
    default:
      return false;
  }
}

}  // namespace

DeviceGuard::DeviceGuard(int device) {
  EDA_CUDA(cudaGetDevice(&previous_));
  if (previous_ != device) {
    EDA_CUDA(cudaSetDevice(device));
    changed_ = true;
  }
}

DeviceGuard::~DeviceGuard() {
  if (changed_) (void)cudaSetDevice(previous_);
}

std::string cuda_version_string(int version) {
  return std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10);
}

int required_driver_version() { return EDA_REQUIRED_DRIVER; }

int current_device(const std::optional<std::string>& visible_devices) {
  int device = 0;
  const cudaError_t status = cudaGetDevice(&device);
  if (status == cudaSuccess) return device;
  (void)cudaGetLastError();
  require_supported_device(0, visible_devices);  // throws the actionable message when it applies
  throw_cuda_error(status, "cudaGetDevice", __FILE__, __LINE__);
}

int device_count() {
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status == cudaSuccess) return count;
  (void)cudaGetLastError();
  int driver = 0;
  (void)cudaDriverGetVersion(&driver);
  if (status == cudaErrorNoDevice || driver == 0) return 0;
  if (status == cudaErrorInsufficientDriver) {
    throw CudaError(status, "eda: needs an NVIDIA driver that supports CUDA " +
                                cuda_version_string(required_driver_version()) +
                                " or newer; the installed driver supports CUDA " + cuda_version_string(driver) +
                                ". Update the NVIDIA driver" EDA_DRIVER_HINT ". [" + cuda_error_text(status) + "]");
  }
  throw CudaError(status, "eda: could not list the CUDA devices: cudaGetDeviceCount failed with " +
                              cuda_error_text(status) +
                              ". Check that the NVIDIA kernel driver and user-mode driver (libcuda.so) match.");
}

DeviceInfo device_info(int device) {
  const int count = device_count();
  if (device < 0 || device >= count) {
    throw std::invalid_argument("eda: CUDA device " + std::to_string(device) + " is not visible (" + std::to_string(count) +
                                " visible device" + (count == 1 ? "" : "s") + ")");
  }
  DeviceInfo info;
  info.device = device;
  cudaDeviceProp prop{};
  EDA_CUDA(cudaGetDeviceProperties(&prop, device));
  info.name = prop.name;
  char bus_id[64] = {};
  if (cudaDeviceGetPCIBusId(bus_id, sizeof bus_id, device) == cudaSuccess) {
    info.pci_bus_id = bus_id;
  } else {
    (void)cudaGetLastError();
  }
  auto attribute = [device](cudaDeviceAttr attr) {
    int value = 0;
    EDA_CUDA(cudaDeviceGetAttribute(&value, attr, device));
    return value;
  };
  info.cc_major = attribute(cudaDevAttrComputeCapabilityMajor);
  info.cc_minor = attribute(cudaDevAttrComputeCapabilityMinor);
  info.sm_count = attribute(cudaDevAttrMultiProcessorCount);
  info.max_threads_per_sm = attribute(cudaDevAttrMaxThreadsPerMultiProcessor);
  info.max_blocks_per_sm = attribute(cudaDevAttrMaxBlocksPerMultiprocessor);
  info.registers_per_sm = attribute(cudaDevAttrMaxRegistersPerMultiprocessor);
  info.smem_per_block = static_cast<size_t>(attribute(cudaDevAttrMaxSharedMemoryPerBlock));
  info.smem_per_block_optin = static_cast<size_t>(attribute(cudaDevAttrMaxSharedMemoryPerBlockOptin));
  info.smem_per_sm = static_cast<size_t>(attribute(cudaDevAttrMaxSharedMemoryPerMultiprocessor));
  info.l2_bytes = static_cast<size_t>(attribute(cudaDevAttrL2CacheSize));
  info.persisting_l2_max_bytes = static_cast<size_t>(attribute(cudaDevAttrMaxPersistingL2CacheSize));
  info.total_memory = prop.totalGlobalMem;
  info.kernel_timeout = attribute(cudaDevAttrKernelExecTimeout) != 0;
  info.memory_pools = attribute(cudaDevAttrMemoryPoolsSupported) != 0;
  info.integrated = attribute(cudaDevAttrIntegrated) != 0;
  EDA_CUDA(cudaDriverGetVersion(&info.driver_version));
  EDA_CUDA(cudaRuntimeGetVersion(&info.runtime_version));
  return info;
}

void require_supported_device(int device, const std::optional<std::string>& visible_devices) {
  {
    std::lock_guard<std::mutex> lock(g_checked_mutex);
    if (g_checked.count(device)) return;
  }
  const int count = device_count();
  if (count == 0) {
    int driver = 0;
    (void)cudaDriverGetVersion(&driver);
    std::string where;
    if (driver == 0) {
      where = ": no NVIDIA driver is loaded";
    } else if (visible_devices) {
      where = " (CUDA_VISIBLE_DEVICES=\"" + *visible_devices + "\")";
    }
    throw CudaError(cudaErrorNoDevice, "eda: found no CUDA device" + where +
                                           ". eda needs an NVIDIA GPU of compute capability 8.0 or newer.");
  }
  if (device < 0 || device >= count) {
    throw std::invalid_argument("eda: CUDA device " + std::to_string(device) + " is not visible (" + std::to_string(count) +
                                " visible device" + (count == 1 ? "" : "s") + ")");
  }
  const DeviceInfo info = device_info(device);
  const std::string label = device_label(device, info.name);
  const int needed = required_driver_version();
  if (info.driver_version < needed) {
    throw CudaError(cudaErrorInsufficientDriver,
                    "eda: needs an NVIDIA driver that supports CUDA " + cuda_version_string(needed) +
                        " or newer (this build's CUDA runtime is " + cuda_version_string(info.runtime_version) +
                        "); the installed driver supports CUDA " + cuda_version_string(info.driver_version) +
                        ". Update the NVIDIA driver" EDA_DRIVER_HINT ".");
  }
  if (info.cc_major < 8) {
    throw CudaError(cudaErrorNotSupported, "eda: needs an NVIDIA GPU of compute capability 8.0 or newer "
                                           "(Ampere or later); " +
                                               label + " has compute capability " + std::to_string(info.cc_major) +
                                               "." + std::to_string(info.cc_minor) + ".");
  }
  if (!info.memory_pools) {
    throw CudaError(cudaErrorNotSupported, "eda: " + label + " does not support stream-ordered memory pools "
                                                             "(cudaDevAttrMemoryPoolsSupported is 0), which eda needs.");
  }
  {
    DeviceGuard guard(device);
    cudaFuncAttributes attributes{};
    const cudaError_t image = cudaFuncGetAttributes(&attributes, eda_device_probe);
    if (image != cudaSuccess) {
      (void)cudaGetLastError();
      const std::string arch = std::to_string(info.cc_major) + std::to_string(info.cc_minor);
      if (is_image_error(image)) {
        throw CudaError(image, "eda: this build has no usable GPU code for " + label + ", compute capability " +
                                   std::to_string(info.cc_major) + "." + std::to_string(info.cc_minor) +
                                   ". Rebuild with -DCMAKE_CUDA_ARCHITECTURES=" + arch +
                                   " (or native on that machine). [" + cuda_error_text(image) + "]");
      }
      throw CudaError(image, "eda: could not check its GPU code on " + label + ": cudaFuncGetAttributes failed: " +
                                 cuda_error_text(image));
    }
  }
  std::lock_guard<std::mutex> lock(g_checked_mutex);
  g_checked.insert(device);
}

}  // namespace eda
