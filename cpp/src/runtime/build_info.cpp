// SPDX-License-Identifier: Apache-2.0
#include "runtime/build_info.hpp"

#include "eda_build_config.h"

#include <cuda/std/version>  // CCCL, from <cuda>/include/cccl in a host (g++) translation unit
#include <cuda_runtime_api.h>

#include <string>

namespace eda {

BuildInfo build_info() {
  BuildInfo info;
  info.version = EDA_VERSION_STRING;
  info.cuda_toolkit = EDA_CUDA_TOOLKIT_VERSION_STRING;
  if (cudaRuntimeGetVersion(&info.cuda_runtime_version) != cudaSuccess) info.cuda_runtime_version = 0;
  if (cudaDriverGetVersion(&info.cuda_driver_version) != cudaSuccess) info.cuda_driver_version = 0;
  info.cuda_architectures = EDA_CUDA_ARCHITECTURES;
  info.sass = EDA_CUDA_SASS;
  info.ptx = EDA_CUDA_PTX;
  info.build_type = EDA_BUILD_TYPE;
  info.cxx_compiler = EDA_CXX_COMPILER;
  info.cuda_compiler = EDA_CUDA_COMPILER;
  info.cccl_version = std::to_string(CCCL_MAJOR_VERSION) + "." + std::to_string(CCCL_MINOR_VERSION) + "." +
                      std::to_string(CCCL_PATCH_VERSION);
  return info;
}

}  // namespace eda
