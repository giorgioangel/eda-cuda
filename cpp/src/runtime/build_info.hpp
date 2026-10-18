// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>

namespace eda {

struct BuildInfo {
  std::string version;             // library version (CMake project version)
  std::string cuda_toolkit;        // toolkit used for the build, e.g. "13.4.59"
  int cuda_runtime_version = 0;    // statically linked cudart, e.g. 13040
  int cuda_driver_version = 0;     // installed driver, 0 when none
  std::string cuda_architectures;  // explicit CMake list, e.g. "89-real"
  std::string sass;                // e.g. "sm_89"
  std::string ptx;                 // e.g. "" or "compute_120"
  std::string build_type;          // CMAKE_BUILD_TYPE
  std::string cxx_compiler;
  std::string cuda_compiler;
  std::string cccl_version;        // CUB/Thrust/libcu++ headers the build used, e.g. "3.4.2"
};

// Works without a GPU (driver version 0 when no driver is loaded).
BuildInfo build_info();

}  // namespace eda
