# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
cmake_minimum_required(VERSION 3.28)
# Target the build machine's GPU unless CMAKE_CUDA_ARCHITECTURES or the CUDAARCHS environment
# variable names the architectures.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA target architectures")
endif()
project(eda VERSION 0.1.0 LANGUAGES CXX CUDA)

# Release (-O3 -DNDEBUG) unless the build type is given.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# CUDA 12.8 is the first toolkit with compute capabilities 10.0 and 12.0.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
  message(FATAL_ERROR "eda requires CUDA Toolkit 12.8 or newer")
endif()
# The package of each CUDA major version: its name, the cuFFT runtime wheel it depends on and the
# oldest driver it supports. pyproject.toml sets EDA_CUDA_MAJOR (13 for eda-cuda, 12 for
# eda-cuda-cu12), so that a package is never built with the other toolkit.
string(REGEX MATCH "^[0-9]+" _eda_cuda_major "${CMAKE_CUDA_COMPILER_VERSION}")
if(DEFINED EDA_CUDA_MAJOR AND NOT EDA_CUDA_MAJOR STREQUAL _eda_cuda_major)
  message(FATAL_ERROR "eda: this package is built with a CUDA ${EDA_CUDA_MAJOR} toolkit, but the CUDA "
    "compiler is ${CMAKE_CUDA_COMPILER_VERSION}. tools/build_wheels.sh builds both packages.")
endif()
if(_eda_cuda_major EQUAL 12)
  set(EDA_DISTRIBUTION eda-cuda-cu12)
  set(EDA_CUFFT_PACKAGE nvidia-cufft-cu12)
  set(EDA_REQUIRED_DRIVER 12030)
  set(EDA_DRIVER_HINT "")
else()
  set(EDA_DISTRIBUTION eda-cuda)
  set(EDA_CUFFT_PACKAGE nvidia-cufft)
  math(EXPR EDA_REQUIRED_DRIVER "${_eda_cuda_major} * 1000")
  set(EDA_DRIVER_HINT ", or install eda-cuda-cu12 in place of eda-cuda (CUDA 12, for drivers of CUDA 12.3 or newer)")
endif()

# Compute capability 8.0 is the minimum. The resolved list is also compiled into the library, so
# the startup check can name it.
set(EDA_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
foreach(_eda_keyword native all all-major)
  if(EDA_CUDA_ARCHITECTURES STREQUAL _eda_keyword)
    string(TOUPPER "${_eda_keyword}" _eda_suffix)
    string(REPLACE "-" "_" _eda_suffix "${_eda_suffix}")
    set(EDA_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES_${_eda_suffix}}")
  endif()
endforeach()
set(EDA_CUDA_SASS)
set(EDA_CUDA_PTX)
set(_eda_too_old)
foreach(_eda_arch IN LISTS EDA_CUDA_ARCHITECTURES)
  if(NOT _eda_arch MATCHES "^([0-9]+)([af]?)(-real|-virtual)?$")
    message(FATAL_ERROR "eda cannot parse the CUDA architecture \"${_eda_arch}\" in "
      "CMAKE_CUDA_ARCHITECTURES=\"${CMAKE_CUDA_ARCHITECTURES}\". Use entries such as 90, 90-real or 120-virtual.")
  endif()
  if(CMAKE_MATCH_1 LESS 80)
    list(APPEND _eda_too_old "${_eda_arch}")
  endif()
  if(NOT CMAKE_MATCH_3 STREQUAL "-virtual")
    list(APPEND EDA_CUDA_SASS "sm_${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
  endif()
  if(NOT CMAKE_MATCH_3 STREQUAL "-real")
    list(APPEND EDA_CUDA_PTX "compute_${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
  endif()
endforeach()
if(NOT EDA_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "eda found no CUDA architecture to build for "
    "(CMAKE_CUDA_ARCHITECTURES=\"${CMAKE_CUDA_ARCHITECTURES}\"). "
    "Set architectures 80 or newer explicitly, for example -DCMAKE_CUDA_ARCHITECTURES=90.")
endif()
if(_eda_too_old)
  string(JOIN ", " _eda_too_old ${_eda_too_old})
  message(FATAL_ERROR "eda requires NVIDIA GPUs of compute capability 8.0 or newer. "
    "CMAKE_CUDA_ARCHITECTURES=\"${CMAKE_CUDA_ARCHITECTURES}\" includes ${_eda_too_old}. "
    "Use architectures 80 or newer, for example "
    "-DCMAKE_CUDA_ARCHITECTURES=\"80-real;86-real;89-real;90-real;100-real;120-real;120-virtual\", "
    "or native on a machine with a supported GPU.")
endif()
string(JOIN " " EDA_CUDA_SASS ${EDA_CUDA_SASS})
string(JOIN " " EDA_CUDA_PTX ${EDA_CUDA_PTX})
message(STATUS "eda CUDA code: SASS [${EDA_CUDA_SASS}], PTX [${EDA_CUDA_PTX}]")
set(EDA_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
configure_file(cpp/cmake/build_config.h.in ${EDA_GENERATED_DIR}/eda_build_config.h @ONLY)

# The results must not depend on the compiler's freedom to reassociate or flush: refuse fast-math
# and flush-to-zero however they reach the build.
foreach(_eda_flags "${CMAKE_CXX_FLAGS}" "${CMAKE_CUDA_FLAGS}" "$ENV{CXXFLAGS}" "$ENV{CUDAFLAGS}")
  if(_eda_flags MATCHES "(-|--)(use_fast_math|ffast-math|Ofast)|(-|--)ftz=true|(-|--)prec-div=false|(-|--)prec-sqrt=false")
    message(FATAL_ERROR "eda refuses fast-math and flush-to-zero flags: \"${_eda_flags}\"")
  endif()
endforeach()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffp-contract=off")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -ftz=false -prec-div=true -prec-sqrt=true -Xcompiler=-ffp-contract=off")

option(EDA_BUILD_PYTHON "Build the eda Python extension" ON)
option(EDA_BUILD_TESTS "Build the C++ tests" OFF)

find_package(CUDAToolkit 12.8 REQUIRED)
# The soname of cuFFT (libcufft.so.11 with CUDA 12, .12 with CUDA 13), which the Python package
# loads from the runtime wheel before the extension.
file(REAL_PATH "${CUDA_cufft_LIBRARY}" _eda_cufft)
if(NOT _eda_cufft MATCHES "/(libcufft\\.so\\.[0-9]+)[^/]*$")
  message(FATAL_ERROR "eda: cannot read the cuFFT soname from ${_eda_cufft}")
endif()
set(EDA_CUFFT_SONAME "${CMAKE_MATCH_1}")
# cuFFT is linked dynamically: the nvidia-cufft wheel supplies it at run time.

if(EDA_BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory(cpp)

if(EDA_BUILD_PYTHON AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/eda)
  # The wheel has no Python packages of its own (wheel.packages is empty): CMake installs them.
  install(DIRECTORY src/eda/ DESTINATION ${SKBUILD_PLATLIB_DIR}/eda COMPONENT Python
    FILES_MATCHING PATTERN "*.py" PATTERN "*.pyi" PATTERN "py.typed")
endif()
