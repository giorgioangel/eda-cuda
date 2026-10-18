# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti

# CUDA 13 ships CUB, Thrust and libcu++ only under <cuda>/include/cccl. nvcc adds that directory for
# .cu files; host (g++) translation units need it explicitly.
find_path(EDA_CCCL_INCLUDE_DIR NAMES cuda/std/version
  PATHS ${CUDAToolkit_INCLUDE_DIRS} PATH_SUFFIXES cccl NO_DEFAULT_PATH)
if(NOT EDA_CCCL_INCLUDE_DIR)
  message(FATAL_ERROR "eda: CCCL headers (cuda/std/version) not found under ${CUDAToolkit_INCLUDE_DIRS}")
endif()

# eda_native: the static CUDA library (no relocatable device code). Device code lives in header-only
# .cuh files; kernel instantiations are split across .cu translation units.
set(EDA_NATIVE_SOURCES
  src/decompose.cpp
  src/field/convert.cu
  src/field/dct.cu
  src/field/field_ops.cu
  src/field/geom.cpp
  src/field/vhgw.cu
  src/method/common.cpp
  src/method/driver.cpp
  src/method/element.cu
  src/method/element_op_f32.cu
  src/method/element_op_f64.cu
  src/method/mode.cu
  src/method/monogenic.cpp
  src/method/orientation.cu
  src/method/sift.cu
  src/method/stats.cu
  src/method/stats_fields.cu
  src/method/stats_rungs.cu
  src/method/stats_scale.cu
  src/runtime/build_info.cpp
  src/runtime/context.cpp
  src/runtime/device.cu
  src/runtime/env.cpp
  src/runtime/error.cpp
  src/runtime/memory.cpp)
set(EDA_NATIVE_HEADERS
  src/decompose.hpp
  src/field/common.cuh
  src/field/convert.hpp
  src/field/dct.cuh
  src/field/dct.hpp
  src/field/field_ops.hpp
  src/field/geom.hpp
  src/field/reduce.hpp
  src/field/state.hpp
  src/field/vhgw.hpp
  src/method/element.cuh
  src/method/element.hpp
  src/method/element_op.cuh
  src/method/method.hpp
  src/method/mode.hpp
  src/method/stats.hpp
  src/method/stats_common.cuh
  src/runtime/build_info.hpp
  src/runtime/context.hpp
  src/runtime/device.hpp
  src/runtime/env.hpp
  src/runtime/env_registry.def
  src/runtime/error.hpp
  src/runtime/memory.hpp)

add_library(eda_native STATIC ${EDA_NATIVE_SOURCES} ${EDA_NATIVE_HEADERS})
target_include_directories(eda_native
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src
  PRIVATE ${EDA_GENERATED_DIR})
target_include_directories(eda_native SYSTEM PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${EDA_CCCL_INCLUDE_DIR}>")
# cuFFT is linked dynamically (the nvidia-cufft wheel supplies it at run time); cudart statically.
target_link_libraries(eda_native PUBLIC CUDA::cudart_static CUDA::cufft)
set_target_properties(eda_native PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RUNTIME_LIBRARY Static)
# Warnings for the library's own translation units.
target_compile_options(eda_native PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wno-unused-parameter>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wno-unused-parameter>")
# The build facts that build_info() reports besides the architectures in eda_build_config.h.
set_property(SOURCE src/runtime/build_info.cpp APPEND PROPERTY COMPILE_DEFINITIONS
  "EDA_VERSION_STRING=\"${PROJECT_VERSION}\""
  "EDA_CUDA_TOOLKIT_VERSION_STRING=\"${CUDAToolkit_VERSION}\""
  "EDA_BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\""
  "EDA_CXX_COMPILER=\"${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\""
  "EDA_CUDA_COMPILER=\"${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}\"")

# eda_add_test(NAME SOURCES <files...> LABEL <cpu|gpu|sanitize|canary> [LIBS <targets...>])
# Builds one test executable and registers it with ctest under the given label. Tests labelled
# cpu run with no GPU visible.
function(eda_add_test name)
  cmake_parse_arguments(ARG "" "LABEL" "SOURCES;LIBS" ${ARGN})
  if(NOT ARG_LABEL MATCHES "^(cpu|gpu|sanitize|canary)$")
    message(FATAL_ERROR "eda_add_test(${name}): LABEL must be cpu, gpu, sanitize or canary")
  endif()
  add_executable(${name} ${ARG_SOURCES})
  target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/cpp/src ${EDA_GENERATED_DIR})
  target_link_libraries(${name} PRIVATE ${ARG_LIBS})
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES LABELS ${ARG_LABEL})
  if(ARG_LABEL STREQUAL "cpu")
    set_tests_properties(${name} PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=")
  endif()
endfunction()

# The extension module and the C++ tests have their own lists (cpp/python, cpp/tests).
if(EDA_BUILD_PYTHON AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/python/CMakeLists.txt)
  add_subdirectory(python)
endif()
if(EDA_BUILD_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
  add_subdirectory(tests)
endif()
