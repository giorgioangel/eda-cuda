# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti

# The extension module eda._core. Python and nanobind are found here.
# Development.SABIModule enables nanobind's stable-ABI (abi3) build on CPython >= 3.12.
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
if(NOT nanobind_DIR AND NOT nanobind_ROOT)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
find_package(nanobind CONFIG REQUIRED)

# The extension is built only from .cpp translation units: nanobind is never included from .cu.
file(GLOB _eda_python_cu CONFIGURE_DEPENDS LIST_DIRECTORIES false "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
if(_eda_python_cu)
  message(FATAL_ERROR "cpp/python must not contain .cu files (nanobind is host-only): ${_eda_python_cu}")
endif()

nanobind_add_module(_core STABLE_ABI NB_STATIC module.cpp decompose.cpp)
target_link_libraries(_core PRIVATE eda_native)
# The C++ runtime is linked into the extension, so that the wheel also loads beside an older
# libstdc++ (a conda Python's, for example). cudart is static as well; cuFFT stays a shared library.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_options(_core PRIVATE -static-libstdc++ -static-libgcc)
endif()

# A runnable package tree in the build directory: <build>/python/eda holds the extension and
# links to the package's files, so that PYTHONPATH=<build>/python imports eda from the build.
set(EDA_PYTHON_BUILD_DIR ${CMAKE_BINARY_DIR}/python/eda)
set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${EDA_PYTHON_BUILD_DIR})
file(MAKE_DIRECTORY ${EDA_PYTHON_BUILD_DIR})
file(GLOB _eda_package_files CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/eda/*.py"
  "${PROJECT_SOURCE_DIR}/src/eda/*.pyi" "${PROJECT_SOURCE_DIR}/src/eda/py.typed")
foreach(_eda_file IN LISTS _eda_package_files)
  get_filename_component(_eda_name ${_eda_file} NAME)
  file(CREATE_LINK ${_eda_file} ${EDA_PYTHON_BUILD_DIR}/${_eda_name} SYMBOLIC)
endforeach()
configure_file(${PROJECT_SOURCE_DIR}/cpp/cmake/build_config.py.in ${EDA_PYTHON_BUILD_DIR}/_build_config.py @ONLY)

if(SKBUILD_PLATLIB_DIR)
  install(TARGETS _core LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/eda COMPONENT Python)
  install(FILES ${EDA_PYTHON_BUILD_DIR}/_build_config.py DESTINATION ${SKBUILD_PLATLIB_DIR}/eda
    COMPONENT Python)
endif()
