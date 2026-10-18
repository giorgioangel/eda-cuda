# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
#
# The C++ tests of the method. eda_add_test takes one label; the sanitizer-capable tests also carry
# the sanitize label, added below. The tests include test_support.hpp and twin.hpp from
# this directory.
function(eda_method_test name source timeout)
  eda_add_test(${name} SOURCES ${source} LABEL gpu LIBS eda_native)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  set_property(TEST ${name} APPEND PROPERTY LABELS sanitize)
  set_tests_properties(${name} PROPERTIES TIMEOUT ${timeout})
endfunction()

eda_method_test(test_cube test_cube.cu 120)
eda_method_test(test_element test_element.cu 300)
eda_method_test(test_stats test_stats.cu 300)
eda_method_test(test_mode test_mode.cu 300)
eda_method_test(test_e2e test_e2e.cu 600)
eda_method_test(test_input test_input.cu 300)
eda_method_test(test_orientation test_orientation.cu 600)
# The twin's self-test of test_orientation: no CUDA call, so it runs without a GPU.
add_test(NAME test_orientation_twin COMMAND test_orientation --twin)
set_tests_properties(test_orientation_twin PROPERTIES LABELS cpu ENVIRONMENT "CUDA_VISIBLE_DEVICES=" TIMEOUT 120)

# The sanitizer canaries of tools/sanitize.sh: the executable eda_canary and the module
# libeda_canary.so. Neither is a ctest (the synccheck canary faults without the tool).
add_executable(eda_canary canary.cu)
set_target_properties(eda_canary PROPERTIES CUDA_RUNTIME_LIBRARY Static)
add_library(eda_canary_module MODULE canary.cu)
set_target_properties(eda_canary_module PROPERTIES OUTPUT_NAME eda_canary CUDA_RUNTIME_LIBRARY Static)
target_compile_definitions(eda_canary_module PRIVATE EDA_CANARY_MODULE)
