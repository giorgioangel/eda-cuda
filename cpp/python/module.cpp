// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Giorgio Angelotti
// The eda._core extension (private; the eda package is the public interface): the CudaError
// translation, the build and device information, and the decomposition entry points of
// decompose.cpp.
#include "lifecycle.hpp"
#include "runtime/build_info.hpp"
#include "runtime/device.hpp"
#include "runtime/error.hpp"

#include <cuda_runtime_api.h>
#include <cufft.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <exception>
#include <string>

namespace nb = nanobind;

namespace {

// eda._errors.CudaError, defined in Python so that it exists without the extension. The reference
// is never released: a static nb::object would drop it after the interpreter has finalized.
PyObject* g_cuda_error_type = nullptr;

// eda::CudaError -> eda.CudaError(message, code, name), or MemoryError for an allocation that
// found the device out of memory. Every other exception keeps nanobind's mapping:
// std::invalid_argument (with eda::EnvError) -> ValueError, std::bad_alloc (with
// eda::OutOfDeviceMemory) -> MemoryError, nb::type_error -> TypeError, and any other
// std::exception -> RuntimeError.
void translate_cuda_error(const std::exception_ptr& error, void*) {
  try {
    std::rethrow_exception(error);
  } catch (const eda::CudaError& e) {
    if (e.code() == cudaErrorMemoryAllocation) {
      PyErr_SetString(PyExc_MemoryError, e.what());
      return;
    }
    const char* name = cudaGetErrorName(e.code());
    PyObject* args = Py_BuildValue("(sis)", e.what(), static_cast<int>(e.code()), name ? name : "");
    if (!args) return;  // the Python error from Py_BuildValue stays set
    PyObject* instance = PyObject_Call(g_cuda_error_type, args, nullptr);
    Py_DECREF(args);
    if (instance) {
      PyErr_SetObject(g_cuda_error_type, instance);
      Py_DECREF(instance);
    }
  }
}

nb::list words(const std::string& text) {
  nb::list out;
  size_t start = 0;
  while (start < text.size()) {
    const size_t end = std::min(text.find(' ', start), text.size());
    if (end > start) out.append(nb::str(text.substr(start, end - start).c_str()));
    start = end + 1;
  }
  return out;
}

nb::dict build_info() {
  note_cuda_used();  // the version queries may initialize the CUDA driver
  const eda::BuildInfo info = eda::build_info();
  int cufft = 0;
  nb::dict d;
  d["version"] = info.version;
  d["cuda_toolkit"] = info.cuda_toolkit;
  d["cuda_runtime"] = eda::cuda_version_string(info.cuda_runtime_version);
  if (info.cuda_driver_version > 0) {
    d["cuda_driver"] = eda::cuda_version_string(info.cuda_driver_version);
  } else {
    d["cuda_driver"] = nb::none();
  }
  d["cuda_architectures"] = info.cuda_architectures;
  d["sass"] = words(info.sass);
  d["ptx"] = words(info.ptx);
  d["build_type"] = info.build_type;
  d["cxx_compiler"] = info.cxx_compiler;
  d["cuda_compiler"] = info.cuda_compiler;
  d["cccl_version"] = info.cccl_version;
  if (cufftGetVersion(&cufft) == CUFFT_SUCCESS) {
    d["cufft_version"] = cufft;
  } else {
    d["cufft_version"] = nb::none();
  }
  return d;
}

int device_count() {
  Call call;
  note_cuda_used();
  nb::gil_scoped_release release;
  return eda::device_count();
}

nb::dict device_info(int device) {
  Call call;
  note_cuda_used();
  eda::DeviceInfo info;
  {
    nb::gil_scoped_release release;
    info = eda::device_info(device);
  }
  nb::dict d;
  d["device"] = info.device;
  d["name"] = info.name;
  d["pci_bus_id"] = info.pci_bus_id;
  d["compute_capability"] = nb::make_tuple(info.cc_major, info.cc_minor);
  d["sm"] = "sm_" + std::to_string(info.cc_major) + std::to_string(info.cc_minor);
  d["multiprocessors"] = info.sm_count;
  d["max_threads_per_multiprocessor"] = info.max_threads_per_sm;
  d["max_blocks_per_multiprocessor"] = info.max_blocks_per_sm;
  d["registers_per_multiprocessor"] = info.registers_per_sm;
  d["shared_memory_per_block"] = info.smem_per_block;
  d["shared_memory_per_block_optin"] = info.smem_per_block_optin;
  d["shared_memory_per_multiprocessor"] = info.smem_per_sm;
  d["l2_cache_bytes"] = info.l2_bytes;
  d["persisting_l2_max_bytes"] = info.persisting_l2_max_bytes;
  d["total_memory_bytes"] = info.total_memory;
  d["kernel_timeout"] = info.kernel_timeout;
  d["memory_pools"] = info.memory_pools;
  d["integrated"] = info.integrated;
  d["driver_version"] = eda::cuda_version_string(info.driver_version);
  d["runtime_version"] = eda::cuda_version_string(info.runtime_version);
  return d;
}

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "eda native extension (private; use the eda package)";
  nb::object cuda_error = nb::module_::import_("eda._errors").attr("CudaError");
  g_cuda_error_type = cuda_error.release().ptr();
  nb::register_exception_translator(translate_cuda_error);
  m.def("build_info", &build_info, "Build information of the extension (works without a GPU).");
  m.def("device_count", &device_count, "Number of visible CUDA devices (0 without a driver or device).");
  m.def("device_info", &device_info, nb::arg("device") = 0, "Properties of a visible CUDA device.");
  bind_decompose(m);
}
