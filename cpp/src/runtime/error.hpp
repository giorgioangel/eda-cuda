// SPDX-License-Identifier: Apache-2.0
// CUDA error handling: every failing runtime call becomes a CudaError that names the CUDA error,
// the failing expression and its source location.
#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>
#include <string_view>

namespace eda {

// A failed CUDA runtime call or kernel launch. what() is the full, human-readable message; code()
// is the cudaError_t.
class CudaError : public std::runtime_error {
 public:
  CudaError(cudaError_t code, const std::string& message) : std::runtime_error(message), code_(code) {}
  cudaError_t code() const noexcept { return code_; }

 private:
  cudaError_t code_;
};

// "cudaErrorInvalidValue (invalid argument)".
std::string cuda_error_text(cudaError_t code);

// Source path from "cpp/" on (absolute and build-relative spellings both shorten to it).
std::string_view source_path(const char* file) noexcept;

// Throws CudaError("eda: <what> failed: <error> at <file>:<line>") and clears the runtime's
// last-error state, so that a later launch check does not report this error again.
[[noreturn]] void throw_cuda_error(cudaError_t code, std::string_view what, const char* file, int line);

inline void cuda_check(cudaError_t code, const char* expression, const char* file, int line) {
  if (code != cudaSuccess) [[unlikely]] {
    throw_cuda_error(code, expression, file, line);
  }
}

// After kernel<<<...>>>: throws CudaError naming `kernel` if the launch was rejected (invalid
// configuration, missing kernel image, too many resources, ...).
void check_launch(const char* kernel, const char* file, int line);

}  // namespace eda

// Checks a CUDA runtime call: EDA_CUDA(cudaMemsetAsync(p, 0, n, stream));
#define EDA_CUDA(expr) ::eda::cuda_check((expr), #expr, __FILE__, __LINE__)
// Checks the launch just issued: kernel<<<g, b, 0, s>>>(...); EDA_CHECK_LAUNCH("kernel");
#define EDA_CHECK_LAUNCH(kernel_name) ::eda::check_launch((kernel_name), __FILE__, __LINE__)
