// SPDX-License-Identifier: Apache-2.0
#include "runtime/error.hpp"

namespace eda {

std::string cuda_error_text(cudaError_t code) {
  const char* name = cudaGetErrorName(code);
  const char* text = cudaGetErrorString(code);
  std::string out = name ? name : "cudaError(" + std::to_string(static_cast<int>(code)) + ")";
  if (text && *text) {
    out += " (";
    out += text;
    out += ")";
  }
  return out;
}

std::string_view source_path(const char* file) noexcept {
  if (!file) return {};
  const std::string_view path(file);
  if (path.starts_with("cpp/")) return path;
  const size_t at = path.rfind("/cpp/");
  return at == std::string_view::npos ? path : path.substr(at + 1);
}

void throw_cuda_error(cudaError_t code, std::string_view what, const char* file, int line) {
  // Reset the non-sticky last-error state: the error travels in the exception.
  (void)cudaGetLastError();
  std::string message("eda: ");
  message += what;
  message += " failed: ";
  message += cuda_error_text(code);
  if (file) {
    message += " at ";
    message += source_path(file);
    message += ":";
    message += std::to_string(line);
  }
  throw CudaError(code, message);
}

void check_launch(const char* kernel, const char* file, int line) {
  const cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) [[unlikely]] {
    throw_cuda_error(code, std::string("launch of kernel ") + (kernel ? kernel : "?"), file, line);
  }
}

}  // namespace eda
