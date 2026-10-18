// SPDX-License-Identifier: Apache-2.0
// Minimal check helpers for the ctest executables (no third-party test framework).
#pragma once

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <string>

namespace eda_test {

inline int& failures() {
  static int count = 0;
  return count;
}
inline int& checks() {
  static int count = 0;
  return count;
}

inline void record(bool ok, const char* what, const char* file, int line) {
  ++checks();
  if (!ok) {
    ++failures();
    std::fprintf(stderr, "CHECK FAILED %s:%d: %s\n", file, line, what);
  }
}

// Runs `body`, expects an exception of type E whose what() contains `needle`.
template <class E>
bool throws_with(const std::function<void()>& body, const std::string& needle) {
  try {
    body();
  } catch (const E& error) {
    if (std::string(error.what()).find(needle) != std::string::npos) return true;
    std::fprintf(stderr, "  exception text lacks \"%s\": %s\n", needle.c_str(), error.what());
    return false;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "  unexpected exception type: %s\n", error.what());
    return false;
  }
  std::fprintf(stderr, "  no exception thrown (expected one containing \"%s\")\n", needle.c_str());
  return false;
}

inline int finish(const char* test) {
  if (failures()) {
    std::fprintf(stderr, "%s: %d of %d checks FAILED\n", test, failures(), checks());
    return 1;
  }
  std::printf("%s: all %d checks passed\n", test, checks());
  return 0;
}

}  // namespace eda_test

#define CHECK(cond) ::eda_test::record(static_cast<bool>(cond), #cond, __FILE__, __LINE__)
