// SPDX-License-Identifier: Apache-2.0
// Strict registry of the EDA_* environment variables (see env_registry.def). Readers validate
// the value of their variable on every read; validate_environment() checks all of them at once
// and also rejects EDA_* names that are not registered (a misspelled switch must not be ignored).
#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace eda {

// An EDA_* variable is unknown or has a value its registry entry rejects (Python: ValueError).
class EnvError : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

struct EnvVar {
  std::string_view name;
  std::string_view default_value;
};

const std::vector<EnvVar>& env_registry();

// The boolean value of a registered variable; an unset or empty variable yields the registered
// default. Throws EnvError for an invalid value and std::logic_error when `name` is not
// registered.
bool env_bool(std::string_view name);

// Validates every EDA_* variable of the process environment (unknown names and invalid values
// raise EnvError).
void validate_environment();

// The library's EDA_* settings, validated and read from the environment in one pass.
//
// The readers above and validate_environment() read the live process environment (getenv,
// environ). That is not safe while another thread modifies the environment (setenv may reallocate
// the environ array), and a call that read its settings piecemeal could see them change midway.
// Code that may run concurrently with environment writers therefore takes one snapshot up front
// and reads only the snapshot: the Python bindings capture it with the GIL held (Python's
// os.environ writes hold the GIL) before they release the GIL, and pass it to Context::current.
struct RuntimeConfig {
  bool sanitize = false;                       // EDA_SANITIZE (also the arena's debug mode)
  bool arena_poison = true;                    // EDA_ARENA_POISON
  std::optional<std::string> visible_devices;  // CUDA_VISIBLE_DEVICES (error messages only)

  // validate_environment(), then every variable (throws EnvError like the readers).
  static RuntimeConfig from_environment();
};

// EDA_SANITIZE: sanitizer support (arena debug mode, zeroed output rows, small test shapes).
bool sanitize_mode();
// EDA_ARENA_POISON: fill debug-mode slices with 0xFF bytes (default on).
bool arena_poison();

}  // namespace eda
