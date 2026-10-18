// SPDX-License-Identifier: Apache-2.0
#include "runtime/env.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>

extern char** environ;

namespace eda {
namespace {

std::string trim(std::string_view text) {
  size_t begin = 0;
  size_t end = text.size();
  while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) ++begin;
  while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) --end;
  return std::string(text.substr(begin, end - begin));
}

std::string lower(std::string text) {
  for (char& c : text) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return text;
}

std::optional<bool> parse_bool(std::string_view text) {
  const std::string value = lower(trim(text));
  if (value == "1" || value == "true" || value == "on" || value == "yes") return true;
  if (value == "0" || value == "false" || value == "off" || value == "no") return false;
  return std::nullopt;
}

const EnvVar* find_var(std::string_view name) {
  for (const EnvVar& var : env_registry()) {
    if (var.name == name) return &var;
  }
  return nullptr;
}

const EnvVar& spec_of(std::string_view name) {
  const EnvVar* var = find_var(name);
  if (!var) throw std::logic_error("eda: the library reads the unregistered environment variable " + std::string(name));
  return *var;
}

void require_valid(const EnvVar& var, const std::string& trimmed, std::string_view raw) {
  if (!parse_bool(trimmed)) {
    throw EnvError("eda: " + std::string(var.name) + "=\"" + std::string(raw) +
                   "\" is invalid: expected a boolean (1/0, true/false, on/off, yes/no) (EDA_* variables are "
                   "listed in cpp/src/runtime/env_registry.def)");
  }
}

// The trimmed environment value if set and non-empty (validated), else the registered default.
std::string effective_value(const EnvVar& var) {
  const char* raw = std::getenv(std::string(var.name).c_str());
  if (raw) {
    std::string value = trim(raw);
    if (!value.empty()) {
      require_valid(var, value, raw);
      return value;
    }
  }
  return std::string(var.default_value);
}

size_t edit_distance(std::string_view a, std::string_view b) {
  std::vector<size_t> row(b.size() + 1);
  for (size_t j = 0; j <= b.size(); ++j) row[j] = j;
  for (size_t i = 1; i <= a.size(); ++i) {
    size_t diagonal = row[0];
    row[0] = i;
    for (size_t j = 1; j <= b.size(); ++j) {
      const size_t above = row[j];
      row[j] = std::min({row[j] + 1, row[j - 1] + 1, diagonal + (a[i - 1] == b[j - 1] ? 0u : 1u)});
      diagonal = above;
    }
  }
  return row[b.size()];
}

[[noreturn]] void reject_unknown(std::string_view name) {
  std::string message = "eda: unknown environment variable " + std::string(name);
  const EnvVar* closest = nullptr;
  size_t best = 4;  // suggest only close spellings
  for (const EnvVar& var : env_registry()) {
    const size_t distance = edit_distance(name, var.name);
    if (distance < best) {
      best = distance;
      closest = &var;
    }
  }
  if (closest) message += " (did you mean " + std::string(closest->name) + "?)";
  message += ". The library rejects unregistered EDA_* variables so that a misspelled switch is not "
             "silently ignored; registered:";
  for (const EnvVar& var : env_registry()) message += " " + std::string(var.name);
  throw EnvError(message);
}

}  // namespace

const std::vector<EnvVar>& env_registry() {
  static const std::vector<EnvVar> table = {
#define EDA_ENV(name, default_value, description) EnvVar{name, default_value},
#include "runtime/env_registry.def"
#undef EDA_ENV
  };
  return table;
}

bool env_bool(std::string_view name) { return parse_bool(effective_value(spec_of(name))).value_or(false); }

void validate_environment() {
  for (char** entry = environ; entry && *entry; ++entry) {
    const std::string_view text(*entry);
    if (!text.starts_with("EDA_")) continue;
    const size_t eq = text.find('=');
    const std::string_view name = text.substr(0, eq);
    const std::string_view raw = eq == std::string_view::npos ? std::string_view() : text.substr(eq + 1);
    const EnvVar* var = find_var(name);
    if (!var) reject_unknown(name);
    const std::string value = trim(raw);
    if (!value.empty()) require_valid(*var, value, raw);
  }
}

RuntimeConfig RuntimeConfig::from_environment() {
  validate_environment();
  RuntimeConfig config;
  config.sanitize = sanitize_mode();
  config.arena_poison = eda::arena_poison();
  if (const char* visible = std::getenv("CUDA_VISIBLE_DEVICES")) config.visible_devices = std::string(visible);
  return config;
}

bool sanitize_mode() { return env_bool("EDA_SANITIZE"); }
bool arena_poison() { return env_bool("EDA_ARENA_POISON"); }

}  // namespace eda
