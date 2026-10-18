// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Giorgio Angelotti
// The process lifecycle of eda._core (defined in decompose.cpp).
//
// Every entry point that releases the GIL holds a Call from its first line to its return: the exit
// hook refuses new calls and waits until none is in progress, because a daemon thread that is
// still outside the GIL when the interpreter finalizes is ended inside the release, which aborts
// the process. Entry points that may initialize CUDA call note_cuda_used(), which also refuses to
// run in a child forked after CUDA was used.
#pragma once

#include <nanobind/nanobind.h>

class Call {
 public:
  Call();  // throws RuntimeError in a forked child and once the exit hook has run
  ~Call();
  Call(const Call&) = delete;
  Call& operator=(const Call&) = delete;
};

void note_cuda_used();

// The decomposition entry points, the exit hook and the fork hook.
void bind_decompose(nanobind::module_& m);
