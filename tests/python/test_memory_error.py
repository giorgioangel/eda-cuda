# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
"""A call that raises MemoryError gives back the device memory eda does not use.

In a fresh process (CuPy and the driver's NVML needed), calls fail after their rows are allocated:
at the workspace check, at the cuFFT plans and at the output of a finished run, each once with
CUDA input and once with host input (``decompose_device`` and ``decompose_host``). Then the free
memory steps across the boundary of the cuFFT plan, where cuFFT reports a shortage either as an
allocation failure or as an internal error: every call there raises MemoryError. A CuPy
allocation sets the free memory of each call, so the failures do not depend on the size of the
device. After each, the device memory the process holds (NVML, this pid only, so other processes
do not enter it) is back within MARGIN of its level after a warm-up. Then a few calls succeed and
equal those of another fresh process bit for bit, the chunk count included.

Not covered: the release after a CudaError of an allocation (the twiddle table of the cuFFT plan,
a small buffer of the noise scale), which the native layer maps to MemoryError as well. These
allocations are small and come after larger ones that fail first: cuFFT keeps tens of MiB free
after a plan it could make, so holding memory does not reach them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.gpu

SANITIZED = os.environ.get("EDA_SANITIZE") not in (None, "", "0")
# The failed call may load kernels the warm-up did not, and cuFFT keeps some state of its own;
# measured: nothing (the release also frees the warm-up's workspace and plans, so the level ends
# 66 MiB below the warm-up's). Without the release a failed call here keeps at least its rows,
# 1.2 GiB or more.
MARGIN = 64 << 20

SCRIPT = r"""
import gc, hashlib, json, sys
import numpy as np, cupy as cp, eda
import device_memory
from eda import _native

mode = sys.argv[1]
core = _native.core()
chunks = []
for name in ("decompose_host", "decompose_device"):
    def spy(*args, _original=getattr(core, name)):
        result = _original(*args)
        chunks.append(int(result[-1]["chunk"]))
        return result
    setattr(core, name, spy)

try:
    memory = device_memory.ProcessMemory(cp.cuda.Device().pci_bus_id)
except OSError as error:
    print(json.dumps({"skip": f"NVML is not available: {error}"}))
    sys.exit(0)
held = memory.held

def settle():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp.cuda.runtime.deviceSynchronize()

rng = np.random.default_rng(5)
line = (np.sin(0.2 * np.arange(1024)) + 0.1 * rng.standard_normal(1024)).astype(np.float32)
row = cp.asarray(line)

def zeros(shape):
    return cp.broadcast_to(cp.zeros(1, cp.float32), shape)

def following():
    # Calls after the failures: host and device input, D 1 to 3, both dtypes.
    x1 = (line + 0.01 * rng.standard_normal((64, 1024))).astype(np.float32)
    x2 = rng.standard_normal((3, 64, 64))
    x3 = cp.asarray(rng.standard_normal((2, 32, 32, 32)).astype(np.float32))
    out = []
    for x, kw in ((x1, {"axes": 1}), (x2, {"axes": (1, 2)}), (x3, {"axes": (1, 2, 3)}),
                  (cp.broadcast_to(row, (64, 1024)), {"axes": 1, "max_imf": 1})):
        y, info = eda.decompose(x, return_info=True, **kw)
        y = cp.asnumpy(cp.asarray(y)) if isinstance(y, eda.DeviceArray) else y
        digest = hashlib.sha256(y.tobytes())
        for name in ("n_imfs", "error", "windows", "spacing", "sifts", "energy"):
            digest.update(np.ascontiguousarray(getattr(info, name)).tobytes())
        out.append([list(y.shape), str(y.dtype), digest.hexdigest(), chunks[-1]])
    return out

report = {}
if mode == "fail":
    # warm-up: the kernels and the plans of the paths below, a small workspace
    eda.decompose(cp.broadcast_to(row, (64, 1024)), axes=1, max_imf=1)
    eda.decompose(zeros((16, 16, 16)), max_imf=1)
    settle()
    report["before"] = held()
    lines = int(0.7 * (2 << 30)) // (3 * 4096)
    cases = [
        # rows 1.8 GiB and the plan (1.2 GiB of work area) fit, the workspace does not
        ("workspace", (603, 256, 1024), {"max_imf": 1}, 6 << 30),
        # rows 1.2 GiB fit, the plan does not (804 MiB of work area with cuFFT 12.0)
        ("cuFFT", (201, 512, 1024), {"max_imf": 1}, 12 * 201 * 512 * 1024 + (600 << 20)),
        # rows 1.4 GiB of 2 GiB free, a finished run, then 2 of the 3 rows of output (or of the
        # device copy of a host output) do not fit
        ("output", (lines, 1024), {"axes": 1, "max_imf": 1}, 2 << 30),
    ]
    # the free memory beside the rows steps across the work area of the plan (804 MiB), up to
    # 70 MiB above it
    rows = 12 * 201 * 512 * 1024
    for mib in range(798, 876, 6):
        free = rows + (mib << 20)
        cases.append((f"plan boundary {mib} MiB", (201, 512, 1024), {"max_imf": 1}, free))
    if cp.cuda.runtime.memGetInfo()[0] < 7 << 30:
        print(json.dumps({"skip": "the test needs 7 GiB of free device memory"}))
        sys.exit(0)
    report["cases"] = {}
    for name, shape, kw, free in cases:
        for kind in ("device", "host") if not name.startswith("plan") else ("device",):
            settle()
            if name == "output":
                x = np.broadcast_to(line, shape) if kind == "host" else cp.broadcast_to(row, shape)
            else:
                x = np.zeros(shape, np.float32) if kind == "host" else zeros(shape)
            hold = cp.cuda.runtime.malloc(cp.cuda.runtime.memGetInfo()[0] - free)
            try:
                eda.decompose(x, **kw)
                message = None
            except MemoryError as error:
                message = str(error)
            finally:
                cp.cuda.runtime.free(hold)
                x = None
            settle()
            report["cases"][f"{name} ({kind} input)"] = {"message": message, "after": held()}
report["following"] = following()
row = x = None  # every CuPy array goes before the exit (leak checks)
settle()
print(json.dumps(report))
"""


def run(mode: str) -> dict:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    done = subprocess.run(
        [sys.executable, "-c", SCRIPT, mode],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    report = json.loads(done.stdout.strip().splitlines()[-1])
    if "skip" in report:
        pytest.skip(report["skip"])
    return report


@pytest.fixture(scope="module")
def cupy():
    return pytest.importorskip("cupy")


@pytest.mark.skipif(SANITIZED, reason="the sanitizer's own tracking uses device memory")
def test_a_memory_error_gives_back_the_memory_eda_does_not_use(cupy):
    failed = run("fail")
    for name, case in failed["cases"].items():
        assert case["message"] is not None, f"{name}: the call did not raise MemoryError"
        if not name.startswith("plan boundary"):
            assert name.split(" (")[0] in case["message"], (name, case["message"])
        excess = case["after"] - failed["before"]
        assert excess <= MARGIN, f"{name}: {excess >> 20} MiB still held after the MemoryError"
    assert failed["following"] == run("fresh")["following"]
