# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
"""The device memory this process holds, read through the driver's NVML (libnvidia-ml).

NVML counts the memory per process, so other programs on a shared GPU do not enter it (the free
memory of cudaMemGetInfo is device-wide).
"""

from __future__ import annotations

import ctypes
import os


class _Process(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("used", ctypes.c_ulonglong),
        ("gi", ctypes.c_uint),
        ("ci", ctypes.c_uint),
    ]


class ProcessMemory:
    """``held()`` gives the bytes of device memory of this process on the device of the PCI bus id
    (for example CuPy's ``cupy.cuda.Device().pci_bus_id``)."""

    def __init__(self, pci_bus_id: str):
        self._nvml = ctypes.CDLL("libnvidia-ml.so.1")
        if self._nvml.nvmlInit_v2() != 0:
            raise OSError("nvmlInit failed")
        self._handle = ctypes.c_void_p()
        bus = pci_bus_id.encode()
        if self._nvml.nvmlDeviceGetHandleByPciBusId_v2(bus, ctypes.byref(self._handle)) != 0:
            raise OSError(f"NVML does not know the device {pci_bus_id}")
        self.held()

    def held(self) -> int:
        count, table = ctypes.c_uint(256), (_Process * 256)()
        status = self._nvml.nvmlDeviceGetComputeRunningProcesses_v3(
            self._handle, ctypes.byref(count), table
        )
        if status != 0:
            raise OSError(f"nvmlDeviceGetComputeRunningProcesses failed ({status})")
        pid = os.getpid()
        return sum(table[i].used for i in range(count.value) if table[i].pid == pid)
