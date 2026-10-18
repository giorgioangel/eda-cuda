# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
"""eda-cuda: N-dimensional empirical decomposition on NVIDIA GPUs.

``eda.decompose(x)`` splits every 1-D, 2-D or 3-D field of an array into intrinsic mode functions
and a residue; ``eda.monogenic(x)`` gives the local amplitude, phase, frequency and direction of
fields such as those modes. Importing the package needs neither a GPU nor the compiled extension;
the extension loads on the first call that needs it.
"""

from __future__ import annotations

import operator
from typing import Any

from . import _native
from ._api import DecompositionInfo, Monogenic, Orientation, decompose, monogenic
from ._device import DeviceArray
from ._errors import CudaError, DecompositionError

__version__ = "0.1.0"

__all__ = [
    "CudaError",
    "DecompositionError",
    "DecompositionInfo",
    "DeviceArray",
    "Monogenic",
    "Orientation",
    "__version__",
    "build_info",
    "decompose",
    "device_count",
    "device_info",
    "monogenic",
]


def build_info() -> dict[str, Any]:
    """How the extension was built: ``version``, the CUDA toolkit, runtime and driver versions,
    the architectures (``cuda_architectures``, ``sass``, ``ptx``), the build type, the compilers,
    the CCCL version and ``cufft_version`` (the cuFFT library mapped into the process). Works
    without a GPU."""
    return _native.core().build_info()


def device_count() -> int:
    """The number of visible CUDA devices; 0 without a driver or device."""
    return _native.core().device_count()


def device_info(device: int = 0) -> dict[str, Any]:
    """Properties of a visible CUDA device: name, PCI bus id, compute capability, multiprocessors,
    memory sizes, driver and runtime versions. Raises ``ValueError`` for a device that is not
    visible and ``TypeError`` for an argument that is not an integer."""
    if isinstance(device, bool):
        raise TypeError("eda: device must be an integer, not bool")
    try:
        number = operator.index(device)
    except TypeError:
        raise TypeError(f"eda: device must be an integer, got {type(device).__name__}") from None
    if not -(2**31) <= number < 2**31:
        raise ValueError(f"eda: CUDA device {number} is not visible")
    return _native.core().device_info(number)
