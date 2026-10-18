# SPDX-License-Identifier: Apache-2.0
"""Lazy loading of the compiled extension eda._core."""

from __future__ import annotations

from types import ModuleType

_module: ModuleType | None = None

# The soname the extension is linked against (DT_NEEDED) and the NVIDIA runtime wheel that provides
# it (pyproject.toml, `dependencies`): cuFFT 12 from `nvidia-cufft` for the CUDA 13 package
# eda-cuda, cuFFT 11 from `nvidia-cufft-cu12` for the CUDA 12 package eda-cuda-cu12. The build
# writes them (cpp/cmake/build_config.py.in); a tree without a build has the CUDA 13 values.
try:
    from ._build_config import CUFFT_PACKAGE as _CUFFT_PACKAGE
    from ._build_config import CUFFT_SONAME as _CUFFT_SONAME
    from ._build_config import DISTRIBUTION as _DISTRIBUTION
except ImportError:
    _DISTRIBUTION, _CUFFT_SONAME, _CUFFT_PACKAGE = "eda-cuda", "libcufft.so.12", "nvidia-cufft"


def _mapped(soname: str) -> bool:
    """Whether a library of this file name is already mapped into the process (Linux)."""
    try:
        with open("/proc/self/maps", encoding="utf-8") as maps:
            return any(line.rstrip().endswith("/" + soname) for line in maps)
    except OSError:
        return False


def _preload_cufft() -> str | None:
    """Load libcufft from the NVIDIA runtime wheel before the extension, with global symbols.

    The extension has a DT_NEEDED entry on the soname and no run path to the wheel; a library
    already loaded under that soname satisfies it. Nothing is loaded when the process already
    maps the library (torch, CuPy: one cuFFT per process) or when the wheel is not installed (the
    system loader then looks for it as usual). Returns the path loaded, or None.
    """
    import ctypes
    import glob
    import importlib.util
    import os

    if _mapped(_CUFFT_SONAME):
        return None
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        return None
    for base in (spec.submodule_search_locations or []) if spec else []:
        # CUDA 13 wheels share nvidia/cu13/lib; the CUDA 12 wheels have one directory each
        # (nvidia/cufft/lib).
        for path in sorted(glob.glob(os.path.join(base, "*", "lib", _CUFFT_SONAME))):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
            return path
    return None


def core() -> ModuleType:
    """The extension module, imported on first use.

    Raises ImportError with build instructions when it is missing or cannot be loaded.
    """
    global _module
    if _module is None:
        _preload_cufft()
        try:
            from . import _core as module
        except ImportError as exc:
            if "libcufft" in str(exc):
                raise ImportError(
                    f"eda: the {_DISTRIBUTION} CUDA extension needs {_CUFFT_SONAME} and the "
                    f"loader did not find it: {exc}. Install the NVIDIA cuFFT runtime package "
                    f"(`pip install {_CUFFT_PACKAGE}`, a dependency of {_DISTRIBUTION}) or make "
                    "the CUDA toolkit's library directory visible to the loader."
                ) from exc
            raise ImportError(
                f"eda: the {_DISTRIBUTION} CUDA extension (eda._core) is not available: {exc}. "
                "Build it with CMake (EDA_BUILD_PYTHON=ON); otherwise install an "
                f"{_DISTRIBUTION} wheel built for this platform."
            ) from exc
        _module = module
    return _module
