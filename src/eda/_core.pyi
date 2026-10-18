"""eda native extension (private; use the eda package)"""

from collections.abc import Sequence
from typing import Annotated

from numpy.typing import NDArray


def build_info() -> dict:
    """Build information of the extension (works without a GPU)."""

def device_count() -> int:
    """Number of visible CUDA devices (0 without a driver or device)."""

def device_info(device: int = 0) -> dict:
    """Properties of a visible CUDA device."""

def decompose_host(x: Annotated[NDArray, dict(order='C', device='cpu', writable=False)], max_imf: int, tie_tolerance: float | None, sifts: int, check_finite: bool, orientation: bool, max_chunk: int) -> tuple:
    """
    Decompose the C-contiguous host array x (B, *spatial) of float32 or float64 on the calling thread's current device: (rows (B, P, *spatial), info); with orientation, info holds direction and coherence.
    """

def decompose_device(ptr: int, device: int, dtype: str, target: str, batch_shape: Sequence[int], batch_strides: Sequence[int], shape: Sequence[int], strides: Sequence[int], fences: Sequence[int], max_imf: int, tie_tolerance: float | None, sifts: int, check_finite: bool, orientation: bool, max_chunk: int) -> tuple:
    """
    Decompose a CUDA array on its device after waiting for the fence streams: (owner, ptr, info), the rows C-contiguous (B, P, *spatial) in target, complete on return; with orientation, info['orientation'] is (owner, ptr) of the direction (B, K, D, *spatial) followed by the coherence (B, K, *spatial).
    """

def monogenic_host(x: Annotated[NDArray, dict(order='C', device='cpu', writable=False)], check_finite: bool, max_chunk: int) -> tuple:
    """
    The monogenic signal of the C-contiguous host array x (B, *spatial) of float32 or float64: (out, info), info holding nonfinite and chunk, out the flat planes amplitude, phase, frequency (B, *spatial), then for 2 or 3 spatial axes direction (B, D, *spatial) and coherence (B, *spatial).
    """

def monogenic_device(ptr: int, device: int, dtype: str, target: str, batch_shape: Sequence[int], batch_strides: Sequence[int], shape: Sequence[int], strides: Sequence[int], fences: Sequence[int], check_finite: bool, max_chunk: int) -> tuple:
    """
    The monogenic signal of a CUDA array after waiting for the fence streams: (owner, ptr, info), the planes of monogenic_host in one device buffer, complete on return.
    """

def dlpack_info(capsule: object) -> tuple:
    """
    Import a DLPack capsule: (owner, ptr, device_type, device_id, dtype, shape, strides in elements).
    """

def pointer_device(ptr: int) -> int:
    """The CUDA device of a device or managed pointer."""

def dlpack_export(owner: object | None, ptr: int, shape: Sequence[int], strides: Sequence[int], dtype: str, device: int) -> object:
    """
    A DLPack exporter of float32/float64 device memory kept alive by owner (strides in elements).
    """
