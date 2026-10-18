# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
"""CUDA arrays in and out: intake of host and device input, stream handles, the result type.

Nothing here imports CuPy or PyTorch. A CUDA array is read through DLPack or the CUDA array
interface; a result of CUDA input is a ``DeviceArray`` that both protocols export.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import _native

#: The dtype names a CUDA array may have (the native layer casts them on the GPU).
SUPPORTED_DTYPES = frozenset(
    (
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
    )
)

LEGACY_STREAM = 1  # handle of the legacy default stream; 0 is normalized to it

_DLPACK_HOST = (1, 3)  # kDLCPU, kDLCUDAHost
_DLPACK_CUDA = (2, 13)  # kDLCUDA, kDLCUDAManaged
_STREAM_HELP = "an integer CUDA stream handle or an object with __cuda_stream__"


def stream_handle(stream: Any) -> int:
    """The integer handle of a ``stream=`` argument: an integer, an object with the CUDA stream
    protocol (``__cuda_stream__() -> (version, handle)``), a PyTorch stream (``cuda_stream``) or a
    CuPy stream (``ptr``). 0 is the legacy default stream (1)."""
    bad = TypeError(f"eda: stream must be {_STREAM_HELP}, got {type(stream).__name__}")
    if isinstance(stream, (bool, np.bool_)):
        raise bad
    if hasattr(stream, "__index__"):
        value = stream
    elif hasattr(stream, "__cuda_stream__"):
        pair = stream.__cuda_stream__()
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise bad
        value = pair[1]
    elif hasattr(stream, "cuda_stream"):
        value = stream.cuda_stream
    elif hasattr(stream, "ptr"):
        value = stream.ptr
    else:
        raise bad
    if isinstance(value, (bool, np.bool_)):
        raise bad
    try:
        handle = operator.index(value)
    except TypeError:
        raise bad from None
    if handle < 0:
        raise ValueError(f"eda: stream handle {handle} is negative")
    return handle or LEGACY_STREAM


@dataclass(frozen=True)
class DeviceInput:
    """A CUDA array as the native layer reads it: strides in elements, ``owner`` keeps the
    producer's memory alive while the call runs, ``fences`` are the streams to wait for.
    ``device`` is ``None`` for a CUDA array interface input, whose device is read from its pointer
    only after the arguments are checked (``device_of``), so that no check touches CUDA."""

    owner: Any
    ptr: int
    device: int | None
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    fences: tuple[int, ...]


def device_of(data: DeviceInput) -> int:
    """The CUDA device of a device input (a CUDA call for a CUDA array interface input)."""
    if data.device is not None:
        return data.device
    return _native.core().pointer_device(data.ptr)


def intake(x: Any, explicit: int | None) -> np.ndarray | DeviceInput:
    """``x`` as a NumPy array (host input) or a ``DeviceInput`` (CUDA input).

    Order: a NumPy array is host input; an object with ``__dlpack__`` and ``__dlpack_device__`` is
    read through DLPack (host memory: ``numpy.from_dlpack``; CUDA memory: ``DeviceInput``); an
    object with ``__cuda_array_interface__`` is CUDA input; anything else goes through
    ``numpy.asarray``. DLPack comes first because its stream semantics are the same for every
    producer.
    """
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "__dlpack__") and hasattr(x, "__dlpack_device__"):
        return _from_dlpack(x, explicit)
    cai = getattr(x, "__cuda_array_interface__", None)
    if cai is not None:
        return _from_interface(x, cai, explicit)
    return np.asarray(x)


def _detached(x: Any) -> Any:
    """A tensor that requires gradients cannot be exported; its detached view holds the values."""
    if getattr(x, "requires_grad", False) is True and hasattr(x, "detach"):
        return x.detach()
    return x


def _from_dlpack(x: Any, explicit: int | None) -> np.ndarray | DeviceInput:
    x = _detached(x)
    kind, _ = x.__dlpack_device__()
    kind = int(kind)
    if kind in _DLPACK_HOST:
        try:
            return np.from_dlpack(x)
        except (BufferError, RuntimeError, TypeError, ValueError, NotImplementedError) as exc:
            raise TypeError(f"eda: the host array cannot be read through DLPack: {exc}") from exc
    if kind not in _DLPACK_CUDA:
        raise TypeError(
            f"eda: arrays on DLPack device type {kind} are not supported; pass a NumPy array or "
            "a CUDA array"
        )
    consumer = LEGACY_STREAM if explicit is None else explicit
    try:
        try:
            capsule = x.__dlpack__(stream=consumer, max_version=(1, 1))
        except TypeError:  # a producer that predates the versioned protocol
            capsule = x.__dlpack__(stream=consumer)
    except (BufferError, RuntimeError, ValueError, NotImplementedError) as exc:
        raise TypeError(f"eda: the CUDA array cannot be exported through DLPack: {exc}") from exc
    owner, ptr, _, device, dtype, shape, strides = _native.core().dlpack_info(capsule)
    return DeviceInput(owner, ptr, device, dtype, shape, strides, (consumer,))


def _interface_field(cai: dict, key: str) -> Any:
    try:
        return cai[key]
    except KeyError:
        raise ValueError(f"eda: __cuda_array_interface__ has no {key!r} entry") from None


def _from_interface(x: Any, cai: dict, explicit: int | None) -> DeviceInput:
    try:
        shape = tuple(operator.index(n) for n in _interface_field(cai, "shape"))
        typestr = _interface_field(cai, "typestr")
        data = _interface_field(cai, "data")
        ptr = operator.index(data[0])
        version = operator.index(cai.get("version", 0))
        strides = cai.get("strides")
        if strides is not None:
            strides = tuple(operator.index(s) for s in strides)
    except (TypeError, IndexError) as exc:
        raise ValueError(f"eda: __cuda_array_interface__ is malformed: {exc}") from exc
    if cai.get("mask") is not None:
        raise ValueError(
            "eda: __cuda_array_interface__ has a mask; masked arrays are not supported"
        )
    if not isinstance(typestr, str):
        raise ValueError("eda: __cuda_array_interface__ typestr must be a string")
    if typestr[:1] == ">":
        raise ValueError(f"eda: __cuda_array_interface__ typestr {typestr!r} is big-endian")
    try:
        dtype = np.dtype(typestr)
    except TypeError as exc:
        raise ValueError(f"eda: __cuda_array_interface__ typestr {typestr!r} is not valid") from exc
    item = dtype.itemsize
    if strides is None:
        elements, acc = [], 1
        for n in reversed(shape):
            elements.append(acc)
            acc *= max(n, 1)
        strides = tuple(reversed(elements))
    else:
        if len(strides) != len(shape):
            raise ValueError("eda: __cuda_array_interface__ strides and shape differ in length")
        if any(s % item for s in strides):
            raise ValueError(
                f"eda: __cuda_array_interface__ strides {strides} are not multiples of the "
                f"item size {item}"
            )
        strides = tuple(s // item for s in strides)
    fences = _interface_fences(cai, version, explicit)
    return DeviceInput(x, ptr, None, dtype.name, shape, strides, fences)


def _interface_fences(cai: dict, version: int, explicit: int | None) -> tuple[int, ...]:
    """The streams a CUDA array interface input waits for: the stream the producer reports
    (version 3; ``None`` means no wait), or the legacy default stream when it reports none
    (version below 3), plus ``stream=`` when given."""
    fences: list[int] = []
    if version >= 3:
        reported = cai.get("stream")
        if reported is not None:
            if isinstance(reported, (bool, np.bool_)) or not hasattr(reported, "__index__"):
                raise ValueError(
                    f"eda: __cuda_array_interface__ stream {reported!r} is not an integer"
                )
            handle = operator.index(reported)
            if handle <= 0:
                raise ValueError(
                    f"eda: __cuda_array_interface__ stream {handle} is not a valid stream "
                    "(version 3 forbids 0)"
                )
            fences.append(handle)
    elif explicit is None:
        fences.append(LEGACY_STREAM)
    if explicit is not None:
        fences.append(explicit)
    return tuple(fences)


class DeviceArray:
    """An eda result in CUDA device memory.

    It owns its memory and exposes it to other libraries without a copy, through DLPack
    (``__dlpack__``, ``__dlpack_device__``) and the CUDA array interface (version 3): for example
    ``cupy.asarray(y)``, ``cupy.from_dlpack(y)``, ``torch.from_dlpack(y)``, ``torch.as_tensor(y)``.
    The data is complete when the eda call returns, so any stream may read it. Once the last
    reference to the array, and to every array a library made from it, is gone, the memory is
    freed by the next eda call on any thread (or at exit). ``shape`` and ``strides`` (in bytes)
    describe the layout like NumPy's; ``device`` is the CUDA device ordinal. Users do not
    construct it.
    """

    __slots__ = ("_owner", "_ptr", "device", "dtype", "shape", "strides")

    def __init__(
        self,
        owner: Any,
        ptr: int,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        dtype: np.dtype,
        device: int,
    ) -> None:
        self._owner = owner
        self._ptr = ptr
        self.shape = shape
        self.strides = strides
        self.dtype = dtype
        self.device = device

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        n = 1
        for extent in self.shape:
            n *= extent
        return n

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.itemsize

    def _contiguous(self) -> bool:
        acc = self.dtype.itemsize
        for n, s in zip(reversed(self.shape), reversed(self.strides), strict=True):
            if n != 1 and s != acc:
                return False
            acc *= n
        return True

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "typestr": self.dtype.str,
            "data": (self._ptr, False),
            "strides": None if self.size == 0 or self._contiguous() else self.strides,
            "version": 3,
            "stream": None,
        }

    def __dlpack_device__(self) -> tuple[int, int]:
        return (2, self.device)  # kDLCUDA

    def __dlpack__(
        self,
        *,
        stream: Any = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> Any:
        """A DLPack capsule over the memory (versioned when ``max_version >= (1, 0)``). The data
        is complete already, so ``stream`` is ignored; a copy or another device is refused."""
        if copy:
            raise BufferError(
                "eda: a DeviceArray cannot be copied by __dlpack__; copy it with the consumer"
            )
        if dl_device is not None and tuple(dl_device) != (2, self.device):
            raise BufferError(
                f"eda: a DeviceArray lives on (2, {self.device}); got dl_device={dl_device}"
            )
        item = self.dtype.itemsize
        exporter = _native.core().dlpack_export(
            self._owner,
            self._ptr,
            self.shape,
            tuple(s // item for s in self.strides),
            self.dtype.name,
            self.device,
        )
        if max_version is None:
            return exporter.__dlpack__()
        return exporter.__dlpack__(max_version=max_version)

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        raise TypeError(
            "eda: a DeviceArray is CUDA memory; convert it with its consumer library, for "
            "example cupy.asnumpy(y) or torch.from_dlpack(y).cpu()"
        )

    def __repr__(self) -> str:
        return f"DeviceArray(shape={self.shape}, dtype={self.dtype.name}, device={self.device})"
