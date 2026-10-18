# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
"""The public calls ``eda.decompose`` and ``eda.monogenic``: argument rules, layout, errors and the
result types."""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import _device, _native
from ._errors import DecompositionError

#: Error codes of an element (``DecompositionInfo.error``).
ERROR_NON_FINITE = 1
ERROR_NON_FINITE_RESULT = 2
ERROR_INTERNAL = 255
_ERROR_MESSAGES = {
    ERROR_NON_FINITE: "the input has inf or NaN; check_finite=True",
    ERROR_NON_FINITE_RESULT: (
        "a non-finite intermediate or output row: the decomposition overflows the dtype's range, "
        "or the input has inf or NaN with check_finite=False"
    ),
}
_FLOAT_DTYPES = ("float32", "float64")
_ERRORS = ("raise", "status")
_INT32_MAX = 2**31 - 1


@dataclass(frozen=True, eq=False)
class DecompositionInfo:
    """Per-call information of ``eda.decompose(..., return_info=True)``.

    Arrays have the batch shape (``n_imfs``, ``error``) or the batch shape plus ``K``, the largest
    IMF count (the others); entries beyond an element's ``n_imfs``, and all entries of a failed
    element, are 0.

    - ``n_imfs``: IMFs found per element (int32); the residue row follows them.
    - ``error``: 0, 1 (the input has inf or NaN and ``check_finite`` is on) or 2 (a non-finite
      intermediate or output row).
    - ``windows``: the window of each mode, in samples (int32).
    - ``spacing``: the extrema-density spacing of the window rule's second pass (float64).
    - ``sifts``: the sifts each mode completed, 1 to the call's ``sifts`` (int32).
    - ``energy``: the sum of the squared IMF, in the units of ``x`` (float64).
    - ``dtype``: the compute dtype, ``"float32"`` or ``"float64"``.
    - ``axes``: the spatial axes, normalized, in the order given.
    """

    n_imfs: np.ndarray
    error: np.ndarray
    windows: np.ndarray
    spacing: np.ndarray
    sifts: np.ndarray
    energy: np.ndarray
    dtype: str
    axes: tuple[int, ...]


@dataclass(frozen=True, eq=False)
class Orientation:
    """The orientation of every mode, from ``eda.decompose(..., return_orientation=True)``.

    With ``m = min(axes)``, ``K`` the largest IMF count of the call (as in ``DecompositionInfo``)
    and ``D`` the number of spatial axes (2 or 3):

    - ``direction``: shape ``x.shape[:m] + (K, D) + x.shape[m:]``, the unit direction ``n`` across
      the stripes that the method computes at the start of each mode; component ``a`` is along
      the ``a``-th axis of ``axes``, and the sign makes the first component of magnitude above
      the dtype's default tie tolerance positive, as in ``eda.monogenic``.
    - ``coherence``: shape ``x.shape[:m] + (K,) + x.shape[m:]``, how clearly the pattern has a
      direction, from 0 (none) to 1 (perfect stripes).

    Rows beyond an element's ``n_imfs``, and all rows of a failed element, are 0. NumPy arrays for
    host input, ``eda.DeviceArray`` for CUDA input, in the compute dtype.
    """

    direction: Any
    coherence: Any


@dataclass(frozen=True, eq=False)
class Monogenic:
    """The monogenic signal of every field, from ``eda.monogenic``.

    - ``amplitude``, ``phase``, ``frequency``: the shape of ``x``. The phase lies in (-pi, pi];
      the frequency is in cycles per sample along ``direction``.
    - ``direction``: shape ``x.shape[:m] + (D,) + x.shape[m:]`` with ``m = min(axes)``, the unit
      direction across the local stripes, component ``a`` along the ``a``-th axis of ``axes``,
      with its first component of magnitude above the dtype's default tie tolerance positive (a
      smaller one is rounding noise); ``None`` in 1-D.
    - ``coherence``: the shape of ``x``, from 0 (no direction) to 1 (perfect stripes); ``None`` in
      1-D.

    NumPy arrays for host input, ``eda.DeviceArray`` for CUDA input, in the compute dtype.
    """

    amplitude: Any
    phase: Any
    frequency: Any
    direction: Any
    coherence: Any


@dataclass(frozen=True)
class _Call:
    """The input of a call after the shared argument rules: ``data`` as ``_device.intake`` gives
    it, the source and compute dtypes, the normalized spatial axes, the batch axes and the
    explicit stream handle."""

    data: Any
    on_device: bool
    src: str
    target: str
    axes: tuple[int, ...]
    batch_axes: tuple[int, ...]
    spatial: tuple[int, ...]
    batch_shape: tuple[int, ...]

    @property
    def ndim(self) -> int:
        return len(self.data.shape)

    def device_args(self) -> tuple:
        """The input arguments of the native ``*_device`` calls."""
        data = self.data
        return (
            data.ptr,
            _device.device_of(data),
            self.src,
            self.target,
            self.batch_shape,
            tuple(data.strides[i] for i in self.batch_axes),
            self.spatial,
            tuple(data.strides[a] for a in self.axes),
            list(data.fences),
        )

    def host_batch(self) -> np.ndarray:
        """The host input as a C-contiguous ``(B, *spatial)`` array in the compute dtype."""
        data, ndim = self.data, self.ndim
        order = list(self.batch_axes) + list(self.axes)
        xc = data if data.dtype == np.dtype(self.target) else data.astype(self.target)
        xt = np.ascontiguousarray(xc.transpose(order)) if order != list(range(ndim)) else xc
        return np.ascontiguousarray(xt).reshape((math.prod(self.batch_shape),) + self.spatial)


def _prepare(x: Any, axes: Any, dtype: Any, stream: Any) -> _Call:
    """The argument rules that ``decompose`` and ``monogenic`` share: the stream, the input, its
    dtype and the axes."""
    explicit = None if stream is None else _device.stream_handle(stream)
    data = _device.intake(x, explicit)
    on_device = isinstance(data, _device.DeviceInput)
    src = data.dtype if on_device else data.dtype.name
    shape = data.shape
    ndim = len(shape)
    _check_source(data, src, on_device)
    target = src if dtype is None else _dtype_name(dtype)
    if target not in _FLOAT_DTYPES:
        if dtype is None:
            raise TypeError(
                f"eda: expected a float32 or float64 array, got dtype {src} (pass dtype='float32' "
                "or dtype='float64' to cast it)"
            )
        raise TypeError(f"eda: dtype= must be float32 or float64, got {target}")
    axes = _normalize_axes(axes, ndim)
    batch_axes = tuple(i for i in range(ndim) if i not in axes)
    return _Call(
        data,
        on_device,
        src,
        target,
        axes,
        batch_axes,
        tuple(shape[a] for a in axes),
        tuple(shape[i] for i in batch_axes),
    )


def decompose(
    x: Any,
    axes: int | Sequence[int] | None = None,
    max_imf: int = -1,
    *,
    dtype: Any = None,
    tie_tolerance: float | None = None,
    sifts: int = 3,
    check_finite: bool = True,
    errors: str = "raise",
    stream: Any = None,
    return_info: bool = False,
    return_orientation: bool = False,
) -> Any:
    """Decompose every 1-D, 2-D or 3-D field of ``x`` into intrinsic mode functions (IMFs) and a
    residue.

    ``x`` is a NumPy array (or anything ``numpy.asarray`` accepts) or a CUDA array that implements
    DLPack or the CUDA array interface (CuPy, PyTorch and others). ``axes`` names the spatial
    axes, one to three of them, in the order the method sees them; ``axes=None`` takes every axis
    and needs ``x.ndim <= 3``. The other axes form the batch. Each batch element is decomposed on
    its own: its result is the same, bit for bit, alone or at any position of any batch, whatever
    the other elements hold, for every shape and dtype (with the same build, GPU and cuFFT). Every
    spatial extent must be at least 2, and at most 8192 (float32) or 4096 (float64).

    The result has the shape of ``x`` with a mode axis inserted before the first spatial axis:
    ``y.shape == x.shape[:m] + (P,) + x.shape[m:]`` with ``m = min(axes)``. Along the mode axis
    each element holds its IMFs from fine to coarse, then its residue, then zero rows. ``P`` is one
    more than the largest IMF count over the elements without an error. The rows of an element add
    up to the element up to rounding. ``max_imf`` limits the IMFs per element; 0 or a negative
    value means no limit.

    float32 and float64 input is decomposed in its own dtype. Any other real dtype (bool, integer,
    float16, and bfloat16 of a CUDA array) raises ``TypeError`` unless ``dtype`` (``"float32"`` or
    ``"float64"``) casts it first; ``dtype`` also sets the compute dtype of float input. The cast
    rounds to nearest, as ``numpy.ndarray.astype`` does, so host and CUDA input with the same
    values give bitwise equal results. Complex and other dtypes raise ``TypeError``. A CUDA array
    that cannot be read in place (another dtype, or batch axes that do not merge into one stride)
    is converted on the GPU into the compute dtype and layout; one that can is read without a
    copy.

    ``tie_tolerance`` is the tolerance, relative to the largest magnitude of the residual, under
    which two samples count as equal when extrema are counted; ``None`` takes ``2**-14`` for
    float32 and ``2**-30`` for float64.

    ``sifts`` is the number of sifts per mode, an integer >= 1. A sift subtracts the mean of the
    upper and lower envelopes; each further sift takes out more of the slower oscillations that
    the first one leaves in the mode, and makes the mode's band narrower. More sifts give sharper
    modes but split noise and smooth fields into more of them, and the sift has no limit to
    converge to: past a few sifts the modes drift instead of settling. The default 3 is the value
    the method was designed and calibrated with; 1 is the single-sift form. A sift that would make
    the mode or the remainder larger than the residual it started from ends the sifting of that
    mode early, so ``info.sifts`` reports between 1 and ``sifts`` per mode. The time per mode grows
    about linearly with ``sifts``.

    With ``check_finite=True`` an element that holds inf or NaN fails with error code 1; a
    non-finite intermediate or output row fails it with code 2. With ``errors="raise"`` a failed
    element makes the call raise ``DecompositionError``; with ``errors="status"`` the rows of a
    failed element are NaN and ``info.error`` holds its code. An internal failure (code 255) always
    raises ``RuntimeError``.

    Host input returns a NumPy array. A CUDA array returns an ``eda.DeviceArray`` on the device of
    the input, which CuPy, PyTorch and other libraries read without a copy. The result is complete
    when the call returns, so any stream can use it.

    Streams. The call runs on its own stream and never on the caller's. Before it reads a CUDA
    array it waits for the producer's pending writes:

    - An object with both DLPack and the CUDA array interface is read through DLPack. The call
      asks the producer for the data with ``__dlpack__(stream=s)``, where ``s`` is ``stream`` when
      given and otherwise the legacy default stream (handle 1), and then waits for ``s``. The
      producer makes ``s`` wait for its current stream, so data written on another stream needs
      ``stream=`` or a ``with`` block that makes that stream current (``with s:`` in CuPy,
      ``torch.cuda.stream(s)`` in PyTorch).
    - An object with only the CUDA array interface is waited for on the stream it reports
      (version 3; ``None`` reports none and the call does not wait). A producer of version 2 or
      lower reports none: the call then waits for the legacy default stream, or only for
      ``stream`` when it is given, so pass ``stream`` if its data was written on another stream.
      Stream handle 0 is refused in the interface (version 3 forbids it); 1 is the legacy default
      stream and 2 the per-thread default stream of the calling thread.
    - ``stream`` is an integer handle (0 means the legacy default stream), an object with
      ``__cuda_stream__``, a PyTorch stream or a CuPy stream. It must be a live stream of the
      device of ``x``: CUDA does not validate handles, and an unknown one can crash the process.
      It is type-checked and ignored for host input.
    - PyTorch exports a CUDA tensor only while the tensor's device is the current device (make it
      current with ``torch.cuda.device(i)``) and refuses ``stream=2``; the call then raises
      ``TypeError``.

    ``return_info=True`` returns ``(y, info)`` with a ``DecompositionInfo``.
    ``return_orientation=True`` (2 or 3 spatial axes) also returns an ``Orientation``: the direction
    and the coherence that the method computed for each mode. The return value is ``y``, then
    ``info`` if ``return_info``, then the ``Orientation`` if ``return_orientation``. Asking for the
    orientation does not change ``y`` or ``info``.

    Device memory. eda keeps the workspace of its calls in a memory pool of its own for later
    calls, until the interpreter exits; a call frees it early when its result would not fit
    beside it. The memory of a dropped ``DeviceArray`` is freed by the next eda call on
    any thread (or at exit), not at the moment it is dropped. A call that raises
    ``MemoryError`` first gives back to the device what the calling thread's eda state does not
    use: the call's own buffers, the workspace and cuFFT plans that thread keeps for later calls,
    and the free memory of eda's pools. The workspaces and plans of other threads, the memory of
    calls still running in them and the memory of live ``DeviceArray`` results stay.

    Raises ``TypeError`` and ``ValueError`` for arguments, ``eda.DecompositionError`` (a
    ``ValueError``) for failed elements with ``errors="raise"``, ``MemoryError`` when the device
    memory does not suffice and ``eda.CudaError`` (a ``RuntimeError``) when a CUDA call fails or no
    usable GPU exists.
    """
    max_imf = _max_imf(max_imf)
    tie = _tie_tolerance(tie_tolerance)
    sifts = _sifts(sifts)
    check_finite = _flag(check_finite, "check_finite")
    return_info = _flag(return_info, "return_info")
    return_orientation = _flag(return_orientation, "return_orientation")
    if not isinstance(errors, str) or errors not in _ERRORS:
        raise ValueError(f"eda: errors={errors!r} is not one of 'raise', 'status'")
    call = _prepare(x, axes, dtype, stream)
    if return_orientation and len(call.axes) < 2:
        raise ValueError(
            "eda: return_orientation=True needs 2 or 3 spatial axes, got 1 (the method has no "
            "orientation in 1-D)"
        )
    target, axes, batch_shape, spatial = call.target, call.axes, call.batch_shape, call.spatial
    ndim, on_device = call.ndim, call.on_device
    out_final = _restore_order(ndim, call.batch_axes, axes)
    core = _native.core()

    if on_device:
        args = call.device_args()
        device = args[1]
        owner, out_ptr, raw = core.decompose_device(
            *args, max_imf, tie, sifts, check_finite, return_orientation, 0
        )
    else:
        computed, raw = core.decompose_host(
            call.host_batch(), max_imf, tie, sifts, check_finite, return_orientation, 0
        )
        rows = computed.shape[1]

    info = _info(raw, batch_shape, target, axes)
    codes = info.error.reshape(-1)
    internal = np.flatnonzero(codes == ERROR_INTERNAL)
    if internal.size:
        raise RuntimeError(
            "eda: an internal invariant failed (error code 255) for batch index "
            f"{', '.join(str(int(i)) for i in internal[:8])}: this is a bug in eda"
        )
    failed = np.flatnonzero(codes != 0)
    if failed.size and errors == "raise":
        raise DecompositionError(
            _error_summary(codes[failed], failed, codes.size),
            indices=failed,
            codes=codes[failed],
            info=info,
        )

    if on_device:
        # P is one more than the largest IMF count; a failed element has n_imfs 0.
        rows = int(info.n_imfs.max(initial=0)) + 1
        computed_shape = batch_shape + (rows,) + spatial
        y = _device_result(owner, out_ptr, computed_shape, out_final, target, device)
    else:
        computed = computed.reshape(batch_shape + (rows,) + spatial)
        y = computed.transpose(out_final) if out_final != list(range(ndim + 1)) else computed
    out: tuple = (y,)
    if return_info:
        out += (info,)
    if return_orientation:
        out += (_orientation(call, raw, info.windows.shape[-1], device if on_device else None),)
    return out if len(out) > 1 else y


def monogenic(
    x: Any,
    axes: int | Sequence[int] | None = None,
    *,
    dtype: Any = None,
    check_finite: bool = True,
    stream: Any = None,
) -> Monogenic:
    """The monogenic signal of every 1-D, 2-D or 3-D field of ``x``: its local amplitude, phase,
    frequency and, in 2-D and 3-D, the direction and the coherence of the local stripes.

    ``x``, ``axes``, ``dtype``, ``check_finite`` and ``stream`` follow the rules of
    ``eda.decompose``: the same input types, dtype casts, axis rules and limits, batch axes and
    stream handling. Each field is processed on its own, and its result is the same, bit for bit,
    alone or at any position of any batch. A typical use is the IMFs of a decomposition,
    ``eda.monogenic(y[:n], axes=...)``.

    For a field f, with the half-sample symmetric extension at the borders (as in the method), the
    first-order Riesz transform R1 = (R_a f), the second-order one R2 = (R_a R_b f) and the
    direction n (the top eigenvector of T = R1 R1^T + R2 R2^T in 2-D and 3-D, 1 in 1-D), with
    q = n . R1:

    - ``amplitude`` = sqrt(f^2 + |R1|^2);
    - ``phase`` = atan2(q, f), in (-pi, pi];
    - ``frequency`` = (f dq/dn - q df/dn) / (2 pi amplitude^2), in cycles per sample along n
      (the classical instantaneous frequency in 1-D);
    - ``direction`` = n, with its first component of magnitude above the dtype's default tie
      tolerance (``2**-14`` float32, ``2**-30`` float64) positive; ``coherence`` =
      (mu1 - mu2) / (mu1 + mu2) from the two largest eigenvalues of T.

    The phase and the frequency are 0 where the amplitude vanishes. For a plane wave
    cos(2 pi k.x + t0) the amplitude is 1, the phase 2 pi k.x + t0, the frequency |k|, the
    direction k/|k| (up to the sign) and the coherence 1.

    Host input returns NumPy arrays and CUDA input ``eda.DeviceArray`` objects, complete on
    return, in the compute dtype. With ``check_finite=True`` a field that holds inf or NaN raises
    ``ValueError``. Raises ``TypeError`` and ``ValueError`` for arguments, ``MemoryError`` when the
    device memory does not suffice and ``eda.CudaError`` when a CUDA call fails.
    """
    check_finite = _flag(check_finite, "check_finite")
    call = _prepare(x, axes, dtype, stream)
    D = len(call.axes)
    core = _native.core()
    if call.on_device:
        args = call.device_args()
        owner, ptr, raw = core.monogenic_device(*args, check_finite, 0)
    else:
        flat, raw = core.monogenic_host(call.host_batch(), check_finite, 0)
    bad = np.flatnonzero(np.asarray(raw["nonfinite"]))
    if check_finite and bad.size:
        shown = ", ".join(str(int(i)) for i in bad[:8]) + (", ..." if bad.size > 8 else "")
        raise ValueError(
            f"eda: the input has inf or NaN at batch index {shown} ({bad.size} of "
            f"{math.prod(call.batch_shape)} field(s)); check_finite=True"
        )
    n = math.prod(call.batch_shape) * math.prod(call.spatial)
    # the planes of the native output: amplitude, phase, frequency, then in 2-D and 3-D the
    # direction (D components per sample) and the coherence
    out, offset = [], 0
    for comps in (0, 0, 0, D, 0) if D >= 2 else (0, 0, 0):
        extra = (comps,) if comps else ()
        computed = call.batch_shape + extra + call.spatial
        perm = _restore_order(call.ndim, call.batch_axes, call.axes, len(extra))
        if call.on_device:
            ptr_k = ptr + offset * np.dtype(call.target).itemsize
            out.append(_device_result(owner, ptr_k, computed, perm, call.target, args[1]))
        else:
            part = flat[offset : offset + n * max(comps, 1)].reshape(computed)
            out.append(part.transpose(perm) if perm != list(range(len(perm))) else part)
        offset += n * max(comps, 1)
    return Monogenic(*out) if D >= 2 else Monogenic(*out, None, None)


# ----------------------------------------------------------------------------------------------
# Arguments


def _max_imf(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("eda: max_imf must be an integer, not bool")
    try:
        number = operator.index(value)
    except TypeError:
        raise TypeError(f"eda: max_imf must be an integer, got {type(value).__name__}") from None
    return max(-_INT32_MAX - 1, min(number, _INT32_MAX))


def _sifts(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("eda: sifts must be an integer, not bool")
    try:
        number = operator.index(value)
    except TypeError:
        raise TypeError(f"eda: sifts must be an integer, got {type(value).__name__}") from None
    if not 1 <= number <= _INT32_MAX:
        raise ValueError(f"eda: sifts must be an integer from 1 to {_INT32_MAX}, got {number}")
    return number


def _tie_tolerance(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            f"eda: tie_tolerance must be a real number or None, got {type(value).__name__}"
        )
    try:
        number = float(value)
    except OverflowError:
        number = math.inf
    if not (math.isfinite(number) and number >= 0):
        raise ValueError(f"eda: tie_tolerance must be a finite number >= 0, got {value}")
    return number


def _flag(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"eda: {name} must be True or False, got {type(value).__name__}")
    return bool(value)


def _dtype_name(dtype: Any) -> str:
    try:
        return np.dtype(dtype).name
    except TypeError:  # a framework dtype such as torch.float32
        return str(dtype).removeprefix("torch.")


def _check_source(data: Any, src: str, on_device: bool) -> None:
    """The dtype kinds eda can cast: bool, integer and float on the host; also bfloat16 on CUDA."""
    if on_device:
        ok = src in _device.SUPPORTED_DTYPES
    else:
        ok = data.dtype.kind in "biuf"
    if not ok:
        raise TypeError(
            f"eda: arrays of dtype {src} are not supported (real bool, integer and float arrays "
            "only)"
        )


def _normalize_axes(axes: Any, ndim: int) -> tuple[int, ...]:
    if ndim < 1:
        raise ValueError("eda: expected an array with at least one dimension")
    if axes is None:
        if ndim > 3:
            raise ValueError(
                f"eda: axes=None takes every axis and needs x.ndim <= 3, got {ndim}: name the "
                "spatial axes (1 to 3) with axes="
            )
        return tuple(range(ndim))
    if isinstance(axes, (int, np.integer)) and not isinstance(axes, bool):
        axes = (axes,)
    try:
        listed = [_axis(a) for a in axes]
    except TypeError:
        raise TypeError(
            f"eda: axes must be an int, a sequence of ints or None, got {type(axes).__name__}"
        ) from None
    out = []
    for a in listed:
        if not -ndim <= a < ndim:
            raise np.exceptions.AxisError(a, ndim, "eda")
        out.append(a % ndim)
    if len(set(out)) != len(out):
        raise ValueError(f"eda: axes must be unique, got {tuple(listed)}")
    if not 1 <= len(out) <= 3:
        raise ValueError(f"eda: 1 to 3 spatial axes, got {len(out)}")
    return tuple(out)


def _axis(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("bool")
    return operator.index(value)


# ----------------------------------------------------------------------------------------------
# Layout and results


def _restore_order(
    ndim: int, batch_axes: tuple[int, ...], axes: tuple[int, ...], extra: int = 1
) -> list[int]:
    """The permutation taking a computed ``(*batch, *E, *spatial)`` array (``extra`` axes ``E``,
    such as the mode axis, then the spatial axes in the order of ``axes``) to the public layout:
    the axes of ``x`` in their order, with ``E`` immediately before ``min(axes)``."""
    nb = len(batch_axes)

    def computed_position(i: int) -> int:
        return nb + extra + axes.index(i) if i in axes else batch_axes.index(i)

    m = min(axes)
    return (
        [computed_position(f) for f in range(m)]
        + [nb + j for j in range(extra)]
        + [computed_position(f) for f in range(m, ndim)]
    )


def _orientation(call: _Call, raw: dict[str, Any], K: int, device: int | None) -> Orientation:
    """The ``Orientation`` of a decomposition from the native information: NumPy arrays
    ``direction`` (B, K, D, *spatial) and ``coherence`` (B, K, *spatial), or the device buffer
    ``orientation`` = (owner, ptr) holding both."""
    D = len(call.axes)
    shapes = (call.batch_shape + (K, D) + call.spatial, call.batch_shape + (K,) + call.spatial)
    perms = [_restore_order(call.ndim, call.batch_axes, call.axes, e) for e in (2, 1)]
    if call.on_device:
        owner, ptr = raw["orientation"]
        item = np.dtype(call.target).itemsize
        offset = math.prod(shapes[0]) * item
        direction = _device_result(owner, ptr, shapes[0], perms[0], call.target, device)
        coherence = _device_result(owner, ptr + offset, shapes[1], perms[1], call.target, device)
        return Orientation(direction, coherence)
    arrays = []
    for name, shape, perm in zip(("direction", "coherence"), shapes, perms, strict=True):
        a = np.asarray(raw[name]).reshape(shape)
        arrays.append(a.transpose(perm) if perm != list(range(len(perm))) else a)
    return Orientation(*arrays)


def _info(
    raw: dict[str, Any], batch_shape: tuple[int, ...], dtype: str, axes: tuple[int, ...]
) -> DecompositionInfo:
    def per_element(name: str, np_dtype: Any) -> np.ndarray:
        a = np.asarray(raw[name], dtype=np_dtype)
        return a.reshape(batch_shape + a.shape[1:])

    return DecompositionInfo(
        n_imfs=per_element("n_imfs", np.int32),
        error=per_element("error", np.int32),
        windows=per_element("windows", np.int32),
        spacing=per_element("spacing", np.float64),
        sifts=per_element("sifts", np.int32),
        energy=per_element("energy", np.float64),
        dtype=dtype,
        axes=axes,
    )


def _error_summary(codes: np.ndarray, indices: np.ndarray, total: int) -> str:
    parts = []
    for code in sorted({int(c) for c in codes}):
        where = indices[codes == code]
        shown = ", ".join(str(int(i)) for i in where[:8]) + (", ..." if where.size > 8 else "")
        text = _ERROR_MESSAGES.get(code, f"error code {code}")
        parts.append(f"code {code} ({text}) at batch index {shown}")
    return f"eda: the decomposition failed for {indices.size} of {total} element(s): " + "; ".join(
        parts
    )


def _device_result(
    owner: Any, ptr: int, computed: tuple[int, ...], out_final: list[int], dtype: str, device: int
) -> _device.DeviceArray:
    """The ``DeviceArray`` over the C-contiguous computed rows, permuted to the public layout."""
    item = np.dtype(dtype).itemsize
    strides, acc = [], item
    for n in reversed(computed):
        strides.append(acc)
        acc *= max(n, 1)
    strides.reverse()
    return _device.DeviceArray(
        owner,
        ptr,
        tuple(computed[i] for i in out_final),
        tuple(strides[i] for i in out_final),
        np.dtype(dtype),
        device,
    )
