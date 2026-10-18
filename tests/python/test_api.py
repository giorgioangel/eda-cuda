# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
"""The public API without a GPU: argument rules, errors, imports, the layout permutation and the
naming rule. Tests that call into the extension and must not see a GPU run in a subprocess with
CUDA_VISIBLE_DEVICES set to the empty string, so that the outcome does not depend on the machine."""

from __future__ import annotations

import codecs
import os
import pickle
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

import eda
from cases import RECIPES
from eda import _api, _device, _native

REPO = Path(__file__).resolve().parents[2]


def _extension() -> object | None:
    try:
        return _native.core()
    except ImportError:
        return None


needs_extension = pytest.mark.skipif(_extension() is None, reason="eda._core is not built")


def run_without_gpu(code: str, **env: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter that sees no CUDA device."""
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": os.pathsep.join(p for p in sys.path if p),
        **env,
    }
    return subprocess.run(
        [sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=120
    )


# ----------------------------------------------------------------------------------------------
# Argument rules (all checked before any GPU work)

X = np.zeros((4, 16), np.float32)

ARGUMENT_ERRORS = [
    (dict(max_imf=1.5), TypeError, "eda: max_imf must be an integer, got float"),
    (dict(max_imf=True), TypeError, "eda: max_imf must be an integer, not bool"),
    (dict(max_imf="2"), TypeError, "max_imf must be an integer, got str"),
    (dict(max_imf=None), TypeError, "max_imf must be an integer, got NoneType"),
    (dict(tie_tolerance="a"), TypeError, "tie_tolerance must be a real number or None, got str"),
    (dict(tie_tolerance=True), TypeError, "tie_tolerance must be a real number or None"),
    (dict(tie_tolerance=[1.0]), TypeError, "tie_tolerance must be a real number or None"),
    (dict(tie_tolerance=-1.0), ValueError, "tie_tolerance must be a finite number >= 0, got -1.0"),
    (dict(tie_tolerance=float("nan")), ValueError, "tie_tolerance must be a finite number >= 0"),
    (dict(tie_tolerance=float("inf")), ValueError, "tie_tolerance must be a finite number >= 0"),
    (dict(tie_tolerance=10**400), ValueError, "tie_tolerance must be a finite number >= 0"),
    (dict(sifts=0), ValueError, "eda: sifts must be an integer from 1 to 2147483647, got 0"),
    (dict(sifts=-2), ValueError, "sifts must be an integer from 1 to 2147483647, got -2"),
    (dict(sifts=2**31), ValueError, "sifts must be an integer from 1 to 2147483647"),
    (dict(sifts=True), TypeError, "eda: sifts must be an integer, not bool"),
    (dict(sifts=2.5), TypeError, "eda: sifts must be an integer, got float"),
    (dict(sifts="3"), TypeError, "sifts must be an integer, got str"),
    (dict(sifts=None), TypeError, "sifts must be an integer, got NoneType"),
    (dict(check_finite=1), TypeError, "eda: check_finite must be True or False, got int"),
    (dict(check_finite=None), TypeError, "check_finite must be True or False, got NoneType"),
    (dict(return_info=None), TypeError, "eda: return_info must be True or False, got NoneType"),
    (dict(return_info=0), TypeError, "return_info must be True or False, got int"),
    (dict(errors="x"), ValueError, "eda: errors='x' is not one of 'raise', 'status'"),
    (dict(errors=None), ValueError, "errors=None is not one of 'raise', 'status'"),
    (dict(stream=1.5), TypeError, "eda: stream must be an integer CUDA stream handle"),
    (dict(stream=True), TypeError, "stream must be an integer CUDA stream handle"),
    (dict(stream="a"), TypeError, "stream must be an integer CUDA stream handle"),
    (dict(stream=-1), ValueError, "eda: stream handle -1 is negative"),
    (dict(dtype="float16"), TypeError, "eda: dtype= must be float32 or float64, got float16"),
    (dict(dtype="int32"), TypeError, "dtype= must be float32 or float64, got int32"),
    (dict(dtype="no-such-dtype"), TypeError, "dtype= must be float32 or float64"),
    (dict(axes="ab"), TypeError, "eda: axes must be an int, a sequence of ints or None, got str"),
    (dict(axes=1.5), TypeError, "axes must be an int, a sequence of ints or None, got float"),
    (dict(axes=True), TypeError, "axes must be an int, a sequence of ints or None, got bool"),
    (dict(axes=(0, 0)), ValueError, "eda: axes must be unique, got (0, 0)"),
    (dict(axes=(0, -2)), ValueError, "axes must be unique, got (0, -2)"),
    (dict(axes=()), ValueError, "eda: 1 to 3 spatial axes, got 0"),
    (dict(axes=5), ValueError, "eda: axis 5 is out of bounds for array of dimension 2"),
    (dict(axes=(0, -3)), ValueError, "axis -3 is out of bounds for array of dimension 2"),
]


@pytest.mark.parametrize(("kwargs", "error", "fragment"), ARGUMENT_ERRORS)
def test_argument_errors(kwargs, error, fragment):
    with pytest.raises(error, match=re.escape(fragment)):
        eda.decompose(X, **kwargs)


def test_axis_errors_are_axis_errors():
    with pytest.raises(np.exceptions.AxisError):
        eda.decompose(X, axes=2)


INPUT_ERRORS = [
    (
        np.zeros((4, 16), np.int32),
        {},
        TypeError,
        "expected a float32 or float64 array, got dtype int32",
    ),
    (np.zeros((4, 16), np.float16), {}, TypeError, "got dtype float16 (pass dtype='float32'"),
    (np.zeros((4, 16), np.bool_), {}, TypeError, "got dtype bool"),
    (
        np.zeros((4, 16), np.complex64),
        {},
        TypeError,
        "eda: arrays of dtype complex64 are not supported",
    ),
    (
        np.zeros((4, 16), np.complex64),
        {"dtype": "float32"},
        TypeError,
        "dtype complex64 are not supported",
    ),
    (np.array([["1.5", "2.5"]]), {"dtype": "float32"}, TypeError, "are not supported"),
    (
        np.array([[1, 2]], dtype=object),
        {"dtype": "float64"},
        TypeError,
        "dtype object are not supported",
    ),
    (
        np.zeros((), np.float32),
        {},
        ValueError,
        "eda: expected an array with at least one dimension",
    ),
    (
        np.zeros((2, 2, 2, 2), np.float32),
        {},
        ValueError,
        "axes=None takes every axis and needs x.ndim <= 3, got 4",
    ),
    (
        np.zeros((2, 2, 2, 2), np.float32),
        {"axes": (0, 1, 2, 3)},
        ValueError,
        "eda: 1 to 3 spatial axes, got 4",
    ),
]


@pytest.mark.parametrize(("x", "kwargs", "error", "fragment"), INPUT_ERRORS)
def test_input_errors(x, kwargs, error, fragment):
    with pytest.raises(error, match=re.escape(fragment)):
        eda.decompose(x, **kwargs)


def test_check_order_scalars_before_input():
    # a bad scalar argument is reported even when the input is also wrong
    with pytest.raises(TypeError, match="check_finite"):
        eda.decompose(np.zeros((2, 2), np.complex64), check_finite=3)
    # the input's dtype is reported before the axes
    with pytest.raises(TypeError, match="expected a float32"):
        eda.decompose(np.zeros((4, 4), np.int8), axes=9)


@pytest.mark.parametrize("dtype", ["float32", "float64", np.float32, "<f8", float])
def test_float_dtype_names_are_accepted(dtype):
    assert _api._dtype_name(dtype) in ("float32", "float64")


def test_framework_dtype_names():
    class FakeTorchDtype:
        def __str__(self) -> str:
            return "torch.float32"

    assert _api._dtype_name(FakeTorchDtype()) == "float32"


@needs_extension
@pytest.mark.parametrize(
    ("x", "kwargs", "fragment"),
    [
        (np.zeros((4, 1), np.float32), {"axes": 1}, "needs at least 2 samples"),
        (
            np.zeros((2, 9000), np.float32),
            {"axes": 1},
            "axis extent 9000 exceeds the limit 8192 for float32",
        ),
        (
            np.zeros((2, 5000), np.float64),
            {"axes": 1},
            "axis extent 5000 exceeds the limit 4096 for float64",
        ),
        (
            np.zeros((2, 5000), np.float32),
            {"axes": 1, "dtype": "float64"},
            "limit 4096 for float64",
        ),
        (np.zeros((0, 1), np.float32), {"axes": 1}, "needs at least 2 samples"),
    ],
)
def test_native_validation(x, kwargs, fragment):
    with pytest.raises(ValueError, match=fragment):
        eda.decompose(x, **kwargs)


@needs_extension
def test_native_sifts_validation():
    core = _native.core()
    with pytest.raises(ValueError, match="eda: sifts must be >= 1"):
        core.decompose_host(np.zeros((1, 64), np.float32), -1, None, 0, True, False, 0)


# ----------------------------------------------------------------------------------------------
# Without a GPU


@needs_extension
def test_without_a_gpu():
    code = """
import numpy as np, eda
assert eda.device_count() == 0
try:
    eda.device_info(0)
except ValueError as exc:
    print("device_info:", exc)
else:
    raise SystemExit("device_info(0) did not raise")
info = eda.build_info()
assert {"version", "cuda_toolkit", "cuda_runtime", "cuda_driver", "cuda_architectures", "sass",
        "ptx", "build_type", "cxx_compiler", "cuda_compiler", "cccl_version",
        "cufft_version"} <= set(info), sorted(info)
assert info["version"] == eda.__version__
try:
    eda.decompose(np.zeros((4, 16), np.float32), axes=1)
except eda.CudaError as exc:
    assert isinstance(exc, RuntimeError)
    print("decompose:", exc)
else:
    raise SystemExit("decompose did not raise")
# an empty batch needs no device
y, info = eda.decompose(np.zeros((0, 16), np.float32), axes=1, return_info=True)
assert y.shape == (0, 1, 16) and y.dtype == np.float32
assert info.n_imfs.shape == (0,) and info.windows.shape == (0, 0)
"""
    done = run_without_gpu(code)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "found no CUDA device" in done.stdout


@needs_extension
@pytest.mark.parametrize("name", ["EDA_FOO", "EDA_GRAPHS", "EDA_ARENA_DEBUG"])
def test_unknown_environment_variable(name):
    code = (
        "import numpy as np, eda\n"
        "for batch in (4, 0):  # an empty batch is checked too\n"
        "    try:\n"
        "        eda.decompose(np.zeros((batch, 16), np.float32), axes=1)\n"
        "    except ValueError as exc:\n"
        f"        assert 'unknown environment variable {name}' in str(exc), exc\n"
        "    else:\n"
        "        raise SystemExit(f'no error for batch {batch}')\n"
    )
    done = run_without_gpu(code, **{name: "1"})
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.parametrize(
    "device, error, fragment",
    [
        (True, TypeError, "not bool"),
        ("0", TypeError, "got str"),
        (1.0, TypeError, "got float"),
        (2**40, ValueError, "is not visible"),
    ],
)
def test_device_info_argument_errors(device, error, fragment):
    with pytest.raises(error, match=f"^eda: .*{fragment}"):
        eda.device_info(device)


def test_missing_extension_message(monkeypatch):
    monkeypatch.setattr(_native, "_module", None)
    monkeypatch.delattr(eda, "_core", raising=False)
    monkeypatch.setitem(sys.modules, "eda._core", None)  # the import then fails
    message = r"^eda: the eda-cuda(-cu12)? CUDA extension \(eda._core\)"
    with pytest.raises(ImportError, match=message):
        _native.core()


def test_import_needs_neither_gpu_nor_extension():
    code = """
import sys
import eda
assert "eda._core" not in sys.modules, "the extension loaded at import"
assert eda.__all__ == [
    "CudaError", "DecompositionError", "DecompositionInfo", "DeviceArray", "Monogenic",
    "Orientation", "__version__", "build_info", "decompose", "device_count", "device_info",
    "monogenic"]
"""
    done = run_without_gpu(code)
    assert done.returncode == 0, done.stdout + done.stderr


def test_public_names():
    assert sorted(eda.__all__) == sorted(
        [
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
    )
    for name in eda.__all__:
        assert hasattr(eda, name)
    assert (Path(eda.__file__).parent / "py.typed").exists()


def test_version_matches_cmake():
    text = (REPO / "CMakeLists.txt").read_text()
    match = re.search(r"project\(eda VERSION ([0-9.]+)", text)
    if match is None:  # an installed package without the source tree
        pytest.skip("CMakeLists.txt is not next to the tests")
    assert eda.__version__ == match.group(1)


@needs_extension
def test_version_matches_build_info():
    assert eda.build_info()["version"] == eda.__version__


# ----------------------------------------------------------------------------------------------
# Exceptions


def test_exception_types():
    assert issubclass(eda.DecompositionError, ValueError)
    assert issubclass(eda.CudaError, RuntimeError)


def test_cuda_error_attributes_and_pickle():
    error = eda.CudaError("eda: boom", 2, "cudaErrorMemoryAllocation")
    assert (error.code, error.name) == (2, "cudaErrorMemoryAllocation")
    again = pickle.loads(pickle.dumps(error))
    assert type(again) is eda.CudaError
    assert (str(again), again.code, again.name) == ("eda: boom", 2, "cudaErrorMemoryAllocation")
    bare = eda.CudaError("eda: x")
    assert (bare.code, bare.name) == (None, None)


def test_decomposition_error_attributes_and_pickle():
    info = _api._info(
        {
            "n_imfs": np.zeros(2, np.int32),
            "error": np.array([0, 1], np.int32),
            "windows": np.zeros((2, 0), np.int32),
            "sifts": np.zeros((2, 0), np.int32),
            "spacing": np.zeros((2, 0)),
            "energy": np.zeros((2, 0)),
        },
        (2,),
        "float32",
        (1,),
    )
    error = eda.DecompositionError("eda: failed", np.array([1]), np.array([1], np.int32), info)
    assert isinstance(error, ValueError)
    again = pickle.loads(pickle.dumps(error))
    assert type(again) is eda.DecompositionError
    assert str(again) == "eda: failed"
    assert again.indices.tolist() == [1] and again.codes.tolist() == [1]
    assert again.info.error.tolist() == [0, 1] and again.info.axes == (1,)


def test_error_summary_text():
    codes = np.array([1] * 10 + [2], np.int32)
    indices = np.arange(11)
    text = _api._error_summary(codes, indices, 40)
    assert text.startswith("eda: the decomposition failed for 11 of 40 element(s): code 1 (")
    assert "at batch index 0, 1, 2, 3, 4, 5, 6, 7, ...; code 2 (" in text
    assert text.endswith("at batch index 10")


def test_info_is_frozen_and_identity_compared():
    info = _api._info(
        {
            "n_imfs": np.zeros(1, np.int32),
            "error": np.zeros(1, np.int32),
            "windows": np.zeros((1, 0), np.int32),
            "sifts": np.zeros((1, 0), np.int32),
            "spacing": np.zeros((1, 0)),
            "energy": np.zeros((1, 0)),
        },
        (),
        "float64",
        (0,),
    )
    assert info.n_imfs.shape == () and info.windows.shape == (0,)
    with pytest.raises(AttributeError):
        info.dtype = "float32"  # type: ignore[misc]


# ----------------------------------------------------------------------------------------------
# The layout permutation


def _layouts():
    seen = set()
    for r in RECIPES:
        key = (len(r.shape), r.axes)
        if key not in seen:
            seen.add(key)
            yield r.shape, r.axes
    yield (3, 4, 5, 6), (3, 0)
    yield (3, 4, 5, 6), (-1, 1, 0)
    yield (3, 4, 5), (2,)


@pytest.mark.parametrize(("shape", "axes"), list(_layouts()))
def test_restore_order_inserts_the_mode_axis_before_the_first_spatial_axis(shape, axes):
    ndim = len(shape)
    norm = _api._normalize_axes(axes, ndim)
    batch_axes = tuple(i for i in range(ndim) if i not in norm)
    order = _api._restore_order(ndim, batch_axes, norm)
    batch_shape = tuple(shape[i] for i in batch_axes)
    spatial = tuple(shape[a] for a in norm)
    modes = 3
    computed = np.arange(int(np.prod(batch_shape + (modes,) + spatial))).reshape(
        batch_shape + (modes,) + spatial
    )
    y = computed.transpose(order)
    m = min(norm)
    assert y.shape == shape[:m] + (modes,) + shape[m:]
    # every entry of y comes from the element at the same position in x, spatial axes in the
    # order of axes
    grid = np.indices(y.shape).reshape(ndim + 1, -1)
    mode = grid[m]
    x_index = np.delete(grid, m, axis=0)
    source = tuple(x_index[i] for i in batch_axes) + (mode,) + tuple(x_index[a] for a in norm)
    assert np.array_equal(y.reshape(-1), computed[source])


def test_axes_forms():
    assert _api._normalize_axes(None, 3) == (0, 1, 2)
    assert _api._normalize_axes(-1, 3) == (2,)
    assert _api._normalize_axes(np.int64(1), 3) == (1,)
    assert _api._normalize_axes([2, 0], 3) == (2, 0)
    assert _api._normalize_axes((np.int32(-3),), 3) == (0,)
    assert _api._normalize_axes(range(2), 3) == (0, 1)


# ----------------------------------------------------------------------------------------------
# Streams and CUDA array interface fields (no GPU needed)


class _Handle:
    def __init__(self, handle):
        self.handle = handle


def test_stream_handle_forms():
    class Protocol(_Handle):
        def __cuda_stream__(self):
            return (0, self.handle)

    class Torch(_Handle):
        @property
        def cuda_stream(self):
            return self.handle

    class Cupy(_Handle):
        @property
        def ptr(self):
            return self.handle

    assert _device.stream_handle(5) == 5
    assert _device.stream_handle(np.int64(7)) == 7
    assert _device.stream_handle(0) == 1
    assert _device.stream_handle(2) == 2
    assert _device.stream_handle(Protocol(9)) == 9
    assert _device.stream_handle(Torch(11)) == 11
    assert _device.stream_handle(Cupy(13)) == 13
    assert _device.stream_handle(Cupy(0)) == 1
    with pytest.raises(TypeError):
        _device.stream_handle(np.bool_(True))

    class BadProtocol:
        def __cuda_stream__(self):
            return 5

    with pytest.raises(TypeError, match="stream must be"):
        _device.stream_handle(BadProtocol())
    for bad in (Protocol(1.5), Protocol(True), Torch("abc"), Cupy(None)):
        with pytest.raises(TypeError, match=f"^eda: stream must be .*got {type(bad).__name__}$"):
            _device.stream_handle(bad)


@pytest.mark.parametrize(
    ("cai", "explicit", "expected"),
    [
        ({"version": 3, "stream": 7}, None, (7,)),
        ({"version": 3, "stream": None}, None, ()),
        ({"version": 3}, None, ()),
        ({"version": 3, "stream": 7}, 9, (7, 9)),
        ({"version": 3, "stream": None}, 9, (9,)),
        ({"version": 3, "stream": 1}, None, (1,)),
        ({"version": 3, "stream": 2}, None, (2,)),
        ({"version": 2}, None, (1,)),
        ({"version": 2, "stream": 5}, None, (1,)),
        ({"version": 2}, 9, (9,)),
        ({}, None, (1,)),
    ],
)
def test_interface_fences(cai, explicit, expected):
    version = cai.get("version", 0)
    assert _device._interface_fences(cai, version, explicit) == expected


def test_interface_stream_zero_is_refused():
    with pytest.raises(ValueError, match="stream 0 is not a valid stream"):
        _device._interface_fences({"version": 3, "stream": 0}, 3, None)


@pytest.mark.parametrize("stream", [1.5, "7", True, (7,)])
def test_interface_stream_must_be_an_integer(stream):
    with pytest.raises(ValueError, match="^eda: __cuda_array_interface__ stream .* not an integer"):
        _device._interface_fences({"version": 3, "stream": stream}, 3, None)


@needs_extension
def test_interface_input_is_checked_before_cuda():
    # the dtype and axes of a CUDA array interface input are refused before any CUDA call
    code = """
import numpy as np, eda
class Interface:
    def __init__(self, typestr):
        self.__cuda_array_interface__ = {
            "shape": (4, 8), "typestr": typestr, "data": (0x7000, False), "version": 3}
for typestr, kwargs, error in [
    ("<c8", {}, TypeError),
    ("<i4", {}, TypeError),
    ("<f4", {"dtype": "float16"}, TypeError),
    ("<f4", {"axes": 7}, np.exceptions.AxisError),
]:
    try:
        eda.decompose(Interface(typestr), **kwargs)
    except error as exc:
        assert str(exc).startswith("eda"), exc
    else:
        raise SystemExit(f"no error for {typestr} {kwargs}")
"""
    done = run_without_gpu(code)
    assert done.returncode == 0, done.stdout + done.stderr


class _Interface:
    def __init__(self, **cai):
        base = {"shape": (4, 8), "typestr": "<f4", "data": (0x7000, False), "version": 3}
        self.__cuda_array_interface__ = {**base, **cai}


@pytest.mark.parametrize(
    ("cai", "fragment"),
    [
        ({"mask": object()}, "has a mask"),
        ({"typestr": ">f4"}, "big-endian"),
        ({"typestr": "nonsense"}, "typestr 'nonsense' is not valid"),
        ({"typestr": 5}, "typestr must be a string"),
        ({"strides": (30, 4)}, "not multiples of the item size 4"),
        ({"strides": (32,)}, "differ in length"),
        ({"data": None}, "malformed"),
        ({"shape": ("a", 2)}, "malformed"),
    ],
)
def test_interface_field_errors(cai, fragment):
    with pytest.raises(ValueError, match=fragment):
        _device.intake(_Interface(**cai), None)


def test_interface_without_required_entries():
    x = _Interface()
    del x.__cuda_array_interface__["typestr"]
    with pytest.raises(ValueError, match="no 'typestr' entry"):
        _device.intake(x, None)


class _DLPackHost:
    """An object that exports a host array through DLPack."""

    def __init__(self, array, kind=1):
        self.array, self.kind = array, kind

    def __dlpack__(self, **kwargs):
        return self.array.__dlpack__(**kwargs)

    def __dlpack_device__(self):
        return (self.kind, 0)


def test_intake_of_host_objects():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert _device.intake(a, None) is a
    assert np.array_equal(_device.intake(_DLPackHost(a), None), a)
    assert np.array_equal(_device.intake([[1.0, 2.0]], None), [[1.0, 2.0]])


def test_intake_of_other_device_types():
    with pytest.raises(TypeError, match="DLPack device type 7 are not supported"):
        _device.intake(_DLPackHost(np.zeros(3), kind=7), None)


def test_requires_grad_is_detached():
    a = np.zeros(3)

    class Tensor(_DLPackHost):
        requires_grad = True

        def detach(self):
            return _DLPackHost(self.array)

        def __dlpack__(self, **kwargs):
            raise RuntimeError("Can't export tensors that require gradient")

    out = _device.intake(Tensor(a), None)
    assert isinstance(out, np.ndarray)


def test_host_dlpack_failure_is_a_type_error():
    class Broken(_DLPackHost):
        def __dlpack__(self, **kwargs):
            raise BufferError("unsupported dtype")

    with pytest.raises(TypeError, match="cannot be read through DLPack"):
        _device.intake(Broken(np.zeros(3)), None)


# ----------------------------------------------------------------------------------------------
# DeviceArray (no device memory behind these)


def test_device_array_description():
    y = _device.DeviceArray(None, 0x1000, (2, 3, 4), (48, 16, 4), np.dtype("float32"), 1)
    assert (y.ndim, y.size, y.nbytes) == (3, 24, 96)
    cai = y.__cuda_array_interface__
    assert cai == {
        "shape": (2, 3, 4),
        "typestr": "<f4",
        "data": (0x1000, False),
        "strides": None,
        "version": 3,
        "stream": None,
    }
    assert y.__dlpack_device__() == (2, 1)
    assert repr(y) == "DeviceArray(shape=(2, 3, 4), dtype=float32, device=1)"
    with pytest.raises(TypeError, match="CUDA memory"):
        np.asarray(y)


def test_device_array_strided_layout():
    y = _device.DeviceArray(None, 0x1000, (3, 2, 4), (32, 96, 8), np.dtype("float64"), 0)
    cai = y.__cuda_array_interface__
    assert cai["strides"] == (32, 96, 8) and cai["typestr"] == "<f8"
    one = _device.DeviceArray(None, 0, (1, 5, 1), (40, 8, 8), np.dtype("float64"), 0)
    assert one.__cuda_array_interface__["strides"] is None  # extent-1 axes never matter
    empty = _device.DeviceArray(None, 0, (2, 0, 1, 64), (256, 256, 256, 4), np.dtype("float32"), 0)
    assert empty.__cuda_array_interface__["strides"] is None  # an empty array is contiguous


def test_device_array_dlpack_refusals():
    y = _device.DeviceArray(None, 0x1000, (2, 3), (12, 4), np.dtype("float32"), 0)
    with pytest.raises(BufferError):
        y.__dlpack__(copy=True)
    with pytest.raises(BufferError):
        y.__dlpack__(dl_device=(1, 0))


def test_result_strides_follow_the_permutation():
    # computed (B=2, P=3, N=5) float32 with out_final moving the mode axis last
    y = _api._device_result(None, 0, (2, 3, 5), [0, 2, 1], "float32", 0)
    assert y.shape == (2, 5, 3) and y.strides == (60, 4, 20)
    empty = _api._device_result(None, 0, (0, 1, 5), [0, 1, 2], "float32", 0)
    assert empty.shape == (0, 1, 5) and empty.size == 0


# ----------------------------------------------------------------------------------------------
# The naming rule: no name of the predecessor anywhere in the tracked tree


def _unrot(text: str) -> str:
    return codecs.decode(text, "rot13")


def _patterns() -> list[re.Pattern[str]]:
    """The words the tree must not contain, stored rot13-encoded so that this file passes its own
    check. Whole-word patterns do not match inside longer words ('monogenic', 'monotone')."""
    anywhere = ["ynfp", "rzq_aq", "fgrre", "qbpf/qrfvta", "erfrnepu/", "pynhqr", "naguebcvp"]
    bounded = ["oaq", "zbab", "bfs", "fbfs"]
    patterns = [re.escape(_unrot(w)) for w in anywhere]
    patterns.append(_unrot("rzq") + "[_-]?" + _unrot("tch"))
    patterns.append("(?<![a-z])(?:" + "|".join(_unrot(w) for w in bounded) + ")(?![a-z])")
    patterns.append("(?<![a-z0-9])" + _unrot("aq_"))
    patterns.append(_unrot("jc-") + "[abc]")
    return [re.compile(p, re.IGNORECASE | re.ASCII) for p in patterns]


def _tracked_files() -> list[str]:
    try:
        done = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "-z"], capture_output=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git or the repository is not available")
    return [p for p in done.stdout.decode().split("\0") if p]


def test_no_legacy_names():
    patterns = _patterns()
    hits: list[str] = []

    def scan(where: str, text: str) -> None:
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                hits.append(f"{where}: {match.group(0)!r}")

    for name in _tracked_files():
        scan(f"path {name}", name)
        path = REPO / name
        if name.endswith(".npz"):
            with zipfile.ZipFile(path) as archive:  # member names only; the data is binary
                for member in archive.namelist():
                    scan(f"{name} member", member)
            continue
        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            continue
        scan(name, text)
    assert not hits, "legacy names found:\n" + "\n".join(hits[:40])
