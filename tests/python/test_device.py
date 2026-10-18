# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
"""CUDA input and output through CuPy and PyTorch (each skipped when not installed): DLPack and the
CUDA array interface, every supported dtype, strided and non-mergeable layouts, streams, result
lifetime and memory. Device results equal the host call on the same values bit for bit."""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

import cases as C
import device_memory
import eda
import oracle
from eda import _api, _native

pytestmark = pytest.mark.gpu

WANT = oracle.load() if oracle.MANIFEST.exists() else {}
needs_manifest = pytest.mark.skipif(not WANT, reason="tests/data/parity_v2.json is not present")
SANITIZED = os.environ.get("EDA_SANITIZE") not in (None, "", "0")


@pytest.fixture(scope="module")
def cp():
    cupy = pytest.importorskip("cupy")
    yield cupy
    cupy.get_default_memory_pool().free_all_blocks()


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def field(shape, seed=3, dtype="float64"):
    rng = np.random.default_rng(seed)
    grids = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape[1:]], indexing="ij")
    out = np.empty(shape)
    for b in range(shape[0]):
        wave = sum(np.sin((0.31 + 0.07 * b + 0.05 * k) * g) for k, g in enumerate(grids))
        out[b] = wave + 0.1 * rng.standard_normal(shape[1:])
    return out.astype(dtype)


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


class CaiOnly:
    """Exposes only the CUDA array interface of a CuPy array (the interface path)."""

    def __init__(self, array, *, stream="keep", readonly=False, version=3):
        cai = dict(array.__cuda_array_interface__)
        cai.pop("descr", None)
        if stream != "keep":
            cai["stream"] = stream
        cai["data"] = (cai["data"][0], readonly)
        cai["version"] = version
        if version < 3:
            cai.pop("stream", None)
        self.__cuda_array_interface__ = cai
        self._array = array


def cupy_read(cp, y):
    assert isinstance(y, eda.DeviceArray)
    return cp.asnumpy(cp.asarray(y))


def torch_read(torch, y):
    return torch.from_dlpack(y).cpu().numpy()


# ----------------------------------------------------------------------------------------------
# The oracle through device input

DEVICE_CASES = [
    "d1-tones-512.f32",
    "d1-chirp-2048.f64",
    "d1-batch4-512.f32",
    "d1-chunk1-512.f64",
    "d1-nan-status.f32",
    "d1-inf-nocheck.f64",
    "d1-twobatch-37.f64",
    "d2-gabor-40x36.f32",
    "d2-batch3-40x36.f64",
    "d2-chunk1-24x31.f32",
    "d2-batchmid-24x31.f32",
    "d2-axes-reversed.f64",
    "d2-twobatch-40x36.f32",
    "d2-nan-status.f64",
    "d3-tones-16x13x11.f32",
    "d3-chunk2-16x13x11.f64",
    "d3-batchmid-16x13x11.f32",
    "d3-batch3-20x18x16.f64",
]


def observe_device(case, x, y_host, info, spy):
    """The manifest fields of a device result (y read back to the host as a NumPy array)."""
    kw = case.kwargs()
    ndim = x.ndim
    axes = _api._normalize_axes(kw["axes"], ndim)
    batch_axes = tuple(i for i in range(ndim) if i not in axes)
    out_final = _api._restore_order(ndim, batch_axes, axes)
    computed = np.ascontiguousarray(y_host.transpose(np.argsort(out_final)))
    batch = int(np.prod([x.shape[i] for i in batch_axes]))
    rows = computed.reshape((batch,) + computed.shape[len(batch_axes) :])
    return oracle.observe(x, y_host, info, rows, spy.chunk, kw["axes"])


def check_device_case(case_id, monkeypatch, make_input, read):
    case = next(c for c in C.all_cases() if c.id == case_id)
    x = case.input()
    kw = case.kwargs()
    spy = oracle.Spy(monkeypatch, _native.core(), "decompose_device")
    spy.max_chunk = case.recipe.max_chunk or 0
    y, info = eda.decompose(make_input(x), **kw, return_info=True)
    assert isinstance(y, eda.DeviceArray) and y.dtype == np.dtype(case.dtype)
    got = observe_device(case, x, read(y), info, spy)
    want = WANT[case_id]
    if not case.recipe.max_chunk:  # an in-place read has no staging slot: its chunking may differ
        got["chunks"] = want["chunks"]
    assert oracle.differences(got, want) == []


@needs_manifest
@pytest.mark.parametrize("case_id", DEVICE_CASES)
def test_oracle_through_cupy_dlpack(cp, monkeypatch, case_id):
    check_device_case(case_id, monkeypatch, cp.asarray, lambda y: cupy_read(cp, y))


@needs_manifest
@pytest.mark.parametrize("case_id", DEVICE_CASES)
def test_oracle_through_the_cuda_array_interface(cp, monkeypatch, case_id):
    check_device_case(
        case_id, monkeypatch, lambda x: CaiOnly(cp.asarray(x)), lambda y: cupy_read(cp, y)
    )


@needs_manifest
@pytest.mark.parametrize("case_id", DEVICE_CASES)
def test_oracle_through_torch(torch, monkeypatch, case_id):
    check_device_case(
        case_id,
        monkeypatch,
        lambda x: torch.as_tensor(x).cuda(),
        lambda y: torch_read(torch, y),
    )


def test_result_is_readable_by_every_consumer(cp, torch):
    x = field((3, 200), dtype="float32")
    expected = eda.decompose(x, axes=1)
    y = eda.decompose(cp.asarray(x), axes=1)
    assert same(cp.asnumpy(cp.asarray(y)), expected)
    assert same(cp.asnumpy(cp.from_dlpack(y)), expected)
    assert same(torch.from_dlpack(y).cpu().numpy(), expected)
    assert same(torch.as_tensor(y, device="cuda").cpu().numpy(), expected)
    assert (y.device, y.shape, y.dtype) == (0, expected.shape, np.dtype("float32"))
    assert cp.asarray(y).strides == tuple(y.strides)


def test_result_layouts_equal_the_host_layout(cp):
    x = field((24, 3, 31))
    for axes in ((0, 2), (2, 0)):
        expected = eda.decompose(x, axes=axes)
        y = eda.decompose(cp.asarray(x), axes=axes)
        assert y.shape == expected.shape
        assert y.strides == expected.strides
        assert same(cupy_read(cp, y), expected)


def test_dlpack_capsule_names(cp):
    y = eda.decompose(cp.asarray(field((2, 100))), axes=1)
    assert "dltensor_versioned" in repr(y.__dlpack__(max_version=(1, 1)))
    assert "dltensor_versioned" not in repr(y.__dlpack__())
    assert y.__dlpack_device__() == (2, 0)
    with pytest.raises(BufferError):
        y.__dlpack__(copy=True)


def test_a_device_array_can_be_decomposed_again(cp):
    x = field((2, 300), dtype="float32")
    first = eda.decompose(cp.asarray(x), axes=1)
    again = eda.decompose(first, axes=2)  # (B, P, N): decompose along the samples of every row
    host = cupy_read(cp, first)
    assert same(cupy_read(cp, again), eda.decompose(host, axes=2))


def test_return_info_for_device_input(cp):
    x = cp.asarray(field((3, 150), dtype="float32"))
    y, info = eda.decompose(x, axes=1, return_info=True)
    assert info.n_imfs.shape == (3,) and info.dtype == "float32"
    assert y.shape == (3, int(info.n_imfs.max()) + 1, 150)


# ----------------------------------------------------------------------------------------------
# dtypes and layouts that are converted on the GPU


# 64-bit integers whose float64 rounding lands on a float32 midpoint: a cast through float64
# rounds them to float32 differently from the direct cast NumPy does
MIDPOINTS_I64 = [2**62 + 2**38 + 1, -(2**62 + 2**38 + 1)]
MIDPOINTS_U64 = [2**63 + 2**39 + 1, 2**62 + 2**38 + 1]


def dtype_values():
    base = field((4, 400), 2)
    values = {
        "bool": base > 0.2,
        "int8": np.round(base * 40).astype(np.int8),
        "int16": np.round(base * 9000).astype(np.int16),
        "int32": np.round(base * 6e8).astype(np.int32),
        "int64": np.round(base * 3e18).astype(np.int64),
        "uint8": np.round(base * 40 + 128).astype(np.uint8),
        "uint16": np.round(base * 9000 + 30000).astype(np.uint16),
        "uint32": np.round(base * 9e8 + 2.1e9).astype(np.uint32),
        "uint64": np.round(base * 2e18 + 4.5e18).astype(np.uint64) * np.uint64(2),
        "float16": base.astype(np.float16),
        "float32": (base * 1.5).astype(np.float32),
        "float64": base * 1.5,
    }
    values["int64"][:, 100:102] = MIDPOINTS_I64
    values["uint64"][:, 100:102] = MIDPOINTS_U64
    return values


def test_midpoints_need_one_rounding():
    for values, dtype in ((MIDPOINTS_I64, np.int64), (MIDPOINTS_U64, np.uint64)):
        a = np.array(values, dtype)
        assert not np.array_equal(a.astype(np.float32), a.astype(np.float64).astype(np.float32))


@pytest.mark.parametrize("target", ["float32", "float64"])
@pytest.mark.parametrize("source", list(dtype_values()))
def test_every_dtype_equals_the_host_call(cp, source, target):
    x = dtype_values()[source]
    expected = eda.decompose(x, axes=1, dtype=target)
    for make in (cp.asarray, lambda a: CaiOnly(cp.asarray(a))):
        y = eda.decompose(make(x), axes=1, dtype=target)
        assert same(cupy_read(cp, y), expected), source
    if source not in ("float32", "float64"):
        with pytest.raises(TypeError, match=f"got dtype {source}"):
            eda.decompose(cp.asarray(x), axes=1)
    elif source != target:
        assert eda.decompose(cp.asarray(x), axes=1).dtype == np.dtype(source)


def test_dtype_argument_changes_the_compute_dtype_of_float_input(cp):
    x = field((2, 200), dtype="float32")
    y, info = eda.decompose(cp.asarray(x), axes=1, dtype="float64", return_info=True)
    assert y.dtype == np.dtype("float64") and info.dtype == "float64"
    assert same(cupy_read(cp, y), eda.decompose(x.astype("float64"), axes=1))


@pytest.mark.parametrize("source", ["float16", "bfloat16", "int64", "uint8", "int32"])
@pytest.mark.parametrize("target", ["float32", "float64"])
def test_torch_dtypes_equal_the_host_call(torch, source, target):
    base = field((3, 300), 3)
    scale = 1000 if source in ("int64", "int32") else 40 if source == "uint8" else 1
    values = torch.as_tensor(base * scale)
    if source == "uint8":
        values = values + 128
    tensor = values.to(getattr(torch, source)).cuda()
    reference = tensor.float() if source == "bfloat16" else tensor  # NumPy has no bfloat16
    expected = eda.decompose(reference.cpu().numpy(), axes=1, dtype=target)
    y = eda.decompose(tensor, axes=1, dtype=target)
    assert same(torch_read(torch, y), expected)
    with pytest.raises(TypeError, match="pass dtype="):
        eda.decompose(tensor, axes=1)


def test_unsupported_device_dtypes(cp, torch):
    complex_array = cp.zeros((2, 40), cp.complex64)
    with pytest.raises(TypeError, match="complex64 are not supported"):
        eda.decompose(complex_array, axes=1)
    with pytest.raises(TypeError, match="complex64 are not supported"):
        eda.decompose(complex_array, axes=1, dtype="float32")
    with pytest.raises(TypeError, match="dtype= must be float32 or float64, got float16"):
        eda.decompose(cp.zeros((2, 40), cp.float32), axes=1, dtype="float16")


def test_non_mergeable_batch_axes_are_converted(cp, monkeypatch):
    values = field((6, 300), 4)
    stack = values.reshape(3, 2, 300)
    view = cp.asarray(stack).transpose(1, 0, 2)  # batch strides (300, 600) do not merge
    reference = np.ascontiguousarray(stack.transpose(1, 0, 2))
    host_spy = oracle.Spy(monkeypatch, _native.core(), "decompose_host")
    device_spy = oracle.Spy(monkeypatch, _native.core(), "decompose_device")
    eda.decompose(reference, axes=2)
    host_chunk = host_spy.chunk
    for forced in (0, 1, 2):
        device_spy.max_chunk = forced
        host_spy.max_chunk = forced
        y = eda.decompose(view, axes=2)
        assert same(cupy_read(cp, y), eda.decompose(reference, axes=2)), forced
        assert device_spy.chunk == host_spy.chunk, forced
    assert host_chunk > 0
    # and a layout whose batch axes are not even adjacent: (2, 37, 3) with axes=(1,)
    cube = field((6, 37), 5).reshape(2, 3, 37).transpose(0, 2, 1)
    assert same(cupy_read(cp, eda.decompose(cp.asarray(cube), axes=1)), eda.decompose(cube, axes=1))


def test_strided_views_equal_the_host_call(cp):
    base = field((8, 60, 54), 6, "float32")
    device = cp.asarray(base)
    views = {
        "slice": (device[::2], base[::2], (1, 2)),
        "reversed rows": (device[:, ::-1], base[:, ::-1], (1, 2)),
        "transposed": (device.transpose(0, 2, 1), base.transpose(0, 2, 1), (1, 2)),
        "cropped": (device[1:5, 3:50, 2:40], base[1:5, 3:50, 2:40], (1, 2)),
        "batch in the middle": (device.transpose(1, 0, 2), base.transpose(1, 0, 2), (0, 2)),
    }
    for name, (d, h, axes) in views.items():
        assert same(cupy_read(cp, eda.decompose(d, axes=axes)), eda.decompose(h, axes=axes)), name
    broadcast = cp.broadcast_to(device[0], (3, 60, 54))  # a zero batch stride
    assert same(
        cupy_read(cp, eda.decompose(broadcast, axes=(1, 2))),
        eda.decompose(np.broadcast_to(base[0], (3, 60, 54)), axes=(1, 2)),
    )


def test_read_only_interface_data_is_accepted(cp):
    x = field((2, 200), dtype="float32")
    wrapped = CaiOnly(cp.asarray(x), readonly=True)
    assert same(cupy_read(cp, eda.decompose(wrapped, axes=1)), eda.decompose(x, axes=1))


def test_old_interface_versions_and_missing_stream(cp):
    x = field((2, 200), dtype="float32")
    device = cp.asarray(x)
    expected = eda.decompose(x, axes=1)
    for wrapped in (CaiOnly(device, version=2), CaiOnly(device, stream=None)):
        assert same(cupy_read(cp, eda.decompose(wrapped, axes=1)), expected)


def test_dlpack_producer_without_the_versioned_protocol(cp):
    class OldProducer:
        def __init__(self, array):
            self.array = array

        def __dlpack__(self, stream=None):
            return self.array.__dlpack__(stream=stream)

        def __dlpack_device__(self):
            return self.array.__dlpack_device__()

    x = field((2, 200), dtype="float32")
    y = eda.decompose(OldProducer(cp.asarray(x)), axes=1)
    assert same(cupy_read(cp, y), eda.decompose(x, axes=1))


def test_managed_memory_input(cp):
    x = field((2, 200), dtype="float32")
    memory = cp.cuda.malloc_managed(x.nbytes)
    managed = cp.ndarray(x.shape, np.float32, memory)
    managed[...] = cp.asarray(x)
    expected = eda.decompose(x, axes=1)
    assert same(cupy_read(cp, eda.decompose(CaiOnly(managed), axes=1)), expected)
    assert same(cupy_read(cp, eda.decompose(managed, axes=1)), expected)


def test_empty_batch_on_the_device(cp, torch):
    for empty in (cp.zeros((0, 3, 40), cp.float32), torch.zeros((0, 3, 40), device="cuda")):
        y, info = eda.decompose(empty, axes=2, return_info=True)
        assert isinstance(y, eda.DeviceArray) and y.shape == (0, 3, 1, 40) and y.size == 0
        assert info.n_imfs.shape == (0, 3)
        assert tuple(cp.from_dlpack(y).shape) == (0, 3, 1, 40)
        assert tuple(torch.from_dlpack(y).shape) == (0, 3, 1, 40)
        assert tuple(cp.asarray(y).shape) == (0, 3, 1, 40)


# ----------------------------------------------------------------------------------------------
# Input that cannot be read


def test_host_pointer_in_a_cuda_array_interface(cp):
    host = np.zeros((2, 40), np.float32)

    class HostCai:
        __cuda_array_interface__ = {
            "shape": (2, 40),
            "typestr": "<f4",
            "data": (host.ctypes.data, False),
            "version": 3,
            "stream": None,
        }

    with pytest.raises(ValueError, match="is not CUDA device or managed memory"):
        eda.decompose(HostCai(), axes=1)


def test_misaligned_pointer(cp):
    device = cp.zeros(2 * 64 * 8 + 8, cp.uint8)
    wrapped = CaiOnly(device[:1024].view(cp.float64).reshape(2, 64))
    wrapped.__cuda_array_interface__["data"] = (device.data.ptr + 4, False)
    with pytest.raises(ValueError, match="align"):
        eda.decompose(wrapped, axes=1)


def test_spatial_extent_of_one_on_the_device(cp):
    with pytest.raises(ValueError, match="needs at least 2 samples"):
        eda.decompose(cp.zeros((3, 1), cp.float32), axes=1)


# ----------------------------------------------------------------------------------------------
# Streams


SPIN = r"""
extern "C" __global__ void spin(long long cycles) {
  long long t0 = clock64();
  while (clock64() - t0 < cycles) {}
}
"""


def written_late(cp, final, stream, cycles=60_000_000):
    """A device array that receives ``final`` on ``stream`` after a spin of about ``cycles``."""
    spin = cp.RawKernel(SPIN, "spin")
    x = cp.zeros_like(final)
    cp.cuda.Device().synchronize()
    with stream:
        spin((1,), (1,), (np.int64(cycles),))
        x[...] = final
    return x


def test_cupy_streams_are_honoured(cp):
    host = field((3, 400), 7, "float32")
    final = cp.asarray(host)
    expected = eda.decompose(host, axes=1)
    s = cp.cuda.Stream(non_blocking=True)
    # an explicit stream: the call waits for it
    x = written_late(cp, final, s)
    assert same(cupy_read(cp, eda.decompose(x, axes=1, stream=s)), expected)
    # the same, with the integer handle and with a __cuda_stream__ object
    x = written_late(cp, final, s)
    assert same(cupy_read(cp, eda.decompose(x, axes=1, stream=s.ptr)), expected)
    # DLPack without stream=: the producer orders its current stream before the legacy stream
    x = written_late(cp, final, s)
    with s:
        assert same(cupy_read(cp, eda.decompose(x, axes=1)), expected)
    # the interface path: the version 3 stream entry names the stream
    x = written_late(cp, final, s)
    with s:
        wrapped = CaiOnly(x)
    assert wrapped.__cuda_array_interface__["stream"] == s.ptr
    assert same(cupy_read(cp, eda.decompose(wrapped, axes=1)), expected)
    # the interface path with stream= and no stream entry
    x = written_late(cp, final, s)
    assert same(cupy_read(cp, eda.decompose(CaiOnly(x, stream=None), axes=1, stream=s)), expected)
    # DLPack with stream= naming another stream than the producer's current one: the producer is
    # handed that stream and orders its current stream before it
    other = cp.cuda.Stream(non_blocking=True)
    x = written_late(cp, final, s)
    with s:
        assert same(cupy_read(cp, eda.decompose(x, axes=1, stream=other)), expected)


def test_per_thread_default_stream(cp):
    host = field((2, 300), 8, "float32")
    final = cp.asarray(host)
    expected = eda.decompose(host, axes=1)
    ptds = cp.cuda.Stream.ptds
    x = written_late(cp, final, ptds)
    assert same(cupy_read(cp, eda.decompose(x, axes=1, stream=2)), expected)
    x = written_late(cp, final, ptds)
    with ptds:
        wrapped = CaiOnly(x)
    assert wrapped.__cuda_array_interface__["stream"] == 2
    assert same(cupy_read(cp, eda.decompose(wrapped, axes=1)), expected)


def test_legacy_stream_handles(cp):
    host = field((2, 300), 9, "float32")
    x = cp.asarray(host)
    expected = eda.decompose(host, axes=1)
    for handle in (0, 1):
        assert same(cupy_read(cp, eda.decompose(x, axes=1, stream=handle)), expected)


@pytest.mark.skipif(SANITIZED, reason="the sanitizer serializes launches: no stream overlap")
def test_a_missing_fence_is_visible(cp):
    """Control: with no stream entry and no stream= the call does not wait, so it reads the data
    from before the late write. This shows that the positive cases above detect a missing fence."""
    old = field((3, 400), 10, "float32")
    new = field((3, 400), 11, "float32")
    eda.decompose(old, axes=1)  # warm up: modules loaded, plans created, the free queue empty
    gc.collect()
    eda.decompose(cp.asarray(old), axes=1)
    s = cp.cuda.Stream(non_blocking=True)
    x = cp.asarray(old)
    spin = cp.RawKernel(SPIN, "spin")
    cp.cuda.Device().synchronize()
    with s:
        spin((1,), (1,), (np.int64(2_000_000_000),))
        x[...] = cp.asarray(new)
    stale = cupy_read(cp, eda.decompose(CaiOnly(x, stream=None), axes=1))
    s.synchronize()
    assert same(stale, eda.decompose(old, axes=1))
    assert not same(stale, eda.decompose(new, axes=1))
    # and with the fence the call reads the new data
    x = written_late(cp, cp.asarray(new), s)
    assert same(cupy_read(cp, eda.decompose(x, axes=1, stream=s)), eda.decompose(new, axes=1))


def test_torch_streams_are_honoured(torch):
    host = field((3, 400), 12, "float32")
    final = torch.as_tensor(host).cuda()
    expected = eda.decompose(host, axes=1)
    s = torch.cuda.Stream()

    def written_late_torch():
        x = torch.zeros_like(final)
        torch.cuda.synchronize()
        with torch.cuda.stream(s):
            torch.cuda._sleep(200_000_000)
            x.copy_(final)
        return x

    x = written_late_torch()
    assert same(torch_read(torch, eda.decompose(x, axes=1, stream=s)), expected)
    x = written_late_torch()
    with torch.cuda.stream(s):
        assert same(torch_read(torch, eda.decompose(x, axes=1)), expected)
    x = written_late_torch()
    assert same(torch_read(torch, eda.decompose(x, axes=1, stream=s.cuda_stream)), expected)
    # stream= names another stream than the current one: torch orders its current stream before it
    other = torch.cuda.Stream()
    x = written_late_torch()
    with torch.cuda.stream(s):
        assert same(torch_read(torch, eda.decompose(x, axes=1, stream=other)), expected)


def test_torch_inputs(torch):
    host = field((4, 90, 70), 13, "float32")
    tensor = torch.as_tensor(host).cuda()
    expected = eda.decompose(host, axes=(1, 2))
    assert same(torch_read(torch, eda.decompose(tensor, axes=(1, 2))), expected)
    permuted = tensor.permute(0, 2, 1)
    assert same(
        torch_read(torch, eda.decompose(permuted, axes=(1, 2))),
        eda.decompose(host.transpose(0, 2, 1), axes=(1, 2)),
    )
    grad = tensor.clone().requires_grad_(True)  # exported detached
    assert same(torch_read(torch, eda.decompose(grad, axes=(1, 2))), expected)
    cpu_grad = torch.as_tensor(host).requires_grad_(True)
    assert same(eda.decompose(cpu_grad, axes=(1, 2)), expected)  # host input: a NumPy array


def need_free_memory(cp, gib):
    """Skip when the device has less free memory than the test needs (another program holds it)."""
    free = cp.cuda.runtime.memGetInfo()[0]
    if free < gib << 30:
        pytest.skip(f"needs {gib} GiB of free device memory, {free / 2**30:.1f} GiB free")


@pytest.mark.skipif(SANITIZED, reason="the sanitizer serializes launches")
def test_a_device_result_is_complete_on_return(cp):
    """The device has no work of the call left when it returns: a device synchronization right
    after the return finds nothing to wait for (copying the rows out takes about 0.5 ms)."""
    need_free_memory(cp, 4)
    x = cp.asarray(field((2048, 4096), 19, "float32"))
    eda.decompose(x, axes=1)
    gc.collect()
    eda.decompose(np.zeros((1, 64), np.float32), axes=1)  # the free queue is empty from here on
    kept, waits = [], []
    for _ in range(5):
        kept.append(eda.decompose(x, axes=1))  # kept: no drop makes a later call synchronize
        start = time.perf_counter()
        cp.cuda.runtime.deviceSynchronize()
        waits.append(time.perf_counter() - start)
    assert min(waits) < 150e-6, waits


# ----------------------------------------------------------------------------------------------
# Lifetime and memory


def test_a_tensor_made_from_a_result_survives_the_result_and_later_calls(torch):
    host = field((3, 300), 14, "float32")
    expected = eda.decompose(host, axes=1)
    y = eda.decompose(torch.as_tensor(host).cuda(), axes=1)
    tensor = torch.from_dlpack(y)
    del y
    gc.collect()
    for _ in range(3):
        eda.decompose(host[:1], axes=1)
        eda.decompose(torch.as_tensor(host[:2]).cuda(), axes=1)
    assert same(tensor.cpu().numpy(), expected)


def test_an_array_made_by_cupy_survives_the_result(cp):
    host = field((3, 300), 15, "float32")
    expected = eda.decompose(host, axes=1)
    y = eda.decompose(cp.asarray(host), axes=1)
    view = cp.asarray(y)
    del y
    gc.collect()
    for _ in range(3):
        eda.decompose(host[:1], axes=1)
    assert same(cp.asnumpy(view), expected)


@pytest.mark.skipif(SANITIZED, reason="the sanitizer's own tracking uses device memory")
def test_dropped_results_go_back_to_the_driver(cp):
    # this process's own device memory (NVML): the device-wide free memory moves with other programs
    try:
        held = device_memory.ProcessMemory(cp.cuda.Device().pci_bus_id).held
    except OSError as error:
        pytest.skip(f"NVML is not available: {error}")
    cp.get_default_memory_pool().free_all_blocks()
    x = cp.asarray(field((4096, 1024), 16, "float32"))  # the output is above 64 MiB
    small = np.zeros((1, 64), np.float32)
    gc.collect()
    y = eda.decompose(x, axes=1)
    del y
    gc.collect()
    eda.decompose(small, axes=1)  # drains
    before = held()
    y = eda.decompose(x, axes=1)
    assert held() - before > 64 << 20
    del y
    gc.collect()
    eda.decompose(small, axes=1)  # the next call frees what was queued
    after = held()
    assert abs(after - before) < 16 << 20, (after - before) >> 20
    # three results dropped together: one call frees all of them
    ys = [eda.decompose(x, axes=1) for _ in range(3)]
    assert held() - before > 3 * (64 << 20)
    del ys
    gc.collect()
    eda.decompose(small, axes=1)
    after = held()
    assert abs(after - before) < 16 << 20, (after - before) >> 20


FITS_WITHOUT_THE_WORKSPACE = """
import numpy as np, cupy as cp, eda
rng = np.random.default_rng(20)
host = (np.sin(0.2 * np.arange(1024)) + 0.1 * rng.standard_normal((20000, 1024))).astype(np.float32)
x = cp.asarray(host) if {kind!r} == "device" else host
hold = cp.cuda.runtime.malloc(cp.cuda.runtime.memGetInfo()[0] - (3 << 29))
try:
    y = eda.decompose(x, axes=1)
    got = cp.asnumpy(cp.asarray(y)) if {kind!r} == "device" else y
    del y
finally:
    cp.cuda.runtime.free(hold)
assert got.tobytes() == eda.decompose(host, axes=1).tobytes()
print("ok")
"""


@pytest.mark.skipif(SANITIZED, reason="the sanitizer's own tracking uses device memory")
@pytest.mark.parametrize("kind", ["device", "host"])
def test_a_result_that_fits_only_without_the_workspace(cp, kind):
    """In a fresh process with 1.5 GiB free, the rows (0.84 GiB) and the workspace fill the
    device, and the result (0.53 GiB) fits only once the workspace is freed: the call frees it
    and returns the same result as without the limit."""
    need_free_memory(cp, 3)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    code = FITS_WITHOUT_THE_WORKSPACE.format(kind=kind)
    done = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=240
    )
    assert done.returncode == 0 and "ok" in done.stdout, done.stdout + done.stderr


@pytest.mark.skipif(SANITIZED, reason="the sanitizer's own tracking uses device memory")
def test_memory_stays_bounded_over_many_calls(cp):
    # this process's own device memory (NVML): the device-wide free memory moves with other programs
    try:
        held = device_memory.ProcessMemory(cp.cuda.Device().pci_bus_id).held
    except OSError as error:
        pytest.skip(f"NVML is not available: {error}")
    x = cp.asarray(field((1024, 1024), 17, "float32"))
    gc.collect()
    eda.decompose(x, axes=1)
    gc.collect()
    before = held()
    for _ in range(20):
        eda.decompose(x, axes=1)
    gc.collect()
    eda.decompose(np.zeros((1, 64), np.float32), axes=1)
    assert held() - before < 64 << 20


def test_input_on_another_device(cp):
    if eda.device_count() < 2:
        pytest.skip("needs two GPUs")
    host = field((3, 300), 18, "float32")
    with cp.cuda.Device(1):
        x = cp.asarray(host)
    y = eda.decompose(x, axes=1)
    assert y.device == 1 and cp.cuda.Device().id == 0
    with cp.cuda.Device(1):
        assert same(cp.asnumpy(cp.asarray(y)), eda.decompose(host, axes=1))


# ----------------------------------------------------------------------------------------------
# One cuFFT per process, whichever library loads first

IMPORT_ORDERS = {
    "eda first": "import eda\nimport cupy, torch",
    "torch first": "import torch\nimport eda\nimport cupy",
    "cupy first": (
        "import cupy\nimport cupy.fft\ncupy.fft.fft(cupy.ones(64, cupy.complex64))\n"
        "import eda\nimport torch"
    ),
}


@needs_manifest
@pytest.mark.parametrize("order", list(IMPORT_ORDERS))
def test_one_cufft_whatever_the_import_order(cp, torch, order):
    code = textwrap.dedent(
        """
        import hashlib, sys
        import numpy as np
        {imports}
        import eda
        x = np.random.default_rng(1).standard_normal((2, 64, 64)).astype(np.float32)
        a = eda.decompose(x, axes=(1, 2))
        b = eda.decompose(torch.as_tensor(x).cuda(), axes=(1, 2))
        assert np.array_equal(a, torch.from_dlpack(b).cpu().numpy())
        paths = sorted({{l.split()[-1] for l in open("/proc/self/maps")
                         if "libcufft" in l and l.split()[-1].startswith("/")}})
        assert len(paths) == 1, paths
        digest = hashlib.sha256(open(paths[0], "rb").read()).hexdigest()
        assert digest == "{digest}", (paths, digest)
        print("ok", paths[0])
        """
    ).format(imports=IMPORT_ORDERS[order], digest=oracle.provenance()["libcufft_sha256"])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    env.pop("LD_LIBRARY_PATH", None)
    done = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600
    )
    assert done.returncode == 0, done.stdout + done.stderr
