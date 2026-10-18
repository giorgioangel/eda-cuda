# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
"""Host input through ``eda.decompose``: layouts, casts, errors, information and threads."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import eda
from eda import _native

pytestmark = pytest.mark.gpu


def field(shape, dtype="float32", seed=3):
    """A smooth multi-tone field plus noise, so that every element has several modes."""
    rng = np.random.default_rng(seed)
    grids = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij")
    out = sum(np.sin((0.9 + 0.35 * k) * g * (0.3 + 0.1 * k)) for k, g in enumerate(grids))
    out = out + np.sin(0.05 * sum(grids)) * 3 + 0.05 * rng.standard_normal(shape)
    return out.astype(dtype)


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def test_result_type_and_shapes():
    x = field((3, 80))
    y, info = eda.decompose(x, axes=1, return_info=True)
    assert isinstance(y, np.ndarray) and y.dtype == np.float32
    assert y.shape[0] == 3 and y.shape[2] == 80
    p = y.shape[1]
    assert p == int(info.n_imfs.max()) + 1
    assert info.n_imfs.shape == (3,) and info.n_imfs.dtype == np.int32
    assert info.error.shape == (3,) and not info.error.any()
    for name, dt in (
        ("windows", np.int32),
        ("sifts", np.int32),
        ("spacing", np.float64),
        ("energy", np.float64),
    ):
        a = getattr(info, name)
        assert a.dtype == dt and a.shape == (3, int(info.n_imfs.max())), name
    assert info.dtype == "float32" and info.axes == (1,)
    # the rows of an element add up to it
    tol = 64 * np.finfo(np.float32).eps * np.abs(x).max()
    assert np.abs(y.astype(np.float64).sum(axis=1) - x).max() <= tol


def test_axes_none_info_is_zero_dimensional():
    x = field((60, 50), "float64")
    y, info = eda.decompose(x, return_info=True)
    assert y.shape == (y.shape[0], 60, 50)
    assert info.n_imfs.shape == () and info.error.shape == ()
    assert info.windows.shape == (int(info.n_imfs),)
    assert info.axes == (0, 1)


def test_negative_axes_equal_positive():
    x = field((2, 40, 36))
    assert same(eda.decompose(x, axes=(-2, -1)), eda.decompose(x, axes=(1, 2)))
    assert same(eda.decompose(x, axes=-1), eda.decompose(x, axes=2))


@pytest.mark.parametrize("axes", [(0, 1), (1, 0), 0])
def test_batch_after_spatial_axes_matches_canonical_layout(axes):
    shape = (36, 30, 3) if axes != 0 else (64, 3, 2)
    x = field(shape)
    y = eda.decompose(x, axes=axes)
    ax = (axes,) if isinstance(axes, int) else axes
    ndim = x.ndim
    batch_axes = [i for i in range(ndim) if i not in ax]
    canon = eda.decompose(
        np.ascontiguousarray(x.transpose(batch_axes + list(ax))),
        axes=tuple(range(len(batch_axes), ndim)),
    )
    m = min(ax)
    assert y.shape == x.shape[:m] + (canon.shape[len(batch_axes)],) + x.shape[m:]
    # element-wise: y[..., p, ...] == canon[batch, p, spatial in the order of axes]
    index = np.indices(y.shape).reshape(ndim + 1, -1)
    mode = index[m]
    xi = np.delete(index, m, axis=0)
    source = tuple(xi[i] for i in batch_axes) + (mode,) + tuple(xi[a] for a in ax)
    assert np.array_equal(y.reshape(-1), canon[source])


def test_non_contiguous_and_fortran_input_equal_their_copy():
    base = field((4, 50, 44))
    expected = eda.decompose(np.ascontiguousarray(base), axes=(1, 2))
    assert same(eda.decompose(np.asfortranarray(base), axes=(1, 2)), expected)
    wide = np.zeros((8, 50, 88), np.float32)
    wide[::2, :, ::2] = base
    assert same(eda.decompose(wide[::2, :, ::2], axes=(1, 2)), expected)
    reversed_ = np.ascontiguousarray(base[:, ::-1, :])
    assert same(eda.decompose(base[:, ::-1, :], axes=(1, 2)), eda.decompose(reversed_, axes=(1, 2)))
    assert same(
        eda.decompose(base.transpose(0, 2, 1), axes=(2, 1)),
        eda.decompose(np.ascontiguousarray(base.transpose(0, 2, 1)), axes=(2, 1)),
    )


def test_list_input_equals_array_input():
    x = field((90,), "float64")
    assert same(eda.decompose(x.tolist()), eda.decompose(x))


@pytest.mark.parametrize("source", ["int8", "int64", "uint16", "float16", "bool"])
@pytest.mark.parametrize("target", ["float32", "float64"])
def test_cast_equals_the_cast_input(source, target):
    values = field((3, 120), "float64") * 20
    x = values.astype(source) if source != "bool" else values > 0.5
    with pytest.raises(TypeError, match="pass dtype="):
        eda.decompose(x, axes=1)
    y = eda.decompose(x, axes=1, dtype=target)
    assert y.dtype == np.dtype(target)
    assert same(y, eda.decompose(x.astype(target), axes=1))


def test_dtype_changes_the_compute_dtype_of_float_input():
    x = field((2, 100), "float32")
    y, info = eda.decompose(x, axes=1, dtype=np.float64, return_info=True)
    assert y.dtype == np.float64 and info.dtype == "float64"
    assert same(y, eda.decompose(x.astype(np.float64), axes=1))


def test_big_endian_input_equals_native():
    x = field((2, 100), "float64")
    assert same(eda.decompose(x.astype(">f8"), axes=1), eda.decompose(x, axes=1))
    assert same(eda.decompose(x.astype(">f4"), axes=1), eda.decompose(x.astype("f4"), axes=1))


def test_errors_raise_lists_the_failed_elements():
    x = field((5, 128))
    x[1, 10] = np.nan
    x[3, 20] = np.inf
    with pytest.raises(eda.DecompositionError) as caught:
        eda.decompose(x, axes=1)
    error = caught.value
    assert error.indices.tolist() == [1, 3] and error.codes.tolist() == [1, 1]
    assert error.info.error.tolist() == [0, 1, 0, 1, 0]
    assert "failed for 2 of 5 element(s): code 1" in str(error) and "batch index 1, 3" in str(error)
    assert isinstance(error, ValueError)


def test_errors_status_gives_nan_rows():
    x = field((4, 128))
    x[2, 5] = np.nan
    y, info = eda.decompose(x, axes=1, errors="status", return_info=True)
    assert info.error.tolist() == [0, 0, 1, 0]
    assert np.isnan(y[2]).all() and not np.isnan(y[[0, 1, 3]]).any()
    assert info.n_imfs[2] == 0 and not info.windows[2].any()


def test_flat_batch_indices_of_a_multi_axis_batch():
    x = field((2, 3, 100))
    x[1, 2, 7] = np.nan
    with pytest.raises(eda.DecompositionError) as caught:
        eda.decompose(x, axes=2)
    assert caught.value.indices.tolist() == [5]
    assert caught.value.info.error.shape == (2, 3)


def test_check_finite_false_reports_code_2_for_a_non_finite_result():
    x = field((3, 128))
    x[1, 9] = np.inf
    y, info = eda.decompose(x, axes=1, check_finite=False, errors="status", return_info=True)
    assert info.error[1] == 2 and info.error[0] == 0
    assert np.isnan(y[1]).all()


def test_internal_error_always_raises(monkeypatch):
    core = _native.core()
    original = core.decompose_host

    def tampered(*args):
        rows, raw = original(*args)
        raw["error"][0] = 255
        return rows, raw

    monkeypatch.setattr(core, "decompose_host", tampered)
    for mode in ("raise", "status"):
        with pytest.raises(RuntimeError, match="error code 255") as caught:
            eda.decompose(field((2, 64)), axes=1, errors=mode)
        assert not isinstance(caught.value, ValueError)


def test_max_imf_limits_the_modes():
    x = field((2, 300))
    full, info_full = eda.decompose(x, axes=1, return_info=True)
    assert info_full.n_imfs.max() >= 2
    y, info = eda.decompose(x, axes=1, max_imf=1, return_info=True)
    assert (info.n_imfs == 1).all() and y.shape[1] == 2
    assert np.array_equal(y[:, 0], full[:, 0])
    for value in (0, -1, -7, 2**40, np.int64(2**62)):
        assert same(eda.decompose(x, axes=1, max_imf=value), full)


def test_tie_tolerance_is_forwarded():
    x = field((2, 200), "float64")
    assert same(eda.decompose(x, axes=1, tie_tolerance=2.0**-30), eda.decompose(x, axes=1))
    assert same(
        eda.decompose(x, axes=1, tie_tolerance=np.float32(2.0**-30)), eda.decompose(x, axes=1)
    )
    eda.decompose(x, axes=1, tie_tolerance=0)
    eda.decompose(x, axes=1, tie_tolerance=0.01)


def test_stream_is_checked_and_ignored_for_host_input():
    x = field((2, 100))
    expected = eda.decompose(x, axes=1)

    class Handle:
        def __cuda_stream__(self):
            return (0, 12345)  # no such stream: host input never touches it

    for stream in (0, 1, 2, 99999, Handle()):
        assert same(eda.decompose(x, axes=1, stream=stream), expected)


def test_empty_batch():
    y, info = eda.decompose(np.zeros((0, 3, 40), np.float32), axes=2, return_info=True)
    assert y.shape == (0, 3, 1, 40) and y.dtype == np.float32
    assert info.n_imfs.shape == (0, 3) and info.windows.shape == (0, 3, 0)
    y, info = eda.decompose(np.zeros((2, 0, 40), np.float64), axes=2, return_info=True)
    assert y.shape == (2, 0, 1, 40)


def test_constant_signal_has_no_modes():
    y, info = eda.decompose(np.full((2, 64), 0.75, np.float32), axes=1, return_info=True)
    assert y.shape == (2, 1, 64) and (info.n_imfs == 0).all()
    assert (y == 0.75).all()


def test_threads_give_the_sequential_results():
    inputs = [field((3, 200 + 16 * k), "float32", seed=k) for k in range(4)]
    expected = [[eda.decompose(x, axes=1) for _ in range(3)] for x in inputs]

    def work(k):
        return [eda.decompose(inputs[k], axes=1) for _ in range(3)]

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(work, range(4)))
    for k in range(4):
        for call in range(3):
            assert same(results[k][call], expected[k][call])


def test_threads_with_errors_do_not_cross():
    good = field((2, 150))
    bad = good.copy()
    bad[0, 3] = np.nan

    def call(x):
        try:
            return eda.decompose(x, axes=1)
        except eda.DecompositionError as exc:
            return exc

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(call, [good, bad, good, bad, good, bad]))
    assert all(isinstance(r, np.ndarray) for r in results[0::2])
    assert all(isinstance(r, eda.DecompositionError) for r in results[1::2])
    assert same(results[0], results[2]) and same(results[2], results[4])
