# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
"""The output rows are allocated without a zero fill: every returned byte must be written by the
device copy, whatever the memory held before. Two checks: the host heap is poisoned with two
different bytes between calls, and equal results prove that no byte is left over from a recycled
block; the rows of every element add up to the element, which also covers rows of 8 MiB and more
(a fresh mapping, aligned to 2 MiB, that no poison reaches and that reads zero where unwritten)."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

native = pytest.importorskip("eda._native")
try:
    core = native.core()
except ImportError:  # extension not built
    pytest.skip("eda._core is not built", allow_module_level=True)


def _field(shape: tuple[int, ...], dtype: str) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.standard_normal(shape).astype(dtype)


def _poison(nbytes: int, value: int) -> None:
    """Fill and free a host buffer of the size of the next allocation, so a recycled block would
    carry the pattern."""
    block = np.full(nbytes, value, dtype=np.uint8)
    del block


def _run(x: np.ndarray, byte: int, nbytes: int):
    _poison(nbytes, byte)
    return core.decompose_host(x, -1, None, 3, True, False, 0)


def _rows_bytes(x: np.ndarray) -> int:
    return core.decompose_host(x, -1, None, 3, True, False, 0)[0].nbytes


def _assert_rows_add_up(rows: np.ndarray, info: dict, x: np.ndarray) -> None:
    """The IMFs and the residue of every element without an error add up to the element, up to
    rounding (64 ulp of its largest magnitude; an unwritten row misses by a whole mode)."""
    tol = 64 * np.finfo(x.dtype).eps
    for b in range(x.shape[0]):
        if info["error"][b]:
            continue
        k = int(info["n_imfs"][b])
        total = rows[b, : k + 1].sum(axis=0, dtype=np.float64)
        assert np.max(np.abs(total - x[b])) <= tol * np.max(np.abs(x[b]))


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((1, 1500), "float32"),
        ((2, 33, 40), "float64"),
        ((1, 1024, 1024), "float32"),  # the rows exceed 8 MiB: the aligned, huge-page path
    ],
)
def test_rows_hold_no_stray_bytes(shape, dtype):
    x = _field(shape, dtype)
    nbytes = _rows_bytes(x)
    first, info_a = _run(x, 0xA5, nbytes)
    second, info_b = _run(x, 0x5A, nbytes)
    assert first.dtype == x.dtype
    assert first.shape[0] == shape[0] and first.shape[2:] == shape[1:]
    assert first.flags["C_CONTIGUOUS"] and first.flags["OWNDATA"] is False
    assert first.tobytes() == second.tobytes()
    assert np.array_equal(info_a["n_imfs"], info_b["n_imfs"])
    _assert_rows_add_up(first, info_a, x)
    # Rows beyond n_imfs of an element are zero (written by the copy, not left by the allocation).
    for b in range(shape[0]):
        k = int(info_a["n_imfs"][b])
        assert not np.any(first[b, k + 1 :])


def test_rows_of_an_element_in_error_are_nan():
    x = _field((3, 600), "float32")
    x[1, 100] = np.nan
    nbytes = _rows_bytes(x)
    first, info = _run(x, 0xA5, nbytes)
    second, _ = _run(x, 0x5A, nbytes)
    assert first.tobytes() == second.tobytes()
    bad = np.asarray(info["error"]) != 0
    assert bad[1] and not bad[0] and not bad[2]
    assert np.all(np.isnan(first[1]))
    assert not np.any(np.isnan(first[0]))
    _assert_rows_add_up(first, info, x)
