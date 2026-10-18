# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
"""``sifts``, the number of sifts per mode, through ``eda.decompose``.

The default is 3. Every count from 1 up keeps the guarantees of the default: the rows add up to
the input within the commit's rounding, an element's result does not depend on the rest of the
batch, and host and CUDA input give the same bits. ``info.sifts`` reports between 1 and ``sifts``
completed sifts per mode (the guard can end a mode's sifting early). Each further sift leaves less
of a slower oscillation in the first mode: one sift keeps a fraction of a tone the window does not
resolve, and every sift multiplies what is left by that fraction again.
"""

from __future__ import annotations

import numpy as np
import pytest

import eda
from test_properties import bits, info_equal, mix, reconstruction_excess

pytestmark = pytest.mark.gpu

SHAPES = [((1000,), "float32"), ((48, 40), "float64"), ((20, 18, 16), "float32")]
IDS = ["1d-f32", "2d-f64", "3d-f32"]


def two_tones(dim):
    """A fast and a slower plane wave, the slow one about 12 times the fast one's period."""
    if dim == 1:
        t = np.arange(4096, dtype=np.float64)
        return np.sin(2 * np.pi * t / 8), np.sin(2 * np.pi * t / 96 + 0.3)
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float64)
    fast = np.sin(2 * np.pi * (0.11 * xx + 0.07 * yy))
    return fast, np.sin(2 * np.pi * (0.009 * xx + 0.017 * yy) + 0.4)


def slow_share_in_first_mode(dim, sifts):
    fast, slow = two_tones(dim)
    y = eda.decompose((fast + slow).astype(np.float32), sifts=sifts)
    return float(abs(np.vdot(y[0].astype(np.float64), slow)) / np.vdot(slow, slow))


@pytest.mark.parametrize(("shape", "dtype"), SHAPES, ids=IDS)
def test_the_default_is_three_sifts(shape, dtype):
    x = mix(shape, 3).astype(dtype)
    y, info = eda.decompose(x, return_info=True)
    y3, info3 = eda.decompose(x, sifts=3, return_info=True)
    assert bits(y) == bits(y3) and info_equal(info, info3)


@pytest.mark.parametrize("sifts", [1, 2, 5])
@pytest.mark.parametrize(("shape", "dtype"), SHAPES, ids=IDS)
def test_every_count_reconstructs_and_reports_its_sifts(shape, dtype, sifts):
    x = mix(shape, 4).astype(dtype)
    y, info = eda.decompose(x, sifts=sifts, return_info=True)
    n = int(info.n_imfs)
    assert n >= 1
    done = info.sifts[:n]
    assert ((done >= 1) & (done <= sifts)).all(), done
    assert reconstruction_excess(x, y, n) <= 1.0


@pytest.mark.parametrize("sifts", [1, 4, 8])
def test_modes_run_the_sifts_asked_for(sifts):
    fast, slow = two_tones(1)
    _, info = eda.decompose((fast + slow).astype(np.float32), sifts=sifts, return_info=True)
    assert info.sifts[: int(info.n_imfs)].max() == sifts


@pytest.mark.parametrize("sifts", [1, 5])
@pytest.mark.parametrize(("shape", "dtype"), [((16, 51), "float32"), ((12, 23, 19), "float64")])
def test_an_element_does_not_see_the_batch(shape, dtype, sifts):
    x = mix(shape, 7).astype(dtype)
    alone, info_alone = eda.decompose(x, sifts=sifts, return_info=True)
    for position in (0, 2, 4):
        batch = np.stack([mix(shape, 20 + b).astype(dtype) for b in range(5)])
        batch[position] = x
        axes = tuple(range(1, len(shape) + 1))
        y, info = eda.decompose(batch, axes=axes, sifts=sifts, return_info=True)
        n = int(info_alone.n_imfs)
        assert int(info.n_imfs[position]) == n
        assert bits(y[position, : n + 1]) == bits(alone[: n + 1])
        assert np.array_equal(info.sifts[position, :n], info_alone.sifts[:n])


@pytest.mark.parametrize("sifts", [1, 5])
def test_cuda_input_gives_the_host_bits(sifts):
    cp = pytest.importorskip("cupy")
    x = mix((48, 40), 9).astype(np.float32)
    host, info_host = eda.decompose(x, sifts=sifts, return_info=True)
    device, info_device = eda.decompose(cp.asarray(x), sifts=sifts, return_info=True)
    assert bits(cp.asnumpy(cp.asarray(device))) == bits(host)
    assert info_equal(info_host, info_device)


@pytest.mark.parametrize("dim", [1, 2])
def test_more_sifts_leave_less_of_a_slower_tone_in_the_first_mode(dim):
    shares = [slow_share_in_first_mode(dim, sifts) for sifts in (1, 2, 3)]
    assert shares[0] > shares[1] > shares[2], shares
