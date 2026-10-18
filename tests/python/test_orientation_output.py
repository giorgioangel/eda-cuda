# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
"""``eda.decompose(..., return_orientation=True)``: the direction n and the coherence c that the
method computes at the start of each mode (2-D and 3-D), returned beside the rows.

- Asking for them changes nothing else: the rows and the information are the same bits.
- Layout: ``direction`` has the shape ``x.shape[:m] + (K, D) + x.shape[m:]`` and ``coherence``
  ``x.shape[:m] + (K,) + x.shape[m:]`` (m = min(axes), K the largest IMF count); rows beyond an
  element's ``n_imfs`` and every row of a failed element are 0; |n| = 1 up to rounding where c >
  0, c lies in [0, 1], and n has the canonical sign of ``eda.monogenic`` (its first component of
  magnitude above the dtype's default tie tolerance is positive).
- Values: every row equals the method's orientation recomputed in float64 from the residual of
  its mode (``band_orientation``), and on a fast oblique plane wave over a slower one the first
  mode's direction is the fast wave's in the interior.
- Bitwise: host and CUDA input, an element alone and in a batch, any chunking.
"""

from __future__ import annotations

import numpy as np
import pytest

import eda
import oracle
from eda import _native
from test_properties import bits, info_equal, mix

pytestmark = pytest.mark.gpu

ZERO = {"float32": 2.0**-14, "float64": 2.0**-30}
EPS = {"float32": float(np.finfo(np.float32).eps), "float64": float(np.finfo(np.float64).eps)}


def two_waves(shape, k_fast, k_slow):
    grids = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij")

    def wave(k, phase):
        return np.cos(2 * np.pi * sum(ki * g for ki, g in zip(k, grids, strict=True)) + phase)

    return wave(k_fast, 0.3) + wave(k_slow, 1.1)


def patchy(shape, seed):
    """Two plane waves (periods from 12 to 45) and a fine patch (period 4 to 6): fields whose modes
    include ones that the fine-population test opens."""
    rng = np.random.default_rng(seed)
    g = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij")
    out = np.zeros(shape)
    for period in rng.choice([12.0, 18.0, 30.0, 45.0], size=2, replace=False):
        k = rng.normal(size=len(shape))
        k = k / np.linalg.norm(k) * 2 * np.pi / period
        out += np.sin(sum(ki * gi for ki, gi in zip(k, g, strict=True)) + rng.uniform(0, 6))
    k = rng.normal(size=len(shape))
    k = k / np.linalg.norm(k) * 2 * np.pi / rng.uniform(4, 6)
    centre = [rng.uniform(0.2, 0.8) * n for n in shape]
    r2 = sum(((gi - ci) / (0.18 * n)) ** 2 for gi, ci, n in zip(g, centre, shape, strict=True))
    return out + 0.8 * np.exp(-r2) * np.sin(sum(ki * gi for ki, gi in zip(k, g, strict=True)))


def band_orientation(r, w, zero):
    """The method's orientation of a residual r in float64 (README, Direction): the one-octave
    log-Gabor band at f0 = 1 / w of the half-sample mirrored r - mean, R1 = (R_a b), R2 =
    (R_a R_b b), T = R1 R1^T + R2 R2^T, its top eigenvector with the canonical sign and c = (mu1 -
    mu2) / (mu1 + mu2). Returns n, c, mu1, mu2."""
    r = np.asarray(r, dtype=np.float64)
    D = r.ndim
    ext = r - r.mean()
    for a in range(D):
        ext = np.concatenate([ext, np.flip(ext, axis=a)], axis=a)
    u = np.meshgrid(*[np.fft.fftfreq(2 * n) for n in r.shape], indexing="ij")
    rad = np.sqrt(sum(v * v for v in u))
    inv = np.divide(1.0, rad, out=np.zeros_like(rad), where=rad > 0)
    sigma = np.log(2.0) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    band = np.exp(-(np.log(np.where(rad > 0, rad * w, 1.0)) ** 2) / (2 * sigma**2)) * (rad > 0)
    F = np.fft.fftn(ext) * band
    crop = tuple(slice(0, n) for n in r.shape)

    def back(multiplier):
        return np.real(np.fft.ifftn(F * multiplier))[crop]

    r1 = np.stack([back(-1j * u[a] * inv) for a in range(D)])
    h = np.zeros(r.shape + (D, D))
    for a in range(D):
        for b in range(D):
            h[..., a, b] = back(-u[a] * u[b] * inv**2)
    mu, V = np.linalg.eigh(np.einsum("a...,b...->...ab", r1, r1) + h @ h)
    n = np.moveaxis(V[..., :, -1], -1, 0)
    sign = np.zeros(r.shape)
    for a in range(D):
        sign = np.where((sign == 0) & (np.abs(n[a]) > zero), np.sign(n[a]), sign)
    n = n * np.where(sign < 0, -1.0, 1.0)
    mu1, mu2 = mu[..., -1], mu[..., -2]
    s = mu1 + mu2
    c = np.clip(np.divide(mu1 - mu2, s, out=np.zeros_like(s), where=s > 0), 0, 1)
    return n, c, mu1, mu2


def check_rows(o, info, axes):
    """The layout and the invariants of one call's orientation (batch axes before the spatial
    ones)."""
    m, D = min(axes), len(axes)
    K = info.windows.shape[-1]
    # the mode axes last: direction (*batch, *spatial, K, D), coherence (*batch, *spatial, K)
    n = np.moveaxis(o.direction, (m, m + 1), (-2, -1))
    c = np.moveaxis(o.coherence, m, -1)
    assert n.shape[-2:] == (K, D) and c.shape[-1] == K
    flat_n = n.reshape(-1, K, D)  # (samples of every element, K, D) per element in batch order
    flat_c = c.reshape(-1, K)
    count = flat_c.shape[0] // max(info.n_imfs.size, 1)
    dtype = info.dtype
    for b, (k_b, err) in enumerate(
        zip(info.n_imfs.reshape(-1), info.error.reshape(-1), strict=True)
    ):
        rows = slice(b * count, (b + 1) * count)
        k_b = 0 if err else int(k_b)
        assert not flat_n[rows, k_b:].any() and not flat_c[rows, k_b:].any()
        cb, nb = flat_c[rows, :k_b], flat_n[rows, :k_b].astype(np.float64)
        assert (cb >= 0).all() and (cb <= 1).all()
        norm = np.sqrt((nb**2).sum(axis=-1))
        assert np.abs(norm[cb > 0] - 1).max(initial=0) <= 8 * EPS[dtype]
        first = np.zeros(nb.shape[:-1])
        for a in range(D):
            first = np.where((first == 0) & (np.abs(nb[..., a]) > ZERO[dtype]), nb[..., a], first)
        assert (first > 0).all(), "the canonical sign"


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [(3, 48, 40), (2, 20, 18, 16)], ids=["2d", "3d"])
def test_rows_and_info_do_not_change(shape, dtype):
    x = np.stack([mix(shape[1:], 30 + b) for b in range(shape[0])]).astype(dtype)
    axes = tuple(range(1, len(shape)))
    y, info = eda.decompose(x, axes=axes, return_info=True)
    y2, info2, o = eda.decompose(x, axes=axes, return_info=True, return_orientation=True)
    assert bits(y) == bits(y2) and info_equal(info, info2)
    y3, o3 = eda.decompose(x, axes=axes, return_orientation=True)
    assert bits(y3) == bits(y) and bits(o3.direction) == bits(o.direction)
    assert int(info.n_imfs.min()) >= 1
    check_rows(o, info, axes)
    assert o.direction.dtype == np.dtype(dtype) and o.coherence.dtype == np.dtype(dtype)


def test_layout_with_batch_axes_in_between():
    x = np.stack([mix((40, 36), 40 + b) for b in range(3)], axis=1)  # (40, 3, 36)
    y, info, o = eda.decompose(x, axes=(2, 0), return_info=True, return_orientation=True)
    K = info.windows.shape[-1]
    assert o.direction.shape == (K, 2, 40, 3, 36) and o.coherence.shape == (K, 40, 3, 36)
    one, info_one, o_one = eda.decompose(
        np.ascontiguousarray(x[:, 1, :].T), return_info=True, return_orientation=True
    )
    k = int(info_one.n_imfs)
    assert int(info.n_imfs[1]) == k
    # element 1 seen in the order of axes: the transposed field
    assert bits(o.direction[:k, :, :, 1, :]) == bits(o_one.direction[:k].transpose(0, 1, 3, 2))
    assert bits(o.coherence[:k, :, 1, :]) == bits(o_one.coherence[:k].transpose(0, 2, 1))


# Every row against band_orientation, in eps of the dtype: n weighted by the gap mu1 - mu2 and c
# by mu1 + mu2 (each relative to the largest mu1 of the mode), the weights that bound how well T
# defines them. Measured over the 13 modes of the two cases, float64 / float32: n 3.8 / 2.7, c 4.9
# / 4.2; a neighbouring row in place of the right one gives at least 1.3e15 / 2.4e6. Unweighted,
# where the gap is above 1e-3 of the largest mu1: |n - n_ref| at most 1.3e-13 / 3.8e-5 (99th
# percentile 6.5e-15 / 3.2e-6), and |c - c_ref| at most 5.5e-13 / 1.8e-4.
ROW_TOL = {"float64": 40, "float32": 40}
NEIGHBOUR_MIN = 1e5


def row_error(o, b, k, ref, eps):
    """The weighted errors of row k of element b against ref = band_orientation(...)."""
    n_ref, c_ref, mu1, mu2 = ref
    top = mu1.max()
    dn = np.abs(o.direction[b, k].astype(np.float64) - n_ref).max(axis=0)
    dc = np.abs(o.coherence[b, k].astype(np.float64) - c_ref)
    return float((dn * (mu1 - mu2)).max() / top / eps), float((dc * (mu1 + mu2)).max() / top / eps)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize(
    ("shape", "make"), [((5, 64, 56), patchy), ((3, 24, 22, 20), mix)], ids=["2d", "3d"]
)
def test_every_row_matches_the_float64_reference(shape, make, dtype):
    """The residual of mode k is x - y_0 - .. - y_(k-1), rounded to the dtype after every
    subtraction as the commit rounds it (the method's power-of-two prescale is exact and leaves n
    and c unchanged). Element 1 is constant (no mode) and element 3 of the 2-D batch holds a NaN
    (a failed element: every row 0)."""
    x = np.stack([make(shape[1:], 100 + b) for b in range(shape[0])]).astype(dtype)
    x[1] = 5.0
    if shape[0] > 3:
        x[(3,) + (2,) * (len(shape) - 1)] = np.nan
    axes = tuple(range(1, len(shape)))
    y, info, o = eda.decompose(
        x, axes=axes, errors="status", return_info=True, return_orientation=True
    )
    assert info.n_imfs[1] == 0 and (info.error[3] == 1 if shape[0] > 3 else True)
    worst, nearest, modes = {"n": 0.0, "c": 0.0}, np.inf, 0
    eps = EPS[dtype]
    for b in range(shape[0]):
        kb = 0 if info.error[b] else int(info.n_imfs[b])
        assert not o.direction[b, kb:].any() and not o.coherence[b, kb:].any()
        r = x[b].copy()
        for k in range(kb):
            ref = band_orientation(r, int(info.windows[b, k]), ZERO[dtype])
            en, ec = row_error(o, b, k, ref, eps)
            worst = {"n": max(worst["n"], en), "c": max(worst["c"], ec)}
            for other in (k - 1, k + 1):
                if 0 <= other < kb:
                    nearest = min(nearest, max(row_error(o, b, other, ref, eps)))
            modes += 1
            r = (r - y[b, k]).astype(dtype)
    assert modes >= 6
    assert worst["n"] <= ROW_TOL[dtype] and worst["c"] <= ROW_TOL[dtype], worst
    assert nearest >= NEIGHBOUR_MIN


# The fast wave (period 8.5) over a slow one (period 50): two modes, windows 9 and 57. Measured:
# the first mode's direction within 1.24e-3 of the fast wave's k/|k| in both dtypes, two fast
# periods from the borders, with c >= 0.99999 there; the tolerance is twice that.
FAST_TOL = 2.5e-3


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_first_mode_follows_the_fast_wave(dtype):
    k_fast, k_slow = np.array([0.07, 0.095]), np.array([0.016, -0.012])
    x = two_waves((160, 150), k_fast, k_slow).astype(dtype)
    y, info, o = eda.decompose(x, return_info=True, return_orientation=True)
    assert int(info.n_imfs) >= 2
    margin = int(np.ceil(2 / np.linalg.norm(k_fast)))
    inner = (slice(margin, -margin), slice(margin, -margin))
    khat = k_fast / np.linalg.norm(k_fast)
    err = max(np.abs(o.direction[0, a][inner] - khat[a]).max() for a in range(2))
    print("FAST", dtype, err, o.coherence[0][inner].min(), info.n_imfs, info.windows)
    assert err <= FAST_TOL, err
    assert o.coherence[0][inner].min() >= 0.99


def test_one_spatial_axis_raises():
    with pytest.raises(ValueError, match="return_orientation=True needs 2 or 3 spatial axes"):
        eda.decompose(np.zeros(64), return_orientation=True)
    with pytest.raises(ValueError, match="needs 2 or 3 spatial axes, got 1"):
        eda.decompose(np.zeros((4, 64)), axes=1, return_orientation=True)
    with pytest.raises(TypeError, match="eda: return_orientation must be True or False, got int"):
        eda.decompose(np.zeros((16, 16)), return_orientation=1)


def test_failed_elements_and_max_imf():
    x = np.stack([mix((40, 36), 50 + b) for b in range(3)])
    x[1, 3, 3] = np.nan
    y, info, o = eda.decompose(
        x, axes=(1, 2), errors="status", return_info=True, return_orientation=True
    )
    assert info.error[1] == 1 and not o.direction[1].any() and not o.coherence[1].any()
    check_rows(o, info, (1, 2))
    full, info_full, o_full = eda.decompose(x[0], return_info=True, return_orientation=True)
    one, o_one = eda.decompose(x[0], max_imf=1, return_orientation=True)
    assert o_one.direction.shape[0] == 1 and int(info_full.n_imfs) >= 2
    assert bits(o_one.direction) == bits(o_full.direction[:1])
    assert bits(o_one.coherence) == bits(o_full.coherence[:1])


@pytest.mark.parametrize(
    ("shape", "dtype", "batch", "positions"),
    [((16, 51), "float32", 70, (0, 63, 64, 69)), ((12, 23, 19), "float64", 52, (0, 48, 49, 51))],
    ids=["2d", "3d"],
)
def test_an_element_does_not_see_the_batch(shape, dtype, batch, positions):
    x = mix(shape, 7).astype(dtype)
    _, info_alone, alone = eda.decompose(x, return_info=True, return_orientation=True)
    k = int(info_alone.n_imfs)
    stack = np.stack([mix(shape, 20 + b).astype(dtype) for b in range(batch)])
    for p in positions:
        stack[p] = x
    axes = tuple(range(1, len(shape) + 1))
    _, info, o = eda.decompose(stack, axes=axes, return_info=True, return_orientation=True)
    for p in positions:
        assert int(info.n_imfs[p]) == k
        assert bits(o.direction[p, :k]) == bits(alone.direction[:k])
        assert bits(o.coherence[p, :k]) == bits(alone.coherence[:k])
        assert not o.direction[p, k:].any()


def test_chunks_do_not_change_the_orientation(monkeypatch):
    x = np.stack([mix((24, 20), 80 + b) for b in range(7)]).astype(np.float32)
    _, ref = eda.decompose(x, axes=(1, 2), return_orientation=True)
    spy = oracle.Spy(monkeypatch, _native.core(), "decompose_host")
    spy.max_chunk = 3
    _, o = eda.decompose(x, axes=(1, 2), return_orientation=True)
    assert spy.chunk == 3
    assert bits(o.direction) == bits(ref.direction) and bits(o.coherence) == bits(ref.coherence)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cuda_input_gives_the_host_bits(dtype):
    cp = pytest.importorskip("cupy")
    for shape in [(3, 48, 40), (2, 20, 18, 16)]:
        x = np.stack([mix(shape[1:], 60 + b) for b in range(shape[0])]).astype(dtype)
        axes = tuple(range(1, len(shape)))
        y, info, o = eda.decompose(x, axes=axes, return_info=True, return_orientation=True)
        yd, info_d, od = eda.decompose(
            cp.asarray(x), axes=axes, return_info=True, return_orientation=True
        )
        assert info_equal(info, info_d)
        assert bits(cp.asnumpy(cp.asarray(yd))) == bits(y)
        for name in ("direction", "coherence"):
            d = getattr(od, name)
            assert isinstance(d, eda.DeviceArray) and d.shape == getattr(o, name).shape
            assert bits(cp.asnumpy(cp.asarray(d))) == bits(getattr(o, name))
    # an empty batch
    _, od = eda.decompose(cp.zeros((0, 16, 16), dtype=dtype), axes=(1, 2), return_orientation=True)
    assert od.direction.shape == (0, 0, 2, 16, 16) and od.coherence.shape == (0, 0, 16, 16)
