# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
"""Properties of the method through ``eda.decompose``: reconstruction, invariance under batching,
chunking, repetition and threads, the symmetries of the method, special inputs, ``max_imf``,
``tie_tolerance`` and device memory.

Notation: u is the unit roundoff of the compute dtype (2**-24 float32, 2**-53 float64), eps = 2u,
P the rows of an element (IMFs and the residue), M the largest magnitude of the input.

Batch, chunk, order, repeat and thread invariance is bitwise on every shape: each element runs
alone in its modes, and the orientation's transforms are complex ones, which compute every batch
entry from that entry alone, on a cuFFT plan fixed by the shape. The tests include shapes whose
real-to-complex cuFFT plans compute neighbouring batch entries together (16x51, 12x23x19); with
``-m full`` an element also sits at every position of a batch on 24 sizes, both dtypes.

Which symmetries hold, from the code (a mode's work is exact selections, integer counts and a
fixed order of floating-point sums):

- gain by a power of two: exact (the prescale and the unscale are exact, the tie tolerance is
  relative): BITWISE. The gain test includes 2**80 (float32) and 2**300 (float64), where the
  squares of float32 overflow unless the prescale takes the exponent out.
- sign: every operator is odd or even as needed: the rows of -x are the negated rows, equal in
  value (a zero may change its sign).
- 1-D flip: the line sums are built flip-symmetric and there is no orientation in 1-D: BITWISE,
  but only while no mode's input has a plateau of an even number of equal samples between a rise
  and a fall (the plateau's peak sits at (left + right) // 2, which a mirror moves to the other
  middle sample). A float32 input without adjacent ties still forms such plateaus in the residues
  it leaves (see ``test_flip_ties_form_in_the_residues``), so the precondition is checked on every
  mode's input, rebuilt from the rows, in both runs.
- point reflection and axis permutation in D >= 2: symmetries of the method in exact arithmetic
  only. The FFT and the interpolation round differently on a mirrored
  or permuted field, and a rounding can flip a discrete choice inside a mode (a rung point, a hat
  weight of zero, a near test) that the information does not report; the rows then move by more
  than a rounding, so the code gives no bound on them. The tests assert the decisions the
  information reports, which agree over the declared family (``waves`` of 48x40 to 64x56 and
  30x28x26 to 40x36x32, both dtypes, nine seeds), and report the largest row difference in units
  of eps M (P - 1) in the terminal summary. Measured: 238 for the float32 point reflection of
  40x36x32 with seed 19 and 177 with seed 0 (a flipped choice spreads over a fifth of the
  voxels), at most 13 in every other combination. So 3-D float32 point reflection is recorded as
  not a symmetry beyond rounding. The row differences are held below regression ceilings
  (``ROW_CEILING``): not bounds of the method, but a guard that catches an asymmetric fault.
- single-axis flip in D >= 2: not a symmetry (the noise level pools the signed forward
  differences of all axes); not tested. Exact ties under any mirror: not a symmetry (the plateau
  rule); not tested.
- a constant offset: not a symmetry (the tie tolerance and the sift guard are relative to
  max|r|): the golden chirp changes its windows (``test_an_offset_is_not_a_symmetry``). On the
  declared family the decisions agree, which is asserted as above; the difference of the IMFs and
  of the residue less the offset is reported (measured: at most 3.7) and held below the ceiling.
"""

from __future__ import annotations

import dataclasses
import gc
import itertools
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import device_memory
import eda
import oracle
from eda import _native

pytestmark = pytest.mark.gpu

SANITIZED = os.environ.get("EDA_SANITIZE") not in (None, "", "0")
EPS = {"float32": float(np.finfo(np.float32).eps), "float64": float(np.finfo(np.float64).eps)}

if SANITIZED:
    COMBOS = [
        ((256,), "float64"),
        ((24, 20), "float64"),
        ((24, 20), "float32"),
        ((16, 16, 16), "float64"),
    ]
    SYMMETRY_SHAPES = [((37, 45), "float64"), ((30, 28, 26), "float64")]
    SYMMETRY_SEEDS = (0, 1)
else:
    COMBOS = [
        (shape, dtype)
        for shape in [(512,), (1000,), (48, 40), (37, 45), (20, 18, 16), (24, 22, 20)]
        for dtype in ("float32", "float64")
    ]
    SYMMETRY_SHAPES = [
        (shape, dtype)
        for shape in [(48, 40), (37, 45), (64, 56), (30, 28, 26), (40, 36, 32)]
        for dtype in ("float32", "float64")
    ]
    SYMMETRY_SEEDS = (*range(8), 19)  # 19: the largest row difference seen, see the docstring
SPARSE = COMBOS if SANITIZED else COMBOS[0:2] + COMBOS[4:6] + COMBOS[8:10]


def ident(combos):
    return [f"{'x'.join(map(str, s))}-{d}" for s, d in combos]


IDS = ident(COMBOS)

# The cases of the tests whose names the sanitizer matrix selects under racecheck (gain, sign,
# batch) are few and small under EDA_SANITIZE: racecheck needs seconds per call.
if SANITIZED:
    GAIN_CASES = [((128,), "float64", 3, None), ((16, 16), "float32", "huge", None)]
    SIGN_CASES = [((128,), "float64", "mix"), ((16, 16), "float32", "quantized")]
else:
    GAIN_CASES = [
        (shape, dtype, j, tie)
        for shape, dtype in COMBOS
        for j in (-8, 3, 20, "huge", "-huge")
        for tie in (None, 0.0, 2.0**-10)
    ]
    SIGN_CASES = [(shape, dtype, kind) for shape, dtype in COMBOS for kind in ("mix", "quantized")]
GAIN_IDS = [f"{'x'.join(map(str, s))}-{d}-{j}-{t}" for s, d, j, t in GAIN_CASES]
SIGN_IDS = [f"{'x'.join(map(str, s))}-{d}-{k}" for s, d, k in SIGN_CASES]


# ----------------------------------------------------------------------------------------------
# Fields and helpers


def waves(shape, seed, noise=0.05, periods=None):
    """Plane waves of random directions at periods 5.5, 17 and min(shape) / 1.7, plus noise, in
    float64: several modes in every dimension."""
    rng = np.random.default_rng(seed)
    grids = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij")
    out = np.zeros(shape)
    for period, amp in periods or ((5.5, 0.3), (17.0, 0.6), (min(shape) / 1.7, 1.0)):
        k = rng.normal(size=len(shape))
        k = k / np.linalg.norm(k) * 2 * np.pi / period
        out += amp * np.sin(sum(ki * g for ki, g in zip(k, grids, strict=True)) + rng.uniform(0, 6))
    return out + noise * rng.standard_normal(shape)


def mix(shape, seed):
    """One sinusoid per axis (periods 4.8 to 7), a diagonal one, a slow one and noise, in
    float64: two or more modes from 12 samples per axis up."""
    rng = np.random.default_rng(seed)
    grids = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij")
    k = rng.uniform(0.9, 1.3, len(shape))
    out = sum(np.sin(k[i] * g) for i, g in enumerate(grids))
    out = out + 0.5 * np.sin(0.13 * sum(grids)) + 0.1 * np.cos(0.031 * grids[0])
    return out + 0.05 * rng.standard_normal(shape)


def quantized(shape, seed):
    """Integer-valued input, full of exact ties."""
    rng = np.random.default_rng(seed)
    grids = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij")
    return np.floor(rng.uniform(0, 5, shape)) - 2 + np.round(3 * np.sin(0.3 * sum(grids)))


def slow_line(n, seed, noise, dtype):
    """Three tones, the slowest with a period of n/3 to n/1.5, plus noise: windows above 1000."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    p1, p2, p3 = rng.uniform(n / 3, n / 1.5), rng.uniform(150, 600), rng.uniform(9, 40)
    x = (
        np.sin(2 * np.pi * t / p1 + rng.uniform(0, 6))
        + 0.4 * np.sin(2 * np.pi * t / p2 + rng.uniform(0, 6))
        + 0.2 * np.sin(2 * np.pi * t / p3)
        + noise * rng.standard_normal(n)
    )
    return x.astype(dtype)


def run(x, **kwargs):
    return eda.decompose(x, return_info=True, **kwargs)


def bits(a):
    return np.ascontiguousarray(a).tobytes()


def decisions_equal(a, b):
    """Every decision the information reports: the IMF counts, the windows and the sifts."""
    return (
        a.n_imfs.shape == b.n_imfs.shape
        and np.array_equal(a.n_imfs, b.n_imfs)
        and np.array_equal(a.error, b.error)
        and a.windows.shape == b.windows.shape
        and np.array_equal(a.windows, b.windows)
        and np.array_equal(a.sifts, b.sifts)
    )


def info_equal(a, b):
    """The whole information, bit for bit."""
    return (
        decisions_equal(a, b)
        and bits(a.spacing) == bits(b.spacing)
        and (bits(a.energy) == bits(b.energy))
    )


def normal_or_zero(a):
    a = np.asarray(a)
    return bool(np.all((a == 0) | (np.abs(a) >= np.finfo(a.dtype).tiny)))


def residual_chain(x, rows, n_imfs):
    """The inputs of the modes of one element: r_0 = x, r_{k+1} = fl(r_k - y_k) in the dtype of x
    (the commit of a mode is one rounded subtraction; the prescale is a power of two)."""
    chain = [x]
    for k in range(n_imfs):
        chain.append((chain[-1] - rows[k]).astype(x.dtype))
    return chain


def has_even_plateau_at_an_extremum(line):
    """A run of an even number (at least 2) of exactly equal samples between a strict rise and a
    strict fall, or between a fall and a rise, not touching either end of the line."""
    n = line.size
    if n < 4:
        return False
    changes = np.flatnonzero(np.diff(line) != 0) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes - 1, [n - 1]))
    for s, e in zip(starts[1:-1], ends[1:-1], strict=True):
        if (e - s + 1) % 2 == 0:
            v = line[s]
            before, after = line[s - 1], line[e + 1]
            if (before < v and after < v) or (before > v and after > v):
                return True
    return False


def reconstruction_excess(x, rows, n_imfs):
    """max over the samples of |sum of the rows - x| / bound, with the bound derived from the
    commit: r_{k+1} = fl(r_k - y_k) loses at most u |r_{k+1}| per mode, so
    |x - sum_k y_k| <= u sum_{k=1}^{n} |r_k| and, as the rows give r_k only up to those losses
    (r~_k = sum_{j>=k} y_j), <= u S / (1 - P u) with S = sum_{k=1}^{n} |r~_k|; the sum of the rows
    in long double adds at most P 2**-63 sum |y_k|. ``rows`` holds the n_imfs + 1 rows of the
    element."""
    dtype = x.dtype.name
    u = EPS[dtype] / 2
    p = n_imfs + 1
    y = rows[:p].astype(np.longdouble)
    tail = np.cumsum(y[::-1], axis=0)[::-1]  # tail[k] = r~_k
    error = np.abs(tail[0] - x.astype(np.longdouble))
    s = np.abs(tail[1:]).sum(axis=0)
    bound = u * s / (1 - p * u) * (1 + 2.0**-20) + p * 2.0**-63 * np.abs(y).sum(axis=0)
    assert not error[bound == 0].any()
    return float(np.max(error[bound > 0] / bound[bound > 0], initial=0.0))


def check_element_rows(x, y, info):
    """The layout of one element's rows (spatial axes only) and the reconstruction bound."""
    n = int(info.n_imfs)
    assert y.shape[1:] == x.shape and y.shape[0] >= n + 1
    assert not y[n + 1 :].any(), "rows above the residue are zero"
    return reconstruction_excess(x, y, n)


# ----------------------------------------------------------------------------------------------
# Layout and reconstruction


@pytest.mark.parametrize(("shape", "dtype"), COMBOS, ids=IDS)
@pytest.mark.parametrize("kind", ["mix", "noise"])
def test_reconstruction_within_the_commit_rounding(shape, dtype, kind):
    x = mix(shape, 5) if kind == "mix" else np.random.default_rng(6).standard_normal(shape)
    x = x.astype(dtype)
    y, info = run(x)
    assert int(info.n_imfs) >= 1
    assert check_element_rows(x, y, info) <= 1.0


def test_layout_and_reconstruction_with_other_axes_around_the_spatial_ones():
    """The mode axis sits before the first spatial axis, whatever surrounds it; each element has
    its IMFs, its residue, then zero rows (uneven counts), and its rows add up."""
    shape = (16, 24) if SANITIZED else (20, 24)
    x = np.empty((3, shape[0], 4, shape[1]), np.float32)
    for b0, b2 in itertools.product(range(3), range(4)):
        x[b0, :, b2, :] = mix(shape, 10 * b0 + b2)
    x[0, :, 3, :] = 0.75  # no mode at all
    y, info = run(x, axes=(1, 3))
    assert info.n_imfs.shape == (3, 4) and info.n_imfs[0, 3] == 0
    assert len(set(info.n_imfs.reshape(-1).tolist())) > 1, "uneven counts"
    p = int(info.n_imfs.max()) + 1
    assert y.shape == (3, p, shape[0], 4, shape[1])
    for b0, b2 in itertools.product(range(3), range(4)):
        element = np.ascontiguousarray(y[b0, :, :, b2, :])
        el_info = run(np.ascontiguousarray(x[b0, :, b2, :]))[1]
        assert int(el_info.n_imfs) == info.n_imfs[b0, b2]
        assert check_element_rows(x[b0, :, b2, :], element, el_info) <= 1.0
    assert (y[0, 0, :, 3, :] == 0.75).all() and not y[0, 1:, :, 3, :].any()


# ----------------------------------------------------------------------------------------------
# Batch, chunk, repeat and thread invariance (bitwise)

# 16x51 and 12x23x19: sizes whose real-to-complex cuFFT plans compute neighbouring batch entries
# together (12x23x19 in float32 only); under EDA_SANITIZE test_orientation covers 16x51
if SANITIZED:
    BATCH_CASES = [((16, 16), "float32")]
    BATCH_ALONE, BATCH_CHUNKS = (0, 64), (64,)
else:
    BATCH_CASES = [((16, 16), "float32"), ((16, 16), "float64")]
    BATCH_CASES += [((16, 16, 16), "float32"), ((16, 16, 16), "float64")]
    BATCH_CASES += [((16, 51), "float32"), ((16, 51), "float64"), ((12, 23, 19), "float32")]
    BATCH_ALONE, BATCH_CHUNKS = (0, 1, 63, 64, 65), (7, 64, 65)
BATCH = 66  # two orientation groups of up to 64 elements: [0, 64) and [64, 66)


def batch_input(shape, dtype):
    x = np.stack([mix(shape, 100 + b).astype(dtype) for b in range(BATCH)])
    return x, tuple(range(1, len(shape) + 1))


def permuted(info, order):
    names = ("n_imfs", "error", "windows", "spacing", "sifts", "energy")
    return dataclasses.replace(info, **{name: getattr(info, name)[order] for name in names})


@pytest.mark.parametrize(("shape", "dtype"), BATCH_CASES, ids=ident(BATCH_CASES))
def test_batch_position_chunk_and_order_do_not_change_an_element(shape, dtype, monkeypatch):
    spy = oracle.Spy(monkeypatch, _native.core(), "decompose_host")
    x, axes = batch_input(shape, dtype)
    y, info = run(x, axes=axes)
    base_rows = spy.rows.copy()
    if len(shape) == 3:
        assert info.n_imfs.min() >= 2, "every element has IMFs beyond the first mode"
    else:
        assert info.n_imfs.min() >= 1 and len(set(info.n_imfs.tolist())) > 1
    p = y.shape[1]
    # alone, at the first and last positions of both groups
    for b in BATCH_ALONE:
        alone, alone_info = run(x[b : b + 1], axes=axes)
        n = int(info.n_imfs[b])
        assert int(alone_info.n_imfs[0]) == n
        assert bits(alone[0, : n + 1]) == bits(y[b, : n + 1])
        assert not y[b, n + 1 :].any()
        assert bits(alone_info.windows[0]) == bits(info.windows[b, :n])
        assert bits(alone_info.energy[0]) == bits(info.energy[b, :n])
        assert bits(alone_info.spacing[0]) == bits(info.spacing[b, :n])
    # chunks of 7 (ragged groups), of exactly one group, and one past it
    for chunk in BATCH_CHUNKS:
        spy.max_chunk = chunk
        yc, info_c = run(x, axes=axes)
        assert spy.chunk == chunk
        assert bits(yc) == bits(y) and info_equal(info_c, info)
        assert bits(spy.rows) == bits(base_rows)
    spy.max_chunk = 0
    # reversed and shuffled batches
    orders = [np.arange(BATCH)[::-1], np.random.default_rng(0).permutation(BATCH)]
    for order in orders[:1] if SANITIZED else orders:
        yp, info_p = run(x[order], axes=axes)
        assert bits(yp) == bits(y[order]) and p == yp.shape[1]
        assert info_equal(info_p, permuted(info, order))


# The position sweep: sizes whose real-to-complex cuFFT plans compute neighbouring batch entries
# together (the first ten, 12x23x19 in float32 only) and sizes drawn once (default_rng(2026),
# extents 2 to 199 in 2-D and 2 to 39 in 3-D).
POSITION_SHAPES = [(16, 39), (16, 51), (16, 57), (16, 65), (16, 185), (67, 65), (8, 8, 51)]
POSITION_SHAPES += [(8, 8, 39), (10, 12, 65), (12, 23, 19), (170, 37), (7, 128), (74, 94)]
POSITION_SHAPES += [(17, 75), (129, 72), (166, 158), (141, 181), (144, 37), (34, 26, 5)]
POSITION_SHAPES += [(13, 8, 38), (29, 36, 12), (26, 24, 30), (6, 21, 26), (33, 26, 19)]
POSITION_CASES = [(s, d) for d in ("float32", "float64") for s in POSITION_SHAPES]


@pytest.mark.full
@pytest.mark.parametrize(("shape", "dtype"), POSITION_CASES, ids=ident(POSITION_CASES))
def test_every_batch_position_gives_an_element_the_same_bits(shape, dtype):
    """An element at every position of a batch of two orientation groups and one element more
    (at most 129): first at the even positions with other elements at the odd ones, then the
    other way round. Every copy has the rows and the information of the element alone, bit for
    bit."""
    group = min(max((1 << 18) // int(np.prod(shape)), 1), 64)
    batch = min(2 * group + 1, 129)
    x0 = mix(shape, 3).astype(dtype)
    y0, info0 = run(x0)
    n = int(info0.n_imfs)
    assert n >= 1
    axes = tuple(range(1, len(shape) + 1))
    for parity in (0, 1):
        others = [mix(shape, 1000 + b).astype(dtype) for b in range(batch)]
        x = np.stack([x0 if b % 2 == parity else others[b] for b in range(batch)])
        y, info = run(x, axes=axes)
        for b in range(parity, batch, 2):
            assert int(info.n_imfs[b]) == n, b
            assert bits(y[b, : n + 1]) == bits(y0[: n + 1]) and not y[b, n + 1 :].any(), b
            for name in ("windows", "sifts", "spacing", "energy"):
                assert bits(getattr(info, name)[b, :n]) == bits(getattr(info0, name)[:n]), (b, name)


@pytest.mark.parametrize(("shape", "dtype"), SPARSE, ids=ident(SPARSE))
def test_repeat_and_other_geometries_in_between_do_not_change_a_result(shape, dtype):
    x = mix(shape, 21).astype(dtype)
    y, info = run(x)
    for _ in range(2):
        again, info_again = run(x)
        assert bits(again) == bits(y) and info_equal(info_again, info)
    # five other geometries and a batch in between (the plan cache holds four geometries)
    for other in range(5):
        run(mix(tuple(n + 1 + other for n in shape), 22 + other).astype(dtype))
    run(np.stack([x, x]), axes=tuple(range(1, x.ndim + 1)))
    again, info_again = run(x)
    assert bits(again) == bits(y) and info_equal(info_again, info)


def test_threads_give_the_sequential_results_in_every_dimension():
    shapes = [(40, 36), (12, 10, 9), (64, 30), (9, 9, 9)]
    inputs = [(mix(s, 20 + k).astype(np.float32), None) for k, s in enumerate(shapes)]
    inputs.append((np.stack([mix((256,), 30 + b) for b in range(5)]).astype(np.float32), 1))
    want = [run(x, axes=axes) for x, axes in inputs]

    def work(k):
        x, axes = inputs[k]
        return [run(x, axes=axes) for _ in range(3)]

    with ThreadPoolExecutor(len(inputs)) as pool:
        got = list(pool.map(work, range(len(inputs))))
    for k in range(len(inputs)):
        for y, info in got[k]:
            assert bits(y) == bits(want[k][0]) and info_equal(info, want[k][1])


# ----------------------------------------------------------------------------------------------
# Exact invariances


@pytest.mark.parametrize(("shape", "dtype", "j", "tie"), GAIN_CASES, ids=GAIN_IDS)
def test_gain_by_a_power_of_two_is_bitwise(shape, dtype, j, tie):
    """x * 2**j gives the rows * 2**j, the same windows, sifts and spacing, and the energy *
    4**j; also with the tie tolerance 0 and 2**-10, which is relative. "huge" is 2**80 (float32)
    or 2**300 (float64): the squares of float32 overflow unless the prescale takes the exponent
    out. Precondition: every value is normal or zero at both scales (a subnormal result would
    round)."""
    if isinstance(j, str):
        j = (80 if dtype == "float32" else 300) * (-1 if j.startswith("-") else 1)
    x = mix(shape, 8).astype(dtype)
    y, info = run(x, tie_tolerance=tie)
    xs = np.ldexp(x, j)
    ys, info_s = run(xs, tie_tolerance=tie)
    assert normal_or_zero(x) and normal_or_zero(xs) and normal_or_zero(y)
    assert normal_or_zero(np.ldexp(y, j)) and normal_or_zero(ys)
    assert int(info.n_imfs) >= 2
    assert bits(ys) == bits(np.ldexp(y, j))
    assert decisions_equal(info_s, info) and bits(info_s.spacing) == bits(info.spacing)
    assert bits(info_s.energy) == bits(np.ldexp(info.energy, 2 * j))


@pytest.mark.parametrize(("shape", "dtype", "kind"), SIGN_CASES, ids=SIGN_IDS)
def test_sign_gives_negated_rows(shape, dtype, kind):
    """-y(-x) == y(x) in value (so the integer ties of the quantized input too); the bytes can
    differ only where both are zero (a zero keeps no sign through the odd operators)."""
    x = (mix(shape, 9) if kind == "mix" else quantized(shape, 9)).astype(dtype)
    y, info = run(x)
    yn, info_n = run(-x)
    assert int(info.n_imfs) >= 1
    assert yn.shape == y.shape and np.array_equal(-yn, y)
    word = np.uint32 if dtype == "float32" else np.uint64
    differ = np.ascontiguousarray(-yn).view(word) != np.ascontiguousarray(y).view(word)
    assert not y[differ].any()
    assert info_equal(info_n, info)


FLIP_LINES = [(512, 3), (1536, 3)] if SANITIZED else [(512, 12), (1536, 12), (4096, 12)]


def test_flip_in_1d_is_bitwise_while_no_mode_input_has_an_even_plateau_at_an_extremum():
    """x[::-1] gives the mirrored rows, the same windows, sifts and spacing, bit for bit, and the
    energy within the rounding of a sum in any order (2 n 2**-53 relative), for every input
    whose mode inputs (rebuilt from the rows, in both runs) have no even plateau at an extremum."""
    held = total = 0
    for (n, seeds), noise, dtype in itertools.product(
        FLIP_LINES, (0.0, 1e-5, 1e-3), ("float32", "float64")
    ):
        for seed in range(seeds):
            x = slow_line(n, seed, noise, dtype)
            y, info = run(x)
            xr = np.ascontiguousarray(x[::-1])
            yr, info_r = run(xr)
            chains = residual_chain(x, y, int(info.n_imfs)) + residual_chain(
                xr, yr, int(info_r.n_imfs)
            )
            total += 1
            if any(has_even_plateau_at_an_extremum(r) for r in chains):
                continue
            held += 1
            assert bits(np.ascontiguousarray(yr[:, ::-1])) == bits(y), (n, noise, dtype, seed)
            assert decisions_equal(info_r, info) and bits(info_r.spacing) == bits(info.spacing)
            assert np.all(np.abs(info_r.energy - info.energy) <= 2 * n * 2.0**-53 * info.energy)
    assert held >= total // 4, f"the precondition held for only {held} of {total} inputs"


def test_flip_ties_form_in_the_residues():
    """The 1-D flip is NOT bitwise for every input without adjacent ties: the residue r_k =
    fl(r_{k-1} - y) can have an even plateau at an extremum when x has none, and the mirror then
    moves the plateau's peak. A recorded non-symmetry (the scipy plateau rule is kept): among 40
    inputs of 4096 float32 samples without adjacent ties, one with such a residue changes its
    rows under the flip, and the precondition check sees it."""
    found = 0
    for seed in range(40):
        x = slow_line(4096, seed, 1e-3, "float32")
        if not np.all(np.diff(x) != 0):
            continue
        y, info = run(x)
        xr = np.ascontiguousarray(x[::-1])
        yr, info_r = run(xr)
        flagged = any(
            has_even_plateau_at_an_extremum(r) for r in residual_chain(x, y, int(info.n_imfs))
        )
        differs = bits(np.ascontiguousarray(yr[:, ::-1])) != bits(y)
        assert flagged or not differs, "a flip changed rows although the precondition held"
        found += flagged and differs
    assert found >= 1


def symmetry_cases():
    for (shape, dtype), seed in itertools.product(SYMMETRY_SHAPES, SYMMETRY_SEEDS):
        yield shape, dtype, seed


SYMMETRY_PARAMS = [
    pytest.param(shape, dtype, seed, id=f"{'x'.join(map(str, shape))}-{dtype}-{seed}")
    for shape, dtype, seed in symmetry_cases()
]


# Regression ceilings of the symmetries' row differences, in eps M (P - 1). Fixed inputs and a
# fixed build give fixed values, so these are not bounds of the method: they sit at four or more
# times the largest value measured over the family (7.9 for the permutations, 7.2 for the point
# reflections except 3-D float32, 3.7 for the offset; 238 for the 3-D float32 point reflection),
# a margin for other cuFFT kernels and compilers, and catch a fault that breaks a mirror or an
# axis order (such a fault moved the rows by 2e14).
ROW_CEILING = 64.0
REFLECTION_3D_F32_CEILING = 1024.0


def row_units(y, ref, scale, dtype, p):
    """max |y - ref| in units of eps * scale * (p - 1)."""
    d = float(np.max(np.abs(y.astype(np.float64) - ref.astype(np.float64)), initial=0.0))
    return d / (EPS[dtype] * scale * max(p - 1, 1))


def report_rows(record_property, y, ref, scale, dtype, p, ceiling=ROW_CEILING):
    """Records the row difference (row_units) for the terminal summary and holds it below the
    ceiling."""
    units = row_units(y, ref, scale, dtype, p)
    record_property("symmetry_rows", units)
    assert units <= ceiling


@pytest.mark.parametrize(("shape", "dtype", "seed"), SYMMETRY_PARAMS)
def test_point_reflection_keeps_the_decisions(shape, dtype, seed, record_property):
    """The mirror of every axis: the same decisions; the row difference is reported."""
    if len(shape) == 1:
        pytest.skip("1-D is the flip test")
    x = waves(shape, seed).astype(dtype)
    y, info = run(x)
    if int(info.n_imfs) == 0:
        pytest.skip("this draw of the family has no mode to compare")
    flip = (slice(None, None, -1),) * x.ndim
    yr, info_r = run(np.ascontiguousarray(x[flip]))
    assert decisions_equal(info_r, info)
    scale = float(np.abs(x).max())
    ceiling = REFLECTION_3D_F32_CEILING if (len(shape), dtype) == (3, "float32") else ROW_CEILING
    report_rows(record_property, yr[(slice(None),) + flip], y, scale, dtype, y.shape[0], ceiling)


@pytest.mark.parametrize(("shape", "dtype", "seed"), SYMMETRY_PARAMS)
def test_axis_permutation_keeps_the_decisions(shape, dtype, seed, record_property):
    """Every other order of the axes: the same decisions; the row difference is reported."""
    x = waves(shape, seed).astype(dtype)
    y, info = run(x)
    if int(info.n_imfs) == 0:
        pytest.skip("this draw of the family has no mode to compare")
    scale = float(np.abs(x).max())
    backs = []
    for perm in itertools.permutations(range(x.ndim)):
        if perm == tuple(range(x.ndim)):
            continue
        yp, info_p = run(np.ascontiguousarray(x.transpose(perm)))
        assert decisions_equal(info_p, info), perm
        backs.append(yp.transpose((0,) + tuple(int(q) + 1 for q in np.argsort(perm))))
    ref = np.stack([y] * len(backs))
    report_rows(record_property, np.stack(backs), ref, scale, dtype, y.shape[0])


@pytest.mark.parametrize(("shape", "dtype", "seed"), SYMMETRY_PARAMS)
def test_an_offset_keeps_the_decisions_on_this_family(shape, dtype, seed, record_property):
    """x + c with c = 0.75 max|x| (rounded to the dtype): the same decisions; the difference of
    the IMFs and of the residue less c is reported. Not a symmetry in general (next test)."""
    x = waves(shape, seed).astype(dtype)
    y, info = run(x)
    if int(info.n_imfs) == 0:
        pytest.skip("this draw of the family has no mode to compare")
    c = 0.75 * float(np.abs(x).max())
    xo = (x.astype(np.float64) + c).astype(dtype)
    yo, info_o = run(xo)
    assert decisions_equal(info_o, info)
    n = int(info.n_imfs)
    shifted = np.concatenate([yo[:n], (yo[n].astype(np.float64) - c)[None]])
    scale = float(np.abs(xo).max())
    report_rows(record_property, shifted, y[: n + 1], scale, dtype, y.shape[0])


def test_an_offset_is_not_a_symmetry():
    """A recorded non-symmetry: the golden chirp with a constant offset of 0.75 max|x| takes other
    windows and sifts (the tie tolerance and the sift guard are relative to max|r|)."""
    chirp = oracle.MANIFEST.parent / "golden" / "chirp_2048.npz"
    x = np.load(chirp, allow_pickle=False)["x"]
    _, info = run(x)
    _, info_o = run(x + 0.75 * np.abs(x).max())
    assert not (
        np.array_equal(info.windows, info_o.windows) and np.array_equal(info.sifts, info_o.sifts)
    )


# ----------------------------------------------------------------------------------------------
# Special inputs


@pytest.mark.parametrize("shape", [(64,), (12, 10), (6, 5, 4)], ids=str)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("value", ["zero", "offset", "subnormal"])
def test_constants_have_no_modes_and_come_back_as_they_are(shape, dtype, value):
    v = {"zero": 0.0, "offset": 0.75, "subnormal": 3 * float(np.finfo(dtype).smallest_subnormal)}[
        value
    ]
    x = np.full(shape, v, dtype)
    y, info = run(x)
    assert int(info.n_imfs) == 0 and y.shape == (1,) + shape and bits(y[0]) == bits(x)


@pytest.mark.parametrize("shape", [(12, 10), (6, 5, 4)], ids=str)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_non_finite_values_in_2d_and_3d(shape, bad, dtype):
    good = mix(shape, 3).astype(dtype)
    x = np.stack([good, good.copy()])
    x[(1,) + (1,) * len(shape)] = bad
    axes = tuple(range(1, len(shape) + 1))
    with pytest.raises(eda.DecompositionError) as caught:
        eda.decompose(x, axes=axes)
    assert caught.value.indices.tolist() == [1] and caught.value.codes.tolist() == [1]
    y, info = run(x, axes=axes, errors="status")
    assert info.error.tolist() == [0, 1] and np.isnan(y[1]).all() and not np.isnan(y[0]).any()
    # without the finite check the failure is found later, as a non-finite result (code 2)
    y, info = run(x, axes=axes, errors="status", check_finite=False)
    assert info.error.tolist() == [0, 2] and np.isnan(y[1]).all() and not np.isnan(y[0]).any()
    assert bits(y[0, : int(info.n_imfs[0]) + 1]) == bits(run(good)[0])


# ----------------------------------------------------------------------------------------------
# max_imf and tie_tolerance


@pytest.mark.parametrize(("shape", "dtype"), COMBOS, ids=IDS)
def test_max_imf_stops_the_loop_and_changes_nothing_before_it(shape, dtype):
    """With max_imf = m < n the first m IMFs and their windows, sifts, spacing and energy are
    bitwise those of the unlimited run, and the residue is the unlimited r_m, which the unlimited
    rows give as sum_{j >= m} y_j within u sum_{k > m} |r_k| / (1 - P u) (the commit losses)."""
    x = mix(shape, 12).astype(dtype)
    y, info = run(x)
    n = int(info.n_imfs)
    assert n >= 2
    u = EPS[dtype] / 2
    tail = np.cumsum(y.astype(np.longdouble)[::-1], axis=0)[::-1]
    for m in range(1, n):
        ym, im = run(x, max_imf=m)
        assert int(im.n_imfs) == m and ym.shape[0] == m + 1
        assert bits(ym[:m]) == bits(y[:m])
        for name in ("windows", "sifts", "spacing", "energy"):
            assert bits(getattr(im, name)) == bits(getattr(info, name)[:m]), name
        bound = u * np.abs(tail[m + 1 :]).sum(axis=0) / (1 - (n + 1) * u) * (1 + 2.0**-20)
        err = np.abs(ym[m].astype(np.longdouble) - tail[m])
        assert (
            err <= bound + (n + 1) * 2.0**-63 * np.abs(y).astype(np.longdouble).sum(axis=0)
        ).all(), m


@pytest.mark.parametrize(("shape", "dtype"), SPARSE, ids=ident(SPARSE))
def test_a_large_tie_tolerance_finds_no_mode_and_the_default_is_explicit(shape, dtype):
    """The tolerance is relative to max|r|; every prominence is at most 2 max|r|, so 2.5 leaves
    the input as the residue. None equals the documented default (2**-14, 2**-30)."""
    x = mix(shape, 13).astype(dtype)
    y, info = run(x)
    assert int(info.n_imfs) >= 1
    none, none_info = run(x, tie_tolerance=2.5)
    assert int(none_info.n_imfs) == 0 and none.shape == (1,) + shape and bits(none[0]) == bits(x)
    default = 2.0**-14 if dtype == "float32" else 2.0**-30
    explicit, explicit_info = run(x, tie_tolerance=default)
    assert bits(explicit) == bits(y) and info_equal(explicit_info, info)


# ----------------------------------------------------------------------------------------------
# Device memory (CuPy: the interop environment)


@pytest.fixture(scope="module")
def cp():
    cupy = pytest.importorskip("cupy")
    yield cupy
    cupy.get_default_memory_pool().free_all_blocks()


SWEEP_SHAPES = [
    ((4, 2048), 1),
    ((256, 256), 2),
    ((40, 40, 40), 3),
    ((3, 96, 80), 2),
    ((1000,), 1),
    ((128, 128, 16), 3),
    ((1024, 1024), 2),  # four 4 MiB rows per result: a dropped device result is above the margin
]


def held_bytes(cp, memory):
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    eda.decompose(np.zeros((1, 64), np.float32), axes=1)  # drains the queued frees
    cp.cuda.runtime.deviceSynchronize()
    return memory.held()


@pytest.mark.skipif(SANITIZED, reason="the sanitizer's own tracking uses device memory")
def test_memory_does_not_grow_over_a_mixed_sweep(cp):
    """Host and CuPy input of seven shapes and both dtypes, results dropped: the device memory
    this process holds (NVML, so other programs on the GPU do not enter it) after three more
    rounds equals that after the first within 64 MiB. A dropped device result of the largest
    shape (above 40 MiB per round) makes a missing drain visible."""
    try:
        memory = device_memory.ProcessMemory(cp.cuda.Device().pci_bus_id)
    except OSError as error:
        pytest.skip(f"NVML is not available: {error}")

    def sweep():
        for shape, nd in SWEEP_SHAPES:
            for dtype in ("float32", "float64"):
                x = mix(shape, 1).astype(dtype)
                axes = tuple(range(x.ndim - nd, x.ndim))
                eda.decompose(x, axes=axes)
                device = cp.asarray(x)
                dropped = eda.decompose(device, axes=axes)
                del dropped, device

    sweep()
    level = held_bytes(cp, memory)
    for _ in range(3):
        sweep()
    growth = held_bytes(cp, memory) - level
    assert growth < 64 << 20, growth >> 20


@pytest.mark.skipif(SANITIZED, reason="the sanitizer's own tracking uses device memory")
def test_impossible_sizes_raise_memory_error_and_the_next_call_works(cp):
    """Rows for far more elements than fit, and an element whose rows alone exceed the device:
    MemoryError (not CudaError) naming the sizes; the next call returns the earlier result. A
    workspace that does not fit beside rows that do is in test_memory_error.py."""
    x = waves((4, 256), 14).astype(np.float32)
    want, want_info = run(x, axes=1)
    rows = cp.broadcast_to(cp.zeros(4096, cp.float32), (10**7, 4096))
    with pytest.raises(MemoryError, match="4096 float32 samples for 10000000 elements"):
        eda.decompose(rows, axes=1)
    after, after_info = run(x, axes=1)
    assert bits(after) == bits(want) and info_equal(after_info, want_info)
    total = cp.cuda.runtime.memGetInfo()[1]
    if total < 80 << 30:  # 11 rows of 8 GiB
        element = cp.broadcast_to(cp.zeros(1, cp.float64), (1024, 1024, 1024))
        with pytest.raises(MemoryError, match="float64 samples for 1 element"):
            eda.decompose(element)
        after, after_info = run(x, axes=1)
        assert bits(after) == bits(want) and info_equal(after_info, want_info)
