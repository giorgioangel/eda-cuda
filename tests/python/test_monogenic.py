# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
"""``eda.monogenic``: the monogenic signal of given fields.

For a field f with D spatial axes and its half-sample symmetric extension (the DCT domain of the
method), R1 = (R_a f), R2 = (R_a R_b f) and the derivatives act on the whole spectrum, every
multiplier 0 at the DC. In 2-D and 3-D n is the top eigenvector of T = R1 R1^T + R2 R2^T with
the canonical sign (its first component of magnitude above the dtype's default tie tolerance,
2**-14 or 2**-30, positive), c = (mu1 - mu2) / (mu1 + mu2); n = 1 in 1-D. With q = n . R1:
amplitude A = sqrt(f^2 + |R1|^2), phase atan2(q, f), frequency (f dq/dn - q df/dn) / (2 pi A^2)
in cycles per sample.

- Exact DCT basis fields cos(pi m (x + 1/2) / n): A = 1, frequency m / (2 n) and phase
  pi m (x + 1/2) / n at every sample, in 1-D and along one axis of 2-D and 3-D fields, where
  n is that axis' unit vector and c = 1.
- Oblique plane waves, which are not DCT basis fields: the extension bends them at the borders,
  so the values hold in the interior only, two periods from every border, within the measured
  border effect.
- A NumPy float64 reference (the FFT of the extension, the same definitions) on smooth random
  fields: every output agrees within a few eps of the field's scale, with the error measures
  of ``compare`` (the phase weighted by A, the frequency by A^2, n by the eigenvalue gap and c by
  mu1 + mu2: each output is only as well defined as that weight).
- Bitwise: an element alone and at any position of a batch (across the transforms' groups of
  64 or 49 elements), any chunking, host and CUDA input, and a gain by a power of two (A scales;
  the phase, the frequency, n and c keep their bits: the fields are prescaled to max |x| in
  [1, 2)).

Tolerances are in eps of the compute dtype (2**-52, 2**-23) and were measured with this build on
an RTX 4090 (2026-10-16); each sits at about eight times the largest value measured.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

import eda
from eda import _native

pytestmark = pytest.mark.gpu

EPS = {"float32": float(np.finfo(np.float32).eps), "float64": float(np.finfo(np.float64).eps)}
ZERO = {"float32": 2.0**-14, "float64": 2.0**-30}  # the canonical sign's zero (tie tolerance)
NAMES = ("amplitude", "phase", "frequency", "direction", "coherence")


def bits(a):
    return None if a is None else np.ascontiguousarray(a).tobytes()


def all_bits(r):
    return [bits(getattr(r, name)) for name in NAMES]


def wrap(d):
    return np.abs(np.angle(np.exp(1j * np.asarray(d, dtype=np.float64))))


def basis(n, m):
    """The DCT basis field cos(pi m (x + 1/2) / n) and its phase, the angle reduced exactly to
    [0, 2 pi) first (an unreduced angle of hundreds of radians carries its own rounding)."""
    theta = np.pi * ((m * (2 * np.arange(n) + 1)) % (4 * n)) / (2 * n)
    return np.cos(theta), theta


def smooth(shape, seed, offset=0.3):
    """Three plane waves of random directions (periods 5, 9 and 23) plus an offset, in float64."""
    rng = np.random.default_rng(seed)
    grids = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij")
    out = np.full(shape, float(offset))
    for period in (5.0, 9.0, 23.0):
        k = rng.normal(size=len(shape))
        k = k / np.linalg.norm(k) * 2 * np.pi / period
        phase = sum(ki * g for ki, g in zip(k, grids, strict=True)) + rng.uniform(0, 6)
        out += rng.uniform(0.5, 1.0) * np.sin(phase)
    return out


def canonical(n, zero):
    """n (components on axis 0) with the first component above ``zero`` positive."""
    sign = np.zeros(n.shape[1:])
    for a in range(n.shape[0]):
        sign = np.where((sign == 0) & (np.abs(n[a]) > zero), np.sign(n[a]), sign)
    return n * np.where(sign < 0, -1.0, 1.0)


def reference(f, zero):
    """The monogenic signal of one field by the FFT of its half-sample symmetric extension, in
    float64: the outputs, and mu1, mu2 of T (None in 1-D)."""
    f = np.asarray(f, dtype=np.float64)
    D = f.ndim
    ext = f
    for a in range(D):
        ext = np.concatenate([ext, np.flip(ext, axis=a)], axis=a)
    F = np.fft.fftn(ext)
    u = np.meshgrid(*[np.fft.fftfreq(2 * n) for n in f.shape], indexing="ij")
    rad = np.sqrt(sum(v * v for v in u))
    inv = np.divide(1.0, rad, out=np.zeros_like(rad), where=rad > 0)
    crop = tuple(slice(0, n) for n in f.shape)

    def back(multiplier):
        return np.real(np.fft.ifftn(F * multiplier))[crop]

    pairs = [(a, b) for a in range(D) for b in range(D)]
    r1 = np.stack([back(-1j * u[a] * inv) for a in range(D)])  # R_a: -i u_a / |u|
    df = np.stack([back(1j * u[a]) for a in range(D)])  # d/dx_a / (2 pi): i u_a
    dr = {(a, b): back(u[a] * u[b] * inv) for a, b in pairs}  # D_b R_a: u_a u_b / |u|
    out = {}
    if D == 1:
        n = np.ones((1,) + f.shape)
        out["mu"] = None
    else:
        h = np.zeros(f.shape + (D, D))
        for a, b in pairs:
            h[..., a, b] = back(-u[a] * u[b] * inv**2)  # R_a R_b: -u_a u_b / |u|^2
        T = np.einsum("a...,b...->...ab", r1, r1) + h @ h
        w, V = np.linalg.eigh(T)
        n = canonical(np.moveaxis(V[..., :, -1], -1, 0), zero)
        s = w[..., -1] + w[..., -2]
        c = np.divide(w[..., -1] - w[..., -2], s, out=np.zeros_like(s), where=s > 0)
        out.update(direction=n, coherence=np.clip(c, 0, 1), mu=(w[..., -1], w[..., -2]))
    q = (n * r1).sum(axis=0)
    a2 = f * f + (r1 * r1).sum(axis=0)
    dnq = sum(n[a] * n[b] * dr[(a, b)] for a, b in pairs)
    dnf = (n * df).sum(axis=0)
    out.update(amplitude=np.sqrt(a2), phase=np.arctan2(q, f), frequency=(f * dnq - q * dnf) / a2)
    return out


def compare(r, ref, scale, eps):
    """The errors of one result against the reference, in eps of ``scale`` (max |f|): A; the
    phase times A; the frequency times A^2 (over scale^2); n times the gap mu1 - mu2 and c times
    mu1 + mu2 (over max mu1)."""
    A = ref["amplitude"]
    errors = {
        "amplitude": np.abs(r.amplitude - A).max() / scale,
        "phase": (wrap(r.phase - ref["phase"]) * A).max() / scale,
        "frequency": (np.abs(r.frequency - ref["frequency"]) * A**2).max() / scale**2,
    }
    if ref["mu"] is not None:
        mu1, mu2 = ref["mu"]
        dn = np.abs(r.direction - ref["direction"]).max(axis=0)
        errors["direction"] = (dn * (mu1 - mu2)).max() / mu1.max()
        errors["coherence"] = (
            np.abs(r.coherence - ref["coherence"]) * (mu1 + mu2)
        ).max() / mu1.max()
    return {k: float(v) / eps for k, v in errors.items()}


# ----------------------------------------------------------------------------------------------
# DCT basis fields: exact values at every sample

# Measured (float64 / float32, in eps; the frequency in eps of 1 cycle per sample): |A - 1| 7.5 /
# 3; frequency 2 / 2.3; phase 5.1 / 3.6; n 0.5 / 0.6; 1 - c 0 / 0.
BASIS_TOL = {"amplitude": 64, "frequency": 24, "phase": 48, "direction": 8, "coherence": 4}


def check_basis(r, theta, m, n, dtype, axis=None):
    eps = EPS[dtype]
    assert np.abs(r.amplitude - 1).max() <= BASIS_TOL["amplitude"] * eps
    assert np.abs(r.frequency - m / (2 * n)).max() <= BASIS_TOL["frequency"] * eps
    assert wrap(r.phase - theta).max() <= BASIS_TOL["phase"] * eps
    if axis is not None:
        unit = np.zeros(r.direction.shape[0])
        unit[axis] = 1
        target = unit.reshape((-1,) + (1,) * (r.direction.ndim - 1))
        assert np.abs(r.direction - target).max() <= BASIS_TOL["direction"] * eps
        assert np.abs(r.coherence - 1).max() <= BASIS_TOL["coherence"] * eps


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize(("n", "m"), [(16, 1), (37, 5), (64, 21), (255, 254), (1000, 3)])
def test_dct_basis_1d(n, m, dtype):
    f, theta = basis(n, m)
    r = eda.monogenic(f.astype(dtype))
    assert r.direction is None and r.coherence is None
    assert r.amplitude.dtype == np.dtype(dtype) and r.amplitude.shape == (n,)
    check_basis(r, theta, m, n, dtype)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("shape", [(12, 20), (6, 9, 10)], ids=["2d", "3d"])
def test_dct_basis_along_one_axis(shape, dtype):
    for axis, n in enumerate(shape):
        m = n // 3 + 1
        f, theta = basis(n, m)
        view = [1] * len(shape)
        view[axis] = n
        x = np.broadcast_to(f.reshape(view), shape).astype(dtype)
        r = eda.monogenic(x)
        assert r.direction.shape == (len(shape),) + shape and r.coherence.shape == shape
        check_basis(r, np.broadcast_to(theta.reshape(view), shape), m, n, dtype, axis)


# ----------------------------------------------------------------------------------------------
# Oblique plane waves: the values in the interior

# The interior starts two periods from every border. Measured there, the same in both dtypes:
# 2-D |A - 1| 3.2e-3, |frequency - |k|| 1.9e-4, |n - k/|k|| 3.3e-3, 1 - c 8.9e-5, |phase - theta|
# 3.7e-3; 3-D 3.0e-4, 1.0e-4, 3.4e-4, 9.5e-7, 3.4e-4. The tolerances are about twice those.
PLANE = {
    2: ((128, 128), (0.031, 0.047), (6.5e-3, 4e-4, 6.6e-3, 1.8e-4, 7.5e-3)),
    3: ((48, 48, 48), (0.05, 0.07, 0.09), (6e-4, 2e-4, 7e-4, 2e-6, 7e-4)),
}


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("dim", [2, 3])
def test_oblique_plane_wave(dim, dtype):
    shape, k, (tol_a, tol_f, tol_n, tol_c, tol_p) = PLANE[dim]
    k = np.asarray(k)
    grids = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij")
    theta = 2 * np.pi * sum(ki * g for ki, g in zip(k, grids, strict=True)) + 0.3
    r = eda.monogenic(np.cos(theta).astype(dtype))
    margin = int(np.ceil(2 / np.linalg.norm(k)))
    inner = tuple(slice(margin, n - margin) for n in shape)
    khat = (k / np.linalg.norm(k)).reshape((-1,) + (1,) * dim)
    assert np.abs(r.amplitude[inner] - 1).max() <= tol_a
    assert np.abs(r.frequency[inner] - np.linalg.norm(k)).max() <= tol_f
    assert np.abs(r.direction[(slice(None),) + inner] - khat).max() <= tol_n
    assert np.abs(r.coherence[inner] - 1).max() <= tol_c
    assert wrap(r.phase[inner] - theta[inner]).max() <= tol_p


# ----------------------------------------------------------------------------------------------
# Against the NumPy reference

# Measured (float64 / float32, in eps, over the three seeds): 1-D amplitude 2.4 / 2.0, phase
# 2.9 / 1.6, frequency 0.6 / 0.6; 2-D 2.2 / 2.2, 15.6 / 13.2, 0.8 / 0.8, direction 2.0 / 1.6,
# coherence 3.1 / 3.3; 3-D 3.3 / 2.5, 62.9 / 40.7, 2.0 / 1.5, 3.8 / 1.6, 6.5 / 2.8. The phase is
# also as uncertain as n where the gap is small, which the weight A does not cover.
REFERENCE_TOL = {"amplitude": 32, "phase": 512, "frequency": 16, "direction": 32, "coherence": 64}


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("shape", [(300,), (37, 45), (17, 20, 23)], ids=["1d", "2d", "3d"])
def test_matches_the_numpy_reference(shape, dtype, seed):
    x = smooth(shape, seed).astype(dtype)
    r = eda.monogenic(x)
    errors = compare(r, reference(x, ZERO[dtype]), float(np.abs(x).max()), EPS[dtype])
    assert len(errors) == (3 if len(shape) == 1 else 5)
    for name, value in errors.items():
        assert value <= REFERENCE_TOL[name], (name, value)


# ----------------------------------------------------------------------------------------------
# Layout, special inputs, the canonical sign


def test_layout_follows_the_axes():
    """Batch axes anywhere and spatial axes in any order: element b is the monogenic signal of
    x[:, b, :] seen in the order of axes, bitwise; the direction axis sits before min(axes)."""
    x = smooth((8, 3, 9), 4)
    r = eda.monogenic(x, axes=(2, 0))
    assert r.amplitude.shape == x.shape and r.coherence.shape == x.shape
    assert r.direction.shape == (2,) + x.shape
    for b in range(3):
        one = eda.monogenic(np.ascontiguousarray(x[:, b, :].T))
        for name in ("amplitude", "phase", "frequency", "coherence"):
            assert bits(getattr(r, name)[:, b, :]) == bits(getattr(one, name).T)
        assert bits(r.direction[:, :, b, :]) == bits(one.direction.transpose(0, 2, 1))
    lines = smooth((4, 100), 5)
    r = eda.monogenic(lines, axes=1)
    assert r.amplitude.shape == (4, 100) and r.direction is None
    assert bits(r.phase[2]) == bits(eda.monogenic(lines[2]).phase)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("shape", [(50,), (9, 13), (5, 6, 7)], ids=["1d", "2d", "3d"])
def test_constant_and_zero_fields(shape, dtype):
    for value in (0.0, 0.75, -32.0):
        r = eda.monogenic(np.full(shape, value, dtype=dtype))
        assert (r.amplitude == abs(value)).all()
        pi = np.asarray(np.pi, dtype)
        assert (r.phase == (pi if value < 0 else 0)).all() and (r.frequency == 0).all()
        if len(shape) > 1:
            assert (r.direction[0] == 1).all() and (r.direction[1:] == 0).all()
            assert (r.coherence == 0).all()


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_outputs_stay_in_range(dtype):
    x = smooth((40, 36), 6).astype(dtype)
    r = eda.monogenic(x)
    assert (r.amplitude >= 0).all() and (r.coherence >= 0).all() and (r.coherence <= 1).all()
    pi = np.asarray(np.pi, dtype)  # the phase lies in (-pi, pi] as the dtype rounds pi
    assert (r.phase > -pi).all() and (r.phase <= pi).all()
    norm = np.sqrt((r.direction.astype(np.float64) ** 2).sum(axis=0))
    assert np.abs(norm - 1).max() <= 8 * EPS[dtype]
    first = np.where(np.abs(r.direction[0]) > ZERO[dtype], r.direction[0], r.direction[1])
    assert (first > 0).all(), "the canonical sign"


# ----------------------------------------------------------------------------------------------
# Bitwise invariances


@pytest.mark.parametrize(
    ("shape", "dtype", "batch", "positions"),
    [
        ((1000,), "float64", 70, (0, 63, 64, 69)),
        ((16, 51), "float32", 70, (0, 63, 64, 69)),
        ((12, 23, 19), "float64", 52, (0, 48, 49, 51)),
    ],
    ids=["1d", "2d", "3d"],
)
def test_an_element_does_not_see_the_batch(shape, dtype, batch, positions):
    """The transforms run in groups of 64 elements (49 for 12x23x19); the positions sit at the
    first and last place of each group."""
    x = smooth(shape, 7).astype(dtype)
    alone = all_bits(eda.monogenic(x))
    stack = np.stack([smooth(shape, 20 + b, offset=b).astype(dtype) for b in range(batch)])
    for p in positions:
        stack[p] = x
    axes = tuple(range(1, len(shape) + 1))
    r = eda.monogenic(stack, axes=axes)
    for p in positions:
        got = [getattr(r, name) for name in NAMES]
        assert [bits(None if a is None else a[p]) for a in got] == alone


def test_chunks_do_not_change_the_bits(monkeypatch):
    """Chunks of 3 and 7 elements (max_chunk, the last argument of the native call, a test hook)
    give the bits of one chunk; the native call reports the chunk it used."""
    x = np.stack([smooth((20, 18), 70 + b) for b in range(10)]).astype(np.float32)
    core = _native.core()
    original = core.monogenic_host
    used = []

    def spy(*args, cap=0):
        flat, info = original(*args[:-1], cap)
        used.append(int(info["chunk"]))
        return flat, info

    monkeypatch.setattr(core, "monogenic_host", spy)
    whole = all_bits(eda.monogenic(x, axes=(1, 2)))
    for chunk in (3, 7):
        monkeypatch.setattr(core, "monogenic_host", lambda *a, c=chunk: spy(*a, cap=c))
        assert all_bits(eda.monogenic(x, axes=(1, 2))) == whole
    assert used == [10, 3, 7]


@pytest.mark.parametrize("j", [-8, 3, 20, "huge"])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("shape", [(300,), (30, 28), (12, 11, 10)], ids=["1d", "2d", "3d"])
def test_gain_by_a_power_of_two(shape, dtype, j):
    if j == "huge":
        j = 80 if dtype == "float32" else 300
    x = smooth(shape, 8).astype(dtype)
    r = eda.monogenic(x)
    g = eda.monogenic(x * np.asarray(2.0**j, dtype=dtype))
    assert bits(g.amplitude) == bits(r.amplitude * np.asarray(2.0**j, dtype=dtype))
    for name in NAMES[1:]:
        assert bits(getattr(g, name)) == bits(getattr(r, name))


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_cuda_input_gives_the_host_bits(dtype):
    cp = pytest.importorskip("cupy")
    for shape in [(2, 300), (2, 30, 28), (2, 12, 11, 10)]:
        x = np.stack([smooth(shape[1:], 9 + b) for b in range(shape[0])]).astype(dtype)
        axes = tuple(range(1, len(shape)))
        host = eda.monogenic(x, axes=axes)
        device = eda.monogenic(cp.asarray(x), axes=axes)
        for name in NAMES:
            h, d = getattr(host, name), getattr(device, name)
            if h is None:
                assert d is None
                continue
            assert isinstance(d, eda.DeviceArray) and d.shape == h.shape and d.dtype == h.dtype
            assert bits(cp.asnumpy(cp.asarray(d))) == bits(h)
    # a converted input (another dtype, a batch axis between the spatial ones) and a strided one
    x = (100 * smooth((20, 3, 18), 10)).astype(np.int16)
    host = eda.monogenic(x, axes=(0, 2), dtype=dtype)
    device = eda.monogenic(cp.asarray(x), axes=(0, 2), dtype=dtype)
    strided = eda.monogenic(
        cp.asarray(np.repeat(x, 2, axis=2).astype(dtype))[:, :, ::2], axes=(0, 2)
    )
    for name in NAMES:
        assert bits(cp.asnumpy(cp.asarray(getattr(device, name)))) == bits(getattr(host, name))
        assert bits(cp.asnumpy(cp.asarray(getattr(strided, name)))) == bits(getattr(host, name))


# ----------------------------------------------------------------------------------------------
# Arguments and errors

X = np.zeros((4, 16), np.float32)

ARGUMENT_ERRORS = [
    (dict(check_finite=1), TypeError, "eda: check_finite must be True or False, got int"),
    (dict(check_finite=None), TypeError, "check_finite must be True or False, got NoneType"),
    (dict(stream=1.5), TypeError, "eda: stream must be an integer CUDA stream handle"),
    (dict(stream=True), TypeError, "stream must be an integer CUDA stream handle"),
    (dict(stream=-1), ValueError, "eda: stream handle -1 is negative"),
    (dict(dtype="float16"), TypeError, "eda: dtype= must be float32 or float64, got float16"),
    (dict(dtype="int32"), TypeError, "dtype= must be float32 or float64, got int32"),
    (dict(axes="ab"), TypeError, "eda: axes must be an int, a sequence of ints or None, got str"),
    (dict(axes=True), TypeError, "axes must be an int, a sequence of ints or None, got bool"),
    (dict(axes=(0, 0)), ValueError, "eda: axes must be unique, got (0, 0)"),
    (dict(axes=()), ValueError, "eda: 1 to 3 spatial axes, got 0"),
    (dict(axes=5), ValueError, "eda: axis 5 is out of bounds for array of dimension 2"),
]


@pytest.mark.parametrize(("kwargs", "error", "fragment"), ARGUMENT_ERRORS)
def test_argument_errors(kwargs, error, fragment):
    with pytest.raises(error, match=re.escape(fragment)):
        eda.monogenic(X, **kwargs)


@pytest.mark.parametrize(
    ("x", "kwargs", "error", "fragment"),
    [
        (np.zeros((4, 16), np.int32), {}, TypeError, "expected a float32 or float64 array"),
        (np.zeros((4, 16), np.complex64), {}, TypeError, "dtype complex64 are not supported"),
        (np.zeros((), np.float32), {}, ValueError, "expected an array with at least one dimension"),
        (np.zeros((2, 2, 2, 2), np.float32), {}, ValueError, "needs x.ndim <= 3, got 4"),
        (np.zeros((4, 1), np.float32), {"axes": 1}, ValueError, "needs at least 2 samples"),
        (np.zeros((3, 1)), {}, ValueError, "spatial axis 1 has extent 1"),
        (np.zeros((2, 9000), np.float32), {"axes": 1}, ValueError, "exceeds the limit 8192"),
        (np.zeros((2, 5000)), {"axes": 1}, ValueError, "exceeds the limit 4096 for float64"),
    ],
)
def test_input_errors(x, kwargs, error, fragment):
    with pytest.raises(error, match=re.escape(fragment)):
        eda.monogenic(x, **kwargs)


def test_integer_input_is_cast_with_dtype():
    x = np.arange(64, dtype=np.int32) % 7
    r = eda.monogenic(x, dtype="float32")
    assert bits(r.amplitude) == bits(eda.monogenic(x.astype(np.float32)).amplitude)


def test_non_finite_input():
    x = np.stack([smooth((40,), 11), smooth((40,), 12)])
    x[1, 5] = np.nan
    with pytest.raises(ValueError, match=re.escape("inf or NaN at batch index 1 (1 of 2")):
        eda.monogenic(x, axes=1)
    r = eda.monogenic(x, axes=1, check_finite=False)
    assert bits(r.amplitude[0]) == bits(eda.monogenic(x[0]).amplitude)
    assert np.isnan(r.amplitude[1]).all()


def test_an_empty_batch():
    r = eda.monogenic(np.zeros((0, 3, 16), np.float32), axes=(1, 2))
    assert r.amplitude.shape == (0, 3, 16) and r.direction.shape == (0, 2, 3, 16)
    assert r.amplitude.dtype == np.float32


def test_the_result_is_frozen():
    r = eda.monogenic(smooth((30,), 1))
    with pytest.raises(AttributeError):
        r.phase = None  # type: ignore[misc]
