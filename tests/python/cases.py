# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
"""Deterministic inputs of the parity cases (tests/data/parity_v1.json).

Every input is built from a recipe. Random numbers are the raw 64-bit words of NumPy's PCG64 bit
generator (seeded by a SeedSequence of the recipe name's SHA-256 and the batch element's index),
turned into doubles in [0, 1) by a shift and an exact scaling. The fields use only elementwise
+ - * /, minimum, ldexp, comparisons and astype on arrays, correctly rounded Python float
arithmetic (and math.sqrt) on scalars, data movement (roll, reshape, transpose), and the
polynomial cosine, sine and exponential of this file. No NumPy transcendental or reduction kernel
is involved, so the bytes of an input depend on IEEE arithmetic only; every input regenerates
bitwise with NumPy 2.5.3 and 2.5.4 (the manifest's input_sha256 is the check).

A case is a recipe and a dtype: the field is built in float64 (optionally rounded down to a
quantum), scaled (gain, offset), cast to the dtype, and then the special elements are written
(NaN, inf, constant elements). The batch elements are numbered in C order over the batch axes.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

DTYPES = ("float32", "float64")
_SHORT = {"float32": "f32", "float64": "f64"}

# ----------------------------------------------------------------------------------------------
# Elementwise polynomial functions (float64)

_TWO_PI = 6.283185307179586  # the double nearest 2 pi
_TWO_PI_LO = 2.4492935982947064e-16  # 2 pi - _TWO_PI
_INV_TWO_PI = 0.15915494309189535
_LN2_HI = 0.6931471805599453
_LN2_LO = 2.3190468138462996e-17
_INV_LN2 = 1.4426950408889634
# Taylor coefficients (Python's int / int division is correctly rounded)
_COS_C = tuple((-1) ** k / math.factorial(2 * k) for k in range(16))
_SIN_C = tuple((-1) ** k / math.factorial(2 * k + 1) for k in range(16))
_EXP_C = tuple((-1) ** k / math.factorial(k) for k in range(23))


def _floor(t: np.ndarray) -> np.ndarray:
    """floor(t) for |t| < 2^62 by truncation and a correction."""
    n = t.astype(np.int64).astype(np.float64)
    return n - (n > t).astype(np.float64)


def _horner(coeffs: tuple[float, ...], z: np.ndarray) -> np.ndarray:
    p = np.full(z.shape, coeffs[-1])
    for c in coeffs[-2::-1]:
        p = p * z + c
    return p


def _reduce(x: np.ndarray) -> np.ndarray:
    """x - 2 pi round(x / 2 pi), in [-pi, pi] up to rounding."""
    n = _floor(x * _INV_TWO_PI + 0.5)
    return (x - n * _TWO_PI) - n * _TWO_PI_LO


def pcos(x: Any) -> np.ndarray:
    """cos(x) to about 1e-16 (Taylor series of degree 30 after reduction to [-pi, pi])."""
    r = _reduce(np.asarray(x, dtype=np.float64))
    return _horner(_COS_C, r * r)


def psin(x: Any) -> np.ndarray:
    """sin(x) to about 1e-16 (Taylor series of degree 31 after reduction to [-pi, pi])."""
    r = _reduce(np.asarray(x, dtype=np.float64))
    return r * _horner(_SIN_C, r * r)


def pexpm(t: Any) -> np.ndarray:
    """exp(-t) for t >= 0: exp(-r) 2^-k with t = k ln 2 + r, 0 <= r < ln 2 (up to rounding)."""
    t = np.minimum(np.asarray(t, dtype=np.float64), 1400.0)
    k = _floor(t * _INV_LN2)
    r = (t - k * _LN2_HI) - k * _LN2_LO
    return np.ldexp(_horner(_EXP_C, r), -k.astype(np.int32))


# ----------------------------------------------------------------------------------------------
# Random numbers


class Uniform:
    """Doubles in [0, 1) from the raw words of a PCG64 bit generator: (word >> 11) 2^-53."""

    def __init__(self, *key: int) -> None:
        self._bits = np.random.PCG64(np.random.SeedSequence(list(key)))

    def array(self, shape: tuple[int, ...]) -> np.ndarray:
        size = math.prod(shape)
        words = self._bits.random_raw(size) if size else np.zeros(0, np.uint64)
        u = (np.asarray(words, dtype=np.uint64) >> np.uint64(11)).astype(np.float64)
        return (u * 2.0**-53).reshape(shape)

    def scalar(self) -> float:
        return float(self.array((1,))[0])

    def between(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.scalar()

    def normal(self, shape: tuple[int, ...]) -> np.ndarray:
        """Unit-variance noise: sqrt(3) times the sum of four uniforms minus 2 (Irwin-Hall)."""
        s = self.array(shape)
        for _ in range(3):
            s = s + self.array(shape)
        return (s - 2.0) * math.sqrt(3.0)

    def direction(self, dim: int) -> list[float]:
        """A unit vector (the normalized Irwin-Hall draws; never the zero vector)."""
        while True:
            g = [float(v) for v in self.normal((dim,))]
            norm2 = 0.0
            for v in g:
                norm2 += v * v
            if norm2 > 1e-6:
                norm = math.sqrt(norm2)
                return [v / norm for v in g]


def recipe_seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")


# ----------------------------------------------------------------------------------------------
# Fields: f(n, u, **params) -> float64 array of shape n (the spatial extents in the order of axes)


def _coords(n: tuple[int, ...]) -> list[np.ndarray]:
    out = []
    for a, m in enumerate(n):
        shape = [1] * len(n)
        shape[a] = m
        out.append(np.arange(m, dtype=np.int64).astype(np.float64).reshape(shape))
    return out


def _dot(d: list[float], c: list[np.ndarray], n: tuple[int, ...]) -> np.ndarray:
    acc = np.zeros(n)
    for a in range(len(n)):
        acc = acc + c[a] * d[a]
    return acc


def _dist2(center: list[float], c: list[np.ndarray], n: tuple[int, ...]) -> np.ndarray:
    acc = np.zeros(n)
    for a in range(len(n)):
        e = c[a] - center[a]
        acc = acc + e * e
    return acc


def _wave(n, c, u, period, amp=1.0):
    d = u.direction(len(n))
    phi = _TWO_PI * u.scalar()
    return amp * pcos(_dot(d, c, n) * (_TWO_PI / period) + phi)


def f_constant(n, u, value=0.75):
    return np.full(n, float(value))


def f_noise(n, u, smooth=0):
    v = u.normal(n)
    for _ in range(smooth):
        for a in range(len(n)):
            v = (np.roll(v, -1, axis=a) + v + np.roll(v, 1, axis=a)) / 3.0
    return v


def f_tones(n, u, count=3, base=5.0, ratio=3.0):
    c = _coords(n)
    v = np.zeros(n)
    for j in range(count):
        period = base * ratio**j * u.between(0.8, 1.2)
        v = v + _wave(n, c, u, period, u.between(0.5, 1.5))
    return v


def f_chirp(n, u, f0=0.01, f1=0.3):
    c = _coords(n)
    d = u.direction(len(n))
    t = _dot(d, c, n)
    span = float(max(n))
    a = f0 * u.between(0.5, 1.5)
    b = f1 * u.between(0.85, 1.15)
    phase = _TWO_PI * (a * t + 0.5 * (b - a) * t * t / span) + _TWO_PI * u.scalar()
    return pcos(phase) + 0.3 * _wave(n, c, u, 0.6 * span)


def f_amfm(n, u, period=7.0):
    c = _coords(n)
    d = u.direction(len(n))
    t = _dot(d, c, n)
    p = period * u.between(0.8, 1.3)
    am = 1.0 + 0.6 * pcos(t * (_TWO_PI / (8.0 * p)))
    fm = 2.0 * psin(t * (_TWO_PI / (5.0 * p)))
    return am * pcos(t * (_TWO_PI / p) + fm + _TWO_PI * u.scalar()) + _wave(n, c, u, 25.0 * p, 0.5)


def f_blobs(n, u, count=6, size=(0.05, 0.17)):
    c = _coords(n)
    v = np.zeros(n)
    small = float(min(n))
    for _ in range(count):
        center = [u.between(0.0, float(m)) for m in n]
        s = max(1.0, small * u.between(size[0], size[1]))
        amp = u.between(0.5, 1.5) * (1.0 if u.scalar() < 0.6 else -1.0)
        v = v + amp * pexpm(_dist2(center, c, n) / (2.0 * s * s))
    return v


def f_fringes(n, u):
    c = _coords(n)
    return _wave(n, c, u, u.between(4.0, 7.0)) + _wave(n, c, u, u.between(14.0, 24.0), 0.7)


def f_gabor(n, u, count=3, background=0.3, p_lo=3.0, p_hi=8.0, width=(1.5, 3.0)):
    c = _coords(n)
    v = np.zeros(n)
    for _ in range(count):
        center = [u.between(0.15, 0.85) * m for m in n]
        p = u.between(p_lo, p_hi)
        s = u.between(width[0], width[1]) * p
        d = u.direction(len(n))
        x0 = [-center[a] for a in range(len(n))]
        t = _dot(d, c, n)
        for a in range(len(n)):
            t = t + x0[a] * d[a]
        env = pexpm(_dist2(center, c, n) / (2.0 * s * s))
        v = v + u.between(0.6, 1.4) * env * pcos(t * (_TWO_PI / p) + _TWO_PI * u.scalar())
    return v + _wave(n, c, u, 0.8 * float(max(n)), background)


def f_edges(n, u, count=4, noise=0.02):
    c = _coords(n)
    v = np.zeros(n)
    for _ in range(count):
        d = u.direction(len(n))
        cut = 0.0
        for a in range(len(n)):
            cut += d[a] * u.between(0.2, 0.8) * n[a]
        step = (_dot(d, c, n) > cut).astype(np.float64)
        v = v + u.between(-1.0, 1.0) * step
    return v + noise * u.normal(n)


def f_flat(n, u, period=5.0):
    """A tone inside a random box, zero outside, and one step: flat regions next to oscillation."""
    c = _coords(n)
    inside = np.ones(n, dtype=bool)
    for a, m in enumerate(n):
        lo = u.between(0.1, 0.35) * m
        hi = u.between(0.6, 0.9) * m
        inside = inside & (c[a] >= lo) & (c[a] < hi)
    tone = _wave(n, c, u, period * u.between(0.8, 1.2))
    d = u.direction(len(n))
    mid = 0.0
    for a, m in enumerate(n):
        mid += 0.5 * d[a] * m
    step = (_dot(d, c, n) > mid).astype(np.float64)
    return tone * inside.astype(np.float64) + 0.5 * step


def f_local(n, u, fine=5.0, frac=(0.35, 0.65), coarse_period=0.45):
    """A coarse wave everywhere and a fine wave over a part of the first axis only."""
    c = _coords(n)
    coarse = _wave(n, c, u, coarse_period * float(max(n)) * u.between(0.8, 1.2))
    cut = u.between(frac[0], frac[1]) * n[0]
    part = (c[0] < cut).astype(np.float64) * np.ones(n)
    return coarse + part * _wave(n, c, u, fine * u.between(0.8, 1.2), 0.7)


def f_growth(n, u, frac=(0.55, 0.62), fine=4.5, noise=0.02):
    """A fine wave, a wave along the first axis whose period is a bit over half its extent, and
    weak noise: the window after the coarse wave would have to grow past the extent."""
    c = _coords(n)
    p = u.between(frac[0], frac[1]) * n[0]
    coarse = pcos(c[0] * (_TWO_PI / p) + _TWO_PI * u.scalar()) * np.ones(n)
    return _wave(n, c, u, fine * u.between(0.9, 1.1), 0.4) + coarse + noise * u.normal(n)


def f_zone(n, u, top=0.35):
    """A zone plate: the local frequency grows linearly to `top` cycles per sample at the rim."""
    c = _coords(n)
    center = [0.5 * m + u.between(-1.0, 1.0) for m in n]
    rim = 0.0
    for m in n:
        rim += 0.25 * m * m
    rim = math.sqrt(rim)
    phase = _dist2(center, c, n) * (_TWO_PI * 0.5 * top / rim) + _TWO_PI * u.scalar()
    return pcos(phase)


def f_mixture(n, u):
    c = _coords(n)
    d = u.direction(len(n))
    trend = _dot(d, c, n) * (0.5 / float(max(n)))
    return f_tones(n, u, count=2) + f_blobs(n, u, count=2) + 0.15 * u.normal(n) + trend


FAMILIES = {
    "constant": f_constant,
    "noise": f_noise,
    "tones": f_tones,
    "chirp": f_chirp,
    "amfm": f_amfm,
    "blobs": f_blobs,
    "fringes": f_fringes,
    "gabor": f_gabor,
    "edges": f_edges,
    "flat": f_flat,
    "local": f_local,
    "zone": f_zone,
    "growth": f_growth,
    "mixture": f_mixture,
}

# ----------------------------------------------------------------------------------------------
# Recipes


@dataclass(frozen=True)
class Recipe:
    name: str
    family: str
    shape: tuple[int, ...]
    axes: tuple[int, ...] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    dtypes: tuple[str, ...] = DTYPES
    max_imf: int = -1
    tie_tolerance: float | None = None
    check_finite: bool = True
    errors: str = "raise"
    max_chunk: int | None = None  # the most elements per chunk (a private test option)
    gain: float | None = None
    gain_pow2: int | None = None  # x 2^k, exact
    offset: float | None = None
    nan_at: tuple[int, ...] = ()  # elements with one NaN sample
    inf_at: tuple[int, ...] = ()  # elements with one +inf sample
    constant_at: tuple[int, ...] = ()  # elements replaced by the constant 0.75
    seed_from: str | None = None  # build the field from this recipe name's seed (same field)
    quantum: float | None = None  # round the field down to multiples of this (ties)
    quick: bool = False

    def spatial_axes(self) -> tuple[int, ...]:
        return tuple(range(len(self.shape))) if self.axes is None else self.axes

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["shape"] = list(self.shape)
        d["axes"] = None if self.axes is None else list(self.axes)
        d["dtypes"] = list(self.dtypes)
        for k in ("nan_at", "inf_at", "constant_at"):
            d[k] = list(d[k])
        return d


@dataclass(frozen=True)
class Case:
    recipe: Recipe
    dtype: str

    @property
    def id(self) -> str:
        return f"{self.recipe.name}.{_SHORT[self.dtype]}"

    @property
    def quick(self) -> bool:
        return self.recipe.quick

    def kwargs(self) -> dict[str, Any]:
        """The public decomposition keywords of the case (max_chunk is separate)."""
        r = self.recipe
        return {
            "axes": r.axes,
            "max_imf": r.max_imf,
            "dtype": self.dtype,
            "tie_tolerance": r.tie_tolerance,
            "check_finite": r.check_finite,
            "errors": r.errors,
        }

    def input(self) -> np.ndarray:
        return build(self.recipe, self.dtype)


def _r(name: str, family: str, shape: tuple[int, ...], **kw: Any) -> Recipe:
    return Recipe(name, family, tuple(shape), **kw)


F32 = ("float32",)
F64 = ("float64",)
_MANY_BLOBS = {"count": 40, "size": (0.04, 0.1)}
_FINEPART = {"frac": (0.15, 0.3), "fine": 4.0}


def _recipes() -> list[Recipe]:
    R = []
    # ---- 1-D, one element
    R += [
        _r("d1-constant-2", "constant", (2,), quick=True),
        _r("d1-noise-2", "noise", (2,), quick=True),
        _r("d1-noise-3", "noise", (3,)),
        _r("d1-tones-3", "tones", (3,), quick=True),
        _r("d1-tones-37", "tones", (37,)),
        _r("d1-noise-37", "noise", (37,), quick=True),
        _r("d1-chirp-37", "chirp", (37,)),
        _r("d1-tones-512", "tones", (512,), quick=True),
        _r("d1-mixture-512", "mixture", (512,)),
        _r("d1-chirp-512", "chirp", (512,)),
        _r("d1-amfm-512", "amfm", (512,)),
        _r("d1-noise-512", "noise", (512,)),
        _r("d1-rednoise-512", "noise", (512,), params={"smooth": 3}),
        _r("d1-edges-512", "edges", (512,)),
        _r("d1-flat-512", "flat", (512,), quick=True),
        _r("d1-gabor-512", "gabor", (512,), quick=True),
        _r("d1-local-512", "local", (512,)),
        _r("d1-blobs-512", "blobs", (512,)),
        _r("d1-constant-512", "constant", (512,)),
        _r("d1-tones-2048", "tones", (2048,)),
        _r("d1-mixture-2048", "mixture", (2048,)),
        _r("d1-chirp-2048", "chirp", (2048,), quick=True),
        _r("d1-amfm-2048", "amfm", (2048,)),
        _r("d1-gabor-2048", "gabor", (2048,)),
        _r("d1-local-2048", "local", (2048,)),
        _r("d1-noise-4096", "noise", (4096,)),
        _r("d1-mixture-4096", "mixture", (4096,)),
        _r("d1-local-4096", "local", (4096,)),
        _r("d1-chirp-8192", "chirp", (8192,), dtypes=F32),
        _r("d1-mixture-8192", "mixture", (8192,), dtypes=F32),
    ]
    # ---- 1-D batches and layouts
    R += [
        _r("d1-batch4-512", "mixture", (4, 512), axes=(1,), quick=True),
        _r("d1-chunk1-512", "gabor", (5, 512), axes=(1,), max_chunk=1, quick=True),
        _r("d1-chunk2-512", "gabor", (5, 512), axes=(1,), max_chunk=2),
        _r("d1-chunk5-512", "gabor", (5, 512), axes=(1,), max_chunk=5),
        _r("d1-twobatch-37", "tones", (2, 37, 3), axes=(1,)),
        _r("d1-batchfirst-axis0", "chirp", (512, 3), axes=(0,)),
    ]
    # ---- 1-D specials
    R += [
        _r(
            "d1-nan-status",
            "mixture",
            (3, 512),
            axes=(1,),
            nan_at=(1,),
            errors="status",
            quick=True,
        ),
        _r("d1-constant-element", "mixture", (3, 512), axes=(1,), constant_at=(2,)),
        _r(
            "d1-inf-nocheck",
            "mixture",
            (2, 512),
            axes=(1,),
            inf_at=(0,),
            check_finite=False,
            errors="status",
            quick=True,
        ),
        _r(
            "d1-nan-nocheck",
            "mixture",
            (2, 512),
            axes=(1,),
            nan_at=(1,),
            check_finite=False,
            errors="status",
        ),
        _r("d1-inf-check", "mixture", (2, 512), axes=(1,), inf_at=(1,), errors="status"),
        _r("d1-gain-p40", "mixture", (512,), gain_pow2=40, seed_from="d1-mixture-512"),
        _r("d1-gain-m40", "mixture", (512,), gain_pow2=-40, quick=True, seed_from="d1-mixture-512"),
        _r("d1-gain-1e30", "mixture", (512,), gain=1e30, seed_from="d1-mixture-512"),
        _r("d1-gain-1e-30", "mixture", (512,), gain=1e-30, seed_from="d1-mixture-512"),
        _r("d1-gain-1e300", "mixture", (512,), gain=1e300, dtypes=F64, seed_from="d1-mixture-512"),
        _r(
            "d1-gain-1e-300", "mixture", (512,), gain=1e-300, dtypes=F64, seed_from="d1-mixture-512"
        ),
        _r("d1-negative", "mixture", (512,), gain=-1.0, seed_from="d1-mixture-512"),
        _r("d1-negative-3", "gabor", (2048,), gain=-3.0),
        _r("d1-offset", "mixture", (512,), offset=100.0, seed_from="d1-mixture-512"),
        _r("d1-maximf1", "mixture", (2048,), max_imf=1, quick=True),
        _r("d1-maximf2", "mixture", (2048,), max_imf=2),
        _r("d1-tie0", "mixture", (512,), tie_tolerance=0.0, quantum=0.0625),
        _r("d1-quantized", "mixture", (512,), quantum=0.0625, seed_from="d1-tie0"),
        _r("d1-tie2m20", "mixture", (512,), tie_tolerance=2.0**-20, quantum=2.0**-12),
        _r("d1-tie1e-3", "edges", (512,), tie_tolerance=1e-3),
        _r("d1-tie1e-3-quant", "edges", (512,), tie_tolerance=1e-3, params={"noise": 0.0}),
    ]
    # ---- 2-D, one element
    R += [
        _r("d2-noise-2x16", "noise", (2, 16), quick=True),
        _r("d2-fringes-2x16", "fringes", (2, 16)),
        _r("d2-constant-16x12", "constant", (16, 12), quick=True),
        _r("d2-noise-16x12", "noise", (16, 12)),
        _r("d2-fringes-16x12", "fringes", (16, 12), quick=True),
        _r("d2-tones-24x31", "tones", (24, 31)),
        _r("d2-blobs-24x31", "blobs", (24, 31), params=_MANY_BLOBS),
        _r("d2-edges-24x31", "edges", (24, 31)),
        _r("d2-fringes-40x36", "fringes", (40, 36)),
        _r("d2-gabor-40x36", "gabor", (40, 36), quick=True),
        _r("d2-mixture-40x36", "mixture", (40, 36)),
        _r("d2-flat-40x36", "flat", (40, 36)),
        _r("d2-zone-64x64", "zone", (64, 64), quick=True),
        _r("d2-gabor-64x64", "gabor", (64, 64)),
        _r("d2-amfm-64x64", "amfm", (64, 64)),
        _r("d2-local-64x64", "local", (64, 64), quick=True),
        _r("d2-edges-64x64", "edges", (64, 64)),
        _r("d2-blobs-64x64", "blobs", (64, 64), params=_MANY_BLOBS),
        _r("d2-flat-64x64", "flat", (64, 64)),
        _r("d2-fringes-128x128", "fringes", (128, 128)),
        _r("d2-gabor-128x128", "gabor", (128, 128)),
        _r("d2-mixture-128x128", "mixture", (128, 128)),
        _r("d2-zone-128x128", "zone", (128, 128)),
        _r("d2-local-128x128", "local", (128, 128)),
        _r("d2-zone-256x256", "zone", (256, 256)),
        _r("d2-mixture-256x256", "mixture", (256, 256)),
        _r("d2-gabor-256x256", "gabor", (256, 256)),
    ]
    # ---- 2-D batches and layouts
    R += [
        _r("d2-batch3-40x36", "gabor", (3, 40, 36), axes=(1, 2), quick=True),
        _r("d2-chunk1-24x31", "mixture", (5, 24, 31), axes=(1, 2), max_chunk=1, quick=True),
        _r("d2-chunk2-24x31", "mixture", (5, 24, 31), axes=(1, 2), max_chunk=2),
        _r("d2-chunk5-24x31", "mixture", (5, 24, 31), axes=(1, 2), max_chunk=5),
        _r("d2-batch70-16x12", "fringes", (70, 16, 12), axes=(1, 2), quick=True),
        _r("d2-batch70-chunk35", "fringes", (70, 16, 12), axes=(1, 2), max_chunk=35),
        _r("d2-batch70-local", "local", (70, 16, 12), axes=(1, 2)),
        _r("d2-batchmid-24x31", "tones", (24, 3, 31), axes=(0, 2)),
        _r("d2-twobatch-40x36", "gabor", (2, 40, 3, 36), axes=(1, 3)),
        _r("d2-axes-reversed", "local", (24, 31), axes=(1, 0)),
    ]
    # ---- 2-D specials
    R += [
        _r(
            "d2-nan-status",
            "mixture",
            (3, 24, 31),
            axes=(1, 2),
            nan_at=(0,),
            errors="status",
            quick=True,
        ),
        _r("d2-constant-element", "gabor", (3, 24, 31), axes=(1, 2), constant_at=(1,)),
        _r(
            "d2-inf-nocheck",
            "mixture",
            (2, 24, 31),
            axes=(1, 2),
            inf_at=(1,),
            check_finite=False,
            errors="status",
        ),
        _r("d2-gain-p40", "gabor", (40, 36), gain_pow2=40, seed_from="d2-gabor-40x36"),
        _r("d2-gain-1e-30", "gabor", (40, 36), gain=1e-30, seed_from="d2-gabor-40x36"),
        _r("d2-gain-1e300", "gabor", (40, 36), gain=1e300, dtypes=F64, seed_from="d2-gabor-40x36"),
        _r("d2-negative", "gabor", (40, 36), gain=-1.0, quick=True, seed_from="d2-gabor-40x36"),
        _r("d2-offset", "gabor", (40, 36), offset=100.0, seed_from="d2-gabor-40x36"),
        _r("d2-maximf1", "mixture", (64, 64), max_imf=1, quick=True),
        _r("d2-maximf2", "mixture", (64, 64), max_imf=2),
        _r("d2-tie0", "mixture", (40, 36), tie_tolerance=0.0, quantum=0.0625),
        _r("d2-quantized", "mixture", (40, 36), quantum=0.0625, seed_from="d2-tie0"),
        _r("d2-tie2m20", "mixture", (40, 36), tie_tolerance=2.0**-20, quantum=2.0**-12),
        _r("d2-tie1e-3", "edges", (40, 36), tie_tolerance=1e-3),
    ]
    # ---- 3-D, one element
    R += [
        _r("d3-constant-8x7x6", "constant", (8, 7, 6), quick=True),
        _r("d3-noise-8x7x6", "noise", (8, 7, 6), quick=True),
        _r("d3-fringes-8x7x6", "fringes", (8, 7, 6)),
        _r("d3-blobs-16x13x11", "blobs", (16, 13, 11), params=_MANY_BLOBS),
        _r("d3-tones-16x13x11", "tones", (16, 13, 11), quick=True),
        _r("d3-gabor-20x18x16", "gabor", (20, 18, 16)),
        _r("d3-fringes-20x18x16", "fringes", (20, 18, 16)),
        _r("d3-edges-20x18x16", "edges", (20, 18, 16)),
        _r("d3-flat-20x18x16", "flat", (20, 18, 16)),
        _r("d3-mixture-32x32x32", "mixture", (32, 32, 32)),
        _r("d3-gabor-32x32x32", "gabor", (32, 32, 32), quick=True),
        _r("d3-zone-32x32x32", "zone", (32, 32, 32)),
        _r("d3-local-32x32x32", "local", (32, 32, 32)),
        _r("d3-gabor-64x64x64", "gabor", (64, 64, 64), dtypes=F32),
        _r("d3-mixture-64x64x64", "mixture", (64, 64, 64), dtypes=F32),
    ]
    # ---- 3-D batches and layouts
    R += [
        _r("d3-batch3-20x18x16", "gabor", (3, 20, 18, 16), axes=(1, 2, 3), quick=True),
        _r("d3-chunk1-16x13x11", "mixture", (5, 16, 13, 11), axes=(1, 2, 3), max_chunk=1),
        _r(
            "d3-chunk2-16x13x11",
            "mixture",
            (5, 16, 13, 11),
            axes=(1, 2, 3),
            max_chunk=2,
            quick=True,
        ),
        _r("d3-chunk5-16x13x11", "mixture", (5, 16, 13, 11), axes=(1, 2, 3), max_chunk=5),
        _r("d3-batch9-32x32x32", "fringes", (9, 32, 32, 32), axes=(1, 2, 3)),
        _r("d3-batchmid-16x13x11", "tones", (16, 2, 13, 11), axes=(0, 2, 3)),
    ]
    # ---- 3-D specials
    R += [
        _r(
            "d3-nan-status",
            "mixture",
            (3, 16, 13, 11),
            axes=(1, 2, 3),
            nan_at=(2,),
            errors="status",
            quick=True,
        ),
        _r(
            "d3-inf-nocheck",
            "mixture",
            (2, 16, 13, 11),
            axes=(1, 2, 3),
            inf_at=(0,),
            check_finite=False,
            errors="status",
        ),
        _r("d3-gain-m40", "gabor", (20, 18, 16), gain_pow2=-40, seed_from="d3-gabor-20x18x16"),
        _r("d3-gain-1e30", "gabor", (20, 18, 16), gain=1e30, seed_from="d3-gabor-20x18x16"),
        _r("d3-negative", "gabor", (20, 18, 16), gain=-1.0, seed_from="d3-gabor-20x18x16"),
        _r("d3-maximf1", "mixture", (20, 18, 16), max_imf=1, quick=True),
        _r("d3-maximf2", "mixture", (20, 18, 16), max_imf=2),
        _r("d3-tie0", "mixture", (20, 18, 16), tie_tolerance=0.0, quantum=0.125),
    ]
    # ---- targeted branches (the name fixes the seed; a -vK suffix is a seed variant): the
    # growth-floor stop, an accepted finest population with its band-edge factor below 1
    R += [
        _r("d1-growth-100-v7", "growth", (100,), params={"frac": (0.4, 0.5)}),
        _r("d1-growth-512-v1", "growth", (512,), params={"frac": (0.4, 0.5)}, quick=True),
        _r("d1-tones-100-v4", "tones", (100,), params={"base": 3.0, "ratio": 2.5}),
        _r("d2-growth-40x36-v4", "growth", (40, 36), quick=True),
        _r("d2-growth-64x64-v3", "growth", (64, 64), params={"frac": (0.4, 0.5)}),
        _r(
            "d2-finepart-64x64",
            "local",
            (64, 64),
            params=_FINEPART | {"coarse_period": 0.2},
            quick=True,
        ),
        _r(
            "d2-finepart-128x128-v7", "local", (128, 128), params=_FINEPART | {"coarse_period": 0.1}
        ),
        _r(
            "d3-growth-32x32x32-v3", "growth", (32, 32, 32), params={"frac": (0.4, 0.5)}, quick=True
        ),
        _r("d3-finepart-32x32x32-v10", "local", (32, 32, 32), params=_FINEPART, quick=True),
        _r(
            "d3-gaborone-32x32x32",
            "gabor",
            (32, 32, 32),
            params={"count": 1, "p_lo": 3.0, "p_hi": 4.0, "width": (1.0, 1.5)},
        ),
    ]
    return R


RECIPES: tuple[Recipe, ...] = tuple(_recipes())


def all_cases() -> list[Case]:
    return [Case(r, dt) for r in RECIPES for dt in r.dtypes]


def case_by_id(case_id: str) -> Case:
    for c in all_cases():
        if c.id == case_id:
            return c
    raise KeyError(case_id)


# ----------------------------------------------------------------------------------------------
# Building an input


def _elements(recipe: Recipe) -> Iterator[tuple[int, tuple[Any, ...], tuple[int, ...]]]:
    """(element number, index into the full array, spatial extents in the order of axes)."""
    shape = recipe.shape
    axes = recipe.spatial_axes()
    batch_axes = [i for i in range(len(shape)) if i not in axes]
    batch_shape = tuple(shape[i] for i in batch_axes)
    n = tuple(shape[a] for a in axes)
    for e, bidx in enumerate(np.ndindex(*batch_shape)):
        idx: list[Any] = [slice(None)] * len(shape)
        for i, b in zip(batch_axes, bidx, strict=True):
            idx[i] = b
        yield e, tuple(idx), n


def _to_full_order(f: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    """A field with its axes in the order of `axes` -> its axes in increasing axis order."""
    return f.transpose([axes.index(a) for a in sorted(axes)])


def build(recipe: Recipe, dtype: str) -> np.ndarray:
    if dtype not in recipe.dtypes:
        raise ValueError(f"recipe {recipe.name} has no {dtype} case")
    fn = FAMILIES[recipe.family]
    seed = recipe_seed(recipe.seed_from or recipe.name)
    axes = recipe.spatial_axes()
    x = np.empty(recipe.shape, dtype=np.float64)
    for e, idx, n in _elements(recipe):
        f = np.asarray(fn(n, Uniform(seed, e), **recipe.params), dtype=np.float64)
        if f.shape != n:
            raise AssertionError(f"{recipe.name}: field shape {f.shape} != {n}")
        x[idx] = _to_full_order(f, axes)
    if recipe.quantum is not None:
        x = _floor(x / recipe.quantum) * recipe.quantum
    if recipe.gain_pow2 is not None:
        x = np.ldexp(x, recipe.gain_pow2)
    if recipe.gain is not None:
        x = x * recipe.gain
    if recipe.offset is not None:
        x = x + recipe.offset
    x = x.astype(dtype)
    for e, idx, _n in _elements(recipe):
        if e in recipe.constant_at:
            x[idx] = np.asarray(0.75, dtype=dtype)
        view = x[idx]
        flat = (len(view.reshape(-1)) * 2) // 5  # one sample, away from the faces
        pos = np.unravel_index(flat, view.shape)
        if e in recipe.nan_at:
            view[pos] = np.nan
        if e in recipe.inf_at:
            view[pos] = np.inf
    return x


def sha256(a: np.ndarray) -> str:
    """SHA-256 of an array's C-order bytes (dtype and shape are recorded separately)."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
