# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
"""The golden corpus (tests/data/golden): float64 inputs and the rows, windows and sifts that the
predecessor implementation's reference prototype produced for them.

``index.json`` lists, per case, the shape, the number of modes, the sha256 of the input and of the
rows, the reconstruction error and the conditioning class; each ``<case>.npz`` holds ``x``,
``modes``, ``windows``, ``sifts``, ``sensitivity`` and ``conditioning_class``. ``sensitivity`` is
the largest change of the prototype's rows, relative to max|x|, under three input perturbations of
relative size 1e-12 (x (1 + 1e-12), x (1 - 1e-12), random signs) that changed no decision, so
``A = sensitivity / 1e-12`` (1.5 to 4) is the amplification of an input perturbation into the rows.

Rules:

1. Integrity (CPU): the index and the files agree and the data is what the index hashed.
2. Decisions, exact (GPU, float64, "stable" cases): the number of modes, the windows and the sifts
   equal the corpus. They are discrete, and the port is another rounding of the same method, a
   perturbation far below the probes' 1e-12.
3. Rows (GPU, float64, "stable" cases): ``max|y - modes| <= A * DELTA * max|x|``, with
   ``DELTA = 1000 u`` (u = 2**-53), the relative input perturbation that stands for the rounding
   differences of the port (about 70 u measured at the worst, gabor_512: a margin of 14). This is a
   measured model bound, not a derived one.

Rules that cannot fail today: a "sensitive" case (no decision or row can be compared across
roundings) is listed as skipped with its class, and there is none in the corpus; the float32 result
of the same inputs is not checked, since the corpus says nothing about it.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

import eda

DATA = Path(__file__).resolve().parents[1] / "data" / "golden"
SANITIZED = os.environ.get("EDA_SANITIZE") not in (None, "", "0")
MEMBERS = {"x", "modes", "windows", "sifts", "sensitivity", "conditioning_class"}
CLASSES = ("stable", "sensitive")
#: The relative input perturbation that stands for the port's rounding differences (see above).
DELTA = 1000 * 2.0**-53
#: The relative size of the perturbation the corpus's sensitivity was measured at.
PROBE = 1e-12
#: Under the sanitizer: the three 512-sample cases and one 2-D case.
SANITIZED_CASES = {"amfm_512", "gabor_512", "noiseflat_512", "gabor_2d_64"}


def load_index() -> dict:
    return json.loads((DATA / "index.json").read_text())


def load_case(name: str) -> dict:
    with np.load(DATA / f"{name}.npz", allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


INDEX = load_index()
NAMES = sorted(INDEX["cases"])


def sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# ----------------------------------------------------------------------------------------------
# Rule 1: integrity (no GPU)


def test_the_index_and_the_files_list_the_same_cases():
    assert INDEX["corpus_version"] == 1
    on_disk = sorted(p.stem for p in DATA.glob("*.npz"))
    assert on_disk == NAMES and len(NAMES) >= 1


@pytest.mark.parametrize("name", NAMES)
def test_a_case_is_what_the_index_says(name):
    entry = INDEX["cases"][name]
    case = load_case(name)
    assert set(case) == MEMBERS
    shape = tuple(entry["shape"])
    x, modes = case["x"], case["modes"]
    assert entry["dtype"] == "float64" and x.dtype == np.float64 and modes.dtype == np.float64
    assert x.shape == shape and modes.shape == (entry["n_modes"],) + shape
    assert sha256(x) == entry["sha256_x"] and sha256(modes) == entry["sha256_modes"]
    assert case["windows"].shape == (entry["n_modes"] - 1,)
    assert case["sifts"].shape == (entry["n_modes"] - 1,)
    assert case["windows"].dtype.kind == "i" and case["sifts"].dtype.kind == "i"
    assert (case["windows"] % 2 == 1).all() and (case["windows"] >= 3).all()
    assert ((case["sifts"] >= 1) & (case["sifts"] <= 3)).all()
    assert str(case["conditioning_class"]) == entry["conditioning_class"] in CLASSES
    assert np.isfinite(x).all() and np.isfinite(modes).all()
    sensitivity = float(case["sensitivity"])
    assert np.isfinite(sensitivity) and 0 < sensitivity < 1e-9


@pytest.mark.parametrize("name", NAMES)
def test_the_recorded_reconstruction_error_is_that_of_the_stored_rows(name):
    """A consistency check of the index's scalar against the stored rows (the sha256 pins the
    bytes, not the scalar). The sum order is a numpy property, so the comparison allows
    n_modes * eps instead of bitwise equality."""
    case = load_case(name)
    x, modes = case["x"], case["modes"]
    recomputed = float(np.abs(modes.sum(axis=0) - x).max() / np.abs(x).max())
    recorded = INDEX["cases"][name]["reconstruction_rel_err"]
    assert abs(recomputed - recorded) <= modes.shape[0] * np.finfo(np.float64).eps


# ----------------------------------------------------------------------------------------------
# Rules 2 and 3: decisions and rows (GPU)


@pytest.mark.gpu
@pytest.mark.parametrize("name", NAMES)
def test_the_port_reproduces_the_corpus(name):
    if SANITIZED and name not in SANITIZED_CASES:
        pytest.skip("outside the sanitizer's subset")
    case = load_case(name)
    conditioning = str(case["conditioning_class"])
    if conditioning != "stable":
        pytest.skip(f"conditioning class {conditioning!r}: decisions and rows are not comparable")
    x, modes = case["x"], case["modes"]
    y, info = eda.decompose(x, return_info=True)
    assert info.error == 0 and info.dtype == "float64"
    # decisions, exact
    assert y.shape == modes.shape
    assert int(info.n_imfs) == modes.shape[0] - 1
    assert info.windows.tolist() == case["windows"].tolist()
    assert info.sifts.tolist() == case["sifts"].tolist()
    # rows, within the bound the corpus's sensitivity defines
    amplification = float(case["sensitivity"]) / PROBE
    bound = amplification * DELTA * float(np.abs(x).max())
    assert float(np.abs(y - modes).max()) <= bound
