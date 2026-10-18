# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
"""The frozen oracle through the public call: ``eda.decompose`` reproduces every entry of
tests/data/parity_v2.json, and the predecessor's record tests/data/parity_v1.json within tolerance.

``test_parity_quick`` runs the manifest's quick subset; ``test_parity_full`` all 287 cases and runs
only with ``-m full``. The record was made with the cuFFT library of the NVIDIA wheel
``nvidia-cufft`` 12.0.0.61. When the process maps that file (by sha256), every entry must match bit
for bit (rows, public output, per-element information, chunk counts): mode "bitwise". With any other
cuFFT (another 12.x, or cuFFT 11 for the CUDA 12 package) the mode is "tolerance", the rule of the
predecessor's record below with parity_v2 as the record: the shapes, the chunk counts and the
decisions exact on every case, the rows within 10 sensitivity max|x| of the record's rows on the
stable cases. Those rows are out of tree (``PARITY_V2_ROWS``, by default
~/.cache/eda-cuda/oracle-v2/cases: ``<id>.a.npz`` whose ``y`` is the public output); without them
the row check skips. The test summary names the mode and counts the cases that matched bit for bit.

The predecessor's record. Its 1-D entries are eda's bit for bit. In 2-D and 3-D the orientation's
transforms differ (complex transforms, field/dct.hpp, where the predecessor's real ones computed
two batch entries together for some sizes), so the rows differ by roundings, and the method
amplifies a rounding difference on some inputs. The record therefore holds within tolerance:

- decisions, exact, on every case: the IMF counts, the error codes, the windows and the sifts;
- rows, on the cases of conditioning class "stable": ``max|y - y_v1| <= 10 sensitivity max|x|``
  over the finite samples, NaN and inf where the record has them. The sensitivity (parity_v2.json)
  is the largest change of the output, relative to max|x|, when every input sample moves by one
  unit in the last place in a random direction (three draws), the smallest change of the input,
  so it measures how far the method carries a rounding. The bound is 10 times that change; the
  worst case measured is 1.9 times. A fixed bound on the rounding of each operator would not do:
  the method amplifies roundings, and seven inputs exceed such a bound (by 941 times for
  d2-gabor-256x256 in float64, 4.4e-11 max|x|) while their one-unit probes move the rows as far
  (up to 784 times). A "sensitive" case, whose decisions a probe changes (six quantized or tie
  inputs), has no row bound.

The record's rows are out of tree (``PARITY_V1_ROWS``, by default ~/.cache/eda-cuda/oracle/cases:
one npz per case whose ``y`` is the public output); without them the row test skips.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

import cases as C
import eda
import oracle
from eda import _native

pytestmark = pytest.mark.gpu

if not oracle.MANIFEST.exists():
    pytest.skip("tests/data/parity_v2.json is not present", allow_module_level=True)
WANT = oracle.load()
WANT_V1 = oracle.load(oracle.MANIFEST_V1)
ALL = C.all_cases()
QUICK = [c for c in ALL if WANT[c.id]["quick"]]
STABLE = [c for c in ALL if WANT[c.id]["conditioning_class"] == "stable"]
V1_ROWS = Path(os.environ.get("PARITY_V1_ROWS", Path.home() / ".cache/eda-cuda/oracle/cases"))
V2_ROWS = Path(os.environ.get("PARITY_V2_ROWS", Path.home() / ".cache/eda-cuda/oracle-v2/cases"))


def mapped_cufft() -> list[str]:
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split()
        if fields and "libcufft" in fields[-1] and fields[-1].startswith("/"):
            paths.add(fields[-1])
    return sorted(paths)


@pytest.fixture(scope="session")
def mode() -> str:
    """The mode: "bitwise" when the process maps the record's cuFFT file, else "tolerance"."""
    eda.decompose(np.random.default_rng(0).standard_normal((24, 24)).astype(np.float32))
    libraries = mapped_cufft()
    assert len(libraries) == 1, f"expected one mapped cuFFT, got {libraries}"
    digest = hashlib.sha256(Path(libraries[0]).read_bytes()).hexdigest()
    if digest != oracle.provenance()["libcufft_sha256"]:
        return "tolerance"
    assert eda.build_info()["cufft_version"] == oracle.provenance()["cufft_version"]
    return "bitwise"


@pytest.fixture(autouse=True)
def report_mode(mode, record_property):
    """The mode and the cuFFT, for the summary (conftest.py)."""
    record_property(
        "parity_mode", f"{mode} (cuFFT {eda.build_info()['cufft_version']}, {mapped_cufft()[0]})"
    )


def run(case: C.Case, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, np.ndarray]:
    spy = oracle.Spy(monkeypatch, _native.core(), "decompose_host")
    spy.max_chunk = case.recipe.max_chunk or 0
    x = case.input()
    kwargs = case.kwargs()
    assert kwargs["errors"] in ("raise", "status")
    y, info = eda.decompose(x, **kwargs, return_info=True)  # errors="raise" must not raise
    assert isinstance(y, np.ndarray) and y.dtype == np.dtype(case.dtype)
    assert info.dtype == case.dtype
    return oracle.observe(x, y, info, spy.rows, spy.chunk, kwargs["axes"]), y


def rows_within_tolerance(case: C.Case, y: np.ndarray, old: np.ndarray) -> None:
    """The rule for rows: equal non-finite samples, finite ones within 10 sensitivity max|x|."""
    assert old.shape == y.shape and old.dtype == y.dtype
    finite = np.isfinite(old)
    assert np.array_equal(finite, np.isfinite(y)) and np.array_equal(np.isnan(old), np.isnan(y))
    assert np.array_equal(old[np.isinf(old)], y[np.isinf(old)])
    x = case.input()
    scale = float(np.abs(x[np.isfinite(x)].astype(np.float64)).max(initial=0.0))
    bound = 10 * WANT[case.id]["sensitivity"] * scale
    diff = np.abs(y[finite].astype(np.float64) - old[finite].astype(np.float64))
    assert float(diff.max(initial=0.0)) <= bound


def check_parity(case, monkeypatch, record_property, mode):
    got, y = run(case, monkeypatch)
    differences = oracle.differences(got, WANT[case.id])
    record_property("parity_bitwise", not differences)
    if mode == "bitwise":
        assert differences == []
        return
    assert oracle.tolerance_differences(got, WANT[case.id]) == []
    if WANT[case.id]["conditioning_class"] != "stable" or not differences:
        return
    path = V2_ROWS / f"{case.id}.a.npz"
    if not path.exists():
        pytest.skip(f"decisions equal; the record's rows are not present ({path})")
    with np.load(path, allow_pickle=False) as z:
        rows_within_tolerance(case, y, z["y"])


@pytest.mark.parametrize("case", QUICK, ids=[c.id for c in QUICK])
def test_parity_quick(case, monkeypatch, record_property, mode):
    check_parity(case, monkeypatch, record_property, mode)


@pytest.mark.full
@pytest.mark.parametrize("case", ALL, ids=[c.id for c in ALL])
def test_parity_full(case, monkeypatch, record_property, mode):
    check_parity(case, monkeypatch, record_property, mode)


@pytest.mark.parametrize("case", QUICK, ids=[c.id for c in QUICK])
def test_predecessor_decisions_quick(case, monkeypatch):
    assert oracle.decision_differences(run(case, monkeypatch)[0], WANT_V1[case.id]) == []


@pytest.mark.full
@pytest.mark.parametrize("case", ALL, ids=[c.id for c in ALL])
def test_predecessor_decisions_full(case, monkeypatch):
    assert oracle.decision_differences(run(case, monkeypatch)[0], WANT_V1[case.id]) == []


@pytest.mark.full
@pytest.mark.parametrize("case", STABLE, ids=[c.id for c in STABLE])
def test_predecessor_rows_within_tolerance(case, monkeypatch):
    path = V1_ROWS / f"{case.id}.npz"
    if not path.exists():
        pytest.skip(f"the predecessor's rows are not present ({path})")
    y = run(case, monkeypatch)[1]
    with np.load(path, allow_pickle=False) as z:
        rows_within_tolerance(case, y, z["y"])


def test_manifest_covers_the_cases():
    assert {c.id for c in ALL} == set(WANT) == set(WANT_V1)
    # the 1-D entries are the predecessor's bit for bit (no orientation in 1-D)
    for c in ALL:
        if len(WANT[c.id]["info_axes"]) == 1:
            assert {k: WANT[c.id].get(k) for k in WANT_V1[c.id]} == WANT_V1[c.id], c.id
