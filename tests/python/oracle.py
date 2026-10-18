# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
"""The frozen oracle (tests/data/parity_v2.json): what a call must reproduce bit for bit, and the
predecessor's record (tests/data/parity_v1.json), which eda reproduces within tolerance.

``observe`` condenses a result into the fields of a manifest entry and ``differences`` compares the
two. The spy records the computed rows and the chunk size of the native call, which the public
result does not carry; it also passes a recipe's ``max_chunk`` (a test option of the native entry).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import cases as C

MANIFEST = Path(__file__).resolve().parents[1] / "data" / "parity_v2.json"
MANIFEST_V1 = MANIFEST.with_name("parity_v1.json")


def load(path: Path = MANIFEST) -> dict[str, dict[str, Any]]:
    manifest = json.loads(path.read_text())
    assert manifest["format"] == path.stem and manifest["count"] == len(manifest["cases"])
    return {entry["id"]: entry for entry in manifest["cases"]}


def provenance() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text())["provenance"]


class Spy:
    """Replaces a native entry point (``decompose_host`` or ``decompose_device``) by a wrapper
    that sets the call's ``max_chunk`` and records what the native call returned."""

    def __init__(self, monkeypatch: Any, core: Any, name: str) -> None:
        self.max_chunk = 0
        self.rows: np.ndarray | None = None
        self.chunk = 0
        original = getattr(core, name)

        def wrapper(*args: Any) -> Any:
            result = original(*args[:-1], self.max_chunk)
            self.info = result[-1]
            self.chunk = int(result[-1]["chunk"])
            self.result = result
            if name == "decompose_host":
                self.rows = result[0]
            return result

        monkeypatch.setattr(core, name, wrapper)


def sha(a: Any) -> str:
    return C.sha256(np.asarray(a))


def observe(
    x: np.ndarray, y: np.ndarray, info: Any, rows: np.ndarray, chunk: int, axes: Any
) -> dict[str, Any]:
    """The manifest fields of one result: ``x`` the input, ``y`` the public output, ``info`` the
    ``DecompositionInfo``, ``rows`` the computed (B, P, *spatial) rows."""
    batch = math.prod(info.n_imfs.shape)
    out: dict[str, Any] = {
        "input_sha256": sha(x),
        "y_shape": list(y.shape),
        "y_sha256": sha(y),
        "rows_sha256": [sha(rows[:, p]) for p in range(rows.shape[1])],
        "n_imfs": [int(v) for v in info.n_imfs.reshape(-1)],
        "error": [int(v) for v in info.error.reshape(-1)],
        "info_shape": list(info.windows.shape),
        "info_axes": [int(a) for a in info.axes],
    }
    for name in ("windows", "sifts", "spacing", "energy"):
        a = np.ascontiguousarray(getattr(info, name))
        out[name + "_sha256"] = sha(a)
        if name in ("windows", "sifts") and a.size <= 64:
            out[name] = [int(v) for v in a.reshape(-1)]
    out["chunks"] = -(-batch // chunk) if chunk > 0 else 0
    return out


def decision_differences(got: dict[str, Any], want: dict[str, Any]) -> list[str]:
    """The decisions in which a result differs from a manifest entry: the IMF counts, the error
    codes, the windows and the sifts."""
    bad = [key for key in ("n_imfs", "error") if got[key] != want[key]]
    for name in ("windows", "sifts"):
        if name in want:
            if got.get(name) != want[name]:
                bad.append(name)
        elif got[name + "_sha256"] != want[name + "_sha256"]:
            bad.append(name + "_sha256")
    return bad


def tolerance_differences(got: dict[str, Any], want: dict[str, Any]) -> list[str]:
    """The fields that must match also under another cuFFT: the input, the shapes, the chunk
    count and the decisions."""
    keys = ("input_sha256", "y_shape", "info_shape", "info_axes")
    bad = [key for key in keys if got[key] != want[key]]
    if len(got["rows_sha256"]) != len(want["rows_sha256"]):
        bad.append(f"rows count {len(got['rows_sha256'])} != {len(want['rows_sha256'])}")
    if got["chunks"] != want["chunks"]:
        bad.append(f"chunks {got['chunks']} != {want['chunks']}")
    return bad + decision_differences(got, want)


def differences(got: dict[str, Any], want: dict[str, Any]) -> list[str]:
    """The names of the fields in which a result differs from its manifest entry."""
    bad = []
    for key in (
        "input_sha256",
        "y_shape",
        "y_sha256",
        "n_imfs",
        "error",
        "info_shape",
        "info_axes",
    ):
        if got[key] != want[key]:
            bad.append(key)
    if len(got["rows_sha256"]) != len(want["rows_sha256"]):
        bad.append(f"rows count {len(got['rows_sha256'])} != {len(want['rows_sha256'])}")
    else:
        bad += [
            f"rows_sha256[{p}]"
            for p, (a, b) in enumerate(zip(got["rows_sha256"], want["rows_sha256"], strict=True))
            if a != b
        ]
    for name in ("windows", "sifts"):
        if name in want:
            if got.get(name) != want[name]:
                bad.append(name)
        elif got[name + "_sha256"] != want[name + "_sha256"]:
            bad.append(name + "_sha256")
    for name in ("spacing", "energy"):
        if got[name + "_sha256"] != want[name + "_sha256"]:
            bad.append(name + "_sha256")
    if got["chunks"] != want["chunks"]:
        bad.append(f"chunks {got['chunks']} != {want['chunks']}")
    return bad
