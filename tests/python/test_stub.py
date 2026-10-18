# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
"""The type stub src/eda/_core.pyi is nanobind stubgen output, kept byte for byte.

``python tests/python/test_stub.py`` rewrites the tracked stub from the extension in the import
path; the test regenerates it in a temporary directory and compares.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TRACKED = REPO / "src" / "eda" / "_core.pyi"

# The generation runs in its own interpreter: stubgen changes the environment and the import
# path, and the extension needs the cuFFT preload of eda._native.
_SCRIPT = (
    "import sys\n"
    "import eda._native as native\n"
    "native.core()\n"
    "import nanobind.stubgen as stubgen\n"
    "stubgen.main(['-q', '-m', 'eda._core', '-o', sys.argv[1]])\n"
)


def generate(path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    done = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if done.returncode != 0:
        raise RuntimeError(f"stubgen failed:\n{done.stdout}{done.stderr}")


def _needs() -> str | None:
    try:
        import nanobind  # noqa: F401
    except ImportError:
        return "nanobind is not installed"
    try:
        from eda import _native

        _native.core()
    except ImportError:
        return "eda._core is not built"
    return None


def test_stub_is_current():
    reason = _needs()
    if reason:
        pytest.skip(reason)
    if not TRACKED.exists():
        pytest.skip("src/eda/_core.pyi is not next to the tests")
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp) / "_core.pyi"
        generate(fresh)
        assert fresh.read_text() == TRACKED.read_text(), (
            "src/eda/_core.pyi is stale: run `python tests/python/test_stub.py`"
        )


def test_stub_shipped_with_the_package():
    import eda

    assert (Path(eda.__file__).parent / "_core.pyi").exists()


if __name__ == "__main__":
    generate(TRACKED)
    print(f"wrote {TRACKED}")
