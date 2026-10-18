# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
"""Markers, skips and a summary of the row differences the symmetry tests report.

``gpu`` tests need the compiled extension and a visible GPU; without either they are skipped.
``full`` tests (the complete oracle run) run only when the ``-m`` expression names ``full``.
"""

from __future__ import annotations

import sys

import pytest


def _gpu_unavailable() -> str | None:
    """Why GPU tests cannot run here, or None."""
    try:
        from eda import _native

        core = _native.core()
    except ImportError as exc:
        return f"the eda extension is not available: {exc}"
    try:
        if core.device_count() == 0:
            return "no CUDA device is visible"
    except RuntimeError as exc:  # a broken driver installation
        return f"CUDA is not usable: {exc}"
    return None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    full_requested = "full" in (config.getoption("-m") or "")
    reason = None
    if any(item.get_closest_marker("gpu") for item in items):
        reason = _gpu_unavailable()
    for item in items:
        if item.get_closest_marker("full") and not full_requested:
            item.add_marker(pytest.mark.skip(reason="the full run needs -m full"))
        elif reason and item.get_closest_marker("gpu"):
            item.add_marker(pytest.mark.skip(reason=reason))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Give the frameworks' cached device memory back, so that leak checks see only real leaks."""
    cupy = sys.modules.get("cupy")
    if cupy is not None:
        try:
            cupy.get_default_memory_pool().free_all_blocks()
            cupy.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:  # noqa: BLE001
            pass
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """The largest value each test function recorded as ``symmetry_rows``, and the parity mode
    and bitwise count of test_parity (record_property)."""
    largest: dict[str, tuple[float, str]] = {}
    modes: set[str] = set()
    bitwise: dict[str, bool] = {}
    for reports in terminalreporter.stats.values():
        for report in reports:
            for name, value in getattr(report, "user_properties", ()):
                if name == "parity_mode":
                    modes.add(value)
                elif name == "parity_bitwise":  # per case, over the quick and the full test
                    case = report.nodeid.partition("[")[2].rstrip("]")
                    bitwise[case] = bitwise.get(case, True) and value
                if name != "symmetry_rows":
                    continue
                function, _, case = report.nodeid.partition("[")
                if value > largest.get(function, (-1.0, ""))[0]:
                    largest[function] = (value, case.rstrip("]"))
    if modes:
        terminalreporter.section("parity with tests/data/parity_v2.json")
        for mode in sorted(modes):
            terminalreporter.line(f"mode: {mode}")
        terminalreporter.line(f"bit for bit: {sum(bitwise.values())} of {len(bitwise)} cases")
    if largest:
        terminalreporter.section("row differences of the symmetries, in eps M (P - 1)")
        for function, (value, case) in sorted(largest.items()):
            terminalreporter.line(f"{value:10.1f}  {function.split('::')[-1]} [{case}]")
