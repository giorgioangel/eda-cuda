# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
"""Process lifecycle, each case in a fresh interpreter: exit while a daemon thread is inside a
call, a call from an exit handler, and forked children."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.gpu

SANITIZED = os.environ.get("EDA_SANITIZE") not in (None, "", "0")
SIZE = 64 if SANITIZED else 1024
BAD_WORDS = ("terminate", "Segmentation", "Fatal Python error", "Traceback", "core dumped")


def run(code: str, timeout: float = 240.0) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def assert_clean(done: subprocess.CompletedProcess[str]) -> None:
    assert done.returncode == 0, done.stdout + done.stderr
    for word in BAD_WORDS:
        assert word not in done.stderr, done.stderr


@pytest.mark.parametrize("run_number", range(5))
def test_exit_while_a_daemon_thread_is_inside_a_call(run_number):
    done = run(
        f"""
        import threading, time
        import numpy as np
        import eda

        x = np.random.default_rng({run_number}).standard_normal(({SIZE}, {SIZE})).astype(np.float32)
        eda.decompose(x)  # warm up: the thread's calls then take their steady time
        started = threading.Event()

        def loop():
            started.set()
            try:
                while True:
                    eda.decompose(x)
            except RuntimeError as exc:  # refused after the exit hook ran
                assert "shutting down" in str(exc), exc

        threading.Thread(target=loop, daemon=True).start()
        started.wait()
        time.sleep(0.05 + 0.03 * {run_number})
        print("main exits")
        """
    )
    assert_clean(done)
    assert "main exits" in done.stdout


LOOPS = {
    "device_count": "eda.device_count()",
    "device_info": "eda.device_info(0)",
    "interface": "eda.decompose(CaiOnly(x), axes=1)",
}


@pytest.mark.parametrize("entry", sorted(LOOPS))
def test_exit_while_a_daemon_thread_loops_over_an_entry_point(entry):
    """Every entry point that releases the GIL is refused after the exit hook, also when the loop
    swallows the refusal as a generic worker loop does (the CUDA array interface path asks for the
    pointer's device before the decomposition)."""
    if entry == "interface" and importlib.util.find_spec("cupy") is None:
        pytest.skip("needs CuPy")
    done = run(
        f"""
        import threading, time
        import numpy as np
        import eda

        class CaiOnly:
            def __init__(self, array):
                self.__cuda_array_interface__ = array.__cuda_array_interface__
                self._array = array

        x = None
        if {entry!r} == "interface":
            import cupy
            x = cupy.asarray(np.random.default_rng(4).standard_normal((4, 64)).astype(np.float32))
        {LOOPS[entry]}  # warm up
        started = threading.Event()

        def loop():
            started.set()
            while True:
                try:
                    {LOOPS[entry]}
                except RuntimeError:
                    pass

        threading.Thread(target=loop, daemon=True).start()
        started.wait()
        time.sleep(0.05)
        print("main exits")
        """
    )
    assert_clean(done)
    assert "main exits" in done.stdout


def test_a_call_from_an_exit_handler_is_refused():
    done = run(
        """
        import atexit
        import numpy as np

        def handler():
            import eda
            try:
                eda.decompose(np.zeros((4, 64), np.float32), axes=1)
            except RuntimeError as exc:
                print("refused:", exc)
            else:
                print("NOT refused")

        atexit.register(handler)  # registered first: runs after eda's hook
        import eda
        eda.decompose(np.random.default_rng(0).standard_normal((4, 64)).astype(np.float32), axes=1)
        """
    )
    assert_clean(done)
    assert "refused: eda: the interpreter is shutting down" in done.stdout


def test_exit_after_use_releases_without_error():
    done = run(
        f"""
        import numpy as np
        import eda
        eda.decompose(np.random.default_rng(1).standard_normal(({SIZE}, 16)).astype(np.float32))
        """
    )
    assert_clean(done)
    assert done.stderr == ""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.skipif(SANITIZED, reason="compute-sanitizer's own fork handling hangs the child")
def test_forked_child_after_cuda_use_refuses_and_exits():
    done = run(
        f"""
        import os, sys, threading, time
        import numpy as np
        import eda

        x = np.random.default_rng(2).standard_normal(({SIZE}, 32)).astype(np.float32)
        eda.decompose(x, axes=1)
        stop = threading.Event()
        started = threading.Event()

        def loop():  # another thread is inside calls when the fork happens
            started.set()
            while not stop.is_set():
                eda.decompose(x, axes=1)

        thread = threading.Thread(target=loop)
        thread.start()
        started.wait()
        time.sleep(0.05)
        pid = os.fork()
        if pid == 0:
            try:
                eda.decompose(x, axes=1)
            except RuntimeError as exc:
                print("child:", exc)
            sys.exit(0)  # a normal exit must not hang or touch CUDA
        deadline = time.time() + 60
        status = None
        while time.time() < deadline:
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                break
            time.sleep(0.05)
        else:
            os.kill(pid, 9)
            os.waitpid(pid, 0)
            raise SystemExit("the child hung")
        stop.set()
        thread.join()
        assert os.waitstatus_to_exitcode(status) == 0, status
        print("parent ok")
        """
    )
    assert_clean(done)
    assert "child: eda: CUDA cannot be used in a process forked after eda used CUDA" in done.stdout
    assert "parent ok" in done.stdout


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_forked_child_before_cuda_use_works():
    done = run(
        f"""
        import os, sys
        import numpy as np
        import eda
        from eda import _native

        _native.core()  # the extension is loaded, CUDA is not initialized
        x = np.random.default_rng(3).standard_normal(({SIZE}, 32)).astype(np.float32)
        pid = os.fork()
        if pid == 0:
            eda.decompose(x, axes=1)
            sys.exit(0)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0, status
        eda.decompose(x, axes=1)
        print("ok")
        """
    )
    assert_clean(done)
    assert "ok" in done.stdout
