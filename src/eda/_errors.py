# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
"""The exceptions of eda (importable without the compiled extension). Both classes pickle."""

from __future__ import annotations

from typing import Any


class CudaError(RuntimeError):
    """A CUDA runtime call or kernel launch failed inside eda, or no usable GPU was found.

    The message names the CUDA error, what eda was doing and where. ``code`` is the numeric
    ``cudaError_t`` and ``name`` its symbolic name (for example ``"cudaErrorInvalidValue"``); both
    are None when unknown.
    """

    def __init__(self, message: str, code: int | None = None, name: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.name = name or None

    def __reduce__(self) -> tuple[type[CudaError], tuple[str, int | None, str | None]]:
        return (type(self), (str(self), self.code, self.name))


class DecompositionError(ValueError):
    """Some elements of an ``eda.decompose`` call with ``errors="raise"`` failed.

    ``indices`` holds the flat (C-order) batch indices of the failed elements, ``codes`` their
    error codes (1: the input holds inf or NaN and ``check_finite`` is on; 2: a non-finite
    intermediate or output row) and ``info`` the ``DecompositionInfo`` of the call.
    """

    def __init__(self, message: str, indices: Any = (), codes: Any = (), info: Any = None) -> None:
        super().__init__(message)
        self.indices = indices
        self.codes = codes
        self.info = info

    def __reduce__(self) -> tuple[type[DecompositionError], tuple[str, Any, Any, Any]]:
        return (type(self), (str(self), self.indices, self.codes, self.info))
