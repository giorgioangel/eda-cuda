# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
"""Decompose a 1-D signal and a batch of 2-D images, and check that the rows add up to the input.

    python examples/quickstart.py

Needs an NVIDIA GPU. Runs in a few seconds and plots nothing.
"""

import numpy as np

import eda

rng = np.random.default_rng(0)

# 1-D: three oscillations of different periods plus noise.
t = np.arange(2048)
signal = (
    np.sin(2 * np.pi * t / 8)
    + 2 * np.sin(2 * np.pi * t / 64)
    + 3 * np.sin(2 * np.pi * t / 512)
    + 0.1 * rng.standard_normal(t.size)
).astype(np.float32)

y, info = eda.decompose(signal, return_info=True)
print("1-D input", signal.shape, "-> rows", y.shape)
print("  IMFs found:", int(info.n_imfs), "(the last row is the residue)")
print("  window of each mode, in samples:", info.windows[: int(info.n_imfs)].tolist())
error = np.abs(y.sum(axis=0, dtype=np.float64) - signal).max() / np.abs(signal).max()
print(f"  reconstruction error, relative to max|x|: {error:.1e}")

# 2-D: a batch of four 128 x 128 images. The first axis is the batch; each image is
# decomposed on its own.
images = rng.standard_normal((4, 128, 128)).astype(np.float32)
y, info = eda.decompose(images, axes=(1, 2), return_info=True)
print("2-D input", images.shape, "-> rows", y.shape)  # (4, P, 128, 128)
print("  IMFs per image:", info.n_imfs.tolist())
error = np.abs(y.sum(axis=1, dtype=np.float64) - images).max() / np.abs(images).max()
print(f"  reconstruction error, relative to max|x|: {error:.1e}")
