# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
"""Renders the README figures into docs/images (needs a GPU, eda, matplotlib and scikit-image):

    python tools/readme_figures.py [OUT_DIR]

The synthetic inputs are built from known parts, so each mode on a figure can be traced to its
source. The photograph is scikit-image's cameraman (skimage.data.camera, CC0, by Lav Varshney).
On the synthetic inputs every field, the input and the residue included, uses blue for negative,
white for zero and red for positive. Each side is stretched on its own (zero_white), so a field
that is mostly positive, such as an input with an offset, still shows its negative part. The
greyscale photograph stays greyscale: the input and the smooth last row on the photograph's own
scale, the modes and bands from black (negative) through mid-grey (zero) to white (positive).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter
from skimage import data, feature

import eda

OUT = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path(__file__).resolve().parents[1] / "docs" / "images"
)
BWR = LinearSegmentedColormap.from_list("bwr_white", ["#0a2cff", "#ffffff", "#ff1a1a"])
INK, MUTED = "#1b2230", "#5d6779"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)


def frequencies_1d(n: int = 4096) -> dict[str, np.ndarray]:
    """Instantaneous frequencies, in cycles per signal length, of the two oscillations of the 1-D
    example. The chirp starts below the frequency the cosine reaches at its end, so no fixed
    frequency band holds either of them."""
    t = np.arange(n) / n
    return {
        "chirp": 50 + 100 * t**2,
        "cosine": 18 + 32 * t**1.5 + 5 * np.sin(2 * np.pi * 1.3 * t),
    }


def signal_1d() -> dict[str, np.ndarray]:
    """A chirp, a cosine with a varying period and a trend."""
    f = frequencies_1d()
    n = f["chirp"].size
    t = np.arange(n) / n
    return {
        "chirp": np.sin(2 * np.pi * np.cumsum(f["chirp"]) / n),
        "cosine": 1.2 * np.cos(2 * np.pi * np.cumsum(f["cosine"]) / n),
        "trend": 3 * (t - 0.35) ** 2 + 0.6 * t,
    }


def relative_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth))


def fourier_bands(x: np.ndarray, parts: dict[str, np.ndarray]):
    """x as three sums of its Fourier modes, one sum per frequency band (below low, low to high,
    above high), with the cutoffs that give the smallest worst error against the true parts,
    which favours Fourier. The modes are those of the even extension (the cosine series), which
    gives Fourier the same reflecting borders as eda."""
    n = x.size
    spectrum = np.fft.rfft(np.concatenate([x, x[::-1]]))
    f = np.fft.rfftfreq(2 * n, 1 / n)
    band = lambda keep: np.fft.irfft(spectrum * keep, 2 * n)[:n]  # noqa: E731
    best = None
    for low in np.arange(1.0, 20.0):
        trend = band(f < low)
        for high in np.arange(low + 2, 120.0):
            chirp = band(f >= high)
            split = {"chirp": chirp, "cosine": x - trend - chirp, "trend": trend}
            worst = max(relative_error(split[k], parts[k]) for k in parts)
            if best is None or worst < best[0]:
                best = (worst, split, (low, high))
    return best[1], best[2]


def image_2d(n: int = 384) -> np.ndarray:
    """Fine rings around a point, wavy stripes and smooth illumination, with a little noise."""
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    r = np.hypot(xx - 0.38 * n, yy - 0.42 * n)
    rings = np.exp(-((r / (0.32 * n)) ** 2)) * np.sin(2 * np.pi * r / 7.0)
    wavy = 0.9 * np.sin(
        2 * np.pi * (0.8 * xx + 0.6 * yy) / 34.0 + 2.5 * np.sin(2 * np.pi * yy / 230.0)
    )
    light = 1.6 * np.exp(-(((xx - 0.75 * n) ** 2 + (yy - 0.25 * n) ** 2) / (2 * (0.22 * n) ** 2)))
    light += 1.0 * np.exp(-(((xx - 0.2 * n) ** 2 + (yy - 0.85 * n) ** 2) / (2 * (0.25 * n) ** 2)))
    noise = 0.05 * np.random.default_rng(2).standard_normal((n, n))
    return (rings + wavy + light + noise).astype(np.float32)


def zero_white(f: np.ndarray) -> TwoSlopeNorm:
    """White at zero, full blue at the field's 0.5th percentile and full red at its 99.5th, so a
    few extreme samples do not wash out the rest. Neither side is stretched more than four times
    the other, so a slight dip below zero in a positive field stays nearly white."""
    lo, hi = (float(v) for v in np.percentile(f, [0.5, 99.5]))
    tiny = 1e-6 * max(abs(lo), abs(hi), 1e-30)
    vmin = min(lo, -abs(hi) / 4, -tiny)
    vmax = max(hi, abs(lo) / 4, tiny)
    return TwoSlopeNorm(0.0, vmin, vmax)


def labels(info) -> list[str]:
    n = int(info.n_imfs)
    return (
        ["input"]
        + [f"mode {k + 1}  ·  window size {int(info.windows[k])}" for k in range(n)]
        + ["residue"]
    )


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)
    # a 256-colour palette keeps the files small without visible banding at this size
    Image.open(path).convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT).save(
        path, optimize=True
    )
    print(f"{path}  {path.stat().st_size // 1024} KiB")


def figure_1d() -> None:
    parts = signal_1d()
    x = sum(parts.values())
    y, info = eda.decompose(x.astype(np.float32), return_info=True)
    n = int(info.n_imfs)
    rows = y[: n + 1].astype(np.float64)
    # mode 1 is the chirp, the residue the trend, and every mode between them the cosine
    ours = {"chirp": rows[0], "cosine": rows[1:n].sum(axis=0), "trend": rows[n]}
    between = "mode 2" if n == 2 else f"modes 2–{n}"
    ours_name = {"chirp": "mode 1", "cosine": between, "trend": "residue"}
    theirs, (low, high) = fourier_bands(x, parts)
    theirs_name = {
        "chirp": f"modes above {high:.0f}",
        "cosine": f"modes {low:.0f} to {high:.0f}",
        "trend": f"modes below {low:.0f}",
    }
    t = np.arange(x.size)

    fig = plt.figure(figsize=(12, 7.6))
    grid = fig.add_gridspec(4, 2, height_ratios=(1.3, 1, 1, 1), hspace=0.62, wspace=0.08)
    top = fig.add_subplot(grid[0, 0])
    top.plot(t, x, lw=0.6, color=INK)
    top.set_title("input = chirp + cosine + trend", fontsize=10, loc="left")
    top.set_yticks([])
    freq = fig.add_subplot(grid[0, 1])
    for key, f in frequencies_1d().items():
        freq.plot(t, f, lw=2.0, color=INK)
        freq.annotate(
            key,
            (t[-1], f[-1]),
            xytext=(-4, 4),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=9,
        )
    for c in (low, high):
        freq.axhline(c, color="#d9480f", lw=1.0, ls=(0, (4, 2)))
    freq.set_ylim(0, 160)
    freq.set_title(
        "frequency in cycles per signal; dashed: the Fourier band cutoffs",
        fontsize=10,
        loc="left",
    )
    for row, key in enumerate(parts, start=1):
        for col, (estimate, colour, label) in enumerate(
            (
                (ours[key], "#0a2cff", f"eda {ours_name[key]}"),
                (theirs[key], "#d9480f", f"sum of Fourier {theirs_name[key]}"),
            )
        ):
            ax = fig.add_subplot(grid[row, col])
            ax.plot(t, parts[key], lw=1.8, color="#c9ced8")
            ax.plot(t, estimate, lw=0.7, color=colour)
            error = relative_error(estimate, parts[key])
            ax.set_title(
                f"{key} = {label}   ·   error {100 * error:.0f} %", fontsize=10, loc="left"
            )
            ax.set_yticks([])
            if row < 3:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("sample", fontsize=9)
    for ax in fig.axes:
        ax.margins(x=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_visible(ax is freq)
        for side in ("bottom", "left"):
            ax.spines[side].set_color("#d9dde5")
    fig.suptitle(
        "Two oscillations that share frequencies, on a trend (grey: the true parts)",
        x=0.02,
        ha="left",
        fontsize=13,
    )
    fig.subplots_adjust(left=0.03, right=0.99, top=0.9, bottom=0.07)
    save(fig, "decomposition-1d.png")


def tiles(fields, names, title, file):
    fig, axes = plt.subplots(1, len(fields), figsize=(2.55 * len(fields), 3.0))
    for ax, f, name in zip(axes, fields, names, strict=True):
        ax.imshow(f, cmap=BWR, norm=zero_white(f), interpolation="nearest")
        ax.set_title(name, fontsize=10, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#d9dde5")
    fig.suptitle(title, x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, file)


# A Gaussian blur of standard deviation sigma halves the amplitude of a wave of period
# DOG_PERIOD * sigma: exp(-2 pi^2 sigma^2 f^2) = 1/2 at f = sqrt(log(2) / 2) / (pi sigma).
DOG_PERIOD = np.pi / np.sqrt(np.log(2) / 2)


def differences_of_gaussians(x: np.ndarray, windows: np.ndarray) -> list[np.ndarray]:
    """Band k is the blur at sigma_(k-1) minus the blur at sigma_k, with sigma_0 = 0 (x itself) and
    sigma_k = w_k / DOG_PERIOD for eda's window sizes w_k; the last row is the last blur. The rows
    add up to x, as eda's do."""
    blurs = [x] + [gaussian_filter(x, w / DOG_PERIOD, mode="reflect") for w in windows]
    return [blurs[k] - blurs[k + 1] for k in range(len(windows))] + [blurs[-1]]


def near_edge_share(rows, x: np.ndarray, reach: int = 4):
    """The share of each row's energy within `reach` pixels of the strong edges of x (Canny), and
    the share of the pixels that lie there."""
    edges = feature.canny(x, sigma=2.0, low_threshold=0.1, high_threshold=0.2)
    strip = binary_dilation(edges, iterations=reach)
    return float(strip.mean()), [float((r[strip] ** 2).sum() / (r**2).sum()) for r in rows]


def figure_camera() -> None:
    x = data.camera().astype(np.float64) / 255
    y, info = eda.decompose(x.astype(np.float32), return_info=True)
    n = int(info.n_imfs)
    windows = info.windows[:n].astype(np.float64)
    rows = {
        "eda": (
            [r.astype(np.float64) for r in y[: n + 1]],
            [f"mode {k + 1}  ·  window size {int(w)}" for k, w in enumerate(windows)] + ["residue"],
        ),
        "difference of Gaussians": (
            differences_of_gaussians(x, windows),
            [f"band {k + 1}  ·  σ {w / DOG_PERIOD:.1f}" for k, w in enumerate(windows)]
            + ["last blur"],
        ),
    }
    fig = plt.figure(figsize=(2.3 * (n + 3), 5.2))
    grid = fig.add_gridspec(2, n + 2, width_ratios=[2] + [1] * (n + 1), wspace=0.06, hspace=0.18)
    ax = fig.add_subplot(grid[:, 0])
    ax.imshow(x, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("input", fontsize=10, color=INK)
    for row, (method, (fields, names)) in enumerate(rows.items()):
        for col, (f, name) in enumerate(zip(fields, names, strict=True), start=1):
            ax = fig.add_subplot(grid[row, col])
            if col == len(fields):  # the residue and the last blur: smooth images
                ax.imshow(f, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.imshow(f, cmap="gray", norm=zero_white(f), interpolation="nearest")
            ax.set_title(name, fontsize=9, color=INK)
            if col == 1:
                ax.set_ylabel(method, fontsize=10, color=INK)
    for ax in fig.axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#d9dde5")
    fig.suptitle(
        "The cameraman: eda's modes and differences of Gaussians at the same scales",
        x=0.01,
        ha="left",
        fontsize=13,
    )
    fig.subplots_adjust(left=0.01, right=0.995, top=0.86, bottom=0.02)
    save(fig, "comparison-dog.png")
    area, ours = near_edge_share(rows["eda"][0][:n], x)
    _, theirs = near_edge_share(rows["difference of Gaussians"][0][:n], x)
    print(f"energy within 4 px of the strong edges ({100 * area:.0f} % of the pixels), per row:")
    print("  eda ", " ".join(f"{100 * v:.0f} %" for v in ours))
    print("  DoG ", " ".join(f"{100 * v:.0f} %" for v in theirs))


def figure_2d() -> None:
    x = image_2d()
    y, info = eda.decompose(x, return_info=True)
    n = int(info.n_imfs)
    tiles(
        [x] + list(y[: n + 1]),
        labels(info),
        "An image split into its oscillations, finest first",
        "decomposition-2d.png",
    )


# The monogenic signal of the modes: amplitude, phase, frequency and direction at every sample.
TWILIGHT = plt.get_cmap("twilight")  # cyclic: light at a crest (phase 0), dark at a trough
REDS = LinearSegmentedColormap.from_list("white_red", ["#ffffff", "#ff1a1a"])


def strength(amplitude: np.ndarray) -> np.ndarray:
    """1 where a mode is strong, falling to 0 where it is weak (against its 99th percentile)."""
    return np.clip(amplitude / max(float(np.percentile(amplitude, 99)), 1e-30), 0, 1) ** 0.7


def fade(rgb: np.ndarray, weight: np.ndarray, ground: float = 0.92) -> np.ndarray:
    """Fades colours to a light grey where the weight (0 to 1) is small."""
    w = np.clip(weight, 0, 1)[..., None]
    return rgb[..., :3] * w + ground * (1 - w)


def key(fig, rect, cmap, ticks, title):
    ax = fig.add_axes(rect)
    ax.imshow(np.linspace(0, 1, 256)[None, :], cmap=cmap, aspect="auto")
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 255, len(ticks)))
    ax.set_xticklabels(ticks, fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")


def figure_monogenic_1d() -> None:
    parts, truth = signal_1d(), frequencies_1d()
    x = sum(parts.values())
    y, info = eda.decompose(x.astype(np.float32), return_info=True)
    n = int(info.n_imfs)
    # the chirp is mode 1 and the cosine every mode before the residue, as in figure_1d
    pair = np.stack([y[0], y[1:n].sum(axis=0)]).astype(np.float32)
    m = eda.monogenic(pair, axes=(1,))
    size, t = x.size, np.arange(x.size)
    inner = slice(size // 20, size - size // 20)
    names = {"chirp": "mode 1", "cosine": "mode 2" if n == 2 else f"modes 2–{n}"}
    level = {"chirp": 1.0, "cosine": 1.2}
    fig = plt.figure(figsize=(12, 7.6))
    grid = fig.add_gridspec(3, 1, height_ratios=(1, 1, 1.6), hspace=0.5)
    for i, part in enumerate(("chirp", "cosine")):
        ax = fig.add_subplot(grid[i])
        ax.plot(t, pair[i], lw=0.6, color="#0a2cff")
        for sign in (1, -1):
            ax.axhline(sign * level[part], lw=2.4, color="#c9ced8")
            ax.plot(t, sign * m.amplitude[i], lw=1.3, color="#ff1a1a")
        err = np.median(np.abs(m.amplitude[i][inner] - level[part])) / level[part]
        ax.set_title(
            f"{part} = eda {names[part]} (blue) and its amplitude ± A (red)   ·   median error "
            f"{100 * err:.1f} %",
            fontsize=10,
            loc="left",
        )
        ax.set_yticks([])
        ax.set_xticklabels([])
        print(f"1-D {part}: median amplitude error {100 * err:.2f} %")
    ax = fig.add_subplot(grid[2])
    errors = []
    for i, (part, colour) in enumerate((("chirp", "#0a2cff"), ("cosine", "#ff1a1a"))):
        ax.plot(t, truth[part], lw=3.2, color="#c9ced8")
        ax.plot(t, m.frequency[i] * size, lw=0.9, color=colour)
        rel = np.abs(m.frequency[i][inner] * size - truth[part][inner]) / truth[part][inner]
        errors.append(f"{part} {100 * np.median(rel):.1f} %")
        print(f"1-D {part}: median frequency error {100 * np.median(rel):.2f} %")
    ax.set_ylim(0, 170)
    ax.set_ylabel("cycles per signal", fontsize=9)
    ax.set_xlabel("sample", fontsize=9)
    ax.set_title(
        "instantaneous frequency of the chirp (blue) and the cosine (red); grey: the true "
        f"frequencies   ·   median error {', '.join(errors)}",
        fontsize=10,
        loc="left",
    )
    for ax in fig.axes:
        ax.margins(x=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color("#d9dde5")
    fig.suptitle(
        "eda.monogenic on the modes of the 1-D example: amplitude and frequency at every sample",
        x=0.02,
        ha="left",
        fontsize=13,
    )
    fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.07)
    save(fig, "monogenic-1d.png")


def monogenic_panels(mode, m, k):
    a = m.amplitude[k]
    return [
        BWR(zero_white(mode)(mode))[..., :3],
        REDS(np.clip(a / max(float(np.percentile(a, 99.5)), 1e-30), 0, 1))[..., :3],
        fade(TWILIGHT(np.mod(m.phase[k] / (2 * np.pi), 1)), strength(a)),
    ]


def direction_segments(ax, mode, n, coherence, amplitude, per_side=28, colour=INK):
    """The direction n as short segments, across the stripes, on a pale copy of the mode. n has no
    sign, so each cell averages the doubled angle, weighted by the amplitude and the coherence; a
    segment's length is how consistent the direction is in its cell, and cells where the mode is
    weak get none."""
    pale = 0.35 * BWR(zero_white(mode)(mode))[..., :3] + 0.65
    ax.imshow(pale, interpolation="nearest")
    h, w = mode.shape
    step = max(h, w) / per_side
    weight = strength(amplitude)
    angle_n = np.arctan2(n[0], n[1])  # in (x = column, y = row) coordinates
    doubled = coherence * weight * np.exp(2j * angle_n)
    xs, ys, us, vs = [], [], [], []
    for r0 in np.arange(0, h - step / 2, step):
        for c0 in np.arange(0, w - step / 2, step):
            cell = (slice(int(r0), int(r0 + step)), slice(int(c0), int(c0 + step)))
            total = weight[cell].sum()
            if total <= 0 or weight[cell].mean() < 0.2:
                continue
            mean = doubled[cell].sum() / total
            length = 0.9 * step * min(abs(mean), 1.0)
            if length < 0.15 * step:
                continue
            angle = np.angle(mean) / 2
            xs.append(c0 + step / 2)
            ys.append(r0 + step / 2)
            us.append(length * np.cos(angle))
            vs.append(length * np.sin(angle))
    ax.quiver(
        xs,
        ys,
        us,
        vs,
        angles="xy",
        scale_units="xy",
        scale=1,
        pivot="middle",
        headwidth=0,
        headlength=0,
        headaxislength=0,
        width=0.006,
        color=colour,
    )
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)


def figure_monogenic_2d() -> None:
    x = image_2d()
    y, info = eda.decompose(x, return_info=True)
    m = eda.monogenic(y[:2], axes=(1, 2))  # mode 1, the rings, and mode 2, the wavy stripes
    titles = ["mode", "amplitude", "phase", "direction"]
    fig = plt.figure(figsize=(10.8, 6.9))
    grid = fig.add_gridspec(
        2, 4, left=0.05, right=0.995, top=0.88, bottom=0.16, wspace=0.04, hspace=0.1
    )
    for k, what in enumerate(("rings", "wavy stripes")):
        strong = strength(m.amplitude[k]) > 0.5
        period = np.median(1.0 / np.abs(m.frequency[k][strong]))
        print(f"2-D {what}: median period 1/frequency {period:.1f} px where the mode is strong")
        panels = monogenic_panels(y[k], m, k)
        for j, title in enumerate(titles):
            ax = fig.add_subplot(grid[k, j])
            if j == 3:
                direction_segments(ax, y[k], m.direction[k], m.coherence[k], m.amplitude[k])
            else:
                ax.imshow(panels[j], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#d9dde5")
            if k == 0:
                ax.set_title(title, fontsize=11)
            if j == 0:
                ax.set_ylabel(
                    f"mode {k + 1} ({what})\nwindow size {int(info.windows[k])}", fontsize=10
                )
    y0, h = 0.045, 0.03
    key(fig, (0.05, y0, 0.18, h), REDS, ["0", "max"], "amplitude")
    key(
        fig,
        (0.3, y0, 0.22, h),
        TWILIGHT,
        ["0 crest", "π/2", "±π trough", "−π/2", "0"],
        "phase (fades to grey where the mode is weak)",
    )
    fig.text(
        0.6,
        y0,
        "direction: segments across the stripes,\nlonger where they have one clear direction",
        fontsize=9,
        color=INK,
    )
    fig.suptitle(
        "eda.monogenic on the first two modes of the image at the top",
        x=0.01,
        ha="left",
        fontsize=13,
    )
    save(fig, "monogenic-2d.png")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--probe":
        one_d = lambda: sum(signal_1d().values()).astype(np.float32)  # noqa: E731
        for name, f in (("1d", one_d), ("2d", image_2d)):
            _, info = eda.decompose(f(), return_info=True)
            print(name, int(info.n_imfs), info.windows.tolist())
        sys.exit(0)
    figure_1d()
    figure_2d()
    figure_camera()
    figure_monogenic_1d()
    figure_monogenic_2d()
