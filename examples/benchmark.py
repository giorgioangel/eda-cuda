# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
"""Time eda.decompose on a table of sizes, in alternating fresh processes, and print a table.

    python examples/benchmark.py [--cases a,b] [--rounds 3] [--reps 5] [--input multi|noise]
        [--compare-python PY --compare-path DIR] [--baseline-path DIR [--baseline-python PY]]
        [--tolerance 0.01] [--noise 0.03] [--gate] [--list] [--json OUT]

A is eda.decompose of the interpreter that runs this script (put a build tree on PYTHONPATH to
time it). B, with --compare-python and --compare-path, is the call of the predecessor
implementation in its own interpreter and source tree. C, with --baseline-path, is another build of
eda (for example the commit before a change), imported from that directory: it detects a
regression of eda against itself. Each side is timed in a fresh process: one warm-up call, then
``--reps`` timed calls (the call returns complete host results), and the median is kept.
A round runs A B C A B C; the speed-up of A over a side in a round is the mean of the side's two
medians over the mean of A's two (above 1: A is faster), and the case reports the median over
``--rounds`` rounds. The spread is the largest relative difference between A's two medians in a
round, which is the noise of one round. The verdict of a side is "A slower" only when A's
speed-up over it is below 1 / (1 + max(tolerance, spread)), so the smallest slowdown of A that a
run can report is max(tolerance, spread). A spread above ``--noise`` makes the verdict "noisy"
unless A is clearly faster than the side (a speed-up above 1 + the spread): run again, with more
``--reps``. A baseline that imports the same eda as A is "same build". "same" tells whether the
outputs of A and the side are bitwise equal ("different" wins over "noisy"). With --gate the exit
status is 1 when a case is "A slower", noisy, different or "same build", when a side that was
asked for is unavailable, or when there is nothing to compare.

Run it on an idle GPU; the table is only as good as the spread column says.
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time

# name: (dtype, batch, spatial shape)
CASES = {
    "d1_1024_f32": ("float32", 1, (1024,)),
    "d1_8192_f32": ("float32", 1, (8192,)),
    "d1_4096_f64": ("float64", 1, (4096,)),
    "b256_d1_4096_f32": ("float32", 256, (4096,)),
    "d2_128_f32": ("float32", 1, (128, 128)),
    "d2_256_f32": ("float32", 1, (256, 256)),
    "d2_512_f32": ("float32", 1, (512, 512)),
    "d2_1024_f32": ("float32", 1, (1024, 1024)),
    "d2_1024_f64": ("float64", 1, (1024, 1024)),
    "d2_2048x512_f32": ("float32", 1, (2048, 512)),
    "d2_2048_f32": ("float32", 1, (2048, 2048)),
    "b64_d2_64_f32": ("float32", 64, (64, 64)),
    "d3_32_f32": ("float32", 1, (32, 32, 32)),
    "d3_64_f32": ("float32", 1, (64, 64, 64)),
    "d3_128_f32": ("float32", 1, (128, 128, 128)),
    "d3_128_f64": ("float64", 1, (128, 128, 128)),
    "b16_d3_32_f32": ("float32", 16, (32, 32, 32)),
}

SIDES = ("A", "B", "C")


def make_input(kind: str, batch: int, shape: tuple[int, ...], dtype: str):
    """ "multi": three sinusoids of periods 128, 24 and 5.3 along the diagonal plus noise of
    standard deviation 0.05; "noise": standard normal. Seed 7."""
    import numpy as np

    rng = np.random.default_rng(7)
    full = (batch, *shape)
    if kind == "noise":
        x = rng.standard_normal(full)
    else:
        t = np.indices(shape, dtype=np.float64).sum(axis=0) / math.sqrt(len(shape))
        signal = sum(
            a * np.sin(2 * np.pi * t / p) for a, p in ((1.0, 128.0), (0.6, 24.0), (0.3, 5.3))
        )
        x = signal[None] + 0.05 * rng.standard_normal(full)
    return np.ascontiguousarray(x.astype(dtype))


def worker(args: argparse.Namespace) -> None:
    """One process: warm-up, timed calls, one JSON line on standard output."""
    import numpy as np

    if args.path:
        sys.path.insert(0, args.path)
    dtype, batch, shape = CASES[args.case]
    x = make_input(args.input, batch, shape, dtype)
    axes = tuple(range(len(shape))) if batch == 1 else tuple(range(1, len(shape) + 1))
    if batch == 1:
        x = x[0]
    if args.impl == "eda":
        import eda

        origin = eda.__file__

        def call():
            return eda.decompose(x, axes=axes)
    else:  # the predecessor implementation; its names are built here and appear nowhere in the tree
        module = __import__(codecs.decode("rzq_tch", "rot13"))
        entry = getattr(module, codecs.decode("rzq_aq", "rot13"))
        method = codecs.decode("ynfp", "rot13")
        origin = module.__file__

        def call():
            return entry(x, method=method, axes=axes)

    y = call()
    times = []
    for _ in range(args.reps):
        start = time.perf_counter()
        y = call()
        times.append(time.perf_counter() - start)
    y = np.ascontiguousarray(y)
    print(
        json.dumps(
            {
                "median": statistics.median(times),
                "times": times,
                "sha256": hashlib.sha256(y.tobytes()).hexdigest(),
                "shape": list(y.shape),
                "origin": origin,
            }
        )
    )


def run_side(side: str, case: str, args: argparse.Namespace) -> dict:
    """Run one worker process for a side; returns its JSON line or an error record."""
    interpreter, impl, path = sys.executable, "eda", ""
    env = dict(os.environ)
    if side == "B":
        interpreter, impl, path = args.compare_python, "pred", args.compare_path
        env.pop("LD_LIBRARY_PATH", None)
    elif side == "C":
        interpreter, path = args.baseline_python or sys.executable, args.baseline_path
    command = [interpreter, os.path.abspath(__file__), "--worker", impl, case]
    command += ["--reps", str(args.reps), "--input", args.input]
    if path:
        command += ["--path", os.path.abspath(path)]
    try:
        done = subprocess.run(command, env=env, capture_output=True, text=True)
    except OSError as error:
        return {"error": str(error)}
    if done.returncode != 0 or not done.stdout.strip():
        return {"error": (done.stderr.strip().splitlines() or ["no output"])[-1]}
    return json.loads(done.stdout.strip().splitlines()[-1])


def summarize(case: str, rounds: list[dict], sides: list[str], args: argparse.Namespace) -> dict:
    """Medians, speed-ups, spread, equality and verdicts of one case."""
    out: dict = {"case": case, "rounds": len(rounds)}
    errors = [r[k]["error"] for r in rounds for k in r if k[0] == "A" and "error" in r[k]]
    if errors:
        out["error"] = errors[0]
        return out
    a_first = [r["A1"]["median"] for r in rounds]
    a_second = [r["A2"]["median"] for r in rounds]
    a_mean = [(p + q) / 2 for p, q in zip(a_first, a_second, strict=True)]
    out["A_ms"] = 1e3 * statistics.median(a_mean)
    out["spread"] = max(abs(q / p - 1) for p, q in zip(a_first, a_second, strict=True))
    out["resolution"] = max(args.tolerance, out["spread"])
    out["origin"] = {"A": rounds[0]["A1"]["origin"]}
    for side in sides[1:]:
        other = [(r[side + "1"]["median"] + r[side + "2"]["median"]) / 2 for r in rounds]
        ratios = [o / a for o, a in zip(other, a_mean, strict=True)]
        out[side + "_ms"] = 1e3 * statistics.median(other)
        out[side + "_speedup"] = statistics.median(ratios)
        out[side + "_same"] = all(
            r[side + k]["sha256"] == r["A1"]["sha256"] for r in rounds for k in "12"
        ) and all(r["A2"]["sha256"] == r["A1"]["sha256"] for r in rounds)
        out["origin"][side] = rounds[0][side + "1"]["origin"]
        if side == "C" and out["origin"]["C"] == out["origin"]["A"]:
            verdict = "same build"
        elif not out[side + "_same"]:
            verdict = "different"
        elif out[side + "_speedup"] < 1 / (1 + out["resolution"]):
            verdict = "A slower"
        elif out["spread"] > args.noise and out[side + "_speedup"] < 1 + out["resolution"]:
            verdict = "noisy"
        else:
            verdict = "ok"
        out[side + "_verdict"] = verdict
    return out


def table(rows: list[dict], sides: list[str], asked: list[str]) -> str:
    head = ["case", "A ms"]
    for side in asked:
        head += [f"{side} ms", f"{side}/A", f"{side} same", f"{side} verdict"]
    head += ["spread %"]
    lines = [head]
    for r in rows:
        if "error" in r:
            lines.append([r["case"], "error: " + r["error"]])
            continue
        line = [r["case"], f"{r['A_ms']:.2f}"]
        for side in asked:
            if side in sides:
                line += [
                    f"{r[side + '_ms']:.2f}",
                    f"{r[side + '_speedup']:.2f}x",
                    "yes" if r[side + "_same"] else "NO",
                    r[side + "_verdict"],
                ]
            else:
                line += ["-", "-", "-", "unavailable"]
        line += [f"{100 * r['spread']:.1f}"]
        lines.append(line)
    widths = [max(len(row[i]) for row in lines if i < len(row)) for i in range(len(head))]
    return "\n".join("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)) for row in lines)


def parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cases", help="comma-separated case names (default: all)")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--input", choices=("multi", "noise"), default="multi")
    p.add_argument(
        "--compare-python", help="interpreter of the predecessor implementation (side B)"
    )
    p.add_argument("--compare-path", help="source directory of the predecessor implementation")
    p.add_argument("--baseline-path", help="directory of another eda build (side C)")
    p.add_argument("--baseline-python", help="interpreter of side C (default: this one)")
    p.add_argument(
        "--tolerance", type=float, default=0.01, help="smallest slowdown to report (default 0.01)"
    )
    p.add_argument(
        "--noise", type=float, default=0.03, help="largest A-vs-A spread that gives a verdict"
    )
    p.add_argument(
        "--gate",
        action="store_true",
        help="exit 1 on A slower, noisy, different, same build or unavailable",
    )
    p.add_argument("--list", action="store_true", help="print the size table and exit")
    p.add_argument("--json", help="write the per-case results to this file")
    p.add_argument("--worker", nargs=2, metavar=("IMPL", "CASE"), help=argparse.SUPPRESS)
    p.add_argument("--path", default="", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.worker:
        args.impl, args.case = args.worker
        return args
    if args.rounds < 1 or args.reps < 1:
        p.error("--rounds and --reps must be at least 1")
    if bool(args.compare_python) != bool(args.compare_path):
        p.error("--compare-python and --compare-path go together")
    args.names = args.cases.split(",") if args.cases else list(CASES)
    unknown = [n for n in args.names if n not in CASES]
    if unknown:
        p.error("unknown case(s): " + ", ".join(unknown))
    return args


def main(argv: list[str]) -> int:
    args = parse(argv)
    if args.worker:
        worker(args)
        return 0
    if args.list:
        for name in args.names:
            dtype, batch, shape = CASES[name]
            print(f"{name:20s} {dtype:8s} batch {batch:4d}  shape {'x'.join(map(str, shape))}")
        return 0
    asked = [s for s, wanted in (("B", args.compare_python), ("C", args.baseline_path)) if wanted]
    order = ["A", *asked, "A", *asked]
    seen: dict[str, int] = {}
    labels = []
    for side in order:
        seen[side] = seen.get(side, 0) + 1
        labels.append(f"{side}{seen[side]}")
    rows = []
    for case in args.names:
        rounds = []
        for _ in range(args.rounds):
            rounds.append({label: run_side(label[0], case, args) for label in labels})
        available = [s for s in asked if all("error" not in r[s + "1"] for r in rounds)]
        rows.append(summarize(case, rounds, ["A", *available], args))
        print(f"  {case} done", file=sys.stderr, flush=True)
    sides = ["A", *[s for s in asked if any(s + "_ms" in r for r in rows)]]
    print(table(rows, sides, asked))
    spreads = [r["spread"] for r in rows if "spread" in r]
    resolution = max([args.tolerance, *spreads])
    print(
        f"\n{args.rounds} rounds of {args.reps} calls; smallest slowdown this run can report: "
        f"{100 * resolution:.1f} % (largest A-vs-A spread {100 * max(spreads, default=0):.1f} %, "
        f"tolerance {100 * args.tolerance:.1f} %); noise bound {100 * args.noise:.1f} %"
    )
    origins = {k: v for r in rows if "origin" in r for k, v in r["origin"].items()}
    print("imported from: " + "; ".join(f"{k} {v}" for k, v in sorted(origins.items())))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=1)
    if not args.gate:
        return 0
    problems = []
    if not asked:
        problems.append("nothing to compare against (give --compare-python or --baseline-path)")
    for r in rows:
        if "error" in r:
            problems.append(f"{r['case']}: {r['error']}")
            continue
        for side in asked:
            verdict = r.get(side + "_verdict", "unavailable")
            if verdict != "ok":
                problems.append(f"{r['case']}: side {side}: {verdict}")
    for p in problems:
        print("gate: " + p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
