#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
#
# tools/sanitize.sh {memcheck|initcheck|racecheck|synccheck} {ctest|pytest}
#
# Runs the C++ tests (ctest) or a covering selection of the Python tests (pytest) under one
# compute-sanitizer tool, one run at a time, each run limited to 300 s. Before the real runs the
# tool must report that tool's canary (cpp/tests/canary.cu: a planted fault per tool); a canary
# that is not reported means the tool cannot be trusted here, and the script stops with exit 1.
# The full matrix is the eight invocations (four tools, two targets).
#
# A run is clean only when its command exits 0 and its log holds no sanitizer report. A report
# (exit 9), a timeout (exit 124), a test that fails under the tool, a selection that does not
# exist and an interop selection without SANITIZE_INTEROP_PYTHON all end the script with exit 1.
# Exit 75 and 76 (a guard in SANITIZE_RUN_PREFIX paused the run or the GPU is busy) are passed
# through; the caller waits and runs the script again.
#
# Environment (the library rejects unknown EDA_* variables, so the script's own inputs are not
# named EDA_*):
#   SANITIZE_BUILD           build tree with cpp/tests (-DEDA_BUILD_TESTS=ON) and python/ (default build/tests)
#   SANITIZE_PYTHON          interpreter of the pytest runs (default .venv/bin/python); its nvidia-cufft
#                            wheel supplies libcufft.so.12 to the C++ tests as well
#   SANITIZE_INTEROP_PYTHON  interpreter with CuPy and pytest for the device-input tests (optional)
#   COMPUTE_SANITIZER        the tool (default compute-sanitizer)
#   SANITIZE_RUN_PREFIX      words put before every GPU command, for example "flock FILE capped --gpu"
#   SANITIZE_CPU_PREFIX      words put before the commands that need no GPU (the library check)
#   SANITIZE_LOGS            log directory (default build/sanitize)
#
# Every run sets EDA_SANITIZE=1 (small test shapes, the workspace arena in debug mode, zeroed
# output rows) and --error-exitcode 9; memcheck adds --leak-check full --padding 512; initcheck
# sets EDA_ARENA_POISON=0 so that a fill does not count as initialization. No report is suppressed.
# The pytest runs use --target-processes all, so that the interpreters that tests start are
# checked as well; the canaries include a child process. Each run's cuFFT must be the one of the
# parity oracle (the sha256 in tests/data/parity_v2.json): the script checks the library that the
# tests map.
#
# What the case sets reach (ctest runs the tests registered with the label sanitize; the
# sizes shrink under EDA_SANITIZE):
#   test_cube      all cube operations, 1-D to 3-D, both dtypes; lines per CTA 1 to 64; rung groups 1 to 32;
#                  the vHGW warp-carry path (more than 32 chunks per segment: cube widths 1023, 1025, 2999)
#   test_element   the element operator, gathers, interpolation, guards
#   test_stats     the statistics kernels (block summaries, ladders, windows)
#   test_mode      one mode: sifts, commit, stop test
#   test_e2e       whole decompositions, 1-D to 3-D, chunks, batches, repeated calls, new threads
#   test_input     input conversion and casts, strides, layouts (quick mode under racecheck)
#   test_orientation  the orientation path (Riesz band, tensor, eigen), 2-D and 3-D, both dtypes; the
#                  monogenic signal (derivatives, the 1-D transforms, the output planes), 1-D to 3-D
# What the pytest selections add: the public API on host input (test_host, test_rows), the parity
# oracle's quick set (test_parity), the properties, the golden corpus (a 512-sample subset under the
# race and sync tools), the process exit (test_exit), device input through CuPy (test_device), and
# eda.monogenic with decompose's return_orientation (test_monogenic, test_orientation_output; their
# batch tests under the race and sync tools).
# Device input is tested through CuPy only: with torch in the process, memcheck's leak check
# reports torch's cached blocks and the result of a call that is still alive at interpreter exit
# (its buffer is released with the process; eda does not free it before the context is torn down).
# Not reached by any case set: Bluestein cuFFT plans (no axis has a prime factor above 127).
# Known blind spots:
# - Under EDA_SANITIZE the driver zeroes the output rows, so initcheck cannot see a missing row
#   write; test_rows covers that in normal mode.
# - The debug arena gives each workspace slot its own allocation, and under EDA_SANITIZE each
#   scratch field of a chunk is a slot of its own (the orientation's tensor fields, which it reads
#   as one block, share one). The memory pool places allocations back to back when their size is a
#   multiple of 512 bytes, so an access past one field would land in the next field's allocation
#   and memcheck would not report it; --padding 512 puts 512 unallocated bytes after every
#   allocation, and that makes the overrun visible. The canary "pool" checks it: two adjacent
#   2048-byte pool allocations and a write one byte past the first, which memcheck reports only
#   with the padding.
#   A call reuses the scratch fields from mode to mode: initcheck does not see a missing write to
#   a scratch field that an earlier step wrote. The other canaries use their own cudaMalloc
#   buffers, so they show that a tool reports, not that it reaches the scratch fields. racecheck
#   and synccheck do report faults planted in eda's kernels.
#
# Measured on the development machine (RTX 4090, one run at a time; seconds for all runs of an
# invocation, the longest single run in brackets; the limit of one run is 300 s):
#   memcheck   ctest   21 (test_input 13)        pytest  165 (properties 49)
#   initcheck  ctest   11 (test_input 5)         pytest  101 (device 28)
#   synccheck  ctest   11 (test_input 6)         pytest    7 (device 2)
#   racecheck  ctest  568 (test_e2e 178)         pytest  449 (host 142)
# The full matrix takes about 22 minutes.
set -uo pipefail

usage() {
  echo "usage: $0 {memcheck|initcheck|racecheck|synccheck} {ctest|pytest}" >&2
  exit 2
}
tool=${1:-}
target=${2:-}
case $tool in memcheck | initcheck | racecheck | synccheck) ;; *) usage ;; esac
case $target in ctest | pytest) ;; *) usage ;; esac

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build=${SANITIZE_BUILD:-$root/build/tests}
python=${SANITIZE_PYTHON:-$root/.venv/bin/python}
interop=${SANITIZE_INTEROP_PYTHON:-}
sanitizer=${COMPUTE_SANITIZER:-compute-sanitizer}
logs=${SANITIZE_LOGS:-$root/build/sanitize}
limit=300
read -r -a gpu_prefix <<<"${SANITIZE_RUN_PREFIX:-}"
read -r -a cpu_prefix <<<"${SANITIZE_CPU_PREFIX:-}"

[ -x "$python" ] || {
  echo "$0: SANITIZE_PYTHON '$python' is not an executable" >&2
  exit 2
}
mkdir -p "$logs"
cd "$root" || exit 2
export EDA_SANITIZE=1
export PYTHONPATH=$build/python
tool_args=()
[ "$tool" = memcheck ] && tool_args+=(--leak-check full --padding 512)
[ "$tool" = initcheck ] && export EDA_ARENA_POISON=0
[ "$target" = pytest ] && tool_args+=(--target-processes all)

failures=0
rc=0
secs=0

say() { printf '%-10s %-7s %-26s %s\n' "$tool" "$target" "$1" "$2"; }
bad() { failures=$((failures + 1)); }
passthrough() {
  if [ "$rc" = 75 ] || [ "$rc" = 76 ]; then
    say "$1" "exit $rc: the guard paused the run or the GPU is busy; run the script again later"
    exit "$rc"
  fi
}

# sanitized LOG COMMAND...: one run under the tool; sets rc and secs.
sanitized() {
  local log=$1 start=$SECONDS
  shift
  "${gpu_prefix[@]}" timeout "$limit" "$sanitizer" --tool "$tool" "${tool_args[@]}" --error-exitcode 9 "$@" >"$log" 2>&1
  rc=$?
  secs=$((SECONDS - start))
}

# The text a tool prints when it reports.
report_pattern='(ERROR|RACECHECK|LEAK) SUMMARY: [1-9]'

# ------------------------------------------------------------------------------------------
# The cuFFT of every run must be the oracle's.

wanted_sha=$("${cpu_prefix[@]}" "$python" -c '
import json, sys
print(json.load(open(sys.argv[1]))["provenance"]["libcufft_sha256"])' "$root/tests/data/parity_v2.json")

# mapped_cufft_ok PYTHON: the library that eda maps in that interpreter has the oracle's digest.
mapped_cufft_ok() {
  local got
  got=$(env CUDA_VISIBLE_DEVICES='' "${cpu_prefix[@]}" "$1" -c '
import hashlib
from eda import _native
_native.core()
paths = {line.split()[-1] for line in open("/proc/self/maps") if "libcufft" in line.split()[-1]}
print(",".join(hashlib.sha256(open(p, "rb").read()).hexdigest() for p in sorted(paths)))' 2>&1)
  [ "$got" = "$wanted_sha" ]
}

# cufft_dir PYTHON: the directory of libcufft.so.12 in that interpreter's nvidia wheels.
cufft_dir() {
  "${cpu_prefix[@]}" "$1" -c '
import glob, importlib.util, os
spec = importlib.util.find_spec("nvidia")
for base in (spec.submodule_search_locations if spec else []):
    for path in sorted(glob.glob(os.path.join(base, "*", "lib", "libcufft.so.12"))):
        print(os.path.dirname(path))
        raise SystemExit(0)
raise SystemExit(1)'
}

# ------------------------------------------------------------------------------------------
# Canaries

canary_modes() { if [ "$tool" = memcheck ]; then echo "memcheck leak pool"; else echo "$tool"; fi; }
canary_text() {
  case $1 in
    memcheck | pool) echo 'Invalid __global__ write' ;;
    leak) echo 'LEAK SUMMARY: [1-9]' ;;
    initcheck) echo 'Uninitialized __global__ memory read' ;;
    racecheck) echo 'RACECHECK SUMMARY: [1-9]' ;;
    synccheck) echo 'Barrier error detected' ;;
  esac
}

load_canary='import ctypes, sys; sys.exit(ctypes.CDLL(sys.argv[1]).eda_canary(sys.argv[2].encode()))'
start_child='import subprocess, sys; sys.exit(subprocess.run([sys.argv[1], "-c", sys.argv[4], sys.argv[2], sys.argv[3]]).returncode)'

# canaries: every canary of the tool, in the form of the target (the executable; the module in
# the interpreter; the module in a child interpreter). All must be reported: exit 9 and the text.
canaries() {
  local mode form log text ok=1
  local exe=$build/cpp/tests/eda_canary lib=$build/cpp/tests/libeda_canary.so
  local forms=(exe)
  [ "$target" = pytest ] && forms=(module child)
  for mode in $(canary_modes); do
    for form in "${forms[@]}"; do
      log=$logs/canary_${tool}_${form}_${mode}.log
      case $form in
        exe) sanitized "$log" "$exe" "$mode" ;;
        module) sanitized "$log" "$python" -c "$load_canary" "$lib" "$mode" ;;
        child) sanitized "$log" "$python" -c "$start_child" "$python" "$lib" "$mode" "$load_canary" ;;
      esac
      passthrough "canary $mode"
      text=$(canary_text "$mode")
      if [ "$rc" = 9 ] && grep -Eq "$text" "$log"; then
        say "canary $mode ($form)" "reported (${secs}s)"
      else
        say "canary $mode ($form)" "MISSED (exit $rc, ${secs}s): $log"
        ok=0
      fi
    done
  done
  [ "$ok" = 1 ]
}

# ------------------------------------------------------------------------------------------
# One run of the matrix

# judge LABEL LOG [REQUIRED]: classify the finished run and count a failure unless it is clean.
judge() {
  local label=$1 log=$2 required=${3:-} note=""
  passthrough "$label"
  if [ "$rc" = 124 ]; then
    say "$label" "NOT CLEAN: timeout of ${limit}s: $log"
    bad
  elif [ "$rc" = 9 ] || grep -Eq "$report_pattern" "$log"; then
    say "$label" "NOT CLEAN: the tool reported (exit $rc, ${secs}s): $log"
    bad
  elif [ "$rc" != 0 ]; then
    say "$label" "NOT CLEAN: the test failed under the tool (exit $rc, ${secs}s): $log"
    bad
  elif [ -n "$required" ] && ! grep -Eq "$required" "$log"; then
    say "$label" "NOT CLEAN: exit 0 but no '$required' in the log: $log"
    bad
  else
    note=$(grep -Eho '[0-9]+ passed[^=]*|all [0-9]+ checks passed' "$log" | tail -1)
    say "$label" "clean (${secs}s) $note"
  fi
}

run_ctest() {
  local names name bin
  names=$(ctest --test-dir "$build" -N -L sanitize 2>/dev/null | sed -n 's/^ *Test *#[0-9]*: *//p')
  if [ -z "$names" ]; then
    say ctest "NOT RUN: no test with the label sanitize in $build"
    bad
    return
  fi
  local lib libdir extra
  libdir=$(cufft_dir "$python") || {
    say cufft "NOT RUN: no libcufft.so.12 in the nvidia wheels of $python"
    bad
    return
  }
  export LD_LIBRARY_PATH=$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  for name in $names; do
    bin=$build/cpp/tests/$name
    lib=$(ldd "$bin" 2>/dev/null | sed -n 's/^[[:space:]]*libcufft\.so\.12 => \(.*\) (0x.*/\1/p')
    # a test that does not use cuFFT does not link it
    if [ -n "$lib" ] && [ "$(sha256sum "$lib" | cut -d' ' -f1)" != "$wanted_sha" ]; then
      say "$name" "NOT RUN: the loader maps '$lib' for libcufft, not the oracle's"
      bad
      continue
    fi
    # the full input test takes more than 300 s under racecheck: its quick mode (the decision and
    # cast tables, one call read in place and one converted call) takes about 110 s
    extra=()
    [ "$name:$tool" = test_input:racecheck ] && extra=(quick)
    sanitized "$logs/${tool}_ctest_$name.log" "$bin" "${extra[@]}"
    judge "$name${extra:+ ${extra[*]}}" "$logs/${tool}_ctest_$name.log" 'checks passed'
  done
}

# pt LABEL {base|interop} PYTEST_ARGUMENTS...: one pytest run.
pt() {
  local label=$1 which=$2 interp arg log=$logs/${tool}_pytest_$1.log
  shift 2
  interp=$python
  if [ "$which" = interop ]; then
    interp=$interop
    if [ -z "$interp" ]; then
      say "$label" "NOT RUN: SANITIZE_INTEROP_PYTHON is not set"
      bad
      return
    fi
  fi
  for arg in "$@"; do
    case $arg in
      *.py) if [ ! -e "$arg" ]; then
        say "$label" "NOT RUN: $arg does not exist"
        bad
        return
      fi ;;
    esac
  done
  if ! mapped_cufft_ok "$interp"; then
    say "$label" "NOT RUN: $interp does not map the oracle's libcufft"
    bad
    return
  fi
  sanitized "$log" "$interp" -m pytest -p no:cacheprovider "$@"
  # a run that skipped its GPU tests proves nothing
  if [ "$rc" = 0 ] && grep -Eq "no CUDA device is visible|extension is not available|CUDA is not usable|could not import 'cupy'" "$log"; then
    say "$label" "NOT CLEAN: tests were skipped for lack of a GPU or of CuPy: $log"
    bad
    return
  fi
  judge "$label" "$log" ' passed'
}

run_pytest() {
  local t=tests/python
  if [ "$tool" = memcheck ] || [ "$tool" = initcheck ]; then
    pt host_rows base $t/test_host.py $t/test_rows.py
    pt parity base $t/test_parity.py
    pt properties base $t/test_properties.py
    pt golden base $t/test_golden.py
    pt exit base $t/test_exit.py
    pt monogenic base $t/test_monogenic.py $t/test_orientation_output.py
    # not the torch tests: torch's caching allocator and a result still alive at interpreter exit
    # are reported as leaks that are not eda's (see the header)
    pt device interop $t/test_device.py -k "not torch and not consumer and not import_order and not unsupported and not empty_batch"
  else
    pt host base $t/test_host.py -k "result_type or axes_none or threads_give"
    pt properties base $t/test_properties.py -k "gain or sign or batch"
    pt golden base $t/test_golden.py -k 512
    pt monogenic base $t/test_monogenic.py $t/test_orientation_output.py -k batch
    pt device interop $t/test_device.py -k "(cupy_dlpack or interface) and (tones or gabor or nan-status or read_only or old_interface or host_pointer)"
  fi
}

# ------------------------------------------------------------------------------------------

if [ ! -x "$build/cpp/tests/eda_canary" ] || [ ! -e "$build/cpp/tests/libeda_canary.so" ]; then
  say canaries "NOT RUN: eda_canary and libeda_canary.so are missing from $build/cpp/tests"
  exit 1
fi
if ! canaries; then
  say canaries "a canary was missed: the verdict of this tool cannot be trusted; no test was run"
  exit 1
fi
if [ "$target" = ctest ]; then run_ctest; else run_pytest; fi

if [ "$failures" = 0 ]; then
  say "$tool $target" "CLEAN"
  exit 0
fi
say "$tool $target" "$failures run(s) not clean"
exit 1
