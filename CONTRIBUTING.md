# Contributing

## Reporting a problem

Open an issue with a minimal example and the output of

```sh
python -c "import eda; print(eda.__version__); print(eda.build_info())"
```

together with the Linux distribution, the Python version and how eda was installed. Report
security problems as described in [SECURITY.md](SECURITY.md), not in an issue.

## Platform

eda supports Linux on x86-64 with NVIDIA GPUs of compute capability 8.0 or newer. A build from
the repository makes the package `eda-cuda` and needs the CUDA toolkit 13.0 or newer. The CUDA 12
package `eda-cuda-cu12` needs a CUDA 12.8 toolkit or newer and is built by `tools/build_wheels.sh`
(see Release).

## Build and test

```sh
uv sync --group dev
uv run pytest
uv run ruff check .
```

`uv sync` builds the extension. The GPU tests need an NVIDIA GPU and are skipped when none is
visible; `pytest -m "not gpu"` runs the tests that need none, `pytest -m "gpu or full"` the GPU
tests and the complete parity run. The tests of device input and of memory behaviour need CuPy
(and PyTorch): `uv sync --group interop`. The Python package builds for the architectures of the
release wheel unless `CUDAARCHS` names others: `CUDAARCHS=native` builds for the GPU of the build
machine only, `CUDAARCHS=89-real` for compute capability 8.9. A CMake build without
`CMAKE_CUDA_ARCHITECTURES` or `CUDAARCHS` targets the GPU of the build machine. The test data in
`tests/data/` is frozen: do not regenerate it.

The C++ tests:

```sh
cmake -S . -B build/tests -G Ninja -DCMAKE_BUILD_TYPE=Release -DEDA_BUILD_PYTHON=OFF -DEDA_BUILD_TESTS=ON
cmake --build build/tests
ctest --test-dir build/tests --output-on-failure
```

`ctest -L cpu` runs only the tests that need no GPU. The C++ tests find cuFFT through
`LD_LIBRARY_PATH`: point it at the `nvidia/cu13/lib` directory of the `nvidia-cufft` wheel, for
example `.venv/lib/python3.12/site-packages/nvidia/cu13/lib`. To test the Python package from a
build tree, build with `-DEDA_BUILD_PYTHON=ON` and set `PYTHONPATH=build/tests/python`.

## Naming

Names in the code, the tests, the docs and the commit messages say `eda`. No name from an earlier
project may appear in a tracked file or a commit message; the earlier implementation is called "the
predecessor implementation". `test_no_legacy_names` in `tests/python/test_api.py` checks the
tracked files (`pytest tests/python/test_api.py::test_no_legacy_names`), and so does CI. Commit
subjects begin with `eda: ` and stay within 72 characters; commit bodies are plain.

## Tools

`tools/sanitize.sh TOOL TARGET` runs the C++ tests (`ctest`) or a covering selection of the
Python tests (`pytest`) under one compute-sanitizer tool (`memcheck`, `initcheck`, `racecheck`,
`synccheck`), one run at a time and at most 300 s each. Planted faults (canaries) must be reported
first, or the script stops. The script's header lists its environment variables (`SANITIZE_BUILD`
for the build tree, `SANITIZE_RUN_PREFIX` for a command put before every GPU command, and others)
and what each case set reaches. The full matrix is the eight invocations.

`tools/readme_figures.py [OUT_DIR]` renders the README figures into `docs/images`; it needs a
GPU, matplotlib and scikit-image, and reports the numbers the captions quote.

`examples/benchmark.py` times `eda.decompose` on a table of sizes in alternating fresh processes.
`--baseline-path DIR` compares with another build of eda (to check that a change did not slow
the library down), `--compare-python PY --compare-path DIR` with the predecessor implementation;
`--gate` turns slower, noisy or differing results, a baseline of the same build and a side that is
unavailable into an exit status of 1. The footer states the smallest slowdown that the run could
have reported. Run it on an idle GPU.

## Release

eda-cuda is not published on PyPI. A release is a Git tag and a GitHub release with the wheels and
the sdists of two packages attached. Both packages come from the same tree and version, and both
install the Python package `eda`:

- `eda-cuda`: CUDA 13.4, depends on `nvidia-cufft>=12,<13` (cuFFT 12, `libcufft.so.12`), needs a
  driver for CUDA 13.0 or newer.
- `eda-cuda-cu12`: CUDA 12.9, depends on `nvidia-cufft-cu12>=11,<12` (cuFFT 11,
  `libcufft.so.11`), needs a driver for CUDA 12.3 or newer.

```sh
tools/build_wheels.sh [OUT]   # default OUT: dist/release
uv run --no-project --with twine twine check OUT/sdist/*.tar.gz OUT/wheelhouse/*.whl
git tag -a vX.Y.Z -m "eda-cuda X.Y.Z" && git push origin main vX.Y.Z
gh release create vX.Y.Z OUT/wheelhouse/*.whl OUT/sdist/*.tar.gz --title "eda-cuda X.Y.Z"
```

`twine check` checks that the metadata and the long description of every file render. The
README's install example links to the wheel of a given release, so update its version when you
make a new one.

The script needs Docker. It builds the images `eda-build:cu13` and `eda-build:cu12` from
`tools/wheel.Dockerfile`: manylinux_2_28 (glibc 2.28, GCC 14) with the CUDA compiler, runtime,
cuFFT and CCCL from NVIDIA's RHEL 8 repository, at the versions the script names. In them it makes
the `eda-cuda` sdist from the tree and the `eda-cuda-cu12` sdist from a copy of it whose
`pyproject.toml` names the CUDA 12 package, its cuFFT dependency and `EDA_CUDA_MAJOR=12`, then each
wheel from its sdist. CMake refuses a package whose `EDA_CUDA_MAJOR` is not the compiler's, and
takes from the toolkit the cuFFT soname that `eda._native` preloads and the driver floor of the
startup check. Each wheel has SASS for compute capability 8.0, 8.6, 8.9, 9.0, 10.0 and 12.0, PTX
for 12.0, libstdc++, libgcc and cudart linked in, and cuFFT as a shared library; `auditwheel
repair` tags it `manylinux_2_28_x86_64` and leaves out libcuda and libcufft. The sdists are in
`OUT/sdist`, the wheels in `OUT/wheelhouse`. With the images built, each package took 31 to 45 s
here (8 CPUs per container). `uv build` on a host makes an `eda-cuda` wheel for that host's glibc only.

## Pull requests

CI runs only the CPU tests; run the GPU tests locally. A pull request that changes numerical
output says so.

## License

Contributions are licensed under Apache-2.0, the license of the project. No DCO sign-off is
required.
