#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
#
# Builds the two release packages in manylinux_2_28 containers, each as an sdist and a wheel:
#   eda-cuda       CUDA 13.4 toolkit, cuFFT 12 (nvidia-cufft), driver for CUDA 13.0 or newer
#   eda-cuda-cu12  CUDA 12.9 toolkit, cuFFT 11 (nvidia-cufft-cu12), driver for CUDA 12.3 or newer
# Both hold the Python package eda from the same tree; eda-cuda-cu12 is built from a copy of the
# eda-cuda sdist whose pyproject.toml names the CUDA 12 package.
#
# Usage: tools/build_wheels.sh [OUT]   (default OUT: dist/release)
# OUT/sdist holds the sdists, OUT/wheelhouse the wheels after `auditwheel repair`.
# Environment: DOCKER_RUN_ARGS, extra `docker run` arguments (default "--cpus 8 --memory 24g");
# CMAKE_BUILD_PARALLEL_LEVEL, build jobs in the container (default 8).
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out=$(mkdir -p "${1:-$root/dist/release}" && cd "${1:-$root/dist/release}" && pwd)
read -r -a run_args <<< "${DOCKER_RUN_ARGS:---cpus 8 --memory 24g}"
jobs=${CMAKE_BUILD_PARALLEL_LEVEL:-8}

# The toolkit packages of each image, at fixed versions.
cu13_packages="cuda-nvcc-13-4-13.4.59 cuda-crt-13-4-13.4.59 libnvvm-13-4-13.4.59
  libnvptxcompiler-13-4-13.4.59 cuda-cudart-13-4-13.4.49 cuda-cudart-devel-13-4-13.4.49
  cuda-culibos-devel-13-4-13.4.49 cccl-13-4-13.3.4.2.1 libcufft-13-4-12.4.0.34
  libcufft-devel-13-4-12.4.0.34"
cu12_packages="cuda-nvcc-12-9-12.9.86 cuda-cudart-devel-12-9-12.9.79 cuda-cccl-12-9-12.9.27
  libcufft-devel-12-9-11.4.1.4"

docker build -t eda-build:cu13 --build-arg CUDA_VERSION=13.4 \
  --build-arg CUDA_PACKAGES="$cu13_packages" -f "$root/tools/wheel.Dockerfile" "$root/tools"
docker build -t eda-build:cu12 --build-arg CUDA_VERSION=12.9 \
  --build-arg CUDA_PACKAGES="$cu12_packages" -f "$root/tools/wheel.Dockerfile" "$root/tools"

# build IMAGE SCRIPT: runs SCRIPT in IMAGE as the calling user, the tree read-only at /src and OUT
# at /out.
build() {
  docker run --rm "${run_args[@]}" --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -e CMAKE_BUILD_PARALLEL_LEVEL="$jobs" -e UV_CACHE_DIR=/tmp/uv-cache \
    -v "$root:/src:ro" -v "$out:/out" "$1" bash -euo pipefail -c "$2"
}

rm -rf "$out/sdist" "$out/wheelhouse" "$out/raw"
mkdir -p "$out/sdist" "$out/wheelhouse" "$out/raw"
py=/opt/python/cp312-cp312/bin/python

# repair CUFFT_SONAME: the command that tags the wheel in /out/raw as manylinux_2_28 and moves it
# to /out/wheelhouse; libcuda and cuFFT stay outside the wheel.
repair() {
  echo "auditwheel repair --plat manylinux_2_28_x86_64 --only-plat --exclude libcuda.so.1 \
    --exclude $1 -w /out/wheelhouse /out/raw/*.whl && rm /out/raw/*.whl"
}

# eda-cuda: the sdist of the tree, then the wheel from that sdist.
time build eda-build:cu13 "
  uv build --python $py --sdist --out-dir /out/sdist /src
  uv build --python $py --wheel --out-dir /out/raw /out/sdist/eda_cuda-*.tar.gz
  $(repair libcufft.so.12)"

# eda-cuda-cu12: the eda-cuda sdist with the CUDA 12 pyproject.toml.
time build eda-build:cu12 "
  mkdir /tmp/cu12 && tar -xzf /out/sdist/eda_cuda-*.tar.gz -C /tmp/cu12 --strip-components 1
  $py - /tmp/cu12/pyproject.toml <<'EOF'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text()
for old, new in [
    ('name = \"eda-cuda\"', 'name = \"eda-cuda-cu12\"'),
    ('\"nvidia-cufft>=12,<13\"', '\"nvidia-cufft-cu12>=11,<12\"'),
    ('NVIDIA CUDA :: 13\"', 'NVIDIA CUDA :: 12\"'),
    ('EDA_CUDA_MAJOR = \"13\"', 'EDA_CUDA_MAJOR = \"12\"'),
]:
    assert text.count(old) == 1, old
    text = text.replace(old, new)
path.write_text(text)
EOF
  uv build --python $py --sdist --out-dir /out/sdist /tmp/cu12
  uv build --python $py --wheel --out-dir /out/raw /out/sdist/eda_cuda_cu12-*.tar.gz
  $(repair libcufft.so.11)"

rm -r "$out/raw"
ls -l "$out/sdist" "$out/wheelhouse"
