# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Giorgio Angelotti
#
# The build image of a release wheel: manylinux_2_28 (glibc 2.28, GCC 14) with the parts of one
# CUDA toolkit that the build needs, from NVIDIA's RHEL 8 repository. tools/build_wheels.sh builds
# it twice (eda-build:cu13, eda-build:cu12) and names the packages and their versions.
FROM quay.io/pypa/manylinux_2_28_x86_64:latest
ARG CUDA_VERSION
ARG CUDA_PACKAGES
RUN dnf config-manager --add-repo \
      https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
 && dnf install -y --setopt=install_weak_deps=False ${CUDA_PACKAGES} \
 && dnf clean all
ENV CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
ENV PATH=/usr/local/cuda-${CUDA_VERSION}/bin:${PATH}
