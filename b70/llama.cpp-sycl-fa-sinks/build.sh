#!/bin/bash
# build.sh SRC: build ggml-sycl, ggml-cpu and test-backend-ops of the llama.cpp tree SRC (into SRC/build-pr) inside
# llama.cpp's own full-intel image (IntelLLVM 2026.1), pinned by digest. We built three trees this way: master
# a34de8618b25 with this branch's tests/test-backend-ops.cpp only, commit 1 only, and both commits.
# Prints the sha256 of the SYCL library and of the test binary. About 3 min on 24 cores.
set -eu
SRC=$(cd "${1:?usage: build.sh SRC}" && pwd)
IMAGE=${IMAGE:-ghcr.io/ggml-org/llama.cpp@sha256:77faff7d38b78820e5b7718f43a4284e27e58f83a643e4cb4bbe7653967f4691}
exec docker run --rm --init -v "$SRC:/src" -w /src --entrypoint bash "$IMAGE" -c "
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
cmake -B build-pr -DGGML_NATIVE=OFF -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_BACKEND_DL=ON \
  -DGGML_CPU_ALL_VARIANTS=OFF -DLLAMA_BUILD_TESTS=ON -DGGML_SYCL_F16=ON -DCMAKE_BUILD_TYPE=Release >build-pr.config.log 2>&1 \
  || { tail -30 build-pr.config.log; exit 1; }
cmake --build build-pr --config Release -j \$(nproc) --target ggml-sycl ggml-cpu test-backend-ops >build-pr.log 2>&1 \
  || { tail -30 build-pr.log; exit 1; }
sha256sum build-pr/bin/libggml-sycl.so build-pr/bin/test-backend-ops"
