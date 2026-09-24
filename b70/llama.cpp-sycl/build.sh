#!/bin/bash
# Build llama.cpp's SYCL backend with the decode patches in ./patches, byte-for-byte the source we
# measured. The patches apply onto a pinned upstream commit, and the patched source TREE must hash
# to the expected value (the tree hash covers file content only, not who committed it or when).
# Usage: build.sh <dest-dir> [--apply-only]
# Needs git and the oneAPI Base Toolkit in /opt/intel/oneapi. LLAMA_REPO overrides the
# clone source (a local mirror works).
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true  # before set -u: setvars trips it
set -eu
BASE=94256114c229674ef96e76eb2dea596e65b43818   # ggml-org/llama.cpp master, 2026-09-23
TREE=35581c4e65af3d6d1a00a0dfcfd42471daf29926   # the patched tree we measured
HERE=$(cd "$(dirname "$0")" && pwd)
DEST=${1:?usage: build.sh <dest-dir> [--apply-only]}
[ -d "$DEST/.git" ] || git clone -q "${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp}" "$DEST"
cd "$DEST"
git cat-file -e "$BASE^{commit}" 2>/dev/null || git fetch -q origin "$BASE"
git checkout -q --detach "$BASE"
git -c user.name=build -c user.email=build@localhost am -q "$HERE"/patches/0*.patch
[ "$(git rev-parse HEAD^{tree})" = "$TREE" ] || { echo "patched tree is $(git rev-parse HEAD^{tree}), expected $TREE"; exit 1; }
echo "patched: $(git log --oneline -1) (tree verified)"
[ "${2:-}" = "--apply-only" ] && exit 0
command -v icpx >/dev/null || { echo "icpx not found: install the oneAPI Base Toolkit"; exit 1; }
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_F16=ON -DGGML_SYCL_DNN=ON \
  -DGGML_SYCL_GRAPH=ON -DGGML_SYCL_HOST_MEM_FALLBACK=ON -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON \
  -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON >build-config.log 2>&1 || { tail -25 build-config.log; exit 1; }
cmake --build build -j "$(nproc)" --target llama-server test-backend-ops >build.log 2>&1 || { tail -30 build.log; exit 1; }
echo "built: $DEST/build/bin/llama-server"
