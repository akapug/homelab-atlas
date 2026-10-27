#!/bin/bash
# Build PrismML's llama.cpp fork (PrismML-Eng/llama.cpp) for Ternary Bonsai 2 27B on an Intel GPU, with our SYCL
# patches, byte-for-byte the source we measured. Two builds:
#   pr235   PR #235's head (the first SYCL support for PTQ1_0) + patches/0001-0019. The long-context numbers.
#   hybrid  prism at the merge of PR #294 (its XMX path) + patches-hybrid/0001-0022: the same 19 patches rebased,
#           plus three that keep our decode kernel for 1-8 tokens and give PR #294's XMX path the prompt chunks.
# The patched source TREE must hash to the expected value (the tree hash covers file content only, not who
# committed it or when).
# Usage: build.sh <dest-dir> [pr235|hybrid] [--apply-only]      (default: hybrid)
# Needs git and Intel's oneAPI Base Toolkit in /opt/intel/oneapi. BONSAI_REPO overrides the clone source (a local
# mirror works). JOBS sets the build's parallel jobs (default: all cores).
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true  # before set -u: setvars trips it
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
DEST=${1:?usage: build.sh <dest-dir> [pr235|hybrid] [--apply-only]}
WHICH=${2:-hybrid}
case $WHICH in
  pr235)
    BASE=42ce76d0196baf2e2a995123dd303e3727b6634a   # PR #235 head (2026-09-24)
    FETCH=pull/235/head
    TREE=8964a8c0f23d5360e7f63a16ff4b2bf3e7ed3a8b   # the patched tree we measured
    DIR=patches ;;
  hybrid)
    BASE=a14c7de990e1b70fa4ebd5472a1cc24bc2388400   # prism: "sycl: XMX path for PQ2_0 and PTQ1_0 on 16-wide DPAS devices (#294)", 2026-10-02
    FETCH=$BASE
    TREE=247ee38674949837b93d3a4dc6635adebaec182f   # the patched tree we measured
    DIR=patches-hybrid ;;
  *) echo "usage: build.sh <dest-dir> [pr235|hybrid] [--apply-only]"; exit 2 ;;
esac
[ -d "$DEST/.git" ] || git clone -q "${BONSAI_REPO:-https://github.com/PrismML-Eng/llama.cpp}" "$DEST"
cd "$DEST"
git cat-file -e "$BASE^{commit}" 2>/dev/null || git fetch -q origin "$FETCH"
git checkout -q --detach "$BASE"
git -c user.name=build -c user.email=build@localhost am -q "$HERE/$DIR"/0*.patch
[ "$(git rev-parse 'HEAD^{tree}')" = "$TREE" ] || { echo "patched tree is $(git rev-parse 'HEAD^{tree}'), expected $TREE"; exit 1; }
echo "patched ($WHICH): $(git log --oneline -1) (tree verified)"
[ "${3:-}" = "--apply-only" ] && exit 0
command -v icpx >/dev/null || { echo "icpx not found: install the oneAPI Base Toolkit"; exit 1; }
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_F16=ON -DGGML_SYCL_DNN=ON \
  -DGGML_SYCL_GRAPH=ON -DGGML_SYCL_HOST_MEM_FALLBACK=ON -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON \
  -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON >build-config.log 2>&1 || { tail -25 build-config.log; exit 1; }
cmake --build build -j "${JOBS:-$(nproc)}" \
  --target llama-server llama-bench llama-batched-bench llama-perplexity test-backend-ops >build.log 2>&1 || { tail -30 build.log; exit 1; }
echo "built: $DEST/build/bin/{llama-server,llama-bench,llama-batched-bench,llama-perplexity,test-backend-ops}"
