#!/usr/bin/env bash
# Build the engine of the 12.69 run: akapug/ik_llama.cpp branch tier-split, CUDA for the Quadro RTX 6000 (sm_75), CPU
# code for Cascade Lake (AVX-512 + VNNI), in the CUDA 12.6.3 container. Our measured package used these flags.
#   OUT=$HOME/build/ik-tier-split ./build-engine.sh        -> $OUT/pkg/bin (llama-server + tests) and $OUT/pkg/lib
# GGML_EXPERT_CHUNKING=OFF matters: ON cost this box about 20-24 % of GLM decode. Change -march and
# CMAKE_CUDA_ARCHITECTURES for another box.
set -euo pipefail
# REF pins the branch tip we published (its code is cde5fc20, the measured build; later commits change only docs).
PIN=6d9992e2b7fc888882d271b3d297ebc382c4e54c
OUT=${OUT:-$HOME/build/ik-tier-split}; REF=${REF:-$PIN}; J=${J:-16}
IMG=${IMG:-nvidia/cuda:12.6.3-devel-ubuntu24.04}
mkdir -p "$OUT"
[ -d "$OUT/src/.git" ] || git clone https://github.com/akapug/ik_llama.cpp "$OUT/src"
git -C "$OUT/src" fetch -q origin tier-split && git -C "$OUT/src" checkout -q "$REF"
git -C "$OUT/src" rev-parse HEAD > "$OUT/SOURCE_SHA"
[ "$REF" != "$PIN" ] || [ "$(cat "$OUT/SOURCE_SHA")" = "$PIN" ] || { echo "checkout is not $PIN"; exit 1; }
docker run --rm -v "$OUT":/o -e J="$J" -e U="$(id -u):$(id -g)" "$IMG" bash -euc '
  trap "chown -R $U /o" EXIT
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends cmake python3 >/dev/null
  cf="-march=cascadelake -mtune=cascadelake"
  cmake -S /o/src -B /o/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX512=ON \
    -DGGML_AVX512_VNNI=ON -DCMAKE_C_FLAGS="$cf" -DCMAKE_CXX_FLAGS="$cf" -DGGML_SCHED_MAX_COPIES=1 \
    -DGGML_IQK_FA_ALL_QUANTS=ON -DLLAMA_BUILD_TESTS=ON -DGGML_CCACHE=OFF -DGGML_EXPERT_CHUNKING=OFF -DGGML_NCCL=OFF \
    -DLLAMA_CURL=OFF -DGGML_OPENMP=ON
  cmake --build /o/build --target llama-server llama-cli llama-perplexity llama-quantize llama-imatrix test-backend-ops -j "$J"
  # the expert-placement and tier-split tests are not CMake targets: link them against the built libraries
  for t in test-expert-place test-tier-split; do
    g++ -std=c++17 $cf -O2 -I/o/src/include -I/o/src/src -I/o/src/ggml/include -I/o/src/ggml/src /o/src/tests/$t.cpp \
      -o /o/build/bin/$t -L/o/build/src -L/o/build/ggml/src -lllama -lggml -L/usr/local/cuda/lib64 \
      -L/usr/local/cuda/lib64/stubs -lcudart -lcublas -lcublasLt -lcuda -fopenmp -lpthread -ldl -lrt
  done
  mkdir -p /o/pkg/bin /o/pkg/lib
  cp /o/build/bin/llama-server /o/build/bin/llama-cli /o/build/bin/llama-perplexity /o/build/bin/llama-quantize /o/build/bin/llama-imatrix /o/build/bin/test-backend-ops \
     /o/build/bin/test-expert-place /o/build/bin/test-tier-split /o/pkg/bin/
  for l in libcudart.so.12 libcublas.so.12 libcublasLt.so.12; do cp -L /usr/local/cuda/lib64/$l /o/pkg/lib/; done'
(cd "$OUT/pkg" && sha256sum bin/* lib/* > SHA256SUMS)
echo "built $(cat "$OUT/SOURCE_SHA") into $OUT/pkg; check: $OUT/pkg/bin/test-tier-split 8 (every arm must print OK)"
