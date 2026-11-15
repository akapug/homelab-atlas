#!/usr/bin/env bash
# Build tail-dump (CPU) against a llama.cpp checkout:  LLAMA_CPP=/path/to/llama.cpp bash build.sh  -> ./bin/tail-dump
# Extra cmake flags go in CMAKE_ARGS (the CPU build is native by default: -DGGML_NATIVE=ON).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LLAMA_CPP=${LLAMA_CPP:?set LLAMA_CPP to a llama.cpp checkout}
B=${BUILD_DIR:-$HERE/build}
cmake -S "$HERE" -B "$B" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CPP="$(cd "$LLAMA_CPP" && pwd)" -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_METAL=OFF ${CMAKE_ARGS:-} > "$B.cfg.log" 2>&1 \
    || { tail -30 "$B.cfg.log"; exit 3; }
cmake --build "$B" --target tail-dump -j "${JOBS:-$(nproc)}" > "$B.build.log" 2>&1 || { grep -E -A3 "error" "$B.build.log" | head -40; exit 4; }
mkdir -p "$HERE/bin"
cp "$B/tail-dump" "$HERE/bin/tail-dump"
echo "built $HERE/bin/tail-dump against llama.cpp $(git -C "$LLAMA_CPP" rev-parse --short HEAD 2>/dev/null || echo '(not a git checkout)')"
