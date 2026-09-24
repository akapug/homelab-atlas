#!/usr/bin/env bash
# build.sh - turn configs/*.conf into the release bundle dist/bundle.tar.gz.
# Every step's output also goes to dist/build.log, which CI keeps as an artifact.
set -e
cd "$(dirname "$0")"

out=dist
stage=$out/stage
log=$out/build.log
rm -rf "$out"
mkdir -p "$stage"

echo "== generate" | tee "$log"
python3 tools/gen.py configs "$stage" 2>&1 | tee -a "$log"

echo "== validate" | tee -a "$log"
for f in "$stage"/*.json; do
    python3 tools/validate.py "$f" >>"$log" 2>&1 && echo "  ok $(basename "$f")" | tee -a "$log"
done

echo "== package" | tee -a "$log"
tar -czf "$out/bundle.tar.gz" -C "$stage" . 2>&1 | tee -a "$log"
(cd "$out" && sha256sum bundle.tar.gz > bundle.sha256)

echo "BUILD OK" | tee -a "$log"
