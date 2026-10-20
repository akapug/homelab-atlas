#!/usr/bin/env bash
# All-GPU decode sweep for the GPU host and per-token MoE terms (build.py gpu_bench). One llama-bench decode per arm,
# every layer on the GPU, no prompt, GEN tokens x REPS repetitions after the bench's own warm-up. Dense models of several
# sizes give the per-engine line (time per token = bytes / rate + host); MoE models read against that line give the
# per-token MoE overhead. Prints one JSON line per arm: the arm, its kind, the model path and the llama-bench jsonl row.
# usage: tools/bench_gpu.sh ARMS_FILE BENCH_CMD...   (BENCH_CMD = a llama-bench binary, or a docker run prefix ending in it)
# ARMS_FILE: label <tab> kind (dense|moe) <tab> model path as BENCH_CMD sees it
set -u
GEN=${GEN:-128}   # decode tokens per repetition
REPS=${REPS:-3}   # llama-bench repetitions
ARMS=$1; shift
rc=0
while IFS=$'\t' read -r arm kind model; do
  [[ -z "$arm" || "$arm" == \#* ]] && continue
  out=$("$@" -m "$model" -ngl 99 -p 0 -n "$GEN" -r "$REPS" -o jsonl 2>/dev/null | tail -n 1); r=$?
  [[ -z "$out" ]] && { r=1; out=null; }
  [[ $r -ne 0 ]] && rc=$r
  printf '{"arm":"%s","kind":"%s","model":"%s","rc":%d,"bench":%s}\n' "$arm" "$kind" "$model" "$r" "$out"
done <"$ARMS"
exit $rc
