#!/bin/bash
# How long one speculative verify step takes at the server's current draft count. One stream decodes
# TOKENS tokens of a log summary (decode-vllm.py); vLLM's spec-decode counters give the tokens each
# step produced (accepted drafts + 1), and step ms = 1000 x tokens per step / decode tok/s. Run it at
# two draft counts (restart the server with each) for the cost of the extra drafts: the gain from a
# higher acceptance elsewhere (agent traffic) is its tokens-per-step ratio times the inverse of this
# step-time ratio.
# The counters are the server's, so any other request in flight is counted too and slows the step:
# the script waits for an idle server and flags a run whose counters saw more tokens than it decoded.
# Hold the model's canary meanwhile (its probes are requests too).
#   spec-step-cost.sh <label> <corpus.jsonl> [port, default 8083]      Env: TOKENS (default 1024)
set -u
L=${1:?label}; C=${2:?corpus.jsonl}; P=${3:-8083}
metric(){ curl -s -m 5 localhost:$P/metrics | awk -v k="$1" '$1 ~ "^"k"[{]" {s += $NF} END {printf "%d", s}'; }
for i in $(seq 1 120); do [ "$(metric 'vllm:num_requests_(running|waiting)')" = 0 ] && break; sleep 5; done
d0=$(metric vllm:spec_decode_num_drafts_total); a0=$(metric vllm:spec_decode_num_accepted_tokens_total)
out=$(STREAMS=1 CHARS=6000 TOKENS=${TOKENS:-1024} python3 "$(dirname "$0")/decode-vllm.py" "$L" "$C" "$P")
d1=$(metric vllm:spec_decode_num_drafts_total); a1=$(metric vllm:spec_decode_num_accepted_tokens_total)
echo "$out" | tail -1
rate=$(echo "$out" | tail -1 | sed -n 's/.*decode aggregate *\([0-9.]*\) tok\/s.*/\1/p')
awk -v l="$L" -v a=$((a1 - a0)) -v d=$((d1 - d0)) -v r="$rate" -v n="${TOKENS:-1024}" 'BEGIN {
  t = 1 + a / d; printf "%s: %d steps, %.2f tokens per step, %.1f tok/s, %.1f ms per step%s\n", l, d, t, r, 1000 * t / r,
    (d + a > 1.05 * (n + 9)) ? sprintf("  CONTAMINATED: the counters saw %d tokens for %d decoded", d + a, n + 9) : "" }'
