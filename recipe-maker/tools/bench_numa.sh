#!/usr/bin/env bash
# NUMA placement sweep for CPU decode (the numa socket term in build.py). One llama-bench decode per arm; each arm binds its
# threads to a CPU list with numactl --physcpubind and places the weights with a memory policy (--membind / --interleave /
# first touch). Weights are loaded with --mmap 0, so the policy decides where they live. Threads = the arm's cores minus 1
# (store cpu-decode-leaves-one-cpu-for-the-system). Prints one JSON line per arm: the arm, its policy, threads, the
# load average before the arm, the llama-bench jsonl row and the per-node MiB of the bench process (numastat -p, sampled mid-run).
# usage: tools/bench_numa.sh [ARMS_FILE]   (run inside a fab job on the box; settings below are env-overridable)
set -u
BENCH=${BENCH:-~/bonsai/llama.cpp/build-cpu/bin/llama-bench}
MODEL=${MODEL:-~/hybridagent/gguf/ha8b-q8_0.gguf}
GEN=${GEN:-48}          # decode tokens per repetition
REPS=${REPS:-2}         # llama-bench repetitions (after its own warm-up)
SAMPLE_S=${SAMPLE_S:-12} # when to read numastat for the bench process
ARMS=${1:-$(dirname "$0")/bench_numa_lassie.tsv}
rc=0
while IFS=$'\t' read -r arm cpus policy note; do
  [[ -z "$arm" || "$arm" == \#* ]] && continue
  ncpu=$(numactl --physcpubind="$cpus" --show | awk '/^physcpubind/{print NF-1}')
  t=$((ncpu - 1))
  pol=()
  [[ "$policy" != first_touch ]] && pol=("--$policy")
  load=$(cut -d' ' -f1-3 /proc/loadavg)
  out=$(mktemp)
  numactl --physcpubind="$cpus" "${pol[@]}" "$BENCH" -m "$MODEL" -p 0 -n "$GEN" -r "$REPS" -t "$t" -mmp 0 -o jsonl >"$out" 2>"$out.err" &
  pid=$!
  sleep "$SAMPLE_S"
  nodes=$(numastat -p "$pid" 2>/dev/null | awk '/^Total/{$1=""; print}' | sed 's/^ *//')
  wait "$pid"; r=$?
  [[ $r -ne 0 ]] && rc=$r
  row=$(tail -n 1 "$out")
  printf '{"arm":"%s","cpus":"%s","policy":"%s","note":"%s","threads":%d,"rc":%d,"load_before":"%s","node_mib":"%s","bench":%s}\n' \
    "$arm" "$cpus" "$policy" "$note" "$t" "$r" "$load" "$nodes" "${row:-null}"
  [[ $r -ne 0 ]] && tail -n 5 "$out.err" >&2
  rm -f "$out" "$out.err"
done <"$ARMS"
exit $rc
