#!/usr/bin/env bash
# Launch DeepSeek V4.1 Flash 1017 the way our fastest run on the Quadro RTX 6000 + Optane box did (12.69-12.71 tok/s n4).
# Box: 2 sockets of 10 cores, 20 threads each (NUMA nodes 0 and 1) plus Optane PMem as CPU-less node 2. The 39 compute
# threads leave CPU 39 free: it is the tier split's prefetch CPU, idle here because this run had prefetch off.
# Change CPUS/T and the node numbers in placement.tsv for another box.
#   ENGINE=<dir with bin/llama-server from the tier-split branch>  STORE=<split store dir>  DRAFT=<draft gguf>  ./launch.sh
# Needs root for the three system settings. Restore them afterwards (numa_balancing, demotion, swap).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ENGINE=${ENGINE:?ENGINE=<build dir>}; STORE=${STORE:?STORE=<split store dir>}; DRAFT=${DRAFT:?DRAFT=<DSpark q2 draft gguf>}
CPUS=${CPUS:-0-9,20-29,10-19,30-38}; T=${T:-39}; PORT=${PORT:-8790}; SPLIT=${SPLIT:-8}

echo 0 | sudo tee /proc/sys/kernel/numa_balancing >/dev/null       # no automatic page migration
echo false | sudo tee /sys/kernel/mm/numa/demotion_enabled >/dev/null
sudo swapoff -a
wi=/sys/kernel/mm/mempolicy/weighted_interleave                  # equal weights, where the kernel has them
[ ! -d $wi ] || for n in 0 1 2; do echo 1 | sudo tee $wi/node$n >/dev/null; done

ctl=$(mktemp)                                                       # the tier split, as the measured run set it
echo "split=$SPLIT prefetch=0" > "$ctl"

exec env GGML_CUDA_GRAPHS_PRE_AMPERE=1 GGML_AFFINITY_ONCE=1 LLAMA_MOE_OVERLAP=1 GGML_SCHED_OVERLAP=1 \
  GGML_SCHED_ASYNC_COPY=1 GGML_OMP_FAST_ENTRY=1 GGML_BARRIER_SPIN=1 GGML_MOE_GROUPS=2 GGML_MAIN_NO_MOE=1 \
  GGML_CUDA_PIN_STAGING=1 GGML_CUDA_PIN_MAX_MIB=4096 GGML_SCHED_INPLACE_FOLLOW=1 LLAMA_NO_HOST_WEIGHTS=1 \
  GGML_TIER_CTL="$ctl" GGML_TIER_RING_MIB=1024 GGML_TIER_PREFETCH_CPUS=39 \
  LLAMA_EXPERT_PLACEMENT="$here/placement.tsv" LLAMA_HOT_EXPERTS="$here/hot-experts.tsv" \
  LD_LIBRARY_PATH="$ENGINE/lib:${LD_LIBRARY_PATH:-}" \
  numactl --interleave=0,1 --physcpubind="$CPUS" \
  "$ENGINE/bin/llama-server" -m "$STORE/DeepSeek-V4.1-Flash-1017-hotmxfp4-coldq2k-00001-of-00010.gguf" \
    -ngl 99 -fa 1 --no-mmap -t "$T" -c 4096 -cr "$CPUS" --numa numactl -dple --cpu-moe \
    -md "$DRAFT" -ngld 99 -draft "-ncmoe 3 --no-mmap" -cd 4096 \
    --spec-type dspark:n_max=4,conf_cum_min=0.5 -ub 4096 -b 4096 -cuda offload-batch-size=100000 \
    --port "$PORT"
