#!/bin/bash
# serving-cpu-guard: keep other work off the inference server's CCD while it is serving, and give
# the CPUs back when it is idle. Runs as root (serving-cpu-guard.service).
#
# For a host that both serves a model (vLLM in docker) and runs other heavy work (builds, tests).
# On a multi-CCD Ryzen each CCD has its own L3. vLLM on XPU in eager mode launches every kernel
# from Python, so a short speculative step (Qwen3.6-35B-A3B: ~15 ms) slows badly when other work
# shares its CCD, and pinning the server alone does not help (README, "The host's CPU"). This pins
# the server containers to SERVE_CPUS and, while any server has a request in flight (plus IDLE_S
# seconds), limits SLICE (by default user.slice: every login session and user service) to
# USER_CPUS. When the servers are idle, the slice gets every CPU again.
#
# Stopping the service restores the slice and the containers: the trap here, and ExecStopPost in
# the unit when the process is killed. Stop it before measuring contention yourself, since it
# re-pins the containers every PIN_EVERY polls.
#
# Required env: CONTAINERS (docker names), PORTS (their vLLM ports), SERVE_CPUS and USER_CPUS
# (one CCD each, as cpuset lists; `lscpu -e` and /sys/devices/system/cpu/cpu*/cache/index3/
# shared_cpu_list show which CPUs share an L3). On a 16-core Ryzen 9 9950X3D: SERVE_CPUS=8-15,24-31
# USER_CPUS=0-7,16-23. Optional: SLICE (user.slice), IDLE_S (20), POLL (1), PIN_EVERY (30).
set -u
: "${CONTAINERS:?}" "${PORTS:?}" "${SERVE_CPUS:?}" "${USER_CPUS:?}"
SLICE=${SLICE:-user.slice}; IDLE_S=${IDLE_S:-20}; POLL=${POLL:-1}; PIN_EVERY=${PIN_EVERY:-30}
ALL=0-$(($(nproc --all) - 1))
log(){ echo "serving-cpu-guard: $*"; }
busy(){ for p in $PORTS; do curl -s -m 1 "localhost:$p/metrics"; done |
  awk '$1 ~ /^vllm:num_requests_(running|waiting)[{]/ {s += $NF} END {exit !(s > 0)}'; }
allow(){ systemctl set-property --runtime "$SLICE" AllowedCPUs="$1"; }
pin(){  # pin(cpus): every named container that exists, if not already there
  for c in $CONTAINERS; do
    cur=$(docker inspect -f '{{.HostConfig.CpusetCpus}}' "$c" 2>/dev/null) || continue
    [ "$cur" = "$1" ] || { docker update --cpuset-cpus "$1" "$c" >/dev/null && log "$c on CPUs $1 (was ${cur:-all})"; }
  done; }
restore(){ allow ""; pin "$ALL"; log "stopped: $SLICE and containers back on every CPU"; exit 0; }
trap restore TERM INT
split=0 last=0 n=0 since=0
allow ""; log "watching ports $PORTS; containers on $SERVE_CPUS; $SLICE on $USER_CPUS while serving, +${IDLE_S}s"
while :; do
  [ $((n++ % PIN_EVERY)) = 0 ] && pin "$SERVE_CPUS"
  now=$(date +%s)
  if busy; then
    last=$now
    [ $split = 1 ] || { allow "$USER_CPUS"; split=1 since=$now; log "split: a model is serving; $SLICE on $USER_CPUS"; }
  elif [ $split = 1 ] && [ $((now - last)) -ge "$IDLE_S" ]; then
    allow ""; split=0; log "open: idle ${IDLE_S}s; $SLICE on every CPU (split lasted $((now - since))s)"
  fi
  sleep "$POLL"
done
