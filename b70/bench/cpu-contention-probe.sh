#!/bin/bash
# Does a busy host CPU slow a vLLM server's speculative step, and what gets it back? Each phase is
# timed by spec-step-cost.sh (one stream, on an idle server):
#   idle       nothing else running
#   busy       stress-ng on every CPU, niced 19 in a transient user scope (like a background job)
#   pinned     the container on PIN (docker update --cpuset-cpus, live, no restart), stress-ng on the rest
#   pin-idle   the container on PIN, no stress: does the pin itself cost anything
#   pin-busy   (not in the default set) the container on PIN, stress-ng on EVERY CPU
# The container's CPU set is restored on exit. Stop serving-cpu-guard first if it runs.
#   cpu-contention-probe.sh <container> <port> <PIN cpus, e.g. 8-15,24-31> [runs per phase, default 2]
# Env: CORPUS (prompts for decode-vllm.py, required), LOAD (stress-ng arguments, default
#      "--cpu 0 --cpu-method matrixprod"; "--stream 0" loads memory bandwidth),
#      PHASES (default "idle busy pinned pin-idle")
set -u
systemctl is-active --quiet serving-cpu-guard &&
  { echo "serving-cpu-guard is running and re-pins the containers: stop it first, start it after"; exit 1; }
M=${1:?container}; P=${2:?port}; PIN=${3:?cpus}; N=${4:-2}; C=${CORPUS:?corpus.jsonl for decode-vllm.py}
D=$(cd "$(dirname "$0")" && pwd)
ALL=0-$(($(nproc --all) - 1))
REST=$(python3 -c "
import sys
p = set()
for x in sys.argv[1].split(','):
    a, _, b = x.partition('-')
    p.update(range(int(a), int(b or a) + 1))
r = [str(c) for c in range(int(sys.argv[2])) if c not in p]
print(','.join(r) if r else sys.exit('PIN leaves no CPU for the load'))" "$PIN" "$(nproc --all)") || exit 1
STRESS=
stop(){ [ -n "$STRESS" ] && systemctl --user stop "$STRESS" 2>/dev/null; STRESS=; }
cleanup(){ stop; docker update --cpuset-cpus "$ALL" "$M" >/dev/null; echo "restored: $M on $(docker inspect -f '{{.HostConfig.CpusetCpus}}' "$M")"; }
trap cleanup EXIT
stress(){  # stress(cpus): stress-ng on those CPUs, niced 19, in a transient user scope
  STRESS=contention-probe-$$-$RANDOM
  systemd-run --user --quiet --unit="$STRESS" --nice=19 taskset -c "$1" stress-ng ${LOAD:---cpu 0 --cpu-method matrixprod} --timeout 30m >/dev/null
  sleep 5
}
phase(){ for i in $(seq 1 "$N"); do bash "$D/spec-step-cost.sh" "$M $1 #$i" "$C" "$P" | tail -1; done; }
for f in ${PHASES:-idle busy pinned pin-idle}; do
  case $f in
    idle) phase idle ;;
    busy) stress "$ALL"; phase busy; stop ;;
    pinned) docker update --cpuset-cpus "$PIN" "$M" >/dev/null; stress "$REST"; phase pinned; stop ;;
    pin-busy) docker update --cpuset-cpus "$PIN" "$M" >/dev/null; stress "$ALL"; phase pin-busy; stop ;;
    pin-idle) docker update --cpuset-cpus "$PIN" "$M" >/dev/null; phase pin-idle ;;
  esac
done
