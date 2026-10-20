#!/usr/bin/env bash
# After a run kit finishes: append ONE ledger row (runs/runs.jsonl) with the recipe the pre-launch gate consumed, the gate's
# launch-time `report.js --check --json` output (its prediction is the launch prediction), and what the logs measured.
#   tools/kit_record.sh RECIPE.json LOG [LOG...] [-- record_run.py options: --arm NAME --tag T --source S --id ID --dry-run]
# LOG = KIND:PATH, KIND one of kit | bench-log | bench-json | kl; PATH is local or host:path (read with ssh cat, read-only),
# PATH#L10-L20 keeps those lines. The gate output is looked up beside the recipe: RECIPE.json.check.json, then
# RECIPE.check.json (FITVIZ_CHECK names it explicitly). Without one the row is recorded post hoc (--no-check) and says so.
# Settings: FITVIZ_RUNS = ledger file (default runs/runs.jsonl of this checkout). The row id defaults to <box>-<model>-<recipe id>.
# Exit: 0 appended (or --dry-run valid), 1 invalid row, 2 usage.
set -eu -o pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
[ $# -ge 2 ] || { sed -n 2,10p "$0"; exit 2; }
recipe=$1; shift
[ -s "$recipe" ] || { echo "kit_record: no recipe $recipe" >&2; exit 2; }
logs=()
while [ $# -gt 0 ] && [ "$1" != -- ]; do logs+=(--from-log "$1"); shift; done
[ "${1:-}" = -- ] && shift
[ ${#logs[@]} -gt 0 ] || { echo "kit_record: no LOG given" >&2; exit 2; }
check=${FITVIZ_CHECK:-}
if [ -z "$check" ]; then
  for c in "$recipe.check.json" "${recipe%.json}.check.json"; do [ -s "$c" ] && { check=$c; break; }; done
fi
if [ -n "$check" ]; then ck=(--check "$check"); echo "kit_record: launch-time gate output $check" >&2
else ck=(--no-check); echo "kit_record: no gate output beside $recipe: the prediction is computed now and marked post-hoc" >&2; fi
id=()
case " $* " in *" --id "*) ;; *) id=(--id "$(python3 -I -c 'import json, sys
r = json.load(open(sys.argv[1])); p = "%s-%s-" % (r["box"], r["model"])
print(r["id"] if r["id"].startswith(p) else p + r["id"])' "$recipe")") ;; esac
exec python3 "$root/tools/record_run.py" --recipe "$recipe" "${ck[@]}" "${logs[@]}" --runs "${FITVIZ_RUNS:-$root/runs/runs.jsonl}" "${id[@]}" "$@"
