#!/usr/bin/env bash
# Tests for bin/archive-logs. Run from anywhere: tests/run_tests.sh
cd "$(dirname "$0")/.." || exit 1

failures=0
ok()   { echo "ok   - $1"; }
fail() { echo "FAIL - $1"; failures=$((failures + 1)); }

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# --- archives every .log file and nothing else ---
mkdir -p "$work/t1/src"
printf 'hello\n' > "$work/t1/src/app.log"
printf 'db line\n' > "$work/t1/src/db.log"
printf 'not a log\n' > "$work/t1/src/notes.txt"
out=$(bash bin/archive-logs "$work/t1/src" "$work/t1/dest")
status=$?
[ "$status" -eq 0 ] && ok "exit 0 on success" || fail "exit 0 on success (got $status)"
[ "$(gzip -dc "$work/t1/dest/app.log.gz")" = "hello" ] && ok "app.log archived" || fail "app.log archived"
[ "$(gzip -dc "$work/t1/dest/db.log.gz")" = "db line" ] && ok "db.log archived" || fail "db.log archived"
[ ! -e "$work/t1/dest/notes.txt.gz" ] && ok "non-log files skipped" || fail "non-log files skipped"
case "$out" in
    *"archived 2 files"*) ok "summary line" ;;
    *) fail "summary line (got: $out)" ;;
esac
[ -f "$work/t1/src/app.log" ] && ok "originals kept" || fail "originals kept"

# --- usage error ---
bash bin/archive-logs "$work/t1/src" 2>/dev/null
status=$?
[ "$status" -eq 2 ] && ok "usage error exits 2" || fail "usage error exits 2 (got $status)"

echo
if [ "$failures" -eq 0 ]; then
    echo "all tests passed"
else
    echo "$failures test(s) failed"
    exit 1
fi
