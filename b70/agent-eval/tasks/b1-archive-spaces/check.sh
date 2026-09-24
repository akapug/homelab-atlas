#!/usr/bin/env bash
# Hidden check for b1-archive-spaces. Run from the root of the agent's
# working copy. Exit 0 only if the required behavior holds.
set -u
root=$(pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
failures=0
ok()   { echo "ok   - $1"; }
fail() { echo "FAIL - $1"; failures=$((failures + 1)); }

# 1. The repo's own tests still pass.
if bash tests/run_tests.sh >"$work/own.txt" 2>&1; then
    ok "repo tests pass"
else
    fail "repo tests pass"; cat "$work/own.txt"
fi

# 2. File names (and directories) with spaces are archived intact.
src="$work/my logs/src"
dest="$work/archive out"
mkdir -p "$src"
printf 'plain\n'  > "$src/app.log"
printf 'one\n'    > "$src/with space.log"
printf 'two\n'    > "$src/two  spaces.log"
printf 'skip\n'   > "$src/read me.txt"
out=$(bash "$root/bin/archive-logs" "$src" "$dest" 2>&1)
status=$?
[ "$status" -eq 0 ] && ok "spaces: exit 0" || fail "spaces: exit 0 (got $status): $out"
for pair in "app.log:plain" "with space.log:one" "two  spaces.log:two"; do
    name=${pair%%:*}; want=${pair#*:}
    got=$(gzip -dc "$dest/$name.gz" 2>/dev/null)
    [ "$got" = "$want" ] && ok "spaces: '$name' archived" || fail "spaces: '$name' archived"
done
[ ! -e "$dest/read me.txt.gz" ] && ok "spaces: non-log skipped" || fail "spaces: non-log skipped"
case "$out" in
    *"archived 3 files"*) ok "spaces: summary counts 3" ;;
    *) fail "spaces: summary counts 3 (got: $out)" ;;
esac
[ -f "$src/with space.log" ] && ok "spaces: originals kept" || fail "spaces: originals kept"

# 3. An empty source directory is not an error.
mkdir -p "$work/empty"
out=$(bash "$root/bin/archive-logs" "$work/empty" "$work/empty-out" 2>&1)
status=$?
[ "$status" -eq 0 ] && ok "empty: exit 0" || fail "empty: exit 0 (got $status)"
case "$out" in
    *"archived 0 files"*) ok "empty: summary counts 0" ;;
    *) fail "empty: summary counts 0 (got: $out)" ;;
esac

# 4. A file that cannot be written makes the run fail. A directory sits
#    where b.log.gz should go, which fails even for root.
mkdir -p "$work/f/src" "$work/f/dest/b.log.gz"
printf 'a\n' > "$work/f/src/a.log"
printf 'b\n' > "$work/f/src/b.log"
printf 'c\n' > "$work/f/src/c.log"
bash "$root/bin/archive-logs" "$work/f/src" "$work/f/dest" >/dev/null 2>&1
status=$?
[ "$status" -ne 0 ] && ok "write failure: nonzero exit" || fail "write failure: nonzero exit (got 0)"

# 5. Same, when the failing file has a space in its name.
mkdir -p "$work/g/src" "$work/g/dest/bad name.log.gz"
printf 'x\n' > "$work/g/src/bad name.log"
bash "$root/bin/archive-logs" "$work/g/src" "$work/g/dest" >/dev/null 2>&1
status=$?
[ "$status" -ne 0 ] && ok "write failure (spaced name): nonzero exit" || fail "write failure (spaced name): nonzero exit (got 0)"

echo
[ "$failures" -eq 0 ] && { echo "CHECK PASSED"; exit 0; }
echo "CHECK FAILED: $failures"
exit 1
