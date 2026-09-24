#!/usr/bin/env bash
# Tests for build.sh and its tools. Run from anywhere: tests/run.sh
cd "$(dirname "$0")/.." || exit 1
repo=$(pwd)

failures=0
ok()   { echo "ok   - $1"; }
fail() { echo "FAIL - $1"; failures=$((failures + 1)); }

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# A copy of the project to build in, so the tests never touch ./dist.
copy() {
    rm -rf "$work/p"
    mkdir -p "$work/p"
    cp -r "$repo/build.sh" "$repo/tools" "$repo/configs" "$work/p/"
}

# --- the tools reject bad input ---
printf 'name = x\nport 80\n' > "$work/bad.conf"
mkdir -p "$work/gen-out"
python3 tools/gen.py "$work" "$work/gen-out" 2>/dev/null
[ $? -eq 1 ] && ok "gen.py exits 1 on a malformed line" || fail "gen.py exits 1 on a malformed line"
printf '{"name": "x", "image": "i", "port": 0, "replicas": 1}\n' > "$work/bad.json"
python3 tools/validate.py "$work/bad.json" 2>/dev/null
[ $? -eq 1 ] && ok "validate.py exits 1 on port 0" || fail "validate.py exits 1 on port 0"
printf '{"name": "x", "image": "i", "port": 80, "replicas": 1}\n' > "$work/good.json"
python3 tools/validate.py "$work/good.json"
[ $? -eq 0 ] && ok "validate.py accepts a good config" || fail "validate.py accepts a good config"

# --- a clean build ---
copy
out=$(bash "$work/p/build.sh" 2>&1)
status=$?
[ "$status" -eq 0 ] && ok "build exits 0" || fail "build exits 0 (got $status): $out"
case "$out" in
    *"BUILD OK"*) ok "build reports BUILD OK" ;;
    *) fail "build reports BUILD OK" ;;
esac
listing=$(tar -tzf "$work/p/dist/bundle.tar.gz" 2>/dev/null | sort | tr '\n' ' ')
[ "$listing" = "./ ./api.json ./web.json ./worker.json " ] && ok "bundle holds every config" \
    || fail "bundle holds every config (got: $listing)"
(cd "$work/p/dist" && sha256sum -c --quiet bundle.sha256) && ok "checksum matches" || fail "checksum matches"
grep -q "wrote 3 configs" "$work/p/dist/build.log" && ok "build.log has the generator output" \
    || fail "build.log has the generator output"

echo
if [ "$failures" -eq 0 ]; then
    echo "all tests passed"
else
    echo "$failures test(s) failed"
    exit 1
fi
