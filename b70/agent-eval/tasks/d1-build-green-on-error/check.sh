#!/usr/bin/env bash
# Hidden check for d1-build-green-on-error. Run from the root of the agent's
# working copy. Exit 0 only if build.sh fails whenever any step fails, still
# succeeds on good input, and keeps logging each step's output to build.log.
set -u
root=$(pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
failures=0
ok()   { echo "ok   - $1"; }
fail() { echo "FAIL - $1"; failures=$((failures + 1)); }

# 1. The repo's own tests pass.
if timeout -k 3 25 bash tests/run.sh </dev/null >"$work/own.txt" 2>&1; then
    ok "repo tests pass"
else
    fail "repo tests pass"; tail -30 "$work/own.txt"
fi

# A fresh copy of the project (without dist/ or .git) with the given extra
# config files, built with build.sh. Sets: dir, status, out, logtext.
build() {
    local name=$1; shift
    dir="$work/$name"
    mkdir -p "$dir"
    tar -C "$root" --exclude=./.git --exclude=./dist -cf - . | tar -C "$dir" -xf -
    while [ $# -gt 0 ]; do
        printf '%b' "$2" > "$dir/configs/$1"
        shift 2
    done
    out=$(cd "$dir" && PATH="${FAKEBIN:+$FAKEBIN:}$PATH" timeout -k 2 6 bash ./build.sh </dev/null 2>&1)
    status=$?
    logtext=$(cat "$dir/dist/build.log" 2>/dev/null)
}

good() { printf 'name = %s\nimage = registry.local/%s:1\nport = %s\nreplicas = %s\n' "$1" "$1" "$2" "$3"; }

# Expect the build to fail: nonzero exit and no BUILD OK anywhere.
expect_fail() {
    local what=$1
    if [ "$status" -ne 0 ] && [ "$status" -ne 124 ]; then
        ok "$what: nonzero exit"
    else
        fail "$what: nonzero exit (got $status)"
    fi
    case "$out$logtext" in
        *"BUILD OK"*) fail "$what: no BUILD OK" ;;
        *) ok "$what: no BUILD OK" ;;
    esac
}

# Expect a step's own message to be in build.log.
expect_logged() {
    case "$logtext" in
        *"$2"*) ok "$1: build.log has '$2'" ;;
        *) fail "$1: build.log has '$2'"; printf '%s\n' "$logtext" | tail -15 ;;
    esac
}

# 2. A clean build with an extra service still succeeds and logs the steps.
build clean "cache.conf" "$(good cache 6379 1)"
[ "$status" -eq 0 ] && ok "clean: exit 0" || fail "clean: exit 0 (got $status): $out"
case "$out" in *"BUILD OK"*) ok "clean: BUILD OK" ;; *) fail "clean: BUILD OK" ;; esac
listing=$(tar -tzf "$dir/dist/bundle.tar.gz" 2>/dev/null | sed 's|^\./||' | awk 'NF' | sort | tr '\n' ' ')
[ "$listing" = "api.json cache.json web.json worker.json " ] && ok "clean: bundle holds every config" \
    || fail "clean: bundle holds every config (got: $listing)"
expect_logged "clean" "gen: wrote 4 configs"

# 3. The generator fails on the last config file (earlier ones were written).
build gen-last "zz-broken.conf" 'name = zz\nimage = registry.local/zz:1\nport: 80\nreplicas = 1\n'
expect_fail "generator fails on last file"
expect_logged "generator fails on last file" 'expected "key = value"'

# 4. The generator fails on the first config file (nothing was written).
build gen-first "aa-broken.conf" 'name = aa\nimage = registry.local/aa:1\nport = eighty\nreplicas = 1\n'
expect_fail "generator fails on first file"
expect_logged "generator fails on first file" "port must be a whole number"

# 5. Validation fails for a config in the middle of the list.
build val-middle "db.conf" "$(good db 0 1)"
expect_fail "validation fails (middle)"
expect_logged "validation fails (middle)" "port must be 1-65535, got 0"

# 6. Validation fails for the last config in the list.
build val-last "zz.conf" "$(good zz 8081 0)"
expect_fail "validation fails (last)"
expect_logged "validation fails (last)" "replicas must be 1 or more, got 0"

# 7. Validation fails for the first config in the list.
build val-first "aa.conf" "$(good aa 70000 1)"
expect_fail "validation fails (first)"

# 8. Packaging fails: a tar that fails after leaving an empty archive behind
#    (so later steps that only read the archive still succeed), if the build
#    calls tar.
mkdir -p "$work/fakebin"
cat > "$work/fakebin/tar" <<SH
#!/bin/sh
touch "$work/fake-tar-ran"
for a in "\$@"; do
    case "\$a" in *.tar.gz|*.tgz) : > "\$a" 2>/dev/null ;; esac
done
echo "tar: simulated failure" >&2
exit 2
SH
chmod +x "$work/fakebin/tar"
FAKEBIN="$work/fakebin" build pkg-fail
if [ -e "$work/fake-tar-ran" ]; then
    expect_fail "packaging fails"
else
    ok "packaging fails: build does not call tar (skipped)"
fi

echo
[ "$failures" -eq 0 ] && { echo "CHECK PASSED"; exit 0; }
echo "CHECK FAILED: $failures"
exit 1
