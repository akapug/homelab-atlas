#!/usr/bin/env bash
# Hidden check for a1-semver-sort. Run from the root of the agent's working
# copy. Exit 0 only if version ordering is numeric everywhere and nothing else broke.
set -u
root=$(pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
failures=0
ok()   { echo "ok   - $1"; }
fail() { echo "FAIL - $1"; failures=$((failures + 1)); }
# A broken fix can loop forever appending output; bound memory as well as time.
ulimit -v 8000000 2>/dev/null || true

# 1. The repo's own tests pass, and the original tests (from the fixture
#    commit) pass against the current code, so editing a test does not help.
if timeout -k 5 60 python3 -m unittest discover -s tests >"$work/own.txt" 2>&1; then
    ok "repo tests pass"
else
    fail "repo tests pass"; tail -30 "$work/own.txt"
fi
first=$(git rev-list --max-parents=0 HEAD 2>/dev/null | tail -1)
if [ -n "$first" ]; then
    mkdir -p "$work/orig"
    for f in test_version.py test_compare.py test_bump.py; do
        git show "$first:tests/$f" >"$work/orig/$f"
    done
    if (cd "$work/orig" && PYTHONPATH="$root" timeout -k 5 60 python3 -m unittest discover -s . >"$work/orig.txt" 2>&1); then
        ok "original tests pass"
    else
        fail "original tests pass"; tail -30 "$work/orig.txt"
    fi
fi

# 2. Hidden behavior tests.
cat >"$work/hidden_test.py" <<'PY'
import unittest

from semverlite import compare, max_version, sort_versions


class Hidden(unittest.TestCase):
    def test_multi_digit_each_part(self):
        self.assertEqual(compare("10.0.0", "9.0.0"), 1)
        self.assertEqual(compare("1.10.0", "1.9.0"), 1)
        self.assertEqual(compare("1.2.10", "1.2.9"), 1)
        self.assertEqual(compare("1.2.9", "1.2.10"), -1)
        self.assertEqual(compare("2.0.0", "10.0.0"), -1)

    def test_leading_zero_free_equality(self):
        self.assertEqual(compare("v12.3.45", "12.3.45"), 0)

    def test_sort(self):
        tags = ["0.10.0", "0.9.9", "0.10.0-beta.2", "0.10.0-beta.11", "0.2.0", "1.0.0", "0.10.1"]
        self.assertEqual(sort_versions(tags),
                         ["0.2.0", "0.9.9", "0.10.0-beta.2", "0.10.0-beta.11", "0.10.0", "0.10.1", "1.0.0"])
        self.assertEqual(sort_versions(tags, reverse=True)[0], "1.0.0")

    def test_max_multi_digit(self):
        self.assertEqual(max_version(["1.9.12", "1.10.3", "1.10.10", "1.2.99"]), "1.10.10")

    def test_max_returns_original_string(self):
        self.assertEqual(max_version(["v1.9.0", "v1.10.0"]), "v1.10.0")

    def test_prerelease_still_before_release(self):
        self.assertEqual(compare("3.0.0-rc.10", "3.0.0-rc.9"), 1)
        self.assertEqual(compare("3.0.0-rc.10", "3.0.0"), -1)


if __name__ == "__main__":
    unittest.main()
PY
if PYTHONPATH="$root" timeout -k 5 120 python3 "$work/hidden_test.py" >"$work/hidden.txt" 2>&1; then
    ok "hidden ordering tests"
else
    fail "hidden ordering tests"; tail -40 "$work/hidden.txt"
fi

echo
[ "$failures" -eq 0 ] && { echo "CHECK PASSED"; exit 0; }
echo "CHECK FAILED: $failures"
exit 1
