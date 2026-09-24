#!/usr/bin/env bash
# Hidden check for a2-retry-backoff. Run from the root of the agent's working
# copy. Exit 0 only if retries back off exponentially per the config and the
# repo's own tests pass.
set -u
root=$(pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
failures=0
ok()   { echo "ok   - $1"; }
fail() { echo "FAIL - $1"; failures=$((failures + 1)); }
# A broken fix can loop forever appending output; bound memory as well as time.
ulimit -v 8000000 2>/dev/null || true
# Settings from the caller's environment must not leak into the checks.
for v in $(env | sed -n 's/^\(SYNC_[A-Z_]*\)=.*/\1/p'); do unset "$v"; done

# 1. The repo's own (updated) tests pass.
if timeout -k 5 60 python3 -m unittest discover -s tests >"$work/own.txt" 2>&1; then
    ok "repo tests pass"
else
    fail "repo tests pass"; tail -30 "$work/own.txt"
fi

# 2. Hidden behavior tests: config file/env -> load_config -> call_with_retry -> sleeps.
cat >"$work/hidden_test.py" <<'PY'
import os
import re
import subprocess
import sys
import tempfile
import unittest

from syncer.client import RetryError, call_with_retry
from syncer.config import ConfigError, load_config

ROOT = os.environ["CHECK_ROOT"]


def sleeps_for(cfg, failures):
    calls = []

    def fn():
        calls.append(1)
        if len(calls) <= failures:
            raise ConnectionError("down")
        return "ok"
    slept = []
    try:
        result = call_with_retry(fn, cfg.retry, sleep=slept.append)
    except RetryError:
        result = None
    return result, slept


class Hidden(unittest.TestCase):
    def write(self, text):
        fd, path = tempfile.mkstemp(suffix=".ini")
        with os.fdopen(fd, "w") as f:
            f.write(text)
        self.addCleanup(os.remove, path)
        return path

    def assertSleeps(self, got, want):
        self.assertEqual(len(got), len(want), f"sleeps {got}, want {want}")
        for g, w in zip(got, want):
            self.assertAlmostEqual(g, w, msg=f"sleeps {got}, want {want}")

    def test_defaults(self):
        result, slept = sleeps_for(load_config(None, env={}), 99)
        self.assertIsNone(result)
        self.assertSleeps(slept, [1, 2, 4])

    def test_default_cap_is_30(self):
        cfg = load_config(None, env={"SYNC_RETRY_MAX_ATTEMPTS": "8"})
        self.assertSleeps(sleeps_for(cfg, 99)[1], [1, 2, 4, 8, 16, 30, 30])

    def test_file_values_and_cap(self):
        path = self.write("[retry]\nmax_attempts = 6\nbackoff_base = 0.5\nbackoff_max = 3\n")
        self.assertSleeps(sleeps_for(load_config(path, env={}), 99)[1], [0.5, 1, 2, 3, 3])

    def test_env_overrides_file(self):
        path = self.write("[retry]\nmax_attempts = 6\nbackoff_base = 0.5\nbackoff_max = 3\n")
        env = {"SYNC_RETRY_BACKOFF_BASE": "2", "SYNC_RETRY_BACKOFF_MAX": "5"}
        self.assertSleeps(sleeps_for(load_config(path, env=env), 99)[1], [2, 4, 5, 5, 5])

    def test_legacy_delay_key_is_base(self):
        path = self.write("[retry]\nmax_attempts = 4\ndelay = 3\n")
        self.assertSleeps(sleeps_for(load_config(path, env={}), 99)[1], [3, 6, 12])

    def test_success_midway(self):
        path = self.write("[retry]\nmax_attempts = 6\nbackoff_base = 0.25\nbackoff_max = 10\n")
        result, slept = sleeps_for(load_config(path, env={}), 2)
        self.assertEqual(result, "ok")
        self.assertSleeps(slept, [0.25, 0.5])

    def test_max_below_base_rejected(self):
        path = self.write("[retry]\nbackoff_base = 5\nbackoff_max = 2\n")
        with self.assertRaises(ConfigError):
            load_config(path, env={})
        with self.assertRaises(ConfigError):
            load_config(None, env={"SYNC_RETRY_BACKOFF_MAX": "0.5"})

    def test_equal_max_and_base_allowed(self):
        cfg = load_config(None, env={"SYNC_RETRY_BACKOFF_BASE": "2", "SYNC_RETRY_BACKOFF_MAX": "2"})
        self.assertSleeps(sleeps_for(cfg, 99)[1], [2, 2, 2])

    def test_show_config(self):
        path = self.write("[retry]\nmax_attempts = 6\nbackoff_base = 0.5\nbackoff_max = 3\n")
        env = {k: v for k, v in os.environ.items() if not k.startswith("SYNC_")}
        env["PYTHONPATH"] = ROOT
        out = subprocess.run([sys.executable, "-m", "syncer.cli", "--config", path, "show-config"],
                             cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertRegex(out.stdout, r"backoff_base\D*0?\.5(?!\d)")
        self.assertRegex(out.stdout, r"backoff_max\D*3(\.0+)?(?![\d.])")


if __name__ == "__main__":
    unittest.main()
PY
if CHECK_ROOT="$root" PYTHONPATH="$root" timeout -k 5 120 python3 "$work/hidden_test.py" >"$work/hidden.txt" 2>&1; then
    ok "hidden backoff tests"
else
    fail "hidden backoff tests"; tail -60 "$work/hidden.txt"
fi

echo
[ "$failures" -eq 0 ] && { echo "CHECK PASSED"; exit 0; }
echo "CHECK FAILED: $failures"
exit 1
