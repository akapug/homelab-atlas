#!/usr/bin/env bash
# Hidden check for d2-config-v2-migration. Run from the root of the agent's
# working copy. Exit 0 only if v1 job files still load (as v2), v2 files load,
# round-trip and are checked, bkp fmt writes v2, plans are right for both, and
# the repo's tests were updated to cover the new format.
set -u
root=$(pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
failures=0
ok()   { echo "ok   - $1"; }
fail() { echo "FAIL - $1"; failures=$((failures + 1)); }

# 1. The repo's own tests pass.
if timeout -k 3 20 python3 -m unittest discover -s tests >"$work/own.txt" 2>&1; then
    ok "repo tests pass"
else
    fail "repo tests pass"; tail -30 "$work/own.txt"
fi

# 2. The repo's tests were updated: run against the original code, they fail.
base=$(git rev-list --max-parents=0 HEAD 2>/dev/null | tail -n 1)
mkdir -p "$work/orig"
if [ -n "$base" ] && git archive "$base" | tar -x -C "$work/orig" 2>/dev/null && [ -d "$work/orig/bkp" ]; then
    rm -rf "$work/orig/tests"
    cp -r tests "$work/orig/tests"
    rm -rf "$work/orig/tests/__pycache__"
    if (cd "$work/orig" && timeout -k 3 20 python3 -m unittest discover -s tests >"$work/orig.txt" 2>&1); then
        fail "repo tests cover v2 (they still pass against the original v1-only code)"
    else
        ok "repo tests cover v2"
    fi
else
    fail "repo tests cover v2 (could not read the original fixture from git)"
fi

# 3. Hidden behavior tests.
cat >"$work/hidden_test.py" <<'PY'
import json
import os
import subprocess
import sys
import tempfile
import unittest

from bkp.config import ConfigError, load, save

ROOT = os.environ["CHECK_ROOT"]
V1_ONLY = {"source_dir", "dest", "exclude"}
V2_KEYS = {"name", "source", "target", "keep", "compression"}

V1_FULL = {"name": "mail", "source_dir": "/var/mail", "dest": "/backup/mail box",
           "exclude": ["*.lock", "tmp dir/"], "keep": 30}
V1_MIN = {"version": 1, "name": "etc", "source_dir": "/etc", "dest": "/backup"}
V2_ZSTD = {"version": 2, "name": "db", "source": {"path": "/var/lib/db", "exclude": ["*.wal"]},
           "target": "/backup/db", "keep": 5, "compression": "zstd"}
V2_NONE = {"version": 2, "name": "iso", "source": {"path": "/srv/iso"}, "target": "/backup/iso",
           "compression": "none"}
V2_GZIP = {"compression": "gzip", "keep": 2, "name": "home", "target": "/backup",
           "source": {"exclude": [], "path": "/home"}, "version": 2}

PLANS = {
    "V1_FULL": ["job mail",
                "  tar -czf '/backup/mail box/mail-{date}.tar.gz' --exclude='*.lock' --exclude='tmp dir/' -C /var/mail .",
                "  prune '/backup/mail box/mail-*.tar.gz' keep 30"],
    "V1_MIN": ["job etc",
               "  tar -czf '/backup/etc-{date}.tar.gz' -C /etc .",
               "  prune '/backup/etc-*.tar.gz' keep 7"],
    "V2_ZSTD": ["job db",
                "  tar --zstd -cf '/backup/db/db-{date}.tar.zst' --exclude='*.wal' -C /var/lib/db .",
                "  prune '/backup/db/db-*.tar.zst' keep 5"],
    "V2_NONE": ["job iso",
                "  tar -cf '/backup/iso/iso-{date}.tar' -C /srv/iso .",
                "  prune '/backup/iso/iso-*.tar' keep 7"],
    "V2_GZIP": ["job home",
                "  tar -czf '/backup/home-{date}.tar.gz' -C /home .",
                "  prune '/backup/home-*.tar.gz' keep 2"],
}


def body(job):
    """A loaded job without its version key (whether load() returns one is not specified)."""
    return {k: v for k, v in job.items() if k != "version"}


def without(data, *keys):
    return {k: v for k, v in data.items() if k not in keys}


def bkp(*argv):
    p = subprocess.run([sys.executable, "-m", "bkp", *argv], cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT),
                       capture_output=True, text=True, timeout=10)
    return p.returncode, p.stdout, p.stderr


class Hidden(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.dir = d.name

    def write(self, data, name="job.json"):
        path = os.path.join(self.dir, name)
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def read(self, path):
        with open(path) as f:
            return json.load(f)

    # --- v1 files still load, in the v2 shape ---
    def test_v1_full_loads_as_v2(self):
        self.assertEqual(body(load(self.write(V1_FULL))), {
            "name": "mail", "source": {"path": "/var/mail", "exclude": ["*.lock", "tmp dir/"]},
            "target": "/backup/mail box", "keep": 30, "compression": "gzip"})

    def test_v1_minimal_loads_with_defaults(self):
        self.assertEqual(body(load(self.write(V1_MIN))), {
            "name": "etc", "source": {"path": "/etc", "exclude": []},
            "target": "/backup", "keep": 7, "compression": "gzip"})

    def test_v1_is_still_checked(self):
        for bad in (without(V1_FULL, "dest"), without(V1_FULL, "source_dir"), dict(V1_FULL, keep=0),
                    dict(V1_FULL, colour="red"), dict(V1_FULL, exclude="*.lock")):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                load(self.write(bad))

    def test_v1_saves_as_v2(self):
        job = load(self.write(V1_FULL))
        out = os.path.join(self.dir, "out.json")
        save(job, out)
        raw = self.read(out)
        self.assertEqual(raw.get("version"), 2)
        self.assertEqual(set(raw), V2_KEYS | {"version"})
        self.assertEqual(raw["compression"], "gzip")
        self.assertEqual(body(load(out)), body(job))

    # --- v2 files load as written, round-trip and are checked ---
    def test_v2_loads_as_written(self):
        self.assertEqual(body(load(self.write(V2_ZSTD))), without(V2_ZSTD, "version"))
        self.assertEqual(body(load(self.write(V2_GZIP))), without(V2_GZIP, "version"))

    def test_v2_defaults(self):
        self.assertEqual(body(load(self.write(V2_NONE))), {
            "name": "iso", "source": {"path": "/srv/iso", "exclude": []},
            "target": "/backup/iso", "keep": 7, "compression": "none"})

    def test_v2_round_trips(self):
        for data in (V2_ZSTD, V2_NONE, V2_GZIP):
            with self.subTest(name=data["name"]):
                job = load(self.write(data))
                out = os.path.join(self.dir, "rt.json")
                save(job, out)
                raw = self.read(out)
                self.assertEqual(raw.get("version"), 2)
                self.assertEqual(body(raw), body(job))
                self.assertEqual(load(out), job)

    def test_v2_needs_compression(self):
        with self.assertRaises(ConfigError):
            load(self.write(without(V2_ZSTD, "compression")))

    def test_v2_bad_compression(self):
        for bad in ("lz4", "", "GZIP", None):
            with self.subTest(compression=bad), self.assertRaises(ConfigError):
                load(self.write(dict(V2_ZSTD, compression=bad)))

    def test_v2_rejects_v1_keys(self):
        for bad in (dict(without(V2_ZSTD, "target"), dest="/backup/db"),
                    dict(V2_ZSTD, dest="/backup/db"),
                    dict(V2_ZSTD, source_dir="/var/lib/db"),
                    dict(V2_ZSTD, exclude=["*.wal"])):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                load(self.write(bad))

    def test_v2_needs_source_path_and_target(self):
        for bad in (dict(V2_ZSTD, source={"exclude": []}), without(V2_ZSTD, "target"),
                    without(V2_ZSTD, "source"), dict(V2_ZSTD, source="/var/lib/db")):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                load(self.write(bad))

    def test_other_versions(self):
        for version in (3, 0, "2"):
            with self.subTest(version=version), self.assertRaises(ConfigError):
                load(self.write(dict(V2_ZSTD, version=version)))

    # --- the command line ---
    def test_plans(self):
        for name, data in (("V1_FULL", V1_FULL), ("V1_MIN", V1_MIN), ("V2_ZSTD", V2_ZSTD),
                           ("V2_NONE", V2_NONE), ("V2_GZIP", V2_GZIP)):
            with self.subTest(job=name):
                code, out, err = bkp("plan", self.write(data))
                self.assertEqual(code, 0, err)
                self.assertEqual(out.splitlines(), PLANS[name])

    def test_fmt_migrates_v1_and_keeps_the_plan(self):
        path = self.write(V1_FULL)
        code, before, err = bkp("plan", path)
        self.assertEqual(code, 0, err)
        code, _, err = bkp("fmt", path)
        self.assertEqual(code, 0, err)
        raw = self.read(path)
        self.assertEqual(raw.get("version"), 2)
        self.assertFalse(V1_ONLY & set(raw), raw)
        self.assertEqual(bkp("plan", path)[1], before)

    def test_fmt_is_idempotent_on_v2(self):
        path = self.write(V2_ZSTD)
        self.assertEqual(bkp("fmt", path)[0], 0)
        with open(path) as f:
            once = f.read()
        self.assertEqual(json.loads(once), V2_ZSTD)
        self.assertTrue(once.endswith("\n"))
        self.assertEqual(bkp("fmt", path)[0], 0)
        with open(path) as f:
            self.assertEqual(f.read(), once)

    def test_fmt_leaves_a_bad_v2_file_alone(self):
        path = self.write(without(V2_ZSTD, "compression"))
        with open(path) as f:
            before = f.read()
        code, _, err = bkp("fmt", path)
        self.assertEqual(code, 1)
        self.assertIn("bkp:", err)
        with open(path) as f:
            self.assertEqual(f.read(), before)

    def test_show(self):
        for data, want in ((V1_FULL, ("mail", "/var/mail", "/backup/mail box", "30")),
                           (V2_ZSTD, ("db", "/var/lib/db", "/backup/db", "5"))):
            with self.subTest(job=data["name"]):
                code, out, err = bkp("show", self.write(data))
                self.assertEqual(code, 0, err)
                for text in want:
                    self.assertIn(text, out)


if __name__ == "__main__":
    unittest.main()
PY
if CHECK_ROOT="$root" PYTHONPATH="$root" timeout -k 3 60 python3 "$work/hidden_test.py" >"$work/hidden.txt" 2>&1; then
    ok "hidden config tests"
else
    fail "hidden config tests"; tail -60 "$work/hidden.txt"
fi

echo
[ "$failures" -eq 0 ] && { echo "CHECK PASSED"; exit 0; }
echo "CHECK FAILED: $failures"
exit 1
