#!/usr/bin/env bash
# Hidden check for a3-export-ties. Run from the root of the agent's working
# copy. Exit 0 only if every order is exported exactly once when many orders
# share an updated_at value, within one run and across capped runs.
set -u
root=$(pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
failures=0
ok()   { echo "ok   - $1"; }
fail() { echo "FAIL - $1"; failures=$((failures + 1)); }
# A broken fix can loop forever appending output; bound memory as well as time.
ulimit -v 8000000 2>/dev/null || true

# 1. The repo's own tests pass.
if timeout -k 5 60 python3 -m unittest discover -s tests >"$work/own.txt" 2>&1; then
    ok "repo tests pass"
else
    fail "repo tests pass"; tail -30 "$work/own.txt"
fi

# 2. Hidden behavior tests.
cat >"$work/hidden_test.py" <<'PY'
import csv
import io
import os
import signal
import subprocess
import sys
import tempfile
import unittest

from orderexp.exporter import export
from orderexp.store import Store

ROOT = os.environ["CHECK_ROOT"]


def stamp(second):
    return f"2026-08-01T12:{second // 60:02d}:{second % 60:02d}Z"


def _alarm(signum, frame):
    raise TimeoutError("export did not finish (endless loop?)")


signal.signal(signal.SIGALRM, _alarm)


class Hidden(unittest.TestCase):
    def setUp(self):
        signal.alarm(8)
        self.addCleanup(signal.alarm, 0)
        self.store = Store()
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.dir = d.name
        self.cp = os.path.join(d.name, "state.json")
        self.n = 0

    def seed(self, groups):
        """groups: [(second, how many orders at that second)]; returns the new ids."""
        ids = []
        for second, count in groups:
            for _ in range(count):
                self.n += 1
                ids.append(self.store.add(f"ORD-{self.n:05d}", stamp(second), 100 + self.n))
        return ids

    def one_run(self, **kw):
        """[(id, updated_at)] written by one export run."""
        buf = io.StringIO()
        n = export(self.store, buf, self.cp, **kw)
        rows = [(int(r[0]), r[2]) for r in csv.reader(io.StringIO(buf.getvalue())) if r and r[0].isdigit()]
        self.assertEqual(n, len(rows), "returned count differs from rows written")
        return rows

    def runs_until_done(self, limit=200, **kw):
        rows = []
        for _ in range(limit):
            got = self.one_run(**kw)
            if not got:
                return rows
            rows += got
        self.fail(f"still exporting after {limit} runs")

    def assertExactlyOnce(self, rows, want):
        got = [i for i, _ in rows]
        dups = sorted({i for i in got if got.count(i) > 1})
        missing = sorted(set(want) - set(got))
        extra = sorted(set(got) - set(want))
        self.assertEqual((missing, dups, extra), ([], [], []), "(missing, duplicated, unexpected) ids")

    def test_ties_across_page_boundaries_one_run(self):
        ids = self.seed([(0, 7), (1, 6), (2, 9), (3, 1), (4, 12), (5, 5)])
        rows = self.one_run(page_size=5)
        self.assertExactlyOnce(rows, ids)
        stamps = [s for _, s in rows]
        self.assertEqual(stamps, sorted(stamps), "not exported oldest first")

    def test_tie_group_larger_than_a_page(self):
        ids = self.seed([(0, 3), (1, 23), (2, 4)])
        self.assertExactlyOnce(self.one_run(page_size=5), ids)

    def test_everything_in_one_second(self):
        ids = self.seed([(7, 50)])
        self.assertExactlyOnce(self.one_run(page_size=8), ids)

    def test_page_size_one(self):
        ids = self.seed([(0, 3), (1, 3), (2, 1)])
        self.assertExactlyOnce(self.one_run(page_size=1), ids)

    def test_capped_runs_resume_mid_burst(self):
        ids = self.seed([(0, 4), (1, 9), (2, 2), (3, 11), (4, 3)])
        self.assertExactlyOnce(self.runs_until_done(page_size=4, max_pages=1), ids)

    def test_capped_runs_two_pages(self):
        ids = self.seed([(0, 5), (1, 13), (2, 1), (3, 7)])
        self.assertExactlyOnce(self.runs_until_done(page_size=3, max_pages=2), ids)

    def test_later_runs_pick_up_new_orders(self):
        first = self.seed([(0, 6), (1, 6)])
        self.assertExactlyOnce(self.one_run(page_size=4), first)
        later = self.seed([(10, 6), (11, 5)])
        self.assertExactlyOnce(self.runs_until_done(page_size=4, max_pages=1), later)

    def test_cli_runs(self):
        db = os.path.join(self.dir, "orders.db")
        store = Store(db)
        want = []
        n = 0
        for second, count in [(0, 5), (1, 8), (2, 3)]:
            for _ in range(count):
                n += 1
                want.append(store.add(f"ORD-{n:05d}", stamp(second), 500 + n))
        del store
        got = []
        for i in range(30):
            out = os.path.join(self.dir, f"run{i}.csv")
            p = subprocess.run([sys.executable, "-m", "orderexp.cli", "--db", db, "--checkpoint", self.cp,
                                "--out", out, "--page-size", "3", "--max-pages", "2"],
                               cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT), capture_output=True,
                               text=True, timeout=6)
            self.assertEqual(p.returncode, 0, p.stderr)
            with open(out, newline="") as f:
                rows = [(int(r[0]), r[2]) for r in csv.reader(f) if r and r[0].isdigit()]
            if not rows:
                break
            got += rows
        self.assertExactlyOnce(got, want)


if __name__ == "__main__":
    unittest.main()
PY
if CHECK_ROOT="$root" PYTHONPATH="$root" timeout -k 5 120 python3 "$work/hidden_test.py" >"$work/hidden.txt" 2>&1; then
    ok "hidden export tests"
else
    fail "hidden export tests"; tail -60 "$work/hidden.txt"
fi

echo
[ "$failures" -eq 0 ] && { echo "CHECK PASSED"; exit 0; }
echo "CHECK FAILED: $failures"
exit 1
