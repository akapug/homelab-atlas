#!/usr/bin/env bash
# Hidden check for b2-ledger-report. Run from the root of the agent's
# working copy. Exit 0 only if the required behavior holds.
set -u
root=$(pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
BASELINE_TESTS=7

# 1. The repo's own suite passes, and it grew tests for `report`.
python3 -m unittest discover -s tests -t . >"$work/own.txt" 2>&1
own=$?
if [ "$own" -ne 0 ]; then
    echo "FAIL - repo test suite does not pass"; tail -n 30 "$work/own.txt"; exit 1
fi
ran=$(python3 - "$work/own.txt" <<'PY'
import re, sys
m = re.search(r"^Ran (\d+) tests?", open(sys.argv[1]).read(), re.M)
print(m.group(1) if m else 0)
PY
)
if [ "$ran" -le "$BASELINE_TESTS" ]; then
    echo "FAIL - no new tests were added (ran $ran, baseline $BASELINE_TESTS)"; exit 1
fi
mentions=$(python3 - <<'PY'
import pathlib
print(sum("report" in p.read_text(errors="replace") for p in pathlib.Path("tests").rglob("*.py")))
PY
)
if [ "$mentions" -eq 0 ]; then
    echo "FAIL - no test file under tests/ exercises the report command"; exit 1
fi
echo "ok   - repo suite passes with $ran tests (baseline $BASELINE_TESTS) and covers report"

# 2. Hidden behavior tests, through the real command line.
cat > "$work/hidden_check.py" <<'PY'
import os, subprocess, sys, tempfile, unittest

ROOT = os.environ["LEDGER_ROOT"]

def ledger(path, *argv):
    env = dict(os.environ, PYTHONPATH=ROOT)
    env.pop("LEDGER_FILE", None)
    p = subprocess.run([sys.executable, "-m", "ledger", "--file", path, *argv],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    return p.returncode, p.stdout, p.stderr

ENTRIES = [
    ("12.50", "food", "2026-08-30"),
    ("-3.50", "food", "2026-09-01"),
    ("1000", "rent", "2026-09-02"),
    ("0.10", "fun", "2026-09-15"),
    ("0.20", "fun", "2026-09-15"),
    ("7.25", "books", "2026-07-04"),
]

class Report(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "l.json")
        for amt, cat, day in ENTRIES:
            code, out, err = ledger(cls.path, "add", amt, cat, "--date", day)
            assert code == 0, (amt, cat, day, out, err)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def report(self, *argv):
        return ledger(self.path, "report", *argv)

    def assertLines(self, argv, want):
        code, out, err = self.report(*argv)
        self.assertEqual(code, 0, err)
        self.assertEqual(out.splitlines(), want)

    def test_by_category(self):
        self.assertLines(["--by", "category"], [
            "books\t7.25", "food\t9.00", "fun\t0.30", "rent\t1000.00", "TOTAL\t1016.55"])

    def test_by_month(self):
        self.assertLines(["--by", "month"], [
            "2026-07\t7.25", "2026-08\t12.50", "2026-09\t996.80", "TOTAL\t1016.55"])

    def test_since_is_inclusive(self):
        self.assertLines(["--by", "category", "--since", "2026-09-01"], [
            "food\t-3.50", "fun\t0.30", "rent\t1000.00", "TOTAL\t996.80"])

    def test_since_by_month(self):
        self.assertLines(["--since", "2026-09-02", "--by", "month"], [
            "2026-09\t1000.30", "TOTAL\t1000.30"])

    def test_since_after_everything(self):
        self.assertLines(["--by", "month", "--since", "2027-01-01"], ["TOTAL\t0.00"])

    def test_bad_since_is_usage_error(self):
        for bad in ("2026-13-01", "yesterday"):
            code, out, err = self.report("--by", "month", "--since", bad)
            self.assertEqual(code, 2, (bad, out, err))
            self.assertEqual(out, "")
            self.assertTrue(err.strip(), "expected an error message on stderr")

    def test_by_is_required_and_checked(self):
        for argv in ([], ["--by", "week"]):
            code, out, err = self.report(*argv)
            self.assertEqual(code, 2, (argv, out, err))
            self.assertEqual(out, "")

    def test_empty_ledger(self):
        empty = os.path.join(self.tmp.name, "none.json")
        code, out, err = ledger(empty, "report", "--by", "category")
        self.assertEqual(code, 0, err)
        self.assertEqual(out.splitlines(), ["TOTAL\t0.00"])

    def test_existing_commands_unchanged(self):
        code, out, _ = ledger(self.path, "total")
        self.assertEqual((code, out), (0, "TOTAL\t1016.55\n"))
        code, out, _ = ledger(self.path, "list", "--category", "fun")
        self.assertEqual(out.splitlines(), ["2026-09-15\tfun\t0.10\t", "2026-09-15\tfun\t0.20\t"])

if __name__ == "__main__":
    unittest.main(verbosity=2)
PY
if LEDGER_ROOT="$root" python3 "$work/hidden_check.py" >"$work/hidden.txt" 2>&1; then
    echo "ok   - hidden report behavior"; echo "CHECK PASSED"; exit 0
fi
echo "FAIL - hidden report behavior"; tail -n 40 "$work/hidden.txt"
exit 1
