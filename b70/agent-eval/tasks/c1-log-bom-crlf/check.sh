#!/usr/bin/env bash
# Hidden check for c1-log-bom-crlf. Run from the root of the agent's working
# copy. Exit 0 only if logstat gives the same summary for a log whatever its
# line endings and byte-order mark, plain or gzipped, and the repo's tests pass.
set -u
root=$(pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
failures=0
ok()   { echo "ok   - $1"; }
fail() { echo "FAIL - $1"; failures=$((failures + 1)); }
ulimit -v 8000000 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8

# 1. The repo's own tests pass.
if timeout -k 2 30 python3 -m unittest discover -s tests -t . >"$work/own.txt" 2>&1; then
    ok "repo tests pass"
else
    fail "repo tests pass"; tail -30 "$work/own.txt"
fi

# 2. Hidden tests: one log written with every combination of line ending, BOM
#    and compression must summarize exactly like the plain LF file does today.
cat >"$work/hidden_test.py" <<'PY'
import gzip
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.environ["CHECK_ROOT"]

# Two malformed lines (a truncated request and a restart marker) and one blank line.
LINES = [
    "2026-09-15T11:00:00Z GET /v2/stock?sku=A1 200 12ms",
    "2026-09-15T11:00:01Z GET /v2/stock?sku=B7 200 15ms",
    "2026-09-15T11:00:01Z POST /v2/reserve 201 84ms",
    "2026-09-15T11:00:02Z GET /v2/stock?sku=A1 200 11ms",
    "2026-09-15T11:00:04Z POST /v2/reserve 409 20ms",
    "2026-09-15T11:00:05Z GET /v2/ping 200 1ms",
    "2026-09-15T11:00:06Z POST /v2/reserve 500 1503ms",
    "2026-09-15T11:02:00Z GET /v2/stock 200",
    "== gateway restarted ==",
    "",
    "2026-09-15T11:02:07Z GET /v2/ping 200 2ms",
    "2026-09-15T11:02:09Z GET /v2/stock?sku=C3 200 14ms",
    "2026-09-15T11:02:10Z POST /v2/reserve 201 91ms",
    "2026-09-15T11:02:12Z PUT /v2/reserve/5521 200 40ms",
    "2026-09-15T11:02:15Z GET /v2/stock?sku=A1 304 3ms",
    "2026-09-15T11:02:18Z DELETE /v2/reserve/5521 204 25ms",
    "2026-09-15T11:02:20Z POST /v2/reserve 502 30001ms",
    "2026-09-15T11:02:22Z GET /v2/ping 200 1ms",
    "2026-09-15T11:02:30Z POST /v2/auth 200 77ms",
]
BOM = b"\xef\xbb\xbf"


def joined(ending, last=True):
    text = ending.join(LINES) + (ending if last else "")
    return text.encode("utf-8")


def mixed():
    return "".join(l + ("\r\n" if i % 2 else "\n") for i, l in enumerate(LINES)).encode("utf-8")


VARIANTS = {
    "lf.log": joined("\n"),
    "crlf.log": joined("\r\n"),
    "bom-crlf.log": BOM + joined("\r\n"),
    "bom-lf.log": BOM + joined("\n"),
    "bom-crlf-no-final-newline.log": BOM + joined("\r\n", last=False),
    "mixed-endings.log": BOM + mixed(),
    "bom-crlf.log.gz": BOM + joined("\r\n"),
    "crlf.log.gz": joined("\r\n"),
}

GOLDEN_ONE = """\
requests:        16
malformed lines: 2
status:          2xx 12  3xx 1  4xx 1  5xx 2

endpoint                  hits  avg ms  max ms
/v2/reserve                  5  6339.8   30001
/v2/stock                    5    11.0      15
/v2/ping                     3     1.3       2
/v2/reserve/5521             2    32.5      40

server errors:
2026-09-15T11:00:06Z POST /v2/reserve 500
2026-09-15T11:02:20Z POST /v2/reserve 502
"""
GOLDEN_TWO = """\
requests:        32
malformed lines: 4
status:          2xx 24  3xx 2  4xx 2  5xx 4

endpoint                  hits  avg ms  max ms
/v2/reserve                 10  6339.8   30001
/v2/stock                   10    11.0      15
/v2/ping                     6     1.3       2
/v2/reserve/5521             4    32.5      40

server errors:
2026-09-15T11:00:06Z POST /v2/reserve 500
2026-09-15T11:02:20Z POST /v2/reserve 502
2026-09-15T11:00:06Z POST /v2/reserve 500
2026-09-15T11:02:20Z POST /v2/reserve 502
"""


def logstat(*paths):
    env = dict(os.environ, PYTHONPATH=ROOT)
    p = subprocess.run([sys.executable, "-m", "logstat", "--top", "4", "--errors", *paths],
                       cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=20)
    return p.returncode, p.stdout, p.stderr


class Hidden(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(dir=os.environ["CHECK_WORK"])
        cls.paths = {}
        for name, data in VARIANTS.items():
            path = os.path.join(cls.dir, name)
            with (gzip.open if name.endswith(".gz") else open)(path, "wb") as f:
                f.write(data)
            cls.paths[name] = path

    def test_every_variant_matches_plain_lf(self):
        for name, path in self.paths.items():
            with self.subTest(file=name):
                code, out, err = logstat(path)
                self.assertEqual(code, 0, err)
                self.assertEqual(out, GOLDEN_ONE)

    def test_several_files_at_once(self):
        code, out, err = logstat(self.paths["bom-crlf.log.gz"], self.paths["lf.log"])
        self.assertEqual(code, 0, err)
        self.assertEqual(out, GOLDEN_TWO)


if __name__ == "__main__":
    unittest.main(verbosity=2)
PY
if CHECK_ROOT="$root" CHECK_WORK="$work" timeout -k 2 80 python3 "$work/hidden_test.py" >"$work/hidden.txt" 2>&1; then
    ok "hidden line-ending and BOM tests"
else
    fail "hidden line-ending and BOM tests"; tail -60 "$work/hidden.txt"
fi

echo
[ "$failures" -eq 0 ] && { echo "CHECK PASSED"; exit 0; }
echo "CHECK FAILED: $failures"
exit 1
