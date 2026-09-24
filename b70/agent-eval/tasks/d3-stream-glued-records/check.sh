#!/usr/bin/env bash
# Hidden check for d3-stream-glued-records. Run from the root of the agent's
# working copy. Exit 0 only if decoding gives the same records however the
# input is split into pieces, through Decoder.feed and through read_records.
set -u
root=$(pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
failures=0
ok()   { echo "ok   - $1"; }
fail() { echo "FAIL - $1"; failures=$((failures + 1)); }
# A broken fix can loop forever growing a buffer; bound memory as well as time.
ulimit -v 8000000 2>/dev/null || true

# 1. The repo's own tests pass.
if timeout -k 3 20 python3 -m unittest discover -s tests >"$work/own.txt" 2>&1; then
    ok "repo tests pass"
else
    fail "repo tests pass"; tail -30 "$work/own.txt"
fi

# 2. Hidden behavior tests. Streams are built by this file's own encoder, so the
#    expected records never come from the code under test.
cat >"$work/hidden_test.py" <<'PY'
import io
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import unittest

from tabstream.decoder import Decoder
from tabstream.escapes import DecodeError
from tabstream.reader import read_records

ROOT = os.environ["CHECK_ROOT"]
WIRE = {"\\": "\\\\", "\t": "\\t", "\n": "\\n", "\r": "\\r"}


def enc(records):
    return "".join("\t".join("\\N" if f is None else "".join(WIRE.get(c, c) for c in f) for f in r) + "\n"
                   for r in records)


CAPTURE = [
    ["88210", "done", "nightly export", "D:\\exports\\2026-09-17\\orders.csv"],
    ["88211", "failed", "disk full\ton D:", None],
    ["88212", "done", None, "D:\\exports\\2026-09-17\\refunds.csv"],
    ["88213", "done", "nightly export", "D:\\exports\\"],
    ["88214", "done", "nightly export", "D:\\exports\\2026-09-17\\stock.csv"],
    ["88215", "skipped", "no changes\nsince 88190", None],
]

TRICKY = [
    CAPTURE[3:5],
    [["a\\"], ["b"]],
    [["\\"], ["\\\\"], ["\\\\\\"], ["x"]],
    [["\\", "\\"], ["\\n", "n\\"], ["\\\\t"]],
    [[None, "\\N", "\\"], ["N\\", None]],
    [["tab\there\\", "nl\nhere", "cr\r\\"], ["\\\r\n\t"]],
    [["é\\", "日本\\\\"], ["ü", "\\ß"]],
    [["", ""], ["\\"], ["", "\\\\"]],
]


def by_pieces(pieces):
    d = Decoder()
    out = []
    for p in pieces:
        out += d.feed(p)
    return out + d.close()


def _alarm(signum, frame):
    raise TimeoutError("decoding did not finish (endless loop?)")


signal.signal(signal.SIGALRM, _alarm)


class Hidden(unittest.TestCase):
    def setUp(self):
        signal.alarm(8)
        self.addCleanup(signal.alarm, 0)

    def test_reported_case(self):
        text = enc(CAPTURE)
        cut = text.index("D:\\\\exports\\\\\n") + len("D:\\\\exports\\")
        self.assertEqual(text[cut - 1:cut + 2], "\\\\\n")
        self.assertEqual(by_pieces([text[:cut], text[cut:]]), CAPTURE)

    def test_every_split_in_two(self):
        for records in TRICKY + [CAPTURE]:
            text = enc(records)
            for cut in range(len(text) + 1):
                with self.subTest(text=text, cut=cut):
                    self.assertEqual(by_pieces([text[:cut], text[cut:]]), records)

    def test_every_split_in_three(self):
        for records in TRICKY:
            text = enc(records)
            for a in range(len(text) + 1):
                for b in range(a, len(text) + 1):
                    got = by_pieces([text[:a], text[a:b], text[b:]])
                    if got != records:
                        self.fail(f"pieces {text[:a]!r} {text[a:b]!r} {text[b:]!r}: got {got!r}")

    def test_one_character_at_a_time(self):
        records = [r for group in TRICKY for r in group] + CAPTURE
        self.assertEqual(by_pieces(list(enc(records))), records)

    def test_last_record_without_lf(self):
        for records in ([["x", "\\"]], [["x", "a\\\\"]], [["\\"], ["y\\"]]):
            text = enc(records)[:-1]
            for cut in range(len(text) + 1):
                with self.subTest(text=text, cut=cut):
                    self.assertEqual(by_pieces([text[:cut], text[cut:]]), records)

    def test_dangling_backslash_at_the_end_is_an_error(self):
        text = "a\tb\\\\c\\"
        for cut in range(len(text) + 1):
            with self.subTest(cut=cut), self.assertRaises(DecodeError):
                by_pieces([text[:cut], text[cut:]])

    def test_random_streams_and_pieces(self):
        rng = random.Random(20260923)
        alphabet = ["a", "b", "\\", "\\", "\\", "\t", "\n", "\r", "N", " ", "é"]
        for _ in range(400):
            records = [[None if rng.random() < 0.1 else "".join(rng.choice(alphabet) for _ in range(rng.randrange(7)))
                        for _ in range(rng.randrange(1, 5))] for _ in range(rng.randrange(1, 6))]
            text = enc(records)
            cuts = sorted(rng.sample(range(len(text) + 1), min(len(text) + 1, rng.randrange(1, 12))))
            pieces = [text[a:b] for a, b in zip([0] + cuts, cuts + [len(text)])]
            self.assertEqual(by_pieces(pieces), records, f"pieces {pieces!r}")

    def test_read_records_any_chunk_size(self):
        records = [r for group in TRICKY for r in group] + CAPTURE
        data = enc(records).encode("utf-8")
        for size in list(range(1, 40)) + [64, 100, 4096, 65536]:
            with self.subTest(chunk_size=size):
                self.assertEqual(list(read_records(io.BytesIO(data), chunk_size=size)), records)

    def test_large_record(self):
        big = ["id", "x\\" * 2000 + "\\", "tail\\"]
        records = [big, ["next", "\\"]]
        data = enc(records).encode("utf-8")
        for size in (7, 8, 63, 64):
            with self.subTest(chunk_size=size):
                self.assertEqual(list(read_records(io.BytesIO(data), chunk_size=size)), records)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "capture.tsv")
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(enc(CAPTURE))
            p = subprocess.run([sys.executable, "-m", "tabstream", path], cwd=ROOT, capture_output=True,
                               text=True, timeout=8, env=dict(os.environ, PYTHONPATH=ROOT))
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual([json.loads(line) for line in p.stdout.splitlines()], CAPTURE)


if __name__ == "__main__":
    unittest.main()
PY
if CHECK_ROOT="$root" PYTHONPATH="$root" timeout -k 3 65 python3 "$work/hidden_test.py" >"$work/hidden.txt" 2>&1; then
    ok "hidden decoding tests"
else
    fail "hidden decoding tests"; tail -60 "$work/hidden.txt"
fi

echo
[ "$failures" -eq 0 ] && { echo "CHECK PASSED"; exit 0; }
echo "CHECK FAILED: $failures"
exit 1
