import contextlib
import io
import json
import os
import unittest

from tabstream.cli import main
from tabstream.escapes import encode_record
from tabstream.reader import read_records

CAPTURE = os.path.join(os.path.dirname(__file__), "..", "samples", "capture-0917.tsv")


class ReaderTest(unittest.TestCase):
    def test_chunks(self):
        records = [["1", "naïve", "日本語"], ["2", None, "tab\tand\nline"], ["3", "", "ü"]]
        data = "".join(map(encode_record, records)).encode("utf-8")
        for size in range(1, 6):
            with self.subTest(size=size):
                self.assertEqual(list(read_records(io.BytesIO(data), chunk_size=size)), records)

    def test_default_chunk_size(self):
        with open(CAPTURE, "rb") as f:
            self.assertEqual(len(list(read_records(f))), 6)

    def test_cli(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main([CAPTURE]), 0)
        lines = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(len(lines), 6)
        self.assertEqual(lines[1], ["88211", "failed", "disk full\ton D:", None])

    def test_cli_missing_file(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["/nonexistent/capture.tsv"]), 1)


if __name__ == "__main__":
    unittest.main()
