import gzip
import os
import shutil
import tempfile
import unittest

from logstat.cli import load
from logstat.report import server_errors, summarize

DATA = os.path.join(os.path.dirname(__file__), "data", "app-2026-09-13.log")


class ReportTest(unittest.TestCase):
    def test_counts(self):
        requests, malformed = load([DATA])
        self.assertEqual((len(requests), malformed), (15, 1))
        text = summarize(requests, malformed)
        self.assertIn("requests:        15", text)
        self.assertIn("status:          2xx 12  3xx 1  4xx 1  5xx 1", text)

    def test_busiest_first_ties_by_name(self):
        requests, malformed = load([DATA])
        rows = summarize(requests, malformed, top=3).splitlines()[-3:]
        self.assertEqual([r.split()[0] for r in rows], ["/api/items", "/api/orders", "/health"])

    def test_server_errors(self):
        requests, _ = load([DATA])
        self.assertEqual(server_errors(requests), "2026-09-13T09:00:15Z POST /api/orders 500")

    def test_gzip_log(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "app.log.gz")
            with open(DATA, "rb") as src, gzip.open(path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            self.assertEqual(load([path]), load([DATA]))


if __name__ == "__main__":
    unittest.main()
