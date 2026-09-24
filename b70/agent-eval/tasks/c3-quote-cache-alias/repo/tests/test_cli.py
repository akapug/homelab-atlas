import io
import json
import os
import tempfile
import unittest

from quoter import catalog, cli


class BatchTest(unittest.TestCase):
    def setUp(self):
        catalog.clear_cache()

    def run_batch(self, orders):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "orders.jsonl")
            with open(path, "w") as f:
                f.writelines(json.dumps(o) + "\n" for o in orders)
            out = io.StringIO()
            code = cli.run_batch(path, out)
        return code, [json.loads(line) for line in out.getvalue().splitlines()]

    def test_one_record_per_order(self):
        code, records = self.run_batch([
            {"id": "Q-1", "customer": "A", "plan": "starter", "seats": 3, "addons": ["sandbox"]},
            {"id": "Q-2", "customer": "B", "plan": "enterprise", "seats": 40, "addons": ["sso"]},
        ])
        self.assertEqual(code, 0)
        self.assertEqual([(r["quote"], r["total_cents"]) for r in records],
                         [("Q-1", 8100), ("Q-2", 49900 + 40 * 2500 + 20000 + 40 * 300)])

    def test_stops_at_the_first_bad_order(self):
        with self.assertLogs("quoter.batch", "ERROR"):
            code, records = self.run_batch([
                {"id": "Q-1", "customer": "A", "plan": "starter", "seats": 3},
                {"id": "Q-2", "customer": "B", "plan": "starter", "seats": 3, "addons": ["sso"]},
                {"id": "Q-3", "customer": "C", "plan": "team", "seats": 3},
            ])
        self.assertEqual(code, 1)
        self.assertEqual([r["quote"] for r in records], ["Q-1"])


if __name__ == "__main__":
    unittest.main()
