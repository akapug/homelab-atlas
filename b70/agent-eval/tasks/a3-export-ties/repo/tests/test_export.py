import unittest

from orderexp.exporter import export
from orderexp.store import Store

from tests.helpers import checkpoint_path, run, stamp


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.cp = checkpoint_path(self)

    def seed(self, count, start=0):
        return [self.store.add(f"ORD-{start + i:05d}", stamp(start + i), 1000 + i) for i in range(count)]

    def test_exports_everything_once(self):
        ids = self.seed(25)
        n, rows = run(export, self.store, self.cp, page_size=10)
        self.assertEqual(n, 25)
        self.assertEqual([int(r[0]) for r in rows], ids)

    def test_csv_columns(self):
        self.store.add("ORD-7", stamp(5), 1999)
        _, rows = run(export, self.store, self.cp)
        self.assertEqual(rows, [["1", "ORD-7", stamp(5), "19.99"]])

    def test_empty_store(self):
        self.assertEqual(run(export, self.store, self.cp), (0, []))

    def test_next_run_only_exports_new_orders(self):
        self.seed(12)
        run(export, self.store, self.cp, page_size=5)
        new = self.seed(3, start=100)
        n, rows = run(export, self.store, self.cp, page_size=5)
        self.assertEqual([int(r[0]) for r in rows], new)

    def test_max_pages_resumes_next_run(self):
        ids = self.seed(25)
        seen = []
        for want in (10, 10, 5, 0):
            n, rows = run(export, self.store, self.cp, page_size=10, max_pages=1)
            self.assertEqual(n, want)
            seen += [int(r[0]) for r in rows]
        self.assertEqual(seen, ids)


if __name__ == "__main__":
    unittest.main()
