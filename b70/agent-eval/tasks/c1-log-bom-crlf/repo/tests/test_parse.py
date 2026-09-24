import unittest

from logstat.parse import parse_line


class ParseTest(unittest.TestCase):
    def test_request(self):
        r = parse_line("2026-09-13T09:00:01Z GET /api/items?page=1 200 41ms")
        self.assertEqual((r.ts, r.method, r.path, r.status, r.ms),
                         ("2026-09-13T09:00:01Z", "GET", "/api/items?page=1", 200, 41))
        self.assertEqual(r.endpoint, "/api/items")

    def test_truncated_line_is_malformed(self):
        self.assertIsNone(parse_line("2026-09-13T09:00:31Z POST /api/orders 502"))

    def test_other_text_is_malformed(self):
        self.assertIsNone(parse_line("-- log rotated --"))
        self.assertIsNone(parse_line("2026-09-13T09:00:01Z GET /api/items 200 41ms trailing"))


if __name__ == "__main__":
    unittest.main()
