import os
import unittest

from shop import invoice, report

EX = os.path.join(os.path.dirname(__file__), "..", "examples")


def ex(name):
    return invoice.load(os.path.join(EX, name))


class InvoiceTest(unittest.TestCase):
    def test_total(self):
        self.assertEqual(invoice.total(ex("inv-0042.json")), 3 * 1250 + 123456 - 2500)

    def test_render_has_lines_and_total(self):
        text = invoice.render(ex("inv-0042.json"))
        self.assertIn("Date:    01 Sep 2026", text)
        self.assertIn("€1,234.56", text)
        self.assertIn("€1,247.06", text)
        self.assertIn("-€25.00", text)


class ReportTest(unittest.TestCase):
    def test_sorted_by_date(self):
        rows = report.summary([ex("inv-0043.json"), ex("inv-0041.json")]).splitlines()
        self.assertTrue(rows[0].startswith("INV-0041   28 Aug 2026"))
        self.assertTrue(rows[1].rstrip().endswith("¥184,280"))


if __name__ == "__main__":
    unittest.main()
