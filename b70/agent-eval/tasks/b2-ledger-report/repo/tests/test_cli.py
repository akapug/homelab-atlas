import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from ledger import cli, money


def run(*argv):
    """Run the CLI in-process. Returns (exit_code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = cli.main(list(argv))
        except SystemExit as e:
            code = e.code
    return code, out.getvalue(), err.getvalue()


class MoneyTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(money.parse_amount("12.5"), 1250)
        self.assertEqual(money.parse_amount("-3"), -300)
        with self.assertRaises(ValueError):
            money.parse_amount("1.234")
        with self.assertRaises(ValueError):
            money.parse_amount("abc")

    def test_format(self):
        self.assertEqual(money.format_cents(1250), "12.50")
        self.assertEqual(money.format_cents(-5), "-0.05")
        self.assertEqual(money.format_cents(0), "0.00")


class CliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.file = os.path.join(self.tmp.name, "ledger.json")

    def tearDown(self):
        self.tmp.cleanup()

    def cli(self, *argv):
        return run("--file", self.file, *argv)

    def test_add_and_list(self):
        self.cli("add", "12.50", "food", "--date", "2026-09-02", "--note", "lunch")
        self.cli("add", "1000", "rent", "--date", "2026-09-01")
        code, out, _ = self.cli("list")
        self.assertEqual(code, 0)
        self.assertEqual(out.splitlines(), [
            "2026-09-01\trent\t1000.00\t",
            "2026-09-02\tfood\t12.50\tlunch",
        ])

    def test_list_category(self):
        self.cli("add", "12.50", "food", "--date", "2026-09-02")
        self.cli("add", "1000", "rent", "--date", "2026-09-01")
        _, out, _ = self.cli("list", "--category", "food")
        self.assertEqual(out.splitlines(), ["2026-09-02\tfood\t12.50\t"])

    def test_total_with_refund(self):
        self.cli("add", "12.50", "food", "--date", "2026-09-02")
        self.cli("add", "-3.50", "food", "--date", "2026-09-03")
        _, out, _ = self.cli("total")
        self.assertEqual(out, "TOTAL\t9.00\n")

    def test_bad_date_is_usage_error(self):
        code, _, err = self.cli("add", "1", "food", "--date", "2026-13-01")
        self.assertEqual(code, 2)
        self.assertIn("invalid date", err)

    def test_bad_amount_is_usage_error(self):
        code, _, _ = self.cli("add", "1.234", "food")
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
