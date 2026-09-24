import unittest

from shop import money


class FmtTest(unittest.TestCase):
    def test_defaults_to_dollars(self):
        self.assertEqual(money.fmt(123456), "$1,234.56")

    def test_other_currencies(self):
        self.assertEqual(money.fmt(-5, "EUR"), "-€0.05")
        self.assertEqual(money.fmt(1500, currency="JPY"), "¥1,500")
        self.assertEqual(money.fmt(100, "CHF"), "CHF 1.00")

    def test_without_symbol(self):
        self.assertEqual(money.fmt(1500, "JPY", False), "1,500")

    def test_pad(self):
        self.assertEqual(money.fmt(250, "GBP", pad=8), "   £2.50")


if __name__ == "__main__":
    unittest.main()
