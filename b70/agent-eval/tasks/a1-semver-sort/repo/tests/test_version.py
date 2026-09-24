import unittest

from semverlite import InvalidVersion, format_version, parse


class ParseTest(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(format_version(parse("1.2.3")), "1.2.3")

    def test_v_prefix_and_whitespace(self):
        self.assertEqual(format_version(parse(" v4.0.1\n")), "4.0.1")

    def test_prerelease(self):
        self.assertEqual(parse("2.0.0-rc.1").prerelease, "rc.1")

    def test_rejects_short(self):
        with self.assertRaises(InvalidVersion):
            parse("1.2")

    def test_rejects_empty_prerelease_ident(self):
        with self.assertRaises(InvalidVersion):
            parse("1.2.3-rc..1")


if __name__ == "__main__":
    unittest.main()
