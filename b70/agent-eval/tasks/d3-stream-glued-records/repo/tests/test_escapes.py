import unittest

from tabstream.escapes import DecodeError, decode_record, encode_record, escape, unescape


class EscapeTest(unittest.TestCase):
    def test_round_trip(self):
        for value in ["plain", "", "C:\\temp\\", "a\tb", "two\nlines", "cr\r", "\\N", "naïve 日本", None]:
            with self.subTest(value=value):
                self.assertEqual(unescape(escape(value)), value)

    def test_wire_form(self):
        self.assertEqual(escape("a\\b\tc\nd"), "a\\\\b\\tc\\nd")
        self.assertEqual(escape(None), "\\N")
        self.assertEqual(escape("\\N"), "\\\\N")

    def test_dangling_backslash(self):
        with self.assertRaises(DecodeError):
            unescape("abc\\")

    def test_unknown_escape(self):
        with self.assertRaises(DecodeError):
            unescape("a\\qb")

    def test_record(self):
        fields = ["88213", "done", None, "D:\\exports\\"]
        line = encode_record(fields)
        self.assertEqual(line, "88213\tdone\t\\N\tD:\\\\exports\\\\\n")
        self.assertEqual(decode_record(line[:-1]), fields)


if __name__ == "__main__":
    unittest.main()
