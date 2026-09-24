import os
import unittest

from tabstream.decoder import Decoder, decode_text
from tabstream.escapes import encode_record

CAPTURE = os.path.join(os.path.dirname(__file__), "..", "samples", "capture-0917.tsv")

RECORDS = [
    ["1", "plain", "text"],
    ["2", "tab\there", None],
    ["3", "two\nlines", "cr\r"],
    ["4", "", "naïve"],
]


def feed_in_pieces(text, size):
    decoder = Decoder()
    records = []
    for i in range(0, len(text), size):
        records += decoder.feed(text[i:i + size])
    return records + decoder.close()


class DecoderTest(unittest.TestCase):
    def test_whole_text(self):
        records = RECORDS + [["5", "C:\\temp\\", "\\N"]]
        self.assertEqual(decode_text("".join(map(encode_record, records))), records)

    def test_last_record_without_lf(self):
        self.assertEqual(decode_text("a\tb\nc\td"), [["a", "b"], ["c", "d"]])

    def test_empty(self):
        self.assertEqual(decode_text(""), [])

    def test_record_split_across_pieces(self):
        decoder = Decoder()
        self.assertEqual(decoder.feed("88210\tdo"), [])
        self.assertEqual(decoder.feed("ne\tx\n882"), [["88210", "done", "x"]])
        self.assertEqual(decoder.feed("11\ty\n"), [["88211", "y"]])
        self.assertEqual(decoder.close(), [])

    def test_escape_split_across_pieces(self):
        for first, second, want in [("a\\", "tb\n", "a\tb"), ("x\\", "n\n", "x\n"), ("\\", "N\n", None)]:
            with self.subTest(first=first):
                decoder = Decoder()
                self.assertEqual(decoder.feed(first) + decoder.feed(second) + decoder.close(), [[want]])

    def test_small_pieces(self):
        text = "".join(map(encode_record, RECORDS))
        for size in range(1, 9):
            with self.subTest(size=size):
                self.assertEqual(feed_in_pieces(text, size), RECORDS)

    def test_capture(self):
        with open(CAPTURE, encoding="utf-8") as f:
            records = decode_text(f.read())
        self.assertEqual([r[0] for r in records], ["88210", "88211", "88212", "88213", "88214", "88215"])
        self.assertEqual(records[3], ["88213", "done", "nightly export", "D:\\exports\\"])
        self.assertTrue(all(len(r) == 4 for r in records))


if __name__ == "__main__":
    unittest.main()
