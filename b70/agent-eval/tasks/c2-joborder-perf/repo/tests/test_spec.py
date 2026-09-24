import unittest

from jobq.spec import Job, SpecError, parse


class ParseTest(unittest.TestCase):
    def test_jobs_in_declared_order(self):
        jobs = parse("# comment\nb 2 a   # trailing comment\n\na 1\nc -1 a,b\n")
        self.assertEqual(jobs, [Job("b", 2, ("a",)), Job("a", 1), Job("c", -1, ("a", "b"))])

    def test_errors(self):
        cases = {
            "a 1\na 2\n": "line 2: duplicate job a",
            "a 1 b\n": "job a: unknown dependency 'b'",
            "a high\n": "line 1: bad priority 'high'",
            "a\n": "line 1: expected NAME PRIORITY [DEPS], got 'a'",
        }
        for text, message in cases.items():
            with self.subTest(text=text):
                with self.assertRaises(SpecError) as ctx:
                    parse(text)
                self.assertEqual(str(ctx.exception), message)


if __name__ == "__main__":
    unittest.main()
