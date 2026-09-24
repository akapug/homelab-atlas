import contextlib
import io
import shutil
import unittest

from bkp.cli import main

from tests.helpers import EXAMPLE, job_file


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


class CliTest(unittest.TestCase):
    def test_plan(self):
        code, out, _ = run("plan", EXAMPLE)
        self.assertEqual(code, 0)
        self.assertEqual(out.splitlines()[0], "job photos")
        self.assertIn("--exclude='*.tmp'", out)

    def test_show(self):
        code, out, _ = run("show", EXAMPLE)
        self.assertEqual(code, 0)
        self.assertIn("/srv/photos (2 excludes)", out)
        self.assertIn("/mnt/backup/photos", out)

    def test_fmt_keeps_a_canonical_file(self):
        path = job_file(self, {})
        shutil.copyfile(EXAMPLE, path)
        self.assertEqual(run("fmt", path)[0], 0)
        with open(path) as f, open(EXAMPLE) as g:
            self.assertEqual(f.read(), g.read())

    def test_bad_file(self):
        code, _, err = run("plan", "/nonexistent/job.json")
        self.assertEqual(code, 1)
        self.assertIn("bkp:", err)


if __name__ == "__main__":
    unittest.main()
