import unittest

from bkp.config import ConfigError, load, save

from tests.helpers import EXAMPLE, job_file

MINIMAL = {"name": "docs", "source_dir": "/home/docs", "dest": "/backup/docs"}


class LoadTest(unittest.TestCase):
    def test_defaults(self):
        job = load(job_file(self, MINIMAL))
        self.assertEqual(job["exclude"], [])
        self.assertEqual(job["keep"], 7)

    def test_example(self):
        job = load(EXAMPLE)
        self.assertEqual(job["source_dir"], "/srv/photos")
        self.assertEqual(job["exclude"], ["*.tmp", "cache/"])
        self.assertEqual(job["keep"], 14)

    def test_missing_required_key(self):
        data = dict(MINIMAL)
        del data["dest"]
        with self.assertRaisesRegex(ConfigError, "dest"):
            load(job_file(self, data))

    def test_unknown_key(self):
        with self.assertRaisesRegex(ConfigError, "colour"):
            load(job_file(self, dict(MINIMAL, colour="red")))

    def test_keep_must_be_positive(self):
        with self.assertRaises(ConfigError):
            load(job_file(self, dict(MINIMAL, keep=0)))

    def test_unsupported_version(self):
        with self.assertRaisesRegex(ConfigError, "version"):
            load(job_file(self, dict(MINIMAL, version=2)))

    def test_not_json(self):
        path = job_file(self, {})
        with open(path, "w") as f:
            f.write("{nope")
        with self.assertRaises(ConfigError):
            load(path)


class SaveTest(unittest.TestCase):
    def test_canonical_form(self):
        path = job_file(self, MINIMAL)
        save(load(path), path)
        with open(path) as f:
            text = f.read()
        self.assertTrue(text.startswith('{\n  "dest": "/backup/docs",\n'), text)
        self.assertTrue(text.endswith('"version": 1\n}\n'), text)

    def test_round_trip(self):
        path = job_file(self, {})
        save(load(EXAMPLE), path)
        self.assertEqual(load(path), load(EXAMPLE))


if __name__ == "__main__":
    unittest.main()
