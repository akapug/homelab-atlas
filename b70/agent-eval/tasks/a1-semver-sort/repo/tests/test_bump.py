import unittest

from semverlite import bump


class BumpTest(unittest.TestCase):
    def test_major(self):
        self.assertEqual(bump("1.4.2", "major"), "2.0.0")

    def test_minor(self):
        self.assertEqual(bump("1.4.2", "minor"), "1.5.0")

    def test_patch(self):
        self.assertEqual(bump("1.4.2", "patch"), "1.4.3")

    def test_patch_releases_prerelease(self):
        self.assertEqual(bump("1.4.2-rc.3", "patch"), "1.4.2")

    def test_unknown_part(self):
        with self.assertRaises(ValueError):
            bump("1.4.2", "build")


if __name__ == "__main__":
    unittest.main()
