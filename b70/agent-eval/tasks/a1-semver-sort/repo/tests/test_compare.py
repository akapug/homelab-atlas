import unittest

from semverlite import InvalidVersion, compare, max_version, sort_versions


class CompareTest(unittest.TestCase):
    def test_equal(self):
        self.assertEqual(compare("1.2.3", "1.2.3"), 0)

    def test_patch_order(self):
        self.assertEqual(compare("1.2.3", "1.2.4"), -1)
        self.assertEqual(compare("1.2.4", "1.2.3"), 1)

    def test_v_prefix_ignored(self):
        self.assertEqual(compare("v1.2.3", "1.2.3"), 0)

    def test_prerelease_before_release(self):
        self.assertEqual(compare("2.0.0-rc.1", "2.0.0"), -1)

    def test_prerelease_numeric_identifiers(self):
        self.assertEqual(compare("1.0.0-alpha.2", "1.0.0-alpha.10"), -1)

    def test_sort_release_train(self):
        tags = ["1.9.0", "1.10.0", "1.2.0", "1.10.0-rc.1"]
        self.assertEqual(sort_versions(tags), ["1.2.0", "1.9.0", "1.10.0-rc.1", "1.10.0"])

    def test_max_version(self):
        self.assertEqual(max_version(["0.3.1", "0.4.0", "0.2.5"]), "0.4.0")

    def test_rejects_garbage(self):
        with self.assertRaises(InvalidVersion):
            compare("1.2", "1.2.0")


if __name__ == "__main__":
    unittest.main()
