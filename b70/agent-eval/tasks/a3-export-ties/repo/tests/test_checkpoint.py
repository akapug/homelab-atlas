import unittest

from orderexp import checkpoint

from tests.helpers import checkpoint_path, stamp


class CheckpointTest(unittest.TestCase):
    def test_missing_file_means_start(self):
        self.assertIsNone(checkpoint.load(checkpoint_path(self)))

    def test_round_trip(self):
        path = checkpoint_path(self)
        checkpoint.save(path, stamp(42))
        self.assertEqual(checkpoint.load(path), stamp(42))


if __name__ == "__main__":
    unittest.main()
