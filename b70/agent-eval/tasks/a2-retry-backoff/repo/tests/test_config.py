import os
import tempfile
import unittest

from syncer.config import ConfigError, load_config


class ConfigTest(unittest.TestCase):
    def write(self, text):
        fd, path = tempfile.mkstemp(suffix=".ini")
        with os.fdopen(fd, "w") as f:
            f.write(text)
        self.addCleanup(os.remove, path)
        return path

    def test_defaults(self):
        cfg = load_config(env={})
        self.assertEqual(cfg.endpoint, "http://localhost:8080/api")
        self.assertEqual(cfg.retry.max_attempts, 4)
        self.assertEqual(cfg.retry.delay, 2.0)

    def test_file_values(self):
        path = self.write("[sync]\ntimeout = 3\n[retry]\nmax_attempts = 6\ndelay = 0.5\n")
        cfg = load_config(path, env={})
        self.assertEqual(cfg.timeout, 3.0)
        self.assertEqual(cfg.retry.max_attempts, 6)
        self.assertEqual(cfg.retry.delay, 0.5)

    def test_env_beats_file(self):
        path = self.write("[retry]\ndelay = 0.5\n")
        cfg = load_config(path, env={"SYNC_RETRY_DELAY": "7"})
        self.assertEqual(cfg.retry.delay, 7.0)

    def test_bad_number(self):
        with self.assertRaises(ConfigError):
            load_config(env={"SYNC_RETRY_MAX_ATTEMPTS": "lots"})

    def test_negative_delay_rejected(self):
        with self.assertRaises(ConfigError):
            load_config(env={"SYNC_RETRY_DELAY": "-1"})

    def test_missing_file(self):
        with self.assertRaises(ConfigError):
            load_config("/nonexistent/settings.ini", env={})


if __name__ == "__main__":
    unittest.main()
