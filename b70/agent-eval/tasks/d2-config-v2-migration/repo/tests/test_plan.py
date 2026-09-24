import unittest

from bkp.plan import plan


def job(**kw):
    base = {"name": "photos", "source_dir": "/srv/photos", "exclude": [], "dest": "/mnt/b", "keep": 7}
    return dict(base, **kw)


class PlanTest(unittest.TestCase):
    def test_lines(self):
        self.assertEqual(plan(job()), [
            "job photos",
            "  tar -czf '/mnt/b/photos-{date}.tar.gz' -C /srv/photos .",
            "  prune '/mnt/b/photos-*.tar.gz' keep 7",
        ])

    def test_excludes_are_quoted(self):
        lines = plan(job(exclude=["*.tmp", "cache/"]))
        self.assertIn(" --exclude='*.tmp' --exclude=cache/ -C", lines[1])

    def test_paths_with_spaces(self):
        lines = plan(job(source_dir="/srv/my photos", dest="/mnt/usb disk"))
        self.assertIn("-C '/srv/my photos' .", lines[1])
        self.assertEqual(lines[2], "  prune '/mnt/usb disk/photos-*.tar.gz' keep 7")


if __name__ == "__main__":
    unittest.main()
