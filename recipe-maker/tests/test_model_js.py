"""runs the node model tests (tests/test_accuracy.js: predictRecipe, checkRecipe, the ledger, the accuracy math) inside the
unittest suite, so `python3 -m unittest discover -s tests` on fab covers src/model.js too"""
import os
import shutil
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ModelJs(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'node not installed')
    def test_accuracy_js(self):
        p = subprocess.run(['node', os.path.join('tests', 'test_accuracy.js')], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout[-3000:] + p.stderr[-3000:])
        self.assertRegex(p.stdout, r'\d+ tests passed')

    @unittest.skipUnless(shutil.which('node'), 'node not installed')
    def test_advisor_js(self):
        """tests/test_advisor.js: the layer map, the route-trim proposer and the upgrade advisor"""
        p = subprocess.run(['node', os.path.join('tests', 'test_advisor.js')], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout[-3000:] + p.stderr[-3000:])
        self.assertRegex(p.stdout, r'\d+ tests passed')


if __name__ == '__main__':
    unittest.main()
