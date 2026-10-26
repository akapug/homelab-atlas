"""runs the node model tests (tests/test_accuracy.js: predictRecipe, checkRecipe, the ledger, the accuracy math) inside the
unittest suite, so `python3 -m unittest discover -s tests` on fab covers src/model.js too"""
import os
import shutil
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ModelJs(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node') and os.path.exists(os.path.join(ROOT, 'tests', 'test_research_memory.js')) and os.path.exists(os.path.join(ROOT, 'recipes', 'snoopy-v41-gsq-iq2xs-strata-ramcopy.json')), 'node or private maintained research fixtures not present')
    def test_research_memory_js(self):
        for script in ('test_expert_residency.js', 'test_research_memory.js'):
            with self.subTest(script=script):
                p = subprocess.run(['node', os.path.join('tests', script)], cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(p.returncode, 0, p.stdout[-10000:] + p.stderr[-10000:])
                self.assertRegex(p.stdout, r'\d+ .* checks passed')
                print(p.stdout)

    @unittest.skipUnless(shutil.which('node') and os.path.exists(os.path.join(ROOT, 'tests', 'test_fallback.js')), 'node or private fallback integration fixture not present')
    def test_fallback_js(self):
        p = subprocess.run(['node', os.path.join('tests', 'test_fallback.js')], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout[-10000:] + p.stderr[-10000:])
        self.assertRegex(p.stdout, r'\d+ fallback checks passed')
        print(p.stdout)

    @unittest.skipUnless(shutil.which('node'), 'node not installed')
    def test_network_js(self):
        p = subprocess.run(['node', os.path.join('tests', 'test_network.js')], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout[-6000:] + p.stderr[-6000:])
        self.assertRegex(p.stdout, r'\d+ tests passed')

    @unittest.skipUnless(shutil.which('node'), 'node not installed')
    def test_accuracy_js(self):
        p = subprocess.run(['node', os.path.join('tests', 'test_accuracy.js')], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout[-3000:] + p.stderr[-3000:])
        self.assertRegex(p.stdout, r'\d+ tests passed')

    @unittest.skipUnless(shutil.which('node'), 'node not installed')
    def test_decode_fit_js(self):
        p = subprocess.run(['node', os.path.join('tests', 'test_decode_fit.js')], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout[-3000:] + p.stderr[-3000:])
        self.assertRegex(p.stdout, r'\d+ tests passed')

    @unittest.skipUnless(shutil.which('node'), 'node not installed')
    def test_advisor_js(self):
        """tests/test_advisor.js: the layer map, the route-trim proposer and the upgrade advisor"""
        p = subprocess.run(['node', os.path.join('tests', 'test_advisor.js')], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout[-3000:] + p.stderr[-3000:])
        self.assertRegex(p.stdout, r'\d+ tests passed')

    @unittest.skipUnless(shutil.which('node') and os.path.exists(os.path.join(ROOT, 'tests', 'test_screen.js')), 'node not installed, or a public copy (the screening ledger stays home)')
    def test_screen_js(self):
        """tests/test_screen.js: the verdict-ready check on the screening rows against the card conditions"""
        p = subprocess.run(['node', os.path.join('tests', 'test_screen.js')], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout[-3000:] + p.stderr[-3000:])
        self.assertRegex(p.stdout, r'\d+ tests passed')


if __name__ == '__main__':
    unittest.main()
