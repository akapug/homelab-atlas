"""build.py on a fresh clone: optional live inputs absent -> snapshot used with a note; snapshot absent too -> note, the
parts built on it are dropped, the build still writes data.json. Runs build.py in a temp copy (never rewrites the repo)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COPY = ['build.py', 'src', 'data', 'runs', 'probe', 'tools', 'quant-priors.json', 'quant-recipes.json', 'QUANT-PRIORS.md', 'QUANT-RECIPES.md']


class FreshClone(unittest.TestCase):
    def build(self, drop_snapshots=(), replacements=()):
        d = tempfile.mkdtemp(prefix='fitviz-build-')
        self.addCleanup(shutil.rmtree, d)
        for f in COPY:
            src = os.path.join(ROOT, f)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(d, f), ignore=shutil.ignore_patterns('__pycache__'))
            elif os.path.exists(src):
                shutil.copy(src, d)
        for f in drop_snapshots:
            os.remove(os.path.join(d, f))
        for name, old, new in replacements:
            path = os.path.join(d, name)
            with open(path) as f:
                text = f.read()
            self.assertEqual(text.count(old), 1, 'build-default mutation must hit the real producer exactly once')
            with open(path, 'w') as f:
                f.write(text.replace(old, new))
        env = dict(os.environ, FITVIZ_DIR_PMEM_LANE=os.path.join(d, 'absent-lane'), FITVIZ_DIR_HOMELAB=os.path.join(d, 'absent-homelab'),
                   FITVIZ_CONFIG=os.path.join(d, 'absent.json'))
        p = subprocess.run([sys.executable, 'build.py'], cwd=d, env=env, capture_output=True, text=True, timeout=600)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        with open(os.path.join(d, 'data.json')) as f:
            return p.stdout, json.load(f)

    def test_snapshots_build_without_live_sources(self):
        out, data = self.build()
        self.assertIn('live source not here', out)
        self.assertIn('imatrix', data['models']['v41']['heat']['sets'])
        self.assertIsNotNone(data['alloc3'])
        self.assertIn('v40', data['models'])
        q3 = data['types']['ud_q3kxl']  # GLM UD-Q3_K_XL expert mix, runnable on mainline SYCL (odie B70s)
        self.assertAlmostEqual(q3['bpw']['v'], 3.5321, places=3)
        self.assertIn('ud_q3kxl', data['engines']['sycl']['types'])
        # XeStrata served the GSQ-RCO IQ3_XXS file on odie B70 #1 (2026-10-19): the engine lists the type the type names it for
        self.assertIn('gsq_iq3xxs', data['engines']['strata']['types'])
        self.assertIn('strata', data['types']['gsq_iq3xxs']['engines'])
        self.assertEqual(q3['kld']['glm']['tag'], 'PRIOR')

    def test_build_default_change_cannot_reauthorize_the_measured_decode_envelope(self):
        _, data = self.build(replacements=(('build.py', "'kernel2x': False", "'kernel2x': True"),))
        self.assertTrue(data['knobs']['kernel2x'])
        with open(os.path.join(ROOT, 'data', 'decode-fits.json')) as f:
            catalog = json.load(f)
        for key, spec in data['decode_fits'].items():
            self.assertIs(spec['knobs']['kernel2x'], False, 'mutable defaults must not become measured fit authority')
            self.assertEqual(spec.get('knobs'), catalog[key].get('knobs'))
        js = ("const fs=require('fs'), FM=require(process.argv[1]), d=JSON.parse(fs.readFileSync(0,'utf8'));"
              "FM.applyCal(d); const r=Object.values(d.decode_fits)[0].training[0].recipe;"
              "console.log(JSON.stringify(FM.predictRecipe(d,r)));")
        p = subprocess.run(['node', '-e', js, os.path.join(ROOT, 'src', 'model.js')], input=json.dumps(data),
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr)
        prediction = json.loads(p.stdout)
        self.assertEqual(prediction['status'], 'UNMODELLED', 'changed build physics must refuse the measured profile')
        self.assertIsNone(prediction['decode_tps'])
        self.assertIsNone(prediction['decode_tps_nodraft'])

    def test_missing_inputs_are_notes_not_crashes(self):
        # the imatrix heat sets come from their hot-expert maps (data/heat); the snapshots only feed convert-all
        out, data = self.build(drop_snapshots=('src/inputs/a5v41-expert-counts.tsv', 'src/inputs/a5-expert-counts.tsv', 'src/inputs/2026-10-08-alloc-3bpw.json',
                                               'data/heat/dsv41-flash__a5-docs-51k__ik-imatrix__2026-10-16.heatmap.json',
                                               'data/heat/dsv4-flash__a5-docs__ik-imatrix__2026-10-14.heatmap.json'))
        self.assertIn('NOT FOUND', out)
        self.assertIn('heat map dsv41-flash__a5-docs-51k__ik-imatrix__2026-10-16: NOT FOUND', out)
        self.assertNotIn('imatrix', data['models']['v41']['heat']['sets'])
        self.assertEqual(data['models']['v41']['heat']['anchor_sel'], 'coding')
        self.assertNotIn('v40', data['models'])
        self.assertIsNone(data['alloc3'])
        self.assertFalse(any(a.get('split') or a['model'] == 'v40' for a in data['anchors']))


if __name__ == '__main__':
    unittest.main()
