"""build.py on a fresh clone: optional live inputs absent -> snapshot used with a note; snapshot absent too -> note, the
parts built on it are dropped, the build still writes data.json. Runs build.py in a temp copy (never rewrites the repo)."""
import json
import hashlib
import copy
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COPY = ['build.py', 'report.js', 'src', 'data', 'runs', 'probe', 'tools', 'quant-priors.json', 'quant-recipes.json', 'QUANT-PRIORS.md', 'QUANT-RECIPES.md']


class FreshClone(unittest.TestCase):
    def build(self, drop_snapshots=(), replacements=()):
        d = tempfile.mkdtemp(prefix='fitviz-build-')
        self.addCleanup(shutil.rmtree, d)
        self.directory = d
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

    def test_built_identity_is_reproducible_sensitive_and_names_actual_splits(self):
        _, data = self.build()
        _, again = self.build()
        identity = data['meta']['calibration']
        self.assertEqual(identity['status'], 'FROZEN', identity)
        self.assertEqual(identity, again['meta']['calibration'], 'fresh-copy mtimes and wall time are not experiment identity')
        excluded = [('meta', 'built'), ('meta', 'calibration'), ('upgrade', 'advice', 'built'),
                    ('siblings', 'quant_priors', 'mtime'), ('siblings', 'quant_recipes', 'mtime')]
        self.assertEqual(identity['excluded_paths'], [list(p) for p in excluded])

        def hashed(value):
            core = copy.deepcopy(value)
            for path in excluded:
                parent = core
                for key in path[:-1]:
                    parent = parent.get(key) or {}
                parent.pop(path[-1], None)
            return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

        self.assertEqual(identity['data_sha256'], hashed(data))
        clock = copy.deepcopy(data)
        clock['meta']['built'] = '2099-01-01T00:00Z'
        clock['upgrade']['advice']['built'] = '2099-01-01T00:00Z'
        clock['siblings']['quant_priors']['mtime'] = '2099-01-01T00:00Z'
        self.assertEqual(hashed(clock), identity['data_sha256'])
        changed = copy.deepcopy(data)
        changed['knobs']['kernel2x'] = not changed['knobs']['kernel2x']
        self.assertNotEqual(hashed(changed), identity['data_sha256'], 'a semantic operand cannot be excluded')
        split = identity['split']
        for key, spec in data['decode_fits'].items():
            self.assertEqual(split['decode'][key]['train'], sorted(spec['train']))
            self.assertEqual(split['decode'][key]['holdout'], sorted(spec['holdout']))
            self.assertFalse(set(spec['train']) & set(spec['holdout']))
        self.assertEqual(split['dllm']['holdout'], sorted(data['dllm']['holdout']['v']))
        self.assertFalse(set(split['dllm']['holdout']) & set(split['dllm']['efficiency_train']))
        for key, cfg in data['window']['configs'].items():
            self.assertEqual(split['window'][key]['check'], sorted(p['id'] for p in cfg['points'] if p['use'] == 'check'))
        original_split = json.dumps(split, sort_keys=True, separators=(',', ':')).encode()
        self.assertEqual(identity['split_sha256'], hashlib.sha256(original_split).hexdigest())
        split['decode'].clear()
        self.assertNotEqual(identity['split_sha256'], hashlib.sha256(json.dumps(split, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
        self.assertIn('src/ledger.js', identity['evaluator_sha256'], 'the accuracy evaluator owns corrections and confounded eligibility')
        for relative, expected in identity['evaluator_sha256'].items():
            with open(os.path.join(ROOT, relative), 'rb') as f:
                self.assertEqual(expected, hashlib.sha256(f.read()).hexdigest())
        _, changed_evaluator = self.build(replacements=(('src/ledger.js', "c.kind === 'confounded'", "c.kind === 'confounded' && false"),))
        self.assertNotEqual(identity['evaluator_sha256']['src/ledger.js'], changed_evaluator['meta']['calibration']['evaluator_sha256']['src/ledger.js'],
                            'changing outcome qualification must change evaluator identity even before any new cohort is measured')

    def prospective_cli(self):
        _, data = self.build()
        d = self.directory
        with open(os.path.join(ROOT, 'recipes', 'droopy-b1b-hot2p2-t20.json')) as f:
            base = json.load(f)
        cohort = {
            'id': 'synthetic-current-byte-control', 'declared_at': '2026-10-24T00:00:00Z',
            'frozen': data['meta']['calibration'],
            'workload': dict(model_sha256='a' * 64, prompt_set_sha256='b' * 64,
                             quality_reference_sha256='c' * 64, quality_prompt_set_sha256='d' * 64,
                             context_tokens=1024, agents=1, sampling={'temperature': 0}),
            'isolation': dict(id='synthetic-exclusive', scope='exclusive', src='synthetic fixture'),
            'objective': dict(mode='single-seat', measured='decode_tps.c1', predicted='decode_tps'),
            'constraints': [dict(id='quality', kind='quality', unit='KL/token', measured='kl', predicted='quality.kld', op='<=', limit=100),
                            dict(id='memory', kind='memory', unit='GiB', measured='vram_post_gib', predicted='vram_post_gib', op='<=', limit=100)],
            'candidates': [dict(id=key, run_id='synthetic-byte-' + key,
                                recipe=dict(base, id='synthetic-recipe-' + key, banks_gib=2.24 + i * 0.01))
                           for i, key in enumerate(('a', 'b'))],
        }
        recipe = dict(cohort['candidates'][0]['recipe'], validation=dict(candidate='a', cohort=cohort))
        file = os.path.join(d, 'synthetic-recipe.json')

        def check(value):
            with open(file, 'w') as f:
                json.dump(value, f)
            p = subprocess.run(['node', os.path.join(d, 'report.js'), '--check', file, '--json'],
                               capture_output=True, text=True, timeout=60)
            self.assertIn(p.returncode, (0, 1, 2, 3), p.stderr)
            return p.returncode, json.loads(p.stdout)

        return data, d, cohort, recipe, check

    def test_prospective_cli_checks_current_evaluator_bytes_not_matching_declarations(self):
        _, d, cohort, recipe, check = self.prospective_cli()
        status, captured = check(recipe)
        self.assertNotEqual(status, 2, captured)
        self.assertIn('validation', captured['prediction'])
        ordinary = cohort['candidates'][0]['recipe']
        baseline_status, baseline = check(ordinary)
        model = os.path.join(d, 'src', 'model.js')
        with open(model) as f:
            original = f.read()
        operand = 'out.decode_tps = dm / T;'
        self.assertEqual(original.count(operand), 1)
        with open(model, 'w') as f:
            f.write(original.replace(operand, 'out.decode_tps = 2 * dm / T;'))
        status, drifted = check(ordinary)
        self.assertEqual(status, baseline_status, 'ordinary checks retain their existing semantics')
        self.assertAlmostEqual(drifted['prediction']['decode_tps'], 2 * baseline['prediction']['decode_tps'])
        self.assertEqual(drifted['prediction']['calibration'], captured['prediction']['calibration'],
                         'matching stale declarations do not prove current source identity')
        status, stale = check(recipe)
        self.assertEqual(status, 2)
        self.assertNotIn('validation', stale['prediction'])
        failure = next(c for c in stale['checks'] if c['id'] == 'validation')
        self.assertEqual(failure['status'], 'FAIL')
        self.assertIn('stale evaluator src/model.js', failure['detail'])
        self.assertIn('python3 build.py', failure['detail'])
        with open(model, 'w') as f:
            f.write(original)
        os.remove(os.path.join(d, 'tools', 'record_run.py'))
        status, missing = check(recipe)
        self.assertEqual(status, 2)
        self.assertNotIn('validation', missing['prediction'])
        failure = next(c for c in missing['checks'] if c['id'] == 'validation')
        self.assertIn('missing or invalid evaluator tools/record_run.py', failure['detail'])
        self.assertIn('python3 build.py', failure['detail'])
        status, unchanged = check(ordinary)
        self.assertEqual(status, baseline_status)
        self.assertEqual(unchanged['prediction']['decode_tps'], baseline['prediction']['decode_tps'])
        self.assertFalse(any(c['id'] == 'validation' for c in unchanged['checks']))

    def test_prospective_cli_checks_effective_data_before_apply_cal(self):
        data, d, cohort, recipe, check = self.prospective_cli()
        status, captured = check(recipe)
        self.assertNotEqual(status, 2, captured)
        self.assertIn('validation', captured['prediction'])
        file = os.path.join(d, 'data.json')
        data['meta']['built'] = '2099-01-01T00:00Z'
        data['upgrade']['advice']['built'] = '2099-01-01T00:00Z'
        data['siblings']['quant_priors']['mtime'] = '2099-01-01T00:00Z'
        data['siblings']['quant_recipes']['mtime'] = '2099-01-01T00:00Z'
        with open(file, 'w') as f:
            json.dump(data, f, sort_keys=True, indent=1)
        status, clocks = check(recipe)
        self.assertNotEqual(status, 2, clocks)
        self.assertIn('validation', clocks['prediction'], 'documented nonsemantic paths and JSON formatting do not change identity')
        ordinary = cohort['candidates'][0]['recipe']
        baseline_status, baseline = check(ordinary)
        data['knobs']['kernel2x'] = not data['knobs']['kernel2x']
        with open(file, 'w') as f:
            json.dump(data, f)
        status, drifted = check(ordinary)
        self.assertEqual(status, baseline_status)
        self.assertNotEqual(drifted['prediction']['decode_tps'], baseline['prediction']['decode_tps'],
                            'the unchanged FROZEN declaration now attributes a different actual prediction')
        self.assertEqual(drifted['prediction']['calibration'], captured['prediction']['calibration'])
        status, stale = check(recipe)
        self.assertEqual(status, 2, 'semantic-data drift must fail prospective capture, not retain the old FROZEN attribution')
        self.assertNotIn('validation', stale['prediction'])
        failure = next(c for c in stale['checks'] if c['id'] == 'validation')
        self.assertEqual(failure['status'], 'FAIL')
        self.assertIn('stale effective data', failure['detail'])
        self.assertIn('python3 build.py', failure['detail'])

    def test_hash_only_mode_uses_python_numbers_without_build_side_effects(self):
        d = tempfile.mkdtemp(prefix='fitviz-hash-only-')
        self.addCleanup(shutil.rmtree, d)
        shutil.copy(os.path.join(ROOT, 'build.py'), d)
        raw = '{"float":1.0,"small":1e-7,"large":1e20,"label":"numeric identity","meta":{"built":"clock","calibration":{"ignored":true}}}'
        p = subprocess.run([sys.executable, '-I', os.path.join(d, 'build.py'), '--calibration-data-sha256'],
                           input=raw, capture_output=True, text=True, timeout=60, cwd=d)
        self.assertEqual(p.returncode, 0, p.stderr)
        expected = dict(float=1.0, small=1e-7, large=1e20, label='numeric identity', meta={})
        self.assertEqual(p.stdout.strip(), hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest())
        self.assertEqual(os.listdir(d), ['build.py'], 'hash-only mode cannot import local inputs, build or write outputs')

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
