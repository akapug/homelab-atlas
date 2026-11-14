import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('check_atlas', Path(__file__).resolve().parents[1] / 'tools/check_atlas.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class ProvenancePins(unittest.TestCase):
    def test_qualified_unpinned_is_unknown_without_mutation(self):
        for status in ['qualified', 'verified']:
            seat = {'status': status, 'role': 'author_under_judge'}
            result, = check.provenance_checks({'models': {'toy': {'seats': [seat]}}})
            self.assertEqual(result['status'], 'UNKNOWN')
            self.assertEqual(result['admission_status'], 'unpinned_provenance')
            self.assertEqual(seat['status'], status)

    def test_value_pins_and_mutations(self):
        import hashlib
        digest = lambda name: hashlib.sha256(name.encode()).hexdigest()
        good = {'weights_sha256': digest('weights'), 'engine_commit': hashlib.sha1(b'engine').hexdigest(),
                'patch_sha256': digest('patch'), 'outcome_probe_receipt': {'sha256': digest('receipt')}}
        self.good = good
        for key in ['weights_sha256', 'engine_commit']:
            full = good[key]
            for bad in ['latest', 'dirty', '/path/to/weights', 'abc1234', None, '0'*64, '0'*40, 'f'*len(full), 42, True,
                        full + '-dirty', full[:-1], full + '0', ' ' + full, full + '\n', 'sha256:' + full, 'g' + full[1:],
                        [full], {'sha256': full}, '']:
                seat = {'status': 'qualified', 'provenance_pins': dict(good, **{key:bad})}
                result = next(check.provenance_checks(seat))
                self.assertEqual(result['status'], 'UNKNOWN', (key, bad))
                self.assertEqual(result['admission_status'], 'unpinned_provenance')
            seat = {'status': 'qualified', 'provenance_pins': {k: v for k, v in good.items() if k != key}}
            self.assertEqual(next(check.provenance_checks(seat))['status'], 'UNKNOWN', 'absent ' + key)
        self.assertEqual(next(check.provenance_checks({'status':'verified','provenance_pins':good}))['status'], 'PASS')
        # optional slots may be absent, an engine commit may be a SHA-256 object id
        minimal = {'weights_sha256': good['weights_sha256'], 'engine_commit': digest('sha256 repository')}
        self.assertEqual(next(check.provenance_checks({'status':'qualified','provenance_pins':minimal}))['status'], 'PASS')
        # but a supplied optional slot must be immutable, and an unknown slot is not a pin
        for extra in [{'patch_sha256': 'dirty'}, {'outcome_probe_receipt': 'latest'}, {'outcome_probe_receipt': {'sha256': 'abc'}}, {'weights_path': '/models/x.gguf'}]:
            self.assertEqual(next(check.provenance_checks({'status':'qualified','provenance_pins':dict(minimal, **extra)}))['status'], 'UNKNOWN', extra)
        # one digest pasted into two slots pins neither object
        pasted = dict(minimal, engine_commit=minimal['weights_sha256'].upper())
        self.assertEqual(next(check.provenance_checks({'status':'qualified','provenance_pins':pasted}))['status'], 'UNKNOWN')
        pasted = dict(minimal, patch_sha256=minimal['weights_sha256'])
        self.assertEqual(next(check.provenance_checks({'status':'qualified','provenance_pins':pasted}))['status'], 'UNKNOWN')

    def test_spelling_cannot_dodge_the_gate(self):
        for status in ['Qualified', 'VERIFIED', ' qualified ', 'verified\n']:
            result, = check.provenance_checks({'seats': [{'status': status}]})
            self.assertEqual(result['status'], 'UNKNOWN', repr(status))
            self.assertEqual(result['claimed_status'], status)
        for active in [0, None, 'false', 'no', True]:
            self.assertEqual(len(list(check.provenance_checks({'status': 'qualified', 'active': active}))), 1, repr(active))
        # pins on a parent or a sibling do not qualify a seat that carries none
        parent = {'provenance_pins': {'weights_sha256': 'ab' * 32, 'engine_commit': 'cd' * 20}, 'seats': [{'status': 'qualified'}]}
        self.assertEqual([r['status'] for r in check.provenance_checks(parent)], ['UNKNOWN'])

    def test_duplicate_keys_are_refused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'atlas.json'
            path.write_text('{"schema":"homelab-atlas/1","cards":[],"results":[],"seats":[{"status":"qualified","status":"measured"}]}')
            with self.assertRaises(ValueError):
                check.load_json(path)
            import contextlib, io
            from unittest import mock
            output = io.StringIO()
            with mock.patch.object(check, 'ROOT', directory), mock.patch.object(check.sys, 'argv', ['check_atlas.py', '--provenance']), contextlib.redirect_stdout(output):
                self.assertEqual(check.main(), 1)
            self.assertIn('duplicate JSON key', output.getvalue())

    def test_recipe_pins_are_validated_and_projected(self):
        import contextlib, io, json, tempfile
        from unittest import mock
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'atlas.json').write_text(json.dumps({'schema': 'homelab-atlas/1', 'cards': [], 'results': []}))
            recipes = root / 'recipe-maker' / 'recipes'
            recipes.mkdir(parents=True)
            (recipes / 'seat.json').write_text(json.dumps({'status': 'qualified', 'provenance_pins': {'weights_sha256': 'ab' * 32, 'engine_commit': 'cd' * 20}}))
            run = lambda: check.main()
            output = io.StringIO()
            with mock.patch.object(check, 'ROOT', directory), mock.patch.object(check.sys, 'argv', ['check_atlas.py', '--provenance']), contextlib.redirect_stdout(output):
                self.assertEqual(run(), 0)
            self.assertIn('"status": "PASS"', output.getvalue())
            (recipes / 'drift.json').write_text(json.dumps({'status': 'measured', 'provenance_pins': {'engine_commit': 'latest'}}))
            output = io.StringIO()
            with mock.patch.object(check, 'ROOT', directory), mock.patch.object(check.sys, 'argv', ['check_atlas.py', '--provenance']), contextlib.redirect_stdout(output):
                self.assertEqual(run(), 1)
            self.assertIn('drift.json', output.getvalue())

    def test_cli_unknown_exit_and_read_only_projection(self):
        import contextlib, io, json, tempfile
        from unittest import mock
        document={'schema':'homelab-atlas/1','cards':[],'results':[],
                  'seats':[{'status':'qualified','role':'author_under_judge'}]}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'atlas.json';path.write_text(json.dumps(document))
            original=path.read_bytes();output=io.StringIO()
            with mock.patch.object(check,'ROOT',directory), mock.patch.object(check.sys,'argv',['check_atlas.py','--provenance']), contextlib.redirect_stdout(output):
                self.assertEqual(check.main(),3)
            self.assertIn('unpinned_provenance',output.getvalue())
            self.assertNotIn('"status": "PASS"',output.getvalue())
            self.assertEqual(path.read_bytes(),original)

    def test_historical_measurement_is_not_serving_qualification(self):
        self.assertEqual(list(check.provenance_checks({'status':'measured'})), [])
        self.assertEqual(list(check.provenance_checks({'status':'qualified','active':False})), [])
        self.assertTrue(check.validate_schema({'schema':'homelab-atlas/1','seats':[{'provenance_pins':{'patch_sha256':'dirty'}}]}))


if __name__ == '__main__': unittest.main()
