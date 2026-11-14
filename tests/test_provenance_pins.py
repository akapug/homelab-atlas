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
        good = {'weights_sha256': 'a'*64, 'engine_commit': 'b'*40,
                'patch_sha256': 'c'*64, 'outcome_probe_receipt': {'sha256': 'd'*64}}
        for key in ['weights_sha256', 'engine_commit']:
            for bad in ['latest', 'dirty', '/path/to/weights', 'abc1234', None, '0'*64, 42]:
                seat = {'status': 'qualified', 'provenance_pins': dict(good, **{key:bad})}
                self.assertEqual(next(check.provenance_checks(seat))['status'], 'UNKNOWN')
        self.assertEqual(next(check.provenance_checks({'status':'verified','provenance_pins':good}))['status'], 'PASS')

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
