"""Measured intake-only runs are recordable, never planner-admitted."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import record_run as R


class MetadataRecording(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='record-metadata-')
        self.addCleanup(self.tmp.cleanup)
        self.recipe = {'id': 'synthetic-gptoss', 'box': 'lassie',
                       'model': 'gptoss120', 'engine': 'ik', 'draft': False}
        self.measured = {'outcome': 'ok', 'decode_tps': {'n0': 12.5},
                         'src': {'decode_tps': 'synthetic fixture'}}

    def record(self, **changes):
        recipe = dict(self.recipe, **changes)
        paths = {k: os.path.join(self.tmp.name, k + '.json') for k in ('recipe', 'measured')}
        for key, value in (('recipe', recipe), ('measured', self.measured)):
            with open(paths[key], 'w') as f:
                json.dump(value, f)
        runs = os.path.join(self.tmp.name, 'runs.jsonl')
        result = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'record_run.py'),
                                 '--recipe', paths['recipe'], '--measured', paths['measured'],
                                 '--no-check', '--source', 'synthetic fixture', '--runs', runs,
                                 '--when', '2026-10-20T00:00Z'],
                                capture_output=True, text=True, timeout=300)
        return result, R.load(runs)

    def test_metadata_only_cli_records_biases_without_prediction(self):
        result, rows = self.record()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        row, = rows
        self.assertEqual(row['recipe'], self.recipe)
        self.assertEqual(row['measured'], self.measured)
        self.assertIsNone(row['check'])
        pred = row['prediction']
        self.assertEqual((pred['status'], pred['basis']), ('UNMODELLED', 'post-hoc'))
        for key in ('decode_tps', 'decode_tps_nodraft', 'prefill_tps'):
            self.assertIsNone(pred[key])
        with open(os.path.join(ROOT, 'data', 'models', 'gptoss120.json')) as f:
            metadata = json.load(f)
        self.assertEqual(pred['intake'], metadata['intake'])
        for tensor in ('expert_gate', 'expert_up', 'expert_down'):
            self.assertEqual(pred['intake']['bytes_by_class'][tensor]['f32'], 53084160)
            self.assertGreater(pred['intake']['bytes_by_class'][tensor]['mxfp4'], 0)
        self.assertEqual(R.validate(row), [])

    def test_recorded_run_remains_unmodelled_in_accuracy_report(self):
        result, rows = self.record()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = os.path.join(self.tmp.name, 'accuracy.json')
        result = subprocess.run(['node', os.path.join(ROOT, 'report.js'), '--accuracy',
                                 os.path.join(self.tmp.name, 'runs.jsonl'), '--json', report],
                                capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with open(report) as f:
            accuracy = json.load(f)
        self.assertEqual(accuracy['runs'][0]['status'], 'UNMODELLED')
        self.assertEqual(accuracy['unmodelled'][0]['id'], rows[0]['id'])
        self.assertTrue(all(not group for group in accuracy['summary'].values()))

    def test_second_pinned_intake_is_recordable(self):
        result, rows = self.record(model='gptoss20')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(rows[0]['prediction']['status'], 'UNMODELLED')

    def test_unknown_inputs_still_refuse_without_a_row(self):
        for changes in ({'model': 'not-in-catalog'}, {'model': '../gptoss120'},
                        {'box': 'not-in-catalog'}, {'engine': 'not-in-catalog'}):
            with self.subTest(changes=changes):
                result, rows = self.record(**changes)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(rows, [])

    def test_prelaunch_gate_and_solver_admission_are_unchanged(self):
        with open(os.path.join(ROOT, 'data.json')) as f:
            data = json.load(f)
        self.assertNotIn('gptoss120', data['models'])
        result = subprocess.run(['node', os.path.join(ROOT, 'report.js'), '--check', '/dev/stdin', '--json'],
                                input=json.dumps(self.recipe), capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 2, result.stderr)
        gate = json.loads(result.stdout)
        self.assertIsNone(gate['prediction'])
        self.assertEqual(gate['checks'][0]['status'], 'FAIL')

    def test_existing_model_still_uses_the_actual_planner(self):
        with open(os.path.join(ROOT, 'recipes', 'droopy-b1b-hot2p2-t20.json')) as f:
            recipe = json.load(f)
        predicted = R.predict(recipe)['prediction']
        self.assertEqual(predicted['status'], 'PREDICTED')
        self.assertGreater(predicted['decode_tps'], 0)
        self.assertNotIn('intake', predicted)


if __name__ == '__main__':
    unittest.main()
