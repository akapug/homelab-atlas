"""runs_tables.draft_twins: in-load draft twins from the run ledger (the draft speed-up fit)."""
import copy
import json
import os
import tempfile
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))
import runs_tables as R  # noqa: E402


def run(rid, d, degraded=(), box='droopy', model='glm'):
    return {'id': rid, 'box': box, 'model': model, 'measured': {'outcome': 'ok', 'decode_tps': d, 'degraded': list(degraded)}}


class DraftTwins(unittest.TestCase):
    def test_aba_mean_over_no_draft(self):
        t = R.draft_twins([run('a', {'m3': 10.0, 'm1': 9.0, 'm0': 8.0, 'm3b': 12.0})])['droopy|glm']
        self.assertEqual([(x['id'], x['mult'], x['what']) for x in t], [('a', 1.375, 'MTP n_max 3')])
        self.assertEqual(t[0]['phases'], 'm3 10 / m3b 12 / m0 8')

    def test_dspark_twin_and_one_sided_runs(self):
        out = R.draft_twins([run('v', {'n4': 9.0, 'n0': 7.5}, model='v41'), run('k', {'M3_p0': 12.2}), run('z', {'m3': 9.0})])
        self.assertEqual(list(out), ['droopy|v41'])
        self.assertEqual(out['droopy|v41'][0]['mult'], 1.2)

    def test_a_note_naming_a_twin_phase_drops_it_a_load_note_does_not(self):
        out = R.draft_twins([run('p', {'m3': 9.0, 'm0': 6.0}, ['m0 prompt1 decoded 3.10 tok/s while the hub staged a package']),
                             run('q', {'m3': 9.0, 'm0': 8.0}, ['overlaps m2 prompt2', "placement off target {'N0': 9.7} GiB"])])
        self.assertEqual([x['id'] for x in out['droopy|glm']], ['q'])

    def test_ledger_rows_feed_the_glm_and_v41_fits(self):
        out = R.draft_twins(R.load_runs(os.path.join(os.path.dirname(HERE), 'runs', 'runs.jsonl')))
        self.assertGreaterEqual(len(out['droopy|glm']), 2)
        self.assertGreaterEqual(len(out['droopy|v41']), 2)
        self.assertNotIn('droopy-glm-glm-be1q3e-1215', [x['id'] for x in out['droopy|glm']], 'its m0 prompt1 ran perturbed')



class DecodeFits(unittest.TestCase):
    def test_current_replay_has_operands_but_ledger_and_saved_predictions_are_unchanged(self):
        root = os.path.dirname(HERE)
        runs = R.load_runs(os.path.join(root, 'runs', 'runs.jsonl'))
        original = copy.deepcopy(runs)
        recipes = R.recipes_from_runs(runs)
        self.assertTrue(callable(getattr(R, 'decode_fits', None)), 'the ledger-to-current-operand producer is missing')
        specs = R.decode_fits(root, runs, recipes)
        self.assertEqual(runs, original)
        spec = specs['droopy-v41-v41fs']
        self.assertEqual(len(spec['training']), 6)
        self.assertFalse(set(spec['train']) & set(spec['holdout']))
        for row in recipes:
            if row['run_id'] in spec['recipes']:
                self.assertEqual(row['recipe']['decode_profile'], 'droopy-v41-v41fs')
                self.assertAlmostEqual(sum(row['recipe']['decode_reads']['shares'].values()), 1)
        changed = copy.deepcopy(runs)
        for row in changed:
            if row['id'] in spec['holdout']:
                row['measured']['decode_tps'] = {'n0': 1000, 'n4': 2000}
        other = R.decode_fits(root, changed, R.recipes_from_runs(changed))
        self.assertEqual(specs, other, 'holdout measurements leaked into the solver fit table')
        twins = R.draft_twins(changed, exclude=spec['holdout'])
        self.assertTrue(all(x['id'] not in spec['holdout'] for xs in twins.values() for x in xs))

    def test_missing_measured_knob_envelope_is_rejected_by_the_producer(self):
        with open(os.path.join(os.path.dirname(HERE), 'data', 'decode-fits.json')) as f:
            specs = json.load(f)
        for spec in specs.values():
            spec.pop('knobs', None)
        with tempfile.TemporaryDirectory() as root:
            os.mkdir(os.path.join(root, 'data'))
            with open(os.path.join(root, 'data', 'decode-fits.json'), 'w') as f:
                json.dump(specs, f)
            runs = R.load_runs(os.path.join(os.path.dirname(HERE), 'runs', 'runs.jsonl'))
            with self.assertRaisesRegex(ValueError, 'decode fit measured knob envelope is missing'):
                R.decode_fits(root, runs, R.recipes_from_runs(runs))

    def test_excluded_twins_do_not_poison_the_old_global_fit(self):
        runs = [run('train', {'n4': 12, 'n0': 10}, model='v41'), run('holdout', {'n4': 9999, 'n0': 1}, model='v41')]
        self.assertEqual([x['id'] for x in R.draft_twins(runs, exclude=['holdout'])['droopy|v41']], ['train'])


class CostText(unittest.TestCase):
    def test_a_start_with_seconds_reads_like_one_without(self):
        r = {'id': 'x', 'measured': {'outcome': 'oom', 'wall_min': 3}}
        self.assertEqual(R.cost_text(dict(r, when='2026-10-21T00:49:22Z')), R.cost_text(dict(r, when='2026-10-21T00:49Z')))
        self.assertTrue(R.cost_text(dict(r, when='2026-10-21T00:49Z')).startswith('00:49-00:52Z'))


if __name__ == '__main__':
    unittest.main()
