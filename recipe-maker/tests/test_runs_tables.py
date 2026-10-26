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

    def test_a_confounded_run_gives_no_twin_and_no_fit_input(self):
        r = run('c', {'n4': 9.0, 'n0': 7.5}, model='v41')
        r['measured']['caveats'] = [{'kind': 'confounded', 'why': 'w'}]
        self.assertEqual(R.draft_twins([r, run('v', {'n4': 9.0, 'n0': 7.5}, model='v41')])['droopy|v41'][0]['id'], 'v')
        rec = R.recipes_from_runs([dict(r, when='2026-10-21T23:34Z', engine='ik', recipe={}, sources=[])])[0]
        self.assertEqual(rec['measured']['caveat'], ['confounded'], 'src/model.js host and hop fits skip it')
        self.assertIn('{confounded: w}', rec['outcome'])

    def test_ledger_rows_feed_the_glm_and_v41_fits(self):
        out = R.draft_twins(R.load_runs(os.path.join(os.path.dirname(HERE), 'runs', 'runs.jsonl')))
        self.assertGreaterEqual(len(out['droopy|glm']), 2)
        self.assertGreaterEqual(len(out['droopy|v41']), 2)
        self.assertNotIn('droopy-glm-glm-be1q3e-1215', [x['id'] for x in out['droopy|glm']], 'its m0 prompt1 ran perturbed')



    def test_prefixed_mtp_groups_make_one_twin(self):
        t = R.draft_twins([run('o', {'std-m3': 47.34, 'std-n0': 41.95}, box='lassie', model='orn35'),
                           run('g', {'d600_m3_mean': 31.0, 'd600_n0_mean': 27.0, 'd4000_m3_mean': 30.0, 'd4000_n0_mean': 26.0})])
        self.assertEqual(round(t['lassie|orn35'][0]['mult'], 4), round(47.34 / 41.95, 4))
        self.assertEqual(t['lassie|orn35'][0]['what'], 'MTP n_max 3')
        self.assertEqual(round(t['droopy|glm'][0]['mult'], 4), round((31 / 27 + 30 / 26) / 2, 4), 'the mean over the groups')


class Served(unittest.TestCase):
    def test_unobserved_churn_phases_are_not_averaged_or_changed(self):
        r = run('refill', {'c30_no_read_median': None, 'c30_reading_median': 6.95})
        before = copy.deepcopy(r)
        self.assertEqual(R.served(r), {'other': 6.95, 'other_phase': 'c30_reading_median', 'guess': True})
        self.assertEqual(r, before)
        self.assertIsNone(R.served(run('missing', {'n0': None, 'bad': float('nan'), 'flag': True})))
        self.assertEqual(R.served(run('stall', {'n0': 0})), {'off': 0, 'off_phase': 'n0'})
        twins = R.draft_twins([run('part', {'n4': None, 'n4b': 9, 'n0': 7.5}), run('none', {'n4': None, 'n0': 7.5})])
        self.assertEqual([x['id'] for x in twins['droopy|glm']], ['part'])
        self.assertEqual(twins['droopy|glm'][0]['mult'], 1.2)

    def test_draft_and_off_arms(self):
        self.assertEqual(R.served(run('a', {'n4': 9.0, 'n4b': 11.0, 'n0': 7.5})), {'on': 10.0, 'on_phase': 'n4/n4b', 'off': 7.5, 'off_phase': 'n0'})
        s = R.served(run('b', {'std-m3': 47.0, 'std-n0': 42.0}))
        self.assertEqual((s['on'], s['off'], s['off_phase']), (47.0, 42.0, 'std-n0'))
        self.assertEqual(R.served(run('c', {'std-n4': 11.5, 'std-n0': 42.0}))['on_phase'], 'std-n4', 'an external draft width is a draft arm too')

    def test_primary_phase_c1_and_guess(self):
        self.assertEqual(R.served(run('c', {'short_c1': 70.0, 'long_c1': 60.0})), {'other': 70.0, 'other_phase': 'short_c1'})
        self.assertEqual(R.served(run('d', {'all': 4.0}))['other'], 4.0)
        g = R.served(run('e', {'DV0_p0': 16.9, 'DV0_p1': 16.2}))
        self.assertTrue(g['guess'], 'per-prompt phases are no served decode: the page lists the run and compares nothing')
        r = run('f', {'DV0_p0': 16.9, 'x': 12.0})
        r['measured']['primary_phase'] = 'x'
        self.assertEqual(R.served(r), {'other': 12.0, 'other_phase': 'x'})
        self.assertIsNone(R.served(run('h', {})))

    def test_recipes_carry_measured(self):
        r = dict(run('a', {'n4': 9.0, 'n0': 7.0}), when='2026-10-20T10:00Z', engine='ik', recipe={'id': 'a'}, sources=['x'])
        r['measured']['top1'] = 0.97
        m = R.recipes_from_runs([r], {'a'})[0]['measured']
        self.assertEqual({k: m[k] for k in ('on', 'on_phase', 'off', 'off_phase', 'top1', 'holdout')},
                         {'on': 9.0, 'on_phase': 'n4', 'off': 7.0, 'off_phase': 'n0', 'top1': 0.97, 'holdout': True})
        self.assertIn('bad', m)


class BoxConditions(unittest.TestCase):
    C = [{'id': 'dimm', 'from': '2026-10-20T03:18Z', 'to': '2026-10-21T19:14Z'}]

    def test_a_window_holds_from_its_start_up_to_its_end(self):
        at = lambda w: R.conditions_at(self.C, w)
        self.assertEqual([at(w) for w in ('2026-10-19T23:39Z', '2026-10-20T03:18Z', '2026-10-21T04:48:54Z', '2026-10-21T19:13:59Z', '2026-10-21T19:14Z')],
                         [[], ['dimm'], ['dimm'], ['dimm'], []])
        self.assertEqual(at(''), [], 'a row with no time is under no condition')

    def test_recipes_carry_the_conditions_of_their_box_at_their_time(self):
        rows = [dict(run(rid, {'n0': 7.0}, box=b), when=w, engine='ik', recipe={'id': rid}, sources=['x'])
                for rid, b, w in (('in', 'lassie', '2026-10-21T04:48:54Z'), ('before', 'lassie', '2026-10-19T03:03Z'), ('other', 'droopy', '2026-10-21T04:48Z'))]
        out = R.recipes_from_runs(rows, (), {'lassie': {'conditions': self.C}, 'droopy': {}})
        self.assertEqual({r['id']: r['box_conditions'] for r in out}, {'in': ['dimm'], 'before': [], 'other': []})
        self.assertEqual([r['box_conditions'] for r in R.recipes_from_runs(rows)], [[], [], []], 'no boxes given: none')


class BuiltLassie(unittest.TestCase):
    """data.json as built (the suite runs python3 build.py first)"""
    @classmethod
    def setUpClass(cls):
        cls.data = json.load(open(os.path.join(os.path.dirname(HERE), 'data.json')))

    def test_the_lassie_ledger_runs_in_the_dimm_window_are_tagged(self):
        c = self.data['boxes']['lassie']['conditions'][0]
        self.assertEqual((c['from'], c['to'], c['short'], c['tag']), ('2026-10-20T03:18Z', '2026-10-21T19:14Z', '15 of 16 DIMMs', 'MEASURED'))
        tag = {r['run_id']: r['box_conditions'] for r in self.data['recipes'] if r['box'] == 'lassie'}
        for rid in ('lassie-v41-split-mm7', 'lassie-v41-q8-fs1', 'lassie-v41-q8-hs5', 'lassie-orn35-q4km-o1', 'lassie-orn35-q4km-o6'):
            self.assertEqual(tag[rid], [c['id']], rid)
        self.assertEqual(tag['lassie-dllm-flash-q4-s-cache'], [], 'run on 10-19: 16 of 16 DIMMs')
        self.assertTrue(all(r['box_conditions'] == [] for r in self.data['recipes'] if r['box'] != 'lassie'))

    def test_lassie_dram_reads_the_reseated_16_dimms(self):
        dram = self.data['boxes']['lassie']['dram']
        self.assertGreater(dram['total_gib']['v'], 1000)
        self.assertEqual(len(dram['nodes']), 8)
        self.assertTrue(all(abs(x['gib'] - 126) < 1 for x in dram['nodes']), [x['gib'] for x in dram['nodes']])
        self.assertEqual(dram['dimms']['v']['populated'], 16)


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
