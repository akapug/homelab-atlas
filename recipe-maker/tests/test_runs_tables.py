"""runs_tables.draft_twins: in-load draft twins from the run ledger (the draft speed-up fit)."""
import os
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


if __name__ == '__main__':
    unittest.main()
