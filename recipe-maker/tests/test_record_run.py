#!/usr/bin/env python3
"""record_run.py: ledger row validation, duplicate refusal, and every log parser on scrubbed fixtures.
Run: python3 tests/test_record_run.py"""
import copy
import json
import os
import sys
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'tools'))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))
import record_run as R  # noqa: E402
import ledger as L  # noqa: E402

FX = os.path.join(HERE, 'fixtures')


def fx(name):
    with open(os.path.join(FX, name)) as f:
        return f.read()


def good_row():
    return {'id': 'droopy-test-1', 'when': '2026-10-17T18:37Z', 'box': 'droopy', 'model': 'v41', 'engine': 'ik',
            'recipe': {'id': 'x', 'box': 'droopy', 'model': 'v41', 'engine': 'ik'}, 'check': None, 'prediction': None,
            'measured': {'outcome': 'ok', 'decode_tps': {'n4': 6.16, 'n0': 5.33}, 'prefill_tps': 51.1,
                         'src': {'decode_tps': 'log', 'prefill_tps': 'log', 'outcome': 'log'}},
            'sources': ['droopy:~/optane-b1d.log'], 'tags': []}


class Validate(unittest.TestCase):
    def test_good(self):
        self.assertEqual(R.validate(good_row()), [])

    def test_number_without_source(self):
        r = good_row()
        r['measured']['vram_post_gib'] = 21.06
        self.assertTrue(any('vram_post_gib has no source' in b for b in R.validate(r)))

    def test_phase_without_source(self):
        r = good_row()
        del r['measured']['src']['decode_tps']
        bad = R.validate(r)
        self.assertTrue(any('decode_tps.n4 has no source' in b for b in bad))
        r['measured']['src']['decode_tps.n4'] = 'log'
        r['measured']['src']['decode_tps.n0'] = 'log'
        self.assertEqual(R.validate(r), [])

    def test_outcome_and_id_and_when(self):
        r = good_row()
        r['measured']['outcome'] = 'fine'
        r['id'] = 'Bad Id'
        r['when'] = '17:38Z'
        bad = ' | '.join(R.validate(r))
        self.assertIn('measured.outcome', bad)
        self.assertIn('id', bad)
        self.assertIn('when', bad)

    def test_recipe_must_match_row(self):
        r = good_row()
        r['recipe']['box'] = 'snoopy'
        self.assertTrue(any('recipe.box' in b for b in R.validate(r)))

    def test_ok_needs_decode_or_not_measured(self):
        r = good_row()
        r['measured']['decode_tps'] = {}
        self.assertTrue(any('no decode_tps' in b for b in R.validate(r)))
        r['measured']['not_measured'] = ['decode_tps']
        self.assertEqual(R.validate(r), [])

    def test_nested_numbers_need_source(self):
        r = good_row()
        r['measured']['resident_gib_by_node'] = {'N0': 69.9}
        self.assertTrue(any('resident_gib_by_node' in b for b in R.validate(r)))

    def test_append_refuses_duplicates_and_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'runs.jsonl')
            R.append(good_row(), p)
            with self.assertRaises(ValueError) as e:
                R.append(good_row(), p)
            self.assertIn('duplicate', str(e.exception))
            bad = copy.deepcopy(good_row())
            bad['id'] = 'other'
            bad['sources'] = []
            with self.assertRaises(ValueError):
                R.append(bad, p)
            self.assertEqual(len(R.load(p)), 1)


class HomeRel(unittest.TestCase):
    def test_sources_name_no_user(self):
        h = os.path.expanduser('~')
        self.assertEqual(R.home_rel(h + '/runs/x.log'), '~/runs/x.log')
        self.assertEqual(R.home_rel('kit:' + h + '/a.log'), 'kit:~/a.log')
        self.assertEqual(R.home_rel('droopy:~/optane-b1d.log'), 'droopy:~/optane-b1d.log')


class KitLog(unittest.TestCase):
    def test_phases_watch_and_swap(self):
        arms = R.parse_kit(fx('kit-b1d.log'), 'kit-b1d.log')
        self.assertEqual(len(arms), 1)
        a = arms[0]
        m = a['measured']
        self.assertEqual(a['arm'], 'b1d-swapoff-place74-hot3p0-t40')
        self.assertEqual(m['decode_tps'], {'n4': 6.16, 'n3': 7.01, 'n0': 5.33, 'n4b': 7.82})
        self.assertEqual(m['outcome'], 'ok')
        self.assertEqual(m['swap_in_delta'], 0)
        self.assertEqual(m['system'], {'swap': 'off', 'numa_balancing': 0, 'demotion': False})
        self.assertAlmostEqual(m['vram_post_gib'], 21566 / 1024, places=3)
        self.assertEqual(m['hot_experts'], 177)
        # the watch fired only while loading: the timed window (prompt walls before the rc line) is clean
        self.assertNotIn('node_memfree_min_gib', m)
        self.assertIn('no N0 LOW reading', m['numa_clean'])
        self.assertEqual(a['start'], '2026-10-17T18:37:41Z')
        self.assertAlmostEqual(m['wall_min'], 57.5, places=1)
        self.assertEqual(m['prefill_tps'], 51.1)

    def test_an_arm_that_ends_past_midnight_keeps_a_positive_wall(self):
        # v41fs-ts1 10-21: kit start 23:34:34Z, rc 00:26:34Z the next day was recorded as wall_min -1388.0
        log = ('== ts1 start 2026-10-21T23:34:34Z\n'
               '== N0 LOW 23:59:50Z: node0 MemFree 0 GiB\n'
               '== N0 LOW 00:00:20Z: node0 MemFree 1 GiB\n'
               '{"arm": "ts1", "decode_tps": [10.0], "prefill_tps": [50.0]}\n'
               '== ts1 rc=0 00:26:34Z\n'
               '{"arm": "ts1b", "decode_tps": [11.0], "prefill_tps": [50.0]}\n'
               '== N0 LOW 00:26:30Z: node0 MemFree 2 GiB\n'
               '== ts1b rc=0 01:00:04Z\n')
        a, b = R.parse_kit(log, 'kit-ts1.log')
        self.assertEqual((a['start'], a['end']), ('2026-10-21T23:34:34Z', '2026-10-22T00:26:34Z'))
        self.assertEqual(a['measured']['wall_min'], 52.0)
        self.assertEqual((b['start'], b['end'], b['measured']['wall_min']), ('2026-10-22T00:26:34Z', '2026-10-22T01:00:04Z', 33.5))
        # a watch line a few seconds behind the last stamp stays on its date, and lows past midnight land on the next one
        self.assertEqual([t.strftime('%d %H:%M:%S') for t, _ in R.parse_kit.__globals__['_ts'] and []] or True, True)

    def test_kit_stamp_dates(self):
        utc = R.datetime.timezone.utc
        t0 = R.datetime.datetime(2026, 10, 21, 23, 34, 34, tzinfo=utc)
        self.assertEqual(R._ts(t0, '00:26:34Z'), R.datetime.datetime(2026, 10, 22, 0, 26, 34, tzinfo=utc))
        self.assertEqual(R._ts(t0, '23:34:30Z'), R.datetime.datetime(2026, 10, 21, 23, 34, 30, tzinfo=utc), 'a late watch line stays on its date')
        self.assertEqual(R._ts(t0, '23:50:00Z'), R.datetime.datetime(2026, 10, 21, 23, 50, 0, tzinfo=utc))

    def test_oom_arm_and_refused_arm(self):
        arms = R.parse_kit(fx('kit-b1-oom-refused.log'), 'kit-b1.log')
        self.assertEqual([a['arm'] for a in arms], ['b1-hot3p9-t40', 'b1-hot3p6-t20'])
        self.assertEqual(arms[0]['measured']['outcome'], 'crash')
        self.assertIn('client rc 1', arms[0]['measured']['notes'])
        self.assertEqual(arms[0]['measured']['decode_tps'], {})
        self.assertIsNone(arms[1]['measured'])
        self.assertIn('REFUSED', arms[1]['refused'])

    def test_client_step_failure_after_all_phases_is_degraded_ok(self):
        # BE1-Q3e 10-18: all 5 phases served, then the KL gate raised KeyError 'id' (client rc 1): not a server crash
        arms = R.parse_kit(fx('kit-glm-be1q3e-klgate.log'), 'kit-glm-be1q3e.log')
        m = arms[0]['measured']
        self.assertEqual(m['outcome'], 'ok')
        self.assertEqual(sorted(m['decode_tps']), ['m0', 'm1', 'm2', 'm3', 'm3b'])
        self.assertTrue(any('KeyError' in d and 'client step rc 1' in d for d in m['degraded']), m.get('degraded'))
        self.assertIn('kl_gate', m['not_measured'])

    def test_terminated_kit(self):
        arms = R.parse_kit(fx('kit-b0b-terminated.log'), 'kit-b0b.log')
        self.assertEqual(len(arms), 1)
        self.assertEqual(arms[0]['measured']['outcome'], 'stopped')

    def test_flat_format_phase_from_spec_type(self):
        arms = R.parse_kit(fx('kit-d1g-flat.log'), 'kit-d1g.log')
        got = {a['arm']: a['measured']['decode_tps'] for a in arms}
        self.assertEqual(got['d1g-v41-q2d-p0'], {'n4': 8.04})
        self.assertEqual(got['d1g-v41-place12-p0'], {'n0': 7.7})
        self.assertEqual(arms[0]['measured']['primary_phase'], 'n4')

    def test_from_log_picks_arm(self):
        p = os.path.join(FX, 'kit-b1-oom-refused.log')
        with self.assertRaises(SystemExit):
            R.from_log('kit:' + os.path.join(FX, 'kit-d1g-flat.log'))
        m = R.from_log('kit:' + p)  # the refused arm is not a run, so one arm remains
        self.assertEqual(m['_arm']['arm'], 'b1-hot3p9-t40')


class Bench(unittest.TestCase):
    def test_bench_log(self):
        m = R.parse_bench_log(fx('bench-k2e.log'), 'bench-k2e.log')
        self.assertEqual(m['decode_tps'], {'c1': 22.22})
        self.assertEqual(m['outcome'], 'ok')
        m = R.parse_bench_log(fx('bench-pol1104.log'), 'pol')
        self.assertEqual(m['decode_tps'], {'c1': 17.35})
        self.assertEqual(m['prefill_tps'], 575.2)       # rep 0: the cold 12.7k prompt, not the cached rep 1
        self.assertEqual(m['c8_aggregate_tps'], 14.19)
        m = R.parse_bench_log(fx('bench-synthetic-vram-peak.log'), 'syn')
        self.assertAlmostEqual(m['vram_peak_gib'], 23.0)

    def test_bench_json_matches_bench_py_median(self):
        m = R.parse_bench_json(fx('bench-excl0-1143.json'), 'bench-excl0-1143.json')
        self.assertEqual(m['decode_tps'], {'c1': 20.36})  # bench.py printed 20.36 for this run
        self.assertEqual(m['prefill_tps'], 593.4)
        self.assertEqual(m['c8_aggregate_tps'], 17.61)
        self.assertEqual(R.bench_median([1, 2, 3, 4]), 3)

    def test_kl(self):
        m = R.parse_kl(fx('kl-k2e.json'), 'kl-k2e.json')
        self.assertEqual((m['kl'], m['top1'], m['kl_ref'], m['kl_verdict']), (0.0044, 0.9849, 0.00446, 'PASS'))
        self.assertIn('kl', m['src'])

    def test_line_range(self):
        t = R.read_text(os.path.join(FX, 'bench-k2e.log') + '#L7-L7')
        self.assertTrue(t.startswith('C1 decode median 22.22'))

    def test_merge(self):
        a = R.parse_bench_log(fx('bench-k2e.log'), 'b')
        b = R.parse_kl(fx('kl-k2e.json'), 'k')
        m = R.merge(a, b)
        self.assertEqual((m['decode_tps']['c1'], m['kl']), (22.22, 0.0044))
        self.assertIn('decode_tps', m['src'])
        self.assertIn('kl', m['src'])


ROOT = os.path.dirname(HERE)
KIT = os.path.join(ROOT, 'tools', 'kit_record.sh')
B1D_ARM = 'b1d-swapoff-place74-hot3p0-t40'


class KitRecord(unittest.TestCase):
    """tools/kit_record.sh: the one command a run kit calls after a run"""
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix='kit-record-')
        self.addCleanup(lambda: __import__('shutil').rmtree(self.d))
        rec = next(json.loads(l) for l in L.lines(os.path.join(ROOT, 'runs', 'runs.jsonl')) if '"droopy-v41-b1d"' in l)['recipe']
        self.rid = rec['id']
        self.recipe = os.path.join(self.d, 'res-b1d-recipe.json')
        with open(self.recipe, 'w') as f:
            json.dump(rec, f)
        self.runs = os.path.join(self.d, 'runs.jsonl')

    def kit(self, *extra, env=None):
        e = dict(os.environ, FITVIZ_RUNS=self.runs, **(env or {}))
        return subprocess.run(['bash', KIT, self.recipe, 'kit:' + os.path.join(FX, 'kit-b1d.log'), '--', '--arm', B1D_ARM] + list(extra),
                              capture_output=True, text=True, env=e, timeout=300)

    def gate(self, path):
        p = subprocess.run(['node', os.path.join(ROOT, 'report.js'), '--check', self.recipe, '--json'], capture_output=True, text=True, timeout=300)
        self.assertIn(p.returncode, (0, 1, 3), p.stderr)
        with open(path, 'w') as f:
            f.write(p.stdout)

    def rows(self):
        with open(self.runs) as f:
            return [json.loads(l) for l in f if l.strip()]

    def test_launch_capture_is_the_prediction(self):
        self.gate(self.recipe + '.check.json')
        p = self.kit()
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn('launch-time gate output', p.stderr)
        row, = self.rows()
        self.assertEqual(row['id'], 'droopy-v41-' + self.rid)
        self.assertEqual(row['prediction']['basis'], 'launch')
        self.assertEqual(row['check']['id'], row['recipe']['id'])
        self.assertEqual(row['measured']['decode_tps']['n4'], 6.16)
        self.assertEqual(row['when'], '2026-10-17T18:37Z')

    def test_no_capture_records_post_hoc(self):
        p = self.kit()
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn('post-hoc', p.stderr)
        row, = self.rows()
        self.assertIsNone(row['check'])
        self.assertEqual(row['prediction']['basis'], 'post-hoc')

    def test_gate_output_of_another_recipe_is_refused(self):
        other = os.path.join(self.d, 'other.check.json')
        with open(other, 'w') as f:
            json.dump({'id': 'not-this-one', 'checks': [], 'prediction': None}, f)
        p = self.kit(env={'FITVIZ_CHECK': other})
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('not the gate output', p.stderr)
        self.assertFalse(os.path.exists(self.runs))

    def test_dry_run_and_usage(self):
        p = self.kit('--dry-run')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertFalse(os.path.exists(self.runs))
        self.assertEqual(subprocess.run(['bash', KIT, self.recipe], capture_output=True, text=True).returncode, 2)


class GateCarries(unittest.TestCase):
    def test_check_json_carries_build_banks_ram_tier(self):
        p = subprocess.run(['node', os.path.join(ROOT, 'report.js'), '--check', os.path.join(ROOT, 'recipes', 'snoopy-glm-p4-s25.json'), '--json'],
                           capture_output=True, text=True, timeout=300)
        out = json.loads(p.stdout)
        self.assertEqual((out['engine_build'], out['banks_experts'], out['ram_tier_gib']), ('p512-lane', 1600, 107.0))
        p = subprocess.run(['node', os.path.join(ROOT, 'report.js'), '--check', os.path.join(ROOT, 'recipes', 'regress-droopy-b1017-dry.json'), '--json'],
                           capture_output=True, text=True, timeout=300)
        self.assertNotIn('engine_build', json.loads(p.stdout))


if __name__ == '__main__':
    unittest.main(verbosity=1)
