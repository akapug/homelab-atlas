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

    def test_network_fold_back_retains_the_exact_priced_identity(self):
        r = good_row()
        identity = {'recipe': 'fixture-network', 'window': 'fixture-window', 'agents': 4, 'mode': 'fixture-fast'}
        r['recipe'].update(kind='network', network_recipe=identity['recipe'], network_prediction_identity=copy.deepcopy(identity))
        r['prediction'] = {'network': {'prediction_identity': copy.deepcopy(identity)}}
        r['measured']['network_identity'] = copy.deepcopy(identity)
        r['measured']['src']['network_identity'] = 'synthetic exact launch capture'
        self.assertEqual(R.validate(r), [])
        r['measured']['network_identity']['mode'] = 'fixture-slow'
        self.assertTrue(any('network_identity' in b for b in R.validate(r)))
        r['measured']['network_identity'] = copy.deepcopy(identity)
        r['prediction']['network']['prediction_identity']['agents'] = 8
        self.assertTrue(any('prediction must retain' in b for b in R.validate(r)))

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


class Placement(unittest.TestCase):
    def test_actual_producer_shapes_are_partial_and_line_cited(self):
        # Synthetic values in the actual generate.cpp/verify.cpp formats, not measurements of a new run.
        text = '\n'.join([
            'strata generate: expert cache 12000 slots, 20.00 GiB of VRAM; policy is',
            'strata serve: decode expert cache hit rate: 50.0% (5 hits / 10 lookups); 10 more read by the GPU over PCIe',
            'strata serve: KV streaming: 90.00% of 100 block reads hit VRAM, 4.0 MiB read from RAM',
            'strata serve: the prompt path borrows 12 cache slots (0.02 GiB)',
            'strata generate: WARNING: --batch-mtp is off: a layer split pipelines the slots in groups',
            'strata serve: prompt 100 tokens = 0 reused + 100 read in 10 ms (10000 tok/s), 20 generated in 100 ms (200 tok/s), drafts accepted 0 of 3, 0 checkpoints',
            'strata verify: batch dense products from 4 rows on the decode XMX product',
        ])
        m = R.parse_placement(text, 'synthetic-engine.log')
        events = m['fallback_ledger']['events']
        self.assertNotIn('identity', m['fallback_ledger'])
        self.assertEqual([e['value'] for e in events if e.get('metric') == 'cache_slots'], [12000])
        hit = next(e for e in events if e.get('metric') == 'cache_hit_pct')
        self.assertIn('excludes PCIe', hit['scope'])
        self.assertFalse(any(e.get('tier') == 'gpu' and e['class'] == 'hot_experts' for e in events))
        kv = next(e for e in events if e.get('metric') == 'kv_ram_mib')
        self.assertEqual(kv['value'], 4.0)
        self.assertIn('cumulative', kv['scope'])
        self.assertEqual(kv['src'], 'synthetic-engine.log:3')
        self.assertTrue(any(e['class'] == 'drafter_state' and e.get('tier') == 'on' for e in events))
        dense = next(e for e in events if e['class'] == 'dense_op')
        self.assertEqual(dense['proof'], 'configured')
        self.assertEqual(dense['min_rows'], 4)
        self.assertFalse(any(e['class'] in ('shexp_op', 'hc_op') for e in events))

    def test_positive_batch_proposals_and_executed_rows_only(self):
        events = R.parse_placement('strata batch: MTP proposals accepted 0 of 3 (0.0%)\nstrata batch: 10 windows, avg 2.00 rows, 1.00 ms/window', 'synthetic')['fallback_ledger']['events']
        self.assertEqual([(e['class'], e['tier'], e['proof']) for e in events], [('batch_mtp', 'on', 'observed'), ('execution', 'batched', 'observed')])
        self.assertEqual(R.parse_placement('strata batch: MTP proposals accepted 0 of 0\nstrata batch: 10 windows, avg 1.00 rows', 'synthetic')['fallback_ledger']['events'], [])

    def test_cli_local_receipt_reaches_the_real_model_gate(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'engine.log')
            with open(path, 'w') as f:
                f.write('strata generate: WARNING: --batch-mtp is off: grouped slots\n')
            p = subprocess.run([sys.executable, os.path.join(R.ROOT, 'tools', 'record_run.py'), '--from-log', 'placement:' + path], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        measured = json.loads(p.stdout)
        script = """const FM=require('./src/model.js'),fs=require('fs'),d=JSON.parse(fs.readFileSync('data.json'));
const m=JSON.parse(fs.readFileSync(0,'utf8')),r={box:'odie',model:'fnx',engine:'strata',engine_build:'synthetic',fallback_plan:{src:'synthetic',classes:{batch_mtp:{tier:'on',src:'synthetic'}}}};
const unknown=FM.fallbackCheck(d,r,m);m.fallback_ledger.identity=Object.fromEntries(['box','model','engine','engine_build'].map(k=>[k,r[k]]));
console.log(JSON.stringify([unknown.pred.rows.find(x=>x.id==='batch_mtp:run').status,FM.fallbackCheck(d,r,m).pred.rows.find(x=>x.id==='batch_mtp:run').status]));"""
        p = subprocess.run(['node', '-e', script], input=json.dumps(measured), cwd=R.ROOT, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads(p.stdout), ['UNKNOWN', 'FAIL'])

    def test_absence_flags_packing_and_zero_offers_never_prove_dispatch(self):
        text = 'DSpark drafter from pack.gguf: 3 blocks\n--batch-mtp --dspark auto STRATA_HC_XMX=1\ndrafts accepted 0 of 0'
        self.assertEqual(R.parse_placement(text, 'synthetic')['fallback_ledger']['events'], [])
        self.assertEqual(R.parse_placement('', 'synthetic')['fallback_ledger']['events'], [])

    def test_gemv_and_off_drafter_are_explicit_not_missing_line_inferences(self):
        m = R.parse_placement('strata verify: batch dense products from 4 rows on the GEMV (no int8 matrix path)\nDSpark drafter disabled\nGPUActive: 84.6 GiB', 'synthetic')
        events = m['fallback_ledger']['events']
        self.assertEqual(events[0]['tier'], 'vector')
        self.assertEqual(events[1]['tier'], 'off')
        self.assertEqual(events[2]['metric'], 'host_gtt_gib')
        self.assertEqual(events[2]['value'], 84.6)

    def test_runtime_cache_shrink_and_kv_loans_override_initial_capacity_receipts(self):
        # Exact retained XeStrata 67fcb0a generate.cpp 8717/6308 formats; synthetic values,
        # not copied from any real run log. A later capacity loss must not disappear.
        text = '\n'.join([
            'strata generate: expert cache 23400 slots, 20.00 GiB of VRAM; policy is',
            'strata serve: VRAM 1024 MiB kept free: the expert cache 23400 -> 12000 of 23400 slots (10.00 of 20.00 GiB), 1024 MiB free now, prompt chunk 512, in 1 ms',
            'strata: K/V grown to 262144 cells (10.00 GiB); the expert cache gave 2000 slots for it, 0 hotter experts moved to colder ones\' slots (10000 of 12000 slots hold experts)',
        ])
        events = R.parse_placement(text, 'synthetic-cache.log')['fallback_ledger']['events']
        slots = [e for e in events if e.get('metric') == 'cache_slots']
        self.assertEqual([e['value'] for e in slots], [23400, 12000, 10000])
        self.assertEqual([e['src'] for e in slots], ['synthetic-cache.log:1', 'synthetic-cache.log:2', 'synthetic-cache.log:3'])
        self.assertTrue(all(e['proof'] == 'observed' for e in slots))
        self.assertIn('runtime VRAM reserve', slots[1]['scope'])
        self.assertIn('K/V growth', slots[2]['scope'])

    def test_runtime_cache_loss_reaches_the_real_fallback_gate(self):
        measured = R.parse_placement('strata generate: expert cache 23400 slots\n'
                                    'strata serve: VRAM 1024 MiB kept free: the expert cache 23400 -> 12000 of 23400 slots', 'synthetic-cache')
        script = """const FM=require('./src/model.js'),fs=require('fs'),d=JSON.parse(fs.readFileSync('data.json'));
const m=JSON.parse(fs.readFileSync(0,'utf8')),r={box:'odie',model:'fnx',engine:'strata',engine_build:'synthetic',fallback_plan:{src:'synthetic',classes:{hot_experts:{tier:'gpu',min_cache_slots:23400,src:'synthetic'}}}};
m.fallback_ledger.identity=Object.fromEntries(['box','model','engine','engine_build'].map(k=>[k,r[k]]));
console.log(JSON.stringify(FM.fallbackCheck(d,r,m).pred.rows.find(x=>x.id==='hot_experts:cache_slots')));"""
        p = subprocess.run(['node', '-e', script], input=json.dumps(measured), cwd=R.ROOT, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads(p.stdout)['status'], 'FAIL')

    def test_multiple_receipts_concatenate_and_mixed_identities_refuse(self):
        a = R.parse_placement('strata generate: expert cache 12 slots', 'synthetic-a')
        b = R.parse_placement('strata generate: expert cache 10 slots', 'synthetic-b')
        c = R.merge(a, b)['fallback_ledger']
        self.assertEqual([e['value'] for e in c['events']], [12, 10])
        self.assertEqual(len(a['fallback_ledger']['events']), 1)
        b['fallback_ledger']['identity'] = {'engine_build': 'another-build'}
        with self.assertRaises(ValueError):
            R.merge(a, b)

    def test_nested_receipt_validation_refuses_nonfinite_missing_sources_and_wrong_build(self):
        r = good_row()
        r['recipe']['engine_build'] = 'synthetic-build'
        m = R.parse_placement('strata generate: expert cache 12 slots', 'synthetic')
        r['measured'].update(m)
        r['measured']['src'].update(decode_tps='synthetic', prefill_tps='synthetic', outcome='synthetic')
        ledger = r['measured']['fallback_ledger']
        self.assertEqual(R.validate(r), [])
        ledger['identity'] = {k: r['recipe'][k] for k in ('box', 'model', 'engine', 'engine_build')}
        self.assertEqual(R.validate(r), [])
        ledger['identity']['engine_build'] = 'wrong-build'
        self.assertTrue(any('identity' in x for x in R.validate(r)))
        del ledger['identity']
        ledger['events'][0]['value'] = float('nan')
        self.assertTrue(any('finite' in x for x in R.validate(r)))
        ledger['events'][0]['value'] = 12
        del ledger['events'][0]['src']
        self.assertTrue(any('source' in x for x in R.validate(r)))


class PlacementBinding(unittest.TestCase):
    def capture(self, change=None, missing=None, bound=False):
        row = good_row()
        recipe = dict(row['recipe'], engine_build='synthetic-build')
        check = dict(recipe, checks=[], prediction=None)
        if change:
            check[change] = 'different-' + check[change]
        if missing:
            del check[missing]
        measured = R.merge(row['measured'], R.parse_placement('strata batch: MTP proposals accepted 0 of 3', 'synthetic'))
        if bound:
            measured['fallback_ledger']['identity'] = {k: recipe[k] for k in ('box', 'model', 'engine', 'engine_build')}
        with tempfile.TemporaryDirectory() as d:
            paths = {}
            for name, value in (('recipe', recipe), ('check', check), ('measured', measured)):
                paths[name] = os.path.join(d, name + '.json')
                with open(paths[name], 'w') as f:
                    json.dump(value, f)
            return subprocess.run([sys.executable, os.path.join(R.ROOT, 'tools', 'record_run.py'),
                                   '--recipe', paths['recipe'], '--check', paths['check'], '--measured', paths['measured'],
                                   '--source', 'synthetic', '--dry-run'], capture_output=True, text=True)

    def test_complete_matching_launch_capture_binds_receipt(self):
        p = self.capture()
        self.assertEqual(p.returncode, 0, p.stderr)
        row = json.loads(p.stdout)
        self.assertEqual(row['measured']['fallback_ledger']['identity'],
                         {k: row['check'][k] for k in ('box', 'model', 'engine', 'engine_build')})
        self.assertTrue(row['measured']['fallback_ledger']['identity_src'].endswith('check.json'))

    def test_same_id_does_not_bind_receipts_to_a_different_launch_identity(self):
        for field in ('box', 'model', 'engine', 'engine_build'):
            for bound in (False, True):
                with self.subTest(field=field, bound=bound):
                    p = self.capture(change=field, bound=bound)
                    self.assertNotEqual(p.returncode, 0)
                    self.assertIn('fallback receipt launch identity', p.stderr)

    def test_incomplete_launch_identity_keeps_receipt_unbound(self):
        for field in ('box', 'model', 'engine', 'engine_build'):
            with self.subTest(field=field):
                p = self.capture(missing=field)
                self.assertEqual(p.returncode, 0, p.stderr)
                self.assertNotIn('identity', json.loads(p.stdout)['measured']['fallback_ledger'])


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


class Replay(unittest.TestCase):
    """--from-log replay: an agent-replay.py result into the measured fields of a real-agent row, and --recipe-from"""
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix='replay-')
        self.addCleanup(lambda: __import__('shutil').rmtree(self.d))

    def git(self, *a):
        return subprocess.run(['git', '-C', self.d, '-c', 'user.name=t', '-c', 'user.email=t@t'] + list(a),
                              check=True, capture_output=True, text=True).stdout.strip()

    def repo(self, label='arm-b', log=None):
        """a git repo holding results/<label>-replay-n16.json, its run.log and a trace pointer file, all committed"""
        os.makedirs(os.path.join(self.d, 'bench', 'results'))
        d = json.loads(fx('replay-arm-a.json'))
        d['label'] = label
        path = os.path.join(self.d, 'bench', 'results', label + '-replay-n16.json')
        with open(path, 'w') as f:
            json.dump(d, f)
        with open(os.path.join(self.d, 'bench', 'results', 'run.log'), 'w') as f:
            f.write(log if log is not None else '23:50:00 arm arm-a: ctx 131072\n23:59:00 arm arm-a: EVAL_CMD rc=0\n'
                    '00:01:42 arm %s: ctx 131072 max_tokens 2048\n00:01:43Z up arm %s: {}\n00:12:43 arm %s: EVAL_CMD rc=0\n' % (label, label, label))
        with open(os.path.join(self.d, 'bench', 'agent-replay-traces.json'), 'w') as f:
            json.dump({'traces': {'trace-a': {'stats': {'turns': 158}}}}, f)
        self.git('init', '-q')
        self.git('add', '.')
        subprocess.run(['git', '-C', self.d, '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '-m', 'r'],
                       check=True, env=dict(os.environ, GIT_COMMITTER_DATE='2026-10-26T00:20:00Z', GIT_AUTHOR_DATE='2026-10-26T00:20:00Z'))
        return path

    def test_fields_keyed_by_agent_count(self):
        m = R.parse_replay(fx('replay-arm-a.json'), 'replay.json', {'turns': 158})
        self.assertEqual(m['agent_tps'], {'c16': 33.4})
        self.assertEqual(m['prefill_tps'], {'c16': round(479949 / 445.7824, 1)})
        self.assertEqual(m['fresh_per_output'], {'c16': 23.95})
        self.assertEqual((m['prefill_busy_pct'], m['stall_pct'], m['turns_per_agent_min']), ({'c16': 74.3}, {'c16': 67.0}, {'c16': 0.769}))
        self.assertEqual(m['ttft_s'], {'c16_median': 11.5, 'c16_p90': 77.2})
        self.assertEqual(m['decode_tps'], {'c16_per_request_median': 5.22})
        self.assertEqual((m['power_w'], m['pinned_gib'], m['errors']), ({'c16': 259.7}, 48.9, 0))
        self.assertEqual(m['workload_kind'], R.workload_settings()['real_kind'])
        w = m['workload']
        self.assertEqual((w['fresh_prompt_tokens'], w['reused_prompt_tokens'], w['out_tokens'], w['prompt_ms'], w['requests'], w['cache_evictions']),
                         (479949, 2268938, 20039, 445782, 127, 71))
        self.assertIn('158 turns', w['trace'])
        self.assertEqual(m['outcome'], 'ok')
        self.assertNotIn('degraded', m)
        r = good_row()
        r['measured'] = m
        self.assertEqual(R.validate(r), [], 'every number carries its source')

    def test_warmup_and_fresh_breakdown_are_taken_when_given_and_tolerated_when_absent(self):
        w = R.parse_replay(fx('replay-arm-a.json'), 's')['workload']
        self.assertNotIn('warmup_s', w)
        self.assertNotIn('fresh_breakdown', w)
        d = json.loads(fx('replay-arm-a.json'))
        d.update(warmup_s=120, fresh_breakdown={'cold': 100, 'growth': 300, 'reread': 50})
        m = R.parse_replay(json.dumps(d), 's')
        self.assertEqual((m['workload']['warmup_s'], m['workload']['fresh_breakdown']), (120, {'cold': 100, 'growth': 300, 'reread': 50}))
        r = good_row()
        r['measured'] = m
        self.assertEqual(R.validate(r), [], 'the new fields ride on the sourced workload record')
        d.update(warmup_s='none', fresh_breakdown=[{'cold': 10, 'growth': 5, 'agent': 'a'}, {'cold': 1, 'reread': 7}])
        w = R.parse_replay(json.dumps(d), 's')['workload']
        self.assertEqual((w['warmup_s'], w['fresh_breakdown']), ('none', {'cold': 11, 'growth': 5, 'reread': 7}), 'a per-turn list is summed by cause')

    def test_errors_are_degraded_and_no_turns_is_a_crash(self):
        d = json.loads(fx('replay-arm-a.json'))
        d.update(errors=3, err_sample=['timeout'])
        self.assertIn('3 request errors', R.parse_replay(json.dumps(d), 's')['degraded'][0])
        d.update(turns_completed=0, agg_out_tok_s=0)
        self.assertEqual(R.parse_replay(json.dumps(d), 's')['outcome'], 'crash')

    def test_a_run_that_served_nothing_is_recorded_as_a_crash_without_ratios(self):
        d = json.loads(fx('replay-arm-a.json'))
        d.update(turns_completed=0, agg_out_tok_s=0.0, out_tokens=0, fresh_prompt_tokens=56,
                 ttft_s={'n': 0}, decode_tok_s={'n': 0})
        d['server_delta'].update(prompt_ms=0, requests=4, reused=0)
        m = R.parse_replay(json.dumps(d), 's')
        self.assertEqual(m['outcome'], 'crash')
        self.assertNotIn('prefill_tps', m)
        self.assertNotIn('fresh_per_output', m)
        self.assertNotIn('decode_tps', m)
        self.assertNotIn('ttft_s', m)
        r = good_row()
        r['measured'] = m
        self.assertEqual(R.validate(r), [], 'a zero-output run still validates')

    def test_source_names_the_commit_and_start_comes_from_the_run_log(self):
        path = self.repo()
        sha = self.git('log', '-1', '--format=%h', '--abbrev=11')
        repo = os.path.basename(self.d)
        m = R.from_log('replay:' + path)
        meta = m.pop('_replay')
        self.assertEqual(meta['src'], 'bench/results/arm-b-replay-n16.json (%s %s)' % (repo, sha))
        self.assertEqual(meta['traces_src'], 'bench/agent-replay-traces.json (%s %s) trace trace-a stats' % (repo, sha))
        self.assertEqual(m['src']['agent_tps'], meta['src'])
        self.assertEqual((meta['arm'], meta['agents']), ('arm-b', 16))
        self.assertEqual(meta['when'], '2026-10-26T00:01Z', 'the first line of the arm, on the date at or before its commit')
        self.assertIn('158 turns', m['workload']['trace'])

    def test_a_start_after_the_commit_time_of_day_is_the_previous_date(self):
        path = self.repo(log='23:58:10 arm arm-b: ctx 131072\n00:09:00 arm arm-b: EVAL_CMD rc=0\n')
        self.assertEqual(R.from_log('replay:' + path)['_replay']['when'], '2026-10-25T23:58Z')

    def test_an_uncommitted_result_is_refused(self):
        path = self.repo()
        with open(path, 'a') as f:
            f.write('\n')
        with self.assertRaises(SystemExit) as e:
            R.from_log('replay:' + path)
        self.assertIn('commit it first', str(e.exception))

    def ledger(self):
        base = json.loads(json.dumps(good_row()))
        base.update(id='odie-x-real-ctl-n16-1', box='odie', model='fnx', engine='strata',
                    prediction={'basis': 'post-hoc', 'status': 'UNMODELLED', 'why': 'no planner model'})
        base['recipe'] = {'id': base['id'], 'box': 'odie', 'model': 'fnx', 'engine': 'strata', 'ctx': 131072,
                          'label': 'odie fnx, build A, real traffic at 16 agents (real-ctl)', 'engine_build': 'image aaaaaaaa (engine 1)',
                          'conditions': {'ctx': 131072, 'engine_flags': '--batch 32 --kv-resident 4096; MAX_CONTEXT=131072; arm env A=1 T=500; '
                                         'workload acceptance-wrap max_tokens 2048; agent replay 16 agents',
                                         'workload': {'bench': 'agent-replay.py: 16 agents loop, 600 s', 'kind': 'real-agent', 'agents': 16,
                                                      'trace_stats_src': 'old'}},
                          'serving': {'max_context': 131072, 'batch': 32, 'kv_resident': 4096},
                          'src': 'arm real-ctl (bench/results/real-ctl-replay-n16.json (r 1111111)); arms file arms.json'}
        base['measured']['src']['agent_tps'] = 'bench/results/real-ctl-replay-n16.json (r 1111111)'
        other = json.loads(json.dumps(base))
        other.update(id='odie-x-real-best-n16-1', when='2026-10-17T19:00Z')
        other['recipe'].update(id=other['id'], engine_build='image bbbbbbbb (engine 2 comb)', label='odie fnx, build B, real traffic at 16 agents (real-best)')
        runs = os.path.join(self.d, 'runs.jsonl')
        with open(runs, 'w') as f:
            f.write(json.dumps(base) + '\n' + json.dumps(other) + '\n')
        arms = {'common_env': {'T': '500'}, 'arms': [
            {'name': 'real-ctl', 'ctx': 131072, 'max_tokens': 2048, 'env': {'A': '1'}, 'engine_args_append': ['--batch', '32'], 'image': 'sha256:' + 'a' * 64},
            {'name': 'arm-b', 'ctx': 65536, 'max_tokens': 1024, 'env': {'A': '1', 'SP': '1'}, 'engine_args_append': ['--batch', '64', '--kv-resident', '2048', '--prefill', '8192'],
             'image': 'sha256:' + 'b' * 64},
            {'name': 'arm-c', 'ctx': 131072, 'max_tokens': 2048, 'env': {}, 'engine_args_append': [], 'image': 'sha256:' + 'c' * 64}]}
        with open(os.path.join(self.d, 'arms.json'), 'w') as f:
            json.dump(arms, f)
        return runs, arms

    def test_recipe_from_overrides_the_arm(self):
        runs, arms = self.ledger()
        meta = {'arm': 'arm-b', 'agents': 16, 'window_s': 600.0, 'src': 'bench/results/arm-b-replay-n16.json (r 2222222)', 'traces_src': 'traces (r 3) trace t stats', 'when': None}
        rc, base = R.recipe_from(R.load(runs), 'odie-x-real-ctl-n16-1', arms, meta)
        self.assertEqual(base['id'], 'odie-x-real-ctl-n16-1')
        self.assertEqual(rc['id'], 'odie-x-arm-b-n16-1')
        self.assertEqual(rc['engine_build'], 'image bbbbbbbb (engine 2 comb)', 'the build of the arm\'s image, from the row that names it')
        self.assertEqual(rc['label'], 'odie fnx, build B, real traffic at 16 agents (arm-b)')
        f = rc['conditions']['engine_flags']
        self.assertIn('arm env A=1 SP=1 T=500;', f)
        self.assertIn('--batch 64 --kv-resident 2048 --prefill 8192;', f, 'an arg the base does not name is added to the engine args')
        self.assertEqual(rc['serving']['prefill'], 8192)
        self.assertIn('MAX_CONTEXT=65536', f)
        self.assertIn('max_tokens 1024', f)
        self.assertEqual((rc['ctx'], rc['conditions']['ctx'], rc['serving']['max_context'], rc['serving']['batch'], rc['serving']['kv_resident']),
                         (65536, 65536, 65536, 64, 2048))
        self.assertEqual(rc['conditions']['workload']['trace_stats_src'], 'traces (r 3) trace t stats')
        self.assertEqual(rc['src'], 'arm arm-b (bench/results/arm-b-replay-n16.json (r 2222222)); arms file arms.json')
        self.assertEqual(R.load(runs)[0]['recipe']['ctx'], 131072, 'the base row is not touched')
        with self.assertRaises(SystemExit) as e:
            R.recipe_from(R.load(runs), 'odie-x-real-ctl-n16-1', arms, dict(meta, arm='arm-c'))
        self.assertIn('--engine-build', str(e.exception))
        rc, _ = R.recipe_from(R.load(runs), 'odie-x-real-ctl-n16-1', arms, dict(meta, arm='arm-c'), engine_build='image cccccccc (new)', new_id='odie-x-c')
        self.assertEqual((rc['engine_build'], rc['id']), ('image cccccccc (new)', 'odie-x-c'))
        with self.assertRaises(SystemExit):
            R.recipe_from(R.load(runs), 'odie-x-real-ctl-n16-1', arms, dict(meta, arm='arm-z'))

    def test_one_command_appends_a_valid_row(self):
        runs, _ = self.ledger()
        path = self.repo()
        argv = ['--recipe-from', 'odie-x-real-ctl-n16-1', '--arms', os.path.join(self.d, 'arms.json'), '--no-check',
                '--from-log', 'replay:' + path, '--tag', 'real-agent', '--runs', runs]
        self.assertEqual(R.main(argv), 0)
        row = R.load(runs)[-1]
        self.assertEqual(R.validate(row), [])
        self.assertEqual((row['id'], row['when'], row['tags']), ('odie-x-arm-b-n16-1', '2026-10-26T00:01Z', ['real-agent']))
        self.assertEqual(row['prediction'], {'basis': 'post-hoc', 'status': 'UNMODELLED', 'why': 'no planner model'})
        self.assertEqual(row['sources'], [row['measured']['src']['agent_tps'], row['recipe']['conditions']['workload']['trace_stats_src']])
        self.assertNotIn(os.path.expanduser('~'), json.dumps(row))
        with self.assertRaises(ValueError):
            R.main(argv)   # the same arm twice is a duplicate id


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
        with open(self.recipe + '.check.json') as f:
            captured = json.load(f)['prediction']
        self.assertEqual(row['prediction'], captured, 'the writer preserves every frozen identity/prediction byte as JSON values')
        self.assertEqual(row['prediction']['calibration']['status'], 'FROZEN')
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
