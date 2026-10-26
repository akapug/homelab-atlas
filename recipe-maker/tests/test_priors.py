"""serving priors (data/priors-serving.json -> build.py serving_priors -> data.priors) and the replay facts the replay_artifact and
steady_state checks read off a real-agent ledger row (build.py replay_facts). The checks themselves are in tests/test_workload.js.
Run after python3 build.py (it reads data.json)."""
import ast
import json
import os
import unittest

from test_evidence import ROOT, evidence_ns


def build_fns(*names):
    """build.py's named top-level functions over its evidence table, without running the build"""
    ns = evidence_ns()
    tree = ast.parse(open(os.path.join(ROOT, 'build.py')).read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), 'build.py', 'exec'), ns)
    return ns


def load(*parts):
    with open(os.path.join(ROOT, *parts)) as f:
        return json.load(f)


class ServingPriors(unittest.TestCase):
    def setUp(self):
        self.fn = build_fns('serving_priors')['serving_priors']
        self.raw, self.wl = load('data', 'priors-serving.json'), load('data', 'workload.json')

    def test_built_rows_carry_their_evidence_and_the_named_expectations(self):
        rows = load('data.json')['priors']['rows']
        self.assertEqual([r['id'] for r in rows], [r['id'] for r in self.raw['rows']])
        self.assertEqual({r['tag']: r['evidence'] for r in rows}, {'MEASURED-ELSEWHERE': 'inferred', 'MEASURED-HERE': 'seen', 'INFERRED': 'inferred'})
        by = {r['id']: r for r in rows}
        self.assertEqual(by['fresh-per-output-published']['value'], [4, 11])
        self.assertEqual((by['fresh-per-output-hosted-here']['value'], by['input-per-output-hosted-here']['value']), (7.84, 382.2))
        self.assertEqual(by['trace-ideal-fresh-per-output']['value'], 4.5)
        self.assertEqual(by['client-stream-idle-timeout']['value'], 300)
        for want in ('moe-experts-touched', 'kv-restore-vs-reread', 'chunked-prefill-published', 'stage-pipelining-dejavu', 'mtp-break-even-rows', 'client-compaction-threshold'):
            self.assertIn(want, by)
        for r in rows:
            self.assertTrue(r['src'].startswith(('http', 'MEASURED', 'INFERRED')), r['id'])

    def test_a_bad_row_or_a_dangling_reference_is_a_build_error(self):
        self.fn(self.raw, self.wl['profiles'])
        good = self.raw['rows'][0]
        for bad in (dict(good, applies_when={'phase_of_moon': True}), dict(good, applies_when={}), dict(good, src=''),
                    dict(good, tag='MEASURED-BY-THEM'), dict(good, id=self.raw['rows'][1]['id'])):
            with self.assertRaises(ValueError, msg=json.dumps(bad)[:80]):
                self.fn(dict(self.raw, rows=[bad] + self.raw['rows'][1:]), self.wl['profiles'])
        with self.assertRaises(ValueError):   # a MEASURED-ELSEWHERE src that opens with MEASURED claims our own measurement
            self.fn(dict(self.raw, rows=[dict(good, src='MEASURED: by someone else')]), {})
        prof = {'p': {'fresh_reference_prior': {'v': 'no-such-prior'}}}
        with self.assertRaisesRegex(ValueError, 'fresh_reference_prior'):
            self.fn(self.raw, prof)


class ReplayFacts(unittest.TestCase):
    def setUp(self):
        self.fn = build_fns('replay_facts')['replay_facts']
        self.S = load('data', 'workload.json')['settings']

    def row(self, wl=None, mw=None):
        return {'recipe': {'conditions': {'workload': wl or {}}}, 'measured': {'workload': mw or {}}}

    def test_trace_comes_from_the_recipe_then_the_stats_source_then_the_measured_trace(self):
        f = self.fn(self.S, self.row({'trace_stats_src': 'bench/agent-replay-traces.json (r 1111111) trace trace-a stats'}, {'trace': 'agent traffic, trace trace-b (3 agents)'}))
        self.assertEqual(f['trace'], 'trace-a')
        self.assertEqual(self.fn(self.S, self.row({}, {'trace': 'agent traffic, trace trace-b (3 agents)'}))['trace'], 'trace-b')
        self.assertEqual(self.fn(self.S, self.row({'trace': 'trace-c', 'trace_stats_src': 'x trace trace-a stats'}))['trace'], 'trace-c')
        self.assertIsNone(self.fn(self.S, self.row({}, {'trace': 'recorded agent traffic (9 agents)'}))['trace'])

    def test_warmup_and_breakdown_are_read_when_present_and_none_when_absent(self):
        f = self.fn(self.S, self.row({}, {'warmup_s': 90, 'fresh_breakdown': {'cold': 1}, 'cache_evictions': 4, 'out_tokens': 10}))
        self.assertEqual((f['warmup_s'], f['fresh_breakdown'], f['cache_evictions'], f['out_tokens']), (90, {'cold': 1}, 4, 10))
        self.assertEqual(self.fn(self.S, self.row({'warmup_s': 'none'}))['warmup_s'], 'none')
        f = self.fn(self.S, self.row())
        self.assertEqual((f['warmup_s'], f['fresh_breakdown']), (None, None))

    def test_every_real_agent_row_in_the_build_names_its_trace(self):
        d = load('data.json')
        ideal = d['workload']['profiles'][d['workload']['settings']['default_profile']['v']]['trace_ideal_fresh_per_output']
        rows = d['workload']['measured']
        if not rows:
            self.skipTest('no real-agent rows in this build')
        self.assertEqual([r['run'] for r in rows if r['trace'] not in ideal], [], 'a replayed trace without a trace-ideal ratio')


if __name__ == '__main__':
    unittest.main()
