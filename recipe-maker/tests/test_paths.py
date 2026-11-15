"""Summit paths (data/paths/, src/paths.py): the seeded targets load and every lever they name is in the catalog; one planted
violation per check refuses; heads sit in the goal's agent band; never-composed pairs skip a second engine or quant and drop
a pair once a node composes it. The chart and text side is tests/test_paths.js. Run: python3 tests/test_paths.py"""
import copy
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import paths as P  # noqa: E402

CATALOG = P.load_catalog(os.path.join(ROOT, 'data', 'paths', 'levers.json'))
SRC = 'lab/README.md:12'


def val(v, tag='MEASURED', src=SRC):
    return {'v': v, 'tag': tag, 'src': src}


def point(n, per, agg, ttft=None):
    return {'agents': n, 'per_agent': per, 'aggregate': agg, 'ttft_s': ttft or val(None, 'UNKNOWN', 'not measured')}


def node(id_, levers, points, parents=(), state='open', why='', recipe='none: a test node'):
    return {'id': id_, 'label': id_, 'parents': list(parents), 'levers': list(levers), 'recipe': recipe, 'results': 'lab/README.md',
            'points': points, 'verdict': {'state': state, 'why': why}}


def target(nodes, goal=None, **kw):
    d = {'id': 't', 'label': 'test target', 'model': 'm', 'metric': 'decode tok/s', 'nodes': nodes,
         'goal': goal or {'agents': [4, 8], 'per_agent_min': 15, 'src': SRC}}
    d.update(kw)
    return d


BASE = ['engine-xestrata', 'quant-ud-q3kxl', 'experts-host-ram']


def good():
    return target([node('a', BASE, [point(1, val(20), val(20)), point(4, val(10), val(40))], state='kept'),
                   node('b', BASE + ['mtp'], [point(8, val([16, 23], 'PREDICTED'), val([130, 185], 'PREDICTED'))], parents=['a'])])


class Load(unittest.TestCase):
    def load(self, d, ledger=(), name=None):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, 'data', 'paths'))
            f = os.path.join(tmp, 'data', 'paths', (name or d['id']) + '.json')
            with open(f, 'w') as fh:
                json.dump(d, fh)
            return P.load_target(f, copy.deepcopy(CATALOG), set(ledger), tmp)

    def refuses(self, d, pattern, **kw):
        with self.assertRaisesRegex(ValueError, pattern):
            self.load(d, **kw)

    def test_clean_target_loads_with_edges_heads_and_pairs(self):
        d = self.load(good())
        self.assertEqual(d['edges'], [['a', 'b']])
        self.assertEqual([nd['head'] for nd in d['nodes']], [4, 8])
        self.assertEqual(d['file'], os.path.join('data', 'paths', 't.json'))

    def test_each_planted_violation_refuses(self):
        cases = [
            (lambda d: d['nodes'][0]['levers'].append('no-such-lever'), 'not in data/paths/levers.json'),
            (lambda d: d['nodes'][0]['levers'].append('engine-glmlite'), 'exclusive kind engine'),
            (lambda d: d['nodes'][0]['levers'].append('quant-exl3-3bpw'), 'exclusive kind quant'),
            (lambda d: d['nodes'].reverse(), 'parent a is not an earlier node'),
            (lambda d: d['nodes'][1].update(id='a'), 'unique id'),
            (lambda d: d['nodes'][0]['points'][0]['per_agent'].update(src='lab/README.md'), 'path:line'),
            (lambda d: d['nodes'][0]['points'][0]['aggregate'].update(tag='GUESS'), 'tag one of'),
            (lambda d: d['nodes'][0]['points'][0]['aggregate'].update(v=[5, 2]), r'\[low, high\]'),
            (lambda d: d['nodes'][0]['points'][0].update(ttft_s={'v': 3, 'tag': 'UNKNOWN', 'src': 'x'}), 'UNKNOWN value has v null'),
            (lambda d: d['nodes'][0]['points'][0].pop('ttft_s'), 'a value is'),
            (lambda d: d['nodes'][0]['points'][0].update(agents=0), 'agent count'),
            (lambda d: d['nodes'][0].update(points=[]), 'at least one point'),
            (lambda d: d['nodes'][0]['verdict'].update(state='dropped'), 'dropped node says why'),
            (lambda d: d['nodes'][0]['verdict'].update(state='maybe'), 'verdict.state'),
            (lambda d: d['nodes'][0].update(recipe='odie-made-up-run'), 'neither a ledger run id'),
            (lambda d: d['nodes'][0].update(results=''), 'results'),
            (lambda d: d['goal'].pop('src'), 'goal needs agents'),
            (lambda d: d.update(candidates=['no-such-lever']), 'candidates not in'),
            (lambda d: d['nodes'][0].update(head=2), 'head 2 is not one of its agent counts'),
        ]
        for plant, pattern in cases:
            d = good()
            plant(d)
            with self.subTest(pattern=pattern):
                self.refuses(d, pattern)
        self.refuses(good(), 'named after its id', name='other')

    def test_recipe_resolves_as_ledger_id_or_repo_file(self):
        d = good()
        d['nodes'][0]['recipe'] = 'odie-real-run'
        self.assertEqual(self.load(d, ledger={'odie-real-run'})['nodes'][0]['recipe'], 'odie-real-run')
        d['nodes'][0]['recipe'] = 'data/paths/t.json'
        self.load(d)

    def test_dropped_node_with_why_loads(self):
        d = good()
        d['nodes'][0]['verdict'] = {'state': 'dropped', 'why': 'slower'}
        self.assertEqual(self.load(d)['nodes'][0]['verdict']['state'], 'dropped')


class Heads(unittest.TestCase):
    def test_head_is_the_best_total_inside_the_band_else_the_best_total(self):
        nd = node('a', BASE, [point(1, val(30), val(30)), point(4, val(9), val(36)), point(8, val(4), val(32)), point(16, val(3), val(48))])
        self.assertEqual(P.head_of(nd, {'agents': [4, 8]}), 4)
        self.assertEqual(P.head_of(nd, {'agents': [2, 3]}), 16, 'no point in the band: the best total anywhere')
        self.assertEqual(P.head_of(nd, {}), 16)
        nd['head'] = 1
        self.assertEqual(P.head_of(nd, {'agents': [4, 8]}), 1, 'a declared head wins')

    def test_a_range_counts_by_its_middle_and_an_unknown_total_is_never_a_head(self):
        nd = node('a', BASE, [point(4, val([5, 7], 'PREDICTED'), val([20, 60], 'PREDICTED')), point(8, val(5), val(39))])
        self.assertEqual(P.head_of(nd, {'agents': [4, 8]}), 4)
        nd = node('a', BASE, [point(4, val(None, 'UNKNOWN', 'x'), val(None, 'UNKNOWN', 'x'))])
        self.assertIsNone(P.head_of(nd, {'agents': [4, 8]}))


class Pairs(unittest.TestCase):
    def test_pairs_skip_exclusive_kinds_and_drop_composed_pairs(self):
        nodes = [node('a', ['engine-xestrata', 'quant-ud-q3kxl', 'mtp'], []), node('b', ['engine-glmlite', 'quant-exl3-3bpw', 'route-trim'], [])]
        pairs = {(p['a'], p['b']) for p in P.never_composed(nodes, CATALOG)}
        flat = {frozenset(p) for p in pairs}
        self.assertNotIn(frozenset(('engine-xestrata', 'engine-glmlite')), flat, 'two engines never pair')
        self.assertNotIn(frozenset(('quant-ud-q3kxl', 'quant-exl3-3bpw')), flat, 'two quants never pair')
        self.assertNotIn(frozenset(('engine-xestrata', 'mtp')), flat, 'composed in node a')
        self.assertIn(frozenset(('mtp', 'route-trim')), flat)
        self.assertIn(frozenset(('engine-xestrata', 'route-trim')), flat)

    def test_a_candidate_joins_the_pool_marked_untried(self):
        nodes = [node('a', ['engine-xestrata', 'mtp'], [])]
        out = P.never_composed(nodes, CATALOG, ['ced'])
        row = [p for p in out if {p['a'], p['b']} == {'ced', 'mtp'}][0]
        self.assertEqual(dict(zip((row['a'], row['b']), row['tried'])), {'ced': False, 'mtp': True})


class Seeded(unittest.TestCase):
    """the three seeded targets of 11-15 against the real catalog (ledger ids not checked here: build.py does with RUNS)"""

    @classmethod
    def setUpClass(cls):
        d = os.path.join(ROOT, 'data', 'paths')
        cls.t = {}
        for f in sorted(os.listdir(d)):
            if f.endswith('.json') and f != 'levers.json':
                with open(os.path.join(d, f)) as fh:
                    raw = json.load(fh)
                ids = {nd['recipe'] for nd in raw['nodes']}
                cls.t[raw['id']] = P.load_target(os.path.join(d, f), copy.deepcopy(CATALOG), ids, ROOT)

    def test_three_targets_with_their_lineages(self):
        self.assertEqual(sorted(self.t), ['ds41-one-agent', 'glm53-batched', 'qfn-odie-agents'])
        self.assertIn(['d2', 'd3'], self.t['glm53-batched']['edges'])
        self.assertIn(['door32', 'conv-door'], self.t['qfn-odie-agents']['edges'])
        self.assertIn(['cpu-control', 'ced'], self.t['ds41-one-agent']['edges'])
        self.assertIn(['ced', 'snoopy-ced'], self.t['ds41-one-agent']['edges'])

    def test_glm_d3_is_predicted_open_and_its_head_is_8_agents_in_the_goal(self):
        d3 = [nd for nd in self.t['glm53-batched']['nodes'] if nd['id'] == 'd3'][0]
        self.assertEqual((d3['verdict']['state'], d3['head']), ('open', 8))
        p = [x for x in d3['points'] if x['agents'] == 8][0]
        self.assertEqual((p['per_agent']['tag'], p['per_agent']['v'], p['aggregate']['v']), ('PREDICTED', [16, 23], [130, 185]))

    def test_seeded_measured_heads_quote_the_record(self):
        head = {(t, nd['id']): [p for p in nd['points'] if p['agents'] == nd['head']][0] for t, d in self.t.items() for nd in d['nodes']}
        self.assertEqual(head[('glm53-batched', 'glmlite-b70-tier')]['aggregate']['v'], 39.99)
        self.assertEqual(head[('glm53-batched', 'odie-sycl')]['per_agent']['v'], 13.07)
        self.assertEqual(head[('qfn-odie-agents', 'door32')]['aggregate']['v'], 66.82)
        self.assertEqual(head[('qfn-odie-agents', 'conv-door')]['aggregate']['v'], 62.05)
        g = [nd for nd in self.t['qfn-odie-agents']['nodes'] if nd['id'] == 'sycl-tensor-graphs'][0]
        self.assertEqual([p['aggregate']['v'] for p in g['points']], [38.96, 15.82, 23.82, 33.04], 'graphs: B=1 only')
        self.assertEqual(head[('ds41-one-agent', 'snoopy-iq2xs')]['per_agent']['v'], 27.4)
        ced = [nd for nd in self.t['ds41-one-agent']['nodes'] if nd['id'] == 'snoopy-ced'][0]['points'][0]
        self.assertEqual((ced['ttft_s']['v'], ced['ttft_s']['tag']), (19.88, 'MEASURED'))

    def test_no_public_copy_hazard_in_the_seeded_files(self):
        # what the export scan refuses (task ids, short internal ids, our scratch dir) never enters a path file
        import re
        # (a pointer into our lab notes, homelab b70-rig/... or homelab docs/..., becomes 'lab notes' in the copy: strip it first)
        bad = re.compile(r'task/\d+|\bhl[/-]?\d+|~/scratch|/home/|\bR8\b')
        ptr = re.compile(r'\bhomelab (?:b70-rig|scripts|docs)/[\w./-]+')
        for f in os.listdir(os.path.join(ROOT, 'data', 'paths')):
            with open(os.path.join(ROOT, 'data', 'paths', f)) as fh:
                text = ptr.sub('lab notes', fh.read())
            self.assertIsNone(bad.search(text), '%s: %s' % (f, bad.search(text) and bad.search(text).group(0)))


if __name__ == '__main__':
    unittest.main()
