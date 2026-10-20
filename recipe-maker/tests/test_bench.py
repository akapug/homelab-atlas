"""bench terms (build.py gpu_lines / numa_gains, tools/record_bench.py): the per-engine GPU line (reach, host ms), the
per-token MoE overhead per MoE layer, borrowed backends tagged ESTIMATE, the NUMA placement gain over one local node
(confounded when rounds disagree), the bytes one decode token reads from a GGUF header, and the built data.json citing
its bench rows. External rows re-score these terms in tests/test_accuracy.js; nothing here reads them."""
import importlib.util
import json
import os
import unittest

from test_build_inputs import ROOT
from test_evidence import evidence_ns

START, END = '# ---------------- bench terms', 'BENCH_ROWS = '


def bench_ns():
    """build.py's bench table and fit functions, without running the build"""
    ns = evidence_ns()
    src = open(os.path.join(ROOT, 'build.py')).read()
    exec(compile(src[src.index(START):src.index(END)], 'build.py', 'exec'), ns)
    return ns


def gpu_row(i, kind, gb, tps, layers=0, backend='cuda', box='bx', gid='g1'):
    return {'id': 'r%d' % i, 'kind': 'gpu_decode', 'rc': 0, 'backend': backend, 'box': box, 'gpu_id': gid, 'model_kind': kind,
            'read_gb': gb, 'decode_tps': tps, 'moe_layers': layers}


BOXES = {'bx': {'gpus': [{'id': 'g1', 'bw': {'v': 500}, 'vendor': 'nvidia'}]}}


class GpuLines(unittest.TestCase):
    def setUp(self):
        self.ns = bench_ns()
        self.B = self.ns['bench']

    def test_line_recovers_rate_host_and_moe_term(self):
        # time = gb / 400 GB/s + 2 ms; a MoE at 2 GB active with 24 layers at 0.05 ms each
        t = lambda gb, extra=0: 1000 / (gb / 400 * 1000 + 2 + extra)
        rows = [gpu_row(1, 'dense', 2, t(2)), gpu_row(2, 'dense', 5, t(5)), gpu_row(3, 'dense', 8, t(8)), gpu_row(4, 'moe', 2, t(2, 24 * 0.05), 24)]
        g = self.ns['gpu_lines'](rows, BOXES, self.B)['cuda']
        self.assertAlmostEqual(g['rate_gbs']['v'], 400, places=1)
        self.assertAlmostEqual(g['eta']['v'], 0.8, places=3)
        self.assertAlmostEqual(g['host_ms']['v'], 2, places=3)
        self.assertAlmostEqual(g['moe_layer_ms']['v'], 0.05, places=4)
        for k in ('eta', 'host_ms', 'moe_layer_ms'):
            self.assertEqual(g[k]['evidence'], 'seen', k)
            self.assertTrue(g[k]['src'].startswith('MEASURED: '), k)
        self.assertIn('r1, r2, r3', g['host_ms']['src'])
        self.assertIn('r4', g['moe_layer_ms']['src'])
        self.assertEqual(g['vendors'], ['nvidia'], 'a line is a driver stack: it carries its card vendor')

    def test_quant_kernels_off_the_line_confound_it_and_host_is_never_negative(self):
        t = lambda gb, extra=0: 1000 / (gb / 400 * 1000 + 2 + extra)
        rows = [gpu_row(1, 'dense', 2, t(2)), gpu_row(2, 'dense', 5, t(5, 6)), gpu_row(3, 'dense', 8, t(8)), gpu_row(4, 'dense', 1, t(1, -5))]
        g = self.ns['gpu_lines'](rows, BOXES, self.B)['cuda']
        self.assertEqual(g['eta']['tag'], 'MEASURED, CONFOUNDED')
        self.assertEqual(g['eta']['evidence'], 'seen')
        self.assertIn('the quant kernels differ', g['eta']['src'])
        self.assertGreaterEqual(g['host_ms']['v'], 0)
        rows = [gpu_row(1, 'dense', 1, 1000 / 0.5), gpu_row(2, 'dense', 3, 1000 / 2.5), gpu_row(3, 'dense', 5, 1000 / 4.5)]   # 1 ms/GB - 0.5 ms
        g = self.ns['gpu_lines'](rows, BOXES, self.B)['cuda']
        self.assertEqual(g['host_ms']['v'], 0, 'a negative intercept is held at 0')
        self.assertIn('below zero, held at 0', g['host_ms']['src'])

    def test_moe_rows_that_disagree_are_confounded(self):
        t = lambda gb, extra=0: 1000 / (gb / 400 * 1000 + 2 + extra)
        dense = [gpu_row(1, 'dense', 2, t(2)), gpu_row(2, 'dense', 8, t(8))]
        agree = dense + [gpu_row(3, 'moe', 2, t(2, 24 * 0.05), 24), gpu_row(4, 'moe', 2, t(2, 48 * 0.055), 48)]
        split = dense + [gpu_row(3, 'moe', 2, t(2, 24 * 0.2), 24), gpu_row(4, 'moe', 2, t(2, 48 * 0.01), 48)]
        self.assertEqual(self.ns['gpu_lines'](agree, BOXES, self.B)['cuda']['moe_layer_ms']['tag'], 'MEASURED')
        mo = self.ns['gpu_lines'](split, BOXES, self.B)['cuda']['moe_layer_ms']
        self.assertEqual(mo['tag'], 'MEASURED, CONFOUNDED')
        self.assertIn('expert quant kernels differ', mo['src'])

    def test_borrowed_backends_are_inferred_and_one_size_is_no_line(self):
        t = lambda gb: 1000 / (gb / 400 * 1000 + 2)
        rows = [gpu_row(1, 'dense', 2, t(2)), gpu_row(2, 'dense', 8, t(8)), gpu_row(3, 'dense', 4, t(4), backend='sycl')]
        out = self.ns['gpu_lines'](rows, BOXES, self.B)
        self.assertNotIn('sycl', out, 'one dense size cannot give a line')
        for be in self.B['borrow']['v']:
            self.assertEqual(out[be]['borrowed_from'], 'cuda')
            self.assertEqual(out[be]['host_ms']['tag'], 'ESTIMATE')
            self.assertEqual(out[be]['host_ms']['evidence'], 'inferred')
            self.assertEqual(out[be]['vendors'], [self.B['borrow_vendor']['v'][be]])
        self.assertEqual(out['cuda']['moe_layer_ms']['evidence'], 'unknown', 'no MoE row: the MoE term is UNKNOWN')

    def test_other_engines_and_speculative_rows_stay_off_the_line(self):
        # a strata row (expert cache + CPU pool + MTP) and a llama.cpp row with a draft model measure their scheduler, not the read
        t = lambda gb: 1000 / (gb / 400 * 1000 + 2)
        rows = [gpu_row(1, 'dense', 2, t(2)), gpu_row(2, 'dense', 8, t(8)),
                dict(gpu_row(3, 'dense', 5, 3 * t(5)), engine='strata'), dict(gpu_row(4, 'dense', 5, t(5) / 2), conditions={'spec': 'draft-simple'})]
        self.assertAlmostEqual(self.ns['gpu_lines'](rows, BOXES, self.B)['cuda']['host_ms']['v'], 2, places=3)
        self.assertFalse(self.ns['on_line'](rows[2], self.B))
        self.assertFalse(self.ns['on_line'](rows[3], self.B))
        self.assertTrue(self.ns['on_line'](dict(rows[0], conditions={'spec': 'none'}), self.B))

    def test_failed_rows_are_left_out(self):
        t = lambda gb: 1000 / (gb / 400 * 1000 + 2)
        rows = [gpu_row(1, 'dense', 2, t(2)), gpu_row(2, 'dense', 8, t(8)), dict(gpu_row(3, 'dense', 5, 1.0), rc=1)]
        self.assertAlmostEqual(self.ns['gpu_lines'](rows, BOXES, self.B)['cuda']['host_ms']['v'], 2, places=3)

    def test_a_line_is_one_stack_and_every_stack_gets_its_own_line(self):
        # legacy rows (no conditions) take stack_of; an A/B stack at 2x the rate stays off the line and gets its own line
        old = lambda gb: 1000 / (gb / 150 * 1000 + 3)
        new = lambda gb: 1000 / (gb / 300 * 1000 + 3)
        cond = {'stack': 'mesa-new', 'mmvq': 'on', 'fa': '1', 'split_mode': 'none'}
        rows = [gpu_row(1, 'dense', 2, old(2), backend='vulkan'), gpu_row(2, 'dense', 8, old(8), backend='vulkan'),
                dict(gpu_row(3, 'dense', 2, new(2), backend='vulkan'), conditions=cond), dict(gpu_row(4, 'dense', 8, new(8), backend='vulkan'), conditions=cond)]
        B = dict(self.B, stack_of={'v': {'vulkan': {'stack': 'mesa-old', 'mmvq': 'on', 'fa': 'auto'}}},
                 line_stack={'v': {'vulkan': {'stack': 'mesa-old', 'mmvq': 'on', 'fa': 'auto'}}})
        self.assertAlmostEqual(self.ns['gpu_lines'](rows, BOXES, B)['vulkan']['rate_gbs']['v'], 150, places=0)
        st = self.ns['gpu_stacks'](rows, BOXES, B)['vulkan']
        self.assertEqual(sorted(x['stack']['stack'] for x in st), ['mesa-new', 'mesa-old'])
        self.assertEqual([x['on_line'] for x in st if x['stack']['stack'] == 'mesa-old'], [True])
        self.assertAlmostEqual([x for x in st if x['stack']['stack'] == 'mesa-new'][0]['rate_gbs']['v'], 300, places=0)
        B2 = dict(B, line_stack={'v': {'vulkan': {'stack': 'mesa-new', 'mmvq': 'on', 'fa': '1'}}})
        self.assertAlmostEqual(self.ns['gpu_lines'](rows, BOXES, B2)['vulkan']['rate_gbs']['v'], 300, places=0)
        self.assertEqual(self.ns['gpu_lines'](rows, BOXES, B2)['vulkan']['rows'], ['r3', 'r4'], 'the refit line cites only its stack rows')


def numa_row(arm, rnd, pol, mem, cpu, sockets, threads, tps):
    return {'id': '%s-%s' % (arm, rnd), 'kind': 'numa_decode', 'rc': 0, 'box': 'bx', 'round': rnd, 'policy': pol, 'mem_nodes': mem, 'cpu_nodes': cpu,
            'sockets': sockets, 'threads': threads, 'decode_tps': tps}


class NumaGains(unittest.TestCase):
    def setUp(self):
        self.ns = bench_ns()

    def test_gain_over_the_same_rounds_local_node_and_confounded_spread(self):
        rows = [numa_row('N1', 'r1', 'membind', [4], [4], 1, 7, 4.0), numa_row('N1h', 'r1', 'membind', [4], [4], 1, 3, 3.0),
                numa_row('N1', 'r2', 'membind', [4], [4], 1, 7, 5.0),
                numa_row('X2', 'r1', 'interleave', [3, 4], [3, 4], 2, 15, 3.2), numa_row('X2', 'r2', 'interleave', [3, 4], [3, 4], 2, 15, 4.0),
                numa_row('X8', 'r1', 'interleave', list(range(8)), list(range(8)), 2, 15, 2.0), numa_row('X8', 'r2', 'interleave', list(range(8)), list(range(8)), 2, 15, 6.0)]
        g = self.ns['numa_gains'](rows, self.ns['bench'])['bx']
        self.assertAlmostEqual(g['membind m1 c1 s1 t7']['v'], 1.0)
        self.assertAlmostEqual(g['membind m1 c1 s1 t3']['v'], 0.75, msg='the thread-limited arm is its own key, never the unit')
        self.assertAlmostEqual(g['interleave m2 c2 s2 t15']['v'], 0.8, msg='each round over its own local arm')
        self.assertEqual(g['interleave m2 c2 s2 t15']['evidence'], 'seen')
        self.assertEqual(g['interleave m8 c8 s2 t15']['tag'], 'MEASURED, CONFOUNDED')
        self.assertIn('X2-r1 3.20 / N1-r1 4.00', g['interleave m2 c2 s2 t15']['src'])


class ReadBytes(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('record_bench', os.path.join(ROOT, 'tools', 'record_bench.py'))
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def test_dense_untied_skips_the_embedding_and_tied_reads_it(self):
        ts = [['token_embd.weight', [], 8, 100], ['output.weight', [], 8, 100], ['blk.0.attn_q.weight', [], 8, 50]]
        self.assertEqual(self.m.read_bytes({'kv': {'general.architecture': 'x'}, 'tensors': ts})[:3], (150, 250, 0))
        self.assertEqual(self.m.read_bytes({'kv': {'general.architecture': 'x'}, 'tensors': [ts[0], ts[2]]})[:3], (150, 150, 0))

    def test_experts_count_k_of_e_and_moe_layers(self):
        kv = {'general.architecture': 'a', 'a.expert_count': 8, 'a.expert_used_count': 2}
        ts = [['token_embd.weight', [], 8, 10], ['output.weight', [], 8, 10], ['blk.0.ffn_up_exps.weight', [], 12, 80],
              ['blk.1.ffn_down_exps.weight', [], 12, 80], ['blk.1.ffn_gate_inp.weight', [], 0, 4]]
        rd, total, L, E, k = self.m.read_bytes({'kv': kv, 'tensors': ts})
        self.assertEqual((rd, total, L, E, k), (10 + 20 + 20 + 4, 184, 2, 8, 2))

    def test_decode_is_the_median_repetition(self):
        self.assertEqual(self.m.decode_tps({'samples_ts': [51.5, 115.8, 114.8], 'avg_ts': 94.0}), 114.8, 'a repetition still paying JIT is an outlier')
        self.assertEqual(self.m.decode_tps({'samples_ts': [1, 2]}), 1.5)
        self.assertEqual(self.m.decode_tps({'avg_ts': 3}), 3)

    def test_numa_fields_from_policy_and_first_touch(self):
        r = {'cpus': '24-39', 'policy': 'interleave=3,4', 'threads': 15, 'node_mib': '0 0 0 4000 4000 0 0 0 8000'}
        f = self.m.numa_fields(r, 8, 4)
        self.assertEqual((f['cpu_nodes'], f['mem_nodes'], f['sockets']), ([3, 4], [3, 4], 2))
        ft = self.m.numa_fields(dict(r, policy='first_touch', node_mib='10 0 0 0 7000 990 0 0 8000'), 8, 4)
        self.assertEqual(ft['mem_nodes'], [4, 5], 'a node under NUMA_FT_SHARE of the pages does not count')


class BuiltData(unittest.TestCase):
    def test_bench_terms_in_data_cite_their_rows(self):
        d = json.load(open(os.path.join(ROOT, 'data.json')))
        rows = {json.loads(x)['id'] for x in open(os.path.join(ROOT, 'runs', 'bench.jsonl')) if x.strip()}
        self.assertEqual(d['bench']['rows'], len(rows))
        for be, g in d['bench']['gpu'].items():
            self.assertTrue(g['vendors'], be)
            for k in ('eta', 'host_ms', 'moe_layer_ms'):
                if g[k]['evidence'] == 'seen':
                    self.assertTrue(g[k].get('runs') and set(g[k]['runs']) <= rows, (be, k))
        for box, G in d['bench']['numa'].items():
            for k, x in G.items():
                self.assertEqual(x['evidence'], 'seen', k)
                self.assertTrue(set(x['runs']) <= rows, k)


if __name__ == '__main__':
    unittest.main()
