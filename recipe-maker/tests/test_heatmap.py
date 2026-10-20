"""hot-expert maps (tools/heatmap.py): converters, validation, the loader the planner builds from, and the repo's maps"""
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import heatmap as HM  # noqa: E402

META = {'id': 'toy__w__eng__2026-10-18', 'title': 'toy',
        'model': {'id': 'toy/Toy-MoE', 'family': 'toy', 'checkpoint': 'toy q8', 'checkpoint_hash': 'sha256:' + '0' * 64, 'quant': 'q8_0', 'k': 2},
        'workload': {'name': 'w', 'description': 'two prompts', 'prompt_set_sha256': '1' * 64},
        'producer': {'tool': 't', 'tool_version': '1', 'engine': 'e', 'box': 'b', 'date': '2026-10-18'},
        'provenance': {'tag': 'MEASURED', 'src': 'unit test'}}


def tmpfile(test, name, data, mode='w'):
    d = tempfile.mkdtemp(prefix='heatmap-')
    test.addCleanup(shutil.rmtree, d)
    p = os.path.join(d, name)
    with open(p, mode) as f:
        f.write(data)
    return p


def trace(records):
    out = b''
    for pid, phase, layer, k, ids in records:
        n = len(ids) // k
        out += struct.pack('<7i', 0x4D4F4532, pid, phase, layer, 0, k, n) + struct.pack('<%di' % len(ids), *ids)
    return out


class Converters(unittest.TestCase):
    def test_moe_trace_phases_categories_and_short_last_layer(self):
        # layers 3 and 4, E=4, k=2. Prompt 0 (code): prefill 2 tokens on both layers; prompt 1 (chat): prefill 2 tokens on
        # layer 3 but 1 on layer 4 (llama.cpp prunes the last layer in prefill). Decode: one token per prompt per layer.
        recs = [(0, 0, 3, 2, [0, 1, 0, 2]), (0, 0, 4, 2, [1, 2, 1, 3]), (1, 0, 3, 2, [3, 2, 3, 0]), (1, 0, 4, 2, [0, 1]),
                (0, 1, 3, 2, [0, 1]), (0, 1, 4, 2, [2, 3]), (1, 1, 3, 2, [1, 2]), (1, 1, 4, 2, [3, 0])]
        p = tmpfile(self, 't.bin', trace(recs), 'wb')
        m = HM.build_map('from-moe-trace', p, META, {'experts': 4, 'categories': {0: 'code', 1: 'chat'}, 'default_slice': 'decode'})
        self.assertEqual(m['model']['layers'], [3, 4])
        self.assertEqual(m['slices']['decode']['counts'], [[1, 2, 1, 0], [1, 0, 1, 2]])
        self.assertEqual(m['slices']['decode_code']['counts'], [[1, 1, 0, 0], [0, 0, 1, 1]])
        self.assertEqual(m['slices']['decode_chat']['category'], 'chat')
        self.assertEqual(m['slices']['decode']['tokens'], 2)
        self.assertEqual(m['slices']['prefill']['counts'][0], [3, 1, 2, 2])
        self.assertTrue(any('prunes the last layer' in x for x in m['provenance']['notes']), m['provenance'])
        self.assertEqual(m['provenance']['converted_from']['sha256'], HM.sha256_file(p))

    def test_moe_trace_bad_magic_refuses(self):
        p = tmpfile(self, 'bad.bin', struct.pack('<7i', 1, 0, 0, 3, 0, 1, 1) + struct.pack('<i', 0), 'wb')
        with self.assertRaises(HM.MapError):
            HM.build_map('from-moe-trace', p, META, {'experts': 4})

    def test_imatrix_tsv_one_row_per_layer_sorted(self):
        tsv = ('blk.10.ffn_down_exps.weight\t9\t9\t9\n'
               'blk.10.ffn_up_exps.weight\t1\t2\t3\n'
               'blk.2.ffn_gate_exps.weight\t4\t0\t2\n'
               'blk.2.attn_q.weight\t7\n')
        m = HM.build_map('from-imatrix-tsv', tmpfile(self, 'c.tsv', tsv), META)
        self.assertEqual(m['model']['layers'], [2, 10])
        self.assertEqual(m['slices']['prefill']['counts'], [[4, 0, 2], [1, 2, 3]], 'up first, then gate, then down')
        self.assertEqual(m['slices']['prefill']['tokens_tag'], 'INFERRED')
        self.assertEqual(m['slices']['prefill']['tokens'], round(6 / 2), 'picks per layer / k')

    def test_route_stats_tokens_measured(self):
        d = {'prefill': {'model.language_model.layers.4.mlp': [5, 1], 'model.language_model.layers.3.mlp': [2, 2]},
             'decode': {'model.language_model.layers.4.mlp': [1, 1], 'model.language_model.layers.3.mlp': [0, 2]},
             'total': {}, 'meta': {'prefill_tokens': 3, 'decode_tokens': 1}}
        m = HM.build_map('from-route-stats', tmpfile(self, 'rs.json', json.dumps(d)), META)
        self.assertEqual(m['model']['layers'], [3, 4])
        self.assertEqual(m['slices']['prefill']['counts'], [[2, 2], [5, 1]])
        self.assertEqual((m['slices']['decode']['tokens'], m['slices']['decode']['tokens_tag']), (1, 'MEASURED'))

    def test_exl3_stats(self):
        d = {'model.layers.1.ffn.experts': [[1, 0], [5, 5]], 'model.layers.0.ffn.experts': [[0, 2], [3, 3]]}
        m = HM.build_map('from-exl3-stats', tmpfile(self, 'x.json', json.dumps(d)), META)
        self.assertEqual(m['slices']['decode']['counts'], [[0, 2], [1, 0]])
        self.assertEqual(m['slices']['prefill']['counts'], [[3, 3], [5, 5]])

    def test_selection_from_banks_ranges_and_citation(self):
        m = HM.build_map('from-imatrix-tsv', tmpfile(self, 'c.tsv', 'blk.1.ffn_up_exps.weight\t1\t2\t3\t4\t5\n'), META)
        tsv = '# 4 experts, 0.1 GiB\nblk.1\t0,2-4\t# comment\n'
        s = HM.selection_from_banks(tmpfile(self, 'b.tsv', tsv), {'id': 'sel', 'model': {'id': 'toy/Toy-MoE', 'family': 'toy'}, 'cites': HM.cite(m),
                                                                  'provenance': {'tag': 'MEASURED', 'src': 't'}})
        self.assertEqual(s['experts'], {'1': [0, 2, 3, 4]})
        self.assertEqual(s['n_experts'], 4)
        self.assertEqual(s['cites']['digest'], m['digest'])
        self.assertIn('4 experts', s['rule'])

    def test_manifest_hash_ignores_paths_order_and_binary_marks(self):
        a = tmpfile(self, 'a.sha256', 'b' * 64 + '  dir/x-2.gguf\n' + 'a' * 64 + '  x-1.gguf\n')
        b = tmpfile(self, 'b.sha256', 'a' * 64 + ' *x-1.gguf\n' + 'b' * 64 + '  x-2.gguf\n')
        self.assertEqual(HM.manifest_hash(a), HM.manifest_hash(b))
        with self.assertRaises(HM.MapError):
            HM.manifest_hash(tmpfile(self, 'e.sha256', 'nothing here\n'))


class Validation(unittest.TestCase):
    def good(self):
        return HM.build_map('from-imatrix-tsv', tmpfile(self, 'c.tsv', 'blk.0.ffn_up_exps.weight\t1\t2\t3\nblk.1.ffn_up_exps.weight\t0\t0\t6\n'), META)

    def test_good_map_round_trips_through_the_file_format(self):
        m = self.good()
        p = tmpfile(self, 'm.heatmap.json', HM.dumps(m))
        self.assertEqual(HM.load(p)['digest'], m['digest'])
        self.assertEqual(HM.slurp(p).count('[1,2,3]'), 1, 'a count row stays on one line')

    def test_tampered_counts_fail_summary_and_digest(self):
        m = self.good()
        m['slices']['prefill']['counts'][0][0] += 1
        m['slices']['prefill']['picks'] += 1
        with self.assertRaisesRegex(HM.MapError, 'summary does not match|digest'):
            HM.validate(m)

    def test_digest_alone_catches_a_metadata_change(self):
        m = self.good()
        m['model']['quant'] = 'q4_K'
        with self.assertRaisesRegex(HM.MapError, 'digest'):
            HM.validate(m)

    def test_shape_and_provenance_rules(self):
        for mut, rx in ((lambda m: m['slices']['prefill']['counts'][1].append(1), 'experts, want'),
                        (lambda m: m['slices']['prefill']['counts'].pop(), 'count rows'),
                        (lambda m: m['model'].update(checkpoint_hash=None), 'checkpoint_hash_note'),
                        (lambda m: m['workload'].update(prompt_set_sha256=None), 'prompt_set_note'),
                        (lambda m: m['provenance'].update(tag='GUESS'), 'provenance.tag'),
                        (lambda m: m['slices']['prefill'].update(phase='warmup'), 'phase'),
                        (lambda m: m['model'].pop('k'), 'missing model.k'),
                        (lambda m: m.update(default_slice='decode'), 'default_slice')):
            m = self.good()
            mut(m)
            with self.assertRaisesRegex(HM.MapError, rx):
                HM.validate(m)

    def test_planner_layers_align_and_refuse_gaps(self):
        m = self.good()
        self.assertEqual(HM.planner_layers(m, 'prefill', [1, 0], 3), [[0, 0, 6], [1, 2, 3]])
        with self.assertRaisesRegex(HM.MapError, 'no counts for planner layers'):
            HM.planner_layers(m, 'prefill', [0, 1, 2], 3)
        with self.assertRaisesRegex(HM.MapError, 'experts'):
            HM.planner_layers(m, 'prefill', [0], 4)
        with self.assertRaisesRegex(HM.MapError, 'no slice'):
            HM.planner_layers(m, 'decode', [0], 3)

    def test_load_dir_refuses_duplicate_ids(self):
        d = tempfile.mkdtemp(prefix='heatdir-')
        self.addCleanup(shutil.rmtree, d)
        m = self.good()
        for fn in ('a.heatmap.json', 'b.heatmap.json'):
            with open(os.path.join(d, fn), 'w') as f:
                f.write(HM.dumps(m))
        with self.assertRaisesRegex(HM.MapError, 'duplicate map id'):
            HM.load_dir(d)


class RepoMaps(unittest.TestCase):
    """the maps in data/heat: all valid, and the converters reproduce the planner's original count files exactly"""
    HEAT = os.path.join(ROOT, 'data', 'heat')

    def test_every_file_validates(self):
        files = [f for f in os.listdir(self.HEAT) if f.endswith('.json')]
        self.assertGreaterEqual(len(files), 9)
        self.assertEqual(HM.main(['validate'] + [os.path.join(self.HEAT, f) for f in files]), 0)

    def test_glm_trace_reproduces_the_original_category_counts(self):
        m = HM.load_dir(self.HEAT)['glm53-flash__skew72__llamacpp-cpu__2026-10-09']
        old = HM.jload(os.path.join(ROOT, 'src', 'counts-glm-decode-20261009.json'))
        self.assertEqual(m['model']['layers'], old['layers'])
        for c in ('code', 'diff', 'tool', 'chat'):
            self.assertEqual(m['slices']['decode_' + c]['counts'], old['cats'][c], c)
        self.assertTrue(m['model']['checkpoint_hash'].startswith('sha256:'))

    def test_v41_maps_match_their_sources(self):
        maps = HM.load_dir(self.HEAT)
        old = HM.jload(os.path.join(ROOT, 'src', 'counts-v41-decode-20261008.json'))
        m = maps['dsv41-flash__skew187__dsv41-exl3__2026-10-08']
        for k in ('coding_odd', 'coding_even', 'general_all'):
            self.assertEqual(m['slices'][k]['counts'], old[k])
        self.assertEqual(m['summary']['all_all']['tokens'], 192638, 'picks / (40 x 6) = the capture accounting 192,638 decode rows')
        im = maps['dsv41-flash__a5-docs-51k__ik-imatrix__2026-10-16']
        self.assertEqual(im['provenance']['converted_from']['sha256'], HM.sha256_file(os.path.join(ROOT, 'src', 'inputs', 'a5v41-expert-counts.tsv')),
                         'the imatrix map is converted from the tracked snapshot (re-run convert-all when it changes)')

    def test_selections_cite_maps_that_exist(self):
        maps = HM.load_dir(self.HEAT)
        for f in os.listdir(self.HEAT):
            if f.endswith('.selection.json'):
                s = HM.jload(os.path.join(self.HEAT, f))
                HM.validate_selection(s)
                self.assertIn(s['cites']['map_id'], maps, f)
                self.assertEqual(s['cites']['digest'], maps[s['cites']['map_id']]['digest'], f)


class Planner(unittest.TestCase):
    """data.json (python3 build.py) carries the map citation on every measured heat set"""
    @unittest.skipUnless(os.path.exists(os.path.join(ROOT, 'data.json')), 'data.json not built')
    def test_heat_sets_cite_maps(self):
        data = HM.jload(os.path.join(ROOT, 'data.json'))
        maps = HM.load_dir(os.path.join(ROOT, 'data', 'heat'))
        for mk in ('v41', 'glm'):
            for name, s in data['models'][mk]['heat']['sets'].items():
                if 'sum' in s:
                    continue
                self.assertIn('map', s, mk + ' ' + name)
                self.assertEqual(s['map']['digest'], maps[s['map']['map_id']]['digest'])
        sets = data['models']['glm']['heat']['sets']
        self.assertIn('nv2_code_prefill', sets)
        self.assertIn('dsv41-flash-1017__a5-docs-51k__ik-imatrix__2026-10-17:prefill', data['models']['v41']['heat']['sets'])


if __name__ == '__main__':
    unittest.main()
