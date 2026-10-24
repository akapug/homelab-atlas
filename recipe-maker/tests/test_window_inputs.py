"""probe/window_inputs.py: the batch-window spec and microbenchmark readers on inputs of another model and card"""
import ast
import importlib.util
import os
from pathlib import Path
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location('window_inputs', os.path.join(HERE, '..', 'probe', 'window_inputs.py'))
WIN = importlib.util.module_from_spec(_s)
_s.loader.exec_module(WIN)

LOG = """noise before the table
start: copy kernel 64.0 MiB -> 64.0 MiB: 100.0 us = 1340.0 GB/s (read+write); memcpy 1300.0 GB/s
method floor (flush, tag, tag): 4.2 us, subtracted from every row below
device: Some Card, 128 compute units
end: copy kernel 64.0 MiB -> 64.0 MiB: 99.0 us = 1355.5 GB/s (read+write)

STAGE ROLL-UP: 8 layers (4 MLA, 4 DENSE), sum of isolated kernel times, ms; slope = (t8 - t1) / 7 per layer it runs in; floor = bytes at 8 rows / 1355 GB/s (the copy kernel)
kernel                       layers        1        2        8 us/row/lyr  MB@8/lyr   floor us  x floor
norm (x3 fused) x3                8    0.100    0.110    0.150        0.9       0.10        0.1     50.0
mla attention                     4    0.400    0.500    0.900       17.9       9.00        6.6     34.0
mla-layer ffn                     4    0.300    0.320    0.400        3.6      40.00       29.5      3.4
dense-layer ffn                   4    0.310    0.330    0.420        3.9      44.00       32.5      3.2
routed experts (random)           8    1.000    1.500    3.000       35.7     100.00       73.8      5.1
TOTAL (random routing)                 2.110    2.760    4.870
TOTAL (rows share experts)             2.100    2.500    3.900
not measured: the head
trailing line that is not a kernel row
"""

SPEC = """dims n_embd=1024 n_expert=64 k=4 n_layers=8
layer 0 attn_kv_a_mqa=8 ffn_up=12
layer 1 ffn_up=12
layer 2 attn_kv_a_mqa=8 ffn_up=12
"""


class WindowInputs(unittest.TestCase):
    def test_ubench_of_another_card(self):
        u = WIN.parse_ubench(LOG)
        self.assertEqual(u['grid'], [1, 2, 8])
        self.assertEqual(u['floor_rows'], 8)
        self.assertEqual(u['floor_gbs'], 1355.0)
        self.assertEqual(u['kinds'], {'MLA': 4, 'DENSE': 4})
        self.assertEqual(u['copy_gbs_end'], 1355.5)
        self.assertEqual(u['method_floor_us'], 4.2)
        self.assertEqual([c['name'] for c in u['classes']], ['norm (x3 fused) x3', 'mla attention', 'mla-layer ffn', 'dense-layer ffn', 'routed experts (random)'])
        self.assertEqual(u['classes'][0]['launches'], 3)
        self.assertEqual(u['classes'][1]['ms'], [0.4, 0.5, 0.9])
        self.assertEqual(u['routing']['class'], 'routed experts (random)')
        self.assertEqual(u['routing']['alt_delta_ms'], [-0.01, -0.26, -0.97])
        self.assertEqual(u['unmeasured'], 'the head')

    def test_kind_by_count_then_by_name(self):
        u = WIN.parse_ubench(LOG)
        kinds = [WIN.class_kind(c, u['kinds'], u['layers']) for c in u['classes']]
        self.assertEqual(kinds, [None, 'MLA', 'MLA', 'DENSE', None])
        with self.assertRaises(ValueError):
            WIN.class_kind({'name': 'ffn', 'layers': 4}, u['kinds'], u['layers'])

    def test_spec_kind_rule_is_the_callers(self):
        s = WIN.parse_spec(SPEC, {'tensor': 'attn_kv_a_mqa', 'kind': 'MLA', 'default': 'DENSE'})
        self.assertEqual(s['dims']['n_layers'], 8)
        self.assertEqual([x['kind'] for x in s['layers']], ['MLA', 'DENSE', 'MLA'])
        self.assertEqual(s['layers'][0]['types'], {'attn_kv_a_mqa': 8, 'ffn_up': 12})

    def test_native_unit_network_attempt_needs_no_kernel_throughput(self):
        # Compile the real compiler seam without running build.py's catalog assembly.
        source = ast.parse((Path(HERE).parent / 'build.py').read_text())
        nodes = [x for x in source.body if isinstance(x, ast.FunctionDef) and x.name == 'window_attempts'
                 or isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'WIN_OUTCOMES' for t in x.targets)]
        self.assertEqual(len(nodes), 2)
        namespace = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'build.py', 'exec'), namespace)
        identity = {'recipe': 'fixture-park', 'agents': 4}
        measured = {'outcome': 'ok', 'network_identity': identity, 'extra_slots': 2,
                    'not_measured': ['decode_tps'], 'src': {'extra_slots': 'synthetic admission'}}
        run = {'id': 'fixture-run', 'recipe': {'kind': 'network', 'network_recipe': 'fixture-park'},
               'prediction': {'network': {'prediction_identity': identity}}, 'measured': measured, 'sources': ['synthetic']}
        cfg = {'attempts': [{'id': 'fixture-attempt', 'classes': ['*'], 'outcome': 'win', 'scope': 'bundle_only',
                             'network_recipe': 'fixture-park', 'ledger': {'id': run['id']}}]}
        actual = namespace['window_attempts']('fixture-window', cfg, [], {run['id']: run})[0]
        self.assertEqual(actual['network_receipt']['measured'], measured)
        self.assertEqual(actual['network_receipt']['identity'], identity)
        self.assertIsNone(actual['tps'])
        self.assertIsNone(actual['sanity'])

    def test_bad_inputs_fail_loudly(self):
        with self.assertRaises(ValueError):
            WIN.parse_ubench('no table here')
        with self.assertRaises(ValueError):
            WIN.parse_spec('layer 0 a=1', {'tensor': 'a', 'kind': 'X', 'default': 'Y'})


if __name__ == '__main__':
    unittest.main()
