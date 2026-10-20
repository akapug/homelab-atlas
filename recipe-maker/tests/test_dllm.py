"""the dLLM term: tools/record_dllm.py (the per-prompt sums, the ledger rows it wrote) and src/model.js (tests/test_dllm.js:
coverage, the kernel-gap fit and pooling, the UNMODELLED rules, the held-out run, --check trust lines)"""
import json
import os
import shutil
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import record_dllm  # noqa: E402
import record_run  # noqa: E402


def prompt(n_gen, n_fwd, ms, wall, pre, sub):
    return {'n_gen': n_gen, 'n_fwd': n_fwd, 'ms_per_fwd': ms, 'wall_ms': wall, 'prefill_ms': pre, 'tok_per_fwd_submitted': sub}


class Summary(unittest.TestCase):
    def test_sums_over_prompts_weight_by_forwards(self):
        s = record_dllm.summary([prompt(256, 64, 100, 7000, 600, 40), prompt(256, 128, 50, 7000, 600, 30)])
        self.assertEqual(s['tpf'], round(512 / 192, 3))
        self.assertEqual(s['ms_per_fwd'], round((100 * 64 + 50 * 128) / 192, 2))
        self.assertEqual(s['decode_ms_per_fwd'], round(12800 / 192, 2), 'decode time = wall - prefill, over all forwards')
        self.assertEqual(s['tokens_per_fwd_submitted'], round((40 * 64 + 30 * 128) / 192, 2))
        self.assertEqual(s['decode_tps'], round(512 / 12800 * 1000, 2))


class LedgerRows(unittest.TestCase):
    def setUp(self):
        self.rows = [r for r in record_run.load() if 'dllm' in (r.get('tags') or [])]
        self.cat = json.load(open(os.path.join(ROOT, 'tools', 'dllm_catalog.json')))

    def test_rows_validate_and_declare_every_condition(self):
        self.assertGreaterEqual(len(self.rows), 30)
        for r in self.rows:
            record_run.validate(r)
            c = r['recipe']['conditions']
            for k in ('ctx', 'kv_type', 'draft_n', 'sampling', 'reasoning_effort', 'workload', 'cards', 'placement'):
                self.assertIn(k, c, r['id'] + ' ' + k)
            self.assertEqual(r['recipe']['kind'], 'dllm')

    def test_kernel_path_comes_from_the_catalog(self):
        paths = {s['moe_path'] for s in self.cat['stages'].values()}
        for r in self.rows:
            moe = 'E' in json.load(open(os.path.join(ROOT, 'data', 'models', r['model'] + '.json')))
            if r['recipe']['backend'] == 'sycl':
                self.assertIn(r['recipe']['moe_path'], paths if moe else {None}, r['id'] + ': a dense model has no MoE kernel path')

    def test_probe_disturbed_rows_are_degraded(self):
        es = [r for r in self.rows if '--expert-stats' in (r['recipe']['schedule'].get('args') or '')]
        self.assertTrue(es)
        for r in es:
            self.assertIn('--expert-stats', r['measured']['degraded'][0], r['id'])


class ModelJs(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'node not installed')
    def test_dllm_js(self):
        p = subprocess.run(['node', os.path.join('tests', 'test_dllm.js')], cwd=ROOT, capture_output=True, text=True, timeout=600)
        self.assertEqual(p.returncode, 0, p.stdout[-3000:] + p.stderr[-3000:])
        self.assertRegex(p.stdout, r'\d+ tests passed')


if __name__ == '__main__':
    unittest.main()
