import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import priors_adapter
import record_run
import record_screen


class Wave2ToolingTests(unittest.TestCase):
    def test_parse_kit_does_not_attribute_earlier_arm_json_to_crashed_arm(self):
        log = (
            '--- arm1 ---\n'
            '{"arm": "arm1", "tps": 42.0}\n'
            '--- arm2 ---\n'
            'Segmentation fault\n'
        )
        rows = record_run.parse_kit(log, 'arm2')
        self.assertEqual(rows, [])

    def test_record_screen_write_is_atomic(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'screen_ledger.jsonl')
            record_screen.write([{'id': 'a1', 'v': 1}], path, False)
            with open(path) as f:
                self.assertEqual(json.loads(f.read().strip())['v'], 1)
            record_screen.write([{'id': 'a1', 'v': 2}], path, True)
            with open(path) as f:
                self.assertEqual([json.loads(ln)['v'] for ln in f if ln.strip()], [2])
            self.assertEqual(os.listdir(td), ['screen_ledger.jsonl'])

    def test_priors_adapter_reads_explicit_schema_and_ignores_err(self):
        data = {
            'siblings': {
                'quant_priors': {
                    'present': True,
                    'data': {
                        'kld_priors': {
                            'v41': {
                                'q4_0': {'kld': 0.015},
                                'q8_0': {'err': 99.0},
                            },
                        },
                        'err': 123.0,
                    },
                },
            },
            'types': {
                'q4_0': {'kld': {'v41': {'v': 0.02, 'tag': 'PRIOR'}}},
                'q8_0': {'kld': {'v41': {'v': 0.005, 'tag': 'PRIOR'}}},
            },
        }
        priors_adapter.apply_priors(data)
        self.assertEqual(data['types']['q4_0']['kld']['v41']['v'], 0.015)
        self.assertEqual(data['types']['q8_0']['kld']['v41']['v'], 0.005)

    def test_bench_gpu_sh_syntax_and_pipefail_present(self):
        path = os.path.join(ROOT, 'tools', 'bench_gpu.sh')
        r = subprocess.run(['bash', '-n', path], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(path) as f:
            src = f.read()
        self.assertIn('pipefail', src)


if __name__ == '__main__':
    unittest.main()
