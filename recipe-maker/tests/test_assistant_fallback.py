#!/usr/bin/env python3
"""Retained household-assistant placement counterexample through the existing ledger/check owners."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import record_run

FIXTURE = ROOT / 'runs/assistant-droopy-fallback-20261024.jsonl'


class AssistantFallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
        code = """
const fs = require('fs'), FM = require('./src/model.js');
const data = JSON.parse(fs.readFileSync('data.json'));
const rows = JSON.parse(fs.readFileSync(0, 'utf8'));
console.log(JSON.stringify(rows.map(r => FM.fallbackCheck(data, r.recipe, r.measured))));
"""
        opposite = copy.deepcopy(cls.rows[1])
        opposite['measured']['fallback_ledger']['identity']['engine_build'] = 'mismatched-control'
        result = subprocess.run(['node', '-e', code], cwd=ROOT,
                                input=json.dumps(cls.rows + [opposite]), text=True,
                                capture_output=True, check=True)
        cls.checks = json.loads(result.stdout)

    def test_both_rows_validate_through_the_existing_recorder(self):
        self.assertEqual(len(self.rows), 2)
        self.assertEqual(len({row['id'] for row in self.rows}), 2)
        for row in self.rows:
            with self.subTest(row=row['id']):
                self.assertEqual(record_run.validate(row), [])

    def test_original_lower_tier_fails_the_maintained_intent(self):
        before = self.checks[0]
        self.assertEqual(before['status'], 'FAIL')
        self.assertTrue(before['pred']['identity_matched'])
        path = next(row for row in before['pred']['rows'] if row['id'] == 'cold_experts:run')
        self.assertEqual(path['status'], 'FAIL')
        self.assertIn('pmem', path['detail'])
        self.assertFalse(before['pred']['live_capacity_inferred'])

    def test_corrected_model_buffer_path_does_not_blanket_pass_other_classes(self):
        after = self.checks[1]
        path = next(row for row in after['pred']['rows'] if row['id'] == 'cold_experts:run')
        self.assertEqual(path['status'], 'PASS')
        self.assertEqual(after['status'], 'UNKNOWN')
        self.assertFalse(after['pred']['live_capacity_inferred'])
        self.assertTrue(any(row['status'] == 'UNKNOWN' for row in after['pred']['rows']))

    def test_a_foreign_build_cannot_inherit_the_observed_path(self):
        result = self.checks[2]
        self.assertFalse(result['pred']['identity_matched'])
        path = next(row for row in result['pred']['rows'] if row['id'] == 'cold_experts:run')
        self.assertEqual(path['status'], 'UNKNOWN')

    def test_mapped_page_count_preserves_exact_bytes_and_distinct_units(self):
        before = self.rows[0]['measured']['placement']
        expected = before['model_mmap_pages_by_node']['2'] * before['page_bytes']
        self.assertEqual(expected, 3862511616)
        self.assertEqual(before['observed_node2_bytes'], expected)
        self.assertEqual(before['observed_node2_gib'], expected / 1024**3)
        self.assertEqual(before['observed_node2_gb'], expected / 1e9)
        self.assertNotEqual(before['observed_node2_gib'], before['observed_node2_gb'])
        self.assertIn('not an exclusive CPU-expert byte count', before['scope'])
        after = self.rows[1]['measured']['placement']
        self.assertFalse(after['primary_model_file_mmap_present'])
        self.assertEqual(set(after['dev_zero_pages_by_node']), {'0'})
        self.assertEqual(after['gpu_free_mib_after_4k'], 3132)

    def test_backfill_never_invents_a_launch_prediction_or_isolated_causality(self):
        for row in self.rows:
            with self.subTest(row=row['id']):
                self.assertIsNone(row['check'])
                self.assertEqual(row['prediction']['basis'], 'post-hoc')
                self.assertEqual(row['prediction']['status'], 'UNMODELLED')
                self.assertIn('decode_tps', row['measured']['not_measured'])
                self.assertIn('per_tensor_read_placement', row['measured']['not_measured'])
                self.assertIn('not a controlled locality experiment', row['measured']['notes'])
                self.assertTrue(all('sha256:' in src for src in row['sources']))
        self.assertEqual(self.rows[0]['measured']['proxy_chat_wall_s'], 96.09)
        self.assertEqual(self.rows[1]['measured']['proxy_chat_wall_s'], 0.586)
        self.assertEqual(self.rows[0]['measured']['proxy_tool_wall_s'], 59.247)
        self.assertEqual(self.rows[1]['measured']['proxy_tool_wall_s'], 3.48)


if __name__ == '__main__':
    unittest.main(verbosity=2)
