#!/usr/bin/env python3
"""record_screen.py: failure shapes per request, the settings an arm ran with, grades and the results row, the ledger
write; and the card files build.py loads (data/cards). Run: python3 tests/test_screen.py"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import record_screen as S  # noqa: E402

RULES = {'loop_tail_chars': S.LOOP_TAIL_CHARS, 'loop_min_line': S.LOOP_MIN_LINE, 'loop_repeats': S.LOOP_REPEATS, 'leak_tags': list(S.LEAK_TAGS)}
TPL = "{%- set e = reasoning_effort|default('xhigh') %}<|im_start|>assistant"
BODY = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20}


def put(p, text):
    with open(p, 'w') as f:
        f.write(text)


def answer(ans='DEFECT: x', finish='stop', tokens=1000, reasoning='think', ts=1792608940, body=BODY):
    return {'extra_body': body, 'max_tokens': 32768, 'finish': finish, 'usage': {'completion_tokens': tokens, 'prompt_tokens': 4000},
            'answer': ans, 'reasoning': reasoning, 'wall_s': 10.0, 'ts': ts}


def round_dir(td, files, grades=(), results=''):
    """a round as model-screen.py leaves it: answers/<arm>/*.json, logs/props-<arm>.json, grades.jsonl, RESULTS.md"""
    a = os.path.join(td, 'answers', 'm1')
    os.makedirs(a)
    os.makedirs(os.path.join(td, 'logs'))
    for name, x in files.items():
        put(os.path.join(a, name + '.json'), json.dumps(x))
    put(os.path.join(a, 'server.log'), 'srv load_model: initializing, n_slots = 1, n_ctx_slot = 49152, kv_unified\n')
    put(os.path.join(td, 'logs', 'props-m1.json'), json.dumps({'chat_template': TPL + '\n', 'model_path': '/m/x.gguf', 'default_generation_settings': {
        'n_ctx': 8192, 'params': {'temperature': 0.8, 'top_p': 0.949999988079071, 'top_k': 40, 'min_p': 0.05000000074505806, 'presence_penalty': 0.0}}}))
    put(os.path.join(td, 'grades.jsonl'), ''.join(json.dumps(g) + '\n' for g in grades))
    put(os.path.join(td, 'RESULTS.md'), results)
    return {'round': 'r1', 'dir': td, 'answers': 'answers', 'grades': 'grades.jsonl', 'results': 'RESULTS.md', 'results_heading': '## FINAL',
            'box': 'snoopy', 'engine': 'llamacpp', 'purpose': 'verdict'}


LEDGER = os.path.join(ROOT, 'runs', 'screen.jsonl')
SPEC = {'card': 'sage-a1.5', 'props': 'logs/props-m1.json', 'results_label': 'M1 Q4'}


class Shapes(unittest.TestCase):
    def test_cap_hit_and_empty(self):
        r = S.request_shapes(answer(ans='', finish='length', tokens=32768), RULES)
        self.assertTrue(r['cap_hit'] and r['empty'])
        self.assertTrue(S.failed(r))

    def test_cap_by_tokens_without_finish(self):
        self.assertTrue(S.request_shapes(answer(finish='stop', tokens=32768), RULES)['cap_hit'])

    def test_clean_answer(self):
        r = S.request_shapes(answer(), RULES)
        self.assertFalse(S.failed(r))
        self.assertEqual((r['loop'], r['leaked_tags']), (None, []))

    def test_loop_in_the_tail(self):
        line = 'Wait, let me re-check the census path once more.'
        r = S.request_shapes(answer(reasoning='start\n' + '\n'.join([line] * S.LOOP_REPEATS)), RULES)
        self.assertEqual(r['loop'], line)

    def test_a_diff_line_repeated_a_few_times_is_no_loop(self):
        line = '-            while has(hi + 1):'
        self.assertIsNone(S.request_shapes(answer(ans='\n'.join([line] * (S.LOOP_REPEATS - 2))), RULES)['loop'])

    def test_leaked_tags(self):
        r = S.request_shapes(answer(ans='verdict </think> DEFECT <|im_end|>'), RULES)
        self.assertEqual(r['leaked_tags'], ['</think>', '<|im_end|>'])

    def test_template_effort(self):
        self.assertEqual(S.template_effort(TPL), 'xhigh')
        self.assertEqual(S.template_effort('<think>'), 'none')


class Row(unittest.TestCase):
    def build(self, files, **kw):
        with tempfile.TemporaryDirectory() as td:
            rnd = round_dir(td, files, **kw)
            return S.build_row(rnd, 'm1', SPEC, RULES)

    def test_settings_request_over_server_defaults(self):
        r = self.build({'p1': answer()})
        st, src = r['settings'], r['settings_src']
        self.assertEqual((st['temperature'], st['top_k']), (1.0, 20), 'the request body wins')
        self.assertEqual(st['min_p'], 0.05, 'a server default the request did not set, rounded from float32')
        self.assertIn('server default', src['min_p'])
        self.assertEqual((st['context'], st['max_output']), (49152, 32768), 'context from server.log n_ctx_slot over props n_ctx')
        self.assertEqual((st['thinking_mode'], st['reasoning_effort'], st['reasoning_budget']), ('on', 'xhigh', None))
        self.assertEqual(st['chat_template'], S.sha_text(TPL))
        self.assertEqual(S.validate(r), [])

    def test_retry_only_and_rate(self):
        r = self.build({'p1': answer(ans='', finish='length', tokens=32768), 'p1-retry': answer(), 'p2': answer(),
                        'c1': answer(ans='', finish='length', tokens=32768), 'c1-retry': answer(ans='', finish='length', tokens=32768)})
        s = r['shapes']
        self.assertEqual((s['requests'], s['items'], s['cap_hit'], s['any'], s['retry_only']), (5, 3, 3, 3, 1))
        self.assertEqual(s['retry_only_items'], ['p1'])
        self.assertEqual([(x['item'], x['attempt']) for x in r['requests']][:2], [('c1', 1), ('c1', 2)])

    def test_mixed_settings_refused(self):
        with self.assertRaises(ValueError):
            self.build({'p1': answer(), 'p2': answer(body={'temperature': 0.6})})

    def test_set_overrides_with_source(self):
        with tempfile.TemporaryDirectory() as td:
            rnd = round_dir(td, {'p1': answer()})
            r = S.build_row(rnd, 'm1', dict(SPEC, set={'reasoning_budget': [16384, 'arm note']}), RULES)
        self.assertEqual((r['settings']['reasoning_budget'], r['settings_src']['reasoning_budget']), (16384, 'arm note'))

    def test_grades_last_row_wins_and_results_row(self):
        g = [{'model': 'm1', 'item': 'p1', 'verdict': 'NOREAD', 'false': 0}, {'model': 'm1', 'item': 'p1', 'verdict': 'PASS', 'false': 0},
             {'model': 'm1', 'item': 'c2', 'verdict': 'SOFT_FP', 'false': 1, 'strict_fp': 1, 'lenient_fp': 0},
             {'model': 'm1', 'item': 'c1', 'verdict': 'DESIGN', 'false': 0, 'design': 1}, {'model': 'other', 'item': 'p1', 'verdict': 'PASS'}]
        md = '# x\n\n## FINAL (all, 23:49Z)\n\n| model | catches/5 |\n|---|---|\n| **M1 Q4** | **5** |\n| M2 | 4 |\n\nafter\n'
        r = self.build({'p1': answer()}, grades=g, results=md)
        self.assertEqual(r['grades']['items'], {'c1': 'DESIGN', 'c2': 'SOFT_FP', 'p1': 'PASS'})
        self.assertEqual((r['grades']['fp_strict'], r['grades']['fp_lenient'], r['grades']['design']), (1, 0, 1))
        self.assertEqual(r['results_row'], {'model': '**M1 Q4**', 'catches/5': '**5**'})


class Ledger(unittest.TestCase):
    def test_write_refuses_an_id_twice_unless_replace(self):
        row = {'id': 'screen-a', 'x': 1}
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, 'screen.jsonl')
            self.assertEqual(S.write([row], p, False), (0, 1))
            with self.assertRaises(SystemExit):
                S.write([row], p, False)
            self.assertEqual(S.write([dict(row, x=2)], p, True), (0, 1))
            self.assertEqual([json.loads(x)['x'] for x in S.read(p).splitlines()], [2])

    def test_validate_sweep_needs_best(self):
        r = {'id': 'screen-a', 'round': 'r', 'card': 'c', 'box': 'b', 'engine': 'e', 'settings': {'t': 1}, 'settings_src': {'t': 's'},
             'requests': [], 'shapes': {'requests': 0}, 'sources': [], 'purpose': 'sweep'}
        self.assertTrue(any('sweep.best' in b for b in S.validate(r)))
        self.assertEqual(S.validate(dict(r, sweep={'best': {'t': 1}})), [])
        self.assertTrue(any('no source' in b for b in S.validate(dict(r, purpose='verdict', settings_src={}))))


class ShippedLedger(unittest.TestCase):
    """runs/screen.jsonl and data/cards as shipped: every row valid, every card a row names is there (the ledger stays
    out of a public copy, so its test skips there)"""
    @unittest.skipUnless(os.path.exists(LEDGER), 'no runs/screen.jsonl in this copy')
    def test_rows_and_cards(self):
        rows = [json.loads(x) for x in S.read(LEDGER).splitlines() if x.strip()]
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(S.validate(r), [], r['id'])
            self.assertTrue(os.path.exists(os.path.join(ROOT, 'data', 'cards', r['card'] + '.json')), r['card'])
        sage = next(r for r in rows if r['arm'] == 'sage-a1.5')
        self.assertEqual((sage['shapes']['requests'], sage['shapes']['cap_hit']), (14, 11), 'the 10-21 Sage round: 11 of 14 requests at the cap')

    def test_cards_have_a_source_for_every_value(self):
        d = os.path.join(ROOT, 'data', 'cards')
        for f in sorted(os.listdir(d)):
            c = json.loads(S.read(os.path.join(d, f)))
            for g in ('sampling', 'thinking', 'template', 'limits'):
                for k, x in c[g].items():
                    self.assertTrue(x['src'], (f, g, k))
                    self.assertTrue(('https://' in x['src']) if x['v'] is not None else x['tag'] == 'UNKNOWN', (f, g, k, x))


if __name__ == '__main__':
    unittest.main()
