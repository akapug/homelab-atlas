"""the one trust field: evidence = seen | inferred | unknown on every tagged value (build.py EVIDENCE), the build refusing an
unknown tag or a src that contradicts its tag, and runs/external.jsonl rows (community measurements: inferred, missing fields
named). The node side (roll-ups, external predictions) is in tests/test_accuracy.js."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from test_build_inputs import COPY, ROOT

sys.path.insert(0, os.path.join(ROOT, 'src'))
import ledger as L  # noqa: E402

HEAD_END = '# ---------------- GGUF headers'


def evidence_ns():
    """build.py's evidence table and evidence_of(), without running the build"""
    src = open(os.path.join(ROOT, 'build.py')).read()
    ns = {'__file__': os.path.join(ROOT, 'build.py')}
    exec(compile(src[:src.index(HEAD_END)], 'build.py', 'exec'), ns)
    return ns


class EvidenceOf(unittest.TestCase):
    def setUp(self):
        self.ns = evidence_ns()
        self.of = self.ns['evidence_of']

    def test_tags_map_to_three_classes(self):
        self.assertEqual(self.of('MEASURED', 'probe 10-19'), 'seen')
        for tag in ('FIT', 'ESTIMATE', 'PRIOR', 'READ', 'MEASURED-BY-THEM', 'CLAIMED', 'M-ext', 'INFERRED-BY-US'):
            self.assertEqual(self.of(tag, 'a source'), 'inferred', tag)
        self.assertEqual(self.of('UNKNOWN', ''), 'unknown')
        self.assertEqual(self.of('MEASURED-UNKNOWN', 'unit unknown'), 'unknown')
        self.assertEqual(set(self.ns['EVIDENCE']['by_tag'].values()), {'seen', 'inferred', 'unknown'})

    def test_read_of_our_own_file_is_seen(self):
        self.assertEqual(self.of('READ', 'GGUF header of /models/x.gguf'), 'seen')
        self.assertEqual(self.of('READ', 'a vendor spec sheet'), 'inferred')

    def test_unknown_tag_is_a_build_error(self):
        with self.assertRaisesRegex(ValueError, 'no evidence class'):
            self.of('GUESS', 'x')

    def test_src_that_contradicts_its_tag_is_a_build_error(self):
        with self.assertRaisesRegex(ValueError, 'contradicts'):
            self.of('MEASURED', 'M-ext: their card')
        with self.assertRaisesRegex(ValueError, 'contradicts'):
            self.of('ESTIMATE', 'UNKNOWN: never measured')
        self.assertEqual(self.of('ESTIMATE', 'ESTIMATE: our guess'), 'inferred')

    def test_n_carries_evidence(self):
        self.assertEqual(self.ns['n'](3, 'FIT', 'fit to 4 points')['evidence'], 'inferred')


def walk(x, out):
    if isinstance(x, list):
        for y in x:
            walk(y, out)
    elif isinstance(x, dict):
        if isinstance(x.get('tag'), str):
            out.append(x)
        for y in x.values():
            walk(y, out)
    return out


class BuiltData(unittest.TestCase):
    """data.json as built (the suite runs python3 build.py first): every tagged value is stamped, the counts add up, seen
    values cite ledger rows"""
    @classmethod
    def setUpClass(cls):
        cls.data = json.load(open(os.path.join(ROOT, 'data.json')))

    def test_every_tagged_value_has_evidence_and_a_source(self):
        vals = walk({k: v for k, v in self.data.items() if k != 'meta'}, [])
        self.assertGreater(len(vals), 500)
        self.assertEqual([v for v in vals if v.get('evidence') not in ('seen', 'inferred', 'unknown')], [])
        nosrc = [v for v in vals if v['evidence'] != 'unknown' and not str(v.get('src') or '').strip()]
        self.assertEqual(nosrc, [])
        counts = self.data['meta']['evidence']['counts']
        for k in ('seen', 'inferred', 'unknown'):
            self.assertEqual(counts.get(k, 0), sum(1 for v in vals if v['evidence'] == k), k)

    def test_seen_values_cite_ledger_rows_where_their_src_names_one(self):
        rows = [r for f in ('runs.jsonl', 'bench.jsonl') for r in L.rows(os.path.join(ROOT, 'runs', f))]
        if not any(len(s) >= 12 for r in rows for s in r.get('sources') or []):
            self.skipTest('no ledger row names a citable source (a curated public copy keeps one placeholder per row)')
        ids = {r['id'] for r in rows}
        cited = [v for v in walk(self.data, []) if v.get('runs')]
        self.assertGreater(len(cited), 10)
        self.assertTrue(all(v['evidence'] == 'seen' and set(v['runs']) <= ids for v in cited))

    def test_external_rows_are_inferred_with_their_url(self):
        X = self.data['external']
        rows = X['rows']
        self.assertGreaterEqual(len(rows), 20)
        self.assertEqual({r['hw_class'] for r in rows}, set(X['classes']))
        for r in rows:
            self.assertEqual((r['tag'], r['evidence'], r['src']), ('MEASURED-BY-THEM', 'inferred', r['url']))
            self.assertTrue(r['url'].startswith('https://'))
            self.assertIsInstance(r['missing'], list)
            self.assertGreater(r['measured']['decode_tps'], 0)


class ExternalRowsBuild(unittest.TestCase):
    def test_missing_fields_are_named(self):
        d = tempfile.mkdtemp(prefix='fitviz-ext-')
        self.addCleanup(shutil.rmtree, d)
        for f in COPY:
            s = os.path.join(ROOT, f)
            if os.path.isdir(s):
                shutil.copytree(s, os.path.join(d, f), ignore=shutil.ignore_patterns('__pycache__'))
            elif os.path.exists(s):
                shutil.copy(s, d)
        full = {'ctx': 4096, 'kv_type': 'f16', 'draft_n': 0, 'sampling': 'greedy', 'reasoning_effort': 'n/a', 'workload': 'tg128'}
        rows = [{'id': 'bare', 'url': 'https://example.org/a', 'hw_class': 'apple', 'settings': {'ctx': 4096}, 'measured': {'decode_tps': 10}},
                {'id': 'full', 'url': 'https://example.org/b', 'hw_class': 'amd', 'date': '2026-01-01', 'engine_version': 'b1', 'quant': 'Q4_0',
                 'settings': full, 'measured': {'decode_tps': 10, 'prefill_tps': 100}}]
        with open(os.path.join(d, 'runs', 'external.jsonl'), 'w') as f:
            f.write('\n'.join(json.dumps(r) for r in rows) + '\n')
        env = dict(os.environ, FITVIZ_DIR_PMEM_LANE=os.path.join(d, 'absent'), FITVIZ_DIR_HOMELAB=os.path.join(d, 'absent'),
                   FITVIZ_CONFIG=os.path.join(d, 'absent.json'))
        p = subprocess.run([sys.executable, 'build.py'], cwd=d, env=env, capture_output=True, text=True, timeout=600)
        self.assertEqual(p.returncode, 0, p.stdout[-2000:] + p.stderr[-2000:])
        got = {r['id']: r for r in json.load(open(os.path.join(d, 'data.json')))['external']['rows']}
        self.assertEqual(set(got['bare']['missing']), {'kv_type', 'draft_n', 'sampling', 'reasoning_effort', 'workload', 'date',
                                                       'engine_version', 'quant', 'measured.prefill_tps'})
        self.assertEqual(got['full']['missing'], [])
        self.assertEqual(got['bare']['evidence'], 'inferred')


if __name__ == '__main__':
    unittest.main()
