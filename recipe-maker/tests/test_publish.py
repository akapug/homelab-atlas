"""What keeps the repo publishable: the run ledger in sealed gzip segments + an open file (src/ledger.py, src/ledger.js),
heat maps over 1 MiB written gzipped (tools/heatmap.py), no tracked file over 1 MiB, the page's data split out of index.html
(build.py page/*.js), the curated public copy (tools/curate.py by publish/manifest.json) and its scan (tools/export_atlas.sh)."""
import datetime
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import ledger as L  # noqa: E402
import heatmap as HM  # noqa: E402

UTC = datetime.timezone.utc
DOC = 'docs' + '/'   # planted lab-note pointers are built at run time, so this file never matches the export scan
CURATE = os.path.join(ROOT, 'tools', 'curate.py')
EXPORT = os.path.join(ROOT, 'tools', 'export_atlas.sh')   # neither ships in the public copy: their tests skip there
NOW = datetime.datetime(2026, 10, 20, 12, 0, tzinfo=UTC)


def row(i, day, model='m1'):
    return {'id': 'r%02d' % i, 'when': '2026-10-%02dT10:00Z' % day, 'model': model, 'pad': 'x' * 200}


def write_rows(path, rows):
    with open(path, 'w') as f:
        f.writelines(json.dumps(r) + '\n' for r in rows)


def node_rows(path):
    js = "console.log(JSON.stringify(require(process.argv[1]).rows(process.argv[2]).map(r => r.id)))"
    p = subprocess.run(['node', '-e', js, os.path.join(ROOT, 'src', 'ledger.js'), path], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


class Ledger(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix='ledger-')
        self.addCleanup(shutil.rmtree, self.d)
        self.p = os.path.join(self.d, 'runs.jsonl')
        self.rows = [row(i, d) for i, d in enumerate((14, 15, 16, 18, 19, 20, 20))]
        write_rows(self.p, self.rows)

    def ids(self):
        return [r['id'] for r in L.rows(self.p)]

    def test_small_open_file_is_left_alone(self):
        self.assertEqual(L.pack(self.p, max_bytes=1 << 20, keep_days=1, now=NOW), (0, None))
        self.assertEqual(L.segments(self.p), [])

    def test_pack_seals_the_older_leading_rows_and_keeps_order(self):
        k, seg = L.pack(self.p, max_bytes=100, keep_days=1, now=NOW)
        self.assertEqual((k, os.path.basename(seg)), (5, 'runs.0001.jsonl'))   # 14..19 at 10:00Z are older than 1 day
        self.assertEqual(self.ids(), [r['id'] for r in self.rows])
        with open(self.p) as f:
            self.assertEqual([json.loads(x)['id'] for x in f], ['r05', 'r06'])
        write_rows(self.p, self.rows[5:] + [row(7, 21)])   # a later append, then a second pack seals into segment 2
        k2, seg2 = L.pack(self.p, max_bytes=100, keep_days=0, now=datetime.datetime(2026, 10, 21, 12, 0, tzinfo=UTC))
        self.assertEqual((k2, os.path.basename(seg2)), (3, 'runs.0002.jsonl'))
        self.assertEqual(self.ids(), [r['id'] for r in self.rows] + ['r07'])
        self.assertEqual(node_rows(self.p), self.ids(), 'node reads the same rows in the same order')

    def test_pack_stops_at_the_first_young_or_undated_row(self):
        rows = [row(0, 14), {'id': 'nodate', 'model': 'm1'}, row(2, 14)]
        write_rows(self.p, rows)
        self.assertEqual(L.pack(self.p, max_bytes=10, keep_days=1, now=NOW)[0], 1)
        self.assertEqual([r['id'] for r in L.rows(self.p)], ['r00', 'nodate', 'r02'])

    def test_young_rows_still_fit_in_git(self):
        rows = [row(i, 20) for i in range(40)]   # all younger than keep_days, together over the limit
        write_rows(self.p, rows)
        size = os.path.getsize(self.p)
        limit = size // 3
        k, seg = L.pack(self.p, max_bytes=100, keep_days=1, now=NOW, limit=limit)
        self.assertLessEqual(os.path.getsize(self.p), limit)
        self.assertGreater(os.path.getsize(self.p) + len(json.dumps(rows[k - 1])) + 1, limit, 'only the fewest rows seal')
        self.assertEqual(self.ids(), [r['id'] for r in rows], 'contents and order kept')
        with open(seg) as f:
            self.assertEqual([json.loads(x) for x in f], rows[:k])
        write_rows(self.p, rows)
        for f in L.segments(self.p):
            os.remove(f)
        self.assertEqual(L.pack(self.p, max_bytes=100, keep_days=1, now=NOW, limit=size)[0], 0, 'under the limit young rows stay')

    def test_soft_pack_threshold_cannot_override_the_hard_limit(self):
        rows = [row(i, 20) for i in range(40)]
        write_rows(self.p, rows)
        size = os.path.getsize(self.p)
        limit = size // 3
        k, seg = L.pack(self.p, max_bytes=size + 1, keep_days=1, now=NOW, limit=limit)
        self.assertLessEqual(os.path.getsize(self.p), limit)
        self.assertGreater(k, 0)
        self.assertTrue(os.path.isfile(seg))
        self.assertEqual(self.ids(), [r['id'] for r in rows])

    def test_pack_all_and_a_ledger_of_segments_only(self):
        self.assertEqual(L.pack(self.p, everything=True)[0], len(self.rows))
        os.remove(self.p)
        self.assertTrue(L.exists(self.p))
        self.assertEqual(self.ids(), [r['id'] for r in self.rows])
        self.assertEqual(node_rows(self.p), self.ids())
        with open(L.segments(self.p)[0], 'rb') as f:
            self.assertEqual(len(f.read().splitlines()), len(self.rows))

    def test_older_gzip_segments_still_read_before_new_plain_ones(self):
        with gzip.open(os.path.join(self.d, 'runs.0001.jsonl.gz'), 'wt') as f:
            f.writelines(json.dumps(r) + '\n' for r in self.rows[:2])
        write_rows(self.p, self.rows[2:])
        k, seg = L.pack(self.p, max_bytes=100, keep_days=1, now=NOW)
        self.assertEqual((k, os.path.basename(seg)), (3, 'runs.0002.jsonl'))
        with open(seg, 'rb') as f:
            self.assertNotEqual(f.read(2), b'\x1f\x8b', 'new segments are plain text')
        self.assertEqual(self.ids(), [r['id'] for r in self.rows])
        self.assertEqual(node_rows(self.p), self.ids())

    def test_plain_pack_for_a_public_copy(self):
        k, seg = L.pack(self.p, everything=True)
        segs = L.segments(self.p)
        self.assertEqual(k, len(self.rows))
        self.assertEqual([os.path.basename(x) for x in segs], ['runs.0001.jsonl'])
        self.assertEqual(seg, segs[-1])
        with open(segs[0], 'rb') as f:
            self.assertEqual(f.read(2) != b'\x1f\x8b', True, 'plain text, not gzip')
        self.assertEqual(self.ids(), [r['id'] for r in self.rows])
        self.assertEqual(node_rows(self.p), self.ids())

    def test_plain_pack_splits_at_the_segment_cap(self):
        keep = L.MAX_OPEN_BYTES
        L.MAX_OPEN_BYTES = 600   # each row is about 260 bytes: two per segment
        self.addCleanup(setattr, L, 'MAX_OPEN_BYTES', keep)
        L.pack(self.p, everything=True)
        segs = L.segments(self.p)
        self.assertEqual([os.path.basename(x) for x in segs], ['runs.%04d.jsonl' % i for i in range(1, 5)])
        self.assertTrue(all(os.path.getsize(x) <= 600 for x in segs))
        self.assertEqual(self.ids(), [r['id'] for r in self.rows])
        self.assertEqual(node_rows(self.p), self.ids())
        self.assertEqual(L.chunks([b'x' * 700, b'y'], 600), [[b'x' * 700], [b'y']], 'a longer line is a group of its own')

    def test_oversized_later_row_refusal_leaves_the_ledger_unchanged(self):
        write_rows(os.path.join(self.d, 'runs.0001.jsonl'), [row(9, 14)])
        write_rows(self.p, [row(0, 14), dict(row(1, 14), pad='x' * L.GIT_FILE_LIMIT)])
        def snapshot():
            return {name: Path(self.d, name).read_bytes() for name in sorted(os.listdir(self.d))}
        before = snapshot()
        ids = self.ids()
        self.assertEqual(node_rows(self.p), ids)
        with self.assertRaisesRegex(SystemExit, r'ledger pack: a row of \d+ bytes is over \d+'):
            L.pack(self.p, everything=True)
        with self.subTest(conservation='inventory and exact bytes'):
            self.assertTrue(snapshot() == before, 'refused packing must not install any earlier chunk')
        with self.subTest(conservation='Python logical order'):
            self.assertEqual(self.ids(), ids)
        with self.subTest(conservation='Node logical order'):
            self.assertEqual(node_rows(self.p), ids)

    def test_record_run_append_packs_the_open_file(self):
        import record_run as RR
        real = json.loads(next(L.lines(os.path.join(ROOT, 'runs', 'runs.jsonl'))))
        write_rows(self.p, [dict(real, id='old-%d' % i, when='2026-10-01T00:00Z') for i in range(3)])
        keep = L.MAX_OPEN_BYTES
        L.MAX_OPEN_BYTES = 10
        self.addCleanup(setattr, L, 'MAX_OPEN_BYTES', keep)
        RR.append(dict(real, id='new', when=datetime.datetime.now(UTC).strftime('%Y-%m-%dT%H:%MZ')), self.p)
        self.assertEqual([os.path.basename(s) for s in L.segments(self.p)], ['runs.000%d.jsonl' % i for i in (1, 2, 3)], 'a 10-byte cap seals each row alone')
        self.assertEqual([r['id'] for r in L.rows(self.p)], ['old-0', 'old-1', 'old-2', 'new'])
        with open(self.p) as f:
            self.assertEqual([json.loads(x)['id'] for x in f], ['new'])
        with self.assertRaises(ValueError):   # a sealed id is still a duplicate
            RR.append(dict(real, id='old-1'), self.p)


class FileSizes(unittest.TestCase):
    def test_no_tracked_file_over_the_limit(self):
        p = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, capture_output=True)
        if p.returncode:
            self.skipTest('not a git checkout')
        ledger = os.path.join('runs', 'runs.jsonl')
        big = [f for f in p.stdout.decode().split('\0') if f and f != ledger and os.path.exists(os.path.join(ROOT, f))
               and os.path.getsize(os.path.join(ROOT, f)) > L.GIT_FILE_LIMIT]
        self.assertEqual(big, [], 'over 1 MiB: build outputs are not tracked, big data ships gzipped')

    def test_the_open_ledger_fits_once_its_writer_packs(self):
        src = os.path.join(ROOT, 'runs', 'runs.jsonl')
        if not os.path.exists(src):
            self.skipTest('no open ledger')
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'runs.jsonl')
            shutil.copyfile(src, p)
            L.pack(p)
            self.assertLessEqual(os.path.getsize(p), L.GIT_FILE_LIMIT)


class HeatGzip(unittest.TestCase):
    def test_big_maps_are_written_gzipped_and_load(self):
        d = tempfile.mkdtemp(prefix='heat-')
        self.addCleanup(shutil.rmtree, d)
        src = sorted(f for f in os.listdir(os.path.join(ROOT, 'data', 'heat')) if f.endswith('.heatmap.json'))[0]
        m = HM.load(os.path.join(ROOT, 'data', 'heat', src))
        m.pop('_file', None)
        p = os.path.join(d, 'a.heatmap.json')
        self.assertEqual(HM.write(p, m), p)
        keep = HM.GZIP_OVER
        HM.GZIP_OVER = 100
        self.addCleanup(setattr, HM, 'GZIP_OVER', keep)
        self.assertEqual(HM.write(p, m), p + '.gz')
        self.assertFalse(os.path.exists(p), 'the stale plain copy is removed')
        got = HM.load_dir(d)
        self.assertEqual(list(got), [m['id']])
        self.assertEqual(got[m['id']]['digest'], m['digest'])
        with gzip.open(p + '.gz', 'rt') as f:
            self.assertEqual(f.read(), HM.dumps(m) + '\n')


@unittest.skipUnless(os.path.exists(CURATE), 'the publishing tool is not in this copy')
class IkBuildIds(unittest.TestCase):
    """ids of our engine builds in the copy: a rewritten commit becomes its sha on the public fork, a public commit stays,
    any other build of ours becomes the unpublished label, and ids of anything else (digests, other projects) stay"""
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix='ikids-')
        self.addCleanup(shutil.rmtree, self.d)
        git = lambda repo, *a: subprocess.run(['git', '-C', repo, '-c', 'user.name=t', '-c', 'user.email=t' + '@' + 'example.org', *a],
                                              check=True, capture_output=True, text=True).stdout.strip()
        priv, fork = os.path.join(self.d, 'priv'), os.path.join(self.d, 'fork')
        sha = {}
        for repo in (priv, fork):
            os.makedirs(repo)
            git(repo, 'init', '-q')
        for name in ('up', 'mapped', 'gone', 'unshipped'):     # up: upstream; mapped: rewritten + on the fork; the rest never public
            git(priv, 'commit', '-q', '--allow-empty', '-m', name)
            sha[name] = git(priv, 'rev-parse', 'HEAD')
            if name == 'up':
                git(priv, 'update-ref', 'refs/remotes/upstream/main', sha['up'])
        git(fork, 'commit', '-q', '--allow-empty', '-m', 'rewritten')
        sha['new'] = git(fork, 'rev-parse', 'HEAD')
        git(fork, 'update-ref', 'refs/remotes/fork/main', sha['new'])
        self.cmap = os.path.join(self.d, 'commit-map')
        with open(self.cmap, 'w') as f:   # 'unshipped' was rewritten too, but its new sha is on no public ref
            f.write('old new\n%s %s\n%s %s\n' % (sha['mapped'], sha['new'], sha['unshipped'], 'e' * 40))
        self.sha = sha
        self.cfg = {'commit_map': self.cmap, 'private': [priv], 'public': {priv: ['refs/remotes/upstream'], fork: ['refs/remotes/fork']},
                    'unpublished': 'unpublished build', 'unpublished_tag': 'unpublished'}
        import curate
        self.C = curate

    def test_ids_map_stay_or_go(self):
        s = self.sha
        R = self.C.rules({'rewrite': [], 'ik_builds': self.cfg}, '')
        text = 'ik main %s; stack %s (dvocab-%s-nc); ik h2dstage-%s; rows-%s; build %s; digest sha256:%s; date 20261017' % (
            s['up'][:8], s['mapped'][:8], s['mapped'][:12], s['gone'][:8], s['unshipped'][:10], s['gone'][:9], 'ab12cd34ef56')
        self.assertEqual(self.C.rewrite(text, R),
                         'ik main %s; stack %s (dvocab-%s-nc); ik h2dstage-unpublished; rows-unpublished; build unpublished build; digest sha256:%s; date 20261017' % (
                             s['up'][:8], s['new'][:8], s['new'][:12], 'ab12cd34ef56'))
        row = self.C.rewrite_json({'id': 'r-' + s['gone'][:8], 'engine_build': 'h2dmain-' + s['mapped'][:8]}, R)
        self.assertEqual(row, {'id': 'r-' + s['gone'][:8], 'engine_build': 'h2dmain-' + s['new'][:8]}, 'ids are never rewritten')

    def test_uniform_character_build_ids_use_the_commit_map(self):
        with open(self.cmap, 'a') as f:
            for old in ('2' * 40, 'a' * 40):
                f.write('%s %s\n' % (old, self.sha['new']))
        R = self.C.rules({'rewrite': [], 'ik_builds': self.cfg}, '')
        for old in ('2' * 8, 'a' * 8):
            with self.subTest(old=old):
                self.assertEqual(self.C.rewrite('stack ' + old, R), 'stack ' + self.sha['new'][:8])
                self.assertEqual(self.C.rewrite('dvocab-' + old + '-nc', R), 'dvocab-' + self.sha['new'][:8] + '-nc')
        text = 'date 20261017; digest sha256:1234567890; digest sha256:abcdefabcdef'
        self.assertEqual(self.C.rewrite(text, R), text)

    def test_shipped_tests_get_the_same_build_names(self):
        man = {'rewrite': [['\\bdocs/\\d+', 'lab notes', 'a doc']], 'ik_builds': self.cfg}
        doc = 'docs/' + '42'   # split: the export scan reads this file
        line = "engine_build: 'h2dstage-%s' // %s" % (self.sha['gone'][:8], doc)
        self.assertEqual(self.C.rewrite(line, self.C.ik_rules(man)), "engine_build: 'h2dstage-unpublished' // " + doc)
        self.assertEqual(self.C.rewrite(line, self.C.rules(man, '')), "engine_build: 'h2dstage-unpublished' // lab notes")

    def test_export_refuses_without_the_commit_map(self):
        os.remove(self.cmap)
        with self.assertRaises(SystemExit):
            self.C.curate(os.path.join(self.d, 'out'), {'ik_builds': self.cfg, 'rewrite': []}, root=self.d)


@unittest.skipUnless(os.path.exists(CURATE), 'the publishing tool is not in this copy')
class Curate(unittest.TestCase):
    """tools/curate.py on a planted repo: the manifest decides what ships, the ledger is curated (skipped models out, fields
    dropped, degraded notes reduced to their class, sources replaced), pointers into lab notes are rewritten, ids are not"""
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix='curate-')
        self.addCleanup(shutil.rmtree, self.d)
        src = self.src = os.path.join(self.d, 'repo')
        man = {'ship': ['README.md', 'build.py', 'runs/runs.jsonl', 'runs/labels.json', 'runs/levers.json', 'tests/*.js'],
               'never': {'runs/secret*': 'x'},
               'runs': {'skip_models': {'zz9': 'held back'}, 'drop': ['impression', 'measured.notes', 'recipe.src'],
                        'sources': 'lab notes', 'placeholder': ['measured.src'], 'labels_keep': ['class', 'phases', 'evidence'],
                        'labels_placeholder': ['evidence']},
               'rewrite': [['\\bdocs/\\d+', 'lab notes', 'a doc']], 'rewrite_skip': ['tests/*.js'],
               'results': 'results/RESULTS.md'}
        def run(i, model, dec, deg=None, rid=None):
            m = {'outcome': 'ok', 'decode_tps': {'n4': dec, 'n0': dec / 2}, 'notes': 'see ' + DOC + '42', 'src': {'decode_tps': 'b1:~/runs/x.json'}}
            if deg:
                m['degraded'] = [deg]
            return {'id': rid or 'r%d' % i, 'when': '2026-10-%02dT10:00Z' % (10 + i), 'box': 'b1', 'model': model, 'engine': 'ik',
                    'recipe': {'id': rid or 'r%d' % i, 'label': 'per ' + DOC + '41', 'src': 'x'}, 'measured': m,
                    'sources': ['a long private run log path'], 'impression': 'felt fast'}
        rows = [run(0, 'ok1', 10.0), run(1, 'zz9', 99.0), run(2, 'ok1', 12.0, 'm0 prompt1 decoded 3.1 tok/s while ' + DOC + '42 ran', 'docs-42-run'),
                run(3, 'ok1', 11.0)]
        files = {
            'README.md': 'see ' + DOC + '45 for more\n',
            'build.py': "SRC = '" + DOC + "41 G6'\n",
            'runs/secret.txt': 'never\n',
            'tests/t.js': "recipe('r3.json'); // " + DOC + "9 stays in test code\n",
            'recipes/r3.json': json.dumps({'id': 'r3', 'note': DOC + '7'}),
            'recipes/r0.json': json.dumps({'id': 'r0'}),
            'runs/runs.jsonl': ''.join(json.dumps(r) + '\n' for r in rows),
            'runs/labels.json': json.dumps({'_doc': 'labels', 'docs-42-run': {'causes': [{'class': 'interference', 'phases': ['m0'], 'evidence': 'private'}]}}),
            'runs/levers.json': json.dumps({'levers': [{'lever': 'a', 'before': [{'run': 'r0'}], 'after': [{'run': 'r3'}]},
                                                      {'lever': 'b', 'before': [{'run': 'r1'}], 'after': [{'run': 'r3'}]}]}),
        }
        for rel, body in files.items():
            os.makedirs(os.path.dirname(os.path.join(src, rel)) or src, exist_ok=True)
            with open(os.path.join(src, rel), 'w') as f:
                f.write(body)
        for c in (['init', '-q'], ['add', '-A']):
            subprocess.run(['git', *c], cwd=src, check=True, capture_output=True)
        import curate
        self.C, self.man = curate, man
        self.out = os.path.join(self.d, 'out')
        self.rows, self.recipes, self.files = curate.curate(self.out, man, root=src, home='/home/' + 'someone')

    def test_what_ships(self):
        have = sorted(os.path.relpath(os.path.join(a, f), self.out) for a, _, fs in os.walk(self.out) for f in fs)
        # recipes: r3 is both the fastest clean run of (b1, ok1) and named by a test; r0 is neither; secret.txt is 'never'
        self.assertEqual(have, ['README.md', 'build.py', 'recipes/r3.json', 'runs/labels.json', 'runs/levers.json',
                                'runs/runs.jsonl', 'tests/t.js'])

    def test_ledger_is_curated(self):
        rows = L.rows(os.path.join(self.out, 'runs', 'runs.jsonl'))
        self.assertEqual([r['id'] for r in rows], ['r0', 'docs-42-run', 'r3'], 'the skipped model is out; ids are never rewritten')
        for r in rows:
            self.assertNotIn('impression', r)
            self.assertNotIn('notes', r['measured'])
            self.assertNotIn('src', r['recipe'])
            self.assertEqual(r['sources'], ['lab notes'])
            self.assertEqual(r['measured']['src'], {'decode_tps': 'lab notes'}, 'each number keeps a source key')
            self.assertEqual(r['recipe']['label'], 'per lab notes')
        bad = rows[1]['measured']['degraded']
        self.assertEqual(bad, ['interference (m0 prompt)'], 'the label class plus the phase the note names, never the note')
        with open(os.path.join(self.out, 'runs', 'labels.json')) as f:
            self.assertEqual(json.load(f)['docs-42-run'], {'causes': [{'class': 'interference', 'phases': ['m0'], 'evidence': 'lab notes'}]})
        with open(os.path.join(self.out, 'runs', 'levers.json')) as f:
            self.assertEqual([lv['lever'] for lv in json.load(f)['levers']], ['a'], 'a lever on a left-out run goes')

    def test_rewrite_and_its_skip_list(self):
        with open(os.path.join(self.out, 'README.md')) as f:
            self.assertEqual(f.read(), 'see lab notes for more\n')
        with open(os.path.join(self.out, 'build.py')) as f:
            self.assertEqual(f.read(), "SRC = 'lab notes G6'\n")
        with open(os.path.join(self.out, 'tests', 't.js')) as f:
            self.assertIn(DOC + '9', f.read(), 'test code ships as written')
        with open(os.path.join(self.out, 'recipes', 'r3.json')) as f:
            self.assertEqual(json.load(f), {'id': 'r3', 'note': 'lab notes'})

    def test_manifest_strips_agent_names_and_ledger_prefixes(self):
        with open(os.path.join(ROOT, 'publish', 'manifest.json')) as f:
            R = self.C.rules(json.load(f), '')
        agent = 'lab-' + 'cla' + 'ude'
        self.assertEqual(self.C.rewrite('package (ik ' + 'hl' + '332-stack 0123abcd)', R), 'package (ik stack 0123abcd)')
        self.assertEqual(self.C.rewrite('stopped (' + agent + ' GO 10-20 23:12Z); x', R), 'stopped (approved); x')
        self.assertEqual(self.C.rewrite('asked by gpu-box-' + 'cla' + 'ude', R), 'asked by an agent')

    def test_every_home_path_becomes_tilde(self):
        R = self.C.rules({'rewrite': []}, '/home/' + 'someone')
        for raw in ('/home/' + 'someone/m.gguf', '/home/' + 'user/m.gguf', '/Users/' + 'x/m.gguf'):
            self.assertEqual(self.C.rewrite(raw, R), '~/m.gguf', raw)

    def test_degraded_class_without_a_label(self):
        self.assertEqual(self.C.degraded_class('CUDA OOM at chunk 8192', []), 'out of memory')
        self.assertEqual(self.C.degraded_class('n4 prompt2 slow: 2.1 tok/s', []), 'slowdown (n4 prompt)')
        self.assertEqual(self.C.degraded_class('something else', []), 'degraded')
        self.assertIn('--expert-stats', self.C.degraded_class('run with --expert-stats on', []))

    def test_results_page(self):
        md = self.C.results_md(self.rows, {'boxes': {'b1': {'cpu': {'desc': '1x test CPU'}}}, 'models': {'ok1': {'name': 'Model One'}}},
                               'accuracy text', self.man)
        self.assertIn('| **b1** | 1x test CPU |', md)
        self.assertIn('| b1 | Model One | ik | per lab notes | **11.00** (n4) |', md)
        self.assertIn('zz9 runs (held back)', md)
        self.assertIn('accuracy text', md)


@unittest.skipUnless(os.path.exists(EXPORT), 'the publishing tool is not in this copy')
class Scan(unittest.TestCase):
    def test_scan_reads_gzip_and_refuses_big_files_and_lab_pointers(self):
        d = tempfile.mkdtemp(prefix='scan-')
        self.addCleanup(shutil.rmtree, d)
        term = 'Zebe' + 'dee'   # built at run time so this file never matches its own scan
        terms = os.path.join(d, 'terms.txt')
        with open(terms, 'w') as f:
            f.write(term + '\n')
        planted = os.path.join(d, 'planted')
        os.makedirs(planted)
        with open(os.path.join(planted, 'a.jsonl.gz'), 'wb') as f:
            f.write(gzip.compress(('{"note": "ask %s"}\n' % term.lower()).encode()))
        with open(os.path.join(planted, 'big.txt'), 'w') as f:
            f.write('x' * (L.GIT_FILE_LIMIT + 1))
        with open(os.path.join(planted, 'p.txt'), 'w') as f:
            f.write('see docs/' + '42 and ' + 'hl/' + '314\n')
        with open(os.path.join(planted, 'q.txt'), 'w') as f:
            f.write('copy of ~/.' + 'arc/x\nfrom ' + 'hl' + '256-providers\nsee the seat ' + 'branch b70-rig/x\n')
        with open(os.path.join(planted, 'r.txt'), 'w') as f:
            f.write('"tags": ["full-stack", "hl-' + '332"]\n')
        p = subprocess.run(['bash', os.path.join(ROOT, 'tools', 'export_atlas.sh'), '--scan-only', planted, '--terms', terms],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 1, p.stdout)
        self.assertIn('a.jsonl.gz:1: private term', p.stdout)
        self.assertIn('a.jsonl.gz:0: archive file', p.stdout)
        self.assertIn('big.txt:0: file over 1 MiB', p.stdout)
        self.assertIn('p.txt:1: lab notes pointer', p.stdout)
        self.assertIn('p.txt:1: internal id', p.stdout)
        self.assertIn('q.txt:1: lab notes pointer', p.stdout)
        self.assertIn('q.txt:2: internal id', p.stdout)
        self.assertIn('q.txt:3: lab notes pointer', p.stdout)
        self.assertIn('r.txt:1: internal id', p.stdout)

    def test_scan_admits_only_the_manifest_signoff(self):
        with open(os.path.join(ROOT, 'publish', 'manifest.json')) as f:
            signoff = json.load(f)['signoff']
        d = tempfile.mkdtemp(prefix='scan-')
        self.addCleanup(shutil.rmtree, d)
        terms = os.path.join(d, 'terms.txt')
        open(terms, 'w').close()
        planted = os.path.join(d, 'planted')
        os.makedirs(planted)
        with open(os.path.join(planted, 'README.md'), 'w') as f:
            f.write('# x\n\n---\n\n' + signoff + '\n')
        run = lambda: subprocess.run(['bash', os.path.join(ROOT, 'tools', 'export_atlas.sh'), '--scan-only', planted, '--terms', terms],
                                     capture_output=True, text=True)
        self.assertEqual(run().returncode, 0, run().stdout)
        with open(os.path.join(planted, 'README.md'), 'a') as f:
            f.write('written by ' + 'cla' + 'ude\n')
        self.assertEqual(run().returncode, 1)


class Page(unittest.TestCase):
    """build.py writes index.html (no data inside) + page/*.js, each under the page file limit; the scripts rebuild data.json"""
    def test_page_parts_rebuild_the_data(self):
        page = os.path.join(ROOT, 'page')
        if not os.path.exists(os.path.join(ROOT, 'data.json')) or not os.path.isdir(page):
            self.skipTest('no build outputs (run python3 build.py)')
        with open(os.path.join(ROOT, 'index.html')) as f:
            html = f.read()
        self.assertLess(len(html), 8 << 10, 'index.html holds markup and script tags only')
        srcs = re.findall(r'<script src="([^"]+)"></script>', html)
        self.assertEqual(srcs[0], 'page/data.js')
        self.assertEqual(srcs[-2:], ['src/model.js', 'src/app.js'])
        for s in srcs:
            self.assertTrue(os.path.exists(os.path.join(ROOT, s)), s)
            self.assertLessEqual(os.path.getsize(os.path.join(ROOT, s)), L.GIT_FILE_LIMIT, s)
        js = ("const vm = require('vm'), fs = require('fs'); const ctx = { window: {} };"   # node -e: argv[1:] are the files
              "process.argv.slice(1).forEach(f => vm.runInNewContext(fs.readFileSync(f, 'utf8'), ctx));"
              "process.stdout.write(JSON.stringify(ctx.window.FIT_DATA));")
        p = subprocess.run(['node', '-e', js] + [os.path.join(ROOT, s) for s in srcs if s.startswith('page/')], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        with open(os.path.join(ROOT, 'data.json')) as f:
            self.assertEqual(json.loads(p.stdout), json.load(f), 'the page scripts put back exactly data.json')


if __name__ == '__main__':
    unittest.main()
