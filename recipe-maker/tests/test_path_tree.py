"""Path tracker: builds from the real families file and RM paths; negative verdicts keep their evidence."""
import json, os, subprocess, sys, tempfile, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import path_tree  # noqa: E402


def nodes(data):
    for f in data['families']:
        for m in f['models']:
            for n in m['nodes']:
                yield m, n


class PathTree(unittest.TestCase):
    def test_builds_from_the_repo_data(self):
        data = path_tree.build(os.path.join(ROOT, 'data', 'path-families.json'), ROOT)
        assert data['families'] and any(m['rm_path'] for m, _ in nodes(data))
        states = {n['state'] for _, n in nodes(data)}
        assert states <= {'kept', 'caveat', 'blocked', 'open', 'dead', 'untested', 'live'}, states


    def test_rm_verdicts_carry_through(self):
        data = path_tree.build(os.path.join(ROOT, 'data', 'path-families.json'), ROOT)
        for m, n in nodes(data):
            if not n.get('from_rm'):
                continue
            v = n['verdict'] if isinstance(n['verdict'], str) else ''
            rm = json.load(open(os.path.join(ROOT, 'data', 'paths', n['from_rm'] + '.json')))
            src = {x['id']: x['verdict'] for x in rm['nodes']}[n['id']]
            if src['state'] == 'blocked':
                assert n['state'] == 'blocked', n['id']
                assert n['next'].startswith('untried: '), n['id']


    def test_a_dead_end_always_lists_its_evidence(self):
        data = path_tree.build(os.path.join(ROOT, 'data', 'path-families.json'), ROOT)
        for _, n in nodes(data):
            if n['state'] == 'dead':
                assert n.get('ruled_out') or n.get('superseded_by'), n['id']


    def test_an_unproven_negative_becomes_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            fam = {'about': 'x', 'families': [{'id': 'f', 'name': 'F', 'models': [
                {'id': 'm', 'name': 'M', 'extra': [{'id': 'e1', 'short': 'tried once', 'state': 'dead'}]}]}]}
            p = os.path.join(d, 'fam.json')
            json.dump(fam, open(p, 'w'))
            data = path_tree.build(p, ROOT)
            n = data['families'][0]['models'][0]['nodes'][0]
            assert n['state'] == 'blocked' and n['next'].startswith('unproven negative')


    def test_cli_writes_one_self_contained_page(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, 'pt.html')
            r = subprocess.run([sys.executable, '-I', os.path.join(ROOT, 'tools', 'path_tree.py'), '--out', out],
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            html = open(out).read()
            assert '/*DATA*/null' not in html and '"families"' in html


if __name__ == '__main__':
    unittest.main()
