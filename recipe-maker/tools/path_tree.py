#!/usr/bin/env python3
"""Path tracker: data/path-families.json + Recipe Maker's summit paths (data/paths/*.json, read live) -> one self-contained
path-tree.html, an interactive collapsible tree-table (family -> model -> recipe lineage). Every best number travels with
what binds it and its next test; a negative result without its evidence shows as blocked (levers left), not dead.
Run from the repo: python3 tools/path_tree.py [--out path-tree.html]"""
import argparse, json, os


def fmt_v(v):
    if isinstance(v, list):
        return f'{v[0]:g}-{v[1]:g}'
    return f'{v:g}' if isinstance(v, (int, float)) else str(v)


def rm_best(n):
    pts = n.get('points') or []
    if not pts:
        return '', ''
    one = [p for p in pts if p['agents'] == 1]
    many = max(pts, key=lambda p: p['agents'])
    bits = []
    if one:
        bits.append(f"1 agent {fmt_v(one[0]['per_agent']['v'])}")
    if many['agents'] > 1:
        bits.append(f"{many['agents']} agents {fmt_v(many['per_agent']['v'])} each ({fmt_v(many['aggregate']['v'])} total)")
    tags = {p['per_agent']['tag'] for p in pts}
    tag = 'PREDICTED' if tags == {'PREDICTED'} else 'MIXED' if 'PREDICTED' in tags else 'MEASURED'
    return '; '.join(bits) + ' tok/s', tag


def rm_src(s):
    s = (s or '').strip()
    return 'homelab/' + s[len('homelab '):] if s.startswith('homelab ') else s


def model_nodes(model, rm_dir):
    nodes = []
    if model.get('rm_path'):
        d = json.load(open(os.path.join(rm_dir, 'data', 'paths', model['rm_path'] + '.json')))
        ov = model.get('overlay', {})
        for n in d['nodes']:
            o = ov.get(n['id'], {})
            v = n['verdict']
            st = o.get('state') or {'kept': 'kept', 'dropped': 'dead', 'blocked': 'blocked', 'open': 'open'}.get(v['state'], 'kept')
            if v['state'] == 'blocked' and o.get('state') == 'dead':
                st = 'blocked'   # RM says levers are left: an overlay may not call it dead
            best, tag = rm_best(n)
            nodes.append({'id': n['id'], 'parents': n.get('parents') or [], 'short': o.get('short') or n['label'][:60],
                          'best': best, 'tag': tag, 'state': st,
                          'limit': o.get('limit') or ((v.get('binding') or {}).get('what') or v['why'] if st in ('dead', 'blocked') else ''),
                          'next': o.get('next') or ('untried: ' + '; '.join(v.get('untried') or []) if v.get('untried') else ''),
                          'ruled_out': v.get('ruled_out') or [], 'superseded_by': v.get('superseded_by', ''),
                          'detail': n['label'], 'box': n.get('box', ''), 'verdict': n['verdict']['why'],
                          'levers': n.get('levers', []), 'recipe': n.get('recipe', ''), 'src': rm_src(n.get('results')),
                          'from_rm': model['rm_path']})
    for e in model.get('extra', []):
        nodes.append(dict({'parents': [], 'best': '', 'limit': '', 'next': '', 'detail': '', 'src': ''}, **e))
    for i, u in enumerate(model.get('untested', [])):
        nodes.append({'id': f'{model["id"]}-u{i}', 'parents': [], 'short': u['short'], 'best': '', 'state': 'untested',
                      'limit': u.get('note', ''), 'next': '', 'detail': '', 'src': ''})
    for n in nodes:   # owner 2026-11-15: a negative is a lead until its confounders are ruled out
        if n.get('state') == 'dead' and not n.get('ruled_out') and not n.get('superseded_by'):
            n['state'] = 'blocked'
            n['next'] = n.get('next') or 'unproven negative: record what was ruled out, or the lever to try next'
    ids = {n['id'] for n in nodes}
    for n in nodes:
        n['parents'] = [p for p in n['parents'] if p in ids]
        if n.get('src', '').startswith(('b70-rig/', 'docs/')):
            n['src'] = 'homelab/' + n['src']
    return nodes


def build(fam_file, rm_dir):
    data = json.load(open(fam_file))
    out = {'about': data['about'], 'families': []}
    for f in data['families']:
        fam = {'id': f['id'], 'name': f['name'], 'models': []}
        for m in f['models']:
            fam['models'].append({k: m.get(k, '') for k in ('id', 'name', 'size', 'role', 'state', 'links', 'rm_path')}
                                 | {'nodes': model_nodes(m, rm_dir)})
        out['families'].append(fam)
    return out


if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument('--rm', default=root, help='the Recipe Maker checkout whose data/paths are read')
    ap.add_argument('--families', default=os.path.join(root, 'data', 'path-families.json'))
    ap.add_argument('--template', default=os.path.join(root, 'src', 'path-tree.html'))
    ap.add_argument('--out', default=os.path.join(root, 'path-tree.html'))
    a = ap.parse_args()
    data = json.dumps(build(a.families, a.rm), ensure_ascii=False).replace('</', '<\\/')
    open(a.out, 'w').write(open(a.template).read().replace('/*DATA*/null', data))
    print(a.out)
