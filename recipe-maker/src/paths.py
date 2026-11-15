"""Summit paths (data/paths/<target>.json): per target, the nodes a lineage tried, the serving levers each composed, its
points (per-agent and aggregate tok/s, TTFT, each tagged), the recipe it ran or would run and a verdict. build.py loads them
into data.paths; src/paths.js draws each path on the per-agent against aggregate chart and report.js --paths prints them.

Derived here, once: each node's head point (the one the path line passes through: its best total inside the target's agent
band, else its best total), the edges, and the lever pairs the target never composed (two levers the target tried, never
together in one node; two levers of an exclusive kind, an engine or a quant, never pair)."""
import itertools
import json
import os
import re

TAGS = ('MEASURED', 'PREDICTED', 'UNKNOWN')
STATES = ('kept', 'dropped', 'open')
SRC_LINE = re.compile(r'\S+:\d+')
VALUE_KEYS = ('per_agent', 'aggregate', 'ttft_s')


def _num(x, where):
    if not isinstance(x, dict) or x.get('tag') not in TAGS:
        raise ValueError('%s: a value is {v, tag, src} with tag one of %s (%r)' % (where, TAGS, x))
    v = x.get('v')
    if x['tag'] == 'UNKNOWN':
        if v is not None:
            raise ValueError('%s: an UNKNOWN value has v null' % where)
    elif not (isinstance(v, (int, float)) or (isinstance(v, list) and len(v) == 2 and all(isinstance(y, (int, float)) for y in v) and v[0] <= v[1])):
        raise ValueError('%s: v is a number or a [low, high] range (%r)' % (where, v))
    if x['tag'] == 'MEASURED' and not SRC_LINE.search(str(x.get('src', ''))):
        raise ValueError('%s: a MEASURED value needs its source as path:line (%r)' % (where, x.get('src')))
    if not x.get('src'):
        raise ValueError('%s: every value names its src' % where)


def mid(x):
    v = x.get('v') if x else None
    return None if v is None else (v[0] + v[1]) / 2 if isinstance(v, list) else v


def _read(path):
    with open(path) as f:
        return json.load(f)


def load_catalog(path):
    d = _read(path)
    seen = {}
    for lv in d['levers']:
        if not lv.get('id') or lv['id'] in seen:
            raise ValueError('%s: a lever needs a unique id (%r)' % (path, lv.get('id')))
        if lv.get('kind') not in d['kinds']:
            raise ValueError('%s: lever %s kind %r is not one of %s' % (path, lv['id'], lv.get('kind'), sorted(d['kinds'])))
        seen[lv['id']] = lv
    for k in d['exclusive_kinds']:
        if k not in d['kinds']:
            raise ValueError('%s: exclusive kind %r is not a kind' % (path, k))
    return d


def head_of(node, goal):
    """the point the path line passes through: node.head (an agent count) if set, else the best total inside the goal's
    agent band, else the best total; a point with no total is never a head"""
    pts = [p for p in node['points'] if mid(p['aggregate']) is not None and mid(p['per_agent']) is not None]
    if not pts:
        return None
    if node.get('head') is not None:
        hit = [p for p in pts if p['agents'] == node['head']]
        if not hit:
            raise ValueError('node %s: head %s is not one of its agent counts' % (node['id'], node['head']))
        return hit[0]['agents']
    lo, hi = (goal.get('agents') or [None, None]) if goal else (None, None)
    band = [p for p in pts if lo is not None and lo <= p['agents'] <= hi]
    return max(band or pts, key=lambda p: mid(p['aggregate']))['agents']


def never_composed(nodes, catalog, candidates=()):
    """[{a, b, tried: [a tried, b tried]}]: pairs of levers this target tried (or names as candidates) that no node composed"""
    kind = {lv['id']: lv['kind'] for lv in catalog['levers']}
    excl = set(catalog['exclusive_kinds'])
    tried = {x for nd in nodes for x in nd['levers']}
    pool = sorted(tried | set(candidates), key=lambda x: (kind[x], x))
    together = {frozenset(p) for nd in nodes for p in itertools.combinations(set(nd['levers']), 2)}
    out = []
    for a, b in itertools.combinations(pool, 2):
        if kind[a] == kind[b] and kind[a] in excl:
            continue
        if frozenset((a, b)) not in together:
            out.append({'a': a, 'b': b, 'tried': [a in tried, b in tried]})
    return out


def load_target(path, catalog, ledger_ids, root):
    d = _read(path)
    name = os.path.relpath(path, root)
    for k in ('id', 'label', 'model', 'metric', 'nodes'):
        if not d.get(k):
            raise ValueError('%s: a target needs %s' % (name, k))
    if d['id'] + '.json' != os.path.basename(path):
        raise ValueError('%s: the file is named after its id (%s.json)' % (name, d['id']))
    levers = {lv['id'] for lv in catalog['levers']}
    goal = d.get('goal') or {}
    if goal and not (isinstance(goal.get('agents'), list) and len(goal['agents']) == 2 and goal.get('src')):
        raise ValueError('%s: goal needs agents [low, high] and its src' % name)
    ids = set()
    for nd in d['nodes']:
        where = '%s node %s' % (name, nd.get('id'))
        if not nd.get('id') or nd['id'] in ids:
            raise ValueError('%s: a node needs a unique id' % where)
        for p in nd.get('parents', []):
            if p not in ids:
                raise ValueError('%s: parent %s is not an earlier node of this target' % (where, p))
        ids.add(nd['id'])
        bad = [x for x in nd.get('levers', []) if x not in levers]
        if bad:
            raise ValueError('%s: levers not in data/paths/levers.json: %s' % (where, ', '.join(bad)))
        kind = {lv['id']: lv['kind'] for lv in catalog['levers']}
        for k in catalog['exclusive_kinds']:
            got = [x for x in nd['levers'] if kind[x] == k]
            if len(got) > 1:
                raise ValueError('%s: two levers of the exclusive kind %s: %s' % (where, k, ', '.join(got)))
        rec = nd.get('recipe') or ''
        if not (rec.startswith('none: ') or rec in ledger_ids or os.path.isfile(os.path.join(root, rec))):
            raise ValueError('%s: recipe %r is neither a ledger run id, a file in this repo nor "none: why"' % (where, rec))
        if not nd.get('results'):
            raise ValueError('%s: name the results dir or file (results)' % where)
        v = nd.get('verdict') or {}
        if v.get('state') not in STATES or (v['state'] == 'dropped' and not v.get('why')):
            raise ValueError('%s: verdict.state is one of %s, and a dropped node says why' % (where, STATES))
        if not nd.get('points'):
            raise ValueError('%s: a node has at least one point' % where)
        for p in nd['points']:
            if not isinstance(p.get('agents'), int) or p['agents'] < 1:
                raise ValueError('%s: a point names its agent count' % where)
            for k in VALUE_KEYS:
                _num(p.get(k), '%s at %s agents %s' % (where, p.get('agents'), k))
        nd['head'] = head_of(nd, goal)
    cand = d.get('candidates', [])
    bad = [x for x in cand if x not in levers]
    if bad:
        raise ValueError('%s: candidates not in data/paths/levers.json: %s' % (name, ', '.join(bad)))
    d['file'] = name
    d['edges'] = [[p, nd['id']] for nd in d['nodes'] for p in nd.get('parents', [])]
    d['never_composed'] = never_composed(d['nodes'], catalog, cand)
    return d


def load_paths(root, ledger_ids):
    """data.paths: {catalog, targets: [target, ...]} from data/paths/ (levers.json + one file per target), or None"""
    d = os.path.join(root, 'data', 'paths')
    if not os.path.isdir(d):
        return None
    catalog = load_catalog(os.path.join(d, 'levers.json'))
    targets = [load_target(os.path.join(d, f), catalog, ledger_ids, root)
               for f in sorted(os.listdir(d)) if f.endswith('.json') and f != 'levers.json']
    return {'catalog': catalog, 'targets': targets}
