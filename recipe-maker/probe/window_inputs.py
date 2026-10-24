#!/usr/bin/env python3
"""Readers for the batch-window cost model (data.window, src/model.js window*): a model's batch-window spec and a card's
batch-window microbenchmark log. Both formats are the ones XeStrata's tools write; nothing here names a model, a card or
a kernel class: every class, layer kind and row count comes from the files.

    python3 probe/window_inputs.py --spec SPEC.txt          # the parsed spec as JSON
    python3 probe/window_inputs.py --ubench UBENCH.log      # the parsed STAGE ROLL-UP as JSON

The spec (tools/batch_window_spec.py SHARD1.gguf [--layers A-B] > spec.txt, in the XeStrata tree):
    dims key=value ...
    layer L tensor=ggml_type_id ...
The microbenchmark log (batch_window_ubench, run through fab or by hand): the "start:" / "end:" copy-kernel lines, the
"method floor" line, and the "STAGE ROLL-UP" table, whose header names the measured layers per kind, the row grid, the
floor's row count and the bandwidth the floor is taken at.
"""
import argparse
import json
import re
import sys

NUM = r'-?\d+(?:\.\d+)?'


def parse_spec(text, kind_rule):
    """{dims: {k: int}, layers: [{layer, types, kind}]}. kind_rule = {'tensor': name, 'kind': K, 'default': D}: a layer
    holding that tensor is of kind K, every other layer D (the spec tool's own rule, carried by the config, never here)"""
    dims, layers = {}, []
    for line in text.splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == 'dims':
            dims = {k: int(x) for k, x in (p.split('=', 1) for p in t[1:])}
        elif t[0] == 'layer':
            types = {k: int(x) for k, x in (p.split('=', 1) for p in t[2:])}
            layers.append({'layer': int(t[1]), 'types': types,
                           'kind': kind_rule['kind'] if kind_rule['tensor'] in types else kind_rule['default']})
    if not dims or not layers:
        raise ValueError('spec: no dims line or no layer lines')
    return {'dims': dims, 'layers': sorted(layers, key=lambda x: x['layer'])}


def _copy_gbs(lines, word):
    for line in lines:
        if line.startswith(word + ':') and 'copy kernel' in line:
            return float(re.search(r'=\s*(' + NUM + r')\s*GB/s', line).group(1))
    return None


def parse_ubench(text):
    """the STAGE ROLL-UP of a batch-window microbenchmark log: rows grid, measured layers per kind, each kernel class
    (layers it runs in, ms summed over those layers per row count, us per row per layer, MB at the floor's rows per layer,
    floor us, x floor), the TOTAL lines, the 'not measured' line, the copy-kernel bandwidths and the method floor"""
    lines = text.splitlines()
    head = [i for i, x in enumerate(lines) if x.startswith('STAGE ROLL-UP')]
    if not head:
        raise ValueError('ubench: no STAGE ROLL-UP section')
    h = lines[head[0]]
    m = re.match(r'STAGE ROLL-UP:\s*(\d+) layers \(([^)]*)\)', h)
    kinds = {k: int(c) for c, k in re.findall(r'(\d+)\s+([^\s,]+)', m.group(2))}
    fl = re.search(r'floor = bytes at (\d+) rows / (' + NUM + r') GB/s', h)
    cols = lines[head[0] + 1].split()
    # the header row: "kernel layers r1 r2 ... rN us/row/lyr MB@F/lyr floor us x floor"
    grid = [int(x) for x in cols[2:] if x.isdigit()]
    ntail = 1 + len(grid) + 4
    classes, totals, unmeasured = [], {}, None
    for line in lines[head[0] + 2:]:
        if line.startswith('not measured'):
            unmeasured = line.split(':', 1)[1].strip()
            continue
        if line.startswith('TOTAL'):
            lab = re.match(r'TOTAL\s*\(([^)]*)\)', line)
            totals[lab.group(1) if lab else 'all'] = [float(x) for x in line.split()[-len(grid):]]
            continue
        t = line.split()
        if len(t) <= ntail or not all(re.fullmatch(NUM, x) for x in t[-ntail:]):
            if classes and (totals or unmeasured):
                break
            continue
        name, nums = ' '.join(t[:-ntail]), [float(x) for x in t[-ntail:]]
        launches = re.search(r'\bx(\d+)$', name)
        classes.append({'name': name, 'layers': int(nums[0]), 'ms': nums[1:1 + len(grid)], 'us_row_lyr': nums[1 + len(grid)],
                        'mb_floor_lyr': nums[2 + len(grid)], 'floor_us': nums[3 + len(grid)], 'xfloor': nums[4 + len(grid)],
                        'launches': int(launches.group(1)) if launches else 1})
    if not classes or not grid:
        raise ValueError('ubench: STAGE ROLL-UP has no kernel rows')
    # routing: the class measured with random routing, and the other TOTAL line (rows that share experts): the gap between the
    # two totals is the class's cost at the other routing, row by row
    rc = [c['name'] for c in classes if '(random)' in c['name']]
    rnd = [k for k in totals if 'random' in k]
    alt = [k for k in totals if k not in rnd]
    routing = {'class': rc[0], 'random': rnd[0], 'alt': alt[0], 'alt_delta_ms': [round(b - a, 4) for a, b in zip(totals[rnd[0]], totals[alt[0]])]} if rc and rnd and alt else None
    mf = [re.search(r':\s*(' + NUM + r')\s*us', x) for x in lines if x.startswith('method floor')]
    dev = [x for x in lines if x.startswith('device:')]
    return {'layers': int(m.group(1)), 'kinds': kinds, 'grid': grid, 'floor_rows': int(fl.group(1)), 'floor_gbs': float(fl.group(2)),
            'copy_gbs_start': _copy_gbs(lines, 'start'), 'copy_gbs_end': _copy_gbs(lines, 'end'),
            'method_floor_us': float(mf[0].group(1)) if mf and mf[0] else None, 'device': dev[0][len('device:'):].strip() if dev else None,
            'classes': classes, 'totals': totals, 'routing': routing, 'unmeasured': unmeasured}


def class_kind(c, kinds, measured_layers):
    """the layer kind a class runs in: every layer when its count is the measured total, else the one kind with that count;
    a tie between kinds falls back on the kind's name in the class name (case-insensitive); None when still ambiguous"""
    if c['layers'] == measured_layers:
        return None
    hit = [k for k, n in kinds.items() if n == c['layers']]
    if len(hit) == 1:
        return hit[0]
    named = [k for k in hit if k.lower() in c['name'].lower()]
    if len(named) == 1:
        return named[0]
    raise ValueError('ubench class %r: %d layers matches kinds %s; name the kind in the class' % (c['name'], c['layers'], hit))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec')
    ap.add_argument('--ubench')
    ap.add_argument('--kind-tensor', default='attn_q', help='spec: a layer holding this tensor is of --kind (batch_window_spec.py: attn_q = QSA)')
    ap.add_argument('--kind', default='QSA')
    ap.add_argument('--default-kind', default='GDN')
    a = ap.parse_args()
    out = {}
    if a.spec:
        out['spec'] = parse_spec(open(a.spec).read(), {'tensor': a.kind_tensor, 'kind': a.kind, 'default': a.default_kind})
    if a.ubench:
        out['ubench'] = parse_ubench(open(a.ubench).read())
    json.dump(out, sys.stdout, indent=1)
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
