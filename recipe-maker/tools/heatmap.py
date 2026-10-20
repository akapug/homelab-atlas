#!/usr/bin/env python3
"""Hot-expert maps: one canonical file format for MoE routing heat, converters from every heat file we have, one loader.

A hot-expert map records how often a model's router picked each expert, per MoE layer, on a named prompt set, with the
model, checkpoint, quant, tool, engine and box that produced it. Recipe Maker plans layouts from these counts (which
experts go to VRAM, DRAM, PMem), and a recipe cites the map it was planned on by id and digest.

Format (JSON, one file per capture; spec and procedure: HOT-EXPERT-MAPS.md):
  format     "hot-expert-map/1"
  id         slug, unique per capture (model__workload__engine__date)
  title      one line
  model      {id, family, checkpoint, checkpoint_hash ("sha256:..." or null), checkpoint_hash_note, quant,
              layers [MoE layer ids, file order of the count rows], experts, k}
  workload   {name, description, prompt_set_sha256 (or null), prompt_set_files, decoding}
  producer   {tool, tool_version, engine, engine_version, box, date}
  provenance {tag MEASURED|PRIOR|ESTIMATE, src, converted_from {path, sha256, converter}, notes [..]}
  slices     {name: {phase decode|prefill|mixed, category?, half?, tokens, tokens_tag, picks, counts [[E ints] per layer]}}
  default_slice  name of the slice a planner uses when none is named
  summary    derived per slice (validate recomputes it): picks, tokens, top10/top25 coverage, never-picked, effective experts
  digest     "sha256:" + sha256 of the canonical JSON of {model, workload, slices}; a recipe cites this

Selection files ("hot-expert-selection/1") record a hot set chosen from a map (GPU banks): {format, id, model {id,
family}, cites {map_id, digest, slice} or null, rule, experts {layer: [ids]}, provenance}.

CLI:
  heatmap.py validate FILE...                         check format, shapes, summary and digest; exit 1 on any error
  heatmap.py summary FILE                             per-slice coverage table
  heatmap.py convert-all SOURCES.json OUTDIR          convert every source in the table that exists here (notes the rest)
  heatmap.py from-imatrix-tsv | from-route-stats | from-moe-trace | from-sets-json | from-exl3-stats  SRC --meta M.json -o OUT
  heatmap.py selection-from-banks TSV --meta M.json -o OUT
Standard library only.
"""
import argparse, copy, gzip, hashlib, json, math, os, re, struct, sys

FORMAT = 'hot-expert-map/1'
SEL_FORMAT = 'hot-expert-selection/1'
TAGS = ('MEASURED', 'PRIOR', 'ESTIMATE')
PHASES = ('decode', 'prefill', 'mixed')
REQUIRED = {
    'model': ('id', 'family', 'checkpoint', 'checkpoint_hash', 'quant', 'layers', 'experts', 'k'),
    'workload': ('name', 'description', 'prompt_set_sha256'),
    'producer': ('tool', 'tool_version', 'engine', 'box', 'date'),
    'provenance': ('tag', 'src'),
}


class MapError(ValueError):
    pass


def slurp(path, mode='r'):
    """whole file contents, handle closed"""
    with open(path, mode) as f:
        return f.read()


GZIP_OVER = 1 << 20   # a map whose text is larger than this is written gzipped (FILE.gz): no tracked file passes 1 MiB


def jload(path):
    """JSON from PATH, or from PATH gzipped when PATH ends in .gz"""
    with (gzip.open(path, 'rt', encoding='utf-8') if path.endswith('.gz') else open(path)) as f:
        return json.load(f)


def write(path, obj):
    """write a map or selection; text over GZIP_OVER bytes goes to PATH.gz (a stale plain PATH is removed). Returns the path
    written"""
    text = (dumps(obj) + '\n').encode()
    out = path + '.gz' if len(text) > GZIP_OVER and not path.endswith('.gz') else path
    with open(out + '.tmp', 'wb') as f:
        f.write(gzip.compress(text, 9, mtime=0) if out.endswith('.gz') else text)
    os.replace(out + '.tmp', out)
    for stale in {path, path + '.gz'} - {out}:
        if os.path.exists(stale):
            os.remove(stale)
    return out


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def manifest_hash(path):
    """checkpoint hash = sha256 of its shard manifest normalized: 'sha256  basename' lines sorted by basename, one per line.
    Accepts sha256sum output (with or without paths, '*' binary marks) or a directory of NAME.sha256 files."""
    rows = []
    files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith('.sha256')] if os.path.isdir(path) else [path]
    for fn in files:
        for line in slurp(fn).splitlines():
            f = line.split()
            if len(f) >= 2 and re.fullmatch(r'[0-9a-f]{64}', f[0]):
                rows.append((os.path.basename(f[1].lstrip('*')), f[0]))
    if not rows:
        raise MapError('no sha256 lines in %s' % path)
    text = ''.join('%s  %s\n' % (h, n) for n, h in sorted(rows))
    return 'sha256:' + hashlib.sha256(text.encode()).hexdigest()


def dumps(obj):
    """indented JSON with every list of numbers on one line (count rows stay one line per layer)"""
    s = json.dumps(obj, indent=1)
    return re.sub(r'\[\s*(-?[0-9.e+]+(?:,\s*-?[0-9.e+]+)*)\s*\]', lambda m: '[' + ','.join(m.group(1).split()).replace(',,', ',') + ']', s)


def digest(m):
    core = {'model': m['model'], 'workload': m['workload'], 'slices': m['slices']}
    return 'sha256:' + hashlib.sha256(json.dumps(core, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def slice_summary(s, k):
    rows = s['counts']
    E = len(rows[0]) if rows else 0
    picks = sum(sum(r) for r in rows)
    cov = lambda r, f: sum(sorted(r, reverse=True)[:max(1, math.ceil(f * E))]) / (sum(r) or 1)
    eff = []
    for r in rows:
        t = sum(r)
        h = -sum(x / t * math.log(x / t) for x in r if x) if t else 0.0
        eff.append(math.exp(h))
    eff.sort()
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {'picks': picks, 'tokens': s.get('tokens'), 'tokens_tag': s.get('tokens_tag'),
            'top10_cover': round(mean([cov(r, 0.10) for r in rows]), 4), 'top25_cover': round(mean([cov(r, 0.25) for r in rows]), 4),
            'never_picked': sum(1 for r in rows for x in r if x == 0),
            'effective_experts_median': round(eff[len(eff) // 2], 1) if eff else 0.0}


def finish(m):
    """fill derived fields (summary, digest) and validate; returns m"""
    k = m['model']['k']
    for s in m['slices'].values():
        if s.get('picks') is None:
            s['picks'] = sum(sum(r) for r in s['counts'])
        if s.get('tokens') is None:
            per_layer = s['picks'] / max(1, len(s['counts']))
            s['tokens'], s['tokens_tag'] = round(per_layer / k), 'INFERRED'
    m['summary'] = {n: slice_summary(s, k) for n, s in m['slices'].items()}
    m['digest'] = digest(m)
    validate(m)
    return m


def validate(m):
    """raise MapError naming every problem; return m"""
    errs = []
    if m.get('format') != FORMAT:
        raise MapError('format is %r, want %r' % (m.get('format'), FORMAT))
    for key in ('id', 'title', 'slices', 'default_slice', 'digest'):
        if key not in m:
            errs.append('missing %s' % key)
    for sec, keys in REQUIRED.items():
        d = m.get(sec)
        if not isinstance(d, dict):
            errs.append('missing section %s' % sec)
            continue
        for k in keys:
            if k not in d:
                errs.append('missing %s.%s' % (sec, k))
    if errs:
        raise MapError('; '.join(errs))
    md = m['model']
    if md['checkpoint_hash'] is None and not md.get('checkpoint_hash_note'):
        errs.append('model.checkpoint_hash is null without model.checkpoint_hash_note (say why it is unknown)')
    if m['workload']['prompt_set_sha256'] is None and not m['workload'].get('prompt_set_note'):
        errs.append('workload.prompt_set_sha256 is null without workload.prompt_set_note')
    if m['provenance']['tag'] not in TAGS:
        errs.append('provenance.tag %r not in %s' % (m['provenance']['tag'], TAGS))
    L, E = len(md['layers']), md['experts']
    if len(set(md['layers'])) != L:
        errs.append('model.layers has duplicates')
    if not m['slices']:
        errs.append('no slices')
    for name, s in m['slices'].items():
        if s.get('phase') not in PHASES:
            errs.append('slice %s: phase %r not in %s' % (name, s.get('phase'), PHASES))
        rows = s.get('counts') or []
        if len(rows) != L:
            errs.append('slice %s: %d count rows, model.layers has %d' % (name, len(rows), L))
        for i, r in enumerate(rows):
            if len(r) != E:
                errs.append('slice %s layer %s: %d experts, want %d' % (name, md['layers'][i] if i < L else i, len(r), E))
                break
            if any((not isinstance(x, int)) or x < 0 for x in r):
                errs.append('slice %s layer %s: counts must be integers >= 0' % (name, md['layers'][i] if i < L else i))
                break
        if s.get('picks') is not None and rows and s['picks'] != sum(sum(r) for r in rows):
            errs.append('slice %s: picks %s != sum of counts' % (name, s['picks']))
    if m['default_slice'] not in m['slices']:
        errs.append('default_slice %r is not a slice' % m['default_slice'])
    if errs:
        raise MapError('; '.join(errs))
    want = {n: slice_summary(s, md['k']) for n, s in m['slices'].items()}
    if m.get('summary') != want:
        errs.append('summary does not match the counts (regenerate it)')
    if m['digest'] != digest(m):
        errs.append('digest %s does not match the content (%s)' % (m['digest'], digest(m)))
    if errs:
        raise MapError('; '.join(errs))
    return m


def load(path):
    m = jload(path)
    try:
        return validate(m)
    except MapError as e:
        raise MapError('%s: %s' % (path, e))


def load_dir(d):
    """every *.heatmap.json (or .heatmap.json.gz) under d, validated, keyed by id (a duplicate id is an error)"""
    out = {}
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if fn.endswith(('.heatmap.json', '.heatmap.json.gz')):
            m = load(os.path.join(d, fn))
            if m['id'] in out:
                raise MapError('duplicate map id %s (%s)' % (m['id'], fn))
            m['_file'] = fn
            out[m['id']] = m
    return out


def planner_layers(m, slice_name, layer_ids, experts):
    """the slice's count rows in the planner's layer order; MapError when a layer or the expert count does not match"""
    md = m['model']
    if md['experts'] != experts:
        raise MapError('%s: %d experts, the planner model has %d' % (m['id'], md['experts'], experts))
    s = m['slices'].get(slice_name)
    if s is None:
        raise MapError('%s: no slice %r (have %s)' % (m['id'], slice_name, ', '.join(m['slices'])))
    idx = {l: i for i, l in enumerate(md['layers'])}
    missing = [l for l in layer_ids if l not in idx]
    if missing:
        raise MapError('%s: no counts for planner layers %s' % (m['id'], missing[:8]))
    return [s['counts'][idx[l]] for l in layer_ids]


def cite(m, slice_name=None):
    """the citation a recipe carries: {map_id, digest, slice}"""
    return {'map_id': m['id'], 'digest': m['digest'], 'slice': slice_name or m['default_slice']}


# ---------------- converters (source file -> slices + layers) ----------------
def _layer_no(key):
    m = re.search(r'(?:blk\.|layers\.)(\d+)', key)
    if not m:
        raise MapError('no layer number in %r' % key)
    return int(m.group(1))


def conv_imatrix_tsv(path, opts):
    """ik_llama.cpp / llama.cpp imatrix with the per-expert count patch: one row per expert tensor, name then counts.
    gate/up/down of one layer count the same tokens; the first of up, gate, down present per layer is used."""
    rows = {}
    with open(path) as fh:
        for line in fh:
            f = line.rstrip('\n').split('\t')
            mm = re.match(r'blk\.(\d+)\.ffn_(up|gate|down)_exps', f[0])
            if mm:
                rows.setdefault(int(mm.group(1)), {})[mm.group(2)] = [int(x) for x in f[1:]]
    layers = sorted(rows)
    counts = [next(rows[l][t] for t in ('up', 'gate', 'down') if t in rows[l]) for l in layers]
    return layers, {opts.get('slice', 'prefill'): {'phase': opts.get('phase', 'prefill'), 'counts': counts}}


def conv_route_stats(path, opts):
    """glm53 route_stats.py output: {prefill|decode|total: {layer key: [E counts]}, meta}"""
    d = jload(path)
    keys = sorted(d['decode'], key=_layer_no)
    layers = [_layer_no(k) for k in keys]
    meta = d.get('meta', {})
    sl = {}
    for ph in ('decode', 'prefill'):
        sl[ph] = {'phase': ph, 'counts': [d[ph][k] for k in keys]}
        tok = meta.get(ph + '_tokens')
        if tok:
            sl[ph]['tokens'], sl[ph]['tokens_tag'] = tok, 'MEASURED'
    return layers, sl


def conv_sets_json(path, opts):
    """the planner's legacy count files: {set name: [[E] per layer]} (V4.1 skew) or {k, layers, cats: {name: ...}} (GLM).
    opts.sets maps output slice name -> {key, phase, category, half}; opts.layers gives the layer ids when the file has none"""
    d = jload(path)
    src = d.get('cats', d)
    layers = d.get('layers') if isinstance(d.get('layers'), list) else opts['layers']
    sl = {}
    for name, spec in opts['sets'].items():
        s = {'phase': spec.get('phase', 'decode'), 'counts': src[spec.get('key', name)]}
        for k in ('category', 'half', 'tokens', 'tokens_tag'):
            if k in spec:
                s[k] = spec[k]
        sl[name] = s
    return list(layers), sl


def conv_exl3_stats(path, opts):
    """EXL3_EXPERT_STATS snapshot as JSON (snap2json.py): {layer name: [[decode hits], [prefill hits]]}"""
    d = jload(path)
    keys = sorted(d, key=_layer_no)
    return [_layer_no(k) for k in keys], {'decode': {'phase': 'decode', 'counts': [d[k][0] for k in keys]},
                                          'prefill': {'phase': 'prefill', 'counts': [d[k][1] for k in keys]}}


def conv_moe_trace(path, opts):
    """llama.cpp router logger (moe-trace.cpp, a cb_eval hook on ffn_moe_topk-<layer>). Record: int32 magic 0x4D4F4532,
    prompt_id, phase (0 prefill, 1 decode), layer, pos0, k, n_tok, then n_tok*k int32 expert ids.
    opts.experts = E; opts.categories = {prompt_id: category} (optional) splits decode per category."""
    E = opts['experts']
    cats = {int(k): v for k, v in (opts.get('categories') or {}).items()}
    data = slurp(path, 'rb')
    acc, tok, off = {}, {}, 0
    while off < len(data):
        magic, pid, phase, layer, pos0, k, n = struct.unpack_from('<7i', data, off)
        if magic != 0x4D4F4532:
            raise MapError('%s: bad record magic at byte %d' % (path, off))
        off += 28
        ids = struct.unpack_from('<%di' % (n * k), data, off)
        off += 4 * n * k
        ph = 'decode' if phase == 1 else 'prefill'
        names = [ph] + (['decode_' + cats[pid]] if ph == 'decode' and pid in cats else [])
        for nm in names:
            row = acc.setdefault(nm, {}).setdefault(layer, [0] * E)
            for e in ids:
                row[e] += 1
            tok.setdefault(nm, {}).setdefault(layer, 0)
            tok[nm][layer] += n
    # llama.cpp drops non-output rows on the last layer in prefill: keep only layers with the full token count per slice
    layers = sorted(set.intersection(*(set(v) for v in acc.values())))
    sl, dropped = {}, {}
    for nm, by in acc.items():
        top = max(tok[nm][l] for l in layers)
        short = [l for l in layers if tok[nm][l] != top]
        if short:
            dropped[nm] = short
        sl[nm] = {'phase': 'decode' if nm.startswith('decode') else 'prefill', 'tokens': top, 'tokens_tag': 'MEASURED',
                  'counts': [by[l] for l in layers]}
        if nm.startswith('decode_'):
            sl[nm]['category'] = nm[7:]
    if dropped:
        opts.setdefault('_notes', []).append('layers with fewer rows than the slice maximum (llama.cpp prunes the last layer in prefill): %s' % json.dumps(dropped))
    return layers, sl


CONVERTERS = {'from-imatrix-tsv': conv_imatrix_tsv, 'from-route-stats': conv_route_stats, 'from-sets-json': conv_sets_json,
              'from-exl3-stats': conv_exl3_stats, 'from-moe-trace': conv_moe_trace}


def deep_merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def build_map(converter, src, meta, opts=None):
    """run one converter on src and wrap the result with meta (model/workload/producer/provenance/id/title)"""
    opts = dict(opts or {})
    layers, slices = CONVERTERS[converter](src, opts)
    m = deep_merge({'format': FORMAT}, copy.deepcopy(meta))  # never share nested dicts with the caller's meta
    m['model'] = dict(m.get('model', {}), layers=layers)
    if 'experts' not in m['model']:
        m['model']['experts'] = len(next(iter(slices.values()))['counts'][0])
    m['slices'] = slices
    m.setdefault('default_slice', opts.get('default_slice') or next(iter(slices)))
    prov = m.setdefault('provenance', {})
    prov['converted_from'] = {'path': opts.get('display_path', src), 'sha256': sha256_file(src), 'converter': 'heatmap.py ' + converter}
    if opts.get('_notes'):
        prov['notes'] = list(prov.get('notes', [])) + opts['_notes']
    order = ['format', 'id', 'title', 'model', 'workload', 'producer', 'provenance', 'default_slice', 'slices']
    m = {k: m[k] for k in order if k in m}
    return finish(m)


def parse_ranges(s):
    out = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def selection_from_banks(path, meta):
    """GPU bank TSV (gen-hot*.py): '# header' then 'blk.N<TAB>ids[ # comment]' with ids as a,b,c-d"""
    header, experts = [], {}
    for line in slurp(path).splitlines():
        line = line.rstrip('\n')
        if line.startswith('#'):
            header.append(line.lstrip('# ').strip())
            continue
        if not line.strip():
            continue
        name, rest = line.split('\t', 1)
        experts[str(_layer_no(name))] = parse_ranges(rest.split('#')[0])
    sel = deep_merge({'format': SEL_FORMAT}, copy.deepcopy(meta))
    sel['rule'] = sel.get('rule') or ' | '.join(header)
    sel['experts'] = experts
    sel['n_experts'] = sum(len(v) for v in experts.values())
    prov = sel.setdefault('provenance', {})
    prov['converted_from'] = {'path': meta.get('_display_path', path), 'sha256': sha256_file(path), 'converter': 'heatmap.py selection-from-banks'}
    sel.pop('_display_path', None)
    validate_selection(sel)
    return sel


def validate_selection(s):
    errs = [k for k in ('id', 'model', 'rule', 'experts', 'provenance') if k not in s]
    if s.get('format') != SEL_FORMAT:
        errs.append('format')
    if 'cites' not in s:
        errs.append('cites (null when the source map is not converted; say why in provenance.notes)')
    if errs:
        raise MapError('selection missing/invalid: ' + ', '.join(errs))
    return s


def convert_all(table_path, outdir, quiet=False):
    """convert every source row present on this machine; returns (written, skipped)"""
    table = jload(table_path)
    written, skipped = [], []
    os.makedirs(outdir, exist_ok=True)
    maps = {}
    root = os.path.dirname(os.path.dirname(os.path.abspath(table_path)))  # relative sources resolve from the repo root
    for row in table['sources']:
        src = os.path.join(root, os.path.expanduser(row['src']))
        if not os.path.exists(src):
            skipped.append((row['out'], 'source not here: ' + row['src']))
            continue
        opts = dict(row.get('opts', {}), display_path=row['src'])
        if opts.get('categories_from'):  # prompt id -> category from a workload JSONL ({"id", "cat"} per line)
            cp = os.path.join(root, os.path.expanduser(opts['categories_from']))
            if not os.path.exists(cp):
                skipped.append((row['out'], 'category file not here: ' + opts['categories_from']))
                continue
            opts['categories'] = {json.loads(l)['id']: json.loads(l)['cat'] for l in slurp(cp).splitlines() if l.strip()}
        meta_row = row['meta']
        if row.get('model_ref'):  # shared model block from the table's "models"
            meta_row = deep_merge({'model': table['models'][row['model_ref']]}, meta_row)
        for key, rel in (row.get('manifest') or {}).items():  # model.checkpoint_hash from a shard manifest found here
            mp = os.path.join(root, os.path.expanduser(rel))
            if os.path.exists(mp):
                meta_row = deep_merge(meta_row, {'model': {'checkpoint_hash': manifest_hash(mp), 'checkpoint_hash_note': 'sha256 of the normalized shard manifest ' + rel}})
        row = dict(row, meta=meta_row)
        if row['converter'] == 'selection-from-banks':
            meta = dict(row['meta'], _display_path=row['src'])
            c = row['meta'].get('cites')
            if isinstance(c, str):  # cite a map converted earlier in this table by id
                if c not in maps:
                    raise MapError('%s cites map %s, which is not converted (order the table or convert it first)' % (row['out'], c))
                meta['cites'] = cite(maps[c], row.get('cites_slice'))
            obj = selection_from_banks(src, meta)
        else:
            obj = build_map(row['converter'], src, row['meta'], opts)
            maps[obj['id']] = obj
        written.append(os.path.relpath(write(os.path.join(outdir, row['out']), obj), outdir))
    if not quiet:
        for w in written:
            print('wrote', w)
        for s, why in skipped:
            print('skipped', s, '(%s)' % why)
    return written, skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('validate'); v.add_argument('files', nargs='+')
    s = sub.add_parser('summary'); s.add_argument('file')
    h = sub.add_parser('manifest-hash'); h.add_argument('path')
    c = sub.add_parser('convert-all'); c.add_argument('table'); c.add_argument('outdir')
    for name in list(CONVERTERS) + ['selection-from-banks']:
        p = sub.add_parser(name)
        p.add_argument('src'); p.add_argument('--meta', required=True); p.add_argument('--opts'); p.add_argument('-o', '--out', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'validate':
        bad = 0
        for f in a.files:
            try:
                d = jload(f)
                (validate_selection if d.get('format') == SEL_FORMAT else validate)(d)
                print('OK', f, d.get('digest', ''))
            except (MapError, ValueError, KeyError) as e:
                bad += 1
                print('BAD', f, e)
        return 1 if bad else 0
    if a.cmd == 'summary':
        m = load(a.file)
        print('%s  %s  digest %s' % (m['id'], m['model']['checkpoint'], m['digest']))
        print('slice\tphase\ttokens\tpicks\ttop10\ttop25\tnever\teff_median')
        for n, x in m['summary'].items():
            print('%s\t%s\t%s%s\t%d\t%.3f\t%.3f\t%d\t%.1f' % (n, m['slices'][n]['phase'], x['tokens'], ' (INFERRED)' if x['tokens_tag'] == 'INFERRED' else '',
                                                              x['picks'], x['top10_cover'], x['top25_cover'], x['never_picked'], x['effective_experts_median']))
        return 0
    if a.cmd == 'manifest-hash':
        print(manifest_hash(a.path))
        return 0
    if a.cmd == 'convert-all':
        convert_all(a.table, a.outdir)
        return 0
    meta = jload(a.meta)
    obj = selection_from_banks(a.src, meta) if a.cmd == 'selection-from-banks' else build_map(a.cmd, a.src, meta, jload(a.opts) if a.opts else None)
    print('wrote', write(a.out, obj), obj.get('digest', ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
