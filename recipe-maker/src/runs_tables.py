"""Tables the report and the page show, generated from the run ledger (runs/runs.jsonl) and runs/levers.json.

recipes_from_runs: one row per measured run for the hard-check section: the recipe the gate consumes, the run's own
observations (node MemFree, decode KL) that overlay the prediction, and outcome / cost text built from the measured fields.
levers_from_runs: the measured-lever table; each cell names ledger runs and the fields to show.
"""
import datetime
import json
import os

import ledger

PHASES = ('n4', 'n3', 'n2', 'n1', 'n0', 'n4b', 'c1', 'prompt1')


def load_runs(path):
    return ledger.rows(path)


def went_bad(run):
    m = run['measured']
    return ([m['outcome']] if m.get('outcome') != 'ok' else []) + list(m.get('degraded', []))


def _g(x):
    return ('%.2f' % x).rstrip('0').rstrip('.') if isinstance(x, float) else str(x)


def field(run, key):
    """text for one measured field of a run (None when the run did not measure it)"""
    m = run['measured']
    d = m.get('decode_tps', {})
    if key in PHASES:
        return '%s %s' % (key, _g(d[key])) if key in d else None
    if key == 'draft_x':
        return 'x%.2f' % (d['n4'] / d['n0']) if 'n4' in d and 'n0' in d else None
    if key in ('kl', 'top1'):
        k = 'kl' if key == 'kl' else 'top1'
        return ('KL %.4f' if key == 'kl' else 'top-1 %.3f') % m[k] if m.get(k) is not None else None
    simple = {'prefill': ('prefill_tps', 'prefill %s'), 'memfree': ('node_memfree_min_gib', 'node0 MemFree %s GiB'),
              'vram': ('vram_post_gib', 'VRAM %s GiB post-warm-up'), 'swapin': ('swap_in_delta', 'pswpin delta %s'), 'c8': ('c8_aggregate_tps', 'c8 aggregate %s'),
              'accept': ('draft_accept', 'acceptance %s')}
    if key in simple:
        k, fmt = simple[key]
        return fmt % _g(m[k]) if m.get(k) is not None else None
    if key == 'outcome':
        return {'ok': 'runs', 'oom': 'OOM', 'stopped': 'stopped', 'invalid': 'stopped invalid', 'crash': 'crashed'}[m['outcome']]
    raise KeyError('unknown lever field ' + key)


def outcome_text(run):
    m = run['measured']
    d = m.get('decode_tps', {})
    parts = []
    if d:
        parts.append('decode ' + ' / '.join('%s %s' % (k, _g(v)) for k, v in d.items()))
    if m.get('prefill_tps') is not None:
        parts.append('prefill %s' % _g(m['prefill_tps']))
    if m.get('kl') is not None:
        parts.append('KL %.4f / top-1 %.3f' % (m['kl'], m['top1']))
    if m.get('node_memfree_min_gib') is not None:
        parts.append('node0 MemFree %s GiB' % _g(m['node_memfree_min_gib']))
    head = {'ok': '', 'oom': 'OOM', 'stopped': 'stopped', 'invalid': 'stopped invalid', 'crash': 'crashed'}[m['outcome']]
    txt = ', '.join(([head] if head else []) + parts)
    if m.get('degraded'):
        txt += ' [' + '; '.join(m['degraded']) + ']'
    if m.get('notes'):
        txt += ' (' + m['notes'] + ')'
    return txt


def cost_text(run):
    bad = went_bad(run)
    if not bad:
        return ''
    t0 = datetime.datetime.strptime(run['when'][:16], '%Y-%m-%dT%H:%M')   # the ledger allows seconds (record_run WHEN_RE)
    w = run['measured'].get('wall_min')
    return '%s-%s, ~%d min (%s)' % (t0.strftime('%H:%M'), (t0 + datetime.timedelta(minutes=w)).strftime('%H:%MZ'), round(w), bad[0]) if w else '%s (%s)' % (t0.strftime('%H:%MZ'), bad[0])


def observed(run):
    """what the run itself showed, overlaid on the pre-launch prediction in the hard-check table"""
    m, o = run['measured'], {}
    if m.get('node_memfree_min_gib') is not None:
        o['numa'] = {'observed_memfree_gib': m['node_memfree_min_gib']}
    elif m.get('numa_clean'):
        o['numa'] = {'observed_clean': m['numa_clean']}
    if m.get('kl') is not None:
        o['decode_kl'] = m['kl']
    return o


# in-load draft twins: the served draft width vs no draft, same load and prompts (the b phase repeats the first: ABA)
DRAFT_TWINS = ((('m3', 'm3b'), 'm0', 'MTP n_max 3'), (('n4', 'n4b'), 'n0', 'DSpark n_max 4'))


def draft_twins(runs, exclude=()):
    """{box|model: [twin]} per ledger run that decoded the served draft width and no draft in one load. A twin is left out when a
    degraded note names one of its phases ("m0 prompt1 ... 3.10 tok/s"); a note about the whole load (placement, MemFree) hits
    both phases alike and stays."""
    out = {}
    for run in runs:
        if run['id'] in exclude:
            continue
        d = run['measured'].get('decode_tps')
        if not isinstance(d, dict):
            continue
        notes = ' '.join(run['measured'].get('degraded', []))
        for on, off, what in DRAFT_TWINS:
            xs = [d[k] for k in on if k in d]
            hit = [k for k in on + (off,) if k + ' prompt' in notes]
            if xs and d.get(off) and not hit:
                out.setdefault(run['box'] + '|' + run['model'], []).append(
                    {'id': run['id'], 'what': what, 'mult': round(sum(xs) / len(xs) / d[off], 4),
                     'phases': ' / '.join('%s %s' % (k, _g(d[k])) for k in on + (off,) if k in d)})
    return out

def recipes_from_runs(runs):
    return [{'id': r['id'], 'run_id': r['id'], 'when': r['when'][11:16] + 'Z', 'box': r['box'], 'model': r['model'], 'engine': r['engine'],
             'label': r['recipe'].get('label', r['id']), 'outcome': outcome_text(r), 'wasted': cost_text(r), 'recipe': r['recipe'], 'observed': observed(r),
             'src': '; '.join(r['sources'])} for r in sorted(runs, key=lambda r: r['when'])]


def decode_fits(root, runs, recipes):
    """Replay current decode operands without changing ledger recipes or saved launch predictions. Holdout measurements never
    enter the solver's fit table; the holdout ids also exclude their twins from the older box-wide fit."""
    specs = json.load(open(os.path.join(root, 'data', 'decode-fits.json')))
    by = {r['id']: r for r in runs}
    replay = {r['run_id']: r for r in recipes}
    for key, spec in specs.items():
        if not isinstance(spec.get('knobs'), dict) or not spec['knobs']:
            raise ValueError('decode fit measured knob envelope is missing: ' + key)
        if set(spec['train']) & set(spec['holdout']):
            raise ValueError('decode fit training overlaps holdout: ' + key)
        for rid, reads in spec['recipes'].items():
            replay[rid]['recipe'] = dict(by[rid]['recipe'], decode_profile=key, decode_reads=reads)
        spec['training'] = [{'id': rid, 'recipe': replay[rid]['recipe'], 'decode_tps': by[rid]['measured']['decode_tps']}
                            for rid in spec['train']]
    return specs


def levers_from_runs(defs, runs, tag):
    by = {r['id']: r for r in runs}

    def cell(c):
        if isinstance(c, str):
            return c, []
        out, src = [], []
        for x in c:
            r = by[x['run']]
            vals = [v for v in (field(r, k) for k in x['show']) if v]
            out.append(', '.join(vals) + (' (%s)' % x['note'] if x.get('note') else '') + ' [%s]' % r['recipe'].get('id', r['id']))
            src.append(r['id'] + ': ' + '; '.join(r['sources']))
        return ' / '.join(out), src

    rows = []
    for d in defs['levers']:
        b, sb = cell(d['before'])
        a, sa = cell(d['after'])
        src = d.get('src') or ' | '.join(dict.fromkeys(sb + sa))
        rows.append({'box': d['box'], 'model': d['model'], 'lever': d['lever'], 'before': b, 'after': a, 'tag': tag, 'src': src,
                     'runs': [x['run'] for c in (d['before'], d['after']) if not isinstance(c, str) for x in c]})
    return rows
