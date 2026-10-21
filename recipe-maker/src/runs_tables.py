"""Tables the report and the page show, generated from the run ledger (runs/runs.jsonl) and runs/levers.json.

recipes_from_runs: one row per measured run for the hard-check section: the recipe the gate consumes, the run's own
observations (node MemFree, decode KL) that overlay the prediction, and outcome / cost text built from the measured fields.
levers_from_runs: the measured-lever table; each cell names ledger runs and the fields to show.
"""
import datetime
import json
import os
import re

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
# kits that run one workload per phase group name MTP phases with the workload around the width: std-m3 / std-n0,
# d600_m3_mean / d600_n0_mean. The no-draft phase of such a group is m0 or n0 (the draft is off either way). Only the MTP
# family is read this way: a prefixed n4 phase names an external draft whose kit we never recorded as a served width
MTP_TOKEN = re.compile(r'(?:^|(?<=[-_]))(m3b?)(?=$|[-_])')
DSPARK_TOKEN = re.compile(r'(?:^|(?<=[-_]))(n4b?)(?=$|[-_])')
OFF_TOKENS = ('m0', 'n0')


def _groups(d, rx=MTP_TOKEN):
    """[(on keys, off key)] per workload group of prefixed draft phases; per-prompt keys yield to the group's _mean key"""
    out = {}
    for k in d:
        m = rx.search(k)
        if not m or k == m.group(1):
            continue
        pre, post = k[:m.start()], k[m.end():]
        off = next((pre + o + post for o in OFF_TOKENS if pre + o + post in d), None)
        if off:
            out.setdefault((pre, post), ([], off))[0].append(k)
    means = {g: x for g, x in out.items() if g[1].endswith('_mean')}
    return list((means or out).values())


def draft_twins(runs, exclude=()):
    """{box|model: [twin]} per ledger run that decoded the served draft width and no draft in one load. A twin is left out when a
    degraded note names one of its phases ("m0 prompt1 ... 3.10 tok/s"); a note about the whole load (placement, MemFree) hits
    both phases alike and stays. A run with prefixed MTP phase groups (std-m3 / std-n0) gives one twin: the mean over its groups"""
    out = {}
    for run in runs:
        if run['id'] in exclude:
            continue
        d = run['measured'].get('decode_tps')
        if not isinstance(d, dict):
            continue
        notes = ' '.join(run['measured'].get('degraded', []))
        found = False
        for on, off, what in DRAFT_TWINS:
            xs = [d[k] for k in on if k in d]
            hit = [k for k in on + (off,) if k + ' prompt' in notes]
            if xs and d.get(off) and not hit:
                found = True
                out.setdefault(run['box'] + '|' + run['model'], []).append(
                    {'id': run['id'], 'what': what, 'mult': round(sum(xs) / len(xs) / d[off], 4),
                     'phases': ' / '.join('%s %s' % (k, _g(d[k])) for k in on + (off,) if k in d)})
        groups = [] if found else [g for g in _groups(d) if not any(k + ' prompt' in notes for k in g[0] + [g[1]]) and d[g[1]]]
        if groups:
            ms = [sum(d[k] for k in on) / len(on) / d[off] for on, off in groups]
            out.setdefault(run['box'] + '|' + run['model'], []).append(
                {'id': run['id'], 'what': 'MTP n_max 3', 'mult': round(sum(ms) / len(ms), 4),
                 'phases': ' / '.join('%s %s' % (k, _g(d[k])) for on, off in groups for k in on + [off])})
    return out


def served(run):
    """the run's measured decode for the page's measured-run markers: {on, on_phase, off, off_phase, other, other_phase}.
    on = the served draft width (n4/m3 and the b repeats, or their prefixed groups std-n4, d600_m3_mean), off = no draft in
    the same load (n0/m0 or the group's), other = the primary phase of a run with neither (measured.primary_phase, else
    c1 / all / a *_c1 phase, else its first phase with guess=True: the page lists it but compares nothing). None = no decode
    measured"""
    m = run['measured']
    d = m.get('decode_tps')
    if not isinstance(d, dict) or not d:
        return None
    mean = lambda ks: round(sum(d[k] for k in ks) / len(ks), 3) if ks else None
    out = {}
    for on, off, _ in DRAFT_TWINS:
        if any(k in d for k in on):
            out['on_phase'] = [k for k in on if k in d]
            if off in d:
                out['off_phase'] = [off]
            break
    else:
        gs = _groups(d) or _groups(d, DSPARK_TOKEN)
        if gs:
            out = {'on_phase': [k for g in gs for k in g[0]], 'off_phase': [g[1] for g in gs]}
        else:
            lone = [k for k in d if DSPARK_TOKEN.search(k) or MTP_TOKEN.search(k)]
            out = {'on_phase': lone} if lone else {}
    if 'off_phase' not in out:
        off = [k for k in OFF_TOKENS if k in d] or [k for k in d if re.search(r'(?:^|(?<=[-_]))[mn]0(?=$|[-_])', k) and not k.endswith(('_p0', '_p1', '_p2'))]
        if off:
            out['off_phase'] = off
    guess = False
    if not out:
        ph = m.get('primary_phase') if m.get('primary_phase') in d else next((k for k in ('c1', 'all') if k in d), None) \
            or next((k for k in d if re.search(r'(?:^|[-_])c1$', k)), None)
        guess = ph is None   # per-prompt or per-pass phases (DV0_p0, M3_p0, V0_0): no single served decode to compare
        out = {'other_phase': [ph or next(iter(d))]}
    r = {k: v for x in ('on', 'off', 'other') if out.get(x + '_phase') for k, v in ((x, mean(out[x + '_phase'])), (x + '_phase', '/'.join(out[x + '_phase'])))}
    return dict(r, guess=True) if guess else r


def recipes_from_runs(runs, holdout=()):
    """one row per run; 'measured' is what the page's measured-run markers read: the served decode (served()), whether the run
    went bad (outcome or a degraded note: its speed is not the recipe's), its measured top-1, and whether a decode fit holds it out"""
    def meas(r):
        s = served(r)
        return s and dict(s, bad=bool(went_bad(r)), top1=r['measured'].get('top1'), holdout=r['id'] in holdout)
    return [{'id': r['id'], 'run_id': r['id'], 'when': r['when'][11:16] + 'Z', 'box': r['box'], 'model': r['model'], 'engine': r['engine'],
             'label': r['recipe'].get('label', r['id']), 'outcome': outcome_text(r), 'wasted': cost_text(r), 'recipe': r['recipe'], 'observed': observed(r),
             'measured': meas(r), 'src': '; '.join(r['sources'])} for r in sorted(runs, key=lambda r: r['when'])]


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
