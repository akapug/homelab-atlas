#!/usr/bin/env python3
"""Read a model-screening round into the screening ledger (runs/screen.jsonl): one row per model arm, beside the speed rows.

A row holds the settings the arm actually ran with (request body over the server defaults, each with its source), every
request's failure shapes (cap hit, empty answer, loop, leaked template tags), the items answered only on retry, the grades
and the verdict the round recorded. The verdict-ready check (src/model.js verdictReady) reads the row against the model's
card conditions (data/cards/<card>.json).

  record_screen.py --spec tools/screen_sources.json                # print the rows
  record_screen.py --spec tools/screen_sources.json --write        # append new rows; an id already there is refused
  record_screen.py --spec tools/screen_sources.json --write --replace

Each arm's answers come from model-screen.py ask: answers/<arm>/<item>.json, a retry as <item>-retry.json, each with
extra_body, max_tokens, finish, usage, answer and reasoning.
"""
import argparse
import collections
import datetime
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LEDGER = os.path.join(ROOT, 'runs', 'screen.jsonl')
ID_RE = re.compile(r'^[a-z0-9][a-z0-9._-]*$')
PURPOSES = ('verdict', 'sweep')
SAMPLING = ('temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'repeat_penalty')
# a server default from /props is a float32: round it so 0.05000000074 reads 0.05
FLOAT_DIGITS = 6
# loop: a line of at least LOOP_MIN_LINE characters seen LOOP_REPEATS times in the last LOOP_TAIL_CHARS characters of the
# reasoning + answer. Code answers repeat a diff line two to four times (10-21 round: at most 4), a loop repeats without end
LOOP_TAIL_CHARS = 6000
LOOP_MIN_LINE = 24
LOOP_REPEATS = 6
# template tags that never belong in the visible answer: their presence means the reasoning parser or the template failed
LEAK_TAGS = ('<think>', '</think>', '<|im_start|>', '<|im_end|>', '<|endoftext|>', '<tool_call>', '</tool_call>')
RETRY_SUFFIX = '-retry'


def read(p):
    with open(p) as f:
        return f.read()


def sha_text(t):
    return 'sha256:' + hashlib.sha256(t.strip().encode()).hexdigest()


def loop_line(text, tail=LOOP_TAIL_CHARS, min_line=LOOP_MIN_LINE, repeats=LOOP_REPEATS):
    """the repeated line when the tail of text loops, else None"""
    c = collections.Counter(x.strip() for x in text[-tail:].splitlines() if len(x.strip()) >= min_line)
    top = c.most_common(1)
    return top[0][0] if top and top[0][1] >= repeats else None


def request_shapes(a, rules):
    """one answer file -> its failure shapes"""
    ans, use = a.get('answer') or '', a.get('usage') or {}
    lp = loop_line((a.get('reasoning') or '') + '\n' + ans, rules['loop_tail_chars'], rules['loop_min_line'], rules['loop_repeats'])
    return {'finish': a.get('finish'), 'completion_tokens': use.get('completion_tokens'), 'prompt_tokens': use.get('prompt_tokens'),
            'wall_s': a.get('wall_s'), 'answer_chars': len(ans),
            'cap_hit': a.get('finish') == 'length' or (use.get('completion_tokens') or 0) >= (a.get('max_tokens') or float('inf')),
            'empty': not ans.strip(), 'loop': lp, 'leaked_tags': [t for t in rules['leak_tags'] if t in ans]}


def failed(r):
    return r['cap_hit'] or r['empty'] or bool(r['loop']) or bool(r['leaked_tags'])


def read_answers(adir):
    """[(item, try, answer json)] in item order, first try before the retry"""
    out = []
    for f in sorted(os.listdir(adir)):
        if f.endswith('.json'):
            stem = f[:-5]
            retry = stem.endswith(RETRY_SUFFIX)
            out.append((stem[:-len(RETRY_SUFFIX)] if retry else stem, 2 if retry else 1, json.loads(read(os.path.join(adir, f)))))
    return sorted(out, key=lambda x: (x[0], x[1]))


def template_effort(tpl):
    """the reasoning_effort a template falls back to when the request sets none; 'none' when it has no effort control"""
    m = re.search(r"reasoning_effort\s*\|\s*default\(\s*['\"](\w+)['\"]", tpl)
    return m.group(1) if m else ('none' if 'reasoning_effort' not in tpl else None)


def settings_used(answers, props, props_src, server_log, set_):
    """the settings the arm ran with, and the source of each"""
    bodies = {json.dumps(a.get('extra_body') or {}, sort_keys=True) for _, _, a in answers}
    caps = {a.get('max_tokens') for _, _, a in answers}
    if len(bodies) > 1 or len(caps) > 1:
        raise ValueError('requests of one arm ran with different settings: %s / max_tokens %s' % (sorted(bodies), sorted(caps)))
    body, params = json.loads(bodies.pop()), (props.get('default_generation_settings') or {}).get('params') or {}
    kw = body.get('chat_template_kwargs') or {}
    st, src = {}, {}
    for k in SAMPLING:
        if k in body:
            st[k], src[k] = body[k], 'request body (answers/*.json extra_body)'
        elif params.get(k) is not None:
            st[k], src[k] = round(params[k], FLOAT_DIGITS), 'server default the request did not override (%s)' % props_src
    st['max_output'], src['max_output'] = caps.pop(), 'request max_tokens (answers/*.json)'
    m = re.search(r'n_ctx_slot = (\d+)', read(server_log)) if server_log and os.path.exists(server_log) else None
    st['context'], src['context'] = (int(m.group(1)), 'server.log n_ctx_slot') if m else ((props.get('default_generation_settings') or {}).get('n_ctx'), props_src + ' n_ctx')
    tpl = props.get('chat_template') or ''
    st['chat_template'], src['chat_template'] = (sha_text(tpl), props_src + ' chat_template (the GGUF template the server ran)') if tpl else (None, 'no chat_template in ' + props_src)
    reasoned = any((a.get('reasoning') or '').strip() for _, _, a in answers)
    st['thinking_mode'] = 'off' if kw.get('enable_thinking') is False else 'on' if reasoned else 'off'
    src['thinking_mode'] = 'chat_template_kwargs.enable_thinking false' if kw.get('enable_thinking') is False else 'answers carry reasoning text' if reasoned else 'no answer carries reasoning text'
    eff = body.get('reasoning_effort', kw.get('reasoning_effort'))
    st['reasoning_effort'], src['reasoning_effort'] = (eff, 'request body') if eff is not None else (template_effort(tpl), 'the served template default (request set none)')
    st['reasoning_budget'], src['reasoning_budget'] = None, 'none recorded for this arm (tools/screen_sources.json set)'
    st['reasoning_parser'], src['reasoning_parser'] = ('llama.cpp --jinja chat parser: reasoning returned apart', 'answers carry a reasoning field') if reasoned else ('none seen', 'no answer carries a reasoning field')
    for k, (val, why) in (set_ or {}).items():
        st[k], src[k] = val, why
    return st, src


def grade_summary(path, arm):
    """last grade row per item for this arm (grades.jsonl, last row wins) and the round's tallies"""
    last = {}
    for x in read(path).splitlines():
        if x.strip():
            g = json.loads(x)
            if g.get('model') == arm:
                last[g['item']] = g
    fp = lambda g, k: g.get(k, g.get('false', 0)) or 0
    return {'items': {k: g['verdict'] for k, g in sorted(last.items())},
            'verdicts': dict(collections.Counter(g['verdict'] for g in last.values())),
            'fp_strict': sum(fp(g, 'strict_fp') for g in last.values()), 'fp_lenient': sum(fp(g, 'lenient_fp') for g in last.values()),
            'design': sum(g.get('design', 0) or 0 for g in last.values())}


def results_row(path, heading, label):
    """the cells of the row whose first cell starts with label, in the first table under the first line starting with heading;
    None when absent"""
    lines = read(path).splitlines()
    at = [i for i, x in enumerate(lines) if x.startswith(heading)]
    if not at:
        return None
    rows = []
    for x in lines[at[0] + 1:]:
        if x.startswith('|'):
            rows.append([c.strip() for c in x.strip().strip('|').split('|')])
        elif rows:
            break
    if len(rows) < 2:
        return None
    hit = [r for r in rows[2:] if r[0].replace('*', '').startswith(label)]
    return dict(zip(rows[0], hit[0])) if hit else None


def when_of(answers):
    ts = [a['ts'] for _, _, a in answers if a.get('ts')]
    return datetime.datetime.fromtimestamp(min(ts), datetime.timezone.utc).strftime('%Y-%m-%dT%H:%MZ') if ts else None


def build_row(rnd, arm, spec, rules, base=None):
    """one ledger row for one arm of one round"""
    d = os.path.expanduser(base or rnd['dir'])
    adir = os.path.join(d, rnd['answers'], arm)
    answers = read_answers(adir)
    props_src = spec['props']
    props = json.loads(read(os.path.join(d, props_src)))
    st, src = settings_used(answers, props, props_src, os.path.join(adir, 'server.log'), spec.get('set'))
    reqs = [dict(request_shapes(a, rules), item=item, attempt=n) for item, n, a in answers]
    by = collections.defaultdict(list)
    for r in reqs:
        by[r['item']].append(r)
    retry_only = sorted(i for i, rs in by.items() if len(rs) > 1 and failed(rs[0]) and not rs[-1]['empty'] and not rs[-1]['cap_hit'])
    shapes = {'requests': len(reqs), 'items': len(by), 'cap_hit': sum(r['cap_hit'] for r in reqs), 'empty': sum(r['empty'] for r in reqs),
              'loop': sum(bool(r['loop']) for r in reqs), 'leaked_tags': sum(bool(r['leaked_tags']) for r in reqs),
              'any': sum(failed(r) for r in reqs), 'retry_only': len(retry_only), 'retry_only_items': retry_only}
    res = os.path.join(d, rnd['results']) if rnd.get('results') else None
    row = {'id': 'screen-%s-%s' % (rnd['round'], arm), 'when': when_of(answers), 'round': rnd['round'], 'arm': arm, 'card': spec['card'],
           'purpose': spec.get('purpose', rnd.get('purpose', 'verdict')), 'box': rnd['box'], 'engine': rnd['engine'], 'gpu': rnd.get('gpu'),
           'model_file': os.path.basename(props.get('model_path') or ''), 'settings': st, 'settings_src': src,
           'prompt_tokens_max': max((r['prompt_tokens'] or 0) for r in reqs) if reqs else None,
           'requests': reqs, 'shapes': shapes, 'shape_rules': rules,
           'grades': grade_summary(os.path.join(d, rnd['grades']), arm) if rnd.get('grades') else None,
           'results_row': results_row(res, rnd['results_heading'], spec['results_label']) if res and spec.get('results_label') else None,
           'verdict': spec.get('verdict'), 'sweep': spec.get('sweep'),
           'sources': [os.path.join(rnd['dir'], rnd['answers'], arm), os.path.join(rnd['dir'], props_src)] +
                      [os.path.join(rnd['dir'], rnd[k]) for k in ('grades', 'results') if rnd.get(k)]}
    return row


def validate(row):
    """a list of problems (empty = valid)"""
    bad = []
    for k, t in (('id', str), ('round', str), ('card', str), ('box', str), ('engine', str), ('settings', dict), ('settings_src', dict),
                 ('requests', list), ('shapes', dict), ('sources', list)):
        if not isinstance(row.get(k), t):
            bad.append('%s: missing or not %s' % (k, t.__name__))
    if bad:
        return bad
    if not ID_RE.match(row['id']):
        bad.append('id %r: lower-case letters, digits, . _ - only' % row['id'])
    if row.get('purpose') not in PURPOSES:
        bad.append('purpose %r: one of %s' % (row.get('purpose'), ', '.join(PURPOSES)))
    bad += ['setting %s has no source' % k for k in row['settings'] if not row['settings_src'].get(k)]
    if row['purpose'] == 'sweep' and not (row.get('sweep') or {}).get('best'):
        bad.append('a sweep row names its best settings (sweep.best)')
    if row['shapes'].get('requests') != len(row['requests']):
        bad.append('shapes.requests %s != %d request rows' % (row['shapes'].get('requests'), len(row['requests'])))
    return bad


def write(rows, path, replace):
    old = [json.loads(x) for x in read(path).splitlines() if x.strip()] if os.path.exists(path) else []
    ids = {r['id'] for r in rows}
    clash = sorted(ids & {r['id'] for r in old})
    if clash and not replace:
        raise SystemExit('already in %s: %s (--replace to rewrite them)' % (path, ', '.join(clash)))
    keep = [r for r in old if r['id'] not in ids]
    with open(path, 'w') as f:
        for r in keep + rows:
            f.write(json.dumps(r, sort_keys=True) + '\n')
    return len(keep), len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--spec', required=True)
    ap.add_argument('--round', help='only this round of the spec')
    ap.add_argument('--dir', help='read the round from this directory instead of its spec dir (a copy)')
    ap.add_argument('--ledger', default=LEDGER)
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--replace', action='store_true')
    ap.add_argument('--loop-tail-chars', type=int, default=LOOP_TAIL_CHARS)
    ap.add_argument('--loop-min-line', type=int, default=LOOP_MIN_LINE)
    ap.add_argument('--loop-repeats', type=int, default=LOOP_REPEATS)
    a = ap.parse_args(argv)
    rules = {'loop_tail_chars': a.loop_tail_chars, 'loop_min_line': a.loop_min_line, 'loop_repeats': a.loop_repeats, 'leak_tags': list(LEAK_TAGS)}
    rounds = [r for r in json.loads(read(a.spec))['rounds'] if not a.round or r['round'] == a.round]
    rows = [build_row(r, arm, spec, rules, a.dir) for r in rounds for arm, spec in r['arms'].items()]
    bad = {r['id']: validate(r) for r in rows if validate(r)}
    if bad:
        raise SystemExit('invalid rows: %s' % json.dumps(bad, indent=1))
    if a.write:
        kept, added = write(rows, a.ledger, a.replace)
        print('%s: %d rows kept, %d written' % (a.ledger, kept, added))
    for r in rows:
        s = r['shapes']
        print('%s: %d requests, %d with a failure shape (cap %d, empty %d, loop %d, leaked %d), answered only on retry %d' %
              (r['id'], s['requests'], s['any'], s['cap_hit'], s['empty'], s['loop'], s['leaked_tags'], s['retry_only']))
    return rows


if __name__ == '__main__':
    main(sys.argv[1:])
