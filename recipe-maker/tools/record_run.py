#!/usr/bin/env python3
"""Append one measured run to the run ledger (runs/runs.jsonl), or parse a run log into measured fields.

A ledger row holds the exact recipe the pre-launch gate consumed, the gate's --check --json output captured at
launch (with the planner's prediction), and what the run measured. Every measured number names its source.

  record_run.py --recipe R.json --check C.json --measured M.json [--id ID] [--when ISO] [--source S]... [--tag T]...
  record_run.py --recipe R.json --no-check --from-log kit:droopy:~/optane-b1d.log --arm b1d-... --source ...
  record_run.py --from-log bench-log:res/k2e-1450-bench.log --from-log kl:res/k2e-1450-kl.json      (print measured JSON)

--check C.json is `node report.js --check R.json --json` saved before the launch. --no-check records a run whose
gate output was not captured (backfill); its prediction is then computed now and marked post-hoc.
Pinned metadata-only intakes can be recorded with --no-check for a known box and engine. Their exact
intake bytes (including biases) travel with an UNMODELLED prediction; this does not admit a launch.

Log kinds for --from-log KIND:PATH (PATH may be host:path, read with `ssh host cat path`, read-only):
  kit         droopy run-kit log (== <kit> start / JSON arm lines / == <arm> rc=N / prompt lines / N0 LOW watch)
  bench-log   bench.py stdout (one JSON line per request, "C1 decode median", "VRAM peak N MiB")
  bench-json  bench.py result file bench-<label>.json (GLM glm-lite, Flash-Next): c1 / prefill / c8 lists
  kl          decode-KL result json (served_Aref_vs_Bvar kl_mean / top1_agree, verdict)
"""
import argparse
import datetime
import json
import os
import re
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))
import ledger as LEDGER  # noqa: E402
RUNS = os.path.join(ROOT, 'runs', 'runs.jsonl')
OUTCOMES = ('ok', 'oom', 'stopped', 'invalid', 'crash')
ID_RE = re.compile(r'^[a-z0-9][a-z0-9._-]*$')
WHEN_RE = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?Z$')
# the run kit's node0 watch prints a line every 30 s while node0 MemFree is below this (run-b1d.sh)
KIT_WATCH_BAR_GIB = 3
MIB_PER_GIB = 1024


# ---------------- validation ----------------
def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate(row):
    """Return a list of problems (empty = valid)."""
    bad = []
    for k, t in (('id', str), ('when', str), ('box', str), ('model', str), ('engine', str), ('recipe', dict),
                 ('measured', dict), ('sources', list), ('tags', list)):
        if not isinstance(row.get(k), t):
            bad.append('%s: missing or not %s' % (k, t.__name__))
    if bad:
        return bad
    if not ID_RE.match(row['id']):
        bad.append('id %r: lower-case letters, digits, . _ - only' % row['id'])
    if not WHEN_RE.match(row['when']):
        bad.append('when %r: UTC ISO like 2026-10-17T15:52Z' % row['when'])
    for k in ('box', 'model', 'engine'):
        if row['recipe'].get(k) != row[k]:
            bad.append('recipe.%s %r != row %s %r' % (k, row['recipe'].get(k), k, row[k]))
    for k in ('check', 'prediction'):
        if k not in row:
            bad.append('%s: missing (null allowed)' % k)
        elif row[k] is not None and not isinstance(row[k], dict):
            bad.append('%s: object or null' % k)
    if not row['sources'] or not all(isinstance(s, str) and s for s in row['sources']):
        bad.append('sources: at least one non-empty string')
    m = row['measured']
    if m.get('outcome') not in OUTCOMES:
        bad.append('measured.outcome %r: one of %s' % (m.get('outcome'), '|'.join(OUTCOMES)))
    src = m.get('src')
    if not isinstance(src, dict):
        bad.append('measured.src: object naming the source of every measured number')
        src = {}
    d = m.get('decode_tps', {})
    if not isinstance(d, dict):
        bad.append('measured.decode_tps: {phase: tok/s}')
        d = {}
    for ph, x in d.items():
        if not _num(x):
            bad.append('measured.decode_tps.%s: not a number' % ph)
        elif not (src.get('decode_tps.' + ph) or src.get('decode_tps')):
            bad.append('measured.decode_tps.%s has no source (measured.src)' % ph)
    for k, x in m.items():
        if k in ('decode_tps', 'src'):
            continue
        if _num(x) and not src.get(k):
            bad.append('measured.%s has no source (measured.src)' % k)
        if isinstance(x, dict) and any(_num(y) for y in x.values()) and not src.get(k):
            bad.append('measured.%s has no source (measured.src)' % k)
    if 'degraded' in m and not (isinstance(m['degraded'], list) and all(isinstance(s, str) for s in m['degraded'])):
        bad.append('measured.degraded: list of reasons')
    if m.get('outcome') == 'ok' and not d and 'decode_tps' not in m.get('not_measured', []):
        bad.append('measured: outcome ok but no decode_tps (list it in measured.not_measured if it truly was not measured)')
    return bad


def load(path=RUNS):
    return LEDGER.rows(path)


def append(row, path=RUNS):
    bad = validate(row)
    if bad:
        raise ValueError('invalid run row %s:\n  ' % row.get('id') + '\n  '.join(bad))
    if any(r['id'] == row['id'] for r in load(path)):
        raise ValueError('duplicate id %s in %s' % (row['id'], path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(row, separators=(',', ':'), sort_keys=False) + '\n')
    k, seg = LEDGER.pack(path)   # the writer keeps the open file under its size bound (src/ledger.py)
    if k:
        print('sealed %d older rows into %s' % (k, seg))


# ---------------- reading logs (local or host:path over ssh, read-only) ----------------
def read_text(spec):
    """PATH or host:PATH (ssh host cat PATH; a local path wins when it exists). PATH#L10-L20 keeps lines 10..20 only."""
    rng = re.search(r'#L(\d+)-L?(\d+)$', spec)
    if rng:
        spec = spec[:rng.start()]
    if os.path.exists(os.path.expanduser(spec)) or ':' not in spec:
        with open(os.path.expanduser(spec), errors='replace') as f:
            text = f.read()
    else:
        host, path = spec.split(':', 1)
        rel = path[2:] if path.startswith('~/') else None
        cmd = 'cat ' + ('~/' + shlex.quote(rel) if rel else shlex.quote(path))
        text = subprocess.run(['ssh', '-o', 'BatchMode=yes', host, cmd], check=True, capture_output=True, text=True, timeout=120).stdout
    if rng:
        text = '\n'.join(text.splitlines()[int(rng.group(1)) - 1:int(rng.group(2))]) + '\n'
    return text


def _ts(day, hms):
    """kit timestamps carry the time only: attach the kit's start date (a later time of day stays on that date)."""
    return datetime.datetime.strptime(day + 'T' + hms.rstrip('Z'), '%Y-%m-%dT%H:%M:%S').replace(tzinfo=datetime.timezone.utc)


def _iso(t):
    return t.strftime('%Y-%m-%dT%H:%M:%SZ')


def _phase_of(extra):
    m = re.search(r'--spec-type \w+:n_max=(\d+)', extra or '')
    return 'n%s' % m.group(1) if m else 'n0'


# ---------------- droopy run-kit log ----------------
def parse_kit(text, src):
    """Split a run-kit log into arms. Lines before an arm's rc line (floor checks, N0 watch, the arm's JSON) belong to
    that arm; lines after it (prompt summaries, placement and hot-expert reports, pswpin) belong to it too."""
    kits, arms = [], []
    kit = None
    pre, cur = [], None
    for ln in text.splitlines():
        s = re.match(r'^== (\S+) start (\S+Z)', ln)
        if s:
            kit = {'name': s.group(1), 'start': datetime.datetime.strptime(s.group(2), '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc),
                   'header': ln, 'system': {}, 'lows': [], 'terminated': False}
            kits.append(kit)
            pre, cur = [], None
            continue
        if kit is None:
            continue
        r = re.match(r'^== (\S+) rc=(-?\d+) (\d\d:\d\d:\d\dZ)', ln)
        if r:
            js = [json.loads(x) for x in pre if x.startswith('{')]
            js = [j for j in js if j.get('arm') == r.group(1)] or js[-1:]
            prev_end = arms[-1]['end'] if arms and arms[-1]['kit'] is kit else kit['start']
            cur = {'kit': kit, 'arm': r.group(1), 'rc': int(r.group(2)), 'end': _ts(kit['start'].strftime('%Y-%m-%d'), r.group(3)),
                   'start': prev_end, 'json': js[0] if js else {}, 'pre': pre, 'post': []}
            arms.append(cur)
            pre = []
            continue
        low = re.match(r'^== N0 LOW (\d\d:\d\d:\d\dZ): node0 MemFree (\d+(?:\.\d+)?) GiB', ln)
        if low:
            kit['lows'].append((_ts(kit['start'].strftime('%Y-%m-%d'), low.group(1)), float(low.group(2))))
            continue
        if ln.startswith('P0: '):
            for k, v in re.findall(r'(numa_balancing|demotion) (\S+)', ln):
                kit['system'][k] = int(v) if v.isdigit() else (v == 'true')
            if 'swap off' in ln:
                kit['system']['swap'] = 'off'
            elif re.search(r'swap on|swap \d+ active', ln):
                kit['system']['swap'] = 'on'
        if ln.strip() == 'Terminated':
            kit['terminated'] = True
        sw = re.match(r'^== pswpin delta (\d+)', ln)
        if sw:
            kit['pswpin_delta'] = int(sw.group(1))
        if ln.startswith('{') or re.match(r'^node\d MemFree', ln) or ln.startswith('P0: '):
            pre.append(ln)
            cur = None
        elif cur is not None:
            cur['post'].append(ln)
        else:
            pre.append(ln)
    out = []
    for a in arms:
        out.append(_kit_arm(a, src))
    for k in kits:
        if k['terminated'] and not any(a['kit'] is k for a in arms):
            out.append({'arm': None, 'kit': k['name'], 'start': _iso(k['start']), 'header': k['header'],
                        'measured': {'outcome': 'stopped', 'notes': 'kit terminated before any arm finished', 'src': {'outcome': src + ' (Terminated, no arm line)'}}})
    return out


def _kit_arm(a, src):
    j, kit = a['json'], a['kit']
    s = '%s [%s]' % (src, a['arm'])
    m = {'decode_tps': {}, 'src': {}}
    if 'REFUSED' in str(j.get('error', '')):
        return {'arm': a['arm'], 'refused': j['error'], 'measured': None}
    phases = j.get('phases')
    if phases:
        for p in phases:
            m['decode_tps'][p['phase']] = p['decode_mean']
        m['src']['decode_tps'] = s + ' phases[].decode_mean (3 prompts each)'
        pf = [x for p in phases for x in p.get('prefill_tps', [])]
        if pf:
            m['prefill_tps'] = round(sum(pf) / len(pf), 1)
            m['src']['prefill_tps'] = s + ' mean of all %d prompt prefills' % len(pf)
        acc = [(sum(p['draft_acc']), sum(p['draft_n'])) for p in phases if p.get('draft_n') and sum(p['draft_n'])]
        if acc:
            m['draft_accept'] = round(sum(x for x, _ in acc) / sum(y for _, y in acc), 3)
            m['src']['draft_accept'] = s + ' sum(draft_acc) / sum(draft_n) over draft phases'
    elif j.get('decode_tps'):
        ph = _phase_of(j.get('extra'))
        m['decode_tps'][ph] = round(sum(j['decode_tps']) / len(j['decode_tps']), 2)
        m['primary_phase'] = ph
        m['src']['decode_tps'] = s + ' decode_tps mean of %s (one phase; n_max from --spec-type)' % j['decode_tps']
        m['prefill_tps'] = round(sum(j['prefill_tps']) / len(j['prefill_tps']), 1)
        m['src']['prefill_tps'] = s + ' prefill_tps mean'
    if j.get('vram_after_warmup'):
        m['vram_post_gib'] = round(j['vram_after_warmup']['used_mib'] / MIB_PER_GIB, 3)
        m['vram_total_gib'] = round(j['vram_after_warmup']['total_mib'] / MIB_PER_GIB, 3)
        m['src']['vram_post_gib'] = s + ' vram_after_warmup %d MiB' % j['vram_after_warmup']['used_mib']
        m['src']['vram_total_gib'] = s + ' vram total %d MiB' % j['vram_after_warmup']['total_mib']
    if j.get('vram_end'):
        m['vram_end_gib'] = round(j['vram_end']['used_mib'] / MIB_PER_GIB, 3)
        m['src']['vram_end_gib'] = s + ' vram_end %d MiB (sampled once after the prompts; not a peak)' % j['vram_end']['used_mib']
    if j.get('resident_GiB_by_node'):
        m['resident_gib_by_node'] = {k.replace('other_', ''): v for k, v in j['resident_GiB_by_node'].items()}
        m['src']['resident_gib_by_node'] = s + ' resident_GiB_by_node'
    walls = [int(x) for x in re.findall(r'^\S+ prompt\d+ .* wall (\d+)s', '\n'.join(a['post']), re.M)]
    timed_from = a['end'] - datetime.timedelta(seconds=sum(walls)) if walls else None
    if kit['lows']:
        win = [v for t, v in kit['lows'] if timed_from and timed_from <= t <= a['end']]
        if win:
            m['node_memfree_min_gib'] = min(win)
            m['src']['node_memfree_min_gib'] = s + ' N0 LOW watch, %d readings in the timed window %s-%s' % (len(win), timed_from.strftime('%H:%MZ'), a['end'].strftime('%H:%MZ'))
            m.setdefault('degraded', []).append('node0 MemFree below %d GiB during timed prompts (min %g)' % (KIT_WATCH_BAR_GIB, min(win)))
        elif timed_from:
            m['numa_clean'] = 'no N0 LOW reading (< %d GiB, 30 s watch) in the timed window %s-%s' % (KIT_WATCH_BAR_GIB, timed_from.strftime('%H:%MZ'), a['end'].strftime('%H:%MZ'))
            m['src']['numa_clean'] = s + ' N0 watch'
    if kit.get('pswpin_delta') is not None:
        m['swap_in_delta'] = kit['pswpin_delta']
        m['src']['swap_in_delta'] = src + ' == pswpin delta (whole kit)'
    if j.get('placement_ok') is False and a['rc'] == 0 and not j.get('client_rc'):
        m.setdefault('degraded', []).append('placement off target %s GiB' % j.get('placement_off_GiB'))
    if kit['system']:
        m['system'] = dict(kit['system'])
        m['src']['system'] = src + ' P0 lines'
    # a client step after the phases (KL gate, long-prompt phase) can fail with every phase measured: the server served the
    # whole run, so that is a degraded ok, not a crash (10-18 BE1-Q3e / q3f: the KL gate's KeyError / null logprobs read as crashes)
    n0 = len((phases or [{}])[0].get('decode_tps') or [])
    served = bool(phases) and n0 > 0 and all(len(p.get('decode_tps') or []) == n0 for p in phases)
    if (j.get('client_rc') or a['rc'] != 0) and served and not j.get('error'):
        m['outcome'] = 'ok'
        why = re.findall(r'^(\w*Error: .*|KL_GATE_ERROR .*)$', '\n'.join(a['post']), re.M)
        m.setdefault('degraded', []).append('client step rc %s after all %d phases were measured (a step after the phases failed: %s)' % (j.get('client_rc'), len(phases), why[-1][:160] if why else 'read the kit log'))
        if not j.get('kl_gate'):
            m['not_measured'] = sorted(set(m.get('not_measured', []) + ['kl_gate']))
    elif j.get('client_rc') or j.get('error') or a['rc'] != 0:
        m['outcome'] = 'crash'
        m['notes'] = 'client rc %s, kit rc %d: %s (OOM or crash: read the server log)' % (j.get('client_rc'), a['rc'], j.get('error', ''))
    else:
        m['outcome'] = 'ok'
    m['src']['outcome'] = s + ' rc line'
    m['wall_min'] = round((a['end'] - a['start']).total_seconds() / 60, 1)
    m['src']['wall_min'] = s + ' arm end minus previous arm end (kit start for the first arm): load + warm-up + prompts'
    hot = re.search(r'llama_hot_experts: \S+: (\d+) hot experts in \d+ layers, ([\d.]+) MiB on GPU', '\n'.join(a['post']))
    if hot:
        m['hot_experts'] = int(hot.group(1))
        m['hot_gib'] = round(float(hot.group(2)) / MIB_PER_GIB, 3)
        m['src']['hot_experts'] = m['src']['hot_gib'] = s + ' llama_hot_experts line'
    return {'arm': a['arm'], 'kit': kit['name'], 'start': _iso(a['start']), 'end': _iso(a['end']), 'extra': j.get('extra'), 'measured': m}


# ---------------- bench.py stdout ----------------
def parse_bench_log(text, src):
    m = {'decode_tps': {}, 'src': {}}
    med = re.search(r'^C1 decode median ([\d.]+) min ([\d.]+) max ([\d.]+) over (\d+)', text, re.M)
    if med:
        m['decode_tps']['c1'] = float(med.group(1))
        m['src']['decode_tps'] = src + ' "C1 decode median" line (bench.py: upper median of %s requests)' % med.group(4)
    pf = [json.loads(l) for l in text.splitlines() if l.startswith('{"prefill_rep"')]
    cold = [p for p in pf if p.get('prefill_rep') == 0]
    if cold:
        m['prefill_tps'] = round(cold[0]['prefill_tps_proxy'], 1)
        m['src']['prefill_tps'] = src + ' prefill_rep 0 (%d-token cold prompt; rep 1 hits the prefix cache)' % cold[0]['prompt_tokens']
    c8 = [json.loads(l) for l in text.splitlines() if l.startswith('{"c8_aggregate_tps"')]
    if c8:
        m['c8_aggregate_tps'] = round(c8[0]['c8_aggregate_tps'], 2)
        m['src']['c8_aggregate_tps'] = src + ' c8_aggregate_tps line'
    vp = re.search(r'^VRAM peak (\d+) MiB', text, re.M)
    if vp:
        m['vram_peak_gib'] = round(int(vp.group(1)) / MIB_PER_GIB, 3)
        m['src']['vram_peak_gib'] = src + ' "VRAM peak" line (gpu.csv sampler)'
    m['outcome'] = 'ok' if med else 'crash'
    m['src']['outcome'] = src + (' C1 summary present' if med else ' no C1 summary line')
    return m


# ---------------- bench.py result json (bench-<label>.json) ----------------
def bench_median(xs):
    """bench.py's own median: sorted(xs)[len // 2] (the upper median for an even count)."""
    s = sorted(xs)
    return s[len(s) // 2]


def parse_bench_json(text, src):
    d = json.loads(text)
    m = {'decode_tps': {}, 'src': {}}
    c1 = [r['decode_tps'] for r in d.get('c1', []) if r.get('decode_tps') is not None]
    if c1:
        m['decode_tps']['c1'] = round(bench_median(c1), 2)
        m['src']['decode_tps'] = src + ' c1[].decode_tps, bench.py median of %d' % len(c1)
    cold = [p for p in (d.get('prefill') or []) if p.get('prefill_rep', 0) == 0 and p.get('prefill_tps_proxy')]
    if cold:
        m['prefill_tps'] = round(cold[0]['prefill_tps_proxy'], 1)
        m['src']['prefill_tps'] = src + ' prefill[rep 0].prefill_tps_proxy'
    c8 = d.get('c8')
    if isinstance(c8, dict) and c8.get('aggregate_tps') is not None:
        m['c8_aggregate_tps'] = round(c8['aggregate_tps'], 2)
        m['src']['c8_aggregate_tps'] = src + ' c8.aggregate_tps'
    if d.get('started') and d.get('finished'):
        t0 = datetime.datetime.strptime(d['started'], '%Y-%m-%dT%H:%M:%SZ')
        t1 = datetime.datetime.strptime(d['finished'], '%Y-%m-%dT%H:%M:%SZ')
        m['bench_min'] = round((t1 - t0).total_seconds() / 60, 1)
        m['src']['bench_min'] = src + ' started..finished (bench only, not the load)'
    m['outcome'] = 'ok' if c1 else 'crash'
    m['src']['outcome'] = src + (' c1 results present' if c1 else ' no c1 results')
    return m


# ---------------- decode-KL json ----------------
def parse_kl(text, src):
    d = json.loads(text)
    s = d.get('served_Aref_vs_Bvar') or {}
    m = {'src': {}}
    if 'kl_mean' in s:
        m['kl'] = round(s['kl_mean'], 5)
        m['top1'] = round(s['top1_agree'], 4)
        m['src']['kl'] = src + ' served_Aref_vs_Bvar.kl_mean (%d positions)' % s.get('positions', 0)
        m['src']['top1'] = src + ' served_Aref_vs_Bvar.top1_agree'
    ref = (d.get('ref') or {}).get('served_A_vs_B1') or {}
    if 'kl_mean' in ref:
        m['kl_ref'] = round(ref['kl_mean'], 5)
        m['src']['kl_ref'] = src + ' ref.served_A_vs_B1.kl_mean (lane on vs exact, the noise floor)'
    if d.get('verdict'):
        m['kl_verdict'] = d['verdict'].get('overall')
    return m


PARSERS = {'kit': parse_kit, 'bench-log': parse_bench_log, 'bench-json': parse_bench_json, 'kl': parse_kl}


def home_rel(text):
    """a path under this user's home as ~/... (ledger rows name no user)"""
    return text.replace(os.path.expanduser('~') + '/', '~/')


def from_log(spec, arm=None):
    kind, path = spec.split(':', 1)
    if kind not in PARSERS:
        raise SystemExit('unknown log kind %s (kinds: %s)' % (kind, ', '.join(PARSERS)))
    src = home_rel(path)
    res = PARSERS[kind](read_text(path), src)
    if kind != 'kit':
        return res
    arms = [a for a in res if a.get('measured')]
    if arm:
        arms = [a for a in arms if a['arm'] == arm]
    if len(arms) != 1:
        raise SystemExit('%s: %d arms match; pick one with --arm (%s)' % (spec, len(arms), ', '.join(str(a['arm']) for a in res)))
    out = arms[0]['measured']
    out['_arm'] = {k: arms[0].get(k) for k in ('arm', 'kit', 'start', 'end', 'extra')}
    return out


def merge(a, b):
    """measured b into a: decode phases and sources merge, other fields from b win, degraded lists concatenate."""
    out = dict(a)
    for k, v in b.items():
        if k in ('decode_tps', 'src') and isinstance(v, dict):
            out[k] = dict(out.get(k, {}), **v)
        elif k == 'degraded':
            out[k] = list(out.get(k, [])) + list(v)
        else:
            out[k] = v
    return out


def predict(recipe):
    """Post-hoc prediction, or pinned intake facts without planner admission."""
    with open(os.path.join(ROOT, 'data.json')) as f:
        data = json.load(f)
    key = recipe.get('model')
    if isinstance(key, str) and ID_RE.fullmatch(key) and key not in data['models'] \
            and recipe.get('box') in data['boxes'] and recipe.get('engine') in data['engines']:
        path = os.path.join(ROOT, 'data', 'models', key + '.json')
        if os.path.isfile(path):
            with open(path) as f:
                model = json.load(f)
            intake = model.get('intake') or {}
            if intake.get('source') and intake.get('bytes_by_class'):
                return {'prediction': {'status': 'UNMODELLED',
                        'why': 'pinned intake only; no planner model for ' + key,
                        'decode_tps': None, 'decode_tps_nodraft': None, 'prefill_tps': None,
                        'intake': intake}}
    p = subprocess.run(['node', os.path.join(ROOT, 'report.js'), '--check', '/dev/stdin', '--json'], input=json.dumps(recipe),
                       capture_output=True, text=True, timeout=300)
    if p.returncode == 2:
        raise SystemExit('report.js --check: bad input\n' + p.stdout + p.stderr)
    return json.loads(p.stdout)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--recipe')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--check', help='report.js --check R --json output captured at launch')
    g.add_argument('--no-check', action='store_true', help='no launch capture: prediction computed now, marked post-hoc')
    ap.add_argument('--measured', help='measured JSON (merged over --from-log results)')
    ap.add_argument('--from-log', action='append', default=[], help='KIND:PATH, repeatable (kinds: %s)' % ', '.join(PARSERS))
    ap.add_argument('--arm', help='kit arm name when a kit log holds several')
    ap.add_argument('--id')
    ap.add_argument('--when', help='UTC ISO start; default the kit arm start or now')
    ap.add_argument('--source', action='append', default=[])
    ap.add_argument('--tag', action='append', default=[])
    ap.add_argument('--runs', default=RUNS)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    meas = {}
    for spec in a.from_log:
        meas = merge(meas, from_log(spec, a.arm))
    if a.measured:
        with open(a.measured) as f:
            meas = merge(meas, json.load(f))
    if not a.recipe:
        print(json.dumps(meas, indent=1))
        return 0
    with open(a.recipe) as f:
        recipe = json.load(f)
    arm = meas.pop('_arm', None)
    if a.check:
        with open(a.check) as f:
            check = json.load(f)
        if not isinstance(check, dict) or check.get('id') != recipe.get('id'):
            raise SystemExit('%s is not the gate output of recipe %s (its id: %s)' % (a.check, recipe.get('id'), check.get('id') if isinstance(check, dict) else 'a list'))
        pred = check.get('prediction')
    elif a.no_check:
        check, pred = None, dict(predict(recipe).get('prediction') or {}, basis='post-hoc')
    else:
        raise SystemExit('give --check C.json (captured at launch) or --no-check')
    when = a.when or (arm and arm['start'][:16] + 'Z') or datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%MZ')
    row = {'id': a.id or recipe.get('id'), 'when': when, 'box': recipe['box'], 'model': recipe['model'], 'engine': recipe['engine'],
           'recipe': recipe, 'check': check, 'prediction': pred, 'measured': meas,
           'sources': a.source + [home_rel(s) for s in a.from_log], 'tags': a.tag}
    if a.dry_run:
        bad = validate(row)
        print(json.dumps(row, indent=1))
        if bad:
            print('INVALID:\n  ' + '\n  '.join(bad), file=sys.stderr)
        return 1 if bad else 0
    append(row, a.runs)
    print('appended %s to %s' % (row['id'], a.runs))
    return 0


if __name__ == '__main__':
    sys.exit(main())
