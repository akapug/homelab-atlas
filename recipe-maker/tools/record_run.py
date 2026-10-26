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
  placement   engine log: placement/dispatch receipts (fallback_ledger)
  replay      agent-replay.py result json (<arm>-replay-n<N>.json): real coding-agent traffic, every field keyed c<agents>.
              Its warmup_s and fresh_breakdown (cold / growth / re-read) go into measured.workload when the result has them.
              Inside git the source names the repo-relative path and the last commit of the file (a file that differs from
              its last commit is refused: commit it first); the start time comes from the "HH:MM:SS arm <arm>:" line of the
              run.log beside it, placed at or before the file's commit time

One command records a replay arm: the recipe is cloned from an existing ledger row and the arm's env, image, context and
batch come from the arms file (JSON: common_env, arms[] with name, env, image, ctx, max_tokens, engine_args_append):
  record_run.py --recipe-from BASE_ROW_ID --arms ARMS.json --no-check --from-log replay:RESULTS/<arm>-replay-n16.json --tag T
"""
import argparse
import datetime
import json
import math
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
    if row['recipe'].get('kind') == 'network':
        identity = row['recipe'].get('network_prediction_identity')
        if not isinstance(identity, dict) or not identity.get('recipe') or identity.get('recipe') != row['recipe'].get('network_recipe'):
            bad.append('network recipe: exact priced network_prediction_identity required')
        if m.get('network_identity') != identity or not src.get('network_identity'):
            bad.append('measured.network_identity must equal the priced identity and cite its capture source')
        prediction = row.get('prediction') or {}
        if prediction.get('network', {}).get('prediction_identity') != identity:
            bad.append('network prediction must retain the exact priced identity')
    validation = row['recipe'].get('validation')
    if validation is not None:
        prediction = row.get('prediction') if isinstance(row.get('prediction'), dict) else {}
        check = row.get('check') if isinstance(row.get('check'), dict) else {}
        capture = prediction.get('validation') if isinstance(prediction.get('validation'), dict) else {}
        cohort = capture.get('cohort') if isinstance(capture.get('cohort'), dict) else {}
        members = cohort.get('candidates') if isinstance(cohort.get('candidates'), list) else []
        member = next((x for x in members if isinstance(x, dict) and x.get('id') == capture.get('candidate')), {})
        bare = {k: v for k, v in row['recipe'].items() if k != 'validation'}
        if (not isinstance(validation, dict) or prediction.get('basis') != 'launch' or not check
                or not isinstance(check.get('prediction'), dict) or check['prediction'].get('validation') != capture
                or cohort != validation.get('cohort') or capture.get('candidate') != validation.get('candidate')
                or member.get('run_id') != row['id'] or member.get('recipe') != bare
                or cohort.get('frozen') != prediction.get('calibration')):
            bad.append('validation: exact declared candidate and frozen launch capture required; no post-hoc cohort')
        observed = m.get('validation')
        if observed is not None and (not isinstance(observed, dict) or observed.get('cohort_sha256') != capture.get('cohort_sha256')
                                     or observed.get('candidate') != capture.get('candidate') or observed.get('workload') != cohort.get('workload')
                                     or observed.get('isolation') != cohort.get('isolation') or not src.get('validation')
                                     or ((capture.get('objective') or {}).get('mode') == 'batch-fleet'
                                         and observed.get('power_scope') != capture['objective'].get('power_scope'))):
            bad.append('measured.validation must match the frozen cohort, workload and isolation with a capture source')
    ledger = m.get('fallback_ledger')
    if ledger is not None:
        if not isinstance(ledger, dict) or not isinstance(ledger.get('events'), list) or not ledger.get('src'):
            bad.append('measured.fallback_ledger: sourced event list required')
        else:
            identity = ledger.get('identity')
            if identity is not None and (not isinstance(identity, dict) or any(not row['recipe'].get(k) or identity.get(k) != row['recipe'].get(k)
                                       for k in ('box', 'model', 'engine', 'engine_build'))):
                bad.append('fallback_ledger.identity must match exact recipe box/model/engine/build')
            for event in ledger['events']:
                if not isinstance(event, dict) or not event.get('class') or not event.get('src') or event.get('proof') not in ('observed', 'configured'):
                    bad.append('fallback_ledger event needs class, source and observed/configured proof')
                elif 'value' in event and (not _num(event['value']) or not math.isfinite(event['value']) or event['value'] < 0):
                    bad.append('fallback_ledger event value must be finite and nonnegative')
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


KIT_WRAP = datetime.timedelta(hours=12)   # a stamp this far before the last one read is past midnight, not a late log line


def _ts(after, hms):
    """kit timestamps carry the time only: the instant with that time of day nearest after `after` (the kit start or the last
    stamp read). A stamp more than KIT_WRAP earlier in the day is past midnight and lands on the next date (v41fs-ts1 10-21:
    start 23:34:34Z, rc 00:26:34Z read as -1388 min before); a watch line a few seconds behind stays on its date."""
    t = datetime.datetime.strptime(after.strftime('%Y-%m-%d') + 'T' + hms.rstrip('Z'), '%Y-%m-%dT%H:%M:%S').replace(tzinfo=datetime.timezone.utc)
    return t + datetime.timedelta(days=1) if after - t > KIT_WRAP else t


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
            kit['clock'] = kit['start']   # the last stamp read: _ts places each time-of-day stamp after it
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
            kit['clock'] = max(kit['clock'], _ts(kit['clock'], r.group(3)))
            cur = {'kit': kit, 'arm': r.group(1), 'rc': int(r.group(2)), 'end': kit['clock'],
                   'start': prev_end, 'json': js[0] if js else {}, 'pre': pre, 'post': []}
            arms.append(cur)
            pre = []
            continue
        low = re.match(r'^== N0 LOW (\d\d:\d\d:\d\dZ): node0 MemFree (\d+(?:\.\d+)?) GiB', ln)
        if low:
            t = _ts(kit['clock'], low.group(1))
            kit['clock'] = max(kit['clock'], t)
            kit['lows'].append((t, float(low.group(2))))
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


def parse_placement(text, src):
    """Engine placement/dispatch receipts, never an inference from flags or a missing log line.

    XeStrata 67fcb0a: generate.cpp 5344, 6829, 7118, 11963, 11977, 12018;
    verify.cpp535 reports a SETUP threshold, not evidence that a row executed there.
    Cache counters exclude PCIe offload; KV counters are cumulative, not per request.
    Identity stays unbound until attached to a captured launch by the existing writer.
    """
    events = []
    for number, line in enumerate(text.splitlines(), 1):
        cite = '%s:%d' % (src, number)
        def event(cls, proof='observed', **fields):
            events.append(dict({'class': cls, 'proof': proof, 'src': cite, 'line': line}, **fields))
        m = re.search(r'decode expert cache hit rate: ([\d.]+)% \((\d+) hits / (\d+) lookups\)', line)
        if m:
            event('hot_experts', metric='cache_hit_pct', value=float(m[1]), scope='decode lookups; excludes PCIe/remote offload')
        m = re.search(r'expert cache:? (\d+) slots', line)
        if m:
            event('hot_experts', metric='cache_slots', value=int(m[1]), scope='one cache allocation; not summed across GPUs')
        # Runtime reserve changes and elastic K/V growth can shrink a cache after the
        # allocation line. Keep every observed size so an earlier high-water mark cannot
        # conceal a later loss; these are per-cache counts, never totals across GPUs.
        for pattern, scope in (
            (r'expert cache \d+ -> (\d+) of \d+ slots', 'one cache after runtime VRAM reserve; not summed across GPUs'),
            (r'K/V grown to .*\((\d+) of \d+ slots hold experts\)', 'one cache after K/V growth; not summed across GPUs'),
        ):
            m = re.search(pattern, line)
            if m:
                event('hot_experts', metric='cache_slots', value=int(m[1]), scope=scope)
        m = re.search(r'KV streaming: ([\d.]+)% of (\d+) block reads hit VRAM, ([\d.]+) MiB read from RAM', line)
        if m:
            event('kv', metric='kv_ram_mib', value=float(m[3]), scope='cumulative process counters, not request joins')
            if float(m[3]) > 0:
                event('kv', tier='dram', scope='some KV block reads; not complete placement')
            if float(m[1]) > 0 and int(m[2]) > 0:
                event('kv', tier='gpu', scope='some KV block reads; not complete placement')
        m = re.search(r'(?:prompt path borrows|borrows) (\d+) (?:cache|expert-cache) slots', line)
        if m:
            event('hot_experts', metric='borrowed_slots', value=int(m[1]), scope='read loan; restoration not proved')
        if re.search(r'WARNING: --batch-mtp is off:', line):
            event('batch_mtp', tier='off')
        if re.search(r'strata serve: --batch-mtp: \d+ slots, each verifying one MTP proposal', line):
            event('batch_mtp', proof='configured', tier='on')
        m = re.search(r'drafts accepted (\d+) of (\d+)', line)
        if m and int(m[2]) > 0:
            event('drafter_state', tier='on', scope='proposals offered, not necessarily accepted')
        m = re.search(r'strata batch: MTP proposals accepted (\d+) of (\d+)', line)
        if m and int(m[2]) > 0:
            event('batch_mtp', tier='on', scope='proposals actually offered')
        m = re.search(r'strata batch: (\d+) windows, avg ([\d.]+) rows', line)
        if m and int(m[1]) > 0 and float(m[2]) > 1:
            event('execution', tier='batched', scope='executed windows with more than one row on average')
        m = re.search(r'batch dense products from (\d+) rows on (.+)', line)
        if m:
            path = m[2].lower()
            tier = 'matrix' if 'xmx' in path or 'dpas' in path else 'vector' if 'gemv' in path or 'mmvq' in path else None
            if tier:
                event('dense_op', proof='configured', tier=tier, min_rows=int(m[1]), scope='setup threshold only, execution unproved')
        if re.search(r'(?:DSpark|dspark).*(?:not loaded|disabled)', line):
            event('drafter_state', tier='off')
        # Explicit runtime memory reports only. MemAvailable is neither GTT size nor allocatable headroom.
        m = re.search(r'(?:GPUActive|GTT)[ :=]+([\d.]+) GiB', line)
        if m:
            event('host_pins', metric='host_gtt_gib', value=float(m[1]), scope='driver pinned allocation outside memcg')
    return {'fallback_ledger': {'src': src, 'events': events}, 'src': {'fallback_ledger': src}}


# ---------------- agent-replay.py result json ----------------
def workload_settings():
    """data/workload.json settings (real_kind, build_key_re): the replay writer and the build read the same values"""
    with open(os.path.join(ROOT, 'data', 'workload.json')) as f:
        return {k: x['v'] for k, x in json.load(f)['settings'].items()}


def fresh_breakdown(b):
    """a replay's fresh breakdown as {cause: tokens}: a dict as given, a per-turn (or per-agent) list summed by key; only numbers count"""
    if isinstance(b, dict):
        return b
    out = {}
    for x in b if isinstance(b, list) else []:
        for k, n in (x.items() if isinstance(x, dict) else []):
            if _num(n):
                out[k] = out.get(k, 0) + n
    return out


def parse_replay(text, src, stats=None):
    """One agent-replay.py run: totals over the window keyed c<agents>. stats = the trace's stats from agent-replay-traces.json
    (its recorded turn count goes into the trace description when given)."""
    d = json.loads(text)
    c, sd = 'c%d' % d['agents'], d['server_delta']
    fresh, out = d['fresh_prompt_tokens'], d['out_tokens']
    turns = ', %d turns' % stats['turns'] if stats and stats.get('turns') else ''
    m = {'agent_tps': {c: d['agg_out_tok_s']},
         'prefill_busy_pct': {c: d['prefill_busy_pct']},
         'stall_pct': {c: d['stall_pct']},
         'ttft_s': {c + '_median': d['ttft_s'].get('median'), c + '_p90': d['ttft_s'].get('p90')},
         'decode_tps': {c + '_per_request_median': d['decode_tok_s'].get('median')},
         'turns_per_agent_min': {c: d['turns_per_agent_min']},
         'power_w': {c: round(sum(d['card_w']), 1)},
         'pinned_gib': d['max_gpuactive_gib'],
         'errors': d['errors'],
         'workload_kind': workload_settings()['real_kind'],
         'workload': {'trace': '%s agent traffic, trace %s (%d recorded agents%s), %s at %d agents' % (
                          d['trace_source'].lower(), d['trace'], d['recorded_agents'], turns, 'looped' if d['loop'] else 'played once', d['agents']),
                      'fresh_prompt_tokens': fresh, 'reused_prompt_tokens': sd['reused'], 'out_tokens': out,
                      'prompt_ms': round(sd['prompt_ms']), 'requests': sd['requests'], 'cache_evictions': sd['cc_evictions']},
         'primary_phase': 'agent',
         'outcome': 'ok' if d['turns_completed'] > 0 and d['agg_out_tok_s'] > 0 else 'crash',
         'notes': '%s; agent_tps = output tokens over the %g s window (all agents); prefill_tps = fresh prompt tokens / server prompt_ms; '
                  'fresh_per_output = fresh prompt tokens / output tokens; power = the %d cards only; %d batch slots' % (
                      d['qualification'], d['window_s'], len(d['card_w']), d['batch_slots'])}
    # the ratios exist only for a run that read and wrote something: a run that served nothing is still recorded (a crash)
    ratios = {'prefill_tps': (fresh, sd['prompt_ms'] / 1000, 1), 'fresh_per_output': (fresh, out, 2)}
    m.update({k: {c: round(a / b, n)} for k, (a, b, n) in ratios.items() if b})
    for k in ('ttft_s', 'decode_tps'):   # a run with no finished turn has no medians: unmeasured, never a null number
        m[k] = {x: v for x, v in m[k].items() if v is not None}
        if not m[k]:
            del m[k]
    # optional replay fields (older results have neither): the warm-up before the timed window (seconds, or the setting
    # warmup_none) and the fresh prompt tokens split by cause; checks steady_state and replay_artifact read them
    if d.get('warmup_s') is not None:
        m['workload']['warmup_s'] = d['warmup_s']
    if d.get('fresh_breakdown'):
        m['workload']['fresh_breakdown'] = fresh_breakdown(d['fresh_breakdown'])
    if d['errors']:
        m['degraded'] = ['%d request errors (err_sample: %s)' % (d['errors'], '; '.join(map(str, d.get('err_sample') or []))[:160])]
    m['src'] = {k: src for k in m if k not in ('workload_kind', 'primary_phase', 'notes', 'outcome')}
    return m


def git_ref(path):
    """(repo-relative path, '<repo> <commit>') of the last commit of a file that matches it on disk; None outside git"""
    p = os.path.realpath(os.path.expanduser(path))
    top = subprocess.run(['git', '-C', os.path.dirname(p), 'rev-parse', '--show-toplevel'], capture_output=True, text=True)
    if top.returncode:
        return None
    rel = os.path.relpath(p, top.stdout.strip())
    git = lambda *a: subprocess.run(['git', '-C', top.stdout.strip()] + list(a), capture_output=True, text=True)  # noqa: E731
    if git('status', '--porcelain', '--', rel).stdout.strip() or not git('ls-files', '--', rel).stdout.strip():
        raise SystemExit('%s differs from its last commit (or is untracked): commit it first, so the source names a commit' % rel)
    common = git('rev-parse', '--path-format=absolute', '--git-common-dir').stdout.strip()
    repo = os.path.basename(os.path.dirname(common)) if os.path.basename(common) == '.git' else os.path.basename(common)[:-len('.git')] or os.path.basename(common)
    log = git('log', '-1', '--format=%h %cI', '--abbrev=11', '--', rel).stdout.split()
    return {'rel': rel, 'top': top.stdout.strip(), 'repo': repo, 'commit': log[0],
            'at': datetime.datetime.fromisoformat(log[1]).astimezone(datetime.timezone.utc)}


def replay_start(path, label, before):
    """the arm's start: the first "HH:MM:SS arm <label>:" line of its last run in the run.log beside the result, on the date that puts it at or
    before `before` (the result's commit or mtime); None without that line"""
    log = os.path.join(os.path.dirname(os.path.abspath(os.path.expanduser(path))), 'run.log')
    if not os.path.exists(log):
        return None
    with open(log, errors='replace') as f:
        lines = re.findall(r'^(\d\d:\d\d:\d\d) arm (\S+): ', f.read(), re.M)
    # an arm's first line after another arm's (or the file start) opens a run of it; the last such line is its start
    hits = [t for i, (t, a) in enumerate(lines) if a == label and (i == 0 or lines[i - 1][1] != label)]
    if not hits:
        return None
    t = datetime.datetime.strptime(before.strftime('%Y-%m-%d') + 'T' + hits[-1], '%Y-%m-%dT%H:%M:%S').replace(tzinfo=datetime.timezone.utc)
    return t - datetime.timedelta(days=1) if t > before else t


def from_replay(path):
    """measured fields of one replay result plus _replay: the arm, agents, source strings and start time"""
    text = read_text(path)
    d = json.loads(text)
    local = os.path.exists(os.path.expanduser(path))   # host:path results are read over ssh: no commit, no run.log
    ref = git_ref(path) if local else None
    src = '%s (%s %s)' % (ref['rel'], ref['repo'], ref['commit']) if ref else home_rel(path)
    found = None
    if ref:   # the trace pointer file agent-replay.py reads: the nearest agent-replay-traces.json up to the repo top
        here = os.path.dirname(os.path.realpath(os.path.expanduser(path)))
        while not found and here.startswith(ref['top']):
            cand = os.path.join(here, 'agent-replay-traces.json')
            found = cand if os.path.exists(cand) else None
            here = os.path.dirname(here)
    stats, traces_src = None, None
    if found:
        with open(found) as f:
            stats = (json.load(f).get('traces', {}).get(d['trace']) or {}).get('stats')
        tref = git_ref(found) if stats else None
        traces_src = '%s (%s %s) trace %s stats' % (tref['rel'], tref['repo'], tref['commit'], d['trace']) if tref else None
    m = parse_replay(text, src, stats)
    before = ref['at'] if ref else datetime.datetime.fromtimestamp(os.path.getmtime(os.path.expanduser(path)), datetime.timezone.utc) if local else None
    start = replay_start(path, d['label'], before) if before else None
    m['_replay'] = {'arm': d['label'], 'agents': d['agents'], 'window_s': d['window_s'], 'src': src, 'traces_src': traces_src,
                    'when': start.strftime('%Y-%m-%dT%H:%MZ') if start else None}
    return m


def recipe_from(rows, base_id, arms, replay, engine_build=None, new_id=None):
    """a replay arm's recipe: a copy of ledger row base_id's recipe with the arm's id, label, build, env, context and batch.
    arms = the parsed arms file; replay = from_replay()['_replay']. The build string and label come from the newest ledger row of
    the arm's image (settings.build_key_re) when the base is another build; engine_build overrides."""
    base = next((r for r in rows if r['id'] == base_id), None)
    if not base:
        raise SystemExit('no ledger row %s to clone' % base_id)
    name, agents = replay['arm'], replay['agents']
    arm = next((a for a in arms['arms'] if a['name'] == name), None)
    if not arm:
        raise SystemExit('arm %s is not in the arms file (%s)' % (name, ', '.join(a['name'] for a in arms['arms'])))
    rc = json.loads(json.dumps(base['recipe']))
    key_re = re.compile(workload_settings()['build_key_re'])
    key = lambda s: (key_re.search(str(s or '')) or [None, None])[1]  # noqa: E731
    img = key('image ' + arm['image'].split(':')[-1])
    tmpl = rc
    if engine_build:
        rc['engine_build'] = engine_build
    elif key(rc.get('engine_build')) != img:
        same = [r for r in rows if r['box'] == rc['box'] and r['model'] == rc['model'] and key(r['recipe'].get('engine_build')) == img]
        if not same:
            raise SystemExit('no ledger row names image %s: give --engine-build' % img)
        tmpl = max(same, key=lambda r: r['when'])['recipe']
        rc['engine_build'] = tmpl['engine_build']
    base_arm = re.search(r'\(([^()]*)\)$', base['recipe']['label']).group(1)
    rc['label'] = re.sub(r'at \d+ agents', 'at %d agents' % agents, re.sub(r'\([^()]*\)$', '(%s)' % name, tmpl['label']))
    rc['id'] = new_id or re.sub(r'-n\d+-', '-n%d-' % agents, base['id'].replace('-%s-' % base_arm, '-%s-' % name))
    if not new_id and '-%s-' % base_arm not in base['id']:
        raise SystemExit('base id %s does not name its arm %s: give --id' % (base['id'], base_arm))
    env = dict(arms.get('common_env') or {}, **arm.get('env') or {})
    extra = dict(zip(arm.get('engine_args_append', [])[::2], arm.get('engine_args_append', [])[1::2]))
    cond, serving = rc['conditions'], rc.setdefault('serving', {})
    flags = re.sub(r'arm env [^;]*', 'arm env ' + ' '.join('%s=%s' % kv for kv in sorted(env.items())), cond['engine_flags'])
    flags = re.sub(r'max_tokens \d+', 'max_tokens %d' % arm['max_tokens'], flags)
    flags = re.sub(r'agent replay \d+ agents', 'agent replay %d agents' % agents, flags)
    flags = re.sub(r'MAX_CONTEXT=\d+', 'MAX_CONTEXT=%d' % arm['ctx'], flags)
    # the arm's own engine args win: replace the flag where the base names it, else add it to the engine-args part (before
    # the first ';'); --prefill N is the prefill chunk (data/workload.json prefill_pass.settings.chunk_re reads it)
    for flag, field in (('--batch', 'batch'), ('--kv-resident', 'kv_resident'), ('--prefill', 'prefill')):
        if flag in extra:
            pat = r'%s (?=\S)[^\s;]+' % re.escape(flag)
            new = '%s %s' % (flag, extra[flag])
            flags = re.sub(pat, new, flags) if re.search(pat, flags) else re.sub(r'^([^;]*?)\s*(?=;|$)', lambda m: (m.group(1) + ' ' + new).strip(), flags, count=1)
            serving[field] = int(extra[flag]) if str(extra[flag]).isdigit() else extra[flag]
    cond['engine_flags'] = flags
    rc['ctx'] = cond['ctx'] = serving['max_context'] = arm['ctx']
    wl = cond.setdefault('workload', {})
    wl['agents'] = agents
    if wl.get('bench'):
        wl['bench'] = re.sub(r'\b\d+ s$', '%g s' % replay['window_s'], re.sub(r'\b\d+ agents', '%d agents' % agents, wl['bench'], count=1))
    if replay.get('traces_src'):
        wl['trace_stats_src'] = replay['traces_src']
    old = base['measured']['src'].get('agent_tps')
    rc['src'] = rc['src'].replace(old, replay['src']) if old else rc['src']
    rc['src'] = re.sub(r'\barm %s\b' % re.escape(base_arm), 'arm ' + name, rc['src'])
    return rc, base


PARSERS = {'kit': parse_kit, 'bench-log': parse_bench_log, 'bench-json': parse_bench_json, 'kl': parse_kl,
           'placement': parse_placement, 'replay': parse_replay}


def home_rel(text):
    """a path under this user's home as ~/... (ledger rows name no user)"""
    return text.replace(os.path.expanduser('~') + '/', '~/')


def from_log(spec, arm=None):
    kind, path = spec.split(':', 1)
    if kind not in PARSERS:
        raise SystemExit('unknown log kind %s (kinds: %s)' % (kind, ', '.join(PARSERS)))
    if kind == 'replay':
        return from_replay(path)
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
        if k == 'fallback_ledger' and k in out:
            previous = out[k]
            if previous.get('identity') != v.get('identity'):
                raise ValueError('cannot merge fallback receipts of different identities')
            out[k] = dict(v, events=previous['events'] + v['events'], src=previous['src'] + '; ' + v['src'])
        elif k in ('decode_tps', 'src') and isinstance(v, dict):
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
    ap.add_argument('--recipe-from', metavar='ROW_ID', help='clone this ledger row\'s recipe for a replay arm (needs --arms and --from-log replay:PATH)')
    ap.add_argument('--arms', help='the arms file of the replay: common_env and arms[] (name, env, image, ctx, max_tokens, engine_args_append)')
    ap.add_argument('--engine-build', help='with --recipe-from: the build string when no ledger row names the arm\'s image')
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
    replay = meas.pop('_replay', None)
    if not a.recipe and not a.recipe_from:
        print(json.dumps(meas, indent=1))
        return 0
    base = None
    if a.recipe_from:
        if not (a.arms and replay):
            raise SystemExit('--recipe-from needs --arms ARMS.json and --from-log replay:PATH')
        recipe, base = recipe_from(load(a.runs), a.recipe_from, json.loads(read_text(a.arms)), replay, a.engine_build, a.id)
    else:
        with open(a.recipe) as f:
            recipe = json.load(f)
    arm = meas.pop('_arm', None)
    if a.check:
        with open(a.check) as f:
            check = json.load(f)
        if not isinstance(check, dict) or check.get('id') != recipe.get('id'):
            raise SystemExit('%s is not the gate output of recipe %s (its id: %s)' % (a.check, recipe.get('id'), check.get('id') if isinstance(check, dict) else 'a list'))
        pred = check.get('prediction')
    elif a.no_check and base and (base.get('prediction') or {}).get('status') == 'UNMODELLED':
        check, pred = None, dict(base['prediction'], basis='post-hoc')   # the base build's engine has no planner model: nothing to predict
    elif a.no_check:
        check, pred = None, dict(predict(recipe).get('prediction') or {}, basis='post-hoc')
    else:
        raise SystemExit('give --check C.json (captured at launch) or --no-check')
    if a.check and meas.get('fallback_ledger') is not None:
        # A reused recipe id is not a launch identity. Bind only what the actual gate captured;
        # old captures without all identity fields leave the receipt unbound (UNKNOWN).
        identity = {k: check.get(k) for k in ('box', 'model', 'engine', 'engine_build')}
        if any(value is not None and value != recipe.get(k) for k, value in identity.items()):
            raise SystemExit('fallback receipt launch identity does not match recipe box/model/engine/build')
        if not meas['fallback_ledger'].get('identity') and all(isinstance(value, str) and value for value in identity.values()):
            meas['fallback_ledger']['identity'] = identity
            meas['fallback_ledger']['identity_src'] = home_rel(a.check)
    when = a.when or (arm and arm['start'][:16] + 'Z') or (replay and replay['when']) or datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%MZ')
    row = {'id': a.id or recipe.get('id'), 'when': when, 'box': recipe['box'], 'model': recipe['model'], 'engine': recipe['engine'],
           'recipe': recipe, 'check': check, 'prediction': pred, 'measured': meas,
           'sources': a.source + [x for s in a.from_log for x in (
               [replay['src']] + ([replay['traces_src']] if replay['traces_src'] else []) if s.startswith('replay:') and replay else [home_rel(s)])],
           'tags': a.tag}
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
