#!/usr/bin/env python3
"""Record llama-dllm-gen (block-diffusion) runs as run-ledger rows (runs/runs.jsonl), every number with its file.

    record_dllm.py --from odie=DIR [--from lassie=DIR] [--catalog tools/dllm_catalog.json] [--runs runs/runs.jsonl] [--dry-run]

DIR is a local copy of the host's run directory (catalog hosts.<host>.dir, e.g. `ssh -n odie 'cd ~/dllm/runs && tar cf - .'`).
Each timed config <prefix>-<cfg>.jsonl (one JSON line per prompt) with its line in <prefix>-conditions.jsonl becomes one row;
catalog extra_runs (no conditions line) and cli_runs (llama-diffusion-cli logs) are added from their catalog entries. A
stripped copy of every source file (per-prompt numbers, no token lists or text) goes to runs/dllm-raw/<host>/, so the
ledger's numbers stay checkable after the box's run directory is gone.

A row's recipe has kind "dllm": the model, schedule (decoder flags), block, kernel path (catalog stages / prefix_env),
cards and every conditions field. measured holds the decode tok/s (all prompts: generated tokens / decode time), tokens
per forward (TPF), ms per forward and tokens submitted per forward, each summed over the prompts. Rows are post hoc
(no launch capture): report.js --accuracy predicts them with today's planner.
"""
import argparse
import datetime
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import record_run  # noqa: E402

RAW = os.path.join(ROOT, 'runs', 'dllm-raw')
KEEP = ('id', 'mode', 'thr', 'edit_thr', 'block', 'device_argmax', 'lev', 'steps', 'max_post', 'n_delete', 'n_split', 'n_loop', 'n_prompt',
        'n_gen', 'n_fwd', 'n_commit', 'fuse_hit', 'fuse_miss', 'edits', 'n_vfwd', 'drafted', 'accepted', 'ms_per_vfwd', 'tpf', 'wall_ms',
        'prefill_ms', 'ms_per_fwd', 'host_ms_per_fwd', 'host_share', 'tok_s', 'tok_per_fwd_submitted', 'experts_per_fwd')
CLI_RE = re.compile(r'total time: ([\d.]+)ms, time per step: ([\d.]+)ms')


def when_of(path):
    return datetime.datetime.fromtimestamp(os.path.getmtime(path), datetime.timezone.utc).strftime('%Y-%m-%dT%H:%MZ')


def prompts_of(path):
    """per-prompt rows of one config file (lines that are JSON objects with n_gen)"""
    out = []
    for line in open(path, errors='replace'):
        line = line.strip()
        if line.startswith('{'):
            r = json.loads(line)
            if 'n_gen' in r:
                out.append(r)
    return out


def summary(P):
    """sums over the prompts: decode time = wall - prefill; TPF = generated / forwards; submitted = token-weighted mean"""
    g, fw = sum(r['n_gen'] for r in P), sum(r['n_fwd'] for r in P)
    dm = sum(r['wall_ms'] - r['prefill_ms'] for r in P)
    fms = sum(r['ms_per_fwd'] * r['n_fwd'] for r in P)
    return {'n_prompts': len(P), 'n_gen': g, 'n_fwd': fw, 'tpf': round(g / fw, 3), 'ms_per_fwd': round(fms / fw, 2),
            'decode_ms_per_fwd': round(dm / fw, 2), 'tokens_per_fwd_submitted': round(sum(r['tok_per_fwd_submitted'] * r['n_fwd'] for r in P) / fw, 2),
            'decode_tps': round(g / dm * 1000, 2), 'n_vfwd': sum(r.get('n_vfwd', 0) for r in P)}


def write_raw(host, name, rows):
    os.makedirs(os.path.join(RAW, host), exist_ok=True)
    with open(os.path.join(RAW, host, name), 'w') as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in KEEP if k in r}, separators=(',', ':')) + '\n')
    return 'runs/dllm-raw/%s/%s' % (host, name)


def schedule(cfg, P, args):
    """the decoder flags that set TPF (the schedule key drops the stop rule, which is a workload field)"""
    r = P[0] if P else {}
    mode = 'spec' if '--verifier' in args else r.get('mode', 'cli')
    return {'name': re.sub(r'-eos$', '', cfg), 'mode': mode, 'thr': r.get('thr'), 'edit_thr': r.get('edit_thr'), 'device_argmax': r.get('device_argmax'),
            'lev': r.get('lev'), 'block': r.get('block'), 'steps': r.get('steps'), 'max_post': r.get('max_post'), 'args': args.strip()}


def conditions(C, wl, ctx, extra):
    """every conditions field (build.py conditions.fields) plus the workload"""
    w = C['workloads'][wl]
    return dict({'ctx': ctx, 'kv_type': 'f16, block-causal block KV (llama_kv_cache_block)', 'draft_n': 0,
                 'sampling': {'temperature': 0, 'set_by': 'llama-dllm-gen greedy argmax (no sampler)'},
                 'reasoning_effort': C['reasoning_effort']['v'],
                 'workload': {'key': w['key'], 'prompt_kind': w['prompt_kind'], 'cached_depth': w['cached_depth'], 'gen_tokens': w['gen_tokens'], 'eos_stop': w['eos_stop']}}, **extra)


def row_for(C, host, tag, cond, P, raw_src, ctx, when, kind_extra=None):
    """one ledger row; cond = the conditions line (or the catalog entry for runs without one)"""
    m = C['models'][cond['gguf']]
    H = C['hosts'][host]
    pre = tag.split('-')[0]
    st = C['stages'].get(cond['stage'], {})
    pe = C['prefix_env'].get(pre, {})
    cards = [C['cards'][c] for c in cond.get('card', '').split('+') if c in C['cards']] if H['backend'] != 'cpu' else []
    placement = cond.get('placement') or ('n_cpu_moe:' + re.search(r'--n-cpu-moe (\d+)', cond['args']).group(1) if '--n-cpu-moe' in cond['args'] else 'all_gpu')
    path = None if H['backend'] == 'cpu' or m['decoder'] == 'dream' else (pe.get('moe_path') or st.get('moe_path'))
    wl = C['workloads'][cond['prompts']]['key']
    sc = schedule(cond['cfg'], P, cond['args'])
    extra = {'env': pe.get('env', 'none'), 'cards': cards, 'placement': placement}
    if cond.get('threads'):
        extra.update(threads=cond['threads'], cpu_reserve='0 (no reserve on the dLLM lane)')
    if cond.get('corunner'):
        extra['corunner'] = cond['corunner']
    rid = tag if tag.startswith(host + '-') else '%s-dllm-%s' % (host, tag)
    recipe = {'kind': 'dllm', 'id': rid, 'label': '%s %s %s on %s (%s, %s)' % (m['model'], sc['name'], m['quant'], host, '+'.join(cards) or 'CPU', path or H['backend']),
              'box': H['box'], 'model': m['model'], 'engine': 'llama-dllm-gen' if not sc['name'].startswith('forkcli') else 'llama-diffusion-cli',
              'engine_build': cond['stage'], 'backend': H['backend'], 'moe_path': path, 'cards': cards, 'quant': m['quant'], 'decoder': m['decoder'],
              'schedule': sc, 'block': sc['block'] or 32, 'workload': wl, 'placement': placement, 'ctx': ctx,
              'conditions': conditions(C, cond['prompts'], ctx, extra), 'src': cond.get('src') or C['hosts'][host]['src']}
    if kind_extra:
        recipe.update(kind_extra)
    S = summary(P) if P else None
    src_line = '%s:%s/%s = %s' % (host, H['dir'], raw_src.split('/')[-1], raw_src)
    meas = {'outcome': 'ok', 'src': {}}
    if S:
        meas.update({'decode_tps': {'all': S['decode_tps']}, 'tpf': S['tpf'], 'ms_per_fwd': S['ms_per_fwd'], 'decode_ms_per_fwd': S['decode_ms_per_fwd'],
                     'tokens_per_fwd_submitted': S['tokens_per_fwd_submitted'], 'n_gen': S['n_gen'], 'n_fwd': S['n_fwd'], 'n_prompts': S['n_prompts']})
        for k in ('decode_tps', 'tpf', 'ms_per_fwd', 'decode_ms_per_fwd', 'tokens_per_fwd_submitted', 'n_gen', 'n_fwd', 'n_prompts'):
            meas['src'][k] = src_line + ' (summed over the prompts by tools/record_dllm.py summary())'
        if S['n_vfwd']:
            meas['n_vfwd'] = S['n_vfwd']
            meas['src']['n_vfwd'] = src_line
    if cond.get('vram_mib'):
        meas['vram_mib'] = cond['vram_mib']
        meas['src']['vram_mib'] = '%s:%s/%s-conditions.jsonl (SYCL0 buffer sizes in the run log)' % (host, H['dir'], tag[:-len(cond['cfg']) - 1])
    rc = cond.get('rc', 0)
    if rc != 0 or not P:
        meas['outcome'] = 'crash'
        meas['notes'] = 'rc %s, %d prompt rows (124 = timeout, 134 = abort, 137 = killed)' % (rc, len(P))
        meas['not_measured'] = ['decode_tps']
    if cond.get('no_conditions_line'):
        meas['outcome'] = 'stopped'
        meas['notes'] = 'no line in the conditions file: the window cut the script before it wrote one, so rc and VRAM are unknown'
    if any(r.get('experts_per_fwd') for r in P):
        meas['degraded'] = ['--expert-stats on: the eval callback reads every ffn_moe_topk tensor, which splits the graph per MoE layer (timing is not production speed)']
    return {'id': rid, 'when': when, 'box': H['box'], 'model': m['model'], 'engine': recipe['engine'], 'recipe': recipe, 'check': None, 'prediction': None,
            'measured': meas, 'sources': [src_line], 'tags': ['post-hoc', 'dllm']}


def ctx_of(log):
    if not os.path.exists(log):
        return None
    for line in open(log, errors='replace'):
        x = re.match(r'llama_context: n_ctx\s+=\s+(\d+)', line)
        if x:
            return int(x.group(1))
    return None


def host_rows(C, host, d):
    rows = []
    conds = {}
    for f in sorted(os.listdir(d)):
        if f.endswith('-conditions.jsonl'):
            for line in open(os.path.join(d, f)):
                if line.strip():
                    c = json.loads(line)
                    conds[c['tag']] = c   # a re-run of a config keeps its last line
            os.makedirs(os.path.join(RAW, host), exist_ok=True)
            with open(os.path.join(RAW, host, f), 'w') as o:
                o.write(open(os.path.join(d, f)).read())
    seen = set()
    for tag, c in sorted(conds.items()):
        if c['cfg'] in C['skip_cfgs']['v'] or c['gguf'] not in C['models']:
            continue
        p = os.path.join(d, tag + '.jsonl')
        P = prompts_of(p) if os.path.exists(p) else []
        raw = write_raw(host, tag + '.jsonl', P)
        rows.append(row_for(C, host, tag, c, P, raw, ctx_of(os.path.join(d, tag + '.log')), when_of(p) if os.path.exists(p) else when_of(os.path.join(d, tag + '.log'))))
        seen.add(tag)
    # a config file with prompt rows but no conditions line: recorded as stopped, from its prefix's other lines
    for f in sorted(os.listdir(d)):
        tag = f[:-6]
        if not f.endswith('.jsonl') or f.endswith('-conditions.jsonl') or tag in seen:
            continue
        pre = {t[:-len(c['cfg']) - 1]: c for t, c in conds.items()}
        p = next((x for x in sorted(pre, key=len, reverse=True) if tag.startswith(x + '-')), None)
        cfg = tag[len(p) + 1:] if p else None
        if not cfg or cfg in C['skip_cfgs']['v']:
            continue
        sib = pre[p]
        P = prompts_of(os.path.join(d, f))
        if not P:
            continue
        c = dict(sib, tag=tag, cfg=cfg, args='(not recorded: no conditions line)', rc=None, vram_mib=None, no_conditions_line=True)
        rows.append(row_for(C, host, tag, c, P, write_raw(host, f, P), ctx_of(os.path.join(d, tag + '.log')), when_of(os.path.join(d, f))))
    return rows


def extra_rows(C, dirs):
    rows = []
    for e in C['extra_runs']:
        d = dirs.get(e['host'])
        if not d:
            continue
        p = os.path.join(d, e['file'])
        P = prompts_of(p)
        rows.append(row_for(C, e['host'], e['id'], dict(e, rc=0), P, write_raw(e['host'], e['file'], P), e['ctx'], when_of(p)))
    for e in C['cli_runs']:
        d = dirs.get(e['host'])
        if not d:
            continue
        p = os.path.join(d, e['log'])
        lines = open(p, errors='replace').read().split('\n')
        i, x = next((i, CLI_RE.search(s)) for i, s in enumerate(lines) if CLI_RE.search(s))
        total, step = float(x.group(1)), float(x.group(2))
        name = e['log'][:-4] + '.timing.txt'
        os.makedirs(os.path.join(RAW, e['host']), exist_ok=True)
        with open(os.path.join(RAW, e['host'], name), 'w') as o:
            o.write('%s line %d: %s\n' % (e['log'], i + 1, lines[i]))
        H = C['hosts'][e['host']]
        src_line = '%s:%s/%s#L%d = runs/dllm-raw/%s/%s' % (e['host'], H['dir'], e['log'], i + 1, e['host'], name)
        m = C['models'][e['gguf']]
        cond = {'gguf': e['gguf'], 'stage': e['stage'], 'card': e['card'], 'cfg': e['cfg'], 'args': e['args'], 'prompts': 'fixed.jsonl', 'src': e['src']}
        row = row_for(C, e['host'], e['id'], cond, [], 'runs/dllm-raw/%s/%s' % (e['host'], name), ctx_of(p), when_of(p))
        row['sources'] = [src_line]
        steps = e['gen_tokens'] / e['tpf']
        row['recipe']['schedule'].update({'mode': 'cli-full', 'steps': int(steps), 'block': 32 if m['decoder'] != 'dream' else None})
        row['recipe']['conditions']['workload'] = {'key': 'cli1x256', 'prompt_kind': e['prompt_kind'], 'cached_depth': 0, 'gen_tokens': e['gen_tokens'], 'eos_stop': False}
        row['recipe']['workload'] = 'cli1x256'
        row['recipe']['conditions'].update(kv_type='none reused: the whole sequence is resubmitted every step', sampling={'temperature': 0, 'set_by': 'llama-diffusion-cli defaults (greedy confidence transfer)'})
        row['measured'] = {'outcome': 'ok', 'decode_tps': {'all': round(e['gen_tokens'] / total * 1000, 2)}, 'tpf': e['tpf'], 'ms_per_fwd': step,
                           'src': {'decode_tps': src_line + ' (gen tokens / total time)', 'tpf': 'catalog cli_runs: gen tokens / steps (%s)' % e['src'], 'ms_per_fwd': src_line + ' (time per step)'},
                           'notes': 'llama-diffusion-cli resubmits the whole sequence every step (no KV reuse); the total time holds the steps only'}
        rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--from', dest='src', action='append', default=[], help='HOST=DIR, a local copy of the host run directory')
    ap.add_argument('--catalog', default=os.path.join(HERE, 'dllm_catalog.json'))
    ap.add_argument('--runs', default=record_run.RUNS)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    C = json.load(open(a.catalog))
    dirs = dict(s.split('=', 1) for s in a.src)
    rows = sum((host_rows(C, h, d) for h, d in dirs.items() if h in C['hosts'] and any(f.endswith('-conditions.jsonl') for f in os.listdir(d))), []) + extra_rows(C, dirs)
    have = {r['id'] for r in record_run.load(a.runs)} if os.path.exists(a.runs) else set()
    bad = {r['id']: record_run.validate(r) for r in rows if record_run.validate(r)}
    if bad:
        raise SystemExit('invalid rows:\n' + '\n'.join('%s: %s' % kv for kv in bad.items()))
    new = [r for r in rows if r['id'] not in have]
    if a.dry_run:
        print(json.dumps([r['id'] for r in new], indent=1))
        return 0
    for r in new:
        record_run.append(r, a.runs)
    print('%d dllm rows (%d new, %d already in %s)' % (len(rows), len(new), len(rows) - len(new), a.runs))
    return 0


if __name__ == '__main__':
    sys.exit(main())
