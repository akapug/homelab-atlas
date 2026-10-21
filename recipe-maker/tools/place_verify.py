#!/usr/bin/env python3
"""place_verify: did the per-expert NUMA placement land where the plan put it? (kit check, run after the server load)

Reads an ik server log's placement lines (llama_expert_numa_apply: one line per policy, node(s) + interleave|bind + MiB;
llama_expert_numa_report: the sampled per-node MiB and the share of sampled pages off their target). Expected MiB per node =
the bind MiB on that node + each interleave policy's MiB split evenly over its nodes (rows:A,B prints as one bind per node).
The report's off-target share counts a page on ANY node of an interleave set as on target, so a skewed interleave passes it:
the per-node comparison is what catches that (10-19 q4z: "0,1" interleave 145 GiB placed as N0 30.8 / N1 121 GiB, 3.26 %
off target; b1017heal: N0 +8.5 / N1 -12.4 GiB). Every apply block that placed tensors is checked, summed (a draft model sharing
the placement env prints a second block: 0 tensors when its weights are on the GPU, its own plan + report when a CPU draft's
experts or dense weights match a rule). Both apply-line forms count: "N expert tensors placed" and the dense-rows
"N expert + M other tensors placed".

FAIL (exit 3) when a node is off by more than max(TOL_GIB, TOL_FRAC x its expected MiB) or more than OFF_MAX_PCT of the sampled
pages are off target; exit 2 when the log has no placement block with a report (nothing verified); 0 PASS.
--pid PID measures the live process instead of the load-time report: /proc/PID/numa_maps pages per node of the bind:N and
interleave ranges (only the placement mbinds in llama-server), off target = bind pages not on N + interleave pages outside the
set. Live, only a node SHORT of its plan fails the per-node bar: with the bind pages on target (off share under the bar) a
node over its plan holds bind ranges the plan does not own (llama-server's own per-node runtime binds; 10-20 orn122 +1.1 GiB on
N0 and N1, 10-21 v41fs-fs1b with a CPU draft +0.7 / +2.5 / +0.3 GiB at 0.13 % off), reported as extra_bind_mib; a skewed
interleave still fails on the short node. Use it after a heal (bind-heal.py moved misplaced pages; 10-19 v41r1: the rows placement landed 11.37 % off, 20 GiB of
bind:1 pages on N0/N2 while N1 had 28 GiB free).
Usage: place_verify.py SERVER_LOG [--pid PID] [--json OUT]   (env PLACE_TOL_GIB, PLACE_TOL_FRAC, PLACE_OFF_MAX_PCT override the defaults)
"""
import json
import os
import re
import sys

TOL_GIB = 1.0       # per-node slack: the report samples 1 page in 64, and tensor-edge pages fall to either side
TOL_FRAC = 0.03     # ... or this share of the node's expected MiB, whichever is larger
OFF_MAX_PCT = 1.0   # sampled pages off target (b1017heal 1.95 %, q4z 3.26 %; r1 and the rows smokes 0.00 %)

APPLY = re.compile(r'llama_expert_numa_apply: .*: (\d+) expert (?:\+ (\d+) other )?tensors placed')
POLICY = re.compile(r'llama_expert_numa_apply:\s+node\(s\) ([\d,]+)\s+(interleave|bind)\s+([\d.]+) MiB')
REPORT = re.compile(r'llama_expert_numa_report: verified .*?: (.*?); ([\d.]+)% of sampled pages are off their target')


def blocks(lines):
    """(policies, report) per apply block that placed tensors and has a report"""
    out, cur = [], None
    for l in lines:
        m = APPLY.search(l)
        if m:
            n = int(m.group(1)) + int(m.group(2) or 0)
            cur = {'tensors': n, 'policies': []} if n > 0 else None
            continue
        m = POLICY.search(l)
        if m and cur is not None:
            cur['policies'].append(([int(x) for x in m.group(1).split(',')], m.group(2), float(m.group(3))))
            continue
        m = REPORT.search(l)
        if m and cur is not None:
            got = {int(k): float(v) for k, v in re.findall(r'node(\d+)=([\d.]+)MiB', m.group(1))}
            out.append(dict(cur, measured=got, off_pct=float(m.group(2))))
            cur = None
    return out


def live(numa_maps):
    """(MiB per node, % off target) of the bind:N / interleave:A-B ranges in a numa_maps text"""
    got, off, tot = {}, 0, 0
    for l in numa_maps:
        f = l.split()
        m = re.fullmatch(r'(bind|interleave):([\d,-]+)', f[1]) if len(f) > 1 else None
        if not m:
            continue
        want = set()
        for part in m.group(2).split(','):
            a, _, b = part.partition('-')
            want.update(range(int(a), int(b or a) + 1))
        ps = int((re.search(r'kernelpagesize_kB=(\d+)', l) or [0, 4])[1]) / 1024
        for n, c in re.findall(r' N(\d+)=(\d+)', l):
            mib = int(c) * ps
            got[int(n)] = got.get(int(n), 0.0) + mib
            tot += mib
            off += mib if int(n) not in want else 0
    return {k: round(v, 1) for k, v in got.items()}, round(100.0 * off / tot, 2) if tot else 0.0


def verify(lines, tol_gib=TOL_GIB, tol_frac=TOL_FRAC, off_max=OFF_MAX_PCT, numa_maps=None):
    bl = blocks(lines)
    if not bl:
        return {'status': 'NONE', 'reasons': ['no llama_expert_numa_apply block with tensors and a report: the placement was not applied or not verified']}
    b = {'tensors': sum(x['tensors'] for x in bl), 'policies': [p for x in bl for p in x['policies']], 'measured': {},
         'off_pct': round(sum(x['off_pct'] * sum(x['measured'].values()) for x in bl) / max(1e-9, sum(sum(x['measured'].values()) for x in bl)), 2)}
    for x in bl:
        for k, v in x['measured'].items():
            b['measured'][k] = b['measured'].get(k, 0.0) + v
    if numa_maps is not None:
        b = dict(b, measured=live(numa_maps)[0], off_pct=live(numa_maps)[1])
    exp = {}
    for nodes, _, mib in b['policies']:
        for x in nodes:
            exp[x] = exp.get(x, 0.0) + mib / len(nodes)
    got = b['measured']
    off = {x: round(got.get(x, 0.0) - exp.get(x, 0.0), 1) for x in sorted(set(exp) | set(got))}
    bar = lambda x: max(tol_gib * 1024, tol_frac * exp.get(x, 0.0))
    bad = [x for x in off if (-off[x] if numa_maps is not None else abs(off[x])) > bar(x)]
    reasons = ['node%d %+.1f GiB vs plan %.1f GiB (measured %.1f)' % (x, off[x] / 1024, exp.get(x, 0.0) / 1024, got.get(x, 0.0) / 1024) for x in bad]
    if b['off_pct'] > off_max:
        reasons.append('%.2f %% of sampled pages off their target (bar %.2f %%)' % (b['off_pct'], off_max))
    return {'status': 'FAIL' if reasons else 'PASS', 'reasons': reasons, 'expected_mib': {str(k): round(v, 1) for k, v in sorted(exp.items())},
            'measured_mib': {str(k): v for k, v in sorted(got.items())}, 'off_mib': {str(k): v for k, v in off.items()}, 'off_pct': b['off_pct'],
            'policies': [{'nodes': n, 'mode': m, 'mib': x} for n, m, x in b['policies']],
            'extra_bind_mib': {str(x): off[x] for x in off if numa_maps is not None and off[x] > bar(x)},
            'bars': {'tol_gib': tol_gib, 'tol_frac': tol_frac, 'off_max_pct': off_max}}


def main(argv):
    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__)
        return 0 if argv else 2
    nm = open('/proc/%s/numa_maps' % argv[argv.index('--pid') + 1]).read().splitlines() if '--pid' in argv else None
    with open(argv[0], errors='replace') as f:
        r = verify(f, float(os.environ.get('PLACE_TOL_GIB', TOL_GIB)), float(os.environ.get('PLACE_TOL_FRAC', TOL_FRAC)),
                   float(os.environ.get('PLACE_OFF_MAX_PCT', OFF_MAX_PCT)), nm)
    r['source'] = 'live numa_maps pid ' + argv[argv.index('--pid') + 1] if nm is not None else 'server log report'
    if '--json' in argv:
        with open(argv[argv.index('--json') + 1], 'w') as f:
            json.dump(r, f, indent=1)
    print('== PLACEMENT %s (%s) %s' % (r['status'], r.get('source', ''), '; '.join(r['reasons']) if r['reasons'] else
                                  'per node within max(%g GiB, %g %%), %.2f %% sampled pages off target' % (r['bars']['tol_gib'], 100 * r['bars']['tol_frac'], r['off_pct'])))
    return {'PASS': 0, 'FAIL': 3, 'NONE': 2}[r['status']]


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
