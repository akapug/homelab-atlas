#!/usr/bin/env python3
"""Append bench sweeps to the bench ledger (runs/bench.jsonl): one row per arm, every number with its log.

The bench ledger holds micro-benchmarks that set physics terms (build.py fits them): all-GPU decode sweeps
(tools/bench_gpu.sh: the per-engine GPU line and the per-token MoE overhead) and NUMA placement sweeps
(tools/bench_numa.sh: the socket term). Run-ledger rows (runs/runs.jsonl) stay recipe runs with a gate capture.

  record_bench.py gpu  --box droopy --device 'Quadro RTX 6000' --backend cuda --engine llamacpp --build b11175 \
                       --when 2026-10-19T08:50Z --log droopy:lab notes --header-host droopy OUT.jsonl
                       [--cond stack=mesa-26.2.5 --cond mmvq=on ...]   (conditions the sweep shares, kept in row.conditions)
  record_bench.py numa --box lassie --backend cpu --engine llamacpp --build 842b188 --when ... --log lassie:... \
                       --cpus-per-node 8 --nodes-per-socket 4 --header-host lassie OUT.jsonl

gpu rows get the bytes one decode token reads, from the model's GGUF header (read over ssh, header bytes only): every
non-expert tensor once (token_embd only when no separate output.weight, since a tied head reads it whole), expert
tensors x experts_used / experts; moe_layers = the blocks holding *_exps tensors.
numa rows get the nodes their threads run on, the nodes holding the weights (the policy's nodes; first touch: nodes
holding over NUMA_FT_SHARE of the process pages, from numastat) and the sockets both span.
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BENCH = os.path.join(ROOT, 'runs', 'bench.jsonl')
GGUF = os.path.join(ROOT, 'probe', 'gguf.py')
NUMA_FT_SHARE = 0.05   # a first-touch node counts as holding weights above this share of the process pages
EXPS = re.compile(r'^blk\.(\d+)\.ffn_(gate|up|down|gate_up)_exps\.')


def header(host, path):
    with open(GGUF) as f:
        out = subprocess.run(['ssh', host, 'python3', '-', '--no-shards', path], stdin=f, capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def read_bytes(h):
    """(bytes one decode token reads, file bytes, moe layers, experts, experts used) from a gguf.py summary"""
    kv, ts = h['kv'], h['tensors']
    arch = kv.get('general.architecture', '')
    E, k = kv.get(arch + '.expert_count') or 0, kv.get(arch + '.expert_used_count') or 0
    names = {t[0] for t in ts}
    tied = 'output.weight' not in names
    total = sum(t[3] for t in ts)
    rd = sum(t[3] * (k / E if E and EXPS.match(t[0]) else 1) for t in ts if t[0] != 'token_embd.weight' or tied)
    layers = {EXPS.match(t[0]).group(1) for t in ts if EXPS.match(t[0])}
    return rd, total, len(layers), E, k


def decode_tps(b):
    """the median repetition: a repetition that still paid kernel JIT (SYCL: a MoE path the bench warm-up did not touch) is an
    outlier, not a sample of decode"""
    s = sorted(b.get('samples_ts') or [])
    return (s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2) if s else b.get('avg_ts')


def cpu_set(spec):
    out = set()
    for part in spec.split(','):
        a, _, b = part.partition('-')
        out |= set(range(int(a), int(b or a) + 1))
    return out


def numa_fields(r, cpn, nps):
    cpu_nodes = sorted({c // cpn for c in cpu_set(r['cpus'])})
    pol, _, arg = r['policy'].partition('=')
    mib = [float(x) for x in r['node_mib'].split()][:-1] if r['node_mib'] else []
    mem = sorted(cpu_set(arg)) if arg else [i for i, x in enumerate(mib) if sum(mib) and x / sum(mib) > NUMA_FT_SHARE]
    sockets = sorted({n // nps for n in cpu_nodes + mem})
    return {'policy': pol, 'cpus': r['cpus'], 'threads': r['threads'], 'cpu_nodes': cpu_nodes, 'mem_nodes': mem, 'sockets': len(sockets),
            'node_mib': mib or None, 'load_before': r.get('load_before') or None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('kind', choices=('gpu', 'numa'))
    ap.add_argument('out', help='the sweep output (one JSON line per arm)')
    for a in ('box', 'backend', 'engine', 'build', 'when', 'log', 'header_host'):
        ap.add_argument('--' + a.replace('_', '-'), required=True)
    ap.add_argument('--device', default=None)
    ap.add_argument('--gpu-id', default=None, help='the card id in build.py boxes[box].gpus (its nameplate bandwidth gives the reach)')
    ap.add_argument('--cpus-per-node', type=int, default=None)
    ap.add_argument('--nodes-per-socket', type=int, default=None)
    ap.add_argument('--round', default='r1')
    ap.add_argument('--note', default='')
    ap.add_argument('--cond', action='append', default=[], metavar='KEY=VALUE',
                    help='a run condition every row of this sweep shares (driver stack, env switches, split mode, device pin); repeatable, kept in row.conditions')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if a.kind == 'numa' and not (a.cpus_per_node and a.nodes_per_socket):
        sys.exit('numa rows need --cpus-per-node and --nodes-per-socket')
    have = {json.loads(x)['id'] for x in open(BENCH) if x.strip()} if os.path.exists(BENCH) else set()
    rows, hdr = [], {}
    for line in open(a.out):
        if not line.startswith('{"arm"'):
            continue
        r = json.loads(line)
        b = r.get('bench') or {}
        rid = '%s-%s-%s-%s-%s' % (a.box, a.backend, r['arm'], a.round, a.when[11:16].replace(':', ''))
        if rid in have:
            sys.exit('row %s already in %s' % (rid, BENCH))
        model = r.get('model') or b.get('model_filename')
        if model not in hdr:
            hdr[model] = read_bytes(header(a.header_host, model))
        rd, total, L, E, k = hdr[model]
        row = {'id': rid, 'when': a.when, 'kind': a.kind + '_decode', 'box': a.box, 'device': a.device or b.get('gpu_info') or b.get('cpu_info'),
               'gpu_id': a.gpu_id, 'round': a.round, 'backend': a.backend, 'engine': a.engine, 'engine_build': a.build, 'build_commit': b.get('build_commit'), 'arm': r['arm'],
               'model_file': model, 'model_kind': r.get('kind') or ('moe' if E else 'dense'), 'read_gb': round(rd / 1e9, 4), 'file_gb': round(total / 1e9, 4),
               'moe_layers': L, 'experts': E, 'experts_used': k, 'rc': r['rc'],
               'decode_tps': decode_tps(b), 'decode_avg': b.get('avg_ts'), 'decode_sd': b.get('stddev_ts'), 'decode_samples': b.get('samples_ts'),
               'gen_tokens': b.get('n_gen'), 'reps': len(b.get('samples_ts') or []),
               'n_gpu_layers': b.get('n_gpu_layers'), 'flash_attn': b.get('flash_attn'),
               'sources': ['%s (arm %s)' % (a.log, r['arm'])], 'tag': 'MEASURED', 'note': a.note}
        if a.cond:
            row['conditions'] = dict(c.split('=', 1) for c in a.cond)
        if a.kind == 'numa':
            row.update(numa_fields(r, a.cpus_per_node, a.nodes_per_socket))
        rows.append(row)
    for row in rows:
        print(json.dumps(row) if a.dry_run else row['id'])
    if not a.dry_run:
        with open(BENCH, 'a') as f:
            f.writelines(json.dumps(x) + '\n' for x in rows)


if __name__ == '__main__':
    main()
