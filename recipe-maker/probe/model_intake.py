#!/usr/bin/env python3
"""Model intake: read a model's layout and per-tensor-class byte totals from headers only, never the weights.

    model_intake.py PATH.gguf                      local GGUF (first shard of a split file is enough)
    model_intake.py --ssh HOST /path/x.gguf        GGUF on another box (probe/gguf.py runs there on stdin; reads headers)
    model_intake.py --hf ORG/REPO                  Hugging Face repo: config.json + safetensors headers (HTTP range
                                                   requests of a few hundred KB per shard)
    model_intake.py --hf ORG/REPO --hf-file X.gguf a GGUF inside a Hugging Face repo (header bytes by range request)
  options: --name KEY (model key), --write data/models/KEY.json, --reference (this file IS the model's reference quant:
           also emit experts_gib / dense_gib under the planner's keys), --diff data.json, --header-out FILE

Emits a model entry in the shape build.py's `models[...]` entries use: L (MoE layers), E, k, P (weights per expert),
layer_ids, tensor_names, plus tagged fields (arch, hidden, expert_ff, n_shared, layers, kv_bytes_per_token, mtp) and
an `intake` block: bytes per tensor class (expert gate/up/down, shared experts, router, attention, linear attention,
embeddings, output, engram / n-gram lookup tables, MTP, vision, norms, other) split by quant type / dtype.
Header values are tagged READ; arithmetic on them that assumes a cache layout is ESTIMATE; what cannot be derived
is UNKNOWN with the reason. Public repos need no token; HF_TOKEN is used if set and never printed.
"""
import argparse
import concurrent.futures
import json
import os
import re
import struct
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gguf  # noqa: E402

R, E, U = 'READ', 'ESTIMATE', 'UNKNOWN'
GiB = 2 ** 30
HF = 'https://huggingface.co'


def n(v, tag, src):
    return {'v': v, 'tag': tag, 'src': src}


# ---------------------------------------------------------------- tensor classes
# (class, regex) in order; the first match wins. Names are matched after stripping "model." / "language_model." prefixes,
# so GGUF (blk.N.ffn_up_exps) and Hugging Face (layers.N.mlp.experts.M.up_proj) names land in the same classes.
CLASSES = [
    ('vision', r'^(visual|vision|vision_tower|vision_model|aligner|image_|mm\.|v\.|multi_modal_projector)'),
    ('mtp', r'^mtp\.|nextn|(^|\.)mtp_'),
    ('shared_expert', r'shared_expert(?!_gate)|shexp(?!.*gate_inp)'),
    ('router', r'shared_expert_gate|ffn_gate_inp|gate_inp_shexp|exp_probs_b|ffn_exps_part|(mlp|ffn)\.gate\.(weight|bias|e_score)|(mlp|ffn)\.gate\.bias'),
    ('expert_gate_up', r'experts\.gate_up_proj'),
    ('expert_gate', r'ffn_gate_exps|experts\.(\d+\.)?(w1|gate_proj)'),
    ('expert_up', r'ffn_up_exps|experts\.(\d+\.)?(w3|up_proj)'),
    ('expert_down', r'ffn_down_exps|experts\.(\d+\.)?(w2|down_proj)'),
    ('engram_table', r'engram[._]embe?d'),
    ('engram', r'engram'),
    ('ngram_table', r'ngram_embedding|per_layer_token_embd'),
    ('ple', r'(^|\.)ple[._]'),
    ('embedding', r'embed_tokens|token_embd|^embed\.weight|^tok_embeddings'),
    ('output', r'^lm_head|^head\.weight|^output\.weight'),
    ('hyper_connection', r'(^|\.|_)hc_|hyper_connection'),
    ('linear_attention', r'linear_attn|(^|\.)ssm_'),
    ('attention', r'attn|attention|indexer|compressor'),
    ('norm', r'norm'),
]
_CRE = [(c, re.compile(p)) for c, p in CLASSES]
LOOKUP = ('engram_table', 'ngram_table', 'embedding')  # a few rows read per token: on file or the slowest tier
EXPERT = ('expert_gate', 'expert_up', 'expert_down', 'expert_gate_up')
NOT_DENSE = EXPERT + LOOKUP + ('mtp', 'vision')      # everything else rides with the dense part (GPU first)


def strip(name):
    return re.sub(r'^(model\.)?(language_model\.)?(model\.)?', '', name)


def layer_of(name):
    m = re.match(r'^(?:blk|layers|h|block)\.(\d+)\.', strip(name))
    return int(m.group(1)) if m else None


def classify(name, layer=None, mtp_from=None, linear_layers=()):
    s = strip(name)
    if mtp_from is not None and layer is not None and layer >= mtp_from:
        return 'mtp'
    for c, rx in _CRE:
        if rx.search(s):
            if c == 'attention' and layer in linear_layers:
                return 'linear_attention'  # GGUF names a linear-attention layer's projections attn_qkv / attn_gate
            return c
    return 'other'


# ---------------------------------------------------------------- sources -> uniform tensor rows [name, dims, type, bytes]
def gguf_rows(hdr):
    return hdr['kv'], [[t[0], t[1], gguf.type_name(t[2]), t[3]] for t in hdr['tensors']]


def gguf_local(path):
    return gguf.read_files([path])


def gguf_ssh(host, path):
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, 'python3', '-', path],
                       input=Path(os.path.join(HERE, 'gguf.py')).read_text(), capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        raise SystemExit('remote header read failed on %s: %s' % (host, p.stderr.strip()[-400:]))
    return json.loads(p.stdout)


def http(url, rng=None, timeout=60):
    hd = {'User-Agent': 'fit-viz-model-intake'}
    if rng:
        hd['Range'] = 'bytes=%d-%d' % rng
    tok = os.environ.get('HF_TOKEN')
    if tok and urllib.parse.urlparse(url).hostname == 'huggingface.co':
        hd['Authorization'] = 'Bearer ' + tok
    with urllib.request.urlopen(urllib.request.Request(url, headers=hd), timeout=timeout) as r:
        return r.read()


def hf_files(repo, rev='main'):
    out, url = [], '%s/api/models/%s/tree/%s?recursive=1' % (HF, repo, rev)
    while url:
        rq = urllib.request.Request(url, headers={'User-Agent': 'fit-viz-model-intake'})
        with urllib.request.urlopen(rq, timeout=60) as r:
            out += [x for x in json.loads(r.read()) if x.get('type') == 'file']
            m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get('Link', ''))
            url = m.group(1) if m else None
    return {x['path']: x.get('size') for x in out}


def hf_url(repo, path, rev='main'):
    return '%s/%s/resolve/%s/%s' % (HF, repo, rev, urllib.parse.quote(path))


def safetensors_header(url):
    hl = struct.unpack('<Q', http(url, (0, 7)))[0]
    if hl > 200 << 20:
        raise ValueError('implausible safetensors header length %d at %s' % (hl, url))
    return json.loads(http(url, (8, 8 + hl - 1)))


def hf_safetensors(repo, files, rev='main', workers=8):
    shards = sorted(p for p in files if p.endswith('.safetensors'))
    if not shards:
        raise SystemExit('no .safetensors files in %s' % repo)
    rows = []
    with concurrent.futures.ThreadPoolExecutor(workers) as ex:
        for f, h in zip(shards, ex.map(lambda p: safetensors_header(hf_url(repo, p, rev)), shards)):
            for k, v in h.items():
                if k != '__metadata__':
                    rows.append([k, v['shape'], v['dtype'], v['data_offsets'][1] - v['data_offsets'][0], f])
    return rows, len(shards)


def hf_gguf(repo, path, files, rev='main'):
    hs = []
    for p in gguf.shard_paths(path):
        if p not in files:
            raise SystemExit('%s not in %s' % (p, repo))
        url = hf_url(repo, p, rev)
        hs.append((os.path.basename(p), gguf.read_header(lambda off, k, u=url: http(u, (off, off + k - 1)), files[p])))
    return gguf.merge(hs)


# ---------------------------------------------------------------- config helpers
def cfg_get(cfg, *keys, default=None):
    """first key found in the text config, then the top level"""
    for c in (cfg.get('text_config') or {}, cfg.get('language_config') or {}, cfg):
        for k in keys:
            if k in c and c[k] is not None:
                return c[k]
    return default


def kv_get(kv, arch, *keys, default=None):
    for k in keys:
        for full in (arch + '.' + k, k):
            if full in kv:
                return kv[full]
    return default


# ---------------------------------------------------------------- summarise
def summarise(rows, shape, src):
    """rows: [name, dims, type, bytes, file]; shape: dict of header/config facts -> model entry"""
    mtp_from = shape.get('mtp_from')
    linear = set(shape.get('linear_layers') or [])
    if not linear:  # GGUF: a layer with ssm_* tensors is a linear-attention layer
        linear = {layer_of(r[0]) for r in rows if re.search(r'(^|\.)ssm_', strip(r[0]))} - {None}
    by_class, layers_exp, unclassified, expert_types = {}, set(), [], {}
    for name, dims, typ, b, *_ in rows:
        ly = layer_of(name)
        c = classify(name, ly, mtp_from, linear)
        d = by_class.setdefault(c, {})
        d[typ] = d.get(typ, 0) + b
        if c in EXPERT and ly is not None:
            layers_exp.add(ly)
            expert_types.setdefault(c, {}).setdefault(typ, 0)
            expert_types[c][typ] += b
        if c == 'other' and len(unclassified) < 20:
            unclassified.append(name)
    gib = lambda d: round(sum(d.values()) / GiB, 3)
    total = sum(sum(d.values()) for d in by_class.values())
    main_layers = shape.get('block_count')
    moe = sorted(layers_exp)
    dense_layers = sorted(set(range(main_layers or 0)) - set(moe)) if main_layers else []
    H, F = shape.get('hidden'), shape.get('expert_ff')
    out = {'name': shape.get('name')}
    if moe:
        out['L'] = len(moe)
        out['layer_ids'] = moe
    if shape.get('E'):
        out['E'] = shape['E']
    if shape.get('k'):
        out['k'] = shape['k']
    if H and F:
        out['P'] = 3 * H * F
        out['P_src'] = '%s: 3 x hidden %d x expert_ff %d' % (src, H, F)
    if shape.get('gguf'):
        names = sorted({re.sub(r'^blk\.\d+\.', 'blk.{L}.', r[0]) for r in rows if classify(r[0], layer_of(r[0]), mtp_from, linear) in EXPERT})
        out['tensor_names'] = names
    out['arch'] = n(shape.get('arch'), R, src)
    for k in ('hidden', 'expert_ff', 'n_shared'):
        out[k] = n(shape.get(k), R, src) if shape.get(k) is not None else n(None, U, '%s: not in the header/config' % k)
    out['layers'] = n({'main': main_layers, 'moe': len(moe), 'dense': dense_layers, 'linear_attention': sorted(linear)}, R, src)
    out['kv_bytes_per_token'] = shape.get('kv') or n(None, U, 'no attention shape in the source')
    mtp_bytes = sum(by_class.get('mtp', {}).values())
    out['mtp'] = n({'layers_config': shape.get('mtp_layers'), 'in_this_source': mtp_bytes > 0, 'gib': round(mtp_bytes / GiB, 3)}, R, src)
    if shape.get('draft_note'):
        out['draft_heads'] = n(shape['draft_note'], R, src)
    out['intake'] = {
        'source': src, 'tag': R, 'files': shape.get('files'), 'total_gib': round(total / GiB, 3),
        'gib_by_class': {c: gib(d) for c, d in sorted(by_class.items())},
        'bytes_by_class': {c: dict(sorted(d.items())) for c, d in sorted(by_class.items())},
        'experts_gib': round(sum(sum(by_class.get(c, {}).values()) for c in EXPERT) / GiB, 3),
        'dense_gib': round(sum(sum(d.values()) for c, d in by_class.items() if c not in NOT_DENSE) / GiB, 3),
        'lookup_gib': {c: gib(by_class[c]) for c in LOOKUP if c in by_class},
        'expert_types': {c: {t: round(b / GiB, 3) for t, b in d.items()} for c, d in sorted(expert_types.items())},
        'unclassified': unclassified,
    }
    if moe and out.get('E') and H and F:
        eb = out['intake']['experts_gib'] * GiB
        out['intake']['expert_bpw'] = round(eb * 8 / (len(moe) * out['E'] * 3 * H * F), 3)
    return out


def kv_estimate(layers_attn, n_kv, head_dim, v_dim=None, kv_lora=None, rope=None, mods=None, src=''):
    """f16 KV cache bytes per token; standard GQA or MLA. Modifiers (compression, windows) -> ESTIMATE or UNKNOWN."""
    if kv_lora:
        b = layers_attn * (kv_lora + (rope or 0)) * 2
        how = '%d layers x (kv_lora_rank %d + rope %d) x 2 B (MLA latent, f16)' % (layers_attn, kv_lora, rope or 0)
    elif n_kv and head_dim:
        b = layers_attn * n_kv * (head_dim + (v_dim or head_dim)) * 2
        how = '%d attention layers x %d KV heads x (K %d + V %d) x 2 B (f16)' % (layers_attn, n_kv, head_dim, v_dim or head_dim)
    else:
        return n(None, U, 'no KV head shape in the source')
    if mods:
        return n(b, E, how + '; NOT applied: ' + mods + ' (the engine may store less)')
    return n(b, R, how + ' (' + src + ')')


def from_gguf(hdr, src):
    kv = hdr['kv']
    arch = kv.get('general.architecture', '?')
    g = lambda *k, default=None: kv_get(kv, arch, *k, default=default)
    L = g('block_count')
    nextn = g('nextn_predict_layers')
    names = [t[0] for t in hdr['tensors']]
    top = max((layer_of(x) for x in names if layer_of(x) is not None), default=-1)
    # llama.cpp counts the nextn (MTP) blocks inside block_count when the blocks stop below it (glm4moe, qwen35moe: GLM-5.3-Flash
    # 46 = 45 + 1, a qwen35moe file 49 = 48 + 1): the last nextn blocks are the MTP head. Files whose blocks run past
    # block_count keep the MTP blocks above it
    main = L - nextn if (L is not None and nextn and top < L) else L
    shape = {'gguf': True, 'name': kv.get('general.name'), 'arch': arch, 'block_count': main, 'E': g('expert_count'), 'k': g('expert_used_count'),
             'hidden': g('embedding_length'), 'expert_ff': g('expert_feed_forward_length'), 'n_shared': g('expert_shared_count'),
             'mtp_layers': nextn, 'mtp_from': main if (main is not None and top >= main) else None,
             'files': [[f[0], f[1], f[2]] for f in hdr['files']]}
    if shape['n_shared'] is None and g('expert_shared_feed_forward_length'):
        shape['n_shared'] = 1  # GGUF stores one fused shared expert with its own width
    rows = [[t[0], t[1], gguf.type_name(t[2]), t[3]] for t in hdr['tensors']]
    # KV: attention layers = all, or every full_attention_interval-th one (hybrid linear-attention models)
    fai = g('full_attention_interval')
    n_attn = (main // fai) if (main and fai) else main
    mods = []
    cr = g('attention.compress_ratios')
    if isinstance(cr, list) and any(cr):
        mods.append('compress_ratios %s' % sorted(set(cr)))
    if g('attention.sliding_window'):
        mods.append('sliding window %s' % g('attention.sliding_window'))
    if g('attention.indexer.head_count'):
        mods.append('sparse-attention indexer (top_k %s)' % g('attention.indexer.top_k'))
    kvl = g('attention.kv_lora_rank')
    if g('attention.sliding_window') and cr and not kvl:
        shape['kv'] = n(None, U, 'per-layer KV compression (%s) + sliding window %s: the cache size depends on the engine layout; not derived from the header' % (
            sorted(set(cr)), g('attention.sliding_window')))
    else:
        shape['kv'] = kv_estimate(n_attn or 0, g('attention.head_count_kv'), g('attention.key_length'), g('attention.value_length'), kvl,
                                  g('rope.dimension_count') if kvl else None, '; '.join(mods), src)
    return rows, shape


def from_hf(cfg, rows, src, nshards):
    tc = lambda *k, default=None: cfg_get(cfg, *k, default=default)
    L = tc('num_hidden_layers')
    lt = tc('layer_types') or []
    linear = [i for i, x in enumerate(lt) if 'linear' in str(x)]
    shape = {'name': (cfg.get('architectures') or ['?'])[0], 'arch': tc('model_type'), 'block_count': L,
             'E': tc('n_routed_experts', 'num_experts', 'num_local_experts'), 'k': tc('num_experts_per_tok', 'moe_topk', 'top_k'),
             'hidden': tc('hidden_size'), 'expert_ff': tc('moe_intermediate_size'),
             'n_shared': tc('n_shared_experts', 'num_shared_experts', default=1 if tc('shared_expert_intermediate_size') else None),
             'mtp_layers': tc('num_nextn_predict_layers', 'mtp_num_hidden_layers'), 'linear_layers': linear, 'files': nshards}
    if tc('dspark_n_routed_experts'):
        shape['draft_note'] = 'DSpark draft config in config.json: %s routed experts, top-%s, block %s, target layers %s' % (
            tc('dspark_n_routed_experts'), tc('dspark_num_experts_per_tok'), tc('dspark_block_size'), tc('dspark_target_layer_ids'))
    full = [i for i, x in enumerate(lt) if 'full' in str(x)] if lt else list(range(L or 0))
    mods = []
    cr = tc('compress_ratios')
    if isinstance(cr, list) and any(cr):
        mods.append('compress_ratios %s' % sorted(set(cr)))
    if tc('indexer_compress_ratio'):
        mods.append('indexer compress ratio %s' % tc('indexer_compress_ratio'))
    if tc('sliding_window') and cr:
        shape['kv'] = n(None, U, 'per-layer KV compression (%s) + sliding window %s: the cache size depends on the engine layout; not derived from config.json' % (
            sorted(set(cr)), tc('sliding_window')))
    else:
        shape['kv'] = kv_estimate(len(full), tc('num_key_value_heads'), tc('head_dim') or (tc('hidden_size') // tc('num_attention_heads') if tc('num_attention_heads') else None),
                                  tc('v_head_dim'), tc('kv_lora_rank'), tc('qk_rope_head_dim'), '; '.join(mods), src)
    return rows, shape


# ---------------------------------------------------------------- diff against the hand table
def diff_model(hand, m):
    rows = []
    v = lambda x: x.get('v') if isinstance(x, dict) and 'tag' in x else x
    for k in ('L', 'E', 'k', 'P'):
        if k in hand and k in m and hand[k] != m[k]:
            rows.append((k, hand[k], m[k]))
    if hand.get('layer_ids') and m.get('layer_ids') and hand['layer_ids'] != m['layer_ids']:
        rows.append(('layer_ids', '%d..%d (%d)' % (hand['layer_ids'][0], hand['layer_ids'][-1], len(hand['layer_ids'])),
                     '%d..%d (%d)' % (m['layer_ids'][0], m['layer_ids'][-1], len(m['layer_ids']))))
    it = m['intake']
    for hk, ik in (('experts_gib', 'experts_gib'), ('dense_gib', 'dense_gib')):
        if hk in hand:
            rows.append((hk + ' (hand, at base type) vs this source', v(hand[hk]), it[ik]))
    hl = sum(v(x['gib']) or 0 for x in hand.get('lookups') or [])
    if hl or it['lookup_gib']:
        rows.append(('lookups gib (hand) vs this source', round(hl, 2), it['lookup_gib']))
    hd = hand.get('draft') or {}
    rows.append(('draft/MTP (hand label) vs this source', hd.get('label'), v(m['mtp'])))
    if 'kv_bytes_per_token' in m:
        rows.append(('kv_bytes_per_token (new)', None, '%s %s' % (v(m['kv_bytes_per_token']), m['kv_bytes_per_token']['tag'])))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('path', nargs='?', help='GGUF path (local, or on --ssh HOST)')
    ap.add_argument('--ssh')
    ap.add_argument('--hf', help='Hugging Face repo id')
    ap.add_argument('--hf-file', help='a GGUF file inside the --hf repo (first shard)')
    ap.add_argument('--rev', default='main')
    ap.add_argument('--name', help='model key in the planner (file name for --write)')
    ap.add_argument('--reference', action='store_true', help='this source is the model reference quant: emit experts_gib/dense_gib too')
    ap.add_argument('--write')
    ap.add_argument('--diff')
    ap.add_argument('--header-out', help='save the raw tensor table (for fixtures)')
    a = ap.parse_args(argv)
    if a.hf and not a.hf_file:
        files = hf_files(a.hf, a.rev)
        cfg = json.loads(http(hf_url(a.hf, 'config.json', a.rev)))
        rows, nsh = hf_safetensors(a.hf, files, a.rev)
        src = 'HF %s@%s config.json + %d safetensors headers' % (a.hf, a.rev, nsh)
        raw = {'config': cfg, 'rows': rows}
        rows, shape = from_hf(cfg, rows, src, nsh)
    else:
        if a.hf:
            hdr = hf_gguf(a.hf, a.hf_file, hf_files(a.hf, a.rev), a.rev)
            src = 'HF %s@%s %s (GGUF headers, range requests)' % (a.hf, a.rev, a.hf_file)
        elif a.ssh:
            hdr = gguf_ssh(a.ssh, a.path)
            src = 'GGUF headers %s:%s' % (a.ssh, a.path)
        elif a.path:
            hdr = gguf_local(a.path)
            src = 'GGUF headers ' + a.path
        else:
            ap.error('give a GGUF path, --ssh HOST PATH, or --hf REPO')
        raw = hdr
        rows, shape = from_gguf(hdr, src)
    if a.header_out:
        json.dump(raw, open(a.header_out, 'w'))
    m = summarise(rows, shape, src)
    if a.reference:
        m['experts_gib'] = n(m['intake']['experts_gib'], R, src + ': routed expert tensors')
        m['dense_gib'] = n(m['intake']['dense_gib'], R, src + ': all tensors except routed experts, lookup tables, MTP and vision')
    js = json.dumps(m, indent=1)
    if a.write:
        os.makedirs(os.path.dirname(os.path.abspath(a.write)), exist_ok=True)
        open(a.write, 'w').write(js + '\n')
    if a.diff:
        key = a.name
        hand = json.loads(Path(a.diff).read_text())['models'].get(key) if key else None
        print('== %s <- %s' % (key, src))
        print('  shape: L %s E %s k %s P %s hidden %s expert_ff %s shared %s; layers %s' % (
            m.get('L'), m.get('E'), m.get('k'), m.get('P'), m['hidden']['v'], m['expert_ff']['v'], m['n_shared']['v'], json.dumps(m['layers']['v'])[:160]))
        print('  GiB by class: ' + ', '.join('%s %.2f' % kv for kv in m['intake']['gib_by_class'].items()))
        print('  expert types: ' + json.dumps(m['intake']['expert_types']) + ('  bpw %.3f' % m['intake']['expert_bpw'] if 'expert_bpw' in m['intake'] else ''))
        print('  kv/token: %s [%s] %s' % (m['kv_bytes_per_token']['v'], m['kv_bytes_per_token']['tag'], m['kv_bytes_per_token']['src'][:200]))
        print('  mtp: %s%s' % (json.dumps(m['mtp']['v']), ('; ' + m['draft_heads']['v']) if 'draft_heads' in m else ''))
        if m['intake']['unclassified']:
            print('  unclassified: %s' % m['intake']['unclassified'][:8])
        if hand:
            for f, x, y in diff_model(hand, m):
                print('  %-48s %s -> %s' % (f, json.dumps(x), json.dumps(y)))
    elif not a.write:
        print(js)
    return 0


if __name__ == '__main__':
    sys.exit(main())
