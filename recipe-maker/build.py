#!/usr/bin/env python3
"""Build fit-viz: data.json, index.html and page/*.js (the page's data, loaded by index.html); all generated, never in git.

Inputs (read-only): GGUF header dumps, decode-count files and input snapshots in src/ (a fresh clone builds from these
alone), plus OPTIONAL live source directories on the research hub (see INPUTS below). Sibling files quant-priors.json /
quant-recipes.json are embedded when present.
Run: python3 build.py   (then: node report.js for the numbers in the report)
"""
import json, os, re, datetime, shutil, math, importlib.util
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'src')
GiB = 2 ** 30
# droopy STATE / PLAN VRAM readings are nvidia-smi MiB / 1000 (a 23,040 MiB card reads 23.04); the planner works in GiB
MK = lambda x: x * 1000 / 1024
UNITS = 'lab notes/PLAN VRAM readings are nvidia-smi MiB / 1000, converted here to GiB (x 1000/1024)'

# ---------------- optional live inputs ----------------
# Each input has a snapshot in src/inputs/ (tracked). When the live source exists, the build reads it and refreshes the
# snapshot (a later commit carries the new numbers); when it does not, the build prints a note and uses the snapshot.
# Source directories are settings: env FITVIZ_DIR_<NAME> (e.g. FITVIZ_DIR_PMEM_LANE), else the "dirs" object of the JSON
# file named by FITVIZ_CONFIG (default: fitviz.local.json beside this file, untracked), else the named default below.
# where a reader of the page finds the public copy (the page links it for "use it on your own box"); env FITVIZ_PUBLIC_HOME
PUBLIC_HOME = os.environ.get('FITVIZ_PUBLIC_HOME') or 'https://github.com/akapug/homelab-atlas/tree/main/recipe-maker'
INPUT_DIRS = {
    'pmem_lane': ('lab notes', 'droopy PMem research lane (imatrix expert-count TSVs)'),
    'homelab': ('lab notes', 'homelab repo (dsv41/skew routing results)'),
}
INPUTS = {  # snapshot name -> (source dir setting, path inside it)
    'a5v41-expert-counts.tsv': ('pmem_lane', 'a5v41-expert-counts.tsv'),
    'a5-expert-counts.tsv': ('pmem_lane', 'a5-expert-counts.tsv'),
    '2026-10-08-alloc-3bpw.json': ('homelab', 'dsv41/skew/results/2026-10-08-alloc-3bpw.json'),
}


def input_dirs():
    cfg_path = os.environ.get('FITVIZ_CONFIG') or os.path.join(HERE, 'fitviz.local.json')
    cfg = {}
    if os.path.exists(cfg_path):
        cfg = json.loads(Path(cfg_path).read_text()).get('dirs', {})
    return {k: os.path.expanduser(os.environ.get('FITVIZ_DIR_' + k.upper()) or cfg.get(k) or d) for k, (d, _) in INPUT_DIRS.items()}


def input_path(name, dirs):
    """the live source when present (refreshing the snapshot), else the snapshot; None (with a note) when neither exists"""
    key, rel = INPUTS[name]
    live, snap = os.path.join(dirs[key], rel), os.path.join(SRC, 'inputs', name)
    if os.path.exists(live):
        if not os.path.exists(snap) or Path(live).read_bytes() != Path(snap).read_bytes():
            os.makedirs(os.path.dirname(snap), exist_ok=True)
            shutil.copyfile(live, snap)
            print('input %s: snapshot src/inputs/%s refreshed from the live source (%s)' % (name, name, INPUT_DIRS[key][1]))
        return live
    if os.path.exists(snap):
        print('input %s: live source not here (setting %s, %s); using the snapshot src/inputs/%s' % (name, key, INPUT_DIRS[key][1], name))
        return snap
    print('input %s: NOT FOUND (no live source, no snapshot): the parts built on it are skipped' % name)
    return None


DIRS = input_dirs()

M, R, P, E, FIT = 'MEASURED', 'READ', 'PRIOR', 'ESTIMATE', 'FIT'

# ---------------- evidence: the one trust field ----------------
# Every tagged value also carries evidence = seen | inferred | unknown (REA method), decided by this one table from its tag;
# its source stays in src. The tag keeps the finer kind (a FIT is inferred, but fitted on our own seen points).
SEEN, INFERRED, UNKNOWN = 'seen', 'inferred', 'unknown'
EVIDENCE = {
    'classes': {SEEN: 'our own measurement: a run (runs/runs.jsonl row), probe, bench or a read of our own file; src cites it',
                INFERRED: 'fitted, derived, estimated, or taken from a vendor spec or a community claim; src names the source',
                UNKNOWN: 'not modelled: no number is given, or none is trusted'},
    'by_tag': {M: SEEN, 'MEASURED (bounded)': SEEN, 'MEASURED, CONFOUNDED': SEEN, 'MEASURED-UNKNOWN': UNKNOWN, 'MEASURED-BY-THEM': INFERRED,
               'INFERRED-BY-US': INFERRED, R: INFERRED, P: INFERRED, 'CLAIMED': INFERRED, 'M-ext': INFERRED, E: INFERRED, FIT: INFERRED,
               'UNKNOWN': UNKNOWN},
    # a src that opens with one of these words declares its own class: a tag that disagrees is a build error (10-19 audit:
    # fnx dense_gib was MEASURED with an M-ext source, strata overlap ESTIMATE with an UNKNOWN one)
    'src_words': {'MEASURED': SEEN, 'M-ext': INFERRED, 'ESTIMATE': INFERRED, 'INFERRED': INFERRED, 'PRIOR': INFERRED, 'CLAIMED': INFERRED,
                  'VENDOR': INFERRED, 'UNKNOWN': UNKNOWN},
    # a READ of our own file or box (GGUF header bytes on our disks, a live kit setting) is our own observation, so seen
    'seen_reads': {'v': ['GGUF header', 'mean expert bpw from GGUF header', 'kit-settings/'],
                   'src': 'READ values whose src starts with one of these read our own files; other READs (docs, vendor specs, code tables, HF configs) stay inferred'},
}


def evidence_of(tag, src=None):
    """seen | inferred | unknown for a tag (EVIDENCE.by_tag); a tag with no class is a build error, never a silent default"""
    if tag not in EVIDENCE['by_tag']:
        raise ValueError('tag %r has no evidence class: add it to EVIDENCE.by_tag in build.py' % (tag,))
    if tag == R and any(str(src or '').startswith(x) for x in EVIDENCE['seen_reads']['v']):
        return SEEN
    ev = EVIDENCE['by_tag'][tag]
    w = re.match(r'\s*(' + '|'.join(map(re.escape, EVIDENCE['src_words'])) + r')\b', str(src or ''))
    if w and EVIDENCE['src_words'][w.group(1)] != ev:
        raise ValueError('tag %s (%s) contradicts its src, which opens %r: %s' % (tag, ev, w.group(1), str(src)[:120]))
    return ev


def n(v, tag, src):
    return {'v': v, 'tag': tag, 'src': src, 'evidence': evidence_of(tag, src)}


# ---------------- GGUF headers ----------------
GLM_MTP_BLOCK = 45  # GLM-5.3-Flash: blocks 0-44 + the nextn (MTP) block 45 (glm5-next.block_count 46, nextn_predict_layers 1)


def hdr_summary(path, skip_lookup, mtp_block=GLM_MTP_BLOCK):
    d = json.load(open(path))
    ex = dense = lookup = mtp = 0
    for name, dims, t, b, fn in d['tensors']:
        if name.startswith('blk.%d.' % mtp_block):
            mtp += b
            continue
        if '_exps' in name:
            ex += b
        elif any(s in name for s in skip_lookup):
            lookup += b
        else:
            dense += b
    return d['kv'], ex / GiB, dense / GiB, lookup / GiB, mtp / GiB


v41kv, v41ex, v41dense, v41lookup, _ = hdr_summary(os.path.join(SRC, 'hdr-v41.json'), ('engram_embd', 'token_embd'))
glmkv, glmex, glmdense, glmlookup, glmmtp = hdr_summary(os.path.join(SRC, 'hdr-glm.json'), ('token_embd',))
_, _, glmq3dense, _, _ = hdr_summary(os.path.join(SRC, 'hdr-glm-q3kxl.json'), ('token_embd',))


def mtp_parts(path, mtp_block=GLM_MTP_BLOCK):
    """MTP block bytes (GLM block 45): (non-expert, routed experts) in GiB, from a GGUF header dump"""
    ne = ex = 0
    for name, dims, t, b, fn in json.load(open(path))['tensors']:
        if name.startswith('blk.%d.' % mtp_block):
            if re.match(r'blk\.%d\.ffn_(gate|up|down)_exps\.weight' % mtp_block, name):
                ex += b
            else:
                ne += b
    return ne / GiB, ex / GiB


glm_mtp_ne, glm_mtp_ex = mtp_parts(os.path.join(SRC, 'hdr-glm.json'))
_, glmq3_mtp_ex = mtp_parts(os.path.join(SRC, 'hdr-glm-q3kxl.json'))
HGLMQ3 = 'GGUF header, odie /models/glm53-gguf/UD-Q3_K_XL (read 2026-10-18; dump src/hdr-glm-q3kxl.json)'
# GLM VRAM at -ub 4096, MEASURED on droopy (glm-be1q3 server.log 10-18 07:46Z, UD-Q3_K_XL, ik h2dstage-unpublished): 22322 of 23040 MiB
# free before the load (CUDA context), CUDA0 model buffer 11871.38 MiB = non-experts 8404 + block 45 non-experts 191 + block 45
# routed experts 3276 (ik --cpu-moe does not move the MTP block's experts to the host), KV 215.56 MiB, compute buffer 2484 MiB
# (cudaMalloc failed at 852 hot experts = 8902 MiB)
GLM_VRAM_SRC = 'droopy glm-be1q3-hot8p7-t40 server.log (10-18 07:46Z, UD-Q3_K_XL, -ub 4096, ik h2dstage-unpublished)'
GLM_CTX_MIB, GLM_KV_MIB, GLM_CB4096_MIB = 23040 - 22322, 215.56, 2484.0
V41_P = 5120 * 2304 * 3
GLM_P = 4096 * 2048 * 3
GLM_LAYERS = sorted({int(t[0].split('.')[1]) for t in json.load(open(os.path.join(SRC, 'hdr-glm.json')))['tensors'] if '_exps' in t[0] and not t[0].startswith('blk.%d.' % GLM_MTP_BLOCK)})
glm_bpw = glmex * GiB * 8 / (42 * 288 * GLM_P)
HV41 = 'GGUF header, odie /mnt/bulk/models/DeepSeek-V4.1-Flash-vcruz-Q8_0 (read 2026-10-17 with src/gguf-header.py; dump src/hdr-v41.json)'
GLMQ3_EXP_B = 134398000000  # UD-Q3_K_XL routed bytes, blocks 3-44 (block 45 = MTP, 3.435 GB, not counted)
GLMQ3_SRC = 'GGUF headers odie /models/glm53-gguf/UD-Q3_K_XL (read 2026-10-18 with probe/model_intake.py --ssh odie; per-block table lab notes)'
HGLM = 'GGUF header, odie /mnt/bulk/models/GLM-5.3-Flash-unsloth-UD-Q4_K_XL-GGUF (read 2026-10-17; dump src/hdr-glm.json)'


# ---------------- heat ----------------
# Every routing-heat set comes from a hot-expert map in data/heat (format: tools/heatmap.py, HOT-EXPERT-MAPS.md),
# loaded and validated by the one loader below. Sources of each map: tools/heat_sources.json (convert-all rewrites them).
import sys as _sys
_sys.path.insert(0, os.path.join(HERE, 'tools'))
import heatmap as HM
# the snapshots maps convert from refresh from their live sources first (a changed snapshot makes its map STALE below)
for _name in ('a5v41-expert-counts.tsv', 'a5-expert-counts.tsv'):
    input_path(_name, DIRS)
_p = input_path('2026-10-08-alloc-3bpw.json', DIRS)
alloc = json.load(open(_p)) if _p else None
alloc_layers = [alloc['layers']['model.layers.%d.ffn.experts' % i] for i in range(40)] if alloc else None
HEAT_DIR = os.path.join(HERE, 'data', 'heat')
HEAT_MAPS = HM.load_dir(HEAT_DIR)
for _m in HEAT_MAPS.values():  # a map converted from a tracked snapshot must match it, else it is stale
    _cf = _m['provenance'].get('converted_from', {})
    _rel = _cf.get('path', '')
    if _rel and not _rel.startswith(('~', '/')) and os.path.exists(os.path.join(HERE, _rel)) and HM.sha256_file(os.path.join(HERE, _rel)) != _cf.get('sha256'):
        print('heat map %s is STALE: %s changed since conversion (run: python3 tools/heatmap.py convert-all tools/heat_sources.json data/heat)' % (_m['id'], _rel))
HEAT_USED = set()


def heat_set(map_id, slice_name, label, layer_ids, experts):
    """one planner heat set from a map slice; None (with a note) when the map is not in data/heat"""
    m = HEAT_MAPS.get(map_id)
    if m is None:
        print('heat map %s: NOT FOUND in data/heat: the sets built on it are skipped' % map_id)
        return None
    HEAT_USED.add((map_id, slice_name))
    s = m['slices'][slice_name]
    return {'label': label, 'tag': m['provenance']['tag'], 'layers': HM.planner_layers(m, slice_name, layer_ids, experts), 'map': HM.cite(m, slice_name),
            'src': '%s [%s, slice %s, %s tokens%s, %s on %s, %s; map %s %s]' % (m['provenance']['src'], m['workload']['name'], slice_name, s['tokens'],
                                                                     ' INFERRED' if s.get('tokens_tag') == 'INFERRED' else '', m['producer']['engine'],
                                                                     m['producer']['box'], m['producer']['date'], m['id'], m['digest'][:19])}


def heat_sets(spec, layer_ids, experts):
    """{set name: (map id, slice, label)} -> the sets whose maps are present"""
    out = {}
    for name, (mid, sl, label) in spec.items():
        h = heat_set(mid, sl, label, layer_ids, experts)
        if h:
            out[name] = h
    return out


def extra_heat_sets(family, layer_ids, experts):
    """every other slice of the maps for this model family, named 'map id:slice' (planner options, never defaults)"""
    out = {}
    for mid, m in HEAT_MAPS.items():
        if m['model']['family'] != family or m['model']['experts'] != experts or not set(layer_ids) <= set(m['model']['layers']):
            continue
        for sl in m['slices']:
            if (mid, sl) not in HEAT_USED:
                out[mid + ':' + sl] = heat_set(mid, sl, '%s, %s (%s)' % (m['title'], sl, m['model']['checkpoint']), layer_ids, experts)
    return out


V41_MAP, V41_IMAT, GLM_MAP, GLM_NV2, V40_IMAT = ('dsv41-flash__skew187__dsv41-exl3__2026-10-08', 'dsv41-flash__a5-docs-51k__ik-imatrix__2026-10-16',
                                               'glm53-flash__skew72__llamacpp-cpu__2026-10-09', 'glm53-flash__coding-kld-agent__glm53-nv2__2026-10-17',
                                               'dsv4-flash__a5-docs__ik-imatrix__2026-10-14')

# ---------------- types ----------------
SMOKE = 'odie AVX2 smoke, fraction of the memory ceiling at batch 1 (lab notes 08:05Z); droopy Cascade Lake K1 pending'
QK = 'ggml block size (q*_K: bytes per 256 weights; q8_0/mxfp4 per 32)'
IKQ = 'ik_llama.cpp examples/quantize/quantize.cpp type table'
KEST = 'ESTIMATE: full-model KLD if every routed expert used this type (imatrix requant), placeholder until quant-priors.json gives priors'


def T(bpw, bpw_t, bpw_s, kap, kap_t, kap_s, k41, kglm, gpu=True, needs_code=False, note='', kld_dense=None, engines=('ik',)):
    t = {'bpw': n(bpw, bpw_t, bpw_s), 'kappa': n(kap, kap_t, kap_s),
         'kld': {'v41': n(k41, E, KEST + '; V4.1 experts are natively FP4 so MXFP4 is lossless for V4.1'),
                 'glm': n(kglm, E, KEST + '; GLM source is BF16')},
         'gpu': gpu, 'needs_code': needs_code, 'note': note, 'engines': list(engines)}
    if kld_dense is not None:
        t['kld_dense'] = kld_dense
    return t


KAPPA_AT_LOAD = 'fitted at load from kappa_from (the glm-lite lane bytes rate / the box DRAM read)'
types = {
    'mxfp4': T(4.25, R, QK, 0.85, E, SMOKE, 0.0, 0.015, engines=('ik', 'sycl')),
    'mxfp4_r8': T(4.25, R, QK, 0.96, E, SMOKE, 0.0, 0.015,
                  note='repacked MXFP4 (-rtr). Runs on CUDA hybrid since hot-unrepack (split-unrepack-unpublished). K1 kernel 0.69 vs 0.47, but END TO END on B0 (10-17 14:58Z vs 15:52Z): no decode gain, prefill 3x worse. Used only with the -rtr knob; the rtr check fails it on split models'),
    'q2_K': T(2.625, R, QK, 0.87, E, SMOKE, 0.09, 0.10, engines=('ik', 'sycl')),
    'q2kq3k': T((2 * 2.625 + 3.4375) / 3, R, 'requant A recipe: gate+up q2_K, down q3_K (lab notes 07:15Z) -> mean of block sizes', 0.87, E, SMOKE + ' (q2_K:q3_K)', 0.05, 0.06,
                note='requant A on odie (~183 GiB experts for V4.1)', engines=('ik', 'sycl')),
    'q3_K': T(3.4375, R, QK, 0.87, E, SMOKE, 0.03, 0.035, engines=('ik', 'sycl')),
    'q4_K': T(4.5, R, QK, 0.95, E, SMOKE, 0.006, 0.010, kld_dense=n(0.009, P, 'neuralll GLM-5.3-Flash GSQ-RCO: Q4_K dense part +0.95 % perplexity (CLAIMED, lab notes)'), engines=('ik', 'sycl')),
    'q5_K': T(5.5, R, QK, 0.90, E, 'ESTIMATE (no smoke row)', 0.002, 0.004, engines=('ik', 'sycl')),
    'q6_K': T(6.5625, R, QK, 0.90, E, 'ESTIMATE (no smoke row)', 0.0008, 0.0015, kld_dense=n(0.002, E, 'ESTIMATE'), engines=('ik', 'sycl')),
    'q8_0': T(8.5, R, QK, 0.95, E, 'ESTIMATE (no smoke row)', 0.0002, 0.0004, kld_dense=n(0.0, R, 'baseline: the files keep non-expert weights at Q8_0'), engines=('ik', 'sycl')),
    'iq2_ks': T(2.1875, R, IKQ, 0.33, E, SMOKE, 0.10, 0.12, note='compute-bound on AVX2 (0.33): fine on GPU, poor on CPU tiers'),
    'iq2_kl': T(2.69, R, IKQ, 0.31, E, SMOKE, 0.06, 0.07),
    'iq3_ks': T(3.19, R, IKQ, 0.45, E, SMOKE, 0.028, 0.032),
    'iq4_ks': T(4.25, R, IKQ, 0.54, E, SMOKE, 0.006, 0.009),
    'iq4_xs': T(4.25, R, QK, 0.80, E, 'ESTIMATE (no smoke row)', 0.006, 0.009, engines=('ik', 'sycl')),
    # GLM-only: unsloth UD-Q4_K_XL expert mix as shipped (gate/up q4_K, down q5_K/q6_K)
    'ud_q4kxl': T(round(glm_bpw, 4), R, 'mean expert bpw from ' + HGLM, 0.93, E, 'ESTIMATE: q4_K 0.95 / q5_K 0.90 mix', 0.004, 0.006, engines=('ik',),
                  note='ik loads the shipped file as is (droopy D2 runs, 10-17); the ledger D2 rows still carry q4_K'),
    # GLM-only: unsloth UD-Q3_K_XL expert mix as shipped (gate/up IQ3_XXS, down IQ4_XS; blocks 11/12/44 down Q6_K, 11 gate/up
    # IQ4_XS). Mainline has no AVX2 repack for IQ3_XXS/IQ4_XS (repack.cpp: Q4_0/Q4_K/IQ4_NL/MXFP4 only), so its CPU rate is
    # an IQ-lookup rate: lab notes G12 read ~17 GB/s of odie's 70 at 16 threads on the Q2 sibling.
    'ud_q3kxl': T(round(GLMQ3_EXP_B * 8 / (42 * 288 * GLM_P), 4), R, GLMQ3_SRC + ': routed bytes of MoE blocks 3-44 (134.40 GB) / (42 x 288 x 3 x 4096 x 2048)',
                  0.80, E, 'ESTIMATE: as iq4_xs (no smoke row); the IQ3_XXS grid lookup is likely slower', 0.10, 0.16, engines=('ik', 'sycl')),
    # glm-lite (0xSero exllamav3 trellis store) types
    'exl3_2.05': T(2.05, R, 'nominal bpw of the 2.05-bpw store variant (lab notes 2.7, unrun)', None, FIT, KAPPA_AT_LOAD, 0.02, 0.02, engines=('glmlite',)),
    'exl3_3.05': T(9474048 * 8 / GLM_P, R, 'store record 9,474,048 B per expert (lab notes NOTES, upstream docs/reference.md:287)', None, FIT, KAPPA_AT_LOAD, 0.0032, 0.0032, engines=('glmlite',)),
    'exl3_4.0': T(4.0, R, 'nominal bpw (hypothetical store)', None, FIT, KAPPA_AT_LOAD, 0.001, 0.001, engines=('glmlite',)),
}
types['exl3_3.05']['kld'] = {'v41': n(0.0032, M, 'lab notes G6 (GLM on snoopy, 6 prompts)'), 'glm': n(0.0032, M, 'lab notes G6: paired decode KL 0.0032 / top-1 0.994 over 1,070 positions, six prompts (VALID)')}
types['exl3_2.05']['kld'] = {'v41': n(0.02, E, 'ESTIMATE'), 'glm': n(0.02, E, 'ESTIMATE: 2.05-bpw store never measured (lab notes 2.7 UNRUN)')}
types['exl3_4.0']['kld'] = {'v41': n(0.001, E, 'ESTIMATE'), 'glm': n(0.001, E, 'ESTIMATE')}
# glm-lite CPU lane, MEASURED 10-17 (the AVX-512 lane kernel made it DRAM-bound; zero-copy admits read the same DRAM)
GR = 'lab notes: '
K2E = GR + 'k2e-1450-run.log (lab notes 15:03Z): excl0 + cs32k + AVX-512 lane kernel, desktop off the GPU; per decode token 126.57 VRAM / 209.4 RAM hits (167.5 CPU lane + 43.3 zero-copy), cpu_busy 34.07 ms, C1 decode median 22.22 tok/s'
K2E_RAM_GB = (167.5 + 43.3) * 9437184 / 1e9
LANE_GBS = K2E_RAM_GB / 34.07e-3
P512 = GR + 'p512-cpubench-1439.txt (lab notes 14:40Z): AVX-512 lane, 14 threads, cold experts: K=3 best shape 0.1665 ms/expert at 56.7 GB/s, K=2 0.1169 ms at 53.8 GB/s, pure-load roofline 59.5 GB/s'
types['exl3_3.05']['kappa_from'] = {'box': 'snoopy', 'rate_gbs': LANE_GBS, 'tag': M,
                                    'src': 'all RAM-hit bytes per token / CPU-lane busy time = %.2f GB / 34.07 ms = %.1f GB/s; ' % (K2E_RAM_GB, LANE_GBS) + K2E + '. DRAM-bound: ' + P512}
types['exl3_2.05']['kappa_from'] = {'box': 'snoopy', 'rate_gbs': LANE_GBS, 'mult': 53.8 / 56.7, 'tag': M,
                                    'src': 'K=2 bytes-rate / K=3 bytes-rate on the AVX-512 lane (53.8 / 56.7 GB/s): both DRAM-bound, so time follows bytes. ' + P512}
types['exl3_4.0']['kappa_from'] = {'box': 'snoopy', 'rate_gbs': LANE_GBS, 'mult': 1.0, 'tag': E, 'src': 'ESTIMATE: the lane is DRAM-bound (p512 bench), so a 4-bit store reads at the same bytes rate'}
types['exl3_2.05']['lowbit'] = True
types['q8_0']['kld_dense'] = n(0.0, R, 'baseline: both files keep non-expert weights at Q8_0 (READ headers)')
types['mxfp4_r8']['rtr'] = True
types['iq2xxs_q2k'] = T((2 * 2.0625 + 2.625) / 3, R, 'antirez / smalinin layout: gate+up iq2_xxs (2.0625), down q2_K (2.625); quant-recipes V22', 0.5, E,
                        'ESTIMATE: iq2_xxs CPU speed unmeasured on our AVX2 boxes (quant-recipes rule 9)', 0.26, 0.45, engines=('ik', 'sycl'))

# Qwen3.8-Flash-Next types: ffn_down_exps has 640 rows (640 % 256 != 0), so down tensors take only block-32 types
# (Q2_0, Q4_0, IQ4_NL, MXFP4, Q8_0); K-quant / IQ-256 files fall back to a block-32 type on down (SCOUT section 1).
SC = 'lab notes'
types['q4_0'] = T(4.5, R, QK, 0.95, E, 'ESTIMATE: as q4_K (no smoke row)', None, None, engines=('ik', 'sycl'))
types['iq4_nl'] = T(4.5, R, QK, 0.80, E, 'ESTIMATE: as iq4_xs (no smoke row)', None, None, engines=('ik', 'sycl'))
types['q2_0'] = T(2.40, R, 'ISTA-DASLab GSQ-RCO Q2_0 file: 2.40 bpw (' + SC + ' section 1)', 0.87, E, 'ESTIMATE: as q2_K (no smoke row)', None, None,
                  note='mainline block-32 2-bit type; ik support UNKNOWN; decodes faster than IQ2_XS (93.79 vs 70.30 tok/s, M-ext ISTA card)', engines=('sycl',))
types['q4k_nl'] = T(4.5, R, 'gate/up q4_K (4.5) + down iq4_nl (4.5): the shape a Q4_K file takes under the 640-row limit (unsloth UD-Q4_K_XL class)', 0.90, E,
                    'ESTIMATE: q4_K / iq4_xs mix', None, None, engines=('ik', 'sycl'))
types['gsq_iq3xxs'] = T(3.0, R, 'ISTA GSQ-RCO IQ3_XXS: 3.0 bpw, 18 of 48 down layers IQ4_NL (' + SC + ')', 0.45, E, 'ESTIMATE: IQ3 lookup types are compute-bound on AVX2 (iq3_ks 0.45)', None, None,
                        note='IQ lookup decode was slow on SYCL before (b70-qwen27 lane); prefer a block-32 requant on B70. SEEN 2026-10-19: XeStrata '
                             'xe0.1.45-1 decodes this file faster than gsq_iq3s on odie B70 #1 (92.7 vs 86.8 tok/s code, runs/bench-raw/odie-strata-xe0145-cookbook-1019.jsonl)',
                        engines=('ik', 'sycl', 'strata'))
types['gsq_iq3s'] = T(3.5, R, 'ISTA GSQ-RCO IQ3_S: 3.5 bpw (' + SC + ')', 0.24, E, 'ESTIMATE: droopy K1-FNX measured the file mix at 0.274 against iq3_ks 0.52 (ratio 0.53, 10-18); odie AVX2 iq3_ks smoke 0.45 x 0.53 = 0.24', None, None,
                      note='also the published Coder variant (256 experts) type', engines=('ik', 'sycl', 'strata'))
types['iq4_kt'] = T(4.0, R, 'ik trellis IQ4_KT (ji-farthing file: experts 52.6 GiB)', 0.30, E, 'ESTIMATE: trellis decode is compute-heavy on CPU', None, None, engines=('ik',))
types['iq3_kt'] = T(3.125, R, 'ik trellis IQ3_KT', 0.28, E, 'ESTIMATE: trellis decode is compute-heavy on CPU', None, None, engines=('ik',))
for _q in ('q4k_nl', 'gsq_iq3xxs', 'gsq_iq3s', 'iq4_kt', 'iq3_kt'):
    types[_q]['mix_handles_rows'] = 'a published or defined per-tensor mix: its down tensors already use a block-32 type'

# ---------------- expert-type KLD priors (from the sibling quant-recipes.json measured tables) ----------------
REC = sibling_json = None
_rp = os.path.join(HERE, 'quant-recipes.json')
if os.path.exists(_rp):
    try:
        REC = json.load(open(_rp))
    except Exception:
        REC = None


def rec_row(table, name, col):
    """value of column col for the row whose first cell == name in quant-recipes measured[table]; None if absent"""
    if not REC:
        return None
    t = REC.get('measured', {}).get(table)
    if not t:
        return None
    cols = t.get('cols')
    for r in t.get('rows', []):
        if r[0] == name:
            if cols and col in cols:
                return r[cols.index(col)]
            return None
    return None


RQ = 'quant-recipes.json measured.'
KLD_NOTE = 'mean KLD of a whole file whose routed experts use this type (non-experts kept high); our quality model treats it as the cost of putting every routed expert at this type'


def prior(table, name, col, minus=0.0, scale=1.0, fallback=None, what=''):
    x = rec_row(table, name, col)
    if x is None:
        return n(fallback, E, 'ESTIMATE (sibling row ' + table + '/' + name + ' missing): ' + what)
    val = max(0.0, (x - minus) * scale)
    how = (' minus %.4f floor' % minus if minus else '') + (' x %.2f' % scale if scale != 1 else '')
    return n(round(val, 4), P, RQ + table + ' "' + name + '" ' + col + ' %.4f' % x + how + ' -> ' + what + ' (PRIOR community)')


# V4 family: ONE harness for all V4 types (quant-priors harness_warning: never compare KLD across harnesses).
# AtomicChat DeepSeek-V4-Flash-0731 ladder (ctx 5632 vs Q8) lists expert bpw per file; AD-MXFP4 (lossless experts) is the floor,
# so delta = file KLD - floor = the cost of the expert type. Interpolate the delta in log space over expert bpw.
QPR = None
_qp = os.path.join(HERE, 'quant-priors.json')
if os.path.exists(_qp):
    try:
        QPR = json.load(open(_qp))
    except Exception:
        QPR = None


def qp(model, quant, metric, harness):
    if not QPR:
        return None
    for x in QPR.get('points', []):
        if x.get('model') == model and x.get('quant') == quant and x.get('metric') == metric and harness in (x.get('harness', '') + x.get('url', '')):
            return x
    return None


import math
AC = 'AtomicChat'
ac_floor = qp('DeepSeek-V4-Flash-0731', 'AD-MXFP4', 'mean_kld', AC)
LAD = []
if QPR and ac_floor:
    for x in QPR['points']:
        if x.get('model') == 'DeepSeek-V4-Flash-0731' and AC in x.get('harness', '') and x.get('metric') == 'mean_kld' and x.get('bpw_experts') and x['quant'] != 'AD-MXFP4':
            LAD.append((x['bpw_experts'], x['value'] - ac_floor['value'], x['quant']))
LAD.sort()
LADTXT = ('quant-priors.json points, AtomicChat DeepSeek-V4-Flash-0731 ladder: delta KLD over AD-MXFP4 (%.4f) by expert bpw: ' % ac_floor['value'] +
          ', '.join('%s %.2f->%.3f' % (q_, b_, d_) for b_, d_, q_ in LAD)) if LAD else ''


def lad(bpw):
    """delta KLD at expert bpw on the AtomicChat V4 scale; log-linear between ladder points, log-linear extrapolation beyond"""
    if len(LAD) < 2:
        return None
    pts = LAD
    if bpw > pts[-1][0]:
        return pts[-1][1] * 0.5 ** (bpw - pts[-1][0])  # ESTIMATE above the ladder: halve per extra bit
    if bpw <= pts[0][0]:
        i = 0
    else:
        i = max(j for j in range(len(pts) - 1) if pts[j][0] <= bpw)
    (x0, y0, _), (x1, y1, _) = pts[i], pts[i + 1]
    if x1 == x0:
        return y0
    t = (bpw - x0) / (x1 - x0)
    return math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0)))


def k41(q, factor, why, fallback):
    bpw = types[q]['bpw']['v']
    x = lad(bpw)
    if x is None:
        return n(fallback, E, 'ESTIMATE (quant-priors ladder missing): ' + why)
    inside = LAD[0][0] <= bpw <= LAD[-1][0]
    if bpw > LAD[-1][0]:
        why += '; above the ladder top (%.2f bpw) the delta is halved per extra bit (ESTIMATE)' % LAD[-1][0]
    tag = P if (inside and factor == 1.0) else E
    return n(round(x * factor, 4), tag, ('PRIOR (community) ladder value' if tag == P else 'ESTIMATE from the PRIOR ladder') +
             ' at %.3f bpw%s; %s. %s' % (bpw, (' x %.2f' % factor) if factor != 1 else '', why, LADTXT))


KQ = 1.15  # K-quants lose to i-quants at equal bytes (quant-recipes rule 9)
kld41 = {
    'mxfp4': n(0.0, R, 'quant-recipes rule 8: V4/V4.1 experts are QAT MXFP4, keeping MXFP4 is lossless'),
    'mxfp4_r8': n(0.0, R, 'repacked MXFP4: same values as MXFP4'),
    'q4_K': k41('q4_K', KQ, 'requantized from MXFP4, K-quant factor 1.15 (rule 9)', 0.01),
    'iq4_ks': k41('iq4_ks', 1.0, 'ik iq4_ks ~ mainline i-quant at equal bpw', 0.01),
    'iq4_xs': k41('iq4_xs', 1.0, 'mainline iq4_xs', 0.01),
    'q5_K': k41('q5_K', KQ, 'K-quant factor 1.15', 0.005),
    'q6_K': k41('q6_K', KQ, 'K-quant factor 1.15', 0.003),
    'q8_0': n(0.0005, E, 'ESTIMATE: q8_0 of an FP4 source is near lossless'),
    'iq2_kl': k41('iq2_kl', 1.0, 'ik iq2_kl ~ mainline i-quant at equal bpw (KeinNiemand V4 Flash IQ2_KL file: 0.205 in its own harness)', 0.13),
    'iq2_ks': k41('iq2_ks', 1.0, 'ik iq2_ks ~ mainline i-quant at equal bpw', 0.26),
    'iq2xxs_q2k': k41('iq2xxs_q2k', 1.0, 'this layout measured on V4.1 itself by Lucebox: KL 0.260 / top-1 78.3 % (other harness, agrees)', 0.26),
    'iq3_ks': k41('iq3_ks', 1.0, 'ik iq3_ks ~ mainline IQ3_XS at 3.2 bpw', 0.05),
    'q3_K': k41('q3_K', KQ, 'K-quant factor 1.15', 0.045),
    'q2_K': k41('q2_K', KQ, 'K-quant factor 1.15', 0.17),
    'q2kq3k': k41('q2kq3k', KQ, 'gate/up q2_K + down q3_K at the mean bpw, K-quant factor 1.15', 0.10),
}
G = 'glm53f_unsloth_kld'
GA = 'glm53f_aessedai_kld'
kldglm = {
    'ud_q4kxl': prior(G, 'UD-Q4_K_XL', 'mean_kld', fallback=0.049, what='this exact file (Unsloth GLM-5.3-Flash UD-Q4_K_XL)'),
    'ud_q3kxl': prior(G, 'UD-Q3_K_XL', 'mean_kld', fallback=0.16, what='this exact file (Unsloth GLM-5.3-Flash UD-Q3_K_XL)'),
    'q4_K': prior(GA, 'Q4_K_M', 'kld', scale=1.1, fallback=0.055, what='AesSedai Q4_K_M = q4_K gate/up + q5_K down; x1.1 for down at q4_K too'),
    'q5_K': prior(GA, 'Q5_K_M', 'kld', scale=1.1, fallback=0.03, what='AesSedai Q5_K_M = q5_K gate/up + q6_K down; x1.1'),
    'q6_K': prior(G, 'UD-Q6_K_XL', 'mean_kld', fallback=0.019, what='Unsloth UD-Q6_K_XL'),
    'q8_0': n(0.006, E, 'ESTIMATE: below the q6_K prior'),
    'iq4_xs': prior(GA, 'IQ4_XS', 'kld', scale=0.65, fallback=0.075, what='AesSedai IQ4_XS = iq3_s gate/up + iq4_xs down; x0.65 for iq4_xs everywhere (ESTIMATE factor)'),
    'iq4_ks': prior(GA, 'IQ4_XS', 'kld', scale=0.6, fallback=0.07, what='as iq4_xs, x0.6 (ik iq4_ks slightly better, rule 9; ESTIMATE factor)'),
    'q3_K': prior(G, 'UD-Q3_K_XL', 'mean_kld', scale=1.1, fallback=0.17, what='Unsloth UD-Q3_K_XL (q3_K experts with some layers higher) x1.1'),
    'iq3_ks': prior(G, 'UD-Q3_K_XL', 'mean_kld', scale=1.0, fallback=0.16, what='ESTIMATE factor 1.0: ik iq3_ks at 3.19 bpw ~ a q3_K mix (rule 9)'),
    'q2kq3k': prior(G, 'UD-Q2_K_XL', 'mean_kld', fallback=0.38, what='Unsloth UD-Q2_K_XL (q2_K experts with q3_K on down/sensitive layers)'),
    'q2_K': prior(G, 'UD-Q2_K_XL', 'mean_kld', scale=1.12, fallback=0.42, what='UD-Q2_K_XL x1.12 for q2_K everywhere'),
    'iq2_kl': prior(GA, 'IQ2_S', 'kld', scale=0.9, fallback=0.34, what='AesSedai IQ2_S (iq2_xs gate/up + iq3_xxs down, 2.83 bpw) x0.9 for ik iq2_kl'),
    'iq2_ks': prior(G, 'UD-IQ2_XXS', 'mean_kld', fallback=0.45, what='Unsloth UD-IQ2_XXS'),
    'iq2xxs_q2k': prior(G, 'UD-IQ2_XXS', 'mean_kld', fallback=0.45, what='Unsloth UD-IQ2_XXS (iq2_xxs experts)'),
}
# turboderp harness -> unsloth scale, through the two GGUF files measured in both harnesses
TD = 'turboderp'
_pairs = []
for qn in ('UD-IQ4_XS', 'UD-IQ3_XXS'):
    t_ = qp('GLM-5.3-Flash', qn + ' (GGUF)', 'mean_kld', TD)
    u_ = rec_row(G, qn, 'mean_kld')
    if t_ and u_:
        _pairs.append((qn, u_, t_['value']))
TDR = sum(u_ / t_ for _, u_, t_ in _pairs) / len(_pairs) if _pairs else None
TDTXT = ('turboderp-harness KLD x %.2f = unsloth scale (ratio from %s)' % (TDR, ', '.join('%s %.4f/%.4f' % p_ for p_ in _pairs))) if TDR else ''


def td(quant, fallback, what):
    x = qp('GLM-5.3-Flash', quant, 'mean_kld', TD)
    if not x or not TDR:
        return n(fallback, E, 'ESTIMATE (quant-priors row missing): ' + what)
    return n(round(x['value'] * TDR, 4), P, 'PRIOR (community): quant-priors.json turboderp GLM-5.3-Flash "%s" %.4f; %s; %s' % (quant, x['value'], TDTXT, what))


kldglm['exl3_3.05'] = td('EXL3 3.05', 0.14, 'the store we run on snoopy. lab notes G6 MEASURED 0.0032 is paired KL on the model\'s own greedy decode, a much gentler measure')
kldglm['exl3_2.05'] = td('EXL3 2.05', 0.40, 'the 2.05-bpw store variant')
kldglm['exl3_4.0'] = td('EXL3 4.05', 0.07, 'a 4-bit store (4.05 bpw)')
_nv = td('NVFP4', 0.1, 'NVFP4')
kldglm['mxfp4'] = n(round(_nv['v'] * 1.2, 4), E, 'ESTIMATE: NVFP4 on the unsloth scale (%.4f, %s) x 1.2 because MXFP4 has a coarser scale (E8M0 per 32 vs FP8 per 16)' % (_nv['v'], _nv['tag']))
kldglm['mxfp4_r8'] = kldglm['mxfp4']
for q, t in types.items():
    t['kld'] = {'v41': kld41.get(q, n(None, E, 'not used: this type is for GLM only')), 'glm': kldglm.get(q, t['kld']['glm'])}
# Flash-Next KLD: two PRIOR points (ji-farthing ik KT files, code KLD, its own harness); log-linear in bpw between them,
# below with the same slope, above halved per extra bit (the V4 ladder rule). ONLY block-32-compatible types get a value.
FNX_PTS = [(3.125, 0.078, 'IQ3_KT'), (4.0, 0.056, 'IQ4_KT')]
FNX_SRC = 'ji-farthing ik_llama KT quants of Qwen3.8-Flash-Next: IQ4_KT code KLD 0.056, IQ3_KT 0.078 (' + SC + ' section 1; [M-ext], its own harness)'


def kfnx(q, factor=1.0, why=''):
    b = types[q]['bpw']['v']
    (x0, y0, _), (x1, y1, _) = FNX_PTS
    if b > x1:
        val = y1 * 0.5 ** (b - x1)
    else:
        val = math.exp(math.log(y0) + (b - x0) / (x1 - x0) * (math.log(y1) - math.log(y0)))
    on = any(abs(b - x) < 1e-6 for x, _, _ in FNX_PTS) and factor == 1.0
    return n(round(val * factor, 4), P if on else E, ('PRIOR (community): ' if on else 'ESTIMATE from two PRIOR points at %.3f bpw%s; ' % (b, (' x %.2f' % factor) if factor != 1 else '')) + (why + '; ' if why else '') + FNX_SRC)


_fnx = {'iq4_kt': kfnx('iq4_kt'), 'iq3_kt': kfnx('iq3_kt'), 'mxfp4': kfnx('mxfp4', 1.1, 'MXFP4 requant of a BF16 model (not QAT): x1.1'),
        'q8_0': kfnx('q8_0'), 'q4_0': kfnx('q4_0', 1.25, 'legacy q4_0 x1.25 (no importance weights)'), 'iq4_nl': kfnx('iq4_nl'),
        'q4k_nl': kfnx('q4k_nl', 1.05, 'K-quant gate/up x1.05'), 'q2_0': kfnx('q2_0', 1.0, 'GSQ Q2_0 task average 89.07 vs BF16 93.12 (ISTA)'),
        'gsq_iq3xxs': kfnx('gsq_iq3xxs', 0.8, 'GSQ-RCO x0.8 (task average 92.57 vs BF16 93.12, ISTA: near lossless)'),
        'gsq_iq3s': kfnx('gsq_iq3s', 0.8, 'GSQ-RCO x0.8 (task average 93.26 vs BF16 93.12, ISTA)')}
for q, t in types.items():
    t['kld']['fnx'] = _fnx.get(q, n(None, E, 'not usable for Flash-Next: the 640-row down tensors take only block-32 types, or not in the scout'))
types['exl3_3.05']['g6'] = n(0.0032, M, 'lab notes G6: paired decode KL 0.0032 / top-1 0.994 over 1,070 positions, six prompts (VALID); own-trajectory measure, not the wikitext scale')

# A llama.cpp Q4_K_M file keeps gate/up at q4_K everywhere and down at q6_K on the use_more_bits layers (q4_K on the rest).
# The planner stores those layers as q4k_q6k and the others as q4_K (a family file reads which layers from its header)
KQM_TY = 'q4k_q6k'
_kq = {'q4_K': (4.5, 0.95), 'q6_K': (6.5625, 0.90)}
types[KQM_TY] = T((2 * _kq['q4_K'][0] + _kq['q6_K'][0]) / 3, R, 'gate/up q4_K + down q6_K (' + QK + '): the Q4_K_M use_more_bits layers',
                  round((2 * _kq['q4_K'][0] * _kq['q4_K'][1] + _kq['q6_K'][0] * _kq['q6_K'][1]) / (2 * _kq['q4_K'][0] + _kq['q6_K'][0]), 3), E,
                  'ESTIMATE: q4_K 0.95 / q6_K 0.90 weighted by bytes (2 x 4.5 : 6.5625)', None, None, engines=('ik', 'llamacpp'))
types[KQM_TY]['kld'] = {f: n(None, E, 'not used: a Q4_K_M layer mix') for f in ('v41', 'glm', 'fnx')}

# quality: projected top-1 agreement with the full model = 100 - a * KLD^b, fitted to community (KLD, top-1) pairs
QP = []
for tb, kc, tc in (('glm53f_unsloth_kld', 'mean_kld', 'top1_pct'), ('v4flash_unsloth', 'mean_kld', 'same_top_pct'), ('v4flash_keinniemand', 'mean_kld', 'top1_pct'),
                   ('kimi_k3_unsloth', 'mean_kld', 'top1_pct'), ('v41_lucebox_kl', 'kl', 'top1_pct')):
    t = REC and REC.get('measured', {}).get(tb)
    if not t:
        continue
    for r in t['rows']:
        k, tp = r[t['cols'].index(kc)], r[t['cols'].index(tc)]
        if k and tp and k > 0 and tp < 100:
            QP.append((k, tp, tb + ' ' + str(r[0])))
if len(QP) >= 4:
    import math
    xs = [math.log(k) for k, _, _ in QP]; ys = [math.log(100 - tp) for _, tp, _ in QP]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    qb = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    qa = math.exp(my - qb * mx)
    qres = math.sqrt(sum((100 - qa * k ** qb - tp) ** 2 for k, tp, _ in QP) / len(QP))
    QSRC = 'fit of 100 - top-1 %% = a x KLD^b to %d community (KLD, top-1) pairs in quant-recipes.json (%s); rms %.2f points' % (len(QP), ', '.join(sorted(set(x[2].split(' ')[0] for x in QP))), qres)
    QTAG = P
else:
    qa, qb, QSRC, QTAG = 31.0, 0.47, 'ESTIMATE (sibling tables missing)', E
QPAIRS = [{'kld': k, 'top1': tp, 'src': s_} for k, tp, s_ in QP]

engines = {
    'ik': {'name': 'ik_llama.cpp hybrid (GPU dense + hot experts, CPU experts; GPU and CPU run one after the other)',
           'types': ['mxfp4', 'mxfp4_r8', 'q2_K', 'q2kq3k', 'iq2xxs_q2k', 'q3_K', 'iq2_ks', 'iq2_kl', 'iq3_ks', 'iq4_ks', 'iq4_xs', 'q4_K', 'q5_K', 'q6_K', 'q8_0',
                     'q4_0', 'iq4_nl', 'q4k_nl', 'gsq_iq3xxs', 'gsq_iq3s', 'iq4_kt', 'iq3_kt', 'ud_q4kxl', 'ud_q3kxl', KQM_TY],
           'overlap': n(0.0, M, 'lab notes lever ledger: GPU/CPU overlap (stack2) worth ~1 %')},
    'glmlite': {'name': 'glm-flash-lite / nv2 (0xSero exllamav3 offload; CPU lane runs beside the GPU)', 'types': ['exl3_2.05', 'exl3_3.05', 'exl3_4.0'],
                'overlap': n(1.0, R, 'nv2 design: CPU lane + zero-copy beside GPU (lab notes NOTES)'),
                'host_ms': n(round(1000 / 22.22 - 34.07, 2), M, 'token 45.00 ms (22.22 tok/s) minus 34.07 ms CPU-lane busy; ' + K2E),
                'zero_copy_gbs': n(0.0, M, 'folded into the lane rate: zero-copy admits read the same DRAM (lab notes 16:46Z: 1.99 GB/token / 59.5 GB/s = 33.4 ms vs 34.07 measured). Arm A (10-17 07:14Z, old kernel) had 9.58 GB/s on top of a compute-bound lane'),
                'dense_gib': n(7.9, R, 'lab notes G16: 7.9-GiB sparse checkpoint'), 'dense_label': 'exl3 checkpoint',
                'kv_gib': n(round(24564 / 1024 - 7.9 - 1600 * 9437184 / GiB, 4), M, 'everything on the 3090 Ti that is not checkpoint or slots, at -cs 32768: 23.99 GiB - 7.9 - 1600 slots x 9.437 MB (excl0-1143.stats.json: 1600 slots, mem_peak 23.81). Includes the start-up RESERVE (1.5 GiB) = the transient margin; default context sizes 1408 slots (+1.7 GiB)'),
                'kv_includes_margin': 1.5,
                'hit_anchor': {'model': 'glm', 'slots': 1600, 'gpu_hits': 126.57, 'reads': 336,
                               'src': K2E + ': decode per token 126.57 GPU hits of 336 routed reads (VRAM cache of 1600 slots, desktop off the GPU)'},
                'no_draft': True, 'sm80': True, 'type_scope': 'layer',
                'scope_note': 'split-store branch (lab notes, split-store): per-layer K=2/K=3 stores MEASURED (P3 15:52Z); per-expert mix (P4 S25/S15) testing 16:46Z'},
    'sycl': {'name': 'mainline llama.cpp SYCL on B70 (no ik types; DeepSeek V4.1 support UNKNOWN)',
             'types': ['mxfp4', 'q2_K', 'q2kq3k', 'iq2xxs_q2k', 'q3_K', 'iq4_xs', 'ud_q3kxl', 'q4_K', 'q5_K', 'q6_K', 'q8_0', 'q4_0', 'iq4_nl', 'q2_0', 'q4k_nl', 'gsq_iq3xxs', 'gsq_iq3s'],
             'overlap': n(0.0, E, 'ESTIMATE: same split as ik'),
             # recipe knobs of mainline llama.cpp (2026-10-18, odie GLM lane); each number names its source
             'recipe_rates': {
                 'upload_gbs': n(8.0, 'MEASURED', 'odie B70 card1 2026-10-18 03:50Z smoke (lab notes): a 35B-A3B MoE Q4_K_M, all ~20 GB of experts in the pinned SYCL_Host buffer, op offload, -ub 2048: llama-server cold prefill 770-837 tok/s (2k) = one ubatch of host experts per 2.5-2.7 s, about 8 GB/s including the GPU MoE compute (INFERRED split; PCIe rx counters N/A)'),
                 'nohost_mult': n(0.5, E, 'ESTIMATE: --no-host puts host experts in pageable CPU (or CPU_REPACK) buffers; the upload then stages through a bounce copy'),
                 'cpu_prefill_tps': n(50.0, E, 'ESTIMATE: CPU-only MoE prefill (-nopo 1); droopy V4.1 B0 48-53 tok/s MEASURED on another box and model; odie 35B-A3B MoE screen 243 tok/s at 3B active'),
                 'upload_touched': n(1, R, 'llama.cpp f74e2fa6a src/llama-context.cpp:2762 sched_copy_experts (the scheduler copy callback for host weights): an offloaded MUL_MAT_ID copies only the experts its ids select'),
                 'offload_min_batch': n(32, R, 'llama.cpp f74e2fa6a ggml/src/ggml-sycl/ggml-sycl.cpp:7721 op_offload_min_batch_size = GGML_OP_OFFLOAD_MIN_BATCH or 32 (offload_op: batch >= it)'),
                 'cache_link_gbs': n(16.0, E, 'ESTIMATE: --moe-cache-mib miss upload of single experts (~11 MB) over PCIe 4.0 x16 (31.5 GB/s READ spec) from the pinned buffer'),
                 'cache_realize': n(0.9, E, 'ESTIMATE: the LRU holds about 90 % of the heat an ideal hottest-expert cache of the same size would; PR #29887 hit rates 72-89 % (READ) at 10-28 % of expert bytes'),
                 'allreduce_ms': n(0.05, E, 'ESTIMATE: -sm tensor adds two cross-card reductions per layer (attention out, FFN out) of one hidden vector over PCIe, ~50 us each'),
                 'kv_model': {'glm': {
                     'ub4096': n(8.43, M, 'odie b1n 2026-10-18 07:02Z llama-server -lv 4, two B70s, -c 40960 -fa on: KV+RS 728 + 624 MiB, compute 3872 + 3404 MiB (SYCL0 + SYCL1)'),
                     'ub512': n(2.24, E, 'ESTIMATE: the KV/RS 1.32 GiB of b1n + its compute 7.11 GiB x 512/4096')}},
                 'host_extra_gib': n(7.5, FIT, 'odie b1n 2026-10-18 07:02Z MEASURED SYCL_Host model 87276.81 MiB + compute 689 MiB = 87.9 GiB vs 80.5 GiB of planner DRAM experts: MTP block 45 experts 3.2 GiB, token_embd 0.6, above-mean host blocks 1.0, rest unexplained'),
                 'pinned_driver_extra_gib': n(11.9, M, 'odie b1n4 2026-10-18 08:57:03Z GPUActive 102451924 kB (97.7 GiB, last sample before the stall; the true peak may be higher) vs SYCL_Host model 87276.81 + compute 344.79 + 280.77 MiB = 85.8 GiB, at the first 2k prefill (+7.1 GiB within ~5 s over the 90.5 GiB loaded); c0 08:49Z gave 4.1 GiB over 50.6 GiB. The larger is used'),
                 'container_anon_gib': n(0.9, M, 'odie c0 2026-10-18 08:47Z server container memory.stat anon 954261504 B (file 53.9 GB = model page cache, reclaimable); the pinned buffer is not in the memcg'),
                 'layers_total': n(46, 'READ', 'GLM-5.3-Flash GGUF block_count incl. MTP block 45 (only GLM is run on sycl today)'),
             }},
    # mainline llama.cpp CUDA (--n-cpu-moe hybrid): ledger-only so prior runs on it are recorded under their real engine; no rate
    # model yet (host, lane and IQ CPU rates unmeasured for it), so it stays out of the solver
    'llamacpp': {'name': 'mainline llama.cpp CUDA hybrid (--n-cpu-moe / -ot: GPU dense + whole MoE layers, CPU experts; check-only, no rate model)',
                 'types': ['mxfp4', 'q2_K', 'q3_K', 'iq4_xs', 'q4_K', 'q5_K', 'q6_K', 'q8_0', 'q4_0', 'iq4_nl', 'q2_0', 'q4k_nl', 'gsq_iq3xxs', 'gsq_iq3s', 'ud_q4kxl', 'ud_q3kxl', KQM_TY],
                 'no_solver': True,
                 'unmodelled_why': 'mainline llama.cpp has no planner rate model (host, lane and IQ CPU kernel rates unmeasured for it); ledger-only',
                 'overlap': n(0.0, E, 'ESTIMATE: GPU and CPU run one after the other, as ik')},
    'strata': {'name': 'Strata (Niko1221/Strata @477ac13c; Qwen3.8-Flash-Next engine: expert cache in VRAM, experts in RAM, PLE table read from NVMe)',
               'types': ['q8_0', 'q6_K', 'q5_K', 'q4_K', 'iq4_nl', 'q2_0', 'gsq_iq3s', 'gsq_iq3xxs'],
               'types_src': 'gsq_iq3xxs SEEN 2026-10-19: XeStrata xe0.1.45-1 served the GSQ-RCO IQ3_XXS file on odie B70 #1 (pack built by its '
                            'entrypoint), 92.7 tok/s code decode, answer checks 6/6 at effort none and high (runs/bench-raw/odie-strata-xe0145-cookbook-1019.jsonl)',
               'models': ['fnx', 'fnxc'], 'no_solver': True, 'self_sizing_cache': True,
               'overlap': n(0.0, 'UNKNOWN', 'UNKNOWN: no rate model yet; Strata is check-only (no solver curves)'),
               'packages': {'snoopy': {'backend': 'cuda', 'path': 'odie:/mnt/bulk/build/strata-477ac13c-sm86-z5 (image strata:477ac13c-sm86-z5, sm_86 + znver5, CUDA 13; SHA256SUMS)', 'src': 'odie ~/qfn-kit/Dockerfile.z5 + build-strata-z5.sh (READ 10-17)'},
                            'odie': {'backend': 'sycl', 'path': 'odie container qfn-xestrata (chriscorbell/intel-arc-pro-b70-qwen38-flash-next @6aad350 = XeStrata xe0.1.40.2.2, the SYCL port of Strata)', 'src': 'lab notes; measured run-1018T1549 (51.3 tok/s, GSQ-RCO IQ3_S)'}},
               'backend_note': 'snoopy: CUDA sm_86 Strata build. odie: XeStrata (SYCL port, chriscorbell recipe 6aad350), measured on B70 #1 2026-10-18 with the GSQ-RCO IQ3_S file; upstream Strata SYCL kernels are still listed "built, not run" in its lab notes.',
               'scope_note': 'self-sizes its expert cache (--expert-cache auto) after dense, MTP, KV and its own prompt buffers, from measured free VRAM'},
}

common_heat_floor = n(0.5, E, 'pseudo-count per expert so a never-seen expert is not free (in-sample zero != never)')
models = {
    'v41': {'name': 'DeepSeek V4.1 Flash 1017', 'short': 'V4.1', 'host_group': 'ds', 'L': v41kv['deepseek41.block_count'], 'E': v41kv['deepseek41.expert_count'], 'k': v41kv['deepseek41.expert_used_count'],
            'P': V41_P, 'P_src': HV41 + ': ffn_*_exps [5120, 2304, 384] x3', 'show': True,
            'dense_gib': n(round(v41dense, 3), R, HV41 + ': all tensors except routed experts, engram tables and token_embd (Q8_0 + BF16 output)'),
            'dense_bpw': n(8.5, R, 'Q8_0'), 'base_type': 'mxfp4', 'kld_family': 'v41',
            'experts_gib': n(round(v41ex, 2), R, HV41 + ': 3 x 89.648 GiB MXFP4'),
            'lookups': [{'label': 'engram tables 2 x [256, 384,016,682] q8_0 + token_embd (lookups: ~13 KB read per token)', 'gib': n(round(v41lookup, 2), R, HV41 + '; read per token: lab notes Model files'),
                         'cls': 'engram', 'on_file': True,
                         'why': 'kept on file (-dple / split shards 8-9), random row reads every token: B0 with the shards linked to the droopy HDD ran prefill 3.51 / decode 3.05 tok/s (lab notes 12:47Z + root cause 13:4xZ); the same recipe on NVMe ran ~50 / 8.39 (15:52Z)'}],
            'kv_gib': n(round(MK(20.79 - 2.24) - v41dense - 5.14e9 / GiB, 2), M, 'KV + compute buffers at -ub 4096, post-warm-up: B0 nortr (lab notes 15:52Z) VRAM 20.79 of 23.04 with 2.24 hot banks (= %.2f GiB without banks), DSpark q2 draft 4.79 GiB, non-experts %.2f GiB (header). %s; the B0 nortr kit log reads the same post-warm-up as 20,790 MiB = 20.30 GiB' % (MK(20.79 - 2.24), v41dense, UNITS)),
            'kv_gib_ub512': n(1.4, 'INFERRED-BY-US', 'INFERRED from d1g q2d-p0 (-ub 512): ~21 GiB used = 8.3 weights + 6.5 hot + 4.8 draft + ~1.4'),
            'draft': {'kind': 'external', 'label': 'DSpark draft (q2_K experts)', 'vram_gib': n(round(5.14e9 / GiB - 9 * 472.5 / 1024, 3), M, 'lab notes 01:14Z: q2 V4.1 draft 5.14 GB (v41r1b CUDA0 draft buffer 4898.66 MiB) minus its routed experts (experts_gib)'),
                      'experts_gib': {'mxfp4': n(round(9 * 472.5 / 1024, 3), R, 'droopy ~/models/drafts/DeepSeek-V4.1-Flash-1017-DSpark-exps-Q2_K.gguf header: blk.0-2 ffn_gate/up/down_exps Q2_K [5120, 2304, 128] 472.5 MiB each')},
                      'experts_on_gpu': n(True, M, 'v41r1b (-draft "-ncmoe 0"): the draft CUDA0 buffer 4898.66 MiB holds its experts. Recipe mtp_experts "cpu" (-draft "-ncmoe 3", run-v41fs.sh DRAFT=cpumoe) moves them to the host'),
                      'acceptance': n(0.90, M, 'B0 rtr 14:58Z: 221 of 242 drafted tokens accepted (91 %); B0 nortr 15:52Z ~90 %; n_max 4, conf_cum_min 0.5'),
                      'mult_default': n(1.21, M, 'droopy n4 / n0 phases in one load: B0 nortr 8.39 / 6.83 = 1.23, E2 7.21 / 6.04 = 1.19 (lab notes 15:52Z, 09:45Z); boxes without their own twin use this')},
            'tensor_names': ['blk.{L}.ffn_gate_exps.weight', 'blk.{L}.ffn_up_exps.weight', 'blk.{L}.ffn_down_exps.weight'],
            'layer_ids': list(range(40)),
            'heat': {'floor': common_heat_floor, 'anchor_score': 'coding', 'anchor_sel': 'imatrix', 'plan_default': 'coding_odd', 'score_default': 'coding_even',
                     'sets': dict(heat_sets({
                         'coding_odd': (V41_MAP, 'coding_odd', 'coding decode, odd prompts (plan half)'),
                         'coding_even': (V41_MAP, 'coding_even', 'coding decode, even prompts (held-out half)'),
                         'general': (V41_MAP, 'general_all', 'general decode (agent/house/long/...)'),
                         'imatrix': (V41_IMAT, 'prefill', 'imatrix routing counts, A5 docs corpus (prefill)')}, list(range(40)), 384), **{
                         'coding': {'label': 'coding decode, all prompts', 'sum': ['coding_odd', 'coding_even']},
                         'all': {'label': 'all decode traffic', 'sum': ['coding_odd', 'coding_even', 'general']},
                     })}},
    'glm': {'name': 'GLM-5.3-Flash', 'short': 'GLM', 'L': 42, 'E': glmkv['glm5-next.expert_count'], 'k': glmkv['glm5-next.expert_used_count'],
            'P': GLM_P, 'P_src': HGLM + ': ffn_*_exps [4096, 2048, 288] x3; 42 MoE layers (3-44; 45 = MTP)', 'show': True,
            'dense_gib': n(round(glmdense, 3), R, HGLM + ': non-expert tensors, Q8_0, without token_embd and the MTP block 45'),
            'dense_bpw': n(8.5, R, 'Q8_0'), 'base_type': 'ud_q4kxl', 'kld_family': 'glm',
            'experts_gib': n(round(glmex, 2), R, HGLM),
            'lookups': [{'label': 'token_embd (lookup)', 'gib': n(round(glmlookup, 2), R, HGLM), 'cls': 'token_embd', 'on_file': False}],
            'dense_gib_by_type': {'ud_q3kxl': n(round(glmq3dense, 3), R, HGLMQ3 + ': non-expert tensors without token_embd and block 45')},
            'kv_gib': n(round((GLM_CTX_MIB + GLM_KV_MIB + GLM_CB4096_MIB) / 1024, 3), M, 'CUDA context %d + KV %.2f + compute buffer %d MiB at -ub 4096 (%s; the compute buffer is the allocation that failed). The old ESTIMATE 4.38 was high; the miss was the MTP experts (draft)' % (GLM_CTX_MIB, GLM_KV_MIB, GLM_CB4096_MIB, GLM_VRAM_SRC)),
            'kv_gib_ub512': n(round((GLM_CTX_MIB + GLM_KV_MIB + GLM_CB4096_MIB * 512 / 4096) / 1024, 3), E, 'INFERRED: context + KV as at -ub 4096, compute buffer scaled linearly with ub (2484 x 512/4096 MiB)'),
            'draft': {'kind': 'mtp', 'label': 'MTP head (block 45: attention and, under ik --cpu-moe, its routed experts on the GPU)',
                      'vram_gib': n(round(glm_mtp_ne, 3), R, HGLM + ': block 45 non-expert tensors (191 MiB in both the Q4 and Q3 headers)'),
                      'experts_gib': {'ud_q4kxl': n(round(glm_mtp_ex, 3), R, HGLM + ': blk.45 ffn_*_exps'), 'ud_q3kxl': n(round(glmq3_mtp_ex, 3), R, HGLMQ3 + ': blk.45 ffn_*_exps')},
                      'experts_on_gpu': n(True, M, GLM_VRAM_SRC + ': CUDA0 model buffer 11871 MiB includes the 3276 MiB block 45 experts; llama_expert_numa_apply skips them as not host-resident. Recipe mtp_experts "cpu" (-ot blk.45 exps=CPU) moves them'),
                      'mult_default': n(1.15, E, 'between lassie MTP1 +12 % and droopy MTP1 +20 % (MEASURED)')},
            'tensor_names': ['blk.{L}.ffn_gate_exps.weight', 'blk.{L}.ffn_up_exps.weight', 'blk.{L}.ffn_down_exps.weight'],
            'layer_ids': GLM_LAYERS,
            'heat': {'floor': common_heat_floor, 'anchor_score': 'code', 'anchor_sel': 'code', 'plan_default': 'code', 'score_default': 'code',
                     'sets': dict(heat_sets({
                         'code': (GLM_MAP, 'decode_code', 'code-reading prompts (18)'),
                         'diff': (GLM_MAP, 'decode_diff', 'diff-review prompts (18)'),
                         'tool': (GLM_MAP, 'decode_tool', 'tool-call JSON prompts (18)'),
                         'chat': (GLM_MAP, 'decode_chat', 'chat prompts (18)'),
                         'nv2_code_decode': (GLM_NV2, 'decode', 'coding agent briefs, decode (glm53 nv2 on snoopy, 10-17)'),
                         'nv2_code_prefill': (GLM_NV2, 'prefill', 'coding text prefill, 87k tokens (glm53 nv2 on snoopy, 10-17)')}, GLM_LAYERS, 288), **{
                         'coding': {'label': 'code + diff + tool', 'sum': ['code', 'diff', 'tool']},
                         'all': {'label': 'all 72 prompts', 'sum': ['code', 'diff', 'tool', 'chat']},
                         'agent_mix': {'label': 'coding decode (72-prompt code+diff+tool) + coding-agent prefill (nv2): both outside the droopy bench decode, which uses the nv2 decode briefs', 'sum': ['coding', 'nv2_code_prefill']},
                     })}},
    'v40': {'name': 'DeepSeek V4 Flash 0731 (calibration only; V4.0 is out)', 'short': 'V4.0', 'host_group': 'ds', 'L': 43, 'E': 256, 'k': 6, 'P': 4096 * 2048 * 3,
            'P_src': 'placement-engine.md: ffn_*_exps [4096, 2048, 256] MXFP4, 13.37 MB per expert', 'show': False,
            'dense_gib': n(7.0, R, 'lab notes: attention and shared experts on the GPU (7.0 GiB)'), 'dense_bpw': n(8.5, R, 'Q8_0'),
            'base_type': 'mxfp4', 'kld_family': 'v41', 'lookups': [], 'kv_gib': n(1.5, E, 'ESTIMATE'),
            'draft': {'kind': 'external', 'label': 'DSpark q2 draft', 'vram_gib': n(6.9e9 / GiB, M, 'lab notes P16c 6.9 GB'), 'mult_default': n(1.28, M, 'snoopy S1 16.55 / 12.93')},
            'heat': {'floor': common_heat_floor, 'anchor_score': 'imatrix', 'anchor_sel': 'imatrix', 'plan_default': 'imatrix', 'score_default': 'imatrix',
                     'sets': heat_sets({'imatrix': (V40_IMAT, 'prefill', 'imatrix routing counts, A5 corpus')}, list(range(43)), 256)}},
}

# V4.1 entry = the 1017 checkpoint from 10-17 on (same layout; measurements so far ran the earlier V4.1 weights)
models['v41']['checkpoint'] = {
    'id': 'deepseek-ai V4.1-Flash-1017', 'adopted': '2026-10-17 16:0xZ',
    'layout': n('byte-identical config/layout; all weights changed (MTP heads most)', R, 'lab notes 15:16Z, lab notes V4.1-Flash-1017 section'),
    'quality_prior': n('vendor claims vs 0731: Terminal-Bench 42.3 vs 31.2, ProgramBench 29.1 vs 20.3; scout table: Terminal-Bench 2.1 90.6 vendor / 74.5 Vals, DeepSWE 74.2 vendor, NL2Repo 64.0 vendor', 'CLAIMED',
                       'VENDOR CLAIMS (deepseek-ai card, 10-17), not our measurement; lab notes 15:16Z + lab notes section 1'),
    'transfer': 'speed, VRAM and storage findings measured on the earlier V4.1 weights transfer (same tensors and shapes, lab notes 15:16Z); the per-type KLD ladder is the quantization cost and is unchanged; the DSpark q2 draft and the imatrix/counts are rebuilt for 1017 (REBUILD-1017.md); routing heat is the pre-1017 capture until the 1017 counts land',
}
models['v41']['heat']['note'] = 'routing counts are from the pre-1017 V4.1 capture (2026-10-08); the 1017 imatrix pass (REBUILD-1017.md) replaces them'
# a missing optional input drops what is built on it (printed above), never the build
if 'imatrix' not in models['v41']['heat']['sets']:
    models['v41']['heat']['anchor_sel'] = 'coding'
if not models['v40']['heat']['sets']:
    del models['v40']
for _mk, _fam in (('v41', 'deepseek-v4.1-flash'), ('glm', 'glm-5.3-flash')):
    _md = models[_mk]
    _md['heat']['sets'].update(extra_heat_sets(_fam, _md.get('layer_ids') or list(range(_md['L'])), _md['E']))


# Qwen3.8-Flash-Next (scout 10-17): no routing counts yet -> a synthetic per-layer heat shaped like GLM's LIVE coding routing on
# snoopy (rs-1340: top2000 0.391 / top3000 0.501 / top4000 0.597 of 12,096 experts). Cumulative share C(x) = x^a over the expert
# fraction x; a fitted to those three points. Expert ids are permuted per layer (deterministic) so layers differ.
_gx = [(2000 / 12096, 0.391), (3000 / 12096, 0.501), (4000 / 12096, 0.597)]
FNX_A = sum(math.log(y) / math.log(x) for x, y in _gx) / len(_gx)


def synth_heat(L, E, frac_of=1.0, seed=17):
    out, st = [], seed
    for l in range(L):
        perm = list(range(E))
        for i in range(E - 1, 0, -1):  # Fisher-Yates with an LCG (deterministic)
            st = (st * 1103515245 + 12345) % 2 ** 31
            j = st % (i + 1)
            perm[i], perm[j] = perm[j], perm[i]
        C = lambda x: (x * frac_of) ** FNX_A / frac_of ** FNX_A
        row = [0.0] * E
        for r in range(E):
            row[perm[r]] = round(1e6 * (C((r + 1) / E) - C(r / E)), 3)
        out.append(row)
    return out


FNX_P = 3 * 2560 * 640
FNXH = 'ESTIMATE: synthetic heat, cumulative share = x^%.3f over the expert fraction x, shaped like GLM-5.3-Flash LIVE coding routing on snoopy (lab notes 13:47Z rs-1340: top2000 0.391 / top3000 0.501 / top4000 0.597 of 12,096). Flash-Next routing not counted yet (scout measurement 3 counts it)' % FNX_A
_fnx_hot = 'not counted'
models['fnx'] = {
    'name': 'Qwen3.8-Flash-Next (125B / 6B active)', 'short': 'Flash-Next', 'host_borrow': 'v41', 'L': 48, 'E': 512, 'k': 10,
    'P': FNX_P, 'P_src': 'config: hidden 2560, expert intermediate 640 -> 3 x 2560 x 640 = 4.915M params per expert (' + SC + ' section 1)', 'show': True,
    'dense_gib': n(4.80, 'M-ext', 'M-ext TSUMUGI quant notes: dense + shared + attention 4.80 GiB (MXFP4 file; ' + SC + ')'), 'dense_bpw': n(8.5, E, 'ESTIMATE: Q8_0-class non-experts (scout: ~5.15 GB read per token at Q8)'),
    'base_type': 'q4k_nl', 'kld_family': 'fnx',
    'experts_gib': n(round(48 * 512 * FNX_P * 4.25 / 8 / GiB, 2), E, 'arithmetic at 4.25 bpw; M-ext 62.11 GiB MXFP4 (TSUMUGI)'),
    'lookups': [{'label': 'n-gram (PLE) table 320M x 160, IQ4_NL shard (16 rows read per token)', 'gib': n(round(28.8e9 / GiB, 2), R, 'ISTA GSQ-RCO card: n-gram shard IQ4_NL 28.8 GB (' + SC + ')'),
                 'cls': 'ngram', 'on_file': True, 'why': 'kept on file (--defer-ple): random row reads every token, the same access shape as the V4.1 engram tables (B0 on HDD: prefill 3.51 / decode 3.05)'}],
    'kv_gib': n(2.0, E, 'ESTIMATE (scout): only the 12 QSA layers hold full KV; Gated DeltaNet layers keep a fixed state'),
    'kv_gib_ub512': n(1.2, E, 'ESTIMATE'),
    'draft': {'kind': 'mtp', 'label': 'MTP head (1 layer, own 512-expert MoE, IQ4_KT)', 'vram_gib': n(2.2, E, 'ESTIMATE (scout): MTP head IQ4_KT ~2.2 GiB (unsloth Q8 head 4.1 GB)'),
              'acceptance': n(0.90, 'M-ext', 'ji-farthing IQ4_KT MTP head: ~90 % on code; collapses near 131k depth (' + SC + ')'),
              'mult_default': n(1.5, E, 'ESTIMATE, conservative: analogs 45 -> 90 (5090 ik + MTP + ngram, code) and 39 no-MTP vs 47-50 MTP3 (3090 hot-32 fork); scout 45-75 with MTP vs 35-45 raw')},
    'host_engine': {'sycl': n(2.0, E, 'ESTIMATE: GPU-resident SYCL decode has little host work (a 35B-A3B MoE served on a B70: 198 tok/s at ~2 GB/token leaves <~1 ms beyond the read; vLLM, not llama.cpp)')},
    'row_limits': {'ffn_down_exps': {'rows': 640, 'allowed': ['q2_0', 'q4_0', 'iq4_nl', 'mxfp4', 'q8_0'], 'src': 'config expert intermediate 640: 640 % 256 != 0, block-32 types only (' + SC + ' section 1)'}},
    'quality_note': 'about GLM-5.3-Flash tier on independent boards (AA index 56 vs 57; LiveBench 76.2 vs 71.6; DeepSWE 58.7 vs 63.4); UNKNOWN in our gate until measured (scout measurement 4: our 12-task agent eval)',
    'tensor_names': ['blk.{L}.ffn_gate_exps.weight', 'blk.{L}.ffn_up_exps.weight', 'blk.{L}.ffn_down_exps.weight'],
    'layer_ids': list(range(48)),
    'heat': {'floor': common_heat_floor, 'anchor_score': 'glm_shape', 'anchor_sel': 'glm_shape', 'plan_default': 'glm_shape', 'score_default': 'glm_shape',
             'sets': {'glm_shape': {'label': 'synthetic: GLM live coding-routing shape (no Flash-Next counts yet)', 'tag': E, 'src': FNXH, 'layers': synth_heat(48, 512)}}},
}
# Coder variant: ISTA GSQ-RCO pruned to 256 of 512 experts (code/agent/vision calibration), IQ3_S 3.5 bpw, 29.6 GB resident
_fnxc_ex = 48 * 256 * FNX_P * 3.5 / 8
models['fnxc'] = dict(models['fnx'])
models['fnxc'].update({
    'name': 'Qwen3.8-Flash-Next Coder (GSQ-RCO, 256 of 512 experts)', 'short': 'Flash-Next Coder', 'E': 256, 'base_type': 'gsq_iq3s', 'types_only': ['gsq_iq3s'],
    'dense_gib': n(round((29.6e9 - _fnxc_ex) / GiB, 2), E, 'ISTA Coder card: 29.6 GB resident minus 256 x 48 experts at 3.5 bpw (%.1f GB) = non-experts incl. vision encoder (read per token is less: ~1.6 GB, scout)' % (_fnxc_ex / 1e9)),
    'dense_bpw': n(8.5, E, 'nominal (the file keeps its own non-expert types; the dense-precision knob does not apply)'),
    'experts_gib': n(round(_fnxc_ex / GiB, 2), E, 'arithmetic: 256 x 48 x 4.915M x 3.5 bpw'),
    'quality_note': 'pruned: SWE-bench Verified 75.6 vs base 82.8 (91.3 %), LCB v6 86.3 vs 87.4 (M-ext ISTA card); the prune cost is NOT in the KLD column (only the 3.5-bpw quant cost is); UNKNOWN in our gate until measured',
    'heat': {'floor': common_heat_floor, 'anchor_score': 'glm_shape', 'anchor_sel': 'glm_shape', 'plan_default': 'glm_shape', 'score_default': 'glm_shape',
             'sets': {'glm_shape': {'label': 'synthetic: the hottest half of the Flash-Next shape (prune keeps the most-used experts)', 'tag': E, 'src': FNXH + '; Coder = top 256 of 512 renormalized', 'layers': synth_heat(48, 256, 0.5, 29)}}},
})

# ---------------- model families in families/*.py ----------------
# Each file adds one model family (its header read, KLD priors, models entry and quality-gate family) and runs in this
# namespace, after the built-in models; a family's quality entry goes to FAMILY_QUALITY. Removing a file removes its family
FAMILY_DIR = os.path.join(HERE, 'families')
FAMILY_QUALITY = {}
for _fn in sorted(os.listdir(FAMILY_DIR)) if os.path.isdir(FAMILY_DIR) else []:
    if _fn.endswith('.py'):
        exec(compile(Path(FAMILY_DIR, _fn).read_text(), os.path.join(FAMILY_DIR, _fn), 'exec'), globals())

# draft.kind = which draft the planner's speed-up stands for: 'mtp' = the model's own MTP head (in the model file), 'external' =
# a separate draft model file. A recipe that serves an external draft file to an 'mtp' model measured another draft: its draft
# speed is UNMODELLED (the no-draft prediction stands)
_nokind = [mk for mk, m in models.items() if m.get('draft') and m['draft'].get('kind') not in ('mtp', 'external')]
assert not _nokind, 'draft.kind (mtp or external) missing for ' + ', '.join(_nokind)

# ---------------- boxes ----------------
D01 = 'lab notes'
PROBE = 'read-only probe 2026-10-17 ~01:10Z'
boxes = {
    'droopy': {
        'name': 'droopy (HP Z8 G4)', 'atlas': 'B', 'role': 'research box (Optane tier)',
        'cpu': {'desc': '2x Xeon Gold 5215 (Cascade Lake, AVX-512 VNNI)', 'threads': n(40, M, PROBE + ' nproc'), 'phys': n(20, R, 'lab notes Box'),
                'used': n(20, R, 'best recipe -t 20 -cr 0-19 (lab notes checkpoint 03:45Z)'), 'smt_mult': n(1.0, E, 'unmeasured on droopy: every arm ran 20 threads on 20 cores'),
                'f_default': n(0.5, E, 'prior before fit')},
        'gpus': [{'id': 'rtx6000', 'name': 'Quadro RTX 6000 (sm_75, PCIe 3 x16)', 'vram_gib': n(23040 / 1024, M, D01 + ' + ' + PROBE + ' (23,040 MiB)'),
                  'bw': n(672, R, 'NVIDIA spec sheet, 672 GB/s'), 'engines': ['ik', 'llamacpp'],
                  'other': [{'what': 'coder30 / embed / AFM (stopped for research windows)', 'gib': n(0.0, M, 'd1g q2d-p0 used ~21 of 22.5 GiB itself (lab notes 06:55Z E2 note)')}]}],
        'dram': {'total_gib': n(round((95138 + 92616) / 1024, 1), M, PROBE + ' numactl -H node 0 95,138 MB + node 1 92,616 MB'),
                 'usable_gib': n(150, R, 'lab notes: 150 GiB is droopy\'s realistic expert budget (placement plan used 142.3)'),
                 'bw': n(148, M, 'P17 moe-mbench: expert bytes read without compute 148 GB/s (both sockets, server layout; lab notes 06:41Z)'),
                 'nodes': [{'name': 'node 0 (socket 0)', 'gib': round(95138 / 1024, 1), 'avail_gib': n(86, M, 'dram-fill plans: N0 76 experts at headroom 10 (12:48Z) and 70 at headroom 16 (17:38Z) -> 86 GiB free at plan time')},
                           {'name': 'node 1 (socket 1)', 'gib': round(92616 / 1024, 1), 'avail_gib': n(89, M, 'dram-fill plans: N1 79 at headroom 10, 73 at headroom 16 -> 89 GiB')}]},
        'pmem': {'gib': n(230, M, PROBE + ' node 2 251,904 MB = 234.6 GiB (230 usable, ESTIMATE margin)'),
                 'bw': n(10, M, 'lab notes finding 2: ~10 GB/s from 2 modules with threads on both sockets (13.4 socket-local, P11)'), 'fit_range': [8, 16]},
        'nvme': {'gib': n(345, M, PROBE + ' df: / 79 GB + /mnt/nvme-p4 88 GB free, plus the V4.1 engram shards already there (2 x 97 GiB, lab notes 20:43Z)'),
                 'bw': n(2.5, E, 'ESTIMATE, unmeasured')},
        'upgrades': {'dram': {'gib': n(192e9 / GiB, R, 'lab notes: 12 of 24 slots empty; 12 x 16 GB DDR4-2666 RDIMM ~$150-250; capacity only (no new channels)'), 'what': '+12 x 16 GB DDR4 (fill empty slots)'},
                     'pmem': {'gib': n(704, R, 'lab notes: six 128 GB PMem 100 sticks on socket 0 = 756 GB, ~$220-620 used'),
                              'bw': n(30, E, 'INFERRED lab notes: 6-way ~39 GB/s socket-local; x10/13.4 for both-socket threads'),
                              # the upgrade rate follows the box's fitted 2-stick rate: x bw_scale (knob optaneBwScale brackets it)
                              'bw_scale': n(round(30 / 13.4, 3), E, 'INFERRED lab notes: 6-way ~39 GB/s socket-local x 10/13.4 for both-socket threads = 30 GB/s over the 13.4 GB/s 2-way rate (low end); linear stripe 6/2 = 3.0 (high end: P11 measured 2 sticks = 2.0x one, 6.85 GB/s per DIMM; 6-way request supply from 10 socket-0 cores unmeasured)'),
                              'bw_scale_hi': n(3.0, E, 'INFERRED: linear stripe 6 of 6 channels / 2 (P11: 2-way = 2 x 6.85 GB/s, balanced)'),
                              'what': '+4 Optane PMem 100 sticks (6-way)'}},
        'host_default': {}, 'threads_note': 'ik expert kernel: 20 threads on 20 physical cores; 20 SMT siblings idle',
    },
    'snoopy': {
        'name': 'snoopy (AM5)', 'atlas': 'A', 'role': 'gaming desktop + test box',
        'cpu': {'desc': 'Ryzen 9 9950X3D (16 cores, AVX-512)', 'threads': n(32, M, PROBE + ' nproc'), 'phys': n(16, M, PROBE + ' lscpu'),
                'used': n(16, R, 'ik p16 layout (lab notes S1); glm-lite: 14 lane + main + ctl, readers on SMT siblings (NOTES)'),
                'smt_mult': n(0.88, M, 'lab notes S1: s32 (SMT on) 11.40 vs p16 12.93 tok/s on V4.0'), 'f_default': n(0.9, E, 'prior before fit')},
        'gpus': [{'id': '3090ti', 'name': 'RTX 3090 Ti (sm_86, PCIe 4 x16)', 'vram_gib': n(24564 / 1024, M, PROBE + ' nvidia-smi 24,564 MiB'),
                  'bw': n(1008, R, D01 + ' (1,008 GB/s)'), 'engines': ['ik', 'glmlite', 'strata'],
                  'other': [{'what': 'desktop compositor (gnome-shell + clients)', 'desktop': True,
                             'gib': n(5.6, M, PROBE + ': gnome-shell 5,549 MiB + ~330 MiB other clients; NOTES: 3.9-5.6 GiB')}]}],
        'dram': {'total_gib': n(182, M, PROBE + ' free -g'), 'usable_gib': n(150, R, 'lab notes 08:37Z: user.slice reserve 16 -> 166G cap; V4-0731 ran with 150 GiB resident'),
                 'bw': n(58.5, M, 'lab notes S1: read memory bandwidth 57-60 GB/s at every layout'), 'nodes': [{'name': 'node 0', 'gib': 182}]},
        'pmem': None,
        'nvme': {'gib': n(306, M, PROBE + ' df: / 251 GB + /scratch 77 GB free'), 'bw': n(5.3, R, 'lab notes D12: QD1 5.3 GB/s (NOT RECORDED conditions)')},
        'upgrades': {'dram': {'gib': n(64e9 / GiB, R, 'lab notes: AM5 max 256 GB (4 x 64 GB); speed at 4 x 64 UNKNOWN'), 'what': '4 x 64 GB DDR5 (182 -> 256 GB)'}},
        'host_default': {}, 'threads_note': 'ik: 16 threads on 16 cores (SMT measured -12 %); glm-lite: 14 lane + 2 control + 14 readers',
    },
    'odie': {
        'name': 'odie (TRX40)', 'atlas': 'C', 'role': 'build primary + central storage + resident model service',
        'cpu': {'desc': 'Threadripper 3960X (24 cores, Zen 2, AVX2)', 'threads': n(44, M, PROBE + ' user.slice cpuset 0-21,24-45'), 'phys': n(24, R, D01),
                'used': n(24, E, 'ESTIMATE'), 'smt_mult': n(1.0, E, 'unmeasured'), 'f_default': n(0.8, E, 'ESTIMATE: the kernel-efficiency table itself was measured on odie (4 threads)'), 'f_fixed': True},
        'gpus': [{'id': 'b70_1', 'name': 'Arc Pro B70 #1 (4c:00.0, PCIe 4 x16)', 'vram_gib': n(29.8, R, D01 + ' 32 GB per card'), 'bw': n(608, R, D01 + ' 608 GB/s'), 'engines': ['sycl', 'strata'],
                  'other': [{'what': 'qwen27 experiment card (not loaded now: 226 MiB used)', 'gib': n(0.0, M, PROBE + ' xpu-smi')}]},
                 {'id': 'b70_2', 'name': 'Arc Pro B70 #2 (23:00.0, PCIe 4 x16)', 'vram_gib': n(29.8, R, D01), 'bw': n(608, R, D01), 'engines': ['sycl', 'strata'],
                  'other': [{'what': 'resident always-on model service (keep)', 'household': True, 'gib': n(26984 / 1024, M, PROBE + ' xpu-smi 26,984 MiB')}]}],
        'dram': {'total_gib': n(121, M, PROBE + ' free -g'), 'usable_gib': n(90, E, 'ESTIMATE: odie is the build primary and hosts the resident model service'),
                 'pinned_margin_gib': n(16, R, 'ruling 2026-10-18 09:0xZ (homelab lead + coordinator) after two stalls (b1n2 07:12Z, b1n4 08:57Z): the xe driver allocates the pinned buffer past any floor in one burst and a SIGKILL cannot land while it is inside that allocation (b1n4: killed 08:57:01Z, exited after 09:02:41Z), so the pinned peak keeps 16 GiB beyond the household floor'),
                 'host_floor_gib': n(27, M, 'odie 2026-10-18 07:12:32Z global OOM report (b1n2): CMA free 11.0 GiB (not usable for the pinned SYCL host buffer or user anon) + anon 12.2 GiB (window stop list applied; shmem tmpfs 7.9 GiB inside it) + kernel/page tables ~3 GiB; b1n at 88 GiB pinned loaded, b1n2 at ~94 GiB livelocked the box. The pinned buffer is driver memory (gpu_active), so a docker --memory cap does not hold it: this floor is the containment'),
                 'bw': n(70.1, M, 'lab notes G14: 70.1 GB/s at 16 threads (CONTAMINATED: beside the resident model service)'), 'nodes': [{'name': 'node 0', 'gib': 121}]},
        'pmem': None,
        'nvme': {'gib': n(1490, M, PROBE + ' df /models 1.6 TB free (SN850X)'), 'bw': n(6, E, 'ESTIMATE: SN850X nameplate ~7 GB/s')},
        'upgrades': {'dram': {'gib': n(128e9 / GiB, R, 'lab notes: 4 free DIMM slots (128 -> 256 GB), ~$225-300 per 64 GB stick'), 'what': '+128 GB DDR4 in the 4 free slots'}},
        'host_default': {}, 'threads_note': 'builds share this box (odie = build primary)',
    },
    'lassie': {
        'name': 'lassie (H11DSi)', 'atlas': 'D', 'role': 'instrument / build box',
        'cpu': {'desc': '2x EPYC 7601 (64 cores, Zen 1, AVX2, 8 NUMA nodes)', 'threads': n(128, M, PROBE + ' nproc'), 'phys': n(64, M, PROBE + ' lscpu'),
                'used': n(64, E, 'ESTIMATE'), 'smt_mult': n(1.0, E, 'unmeasured'), 'f_default': n(0.2, E, 'prior before fit')},
        'gpus': [{'id': '1080ti', 'name': 'GTX 1080 Ti (sm_61, PCIe 3)', 'vram_gib': n(10.9, M, D01 + ' 11 GB'), 'bw': n(484, R, 'lab notes (484 GB/s)'), 'engines': ['ik'],
                  'other': [], 'note': 'probe 2026-10-17: nvidia-smi "Driver/library version mismatch" (MEASURED) - card unusable until the driver is fixed'},
                 {'id': 'mi25', 'vendor': 'amd', 'name': 'Radeon Instinct MI25 (gfx900 Vega10, HBM2, PCIe 3 x16)', 'vram_gib': n(16368 / 1024, M, 'llama.cpp Vulkan device list 2026-10-20 (16,368 MiB)'),
                  'bw': n(484, R, 'lab notes: 945 MHz HBM2 x 2048-bit = 484 GB/s plate'), 'engines': ['llamacpp'], 'other': [],
                  'note': 'Vulkan RADV only (no ROCm installed). 110 W cap = cap max (vBIOS 113-D0513500-N00); passive heatsink + owner shroud with a 120 mm fan: continuous load reaches HBM 93 C in ~4 min, fit plateau ~94.5 C vs 95 C crit (lab notes R1b, 2026-10-20): minutes of load are safe, sustained 110 W is marginal'}],
        'dram': {'total_gib': n(938, M, PROBE + ' free -g 1007 GB'), 'usable_gib': n(850, E, 'ESTIMATE: builds run here'),
                 'bw': n(257.7, M, D01 + ' NUMA-local read'), 'nodes': [{'name': 'node %d' % i, 'gib': 117.3} for i in range(8)]},
        'pmem': None,
        'nvme': {'gib': n(70, M, PROBE + ' df / 76 GB free (SATA SSD, no NVMe)'), 'bw': n(0.5, E, 'ESTIMATE SATA')},
        'upgrades': {}, 'host_default': {}, 'host_fixed': {'glm': n(15, E, 'ESTIMATE: one anchor cannot fit both host and f; droopy V4.0 host is ~16 ms')},
        'threads_note': 'Naples: expert kernel 0.62x a reference EPYC at 64 threads (lab notes D2)',
    },
}

# ---------------- measured 10-17: per-box kernel table, disks ----------------
K1 = 'droopy K1 kernel bench (Cascade Lake, 20 threads on cores 0-19, experts in DRAM first-touch, batch 1; frac of the 144 GB/s read ceiling; lab notes k1-results.jsonl, lab notes 09:4xZ)'
_k1 = {'mxfp4': 0.4645, 'q2_K': 0.5308, 'q2kq3k': 0.5308, 'iq2_ks': 0.3391, 'iq2_kl': 0.3765, 'iq3_ks': 0.5217, 'q4_K': 0.5406, 'iq4_ks': 0.6535}
boxes['droopy']['kappa'] = {q: n(x, M, K1 + (' (row q2_K gate/up + q3_K down)' if q in ('q2_K', 'q2kq3k') else '')) for q, x in _k1.items()}
boxes['droopy']['kappa']['q3_K'] = n(0.5308, E, 'ESTIMATE: the K1 q2_K:q3_K row')
boxes['droopy']['kappa']['mxfp4_r8'] = n(0.4645, M, 'END TO END = mxfp4: B0 -rtr arm decode n0 6.91 vs -rtr off 6.83 (lab notes 14:58Z / 15:52Z); K1 kernel-only 0.6885 did not carry to the server')
boxes['droopy']['kappa']['gsq_iq3s'] = n(0.274, M, 'K1-FNX 10-18 17:20-17:27Z (droopy, the K1 bench: ik llama-moe-mbench unpublished build, T20 on CPUs 0-19, ft, B1, xread ceiling 141-147 GB/s; lab notes): Flash-Next shapes (embd 2560, ff 640, k 10 of 512), cells IQ3_XXS 0.27 / IQ3_S 0.24 / IQ2_S 0.29 / IQ4_XS 0.50 gate-up with IQ4_NL down, weighted by the ISTA GSQ-RCO IQ3_S file\'s layer mix (gate/up 17 IQ3_XXS, 10 IQ3_S, 20 IQ2_S, 1 IQ4_XS; down IQ4_NL 39 + Q2_0 9, IQ4_NL standing in for Q2_0, which the ik bench lacks) = 0.274; experts 25.9 ms per token for 48 layers. Drift anchor: iq3_ks at V4.1 shapes 0.52 = K1 0.5217. ik iqk kernels; mainline llama.cpp IQ kernels unmeasured')
_kr = _k1['mxfp4'] / 0.85
for q in types:
    if q not in boxes['droopy']['kappa'] and types[q]['kappa']['v'] is not None and 'kappa_from' not in types[q]:
        boxes['droopy']['kappa'][q] = n(round(types[q]['kappa']['v'] * _kr, 3), E, 'ESTIMATE: odie AVX2 smoke value x %.3f (droopy K1 mxfp4 / odie mxfp4)' % _kr)
boxes['droopy']['cpu']['used'] = n(40, M, 'B0 ran T40 (20 cores + SMT siblings, lab notes 15:00Z); T20 vs T40 is B1b arm 2')
# CPUs kept for the system during a CPU-decode run (the cpu_reserve check); a box without its own row takes
# checks.cpu_reserve.default_reserve
boxes['droopy']['cpu']['reserve'] = n(1, R, 'kit-settings/droopy.env KIT_CPU_RESERVE=1 with KIT_SYS_PIN=1 (installed as ~/.config/optane-kit/settings.env; srv-arm.sh CPUS=auto = every online CPU but the highest CPU_RESERVE), set after runs row droopy-glm-glm-be1q4t-0330 (fit-viz 55f2358)')

# NUMA page layout of the DRAM experts: CPU expert time relative to interleaved pages (the layout every anchor the box f is
# fitted on ran with). rows = each expert's row halves bound to the socket whose MoE thread group reads them (placement
# rows:0,1 + socket-ordered -cr); first_touch = the kernel bench's socket-local pages; node = every expert whole on one node
_K1N = 'runs/bench-raw/droopy-k1glm-2026-10-19.jsonl K1-GLM-N (ik llama-moe-mbench unpublished build, GLM-5.3 expert shapes, T38): B1 / B4 us per layer '
boxes['droopy']['numa_layout'] = {
    'interleave': n(1.0, M, 'reference: the layout of every droopy anchor the box f is fitted on (dram-fill "0,1" interleave)'),
    'rows': n(0.737, M, 'runs rows droopy-glm-glm-be1r1-1140 vs droopy-glm-glm-be1q4z-0956 (same recipe, same greedy tokens, only the layout and -cr order differ): '
                        'CPU MoE ms per pass (up/gate + fork/join) at depth 32k tolerance 0, K1 62.7 / 85.1 = 0.737, K4 177.0 / 244.6 = 0.724, '
                        'K8 306.5 / 407.9 = 0.751, mean 0.737; the kernel bench agrees (first touch / interleave 0.71-0.73)'),
    'first_touch': n(0.722, M, _K1N + 'first touch 1222 / 3689 vs interleave 1715 / 5041 = 0.713 / 0.732, mean 0.722'),
    'node': n(1.344, M, _K1N + 'every expert on node 1 2326 / 6711 vs interleave 1715 / 5041 = 1.356 / 1.331, mean 1.344'),
}

# ik build flag GGML_EXPERT_CHUNKING (cmake default ON): decode tok/s with ON / with OFF on this box, same source and layout.
# The decode model assumes OFF (every anchor ran an OFF package); a recipe declares expert_chunking 'on' | 'off'.
boxes['droopy']['ik_expert_chunking_on'] = n(0.797, M, 'runs droopy-glm-glm-be1dv-1751 (package dvocab-ee53cdd6, chunking ON) vs droopy-glm-glm-be1dvnc-1202 '
                                             '(dvocab-ee53cdd6-nc, OFF; same source, r1 layout, same cells): MTP3 p_min 0.75 DV0 12.71 / 15.95 = 0.797, '
                                             'no-spec 9.41 / 12.30 = 0.765; OFF = r1 15.73 (+1.4 %). Every droopy ik package builds OFF')

# split GGUF host time per model: the box fit (calibrate, from the V4.1 split anchor d-v41-b0) holds for that model; a model with
# its own row here uses it instead of borrowing V4.1's
models['glm']['split_host_ms'] = n(None, 'UNKNOWN', 'UNKNOWN: no GLM split-GGUF run exists, so the V4.1 fit (37.6 ms, d-v41-b0, one point confounded with T40) '
                                   'is not borrowed. Bounds from GLM measurements: about 0.8 ms if a split adds one CPU expert chain per MoE layer at the measured '
                                   'fork cost (42 layers x 17.6 us, lab notes 1a, r1/q4z oprof), up to about 39 ms if it costs what V4.1 paid per MoE layer '
                                   '(37.6 x 42 / 40); one GLM split arm measures it')

DM = 'homelab inventory/drive-map.toml + lab notes storage table'
boxes['droopy']['disks'] = [
    {'mount': '/', 'kind': 'nvme', 'src': DM + '; holds V4.1 shard 8 (~/models/v41)'},
    {'mount': '/mnt/nvme-p4', 'kind': 'nvme', 'src': 'lab notes 20:43Z (10-16): shard 9 on /mnt/nvme-p4'},
    {'mount': '/mnt/datadrive1', 'kind': 'hdd', 'src': 'inventory/nodes.toml: HGST HUS726T4TAL 4 TB 7200 rpm, NTFS through ntfs-3g FUSE (lab notes 10-15)'}]
boxes['snoopy']['disks'] = [
    {'mount': '/', 'kind': 'nvme', 'src': DM}, {'mount': '/scratch', 'kind': 'unknown', 'src': DM + ' (device type not recorded; UNKNOWN until read: findmnt + lsblk -d -o NAME,ROTA,TRAN on the recipe paths, e.g. qfn-e2e.sh QFN_PLE_DIR=/scratch/lab notes)'},
    {'mount': '/mnt/cold', 'kind': 'hdd', 'src': 'lab notes: internal 500 GB SATA HDD'}]
boxes['odie']['disks'] = [
    {'mount': '/', 'kind': 'nvme', 'src': DM + ' (WD SN750)'}, {'mount': '/models', 'kind': 'nvme', 'src': DM + ' (WD SN850X 2 TB)'},
    {'mount': '/mnt/bulk', 'kind': 'hdd', 'src': DM + ': 2 x 8 TB SMR btrfs RAID1'}]
boxes['lassie']['disks'] = [{'mount': '/', 'kind': 'sata-ssd', 'src': PROBE + ' (SATA SSD, no NVMe)'}]
for bk, kind in (('droopy', 'nvme'), ('snoopy', 'nvme'), ('odie', 'nvme'), ('lassie', 'sata-ssd')):
    boxes[bk]['nvme']['kind'] = kind

# ---------------- intake overlays: data/boxes/*.json (probe/probe_box.py), data/models/*.json (probe/model_intake.py) ----------------
# Probed / header-read values override hand values of the same key (rules in probe/overlay.py); hand-only fields (fitted
# bandwidths, usable budgets, notes, heat) stay. A file with no hand entry is skipped: the planner needs fields a probe
# cannot measure. FITVIZ_OVERLAYS=0 builds from the hand tables alone (what probe --diff compares against).
if os.environ.get('FITVIZ_OVERLAYS', '1') != '0':
    import importlib.util
    _spec = importlib.util.spec_from_file_location('fitviz_overlay', os.path.join(HERE, 'probe', 'overlay.py'))
    _ov = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_ov)
    for _tbl, _dir, _kind in ((models, 'models', 'model intake'), (boxes, 'boxes', 'box probe')):
        _skip, _logs = _ov.apply(_tbl, os.path.join(HERE, 'data', _dir), _kind)
        for _k, _lg in _logs.items():
            print('overlay data/%s/%s.json: %d replaced, %d added, %d conflicts kept as hand' % (
                _dir, _k, sum(x[0] == 'replace' for x in _lg), sum(x[0] == 'add' for x in _lg), sum(x[0] == 'conflict' for x in _lg)))
        for _k in _skip:
            print('overlay data/%s/%s.json: no hand entry for %s, skipped' % (_dir, _k, _k))

b70move = {'gpu': {'id': 'b70_moved', 'name': 'Arc Pro B70 moved from odie (32 GB)', 'vram_gib': n(29.8, R, D01), 'bw': n(608, R, D01), 'engines': ['ik', 'sycl'],
                   'other': [], 'needs_code': {'ik': 'ik_llama.cpp is CUDA/CPU only: experts on a B70 beside the CUDA card need a SYCL/XPU expert worker (the 0xSero dsv41 recipe has one, lab notes)'},
                   'note': 'glm-lite cannot use it (CUDA trellis kernels). ik needs a SYCL/XPU worker (code). Mainline SYCL can drive it alone (the CUDA card then idles).'}}

# ---------------- anchors ----------------
S = 'lab notes'
anchors = [
    # droopy V4.0 (calibration of droopy's CPU factor; all experts in DRAM)
    dict(id='d-v40-k6', box='droopy', model='v40', meas=1000 / 87.0, fit='speed', tag=M, src='lab notes cost model (P6): k=6 87 ms/token, all DRAM', label='droopy V4.0, k=6, all DRAM'),
    dict(id='d-v40-k1', box='droopy', model='v40', k=1, meas=1000 / 43.7, fit='speed', tag=M, src='lab notes cost model (P6): k=1 43.7 ms/token', label='droopy V4.0, k=1 (lossy, info)'),
    # droopy V4.1 (MXFP4 experts)
    dict(id='d-v41-ctl', box='droopy', model='v41', pmem={'mode': 'uniform', 'share': 191.8 / (43.3 + 36.9 + 191.8)}, meas=3.25, fit='speed', tag=M,
         src=S + ' 21:17Z d1-v41-ctl: 3.25 tok/s, N0 43.3 / N1 36.9 / N2 191.8 GiB (tiering on)', label='droopy V4.1 control (70 % on PMem)'),
    dict(id='d-v41-dram', box='droopy', model='v41', pmem={'mode': 'uniform', 'share': 147.4 / (65.4 + 59.1 + 147.4)}, meas=3.99, fit='speed', tag=M,
         src=S + ' 01:17Z d1-v41-dram 5:5:11: 3.99, N2 147.4 GiB (54 %)', label='droopy V4.1 plain DRAM-heavier (54 % PMem)'),
    dict(id='d-v41-place12fb', box='droopy', model='v41', hot_n=685, pmem={'mode': 'uniform', 'share': 112.8 / (80.7 + 78.5 + 112.8)}, meas=5.48, fit='speed', tag=M,
         src=S + ' 01:14Z d1-v41-place12 (placement skipped, hot12 685 experts, 41 % PMem, tiering on)', label='droopy V4.1 hot12, unplaced (41 % PMem)'),
    dict(id='d-v41-place12', box='droopy', model='v41', hot_n=685, pmem={'mode': 'coldest', 'gib': 112}, meas=7.70, fit='speed', tag=M,
         src=S + ' 06:55Z d1g place12-p0: 7.44/7.95/7.70 (n112 placement + hot12, no draft, P0)', label='droopy V4.1 place12-p0 (placed, hot12, no draft)'),
    dict(id='d-v41-q2dfb', box='droopy', model='v41', hot_n=round(6.5 * GiB / (V41_P * 4.25 / 8)), pmem={'mode': 'uniform', 'share': 107.4 / (80.8 + 84.6 + 107.4)}, draft=True, meas=5.33, fit='draft', tag=M,
         src=S + ' 06:55Z d1g q2d-fb-p0: 5.33 (no placement, hot6.5, q2 draft; node split from d1e 107.4 GiB on N2)', label='droopy V4.1 same, unplaced (q2 draft)'),
    dict(id='d-v41-q2d', box='droopy', model='v41', hot_n=round(6.5 * GiB / (V41_P * 4.25 / 8)), pmem={'mode': 'coldest', 'gib': 112}, draft=True, meas=8.04, fit='draft', tag=M,
         src=S + ' 06:55Z d1g q2d-p0: 7.98/8.62/7.53 = 8.04 BEST (n112 placement + hot6.5 + q2 draft)', label='droopy V4.1 d1g q2d-p0 BEST'),
    # droopy V4.1 10-17 (lab-best + split + hot-unrepack; -ub 4096, DSpark n_max 4, P0, NVMe engram)
    dict(id='d-v41-e2', box='droopy', model='v41', hot_n=228, pmem={'mode': 'coldest', 'gib': 129.3}, meas=6.04, fit='speed', tag=M, prefill=36,
         src=S + ' 09:45Z E2 (Q8_0 master = MXFP4 experts, n112-hot4 = 228 experts / 4 GiB on GPU, N2 129.3 GiB, -ub 4096, T20): n0 phase 6.04, prefill ~35', label='droopy V4.1 E2 n0 (no draft phase)'),
    dict(id='d-v41-e2d', box='droopy', model='v41', hot_n=228, pmem={'mode': 'coldest', 'gib': 129.3}, draft=True, twin='d-v41-e2', meas=7.21, fit='draft', tag=M, prefill=36,
         src=S + ' 09:45Z E2 n4 phase 7.21 (same load)', label='droopy V4.1 E2 n4 (DSpark)'),
    dict(id='d-v41-b0', box='droopy', model='v41', meas=6.83, fit='split', tag=M, prefill=50, lowbit_share=0.225,
         split={'hot_set': 'alloc3', 'hot_type': 'mxfp4', 'cold_type': 'q2_K', 'gpu_n': 130, 'pmem_cold_gib': 62}, sel='imatrix', score='coding_even',  # banks ranked by the a5 counts (gen-hot-split.py W=6), = the imatrix set
         src=S + ' 15:52Z B0 nortr: n0 phase 6.83 (split GGUF: the 7,187 alloc-3bpw coding experts MXFP4, the rest q2_K; dram-fill placement N0 81.5 / N1 77 / N2 62 GiB; 130 hot experts = 2.24 GiB banks; T40). Layout rebuilt here from the alloc set + held-out heat (the lane quoted 13.6 %% GPU / 5.1 %% PMem reads in-sample). q3_K down on cold L0-9 ignored (+3 %% cold bytes)'.replace('%%', '%'),
         label='droopy V4.1 B0 split, -rtr off, n0'),
    dict(id='d-v41-b0d', box='droopy', model='v41', draft=True, twin='d-v41-b0', meas=8.39, fit='draft', tag=M, prefill=50, lowbit_share=0.225,
         src=S + ' 15:52Z B0 nortr: decode n4 8.39 [8.26 8.76 8.16], n4b 8.36, prefill 47.8-52.8, draft acceptance ~90 %, VRAM 20.79/23.04', label='droopy V4.1 B0 split, -rtr off, n4 (BEST)'),
    dict(id='d-v41-b0rtr', box='droopy', model='v41', meas=6.91, fit='check', tag=M, prefill=16, rtr=True, lowbit_share=0.225,
         src=S + ' 14:58Z B0 -rtr arm: n0 3.91/6.91/7.04, n4 8.14/6.35/6.17, n4b 8.75/9.27, prefill 15-17 (same layout as B0; predicted with the -rtr-off layout)', label='droopy V4.1 B0 split, -rtr ON, n0'),
    dict(id='d-v41-ds', box='droopy', model='v41', pmem={'mode': 'uniform', 'share': 198 / (40 + 35 + 198)}, draft=True, meas=3.68, fit='check', tag=M,
         src=S + ' 22:53Z d1-v41-ds: 3.68 (control layout + Q8 DSpark draft)', label='droopy V4.1 control + Q8 draft'),
    # droopy GLM
    dict(id='d-glm-ctl', box='droopy', model='glm', pmem={'mode': 'uniform', 'share': 0.47}, meas=3.64, fit='speed', tag=M,
         src=S + ' 06:55Z D2 GLM UD-Q4_K_XL ctl 3.64 (-w 0,1,2 1:1:1, ~47 % PMem, no hot)', label='droopy GLM control (47 % PMem)'),
    dict(id='d-glm-mtp1', box='droopy', model='glm', pmem={'mode': 'uniform', 'share': 0.47}, draft=True, twin='d-glm-ctl', meas=4.38, fit='draft', tag=M, src=S + ' 06:55Z D2 mtp1 4.38', label='droopy GLM MTP1'),
    dict(id='d-glm-mtp3', box='droopy', model='glm', pmem={'mode': 'uniform', 'share': 0.47}, draft=True, twin='d-glm-ctl', meas=4.57, fit='check', tag=M, src=S + ' 07:15Z d2 mtp3 4.57', label='droopy GLM MTP3'),
    # snoopy V4.0 (calibration of snoopy's CPU factor)
    dict(id='s-v40-p16', box='snoopy', model='v40', meas=12.93, fit='speed', tag=M, src='lab notes S1: no draft p16 12.93 (V4.0 all DRAM)', label='snoopy V4.0 no draft'),
    dict(id='s-v40-hot13', box='snoopy', model='v40', hot_n=1044, meas=16.29, fit='speed', tag=M, src='lab notes S1: hot13 alone 16.29 (1,044 experts, placement-engine.md)', label='snoopy V4.0 hot13'),
    dict(id='s-v40-q2', box='snoopy', model='v40', draft=True, twin='s-v40-p16', meas=16.55, fit='draft', tag=M, src='lab notes S1: q2 draft only 16.55', label='snoopy V4.0 q2 draft'),
    dict(id='s-v40-best', box='snoopy', model='v40', hot_n=481, draft=True, meas=19.28, fit='check', tag=M, src='lab notes S1: q2 draft + stack + hot6 (481 experts) 19.28', label='snoopy V4.0 best (q2 draft + hot6 + stack)'),
    # snoopy GLM glm-lite (its own engine)
    dict(id='s-glm-lite', box='snoopy', model='glm', engine='glmlite', meas=15.55, fit='ref', tag=M,
         explicit={'dense_gb': 7.9 * GiB / 1e9, 'gpu_gb': 100.6 * 9.474048e-3, 'dram_gb': 234.7 * 9.474048e-3, 'pmem_gb': 0, 'nvme_gb': 0.7 * 9.474048e-3, 'type': 'exl3_3.05'},
         eng_over={'host_ms': 64.3 - 51.7, 'zero_copy_gbs': 52.3 * 9.474048 / 51.7, 'lane_gbs': 184.2 * 9.474048 / 51.7},
         src='lab notes NOTES arm A: 15.55 tok/s; per token VRAM 100.6 / RAM 234.7 / NVMe 0.7 hits, CPU lane 51.7 of 64.5 ms (desktop on, exclusive RAM tier, AVX2 lane kernel); predicted with its own measured host/lane/zero-copy rates',
         label='snoopy GLM glm-lite 3.05, arm A (desktop on, old lane; history)'),
    dict(id='s-glm-ctl', box='snoopy', model='glm', engine='glmlite', meas=17.88, fit='ref', tag=M,
         explicit={'dense_gb': 7.9 * GiB / 1e9, 'gpu_gb': 117.569 * 9.437184e-3, 'dram_gb': 218.36 * 9.437184e-3, 'pmem_gb': 0, 'nvme_gb': 0.071 * 9.437184e-3, 'type': 'exl3_3.05'},
         eng_over={'host_ms': 1000 / 17.88 - 44.802, 'zero_copy_gbs': 0, 'lane_gbs': 218.36 * 9.437184e-3 / 44.802e-3},
         src=GR + 'ctl-0929.json c1 median 17.88 (bench.py median of 6, run 09:31-09:33Z) + ctl-0929.stats.json decode_per_token of the SAME run (vram 117.6 / ram 218.4 hits, cpu_busy 44.80 ms); lab notes 09:3xZ ctl desktop off the GPU. The first load (17.45, kit-0925.log) has no stats file; until 10-17 21Z this anchor paired 17.45 with these stats', label='snoopy GLM glm-lite ctl, desktop out (history)'),
    dict(id='s-glm-excl0', box='snoopy', model='glm', engine='glmlite', meas=19.51, fit='ref', tag=M,
         explicit={'dense_gb': 7.9 * GiB / 1e9, 'gpu_gb': 99.42 * 9.437184e-3, 'dram_gb': 220.412 * 9.437184e-3, 'pmem_gb': 0, 'nvme_gb': 0.016 * 9.437184e-3, 'type': 'exl3_3.05'},
         eng_over={'host_ms': 1000 / 19.51 - 39.602, 'zero_copy_gbs': 0, 'lane_gbs': 220.412 * 9.437184e-3 / 39.602e-3},
         src=GR + 'excl0-1137.stats.json (1408 slots, cpu_busy 39.60 ms); lab notes 11:50Z excl0 C1 19.51', label='snoopy GLM excl0 (history)'),
    dict(id='s-glm-cs32k', box='snoopy', model='glm', engine='glmlite', meas=20.36, fit='ref', tag=M,
         explicit={'dense_gb': 7.9 * GiB / 1e9, 'gpu_gb': 109.234 * 9.437184e-3, 'dram_gb': 212.067 * 9.437184e-3, 'pmem_gb': 0, 'nvme_gb': 0.017 * 9.437184e-3, 'type': 'exl3_3.05'},
         eng_over={'host_ms': 1000 / 20.36 - 38.06, 'zero_copy_gbs': 0, 'lane_gbs': 212.067 * 9.437184e-3 / 38.06e-3},
         src=GR + 'excl0-1143.stats.json (1600 slots, cpu_busy 38.06 ms, AVX2 lane compute-bound 0.228 ms/expert); lab notes 11:50Z excl0 + cs32k C1 20.36', label='snoopy GLM excl0 + cs32k (history)'),
    dict(id='s-glm-k2e', box='snoopy', model='glm', engine='glmlite', meas=22.22, fit='engine', tag=M, decode_kl=0.00440,
         explicit={'dense_gb': 7.9 * GiB / 1e9, 'gpu_gb': 126.57 * 9.437184e-3, 'dram_gb': K2E_RAM_GB, 'pmem_gb': 0, 'nvme_gb': 0.03 * 9.437184e-3, 'type': 'exl3_3.05'},
         src=K2E + '; decode-KL PASS served 0.00440 vs ref 0.00446 (k2e-1450-kl.json)', label='snoopy GLM 3.05 excl0 + cs32k + AVX-512 lane (BEST, KL pass)'),
    dict(id='s-glm-p3', box='snoopy', model='glm', engine='glmlite', meas=25.13, fit='check', tag=M, decode_kl=0.07217, lowbit_share=30 / 42,
         explicit={'dense_gb': 7.9 * GiB / 1e9, 'parts': [['gpu', 'exl3_3.05', 128.26 * 12 / 42 * 9.474048e-3], ['gpu', 'exl3_2.05', 128.26 * 30 / 42 * 2.05 * GLM_P / 8 / 1e9],
                                                        ['dram', 'exl3_3.05', 207.72 * 12 / 42 * 9.474048e-3], ['dram', 'exl3_2.05', 207.72 * 30 / 42 * 2.05 * GLM_P / 8 / 1e9]],
                   'gpu_gb': 0, 'dram_gb': 0, 'pmem_gb': 0, 'nvme_gb': 0.03 * 9.474048e-3, 'type': 'exl3_3.05'},
         src=GR + 'p3-run.log + p3-1503-bench.log (lab notes 15:52Z): middle 30 of 42 layers K=2 2.05 bpw; per token vram 128.26 / ram 207.72, cpu_busy 27.6 ms; C1 25.13; QUALITY FAIL KL 0.0722 / top-1 0.940 (p3-1503-kl.json). Hits split 12:30 by layer (ASSUMED even per layer)',
         label='snoopy GLM P3 per-layer K=2 split (QUALITY FAIL)'),
    # lassie GLM (ik card arm)
    dict(id='l-glm', box='lassie', model='glm', meas=8.63, fit='speed', tag=M,
         explicit={'dense_gb': 8.81, 'gpu_gb': 0, 'dram_gb': 3.74, 'pmem_gb': 0, 'nvme_gb': 0, 'type': 'q3_K'},
         src='lab notes G2: 8.63 tok/s, dense 8.81 GB on the 1080 Ti, 3.74 GB experts/token from RAM (NOT RECORDED conditions)', label='lassie GLM UD-Q3_K_XL card arm'),
    dict(id='l-glm-mtp', box='lassie', model='glm', draft=True, twin='l-glm', meas=9.65, fit='draft', tag=M,
         explicit={'dense_gb': 8.81, 'gpu_gb': 0, 'dram_gb': 3.74, 'pmem_gb': 0, 'nvme_gb': 0, 'type': 'q3_K'}, src='lab notes 1b: 9.65 with MTP 1', label='lassie GLM MTP1'),
    # odie reference (not fitted)
    dict(id='o-glm-g12', box='odie', model='glm', meas=7.5, fit='ref', tag=M,
         explicit={'dense_gb': 6.45, 'gpu_gb': 0, 'dram_gb': 9.2 - 6.45, 'pmem_gb': 0, 'nvme_gb': 0, 'type': 'q2_K'},
         src='lab notes G12: odie B70 #1 SYCL UD-Q2_K_XL 7.5 tok/s at 16 threads (CONTAMINATED; experts partly not resident)', label='odie GLM Q2 on B70 SYCL (reference only)'),
]

_ab = {a['id']: a for a in anchors}
for _i in ('d-v41-b0d', 'd-v41-b0rtr'):
    for _k in ('split', 'sel', 'score'):
        _ab[_i][_k] = _ab['d-v41-b0'][_k]
anchors = [a for a in anchors if a['model'] in models and not (alloc is None and a.get('split'))]

# ---------------- measured 10-17: hard checks ----------------
NUMA_B = lambda h: {'headroom_gib': h}
B1B = 'B1b (droopy, launched 16:3xZ: 3.0 GiB banks T40, then 2.2 GiB T20). Survives -> margin <= 1.49 GiB; OOM -> margin > 1.49 GiB'
checks = {
    'cpu_reserve': {
        'label': 'CPU decode keeps the box reserve free of decode threads (threads + cpu_reserve declared in recipe.conditions)',
        'default_reserve': n(1, E, 'ESTIMATE for a box without its own cpu.reserve: the droopy measurement (one spare CPU with the system pinned to it) applied to every ggml-style CPU decode'),
        'stall_evidence': n(6.6, M, 'droopy GLM UD-Q4_K_XL T40 on 40 CPUs (runs rows q4l/q4i/q4j, 10-18/19): any co-runner that takes one CPU stalls every spin barrier: 2-3 s token stalls (q4l, reaper / hub ssh), first token 6646 ms (q4i, 19-47 ssh logins per window), 264-420 ms/token through a reaper pass (q4j)'),
        'fix_evidence': n(123.2, M, 'runs row droopy-glm-glm-be1q4t-0330 (fit-viz 55f2358): T39 on CPUs 0-38, every other task pinned to CPU 39: 123.2 ms/token median, 0 tokens over 300 ms under the reaper and /proc walkers; T40 clean (q4i, 0 ssh logins) 125.5 ms, so the reserve costs nothing'),
    },
    'storage': {
        'label': 'storage tier per tensor class',
        'rule': 'tensors kept on file and read every token (V4.1 engram tables, -dple / split shards 8-9) must sit on NVMe or in memory (DRAM, PMem), never on a rotational disk',
        'allowed': ['nvme', 'sata-ssd', 'memory'], 'forbidden': ['hdd'],
        'evidence': n(3.51, M, S + ' 12:47Z + 13:4xZ: B0 with engram shards 8/9 linked to the droopy HDD (/mnt/datadrive1): prefill 3.51 tok/s, decode 3.05 (both -rtr arms); same recipe with the shards on NVMe: prefill 47.8-52.8, decode n4 8.39 (15:52Z)')},
    'vram_peak': {
        'label': 'VRAM peak, not the post-warm-up reading',
        'rule': 'peak = post-warm-up VRAM + transient margin (the first real -ub 4096 prompt grows the compute graph past what init reserved) must stay within the card',
        'margin': {
            'ik_split': {'lo': round(MK(23.04 - 22.54), 3), 'hi': round(MK(23.04 - 21.56), 3), 'candidate': round(1180 / 1024, 2), 'tag': 'MEASURED (bounded)',
                         'src': UNITS + ' (bounds 0.50 / 1.48 in that unit = %.3f / %.3f GiB). Bounds: B1b' % (MK(0.50), MK(1.48)) + ' (3.0 GiB banks, T40) ran timed prompts at -ub 4096 with VRAM 21.56 of 23.04 in use and no OOM -> margin <= 1.48 (lab notes 17:38Z; the run was stopped for node0 memory, not VRAM); B0 nortr survived at 20.79 with 2.24 GiB banks -> <= 2.25 (15:52Z); B1 OOMed on the first timed prompt with 3.99 GiB banks -> post-warm-up ~22.54 (INFERRED +1.75 GiB banks) -> margin > 0.50 (16:32Z). Candidate 1.15 = B0 #1 allocator: the ub4096 graph needed 3,360 MiB vs 2,180 reserved at init (12:44Z)',
                         'pending': 'a run between 21.56 and 22.54 post-warm-up (B1c at 3.0 GiB banks re-reads 21.5x)'},
            'ik_gguf': {'lo': 0.0, 'hi': round(MK(23.04 - 22.4), 3), 'candidate': None, 'tag': 'MEASURED-UNKNOWN',
                        'src': UNITS + ' (0.64 in that unit = %.3f GiB; the E2 kit log reads 22,398 MiB = 21.87 GiB post-warm-up). ' % MK(0.64) + 'one-chain GGUF (no split): E2 survived at 22.4 of 23.04 GiB post-warm-up (lab notes B0 table) -> margin <= 0.64; the split model builds a chain per partition, so its graph grows more (lab notes 12:44Z)'},
            'glmlite': {'lo': 1.0 + 128 / 1024, 'hi': 1.5, 'candidate': None, 'tag': 'MEASURED',
                        'src': 'glm-lite start-up RESERVE: 1.0 GiB OOMed at the first prefill, 128 MiB short (lab notes 10:58Z); 1.5 ran (11:12Z, 11:50Z). The engine sizes its slot pool after this reserve, so it is inside kv_gib'},
            'sycl': {'lo': 0.0, 'hi': 2.25, 'candidate': None, 'tag': 'ESTIMATE', 'src': 'ESTIMATE: ik split bounds borrowed; nothing measured on the B70'},
            'strata': {'lo': 0.0, 'hi': None, 'candidate': None, 'self_sizing': True, 'tag': 'MEASURED-UNKNOWN',
                       'src': 'Strata sizes its expert cache itself (--expert-cache auto) from measured free VRAM after reserving dense, MTP, KV and its own prompt buffers, so post-warm-up VRAM is full by design and a post + margin sum says nothing. The check needs the measured peak under a long prompt (vram.peak_gib) against the card total',
                       'pending': 'the snoopy qfn-e2e.sh run: bench.py --prefill-first samples the peak under the 12.7k-token prompt'}},
        # parts of the VRAM a PREDICTED post-warm-up must add for one model|engine (planner reserves them on the first GPU):
        # the prefill memory pool (ggml CUDA VMM pool: per-op temporaries of the prefill matmuls, grown at the first prompt and
        # kept), the MTP draft context's compute buffer (scales with its micro-batch, LLAMA_MTP_UBATCH, default = the target -ub),
        # the speculative checkpoints + unlogged CUDA allocations of an MTP run, and the extra compute graph of GPU prefill offload
        'runtime': {
            'glm|ik': {
                'pool_gib_per_tok': dict(n(round(456 / 1024 / 2048, 7), M, S + ' 10-18 12:4xZ BE1-Q3e (srv/glm-be1q3e-hot6p9-t40-ub2k, -ub 2048): nvidia-smi 21,744 MiB after the load -> 22,200 MiB after the first prompts (+456 MiB = the VMM pool kept by ggml_cuda_pool_vmm); linear in -ub (INFERRED). Bound: BE1-Q3d at -ub 4096 crashed at its first prefill in ggml_cuda_pool_vmm::alloc <- ggml_cuda_op_mul_mat (cuMemCreate OOM) with 574 MiB free -> pool(4096) > 0.56 GiB; the line gives 0.89'), lo=round(574 / 1024 / 4096, 7)),
                'mtp_compute_gib_per_tok': n(round(2484 / 1024 / 4096, 8), M, S + ' 10-18: MTP draft context CUDA0 compute buffer 2,484 MiB at its inherited -ub 4096 (BE1-Q3c 09:54Z: cudaMalloc failed) and 77.62 MiB at LLAMA_MTP_UBATCH=128 (BE1-Q3d 11:52Z, BE1-Q3e 12:46Z): exactly linear (2484 / 32 = 77.6). Recipe field mtp_ubatch (ik unpublished build); unset = the recipe ub'),
                'draft_fixed_gib': n(round((21744 - 21115) / 1024 - 77.62 / 1024, 3), M, S + ' 10-18 12:46Z BE1-Q3e: 21,744 MiB used after the load vs 21,115 MiB (20.62 GiB) predicted by the parts above minus the MTP compute buffer (77.62 MiB) = 0.538 GiB with MTP on: speculative per-step checkpoint buffer 446.25 MiB (max_tokens 4 = n_max 3 + 1; BE1-Q3c/d fell back to a 145.56 MiB shadow when it did not fit) + conv-state shadow 9.56 MiB + ~95 MiB unlogged (CUDA context, cuBLAS). Draft off: unmeasured, not added'),
                # context length (recipe field ctx, default 4096 = every measured GLM load): KV grows per context token (MEASURED rate);
                # the prefill graph's context-sized buffers are an ESTIMATE until a long-context load prints its compute buffer
                'ctx_base': 4096,
                'kv_gib_per_ctx_tok': n(round((525.00 + 30.00) / 1024 / 30720, 10), M, S + ' 10-18 two points per cache: main CUDA0 KV 215.56 MiB at n_ctx 4096 (BE1-Q3e, a2e) and 740.56 MiB at 34816 (a2f 14:31Z) -> 145.56 + 0.017090 MiB per context token; MTP context KV 4.00 -> 34.00 MiB (+0.000977 per token). Beyond 4096: 555 MiB at 34816. Was 219.56 / 4096 through 0 (1,647 MiB at 34816): the 145.56 MiB part does not grow'),
                'compute_gib_per_ctx_tok': n(0.0, M, S + ' 10-18 a2f 14:31Z: main CUDA0 compute buffer 3,032.01 MiB at n_ctx 34816 (offload 32, -ub 2048, MTP experts on the host), below the 4,102.01 MiB a2e printed at n_ctx 4096 (offload 32, -ub 2048, MTP experts on the GPU); the MTP context buffer stays 77.62 MiB at micro-batch 128. No context-sized prefill buffer shows (the MLA path does not decompress K/V for the whole context at reserve). Was ESTIMATE 64 KiB per context token (+1,920 MiB at 34816)'),
                # GPU prefill offload: the main context's compute buffer is F + s x ub with offload on, against b x ub off; the planner
                # adds the difference (F + (s - b) x ub, >= 0) beside the upload buffers (prefill.vram_gib)
                'offload_compute_fixed_gib': n(round(3654.0 / 1024, 4), M, S + ' 10-18 two points, ik mtpub-unpublished / h2dstage-unpublished, -cuda offload-batch-size 32: a2b CUDA0 compute buffer 4,550.02 MiB at -ub 4096 (09:23Z, cudaMalloc failed), a2e 4,102.01 MiB at -ub 2048 (13:35Z, then a CUDA pool OOM at the first prefill with ~584 MiB free) -> 3654.0 + 0.21876 MiB x ub. Fixed part = the GPU copy of one layer\'s CPU-resident experts (INFERRED: GLM routed experts ~2.9 GiB per layer). Was modelled as linear through 0 (+2,066 MiB at 4096 -> 1,033 at 2048): a2e read 1,827 MiB more'),
                'offload_compute_fixed_gib_mtp_cpu': n(round((3032.01 - 1242.0 + 794.0) / 1024, 4), M, S + ' 10-18 a2f 14:31Z (recipe mtp_experts cpu, offload 32, -ub 2048): CUDA0 compute buffer 3,032.01 MiB = off buffer 1,242 + F + (s - b) x 2048 with the same slope -> F = 2,584 MiB, 1,070 MiB below the MTP-on-GPU fit (one point; INFERRED cause: with block 45 experts on the GPU the prefill graph runs the MTP layer FFN on the GPU for every prompt token, which ik main 8d21687c (#2678) skips; the rebased build prints the buffer again)'),
                'offload_compute_gib_per_tok': n(round((0.218755 - 2484 / 4096) / 1024, 9), M, S + ' 10-18: slope with offload on 0.21876 MiB per ub token (a2b, a2e) minus the offload-off compute 0.60645 MiB per token (2,484 at 4096, 1,242 at 2048: BE1-Q3c/d/e) = -0.38769 MiB per token over the off buffer; fitted on ub 2048-4096'),
            }},
        'evidence': n(3.99, M, S + ' 16:32Z: B1 3.99 GiB banks OOM on the first timed prompt; warm-up had read 20.79 of 23.04 GiB with 2.24 GiB banks (15:52Z). B0 #1 (4.39 GiB banks) OOMed on prompt 2 (12:44Z)')},
    'numa_free': {
        'label': 'DRAM headroom per NUMA node',
        'rule': 'predicted node MemFree = node memory free at plan time - planned experts on the node - load extra (interleaved non-expert share + CUDA host compute/staging buffers + page cache of on-file tensors) must stay >= min_free',
        'min_free_gib': n(3.0, E, 'ESTIMATE (coordinator 10-17): below ~3 GiB MemFree kswapd starts reclaiming and prefill collapses'),
        'load_extra_scope': 'node 0 only: the CUDA host buffers, the main thread and first-touch non-expert pages land on node 0 (B1c steady state: +12.6 GiB on N0 over N1). Every other DRAM node is planned at load extra other_nodes',
        'other_nodes': n(0.0, M, 'resident minus planned experts on node 1, all 7 droopy runs with a per-node plan: B0c-rtr -0.88, B0c-nortr -1.07, B1c -5.33, B1d -2.10, d1g q2d-p0 -9.65, d1g place12-p0 -6.15, E2 -5.83 GiB (ledger runs/runs.jsonl resident_gib_by_node); 0 is the bound above all of them'),
        'load_extra': {
            # load style: '|chain' = one-chain GGUF (no split; llama_expert_numa_apply interleave); split loads use the banks keys, then the bare key
            'droopy|v41|ik|chain': dict(n(4.54, M, 'one-chain GGUF loads that ran fine (ledger, resident_gib_by_node minus the per-node plan of 78.5 GiB experts): d1g q2d-p0 N0 83.04 -> +4.54, d1g place12-p0 N0 81.80 -> +3.30 (lab notes 06:55Z). Process memory beyond the expert bytes, all nodes: 3.85 / 3.02 GiB. E2 (N2 took 17 GiB over its plan) and d1g q2d-fb (no placement) carry no per-node figure. The value is the high end of the band; lo is the low end'), lo=3.30),
            'droopy|v41|ik': n(9.5, M, S + ' 17:38Z B1b: dram-fill headroom 10 GiB on node0 -> MemFree 0.5 GiB (kswapd active, prefill 10-12 vs ~50, decode 4.9-8.8); node0 resident 85.2 GiB vs 76 planned experts (B0 nortr at the same headroom: 82.0 resident, +6.0, ran clean; B1b +9.2 process: interleaved non-experts + CUDA host buffers incl. pinned staging GGML_CUDA_PIN_MAX_MIB=4096) + page cache of the on-file engram. Measured on node0 only; node1 assumed the same (ESTIMATE)'),
            'droopy|glm|ik|chain': dict(n(8.54, E, 'ESTIMATE: V4.1 one-chain node0 +4.54 (MEASURED, d1g q2d-p0) + 4.0 GiB pinned H2D staging ring (GGML_CUDA_PIN_MAX_MIB=4096, h2dstage-unpublished) if it lands on node 0. lo = the GLM D2 process total beyond expert bytes over all nodes: resident 48.17 + 54.33 + 77.9 = 180.40 GiB vs 176.82 GiB experts incl. MTP blk.45 (UD-Q4_K_XL header) = +3.58 (MEASURED, D2 mtp3, interleave 1:1:1 so no per-node split)'), lo=3.58),
            'droopy|v41|ik|banks<=2.25': dict(n(10.55, M, S + ' 03:20Z-04:21Z 10-18 b1017 (split, 2.2 GiB banks, dram-fill headroom 10, swap off): node0 MemFree 0.59 GiB (b1017sw 2 s telemetry, 595 MiB) + 1.14 GiB of runtime pages that found no room and fell back to PMem node 2 -> node 0 needed 10 - 0.59 + 1.14 = 10.55 GiB beyond its experts; prefill 8-10 vs 46-51 once healed. lo = 9.41 = the part that fit (10 - 0.59). B0 nortr read +6.0 at the same headroom (15:52Z 10-17) with swap ON, so pages were out: not comparable under the swap-off P0 rule'), lo=9.41),
            'ik': n(9.5, E, 'ESTIMATE: borrowed from droopy V4.1 B1b node0, a split load (no reading on this box/model)'),
            'sycl': n(4.0, E, 'ESTIMATE: host staging + page cache of on-file tables; non-experts live on the B70s'),
            'glmlite': n(None, E, 'glm-lite sizes its own RAM tier (mem cap); no per-node reading')},
        'headrooms': [{'gib': 10, 'verdict': 'MEASURED-INSUFFICIENT on node 0 without SPILL_HEAL (B1b 3.0 GiB banks: MemFree 0.5; b1017 2.2 GiB banks, swap off: MemFree 0.59 + 1.14 GiB runtime pages on PMem, prefill 8-10); clean with SPILL_HEAL=1 (b1017heal prefill 46-51)', 'src': S + ' 17:38Z B1b, 10-18 04:21Z b1017sw, 05:49Z b1017heal'},
                      {'gib': 16, 'verdict': 'MEASURED-INSUFFICIENT on node0 (N0 MemFree 0 during prompts) and slower', 'src': S + ' 18:38Z B1c: N0 resident 80.25 (+10 over plan), N1 67.7; decode n4 5.61'},
                      {'gib': '74/89 (N0 free reduced by 12)', 'verdict': 'MEASURED clean, swap off, but slower (PMem share 9.8 %)', 'src': S + ' 19:35Z B1d: no N0 starve, prefill ~50, decode n4 6.16'}],

        # SPILL_HEAL=1 (srv-arm.sh default since 10-18 05:5xZ): after warm-up spill-heal.py moves runtime pages off PMem onto the DRAM
        # node with the most MemFree, so node 0 may run full and the bar applies to DRAM free summed over the DRAM nodes. A per-node
        # floor stays: PMem node 2 is ZONE_MOVABLE only, so kernel (unmovable) allocations need DRAM on some node (b1017pm OOM)
        'spill_heal': {
            'rule': 'recipe numa.spill_heal true: predicted DRAM free summed over the DRAM nodes (node free at plan - experts - load extra) must stay >= min_free_gib; node 0 overflow beyond (headroom - kernel floor) is moved to the DRAM node with the most free, and every node must keep kernel_floor_gib',
            'kernel_floor_gib': n(0.5, M, S + ' 10-18 05:49Z b1017heal: node0 MemFree 0.5 GiB through prefill 46-51 and decode, no reclaim, no OOM. b1017pm 04:57Z: both DRAM nodes under the min watermark (41 / 43 MB) -> global OOM, 245 GB free on PMem'),
            'sum_extra': {
                'droopy|v41|ik|banks<=2.25': dict(n(9.93, M, S + ' 10-18 b1017sw (swap off, no heal): DRAM free at plan 10 + 10 GiB; MemFree N0 0.59 + N1 10.62 = 11.21, plus 1.14 GiB spilled to PMem -> 20 - 11.21 + 1.14 = 9.93 GiB summed load extra. lo 8.2 = b1017heal after the heal (N0 0.5 + N1 11.4, 0.115 left on PMem)'), lo=8.2)},
            'evidence': n(48.6, M, S + ' 10-18 05:49Z b1017heal: prefill 46.2 / 51.0 at node0 MemFree 0.5 after spill-heal moved 4774 MiB N2 -> N1')},
        'node0_bias_gib': n(12.6, M, S + ' 18:18Z B1c steady state: N0 carries ~12.6 GiB more non-expert/host/first-touch anon than N1'),
        'pmem_share_points': {'tag': 'MEASURED, CONFOUNDED', 'note': 'decode n4 vs share of experts on PMem; B1c/B1d also used 3.0 GiB banks (B0 2.2) and B1c ran with swap on, so bank size and PMem share are not separated. Isolating run owed: B0 placement + 3.0 GiB banks + swap off.',
                              'points': [{'pmem_pct': 5.1, 'n4': 8.39, 'banks': 2.2, 'swap': 'on', 'src': S + ' 15:52Z B0 nortr'},
                                         {'pmem_pct': 8.8, 'n4': 5.61, 'banks': 3.0, 'swap': 'on', 'src': S + ' 18:38Z B1c'},
                                         {'pmem_pct': 9.8, 'n4': 6.16, 'banks': 3.0, 'swap': 'off', 'src': S + ' 19:35Z B1d'}]},
        'evidence': n(0.5, M, S + ' 17:38Z: node0 MemFree 0.5 GiB at 10 GiB headroom')},
    # the per-expert placement lands where the plan put it (10-19): a DRAM-first recipe declares that its kit verifies the per-node
    # placement after the load and stops the arm off target (numa.verify "fail", tools/place_verify.py), and that the expert host
    # buffers are pageable (numa.pinned_host false: GGML_CUDA_NO_PINNED=1); undeclared = UNKNOWN, pinned or verify off = FAIL
    'placement': {
        'label': 'per-expert placement verified on landing (pageable host buffers, kit stops off target)',
        'pinned': n('pinned CUDA_Host expert buffers cannot be migrated by the placement', M,
                    'r1 smoke v1 10-19 (lab notes 11:3xZ): rows placement 50 %% off target with pinned per-tensor buffers, numa_maps bind:1 pages on N0, pgmigrate_fail +4.8M pages; with GGML_CUDA_NO_PINNED=1 0 %% off'.replace('%%', '%')),
        'verify': n('without a per-node check a skewed placement is measured as if placed', M,
                    'runs rows droopy-glm-glm-be1q4z-0956 ("0,1" interleave 145 GiB landed N0 30.8 / N1 121 GiB; report 3.26 % off) and droopy-v41-b1017-0042 / b1017heal (N0 +8.5 / N1 -12.4 GiB vs plan), both timed as placed; tools/place_verify.py catches both'),
        'interleave': n('an mbind interleave policy is not split evenly on droopy (q4z, b1017heal, the Qwen il smoke all on node 0); the verify decides, rows:A,B places exact halves', M,
                        'runs rows droopy-glm-glm-be1q4z-0956, droopy-glm-glm-be1r1-1140 (rows: N0 72.9 / N1 85.2 GiB, 0 % off)'),
    },
    'system': {
        'label': 'box P0 settings during a timed run',
        'boxes': {'droopy': {'required': {'numa_balancing': 0, 'demotion': False, 'swap': 'off'},
                             'why': 'numa_balancing 2 + demotion moved ~636 MiB/min DRAM->PMem during decode (lab notes 03:35Z 10-17); swap on: pswpin 31.6M pages since boot, model pages swapped under node pressure (B1c 18:38Z); B1d swapoff: pswpin delta 0 (19:35Z)',
                             # a wrong value of these keys FAILs only when the recipe fills DRAM to a headroom (numa.headroom_gib); without
                             # dram-fill it is UNKNOWN: the harm is measured only under node pressure
                             'dram_fill_only': ['swap'],
                             'dram_fill_only_src': 'runs rows with swap on and no dram-fill headroom ran fine: d1g q2d-p0 / q2d-fb-p0 / place12-p0, d2 glm ctl / mtp1 / mtp3, e2 (7 false alarms of the strict check; e1 went bad from placement short, b0-1 / b0b from the HDD store). With dram-fill (headroom 10-16) swap on: b1b-1 node0 MemFree 0.5 + kswapd, prefill 10-12 vs ~50; b1c MemFree 0 during timed prompts and pswpin 31.6M; b1 oom; b0c-nortr ran fine (1 of 4)'}},
        'evidence': n(31.6, M, S + ' 18:38Z: droopy pswpin 31.6M pages since boot with swappiness 60')},
    'rtr': {
        'label': 'CPU repack (-rtr)',
        'rule': '-rtr on the split MXFP4/q2_K model cuts prefill about 3x and gains no decode; on the one-chain GLM UD-Q4_K_XL rows layout it cuts MTP decode a third: off unless a new measurement says otherwise',
        'chain_decode_mult': n(round(10.039 / 15.03, 3), M, 'runs droopy-glm-glm-be1r2-1354 vs droopy-glm-glm-be1r1-1140 (same package rows-17acecef, placement, prompts): MTP n_max 3 10.04 vs 15.03 tok/s; CPU MoE up/gate per distinct expert 195 vs 129 us (q4_k_r4 path under GGML_MOE_GROUPS chunks)'),
        'chain_prefill_mult': n(round(69.8 / 67.9, 2), M, 'r2 warm-up 2933 tokens 69.8 tok/s vs q4c 67.9 (offload off; not a matched pair: r1 logged no warm-up prefill)'),
        'prefill_mult': n(round(16 / 50.0, 2), M, S + ' 14:58Z vs 15:52Z: B0 -rtr prefill 15-17 tok/s vs -rtr off 47.8-52.8'),
        'decode_mult': n(round(6.91 / 6.83, 2), M, S + ' n0 phases: -rtr 6.91 vs off 6.83 (n4 medians 6.35 vs 8.39)'),
        'kernel_only': n(0.6885 / 0.4645, M, 'K1: mxfp4_r8 0.69 vs mxfp4 0.46 of the read ceiling in an isolated kernel (-29 % expert ms); did not carry to the server'),
        'evidence': n(16, M, S + ' 14:58Z B0 -rtr arm')},
    'quality': {
        'label': 'quality gate: KL vs EXL3-3.05 vs share of routed reads at 2 bits',
        'rule': 'paired decode KL vs EXL3-3.05 (the exact GPU path of the EXL3 3.05 bpw store on snoopy, ref-topk512.json) <= 0.03 and top-1 >= 0.97 (p3-run.log thresholds); KL grows with the share of routed reads served by 2-bit experts. The gate\'s reference is the EXL3 3.05 bpw store, not BF16, so every report calls it "KL vs EXL3-3.05", never plain "KL"',
        'bar_kl': 0.03, 'bar_top1': 0.97,
        # whole store files measured on the same gate (paired decode KL vs the gate reference), keyed model|store type
        'stores': {
            'glm|ud_q3kxl': {'kl': n(0.05029, M, 'a2f 10-18: droopy glm-be1q3a2f kl-gate.json'), 'top1': 0.9492, 'n': 926,
                             'src': 'a2f + q3f 10-18 (droopy, ik mtpub-unpublished): Unsloth UD-Q3_K_XL as shipped vs ref-topk512.json (A = the EXL3 3.05 store exact GPU path, snoopy kl-1328; NOT a BF16 reference), 6 standard prompts, 926 positions: KL 0.0503 / top-1 0.949 (a2f, GPU prefill offload 32). Offload ruled out: on the same first 5 prompts (768 positions) q3f without offload read KL 0.0489 / top-1 0.953 vs a2f 0.0504 / 0.949. Agrees with Unsloth\'s own KLD vs BF16 (0.160 / top-1 86.3 % for UD-Q3_K_XL against 0.049 / 92.2 % for UD-Q4_K_XL)'}},
        'families': {
            'glm': {'engines': ['glmlite'], 'tag': 'FIT',
                    'base_kl': n(0.00446, M, GR + 'k2e-1450-kl.json ref served_A_vs_B1: 3.05 lane on vs exact = 0.00446 (top-1 0.9914); two-run floor 0.00472'),
                    'base_top1': 0.9914,
                    'points': [{'share': round(30 / 42, 3), 'kl': 0.07217, 'top1': 0.9397, 'run': 'snoopy-glm-p3-1503', 'src': GR + 'p3-1503-kl.json served (lane on): P3 middle 30 of 42 layers K=2 -> 0.714 of routed reads at 2.05 bpw', 'tag': M},
                               {'share': 0.304, 'kl': 0.03926, 'top1': 0.9612, 'run': 'snoopy-glm-p4-s25-2047', 'src': GR + 'p4-2001-s25-kl.json served_Aref_vs_Bvar (lane on, n 928; p4-run3.log line 29): P4 S25 per-expert coldest-first mix, store-2.05bpw-k2share-25, engine f20ef3f; share = the planned K=2 share of decode reads (p4-mix-e2e.sh). Lane off: KL 0.04073 / top-1 0.9634 (line 28)', 'tag': M},
                               {'share': 0.192, 'kl': 0.02983, 'top1': 0.9623, 'run': 'snoopy-glm-p4-s15-2047', 'src': GR + 'p4-2001-s15-kl.json served_Aref_vs_Bvar (lane on, n 928; p4-run3.log line 47): P4 S15 per-expert coldest-first mix, store-2.05bpw-k2share-15, engine f20ef3f; share = the planned K=2 share of decode reads (lab notes 16:46Z pack estimate). Lane off: KL 0.03012 / top-1 0.9741 (line 46)', 'tag': M}],
                    'pending': [],
                    'note': 'linear in share through (0, base), least squares over the measured points. P4 S25 (per expert, coldest first) cost MORE KL per 2-bit read than P3 (whole layers): (0.0393 - 0.0045) / 0.304 = 0.114 vs (0.0722 - 0.0045) / 0.714 = 0.095 per unit share; P4 S15 (0.192 -> 0.0298) sits above the line too. The layer map (data/layer-maps.json, src/model.js layerMap) explains the gap: P3 cut the middle layers 8-37 whole, S15/S25 spread the coldest experts over every layer including the sensitive early band, so the gate reads the map-weighted share when the map calibrates'},
            'fnx': {'engines': [], 'tag': 'UNKNOWN', 'unmeasured': True,
                    'note': 'Flash-Next: no KL vs 2-bit share and no agent eval yet (scout measurements 3-4); quality ~GLM-5.3-Flash on public boards only'},
            'v41': {'engines': ['ik'], 'tag': 'FIT',
                    'base_kl': n(0.0, M, 'v41kl ref-r1.json floor_gen_vs_forced: the reference (1017 Q8_0, MXFP4 experts as released) generation path vs the forced scoring path on the same server, 320 positions, KL 0.0 / top-1 1.0 (lassie CPU build unpublished build, slot erased per prompt). External check of this reference (10-18 19:45Z, MEASURED, v41kl res/unsloth-u2.json sha 4b73f058): unsloth DeepSeek-V4.1-Flash-1017 UD-Q4_K_XL (experts = the release MXFP4) served by unslothai/llama.cpp 4a314d3a (CPU, FA off, no repack, mmap, swa-full) vs ref-r1.json over the same 1376 positions: KL 0.00276 (median 1.9e-6, p99 0.053, max 0.224), top-1 0.984; std 0.00289 / 0.982, code 0.00247 / 0.988. Below the 6.1e-3 / 0.977 that unsloth measured against the DeepSeek official reference, so the ik V4.1 forward and this reference are confirmed; ~0.003 is the cross-engine floor under the store points'),
                    'base_top1': 1.0,
                    'points': [{'share': 0.225, 'kl': 0.01689, 'top1': 0.9746, 'tag': M, 'run': 'droopy-v41-b1017-kl-lassie',
                                'src': 'v41kl var-s1017.json (lassie 10-18 11:22Z, 1376 positions, 6 standard + 4 coding-agent prompts x <= 160 greedy tokens): the 1017 split store (hot experts MXFP4 kept, cold q2_K / q3_K down 0-9) vs the 1017 Q8_0 reference, same CPU build unpublished build; share = the b1017 recipe lowbit_share (22.5 % of routed reads at q2_K). p99 0.272, max 4.69 (a tool-call opening in coding-agent prompt 4: ref ｜DSML｜ 0.999, store Read 0.974); groups std 0.0163 / 0.9737, code 0.0183 / 0.9766; rerun bit-identical'},
                               {'share': 0.65, 'kl': 0.04036, 'top1': 0.9535, 'tag': M, 'run': 'droopy-v41-q3q2-kl-lassie',
                                'src': 'v41kl var-q3q2.json (lassie 10-18 15:12Z, the same 1376 positions and reference ref-r1.json, same CPU build unpublished build): the 1017 NON-split requant, routed experts q3_K on layers 0-13 and q2_K on 14-39 (PLAN step 5 requant A; ik llama-quantize unpublished build, imatrix-1017-fixed 6a440a1c), non-experts as the source; share = the planner layer_types share 0.650. p99 0.639, max 3.23; groups std 0.0478 / 0.9442, code 0.0238 / 0.9742 (the coding prompts alone clear the bar); the 1-point line had predicted KL 0.0488 / top-1 0.927 here'}],
                    'note': 'V4.1 line through the measured reference floor (0) and TWO store points on one reference and one engine: the 1017 split (22.5 % of reads at q2_K) and the non-split q3_K L0-13 / q2_K requant (65 %). Both points sit near the line (the 65 % point slightly below it: KL 0.0404 measured vs 0.0488 predicted from the first point); the split point keeps MXFP4 hot experts, the requant has none, so share alone carries the fit'}}},
}
checks['quality']['families'].update(FAMILY_QUALITY)
# every family with measured points gets its own least-squares line through its base (share 0) and a max 2-bit share
for _g in checks['quality']['families'].values():
    _pts = _g.get('points') or []
    if not _pts or 'base_kl' not in _g:
        continue
    _sxx = sum(p_['share'] ** 2 for p_ in _pts)
    _ptxt = ', '.join('(%.3f, %.4f / %.4f)' % (p_['share'], p_['kl'], p_['top1']) for p_ in _pts)
    _g['slope_kl'] = n(round(sum(p_['share'] * (p_['kl'] - _g['base_kl']['v']) for p_ in _pts) / _sxx, 4), FIT,
                       'least-squares line through base %.5g at share 0 and %d measured points (share, KL / top-1): %s' % (_g['base_kl']['v'], len(_pts), _ptxt))
    _g['slope_top1'] = n(round(sum(p_['share'] * (_g['base_top1'] - p_['top1']) for p_ in _pts) / _sxx, 4), FIT,
                         'least-squares line through base top-1 %.4f and the same %d points' % (_g['base_top1'], len(_pts)))
    _mk = (checks['quality']['bar_kl'] - _g['base_kl']['v']) / _g['slope_kl']['v'] if _g['slope_kl']['v'] > 0 else float('inf')
    _mt = (_g['base_top1'] - checks['quality']['bar_top1']) / _g['slope_top1']['v'] if _g['slope_top1']['v'] > 0 else float('inf')
    _g['max_share'] = n(round(min(_mk, _mt, 1.0), 3), FIT, 'largest 2-bit share of routed reads inside both bars on the fitted lines (%s binds: KL %.3f, top-1 %.3f)' % ('KL' if _mk <= _mt else 'top-1', _mk, _mt))

# measured runs and levers: generated from the run ledger (runs/runs.jsonl, tools/record_run.py) and runs/levers.json
import sys
sys.path.insert(0, SRC)
import runs_tables  # noqa: E402
RUNS = runs_tables.load_runs(os.path.join(HERE, 'runs', 'runs.jsonl'))
_holdout = {rid for spec in json.load(open(os.path.join(HERE, 'data', 'decode-fits.json'))).values() for rid in spec['holdout']}
recipes = runs_tables.recipes_from_runs(RUNS, _holdout)
decode_fits = runs_tables.decode_fits(HERE, RUNS, recipes)
draft_twins = runs_tables.draft_twins(RUNS, _holdout)
levers = runs_tables.levers_from_runs(json.load(open(os.path.join(HERE, 'runs', 'levers.json'))), RUNS, M)
# the planner's glm-lite rates (host ms, lane GB/s) were measured on this build only; other builds are UNMODELLED in predictions
engines['glmlite']['calibrated_builds'] = ['p512-lane']
# does the engine's decode run expert work on CPU threads (the cpu_reserve check)? ggml/ik: OpenMP-style worker threads that
# spin on ggml_barrier (q4d perf: ggml_barrier 77 % of CPU samples in a slow decode). None = UNKNOWN for that engine
for _e, _c in {'ik': True, 'llamacpp': True, 'sycl': True, 'glmlite': True, 'strata': None}.items():
    engines[_e]['cpu_decode'] = _c
engines['glmlite']['builds'] = {'p512-lane': K2E, 'avx2-lane': 'AVX2 lane kernel before 14:40Z (compute-bound 0.228 ms/expert, excl0-1143.stats.json): rates not in the planner'}

assume = {
    'gpu_eta': n(0.6, E, 'ESTIMATE: share of nameplate VRAM bandwidth reached by decode matvecs'),
    'quality_a': n(round(qa, 3), QTAG, QSRC), 'quality_b': n(round(qb, 4), QTAG, QSRC), 'quality_pairs': QPAIRS,
    'cliff_bpw': n(2.6, P, 'quant-recipes rule 3: below about 2.6 bits average on the experts, coding and agent quality fall off a cliff'),
    'host_ref_box': 'droopy',
}

knobs = {'freeDesktop': False, 'moveB70': 'none', 'addDRAM': False, 'addOptane': False, 'optaneBwScale': None, 'kernel2x': False,
         'draft': True, 'draftMult': None, 'smt': False, 'densePrec': 'q8_0', 'hostCopy': True, 'allowCodeTypes': False, 'perExpertTypes': False,
         'rtr': False, 'ub': 'auto', 'vramMargin': 'safe', 'freeHousehold': False, 'numaHeadroom': 10}

# serving conditions a run must declare (recipe.conditions; a top-level ctx counts): each one moves decode tok/s, quality or
# output length, so two runs that differ in an undeclared one do not compare (owner 2026-10-18: "make sure our test recipes
# are taking all those potential settings into account when thinking about design and result validity")
conditions = {
    'fields': {
        'ctx': 'context window: KV size and the attention cost per token scale with it',
        'kv_type': 'KV cache type (f16 / q8_0 / q4_0): VRAM fit, attention bytes per token, long-context quality',
        'draft_n': 'drafts per step: the verify width (droopy GLM one pass at depth 600: K4 = 2.0x the K1 pass, K8 = 3.0x, K8/K4 1.50 with CPU MoE time tracking distinct experts, runs row droopy-glm-glm-be1q4y-0909; the older K8 = 4.6-4.9x of droopy-glm-glm-be1q4k-2146 was ik\'s prompt checkpoint split at ctx_checkpoints_tolerance 5, not verify cost)',
        'sampling': 'temperature / top_p / top_k / min_p: draft acceptance falls as temperature rises, so a greedy bench overstates served speed',
        'reasoning_effort': 'thinking level and who set it (server default, request, client passthrough): it sets output length, so wall time and agent pass rates',
        'workload': 'what the timed tokens were (prompt kind, cached depth, prompt and generation lengths): acceptance and expert heat follow the text',
    },
    'draft_only': ['draft_n'],
}


# causes of runs that went bad (runs/labels.json labels runs with them, from each row's notes and logs); decided_by = the
# pre-launch checks that can refuse the cause before launch ([] = not decidable before launch)
causes = {
    'interference': {'label': 'another process took CPU time during timed decode', 'decided_by': ['cpu_reserve'],
                     'src': 'runs rows q4l/q4i/q4j/q4t (droopy, 10-18/19): T40 stalls under any co-runner, T39 + a pinned reserve does not'},
    'unexplained_slowdown': {'label': 'timed prompts read off-speed and the notes name no cause', 'decided_by': [],
                             'src': 'runs row droopy-glm-glm-be1q4c-1831 (2 of 15 prompts at half speed, cause UNKNOWN)'},
    # crash taxonomy (2026-10-19, from the crash rows' server logs and lab notes); decided_by = the check whose
    # MEASURED term now covers the cause. Those terms were measured from these very crashes, so their FAIL on these rows is
    # hindsight: the report prints the launch verdict beside today's
    'vram_load': {'label': 'a load-time GPU buffer the plan did not count (MTP experts or banks allocated twice, a compute buffer at -ub 4096)', 'decided_by': ['vram_peak'],
                  'src': 'checks.vram_peak.runtime glm|ik terms (MEASURED from be1q3-0644, be1q3b, a2b, be1q3c server logs)'},
    'vram_prefill': {'label': 'the first prefill or warm-up allocates past the free VRAM (CUDA pool cuMemCreate, level_zero OUT_OF_DEVICE_MEMORY)', 'decided_by': ['vram_peak'],
                     'src': 'checks.vram_peak.runtime prefill pool 456 MiB per 2048 ub (lab notes 13:00Z) and the odie b1n buffer lines'},
    'host_pinned_dram': {'label': 'pinned host buffers (SYCL) exhaust DRAM: global OOM or a box livelock', 'decided_by': ['mem_cap'],
                         'src': 'boxes.odie mem_cap host_floor_gib (b1n2 07:12:32Z global OOM report)'},
    'dram_node_pressure': {'label': 'a DRAM node at its floor during load or prompts: spill to PMem, kswapd, global OOM', 'decided_by': ['numa_free', 'system'],
                           'src': 'checks.numa_free load_extra / spill_heal (b1b, b1017 family); checks.system swap under dram-fill (b1c pswpin 31.6M, 18:38Z)'},
    'long_prompt_vram': {'label': 'a long-prompt prefill (16k) allocates past the free VRAM after the timed phases ran', 'decided_by': [],
                         'src': 'runs row droopy-glm-glm-be1q4c-1831: CUDA OOM in cudaGraphLaunch at depth 14,336-16,384 while vram_peak PASSed (21.86-22.x of 22.50): no context-length graph term is measured'},
    'vram_other_process': {'label': 'another program\'s VRAM grew after the engine sized its pool', 'decided_by': [],
                           'src': 'runs row snoopy-glm-arma-0719: glm-lite sizes its slot pool to free VRAM minus a 1.5 GiB reserve at start; the desktop (3.9-5.6 GiB) then took more; the prefill chunk-8192 buffer is unmeasured'},
    'placement_short': {'label': 'experts landed off the planned tiers after load', 'decided_by': ['placement'],
                        'src': 'runs row droopy-v41-e1: node2 (PMem) resident 148.0 GiB vs 112 planned; seen only after load'},
    'tool_bug': {'label': 'the measuring harness crashed, not the engine', 'decided_by': [],
                 'src': 'runs rows snoopy-glm-p4-1646 / p4-1741: S25 decode-KL step (k_twin.build) assertion and shape errors'},
}

# the run-ledger accuracy report (report.js --accuracy): the time bucket of its error trend; replay_checks = checks added
# after the launch-time captures, evaluated post hoc on launch rows too (labelled in the report)
accuracy = {
    'replay_checks': {'v': ['cpu_reserve', 'placement'], 'src': 'cpu_reserve added 2026-10-19 after q4t, placement after r1; launch captures before them lack them'},
    'trend_bucket': {'v': 'day', 'src': 'owner brief 2026-10-19 (per time bucket: week or launch batch). The ledger spans 2026-09-29..10-19, so one ISO week holds most rows; a UTC day = one launch batch here. --bucket week overrides'},
}


def sibling(name):
    p = os.path.join(HERE, name)
    if os.path.exists(p):
        try:
            return {'present': True, 'mtime': datetime.datetime.fromtimestamp(os.path.getmtime(p), datetime.timezone.utc).strftime('%Y-%m-%dT%H:%MZ'), 'data': json.load(open(p))}
        except Exception as ex:
            return {'present': True, 'error': str(ex)}
    return {'present': False}


def sibling_md(name):
    p = os.path.join(HERE, name)
    return open(p).read() if os.path.exists(p) else None


# ---------------- prefill: CPU path vs GPU expert offload ----------------
# ik prefill either runs the routed experts on the CPUs, or (ubatch >= -cuda offload-batch-size) copies every CPU-resident
# expert a ubatch touches to the GPU and runs it there. At ub 4096 nearly every expert is active (V4.1: 1 - (1 - 6/256)^1532
# ~ 1), so one ubatch costs one upload of the CPU-resident expert bytes; the GPU compute overlaps the copy (b1017off2: the
# on-phase prompt time matches the upload time), so offload tok/s = P / (ceil(P / ub) x upload_s). It pays when that beats
# the CPU rate: crossover P* = cpu_tps x upload_s inside the first ubatch; above ub every extra ubatch pays the upload again.
SP = S + ' 10-18 06:43Z b1017off2 (srv/b1017off-t40: V4.1-1017 split, h2dstage-unpublished, SPILL_HEAL, banks off, ub 4096)'
prefill = {
    'rule': 'offload tok/s = P / (ceil(P / ub) x upload_s), upload_s = CPU-resident routed expert bytes / h2d rate; CPU tok/s = cpu_tps[box|model|engine|ub] inside its envelope (x checks.rtr.prefill_mult with -rtr on a split model), else UNMODELLED; crossover P* = cpu_tps x upload_s (offload pays above it, up to ub)',
    'h2d_gbs': {
        'ring': dict(n(6.67, M, SP + ': stager 477.18 GiB in 12592 copies at 6.67 GB/s incl. DMA waits (dmon rx 6.73 GB/s over 60 busy s); microbench ceiling 7.63-7.67 (04:2xZ)'), hi=7.65),
        'driver': dict(n(2.0, M, S + ' 10-18 04:2xZ microbench: pageable -> device through the driver 2.30 GB/s (interleave), 1.76 (membind 1); P17 implied 1.6-1.8'), lo=1.7, hi=2.3)},
    'ring_builds': ['h2dstage-unpublished'],
    # CPU path: one MEASURED rate per box|model|engine|ub cell, trusted only inside its measured envelope (the PMem expert
    # bytes of the runs that read it within its band). Any other -ub, PMem share, or -rtr outside the measured split pair is
    # UNMODELLED (prefill_tps null): the ledger holds no second -ub per model, and its two -ub 512 V4.1 points (d1g q2d 23.2,
    # place12 38.1) sit at 112 GiB PMem and disagree by 64 % with each other, so neither a -ub nor a PMem term can be fitted
    'cpu_tps': {
        'droopy|v41|ik|4096': dict(n(57.3, M, SP + ' offload-off phases 53.0 / 59.2 (off1) and 57.7 / 64.8 (off2), banks off; b1017heal with 2.2 GiB banks 46.2 / 51.0 (05:49Z)'), lo=46.2, hi=64.8,
                                   # every ledger run held banks (2.19-2.95 GiB) and read 48.6-51.1, 12 % under the banks-off cell (b1017heal vs
                                   # b1017off2: same package, same day, 48.6 vs 57.3); the banks-on rate is fitted on three rows, b1d held out
                                   banks_on=dict(n(50.5, FIT, 'median of runs rows b0c-nortr 50.5, b1c 50.5, b1017heal 48.6 (banks 2.19-2.95 GiB, -ub 4096, no -rtr); held out b1d 51.1 (banks 2.95): predicted 50.5, -1.2 %. One banks-off reading (b1017off2), so no per-GiB banks term is fitted'), banks_gib=[2.19, 2.95], holdout={'id': 'droopy-v41-b1d', 'meas': 51.1}),
                                   envelope={'pmem_gib': [62.0, 70.3], 'src': 'the planner PMem expert GiB of the runs rows that read inside the band: b0c-nortr 50.5 (62.0), b1c 50.5 (68.3), b1d 51.1 (70.3), b1017heal 48.6 (62.0); outside it E2 read 34.5 at 112.0 and the -ub 512 rows 23.2 / 38.1 at 112.0'}),
        'droopy|glm|ik|2048': dict(n(73.4, M, S + ' 10-18 12:4x-13:03Z BE1-Q3e (srv/glm-be1q3e-hot6p9-t40-ub2k: UD-Q3_K_XL all-DRAM N0/N1, 6.9 GiB banks, T40, offload off, -ub 2048): 15 prompts of 1539 / 2146 / 1944 tokens, mean 73.4, per phase m3 71.5/75.0/77.4, m1 71.1/75.2/81.6, m2 71.6/59.1/81.7, m0 71.7/73.5/62.6, m3b 71.5/75.5/81.4'), lo=59.1, hi=81.7,
                                   envelope={'pmem_gib': [0, 16.3], 'src': 'the planner PMem expert GiB of the runs rows that read inside the band: q3e 73.4 / q3f 74.1 / q3m 73.0 (0), q4c 67.9 (13.2), q4d 67.7 (16.3). Every GLM -ub 4096 arm (Q3, Q3b, Q3c, Q3d, a2b) died before its first prefill: -ub 4096 is unmeasured'})},
    'calibration': {'points': [{'tokens': 1532, 'on_tps': 46.6, 'off_tps': [53.0, 57.7]}, {'tokens': 2210, 'on_tps': 62.2, 'off_tps': [59.2, 64.8]}],
                    'upload_gib': 220, 'src': SP + ': ~220 GiB staged per ubatch = ~33-36 s fixed per prompt; ON at par or worse for 1532 / 2210-token prompts'},
    'vram_gib': n(0.84, M, SP + ': vram_end 19496 vs after warm-up 18634 MiB with offload ON in between (+862 MiB for the upload buffers). GLM expert tensors are smaller than V4.1 ones, so for GLM an upper bound (INFERRED)'),
    'prompts': [512, 1024, 1532, 2048, 2210, 4096, 8192, 16384],
}

# ---------------- external evidence: community measurements on hardware we do not own (REA: reverse-engineer good outputs) ----
# runs/external.jsonl: one row per published measurement (URL, date, hardware, engine + version, quant, settings, measured
# tok/s). Every row is MEASURED-BY-THEM (inferred): a validation set for the physics, never a fitted term. report.js --external
# runs our physics on each row's hardware spec and prints the gap; a prediction on a box of the same class carries that gap.
EXT_SRC = 'runs/external.jsonl'
external = {
    'file': EXT_SRC,
    'tag': 'MEASURED-BY-THEM',
    'classes': {'apple': 'Apple Silicon (unified memory, Metal / MLX)', 'amd': 'AMD GPU or APU (ROCm / Vulkan; Strix Halo unified)',
                'nvidia_consumer': 'Nvidia GeForce / workstation cards (CUDA), incl. hybrid CPU-expert runs',
                'cpu_only': 'no GPU in the run', 'intel_arc': 'Intel Arc / Arc Pro (SYCL / Vulkan / XPU)'},
    # our boxes in the same classes (a prediction there carries the class gap)
    'box_class': {'v': {'droopy': 'nvidia_consumer', 'snoopy': 'nvidia_consumer', 'lassie': 'nvidia_consumer', 'odie': 'intel_arc'},
                  'src': 'boxes table: droopy Quadro RTX 6000, snoopy RTX 3090 Ti, lassie GTX 1080 Ti (CUDA); odie Arc Pro B70 (SYCL)'},
    # the host overhead an external row borrows: the only GPU-resident host term we hold, or the reference box's fitted ik host
    'host_borrow': {'v': {'gpu': ['fnx', 'sycl'], 'cpu': ['v41', 'ik']},
                    'src': 'no host term exists for hardware we do not own: GPU-resident rows borrow models.fnx.host_engine.sycl, rows with CPU-side weights borrow assume.host_ref_box\'s fitted v41|ik host'},
    # DRAM: an external row states a spec (channels x MT/s x 8); our boxes read their measured rate. The spec -> read ratio is the
    # median of our boxes that hold both (dram.bw MEASURED / dram.bw_peak_gbs), computed at run time
    'dram_eta_from': {'v': 'median of boxes[*].dram.bw / dram.bw_peak_gbs', 'src': 'snoopy 58.5 / 83.2, odie 70.1 / 102.4, lassie 257.7 / 307.2 (probe + lab notes)'},
    # CPU box factor: the median of our fitted f (FIT rows only), computed at run time
    'cpu_f_from': {'v': 'median of FIT cal.boxes[*].f', 'src': 'calibrate(): droopy and snoopy fit near 1, lassie (8 NUMA nodes, Zen 1) 0.24'},
    # NUMA: a row's CPU side reads one node's share of its DRAM spec x the measured placement gain (bench.numa), not the sum
    'sockets_default': {'v': 1, 'src': 'a row that does not state its sockets is a one-socket desktop or workstation'},
    'nodes_per_socket_default': {'v': 1, 'src': 'EPYC Rome and later, and Xeon with SNC off, default to one NUMA node per socket (NPS1); a row may state nodes_per_socket'},
    'numa_policy_default': {'v': 'interleave', 'src': 'a multi-socket row that states no policy: llama.cpp reads the weights through the page cache, which spreads them over the nodes; first touch over one node per socket is not measured, interleave is (lassie interleave m2 c2 s2)'},
    # published quant label (longest prefix, upper case) -> our type, for bits per weight and the CPU kernel efficiency
    'quant_types': {'v': {'Q4_0': 'q4_0', 'Q4_1': 'q4_0', 'Q4_K': 'q4_K', 'UD-Q4_K_XL': 'ud_q4kxl', 'UD-Q4_K_M': 'q4_K', 'UD-Q8_K_XL': 'q8_0', 'Q5_K': 'q5_K', 'Q6_K': 'q6_K', 'Q8_0': 'q8_0',
                              'MXFP4': 'mxfp4', 'Q3_K': 'q3_K', 'UD-Q3_K_XL': 'ud_q3kxl', 'Q2_K': 'q2_K', 'UD-Q2_K_XL': 'q2_K', 'IQ4_XS': 'iq4_xs',
                              'IQ4_NL': 'iq4_nl', 'IQ4_KS': 'iq4_ks', 'IQ3_KS': 'iq3_ks', 'IQ2_KS': 'iq2_ks', 'IQ2_KL': 'iq2_kl', 'IQ2_XXS': 'iq2xxs_q2k',
                              'UD-IQ2_XXS': 'iq2xxs_q2k', 'IQ3_XXS': 'gsq_iq3xxs', 'UD-IQ3_XXS': 'gsq_iq3xxs', '4BIT': 'q4_0', 'Q4': 'q4_0', 'INT4': 'q4_0'},
                    'src': 'llama.cpp / ik_llama.cpp quant names; MLX 4-bit (group 64, ~4.5 bpw) and INT4 map to q4_0 for the CPU kernel only'},
}


# ---------------- bench terms: the GPU line, the per-token MoE overhead and the NUMA socket term ----------------
# Fitted from the bench ledger (runs/bench.jsonl, written by tools/record_bench.py from tools/bench_gpu.sh and
# tools/bench_numa.sh sweeps), our own boxes only. The external rows (runs/external.jsonl) re-score these terms; they
# never set one.
BENCH_SRC = 'runs/bench.jsonl'
bench = {
    'file': BENCH_SRC,
    # the backend label an external row states -> our backend key
    'backend_of': {'v': {'CUDA': 'cuda', 'SYCL': 'sycl', 'Vulkan': 'vulkan', 'Metal': 'metal', 'ROCm': 'rocm'}, 'src': 'llama.cpp backend names as the rows state them'},
    # a backend with no box of ours borrows a measured one (the borrowed terms are ESTIMATE, so inferred)
    'borrow': {'v': {'metal': 'cuda', 'rocm': 'cuda'},
               'src': 'no Apple or AMD GPU box: Metal and ROCm borrow the CUDA line and MoE term (llama.cpp runs one graph per backend; the reach of each card is unmeasured)'},
    # a line is a driver stack (backend x card vendor): it predicts rows on its own vendor's cards, or the vendor a borrow names
    'vendor_of_class': {'v': {'apple': 'apple', 'amd': 'amd', 'intel_arc': 'intel', 'nvidia_consumer': 'nvidia'}, 'src': 'the card vendor of each external hardware class'},
    'borrow_vendor': {'v': {'metal': 'apple', 'rocm': 'amd'}, 'src': 'the vendor a borrowed backend stands in for'},
    'line_resid_max_ms': {'v': 1.0, 'src': 'a dense row off its line by more than this is a quant kernel, not bytes: the line is MEASURED, CONFOUNDED (odie Vulkan: Q8_0 and Q4_K_M read at 150-165 GB/s, Q4_0 at 190-200 GB/s)'},
    'min_dense': {'v': 2, 'src': 'a line needs dense models of two sizes on one card'},
    'numa_spread_max': {'v': 0.25, 'src': 'two rounds of one arm that differ by more than this share of their median are MEASURED, CONFOUNDED (lassie X8 arms: threads on node 0 share cores with the system)'},
    'moe_spread_max': {'v': 0.5, 'src': 'per-layer MoE terms of one backend that differ by more than this share of their median are MEASURED, CONFOUNDED: the MoE rows carry different expert quants (gpt-oss MXFP4, Qwen3-30B-A3B Q4_K_S), so the spread is the quant kernel, not the MoE path'},
    # a line is one driver stack: rows carry their run conditions (record_bench.py --cond); rows recorded before that carry stack_of
    'stack_of': {'v': {'vulkan': {'stack': 'mesa-26.0.8', 'mmvq': 'on', 'fa': 'auto'}},
                 'src': 'conditions of rows recorded before row.conditions existed: the odie Vulkan sweeps 0915/0929 ran ghcr.io/ggml-org/llama.cpp:full-vulkan b11706 (Mesa 26.0.8 in the image, seen), GGML_VK_DISABLE_MMVQ unset, -fa unset (auto)'},
    'line_stack': {'v': {'vulkan': {'stack': 'mesa-26.2.5', 'mmvq': 'on', 'fa': '1'}},
                   'src': 'the conditions each backend line is fitted on; rows of other stacks stay in the ledger and in bench.stacks (one line per stack). '
                          'Vulkan = the B0 arm of the 2026-10-19 Mesa A/B on odie B70 #1: Mesa 26.2.5 (kisak) beat the image Mesa 26.0.8 on every dense model '
                          '(gate: test-backend-ops MUL_MAT 5/5 clean per arm, llama2-7B Q4_0 perplexity bit-identical across arms); -fa 1 as llama-server runs it'},
    'stack_keys': {'v': ['stack', 'mmvq', 'fa'], 'src': 'the conditions that make a stack (driver, kernel switch, flash attention); split mode, device pin and image are recorded, constant within a sweep'},
    'line_engines': {'v': ['llamacpp'],
                     'src': 'SEEN 2026-10-19 (B70 cookbook window, runs/bench-raw/odie-strata-*-1019.jsonl): XeStrata decodes Flash-Next at 80.9-92.7 tok/s on '
                            'odie B70 #1 with an expert cache, CPU expert pool and MTP draft; its speed is not bytes over a card rate, so a strata row '
                            'kept the SYCL line and MoE term from being a llama.cpp graph. Rows of other engines stay in the ledger, off the line'},
    'line_spec': {'v': 'none', 'src': 'SEEN 2026-10-19 (runs/bench-raw/odie-sycl-*-mtverify-1019.jsonl): Qwen3-30B-A3B Q4_K_S on B70 #1 decodes 116.6 tok/s '
                                      'plain and 57.2 with a Qwen3-1.7B draft (n 3): a speculative row (conditions.spec) measures the draft, not the read'},
    'numa_ref': {'v': 'membind, the threads and the weights on one node, the most threads', 'src': 'the gain unit: one NUMA node read locally (lassie N1-bind4, 7 threads on 8 cores)'},
}
_med = lambda xs: sorted(xs)[len(xs) // 2] if len(xs) % 2 else sum(sorted(xs)[len(xs) // 2 - 1:len(xs) // 2 + 1]) / 2


def moe_term(cards, B):
    """the backend's per-layer MoE term: the median card's median MoE row; MoE rows spread past moe_spread_max -> MEASURED, CONFOUNDED"""
    m = _med([c['moe'] for c in cards])
    per = sum((c['moe_all'] for c in cards), [])
    spread = (max(per) - min(per)) / abs(m) if m else 0
    tag = M if spread <= B['moe_spread_max']['v'] else 'MEASURED, CONFOUNDED'
    return n(round(m, 4), tag, 'MEASURED: median per MoE layer of ' + ' | '.join(c['moe_src'] for c in cards) +
             ('' if tag == M else ' (MoE rows spread %.0f %% > %.0f %%: the expert quant kernels differ)' % (100 * spread, 100 * B['moe_spread_max']['v'])))


def row_stack(r, B):
    """the stack conditions of a bench row: its own conditions, else the backend's stack_of (rows recorded before conditions)"""
    c = r.get('conditions') or B['stack_of']['v'].get(r['backend']) or {}
    return {k: str(c[k]) for k in B['stack_keys']['v'] if k in c}


def plain_decode(r, B):
    """a row is a plain decode of resident weights: an engine whose graph the line models, no speculative decoding"""
    return r.get('engine', 'llamacpp') in B['line_engines']['v'] and str((r.get('conditions') or {}).get('spec', 'none')) == B['line_spec']['v']


def on_line(r, B):
    """a row joins its backend line when it is a plain decode and its stack matches line_stack (a backend with no
    line_stack takes every plain row)"""
    want = B['line_stack']['v'].get(r['backend'])
    return plain_decode(r, B) and (not want or all(row_stack(r, B).get(k) == str(v) for k, v in want.items()))


def gpu_stacks(rows, boxes, B):
    """per backend with more than one stack: one dense line per stack (rate, reach, host), so an A/B of drivers reads as lines"""
    out = {}
    for be in sorted({r['backend'] for r in rows if r['kind'] == 'gpu_decode'}):
        R = [r for r in rows if r['kind'] == 'gpu_decode' and r['backend'] == be]
        keys = sorted({json.dumps(row_stack(r, B), sort_keys=True) for r in R})
        if len(keys) < 2:
            continue
        for k in keys:
            st = json.loads(k)
            L = gpu_lines([r for r in R if row_stack(r, B) == st], boxes, dict(B, line_stack={'v': {be: st}})).get(be)
            if L:
                out.setdefault(be, []).append({'stack': st, 'on_line': st == {x: str(y) for x, y in B['line_stack']['v'].get(be, {}).items()},
                                               'cards': L['cards'], 'rows': L['rows'], 'rate_gbs': L['rate_gbs'], 'eta': L['eta'], 'host_ms': L['host_ms']})
    return out


def gpu_lines(rows, boxes, B):
    """per backend: time per token (ms) = read GB x 1000 / rate + host, least squares over the dense rows of each card (the
    backend takes the median card); moe_layer_ms = median over its MoE rows of (measured - line at the active read) / MoE
    layers. Rows off the backend's line_stack stay out. Backends with no box take B.borrow (ESTIMATE)"""
    ok = [r for r in rows if r['kind'] == 'gpu_decode' and r['rc'] == 0 and r.get('decode_tps') and on_line(r, B)]
    out = {}
    for be in sorted({r['backend'] for r in ok}):
        cards = []
        for box, gid in sorted({(r['box'], r['gpu_id']) for r in ok if r['backend'] == be}):
            R = [r for r in ok if r['backend'] == be and r['box'] == box and r['gpu_id'] == gid]
            D = [r for r in R if r['model_kind'] == 'dense']
            if len({r['read_gb'] for r in D}) < B['min_dense']['v']:
                continue
            xs, ys = [r['read_gb'] for r in D], [1000 / r['decode_tps'] for r in D]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
            h = my - a * mx
            card = next(g for g in boxes[box]['gpus'] if g['id'] == gid)
            plate = card['bw']['v']
            res = [y - a * x - h for x, y in zip(xs, ys)]
            line = 'least squares over %d dense rows on %s %s (%s): %s; residuals %s ms' % (
                len(D), box, gid, ', '.join(r['id'] for r in D), ', '.join('%.2f GB %.1f tok/s' % (r['read_gb'], r['decode_tps']) for r in D),
                ', '.join('%+.2f' % e for e in res))
            M_ = [r for r in R if r['model_kind'] == 'moe' and r['moe_layers']]
            per = [(1000 / r['decode_tps'] - a * r['read_gb'] - h) / r['moe_layers'] for r in M_]
            cards.append({'card': box + ' ' + gid, 'vendor': card['vendor'], 'resid': max(abs(e) for e in res), 'rate': 1000 / a, 'eta': 1000 / a / plate, 'host': h, 'plate': plate, 'line': line,
                          'moe': _med(per) if per else None, 'moe_all': per, 'moe_src': '; '.join('%s: %.2f ms measured vs %.2f on the line at %.2f GB active, %d MoE layers -> %.4f ms/layer' % (
                              r['id'], 1000 / r['decode_tps'], a * r['read_gb'] + h, r['read_gb'], r['moe_layers'], p) for r, p in zip(M_, per)),
                          'rows': [r['id'] for r in R]})
        if not cards:
            continue
        pick = lambda k: _med([c[k] for c in cards])
        src = 'MEASURED: ' + ' | '.join(c['line'] for c in cards)
        rmax = max(c['resid'] for c in cards)
        lt = M if rmax <= B['line_resid_max_ms']['v'] else 'MEASURED, CONFOUNDED'
        if lt != M:
            src += ' (a dense row sits %.2f ms off the line > %.2f ms: the quant kernels differ, so one line is an average over them)' % (rmax, B['line_resid_max_ms']['v'])
        h = pick('host')
        mo = [c for c in cards if c['moe'] is not None]
        out[be] = {'cards': [c['card'] for c in cards], 'vendors': sorted({c['vendor'] for c in cards}), 'rows': sum((c['rows'] for c in cards), []),
                   'rate_gbs': n(round(pick('rate'), 1), lt, src),
                   'eta': n(round(pick('eta'), 3), lt, src + ' (rate / nameplate ' + ', '.join('%s %g GB/s' % (c['card'], c['plate']) for c in cards) + ')'),
                   'host_ms': n(round(max(h, 0), 3), lt, src + (' (the intercept)' if h >= 0 else ' (the intercept %.3f ms is below zero, held at 0: no host work is negative)' % h)),
                   'moe_layer_ms': moe_term(mo, B) if mo else n(None, 'UNKNOWN', 'UNKNOWN: no MoE row on a ' + be + ' card')}
    for be, frm in B['borrow']['v'].items():
        if be not in out and frm in out:
            out[be] = {'cards': [], 'rows': [], 'borrowed_from': frm, 'vendors': [B['borrow_vendor']['v'][be]]}
            for k in ('eta', 'host_ms', 'moe_layer_ms'):
                out[be][k] = n(out[frm][k]['v'], E, 'ESTIMATE: borrowed from the %s %s (%s); %s' % (frm, k, ', '.join(out[frm]['cards']), B['borrow']['src']))
    return out


def numa_gains(rows, B):
    """per placement key 'policy mM cC sS tT' (M weight nodes, C thread nodes, S sockets spanned, T threads): decode tok/s over the
    same round's one-node local arm, median over rounds; rounds spread past numa_spread_max -> MEASURED, CONFOUNDED"""
    ok = [r for r in rows if r['kind'] == 'numa_decode' and r['rc'] == 0 and r.get('decode_tps')]
    local = lambda r: r['policy'] == 'membind' and r['cpu_nodes'] == r['mem_nodes'] and len(r['mem_nodes']) == 1
    ref = {}
    for r in sorted(filter(local, ok), key=lambda r: r['threads']):
        ref[(r['box'], r['round'])] = r
    by = {}
    for r in ok:
        R = ref.get((r['box'], r['round']))
        if R:
            k = '%s m%d c%d s%d t%d' % (r['policy'], len(r['mem_nodes']), len(r['cpu_nodes']), r['sockets'], r['threads'])
            by.setdefault((r['box'], k), []).append((r, r['decode_tps'] / R['decode_tps'], R))
    out = {}
    for (box, k), xs in sorted(by.items()):
        g = [x[1] for x in xs]
        m = _med(g)
        spread = (max(g) - min(g)) / m
        tag = M if spread <= B['numa_spread_max']['v'] else 'MEASURED, CONFOUNDED'
        out.setdefault(box, {})[k] = n(round(m, 3), tag, 'MEASURED: %s decode over the one-node local arm, %s: %s' % (
            box, k, '; '.join('%s %.2f / %s %.2f = %.3f (%d threads, load %s)' % (r['id'], r['decode_tps'], R['id'], R['decode_tps'], x, r['threads'], r.get('load_before') or 'n/a')
                              for r, x, R in xs)) + ('' if tag == M else ' (rounds spread %.0f %% > %.0f %%)' % (100 * spread, 100 * B['numa_spread_max']['v'])))
    return out


BENCH_ROWS = [json.loads(x) for x in open(os.path.join(HERE, BENCH_SRC)) if x.strip()] if os.path.exists(os.path.join(HERE, BENCH_SRC)) else []
bench['gpu'] = gpu_lines(BENCH_ROWS, boxes, bench)
bench['stacks'] = gpu_stacks(BENCH_ROWS, boxes, bench)
bench['numa'] = numa_gains(BENCH_ROWS, bench)
bench['rows'] = len(BENCH_ROWS)


# ---------------- dLLM: block-diffusion decode (tokens per forward x forward cost) ----------------
# Source: lab notes 3a (llama-dllm-gen on the n-engine fork, lab notes feat/dllm-block-kv), recorded as run-ledger
# rows by tools/record_dllm.py (tag "dllm", recipe.kind "dllm"). decode tok/s = TPF x 1000 / ms per forward, where TPF (tokens
# committed per forward) is a measured property of the decoder schedule (model, thresholds, mode, block, workload) and
#   ms per forward = host + non-expert bytes / line + expert bytes touched by B tokens / (line x kernel gap)
# with the line = the box's measured backend line (bench.gpu, or the box DRAM read for CPU) and the kernel gap = the measured
# share of that line the MoE batch kernel reaches. model.js dllmTerms() fits TPF, B and the gap from these row summaries at
# run time (like calibrate()), holding out dllm.holdout; build.py only gathers the inputs and the settings.
DLLM_KINDS = ('cache', 'fuse')
dllm = {
    'file': 'runs/runs.jsonl (rows tagged dllm, written by tools/record_dllm.py)',
    'holdout': {'v': ['odie-dllm-n1p-flash2-q4-lev-s-fuse'],
                'src': 'kept out of every dLLM term to test the planner: LLaDA2.2-flash on 2x B70 with the per-pair MoE kernel (fixed prompts). Its TPF comes from the stock-kernel run of the same schedule, its kernel gap from the other per-pair rows'},
    'fit_modes': {'v': list(DLLM_KINDS), 'src': 'block-KV schedules (one 32-row block, or a fused confirm + next block, per forward); full mode resubmits the prefix, spec runs two models, the fork CLI resubmits the whole sequence'},
    'fit_blocks': {'v': [32], 'src': 'the block length the models were trained with (GGUF diffusion.block_length); block 64 is a what-if (lab notes 3a)'},
    'b_slack': {'v': 1.25, 'src': 'a forward wider or narrower than the fitted rows by more than this factor is UNMODELLED: the MoE batch kernel is not linear in the rows (stock block 64: 409 ms per forward vs 151 ms at block 32)'},
    'eff_min_rows': {'v': 2, 'src': 'a kernel-gap group (expert shape x quant type) stands alone with this many rows; otherwise the path pools its groups'},
    'eff_spread_max': {'v': 0.25, 'src': 'kernel-gap groups of one path that differ by more than this share of their median do not pool: the gap is then a property of the expert shape (stock path: small experts pay the per-expert loop)'},
    'tpf_spread_max': {'v': 0.25, 'src': 'TPF rows of one cell that differ by more than this share of their median are MEASURED, CONFOUNDED: kernel numerics flip near-threshold commits (mini Q fuse 4.04-4.84 across kernels)'},
    'coverage_file': {'v': 'runs/dllm-raw/coverage-2026-10-19.jsonl', 'src': 'the expert-coverage ledger: --expert-stats on the fixed fork (66972be94 reads the strided ffn_moe_topk view row by row), CPU on odie'},
    'rho_spread_max': {'v': 0.25, 'src': 'locality rows that differ by more than this share of their median are MEASURED, CONFOUNDED'},
    'cards_serial': n(1, E, 'ESTIMATE: a layer split runs the cards in turn for one sequence, so a forward costs the summed bytes at one card\'s line'),
    'cpu_host_ms': n(0, E, 'ESTIMATE: no CPU host term is measured; the CPU kernel gap absorbs it'),
    'line_of': {'v': {'sycl': 'bench.gpu.sycl', 'cpu': 'boxes.<box>.dram.bw'}, 'src': 'the measured line a backend reads: the bench GPU line (rate + host), or the box DRAM read for CPU'},
    'models': {}, 'rows': [],
}


def routing_locality(path, D):
    """the routing locality rho: B tokens of one forward touch as many experts per MoE layer as rho x B independent uniform top-k
    picks, coverage(B) = 1 - (1 - k/E)^(rho B). Per row rho = ln(1 - experts / (L E)) / (B ln(1 - k/E)). The diffusion block
    forwards (fit_modes rows) set it, median; the AR decode windows of the same family are shown beside them, never fitted"""
    rows = [json.loads(x) for x in open(path) if x.strip()]
    rho = lambda r: math.log(1 - r['experts'] / (r['moe_layers'] * r['E'])) / (r['b'] * math.log(1 - r['k'] / r['E']))
    fit = [r for r in rows if r['mode'] in D['fit_modes']['v']]
    ar = [r for r in rows if r['mode'] == 'ar-window' and r['b'] == D['fit_blocks']['v'][0]]
    xs = [rho(r) for r in fit]
    m = _med(xs)
    spread = (max(xs) - min(xs)) / m
    tag = M if spread <= D['rho_spread_max']['v'] else 'MEASURED, CONFOUNDED'
    cov = lambda R: _med([r['experts'] / (r['moe_layers'] * r['E']) for r in R])
    return n(round(m, 3), tag, 'MEASURED: routing locality rho = median of %d LLaDA2.1-mini block forwards (%s; %s), coverage at B 32 = %.0f %% '
             'per MoE layer vs %.0f %% for uniform routing; Ling-mini-2.0 AR decode, 32-token windows teacher-forced on its own KV: median '
             'coverage %.0f %% (rho %.2f, %d windows). Other models borrow the mini value (ESTIMATE until measured). The old "nearly all 256" '
             'came from the probe bug (4864 = 19 x 256 on every prompt)' % (
                 len(fit), ', '.join('%s %s B %.1f rho %.3f' % (r['mode'], r['prompt'], r['b'], x) for r, x in zip(fit, xs)), D['coverage_file']['v'],
                 100 * cov([r for r in fit if r['b'] == 32]), 100 * (1 - (1 - fit[0]['k'] / fit[0]['E']) ** 32), 100 * cov(ar), _med([rho(r) for r in ar]), len(ar)))


dllm['routing'] = routing_locality(os.path.join(HERE, dllm['coverage_file']['v']), dllm)
for _k in ('llada21mini', 'llada21mini_q5km', 'llada22flash', 'lingmini2', 'dreamcoder7b'):
    _p = os.path.join(HERE, 'data', 'models', _k + '.json')
    if not os.path.exists(_p):
        continue
    _m = json.load(open(_p))
    _i = _m['intake']
    _src = _i['source']
    _lk = sum((_i.get('lookup_gib') or {}).values())
    dllm['models'][_k] = {'arch': _m['arch']['v'], 'moe': bool(_m.get('E')), 'L': _m.get('L') or 0, 'E': _m.get('E') or 0, 'k': _m.get('k') or 0,
                          'shape': '%sx%s' % (_m['hidden']['v'], (_m.get('expert_ff') or {}).get('v')),
                          'experts_gib': n(_i['experts_gib'], R, _src + ': routed expert tensors'),
                          'nonexpert_gib': n(round(_i['total_gib'] - _i['experts_gib'] - _lk, 3), R, _src + ': every non-expert tensor except the lookup tables (%s, rows read per token only)' % ', '.join(_i.get('lookup_gib') or {})),
                          'total_gib': n(_i['total_gib'], R, _src)}
for _r in runs_tables.ledger.rows(os.path.join(HERE, 'runs', 'runs.jsonl')):
    if 'dllm' not in (_r.get('tags') or []):
        continue
    _c, _ms = _r['recipe'], _r['measured']
    dllm['rows'].append({'id': _r['id'], 'when': _r['when'], 'box': _r['box'], 'model': _r['model'], 'backend': _c['backend'], 'path': _c['moe_path'],
                         'cards': len(_c['cards']), 'quant': _c['quant'], 'schedule': _c['schedule']['name'], 'mode': _c['schedule']['mode'], 'block': _c['block'],
                         'workload': _c['workload'], 'placement': _c['placement'], 'ok': _ms['outcome'] == 'ok' and not _ms.get('degraded'),
                         'tpf': _ms.get('tpf'), 'ms_per_fwd': _ms.get('decode_ms_per_fwd') or _ms.get('ms_per_fwd'), 'b': _ms.get('tokens_per_fwd_submitted'), 'decode_tps': (_ms.get('decode_tps') or {}).get('all'),
                         'src': _r['sources'][0]})


def external_rows(path, fields):
    """runs/external.jsonl rows, each tagged MEASURED-BY-THEM with src = its URL, and missing = the conditions fields (and core
    fields) the source does not declare"""
    if not os.path.exists(path):
        print('external: %s not found (no validation rows)' % EXT_SRC)
        return []
    rows = [json.loads(x) for x in open(path) if x.strip()]
    for r in rows:
        st = r.get('settings') or {}
        miss = [k for k in fields if st.get(k) is None]
        miss += [k for k in ('date', 'engine_version', 'quant') if r.get(k) is None]
        miss += ['measured.prefill_tps'] if (r.get('measured') or {}).get('prefill_tps') is None else []
        r.update({'tag': external['tag'], 'src': r['url'], 'missing': miss})
    return rows


external['rows'] = external_rows(os.path.join(HERE, EXT_SRC), list(conditions['fields']))

# the layer map (per-layer sensitivity, CLAIMED / MEASURED / INFERRED bands; src/model.js layerMap calibrates it) and the
# upgrade advisor's settings and part catalog (src/advisor.js)
layer_maps = json.load(open(os.path.join(HERE, 'data', 'layer-maps.json')))
upgrade = json.load(open(os.path.join(HERE, 'data', 'upgrades.json')))
# a public copy names the models whose runs it leaves out, per box (tools/curate.py writes the file; our own repo has none)
_lo = os.path.join(HERE, 'runs', 'left_out.json')
left_out = json.load(open(_lo)) if os.path.exists(_lo) else {}

data = {
    'meta': {'built': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%MZ'), 'by': 'fit-viz build.py', 'home': PUBLIC_HOME,
             'tags': {M: 'we ran it or read it live', R: 'read from a doc, file header or spec', P: 'community prior', E: 'our estimate', FIT: 'fitted to measured points'}, 'left_out': left_out},
    'types': types, 'engines': engines, 'models': models, 'boxes': boxes, 'b70move': b70move, 'anchors': anchors, 'assume': assume, 'knobs': knobs, 'conditions': conditions, 'accuracy': accuracy, 'causes': causes, 'external': external, 'bench': bench,
    'checks': checks, 'levers': levers, 'recipes': recipes, 'draft_twins': draft_twins, 'decode_fits': decode_fits, 'prefill': prefill, 'dllm': dllm, 'layer_maps': layer_maps, 'upgrade': upgrade,
    'alloc3': {'src': 'homelab dsv41/skew/results/2026-10-08-alloc-3bpw.json (' + alloc['rule'] + ')', 'layers': alloc_layers, 'count': sum(len(x) for x in alloc_layers)} if alloc else None,
    'siblings': {'quant_priors': sibling('quant-priors.json'), 'quant_recipes': sibling('quant-recipes.json'),
                 'quant_priors_md': sibling_md('QUANT-PRIORS.md'), 'quant_recipes_md': sibling_md('QUANT-RECIPES.md')},
}

# apply quant priors (if the sibling file has per-type KLD priors) -- see apply_priors()
exec(open(os.path.join(SRC, 'priors_adapter.py')).read())
apply_priors(data)



def ledger_cites(path):
    """(run id, the source strings it lists) per runs/runs.jsonl row and runs/bench.jsonl row: a seen value whose src names
    either cites that row"""
    rows = runs_tables.ledger.rows(path) + BENCH_ROWS
    norm = lambda s: s.replace('kit:', '').replace('lab notes', 'lab notes')
    return [(r['id'], [norm(x) for x in r.get('sources') or [] if len(x) >= EVIDENCE_CITE_MIN]) for r in rows]


EVIDENCE_CITE_MIN = 12   # a ledger source string shorter than this ("lab notes") names no one run


EVIDENCE_SRC_KEYS = ('url', 'source', 'note', 'method')  # a tagged row from another writer names its source in one of these


def stamp_evidence(x, cites, counts, dflt=None):
    """every dict with a tag gets evidence (n() already set it; literal rows, sibling files and planner rows get it here);
    a row without src gets the first of EVIDENCE_SRC_KEYS it holds, else dflt (the sibling file it came from);
    a seen value also gets runs = the ledger rows its src cites. counts[evidence] tallies the catalog"""
    if isinstance(x, list):
        for y in x:
            stamp_evidence(y, cites, counts, dflt)
        return
    if not isinstance(x, dict):
        return
    if isinstance(x.get('tag'), str):
        if not str(x.get('src') or '').strip():
            k = next((k for k in EVIDENCE_SRC_KEYS if str(x.get(k) or '').strip()), None)
            if k or dflt:
                x['src'] = str(x[k]) if k else dflt
        x['evidence'] = evidence_of(x['tag'], x.get('src'))
        counts[x['evidence']] = counts.get(x['evidence'], 0) + 1
        src = str(x.get('src') or '')
        ids = [i for i, ss in cites if i in src or any(s in src for s in ss)] if x['evidence'] == SEEN else []
        if ids:
            x['runs'] = ids
    for k, y in x.items():
        if k != 'layers' or not isinstance(y, list):  # heat 'layers' are big number arrays; a tagged layers row is stamped
            stamp_evidence(y, cites, counts, dflt)


_counts = {}
SIBLING_FILES = {'quant_priors': 'quant-priors.json', 'quant_recipes': 'quant-recipes.json', 'quant_priors_md': 'QUANT-PRIORS.md', 'quant_recipes_md': 'QUANT-RECIPES.md'}
data['external']['src'] = EXT_SRC
_cites = ledger_cites(os.path.join(HERE, 'runs', 'runs.jsonl'))
for _k, _x in data.items():
    if _k == 'siblings':
        for _s, _y in (_x or {}).items():
            stamp_evidence(_y, _cites, _counts, SIBLING_FILES.get(_s, _s))
    elif _k != 'meta':
        stamp_evidence(_x, _cites, _counts)
data['meta']['evidence'] = dict(EVIDENCE, counts=_counts)
json.dump(data, open(os.path.join(HERE, 'data.json'), 'w'), separators=(',', ':'))

# planner outputs the data carries for its readers (the layer map calibration, the route-trim proposals, the upgrade advice):
# src/advisor.js on the data.json just written; without node they are UNKNOWN with the reason
ADVISOR_TIMEOUT_S = 300


def planner_outputs():
    import subprocess
    try:
        p = subprocess.run(['node', os.path.join(SRC, 'advisor.js'), '--json'], cwd=HERE, capture_output=True, text=True, timeout=ADVISOR_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'status': 'UNKNOWN', 'why': 'src/advisor.js did not run: %s' % e}
    if p.returncode:
        return {'status': 'UNKNOWN', 'why': 'src/advisor.js rc %d: %s' % (p.returncode, p.stderr.strip()[-400:])}
    return dict(json.loads(p.stdout), status='computed')


_po = planner_outputs()
data['layer_maps']['calibration'] = _po.get('layer_maps') or {'status': _po['status'], 'why': _po.get('why')}
data['layer_maps']['trim_proposals'] = _po.get('trim') if 'trim' in _po else {'status': _po['status'], 'why': _po.get('why')}
data['upgrade']['advice'] = _po.get('upgrades') or {'status': _po['status'], 'why': _po.get('why')}
json.dump(data, open(os.path.join(HERE, 'data.json'), 'w'), separators=(',', ':'))
PAGE_DIR = os.path.join(HERE, 'page')   # the page's data scripts (generated, never in git); index.html loads them
PAGE_FILE_LIMIT = 1 << 20                # no page file is larger: a part over it is a build error, so split it further
PAGE_CODE = ('src/model.js', 'src/app.js')


def page_parts(data):
    """[(file name, script)]: data.js holds data.json without its bulky parts (each model's heat sets, the sibling files, the
    recipe rows); each later file puts one of them back on window.FIT_DATA before src/model.js and src/app.js run"""
    j = lambda x: json.dumps(x, separators=(',', ':'))
    heat = [k for k, m in data['models'].items() if (m.get('heat') or {}).get('sets')]
    core = dict(data, siblings={}, recipes=[], models={k: dict(m, heat=dict(m['heat'], sets={})) if k in heat else m
                                                       for k, m in data['models'].items()})
    return ([('data.js', 'window.FIT_DATA = ' + j(core) + ';\n')]
            + [('heat-%s.js' % k, 'window.FIT_DATA.models[%s].heat.sets = %s;\n' % (j(k), j(data['models'][k]['heat']['sets']))) for k in heat]
            + [('%s.js' % k, 'window.FIT_DATA.%s = %s;\n' % (k, j(data[k]))) for k in ('siblings', 'recipes')])


def write_page(data):
    parts = page_parts(data)
    big = [(n, len(s.encode())) for n, s in parts if len(s.encode()) > PAGE_FILE_LIMIT]
    if big:
        raise SystemExit('page parts over %d bytes: %s (split them further in page_parts)' % (PAGE_FILE_LIMIT, big))
    os.makedirs(PAGE_DIR, exist_ok=True)
    for f in os.listdir(PAGE_DIR):
        if f.endswith('.js') and f not in dict(parts):
            os.remove(os.path.join(PAGE_DIR, f))
    for n, s in parts:
        with open(os.path.join(PAGE_DIR, n), 'w') as f:
            f.write(s)
    tags = ['page/' + n for n, _ in parts] + list(PAGE_CODE)
    html = open(os.path.join(SRC, 'template.html')).read().replace(
        '<!--__SCRIPTS__-->', '\n'.join('<script src="%s"></script>' % t for t in tags))
    with open(os.path.join(HERE, 'index.html'), 'w') as f:
        f.write(html)
    return parts


_parts = write_page(data)
print('data.json %.0f KB; index.html %.0f KB + page/ %s' % (os.path.getsize(os.path.join(HERE, 'data.json')) / 1024, os.path.getsize(os.path.join(HERE, 'index.html')) / 1024,
      ', '.join('%s %.0f KB' % (n, len(s.encode()) / 1024) for n, s in _parts)))
print('siblings:', {k: (v.get('present') if isinstance(v, dict) else bool(v)) for k, v in data['siblings'].items()})
