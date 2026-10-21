"""Ornith-35B: reuse the existing qwen35moe family contract, with its own header facts.

No Lassie run, routing counts or local quality calibration is implied. GPT-OSS
intake stays separate: biased experts and its engine path are not modelled here.
"""
import copy

O35_SRC = ('HF GGUF header ornith-ai/Ornith-1.5-35B-A3B-GGUF@'
           'f60452672689e0040be350560267a614cc172a18 '
           'Ornith-1.5-35B-Q4_K_M.gguf; verified bounded ranges, sanitized src/hdr-orn35.json')
O35_PATH = os.path.join(SRC, 'hdr-orn35.json')
with open(O35_PATH) as f:
    o35h = json.load(f)
o35kv = o35h['kv']
o35g = lambda k: o35kv['qwen35moe.' + k]
o35main = o35g('block_count') - o35g('nextn_predict_layers')
_, o35ex, o35dense, o35lookup, _ = hdr_summary(O35_PATH, ('token_embd',), o35main)
o35ne, o35mtp = mtp_parts(O35_PATH, o35main)
o35down = {int(t[0].split('.')[1]): GG.type_name(t[2]) for t in o35h['tensors']
           if re.fullmatch(r'blk\.\d+\.ffn_down_exps\.weight', t[0]) and int(t[0].split('.')[1]) < o35main}
o35non = [t for t in o35h['tensors'] if not ('_exps' in t[0] or t[0].startswith('token_embd') or t[0].startswith('blk.%d.' % o35main))]
o35full = o35main // o35g('full_attention_interval')
o35linear = o35main - o35full
o35kvtok = o35g('attention.head_count_kv') * (o35g('attention.key_length') + o35g('attention.value_length')) * 2
o35state = o35linear * 4 * (o35g('ssm.state_size') * o35g('ssm.inner_size')
    + (o35g('ssm.conv_kernel') - 1) * (o35g('ssm.inner_size') + 2 * o35g('ssm.group_count') * o35g('ssm.state_size')))
o35vocab = next(t[1][1] for t in o35h['tensors'] if t[0] == 'output.weight')
o35cb = GLM_CB4096_MIB * o35vocab / GLM_VOCAB
o35kvbytes = (o35full + o35g('nextn_predict_layers')) * o35kvtok * ORN_CTX + o35state
O35_KVSRC = ('ESTIMATE at ctx %d: %d full-attention + %d MTP layers x %d f16 B/token, '
             '%d B f32 recurrent/conv state; GLM CUDA context and vocab-scaled compute buffer borrowed, '
             'not measured on Pascal; shapes from %s' % (ORN_CTX, o35full, o35g('nextn_predict_layers'), o35kvtok, o35state, O35_SRC))

# Inherit numerical analogs explicitly, never the 122B model's shapes or evidence.
for q, t in types.items():
    x = t['kld'].get('orn122')
    t['kld']['orn35'] = n(x['v'] if x else None, E,
        'ESTIMATE: no Ornith-35B KLD; same numerical GLM K-quant analog as Ornith-122B, not a quality measurement: ' + (x['src'] if x else q))
o35 = copy.deepcopy(models['orn122'])
o35.update({
    'name': 'Ornith-1.5-35B-A3B (Qwen3.5 hybrid MoE fine-tune, ornith-ai)', 'short': 'Ornith-35B',
    'host_borrow_src': 'ESTIMATE: no Lassie Ornith-35B anchor; GLM host analog, not engine or throughput qualification',
    'L': len(o35down), 'E': o35g('expert_count'), 'k': o35g('expert_used_count'),
    'P': 3 * o35g('embedding_length') * o35g('expert_feed_forward_length'), 'P_src': O35_SRC,
    'types_only': [KQM_TY, 'q4_K', 'q6_K'], 'kld_family': 'orn35',
    'dense_gib': n(round(o35dense, 3), R, O35_SRC + ': main non-expert tensors excluding embedding lookup'),
    'dense_bpw': n(sum(t[3] for t in o35non) * 8 / sum(math.prod(t[1]) for t in o35non), R, O35_SRC),
    'dense_gib_by_type': {KQM_TY: n(round(o35dense, 3), R, O35_SRC)},
    'experts_gib': n(round(o35ex, 3), R, O35_SRC + ': main routed gate/up/down, exact mixed Q4_K/Q6_K bytes'),
    'store_layers': {'v': {q + '_down': sorted(l for l, typ in o35down.items() if typ == q) for q in ('q4_K', 'q6_K')}, 'tag': R, 'src': O35_SRC},
    'lookups': [{'label': 'token_embd (lookup)', 'gib': n(round(o35lookup, 3), R, O35_SRC), 'cls': 'token_embd', 'on_file': False}],
    'kv_gib': n((GLM_CTX_MIB + o35kvbytes / 2 ** 20 + o35cb) / 1024, E, O35_KVSRC + '; -ub 4096'),
    'kv_gib_ub512': n((GLM_CTX_MIB + o35kvbytes / 2 ** 20 + o35cb * 512 / 4096) / 1024, E, O35_KVSRC + '; -ub 512'),
    'layer_ids': sorted(o35down),
    'quality_note': 'UNKNOWN on Lassie: B70 household/coding results are different hardware/engine conditions; no local KLD or Lassie agent eval',
    'heat': {'floor': common_heat_floor, 'anchor_score': 'glm_shape', 'anchor_sel': 'glm_shape', 'plan_default': 'glm_shape', 'score_default': 'glm_shape',
             'sets': {'glm_shape': {'label': 'synthetic GLM-shaped routing, not Ornith-35B counts', 'tag': E, 'src': 'ESTIMATE: routing unmeasured',
                                    'layers': synth_heat(len(o35down), o35g('expert_count'), seed=35)}}},
})
o35['draft'].update({
    'label': 'Ornith-35B nextn block %d (unmeasured)' % o35main,
    'vram_gib': n(round(o35ne, 3), R, O35_SRC),
    'experts_gib': {KQM_TY: n(round(o35mtp, 3), R, O35_SRC)},
    'experts_on_gpu': n(True, E, 'ESTIMATE: the existing ik main-block-only CPU override leaves nextn experts on the GPU; verify the actual launch'),
    'acceptance': n(None, E, 'ESTIMATE: UNKNOWN, no Ornith-35B speculative acceptance measurement'),
    'mult_default': n(1.0, E, 'ESTIMATE: no measured draft gain; neutral placeholder, not an acceptance claim'),
})
models['orn35'] = o35
FAMILY_QUALITY['orn35'] = {'engines': [], 'tag': 'UNKNOWN', 'unmeasured': True,
    'note': 'No Lassie agent/KLD qualification; GLM numerical priors, synthetic routing and borrowed host/compute terms are estimates'}
