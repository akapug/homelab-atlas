"""Planner family orn122: Ornith-1.5-122B (Qwen3.5-122B-A10B fine-tune, GGUF arch qwen35moe), Q4_K_M.

Loaded by build.py (families/*.py, run in the build namespace after the built-in models): every name below that is not
defined here (n, R, E, models, types, hdr_summary, ...) is a build.py global. A family file holds everything one model
family adds to the planner (header read, KLD priors, the models entry, its quality-gate family), so dropping the file
drops the family and nothing else."""

# Ornith-1.5-122B (ornith-ai fine-tune of Qwen3.5-122B-A10B; GGUF arch qwen35moe): Gated DeltaNet linear attention on 3 of 4
# layers, 256 experts top-8 + 1 shared, one nextn (MTP) block. The header was read from Hugging Face (not yet from our disks),
# so its values are READ but inferred until a header read on odie / droopy
HORN = ('HF GGUF header ornith-ai/Ornith-1.5-122B-GGUF@5514dc9eb187 Ornith-1.5-122B-Q4_K_M.gguf (75.84 GB; range read 2026-10-20 with '
        'probe/model_intake.py --hf --hf-file; dump src/hdr-orn122.json)')
ORN_HDR = os.path.join(SRC, 'hdr-orn122.json')
_gs = importlib.util.spec_from_file_location('fitviz_gguf', os.path.join(HERE, 'probe', 'gguf.py'))
GG = importlib.util.module_from_spec(_gs)
_gs.loader.exec_module(GG)
_ornh = json.load(open(ORN_HDR))
ornkv = _ornh['kv']
og = lambda k: ornkv[ornkv['general.architecture'] + '.' + k]
ORN_MAIN = og('block_count') - og('nextn_predict_layers')   # llama.cpp counts the nextn block inside block_count (49 = 48 + 1)
ORN_P = 3 * og('embedding_length') * og('expert_feed_forward_length')
_, ornex, orndense, ornlookup, ornmtp = hdr_summary(ORN_HDR, ('token_embd',), ORN_MAIN)
orn_mtp_ne, orn_mtp_ex = mtp_parts(ORN_HDR, ORN_MAIN)
ORN_LAYERS = sorted({int(t[0].split('.')[1]) for t in _ornh['tensors'] if '_exps' in t[0] and int(t[0].split('.')[1]) < ORN_MAIN})
_onx = [t for t in _ornh['tensors'] if not ('_exps' in t[0] or t[0].startswith('token_embd') or t[0].startswith('blk.%d.' % ORN_MAIN))]
ORN_DENSE_BPW = sum(t[3] for t in _onx) * 8 / sum(math.prod(t[1]) for t in _onx)
# Q4_K_M expert mix as shipped: gate/up q4_K everywhere, down q6_K on llama.cpp's use_more_bits layers and q4_K on the rest
ORN_DOWN = {int(t[0].split('.')[1]): GG.type_name(t[2]) for t in _ornh['tensors'] if re.match(r'blk\.\d+\.ffn_down_exps\.weight', t[0]) and int(t[0].split('.')[1]) < ORN_MAIN}


# Ornith KLD: nothing measured for Ornith or Qwen3.5-122B-A10B (quant-priors holds only ubergarm PPL ratios for Qwen3.5-397B,
# IQ4_KSS 1.0072 / Q3_K 1.016: the family takes low bits well), so the K-quant types its file uses borrow the GLM-5.3-Flash
# priors (also a BF16-source top-8 MoE, same llama.cpp K-quant kernels): ESTIMATE until a KL run on our gate
ORN_KLD = 'ESTIMATE: no KLD measured for Ornith-1.5-122B or Qwen3.5-122B-A10B; borrowed from the GLM-5.3-Flash prior of the same type (BF16-source top-8 MoE, same K-quant kernels; Qwen3.5-397B PPL ratios in quant-priors, IQ4_KSS 1.0072 / Q3_K 1.016, say the family takes low bits at least as well): '


def korn(x):
    return n(x['v'], E, ORN_KLD + x['src'])


_orn = {'q4_K': korn(kldglm['q4_K']), 'q6_K': korn(kldglm['q6_K']),
        KQM_TY: korn(prior(GA, 'Q4_K_M', 'kld', fallback=0.05, what='AesSedai GLM-5.3-Flash Q4_K_M = q4_K gate/up + q5_K down, unscaled: q4k_q6k has q6_K down'))}
for q, t in types.items():
    t['kld']['orn122'] = _orn.get(q, n(None, E, 'not priored for Ornith-122B: only the K-quant types of its Q4_K_M file'))


# Ornith-1.5-122B (Qwen3.5-122B-A10B fine-tune): every shape and byte figure from the Q4_K_M header (HORN). No run, no routing
# counts, no KL on our gate yet: heat is the synthetic GLM shape (as Flash-Next), quality UNKNOWN, host borrowed
# KV + compute on the first GPU, at ORN_CTX (the r0 recipe context): f16 KV of the full-attention layers (every
# full_attention_interval-th block) + the MTP block, the Gated DeltaNet state of the other blocks, the CUDA context GLM measured,
# and a compute buffer that is the logits buffer (GLM: 154,880 vocab x 4096 x 4 B = 2,420 MiB vs 2,484 MiB measured), so GLM's
# measured buffer scaled by the vocab
ORN_CTX = 32768
ORN_VOCAB = next(t[1][1] for t in _ornh['tensors'] if t[0] == 'output.weight')
GLM_VOCAB = glmkv['glm5-next.vocab_size']
_n_full = ORN_MAIN // og('full_attention_interval') + og('nextn_predict_layers')
_kv_tok = og('attention.head_count_kv') * (og('attention.key_length') + og('attention.value_length')) * 2
_gdn = (ORN_MAIN - ORN_MAIN // og('full_attention_interval')) * 4 * (og('ssm.time_step_rank') * og('ssm.state_size') * og('ssm.inner_size') // og('ssm.time_step_rank')
                                                                  + (og('ssm.conv_kernel') - 1) * (og('ssm.inner_size') + 2 * og('ssm.group_count') * og('ssm.state_size')))
ORN_KV_MIB = _n_full * _kv_tok * ORN_CTX / 2 ** 20
ORN_GDN_MIB = _gdn / 2 ** 20
ORN_CB4096_MIB = GLM_CB4096_MIB * ORN_VOCAB / GLM_VOCAB
_orn_kv = lambda cb: round((GLM_CTX_MIB + ORN_KV_MIB + ORN_GDN_MIB + cb) / 1024, 3)
ORN_KVSRC = ('ESTIMATE at ctx %d: CUDA context %d MiB (GLM, MEASURED on droopy) + f16 KV %d layers (%d full-attention + %d MTP) x %d B/token = %.0f MiB '
             '+ Gated DeltaNet state %d layers x f32 (%d heads x %d x %d recurrent + conv %d x %d) = %.0f MiB + compute buffer %s; the shapes are READ from %s' % (
                 ORN_CTX, GLM_CTX_MIB, _n_full, ORN_MAIN // og('full_attention_interval'), og('nextn_predict_layers'), _kv_tok, ORN_KV_MIB,
                 ORN_MAIN - ORN_MAIN // og('full_attention_interval'), og('ssm.time_step_rank'), og('ssm.state_size'), og('ssm.inner_size') // og('ssm.time_step_rank'),
                 og('ssm.conv_kernel') - 1, og('ssm.inner_size') + 2 * og('ssm.group_count') * og('ssm.state_size'), ORN_GDN_MIB, '%s', HORN))
models['orn122'] = {
    'name': 'Ornith-1.5-122B (Qwen3.5-122B-A10B fine-tune, ornith-ai)', 'short': 'Ornith-122B', 'host_borrow': 'glm',
    'host_borrow_src': 'ESTIMATE: no Ornith run on any box, so each box lends its GLM-5.3-Flash ik host (the closest analog: one-chain K-quant GGUF + MTP on droopy); Flash-Next borrows V4.1 the same way',
    'L': len(ORN_LAYERS), 'E': og('expert_count'), 'k': og('expert_used_count'), 'P': ORN_P,
    'P_src': HORN + ': ffn_*_exps [%d, %d, %d] x3; %d MoE blocks 0-%d (block %d = nextn MTP)' % (og('embedding_length'), og('expert_feed_forward_length'), og('expert_count'), len(ORN_LAYERS), ORN_LAYERS[-1], ORN_MAIN),
    'show': True,
    'dense_gib': n(round(orndense, 3), R, HORN + ': non-expert tensors without token_embd and the MTP block %d (attention, Gated DeltaNet, shared expert, router, output; q4_K/q6_K/f32)' % ORN_MAIN),
    'dense_bpw': n(round(ORN_DENSE_BPW, 3), R, HORN + ': mean bits per weight of those tensors'),
    'dense_gib_by_type': {KQM_TY: n(round(orndense, 3), R, HORN + ': the shipped file keeps these non-expert bytes')},
    'base_type': KQM_TY, 'kld_family': 'orn122',
    'experts_gib': n(round(ornex, 3), R, HORN + ': routed experts of blocks 0-%d, mean %.3f bpw (gate/up q4_K, down q6_K on %d layers / q4_K on %d)' % (
        ORN_LAYERS[-1], ornex * GiB * 8 / (len(ORN_LAYERS) * og('expert_count') * ORN_P), sum(x == 'q6_K' for x in ORN_DOWN.values()), sum(x == 'q4_K' for x in ORN_DOWN.values()))),
    'store_layers': {'v': {'q6_K_down': sorted(l for l, x in ORN_DOWN.items() if x == 'q6_K'), 'q4_K_down': sorted(l for l, x in ORN_DOWN.items() if x == 'q4_K')}, 'tag': R, 'src': HORN + ': ffn_down_exps type per block'},
    'lookups': [{'label': 'token_embd (lookup)', 'gib': n(round(ornlookup, 3), R, HORN), 'cls': 'token_embd', 'on_file': False}],
    'kv_gib': n(_orn_kv(ORN_CB4096_MIB), E, ORN_KVSRC % ('-ub 4096 %.0f MiB = GLM %d MiB x vocab %d / %d' % (ORN_CB4096_MIB, GLM_CB4096_MIB, ORN_VOCAB, GLM_VOCAB))),
    'kv_gib_ub512': n(_orn_kv(ORN_CB4096_MIB * 512 / 4096), E, ORN_KVSRC % ('-ub 512 %.0f MiB (the -ub 4096 figure x 512/4096)' % (ORN_CB4096_MIB * 512 / 4096))),
    'draft': {'label': 'MTP head (nextn block %d: attention, shared expert and, under ik -ot exps=CPU on blocks 0-%d, its routed experts on the GPU)' % (ORN_MAIN, ORN_MAIN - 1),
              'vram_gib': n(round(orn_mtp_ne, 3), R, HORN + ': block %d non-expert tensors' % ORN_MAIN),
              'experts_gib': {KQM_TY: n(round(orn_mtp_ex, 3), R, HORN + ': blk.%d ffn_*_exps (gate/up q4_K, down q6_K)' % ORN_MAIN)},
              'experts_on_gpu': n(True, E, 'ESTIMATE: the r0 recipe sends only blocks 0-%d experts to the CPU (-ot), as ik --cpu-moe left GLM block 45 experts on the GPU (%s)' % (ORN_MAIN - 1, GLM_VRAM_SRC)),
              'acceptance': n(0.85, E, 'ESTIMATE: Ornith\'s own MTP head (checkpoint "MTP1740", 1 nextn layer) is unmeasured; analog GLM-5.3-Flash MTP on droopy ik, n_max 3, coding-agent greedy: 0.846-0.954 (runs droopy-glm-glm-be1q3e-1215 0.846 at p_min 0, droopy-glm-glm-be1dvnc-1202 0.954 at p_min 0.75); the low end'),
              'mult_default': n(1.30, E, 'ESTIMATE: analog GLM-5.3-Flash MTP n_max 3 p_min 0.75 on droopy ik, DV0 15.95 vs no-spec 12.30 tok/s = 1.30 (runs droopy-glm-glm-be1dvnc-1202, CPU-MoE decode as here); Ornith unmeasured')},
    'quality_note': 'vendor claims only (ornith-ai card: stronger than Qwen3.5-122B-A10B on its boards); UNKNOWN in our gate until a KL run and our agent eval',
    'tensor_names': ['blk.{L}.ffn_gate_exps.weight', 'blk.{L}.ffn_up_exps.weight', 'blk.{L}.ffn_down_exps.weight'],
    'layer_ids': ORN_LAYERS,
    'heat': {'floor': common_heat_floor, 'anchor_score': 'glm_shape', 'anchor_sel': 'glm_shape', 'plan_default': 'glm_shape', 'score_default': 'glm_shape',
             'sets': {'glm_shape': {'label': 'synthetic: GLM live coding-routing shape (no Ornith routing counts yet)', 'tag': E,
                                    'src': FNXH.replace('Flash-Next routing not counted yet (scout measurement 3 counts it)', 'Ornith routing not counted yet: UNKNOWN routing'),
                                    'layers': synth_heat(len(ORN_LAYERS), og('expert_count'), seed=122)}}},
}

FAMILY_QUALITY['orn122'] = {'engines': [], 'tag': 'UNKNOWN', 'unmeasured': True,
                            'note': 'Ornith-122B: no KL on our gate and no agent eval yet; the Q4_K_M store is the only one modelled (lowbit share 0), its KLD per type is an ESTIMATE borrowed from GLM-5.3-Flash measurements'}
