# How the best MoE quantizers choose what to keep and what to cut

2026-10-17. Machine-readable twin: `quant-recipes.json` (rules, recipes, measured tables, per-class header tables).

Method: I read GGUF headers only, never model data. Hugging Face files were read with HTTP Range requests. On odie, the local GLM-5.3-Flash UD-Q4_K_XL and Kimi-K3 UD-Q2_K_XL were read with `gguf-names.py`, plus a small header-only reader that prints the type per layer (`gguf-names.py` collapses layers and prints ik type ids as integers). Both local files match the HF files exactly. Every size below is tensor element count x the block size of the ggml type. Every quality number has a link. "Not measured" means nobody has published one.

## 1. The short answer

1. **Non-expert tensors stay at q8_0.** That covers attention, shared experts, dense layers, router and output. They hold under 4% of the weights (9.2 GiB on GLM-5.3-Flash, 9.6 GiB on V4.1 without engram), but per parameter they are 100-5000x more sensitive than routed experts. Every serious quantizer keeps them high.
2. **The routed experts carry the cuts.** Gate/up go lowest. Down gets the same type or one tier more.
3. **A few layers get extra bits.** Which layers depends on the model, and the good quantizers measure it rather than guess.
4. **Below about 2.6 bpw on the experts, coding and agent quality collapse.** The big quality step is from about 2.7 to about 3.5 bpw.
5. **Unsloth's "dynamic" method is mostly rules 1-3, plus a large chat/code imatrix and per-layer bumps.** Their bumps come from their own sensitivity runs; the method is not published as code. On GLM-5.3-Flash, a plain uniform mix (AesSedai) lands on about the same KLD-vs-size curve (section 3). The upshot: the class-level choices matter most, and the layer bumps add a smaller gain.

## 2. What Unsloth actually does (from their docs + headers)

Stated method:
- R1 blog ([unsloth.ai/blog/deepseekr1-dynamic](https://unsloth.ai/blog/deepseekr1-dynamic)):
  - First 3 dense layers at 4-6 bit; shared experts 6-bit; MLA attention 4-6 bit.
  - Embedding 4-bit, lm_head 6-bit, router and norms in 32-bit.
  - "down_proj is the most sensitive", so the first 3-6 MoE down_proj stay higher.
  - Flappy Bird score /10: dynamic 131 GB 6.92, 158 GB 9.08, 183 GB 9.17, vs naive uniform 133 GB 0, 149 GB 1.67, 175 GB 6.17.
- Dynamic 2.0: the type is chosen per layer and per model; the imatrix uses more than 1.5M chat tokens.
- Dynamic 3.0 ([docs](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs)): calibration adds agentic coding and multilingual data, and layer selection improved. 1-bit is "not for agentic or tool use".
- V4 doc ([unsloth.ai/docs/models/deepseek-v4](https://unsloth.ai/docs/models/deepseek-v4)): requantizing the QAT MXFP4 experts costs "5% for Q4_K, over 30% for IQ2_XXS". Q8_0 for some tensors is "not lossless", so those stay in BF16.
- Qwen3.5 benchmarks ([docs](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks)):
  - Up/gate are fine at 3-bit; down is slightly more sensitive.
  - Attention is very sensitive in hybrid models.
  - MXFP4 is worse than Q4_K on non-expert tensors.
  - On GPU, K-quants decode 5-10% faster than iq types.
  - PPL and KLD can mislead (their MiniMax example).

What the headers show (dominant type, `type@layers` = exceptions, bpw in brackets):

**GLM-5.3-Flash, Unsloth UD.** 46 blocks, 288 experts, top-8, layer 45 = MTP.

| class | UD-IQ1_S 86.7 GiB | UD-IQ2_XXS 94.8 | UD-Q2_K_XL 101.2 | UD-IQ3_XXS 112.1 | UD-Q3_K_XL 137.4 | UD-Q4_K_XL 186.0 |
|---|---|---|---|---|---|---|
| ffn_down_exps | iq3_xxs | iq3_xxs | iq3_xxs; iq4_xs@11-12,44 (3.15) | iq3_s; iq4_xs@11-12,44 | iq4_xs; q6_K@11-12,44 | q5_K; q6_K@11-12,44 |
| ffn_gate/up_exps | iq1_s; iq2_xxs on 14 layers | iq2_xxs; iq2_s@11 | iq2_xs; iq3_xxs@11 | iq2_s; iq3_s@11 | iq3_xxs; iq4_xs@11 | q4_K; q5_K@11 |
| MTP layer 45 experts | q2_K / q3_K down | same | q2_K gate/up, q3_K down | q2_K / q3_K | q3_K / q4_K | q4_K gate/up, q5_K down |
| attention | q5_K (k/q/v), MLA q8_0 | same | q5_K-q6_K, MLA q8_0 | q6_K, MLA q8_0 | q8_0 | q8_0 |
| shared expert | q5_K-q6_K | same | q5_K-q6_K | q6_K | q8_0 | q8_0 |
| output / embd | q4_K / q4_K | q4_K / q4_K | q4_K / q5_K | q6_K / q6_K | q6_K / q8_0 | q8_0 / q8_0 |

Layer 11 gets more bits in every tier, and down on layers 12 and 44 too. Kimi K2 and GLM-4.6 show the same pattern, with different layers (tables in the JSON).

**DeepSeek V4 Flash 0731, Unsloth UD** (V4, not V4.1: 43 blocks, 256 experts):
- Gate/up per tier:
  - IQ1_S: iq1_s, with iq2_xxs on 13 layers.
  - IQ2_XXS: iq2_xxs, with iq2_s@26.
  - Q2_K_XL: iq2_xs, with iq3_xxs@26.
  - Q3_K_XL: iq3_xxs, with mxfp4@26.
  - Q4_K_XL: native mxfp4.
- Down: iq3_xxs with mxfp4@26,42 in the three low tiers; mxfp4 in Q3/Q4.
- Attention, compressor and indexer: q8_0 in every tier. The router stays bf16.

**DeepSeek V3.1, Unsloth UD-Q2_K_XL (238 GiB):**
- Gate/up q2_K.
- Down q3_K, with q4_K on 17 "bump" layers (8, 12, 14, 15, 21, ... 60). The same bump layers recur in every V3.1 tier.
- Attention q4_K/q5_K.

**R1 (2025):** down is q2_K with q3_K on layers 3-5 in IQ1_M through Q2_K_XL. This is the "first 3-6 down_proj" rule from the blog.

**Counter-example: Kimi K3 UD-Q2_K_XL (local, 802.1 GiB).**
- Down iq2_xs, the same type as gate/up. The only raises are iq3_xxs on 11 down layers and iq3_xxs on gate/up@91.
- Every non-expert tensor is q8_0.
- Unsloth did not raise down across the board here.

## 3. The other schools

- **ubergarm** (ik_llama `--custom-q`, recipes in each model card):
  - Attention q8_0, with attn_output/q at iq6_k; shared/dense iq5_ks; embd iq4_k; output iq6_k; MTP layer iq5_ks.
  - Down one tier above gate/up: IQ2_KL = down iq3_ks + gate/up iq2_kl. The "smol" tiers keep down at the gate/up type.
- **Thireus GGUF-Tool-Suite** ([github](https://github.com/Thireus/GGUF-Tool-Suite)):
  - Measures every tensor group: whole-model KLD when that one group alone drops to a low reference type (baseline iq6_k, reference iq1_kt, 250+ chunks of a chat corpus).
  - `quant_assign.py` then picks per-tensor types for a RAM/VRAM budget.
  - Its GLM-5.1 recipe gives early layers 3-17 iq2_kl and late layers 74-77 iq1_kt. Its V3.1 recipe does the opposite: mid-late layers 26-55 get iq3_kt/iq4_kt.
- **AesSedai** ([GLM-5.3-Flash](https://huggingface.co/AesSedai/GLM-5.3-Flash-GGUF)):
  - Uniform per class, no layer bumps: default / gate / up / down, e.g. IQ2_S = Q6_K / IQ2_XS / IQ2_XS / IQ3_XXS.
  - Keeps MLA k_b/v_b, indexer and ssm tensors at bf16, and MTP layer experts at q8_0.
- **mainline llama.cpp** ([llama-quant.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-quant.cpp)):
  - Q2_K puts ffn_down at q3_K.
  - IQ2/IQ1 put attn_v at q4_K when there are more than 4 experts, and the first n/8 of ffn_down at q2_K.
  - `use_more_bits` = first n/8, last n/8, and every third layer between.
- **V4.1 community** (headers read):

| file | experts gate/up / down | rest | engram | total |
|---|---|---|---|---|
| vcruz Q8_0 (our source) | mxfp4 / mxfp4 | q8_0, embd/output/router bf16 | q8_0 | 473.1 GiB |
| antirez Q2 | iq2_xxs / q2_K | q8_0 | raw FP8 | 340.6 GiB |
| smalinin IQ2XXS_GU-Q2K_Down | iq2_xxs / q2_K | q8_0 | q8_0 | 345.9 GiB |
| smalinin Q2_K-Q5 | q2_K / q2_K | q2_K incl. attention | q5_K | 312.3 GiB |
| smalinin Q2K_Protected | q2_K / q3_K | attn q6_K, shexp q5_K/q6_K, router+embd bf16 | q8_0 | 385.0 GiB |
| pfeifferj GSQ-RCO 3.0bit | q2_K, q3_K on layers 0-13 (all three) | q8_0 | q8_0 | 387.2 GiB |
| apetersson mixedq2 (no imatrix) | iq2_xxs / q2_K | bf16 | none | 157.3 GiB |
| JigSawPT MXFP4-engram | mxfp4 / mxfp4 | q8_0/bf16 | raw FP8 + scales | 467.3 GiB |

## 4. Measured quality (the numbers the rules rest on)

**GLM-5.3-Flash, Unsloth** ([doc](https://unsloth.ai/docs/models/glm-5.3-flash)):

| quant | GB | top-1 | mean KLD |
|---|---|---|---|
| UD-IQ1_S | 93.09 | 70.89% | 0.669714 |
| UD-IQ2_XXS | 101.84 | 76.30% | 0.450148 |
| UD-Q2_K_XL | 108.72 | 78.34% | 0.380134 |
| UD-IQ3_XXS | 120.37 | 81.63% | 0.283772 |
| UD-Q3_K_XL | 147.54 | 86.25% | 0.159697 |
| UD-IQ4_XS | 156.82 | 88.18% | 0.116652 |
| UD-Q4_K_XL | 199.71 | 92.22% | 0.049294 |

**GLM-5.3-Flash, AesSedai** (own eval set, so not directly comparable with Unsloth):

| quant | GiB | KLD |
|---|---|---|
| IQ2_S | 105.81 | 0.375406 |
| IQ3_S | 116.14 | 0.283438 |
| IQ4_XS | 148.24 | 0.117358 |
| Q4_K_M | 188.10 | 0.050181 |

At about the same KLD, Unsloth's files are 4-5 GB smaller (Q2_K_XL 108.7 GB vs IQ2_S 113.6 GB; IQ3_XXS 120.4 GB vs IQ3_S 124.7 GB). That is a small edge from the layer bumps, if the two eval sets behave alike.

**V4.1 Flash, Lucebox** ([card](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-ROCMFP23-GGUF)). These are the only published V4.1 quality numbers. They use the 262,144-token smalinin imatrix (the same one we use) and 8,184 held-out tokens, measured against the fp4 checkpoint (PPL 8.45).

| file | experts | expert GiB | KL | top-1 | PPL |
|---|---|---|---|---|---|
| antirez Q2 | iq2_xxs gate/up, q2_K down | 142 | 0.260 | 78.3% | 9.71 |
| ROCMFP23 | fp2 gate/up, fp3 down | 179 | 0.272 | 77.9% | 9.80 |
| ROCMFP2S | fp2s everywhere | 158 | 0.266 | 77.7% | 9.59 |
| ROCMFP2S-MIX10 | fp2s, fp3 down on layers 0-9 | 164 | 0.246 | 78.3% | 9.45 |

**V4 Flash** ([Unsloth](https://unsloth.ai/docs/models/deepseek-v4), wikitext-2; official 156.4 GB, PPL 4.5319):

| file | GB | KLD |
|---|---|---|
| UD-Q4_K_XL (MXFP4 experts) | 155.1 | 0.0102 |
| antirez Q4KExperts | 164.6 | 0.0291 |
| antirez IQ2XXS (imatrix) | 86.7 | 0.4079 |
| antirez IQ2XXS (no imatrix) | 86.7 | 0.4207 |
| antirez IQ2XXS, layers 37-42 at Q4K | 97.6 | 0.3605 |

[KeinNiemand IK](https://huggingface.co/KeinNiemand/DeepSeek-V4-Flash-0731-IK_GGUF):

| quant | GB | KLD |
|---|---|---|
| IQ2_KS (late layers 20, 22, 26-42 raised) | 93.4 | 0.268909 |
| Atomic AD-IQ2_S | 93.4 | 0.293420 |
| IQ2_KL | 106.3 | 0.205305 |

**ubergarm PPL ladders** (wiki.test.raw, ctx 512):
- [GLM-5.1](https://huggingface.co/ubergarm/GLM-5.1-GGUF), BF16 2.7257:

  | tier | bpw | PPL |
  |---|---|---|
  | IQ3_KS | 3.649 | 2.8780 |
  | IQ2_KL | 2.985 | 3.1275 |
  | smol-IQ2_KS | 2.344 | 3.8818 |
  | smol-IQ1_KT | 1.928 | 4.6654 |

- [DeepSeek V3.1](https://huggingface.co/ubergarm/DeepSeek-V3.1-GGUF), Q8_0 3.3473:

  | tier | bpw | PPL |
  |---|---|---|
  | IQ3_KS | 3.551 | 3.4534 |
  | IQ2_KL | 2.960 | 3.6312 |
  | IQ2_KS | 2.472 | 3.9583 |
  | IQ1_S | 1.710 | 5.3113 |

- [Kimi K2-Thinking](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF), Q4_X 2.0818. This is the clean "down one tier up" A/B:

  | tier | down type | bpw | PPL |
  |---|---|---|---|
  | smol-IQ2_KL | iq2_kl | 2.755 | 2.4550 |
  | IQ2_KL | iq3_ks | 2.920 | 2.3735 |

**Coding, Unsloth V3.1 Aider Polyglot** ([doc](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot)):

| quant | non-thinking | thinking |
|---|---|---|
| 1-bit | 55.7 | 57.8 |
| 2-bit | 65.8 | 66.7 |
| 3-bit | 68.4 | 75.6 |
| full model | 71.6 | 76.1 |

For thinking-mode coding, the 2-to-3-bit step is worth 8.9 points.

## 5. Rules, ordered by evidence strength

1. **Keep non-experts at q8_0.** [strong]
   - Thireus per-Gparam sensitivity: GLM-5.1 attn_k_b 1.78 vs ffn_down_exps 0.0028; Kimi attn_kv_a_mqa 1.69 vs 0.0003; V3.1 shared expert about 0.85 vs routed about 0.005.
   - Unsloth V3.1 Aider: attn_k_b at 4-bit instead of 8-bit lost about 2-3 points. I read this off a chart, so it is approximate; the tensor is about 100 MB.
   - [QuantMoE-Bench](https://arxiv.org/abs/2406.08155): bits given to attention beat bits given to FFN (>5%).
   - Cost: 9.2 GiB (GLM-5.3-Flash), 9.6 GiB (V4.1 without engram).
2. **output.weight is the most sensitive single tensor.** [strong] Keep it at q8_0/bf16, never below q6_K.
   - Thireus: dropping it alone costs +0.542 PPL on V3.1 and +0.516 on R1, vs 0.0345 / 0.0227 for the worst single expert layer.
   - GLM-5.2: KLD 0.1248 vs 0.0121.
3. **Do not go below about 2.6 bpw average on experts for coding.** [strong] Use section 4's tables:
   - GLM-5.3-Flash top-1: 70.9% → 78.3% → 86.3% → 92.2% across IQ1_S, Q2_K_XL, Q3_K_XL, Q4_K_XL.
   - Aider 1/2/3-bit: 55.7 / 65.8 / 68.4.
   - Kimi K3 KLD: 0.5645 (IQ1_S) vs 0.1779 (Q2_K_XL).
4. **Shared experts and dense layers stay at q8_0 (q6_K minimum).** [strong] Check the last layer's shared expert.
   - Thireus outliers: R1 ffn_down_shexp layer 60 0.0972 vs median 0.0033; GLM-5.2 layers 75-76 1.072 vs median 0.0104.
5. **Down gets the gate/up type or one tier more.** [medium]
   - Kimi K2-Thinking: 2.4550 → 2.3735 for +6% size; K2-Instruct 0.2: 3.4086 → 3.1813; K2-0905: 2.9294 → 2.7993.
   - Thireus: down median 1.1-1.3x gate or up alone.
   - Practice: Unsloth GLM keeps down at 3.15 bpw even in IQ1_S, and mainline Q2_K uses a q3_K down.
   - Against: Unsloth Kimi K3 does not raise down. Lucebox's "fp3 down everywhere" did not beat "fp2s everywhere", but its gate/up format also differed, so that comparison is confounded.
6. **Spend extra bits on the measured sensitive layers; the layer order differs by model.** [medium]
   - GLM-5.x and Kimi: early layers. Thireus GLM-5.1 sensitivity by thirds: 0.0155 / 0.0073 / 0.0036; top layers 6-14.
   - DeepSeek V3.1/R1: middle and late layers. V3.1 by thirds: 0.0134 / 0.0248 / 0.0239.
   - V4.1: early down on layers 0-9 is the only V4.1 measurement. KL 0.266 → 0.246 for +6 GiB.
   - V4 Flash: late layers. antirez L37-42 cut KLD 0.4079 → 0.3605 for +10.9 GB.
   - Measured allocation vs a uniform type at equal size: GLM-4.7 PPL 5.1061 vs 5.9716; Kimi K2-Thinking 2.8375 vs 2.9361; GLM-5.1 3.8394 vs 3.8818 (marginal).
7. **Use an imatrix from a chat/code corpus.** [medium] V4 Flash IQ2_XXS with vs without an imatrix: KLD 0.4079 vs 0.4207, PPL 6.0808 vs 6.1518.
8. **V4/V4.1 experts are QAT MXFP4: keep native mxfp4 at the 4-bit tier.** [medium] Requant costs more than on a bf16 model: Unsloth's MXFP4 Q4_K_XL has KLD 0.0102, vs 0.0291 for antirez's larger Q4K-experts file.
9. **For CPU-resident experts, prefer a K-quant one tier up over an iq type.** [medium]
   - Quality favors iq types at equal bytes. In [ik #715](https://github.com/ikawrakow/ik_llama.cpp/discussions/715) on R1-0528 (Q8_0 3.2130), the ik IQ2_KL recipe at 2.930 bpw scores 3.4379 vs UD_Q2_K_XL at 2.994 bpw 3.5278: the PPL gap is 29% smaller.
   - Speed favors K-quants on our AVX2 boxes: they reach 0.87-0.95 of the memory ceiling vs 0.31-0.54 for iq*_ks (1.6-3x).
   - Against: [ik PR 602](https://github.com/ikawrakow/ik_llama.cpp/pull/602) measured AVX2 tg-128 IQ2_KL 16.79 vs Q2_K 14.18 t/s on Llama-8B. We have measured neither iq2_xxs nor iq2_kl ourselves.
10. **Per-expert allocation by routing frequency works, but GGUF cannot express it.** [medium-weak] One type covers all experts of one layer and projection.
    - [MC-MoE](https://arxiv.org/abs/2410.06270) on Mixtral at 2.54 bits: MMLU 61.19 (frequency x routing-weight LP) vs 58.22 (Hessian) vs 31.57 (BSP); 16-bit is 70.60.
    - QuantMoE-Bench: raising the most frequent experts from 4 to 8 bits adds little.
    - [buddalaw](https://huggingface.co/buddalaw/DeepSeek-V4.1-Flash-buddalaw-K160-Q2-GGUF): pruning V4.1 to a domain keeps KLD at 0.098-0.119 in domain but loses badly outside it.
11. **Quantize the MTP/nextn layer with K-quants or q8_0.** [weak, practice only] The imatrix has no data for it. Unsloth uses K-quants there, AesSedai q8_0, ubergarm iq5_ks.
12. **Do not drop engram below q5_K without a KLD test.** [no measurement exists] See section 7.

## 6. Recommended `--custom-q` recipes

Syntax facts, read from ik_llama source (`examples/quantize/quantize.cpp`, `src/llama-quantize.cpp`):
- Rules are split on commas, matched with `std::regex_search`, and the **first match wins**.
- So list layer-specific rules first, and write layer sets as `(0|1|2)`. A comma inside a regex such as `{1,2}` breaks parsing.
- Base type Q8_0 covers everything the rules do not name.
- Keep our dry-run guard that refuses any non-expert type change. With a Q8_0 base, the bf16 embd/output/router in the vcruz source would otherwise turn into q8_0.

Measured on: K-quant rows have never been measured on these models unless the row says so. "Bytes read/token" = routed expert bytes per decoded token (V4.1: top-6 x 40 layers; GLM: top-8 x 42 layers; both about 8.46-8.49 B active expert parameters). Decode tok/s is roughly bandwidth divided by this, plus the non-expert reads.

### (a) GLM-5.3-Flash

Quantize from `unsloth/GLM-5.3-Flash-GGUF` `BF16/` with its `imatrix_unsloth.gguf_file` (both are in the repo). Never requantize from the UD-Q4_K_XL. MTP layer 45 is set to q4_K (+3.8 GiB, held on disk and only touched when MTP is used). Non-expert tensors stay at q8_0: 9.16 GiB, as in the UD-Q4_K_XL header.

| id | expert avg bpw (L3-44) | routed GiB | total GiB | bytes read/token | `--custom-q` |
|---|---|---|---|---|---|
| G22 | 2.250 | 79.7 | 92.7 | 2.38 GB | `blk\.45\.ffn_(gate\|up\|down)_exps=q4_K,ffn_gate_exps=iq2_xxs,ffn_up_exps=iq2_xxs,ffn_down_exps=q2_K` |
| G22u | 2.396 | 84.9 | 97.9 | 2.53 GB | `blk\.45\.ffn_(gate\|up\|down)_exps=q4_K,ffn_gate_exps=iq2_xxs,ffn_up_exps=iq2_xxs,ffn_down_exps=iq3_xxs` |
| **G27** | 2.728 | 96.7 | 109.6 | 2.88 GB | `blk\.45\.ffn_(gate\|up\|down)_exps=q4_K,ffn_gate_exps=q2_K,ffn_up_exps=q2_K,blk\.(3\|4\|5\|6\|7\|8\|9\|10\|11\|12\|13\|14\|15\|16\|17\|44)\.ffn_down_exps=q3_K,ffn_down_exps=q2_K` |
| G29 | 2.896 | 102.6 | 115.6 | 3.06 GB | `blk\.45\.ffn_(gate\|up\|down)_exps=q4_K,ffn_gate_exps=q2_K,ffn_up_exps=q2_K,ffn_down_exps=q3_K` |
| **G32** | 3.237 | 114.7 | 127.7 | 3.42 GB | `blk\.45\.ffn_(gate\|up\|down)_exps=q4_K,blk\.(3\|...\|17\|44)\.ffn_gate_exps=q3_K,ffn_gate_exps=q2_K,` (same for up) `,blk\.(3\|...\|17\|44)\.ffn_down_exps=q4_K,ffn_down_exps=q3_K` |

(The table escapes `|` as `\|`; the JSON holds the exact strings.)

- Layers 3-17 + 44 come from Thireus (GLM-5.1/5.2 top layers 6-14, early third most sensitive) and Unsloth's GLM-5.3-Flash bumps at 11, 12 and 44. GLM-5.3-Flash itself has no sweep yet.
- G22 is the only sub-2.6 option, and it needs iq2_xxs. Our AVX2 speed for it is unknown.
- Expected quality bands come from Unsloth's measured tiers at similar sizes. They are not measurements of these recipes:
  - G22u is close to UD-IQ2_XXS: KLD 0.450, top-1 76.3%.
  - G27/G29 sit between UD-Q2_K_XL (KLD 0.380) and UD-IQ3_XXS (KLD 0.284).

### (b) DeepSeek V4.1 Flash

Source: the vcruz Q8_0 (experts MXFP4) plus the 262,144-token smalinin imatrix, as in `quant-v41.sh`. Non-expert tensors stay as in the source (9.56 GiB). Engram is a separate choice (section 7).

| id | expert avg bpw | routed GiB | +rest GiB | bytes read/token | `--custom-q` | evidence |
|---|---|---|---|---|---|---|
| **V22** | 2.250 | 142.4 | 151.9 | 2.39 GB | `ffn_gate_exps=iq2_xxs,ffn_up_exps=iq2_xxs,ffn_down_exps=q2_K` | measured on V4.1 (antirez layout): KL 0.260, top-1 78.3% |
| V22b (our B) | 2.188 | 138.4 | 148.0 | 2.32 GB | `ffn_(gate\|up\|down)_exps=iq2_ks` | not recommended: uniform iq2_ks is the worst ubergarm tier everywhere, and compute-bound on AVX2 |
| **V27** | 2.693 | 170.4 | 180.0 | 2.86 GB | `ffn_gate_exps=q2_K,ffn_up_exps=q2_K,blk\.(0\|1\|2\|3\|4\|5\|6\|7\|8\|9)\.ffn_down_exps=q3_K,ffn_down_exps=q2_K` | layer choice measured on V4.1 (Lucebox MIX10) |
| V27L | 2.693 | 170.4 | 180.0 | 2.86 GB | same with `blk\.(30\|...\|39)` | A/B twin; late layers win on V4 Flash |
| V29 (our A) | 2.896 | 183.3 | 192.8 | 3.08 GB | `ffn_(gate\|up)_exps=q2_K,ffn_down_exps=q3_K` | layout of smalinin Protected; no quality number |
| **V32** | 3.174 | 200.9 | 210.4 | 3.37 GB | `blk\.(0\|...\|13)\.ffn_gate_exps=q3_K,ffn_gate_exps=q2_K,` (same for up) `,blk\.(0\|...\|9)\.ffn_down_exps=q4_K,ffn_down_exps=q3_K` | layers 0-13 as in pfeifferj (MMLU-Pro 61.00%, PPL 3.4846; RCO kept its first guess) |

- Cross-checks against published sizes: V22 = 142.4 GiB matches Lucebox's "antirez Q2 142 GiB"; V29 = 183.3 GiB matches our script's estimate.
- A 4-bit tier would keep native mxfp4 (268.9 GiB of experts, lossless).
- V27 trades its +0.44 bpw against V22 differently from V29: it puts the extra bits into gate/up quality (q2_K instead of iq2_xxs) and the 10 measured-sensitive down layers. V29 spends about 13 GiB more on the 30 remaining down layers.

## 7. Engram (V4.1 only)

Header facts:
- Two tables, `blk.1.engram_embd` and `blk.14.engram_embd`, each [256, 384,016,682]: 196.6 B parameters, 26% of all V4.1 parameters. (The brief said 384,006,168; the header says 384,016,682.)
- One row is 256 values, exactly one K-quant super-block, so every K-quant fits.

Pair size by type:

| type | GiB | source |
|---|---|---|
| q8_0 | 194.6 | as shipped by vcruz |
| q6_K | 150.2 | |
| q5_K | 125.9 | 135.172 GB in smalinin, which matches |
| q4_K | 103.0 | our variant B |

What exists:
- Engram stored as raw FP8 + scales (antirez, [JigSawPT](https://huggingface.co/JigSawPT/DeepSeek-V4.1-Flash-GGUF)), MXFP8 ([SaturnHeaven](https://huggingface.co/SaturnHeaven/DeepSeek-V4.1-Flash-RANMA-Engram-GGUF), 188.8 GiB), q8_0, and q5_K ([smalinin](https://huggingface.co/smalinin/DeepSeek-V4.1-Flash-GGUF)).
- **No one has published a quality number for any engram type.**
- JigSawPT reads 48 rows per token from the memory-mapped tables. At q8_0 that is about 13 KB per token, so the type has no effect on bandwidth; it only decides how much page cache the tables need. JigSawPT also notes that taking page cache away from the tables slows decode.
- Closest analogue, token_embd: Thireus per-Gparam sensitivity is 9-11x a routed expert layer (R1 0.0298 vs 0.0028; GLM-5.1 0.0255 vs 0.0028).

Advice:
- Keep q8_0 where it fits in page cache.
- If capacity forces a cut, use q6_K or q5_K and run a KLD check against q8_0 first. This is one cheap run: it changes one tensor pair.
- Do not use q4_K (variant B) blind.

## 8. Where the evidence pushes against our plan

Our plan: coding-hot experts at 3-4 bit, the rest at 2 bit, all non-expert tensors high.

1. **"All non-expert high" is confirmed (rules 1, 2, 4)**, except for engram. Engram is non-expert, yet at q8_0 it is larger than all the experts at V29 (194.6 vs 183.3 GiB), and nobody has measured it. That is the one "high" item worth testing before we pay for it.
2. **Per-expert precision ("coding-hot") cannot be expressed in standard GGUF.** One type covers all 384 (or 288) experts of a layer's projection. It needs our split-expert path. Evidence that it pays comes only from Mixtral-era papers (MC-MoE, QuantMoE-Bench). Those also say 4→8 bits on hot experts adds little, so 3-4 bit hot is the right ceiling.
3. **"The rest at 2 bit" depends on which 2 bits.**
   - Only q2_K-class (2.6 bpw) gate/up with down at q2_K/q3_K has evidence behind it.
   - Uniform iq2_ks everywhere (our variant B) goes against rules 5 and 9. It is the worst tier on every ubergarm ladder, and it is compute-bound on our CPUs.
   - Experts below about 2.6 bpw on average fall off the coding cliff (rule 3).
4. **Layer position matters about as much as expert identity, and the plan has no layer axis.** On V4.1, fp3 down on just 10 layers bought 7.5% lower KL for 6 GiB. V4 Flash points at late layers, V4.1 at early ones, so the V27 vs V27L A/B decides this on V4.1.
5. **Our CPU constraint narrows the choices.** Below 2.625 bpw (q2_K) there is no K-quant. A "2-bit" CPU tier therefore means q2_K. The cheaper V22 needs iq2_xxs, whose AVX2 speed in ik we have not measured.

## 9. What to measure next (in order, each one run)

1. **AVX2 decode speed of iq2_xxs and iq2_kl vs q2_K** on one expert-heavy model. This decides whether V22/G22 are CPU options at all, and checks ik PR 602 against our iq*_ks result.
2. **KLD of V27 vs V27L vs V29 on V4.1.** Use the Lucebox method: held-out tokens against the vcruz Q8_0, top-1, KL.
3. **Engram q5_K vs q8_0 KLD,** with everything else fixed.
4. **A Thireus-style per-layer sweep on GLM-5.3-Flash and V4.1.** Drop one layer's down (then gate+up) to iq1/q2 and record whole-model KLD. Spend bits by the result, then size the hot-expert tier inside the chosen layers.

## Sources not linked inline

- Thireus recipes: `recipe_examples/ik_harmonized_recipes` (DeepSeek-V3.1 2.6124 bpw, GLM-5.1 2.3453 bpw).
- [ubergarm GLM-4.6/4.7/GLM-5/V3.1-Terminus/K2 cards](https://huggingface.co/ubergarm).
- [GSQ](https://arxiv.org/abs/2604.18556) and [RCO](https://arxiv.org/abs/2605.00649) (IST-DASLab, KL-driven per-tensor allocation under an exact budget).
- [pfeifferj GSQ-RCO](https://huggingface.co/pfeifferj/DeepSeek-V4.1-Flash-GSQ-RCO-GGUF).
- [apetersson mixedq2](https://huggingface.co/apetersson/DeepSeek-V4.1-Flash-MixedQ2-GGUF). Reconstruction cosine minimum: IQ2_XXS 0.9407 vs Q2_K 0.9633.
- [antirez V4.1](https://huggingface.co/antirez/deepseek-v4.1-flash-gguf).
- [vcruz305 V4.1](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF).
- Unsloth repos: [R1](https://huggingface.co/unsloth/DeepSeek-R1-GGUF), [V3.1](https://huggingface.co/unsloth/DeepSeek-V3.1-GGUF), [V3.2](https://huggingface.co/unsloth/DeepSeek-V3.2-GGUF), [V4 Flash 0731](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF), [GLM-4.6](https://huggingface.co/unsloth/GLM-4.6-GGUF), [Kimi K2](https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUF), [Kimi K3](https://huggingface.co/unsloth/Kimi-K3-GGUF).
