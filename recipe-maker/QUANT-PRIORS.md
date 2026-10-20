# Quant priors: what the internet already measured

Desk research, 2026-10-17. Nothing was run on any homelab box. Machine-readable twin: `quant-priors.json` (same folder).

Two tags. **MEASURED** = copied from the linked source. **INFERRED** = our derivation from the linked anchors; nobody measured it. Unknown stays unknown.

**Harness warning.** KLD is not comparable across harnesses: DeepSeek V4 Flash UD-Q4_K_XL scores [0.0102](https://unsloth.ai/docs/models/deepseek-v4) in unsloth's ctx-512 harness and [0.1557](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) in AtomicChat's ctx-5632 harness. Compare KLD only inside one table. Across sources use PPL ratio, top-1 agreement or benchmark retention.

## 1. Quality curve (INFERRED, large sparse MoE)

bpw = whole-model bits per weight as the cards report it, for large sparse MoE (routed experts ~95% of weights, so expert bpw is ~0.1-0.3 lower). pct_quality = coding/agentic benchmark score as % of the full model, for a GOOD recipe (imatrix + attention/shared kept at 6-8 bit); lo = plain or unlucky recipes, hi = best measured recipe. kld_typical = mean KLD vs BF16 in a llama.cpp-style harness at ctx 512 (unsloth/AesSedai tables); other harnesses differ by up to 10x. ppl_delta_pct = median PPL increase across 17 ubergarm ik_llama.cpp model tables in that bin. All INFERRED-BY-US from the anchors listed.

| bpw | % of full on coding | spread | typical KLD (llama.cpp ctx 512) | KLD spread | median PPL rise | anchors |
|---|---|---|---|---|---|---|
| 8 | 100% | 99.5-100% | 0.01 | 0.005-0.015 | +0.02% | [1](https://huggingface.co/AesSedai/GLM-5.3-GGUF) [2](https://unsloth.ai/docs/models/deepseek-v4) |
| 6 | 99% | 96-100% | 0.025 | 0.019-0.03 | +0.2% | [1](https://unsloth.ai/docs/models/glm-5.3-flash) [2](https://unsloth.ai/docs/models/glm-5.3) [3](https://huggingface.co/AesSedai/GLM-5.3-Flash-GGUF) [4](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) |
| 5 | 98% | 95-99.8% | 0.045 | 0.035-0.05 | +0.8% | [1](https://unsloth.ai/docs/models/glm-5.3-flash) [2](https://huggingface.co/AesSedai/GLM-5.3-Flash-GGUF) [3](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) [4](https://arxiv.org/abs/2505.02390) [5](https://huggingface.co/Intel/GLM-5.3-Flash-W4A16-AutoRound) [6](https://huggingface.co/unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF/discussions/8) |
| 4 | 96% | 92-99.5% | 0.1 | 0.05-0.12 | +2.4% | [1](https://unsloth.ai/docs/models/glm-5.3-flash) [2](https://unsloth.ai/docs/models/glm-5.3) [3](https://huggingface.co/AesSedai/GLM-5.3-Flash-GGUF) [4](https://huggingface.co/AesSedai/Kimi-K2.7-Code-GGUF) [5](https://huggingface.co/ubergarm/DeepSeek-V3.1-GGUF) [6](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) |
| 3.5 | 95% | 93-99% | 0.15 | 0.06-0.18 | +4.3% | [1](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) [2](https://arxiv.org/abs/2505.02390) [3](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF) [4](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF/discussions/14) [5](https://unsloth.ai/docs/models/glm-5.3-flash) [6](https://unsloth.ai/docs/models/glm-5.3) |
| 3 | 92% | 85-97% | 0.27 | 0.18-0.29 | +14.0% | [1](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) [2](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF) [3](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF/discussions/14) [4](https://unsloth.ai/docs/models/glm-5.3-flash) [5](https://unsloth.ai/docs/models/glm-5.3) [6](https://huggingface.co/AesSedai/GLM-5.3-Flash-GGUF) [7](https://huggingface.co/AesSedai/Kimi-K2.7-Code-GGUF) |
| 2.5 | 87% | 74-96% | 0.4 | 0.27-0.45 | +32.9% | [1](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) [2](https://arxiv.org/abs/2505.02390) [3](https://unsloth.ai/docs/models/glm-5.3-flash) [4](https://unsloth.ai/docs/models/glm-5.3) [5](https://huggingface.co/AesSedai/Kimi-K2.7-Code-GGUF) [6](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) |
| 2 | 77% | 61-89% | 0.55 | 0.39-0.7 | +67.6% | [1](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) [2](https://arxiv.org/abs/2410.06270) [3](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) [4](https://unsloth.ai/docs/models/glm-5.3-flash) [5](https://huggingface.co/KeinNiemand/DeepSeek-V4-Flash-0731-IK_GGUF) |
| 1.5 | 70% | 55-76% | 0.65 | 0.54-unknown | +61.6% | [1](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) [2](https://arxiv.org/abs/2410.06270) [3](https://unsloth.ai/docs/models/kimi-k3) [4](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) |

- **8 bpw**: Q8_0 is noise-level: GLM-5.3 Q8_0 PPL +0.017%, KLD 0.0131 ([src](https://huggingface.co/AesSedai/GLM-5.3-GGUF), [src](https://unsloth.ai/docs/models/deepseek-v4)).
- **6 bpw**: V3.1 Aider UD-Q5_K_XL 98.7% (within run noise of full) ([src](https://unsloth.ai/docs/models/glm-5.3-flash), [src](https://unsloth.ai/docs/models/glm-5.3), [src](https://huggingface.co/AesSedai/GLM-5.3-Flash-GGUF)).
- **5 bpw**: Q4_K_M/Q4_K_XL class (~4.8-5.0 bpw overall): V3.1 Aider 97.3%, R1 LCB 97.3%, V3-0324 LCB 96.3%, Qwen3-Coder Aider 98.5%, Intel W4A16 99.84% ([src](https://unsloth.ai/docs/models/glm-5.3-flash), [src](https://huggingface.co/AesSedai/GLM-5.3-Flash-GGUF), [src](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot)).
- **4 bpw**: IQ4_XS class: GLM-5.3-Flash KLD 0.117, PPL +7.0%; Kimi-K2.7 Q3_K_L 3.85 KLD 0.063; V3.1 community IQ4_XS Aider 92.6% ([src](https://unsloth.ai/docs/models/glm-5.3-flash), [src](https://unsloth.ai/docs/models/glm-5.3), [src](https://huggingface.co/AesSedai/GLM-5.3-Flash-GGUF)).
- **3.5 bpw**: V3.1 Aider UD-Q3_K_XL 95.5% (thinking 99.3%); R1 DQ3_K_M 98.4% LCB; V3-0324 Q3_K_M LCB 93.0%; Kimi-K2-Thinking smol-IQ3_KS 96.6%; GLM-5.3-Flash GSQ 3.5 MMLU-Pro 97.7% ([src](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot), [src](https://arxiv.org/abs/2505.02390), [src](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF)).
- **3 bpw**: IQ3_XXS/IQ3_S class: V3.1 Aider UD-IQ3_XXS 93.3%; GLM-5.3-Flash GSQ 3.0 MMLU-Pro 96.9% (math 88%); Kimi-K2-Thinking UD-IQ3_XXS ~85%; GLM-5.3-Flash KLD 0.284, PPL +22.9% ([src](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot), [src](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF), [src](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF/discussions/14)).
- **2.5 bpw**: V3.1 Aider UD-Q2_K_XL 91.9% (thinking 87.6%); R1 UD-Q2_K_XL LCB 95.7%; plain V3-0324 Q2_K_L (2.91) LCB 74.3%; GLM-5.3-Flash UD-IQ2_XXS KLD 0.450 ([src](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot), [src](https://arxiv.org/abs/2505.02390), [src](https://unsloth.ai/docs/models/glm-5.3-flash)).
- **2 bpw**: V3.1 Aider UD-IQ1_M (1.93 MoE bpw) 77.8%; community IQ2_XXS 60.9%; MC-MoE PMQ 2.05 on Mixtral 88.7% (general, not coding) ([src](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot), [src](https://arxiv.org/abs/2410.06270), [src](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF)).
- **1.5 bpw**: V3.1 Aider UD-TQ1_0 (1.66 MoE bpw) 70.8%; MC-MoE PMQ 1.57 76.4%; ppl_delta is the ~1.75 bpw bin (no 1.5 tables); kld_hi unknown ([src](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot), [src](https://arxiv.org/abs/2410.06270), [src](https://unsloth.ai/docs/models/kimi-k3)).

## 2. Verdict: does 3-bit do ~95% as well as 4-bit?

**TRUE on benchmark scores for a good 3-bit (imatrix/dynamic, 3.0-3.7 bpw); FALSE as a statement about how close the model stays to the original token by token.**

For:

- [DeepSeek-V3.1 Aider (non-thinking): UD-Q3_K_XL 68.4 / UD-Q4_K_XL 69.7 = 98.1%; UD-IQ3_XXS 66.8 / 69.7 = 95.8%](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot)
- [DeepSeek-V3.1 Aider community quants: IQ3_XXS 65.6 / Q4_K_M 67.7 = 96.9%; Q3_K_S 65.2 / 67.7 = 96.3%](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot)
- [DeepSeek-R1 LiveCodeBench: Q3_K_M 61.95 / Q4_K_M 62.41 = 99.3%](https://arxiv.org/abs/2505.02390)
- [DeepSeek-V3-0324 LiveCodeBench: Q3_K_M 46.23 / Q4_K_M 47.88 = 96.6%](https://arxiv.org/abs/2505.02390)
- [Kimi-K2-Thinking Aider (native INT4, so 4-bit = full): smol-IQ3_KS 77.3 vs ~80 = 96.6%](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF/discussions/14)
- [GLM-5.3-Flash MMLU-Pro: GSQ-RCO 3.0-bit 96.9% and 3.5-bit 97.7% of Q8_0](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF)

Against:

- [GLM-5.3-Flash KLD: UD-IQ3_XXS (3.01 bpw) 0.284 vs UD-Q4_K_XL (4.99) 0.049, 5.8x; PPL IQ3_S +22.9% vs Q4_K_M +1.83%](https://unsloth.ai/docs/models/glm-5.3-flash)
- [Kimi-K2-Thinking Aider: unsloth UD-IQ3_XXS ~68 vs ~80 = ~85% (single community run)](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF/discussions/14)
- [Hard tasks lose more: GSQ 3.0 math 49.8 -> 44.0 (-11.6%) while overall MMLU-Pro lost 3.1%](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF)
- [Uniform (non-mixed) 3-bit GPTQ on Mixtral: MMLU 60.58 vs 67.88 (89%)](https://arxiv.org/abs/2410.06270)
- [Quantization Hurts Reasoning: W4 near-lossless, W3 risky, harder tasks degrade up to 4x more](https://arxiv.org/abs/2504.04823)

Plan on a good 3.5-bit (Q3_K_XL / IQ3_KS / DQ3 class) scoring 95-99% of the 4-bit on coding, and a 3.0-bit (IQ3_XXS class) 90-97%. But the 3-bit drifts 3-6x further from the original per token, so long agentic runs and hard reasoning fall off faster than the benchmark ratio suggests. For DeepSeek V4/V4.1 and Kimi K2-Thinking+ the 4-bit IS the full model (quantization-aware trained), so there '3 vs 4' means '3 vs full'; no coding benchmark of a 3-bit V4/V4.1 exists yet (unknown).

## 3. Mixed precision: what keeping some tensors high buys (MEASURED)

- **A 3.6-bit dynamic recipe beat a uniform 4.8-bit one**: [DQ3_K_M (3.59 bpw) avg drop 0.34% vs Q4_K_M (4.82) 0.68% on R1; 0.30% vs 1.35% on V3-0324](https://arxiv.org/abs/2505.02390). Recipe: routed down_exps q3_k/q4_k/q6_k by layer, attention kv q6_k, shared experts q6_k/q4_k.
- **Dynamic 2-bit vs plain 2-bit (cross-model, same paper)**: [R1 UD-Q2_K_XL (2.70) avg drop 0.94%, LCB 95.7%; V3-0324 plain Q2_K_L (2.91) avg drop 14.66%, LCB 74.3%](https://arxiv.org/abs/2505.02390). Recipe: unsloth keeps attention, shared experts and first dense layers high.
- **Same Aider bench, unsloth dynamic vs community recipes at matching names**: [UD-Q4_K_XL 69.7 vs Q4_K_M 67.7; UD-IQ3_XXS 66.8 vs IQ3_XXS 65.6; UD-Q2_K_XL 65.8 vs Q2_K_L 64.0 vs IQ2_M 56.6 vs IQ2_XXS 43.6 (full 71.6)](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot). Recipe: unsloth dynamic.
- **One rung up on ffn_down_exps only**: [+4.03 GiB buys KLD 0.2629 -> 0.2386 (-9%), top-1 83.91 -> 84.65% on V4 Flash 0731](https://huggingface.co/TacoTakumi/DeepSeek-V4-Flash-0731-GGUF). Recipe: IQ3_XXS gate/up + IQ3_S down; attention/shared Q8_0.
- **fp3 down projection on the first 10 layers only (V4.1 Flash)**: [KL 0.266 -> 0.246, PPL 9.59 -> 9.45](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-ROCMFP23-GGUF). Recipe: ROCMFP2S + fp3 down layers 0-9.
- **IQ2_XXS gate/up + Q2_K down beat a larger 2/3-bit FP format (V4.1 Flash)**: [antirez Q2 142 GiB KL 0.260 vs ROCMFP23 179 GiB KL 0.272](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-ROCMFP23-GGUF). Recipe: antirez Q2.
- **ik_llama.cpp trellis quants beat mainline i-quants per bit (same harness)**: [V4 Flash IQ2_KL 2.99 bpw KLD 0.2053 vs AtomicChat IQ3_XXS 2.91 0.2495; IQ1_KT 2.06 0.3932 vs IQ2_XXS 2.06 0.4544](https://huggingface.co/KeinNiemand/DeepSeek-V4-Flash-0731-IK_GGUF). Recipe: ik IQ*_KT/KS/KL.
- **EXL3 (trellis, GPU-only) vs GGUF at the same size (turboderp harness)**: [GLM-5.3-Flash EXL3 3.05 KLD 0.068 vs UD-IQ3_XXS ~2.98 0.126; EXL3-SC 2.60 0.093](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3). Recipe: exllamav3 trellis + SC variant.
- **Mixed per-expert precision vs uniform at the same average bits (Mixtral)**: [PMQ 2.54 bits avg 67.50 vs GPTQ uniform 2-bit ~43.2 and block-sparsity allocation 49.07 (16-bit 71.29)](https://arxiv.org/abs/2410.06270). Recipe: activation frequency + importance per expert.
- **Frequent-expert-at-4-bit helps at 2.5 bits, not at 3**: [Mixtral 2.54 avg: 54.20 vs random 49.10 +/- 7.73; 3.03 avg: 63.82 vs 63.70 +/- 0.49; DeepSeek-MoE-16B 2.53: 62.99 vs 62.86 +/- 0.60](https://arxiv.org/abs/2406.08155). Recipe: top-k frequent experts per layer at 4 bit, rest 2.
- **Attention bits are cheap and valuable**: [+Attn at 2.06 bits: +5.47 points (46.00); shared experts and early layers need more bits](https://arxiv.org/abs/2406.08155). Recipe: attention to 4 bit.
- **Hot experts are concentrated (GLM-5.3-Flash)**: [~100 most-used of 288 experts per layer cover ~74% of picks in real chat output](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF). Recipe: usage-ranked hot-expert cache.
- **Pruning by usage frequency fails; router-weighted saliency on coding data holds**: [REAP coding-calibrated 50% prune: -1.2% on coding (25%: -0.16%); frequency pruning at 25% on Kimi-K2: 0.486 vs 0.828; C4 calibration makes code generation 'abysmal'; 4-bit + 50% prune beats 2-bit](https://arxiv.org/abs/2510.13999). Recipe: REAP.
- **Attention k_b kept at 8 bit (V3.1)**: [unsloth reports a large Aider gain for ~100 MB extra (exact delta not given in text)](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot). Recipe: attn_k_b 8-bit instead of 4.

## 4. Coding degrades faster than general knowledge (MEASURED)

Coding, maths and agentic work lose about 3-10x more than MMLU-style knowledge at the same quant (median ~3x for R1, ~9x for V3-0324), and the gap widens below 3 bpw. Judge quants for coding on coding benchmarks or KLD, never on MMLU.

| model | quant | coding/maths loss | general loss | ratio |
|---|---|---|---|---|
| DeepSeek-V3-0324 | Q2_K_L 2.91 | [LiveCodeBench -25.7%](https://arxiv.org/abs/2505.02390); AIME -46% | [MMLU -3.9%](https://arxiv.org/abs/2505.02390) | 6.6x |
| DeepSeek-V3-0324 | Q3_K_M 3.81 | [LiveCodeBench -7.0%](https://arxiv.org/abs/2505.02390) | [MMLU -0.7%](https://arxiv.org/abs/2505.02390) | 10.0x |
| DeepSeek-V3-0324 | Q4_K_M 4.82 | [LiveCodeBench -3.7%](https://arxiv.org/abs/2505.02390) | [MMLU -0.4%](https://arxiv.org/abs/2505.02390) | 9.0x |
| DeepSeek-R1 | Q4_K_M 4.82 | [LiveCodeBench -2.7%](https://arxiv.org/abs/2505.02390) | [MMLU -0.9%](https://arxiv.org/abs/2505.02390) | 2.9x |
| DeepSeek-R1 | Q3_K_M 3.81 | [LiveCodeBench -3.4%](https://arxiv.org/abs/2505.02390) | [MMLU -1.2%](https://arxiv.org/abs/2505.02390) | 2.8x |
| DeepSeek-R1 | UD-Q2_K_XL 2.70 | [LiveCodeBench -4.3%](https://arxiv.org/abs/2505.02390) | [MMLU -1.4%](https://arxiv.org/abs/2505.02390) | 3.1x |
| GLM-5.3-Flash | GSQ-RCO 3.0 | [MMLU-Pro math -11.6% (no coding category reported)](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF) | [MMLU-Pro -3.1%](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF) | 3.7x |
| Qwen3.8-27B (dense) | UD-Q2_K_XL vs UD-IQ2_S | [32-token greedy match on agentic/coding prompts ~25% -> under 8-10%](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs); unsloth: 1-bit should not be used for agentic or tool calling | [top-1 still ~72-77%](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs) | n/ax |

## 5. What to expect from the builds we plan (INFERRED)

### DeepSeek V4.1 Flash: coding-hot experts at native 4-bit (MXFP4, untouched), the rest at ~2.25-bit (IQ2_XXS gate/up + Q2_K down)

- Routed experts average **2.94 bpw = ~200 GB** (native fp4 all experts ~289 GB, all ~2.25-bit ~153 GB; 543.58B routed params from the [official config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)).
- Expected: `{"kl": 0.068, "kl_lo": 0.039, "kl_hi": 0.13, "ppl_delta_pct": 3.9, "ppl_delta_lo": 2.2, "ppl_delta_hi": 7.5, "top1_pct": 94.4, "coding_retention_pct": [90, 96]}`
- How: Linear in coverage: error ~= (share of routing picks that land on 2-bit experts) x (error of the all-2-bit build). All-2-bit V4.1 = KL 0.260, PPL +14.9%, top-1 78.3% (Lucebox); hot share 74% (range 50-85%) from GLM-5.3-Flash chat; hot set = 100/288 of experts. KL ~0.07 sits where a uniform ~3-bit V4 build sits (AtomicChat delta-KLD 0.050-0.093 at 3.2-2.9 expert bits), so retention is read off the 3-bit row of the curve.
- Anchors: [1](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-ROCMFP23-GGUF) [2](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF) [3](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) [4](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) [5](https://arxiv.org/abs/2510.13999) [6](https://arxiv.org/abs/2406.08155)
- Caveat: V4.1 routing concentration is UNMEASURED (384 experts, 6 active); the 74% is GLM-5.3-Flash chat traffic
- Caveat: at the same bytes this is about equal to a uniform ~3-bit build; the gain is speed (hot experts in VRAM), not quality per byte
- Caveat: pick the hot set from coding traffic with router-weighted saliency, never raw frequency or C4/chat text (REAP)
- Caveat: frequency allocation only helps when routing is unbalanced (QuantMoE-Bench): measure V4.1's balance first
- Caveat: stock llama.cpp cannot give per-expert types inside one merged tensor; needs a fork (Lucebox PR 593 codebooks, hot-expert cache PR) or our split engine

### DeepSeek V4.1 Flash: coding-hot experts at ~3-bit, the rest at ~2.25-bit

- Routed experts average **2.55 bpw = ~173 GB** (native fp4 all experts ~289 GB, all ~2.25-bit ~153 GB; 543.58B routed params from the [official config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)).
- Expected: `{"kl": 0.12, "kl_lo": 0.105, "kl_hi": 0.136, "ppl_delta_pct": [6, 8], "coding_retention_pct": [87, 93]}`
- How: Same coverage model: 0.74 x (3-bit hot error) + 0.26 x 0.260. 3-bit hot error = V4 Flash 0731 delta-KLD over native at 3.20 and 2.91 expert bits (0.050, 0.093). No 3-bit V4.1 measurement exists. Result KL 0.105-0.136 matches a uniform ~2.7-2.8-bit V4 build (delta 0.100 at 2.79), better than uniform 2.5 (0.178).
- Anchors: [1](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-ROCMFP23-GGUF) [2](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) [3](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF)
- Caveat: requantizing native MXFP4 hot experts to 3-bit throws away lossless bits for ~25% less hot-set memory; worth it only if the hot set then fits VRAM and the 4-bit one does not
- Caveat: PPL range is our scaling of the all-2-bit +14.9% by the KL ratio

### GLM-5.3-Flash at 3.05 bpw

- Expected: `{"gguf": {"kld_unsloth_harness": 0.284, "top1_pct": 81.6, "ppl_delta_pct": [9, 23], "general_retention_pct": [96, 97], "coding_retention_pct": [90, 95]}, "exl3": {"kld_turboderp_harness": 0.068, "coding_retention_pct": [93, 97]}}`
- How: Measured directly, then mapped to coding with the curve. GGUF near 3.0: unsloth UD-IQ3_XXS 3.01 bpw KLD 0.284, top-1 81.63%; AesSedai IQ3_S 3.11 PPL +22.88%; GSQ-RCO 3.0 (2.94 bpw) MMLU-Pro 96.9%, PPL +9.07%, math -11.6%. EXL3 3.05: KLD 0.0681 where AWQ int4 scores 0.0565 and UD-IQ4_XS 0.0615 in the same harness, so EXL3 3.05 ~ a GGUF 4-bit. Coding range from V3.1 Aider UD-IQ3_XXS 93.3% and the 3-10x coding multiplier on the 3.1% MMLU-Pro loss.
- Anchors: [1](https://unsloth.ai/docs/models/glm-5.3-flash) [2](https://huggingface.co/AesSedai/GLM-5.3-Flash-GGUF) [3](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF) [4](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) [5](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) [6](https://arxiv.org/abs/2505.02390)
- Caveat: GLM is BF16-native (no QAT), so even 4-bit loses: Q4_K_M PPL +1.83%
- Caveat: GLM-family PPL rises faster per bit than DeepSeek in ubergarm's tables (IQ2_KL ~3.0: GLM-4.6 +20%, GLM-5.1 +15% vs V3.1 +8.5%)
- Caveat: EXL3 is GPU-only: 3.05 bpw x 320B ~ 122 GB must sit in VRAM

### GLM-5.3-Flash at ~2.5 bpw

- Expected: `{"gguf": {"kld_unsloth_harness": 0.45, "top1_pct": 76.3, "ppl_delta_pct": [33, 44], "coding_retention_pct": [78, 88]}, "exl3_sc_2_60": {"kld_turboderp_harness": 0.0929, "coding_retention_pct": [90, 95]}}`
- How: GGUF: unsloth UD-IQ2_XXS 2.55 bpw KLD 0.4501, top-1 76.30%; AesSedai IQ2_S 2.83 PPL +33.37%; ubergarm GLM-5/5.1 smol-IQ2_KS 2.344 PPL +43.7%/+42.4%. Coding range from V3.1 Aider 2-bit 87.6-91.9%, lowered because GLM loses more PPL per bit than DeepSeek. EXL3-SC 2.60 KLD 0.0929 is better than GGUF UD-IQ3_XXS (0.1258) in the same harness, so it maps to the GGUF ~3-3.5 bpw row.
- Anchors: [1](https://unsloth.ai/docs/models/glm-5.3-flash) [2](https://huggingface.co/AesSedai/GLM-5.3-Flash-GGUF) [3](https://huggingface.co/ubergarm/GLM-5.1-GGUF) [4](https://huggingface.co/ubergarm/GLM-5-GGUF) [5](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) [6](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot)
- Caveat: unsloth: below UD-Q2_K_XL agentic 32-token agreement collapses (25% -> under 10% on Qwen3.8-27B)
- Caveat: EXL3-SC 2.6 needs ~104 GB VRAM

## 6. Do not bother testing

- **Any 1-bit build (IQ1_S/IQ1_M/TQ1_0/IQ1_KT) for coding or agents**. [V3.1 Aider 70.8-77.8% of full; unsloth: 'should not be used for agentic use-cases', tool calling breaks](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs)
- **Plain (non-imatrix, non-dynamic) IQ2_XXS or IQ2_M**. [V3.1 Aider 43.6 and 56.6 vs 71.6 full, while UD-Q2_K_XL at similar size scores 65.8](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot)
- **Plain Q2_K_L on DeepSeek V3-class**. [V3-0324 LiveCodeBench 74.3% of FP8, AIME -46%](https://arxiv.org/abs/2505.02390)
- **Requantizing V4/V4.1 native MXFP4 experts to Q4_K**. [antirez Q4KExperts is larger (164.6 GB vs 155.1) and worse (KLD 0.029 vs 0.0102) than keeping MXFP4](https://unsloth.ai/docs/models/deepseek-v4)
- **Anything above native 4.25 bits for V4/V4.1 routed experts**. [native MXFP4 is the trained weight; UD-Q8_K_XL is bit-exact, so more bits buy nothing](https://unsloth.ai/docs/models/deepseek-v4)
- **ROCMFP23-style 2/3-bit FP experts for V4.1 on our CPU/GPU path**. [179 GiB for KL 0.272 vs antirez Q2 142 GiB KL 0.260](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-ROCMFP23-GGUF)
- **REAP-pruned models re-quantized to 2-bit experts, and any pruning or low-bit build calibrated on C4/chat text for coding**. [Tinker-Stack: IQ2_S experts +7.5% PPL on the pruned model, 'Do not requantize the experts down to 2-bit'; REAP: C4 calibration makes code generation 'abysmal'](https://arxiv.org/abs/2510.13999)
- **Expert ranking by raw activation frequency (prune or demote)**. [REAP: frequency pruning at 25% on Kimi-K2 scored 0.486 vs 0.828; QuantMoE-Bench: no gain at 3 bits](https://arxiv.org/abs/2510.13999)
- **Q5 and up for GLM-5.3-Flash**. [UD-Q5_K_XL 240 GB buys KLD 0.049 -> 0.027 and top-1 +2.1 pts over UD-Q4_K_XL (200 GB); Q6 adds 52 GB more for +0.9 pts](https://unsloth.ai/docs/models/glm-5.3-flash)
- **Nested-bit (Matryoshka) 2-bit prefixes as the V4.1 2-bit tier** (MEASURED BY US, homelab lab notes 2a: 192 V4.1-1017 expert tensors vs native MXFP4, weight-space rel MSE). The best nested prefix only equals a direct scalar 2-bit (0.115) and has 1.58x the error of Q2_K (0.073); the native-MXFP4 relabelled prefix has 2.05x. IQ2_S has 0.94x the Q2_K error at 0.125 bpw less: KL-test that instead.
- **Re-testing quants the internet already ranked inside one harness**. Use the tables in this file; spend our runs on recipes nobody measured (per-expert hot/cold on V4.1, coding-calibrated)

## 7. Every measured point (MEASURED)

bpw columns are as the source reports them (overall = whole file; experts = routed experts only); blank = not given. Where we computed bpw from file size the note says so.

### GLM-5.3-Flash

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| UD-IQ1_S | 2.33 |  | mean_kld | [0.6697](https://unsloth.ai/docs/models/glm-5.3-flash) |  | unsloth dynamic: attention/shared/dense layers higher | unsloth: llama.cpp KLD vs BF16, top-1 agreement | size 93.09 GB; bpw computed GB*8/320 | 2026-08-26 |
| UD-IQ1_S | 2.33 |  | top1_pct | [70.89](https://unsloth.ai/docs/models/glm-5.3-flash) | 100 | unsloth dynamic | unsloth: llama.cpp KLD vs BF16, top-1 agreement |  | 2026-08-26 |
| UD-IQ1_M | 2.44 |  | mean_kld | [0.5724](https://unsloth.ai/docs/models/glm-5.3-flash) |  | unsloth dynamic: attention/shared/dense layers higher | unsloth: llama.cpp KLD vs BF16, top-1 agreement | size 97.58 GB; bpw computed GB*8/320 | 2026-08-26 |
| UD-IQ1_M | 2.44 |  | top1_pct | [73.06](https://unsloth.ai/docs/models/glm-5.3-flash) | 100 | unsloth dynamic | unsloth: llama.cpp KLD vs BF16, top-1 agreement |  | 2026-08-26 |
| UD-IQ2_XXS | 2.55 |  | mean_kld | [0.4501](https://unsloth.ai/docs/models/glm-5.3-flash) |  | unsloth dynamic: attention/shared/dense layers higher | unsloth: llama.cpp KLD vs BF16, top-1 agreement | size 101.84 GB; bpw computed GB*8/320 | 2026-08-26 |
| UD-IQ2_XXS | 2.55 |  | top1_pct | [76.3](https://unsloth.ai/docs/models/glm-5.3-flash) | 100 | unsloth dynamic | unsloth: llama.cpp KLD vs BF16, top-1 agreement |  | 2026-08-26 |
| UD-Q2_K_XL | 2.72 |  | mean_kld | [0.3801](https://unsloth.ai/docs/models/glm-5.3-flash) |  | unsloth dynamic: attention/shared/dense layers higher | unsloth: llama.cpp KLD vs BF16, top-1 agreement | size 108.72 GB; bpw computed GB*8/320 | 2026-08-26 |
| UD-Q2_K_XL | 2.72 |  | top1_pct | [78.34](https://unsloth.ai/docs/models/glm-5.3-flash) | 100 | unsloth dynamic | unsloth: llama.cpp KLD vs BF16, top-1 agreement |  | 2026-08-26 |
| UD-IQ3_XXS | 3.01 |  | mean_kld | [0.2838](https://unsloth.ai/docs/models/glm-5.3-flash) |  | unsloth dynamic: attention/shared/dense layers higher | unsloth: llama.cpp KLD vs BF16, top-1 agreement | size 120.37 GB; bpw computed GB*8/320 | 2026-08-26 |
| UD-IQ3_XXS | 3.01 |  | top1_pct | [81.63](https://unsloth.ai/docs/models/glm-5.3-flash) | 100 | unsloth dynamic | unsloth: llama.cpp KLD vs BF16, top-1 agreement |  | 2026-08-26 |
| UD-Q3_K_XL | 3.69 |  | mean_kld | [0.1597](https://unsloth.ai/docs/models/glm-5.3-flash) |  | unsloth dynamic: attention/shared/dense layers higher | unsloth: llama.cpp KLD vs BF16, top-1 agreement | size 147.54 GB; bpw computed GB*8/320 | 2026-08-26 |
| UD-Q3_K_XL | 3.69 |  | top1_pct | [86.25](https://unsloth.ai/docs/models/glm-5.3-flash) | 100 | unsloth dynamic | unsloth: llama.cpp KLD vs BF16, top-1 agreement |  | 2026-08-26 |
| UD-IQ4_XS | 3.92 |  | mean_kld | [0.1167](https://unsloth.ai/docs/models/glm-5.3-flash) |  | unsloth dynamic: attention/shared/dense layers higher | unsloth: llama.cpp KLD vs BF16, top-1 agreement | size 156.82 GB; bpw computed GB*8/320 | 2026-08-26 |
| UD-IQ4_XS | 3.92 |  | top1_pct | [88.18](https://unsloth.ai/docs/models/glm-5.3-flash) | 100 | unsloth dynamic | unsloth: llama.cpp KLD vs BF16, top-1 agreement |  | 2026-08-26 |
| UD-Q4_K_XL | 4.99 |  | mean_kld | [0.0493](https://unsloth.ai/docs/models/glm-5.3-flash) |  | unsloth dynamic: attention/shared/dense layers higher | unsloth: llama.cpp KLD vs BF16, top-1 agreement | size 199.71 GB; bpw computed GB*8/320 | 2026-08-26 |
| UD-Q4_K_XL | 4.99 |  | top1_pct | [92.22](https://unsloth.ai/docs/models/glm-5.3-flash) | 100 | unsloth dynamic | unsloth: llama.cpp KLD vs BF16, top-1 agreement |  | 2026-08-26 |
| UD-Q5_K_XL | 6.01 |  | mean_kld | [0.0271](https://unsloth.ai/docs/models/glm-5.3-flash) |  | unsloth dynamic: attention/shared/dense layers higher | unsloth: llama.cpp KLD vs BF16, top-1 agreement | size 240.31 GB; bpw computed GB*8/320 | 2026-08-26 |
| UD-Q5_K_XL | 6.01 |  | top1_pct | [94.35](https://unsloth.ai/docs/models/glm-5.3-flash) | 100 | unsloth dynamic | unsloth: llama.cpp KLD vs BF16, top-1 agreement |  | 2026-08-26 |
| UD-Q6_K_XL | 7.3 |  | mean_kld | [0.019](https://unsloth.ai/docs/models/glm-5.3-flash) |  | unsloth dynamic: attention/shared/dense layers higher | unsloth: llama.cpp KLD vs BF16, top-1 agreement | size 291.83 GB; bpw computed GB*8/320 | 2026-08-26 |
| UD-Q6_K_XL | 7.3 |  | top1_pct | [95.23](https://unsloth.ai/docs/models/glm-5.3-flash) | 100 | unsloth dynamic | unsloth: llama.cpp KLD vs BF16, top-1 agreement |  | 2026-08-26 |
| Q5_K_M | 6.01 |  | ppl_delta_pct | [0.55](https://huggingface.co/AesSedai/GLM-5.3-Flash-GGUF) | 0 | mix default/up/gate/down = Q8_0/Q5_K/Q5_K/Q6_K | AesSedai: llama.cpp PPL/KLD vs BF16 |  | 2026-08-30 |
| Q5_K_M | 6.01 |  | mean_kld | [0.02786](https://huggingface.co/AesSedai/GLM-5.3-Flash-GGUF) |  | mix Q8_0/Q5_K/Q5_K/Q6_K | AesSedai: llama.cpp PPL/KLD vs BF16 |  | 2026-08-30 |
| Q4_K_M | 5.04 |  | ppl_delta_pct | [1.83](https://huggingface.co/AesSedai/GLM-5.3-Flash-GGUF) | 0 | mix default/up/gate/down = Q8_0/Q4_K/Q4_K/Q5_K | AesSedai: llama.cpp PPL/KLD vs BF16 |  | 2026-08-30 |
| Q4_K_M | 5.04 |  | mean_kld | [0.05018](https://huggingface.co/AesSedai/GLM-5.3-Flash-GGUF) |  | mix Q8_0/Q4_K/Q4_K/Q5_K | AesSedai: llama.cpp PPL/KLD vs BF16 |  | 2026-08-30 |
| IQ4_XS | 3.97 |  | ppl_delta_pct | [6.98](https://huggingface.co/AesSedai/GLM-5.3-Flash-GGUF) | 0 | mix default/up/gate/down = Q8_0/IQ3_S/IQ3_S/IQ4_XS | AesSedai: llama.cpp PPL/KLD vs BF16 |  | 2026-08-30 |
| IQ4_XS | 3.97 |  | mean_kld | [0.1174](https://huggingface.co/AesSedai/GLM-5.3-Flash-GGUF) |  | mix Q8_0/IQ3_S/IQ3_S/IQ4_XS | AesSedai: llama.cpp PPL/KLD vs BF16 |  | 2026-08-30 |
| IQ3_S | 3.11 |  | ppl_delta_pct | [22.88](https://huggingface.co/AesSedai/GLM-5.3-Flash-GGUF) | 0 | mix default/up/gate/down = Q6_K/IQ2_S/IQ2_S/IQ3_S | AesSedai: llama.cpp PPL/KLD vs BF16 |  | 2026-08-30 |
| IQ3_S | 3.11 |  | mean_kld | [0.2834](https://huggingface.co/AesSedai/GLM-5.3-Flash-GGUF) |  | mix Q6_K/IQ2_S/IQ2_S/IQ3_S | AesSedai: llama.cpp PPL/KLD vs BF16 |  | 2026-08-30 |
| IQ2_S | 2.83 |  | ppl_delta_pct | [33.37](https://huggingface.co/AesSedai/GLM-5.3-Flash-GGUF) | 0 | mix default/up/gate/down = Q6_K/IQ2_XS/IQ2_XS/IQ3_XXS | AesSedai: llama.cpp PPL/KLD vs BF16 |  | 2026-08-30 |
| IQ2_S | 2.83 |  | mean_kld | [0.3754](https://huggingface.co/AesSedai/GLM-5.3-Flash-GGUF) |  | mix Q6_K/IQ2_XS/IQ2_XS/IQ3_XXS | AesSedai: llama.cpp PPL/KLD vs BF16 |  | 2026-08-30 |
| EXL3 2.05 | 2.05 |  | mean_kld | [0.1929](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) |  |  | turboderp: KLD on self-generated in-domain trace 28,154 in + 45,470 out tokens; noise floor 0.01378 | median KLD 0.03247 | 2026-08-28 |
| EXL3 3.05 | 3.05 |  | mean_kld | [0.0681](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) |  |  | turboderp: KLD on self-generated in-domain trace 28,154 in + 45,470 out tokens; noise floor 0.01378 | median KLD 0.0074 | 2026-08-28 |
| EXL3 4.05 | 4.05 |  | mean_kld | [0.0346](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) |  |  | turboderp: KLD on self-generated in-domain trace 28,154 in + 45,470 out tokens; noise floor 0.01378 | median KLD 0.003 | 2026-08-28 |
| EXL3-SC 2.10 | 2.1 |  | mean_kld | [0.157](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) |  |  | turboderp: KLD on self-generated in-domain trace 28,154 in + 45,470 out tokens; noise floor 0.01378 | median KLD 0.02038 | 2026-08-28 |
| EXL3-SC 2.60 | 2.6 |  | mean_kld | [0.0929](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) |  |  | turboderp: KLD on self-generated in-domain trace 28,154 in + 45,470 out tokens; noise floor 0.01378 | median KLD 0.01013 | 2026-08-28 |
| EXL3-SC 3.10 | 3.1 |  | mean_kld | [0.0574](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) |  |  | turboderp: KLD on self-generated in-domain trace 28,154 in + 45,470 out tokens; noise floor 0.01378 | median KLD 0.00581 | 2026-08-28 |
| UD-IQ1_M (GGUF) | 2.4 |  | mean_kld | [0.232](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) |  |  | turboderp: KLD on self-generated in-domain trace 28,154 in + 45,470 out tokens; noise floor 0.01378 |  | 2026-08-28 |
| UD-IQ3_XXS (GGUF) | 2.98 |  | mean_kld | [0.1258](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) |  |  | turboderp: KLD on self-generated in-domain trace 28,154 in + 45,470 out tokens; noise floor 0.01378 | median KLD 0.01663 | 2026-08-28 |
| UD-IQ4_XS (GGUF) | 3.89 |  | mean_kld | [0.0615](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) |  |  | turboderp: KLD on self-generated in-domain trace 28,154 in + 45,470 out tokens; noise floor 0.01378 |  | 2026-08-28 |
| AWQ int4 g128 | 4.42 |  | mean_kld | [0.0565](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) |  |  | turboderp: KLD on self-generated in-domain trace 28,154 in + 45,470 out tokens; noise floor 0.01378 |  | 2026-08-28 |
| NVFP4 | 4.78 |  | mean_kld | [0.0452](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) |  |  | turboderp: KLD on self-generated in-domain trace 28,154 in + 45,470 out tokens; noise floor 0.01378 |  | 2026-08-28 |
| Sushi 2 bpw |  |  | mean_kld | [0.1846](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) |  |  | turboderp: KLD on self-generated in-domain trace 28,154 in + 45,470 out tokens; noise floor 0.01378 |  | 2026-08-28 |
| Q8_0 | 8.5 |  | mmlu_pro_pct | [61.95](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF) |  |  | pfeifferj: MMLU-Pro 2000-q subset, no reasoning; PPL vs Q8_0 3.3909 |  | 2026-09-13 |
| GSQ-RCO 3.5-bit (137.07 GB) | 3.43 | 3.5 | mmlu_pro_pct | [60.55](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF) | 61.95 | mostly Q3_K experts | pfeifferj: MMLU-Pro 2000-q subset, no reasoning; PPL vs Q8_0 3.3909 | retention 97.7%; PPL +4.49%; approx KL 0.0713; math 49.8 -> 44.9 | 2026-09-13 |
| GSQ-RCO 3.0-bit (117.48 GB) | 2.94 | 3.0 | mmlu_pro_pct | [60.0](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF) | 61.95 | mostly Q2_K experts (108 tensors) | pfeifferj: MMLU-Pro 2000-q subset, no reasoning; PPL vs Q8_0 3.3909 | retention 96.9%; PPL +9.07%; approx KL 0.1424; math 49.8 -> 44.0 | 2026-09-13 |
| GSQ-RCO 3.0-bit Q4_K attention |  | 3.0 | ppl | [3.5871](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF) | 3.5534 | attention at Q4_K | neuralll: llama.cpp PPL | +0.95% vs its parent | 2026-09-24 |
| Intel W4A16 AutoRound | 4.0 | 4.0 | lm_eval_avg | [0.8386](https://huggingface.co/Intel/GLM-5.3-Flash-W4A16-AutoRound) | 0.8399 | AutoRound int4 g128 | Intel: lm-eval task average | retention 99.84%; MMLU 0.8620 vs 0.8666 | 2026-08-31 |

### GLM-5.3

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| UD-IQ1_S | 2.3 |  | mean_kld | [0.688](https://unsloth.ai/docs/models/glm-5.3) |  | unsloth dynamic | unsloth: llama.cpp KLD vs BF16/FP8, top-1 | top-1 72.56%; size 216.7 GB; bpw computed GB*8/753 | 2026-08-28 |
| UD-IQ1_M | 2.43 |  | mean_kld | [0.565](https://unsloth.ai/docs/models/glm-5.3) |  | unsloth dynamic | unsloth: llama.cpp KLD vs BF16/FP8, top-1 | top-1 75.64%; size 228.5 GB; bpw computed GB*8/753 | 2026-08-28 |
| UD-IQ2_M | 2.53 |  | mean_kld | [0.454](https://unsloth.ai/docs/models/glm-5.3) |  | unsloth dynamic | unsloth: llama.cpp KLD vs BF16/FP8, top-1 | top-1 78.53%; size 238.6 GB; bpw computed GB*8/753 | 2026-08-28 |
| UD-Q2_K_XL | 2.7 |  | mean_kld | [0.374](https://unsloth.ai/docs/models/glm-5.3) |  | unsloth dynamic | unsloth: llama.cpp KLD vs BF16/FP8, top-1 | top-1 80.93%; size 253.9 GB; bpw computed GB*8/753 | 2026-08-28 |
| UD-IQ3_XXS | 2.99 |  | mean_kld | [0.273](https://unsloth.ai/docs/models/glm-5.3) |  | unsloth dynamic | unsloth: llama.cpp KLD vs BF16/FP8, top-1 | top-1 84.15%; size 281.7 GB; bpw computed GB*8/753 | 2026-08-28 |
| UD-Q3_K_XL | 3.64 |  | mean_kld | [0.141](https://unsloth.ai/docs/models/glm-5.3) |  | unsloth dynamic | unsloth: llama.cpp KLD vs BF16/FP8, top-1 | top-1 88.86%; size 343.0 GB; bpw computed GB*8/753 | 2026-08-28 |
| UD-IQ4_XS | 3.88 |  | mean_kld | [0.1015](https://unsloth.ai/docs/models/glm-5.3) |  | unsloth dynamic | unsloth: llama.cpp KLD vs BF16/FP8, top-1 | top-1 90.59%; size 365.3 GB; bpw computed GB*8/753 | 2026-08-28 |
| UD-Q4_K_XL | 4.96 |  | mean_kld | [0.0369](https://unsloth.ai/docs/models/glm-5.3) |  | unsloth dynamic | unsloth: llama.cpp KLD vs BF16/FP8, top-1 | top-1 94.29%; size 467.3 GB; bpw computed GB*8/753 | 2026-08-28 |
| UD-Q5_K_XL | 5.98 |  | mean_kld | [0.0197](https://unsloth.ai/docs/models/glm-5.3) |  | unsloth dynamic | unsloth: llama.cpp KLD vs BF16/FP8, top-1 | top-1 95.82%; size 562.5 GB; bpw computed GB*8/753 | 2026-08-28 |
| UD-Q6_K_XL | 7.27 |  | mean_kld | [0.0133](https://unsloth.ai/docs/models/glm-5.3) |  | unsloth dynamic | unsloth: llama.cpp KLD vs BF16/FP8, top-1 | top-1 96.59%; size 684.4 GB; bpw computed GB*8/753 | 2026-08-28 |
| Q8_0 | 8.5 |  | ppl_delta_pct | [0.017](https://huggingface.co/AesSedai/GLM-5.3-GGUF) | 0 | AesSedai mix: attention Q8_0/Q6_K | AesSedai: llama.cpp PPL/KLD | KLD 0.0131 | 2026-08-29 |
| Q5_K_M | 5.96 |  | ppl_delta_pct | [0.24](https://huggingface.co/AesSedai/GLM-5.3-GGUF) | 0 | AesSedai mix: attention Q8_0/Q6_K | AesSedai: llama.cpp PPL/KLD | KLD 0.0203 | 2026-08-29 |
| Q4_K_M | 4.98 |  | ppl_delta_pct | [0.94](https://huggingface.co/AesSedai/GLM-5.3-GGUF) | 0 | AesSedai mix: attention Q8_0/Q6_K | AesSedai: llama.cpp PPL/KLD | KLD 0.0371 | 2026-08-29 |
| IQ4_XS | 3.9 |  | ppl_delta_pct | [6.23](https://huggingface.co/AesSedai/GLM-5.3-GGUF) | 0 | AesSedai mix: attention Q8_0/Q6_K | AesSedai: llama.cpp PPL/KLD | KLD 0.1016 | 2026-08-29 |
| IQ3_S | 3.03 |  | ppl_delta_pct | [21.8](https://huggingface.co/AesSedai/GLM-5.3-GGUF) | 0 | AesSedai mix: attention Q8_0/Q6_K | AesSedai: llama.cpp PPL/KLD | KLD 0.2752 | 2026-08-29 |
| IQ2_S | 2.75 |  | ppl_delta_pct | [31.3](https://huggingface.co/AesSedai/GLM-5.3-GGUF) | 0 | AesSedai mix: attention Q8_0/Q6_K | AesSedai: llama.cpp PPL/KLD | KLD 0.3742 | 2026-08-29 |

### DeepSeek-V4-Flash-0731

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| UD-Q4_K_XL | 4.37 |  | mean_kld | [0.0102](https://unsloth.ai/docs/models/deepseek-v4) |  | attention/shared kept FP8/Q8 | unsloth: wikitext-2 ctx 512 vs UD-Q8_K_XL (bit-exact), ref PPL 4.5319 | PPL 4.5335; top-1 96.28%; size 155.1 GB | 2026-07-31 |
| bartowski MXFP4 |  |  | mean_kld | [0.0105](https://unsloth.ai/docs/models/deepseek-v4) |  | attention/shared kept FP8/Q8 | unsloth: wikitext-2 ctx 512 vs UD-Q8_K_XL (bit-exact), ref PPL 4.5319 | PPL 4.5351; top-1 96.18%; size None GB | 2026-07-31 |
| antirez Q4KExperts | 4.64 |  | mean_kld | [0.029](https://unsloth.ai/docs/models/deepseek-v4) |  | attention/shared kept FP8/Q8 | unsloth: wikitext-2 ctx 512 vs UD-Q8_K_XL (bit-exact), ref PPL 4.5319 | PPL 4.57; top-1 93.9%; size 164.6 GB | 2026-07-31 |
| antirez mixed L37-42-Q4K | 2.75 |  | mean_kld | [0.3605](https://unsloth.ai/docs/models/deepseek-v4) |  | attention/shared kept FP8/Q8 | unsloth: wikitext-2 ctx 512 vs UD-Q8_K_XL (bit-exact), ref PPL 4.5319 | PPL 5.8169; top-1 79.74%; size 97.6 GB | 2026-07-31 |
| antirez IQ2XXS imatrix | 2.44 |  | mean_kld | [0.4079](https://unsloth.ai/docs/models/deepseek-v4) |  | attention/shared kept FP8/Q8 | unsloth: wikitext-2 ctx 512 vs UD-Q8_K_XL (bit-exact), ref PPL 4.5319 | PPL 6.0808; top-1 78.15%; size 86.7 GB | 2026-07-31 |
| antirez IQ2XXS no imatrix |  |  | mean_kld | [0.4207](https://unsloth.ai/docs/models/deepseek-v4) |  | attention/shared kept FP8/Q8 | unsloth: wikitext-2 ctx 512 vs UD-Q8_K_XL (bit-exact), ref PPL 4.5319 | PPL 6.1518; top-1 None%; size None GB | 2026-07-31 |
| AD-MXFP4 |  | 4.25 | mean_kld | [0.1564](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) |  | attention/shared high | AtomicChat: ctx 5632 vs Q8 reference, ref PPL 4.5289 (KLD floor ~0.156 even for native MXFP4) | PPL 4.5446; top-1 87.37%; 154.5 GB; delta-KLD over MXFP4 floor 0.0 | 2026-08-01 |
| IQ3_M_XL |  | 3.94 | mean_kld | [0.1675](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) |  | attention/shared high | AtomicChat: ctx 5632 vs Q8 reference, ref PPL 4.5289 (KLD floor ~0.156 even for native MXFP4) | PPL 4.549; top-1 86.86%; 143.6 GB; delta-KLD over MXFP4 floor 0.0111 | 2026-08-01 |
| IQ3_M |  | 3.71 | mean_kld | [0.1798](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) |  | attention/shared high | AtomicChat: ctx 5632 vs Q8 reference, ref PPL 4.5289 (KLD floor ~0.156 even for native MXFP4) | PPL 4.5695; top-1 86.32%; 135.8 GB; delta-KLD over MXFP4 floor 0.0234 | 2026-08-01 |
| IQ3_S |  | 3.56 | mean_kld | [0.1891](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) |  | attention/shared high | AtomicChat: ctx 5632 vs Q8 reference, ref PPL 4.5289 (KLD floor ~0.156 even for native MXFP4) | PPL 4.6016; top-1 85.95%; 130.8 GB; delta-KLD over MXFP4 floor 0.0327 | 2026-08-01 |
| IQ3_XS |  | 3.2 | mean_kld | [0.2065](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) |  | attention/shared high | AtomicChat: ctx 5632 vs Q8 reference, ref PPL 4.5289 (KLD floor ~0.156 even for native MXFP4) | PPL 4.6657; top-1 85.38%; 118.2 GB; delta-KLD over MXFP4 floor 0.0501 | 2026-08-01 |
| IQ3_XXS |  | 2.91 | mean_kld | [0.2495](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) |  | attention/shared high | AtomicChat: ctx 5632 vs Q8 reference, ref PPL 4.5289 (KLD floor ~0.156 even for native MXFP4) | PPL 4.8491; top-1 83.76%; 108.1 GB; delta-KLD over MXFP4 floor 0.0931 | 2026-08-01 |
| IQ2_M |  | 2.79 | mean_kld | [0.2567](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) |  | attention/shared high | AtomicChat: ctx 5632 vs Q8 reference, ref PPL 4.5289 (KLD floor ~0.156 even for native MXFP4) | PPL 4.8822; top-1 83.56%; 104.0 GB; delta-KLD over MXFP4 floor 0.1003 | 2026-08-01 |
| IQ2_S_XL |  | 2.58 | mean_kld | [0.3187](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) |  | attention/shared high | AtomicChat: ctx 5632 vs Q8 reference, ref PPL 4.5289 (KLD floor ~0.156 even for native MXFP4) | PPL 5.1406; top-1 81.46%; 96.8 GB; delta-KLD over MXFP4 floor 0.1623 | 2026-08-01 |
| IQ2_S |  | 2.48 | mean_kld | [0.3343](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) |  | attention/shared high | AtomicChat: ctx 5632 vs Q8 reference, ref PPL 4.5289 (KLD floor ~0.156 even for native MXFP4) | PPL 5.2152; top-1 81.03%; 93.4 GB; delta-KLD over MXFP4 floor 0.1779 | 2026-08-01 |
| IQ2_XS |  | 2.25 | mean_kld | [0.3947](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) |  | attention/shared high | AtomicChat: ctx 5632 vs Q8 reference, ref PPL 4.5289 (KLD floor ~0.156 even for native MXFP4) | PPL 5.4917; top-1 79.24%; 85.1 GB; delta-KLD over MXFP4 floor 0.2383 | 2026-08-01 |
| IQ2_XXS |  | 2.06 | mean_kld | [0.4544](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) |  | attention/shared high | AtomicChat: ctx 5632 vs Q8 reference, ref PPL 4.5289 (KLD floor ~0.156 even for native MXFP4) | PPL 5.7878; top-1 77.46%; 78.5 GB; delta-KLD over MXFP4 floor 0.298 | 2026-08-01 |
| IQ1_M_XL |  | 1.89 | mean_kld | [0.5351](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) |  | attention/shared high | AtomicChat: ctx 5632 vs Q8 reference, ref PPL 4.5289 (KLD floor ~0.156 even for native MXFP4) | PPL 6.1786; top-1 75.16%; 72.8 GB; delta-KLD over MXFP4 floor 0.3787 | 2026-08-01 |
| IQ1_M |  | 1.81 | mean_kld | [0.5641](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) |  | attention/shared high | AtomicChat: ctx 5632 vs Q8 reference, ref PPL 4.5289 (KLD floor ~0.156 even for native MXFP4) | PPL 6.3813; top-1 74.55%; 70.2 GB; delta-KLD over MXFP4 floor 0.4077 | 2026-08-01 |
| unsloth UD-Q4_K_XL |  |  | mean_kld | [0.1557](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) |  | unsloth dynamic | AtomicChat: ctx 5632 vs Q8 reference, ref PPL 4.5289 (KLD floor ~0.156 even for native MXFP4) | unsloth quant re-scored in AtomicChat harness; compare with unsloth's own 0.0102 for UD-Q4_K_XL | 2026-08-01 |
| unsloth UD-IQ4_XS |  |  | mean_kld | [0.1779](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) |  | unsloth dynamic | AtomicChat: ctx 5632 vs Q8 reference, ref PPL 4.5289 (KLD floor ~0.156 even for native MXFP4) | unsloth quant re-scored in AtomicChat harness; compare with unsloth's own 0.0102 for UD-Q4_K_XL | 2026-08-01 |
| unsloth UD-Q3_K_XL |  |  | mean_kld | [0.1981](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) |  | unsloth dynamic | AtomicChat: ctx 5632 vs Q8 reference, ref PPL 4.5289 (KLD floor ~0.156 even for native MXFP4) | unsloth quant re-scored in AtomicChat harness; compare with unsloth's own 0.0102 for UD-Q4_K_XL | 2026-08-01 |
| unsloth UD-IQ3_S |  |  | mean_kld | [0.2565](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) |  | unsloth dynamic | AtomicChat: ctx 5632 vs Q8 reference, ref PPL 4.5289 (KLD floor ~0.156 even for native MXFP4) | unsloth quant re-scored in AtomicChat harness; compare with unsloth's own 0.0102 for UD-Q4_K_XL | 2026-08-01 |
| unsloth UD-IQ3_XXS |  |  | mean_kld | [0.261](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) |  | unsloth dynamic | AtomicChat: ctx 5632 vs Q8 reference, ref PPL 4.5289 (KLD floor ~0.156 even for native MXFP4) | unsloth quant re-scored in AtomicChat harness; compare with unsloth's own 0.0102 for UD-Q4_K_XL | 2026-08-01 |
| unsloth UD-Q2_K_XL |  |  | mean_kld | [0.3216](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) |  | unsloth dynamic | AtomicChat: ctx 5632 vs Q8 reference, ref PPL 4.5289 (KLD floor ~0.156 even for native MXFP4) | unsloth quant re-scored in AtomicChat harness; compare with unsloth's own 0.0102 for UD-Q4_K_XL | 2026-08-01 |
| unsloth UD-IQ2_M |  |  | mean_kld | [0.37](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) |  | unsloth dynamic | AtomicChat: ctx 5632 vs Q8 reference, ref PPL 4.5289 (KLD floor ~0.156 even for native MXFP4) | unsloth quant re-scored in AtomicChat harness; compare with unsloth's own 0.0102 for UD-Q4_K_XL | 2026-08-01 |
| unsloth UD-IQ1_S |  |  | mean_kld | [0.4863](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) |  | unsloth dynamic | AtomicChat: ctx 5632 vs Q8 reference, ref PPL 4.5289 (KLD floor ~0.156 even for native MXFP4) | unsloth quant re-scored in AtomicChat harness; compare with unsloth's own 0.0102 for UD-Q4_K_XL | 2026-08-01 |
| XS_IQ1_KT |  | 1.841 | mean_kld | [0.4978](https://huggingface.co/KeinNiemand/DeepSeek-V4-Flash-0731-IK_GGUF) |  | ik trellis/KS; attention high | KeinNiemand: ik_llama.cpp, AtomicChat setup ctx 5632, 31 chunks | PPL 6.029; top-1 76.08% | 2026-08-27 |
| IQ1_KT |  | 2.059 | mean_kld | [0.3932](https://huggingface.co/KeinNiemand/DeepSeek-V4-Flash-0731-IK_GGUF) |  | ik trellis/KS; attention high | KeinNiemand: ik_llama.cpp, AtomicChat setup ctx 5632, 31 chunks | PPL 5.464; top-1 79.08% | 2026-08-27 |
| IQ2_KT |  | 2.618 | mean_kld | [0.27](https://huggingface.co/KeinNiemand/DeepSeek-V4-Flash-0731-IK_GGUF) |  | ik trellis/KS; attention high | KeinNiemand: ik_llama.cpp, AtomicChat setup ctx 5632, 31 chunks | PPL 4.931; top-1 83.0% | 2026-08-27 |
| IQ2_KS |  | 2.629 | mean_kld | [0.2689](https://huggingface.co/KeinNiemand/DeepSeek-V4-Flash-0731-IK_GGUF) |  | ik trellis/KS; attention high | KeinNiemand: ik_llama.cpp, AtomicChat setup ctx 5632, 31 chunks | PPL 4.914; top-1 83.03% | 2026-08-27 |
| IQ2_KL |  | 2.992 | mean_kld | [0.2053](https://huggingface.co/KeinNiemand/DeepSeek-V4-Flash-0731-IK_GGUF) |  | ik trellis/KS; attention high | KeinNiemand: ik_llama.cpp, AtomicChat setup ctx 5632, 31 chunks | PPL 4.665; top-1 85.43% | 2026-08-27 |
| EXL3 2.04 | 2.04 |  | mean_kld | [0.211](https://huggingface.co/turboderp/DeepSeek-V4-Flash-0731-exl3) |  |  | turboderp: openwebtext 10x2048, noise floor 0.0070 |  | 2026-08-04 |
| EXL3 2.52 | 2.52 |  | mean_kld | [0.143](https://huggingface.co/turboderp/DeepSeek-V4-Flash-0731-exl3) |  |  | turboderp: openwebtext 10x2048, noise floor 0.0070 |  | 2026-08-04 |
| EXL3 2.77 | 2.77 |  | mean_kld | [0.101](https://huggingface.co/turboderp/DeepSeek-V4-Flash-0731-exl3) |  |  | turboderp: openwebtext 10x2048, noise floor 0.0070 |  | 2026-08-04 |
| EXL3 3.04 | 3.04 |  | mean_kld | [0.058](https://huggingface.co/turboderp/DeepSeek-V4-Flash-0731-exl3) |  |  | turboderp: openwebtext 10x2048, noise floor 0.0070 |  | 2026-08-04 |
| UD-IQ1_M (GGUF) |  |  | mean_kld | [0.302](https://huggingface.co/turboderp/DeepSeek-V4-Flash-0731-exl3) |  |  | turboderp: openwebtext 10x2048, noise floor 0.0070 |  | 2026-08-04 |
| UD-IQ2_XXS (GGUF) |  |  | mean_kld | [0.247](https://huggingface.co/turboderp/DeepSeek-V4-Flash-0731-exl3) |  |  | turboderp: openwebtext 10x2048, noise floor 0.0070 |  | 2026-08-04 |
| UD-Q2_K_XL (GGUF) |  |  | mean_kld | [0.203](https://huggingface.co/turboderp/DeepSeek-V4-Flash-0731-exl3) |  |  | turboderp: openwebtext 10x2048, noise floor 0.0070 |  | 2026-08-04 |
| IQ3_XXS with down IQ3_S (111.37 GiB) |  |  | mean_kld | [0.2386](https://huggingface.co/TacoTakumi/DeepSeek-V4-Flash-0731-GGUF) |  | down_exps one rung up; attention/shared Q8_0 | TacoTakumi: attention/shared Q8_0, router BF16 | top-1 84.65%; dPPL +0.536 | 2026-08-01 |
| IQ3_XXS (107.34 GiB) |  |  | mean_kld | [0.2629](https://huggingface.co/TacoTakumi/DeepSeek-V4-Flash-0731-GGUF) |  | attention/shared Q8_0 | TacoTakumi: attention/shared Q8_0, router BF16 | top-1 83.91%; dPPL +0.641; pre-0731 checkpoint at same recipe 0.177 | 2026-08-01 |
| ROCmFP3 Strix |  | 2.766 | internal_eval | [82](https://huggingface.co/Lucebox/DeepSeek-V4-Flash-0731-ROCmFP3) | 92 |  | Lucebox internal eval /92 | matches their reference score; COMPSEC 17/17 | 2026-07-08 |

### DeepSeek-V4-Flash (pre-0731)

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| antirez IQ2XXS-w2Q2K |  |  | mean_kld | [0.4223](https://huggingface.co/sokann/DeepSeek-V4-Flash-GGUF) |  | attention Q8 | sokann: vs near-lossless base PPL 4.7384 | same-top 77.59%; PPL 6.155 | 2026-06-16 |
| antirez Q4KExperts |  |  | mean_kld | [0.1151](https://huggingface.co/sokann/DeepSeek-V4-Flash-GGUF) |  | attention Q8 | sokann: vs near-lossless base PPL 4.7384 | same-top 88.79%; PPL 4.5706 | 2026-06-16 |

### DeepSeek-V4-Flash-0731-REAP-150B

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| TQ3_4S | 3.49 |  | humaneval_plus | [0.9207](https://huggingface.co/Tinker-Stack/DeepSeek-V4-Flash-0731-REAP-150b-TQ3_4S-GGUF) | 0.8963 | 50% experts pruned (REAP) | Tinker-Stack: HumanEval+ thinking vs FP8 (protocol advantaged) | card: IQ2_S experts cost +7.5% PPL on the pruned model; 'Do not requantize the experts down to 2-bit' | 2026-08-29 |

### DeepSeek-V4.1-Flash

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| antirez Q2 (IQ2_XXS gate/up + Q2_K down) |  |  | kl_top256 | [0.26](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-ROCMFP23-GGUF) | 0 | attention/shared Q8; engram FP8 | Lucebox: top-256 KL vs fp4 teacher, 8,184 fineweb-edu tokens; teacher PPL 8.45 | experts 142 GiB vs teacher 269 GiB; top-1 78.3%; PPL 9.71 | 2026-09-17 |
| ROCMFP23 |  |  | kl_top256 | [0.272](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-ROCMFP23-GGUF) | 0 | attention/shared Q8; engram FP8 | Lucebox: top-256 KL vs fp4 teacher, 8,184 fineweb-edu tokens; teacher PPL 8.45 | experts 179 GiB vs teacher 269 GiB; top-1 77.9%; PPL 9.8 | 2026-09-17 |
| ROCMFP2S |  |  | kl_top256 | [0.266](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-ROCMFP23-GGUF) | 0 | attention/shared Q8; engram FP8 | Lucebox: top-256 KL vs fp4 teacher, 8,184 fineweb-edu tokens; teacher PPL 8.45 | experts 158 GiB vs teacher 269 GiB; top-1 77.7%; PPL 9.59 | 2026-09-17 |
| FP2S-MIX10 (fp3 down layers 0-9) |  |  | kl_top256 | [0.246](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-ROCMFP23-GGUF) | 0 | attention/shared Q8; engram FP8 | Lucebox: top-256 KL vs fp4 teacher, 8,184 fineweb-edu tokens; teacher PPL 8.45 | experts None GiB vs teacher 269 GiB; top-1 78.3%; PPL 9.45 | 2026-09-17 |
| Lucebox final IQ2/IQ3 + MXFP8 (366.1 GB) |  |  | hard_bench_177 | [164](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-ROCMFP23-GGUF) |  |  | Lucebox internal hard bench /177 | no teacher score given; antirez Q2 scored 141, minus10gib 155, DSpark 158 | 2026-09-17 |
| antirez Q2 (in Lucebox hard bench) |  |  | hard_bench_177 | [141](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-ROCMFP23-GGUF) |  | attention/shared Q8 | Lucebox internal hard bench /177 |  | 2026-09-17 |
| GSQ-RCO 3.0-bit backbone |  | 3.0 | mmlu_pro_pct | [61.0](https://huggingface.co/pfeifferj/DeepSeek-V4.1-Flash-GSQ-RCO-GGUF) |  |  | pfeifferj MMLU-Pro subset | no full-precision baseline in the same harness: retention unknown | 2026-09-13 |
| apetersson MixedQ2 |  | 2.25 | weight_cosine_min | [0.9407](https://huggingface.co/apetersson/DeepSeek-V4.1-Flash-MixedQ2-GGUF) |  | no imatrix | weight cosine vs MXFP4 | IQ2_XXS min 0.9407, Q2_K min 0.9633; no behavioural eval | 2026-09-10 |

### DeepSeek-V3.1

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| UD-Q5_K_XL |  | 5.5 | aider_polyglot_nonthinking | [70.7](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) | 71.6 | unsloth dynamic; MoE bpw from unsloth size table | unsloth: Aider polyglot pass-2 |  | 2025-09 |
| UD-Q4_K_XL |  | 4.5 | aider_polyglot_nonthinking | [69.7](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) | 71.6 | unsloth dynamic; MoE bpw from unsloth size table | unsloth: Aider polyglot pass-2 |  | 2025-09 |
| UD-Q3_K_XL |  | 3.5 | aider_polyglot_nonthinking | [68.4](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) | 71.6 | unsloth dynamic; MoE bpw from unsloth size table | unsloth: Aider polyglot pass-2 |  | 2025-09 |
| UD-IQ3_XXS |  | 3.12 | aider_polyglot_nonthinking | [66.8](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) | 71.6 | unsloth dynamic; MoE bpw from unsloth size table | unsloth: Aider polyglot pass-2 |  | 2025-09 |
| UD-Q2_K_XL |  | 2.71 | aider_polyglot_nonthinking | [65.8](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) | 71.6 | unsloth dynamic; MoE bpw from unsloth size table | unsloth: Aider polyglot pass-2 |  | 2025-09 |
| UD-IQ1_M |  | 1.93 | aider_polyglot_nonthinking | [55.7](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) | 71.6 | unsloth dynamic; MoE bpw from unsloth size table | unsloth: Aider polyglot pass-2 |  | 2025-09 |
| UD-TQ1_0 |  | 1.66 | aider_polyglot_nonthinking | [50.7](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) | 71.6 | unsloth dynamic; MoE bpw from unsloth size table | unsloth: Aider polyglot pass-2 |  | 2025-09 |
| IQ2_XXS (community) |  |  | aider_polyglot_nonthinking | [43.6](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) | 71.6 | standard llama.cpp recipe | unsloth: Aider polyglot pass-2 |  | 2025-09 |
| IQ2_M (community) |  |  | aider_polyglot_nonthinking | [56.6](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) | 71.6 | standard llama.cpp recipe | unsloth: Aider polyglot pass-2 |  | 2025-09 |
| Q2_K_L (community) |  |  | aider_polyglot_nonthinking | [64.0](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) | 71.6 | standard llama.cpp recipe | unsloth: Aider polyglot pass-2 |  | 2025-09 |
| IQ3_XXS (community) |  |  | aider_polyglot_nonthinking | [65.6](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) | 71.6 | standard llama.cpp recipe | unsloth: Aider polyglot pass-2 |  | 2025-09 |
| Q3_K_S (community) |  |  | aider_polyglot_nonthinking | [65.2](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) | 71.6 | standard llama.cpp recipe | unsloth: Aider polyglot pass-2 |  | 2025-09 |
| IQ4_XS (community) |  |  | aider_polyglot_nonthinking | [66.3](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) | 71.6 | standard llama.cpp recipe | unsloth: Aider polyglot pass-2 |  | 2025-09 |
| Q4_K_M (community) |  |  | aider_polyglot_nonthinking | [67.7](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) | 71.6 | standard llama.cpp recipe | unsloth: Aider polyglot pass-2 |  | 2025-09 |
| Q5_K_M (community) |  |  | aider_polyglot_nonthinking | [68.9](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) | 71.6 | standard llama.cpp recipe | unsloth: Aider polyglot pass-2 |  | 2025-09 |
| unsloth 3-bit |  | 3.5 | aider_polyglot_thinking | [75.6](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) | 76.1 | unsloth dynamic | unsloth: Aider polyglot pass-2 |  | 2025-09 |
| unsloth 2-bit |  | 2.71 | aider_polyglot_thinking | [66.7](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) | 76.1 | unsloth dynamic | unsloth: Aider polyglot pass-2 |  | 2025-09 |
| unsloth 1-bit |  | 1.93 | aider_polyglot_thinking | [57.8](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) | 76.1 | unsloth dynamic | unsloth: Aider polyglot pass-2 |  | 2025-09 |
| IQ5_K | 5.944 |  | ppl_ratio | [1.0023](https://huggingface.co/ubergarm/DeepSeek-V3.1-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.3473 | PPL 3.355 | 2025-08-21 |
| IQ4_K | 4.925 |  | ppl_ratio | [1.0072](https://huggingface.co/ubergarm/DeepSeek-V3.1-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.3473 | PPL 3.3715 | 2025-08-21 |
| IQ4_KS | 4.649 |  | ppl_ratio | [1.0099](https://huggingface.co/ubergarm/DeepSeek-V3.1-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.3473 | PPL 3.3806 | 2025-08-21 |
| IQ4_KSS | 4.162 |  | ppl_ratio | [1.0124](https://huggingface.co/ubergarm/DeepSeek-V3.1-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.3473 | PPL 3.3887 | 2025-08-21 |
| smol-IQ4_KSS | 4.08 |  | ppl_ratio | [1.0127](https://huggingface.co/ubergarm/DeepSeek-V3.1-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.3473 | PPL 3.3898 | 2025-08-21 |
| IQ3_K | 3.753 |  | ppl_ratio | [1.0235](https://huggingface.co/ubergarm/DeepSeek-V3.1-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.3473 | PPL 3.426 | 2025-08-21 |
| IQ3_KS | 3.551 |  | ppl_ratio | [1.0317](https://huggingface.co/ubergarm/DeepSeek-V3.1-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.3473 | PPL 3.4534 | 2025-08-21 |
| IQ2_KL | 2.96 |  | ppl_ratio | [1.0848](https://huggingface.co/ubergarm/DeepSeek-V3.1-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.3473 | PPL 3.6312 | 2025-08-21 |
| IQ2_KT | 2.619 |  | ppl_ratio | [1.1385](https://huggingface.co/ubergarm/DeepSeek-V3.1-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.3473 | PPL 3.8109 | 2025-08-21 |
| IQ2_KS | 2.472 |  | ppl_ratio | [1.1825](https://huggingface.co/ubergarm/DeepSeek-V3.1-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.3473 | PPL 3.9583 | 2025-08-21 |
| IQ1_KT | 1.984 |  | ppl_ratio | [1.3141](https://huggingface.co/ubergarm/DeepSeek-V3.1-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.3473 | PPL 4.3987 | 2025-08-21 |
| IQ1_S | 1.71 |  | ppl_ratio | [1.5867](https://huggingface.co/ubergarm/DeepSeek-V3.1-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.3473 | PPL 5.3113 | 2025-08-21 |

### DeepSeek-R1

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| Q4_K_M | 4.82 |  | livecodebench | [62.41](https://arxiv.org/abs/2505.02390) | 64.16 |  | 2505.02390: benchmark suite vs FP8 | MMLU 90.14 vs 90.99; avg drop 0.68% | 2025-05 |
| Q3_K_M | 3.81 |  | livecodebench | [61.95](https://arxiv.org/abs/2505.02390) | 64.16 |  | 2505.02390: benchmark suite vs FP8 | MMLU 89.87 vs 90.99; avg drop 1.8% | 2025-05 |
| UD-Q2_K_XL | 2.7 |  | livecodebench | [61.4](https://arxiv.org/abs/2505.02390) | 64.16 |  | 2505.02390: benchmark suite vs FP8 | MMLU 89.72 vs 90.99; avg drop 0.94% | 2025-05 |
| DQ3_K_M | 3.59 |  | livecodebench | [63.15](https://arxiv.org/abs/2505.02390) | 64.16 | DQ3_K_M: down q3/q4/q6, attn kv q6_k, shared q6_k/q4_k | 2505.02390: benchmark suite vs FP8 | MMLU 91.03 vs 90.99; avg drop 0.34% | 2025-05 |

### DeepSeek-V3-0324

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| Q4_K_M | 4.82 |  | livecodebench | [47.88](https://arxiv.org/abs/2505.02390) | 49.73 |  | 2505.02390: benchmark suite vs FP8 | MMLU 88.71 vs 89.08; avg drop 1.35% | 2025-05 |
| Q3_K_M | 3.81 |  | livecodebench | [46.23](https://arxiv.org/abs/2505.02390) | 49.73 |  | 2505.02390: benchmark suite vs FP8 | MMLU 88.47 vs 89.08; avg drop 1.85% | 2025-05 |
| Q2_K_L | 2.91 |  | livecodebench | [36.95](https://arxiv.org/abs/2505.02390) | 49.73 |  | 2505.02390: benchmark suite vs FP8 | MMLU 85.59 vs 89.08; avg drop 14.66%; AIME 31.25 vs 57.9; C-Eval 73.60 vs 89.60 | 2025-05 |
| DQ3_K_M | 3.59 |  | livecodebench | [47.89](https://arxiv.org/abs/2505.02390) | 49.73 |  | 2505.02390: benchmark suite vs FP8 | MMLU None vs 89.08; avg drop 0.3% | 2025-05 |

### DeepSeek-V3

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| Q2_K_L | 2.91 |  | livecodebench | [29.14](https://arxiv.org/abs/2505.02390) | 36.21 |  | 2505.02390: benchmark suite vs FP8 | MMLU 84.25 vs 88.06 | 2025-05 |

### Kimi-K3

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| UD-IQ1_S |  |  | mean_kld | [0.5645](https://unsloth.ai/docs/models/kimi-k3) |  | unsloth dynamic | unsloth: vs Q8_K_XL (lossless, PPL 1.4581) | 594 GB; top-1 78.9%; PPL 2.579 | 2026-07-27 |
| UD-IQ1_M |  |  | mean_kld | [0.4789](https://unsloth.ai/docs/models/kimi-k3) |  | unsloth dynamic | unsloth: vs Q8_K_XL (lossless, PPL 1.4581) | 648.9 GB; top-1 81.2%; PPL None | 2026-07-27 |
| UD-IQ2_XXS |  |  | mean_kld | [0.3784](https://unsloth.ai/docs/models/kimi-k3) |  | unsloth dynamic | unsloth: vs Q8_K_XL (lossless, PPL 1.4581) | 711.1 GB; top-1 84.1%; PPL None | 2026-07-27 |
| UD-Q2_K_XL |  |  | mean_kld | [0.1779](https://unsloth.ai/docs/models/kimi-k3) |  | unsloth dynamic | unsloth: vs Q8_K_XL (lossless, PPL 1.4581) | 861.3 GB; top-1 90.4%; PPL 1.736 | 2026-07-27 |
| UD-Q4_K_XL |  |  | ppl | [1.4579](https://unsloth.ai/docs/models/kimi-k3) | 1.4581 | native MXFP4 experts | unsloth: vs Q8_K_XL (lossless, PPL 1.4581) | lossless (native 4-bit) | 2026-07-27 |
| IQ1_S |  |  | mean_kld | [0.5392](https://huggingface.co/AtomicChat/Kimi-K3-GGUF) |  |  | AtomicChat: vs Q8_0 | 590 GB; top-1 78.7%; PPL ratio 1.738 | 2026-07-27 |
| IQ1_M |  |  | mean_kld | [0.3957](https://huggingface.co/AtomicChat/Kimi-K3-GGUF) |  |  | AtomicChat: vs Q8_0 | 690 GB; top-1 None%; PPL ratio 1.495 | 2026-07-27 |
| IQ2_XXS |  |  | mean_kld | [0.2594](https://huggingface.co/AtomicChat/Kimi-K3-GGUF) |  |  | AtomicChat: vs Q8_0 | 769 GB; top-1 87.2%; PPL ratio 1.294 | 2026-07-27 |
| Q3_K_S |  |  | mean_kld | [0.2384](https://huggingface.co/AtomicChat/Kimi-K3-GGUF) |  |  | AtomicChat: vs Q8_0 | 1197 GB; top-1 87.7%; PPL ratio 1.276 | 2026-07-27 |

### Kimi-K2.7-Code

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| Q3_K_L | 3.85 |  | ppl_delta_pct | [3.9](https://huggingface.co/AesSedai/Kimi-K2.7-Code-GGUF) | 0 | AesSedai mix | AesSedai: vs Q4_X (native INT4, 4.55 bpw) | KLD 0.063 | 2026-06-15 |
| IQ3_S | 3.16 |  | ppl_delta_pct | [15.1](https://huggingface.co/AesSedai/Kimi-K2.7-Code-GGUF) | 0 | AesSedai mix | AesSedai: vs Q4_X (native INT4, 4.55 bpw) | KLD 0.1787 | 2026-06-15 |
| IQ2_S | 2.61 |  | ppl_delta_pct | [34.1](https://huggingface.co/AesSedai/Kimi-K2.7-Code-GGUF) | 0 | AesSedai mix | AesSedai: vs Q4_X (native INT4, 4.55 bpw) | KLD 0.3432 | 2026-06-15 |
| IQ2_XXS | 2.2 |  | ppl_delta_pct | [70.5](https://huggingface.co/AesSedai/Kimi-K2.7-Code-GGUF) | 0 | AesSedai mix | AesSedai: vs Q4_X (native INT4, 4.55 bpw) | KLD 0.5873 | 2026-06-15 |

### Kimi-K2.6

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| Q3_K_L |  |  | ppl_delta_pct | [3.8](https://huggingface.co/AesSedai/Kimi-K2.6-GGUF) | 0 | AesSedai mix | AesSedai: vs Q4_X (native INT4, 4.55 bpw) | KLD 0.0516 | 2026-04-20 |
| IQ3_S |  |  | ppl_delta_pct | [17.0](https://huggingface.co/AesSedai/Kimi-K2.6-GGUF) | 0 | AesSedai mix | AesSedai: vs Q4_X (native INT4, 4.55 bpw) | KLD 0.1752 | 2026-04-20 |
| IQ2_S |  |  | ppl_delta_pct | [35.3](https://huggingface.co/AesSedai/Kimi-K2.6-GGUF) | 0 | AesSedai mix | AesSedai: vs Q4_X (native INT4, 4.55 bpw) | KLD 0.3218 | 2026-04-20 |
| IQ2_XXS |  |  | ppl_delta_pct | [75.5](https://huggingface.co/AesSedai/Kimi-K2.6-GGUF) | 0 | AesSedai mix | AesSedai: vs Q4_X (native INT4, 4.55 bpw) | KLD 0.5826 | 2026-04-20 |
| IQ3_K | 3.849 |  | ppl_ratio | [1.0314](https://huggingface.co/ubergarm/Kimi-K2.6-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 1.8433 | PPL 1.9012 | 2026-04-20 |
| smol-IQ3_KS | 3.249 |  | ppl_ratio | [1.0747](https://huggingface.co/ubergarm/Kimi-K2.6-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 1.8433 | PPL 1.981 | 2026-04-20 |
| smol-IQ2_KL | 2.755 |  | ppl_ratio | [1.2038](https://huggingface.co/ubergarm/Kimi-K2.6-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 1.8433 | PPL 2.219 | 2026-04-20 |
| smol-IQ2_KS | 2.261 |  | ppl_ratio | [1.4497](https://huggingface.co/ubergarm/Kimi-K2.6-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 1.8433 | PPL 2.6723 | 2026-04-20 |
| smol-IQ1_KT | 1.832 |  | ppl_ratio | [1.8039](https://huggingface.co/ubergarm/Kimi-K2.6-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 1.8433 | PPL 3.3252 | 2026-04-20 |

### Kimi-K2-Thinking

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| ubergarm smol-IQ3_KS | 3.249 |  | aider_polyglot | [77.3](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF/discussions/14) | 80 | ik recipe | community Aider polyglot, single runs; reference reported ~80 |  | 2025-11 |
| unsloth UD-IQ3_XXS |  |  | aider_polyglot | [68](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF/discussions/14) | 80 | unsloth dynamic | community Aider polyglot, single runs; reference reported ~80 | approximate (~68) | 2025-11 |
| unsloth UD-IQ2_XXS |  |  | aider_polyglot | [50](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF/discussions/14) | 80 | unsloth dynamic | community Aider polyglot, single runs; reference reported ~80 | approximate (~50) | 2025-11 |
| smol-IQ4_KSS | 4.059 |  | ppl_ratio | [1.0252](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.0818 | PPL 2.1343 | 2025-11-07 |
| IQ3_K | 3.845 |  | ppl_ratio | [1.0306](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.0818 | PPL 2.1456 | 2025-11-07 |
| smol-IQ3_KS | 3.249 |  | ppl_ratio | [1.0742](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.0818 | PPL 2.2363 | 2025-11-07 |
| IQ2_KL | 2.92 |  | ppl_ratio | [1.1401](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.0818 | PPL 2.3735 | 2025-11-07 |
| smol-IQ2_KL | 2.755 |  | ppl_ratio | [1.1793](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.0818 | PPL 2.455 | 2025-11-07 |
| smol-IQ2_KS | 2.261 |  | ppl_ratio | [1.4104](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.0818 | PPL 2.9361 | 2025-11-07 |
| smol-IQ1_KT | 1.832 |  | ppl_ratio | [1.726](https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.0818 | PPL 3.5931 | 2025-11-07 |

### DeepSeek-V3.1-Terminus

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| IQ5_K | 5.941 |  | ppl_ratio | [1.0021](https://huggingface.co/ubergarm/DeepSeek-V3.1-Terminus-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.3929 | PPL 3.4 | 2025-09-23 |
| smol-IQ5_KS | 5.339 |  | ppl_ratio | [1.0038](https://huggingface.co/ubergarm/DeepSeek-V3.1-Terminus-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.3929 | PPL 3.4059 | 2025-09-23 |
| IQ4_K | 4.896 |  | ppl_ratio | [1.0079](https://huggingface.co/ubergarm/DeepSeek-V3.1-Terminus-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.3929 | PPL 3.4198 | 2025-09-23 |
| smol-IQ4_KSS | 4.08 |  | ppl_ratio | [1.0152](https://huggingface.co/ubergarm/DeepSeek-V3.1-Terminus-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.3929 | PPL 3.4445 | 2025-09-23 |
| IQ3_K | 3.724 |  | ppl_ratio | [1.0251](https://huggingface.co/ubergarm/DeepSeek-V3.1-Terminus-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.3929 | PPL 3.4781 | 2025-09-23 |
| IQ3_KS | 3.545 |  | ppl_ratio | [1.0367](https://huggingface.co/ubergarm/DeepSeek-V3.1-Terminus-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.3929 | PPL 3.5174 | 2025-09-23 |
| IQ2_KL | 2.962 |  | ppl_ratio | [1.0938](https://huggingface.co/ubergarm/DeepSeek-V3.1-Terminus-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.3929 | PPL 3.7112 | 2025-09-23 |
| IQ2_KS | 2.472 |  | ppl_ratio | [1.1872](https://huggingface.co/ubergarm/DeepSeek-V3.1-Terminus-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.3929 | PPL 4.028 | 2025-09-23 |
| IQ1_KT | 1.987 |  | ppl_ratio | [1.3354](https://huggingface.co/ubergarm/DeepSeek-V3.1-Terminus-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.3929 | PPL 4.531 | 2025-09-23 |
| IQ1_S | 1.745 |  | ppl_ratio | [1.616](https://huggingface.co/ubergarm/DeepSeek-V3.1-Terminus-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.3929 | PPL 5.4829 | 2025-09-23 |

### DeepSeek-V3.2-Speciale

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| IQ5_K | 5.946 |  | ppl_ratio | [1.0012](https://huggingface.co/ubergarm/DeepSeek-V3.2-Speciale-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.2082 | PPL 3.2122 | 2026-01-09 |
| IQ3_K | 3.724 |  | ppl_ratio | [1.0222](https://huggingface.co/ubergarm/DeepSeek-V3.2-Speciale-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.2082 | PPL 3.2793 | 2026-01-09 |
| smol-IQ2_KS | 2.297 |  | ppl_ratio | [1.2141](https://huggingface.co/ubergarm/DeepSeek-V3.2-Speciale-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.2082 | PPL 3.895 | 2026-01-09 |
| smol-IQ1_KT | 1.871 |  | ppl_ratio | [1.3597](https://huggingface.co/ubergarm/DeepSeek-V3.2-Speciale-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.2082 | PPL 4.3623 | 2026-01-09 |

### DeepSeek-R1-0528

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| IQ4_KS_R4 | 4.701 |  | ppl_ratio | [1.0049](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.213 | PPL 3.2286 | 2025-05-29 |
| IQ3_K_R4 | 3.847 |  | ppl_ratio | [1.0187](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.213 | PPL 3.273 | 2025-05-29 |
| IQ3_KS | 3.598 |  | ppl_ratio | [1.0265](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.213 | PPL 3.2983 | 2025-05-29 |
| IQ2_K_R4 | 2.799 |  | ppl_ratio | [1.0915](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.213 | PPL 3.5069 | 2025-05-29 |
| IQ1_S_R4 | 1.664 |  | ppl_ratio | [1.519](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.213 | PPL 4.8805 | 2025-05-29 |

### GLM-4.5

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| IQ5_K | 6.0 |  | ppl_ratio | [0.9969](https://huggingface.co/ubergarm/GLM-4.5-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.1788 | PPL 3.169 | 2025-08-01 |
| IQ4_K | 4.932 |  | ppl_ratio | [1.0126](https://huggingface.co/ubergarm/GLM-4.5-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.1788 | PPL 3.2189 | 2025-08-01 |
| IQ4_KSS | 4.164 |  | ppl_ratio | [1.0463](https://huggingface.co/ubergarm/GLM-4.5-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.1788 | PPL 3.3261 | 2025-08-01 |
| IQ3_KT | 3.537 |  | ppl_ratio | [1.0812](https://huggingface.co/ubergarm/GLM-4.5-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.1788 | PPL 3.4369 | 2025-08-01 |
| IQ2_KL | 3.062 |  | ppl_ratio | [1.1819](https://huggingface.co/ubergarm/GLM-4.5-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.1788 | PPL 3.7569 | 2025-08-01 |
| IQ2_KS | 2.671 |  | ppl_ratio | [1.3589](https://huggingface.co/ubergarm/GLM-4.5-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.1788 | PPL 4.3196 | 2025-08-01 |
| IQ1_KT | 2.009 |  | ppl_ratio | [1.6758](https://huggingface.co/ubergarm/GLM-4.5-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.1788 | PPL 5.327 | 2025-08-01 |

### GLM-4.6

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| IQ5_K | 5.997 |  | ppl_ratio | [0.9992](https://huggingface.co/ubergarm/GLM-4.6-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.4454 | PPL 3.4428 | 2025-10-01 |
| IQ4_K | 5.001 |  | ppl_ratio | [1.0088](https://huggingface.co/ubergarm/GLM-4.6-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.4454 | PPL 3.4758 | 2025-10-01 |
| IQ4_KS | 4.646 |  | ppl_ratio | [1.0248](https://huggingface.co/ubergarm/GLM-4.6-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.4454 | PPL 3.5309 | 2025-10-01 |
| smol-IQ4_KSS | 4.09 |  | ppl_ratio | [1.0423](https://huggingface.co/ubergarm/GLM-4.6-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.4454 | PPL 3.5911 | 2025-10-01 |
| IQ3_KS | 3.573 |  | ppl_ratio | [1.0573](https://huggingface.co/ubergarm/GLM-4.6-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.4454 | PPL 3.6427 | 2025-10-01 |
| IQ2_KL | 3.07 |  | ppl_ratio | [1.2032](https://huggingface.co/ubergarm/GLM-4.6-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.4454 | PPL 4.1456 | 2025-10-01 |
| smol-IQ2_KS | 2.359 |  | ppl_ratio | [1.5313](https://huggingface.co/ubergarm/GLM-4.6-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.4454 | PPL 5.276 | 2025-10-01 |
| smol-IQ1_KT | 1.948 |  | ppl_ratio | [1.7134](https://huggingface.co/ubergarm/GLM-4.6-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.4454 | PPL 5.9034 | 2025-10-01 |

### GLM-4.7

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| IQ5_K | 6.008 |  | ppl_ratio | [1.0045](https://huggingface.co/ubergarm/GLM-4.7-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.9267 | PPL 3.9445 | 2025-12-23 |
| IQ4_K | 5.021 |  | ppl_ratio | [1.0156](https://huggingface.co/ubergarm/GLM-4.7-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.9267 | PPL 3.9879 | 2025-12-23 |
| big-IQ3_KS | 4.12 |  | ppl_ratio | [1.0291](https://huggingface.co/ubergarm/GLM-4.7-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.9267 | PPL 4.041 | 2025-12-23 |
| IQ3_KS | 3.721 |  | ppl_ratio | [1.0525](https://huggingface.co/ubergarm/GLM-4.7-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.9267 | PPL 4.133 | 2025-12-23 |
| IQ2_KL | 3.099 |  | ppl_ratio | [1.1624](https://huggingface.co/ubergarm/GLM-4.7-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.9267 | PPL 4.5644 | 2025-12-23 |
| smol-IQ2_KS | 2.379 |  | ppl_ratio | [1.5208](https://huggingface.co/ubergarm/GLM-4.7-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.9267 | PPL 5.9716 | 2025-12-23 |
| smol-IQ1_KT | 1.976 |  | ppl_ratio | [1.7246](https://huggingface.co/ubergarm/GLM-4.7-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.9267 | PPL 6.772 | 2025-12-23 |

### GLM-5.1

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| smol-IQ4_K | 4.621 |  | ppl_ratio | [1.0106](https://huggingface.co/ubergarm/GLM-5.1-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.7257 | PPL 2.7547 | 2026-04-07 |
| IQ3_KS | 3.649 |  | ppl_ratio | [1.0559](https://huggingface.co/ubergarm/GLM-5.1-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.7257 | PPL 2.878 | 2026-04-07 |
| IQ2_KL | 2.985 |  | ppl_ratio | [1.1474](https://huggingface.co/ubergarm/GLM-5.1-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.7257 | PPL 3.1275 | 2026-04-07 |
| smol-IQ2_KS | 2.344 |  | ppl_ratio | [1.4241](https://huggingface.co/ubergarm/GLM-5.1-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.7257 | PPL 3.8818 | 2026-04-07 |
| smol-IQ1_KT | 1.928 |  | ppl_ratio | [1.7116](https://huggingface.co/ubergarm/GLM-5.1-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.7257 | PPL 4.6654 | 2026-04-07 |

### GLM-5

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| IQ3_KS | 3.649 |  | ppl_ratio | [1.0586](https://huggingface.co/ubergarm/GLM-5-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.6298 | PPL 2.7839 | 2026-02-14 |
| IQ2_KL | 2.985 |  | ppl_ratio | [1.149](https://huggingface.co/ubergarm/GLM-5-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.6298 | PPL 3.0217 | 2026-02-14 |
| smol-IQ2_KS | 2.344 |  | ppl_ratio | [1.4371](https://huggingface.co/ubergarm/GLM-5-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.6298 | PPL 3.7792 | 2026-02-14 |
| smol-IQ1_KT | 1.928 |  | ppl_ratio | [1.7504](https://huggingface.co/ubergarm/GLM-5-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.6298 | PPL 4.6032 | 2026-02-14 |

### Kimi-K2.5

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| IQ3_K | 3.845 |  | ppl_ratio | [1.0296](https://huggingface.co/ubergarm/Kimi-K2.5-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 1.8235 | PPL 1.8775 | 2026-02-06 |
| smol-IQ3_KS | 3.249 |  | ppl_ratio | [1.0728](https://huggingface.co/ubergarm/Kimi-K2.5-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 1.8235 | PPL 1.9562 | 2026-02-06 |
| smol-IQ2_KL | 2.755 |  | ppl_ratio | [1.1962](https://huggingface.co/ubergarm/Kimi-K2.5-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 1.8235 | PPL 2.1813 | 2026-02-06 |
| smol-IQ2_KS | 2.261 |  | ppl_ratio | [1.4373](https://huggingface.co/ubergarm/Kimi-K2.5-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 1.8235 | PPL 2.6209 | 2026-02-06 |
| smol-IQ1_KT | 1.832 |  | ppl_ratio | [1.7795](https://huggingface.co/ubergarm/Kimi-K2.5-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 1.8235 | PPL 3.245 | 2026-02-06 |

### Kimi-K2-Instruct-0905

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| smol-IQ5_KS | 5.295 |  | ppl_ratio | [1.0034](https://huggingface.co/ubergarm/Kimi-K2-Instruct-0905-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.4443 | PPL 2.4526 | 2025-09-05 |
| IQ4_KS | 4.633 |  | ppl_ratio | [1.0081](https://huggingface.co/ubergarm/Kimi-K2-Instruct-0905-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.4443 | PPL 2.4641 | 2025-09-05 |
| smol-IQ4_KSS | 4.059 |  | ppl_ratio | [1.0304](https://huggingface.co/ubergarm/Kimi-K2-Instruct-0905-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.4443 | PPL 2.5185 | 2025-09-05 |
| IQ3_KS | 3.52 |  | ppl_ratio | [1.049](https://huggingface.co/ubergarm/Kimi-K2-Instruct-0905-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.4443 | PPL 2.564 | 2025-09-05 |
| smol-IQ3_KS | 3.249 |  | ppl_ratio | [1.0597](https://huggingface.co/ubergarm/Kimi-K2-Instruct-0905-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.4443 | PPL 2.5902 | 2025-09-05 |
| IQ2_KL | 3.0 |  | ppl_ratio | [1.1452](https://huggingface.co/ubergarm/Kimi-K2-Instruct-0905-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.4443 | PPL 2.7993 | 2025-09-05 |
| smol-IQ2_KL | 2.755 |  | ppl_ratio | [1.1985](https://huggingface.co/ubergarm/Kimi-K2-Instruct-0905-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.4443 | PPL 2.9294 | 2025-09-05 |
| IQ2_KS | 2.425 |  | ppl_ratio | [1.3287](https://huggingface.co/ubergarm/Kimi-K2-Instruct-0905-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.4443 | PPL 3.2478 | 2025-09-05 |
| smol-IQ2_KS | 2.261 |  | ppl_ratio | [1.431](https://huggingface.co/ubergarm/Kimi-K2-Instruct-0905-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.4443 | PPL 3.4977 | 2025-09-05 |
| smol-IQ1_KT | 1.832 |  | ppl_ratio | [1.7274](https://huggingface.co/ubergarm/Kimi-K2-Instruct-0905-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.4443 | PPL 4.2224 | 2025-09-05 |

### Kimi-K2-Instruct

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| IQ4_KS | 4.638 |  | ppl_ratio | [1.0026](https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.9507 | PPL 2.9584 | 2025-07-14 |
| IQ3_KS | 3.604 |  | ppl_ratio | [1.0244](https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.9507 | PPL 3.0226 | 2025-07-14 |
| IQ2_KL | 2.923 |  | ppl_ratio | [1.0782](https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.9507 | PPL 3.1813 | 2025-07-14 |
| smol-IQ2_KL | 2.758 |  | ppl_ratio | [1.1552](https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.9507 | PPL 3.4086 | 2025-07-14 |
| IQ2_KS | 2.429 |  | ppl_ratio | [1.2481](https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.9507 | PPL 3.6827 | 2025-07-14 |
| IQ1_KT | 1.959 |  | ppl_ratio | [1.3466](https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.9507 | PPL 3.9734 | 2025-07-14 |
| smol-IQ1_KT | 1.835 |  | ppl_ratio | [1.4297](https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 2.9507 | PPL 4.2187 | 2025-07-14 |

### Qwen3-235B-A22B-2507

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| IQ5_K | 5.909 |  | ppl_ratio | [1.0063](https://huggingface.co/ubergarm/Qwen3-235B-A22B-Instruct-2507-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 4.3079 | PPL 4.3351 | 2025-07-21 |
| IQ4_K | 4.903 |  | ppl_ratio | [1.0137](https://huggingface.co/ubergarm/Qwen3-235B-A22B-Instruct-2507-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 4.3079 | PPL 4.3668 | 2025-07-21 |
| IQ4_KSS | 4.205 |  | ppl_ratio | [1.0218](https://huggingface.co/ubergarm/Qwen3-235B-A22B-Instruct-2507-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 4.3079 | PPL 4.4017 | 2025-07-21 |
| IQ3_K | 3.897 |  | ppl_ratio | [1.0344](https://huggingface.co/ubergarm/Qwen3-235B-A22B-Instruct-2507-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 4.3079 | PPL 4.4561 | 2025-07-21 |
| IQ3_KS | 3.702 |  | ppl_ratio | [1.0426](https://huggingface.co/ubergarm/Qwen3-235B-A22B-Instruct-2507-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 4.3079 | PPL 4.4915 | 2025-07-21 |
| IQ2_KL | 2.991 |  | ppl_ratio | [1.1122](https://huggingface.co/ubergarm/Qwen3-235B-A22B-Instruct-2507-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 4.3079 | PPL 4.7912 | 2025-07-21 |

### Qwen3-Coder-480B

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| IQ5_K | 5.9 |  | ppl_ratio | [1.0019](https://huggingface.co/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 5.0975 | PPL 5.1073 | 2025-07-23 |
| IQ4_K | 4.885 |  | ppl_ratio | [1.0055](https://huggingface.co/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 5.0975 | PPL 5.1257 | 2025-07-23 |
| IQ4_KSS | 4.18 |  | ppl_ratio | [1.0118](https://huggingface.co/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 5.0975 | PPL 5.1579 | 2025-07-23 |
| IQ3_K | 3.865 |  | ppl_ratio | [1.0163](https://huggingface.co/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 5.0975 | PPL 5.1808 | 2025-07-23 |
| IQ2_KL | 3.034 |  | ppl_ratio | [1.0616](https://huggingface.co/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 5.0975 | PPL 5.4113 | 2025-07-23 |
| IQ2_K | 2.588 |  | ppl_ratio | [1.1099](https://huggingface.co/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 5.0975 | PPL 5.6578 | 2025-07-23 |
| IQ2_KS | 2.578 |  | ppl_ratio | [1.1115](https://huggingface.co/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 5.0975 | PPL 5.6658 | 2025-07-23 |
| IQ1_KT | 1.945 |  | ppl_ratio | [1.2432](https://huggingface.co/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 5.0975 | PPL 6.337 | 2025-07-23 |
| unsloth UD-Q4_K_XL |  |  | aider_polyglot | [60.9](https://huggingface.co/unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF/discussions/8) | 61.8 | unsloth dynamic | community Aider vs reported BF16 |  | 2025-07 |

### Qwen3.5-397B

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| IQ4_KSS | 4.206 |  | ppl_ratio | [1.0072](https://huggingface.co/ubergarm/Qwen3.5-397B-A17B-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.4852 | PPL 3.5102 | 2026-02-16 |
| Q3_K | 3.9 |  | ppl_ratio | [1.016](https://huggingface.co/ubergarm/Qwen3.5-397B-A17B-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.4852 | PPL 3.5409 | 2026-02-16 |
| IQ2_KL | 2.994 |  | ppl_ratio | [1.0483](https://huggingface.co/ubergarm/Qwen3.5-397B-A17B-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.4852 | PPL 3.6536 | 2026-02-16 |
| smol-IQ2_XS | 2.46 |  | ppl_ratio | [1.1109](https://huggingface.co/ubergarm/Qwen3.5-397B-A17B-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.4852 | PPL 3.8717 | 2026-02-16 |
| smol-IQ2_KS | 2.344 |  | ppl_ratio | [1.1234](https://huggingface.co/ubergarm/Qwen3.5-397B-A17B-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.4852 | PPL 3.9153 | 2026-02-16 |
| smol-IQ1_KT | 1.925 |  | ppl_ratio | [1.2201](https://huggingface.co/ubergarm/Qwen3.5-397B-A17B-GGUF) | 1.0 | ubergarm ik recipe: attention/shared q8_0/iq6_k/iq5_ks, routed experts at named type | ubergarm: ik_llama.cpp wiki.test.raw ctx 512; reference PPL 3.4852 | PPL 4.2523 | 2026-02-16 |

### Qwen3-235B-A22B

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| unsloth Q5_K_M (llama.cpp) |  |  | aider_polyglot_whole_nothink | [59.1](https://aider.chat/2025/05/08/qwen3.html) | 65.3 |  | aider: vLLM bf16 vs llama.cpp Q5_K_M | runtime/template differences swamp the quant effect | 2025-05-08 |

### Mixtral-8x7B

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| MC-MoE PMQ 2.54-bit | 2.54 | 2.54 | avg8_zero_shot | [67.5](https://arxiv.org/abs/2410.06270) | 71.29 | pre-loading mixed precision (frequency + importance) | MC-MoE: 8-task zero-shot avg | PMQ 2.05: 63.25; 1.57: 54.49; BSP 2.54: 49.07; Hessian 2.54 ~66.4; GPTQ uniform 2-bit ~43.2 | 2024-10 |
| GPTQ uniform 3-bit | 3.0 | 3.0 | mmlu | [60.58](https://arxiv.org/abs/2410.06270) | 67.88 |  | MC-MoE |  | 2024-10 |
| frequent-expert 4-bit, rest 2-bit, attn 4-bit (2.54 avg) | 2.54 |  | avg_benchmarks | [54.2](https://arxiv.org/abs/2406.08155) |  | top-2 frequent experts/layer at 4 bit | QuantMoE-Bench | random allocation at same bits 49.10 +/- 7.73; uniform GPTQ 3-bit 70.69, 2-bit 40.53 | 2024-06 |
| frequent-expert 4-bit (3.03 avg) | 3.03 |  | avg_benchmarks | [63.82](https://arxiv.org/abs/2406.08155) |  | frequent experts at 4 bit | QuantMoE-Bench | random 63.70 +/- 0.49: no gain at 3 bits | 2024-06 |

### DeepSeek-MoE-16B

| quant | bpw overall | bpw experts | metric | value | reference | kept higher | harness | note | date |
|---|---|---|---|---|---|---|---|---|---|
| frequent-10 4-bit (2.53 avg) | 2.53 |  | avg_benchmarks | [62.99](https://arxiv.org/abs/2406.08155) |  | frequent experts at 4 bit | QuantMoE-Bench | random 62.86 +/- 0.60 | 2024-06 |

