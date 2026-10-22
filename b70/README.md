# Intel Arc Pro B70 (32 GB): serving Qwen3.8-27B

The Arc Pro B70 is a 32 GB Xe2 (Battlemage) workstation card. One card holds a 27B model at
4 bits with room for a useful KV cache. This entry is what we measured serving
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) on it (a hybrid model: 48 Gated DeltaNet
layers and 16 attention layers, plus a multi-token-prediction head usable for speculative
decoding), and the patches that got us there.

**Box:** since 2026-10-07 the two B70s sit in a Threadripper 3960X box (TRX40, 24 cores, 128 GB DDR4 as 4 x 32 GB,
both cards on PCIe 4 x16); the [MoE offload entry](../moe-offload/) calls it box C. Before that they were in the
Ryzen 9 9950X3D box that now holds the [RTX 3090 Ti](../3090-ti/), and the results under "Results" below were
measured there.

## At a glance

One B70 per row, one request at a time unless the row says otherwise. Terms: [GLOSSARY](../GLOSSARY.md).

| model | engine and weights | result | conditions | quality check | details |
|---|---|---|---|---|---|
| Qwen3.8-27B (dense) | vLLM (llm-scaler 0.26.0-b2) + [our changes](vllm-xpu/), AutoRound int4 | **61.2-61.5 decode** at 1 stream, **351.4** summed at 8 | MTP 3, compiled mode with XPU graphs (not yet supported by Intel; 54.4 at 1 stream in eager mode, which our servers ran), 1.5k-token prompts, Ryzen host, 2026-09 | 41 of 41 exact on our correctness eval; 22 of 24 [agent tasks](agent-eval/) | [Results](#results) |
| Qwen3.8 Flash-Next (125B MoE, 6B active) | [XeStrata](https://github.com/chriscorbell/intel-arc-pro-b70-qwen38-flash-next) (the SYCL port of Strata), GSQ-RCO IQ3_S; about 10,900 experts in VRAM, the rest computed on the CPU | **87.1 decode** (median of 3 runs, 512 tokens, one code prompt); 79.9 on the installed server at a short prompt, 70.4 after a 67.6k prompt | MTP 4, int8 KV, 262,144-token window, Threadripper host; speed 2026-10-19, agent tasks 2026-10-18; card 1 has served the smaller IQ3_XXS since 2026-10-20 | **11 of 12** agent tasks in one pass; 3 of 3 real repository changes | [moe-offload point 22](../moe-offload/#what-we-learned-measured) |
| Ornith-1.5-35B-A3B (35B MoE, 3B active) | vLLM (llm-scaler 0.26.0-b2), int4 | **198 decode** at 1 stream, 290 summed at 2; 164-167 with 4 host cores while builds ran on the other CCD | MTP 3 with the smaller drafter vocabulary, 1,200-1,500-token replies, Ryzen host, 2026-10-04 | 17 of 24 agent tasks; 31 of 40 household questions | [Ornith-1.5-35B-A3B](ornith-35b-a3b.md) |

What the two cards serve today (2026-10-22): card 1 runs Qwen3.8 Flash-Next (IQ3_XXS since 2026-10-20) as a coding
model; card 2 runs Ornith-1.5-35B-A3B (a prequantized GPTQ copy since 2026-10-19, MTP off for the reason in
[its entry](ornith-35b-a3b.md#a-bug-mtp-garbled-about-3--of-replies-from-the-first-token)) for a household assistant
and a coding agent.

Card 2 has also run block-diffusion models on a stock diffusion engine (q4_0, 2026-10-19): LLaDA2.1-mini at 59.71 tok/s
over 33 HumanEval prompts, Ling-mini-2.0 at 147.46 over 4 short prompts. Their conditions are in
[RESULTS.md](../recipe-maker/results/RESULTS.md); nothing tuned yet.

## Results

Aggregate decode tok/s, one B70, prompts at ~1.5k and ~16k tokens, 4-bit weights. "MTP 3" is
speculative decoding with the model's own MTP head, 3 draft tokens.

| stack | 1 stream, 1.5k / 16k | 4 streams, 1.5k / 16k | 8 streams, 1.5k |
|---|---|---|---|
| llama.cpp SYCL, upstream `94256114c2`, no speculation | | 39.3 / | |
| llama.cpp SYCL + [our patches](llama.cpp-sycl/) 0001-0017, no speculation | 24.4 / 22.7 | 64 / 57 | |
| llama.cpp SYCL + all 18 patches, MTP 3 | 43.1-44.2 / 36.1 | 100.7-101.6 / 91.6-91.7 | |
| vLLM ([llm-scaler 0.26.0-b2](vllm-xpu/)), BF16 quantized at load, compiled, MTP 3 | 60.4 / 56.1 | 194.4 / 178.9 | |
| vLLM, AutoRound int4 + [inc-q40](vllm-xpu/), compiled, MTP 3 | 61.2-61.5 / 58.6 | 197.0-201.0 / 192.1 | 351.4 |

The card's limit for one stream without speculation is ~35 tok/s: each step reads ~16.5 GB of
weights, and we measured ~590 GB/s sustained (608 nameplate). vLLM compiled reaches 32.8
without speculation, ~94 % of that; our llama.cpp build reaches 24.4.

**Which to use.** vLLM with an AutoRound int4 checkpoint and both fixes in
[`vllm-xpu/`](vllm-xpu/) is the fastest at every stream count we measured, and it matched the
llama.cpp GGUF on our correctness eval (41 of 41 exact; 186 of 187 reasons quoted verbatim against
187). llama.cpp is the simpler install and keeps working without a patched image; its patches are
in [`llama.cpp-sycl/`](llama.cpp-sycl/). The speed-up from speculation depends on the text (see
[`bench/`](bench/)).

## Contents

- [`vllm-xpu/`](vllm-xpu/): compiled mode for llm-scaler (issue #699), AutoRound int4 on the fast
  kernels (`inc-q40`), a smaller vocabulary for the MTP drafter, two tool-call fixes for agent
  clients, a chat-template fix for `reasoning_effort`, a guard that keeps other work off the server's
  CPUs, our run of the DeltaNet mixed-batch test ([`gdn-mixed-batch/`](vllm-xpu/gdn-mixed-batch/)),
  and a CPU check for quantized checkpoints.
- [`llama.cpp-sycl/`](llama.cpp-sycl/): 18 patches, each with its measured effect, and a build
  script that verifies the patched source tree.
- [`bench/`](bench/): the decode benchmarks, the speculative step-time and CPU-contention probes, the
  structured-output probe and the tool-call length check.
- [`qwen36-35b-a3b.md`](qwen36-35b-a3b.md): the MoE model our agents run on (Qwen3.6-35B-A3B): vLLM
  at 2.1-5.1x llama.cpp Vulkan on one card, its full 262k window, a silent load hang to watch for,
  and running Claude Code on it.
- [`ornith-35b-a3b.md`](ornith-35b-a3b.md): Ornith-1.5-35B-A3B, a fine-tune of the MoE above, serving a household
  assistant and a coding agent from one card with the house first: it coded 17 of 24 against Qwen3.6's 13, a
  speculative-decoding bug that garbled 3 % of replies and its repro, thinking with a budget, photos, and voice on CPUs.
- [`agent-eval/`](agent-eval/): can a local model finish real repository work? Twelve tasks with
  hidden checks, a sandboxed runner, and 72 runs: on all twelve, the dense 27B finished 22 of 24
  and the MoE 15 of 24 (p = 0.036), at little more speed once the tasks got hard.
- [`../PATCHES.md`](../PATCHES.md): every patch and tool above, sorted by the hardware and software it
  needs, with how to apply, check and undo each.

## Related work

- [CySpiegel/vllm-intel](https://github.com/CySpiegel/vllm-intel): upstream vLLM tuned on two B70s
  (tensor parallel 2) for Qwen3.8-27B, with XPU fixes proposed to vLLM and to
  vllm-xpu-kernels. One matters to anyone serving this model with speculative decoding:
  [vllm-xpu-kernels#552](https://github.com/vllm-project/vllm-xpu-kernels/pull/552), an
  out-of-bounds write in the DeltaNet kernel for batches that mix speculative and
  non-speculative sequences. Our run of its test on this image is in
  [`vllm-xpu/gdn-mixed-batch/`](vllm-xpu/gdn-mixed-batch/).
- intel/llm-scaler issues we have added data to:
  [#699](https://github.com/intel/llm-scaler/issues/699) (compiled mode),
  [#698](https://github.com/intel/llm-scaler/issues/698) (XPU graphs; Intel: not supported yet),
  [#636](https://github.com/intel/llm-scaler/issues/636) (AutoRound checkpoints).
- For NVIDIA cards, [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen) serves the same model
  family on vLLM with many patches; we measured it on a Turing Quadro RTX 6000 and sent the one
  fix it needed there ([HyperQwen#188](https://github.com/syv-ai/HyperQwen/pull/188)).

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [arc](https://github.com/akapug/arc) + [fab](https://github.com/akapug/fab) for [@akapug](https://github.com/akapug)
