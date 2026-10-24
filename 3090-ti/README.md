# NVIDIA RTX 3090 Ti (24 GB, Ampere)

**Box:** snoopy, an AM5 desktop: Ryzen 9 9950X3D (16 cores, AVX-512), 192 GB DDR5 as 4 DIMMs at 5200 MT/s, the card
on PCIe 4 x16. The [MoE offload entry](../moe-offload/) calls this box A. It is also a desktop in daily use, so the
servers on it stop when the desktop needs the card. The two B70s of [the B70 entry](../b70/) sat in this box until
2026-10-07; this card took their place.

The 3090 Ti has the most memory bandwidth of our cards (1,008 GB/s nameplate). On this box it is where a dense
27B model runs fastest, and where GLM-5.3-Flash (321B, MoE) cleared our 25 tok/s bar with its quality check passing.

## At a glance

The headline is the fastest result that passes our quality checks: Qwen3.8-27B at 103-105 in one pass (95-107 over 10 runs). Sage-A1.5 decodes faster
(154-171) but failed our code-review screen, so it is listed, not crowned.

All figures are measured on this box, one request at a time unless the row says otherwise. Decode and prefill are
tokens per second. "MTP 3" = speculative decoding with the model's own multi-token-prediction head, 3 drafted tokens
a step. Terms: [GLOSSARY](../GLOSSARY.md).

| model | engine and weights | result | conditions | quality check | details |
|---|---|---|---|---|---|
| Qwen3.8-27B (dense) | vLLM 0.31.0 (official CUDA image) + [`draft-vocab`](../b70/vllm-xpu/draft-vocab.py) + [`length-finish`](../b70/vllm-xpu/length-finish.py); AutoRound int4 (the same checkpoint as the B70) | **103-105 decode** at 1 stream (one pass; 95-107 over 10 runs); **279-288** for 3 streams together; prefill 1,407-1,410 on a 16k prompt; 88-95 decode at 32k depth | MTP 3 with a 50,521-token draft vocabulary, fp8 KV, 57,344-token window, temperature 0, thinking off, 512 tokens out, idle server, 2026-10-07 | the checkpoint passed the [B70 entry's](../b70/#at-a-glance) correctness eval (41 of 41 exact) | [Qwen3.8-27B on vLLM CUDA](#qwen38-27b-on-vllm-cuda-about-104-toks) |
| GLM-5.3-Flash (321B MoE) | the `glm53-flash-offload` engine (exllamav3-based), EXL3 3.05 bpw, 1,600 experts cached in VRAM, the rest computed on the CPU from RAM (a CPU lane on 14 cores, the same in the base and the trim runs) | **26.69 decode** with a route trim (keep 4 of 8 picks that miss VRAM, layers 29-44; not yet published); 23.08 without | 3 coding prompts x 300 tokens, two rounds (the arm's 4 agent briefs: 24.65), greedy, thinking off, 32,768-token window, 2026-10-20 | KL 0.0154, top-1 0.975 against the same weights computed exactly on the GPU: **passes** our bar (KL at most 0.03, top-1 at least 0.97) | [GLM route trim](#glm-53-flash-a-route-trim-clears-25-toks), [moe-offload points 1-3](../moe-offload/#what-we-learned-measured) |
| DeepSeek V4.1 Flash (748B MoE) | Strata built for sm_86 by us (not published yet), GSQ-RCO IQ2_XS experts; dense layers and hot experts in the card, the other 13,827 experts (134.2 GB) copied into RAM at load and computed on the CPU | **27.39 decode** (median over 217 agent requests with at least 50 tokens out; p10 25.57); prefill 240.5 (median over 85 reads of at least 1,000 tokens) | DSpark Q8_0 draft model (auto, confidence 0.65), 65,536-token window, served to a coding agent during our agent eval, 2026-10-24 | **13 of 15** agent tasks; no KL check against the release weights (not measured), and the experts are 2-bit | [V4.1 on a gaming card](#deepseek-v41-flash-27-toks-on-a-gaming-card) |
| Qwen3.8 Flash-Next (125B MoE, 6B active) | [Strata](https://github.com/Niko1221/Strata) `477ac13c` built for sm_86, Q8_0, every expert in RAM, a VRAM expert cache | **49.1 decode** (upper median of 6 requests: 3 coding prompts x 2, 300 tokens each; 42.3-55.4); prefill about 1,620 at 15k | MTP 4, the model card's thinking sampling; speed at a 32,768-token window, agent runs at 131,072; 2026-10-18 | **23 of 24** agent tasks over 2 passes (21 of 24 inside the 20-minute limit) | [Flash-Next](#qwen38-flash-next-agent-work-from-ram) |
| sabi-noema (a Qwen3.8-27B fine-tune) | llama.cpp `d454a20` CUDA, i1-Q4_K_M (16.55 GB) | 35.9-47.3 decode | no speculation, f16 KV, 49,152-token window, card sampling, thinking on, 2026-10-21 | **5 of 5** planted defects caught, each with a working test; 1 soft false positive on 2 clean controls | [Code-review rounds](#code-review-rounds-four-models-on-one-card) |
| Qwen3.8-27B and a coding fine-tune (GGUF) | the same llama.cpp build, UD-Q4_K_XL and Q3 GGUFs | 38.0-46.1 decode | as above | 4 of 5 each | [Code-review rounds](#code-review-rounds-four-models-on-one-card) |
| Sage-A1.5 | the same llama.cpp build, Q4_K_M (21.7 GB) | **154-171 decode**, no speculation; 202-215 with its MTP head in a separate speed-only probe over the first 4k output tokens (quality not screened with MTP) | card conditions: thinking on, 131,072-token context, up to 118,000 tokens out | 2 of 5 with 2 false claims: **fails** our bar. Fast, not reliable enough to review code | [Code-review rounds](#code-review-rounds-four-models-on-one-card) |

## Qwen3.8-27B on vLLM CUDA: about 104 tok/s

The same AutoRound int4 checkpoint the B70 serves (all files sha256-identical), so this card and a B70 are an
XPU-vs-CUDA comparison on one set of weights.

Measured 2026-10-07 with [`serve-ab.py`](../b70/bench/serve-ab.py), the same client on both cards: temperature 0, thinking off, `ignore_eos`,
salted prompts so no prefix cache hits, an idle server, 512 tokens out.

| measure | result |
|---|---|
| decode, 1 stream | 103-105 tok/s: the three decode requests of one pass (103.1, 104.2, 104.7); 95-107 over 10 runs |
| decode, 3 streams | 279-288 tok/s together: output tokens over the wall time, each request's prefill included, so below three times one stream; each stream decodes at 96-100 once it starts |
| prefill, 16k-token prompt | 1,407-1,410 tok/s (11.3 s) |
| decode at 32k depth | 88-95 tok/s (24.7 s to the first token) |
| decode at 55k depth | 88 tok/s (47 s to the first token) |
| MTP acceptance | 2.6-2.8 tokens a step |
| the reduced draft vocabulary | +28 %: 102-105 against 78-82 tok/s without it, at no window cost |
| card power | about 357-360 W decoding, about 108 W idle with the model loaded |

- **Against one B70** (B70 figures from other prompts, so provisional): CUDA decodes about 1.4x faster (B70: 74.6
  tok/s at 1 stream with MTP 3 and the same reduced draft vocabulary, from the
  [draft-vocab measurement](../PATCHES.md#draft-vocabpy-cheaper-drafts-for-speculative-decoding); the B70's 61 is its
  compiled-mode figure without the reduced vocabulary), and the B70 prefills about 1.15x faster (1,621 tok/s at 17.7k). Our reading, not
  measured directly: decode follows memory bandwidth (1,008 against 608 GB/s), prefill follows matrix throughput.
- **The window is the cost.** 17.8 GiB of weights leave about 2 GiB for the fp8 KV cache, so the largest window that
  starts is 57,344 tokens (45,056 while the desktop holds about 1.1 GB of the card). One B70 holds 212,992. A
  32 GB card's extra 8 GB buys 3.7x the window, because the weights are a fixed cost.
- **Impression** (from watching it work, not a number): snappy. A 512-token answer lands in about 5 s, three at once
  barely slow each other, and 32k of context hardly dents decode. Reading is its slow side: 25 s of silence before
  the first word at 32k. A fast answerer with a short memory, better for a review pass or a contained coding task
  than for a long agent session.

### Reproduce

The image is vLLM's official `vllm/vllm-openai:v0.31.0` plus two patches from this repo:
[`vllm-cuda/Dockerfile`](vllm-cuda/Dockerfile).

```bash
docker build -f 3090-ti/vllm-cuda/Dockerfile -t vllm-cuda:0.31.0-atlas .   # from the repo root
docker run --rm --gpus all --ipc=host --network=host -v /path/to/models:/models:ro \
  -e VLLM_DRAFT_VOCAB=/models/qwen38-freq32k.txt vllm-cuda:0.31.0-atlas \
  /models/Qwen3.8-27B-int4-AutoRound --served-model-name qwen27 --port 8083 \
  --dtype float16 --mamba-ssm-cache-dtype float16 --language-model-only \
  --max-model-len 57344 --kv-cache-dtype fp8 --max-num-seqs 3 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.95 --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --tool-strict-level parameter --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

- The draft list is [`b70/vllm-xpu/draft-vocab/qwen38-freq32k.txt`](../b70/vllm-xpu/draft-vocab/); copy it next to
  the model. Leave out `VLLM_DRAFT_VOCAB` to compare without it.
- The weights: `Frozenlock/Qwen3.8-27B-int4-AutoRound`, as in [the B70 entry](../b70/).
- Our server also passes a chat template that accepts the usual `reasoning_effort` words
  ([`effort-template.py`](../b70/vllm-xpu/effort-template.py) explains the fix) and three output gates that are not in
  this repo. Neither changes decode speed.
- vLLM 0.31.0 already has `strict-tools` upstream (`--tool-strict-level parameter`) and the streaming half of
  `length-finish`; the script skips the hunk that is already there.

## GLM-5.3-Flash: a route trim clears 25 tok/s

GLM-5.3-Flash routes each token to 8 experts per layer. On this box 1,600 of them sit in a VRAM cache and the rest are
computed on the CPU from a 107 GiB RAM tier. [moe-offload point 1](../moe-offload/#what-we-learned-measured) shows the
box is bound by that CPU expert work at the RAM-bandwidth wall: 23.08 tok/s.

A **route trim** keeps only the top n of a token's routed picks that miss the VRAM cache, in the layers named, with no
renormalisation. Picks already in VRAM are never dropped. It reads fewer experts from RAM per token. Which layers
tolerate it comes from the planner's per-layer sensitivity map ([Recipe Maker, layer map and route
trim](../recipe-maker/docs/how-it-works.md#layer-map-and-route-trim)). Measured 2026-10-20, EXL3 3.05 bpw, layer-ahead
warm-up on, the CPU lane on 14 cores, 3 coding prompts x 300 tokens plus 4 real coding-agent briefs (1.5-2.2k tokens) x 256 tokens,
greedy, thinking off. Each trim ran interleaved with an untrimmed arm on the same server, two rounds each (untrimmed,
trimmed, untrimmed, trimmed); a round is the median of 6 requests, and a figure is the mean of its two rounds. The
percent compares a trim with the untrimmed arm beside it:

| trim | decode (coding prompts): trim, its two rounds | untrimmed beside it | agent briefs, base / trim | KL | top-1 | verdict |
|---|---|---|---|---|---|---|
| none | - | 22.68-23.11 over the three sessions | 20.17-21.37 | 0.0045 (the floor: this session's untrimmed run; 0.0044 in moe-offload's earlier one) | - | pass |
| keep 5 on layers 27-44 | **25.65** (25.94, 25.35) | 22.68 (23.04, 22.32): +13 % | 21.15 / 24.79 | 0.0103 | 0.984 | **pass** |
| keep 4 on layers 29-44 | **26.69** (26.92, 26.45) | 22.77 (22.70, 22.85): +17 % | 20.17 / 24.65 | 0.0154 | 0.975 | **pass** |
| keep 4 on 27-44 and keep 7 on 26 | 27.60 (27.47, 27.72) | 23.11 (23.07, 23.14): +19 % | 21.37 / 25.62 | 0.0193 | 0.969 | **fail** (top-1) |

- KL and top-1 are against the same EXL3 weights computed exactly on the GPU, on our coding text; see
  [the quality reference problem](../moe-offload/#the-quality-reference-problem) for what that reference does and
  does not prove.
- The trim cuts CPU expert reads from about 170 to 137-145 per token.
- The planner predicted the KL of each arm before the run (0.011-0.017); it under-predicted the failing arm's.
- **Not yet published:** the trim is our change to the engine's routing, switched on by an environment variable we
  will name when the change is public. The untrimmed 23.08 is not the public image alone either: every arm here and
  in [moe-offload](../moe-offload/) ran the public `glm53-flash-offload` image with our unpublished CPU-lane changes
  mounted over it (the layer-ahead warm-up among them). These pages give no timing of the public image
  alone.

## Qwen3.8 Flash-Next: agent work from RAM

Flash-Next is a 125B MoE with about 6B active per token. Here it runs from RAM with the 3090 Ti holding the dense
weights, the KV cache and a cache of hot experts.

- **Engine:** [Strata](https://github.com/Niko1221/Strata) at `477ac13c`, built for sm_86. Q8_0 GGUF from
  `unsloth/Qwen3.8-Flash-Next-GGUF`, every expert in RAM, expert cache sized automatically, MTP 4, a coding expert
  profile.
- **Speed** (2026-10-18, one client, 32,768-token window): decode 49.13 tok/s (the upper median of 6 requests, 3 coding prompts x 2 of 300 tokens; 42.27-55.38), cold prefill about 1,620 tok/s
  at 15k tokens.
- **Agent work** (2026-10-18): our [agent eval](../b70/agent-eval/), Claude Code 2.1.291 in a sandbox on another box,
  the model card's thinking sampling (temperature 1.0, top_p 0.95, top_k 20), a 131,072-token window, 1,200 s per task.
  - 23 of 24 over 2 passes (11 of 12, then 12 of 12). Counting only passes inside the limit, 21 of 24: two tasks
    passed their checks on the agent's edits when the limit cut them.
  - Median task 477 s and 409 s.
  - Three real changes from our own repositories (1,800 s each, 2 passes): 4 of 6.
- The same model on one B70 with XeStrata decodes at 80-87 tok/s; see [the B70 entry](../b70/) and
  [moe-offload point 22](../moe-offload/#what-we-learned-measured).
- The recipe as the planner records it: [`recipe-maker/recipes/snoopy-flash-next-strata-q8.json`](../recipe-maker/recipes/snoopy-flash-next-strata-q8.json).

## DeepSeek V4.1 Flash: 27 tok/s on a gaming card

Measured 2026-10-24. DeepSeek V4.1 Flash is a 748B mixture-of-experts model. Here it runs on one RTX 3090 Ti plus 192 GB
of DDR5, served to a coding agent for real tasks: **27.39 tok/s decode**, the median over 217 requests that each
generated at least 50 tokens (p10 25.57). Prefill was 240.5 tok/s, the median over 85 prompt reads of at least
1,000 tokens. That is more than twice our best V4.1 result on the [RTX 6000 + Optane box](../rtx-6000/) (12.69).

- **Weights.** GSQ-RCO IQ2_XS experts: 2-bit, so much smaller than the release precision. That size is what lets the
  CPU part fit in RAM. We have not checked these weights against the release weights with KL and top-1 (not measured),
  so this row is not comparable with the RTX 6000 row's quality check.
- **Placement.** Dense layers and the hot experts are in the card (23.0 of 24.0 GB in use after load). The other 13,827
  experts (134.2 GB) are copied into RAM at load and computed on the CPU (Ryzen 9 9950X3D, native Zen 5 build). No
  expert is read from disk while decoding.
- **Engine.** [Strata](https://github.com/Niko1221/Strata), built for sm_86 by us; that build is not published yet.
  The draft model is the DSpark Q8_0 drafter, set to auto with confidence 0.65.
- **Workload.** Real agent tasks, not a benchmark prompt: 65,536-token window, prompts up to 34,588 tokens, with a
  median of 97.6 % of each prompt reused from the cache.
- **Quality.** **13 of 15** agent tasks passed: 11 of 12 on our task battery and 2 of 3 on real-repository tasks.
  One failed its hidden check. In the other, the context was compacted twice on a 17,000-line file at the 65,536 window
  and the agent made no edit. A bigger window is the obvious next arm. That is the same score as Flash-Next
  (Q8_0 + MTP, 131,072 window) on the same harness.

## Code-review rounds: four models on one card

Can a model that fits this card find a real bug in a diff? We ask each model the same 8 questions from our own
private test kit (not published: it is built from our own code):
- 5 diffs, each with one planted defect;
- 2 clean diffs, as controls;
- 1 repeat of an earlier screen question.

Each answer is graded against a key written before the runs. Its proposed test is run against the planted code and
the real code: a test "works" when it fails on the plant and passes on the real code.

**Conditions:** llama.cpp `d454a20` (upstream, CUDA 12.6, sm_86), `-ngl 99 -fa on -np 1 --jinja`, f16 KV, no
speculation, each model's card sampling with thinking on, one request at a time, an empty answer retried once,
2026-10-21. 49,152-token window and 32,768 tokens out, except Sage-A1.5's best-config row (below). Departures: build
jobs on the same box stalled decode once, so the server ran at a raised CPU priority from then on; two outages (a box
hang and a power cut) were resumed from the last answered question, discarding the partial request.

| model | planted defects caught | false positives, strict / lenient | design claims | repeat | tokens | wall | decode tok/s |
|---|---|---|---|---|---:|---:|---|
| sabi-noema i1-Q4_K_M | **5 of 5** | 1 / 0 | 0 | pass | 93,641 | 37 min | 35.9-47.3 |
| Qwen3.8-27B-Coder390 Q3 | 4 of 5 (one test does not work) | 1 / 0 | 1 | pass | 143,721 | 57 min | 38.0-46.1 |
| Qwen3.8-27B UD-Q4_K_XL | 4 of 5 (one test does not work) | 1 / 0 | 0 | pass | 203,808 | 82 min | 39.3-44.3 |
| Sage-A1.5 Q4_K_M, 32,768 out | 2 (11 of 14 requests ran out of tokens) | 0 / 0 (no control answered) | 0 | no answer | 445,911 | 49 min | 145-165 |
| Sage-A1.5 Q4_K_M, card conditions: 131,072 context, 118,000 out | 2 | 2 / 2 | 1 | pass | 339,479 | 35 min | 154-171 |

- **False positives.** A *soft* one calls real, pre-existing behaviour outside the diff a defect; a *hard* one claims
  something untrue, or blames the diff for it. Strict counts both, lenient counts only hard ones. A *design claim*
  calls intended, tested behaviour a defect and is counted apart.
- **Our bar** for a model that reads code for us: at least 4 of 5, at most 1 invented claim, and the repeat passed.
  sabi-noema, Coder390 and the stock 27B pass it. Sage-A1.5 fails it.
- **sabi-noema** used less than half the stock 27B's tokens for the same 8 answers (93,641 against 203,808), and wrote
  the only working test for the hardest plant.
- **Sage-A1.5: what we tried.** Its first round failed by running out of tokens, so we looked for the configuration
  it needs before judging it. Thinking off, an 8,192-token thinking budget with a stop message, a two-pass budget
  with a finalizer, its verification mode, and thinking off with DRY sampling all did worse: every budget or
  thinking-off arm committed to half-formed answers (3 hard false claims on one clean diff), and DRY corrupted code
  tokens. Only the card's own conditions, with room for up to 118,000 tokens out, answered every question. Under
  them it still caught 2 of 5, often naming a side issue as the defect and filing the real bug as a side note.
- **Impression** (read from the answers): sabi-noema goes straight to the line and stops. The stock 27B is the most
  thorough reader and the most expensive. Sage-A1.5 is thrilling to watch at 160 tok/s and calm on clean code, but
  not reliable enough to be the reviewer.

## What did not work

Banked with their conditions, because a loser on this box can be the right answer on another.

- **GLM's MTP draft loses here**: 17.22 tok/s at 3 drafted tokens, 20.25 at 2, against 22.20 without it: the figure before layer-ahead
  warm-up (the 23.08 base adds it; [moe-offload](../moe-offload/#best-measured-result-per-model) has both). The draft took 512 of the 1,600 VRAM expert slots ([moe-offload point 2](../moe-offload/#what-we-learned-measured)).
- **Fewer bits in GLM's middle layers fail quality**: 25.13 tok/s with the middle 30 layers at 2.05 bpw, but KL
  0.0716, top-1 0.943 ([point 3](../moe-offload/#what-we-learned-measured)). The route trim above is the version
  that passes.
- **The most aggressive route trim fails**: 27.60 tok/s, top-1 0.969 (table above).
- **Sage-A1.5 as a code reviewer**: fast, but 2 of 5 under every configuration we tried.

## Not measured yet

- Flash-Next on this card with a smaller quant (the B70 serves IQ3_XXS).
- Qwen3.8-27B with a window above 57,344 tokens (a smaller KV type, or a smaller quant).
- sabi-noema as an agent, rather than as a reviewer.

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [arc](https://github.com/akapug/arc) + [fab](https://github.com/akapug/fab) for [@akapug](https://github.com/akapug)
