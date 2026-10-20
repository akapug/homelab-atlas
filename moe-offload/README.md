# Big MoE models on one GPU plus system memory

Three mixture-of-experts models that do not fit in one card's VRAM, each served by keeping part of the expert
weights on the GPU and computing the rest on the CPU from system memory:

- GLM-5.3-Flash, 321B total parameters;
- DeepSeek V4.1 Flash, 748B;
- Qwen3.8 Flash-Next.

The question on every box: how close does one mid-range GPU plus the memory already in the machine get to
25 tok/s at one stream, without giving up output quality? 25 tok/s is our bar for a coding model we would
hand an agent's work to.

Box D asks it from the other side: a two-socket server with 1 TB of DRAM and only an 11 GB Pascal card, running
the largest model with its experts at release precision, almost entirely from DRAM.

## The boxes

| box | CPU | system memory | GPU |
|---|---|---|---|
| A: AM5 desktop | Ryzen 9 9950X3D, 16 cores | 192 GB DDR5 as 4 DIMMs, running at 5200 MT/s | RTX 3090 Ti 24 GB |
| B: 2019 workstation (HP Z8 G4) | 2x Xeon Gold 5215, 10 cores each | 192 GB DDR4-2666 (12 x 16 GB), plus 256 GB Optane PMem 100 (2 x 128 GB, one App Direct region) | Quadro RTX 6000 24 GB (Turing) |
| C: TRX40 workstation | Threadripper 3960X, 24 cores | 128 GB DDR4 as 4 x 32 GB | Intel Arc Pro B70 32 GB (one of two in the box) |
| D: two-socket EPYC server (Supermicro H11DSi) | 2x EPYC 7601 (Zen 1), 32 cores each, 8 NUMA nodes | 1 TB DDR4-2400 (16 x 64 GB LRDIMM, 16 channels) | GTX 1080 Ti 11 GB (Pascal). The box also holds an [MI25](../mi25/), not used here |

## Best measured result per model

All decode figures are one request at a time, greedy, thinking off, on coding prompts. How each row aggregates its
requests is named in the row.

| model | box | engine | weights | decode tok/s | quality check |
|---|---|---|---|---|---|
| GLM-5.3-Flash | A | the `glm53-flash-offload` image (ghcr.io/sybil-solutions, exllamav3-based): a VRAM expert cache of 1,600 slots, the other experts computed on 14 CPU cores from a 107 GiB RAM tier | EXL3 3.05 bpw | **23.08**, upper median of 6 requests of 300 tokens (22.20 without layer-ahead warm-up) | served vs the same weights computed exactly on the GPU: KL 0.0044, top-1 0.985 (928 positions) |
| GLM-5.3-Flash | B | ik_llama.cpp with our patches: hot experts on the GPU, each CPU expert's rows split between the two sockets, MTP draft 3 tokens with the reduced draft vocabulary (point 6), p_min 0.75, built with expert chunking off (point 13) | unsloth UD-Q4_K_XL | **16.06**, mean of 4 prompts, 3 rounds each (15.95 with the full draft vocabulary; 15.73 on our earlier build of the same recipe) | KL 0.030, top-1 0.960 against the EXL3 3.05 bpw store above (925 positions). See [the reference problem](#the-quality-reference-problem) |
| DeepSeek V4.1 Flash | B | ik_llama.cpp with our patches: as above, plus the DSpark draft model at n_max 4 (acceptance 0.95) | MXFP4 for hot experts, q2_K for cold ones | **10.6**, mean of two runs (10.58, 10.62), each over 3 prompts of 300 tokens (8.75 without the draft) | KL 0.0169, top-1 0.975 against the all-MXFP4 store (1,376 positions) |
| DeepSeek V4.1 Flash | D | ik_llama.cpp with our patches, CPU only: 64 threads, each expert's rows split across all 8 NUMA nodes, weights in a RAM disk, the DSpark draft model (Q2_K, on the CPU) at n_max 4 (acceptance 0.92) | routed experts at release precision (MXFP4, as DeepSeek released them), other tensors Q8_0 | **6.64**, mean of 3 prompts of 256 tokens, one request each (6.92, 7.58, 5.42); 4.45 without the draft | not measured on box D. The routed experts are as released (only the other tensors are quantized, to Q8_0); this store is the reference the box B row is measured against |
| Qwen3.8 Flash-Next | C | XeStrata (the SYCL port of Strata), one B70, about 10,900 experts in VRAM, the rest on the CPU from RAM, MTP 4, int8 KV, 262,144-token window | ISTA-DASLab GSQ-RCO IQ3_S | **87.1**, median of 3 runs (88.5, 87.1, 86.7) of 512 tokens on one code prompt, XeStrata xe0.1.45 with `--prefill auto` | not measured yet |

Box A and box B each miss 25 tok/s for GLM; box C clears it with a smaller model. Box D, on its CPUs, runs V4.1
at about a quarter of the bar.

## What we learned, measured

**1. On box A, GLM is bound by CPU expert work at the RAM-bandwidth wall.**
- At 23.08 tok/s a token takes about 43 ms.
- About 34 ms of that is the CPU computing the experts that are not in VRAM, reading roughly 2 GB per token at
  about 59.5 GB/s.
- Faster DIMMs or fewer CPU expert reads are the only levers left that are big enough.
- On longer, real agent prompts (1.5-2.2k tokens in) decode runs about 9 % below the short-prompt figure: 20.19 tok/s.

**2. Speculative decoding can lose when the draft competes for the VRAM expert cache.**
- On box A, the engine's MTP drafting at 3 tokens gave 17.22 tok/s and at 2 tokens 20.25, against 22.20 without it.
- The draft layer and its buffers took 512 of the 1,600 VRAM expert slots, so VRAM hits per token fell from 121 to 63.
- Drafting also switched off the engine's lookahead.
- On box B, where the draft costs no expert slots, MTP helps: GLM ran at 12.30 tok/s with no draft and 15.95 with
  MTP at 3 tokens (+30 %). A build with expert chunking on gave 9.41 and 12.71 (+35 %); see point 13.

**3. Lowering the bits of the wrong layers fails quality.**
- On box A, a split store with the middle 30 layers' experts at 2.05 bpw reached 25.13 tok/s.
- It failed the quality gate: KL 0.0716, top-1 0.943 (our bar: KL at most 0.03, top-1 at least 0.97).
- The same weights at 3.05 bpw everywhere pass at KL 0.0044.
- Project Maya reports that GLM's late layers tolerate much more pruning than its early and middle ones. That fits
  this failure, but we have not yet measured a late-layer-only cut.

**4. On a two-socket box, split each CPU expert's rows between the sockets.**
- With each socket computing the half of every expert that sits in its own memory, a kernel benchmark on box B ran
  1.37-1.40x faster than with interleaved pages.
- The served runs also improved: GLM 11.10 → 15.03 tok/s, V4.1 7.82 → 10.58.
- Those served numbers come from runs that changed other things too (thread layout, placement), so treat the kernel ratio as
  the fair measure of this one change.

**5. CPU weight repacking (`-rtr`) was slower on this path.**
- GLM on box B: up/gate time per distinct CPU expert went from 129 µs to 195 µs.
- Decode went from 15.03 to 10.04 tok/s.
- We serve with it off.

**6. A smaller draft vocabulary only pays where the draft head is a large share of a step.**
- A reduced MTP head of 61,039 token ids, chosen by token frequency, gave +0.7 % on box B. There the head is about 1.5 % of a ~245 ms step.
- The same idea gave +25-29 % on the B70 with vLLM; see [the B70 vLLM entry](../b70/vllm-xpu/).
- Choosing the ids by frequency kept acceptance at 0.940. Keeping the first 61,039 ids instead dropped it to 0.914.
- The quick test before building it: head bytes x draft calls per step, against the step time.
- Repeated on box B with expert chunking off: +0.7 % (+0.5 to +0.8 %, higher on all 12 prompt pairs), with
  identical drafts and identical output text.

**7. More Optane would add capacity, not speed, on box B.**
- Box B has 2 PMem sticks. We modelled filling socket 0 to six DDR4 + six PMem DIMMs.
- Our planner predicts V4.1 at 12.1-12.5 tok/s (+15-18 %) and GLM at about 16.2 (+3 %).
- After that, the bind is per-token fixed overhead plus CPU expert compute. Even infinitely fast PMem gives V4.1 only +23 %.
- What the sticks would buy is room: V4.1 with every expert at release precision, or a 1T-class model at 1-2 bits.
- These are predictions, not measurements, from [Recipe Maker](../recipe-maker/).

**8. Flash-Next on the B70 keeps its speed with a modest RAM budget.**
- With the container limited to 42 GiB and the model's pages dropped before start, decode stayed at 70-80 tok/s from short prompts
  out to 64k depth.
- At 12 GiB it fell to 20-22 tok/s, and at 8 GiB to 17-21.
- It held 71-72 tok/s out to 128k depth through a 79-minute soak with the box's normal build load running beside it.

**9. On box D, V4.1 runs from DRAM at 6.6 tok/s, and expert placement is the biggest lever.**

Conditions for every box D figure: ik_llama.cpp with our patches, built for the 1080 Ti (sm_61) with expert
chunking off; 64 threads on cores 0-63; 8,192-token window, flash attention; the DSpark Q2_K draft on the CPU at
n_max 4 (conf_cum_min 0.5); greedy, 256 tokens, one request per prompt. The coding prompts are 3 agent briefs of
1.5-2.2k tokens. The weights sit in a RAM disk (tmpfs) interleaved over 7 of the 8 NUMA nodes. "Split" is box B's
store (MXFP4 hot experts, q2_K cold ones); "release" is the store in the table above.

| layout | store | coding prompts, draft / none | ~600-token prompt | ~3.7k-token prompt | prefill tok/s | acceptance |
|---|---|---|---|---|---|---|
| experts read in place from the RAM disk | split | 5.16 / 4.06 | 3.89 / 4.14 | 5.58 / 4.03 | 37.3 | 0.89 |
| each expert's rows split across all 8 nodes | split | 6.53 / 4.39 | 4.85 / 4.46 | 7.01 / 4.43 | 40.0 | 0.89 |
| read in place, 1080 Ti without attention (point 11) | split | 5.58 / 4.56 | 5.27 / 4.55 | 5.26 / 4.61 | 39.3 | 0.93 |
| read in place | release | 5.31 / 4.12 | 6.63 / 4.18 | 5.54 / 4.08 | 38.8 | 0.92 |
| each expert's rows split across all 8 nodes | release | **6.64** / 4.45 | 8.28 / 4.52 | 6.85 / 4.41 | 41.8 | 0.92 |

- Splitting expert rows across the 8 nodes: +27 % with the draft, +8 % without (split store). Its load took 715 s
  against 23 s.
- The draft gains more here: 1.49x on box D, against 1.21x on box B (10.58 / 8.75).
- Without the draft a token takes about 227 ms on box D, against about 114 ms on box B. Our reading, not measured:
  16 DDR4 channels do not win, because every layer pays a fixed cost across 8 small NUMA nodes and the links
  between them.
- Wall power peaked at 522-587 W in these runs.

**10. On box D the low-bit store buys no speed: serve the experts as released.**
- The release store runs at the split store's speed: 6.64 vs 6.53 placed, 5.31 vs 5.16 read in place.
- It reads only about 1.2-1.3x the split store's routed bytes per token, not 2x: the split store keeps 22.5 % of
  routed reads at 2 bits.
- Its draft acceptance is higher: 0.92 against 0.89.
- Equal speed at more bytes is weak evidence that decode on this box is not bound by DRAM bytes. It is not settled.

**11. The GTX 1080 Ti helps a little, and only without attention.**
- With the experts read in place, the 1080 Ti holding the router, shared experts, norms and output (4.5 GiB with
  its buffers) gave +8 % with the draft and +12 % without (5.58 vs 5.16, 4.56 vs 4.06). A large file copy ran on
  the box during most of that run.
- Any V4.1 attention weight, or its KV cache, on the 1080 Ti gave garbage text in this build. We bisected it: only
  attention weights and KV cache both on the CPU gave correct text. A small Qwen3 model ran correctly on the same
  build and card, so the fault is in V4.1's attention path on Pascal (sm_61). The cause is not found.
- The placed layout plus the GPU crashed at warm-up with a GPU memory fault on host memory. Our diagnosis from the
  logs: with weights loaded without mmap, our build's pinned-memory cap was used up by weight buffers, so a buffer
  the GPU reads directly stayed unpinned. It needs an engine fix, so the gains of points 9 and 11 do not stack yet.

**12. Verifying many tokens in one pass costs almost as much per token: a block drafter would not pay here.**

One forward pass over K new tokens on a cached 600-token prompt, drafts off. Median of 4 rounds on box B, 3 on
box D. Each cell is ms per pass, and the ratio to K = 1.

| K | GLM, box B | V4.1, box B | V4.1, box D |
|---|---|---|---|
| 1 | 174 | 192 | 269 |
| 2 | 232 (1.33) | 291 (1.51) | 486 (1.81) |
| 4 | 307 (1.76) | 389 (2.03) | 580 (2.15) |
| 8 | 468 (2.68) | 562 (2.93) | 727 (2.70) |
| 16 | 1023 (5.87) | 916 (4.77) | 1019 (3.78) |
| 32 | 1493 (8.56) | 1506 (7.84) | 1522 (5.65) |

- If every drafted token were accepted, the best possible rate is K divided by the pass time: at K = 32 about
  21 tok/s in all three columns.
- Our test before building a block drafter is K = 8 at most 1.5x K = 1. All three are at 2.7-2.9x, so no.
- Why, from an op profile of the GLM pass on box B: a pass costs about 120 ms fixed plus about 0.12 ms per distinct
  CPU expert it reads, and the distinct experts grow 1.7-1.9x per doubling of K up to K = 8.
- Depth barely matters: on box B, a 4,000-token prompt added at most 2 ms per pass for GLM and 13-18 ms for V4.1.
- Derived, not measured: for GLM on box B a block drafter would beat MTP at 3 tokens (15.8 tok/s) only with
  per-token acceptance of about 0.95 or better across 8 or more tokens. The best deep draft we measured, MTP at 7
  tokens, accepted 0.69.
- These passes carry per-request overhead that decode verification mostly skips. Our estimate is that taking it
  out makes the ratios worse, not better. On these boxes the lever stays bytes per token.

**13. Build ik_llama.cpp with expert chunking off.**
- Stock ik_llama.cpp has a build option, `GGML_EXPERT_CHUNKING`, on by default. It is upstream's, not one of our
  patches.
- GLM on box B, two loads of the same source and layout, with the option the only intended difference, 4 coding
  prompts of 256 tokens: MTP at 3 tokens, p_min 0.75, went from 12.71 tok/s with chunking on to 15.95 with it off
  (+25.5 %); no draft, from 9.41 to 12.30 (+31 %).
- Build with `-DGGML_EXPERT_CHUNKING=OFF`. The box B GLM figures in the table and all box D figures were built
  with it off. We measured the effect on this box and model only.

**14. For GLM's MTP draft, p_min 0.75-0.85 is a plateau; 0.9 loses.**

Box B, chunking off, reduced draft vocabulary, MTP at 3 tokens, 4 prompts of 256 tokens, 2 rounds per value in
reversed order. "Paired" is the range over prompt pairs against 0.75.

| p_min | tok/s | paired vs 0.75 | acceptance |
|---|---|---|---|
| 0.75 | 16.07 | | 0.954 |
| 0.80 | 16.16 | +0.6 % (-0.3 to +2.4) | 0.964 |
| 0.85 | 16.06 | -0.0 % (-1.1 to +2.0) | 0.962 |
| 0.90 | 15.67 | -2.4 % (-4.8 to -1.4) | 0.947 |

- No draft ran at 12.27 before the sweep and 12.18 after it. MTP at 3 tokens is 1.31x that.
- We keep 0.75: 0.80 is inside the noise.

## The quality reference problem

Our GLM quality gate compares against the EXL3 3.05 bpw store computed exactly on the GPU, because a BF16
reference of a 321B model fits none of our cards.

So the gate measures agreement with a 3-bit store, not fidelity to the original weights. A store that is closer
to BF16 can be penalized for disagreeing with the reference's own errors. The box B GLM line is a measured
miss against that reference (KL 0.02997, top-1 0.960, 925 positions); its fidelity to BF16 is unmeasured.

For scale, unsloth reports KL against BF16 of 0.049 for UD-Q4_K_XL and 0.160 for UD-Q3_K_XL, on their own prompts.
A near-lossless reference (Q8_0 on a large-memory CPU box) is the next step.

## Not published yet

- The ik_llama.cpp patches behind box B's numbers: socket-split expert rows, the reduced draft vocabulary, and
  hot-expert placement. They will follow with build steps.
