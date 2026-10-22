# Big MoE models on one GPU plus system memory

Four mixture-of-experts models that do not fit in one card's VRAM, each served by keeping part of the expert
weights on the GPU and computing the rest on the CPU from system memory:

- GLM-5.3-Flash, 321B total parameters;
- DeepSeek V4.1 Flash, 748B;
- Qwen3.8 Flash-Next;
- Ornith-1.5-122B, a fine-tune of Qwen3.5-122B-A10B (122B total, about 10B active per token).

The question on every box: how close does one mid-range GPU plus the memory already in the machine get to
25 tok/s at one stream, without giving up output quality? 25 tok/s is our bar for a coding model we would
hand an agent's work to.

Box D asks it from the other side: a two-socket server with 1 TB of DRAM and only an 11 GB Pascal card, running
the largest model with its routed experts at release precision, all from DRAM.

## The boxes

Each box also has its own page, with its best results first: A = [RTX 3090 Ti](../3090-ti/), B = [RTX 6000 +
Optane](../rtx-6000/), C = [B70](../b70/), D = [GTX 1080 Ti](../1080-ti/) and [its CPUs alone](../epyc-cpu/).

| box | CPU | system memory | GPU |
|---|---|---|---|
| A: AM5 desktop | Ryzen 9 9950X3D, 16 cores | 192 GB DDR5 as 4 DIMMs, running at 5200 MT/s | RTX 3090 Ti 24 GB |
| B: 2019 workstation (HP Z8 G4) | 2x Xeon Gold 5215, 10 cores each | 192 GB DDR4-2666 (12 x 16 GB), plus 256 GB Optane PMem 100 (2 x 128 GB, one App Direct region) | Quadro RTX 6000 24 GB (Turing) |
| C: TRX40 workstation | Threadripper 3960X, 24 cores | 128 GB DDR4 as 4 x 32 GB | Intel Arc Pro B70 32 GB (one of two in the box) |
| D: two-socket EPYC server (Supermicro H11DSi) | 2x EPYC 7601 (Zen 1), 32 cores each, 8 NUMA nodes | 1 TB DDR4-2400 (16 x 64 GB LRDIMM, 16 channels) | GTX 1080 Ti 11 GB (Pascal). The box also holds an [MI25](../mi25/), used only in points 18 and 23 |

## Best measured result per model

All decode figures are one request at a time, greedy, thinking off, on coding prompts. How each row aggregates its
requests is named in the row.

| model | box | engine | weights | decode tok/s | quality check |
|---|---|---|---|---|---|
| GLM-5.3-Flash | A | the `glm53-flash-offload` image (ghcr.io/sybil-solutions, exllamav3-based): a VRAM expert cache of 1,600 slots, the other experts computed on 14 CPU cores from a 107 GiB RAM tier | EXL3 3.05 bpw | **23.08**, upper median of 6 requests of 300 tokens (22.20 without layer-ahead warm-up) | served vs the same weights computed exactly on the GPU: KL 0.0044, top-1 0.985 (928 positions) |
| GLM-5.3-Flash, route trim | A | as above, plus a route trim: on layers 29-44, of the 8 experts a token picks, only the top 4 of those missing VRAM are computed (not yet published) | EXL3 3.05 bpw | **26.69**, same prompts and aggregation, 2026-10-20 | KL 0.0154, top-1 0.975 against the same weights computed exactly on the GPU: passes our bar ([the 3090 Ti entry](../3090-ti/#glm-53-flash-a-route-trim-clears-25-toks)) |
| GLM-5.3-Flash | B | ik_llama.cpp with our patches: hot experts on the GPU, each CPU expert's rows split between the two sockets, MTP draft 3 tokens with the reduced draft vocabulary (point 6), p_min 0.75, built with expert chunking off (point 13) | unsloth UD-Q4_K_XL | **16.06**, mean of 4 prompts, 3 rounds each (15.95 with the full draft vocabulary; 15.73 on our earlier build of the same recipe) | KL 0.030, top-1 0.960 against the EXL3 3.05 bpw store above (925 positions). See [the reference problem](#the-quality-reference-problem) |
| DeepSeek V4.1 Flash | B | ik_llama.cpp with our public patches, including the newer public ones ([below](#engine-changes)): hot experts on the GPU (7.0 GiB), each CPU expert's rows split between the two sockets, the draft model's experts on the CPU, the DSpark draft model at n_max 4 (acceptance 0.90) | MXFP4 for hot experts, q2_K for cold ones | **11.88**, mean of 3 prompts of 300 tokens, one load (12.88, 12.62, 10.13); 9.72 without the draft (point 21). Before the newer changes: 10.6 and 8.75 | KL 0.0169, top-1 0.975 against the all-MXFP4 store (1,376 positions), measured with the earlier build; not re-measured on the new one |
| DeepSeek V4.1 Flash, newest build | B | the recipe above plus two changes not yet in our public fork (expert pre-binding, a PMem-aware thread split) and one upstream change (an F32 indexer, ik_llama.cpp #2712) | as above | **12.69**, mean of 3 prompts of 300 tokens, 2026-10-22; 10.29 without the draft; prefill 69.0 | KL 0.0169, top-1 0.975, measured on the earlier build only, not re-measured on this one ([the RTX 6000 entry](../rtx-6000/#deepseek-v41-flash-the-best-run)) |
| DeepSeek V4.1 Flash | D | ik_llama.cpp with our patches ([below](#engine-changes)): the GTX 1080 Ti holds attention, KV cache, the dense and shared-expert weights and the output head; the routed experts stay on the CPUs, each expert's rows split across all 8 NUMA nodes; 64 threads; the DSpark draft model (Q2_K, on the CPU) at n_max 4 (acceptance 0.92) | routed experts at release precision (MXFP4, as DeepSeek released them), other tensors Q8_0 | **7.13**, mean of 3 prompts of 300 tokens, one request each (7.24, 7.46, 6.70); 6.97 without the draft (point 17). CPU only: 7.02 and 5.16 (point 16) | not measured on box D. The routed experts are as released (only the other tensors are quantized, to Q8_0); this store is the reference the box B row is measured against |
| Qwen3.8 Flash-Next | C | XeStrata (the SYCL port of Strata), one B70, about 10,900 experts in VRAM, the rest on the CPU from RAM, MTP 4, int8 KV, 262,144-token window | ISTA-DASLab GSQ-RCO IQ3_S | **87.1**, median of 3 runs (88.5, 87.1, 86.7) of 512 tokens on one code prompt, XeStrata xe0.1.45 with `--prefill auto` | our 12 agent tasks 11 of 12, and 3 of 3 real repository changes (IQ3_S, point 22); IQ3_XXS served since 2026-10-20 |
| Ornith-1.5-122B | B | ik_llama.cpp with our patches: the RTX 6000 holds attention, the dense and shared-expert weights, the MTP layer's non-expert weights and the routed experts of layers 0-8; the other routed experts (the MTP layer's too) in DRAM, each expert's rows split between the two sockets; 39 threads; its own MTP layer at n_max 3, p_min 0.75; 49,152-token window | ornith-ai's Q4_K_M GGUF (70.6 GiB) | **31.6** at 4,000 tokens of context and 31.0 at 600, mean of 3 prompts of 300 tokens (33.72, 29.84, 31.23 at 4,000); 27.1-27.3 without the draft | code-review screens: both planted defects found only with an 8k thinking budget; agent tasks 0 of 3 in a configuration that limited it, so not yet a verdict on the model (points 19 and 20) |

Box A clears 25 tok/s for GLM with a route trim that passes our quality bar (26.69; 23.08 without it); box B misses
it (16.06); box C clears it with a smaller model. Box B clears it with
Ornith-1.5-122B; its first agent runs failed in a configuration of ours that limited it (point 20). Box D runs V4.1 at under a third of the bar.

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

**11. Two engine faults kept V4.1's attention off the GTX 1080 Ti. Both are fixed now.**
- First, without attention: with the experts read in place, the 1080 Ti holding the router, shared experts, norms
  and output (4.5 GiB with its buffers) gave +8 % with the draft and +12 % without (5.58 vs 5.16, 4.56 vs 4.06). A
  large file copy ran on the box during most of that run.
- Garbage text. Any V4.1 attention weight, or its KV cache, on the 1080 Ti gave garbage text; a small Qwen3 model ran
  correctly on the same build and card. We compared per-op sums of a GPU-attention run against a CPU-attention run on
  a 7-token prompt. They agree through the attention itself; the first op that departs is the attention output
  projection (49.22 on the CPU, 15.36 on the GPU), and the same matrix multiply is correct on this card when tested
  alone. The cause, from the backend each op ran on and the engine source: this card has no CUDA kernel for V4's
  flash attention (head size 512 with sinks), so that op runs on the CPU and writes its output into host memory. The
  in-place inverse rope that follows was scheduled on the GPU, and the next GPU step had already copied its input
  from host memory before that write landed, so the projection read the un-rotated values. The fix is in the
  engine's scheduler ([below](#engine-changes)). With it the per-op sums agree to within rounding,
  and the greedy text matches the CPU run (identical on 2 of 3 prompts in point 17, the third forking at a near-tie).
  Our reading, not measured: any device that cannot run this attention kernel would hit the same bug, not only
  Pascal.
- A crash at warm-up. With weights loaded without mmap, the placed layout plus the GPU crashed with a GPU memory fault
  on host memory (Xid 31). Our build had pinned 79 GiB of weight buffers up to its pin cap, so the compute buffer
  the GPU reads directly stayed unpinned. With weights kept in plain, never-pinned host memory
  ([below](#engine-changes)) the same arm ran with no fault. Every fault we logged was a GPU write to
  host memory, which is also the in-place bug's signature, so which of the two caused each crash is not settled.
- With both fixed, the gains of points 9 and 11 stack: point 17.

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

Points 15-18 are box D with V4.1 on the release store. Conditions, unless a point says otherwise: weights in a RAM
disk and loaded without mmap; routed experts row-split across all 8 NUMA nodes (point 9); 64 threads on cores 0-63;
8,192-token window, flash attention; the 3 coding-agent prompts of point 9 (1.5-2.2k tokens), 300 greedy tokens each,
one request at a time, first with the DSpark draft at n_max 4 ("draft") and then without it on the cached prompt ("no
draft"). Each figure is the mean of the 3 prompts.

**15. On box D, more threads never help V4.1; 32 threads match 64.**

A thread sweep, CPU only, experts read in place from the RAM disk, threads spread evenly over the 8 nodes (96 and 128
add SMT siblings):

| threads | no draft | draft | prefill tok/s |
|---|---|---|---|
| 32 | 4.15 | 5.45 | 24.0 |
| 48 | 4.04 | 4.86 | 33.1 |
| 64 | 4.00 | 5.21 | 38.9 |
| 96 | 3.86 | 4.89 | 32.2 |
| 128 | 3.42 | 4.63 | 36.2 |

- With experts row-split, 32 threads (4 cores per node) against 64: no draft 4.73 against 4.42 (+7 %), prefill 23.7
  against 36.8, wall power 406 W against 532 W at peak. The 4.42 leaves out one outlier prompt at 3.32; the same
  prompt ran at 4.46 earlier in the day. With the draft, 6.55 against 6.45 over the 3 prompts, but 32 threads were
  about 6 % slower on the 2 prompts whose texts are identical at both thread counts (6.37 and 6.90 against 6.76 and
  7.31).
- An op profile at 32 threads, no draft, 245 ms a token: routed experts 25 %, the Q8_0 attention matrix multiplies
  about 31 %, shared experts 9 %, flash attention 7 %, the BF16 output head 5 %.
- Our reading, not measured directly: decode here is bound neither by compute nor by thread synchronization, but by
  reads that cross NUMA nodes. The attention weights sat interleaved over all nodes, so about 7 of every 8 of their
  reads were remote. Point 16 tests that.
- We keep 64 threads for agent turns (prefill +55 %, drafts verify faster on the matched prompts) and would use 32
  for long generation without a draft.

**16. Place the dense weights per NUMA node too: +9 % with the draft, +17 % without.**
- An engine change ([below](#engine-changes)) lets a placement rule name any weight, not only routed
  experts, and split a dense matrix into 8 row slices, one per node. With `-cr` listing the cores in node order, the
  engine's static row split then has each node's threads read its own slice.
- We placed the attention projections, the shared experts and the output head this way, beside the row-split
  routed experts. Same session, 64 threads:

| layout | draft | no draft | prefill tok/s |
|---|---|---|---|
| routed experts row-split | 6.45 (6.76, 7.31, 5.28) | 4.42 (4.44, 4.40; the outlier of point 15 left out) | 36.8 |
| + dense weights row-split | **7.02** (7.45, 7.70, 5.91) | **5.16** (5.18, 5.13, 5.17) | 36.2 |

- With the draft, all 3 texts and draft counts are identical to the run without the change. Without it, 2 of 3
  texts are identical; the third forks at a near-tie, as it also does between thread counts.
- The same change on Qwen3-30B-A3B Q4_K_S (64 threads, 128 tokens, one text in all four arms): 21.31 and 21.42 tok/s
  with two unpatched builds, 21.26 patched without dense rules, 23.81 with attention and head row-split (+11 %).
- This was the best CPU-only V4.1 recipe on box D: 7.02 tok/s with the draft, 5.16 without. Box B keeps its dense
  weights on its GPU, so the change does nothing there today; we have not measured it on box B.

**17. The 1080 Ti holding attention, KV cache and the dense weights: +35 % without the draft, and the draft stops paying.**

With both faults of point 11 fixed, the 1080 Ti holds attention, the KV cache, the router, the shared experts, the
norms and the output head (`-ngl 99 --cpu-moe`); routed experts stay row-split on the CPUs, and the draft model's
weights on the CPU. Micro-batch 1024 with attention on the GPU, 2048 in the row without it, 4096 on the CPU only.

| arm | draft | no draft | prefill tok/s | 1080 Ti memory |
|---|---|---|---|---|
| CPU only, point 16 | 7.02 | 5.16 | 36.2 | - |
| GPU holds everything above except attention and KV | 6.59 | 5.60 | 37.2 | 6,385 MiB peak |
| **GPU holds attention and KV too** | **7.13** (7.24, 7.46, 6.70) | **6.97** (6.95, 7.00, 6.95) | **48.2** | 9,579 MiB after load, 11,129 peak, of 11,264 |
| the same, KV cache in host memory (`-nkvo`) | 7.05 | 6.67 | 48.8 | |
| the same, the draft model's dense weights on the GPU too (micro-batch 512) | 5.57 | 6.90 | 46.7 | |

- Against CPU only: no draft +35 %, prefill +33 %. No GPU memory faults in any of these runs. Wall power with
  attention on the GPU: median 574 W, peak 590 W; CPU only peaked at 529 W.
- Without the draft, 2 of 3 texts are identical to the CPU-only run; the third forks at the near-tie of point 16.
- The draft now buys 1.02x (7.13 against 6.97). On the CPU it bought 1.36x. Our reading: with the dense work on the
  GPU, verifying 5 tokens costs mostly the CPU reads of the union of the experts those 5 tokens route to, and the
  dense work the draft used to amortize is gone. Putting the draft's own weights on the GPU made it slower still
  (5.57). So on this layout we would serve without a draft.
- Keeping the KV cache on the GPU wins (6.97 against 6.67 without the draft), even though the flash-attention op
  itself runs on the CPU on this card (point 11).
- Hot experts on the 1080 Ti cost 3-4 %. The memory left over holds 28 hot experts (502 MiB), 4.0 % of the routed
  picks on held-out coding text. With them, no draft ran at 6.72, against 6.97 for the same layout without them and
  6.93 in a third run. (In the run with them the draft model ran out of GPU memory at every step, so only its no-draft
  figure is valid.) The hot-expert idea needs a bigger card: point 18.

**18. The MI25 as a hot-expert store for V4.1: correct text, but slower in every version. We stopped it.**
- The 16 GB MI25 in the same box holds the 799 hottest routed experts (14 GiB, 31.6 % of the routed picks on
  held-out coding text) under Vulkan, in the same process as the CUDA 1080 Ti (an engine change,
  [below](#engine-changes)). The rest is the layout of point 17. The text is identical to the run without it.
- First run: 2.66 tok/s with the draft and 4.08 without, against 7.24 and 6.95 for that prompt without the MI25.
- After three fixes (the MI25 forced to its high clock state, a mat-vec Vulkan kernel for small batches, and a bank
  order that really runs the MI25 first): 6.19 with the draft and 5.61 without, mean of 3 prompts, against 7.13 and
  6.97 without the MI25.
- The op profile per step: the banks save about 13.5 ms of CPU expert time and add about 40 ms of waiting across
  devices. With 31.6 % coverage every layer still runs its CPU expert pass, and the bank adds hops in series with it.
  Details: [the MI25 entry](../mi25/#as-a-hot-expert-store-for-deepseek-v41-stopped-bound-by-hop-latency).
- The same two cards win on a model whose experts they hold completely: point 23.

**19. Ornith-1.5-122B on box B: the first model above 25 tok/s on that box.**

ornith-ai's Q4_K_M GGUF, loaded without mmap from a RAM disk in PMem. The RTX 6000 holds attention, the dense and
shared-expert weights, the MTP layer's non-expert weights and the routed experts of layers 0-8 (21,066 of 23,040 MiB
at peak); the other routed experts sit in DRAM, each expert's rows split between the two sockets, none in PMem. 39
threads, micro-batch 2048, 49,152-token window, expert chunking off (point 13). Three coding-agent prompts cut to 600
and to 4,000 tokens, 300 greedy tokens each; each figure is the mean of the 3 prompts.

| draft | 600 tokens deep | 4,000 deep | acceptance (600 / 4,000) |
|---|---|---|---|
| none | 27.32 | 27.11 | |
| MTP n_max 1 | 30.70 | 30.78 | 0.75 / 0.76 |
| MTP n_max 2 | 30.82 | 30.79 | 0.75 / 0.76 |
| MTP n_max 3, p_min 0.75 | 30.99 | 31.60 | 0.73 / 0.77 |
| the same, repeated at the end | 31.13 | 31.55 | |

- Prefill of 2,048-token prompts: 165.5 tok/s (164.5, 166.0, 166.0).
- Context depth costs almost nothing here (-0.8 % without the draft). 36 of its 48 layers are Gated DeltaNet, whose
  state does not grow with context.
- The draft adds 12-17 %, and n_max 1, 2 and 3 land within 3 % of each other. Our reading: the model has one MTP
  layer, and at p_min 0.75 deeper drafts are rarely made.
- Our planner predicted 20.7 tok/s with the draft and 15.9 without; it borrows its per-token host cost from GLM,
  and for this model that is 35-42 % low.

**20. Ornith-1.5-122B as a reviewer and as an agent on box B: our configuration failed first.**

These use our own private tests, not published here.
- Code-review screens: three diffs of our own code, two with one planted defect each and one clean control. Card
  sampling (temperature 0.6, top_p 0.95, top_k 20).
  - Thinking on, no budget, up to 32,768 tokens: no answer on one planted diff and on the clean control (all
    32,768 tokens spent reasoning), a partial answer on the other planted diff. On the first, its reasoning found the
    planted defect within about 1,000 tokens and then kept deliberating.
  - Thinking off: it said "no defect" to all three, so it missed both plants and was right only on the clean control.
  - Thinking on with an 8,192-token reasoning budget (`--reasoning-budget 8192` and a message telling it to commit):
    it found both planted defects, with a fix and a test, but promoted a wording note to a defect on the clean
    control. 8,954-9,435 tokens and 334-397 s per review.
- Agent tasks: three tasks of real repository work from our private battery, run through an agent client and a
  proxy with a 45-minute wall each, the 8,192-token budget, a 262,144-token window with a Q8_0 KV cache, and the routed
  experts of 7 or 8 layers on the GPU. 0 of 3 passed, and no task got as far as editing a file in 17-28 turns: it
  read and searched for 15-25 turns, then deliberated up to the budget.
  - Decode fell from 27.5 tok/s at 21k tokens of context to about 18 past 60k; turns that reached the budget ran at
    18.0-19.3 tok/s, about 8 minutes each. Incremental prefill about 130 tok/s.
  - The server died twice in prefills at 66-76k tokens of context: a CUDA out-of-memory after 55 minutes with 8
    layers of experts on the GPU (its memory use had crept from 22,098 to 22,474 of 23,040 MiB), and a cuBLAS
    internal error after 85 minutes with 7. Two more runs were void because of the first crash.
  - Qwen3.8 Flash-Next on box C ran the same tasks at 60-68 tok/s, 38-101 turns in the same wall.
- Our reading: these runs judge our configuration more than the model.
  - At 18-19 tok/s, every turn that ran to the 8,192-token budget took about 8 minutes, so a 45-minute wall left
    room for only a few such turns.
  - The 262,144-token window with a Q8_0 KV cache does not fit beside these weights on a 24 GB card: both crashes
    came in prefills past 65k.
  - A re-run with a configuration fixed for both is queued. Until then, the model's agent ability on this box is
    not measured.

**21. The same engine changes on box B: V4.1 from 10.6 to 11.9 tok/s.**

Box B's V4.1 recipe of the table (the split store, hot experts on the RTX 6000, CPU experts row-split between the
sockets, 39 threads, the DSpark draft at n_max 4) on a build that carries every engine change below, with weights
never pinned and in-place ops following their target switched on. Three more changes ride with it: the draft model's experts on the CPU of socket 1, hot experts grown from 2.2 to 7.0 GiB (420 experts), and
DRAM filled with the hottest experts the GPU does not hold, the rest in PMem. Micro-batch 4096. The 3 coding-agent
prompts, 300 greedy tokens each, one load; each figure is the mean of the 3 prompts.

| build | draft | no draft | prefill tok/s |
|---|---|---|---|
| earlier (the 10.6 row) | 10.58 | 8.75 | about 52 |
| with the newer changes | **11.88** (12.88, 12.62, 10.13) | **9.72** (9.44, 9.83, 9.90) | **64.9** |

- +12 % with the draft, +11 % without, +25 % prefill. Draft acceptance 0.90. The GPU used 21,512 of 23,040 MiB.
- Without the draft, prompt 0 gives the same text as the earlier build; prompts 1 and 2 share their first 850 and 244
  characters and then drift, both coherent. Our reading: rounding from the 4.8 GiB more experts computed on the GPU.
- Which change buys how much: the same recipe and build, each run with one change turned off, same prompts, one load
  each (2026-10-21).

| turned off | draft | no draft | texts |
|---|---|---|---|
| nothing (the row above) | 11.88 | 9.72 | |
| no host weights | 10.99 | 9.04 | same |
| in-place ops follow their target | 11.92 | 9.78 | same |
| the DRAM fill that skips experts the GPU holds (plain placement instead) | 11.31 | 9.31 | same |
| the draft model's experts on the CPU (the whole draft on the GPU, so only 3.7 GiB of hot experts fit) | 11.65 | 9.40 | differ |
| all of these, with 2.2 GiB of hot experts (the earlier recipe on the new build) | 10.70 | 8.80 | |

  - No host weights is worth +8 % with the draft and +7.5 % without; the fill that skips GPU-held experts +5 % and +4 %; the CPU draft
    experts with the larger hot set +2 % and +3 %. In-place ops following their target changes nothing here (11.92 off against 11.88 on is within run-to-run noise): it is
    the fix that lets the GTX 1080 Ti run V4.1's attention ([engine changes](#engine-changes)), and on this box the
    graph is the same with or without it.
  - With everything off, the new build matches the earlier one (10.70 and 8.80 against 10.58 and 8.75, the same
    texts without the draft). So the prefill gain, 62-64 against 52-54, comes from the rebased build, not from any of
    these changes.
  - Moving the whole draft model to the CPU is a dead end: 2.44-2.69 tok/s, because a CPU draft round costs more than
    a main-model token, and its drafts were never accepted (0 of 1,086, not chased).
  - Two arms were not run: 7.6 GiB of hot experts, and 2.2 GiB of hot experts with every other change on.
- One more engine change, not yet in our public fork, binds each expert weight to its NUMA node before the load reads
  it. On it the same recipe ran 12.13 (13.15, 12.90, 10.33) with the draft and 10.00 without, prefill 65.5, with the
  same texts. That is +2 %, inside the run-to-run spread we have not measured, so the table keeps 11.88.
- Our planner predicted 8.88 and 7.05, 25-27 % low.

**22. Flash-Next on one B70 does real agent work, and a smaller quant was as good and faster.**

Flash-Next on box C as a standing server: XeStrata xe0.1.40.2.2, the GSQ-RCO IQ3_S store, about 10,900
experts in VRAM, MTP 4, int8 KV. The agent client is Claude Code in a sandbox on another box, with the model card's
thinking sampling. These use our own private tests, not published here. One pass of each unless noted.
- **Our 12-task battery** (the tasks of the [B70 agent eval](../b70/agent-eval/), 1,200 s per task, a
  229,376-token window): 11 of 12, median 410 s a task (2026-10-18).
  - The one failure is the same task that Strata with a Q8_0 store missed on box A.
  - Two tasks reached the 1,200 s limit and still passed their checks.
- **Three real changes from our own repositories**, replayed from their parent commit and judged by the change's
  own tests (1,800 s each), with the box's normal build load beside the server: 3 of 3, 22-30 minutes each.
- **Speed on the installed server**, cold page cache:
  - quiet box: decode 79.9 tok/s at a short prompt, 77.4 after a cold 33.8k prompt, 70.4 after 67.6k;
  - beside normal build load: a cold 135,454-token prompt prefilled in 65 s (2,094 tok/s), with decode 71.2 after it.
- **A harder tier** (8 harder tasks at 1,200 s, and 3 multi-file changes at 2,700 s): the wall decides.
  - At the highest reasoning effort (`xhigh`), 8 of 11 runs were stopped by the limit and 4 of 11 passed, mean
    1,463 s a task. Hosted DeepSeek V4.1 Flash and GLM-5.3-Flash took 550 and 744 s on the same tasks.
  - At medium effort, 5 of 11 were stopped. 5 of the 6 runs that finished in time passed.
  - On the three multi-file changes it passed none at medium or `xhigh`, nor in a low-effort probe on one of them.
    96-99 % of its output tokens were thinking, and at medium it made no edit at all on two of them.
  - With a cap on thinking per turn and a nudge to act (our client-side scaffold): 2 of the 3 passed, n = 1 each.
    One of them no other model we ran has passed, hosted ones included. The third ended when a capped turn
    produced text and no tool call, which ends a headless run.
- **IQ3_XXS against IQ3_S** (2026-10-20, XeStrata xe0.1.45, the two multi-file changes, 2,700 s, one run per
  quant per change): one pass each. The failing run always made many capped turns (14 and 27) and the passing
  one few (1 and 2): the capping tracked the change, not the quant.
  - Per-request decode medians on this agent traffic were 64-69 tok/s for IQ3_XXS against 60-61 for IQ3_S, about
    8 % faster. The depths were not matched.
  - Equal pass counts at n = 1 per change show no quality difference either way; they do not show equality. We
    serve IQ3_XXS, the faster and smaller file, on that basis.

**23. Box D with a model that fits its GPUs: Ornith-1.5-35B-A3B at 59-60 tok/s on the MI25 and the 1080 Ti.**

Ornith-1.5-35B-A3B Q4_K_M has about 18.2 GiB of routed experts: too much for either card, enough for both. The MI25
holds the experts of 30 whole layers (13.5 GiB), the 1080 Ti holds the other 10 (4.7 GiB) beside attention, the
DeltaNet layers, the KV cache and the dense weights, and the model's own MTP head drafts. 64 threads, `-c 8192`,
the 3 coding-agent prompts of 300 greedy tokens, one load per arm, each figure the mean of 3 prompts.

| arm | routed experts | decode tok/s | wall power (median) |
|---|---|---|---|
| CPUs only for experts | CPUs, rows over 8 nodes | 41.99 without a draft; 47.34 with MTP 3 | 542 / 598 W |
| both GPUs | MI25 + 1080 Ti banks | 44.62 without a draft; **59.13 with MTP 3, 60.01 with MTP 2** (acceptance 85 %) | 484 / 508 W |

- In real chat requests at the card's sampling, with an 8,192-token thinking budget: 45.7-53.3 tok/s at 13-21k
  tokens of context, prefill 318-354 tok/s.
- On box D, gpt-oss-120b (MXFP4, experts on the CPUs, no draft) decodes 31.0 tok/s, 34.6 with the experts of
  layers 0-2 on the 1080 Ti. Its experts cannot go in our banks yet (biases, `SWIGLU_OAI`).
- Code-review screens (point 20's kit): Ornith-1.5-35B-A3B passed one plant, ranked the other below a wrong primary,
  and claimed a defect in the clean control. gpt-oss-120b missed one plant, found the other, and was right on the
  clean control. One sample each.
- So box D's choice depends on the job: Ornith-1.5-35B-A3B for speed, about 8.3x V4.1 on the same box (59.13 against 7.13, both with a draft);
  gpt-oss-120b where a false alarm on clean code costs more. Neither is shown to be the better reviewer.
- Everything else, with the commands and the switches: [the MI25 entry](../mi25/#ornith-15-35b-a3b-on-the-mi25-and-a-gtx-1080-ti-about-60-toks).

## The quality reference problem

Our GLM quality gate compares against the EXL3 3.05 bpw store computed exactly on the GPU, because a BF16
reference of a 321B model fits none of our cards.

So the gate measures agreement with a 3-bit store, not fidelity to the original weights. A store that is closer
to BF16 can be penalized for disagreeing with the reference's own errors. The box B GLM line is a measured
miss against that reference (KL 0.02997, top-1 0.960, 925 positions); its fidelity to BF16 is unmeasured.

For scale, unsloth reports KL against BF16 of 0.049 for UD-Q4_K_XL and 0.160 for UD-Q3_K_XL, on their own prompts.
A near-lossless reference (Q8_0 on a large-memory CPU box) is the next step.

## Engine changes

Our changes to ik_llama.cpp behind the box B and box D numbers are public in our fork, [akapug/ik_llama.cpp](https://github.com/akapug/ik_llama.cpp).
Each branch contains the one before it, so the last one carries them all:

| branch | commit | adds |
|---|---|---|
| [`mtp-draft-vocab`](https://github.com/akapug/ik_llama.cpp/tree/mtp-draft-vocab) | `ee53cdd6` | expert rows per node, the reduced draft vocabulary, hot-expert placement |
| [`expert-placement-dense-rows`](https://github.com/akapug/ik_llama.cpp/tree/expert-placement-dense-rows) | `3e3c25b1` | dense rows per node |
| [`no-host-weights`](https://github.com/akapug/ik_llama.cpp/tree/no-host-weights) | `d93aeb48` | no host weights |
| [`sched-inplace-follow`](https://github.com/akapug/ik_llama.cpp/tree/sched-inplace-follow) | `5716ef61` | in-place ops follow their target (the V4.1 attention fix for sm_61 cards) |
| [`mixed-cuda-vulkan-expert-banks`](https://github.com/akapug/ik_llama.cpp/tree/mixed-cuda-vulkan-expert-banks) | `63bf5755` (later commits change only docs) | CUDA and Vulkan in one process, the MI25 bank fixes, a bank device per layer, whole-layer banks |

The fork's [PATCHES.md](https://github.com/akapug/ik_llama.cpp/blob/mixed-cuda-vulkan-expert-banks/PATCHES.md) gives each switch, its default, the
measured effect and the known limits. Its box letters are its own: its box A is our box D, and its box B is our box B.
Each change is off by default unless it says otherwise, so a build with them and no switches set runs as before.

| change | what it does | switch | measured effect |
|---|---|---|---|
| expert rows per node | places each routed expert's rows across the sockets or NUMA nodes you name, so each node's threads compute the slice in their own memory | `LLAMA_EXPERT_PLACEMENT` rules, `GGML_MOE_GROUPS` | kernel 1.37-1.40x on box B (point 4); V4.1 on box D +27 % with the draft, +8 % without (point 9) |
| reduced draft vocabulary | the MTP head scores only a frequency-chosen subset of token ids; the target still checks every draft | a token-id list | +0.7 % on box B, identical output (point 6) |
| hot-expert placement | keeps the most-routed experts, from a hot-expert map, on the GPU | placement rules | used in the box B layouts; not measured on its own |
| dense rows per node | lets a placement rule name any weight (by tensor name or regex), not only routed experts, and split a dense matrix into one row slice per node | `LLAMA_EXPERT_PLACEMENT` rules with `*` or `/G` | V4.1 on box D +9 % with the draft, +17 % without; Qwen3-30B-A3B +11 % (point 16) |
| no host weights | keeps model weights in plain host memory, never pinned, so the pin budget goes to the buffers the GPU reads directly; logs a warning when such a buffer falls back to pageable memory | `LLAMA_NO_HOST_WEIGHTS=1` or `--no-host` | the box D arm that crashed at warm-up (Xid 31) ran with no fault (point 11) |
| in-place ops follow their target | the scheduler runs an in-place op on the backend of the tensor it writes into. Fixes V4.1 attention on cards without a kernel for its flash attention (garbage text on the GTX 1080 Ti) | on; `GGML_SCHED_INPLACE_FOLLOW=0` turns it off | correct text with V4.1 attention on the 1080 Ti; moving attention and the KV cache onto the card then took decode without the draft from 5.60 to 6.97 tok/s, and prefill from 37.2 to 48.2 (point 17) |
| CUDA and Vulkan in one process | a CUDA build also opens Vulkan devices and puts hot-expert banks on one of them; MXFP4 expert banks on Vulkan | `LLAMA_EXTRA_VULKAN=1`, `GGML_VK_VISIBLE_DEVICES`, `LLAMA_HOT_EXPERTS_DEVICE=Vulkan0`, `GGML_VK_OFFLOAD_MIN_BATCH` | V4.1: correct text with 14 GiB of experts on the MI25, but slower: 5.61 against 6.97 tok/s without the draft, after the fixes (point 18). Ornith-1.5-35B-A3B with every expert on the two GPUs: 59.13 tok/s against 47.34 with the experts on the CPUs, both with MTP (point 23) |
| a bank device per layer, whole-layer banks | a third column on a heat-file line names the backend for that layer's bank; a layer whose experts are all banked runs no CPU expert pass | `LLAMA_HOT_EXPERTS=<file>` lines `blk.N <ids> <backend>` | the Ornith-1.5-35B-A3B layout of point 23, split between the MI25 and the 1080 Ti |

On box B, the newer changes together lift V4.1 by 11-12 % (point 21); how much each one gives there is not yet
measured.

Build option, upstream's and not ours: `-DGGML_EXPERT_CHUNKING=OFF` (point 13).

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [arc](https://github.com/akapug/arc) + [fab](https://github.com/akapug/fab) for [@akapug](https://github.com/akapug)
