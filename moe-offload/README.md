# Big MoE models on one GPU plus system memory

Three mixture-of-experts models that do not fit in one card's VRAM, each served by keeping part of the expert
weights on the GPU and computing the rest on the CPU from system memory:

- GLM-5.3-Flash, 321B total parameters;
- DeepSeek V4.1 Flash, 748B;
- Qwen3.8 Flash-Next.

The question on every box: how close does one mid-range GPU plus the memory already in the machine get to
25 tok/s at one stream, without giving up output quality? 25 tok/s is our bar for a coding model we would
hand an agent's work to.

## The boxes

| box | CPU | system memory | GPU |
|---|---|---|---|
| A: AM5 desktop | Ryzen 9 9950X3D, 16 cores | 192 GB DDR5 as 4 DIMMs, running at 5200 MT/s | RTX 3090 Ti 24 GB |
| B: 2019 workstation (HP Z8 G4) | 2x Xeon Gold 5215, 10 cores each | 192 GB DDR4-2666 (12 x 16 GB), plus 256 GB Optane PMem 100 (2 x 128 GB, one App Direct region) | Quadro RTX 6000 24 GB (Turing) |
| C: TRX40 workstation | Threadripper 3960X, 24 cores | 128 GB DDR4 as 4 x 32 GB | Intel Arc Pro B70 32 GB (one of two in the box) |

## Best measured result per model

All decode figures are one request at a time, greedy, thinking off, on coding prompts. How each row aggregates its
requests is named in the row.

| model | box | engine | weights | decode tok/s | quality check |
|---|---|---|---|---|---|
| GLM-5.3-Flash | A | the `glm53-flash-offload` image (ghcr.io/sybil-solutions, exllamav3-based): a VRAM expert cache of 1,600 slots, the other experts computed on 14 CPU cores from a 107 GiB RAM tier | EXL3 3.05 bpw | **23.08**, upper median of 6 requests of 300 tokens (22.20 without layer-ahead warm-up) | served vs the same weights computed exactly on the GPU: KL 0.0044, top-1 0.985 (928 positions) |
| GLM-5.3-Flash | B | ik_llama.cpp with our patches: hot experts on the GPU, each CPU expert's rows split between the two sockets, MTP draft 3 tokens, p_min 0.75 | unsloth UD-Q4_K_XL | **15.73**, mean of 4 prompts (median 15.81) | KL 0.030, top-1 0.960 against the EXL3 3.05 bpw store above (925 positions). See [the reference problem](#the-quality-reference-problem) |
| DeepSeek V4.1 Flash | B | ik_llama.cpp with our patches: as above, plus the DSpark draft model at n_max 4 (acceptance 0.95) | MXFP4 for hot experts, q2_K for cold ones | **10.6**, mean of two runs (10.58, 10.62), each over 3 prompts of 300 tokens (8.75 without the draft) | KL 0.0169, top-1 0.975 against the all-MXFP4 store (1,376 positions) |
| Qwen3.8 Flash-Next | C | XeStrata (the SYCL port of Strata), one B70, about 10,900 experts in VRAM, the rest on the CPU from RAM, MTP 4, int8 KV, 262,144-token window | ISTA-DASLab GSQ-RCO IQ3_S | **87.1**, median of 3 runs (88.5, 87.1, 86.7) of 512 tokens on one code prompt, XeStrata xe0.1.45 with `--prefill auto` | not measured yet |

Box A and box B each miss 25 tok/s for GLM; box C clears it with a smaller model.

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
- On box B, where the draft costs no expert slots, MTP helps. In one binary, GLM ran at 9.41 tok/s with no draft
  and 12.71 with MTP at 3 tokens (+35 %).

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

**7. More Optane would add capacity, not speed, on box B.**
- Box B has 2 PMem sticks. We modelled filling socket 0 to six DDR4 + six PMem DIMMs.
- Our planner predicts V4.1 at 12.1-12.5 tok/s (+15-18 %) and GLM at about 16.2 (+3 %).
- After that, the bind is per-token fixed overhead plus CPU expert compute. Even infinitely fast PMem gives V4.1 only +23 %.
- What the sticks would buy is room: V4.1 at full Q8_0, or a 1T-class model at 1-2 bits.
- These are predictions, not measurements.

**8. Flash-Next on the B70 keeps its speed with a modest RAM budget.**
- With the container limited to 42 GiB and the model's pages dropped before start, decode stayed at 70-80 tok/s from short prompts
  out to 64k depth.
- At 12 GiB it fell to 20-22 tok/s, and at 8 GiB to 17-21.
- It held 71-72 tok/s out to 128k depth through a 79-minute soak with the box's normal build load running beside it.

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
- Recipe Maker, the planner used for the predictions in point 7 and for choosing these layouts. It fits a model
  and a recipe to a box's GPU, RAM and PMem, and predicts tok/s and quality from measured runs. It will follow as
  its own entry.
