# Homelab Atlas

*Old hardware. New tricks. Receipts attached.*

Big local models on hardware you can actually get: a 2017 server off the used market, a 2019 workstation full of
Optane, a gaming desktop, and a pair of Intel cards most people have not heard of. No rack of H100s was harmed in
the making of this repo.

The bet: as models lean on sparsity, speculative decoding and low-bit weights, memory bandwidth and software decide
more than raw GPU compute, and most modest hardware runs well below what it can do, because tuning it is fiddly work.
We did the fiddly work and wrote down exactly what happened. Every number names its conditions, every win links the patch behind
it or says it is unpublished, and the losers stay in the notebook, because the next person with the same box needs those most. Some lessons
were measured in tokens per second, and some in pages of confident garbage text.

**Try Recipe Maker in your browser:** [akapug.github.io/homelab-atlas](https://akapug.github.io/homelab-atlas/).
Pick one of our four boxes and see which layouts of a big MoE model fit it, how fast each should run, and how
the predictions compare with our measured runs. Nothing to install.

## Where it stands (2026-11-27)

The target since 11-10: many coding agents at once from the boxes that run around the clock. Newest first:

- **32 real coding agents on two B70s: 645.1 tok/s for two hours.** Inkling 2 Mini on vLLM for XPU, 16 pi agents a
  card on our coding-task set, no drift between quarters (630 to 647), 943 of 1,073 attempts accepted; the B70 stack
  agrees with a CUDA reference at that reference's own noise floor (98.8 % top-1). Adding the RTX 3090 Ti as a third
  card: 874.7 tok/s for 40 agents in a 20-minute smoke. [Inkling 2 Mini pool](b70/inkling-2-mini.md)
- **Close the thinking instead of lengthening it.** On LiveCodeBench (104 problems), ending a reply's thinking at
  10,000 tokens and letting it answer took effort high from 53.8 % to 64.4 % (p 0.031 against effort low's 54.8 %), at
  1.27x low's tokens; a 31.5k cap gave 58.7 % at 3.12x. Effort low with the same close: 63.5 %, the best hard score, at
  1.15x. [LiveCodeBench](b70/inkling-2-mini.md#quality-is-the-b70-stack-the-model)
- **KV room, not compute, is what each replica ran out of.** Re-quantizing Inkling's MXFP8 tail to NVFP4 freed 1.8 GiB
  on the 3090 Ti, enough for its draft head (155.6 -> 231.1 tok/s for 16 agents, 98.3 % top-1 agreement); the RTX 6000
  is held to 8 slots by a llama.cpp buffer sized by the whole pool (66.0 tok/s for 8 agents); the MI25 is too hot
  (11-15 tok/s at 8 streams against the 80 needed). [Replicas](b70/inkling-2-mini.md#replicas-on-other-cards)
- **A scripted load can hide a collapse.** QFN-Coder reached 310.2 tok/s over two hours for 64 agents on a scripted
  assistant workload; on real coding tasks the same pool served 96.4 tok/s and accepted none of 86 attempts, because 64
  fresh agents queued ~644k prompt tokens behind its serial prompt reads. Its idle time was named first: 51 of 52 long
  stops were one serial read. [QFN-Coder pool](b70/many-agents.md#qfn-coder-pool-2026-11-16-to-11-26)
- **One fast agent: Ornith-1.5-35B-A3B with its DFlash drafter, 416.6 tok/s on a 3090 Ti** (2.17x without it), and still
  1.42x at 63k tokens of context once the target's attention runs on FlashInfer; 296.6 on one B70.
  [Ornith + DFlash](3090-ti/#ornith-15-35b-a3b-with-dflash-one-agent-at-417-toks)
- **Peers' engines, measured on our cards.** maya-inkling-mini decodes Inkling 2 Mini at 140.3 tok/s on our 3090 Ti
  (their 3090: 128.7), 2.2-2.6x our llama.cpp from short to 128k context.
  [Inkling 2 Mini on maya-ik2](3090-ti/#inkling-2-mini-one-fast-agent-and-a-pool-replica)
- **GLM-5.3-Flash: one agent at ~30 tok/s on the 3090 Ti, parked on the B70s.** On Maya with its 1.73-bit quant one
  pi agent gets ~30 tok/s, and more seats do not pay (4 seats: 35.1 in total). On a B70 with a CPU expert lane the best
  was 7.07 against a 14 tok/s bar, bound by host work outside the expert kernels. A block drafter (DFly2) forecasts
  0.75-0.95x on the 3090 Ti, so we did not port it. [GLM on Maya](3090-ti/#glm-53-flash-on-maya-one-agent-at-about-30-toks),
  [GLM on a B70](b70/many-agents.md#glm-53-flash-on-a-b70-with-a-cpu-expert-lane-parked)
- **Losers, kept:** the Inkblot-2-mini edit is terser, not smarter (faster agent work, -10.6 points on
  LiveCodeBench); LoopCD-Logits gave nothing measurable; DFlash2 in vLLM accepts no drafted token; a fine-tuned Ornith
  drafter gained 0.035 tokens a step against a 0.30 bar; llama.cpp before 690b87d88 decodes Inkling's NVFP4 GGUFs to
  noise. [What did not work](b70/inkling-2-mini.md#what-did-not-work)

On 11-15: on Intel GPUs, count kernel launches before tuning kernels (SYCL graphs took two-B70 decode 16.9 -> 39.0
tok/s, [host-bound decode](b70/host-submission-bound-decode.md)); 2 bits is too few for GLM-5.3-Flash as a coding
agent (3 of 5 planted bugs at UD-IQ2_XXS, a direction from 8 private items, not a rate).

Older lessons that still shape how to read everything below:

- **Decode speed is not agent speed.** Real coding agents read 4 to 5 new prompt tokens for every token they write.
  On the two B70s, the batched decode reaches 377.9 tok/s over 32 agents on a synthetic load, but recorded agent
  sessions replayed against the same box (a later build) get 80 to 86 tok/s in total at 16 agents (about 5 each), and 32.6 tok/s
  for one agent alone, because prompt reading sets the pace. [Real coding-agent traffic](b70/many-agents.md#real-coding-agent-traffic-the-number-to-plan-with)
- **The table below is one-request decode,** with prompt reading beside it where we measured it. Where we have a
  real-agent figure, it is in its own column. A recipe with slow prompt reading (tens of tok/s) suits chat and batch
  work, not an interactive coding agent, whatever its decode.
- **No silent fallback.** A 35B MoE moved to the Optane box took 96 s for a chat request because 3.7 GB of its CPU
  experts had landed on Optane while DRAM had room; bound to DRAM, plus a smaller window, the same request took
  0.59 s. Our recipes declare where each part lives, and the run checks it.
  [RTX 6000](rtx-6000/#no-silent-fallback-experts-on-optane-by-accident)

## Best result per model

The fastest one-request decode per model, with its prompt reading, its quality evidence (or "none") and, where we
measured it, what agents doing real work get. The card pages hold every condition.

| model | best one-request decode | prompt reading | where | real agent work | quality evidence | date |
|---|---|---|---|---|---|---|
| Inkling 2 Mini (looped MoE, NVFP4) | **140.3** | 3,020 at 4k | [RTX 3090 Ti](3090-ti/#inkling-2-mini-one-fast-agent-and-a-pool-replica), alesha-pro's maya-ik2 engine, every expert in VRAM; 115.8 at 64k | **645.1** tok/s in total for **32 real coding agents** over two hours (28.2 each at the median) on [two B70s](b70/inkling-2-mini.md), vLLM XPU with its draft head; 874.7 for 40 on three cards (20 min) | B70 stack 98.8 % top-1 against a CUDA reference (its floor 98.8 %); LiveCodeBench 104: 64.4 % with thinking closed at 10k | 2026-11-27 |
| Ornith-1.5-35B-A3B with its DFlash drafter | **416.6** on short agent turns; 215.5 at 63k | not measured in this run | [RTX 3090 Ti](3090-ti/#ornith-15-35b-a3b-with-dflash-one-agent-at-417-toks), vLLM nightly, int4 + DFlash K=7, FlashInfer target attention; 296.6 on [one B70](3090-ti/#ornith-15-35b-a3b-with-dflash-one-agent-at-417-toks) | 226.0 tok/s a turn over a replayed 29-49k pi session (1.43x without the drafter) | planted-defect screen passes in this config | 2026-11-26 |
| DeepSeek V4.1 Flash (748B MoE) | **27.39** | 240.5 | [RTX 3090 Ti](3090-ti/#deepseek-v41-flash-27-toks-on-a-gaming-card) + 192 GB DDR5, 2-bit experts; our sm_86 Strata build, not published | not measured | 13 of 15 agent tasks; no KL check | 2026-10-24 |
| the same, release-precision hot experts | 12.69 (4,096-token window); 7.18 at 189k tokens | 69.0; 57.7 at 110-190k | [Quadro RTX 6000](rtx-6000/#deepseek-v41-flash-the-best-run) + DDR4 + Optane, public fork | not measured | KL 0.0169, top-1 0.975 (earlier build of the same store); 16 of 16 facts at 189k | 2026-10-22 |
| the same, no GPU | 7.02 | 36.2 | [2x EPYC 7601](epyc-cpu/), public fork | not measured | release-precision weights | 2026-10-20 |
| GLM-5.3-Flash (321B MoE) | **26.69** | not measured in this run | [RTX 3090 Ti](3090-ti/#glm-53-flash-a-route-trim-clears-25-toks) + DDR5, route trim and CPU-lane changes not published (23.08 without the trim) | about 20-22 tok/s for one agent before tool time (our estimate from the coding-seat setup's 25.9 decode and 761 prompt reading; no replay) | KL 0.0154, top-1 0.975: passes | 2026-10-20 |
| Qwen3.8 Flash-Next (125B MoE, 6B active) | **87.1** | not measured in this run | [one Arc Pro B70](b70/), public XeStrata | **32.6** tok/s for one agent with its tool time (about 42 while the server works for it); **80-86** (mean 83.4, three runs) in total over 16 agents, about 5 each, on [two B70s](b70/many-agents.md#real-coding-agent-traffic-the-number-to-plan-with) with our experimental batched build | 11 of 12 agent tasks (one-card build); none on the batched build | 2026-10-26 |
| Qwen3.8-27B (dense) | **103-105** | 1,407-1,410 at 16k | [RTX 3090 Ti](3090-ti/#qwen38-27b-on-vllm-cuda-about-104-toks), 57,344-token window at most | not measured | 41 of 41 correctness eval | 2026-10-07 |
| Ternary Bonsai 2 27B (dense, 6.5 GiB) | **71.5-71.6** with MTP 2; 52.8-52.9 without | 907 cold at 49k tokens; 574-1,231 at 32-512 tokens (hybrid build) | [one Arc Pro B70](b70/bonsai-2-27b.md), PrismML's llama.cpp fork + our published SYCL patches | not measured | test-backend-ops; perplexity unchanged by the hybrid; no agent eval on these builds | 2026-09-27 |
| Flint (39.8B MoE, about 2.5B active) | **148.9** on an RTX 3090 Ti (UD-Q3_K_XL); 92.3 on one B70 (Q4_K_M) | 4,052 (3090 Ti); 1,922 on the B70, 2,087 with our attention-sinks change | [RTX 3090 Ti and Arc Pro B70](b70/llama.cpp-sycl-fa-sinks/#flint-on-two-b70s-and-on-an-rtx-3090-ti), llama.cpp b12004; the B70 prompt figure needs [our published SYCL change](b70/llama.cpp-sycl-fa-sinks/) | replayed coding-agent sessions, one agent: 15.9 turns a minute at 87.3 tok/s decode (3090 Ti), 10.8 at 66.1 (B70 with the change) | a few short prompts correct; coding quality not measured | 2026-10-30 |
| Ornith-1.5-35B-A3B | **198** with MTP 3; we serve with MTP off (about 118 on the same architecture) after MTP garbled 3 of 106 replies under mixed load | not measured | [one Arc Pro B70](b70/ornith-35b-a3b.md#speed) | not measured | 17 of 24 agent tasks | 2026-10-04 |
| the same, on retired cards | 59-60 | 331 | [GTX 1080 Ti + MI25](mi25/); 59.6 on the [RTX 6000](rtx-6000/#no-silent-fallback-experts-on-optane-by-accident) | not measured | code-review screen 1 of 2 | 2026-10-21 |
| Ornith-1.5-122B (122B MoE, 10B active) | **37.2** | 261 at 4k | [two Arc Pro B70](b70/host-submission-bound-decode.md#a-second-model-ornith-15-122b-with-experts-on-the-cpu), layer split + our SYCL graphs patch, no draft; 14 of 49 layers' experts in the host's 4-channel DDR4, which sets most of the pace, so this is not what two B70s can do with the whole model in VRAM | not measured | none on this build | 2026-11-15 |
| the same, one card | **31.6** | 165.5 | [Quadro RTX 6000](rtx-6000/) + DDR4, 40 of 49 layers' experts in DRAM, with its MTP draft (27.2 without) | 0 of 3 agent tasks in a limited setup | both planted defects found | 2026-10-20 |

"Not measured" means exactly that; it is not a zero. Decode is tokens per second for one request once it is
writing. Real agents also wait for prompt reading and tool time, so they get less: on Flash-Next, one agent got 32.6
tok/s against its build's 68.8 decode.

## The hardware

Four boxes, six GPUs from three vendors and five architectures, 1.5 TB of DRAM between them. Each card has its own
page with its best results first, the commands to reproduce them, and what did not work. Every result is one request
at a time unless it says otherwise.

| box | CPU and memory | card | best measured result |
|---|---|---|---|
| **AM5 desktop** | Ryzen 9 9950X3D (16 cores), 192 GB DDR5 | [**RTX 3090 Ti** 24 GB](3090-ti/) | Qwen3.8-27B at **103-105 tok/s** (95-107 over 10 runs); DeepSeek V4.1 Flash (748B MoE) at **27.39 tok/s** with 2-bit experts; GLM-5.3-Flash (321B MoE) at **26.69 tok/s** |
| **2019 workstation** (HP Z8 G4) | 2x Xeon Gold 5215, 192 GB DDR4 + 256 GB Optane PMem | [**Quadro RTX 6000** 24 GB](rtx-6000/) | DeepSeek V4.1 Flash (748B MoE) at **12.69 tok/s** in a 4,096-token window; **16 of 16** facts recalled from a 189k-token prompt at 7.18 tok/s, prompt reading 57.7 tok/s (the same store, the build before the tier split) |
| **Threadripper workstation** | Threadripper 3960X (24 cores), 128 GB DDR4 | [**2x Arc Pro B70** 32 GB](b70/) | Inkling 2 Mini serving **32 real coding agents at 645.1 tok/s** in total for two hours ([pool](b70/inkling-2-mini.md)); Qwen3.8 Flash-Next at **87.1 tok/s** decode for one request, **11 of 12** agent tasks (one card); on real agent traffic, **32.6 tok/s** for one agent and **80-86** in total over 16 (about 5 each), both cards (377.9 decode over 32 agents on a synthetic load); Qwen3.8-27B at **74.6 tok/s**, **351** summed over 8 streams in a compiled mode Intel does not support, on 1.5k-token prompts (the 27B was measured with the cards in the AM5 box, before they moved here) |
| **2017 two-socket server** (Supermicro H11DSi) | 2x EPYC 7601, 1 TB DDR4 on 16 channels | [**GTX 1080 Ti** 11 GB](1080-ti/) and [**Radeon Instinct MI25** 16 GB](mi25/) | Ornith-1.5-35B-A3B at **59-60 tok/s** with every expert on the two retired cards; DeepSeek V4.1 Flash at **12.09 tok/s** (11.74 without a draft) |
| the same server | [**its CPUs alone**](epyc-cpu/) | none | DeepSeek V4.1 Flash at **7.02 tok/s**, no GPU at all |

The fine print, card by card (the card pages carry the rest):
- **RTX 3090 Ti.** Qwen3.8-27B: vLLM CUDA, int4, MTP 3 with a reduced draft vocabulary; the same checkpoint scored
  41 of 41 on our correctness eval on the B70. GLM-5.3-Flash: EXL3 3.05 bpw with a route trim that passes our quality
  bar but is not published yet; 22.77 without it in the same session, 23.08 in the base run (the public image plus our CPU-lane changes, also unpublished; 14 cores, 3 coding prompts x 300 tokens,
  greedy, thinking off, 32,768 window).
- **RTX 6000.** 11.88 without, 12.69 with two newer changes of ours plus upstream ik_llama.cpp #2712, all public (our
  fork's branch `tier-split`); its quality check comes from the earlier build of the same store and layout.
- **B70.** Flash-Next: IQ3_S on XeStrata; the speed run (xe0.1.45, 262,144-token window, 2026-10-19) and the agent
  tasks (xe0.1.40.2.2, 229,376-token window, 2026-10-18) used two builds of the same setup. Qwen3.8-27B: 74.6 in eager
  mode with MTP 3 and the reduced draft vocabulary, which leaves the output unchanged; the 351 is compiled mode without
  it. Both were measured while the B70s sat in the Ryzen 9950X3D box, before they moved on 2026-10-07. The
  many-agent runs: IQ3_XXS across both cards on our batched XeStrata branches (published as an
  [experimental fork](b70/many-agents.md#engine-code-experimental)), k8v4 KV. The 377.9 is a synthetic load (each agent
  reads one ~14k prompt once, then writes) over one 60 s window; the 80-86 and 32.6 replay recorded coding-agent
  sessions with their own prompts, outputs and tool time. No agent-task or KL run on the batched builds.
- **1080 Ti and MI25.** Ornith: Q4_K_M with its own MTP draft head; on our code-review screens it found 1 of 2 planted
  defects and raised one false alarm. V4.1: the DSpark draft model, experts at release precision (the quality
  reference itself); the 12.09 adds our decode-attention kernel for the 1080 Ti, checked against the CPU attention on
  16 shapes, with all 16 DIMMs. The Ornith run had 15 of 16 DIMMs working.
- **EPYC alone.** V4.1 with the same release-precision weights and draft model, on 15 of 16 DIMMs (the 12.09 had
  all 16).

Names you will meet on the card pages and in Recipe Maker: the desktop is **snoopy**, the workstation **droopy**, the
Threadripper **odie** and the server **lassie**; the [MoE offload entry](moe-offload/) calls them boxes A, B, C and D.
The 3090 Ti box is also somebody's desktop, so its servers step aside when the desktop wants the card. The two B70s
lived in that box until 2026-10-07; their older pages say so where it matters.

## Pick a path

| you want | start here |
|---|---|
| The fastest coding model one 24 GB NVIDIA card can serve | [3090 Ti: Qwen3.8-27B on vLLM CUDA](3090-ti/#qwen38-27b-on-vllm-cuda-about-104-toks), one Dockerfile; a 57,344-token window at most, so review passes and contained tasks rather than long agent sessions |
| A model that finishes real repository work, locally | [the agent eval](b70/agent-eval/), then Flash-Next on [a 3090 Ti](3090-ti/#qwen38-flash-next-agent-work-from-ram) or [a B70](moe-offload/#what-we-learned-measured): 11 of 12 everyday tasks, but 4 of 11 harder ones at the highest effort and 0 of 3 multi-file changes without our thinking cap |
| A frontier-size MoE (DeepSeek V4.1 Flash, GLM-5.3-Flash) on a box with lots of RAM | [MoE offload](moe-offload/), then [the RTX 6000 + Optane page](rtx-6000/#reproduce-the-best-v41-run) for the full command; at 36-70 tok/s of prompt reading these suit chat and batch work, not an interactive coding agent |
| To plan a layout before downloading 400 GB of weights | [Recipe Maker](recipe-maker/), or [in the browser](https://akapug.github.io/homelab-atlas/) |
| To serve many coding agents from one box | [the Inkling 2 Mini pool on two B70s](b70/inkling-2-mini.md): 32 real coding agents at 645.1 tok/s in total over two hours, 28 tok/s each at the median, first tokens about 1 s; size the KV pool before adding agents, and test on real tasks, not a scripted load ([why](b70/many-agents.md#qfn-coder-pool-2026-11-16-to-11-26)) |
| A fast link between boxes for little money | [a 56 Gb/s triangle from three $10 cards](network/), no switch |
| To make an Intel Arc Pro B70 go fast | [the B70 entry](b70/) and its [patches](PATCHES.md#2-intel-arc-pro-b70-with-vllm) |
| To put a cheap old card to work | [MI25](mi25/) ($50, 16 GB HBM2, Vulkan), [GTX 1080 Ti](1080-ti/) |
| No GPU at all | [EPYC CPUs alone](epyc-cpu/): where the bytes sit matters more than how many cores compute; at 36 tok/s of prompt reading, for batch or offline use |
| A specific fix for your hardware | [Find your patch](#find-your-patch) |

## Highlights

The ones we are proudest of, each with its conditions and quality check one click away.

- **32 coding agents from two Intel cards, for two hours.** Inkling 2 Mini on two Arc Pro B70s served 32 real pi
  agents working coding tasks at 645.1 tok/s in total, 28 tok/s each at the median, first tokens about a second, no
  drift over two hours and no GPU fault; the Intel stack matches a CUDA reference at that reference's own noise floor.
  With a 3090 Ti as a third card, 40 agents at 874.7. [Inkling 2 Mini pool](b70/inkling-2-mini.md)
- **One coding agent at 417 tok/s on a gaming card.** Ornith-1.5-35B-A3B with its DFlash drafter on an RTX 3090 Ti,
  and still 1.42x faster than without the drafter at 63k tokens of context.
  [3090 Ti](3090-ti/#ornith-15-35b-a3b-with-dflash-one-agent-at-417-toks)
- **A 748B model on a 2019 workstation, at long context.** DeepSeek V4.1 Flash decodes 12.69 tok/s on one Quadro
  RTX 6000 plus DDR4 and Optane, from public code, in a 4,096-token window; at 189k tokens it decodes 7.18 and reads
  prompts at 57.7 tok/s, so a long prompt takes many minutes. [One folder](rtx-6000/v41-recipe/) holds every input the run
  loaded but the prompts and the importance matrix (our quality harness and the lab launcher's checks are not in it either), with scripts that build the store from the public 508 GB weights (233 GB of new shards; the other three are the
  source's own, linked). At 189k tokens, on the build before the tier split, it recalled all 16 planted facts. [RTX 6000](rtx-6000/)
- **GLM-5.3-Flash over the 25 tok/s line on a gaming card.** 26.69 tok/s on an RTX 3090 Ti plus 192 GB of DDR5
  (+17 % on the untrimmed arm run beside it, 22.77),
  by computing only the top 4 of the expert picks that miss VRAM on 16 of its layers, with its output still within
  our quality bar, on 300-token coding prompts. The trim is not published yet, nor are the CPU-lane changes under the
  23.08 it starts from. Served to one coding agent, the same box's GLM setup works out to about 20 tok/s of useful
  output (our estimate from its measured decode and prompt reading).
  [3090 Ti](3090-ti/#glm-53-flash-a-route-trim-clears-25-toks)
- **Agent work from a model mostly in RAM.** Qwen3.8 Flash-Next passed 23 of 24 agent tasks on the 3090 Ti box
  (21 inside the time limit),
  and 11 of 12 on one B70, with 10,900 of its 24,576 experts in VRAM and the rest computed on the CPU (the next build of
  the same engine decodes it at 87.1 tok/s). [3090 Ti](3090-ti/),
  [B70](b70/)
- **Two retired cards from rival vendors in one process.** A $50 MI25 and a GTX 1080 Ti hold every routed expert of
  Ornith-1.5-35B-A3B in a 2017 server: 59-60 tok/s with the model's own draft head, against 47 with the same draft
  and the experts on the CPUs. [MI25](mi25/)
- **An 11 GB Pascal card earns its slot.** Moving DeepSeek V4.1's attention onto a GTX 1080 Ti first gave garbage
  text; one scheduler fix later it decoded 35 % faster than the CPUs alone
  without a draft (6.97 against 5.16; with the draft, 7.13 against 7.02). Then a profile showed the card spending 92
  ms a token in F16 math it runs at 1/64 speed, so we wrote it a decode-attention kernel. With all 16 DIMMs back and `-ub 512` (7.96 without the kernel, against
  6.97 before), it went 7.96 to 11.74 tok/s without a draft, 12.09 with one, same binary, same prompts, in an 8,192-token
  window with prompt reading at 52 tok/s. A card from 2017, one kernel, +47 % decode.
  [1080 Ti](1080-ti/#one-kernel-47-)
- **What many agents really get from one box.** Two Arc Pro B70s decode Qwen3.8 Flash-Next at 377.9 tok/s over 32
  agents on a synthetic load, but recorded coding-agent sessions replayed against the same box (a later build) get 80 to 86 tok/s
  in total at 16 agents, and 32.6 for one. The gap is prompt reading: real agents read 4 to 5 new prompt tokens per
  token they write. The page has both, the switches that won and lost on each, and the engine code as an
  experimental fork. [Many agents](b70/many-agents.md)
- **56 Gb/s between three boxes for about $136.** Used ConnectX-3 cards at $10.19 each, cross-flashed to their
  faster sibling's firmware, and copper cables, no switch: 52.3 to 52.7 Gb/s of RDMA per link. [Network](network/)
- **The B70, tuned.** Qwen3.8-27B at 74.6 tok/s at one stream once the drafter stopped reading a 2.5 GB output
  layer for every guess (eager mode, 59.5 without the trick, same output; the B70 page's 54.4 eager is another workload), and 351 summed at eight streams (full drafter vocabulary, 1.5k-token prompts) in vLLM's
  compiled mode with XPU graphs, which Intel does not support yet (our servers run eager mode). Five vLLM changes and 18 llama.cpp SYCL patches
  got it there. [B70](b70/)

## Find your patch

[PATCHES.md](PATCHES.md) lists every patch and tool here by the hardware it needs, then the software,
with the symptom it fixes and how to apply it, check it and undo it. Start from your hardware:

| your hardware | what applies | where |
|---|---|---|
| Intel Arc Pro B70 (our card), with vLLM | two Intel-only fixes, four fixes that are not tied to a GPU, two settings for Intel's `xe` driver | [B70 with vLLM](PATCHES.md#2-intel-arc-pro-b70-with-vllm), [any GPU](PATCHES.md#1-any-gpu-serving-software-and-chat-template-fixes) |
| Intel Arc Pro B70, with llama.cpp | 18 patches and a build script; 22 more for ternary Bonsai models on PrismML's fork; 2 on upstream master for attention sinks and broadcast masks in MKL flash attention | [B70 with llama.cpp](PATCHES.md#3-intel-arc-pro-b70-with-llamacpp), [ternary Bonsai](PATCHES.md#ternary-bonsai-on-prismmls-fork), [attention sinks](PATCHES.md#attention-sinks-on-upstream-llamacpp) |
| Other Intel Arc (B580, Pro B60, A-series) | untested, with what may carry over and why | [other Intel Arc](PATCHES.md#other-intel-arc-cards) |
| NVIDIA with vLLM: our RTX 3090 Ti (Ampere) | two of the any-GPU patches on vLLM's official 0.31.0 image, in one Dockerfile | [3090 Ti](PATCHES.md#nvidia-rtx-3090-ti-ampere-24-gb) |
| NVIDIA with big MoE models: our Quadro RTX 6000 (Turing) and GTX 1080 Ti (Pascal) | our ik_llama.cpp fork's engine changes, a Turing fix sent to another project | [RTX 6000](PATCHES.md#nvidia-quadro-rtx-6000-turing-24-gb), [1080 Ti](PATCHES.md#nvidia-gtx-1080-ti-pascal-11-gb) |
| AMD or Apple GPUs | the fixes not tied to a GPU (untested there), the tools | [other GPUs](PATCHES.md#amd-apple-and-other-nvidia-gpus) |
| PyTorch (training) on a B70, a Turing or Pascal card, or an MI25 | which torch build works on each, install scripts, an op probe | [PyTorch per card](PATCHES.md#7-pytorch-on-older-and-other-cards-which-torch-build-works) |
| A Ryzen with two CCDs that serves a model and also runs builds | a CPU guard | [host CPU](PATCHES.md#4-the-host-cpu-a-ryzen-with-two-ccds) |
| A used AMD Radeon Instinct MI25 (Vega 10, 16 GB HBM2) | a cold-boot fix for `amdgpu`, llama.cpp Vulkan numbers, heat; a MoE model's experts split between it and a CUDA card | [MI25](mi25/) |
| A MoE model too big for your VRAM, on any box | a planner that places it across VRAM, DRAM, PMem and disk before you run it | [Recipe Maker](PATCHES.md#recipe-maker-plan-a-moe-layout-before-you-run-it) |
| Any, or none: you want to test a local model on real work | the agent eval, benchmarks and probes | [tools](PATCHES.md#6-tools-that-need-no-particular-gpu) |

Before you start:
- A Linux machine where the GPU driver works. For the B70 that is the kernel's `xe` driver: `lspci -k`
  shows `Kernel driver in use: xe` for each card.
- For vLLM on the B70: Docker and Intel's `intel/llm-scaler-vllm:0.26.0-b2` image. The vLLM patches are
  written for exactly this version. For llama.cpp on the B70: git and Intel's oneAPI Base Toolkit.
- A model from Hugging Face, for example `Frozenlock/Qwen3.8-27B-int4-AutoRound` (18 GB) for vLLM, or
  `Qwen3.8-27B-UD-Q4_K_XL.gguf` from `unsloth/Qwen3.8-27B-GGUF` for llama.cpp.
- To drive it from Claude Code: a proxy that translates Anthropic's API to OpenAI's, set up as in
  [Claude Code on it](b70/qwen36-35b-a3b.md#claude-code-on-it).

## Tools we use

[TOOLS.md](TOOLS.md) lists the outside software behind these results (serving engines, model formats, agent
harnesses and memory tools), each with what we ran it on, our verdict and the evidence.

## Entries

| folder | what is in it |
|---|---|
| [`3090-ti/`](3090-ti/) | RTX 3090 Ti: Ornith with DFlash for one fast agent, Inkling 2 Mini on maya-ik2 and as a pool replica, GLM-5.3-Flash on Maya, Qwen3.8-27B on vLLM CUDA, GLM-5.3-Flash with a route trim, Flash-Next agent runs, four models reviewing code |
| [`rtx-6000/`](rtx-6000/) | Quadro RTX 6000 + Optane: DeepSeek V4.1 Flash best run and its full command, 189k-token recall, GLM, Ornith-1.5-122B |
| [`b70/`](b70/) | Arc Pro B70: [the Inkling 2 Mini coding-agent pool](b70/inkling-2-mini.md), [many agents on two cards](b70/many-agents.md), vLLM and llama.cpp patches, Qwen3.8-27B, [Qwen3.6-35B-A3B](b70/qwen36-35b-a3b.md), [Ornith-1.5-35B-A3B](b70/ornith-35b-a3b.md) serving a household assistant and a coding agent from one card, [Ternary Bonsai 2 27B](b70/bonsai-2-27b.md) on PrismML's llama.cpp fork, [Flint and attention sinks on llama.cpp SYCL](b70/llama.cpp-sycl-fa-sinks/), [tensor against layer split on two cards and three outside claims](b70/split-modes-and-backends.md), [the agent eval](b70/agent-eval/), [benchmarks](b70/bench/) |
| [`tail-trainer/`](tail-trainer/) | train a LoRA on the last blocks of a MoE GGUF in torch on any card, checked against llama.cpp and loaded back with `llama-server --lora`; smoke on an RTX 6000 and an MI25, two cards at once |
| [`tail-portable/`](tail-portable/) | one PyTorch trainer on a B70, an RTX 6000, an MI25 and a 1080 Ti: which torch build works on each, an op probe, a B70 wrong-result repro (driver 26.05 with torch 2.13-2.15) |
| [`behaviour-levers/`](behaviour-levers/) | changing a local coding model's habits: a trained LoRA that changes nothing (NULL), and the habit a harder task set exposes |
| [`network/`](network/) | three used ConnectX-3 cards as a 56 Gb/s triangle between three boxes: flash, setup, every link measured |
| [`1080-ti/`](1080-ti/) | GTX 1080 Ti: V4.1 attention on a card with no kernel for it, and the fix |
| [`mi25/`](mi25/) | Radeon Instinct MI25: a $50 card under Vulkan, a cold-boot fix, experts banked beside a CUDA card, heat |
| [`epyc-cpu/`](epyc-cpu/) | 2x EPYC 7601 with no GPU: placing bytes across 8 NUMA nodes |
| [`moe-offload/`](moe-offload/) | the cross-box story of big MoE models: 23 measured lessons, the quality reference problem, every engine change |
| [`recipe-maker/`](recipe-maker/) | the planner: probe a box, place a model's experts, predict speed and quality, refuse bad launches |
| [`PATCHES.md`](PATCHES.md) | every patch and tool, by hardware |
| [`GLOSSARY.md`](GLOSSARY.md) | the words: MTP, n4, KL, route trim, row split, PMem and the rest |
| [`AGENTS.md`](AGENTS.md), [`atlas.json`](atlas.json) | for coding agents: a dense map, naming crosswalk, and every headline result as data |
| [`CHANGELOG.md`](CHANGELOG.md), [`CONTRIBUTING.md`](CONTRIBUTING.md) | what changed when, and how a result gets in |

## How numbers are reported here

- Every number names its conditions: card, software version, model and quantization, context
  depth, stream count, speculation settings.
- Speed claims name their correctness evidence and its limits. A check from an earlier build or another
  backend says so, and is not passed off as a check of the measured configuration. A new kernel path or a
  compiled mode that is faster and wrong is worse than useless.
- Measured means measured. Estimates are labelled as such, and so are the limits of what we tried.
- Repeated runs are all reported, as ranges.
- What did not work stays in, on the page of the card it failed on.

## Reviewed by its own hardware

Before this README went out, we asked the two B70s what they thought of it, both at once: Qwen3.8 Flash-Next on one
card, Ornith-1.5-35B-A3B on the other (the card that also runs our household assistant, which stayed up). Each got
the README, AGENTS.md, the glossary and three entries, about 40k tokens, and was asked to find results, reproduce the
best V4.1 run and say what is wrong. They read it twice. The second time both agent scores went down: with their
first findings fixed, they dwelt on what we cannot fix today, that the best V4.1 run needs files we have not
published. Fair, and the RTX 6000 page now lists every missing piece.

| reader | where it ran | speed while reviewing (1st; 2nd read) | human / agent score (1st; 2nd) | what it caught |
|---|---|---|---|---|
| Qwen3.8 Flash-Next (IQ3_XXS) | one B70, XeStrata | 70.1 tok/s; 71.1 tok/s | 84 / 58; 86 / 52 | the 12.69 run's quality check came from an earlier build; a speed run quoted at the wrong window; the GLM trim's public command gives 23.08 |
| Ornith-1.5-35B-A3B (GPTQ int4) | the other B70, vLLM, no draft | 15.6 tok/s; 15.8 tok/s, at about 40k tokens of context | 86 / 82; 80 / 74 | the 1080 Ti's "+35 %" holds only without a draft; a table credited the public build with unpublished changes |

Each model card's sampling, thinking on, one request each, 2026-10-22. Every finding in the table is fixed. Cloud
models from several families read it too, over four rounds; what they caught is in the [CHANGELOG](CHANGELOG.md).

## How it evolves

New results land as small commits on `main`, each with a line in [CHANGELOG.md](CHANGELOG.md); a batch that lands
together gets a dated tag (`v2026.10.22`). [CONTRIBUTING.md](CONTRIBUTING.md) says what a result needs to get in.
Issues and corrections are welcome, especially numbers from cards we do not have.

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [fleet](https://github.com/akapug/fleet) for [@akapug](https://github.com/akapug)
