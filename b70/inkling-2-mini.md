# Inkling 2 Mini: a coding-agent pool on two B70s, then three cards

Measured 2026-11-21 to 2026-11-27 (UTC). Thinking Machines' Inkling 2 Mini is a looped MoE: some of its layers run
more than once a token, so a small file does more work than its size suggests. It ships in NVFP4 with an MXFP8 tail
(layers 20-23), 17.6 GiB of weights. On the Threadripper box's two Arc Pro B70s (odie) it became the model our
coding-agent pool runs, replacing QFN-Coder ([many agents](many-agents.md#qfn-coder-pool-2026-11-16-to-11-26)).

**The number to plan with: 32 real coding agents on two B70s, 645.1 tok/s for two hours** (636.8 counted from the
agents' own replies), no drift from quarter to quarter (630, 641, 629, 647), no GPU fault, and 943 of 1,073 task
attempts accepted by the grader. Adding the RTX 3090 Ti box as a third card behind a sticky router took 40 agents to
874.7 tok/s in a 20-minute smoke.

The load is real: 32 [pi](https://www.npmjs.com/package/@earendil-works/pi-coding-agent) coding agents working through our frozen coding-task
set (read-only audits, derivations, dead-reference and test-gap tasks with a grader; the items are not published), at
temperature 1. Server tok/s is the engine's own count of generated tokens over the window; the ledger figure counts
only tokens in replies the agents received.

## At a glance

| setup | agents | result | window | quality | date |
|---|---:|---|---|---|---|
| two B70s, one vLLM server a card, 16 agents each | 32 | **645.08 server tok/s**, 636.75 ledger; reply decode median 28.2 tok/s (p10 16.9); first token p50 1.11 s | **2 h** (7,210 s) | 943 of 1,073 attempts accepted; the B70 stack matches the 3090 Ti reference at its noise floor (below) | 2026-11-27 |
| the same, effort low, 20 min | 32 | 527.55 server; reply median 25.4 | 20 min | 443 of 470 accepted (1,317.6 an hour) | 2026-11-27 |
| two B70s + the RTX 3090 Ti, sticky router 16 : 16 : 8 | 40 | **874.67 server tok/s**, 829.53 ledger; reply median 40.3 | 20 min | 214 of 280 accepted | 2026-11-27 |
| one B70, 16 agents (first smoke) | 16 | 153.73 server; reply median 16.3 | 20 min | 39 of 55 accepted | 2026-11-26 |
| one RTX 3090 Ti, 16 agents, re-quantized tail, drafter on | 16 | 279.8 server (desktop VRAM 0.35 GiB); 231.1 with the desktop using 0.96 GiB | 20 min | 110.1 accepted tasks an hour | 2026-11-27 |
| one Quadro RTX 6000, llama.cpp, 8 agents | 8 | 66.0 server; reply median 10.8 | 20 min | 38.6 accepted tasks an hour | 2026-11-27 |
| one MI25, llama.cpp Vulkan, 8 streams | 8 | 11.4 (cold prompts) to 15.4 tok/s (shared prefix) | ladder | not screened | 2026-11-26 |

## The two-B70 pool

**Engine.** vLLM nightly for XPU (0.34.0rc1.dev1133, g647a76866) with
[mgaruccio/vllm-inkling2-xpu](https://github.com/mgaruccio/vllm-inkling2-xpu) at 2e9d28b and vLLM PR #69186, the
plugin's CUTLASS Xe2 MoE and ESIMD decode-attention kernels and its hybrid prefix SDPA, the model's own DSpark draft head
at k=7. One server per card. Flags: `--dtype bfloat16 --kv-cache-dtype fp8 --max-model-len 32768
--max-num-batched-tokens 2048 --block-size 64 --gpu-memory-utilization 0.97`, prefix caching on. Weights: the official
checkpoint with the MXFP8 tail re-quantized to NVFP4 (below; our script, not published yet).

**Agents.** pi with a 24,576-token history budget and a fix to how it folds old history (ours, not published), effort
high for the certificate, sixteen agents a card.

The settings that got there, each a 20-minute arm on the same task set unless marked. The rows below the first five ran
earlier configurations (the smaller history budget, before the fold fix; some at 1,024 batched tokens), so read each row
against its neighbours, not as one change from the top row:

| arm | agents | server tok/s | reply median / p10 | first token p50 / p95 | accepted |
|---|---:|---:|---|---|---|
| **budget 24,576 + the fold fix, 2 h** | 32 | **645.08** | 28.21 / 16.92 | 1.11 / 8.39 s | 943 / 1,073 |
| the same, 20 min (the best 20-minute arm) | 32 | 660.70 | 30.08 / 18.60 | 0.99 / 7.06 s | 162 / 208 |
| the same at 20 agents a card | 40 | 603.60 | 22.07 / 13.66 | 1.78 / 11.21 s | 144 / 186 |
| effort medium | 32 | 603.39 | 27.20 / 16.11 | 1.22 / 8.35 s | 206 / 233 |
| effort low | 32 | 527.55 | 25.44 / 14.27 | 1.66 / 9.17 s | 443 / 470 |
| the earlier config (1,024 batched tokens, before the budget and fold change), 2 h | 32 | 492.80 | 23.24 / 13.10 | 2.13 / 16.27 s | 740 / 861 |
| 12 agents a card, 2 h | 24 | 439.32 | 27.47 / 14.03 | 1.49 / 9.94 s | 676 / 775 |
| adaptive draft length (k 7 then 5) | 32 | 514.72 | 24.80 / 13.44 | 1.38 / 14.08 s | 127 / 168 |
| 512 batched tokens | 32 | 485.93 | 25.25 / 15.36 | 3.96 / 22.78 s | 121 / 157 |
| prefix caching off | 32 | 342.73 | 16.59 / 9.75 | 8.51 / 25.23 s | 88 / 129 |

How to read it:
- **16 agents a card is the knee.** 20 a card serves less (603.6 against 660.7) and slows every reply.
- **The final config is 1.31x the earlier one over two hours** (645.08 against 492.80: the 24,576-token budget with the
  fold fix, and 2,048 batched tokens). It held its rate for the whole window: quarters 630, 641, 629 and 647 tok/s.
- **Effort trades tokens for tasks.** At effort low the same 32 agents write fewer tokens (527.6 tok/s) but finish
  far more tasks (443 accepted in 20 minutes, against 162 at effort high). A pool that is paid in finished work should
  start at low and raise effort per request; LiveCodeBench below says where high pays.
- **Prefix caching is not optional for agents:** an earlier arm with it off served 342.7 tok/s with first tokens at
  8.5 s.
- The cards stayed fault-free: the `xe` driver logged no page fault before or after any arm.

## Three cards

The two B70 servers plus the 3090 Ti's vLLM server (below) behind one sticky router that keeps each agent on its card,
16 : 16 : 8 agents: **874.67 server tok/s** (829.53 ledger) for 40 agents, reply median 40.25 (p10 18.66), first token
p50 2.74 s, prefix hits 61 % on each B70 and 27 % on the 3090 Ti, 214 of 280 attempts accepted. 20 minutes, one run.
The 3090 Ti is a desktop in daily use, so its share comes and goes.

## Quality: is the B70 stack the model?

- **Agreement with a CUDA reference.** Six arms that switch the B70 stack's pieces off one at a time, scored against the
  same checkpoint served by vLLM on the RTX 3090 Ti: the full B70 stack's greedy token matches the reference at
  **98.76 %** of positions (mean |log-prob difference| 0.0191 nats), where the 3090 Ti matches itself at 98.83 %
  (decode against prefill numerics). No piece of the stack costs measurable quality.
- **The re-quantized tail.** Layers 20-23's experts re-quantized MXFP8 -> NVFP4 in the checkpoint's own layout (the
  rule reproduces layer 19's official NVFP4 bytes from their own dequant; the other 1,228 tensors byte-identical):
  98.34 % agreement with the original, against the original's 98.44 % against itself, 0.009 nats. It frees 1.8 GiB of
  VRAM, which is what lets the drafter fit on a 24 GB card.

**LiveCodeBench, 104 problems** (release v5 + v6, the first 104 of a seeded order; 26 easy, 31 medium, 47 hard), greedy,
on the certified two-B70 stack, graded by lcb_runner's own grader:

| arm | pass@1 | easy / medium / hard | tokens against low | against low (McNemar p) |
|---|---:|---|---:|---|
| effort low, 10,240-token cap | 54.8 % (57) | 96.2 / 64.5 / 25.5 % | 1.00x | - |
| effort high, same cap | 53.8 % (56) | 96.2 / 61.3 / 25.5 % | 1.20x | -1.0 points (p 1.0) |
| effort high, 31.5k cap | 58.7 % (61) | 96.2 / 77.4 / 25.5 % | 3.12x | +3.8 points (p 0.42) |
| effort high, thinking closed at 6k (+2k answer) | 59.6 % (62) | 100 / 71.0 / 29.8 % | 0.94x | +4.8 points (p 0.30) |
| **effort high, thinking closed at 10k (+2k answer)** | **64.4 % (67)** | 100 / 80.6 / 34.0 % | 1.27x | **+9.6 points [+1.9, +17.3], p 0.031** |
| effort high, thinking closed at 14k (+2k answer) | 62.5 % (65) | 100 / 77.4 / 31.9 % | 1.74x | +7.7 points (p 0.057) |
| effort low, thinking closed at 10k (+2k answer) | 63.5 % (66) | 100 / 71.0 / **38.3 %** | 1.15x | +8.7 points (p 0.064) |

**Closing the thinking, not lengthening it, is the lever.** When a reply reaches 10,000 tokens still thinking, the
harness ends the thinking block and lets the model write a 2,000-token answer: in vLLM, a second request with
`continue_final_message: true` that appends the model's own end-of-thinking markers. On the hard problems the model
otherwise loops in its reasoning until the cap; with the close, effort high gains 9.6 points at 1.27x the tokens, more
than raising the cap to 31.5k (3.12x the tokens, +3.8). 10k is the knee: 6k and 14k both score lower. Effort low with
the same close scores within a point (63.5 %) at 1.15x the tokens and has the best hard score, so one pool setting
covers both kinds of request. One run an arm; the 10k-close against low difference is the only one under p 0.05.

## Replicas on other cards

### RTX 3090 Ti (snoopy): vLLM nightly CUDA

vLLM nightly's Inkling 2 kernels need sm_90; [agent2x0r/inkling2-sm86-kernels](https://github.com/agent2x0r/inkling2-sm86-kernels)
patches 19 files to run them on sm_86. With the official checkpoint the KV pool is 2.14 GiB (49,298 tokens), only 4.7
of 16 agents run at once, first tokens wait 42.8 s at the median, and the drafter does not fit at all: **155.6 tok/s**
(2026-11-26, 10 minutes). With the tail re-quantized the pool is 4.12 GiB (94,997 tokens) without the drafter, and the
drafter fits at 512 batched tokens: **231.1 tok/s**, 3.80 tokens a draft step, reply median 76.0 (2026-11-27).

Which KV shape to give 16 agents, three full 20-minute arms on one evening with the desktop using only 0.35 GiB of the card:

| shape | server tok/s | running at once | reply median | first token p50 | accepted tasks an hour |
|---|---:|---:|---:|---:|---:|
| drafter off, 109k-token pool | 247.9 | 13.4 | 20.1 | 6.3 s | 104.07 |
| drafter on, 24k window | 295.7 | 6.2 | 70.0 | 20.8 s | 104.12 |
| **drafter on, 32k window** | 279.8 | 6.1 | 67.0 | 22.9 s | **110.06** |

Width buys first tokens, the drafter buys reply speed, and finished work ties within 6 %. A smaller
`--max-model-len` buys no width: vLLM V1 reserves no KV for a request's maximum length, so 24k and 32k ran on the
same pool with the same 6 requests at a time. We keep the drafter at 32k.

### Quadro RTX 6000 (droopy): llama.cpp CUDA

vLLM has no path for this model on Turing (sm_75), so this is llama.cpp with ggml-org's IQ4_NL file and per-slot
q4_0 K/V, the agents on another box. llama.cpp 4c7344447 against the 212c547fe of the day before: llama-bench decode 53.7
against 40.8 tok/s, prompt reading 1,555 against 911 (pp512), and at 8k depth 991 against 450 (the flash-attention
change in 2d5b3ddde). For 8 agents at 32,768 tokens a slot: **66.0 server tok/s** against 46.8 (1.41x), reply median
10.8, 38.6 accepted tasks an hour (the old build's arm ran 900 s, the new one 1,212 s; on the common 900 s the
gain is 1.21x on tokens in finished replies).

- **16 x 24,576 does not fit, by 63 MiB.** llama.cpp's flash-attention compute buffer is an f16 copy of one
  full-attention layer's K/V for every cell of the pool (1,646 MiB, about 4.1 KB a cell, read from how it scales; not
  traced). 16 agents at 22,528 a slot serve 72.4 tok/s, only 1.10x the 8-agent arm, at half the per-agent speed
  (median 5.7; none of the 16 at 10 tok/s or more).
- **q8_0 K/V costs nothing in speed here:** 8 agents at 28,672 a slot 62.8 tok/s (0.95x), and its KL divergence
  against f16 is 0.0061 against q4_0's 0.0113. For this card: 8 slots of 28,672 at q8_0.
- An earlier 16-agent arm that collapsed to 30.5 tok/s was the box's own CPU load, not the card: with the host quiet,
  the same card was 85 % busy.

### MI25 (lassie): no

llama.cpp Vulkan (Atomic Engine e110381 with a strided SCALE patch), AtomicChat's AD-IQ3_S, 8 streams of ~2k-token
prompts: 11.4 tok/s with fresh prompts, 15.4 with a shared prefix, 26.5 decode alone. The pool needed 80. HBM reaches
the card's 93 C guard within seconds of load and the guard holds the server 20-30 % of the time; prompt reading at
~190 tok/s holds the whole card ~11 s for each new 2k prompt. The card stays a research card until it gets better
cooling. Details on the [MI25 page](../mi25/#inkling-2-mini-as-a-pool-replica-too-hot).

## KV room is the binding limit

On every card the pool ran out of KV room before it ran out of compute:
- one B70 at 16 agents: a 64,507-token FP8 pool (8.02 GiB), 305 preemptions in 20 minutes, prefix hits 9.7 %, first
  tokens 10.8 s at the median. On two cards no arm preempted at all; at 12 agents a card the pool peaked at 95-97 %
  full over two hours;
- the 3090 Ti: 49,298 tokens and 758 preemptions in 10 minutes until the re-quantized tail freed 1.8 GiB;
- the RTX 6000: a compute buffer sized by the whole pool, not by the active stream.
Measure the pool size the engine reports and the preemption count before adding agents.

## What did not work

- **QFN-Coder on the same task set.** The model the pool ran before (on a scripted workload, 310 tok/s for two hours)
  served 64 coding agents at 96.4 tok/s with first tokens at 24.6 s (p90 170.8 s) and none of 86 attempts accepted:
  64 fresh agents pushed ~644k new prompt tokens through its serial 512-token prompt reads.
  [Many agents](many-agents.md#qfn-coder-pool-2026-11-16-to-11-26)
- **The Inkblot-2-mini edit** (Ex0bit's loop-control fine-tune, 327 attention, router and loop-controller tensors
  changed by 0.84 % on average, no expert changed), grafted onto the pool checkpoint: faster on agent work (on the
  two-B70 pool at effort high, 526.5 accepted tasks an hour a card against the base model's 232.0; on the 3090 Ti,
  133.9 against 101.1 an hour; one agent in maya-ik2 did the same 18 tasks in 157 s on a quarter of the tokens) but
  **10.6 points worse on LiveCodeBench** (52.9 against 63.5 %, hard 19.1 against 36.2 %, p 0.003). The edit was
  trained with every answer capped at 1,536 tokens, so it is terser, not smarter. The pool keeps the base model at
  effort low, which beat the graft on accepted work outright (1,317.6 accepted an hour on two cards against 1,053.0).
- **LoopCD-Logits** (arXiv 2610.02185, the omlx rule, omega_max 0.5) built into llama.cpp: no gain. HumanEval+ is at
  its ceiling for this model (0 discordant pairs of 164); on LiveCodeBench 104 with thinking skipped -1.0 points
  [-2.9, 0.0], 1 discordant pair, at 10 % decode cost. A fairer retest at Q8_0 on a B70 stopped when the box locked up;
  that question is open.
- **DFlash2 in vLLM nightly** boots for this model and accepts no drafted token.
- **GSQ-RCO GGUFs** (IQ3_S): 1.20x the decode of AtomicChat's AD-IQ3_XXS at 0.87x its VRAM, but 1.50x its KL divergence on
  code; AD-IQ3_XXS stays the small file.
- **llama.cpp before 690b87d88 decodes the mainline NVFP4 GGUFs to noise** (38 % top-1 against the official
  weights): a loop-slot scale bug, fixed upstream by 690b87d88 (#32704). Bisected: the parent at 58.9 % top-1, the fix at 99.0 %.

## Reproduce

Public: vLLM nightly XPU, the plugin and PR #69186 above, the flags above, the official checkpoint
(`thinkingmachines/Inkling-2-Mini`), pi. Not public: the re-quantization script, our coding-task set and grader, the
history fold fix, the sticky router, and the forced-close splice (described above; the vLLM side is one
`continue_final_message` request). Raw logs stay in our lab notes.

## Not measured yet

- Each card's share inside the three-card pool over two hours.
- The close-at-10k setting under the pool's real agent traffic (it was measured on LiveCodeBench, one request at a time).
- The GTX 1080 Ti as a replica; the MI25 after a cooling fix.
- The B70 pool at 32k-48k context per agent (the task set's replies stay under 32k).

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [fleet](https://github.com/akapug/fleet) for [@akapug](https://github.com/akapug)
