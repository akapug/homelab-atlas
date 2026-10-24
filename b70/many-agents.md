# 32 agents on two B70s: Qwen3.8 Flash-Next, batched

How many coding agents can one box serve at once, each fast enough to work with? On the Threadripper box (a 2019
Threadripper 3960X, 128 GB DDR4, two Arc Pro B70 32 GB cards), the answer so far is **32 agents at 10.92 tok/s each,
349.6 tok/s in total**, every agent holding a 131,072-token context slot, at **0.653 tok/s per watt** of card plus CPU
package power. The model is Qwen3.8 Flash-Next (125B MoE, 6B active) at IQ3_XXS, and the engine is XeStrata with our
batching changes.

Measured 2026-10-23 and 2026-10-24 (UTC). The engine changes behind these numbers are on our XeStrata branches and are
**not published yet**. The work is still running: the last section
lists what is in progress.

## At a glance

| | |
|---|---|
| box | Threadripper 3960X (24 cores), 128 GB DDR4, 2x Arc Pro B70 32 GB on PCIe 4 x16 (box C, `odie` in Recipe Maker) |
| model | Qwen3.8 Flash-Next, GSQ-RCO IQ3_XXS, its layers split across the two cards |
| engine | [XeStrata](https://github.com/chriscorbell/intel-arc-pro-b70-qwen38-flash-next) plus our batched-decode branches (unpublished) |
| result | **349.6 tok/s total, 10.92 tok/s per agent (slowest agent 10.92), 32 agents** (run C2bs below) |
| power | 404.4 W for the two cards + 131.1 W CPU package = 535.5 W; **0.653 tok/s per W** (0.864 per card W) |
| against its control | the same build with its four new switches off (C2a): 337.1 tok/s, so the switches are worth +3.7 % |
| memory | 47.7 GiB of host RAM pinned for older KV |
| quality check | the engine's card parity suites pass with 0 failures on this build; no agent or KL check on the batched build yet (not measured) |

## Conditions

These hold for every row on this page unless the row says otherwise.

- **Layout.** The model's layers split at layer 25 between the two cards, which work as a two-stage pipeline. The 32
  agents run as two groups of 16 (`--batch-groups 2`), so the two cards can work on different groups at the same time.
- **Context.** Every agent gets a slot of 131,072 tokens. The newest 4,096 tokens of each agent's KV cache stay in VRAM
  (`--kv-resident 4096`); older KV streams to pinned host RAM.
- **KV format.** `k8v4`: 8-bit keys, 4-bit values. int8 rows say so.
- **No speculation in the batch** (`--no-batch-mtp`). Thinking on and sampling at the server defaults; requests set no
  token limit.
- **Workload: synthetic, not live agents.** All agents start at once, each on one of 8 coding-agent prompts of about
  14k tokens. In the U rows, 3 of the agents instead get a prompt of about 114k tokens. Once every agent is decoding,
  the client counts the streamed chunks of all agents over a 60 s window, then cancels. The slot size is what each agent
  may grow to; the prompts do not fill it.
- **Watts** are the two cards' own energy counters plus the CPU package (RAPL) over the same 60 s, not wall power. The
  rest of the box (DRAM, drives, fans, power-supply loss) is not counted.
- **One run per row.** The rate probe is one 60 s window per configuration, so differences of about 1 % or less are not
  resolved (our reading; no repeat runs yet).
- The box served nothing else during the bench.

## The ladder

| run | build | KV | prompts | agents | total tok/s | per agent | tok/s per W (cards + CPU) | pinned host RAM |
|---|---|---|---|---|---|---|---|---|
| U24i | first batched build | int8 | 3 at ~114k, rest ~14k | 24 | 295.6 | 12.32 | 0.564 | 45.6 GiB |
| U32i | first batched build | int8 | 3 at ~114k, rest ~14k | 32 | 318.7 | 9.96 | 0.610 | 61.0 GiB |
| U32k | first batched build | k8v4 | 3 at ~114k, rest ~14k | 32 | 330.1 | 10.32 | 0.624 | 47.7 GiB |
| C2a | second build, its new switches off | k8v4 | all ~14k | 32 | 337.1 | 10.53 | 0.647 | 47.7 GiB |
| C2b | second build, 3 new switches on | k8v4 | all ~14k | 32 | 347.5 | 10.86 | 0.650 | 47.7 GiB |
| **C2bs** | **second build, 4 new switches on** | **k8v4** | **all ~14k** | **32** | **349.6** | **10.92** | **0.653** | **47.7 GiB** |

How to read it:
- **24 to 32 agents costs no power.** U24i and U32i drew 524.0 W and 522.8 W: the cards draw about the same whether
  fed 24 agents or 32. Tok/s per watt rises with the agent count as long as each agent stays fast enough. That makes
  the agent count the knob, and the floor (here about 10 tok/s each) decides where to stop.
- **k8v4 beats int8 on every axis** at 32 agents: +3.6 % tok/s (330.1 against 318.7) and 13.3 GiB less pinned host RAM.
  It is our standing KV format now.
- **The fair comparison for the new switches is against C2a**: same build, same prompts. Three switches give +3.1 %
  (C2b), the fourth brings it to +3.7 % (C2bs). C2b and C2bs are within one run's resolution of each other; we lead with
  C2bs as the best reading, not as a proven gain over C2b. Most of the gain is more tokens at about the same
  power (per watt +0.9 %). C2bs against U32k (+5.9 %) also changed the prompts, so part of that gap is the workload, not
  the engine.

## What the switches do, and what did not work

Per-kernel numbers come from the engine's batch-window microbenchmark at 16 agents' rows, in ms per decode step, after
50 warm-up calls. Server numbers are the 32-agent rate probe above.

| change | microbenchmark | at the server (32 agents) | verdict |
|---|---|---|---|
| Gated DeltaNet alpha/beta computed inside the convolution launch (`GDN_AB_FORM=4`) | -1.2 ms | part of C2a to C2b, +3.1 % | on |
| faster sparse-attention kernel and block-score stride (`QSA_ATTN_FAST`, `QSA_SCORES_STRIDE`) | -0.78 ms | part of C2a to C2b | on; the fast form matches the reference within tolerance, not bit for bit |
| streamed sparse-attention rows (`BATCH_QSA_STREAMED`) | | 349.6 tok/s against 347.5 (C2bs against C2b, +0.6 %, inside one run's resolution) | on |
| narrower dense kernels for three weight types (`NARROW_TYPES`) | | 348.0 tok/s against C2b's 347.5, but 549.5 W: 0.633 tok/s per W against 0.650 | off |
| wider hc weight-read window (`HC_Q8_WIN=10`) | -0.6 ms | 345.3 tok/s against C2bs's 349.6 (-1.2 %) | off |
| router fused into the next launch (`ROUTER_FUSE`) | +0.7 to +0.9 ms | not run | off |
| wider dense matrix-vector kernel (`MMVF_WIDE`) | +0.5 ms | not run | off |
| shared expert through shared local memory | +0.08 ms | not run | off |

Why the losers lost, and what that taught us:

- **The warm-up trap.** On these cards the first 25 to 50 calls of a kernel shape in a process run up to 1.3x slow.
  Our early switch readings used 3 warm-up calls and were noise: `HC_Q8_WIN=10` read as a 0.8 ms loss at 3 and a 0.6 ms
  gain at 50. The microbenchmark now warms up 50 calls by default. Even then, its 0.6 ms did not show at the server.
- **Fusing across workgroups loses on Xe2.** Fusing two kernels whose second half needs every workgroup of the first
  to finish takes a device-wide atomic plus a fence in the last workgroup. Inside a recorded graph, that costs more
  than the one extra launch it saves. Both fusions of this kind (the router and the hc reads) came out slower.
- **Cold L2 costs about 25 us per hc weight read** (112 us cold against 85 to 88 us warm), about 1.2 ms a step. A
  prefetch launch ahead of the read is being built to win back about half of it (in progress, no result yet).
- **A kernel win is not a server win until the server says so.** On an earlier build we turned off its six
  switches one at a time at 32 agents. Only the hc read window carried a gain (325.9 with it, 313.3 without); the
  other five moved the total by under 1 %, the expert-kernel switches among them included, which had won 1.18x to 1.49x
  per call in the microbenchmark.

## Mixed context: 2 agents at 262,144, 30 at 131,072

Top-level agents want more context than their subagents, so we also ran 2 big slots of 262,144 tokens beside 30 slots
of 131,072 (C2bs's build and switches; k8v4). One agent had a 248k-token prompt and two had
114k-token prompts.

- Both slot sizes batch in one window. The 248k-token agent decoded at the same rate as the rest: 10.03 tok/s (the
  median agent).
- **295.6 tok/s over the 31 agents that got a slot**, 0.543 tok/s per W, 52.0 GiB pinned. Loading took 1,760 s,
  most of it the 248k-token prompt.
- One agent got no slot for the whole window while a big slot sat idle, and two started late. The server gives a big
  slot only to a request that does not fit a small one. Letting a short request borrow an idle big slot should fix it;
  that change is not written yet.

## Reproduce

Not yet: the batched engine is on our branches and not published, and the benchmark client is ours. What you can check
today:
- Every row on this page is in Recipe Maker's run ledger with its full engine flags and environment
  (`recipe-maker/runs/`, run ids `odie-fnx-iq3xxs-2xb70-sweep-*`); `node report.js --curve all` reads them.
- The single-card Flash-Next setup that this builds on is on the [B70 page](README.md) and in
  [MoE offload point 22](../moe-offload/#what-we-learned-measured).

## In progress (no results yet)

Running or queued when this page was written (2026-10-24 05:00 UTC): 48 and 64 agents at 131,072 each (a first
48-agent run hit the server's slot cap and does not count); every slot at
262,144; the big-slot fix above; the L2 prefetch for the hc reads; moving more layers to the first card, since card 2's
stage takes 47.9 ms of each 16-row window against card 1's 40.1 ms (measured under the profiler, which itself costs
about 4.6 %); and the shared-memory shared expert at the server level. Each lands here with its numbers, and the losers
stay in.

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [arc](https://github.com/akapug/arc) + [fab](https://github.com/akapug/fab) for [@akapug](https://github.com/akapug)
