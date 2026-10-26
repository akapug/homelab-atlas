# Many agents on two B70s: Qwen3.8 Flash-Next, batched

How many coding agents can one box serve at once, each fast enough to work with? We spent a week on that question
with the Threadripper box (a 2019 Threadripper 3960X, 128 GB DDR4, two Arc Pro B70 32 GB cards), Qwen3.8 Flash-Next
(125B MoE, 6B active) at IQ3_XXS, and the XeStrata engine plus our batching changes. The work is paused as of
2026-10-26. This page is where it stopped.

**On real coding-agent traffic, 16 agents get 80 to 86 tok/s in total (mean 83.4, three runs), about 5 tok/s each, and one agent alone gets
32.6 tok/s.** That is the number to plan with. At about 5 tok/s each, a typical turn takes minutes: fine for
unattended batch agents, slow for an interactive one. Real agents read 4 to 5 new prompt tokens for every token they write,
and this box reads prompts at about 0.85k tok/s while it is also decoding, so prompt reading, not decode, sets the
pace. See [Real coding-agent traffic](#real-coding-agent-traffic-the-number-to-plan-with).

**On a synthetic load that reads each prompt once and then only writes, the same box decodes 377.9 tok/s over 32
agents** (11.81 each, 0.729 tok/s per watt). That is a decode ceiling, not what agents doing real work get, and it is
the half of the page we measured first. See [Decode ceiling](#the-decode-ceiling-a-synthetic-load).

The engine changes are published as an experimental fork, each one behind its own switch, with the measured effect of
every switch on real traffic: [Engine code](#engine-code-experimental).

## Real coding-agent traffic: the number to plan with

Measured 2026-10-26 (UTC) on the build the box serves now. Every row in this section is in Recipe Maker's run ledger
(`recipe-maker/runs/`, run ids `odie-fnx-iq3xxs-2xb70-real-*-ss16-*`).

**The load.** 16 recorded coding-agent sessions (long subagents of our own fleet doing real build work, 923 turns in
all), replayed against the server with their own prompts, their own output lengths and their recorded tool time
between turns. The server keeps each agent's earlier prompt in its cache, so it reads only what is new each turn. The
first 600 s are a warm-up and are not counted; the measured window is the next 600 s. In that window the agents read
3.9 to 4.9 new prompt tokens per token they wrote; with a perfect cache this trace would read 4.33.

| run | agents | total tok/s | per agent | prompt reading | errors |
|---|---|---|---|---|---|
| the served build (three runs) | 16 | **86.0, 84.6, 79.6** (mean 83.4) | about 5.2 | about 0.87k tok/s | 0 |
| the build served before it (two runs) | 16 | 78.6, 70.9 (mean 74.8) | about 4.7 | | 0 |
| the served build, prompts read in chunks of 4,096 / 8,192 / 16,384 tokens (default 2,048) | 16 | 77.8 / 89.7 / 87.6 | | 0.81k / 0.84k / 0.86k tok/s | 0 |
| the served build, one agent alone | 1 | **32.6** | 32.6 | about 0.85k tok/s | 0 |

How to read it:
- **One agent gets 32.6 tok/s of useful output.** In the 600 s window it spent 47 % of the time decoding (at 68.8
  tok/s), 31 % reading prompts, and about 22 % in its recorded tool time, when the server has nothing to do for it.
- **16 agents get about 2.6 times one agent's total, not 16 times.** The server reads longer prompts one agent at a
  time (it packs only short reads beside decode), and every agent needs a prompt read each turn. With prompt reading
  at about 0.85k tok/s and 4 to 5 prompt tokens per token written, the box tops out near 170 to 210 tok/s of output
  even with free decode (our reading, from those two measured figures); the measured 80 to 86 sits under that because
  reads and decode also slow each other.
- **The served build is +11.5 % over the one before** (mean of 3 runs against 2; replay noise is about 11 % run to
  run, so treat it as likely, not proven), with 0 errors and 1.12 turns per agent-minute against 0.95.
- **Bigger prompt chunks are not a lever here.** 4,096 to 16,384 moved prompt reading by -6 % to -1 % against the
  2,048 default; the totals follow how much each window happened to read, not the chunk.
- **Why prompt reading is slow on this box.** The model does not fit the two cards: their expert caches hold 18.6 GiB
  each, the rest of the experts sit in host RAM, and by our estimate about 7 GB of them stream over PCIe for every
  prompt chunk, so short reads (agents mostly add a few hundred to a few thousand tokens a turn) run at the slow end.
  XeStrata's own release notes for one B70 give 1.2k to 2.5k tok/s for 1k to 32k-token prompts and 566 tok/s for
  agent-style reads on a newer version than ours (0.2.2 against our 0.1.45 base).

**For comparison, one agent on GLM-5.3-Flash** (the [3090 Ti box](../3090-ti/#glm-53-flash-a-route-trim-clears-25-toks),
decode 25.9 and prompt reading 761 tok/s in our coding-seat setup) works out to about 20 to 22 tok/s of useful output
at the same 4 to 5 prompt tokens per token written, before tool time (our estimate from those two measured rates, not
a replay of this trace). The one Flash-Next agent above got 32.6 with its tool time and about 42 while the server was
working for it (measured). So before tool time Flash-Next on the two B70s is roughly twice as fast for one agent, from
a smaller model; one side is an estimate on another box, so this is not a matched comparison.

**Recipe Maker's part.** Before the chunk-size runs, Recipe Maker wrote down predictions from the hardware alone
(`recipe-maker/data/predictions-prefill-chunk-1026.json`): prompt reading 913 to 922 tok/s, and a fail line at
anything outside 780 to 1,000. All three runs landed inside the line, but the predictions read 7.5 to 12.2 % high, and the
small gain it predicted from bigger chunks did not appear. Its agent formula, applied after the fact to the one-agent
run without refitting, read 33 % high until the tool time between turns was added (4.5 % high with it).

## The decode ceiling: a synthetic load

Everything from here down was measured first, on 2026-10-23 and 2026-10-24, with a synthetic load: every agent reads
one prompt once and then only writes. It measures how fast the batched decode runs, which is the ceiling for the
rows above, not what agents doing real work get.

### At a glance

| | |
|---|---|
| box | Threadripper 3960X (24 cores), 128 GB DDR4, 2x Arc Pro B70 32 GB on PCIe 4 x16 (box C, `odie` in Recipe Maker) |
| model | Qwen3.8 Flash-Next, GSQ-RCO IQ3_XXS, its layers split across the two cards |
| engine | [XeStrata](https://github.com/chriscorbell/intel-arc-pro-b70-qwen38-flash-next) plus our batched-decode branches ([experimental public fork](#engine-code-experimental); the synthetic rows ran earlier builds of it) |
| useful shape (run M32c) | **329.6 tok/s total over 32 agents, each 10.27 to 10.32 tok/s**: 2 slots of 262,144 and 30 of 131,072, one 248k-token and two 114k-token prompts among them; 400.1 W for the cards + 129.1 W CPU package = 529.2 W, **0.623 tok/s per W**; 52.0 GiB of host RAM pinned for older KV |
| newer build, uniform (run served-c8-base) | **377.9 tok/s total, 11.81 per agent (slowest 11.80)**, every slot 131,072, prompts of about 14k; 404.3 W + 114.3 W = 518.6 W, **0.729 tok/s per W**; the build combines the switches that won (below); no parity or quality run on this build yet (not measured) |
| uniform peak (run C2bs) | **349.6 tok/s total, 10.92 tok/s per agent (slowest 10.92)**, every slot 131,072, prompts of about 14k; 404.4 W + 131.1 W = 535.5 W, **0.653 tok/s per W** (0.864 per card W); 47.7 GiB pinned |
| against its control | the same build with its four new switches off (C2a): 337.1 tok/s, so the switches are worth +3.7 % |
| quality check | the engine's card parity suites pass with 0 failures on the C2bs build; the M32c build adds the server's slot fix and later engine changes, and its own parity run is not part of this page; no agent or KL check on either build yet (not measured) |

### Conditions

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

### The ladder

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
  fed 24 agents or 32. On this load, tok/s per watt rises with the agent count as long as each agent stays fast enough.
  It did not carry over to real agents, whose prompt reading sets the pace (see the first section).
- **k8v4 beats int8 on every axis** at 32 agents: +3.6 % tok/s (330.1 against 318.7) and 13.3 GiB less pinned host RAM.
  It is our standing KV format now.
- **The fair comparison for the new switches is against C2a**: same build, same prompts. Three switches give +3.1 %
  (C2b), the fourth brings it to +3.7 % (C2bs). C2b and C2bs are within one run's resolution of each other; we lead with
  C2bs as the best reading, not as a proven gain over C2b. Most of the gain is more tokens at about the same
  power (per watt +0.9 %). C2bs against U32k (+5.9 %) also changed the prompts, so part of that gap is the workload, not
  the engine.

### What the switches do, and what did not work

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
  prefetch launch ahead of the read was started to win back about half of it; it was not finished before the pause.
- **A kernel win is not a server win until the server says so.** On an earlier build we turned off its six
  switches one at a time at 32 agents. Only the hc read window carried a gain (325.9 with it, 313.3 without); the
  other five moved the total by under 1 %, the expert-kernel switches among them included, which had won 1.18x to 1.49x
  per call in the microbenchmark.

### Mixed context: 2 agents at 262,144, 30 at 131,072

Top-level agents want more context than their subagents, so the shape we care most about is 2 big slots of 262,144
tokens beside 30 slots of 131,072 (k8v4, C2bs's switches). One agent had a 248k-token prompt and two had 114k-token
prompts; the other 29 had the usual ~14k.

| run | build | agents decoding | total tok/s | per agent | tok/s per W (cards + CPU) | pinned host RAM | loading |
|---|---|---|---|---|---|---|---|
| M32b | second build | 31 of 32 | 295.6 | 10.03 (median; one agent 0) | 0.543 | 52.0 GiB | 1,760 s |
| **M32c** | **a later build of the same branches, with the slot fix below** | **32 of 32** | **329.6** | **10.27 to 10.32** | **0.623** | **52.0 GiB** | **957 s** |

- Both slot sizes batch in one window, and the 248k-token agent decodes at the same rate as the rest (10.30 tok/s in
  M32c).
- In M32b one agent got no slot for the whole window while a big slot sat idle, and two started late: the server gave
  a big slot only to a request that did not fit a small one. The fix lets a short request use an idle big slot. With it,
  every agent decodes and the total is 11.5 % higher (M32c, which also runs on a later engine build, so the fix is not
  the only change).
- M32c is 5.7 % below the uniform peak (C2bs, 349.6). M32c and C2bs differ in
  build as well as shape, so that figure is our reading, not a matched comparison.

### Newer build: 377.9 tok/s at 32 agents

Measured 2026-10-24. One build carries every switch that won its own arm; its full environment is in the run ledger
(`odie-fnx-iq3xxs-2xb70-sweep-served-c8-base-1024`). It
uses the uniform shape and the same rate probe as C2bs: 32 slots of 131,072 tokens, prompts of about 14k, one 60 s window.

| run | total tok/s | per agent (slowest) | watts (cards + CPU package) | tok/s per W |
|---|---|---|---|---|
| served-c8-base (the combined build) | **377.9** | 11.81 (11.80) | 404.3 + 114.3 = 518.6 | **0.729** |
| served-c8-oasync (plus an asynchronous output copy) | 378.4 | 11.82 (11.82) | 515.0 | 0.735 |
| served-c8-q81 | 373.6 | 11.67 | 512.7 | 0.729 |
| served-c8-all (every candidate switch on) | 375.2 | 11.72 | 513.6 | 0.731 |

The four runs sit within about 1 %, which one window per run does not resolve, so the asynchronous copy is not a
measured win (our reading). Against C2bs (349.6) the combined build is +8.1 %, with build and switches changed together.

**What the rate probe leaves out.** It counts decode tokens once every agent is decoding. Real agents also wait while
the server reads their prompts, and that turned out to be most of the story: see
[Real coding-agent traffic](#real-coding-agent-traffic-the-number-to-plan-with).

## Engine code (experimental)

This is research code, published so another agent or researcher can read it, build on it or take one piece of it. It
is not a supported engine. Our claims about it stop at the numbers below, measured on one box (two B70s, Qwen3.8
Flash-Next IQ3_XXS) on the dates given; we have not run it on any other model, card or engine version.

The changes live on our public fork of [XeStrata](https://github.com/MistVVK/XeStrata) (LGPL-3.0, itself built on
[Strata](https://github.com/Niko1221/Strata)), each branch one squashed commit on upstream `180d077` (version 0.1.45):

| branch | what it holds | ran in serving |
|---|---|---|
| [`homelab-batched`](https://github.com/akapug/XeStrata/tree/homelab-batched) | batched decode for many agents at once, the kernel forms, the slot and prompt-reading switches on this page | yes: the box has served it since 2026-10-26 |
| [`homelab-l5-prefill-export`](https://github.com/akapug/XeStrata/tree/homelab-l5-prefill-export) | `PREFILL` and `RESTORE_PREFILL` serve commands: one engine reads a prompt and saves the session to a file, another restores it and decodes | no; the restore refuses a server split across two cards, so it never ran end to end |
| [`homelab-kvib-prefetch`](https://github.com/akapug/XeStrata/tree/homelab-kvib-prefetch) | parked conversations written to and read from another box over the network, with a read-ahead switch | no server measurement |

The two smaller branches split off `homelab-batched` before its newest switches, so each holds a subset of them.

**Switches.** Almost every change sits behind an environment variable that is off when unset, so an unset build
behaves like upstream. Five act by default; each commit message lists them: batch slots share the main session's
rotary table (`STRATA_SLOT_OWN_ROPE=1` restores upstream), an idle slot's pad row no longer writes into that slot's
conversation, a request waits up to 2 s for its own conversation's slot (`STRATA_SLOT_OWN_WAIT_S`), a pinned host copy
of the KV cache that would leave less than 16 GiB of RAM free is refused (`STRATA_KV_HOST_RESERVE_GIB`), and
conversation files are copied on 4 threads (`STRATA_CONV_IO_THREADS`). Those five ran in every server measurement on
this page; we did not measure them on their own.

The served build turns on 37 switches. In `STRATA_` variables, without the prefix:

```
BATCH_DENSE_MMQ_MIN=4 BATCH_DENSE_XMX=7 BATCH_GDN_FORM=4 BATCH_GDN_SLOTS=1 BATCH_HANDOFF_ONE=1 BATCH_HC_XMX=7
BATCH_PLE_LATE=1 BATCH_QSA_ROWS=1 BATCH_QSA_STREAMED=1 BATCH_RESIDENT_GRAPH=1 BATCH_SELF_COMMIT=1 BATCH_TOPK_ONCE=1
EXPERT_GRID_LEAN=1 EXPERT_IQ4NL_TABLE=1 GDN_AB_FORM=4 GDN_BLOCK_GRF=1 GROUPED_ENTRIES=8 HC_Q8_WIN=2 MMVQ_ROWS=2
MMVQ_TILE_COLS=16 MMVQ_WG_ROWS=2 POOL_SPIN_US=0 QSA_ATTN_FAST=1 QSA_SCORES_STRIDE=1 SHEXP_XMX=7 SKIP_NO_PCIE_SHARE=1
STAGE_LEND_DROP=1 SLOT_SPLIT_REUSE=1 BATCH_PREFILL=1 SHARED_PREFIX=1 BATCH_MAX_T=32 REFILL_STAGED=1
PARK_SKIP_SLOT_HELD=1 SLOT_DIRECT_COPY=1 DMMQ_LEAN=7 QSA_ATTN_LANES=1 PREFILL_STREAM_CPU_SHARE=0.25
```

**What each tested switch did on real agent traffic.** Each row is the 16-agent replay above with one switch changed
from the served build; one run each, so with about 11 % run-to-run noise only the large moves mean anything.

| switch | served build with it changed (served: 83.4 tok/s, mean of 3) | verdict |
|---|---|---|
| slot refill through staged copies (`REFILL_STAGED`, `PARK_SKIP_SLOT_HELD`, `SLOT_DIRECT_COPY`) | off: 78.8 (about -6 %) | on; inside noise |
| leaner dense decode kernel (`DMMQ_LEAN=7`) | off: 73.0 (about -12 %) | on; one run, just past the noise |
| decode attention in 16-lane groups (`QSA_ATTN_LANES=1`, same output bit for bit) | off: 72.5 (about -13 %) | on; one run, just past the noise |
| share of streamed experts computed on the CPU while reading prompts (`PREFILL_STREAM_CPU_SHARE=0.25`) | off: 81.6 | on; no measured effect |
| window of 32 rows, not 8 (`BATCH_MAX_T=32`) | 8: 80.4 | on; no measured effect at 16 agents |
| MoE combine fused into the next norm while reading prompts (`PREFILL_COMBINE_HC=1`) | on: 69.3 (about -17 %) | off |
| chunked prompt reading for the gated-delta layers (`GDN_CHUNK=1`) | served nothing: the first prompt never progressed | off; broken in our build, cause not found |
| pipelined reads plus shortest-new-first admission (`BATCH_READ_PIPE`, `ADMIT_POLICY=shortest-new`) | prompt reading 23 % faster, but decode starved | off; the two halves were not tested apart |

The decode-kernel switches (the rest of the list) were chosen on the synthetic load in the
[switch table](#what-the-switches-do-and-what-did-not-work); turning them off one at a time on real traffic was not
done.

**Where it might lead.** The open problem on this box is prompt reading, not decode: see
[Where it stopped](#where-it-stopped-not-run). The parts most likely to be worth taking elsewhere are the one-launch
batch kernels in `src/core/verify.cpp` and `src/kernels/xe/`, the slot handling in `serve/server.py`, and
`window_prefill.hpp`, which packs short prompt reads into spare decode rows and is the natural place to grow a
several-agents-per-chunk prompt read.

## Reproduce

- The engine: build the `homelab-batched` branch above as you would XeStrata itself. It is experimental code with no
  support; only the switches the served build uses (listed above) have run in serving, and that full set only since
  2026-10-26.
- Every server-rate row on this page is in Recipe Maker's run ledger with its full engine flags and environment
  (`recipe-maker/runs/`, run ids `odie-fnx-iq3xxs-2xb70-sweep-*` for the synthetic load and
  `odie-fnx-iq3xxs-2xb70-real-*-ss16-*` for the real-agent replays); `node report.js --curve all` reads them. The
  microbenchmark, profiler, parity and earlier-build figures are our lab results and are not in the public ledger.
- What you cannot get from us: the recorded agent sessions behind the replay (they are our own work sessions) and the
  replay client. To measure your own box the same way, record a few of your own agent sessions and replay them with
  their prompts, outputs and tool time; a load that reads each prompt once will overstate what agents get, as it did
  for us.
- The single-card Flash-Next setup that this builds on is on the [B70 page](README.md) and in
  [MoE offload point 22](../moe-offload/#what-we-learned-measured).

## Where it stopped (not run)

The research paused on 2026-10-26. These were designed or started and have no results; each is a lead, not a claim.

- **Read several agents' prompts in one pass.** Each prompt chunk pays to stream the experts that do not fit the
  cards; reading several agents' new tokens in the same chunk would pay that once for all of them. The engine today
  packs only short reads (up to 512 new tokens) into spare decode rows and reads longer ones one agent at a time. Our
  estimate is about 125 tok/s at 16 agents if prompt reading reached the 2.5k tok/s long-prompt rate; not built.
- **Move to XeStrata 0.2.2,** which reads prompts with oneMKL and lists 1.6k to 2.5k tok/s for 4k to 32k-token
  prompts on one B70 in its release notes, against the 0.1.45 base under our branches. Not rebased.
- **Read prompts on another box's GPU.** A CUDA build of the same engine read an 11,565-token prompt at 640 tok/s on the
  RTX 6000 box, and the 380 MB of state it produces would cross our [56 Gb/s link](../network/) in well under a second.
  That rate is below the B70s' own, so it would add prompt-reading capacity in parallel, not speed. The engine refuses to restore such state into a server split across two cards, so it never ran end to end.
- **A model whose experts fit both cards,** which removes the streaming from every prompt chunk; a quality question
  before a speed one.
- From the synthetic-load work: 48 and 64 agents at 131,072 each, every slot at 262,144, moving more layers to the
  first card (card 2's stage took 47.9 ms of each 16-row window against card 1's 40.1 ms, under the profiler), and the
  shared-memory shared expert at the server level.

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [arc](https://github.com/akapug/arc) + [fab](https://github.com/akapug/fabric) for [@akapug](https://github.com/akapug)
