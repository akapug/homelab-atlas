# Multi-box plan: one model over every card, and how throughput moves with batch size

Written 2026-11-11, before any multi-box QFN run. The two frozen predictions are in
`data/predictions-roofline-mb-*-1111.json` (written 09:14:19Z, RM 401e3fe).

**The question.** Which model, quant and placement over all our cards serves the most useful tok/s to coding agents, at
QFN quality or better? And how does that change as the number of agents B goes from 1 to 128?

**Short answer.**
- QFN UD-Q4_K_XL, split by free memory over lassie's 1080 Ti, droopy's RTX 6000 and odie's two B70s, run on upstream
  llama.cpp RPC. This ranks first among the plans whose fidelity is still open:
  - about 97 useful tok/s at 8 agents (ESTIMATE);
  - 53 tok/s for a single stream (ESTIMATE);
  - 15 streams at the trace's mean depth (ESTIMATE).
- If Q4_K_XL fails the fidelity bar tonight, the fallback is UD-Q5_K_XL over all five IB-side cards. It holds only about
  4 streams at depth (ESTIMATE), so it suits a few agents, not a fleet.
- Q6_K_XL and Q8_0 do not fit the IB side. Q6_K_XL fits only with snoopy's 3090 Ti, and snoopy is a household machine.
- Hops are not what limits these plans; they cost 1-4 % of a step. Three things do, in this order:
  1. the stages run one after another (serial);
  2. there is only enough memory for that many streams at agent depth (capacity);
  3. the MI25 stage is slow whenever the MI25 is in the plan.

## Replica gate (added 2026-11-11 after the S1-S4 runs; supersedes the ranking below)

The ranking below sorts plans by raw useful tok/s and never asks whether the same cards would serve more running apart.
They would. A llama.cpp RPC split runs its stages one after another, so it buys room (a model no single card holds), not
throughput. S2 measured it: two B70s pipelined ran Qwen3.8-27B at 82.9 tok/s at B=8, one B70 alone at 84.1 (MEASURED).

- **`replicaBaseline` / `replicaGate`** (`src/model.js`) price the other use of the same cards. Each card runs its own
  replica of the best single-card recipe at the split model's coding rung or higher (`models.*.coding_rung`, the owner's
  coding-floor ladder of 2026-10-27, kept in our lab notes; `settings.replica_floor` fixes a rung instead). Each card serves a agents
  (`settings.replica_agents_per_card`), and the split serves the same total, N × a.
- A single-card recipe is one of three kinds:
  - a modelled one-card plan (VRAM only, the same card curves as the split);
  - a measured one-card anchor;
  - a measured one-stage `data.curves` config. Today that means QFN IQ3_XXS on one B70 with the experts' complement in
    odie's DRAM.
- The verdict:
  - **GAINS**: the split beats the replicas by more than `settings.replica_gain_margin` (15 % when RM priced either side,
    the card-curve hold-out error; 5 % when both sides are measured).
  - **CAPACITY-ONLY**: it does not.
  - **CAPACITY-REQUIRED**: no card holds a recipe at the rung alone.
  - When two DRAM-assisted replicas share one box, the gate also prints the verdict with one DRAM replica on that box.
    Two at once were never measured.
  - The rung is per model, not per store. Suppose the split's tool-argument fidelity (arg_values EARf, the 2026-11-11 fidelity panels) is
    measured, and a replica runs a store measured below it. Then the gate also prints the verdict over replicas at the
    split's fidelity or better.
- **`report.js --multibox`** ranks by the gate: GAINS by gain, then CAPACITY-REQUIRED, then CAPACITY-ONLY by gain. Then it
  gives every measured split the same verdict: the card_curves anchors and the 2×B70 XeStrata configs in `data.curves`.

Verdicts on 2026-11-11 data (aggregate decode; `node report.js --multibox`):
- Every QFN llama.cpp RPC candidate is **CAPACITY-ONLY**. The best, UD-IQ4_XS over 4 cards, reaches 125 tok/s (ESTIMATE).
  QFN IQ3_XXS on each B70 alone reaches 167 (2 × 83.6, MEASURED one card at a time). With one DRAM replica on odie
  (83.6), IQ4_XS over 4 cards and Q4_K_XL over 6 cards would read GAINS (+50 % and +31 %; the 6-card point is UNCALIBRATED).
- The measured S4 (QFN Q4_K_XL over RTX 6000 + 2×B70, 117.8 at B=16) is **CAPACITY-ONLY** at −30 %. With one DRAM
  replica on odie it reads GAINS. S2, S2-split, S1-split, S2a-3card and S3 are all **CAPACITY-ONLY**. S2a (27B over
  lassie's 1080 Ti + MI25) is **CAPACITY-REQUIRED**: neither card holds the 27B alone.
- XeStrata's in-box 2×B70 layer split is the one split that gains. 105 of its 123 measured configs read **GAINS**, up to
  378 tok/s against 167, on the same IQ3_XXS store as the replicas.
- At the split's own fidelity, every QFN plan with a measured fidelity (all but UD-IQ4_XS) reads **CAPACITY-REQUIRED**. Q4_K_XL scores EARf 0.955 and Q5/Q6
  0.966/0.968; the B70 replicas run GSQ IQ3_XXS at 0.894 (MEASURED, fidelity panels 2-3). No card runs QFN alone at
  Q4_K_XL fidelity or better in any measured run. So these splits trade throughput for tool-argument fidelity; they gain
  no throughput.

Caveats:
- A measured replica ran at its bench depth (agent-length prompts), while a candidate split is priced at depth 56,445.
  That favours the replica. The measured splits compare like with like.
- The baseline is a floor. The RTX 6000, 3090 Ti and lassie cards have no measured QFN run with DRAM, so they add 0 at the
  QFN rung.
- No B70, RTX 6000 or 3090 Ti has a measured single-card run of QFN Q4_K_XL with its experts in DRAM. That is the run that
  would settle the fidelity-matched verdict.

## Tags

Every number below carries one of these tags.

| tag | meaning |
|---|---|
| MEASURED | taken from a run, with a source line in data/ |
| TRACED | read from engine code |
| READ | read from a file header or an API |
| INFERRED | follows from MEASURED or TRACED facts by a rule we state |
| ESTIMATE | a model output, or a borrowed rate |

Every RM output is an ESTIMATE. The RM inputs carry their own tags in `data/multibox.json`, `data/interconnects.json`
`hop_rtt` and `data.bench.gpu`.

## What RM gained (code: `src/model.js`, branch `multibox`)

- **`pipelinePlan(data, spec)`** places a model's layers over cards on several boxes, in stage order, with the client's
  cards last. It splits by free memory (llama.cpp's `-ts` default) or by measured decode rate. It refuses a plan when:
  - a stage's weights plus the KV and state of the plan's streams pass its card;
  - a remote stage's link to the client has no measured round trip in `data/interconnects.json` `hop_rtt`;
  - the last stage (the LM head) is not on the client. llama.cpp RPC puts the head on the last device, and a remote head
    would ship n_vocab × B × 4 bytes of logits every step.
- **`pipelineRoof(data, plan, B)`** prices the decode step:
  - Each stage reads its trunk share, the distinct experts of B streams on its layers (E(1-(1-k/E)^B)), the head if it has
    it, and every stream's KV and recurrent state. It reads at its card's measured llama.cpp backend line
    (`data.bench.gpu`).
  - That read is weighed against compute at the card's int8 ceiling × an eta. The stage time is the larger of the two,
    plus the line's per-step intercept and per-MoE-layer cost.
  - Each remote stage costs one blocking round trip that carries the hidden stream: B × 10,240 × 4 bytes (TRACED: QFN's
    residual is 2,560 × 4 hyper-connections, f32).
  - The `serial` mode is upstream llama.cpp: step = Σ stages + Σ hops. The `ring` mode is the overlapped arm (K = number of
    stages microbatches in flight, as in prima.cpp's piped ring). Upstream llama.cpp does not ship it.
- **`pipelinePrefill`**: per 512-token ubatch, every stage runs in turn (RPC copies block), plus the hops at the ubatch
  payload.
- **`pipelineLandscape`**: decode at every B, serial and ring, with the hop share and the slowest stage. It also runs the
  agent duty cycle at the plan's own rates (below).
- **`agentLandscape`**, the duty cycle (asked for by the R8 sweep's B=1 miss):
  - B real agents replay trace `agent-long-16`, with tool gaps and fresh tokens per turn. Decode is priced at the running
    number of requests, capped at the server's slots.
  - That gives B_eff (the mean number decoding at once), duty = B_eff / B, and F(B, cache).
  - F(B, cache) = the trace's prompt growth + the server's one-agent re-read / B + a whole-prompt re-read for the evicted
    share (1 - C/B) once B passes the conversations the cache holds (C).
- **`rooflinePredict` phases `agents` and `pipeline`.** These freeze a duty-cycle record and a multi-box record before a
  run. `rooflineScore` scores them on useful tok/s and on aggregate decode.
- **`report.js --multibox`** prints every candidate in `data/multibox.json`: its plan, stages, B landscape, agent rows and
  best useful rate.
- **Data added:**
  - `data/multibox.json`: the QFN geometry, five unsloth stores with byte counts READ from the GGUF headers, settings and
    candidates.
  - `data/interconnects.json` `hop_rtt`: the measured round trips (08:55Z, links A, B and C, plus the LAN).
- **Tests:** `tests/test_multibox.js`.

## Inputs, and where each comes from

- **QFN geometry** (TRACED; `data/multibox.json` models.fnx):
  - 48 layers, 512 experts, top-10;
  - 12 full-attention layers with 2 KV heads × (256 + 256), plus a 128-wide indexer key;
  - recurrent state 117,669,888 B per stream;
  - about 6.0 B active parameters per token, plus a 0.636 B head.
- **Store bytes** (READ, HF API plus GGUF headers): experts + trunk + head on VRAM. PLE lookups stay in host RAM.

| store | on VRAM GB | of which experts GB (bpw) |
|---|---|---|
| UD-IQ4_XS | 64.5 | 60.4 (4.00; 18 GB of it IQ3_XXS) |
| UD-Q4_K_XL | 81.8 | 77.0 (5.10) |
| UD-Q5_K_XL | 103.2 | 98.4 (6.51) |
| UD-Q6_K_XL | 114.1 | 109.2 (7.24) |
| Q8_0 | 133.1 | 128.3 (8.5) |

  unsloth ships no plain Q4_K_M, Q5_K_M or Q6_K for QFN. The UD-*_K_XL files are the K-quant stores of those sizes. The
  640-row down tensors cannot take K-quants.
- **Free VRAM per card:** vram_gib - 1.5 GiB reserve (ESTIMATE) - the other services the plan keeps.
  - odie b70_2 carries qwenlocal (26.35 GiB) today. Every plan displaces it; per homelab lab notes, B70 #2 is a test card.
  - droopy's RTX 6000 also hosts the household assistant and the embedder (lab notes). Every plan that uses the RTX 6000 moves them; the plan
    has to say where they go.
- **Card decode lines** (`data.bench.gpu`, llama.cpp, MEASURED):

| backend | card | rate GB/s | host ms |
|---|---|---|---|
| CUDA | RTX 6000 | 486 | 1.75 |
| SYCL | B70 | 586 | 2.84 |
| Vulkan RADV | MI25 | 197 | 1.54 |

  The 1080 Ti borrows the CUDA line's eta × its nameplate: 350 GB/s (ESTIMATE). So does the 3090 Ti: 729 GB/s (ESTIMATE).
- **Hops** (MEASURED 08:55Z, python ping-pong p50, 64 B payload):

| link | ends | round trip at 64 B |
|---|---|---|
| A | lassie-droopy | 92.5 us |
| B | lassie-odie | 47.5 us |
| C | droopy-odie | 60.7 us |
| LAN | snoopy over 10G | 78 us |

  - Payload rate: single-stream TCP, 19.9-22.6 Gb/s on IB-side Ethernet (MEASURED); 9.41 Gb/s on the LAN.
  - llama.cpp RPC is one TCP connection per server. The python ping-pong curve (0.6 GB/s past 20 KB, from python copies) is
    the upper-bound arm, `hop_model: pingpong`.
- **Engine:** upstream llama.cpp f3b407c with RPC (sibling verdict, homelab `b70-rig/multibox/README.md`, 08:45Z). It is
  the only engine that spans SYCL, CUDA sm75/sm61 and Vulkan. ik_llama.cpp has no SYCL. vLLM cannot mix XPU and CUDA.
- **Prefill eta per backend:** SYCL 0.25, CUDA 0.35, Vulkan 0.2 (ESTIMATE). Prefill is the least grounded number here.
- **Agent workload:** trace `agent-long-16` (READ) with the `llamacpp-sycl-slots-swa` re-read row (MEASURED, 556 tokens a
  turn at 1 agent). llama-server keeps a conversation only in its slot, so cache = slots = the plan's capacity (INFERRED).

## Per-hop bytes and time (decode, serial)

The hidden stream per stream is 10,240 × 4 = 40,960 B each way (TRACED). The table gives the cost of one round trip
(ESTIMATE, stream model).

| B | payload | odie↔lassie (B) | odie↔droopy (C) | odie B70→B70 (PCIe) |
|---|---|---|---|---|
| 1 | 41 KB | 76 us | 90 us | 2 us |
| 8 | 328 KB | 279 us | 296 us | 13 us |

A step at B=8 and 56k depth takes 52 ms for the top plan (ESTIMATE). Its two RPC hops take 0.58 ms of that, about 1.1 %.
**IB bandwidth is not what limits these plans.** RoCE (1.04 us one way, MEASURED) would save under 0.2 ms a step. That is
why llama.cpp RPC's TCP-only path is acceptable for now.

## Ranking (decode, llama.cpp RPC serial, KV q8_0, agent depth 56,445 tokens = the trace mean)

| # | plan | fits? | capacity at depth | B=1 tok/s | aggregate B=8 / 16 / 32 (serial) | ring B=32 | prefill tok/s | best useful (agents) | fidelity |
|---|---|---|---|---|---|---|---|---|---|
| - | **UD-IQ4_XS** over 1080 Ti + RTX 6000 + 2×B70 | yes | 34 | 59.2 | 174 / 207 / 240 | 547 | 2,878 | 152 at B 16 | **likely fails**: 18 GB of its experts are IQ3_XXS, and every 3-bit QFN file failed the 0.98 bar (lab notes, MEASURED: best EARf 0.925) |
| 1 | **UD-Q4_K_XL over 1080 Ti + RTX 6000 + 2×B70** (frozen) | yes | 15 | 53.2 | 154 / 184 / 214 | 483 | 2,491 | 97.5 at B 8 | UNKNOWN (Q4 arm runs tonight). Floor: Q8 on ik vs upstream is EARf 0.975 (MEASURED) |
| 2 | UD-Q4_K_XL over all 6 cards (+ snoopy over the LAN) | yes | 43 | 48.3 | 133 / 157 / 181 | 470 | 1,333 | 90.2 at B 16 | as #1 |
| 3 | UD-Q4_K_XL over all 5 IB-side cards (+ MI25) | yes | 26 | 46.4 | 128 / 151 / 174 | 423 | 1,363 | 86.0 at B 16 | as #1 |
| 4 | UD-Q4_K_XL over RTX 6000 + 2×B70 | yes | 4 | 55.1 | (163 / 195 past capacity) | - | 3,092 | 81.3 at B 4 | as #1 |
| 5 | UD-Q5_K_XL over all 6 cards | yes | 21 | 46.2 | 119 / 139 / 161 | 420 | 1,239 | 78.4 at B 8 | UNKNOWN (Q5 arm tonight); expected above Q4 |
| 6 | **UD-Q5_K_XL over all 5 IB-side cards** (frozen) | yes | 4 | 44.4 | (114 / 134 past capacity) | - | 1,264 | 57.6 at B 4 | as #5 |
| 7 | UD-Q6_K_XL over all 6 cards | yes | 9 | 45.2 | 113 / (131 / 152 past capacity) | - | 1,196 | 67.2 at B 8 | UNKNOWN (Q6 arm tonight) |
| - | UD-Q6_K_XL over the 5 IB-side cards | **REFUSED** | - | - | - | - | - | - | the MI25 and both B70 stages pass their cards |
| - | Q8_0 over all 6 cards | **REFUSED** | - | - | - | - | - | - | four stages pass their cards |
| ref | R8: XeStrata GSQ IQ3_XXS on 2×B70 (served today) | yes | - | 70.6 decode, 30.25 useful (MEASURED) | useful 61.0 at B 8 (MEASURED 09:11Z) | - | - | - | **fails**: 3-bit, EARf 0.894 (MEASURED) |

**How the table is ranked.** Fidelity gates first: a plan that cannot clear the 0.98 tool-argument bar is not a coding
seat, at any speed. Within the plans whose fidelity is still open, the best useful rate ranks. That is why IQ4_XS, the
fastest row, is not ranked, and why the fallback is the highest-fidelity store that still fits.

**Why each plan lands where it does** (ESTIMATE):
- **The slowest stage.** At B=1 every plan is bound by odie b70_1's read: it holds the head and the most layers. From B=8,
  whenever the MI25 is in a plan, its Vulkan line (197 GB/s) makes it the slowest stage.
- **The MI25 trade.** Adding it buys capacity (15 → 26 streams for Q4) but costs prefill. Its prefill stage takes 182 ms per
  512 tokens, against 33-53 ms for each other card, so Q4 prefill drops from 2,491 to 1,363 tok/s. Prefill is roughly half
  of an agent's server time, so the 5-card plan loses at B ≤ 8 and only catches up at B=16.
- **The RTX 6000 trade.** The 3-card plan has the fastest prefill, but only 4 streams fit at agent depth. Adding the
  1080 Ti frees about 10 GB of KV room on the other cards (capacity 4 → 15).
- **Capacity sets the useful peak.** Past capacity, llama-server queues requests and evicts conversations, so every
  evicted turn re-reads its whole prompt: 56k tokens (the F(B, cache) term). That is why useful tok/s collapses at
  B ≈ capacity + 8 in every row. A unified KV pool (`-kvu`) holding actual depths, instead of the mean, would soften the
  edge. It is not priced.
- **Serial vs ring.** The ring arm doubles the aggregate at B=16-32 (361 vs 184 for #1). That is the size of the prize
  for an overlapped RPC, which upstream lacks: a candidate engine change. It is not a recipe tonight.

## The B landscape of the top plan (#1, UD-Q4_K_XL, 4 cards; ESTIMATE)

| B | per stream (synthetic, 56k depth) | aggregate serial | aggregate ring | hops | agents: useful tok/s | B_eff | slowest stage |
|---|---|---|---|---|---|---|---|
| 1 | 53.2 | 53 | 53 | 0.9 % | 34.9 | 0.7 | odie b70_1 (read) |
| 2 | - | - | - | - | 57.9 | 1.4 | |
| 4 | - | - | - | - | 77.3 | 2.4 | |
| 8 | 19.2 | 154 | 243 | 1.1 % | **97.5** | 5.0 | odie b70_1 (read) |
| 16 | 11.5 (past capacity) | 184 | 361 | 1.2 % | 70.4 | 10.9 | capacity: queue + evictions |
| 32 | 6.7 (past capacity) | 214 | 483 | 1.4 % | 4.8 | 5.6 | evictions dominate |

Stage times at B=8, 56k depth (ms; read / compute / fixed):

| stage | time ms | read | compute | fixed |
|---|---|---|---|---|
| lassie 1080 Ti, 5 layers | 7.4 | 7.1 | 0.6 | 0.3 |
| droopy RTX 6000, 11 layers | 12.0 | 11.3 | 0.2 | 0.7 |
| odie b70_2, 16 layers | 15.4 | 13.6 | 0.5 | 1.8 |
| odie b70_1, 16 layers + head | 16.6 | 14.8 | 0.6 | 1.8 |

Hops add 0.6 ms, for a step of 52 ms. Every stage is read-bound. Compute stays under 5 % of each stage.

## The two frozen recipes (written 09:14:19Z, before any multi-box run)

Both use upstream llama.cpp f3b407c plus homelab multibox patch 0001 (RPC) with: `-ngl` all, `-sm layer`, `-ts` by free
memory, `-ctk`/`-ctv` q8_0, `-ub 512`, MTP off, and the client holding the last stage. The predicted numbers are ESTIMATEs;
the band is ±25 %.

### 1. `mb-qfn-udq4kxl-4card-b8-d4k-1111`

QFN UD-Q4_K_XL. Stages:

| stage | layers | role |
|---|---|---|
| lassie 1080 Ti | 5 | rpc-server |
| droopy RTX 6000 | 11 | rpc-server |
| odie b70_2 | 16 | client |
| odie b70_1 | 16 + head | client |

Predictions:
- Aggregate decode at B=8 with ~4k-token depth: **198.7 tok/s**. The primary number; tested term `stage_read`.
- Single stream: 57.7 tok/s.
- Ring arm: 272.
- Hop share: 1.5 %.
- Prefill: 2,491 tok/s over 2,048 tokens.
- Capacity: 15 streams at 56k depth.
- Agent replay useful tok/s: 34.9 at 1, 77.3 at 4, 97.5 at 8, 70.4 at 16.

### 2. `mb-qfn-udq5kxl-5card-b4-d4k-1111`

QFN UD-Q5_K_XL. Stages:

| stage | layers | role |
|---|---|---|
| lassie 1080 Ti | 4 | rpc-server |
| lassie MI25 | 7 | rpc-server, Vulkan RADV |
| droopy RTX 6000 | 9 | rpc-server |
| odie b70_2 | 14 | client |
| odie b70_1 | 14 + head | client |

Predictions:
- Aggregate at B=4, ~4k depth: **110.7 tok/s** (tested term `stage_read`; the MI25 stage is slowest).
- Single stream: 48.3.
- Prefill: 1,264.
- Capacity: 4 at 56k.
- Agent useful tok/s: 28.9 at 1, 57.6 at 4.

To score a run: `node report.js --roofline --score <id> --measured <tok/s> --measured-utc <UTC> --run <what> --write`.

## What tonight's R8 sweep says about these numbers

(Correction 2026-11-11: the useful tok/s of the R8 points below counted turns that ended after the window; recounted
numbers and the rescore: "Correction 2026-11-11" at the end of the per-card batch curve section. The numbers here stay
as reported.)

The R8 agent-cycle predictions (`r8-agents-b{8,16,24,33,40}-1111`) were frozen at 09:00:35Z, before the B=8 window opened
at 09:00:53Z.

| quantity at B=8 | predicted | measured (09:11Z) | verdict |
|---|---|---|---|
| useful tok/s | 91.8 | 61.03 | **FALSIFIED**, -33.5 %, outside the ±20 % band |
| B_eff | 4.92 | 5.04 (decode_ms 3,021 s / 600 s) | holds |
| F | 3.94 | 4.67 (split), 4.98 (server) | holds |
| decode at B_eff, per stream | 30.8 (the served-build synthetic curve) | 14.0 (42,284 tokens / 3,021 s), 12.1 client median | **the term that missed** |

At B=16 (09:31Z, `r8-agents-b16-1111`): useful 69.35 vs predicted 89.4, -22.4 %, FALSIFIED. F 4.61 vs 4.86 holds
(cc_evictions 0). B_eff 14.17 (decode_ms 8,500.6 s / 600 s) vs predicted 10.55: slower decode keeps more requests decoding
at once. Decode per stream 5.66 tok/s (48,105 tokens / 8,500.6 s), so about 80 tok/s aggregate while 14 decode, against
the synthetic curve's ~255 at 14 (ESTIMATE from log-log between its B 1 and B 16 points): 3.2x off, with 44.7 % of the
window stalled by prefill.

So the duty cycle and F(B, cache) terms hold, and the miss is decode at B_eff, which widens with B. Two candidate causes, neither yet
separated:
- the synthetic curve was measured on image 06103b88, not the served 0de528f72871;
- 37.8 % of the window was stalled by prefill ubatches inside decode steps.

**What this means for the multi-box numbers.** Their decode also comes from a B=1-anchored line plus the
distinct-expert read model. Batched MoE decode on llama.cpp SYCL and Vulkan is unmeasured: llama.cpp leaves the MoE
mat-vec path above 8 tokens on Vulkan/HIP. So the `stage_read` term at B=8 is the one most likely to miss.

**The first multi-box run should measure:**
- synthetic decode at B=1 and B=8 on the plan (the frozen primary);
- one card's llama.cpp decode at B=8, alone, to separate the line from the pipeline.

## The served engine, turn by turn (branch `evict`)

The R8 sweep fell off a cliff at 33 agents on 32 slots (79.1 tok/s at B=24, 9.7 at 33, 8.0 at 40). The fluid agent cycle
above priced a point past the slots as a queue that batches, so it could not see that. `agentEngineSim` (src/model.js)
runs the served engine turn by turn instead, with three terms, none fitted:

- **Slot eviction.** A request takes its own conversation's slot, else an empty one, else the slot the served frontend
  picks (`slot_groups` 2: XeStrata's `pick_slot` from the served image, `serve/server.py` sha256 c5bcb67e: a hole below
  a busy slot of its group, then the least busy group's least recently used slot). An evicted conversation re-reads its
  whole prompt less the 4,924 tokens every slot shares. The evicted share comes out of S, B and every agent's cycle.
- **Reads block decode.** Reads run one at a time at P. While a read over `read_batch_max` (512, XeStrata
  `STRATA_BATCH_PREFILL_MAX_TOKENS`) runs, no slot decodes.
- **Admission, declared.** `opts.admission` is `engine` (the engine's own) or `door` (at most N conversations hold a
  lease; the rest wait). A landscape point with B > slots and no admission declared is REFUSED.

The duty cycle was already wired in (`agentLandscape`). Below, the six R8 points are rescored with the new terms: the
decode curve and prefill are the ones the `r8-agents-*-1111` records froze, admission is `engine`, and nothing is refitted.
Measured / predicted:

| B | measured useful | old frozen (err) | new (err) | B_eff | F | stall % | read busy % | evicted share (new) |
|---|---|---|---|---|---|---|---|---|
| 1 | 30.25 | - | 34.2 (-11.6 %) | 0.43 / 0.5 | 10.77 / 7.3 | 0 / 0.0 | 41.2 / 29.0 | 0.0 |
| 8 | 61.03 | 91.8 (-33.5 %) | 91.4 (-33.2 %) | 5.04 / 5.8 | 4.98 / 4.2 | 37.8 / 41.0 | 44.4 / 45.0 | 0.0 |
| 16 | 69.35 | 89.4 (-22.4 %) | 95.8 (-27.6 %) | 14.17 / 13.8 | 5.37 / 5.0 | 44.7 / 55.6 | 48.6 / 59.1 | 0.0 |
| 24 | 79.09 | 72.4 (9.3 %) | 73.6 (7.4 %) | 21.58 / 21.7 | 7.52 / 7.2 | 57.2 / 70.6 | 69.9 / 73.4 | 0.0 |
| 33 | 9.66 | 33.4 (-71.1 %) | 1.0 (836 %) | 15.21 / 30.2 | 111.31 / 165 | 78.8 / 96.5 | 116.7 / 96.5 | 1.0 |
| 40 | 7.99 | 21.0 (-62.0 %) | 0.5 (1604 %) | 9.26 / 30.2 | 134.97 / 211 | 81.3 / 97.1 | 111.4 / 97.1 | 1.0 |

err = measured / predicted - 1. What the new terms get right: the cliff now sits at 33 (the old record predicted 33.4).
B_eff at 8, 16 and 24 lands within 15 % (5.8 / 13.8 / 21.7 against 5.04 / 14.17 / 21.58). The stall share runs 3-14
points high (41 / 56 / 71 % against 38 / 45 / 57 %). What it still misses:
- **Useful at 8 and 16, by 27-33 %.** The frozen synthetic decode curve is ~1.7x too fast at the served depth: aggregate
  decode while 14 decode is ~80 tok/s measured. Blocking reads explain only part of the gap.
- **The cliff is too deep.** At 33 the model gives 1.0 tok/s against 9.7 measured (40: 0.5 against 8.0), with every
  returning turn evicted against 43-52 % measured. Once one read runs, the model's engine admits no decode at all. The
  served engine seems to keep some decode going under a long read, or its reads overlap across the two batch groups
  (prefill busy measured 117 % > 100 %). Not separated.
- **B=1, -11.6 %.** F is 7.3 against 10.8: the 1-agent re-read the model dilutes over B is not modelled per turn.

The door run that tests the cure was frozen in `r8-door32-b{40,33}-1111` (11:28:19Z) and ran 11:47-12:28Z. Measured,
holding the agents to 32 conversations: **85.3 tok/s at B=40** (frozen 37.8, FALSIFIED +126 %) and **79.0 at B=33**
(frozen 30.2, +162 %), against 8.0 and 9.7 with no door. Re-read per output token was 0.02 at both points, so no
evictions, as the model said. The useful rate is what the model got wrong: the measured run is close to its steady
state (the 1200 s warm-up variant gave 100.8 and 93.9). The causes are INFERRED, none refitted:
- **Cold first turns.** The run's cold first turns had 4,924 cached tokens each and were done by ~450 s, where the model
  gave them no shared-prefix credit and read them at P for ~554 s.
- **No decode under a read.** The model stops every decode for the length of each read, which keeps the 32 agents in
  lock step through the warm-up.

The cure holds; the transient term is the next miss to fix.

## The per-card batch curve (branch `cardcurve`)

The multi-box runs falsified the bandwidth-line batched step: it was 2.1-3.4x optimistic at B=8 (S4a froze 198.7 against
87.9 measured). The step per stage is now a curve per card, backend and kernel class (`cardStage`, data in
`data/multibox.json` `card_curves`). Its regimes are TRACED from llama.cpp f3b407c, not fitted:

- **Mat-vec (B <= 8 on every backend).** `MMVQ_MAX_BATCH_SIZE` 8 (SYCL `common.hpp`, CUDA `mmvq.cuh`). Dense:
  `max(W / R, b * W * col)`. MoE `MUL_MAT_ID` reads B x k expert columns while the backend's id mat-vec holds: CUDA
  8 tokens (Turing, Q4_K/Q5_K; 5 on Pascal), Vulkan 8, SYCL **1** (`ggml_sycl_mul_mat_id_mmvq_eligible`: one token).
- **SYCL dense past 8: DERIVED, no constant.** `ggml_sycl_supports_mmq` returns false, so the quantized mat-mul
  dequantizes to f16 and runs a GEMM: `max((W + 4 P) / R, 2 P b / (fp16 peak x prefill eta))`. W + 4P is the read of
  the quantized weights plus the f16 write and read-back. INFERRED (CL 60): the reorder XMX path is not taken for
  UD-Q4_K_XL's mixed types.
- **MoE past the mat-vec limit.** The distinct experts `E (1 - (1 - k / E)^b)` go through the batched kernel: SYCL
  grouped dequant GEMM (`grp`), CUDA MMQ (`mmq`), each priced `bytes x max(1 / R, const)`.
- **Fixed per step.** The host time share, `moe_layer_ms`, and `overhead.<model>` x layers (graph launches beyond the
  line's intercept).
- **A missing constant** prices the stage at its bandwidth line and marks it UNCALIBRATED. A pipeline record at B > 1
  with an UNCALIBRATED stage is REFUSED; so is a decode record at batch > 1 with no measured `curves` entry, and a
  recipe's `phase_bound` at agents > 1 with no batch curve prints UNCALIBRATED and makes no claim.

**Calibration.** Each constant is solved by bisection from ONE named point, in anchor order (`cardCurveFit`); every
other measured point is a hold-out (`node report.js --cardcurve`):

| constant | value | calibration point |
|---|---|---|
| cuda/gtx1080ti.rate | 298.3 GB/s | S1 lassie alone, B=1 (123.3) |
| cuda/gtx1080ti.col | 0.681 ms/GB/col | S1 lassie alone, B=8 (593) |
| cuda/gtx1080ti.mmq | 7.26 ms/GB | S1 lassie alone, B=16 (861) |
| cuda/rtx6000.col | 0.409 | S1 droopy alone, B=8 (906) |
| cuda/rtx6000.mmq | 4.79 | S1 droopy alone, B=16 (1260) |
| overhead.q38_27b | 0.186 ms/layer | S2 one B70, B=1 (22.2) |
| sycl/b70.col | 0.566 | S2 one B70, B=8 (84.1) |
| vulkan-radv/mi25.col | 4.30 | S2a lassie local, B=8 (22.1) |
| overhead.fnx | 0.255 ms/layer | S4b server, B=1 (34.6) |
| sycl/b70.grp | 4.53 ms/GB | S4b bench, B=8 (93.2) |

**Hold-outs** (aggregate tok/s, measured / predicted (err) / bandwidth line (err)):

| point | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| S2 one B70 | cal | - | cal | 60.2 / 69.5 (+15 %) / 73.3 (+22 %), DERIVED |
| S2 2x B70 [33,32] | 22.3 / 22.2 (-1 %) | - | 82.9 / 84.1 (+1 %) / 240 | 58.9 / 69.5 (+18 %), DERIVED |
| S2 split [16,49] | 20.7 / 19.2 (-7 %) | - | 77.1 / 80.7 (+5 %) / 209 | 78.8 / 76.5 (-3 %) |
| S1 split 1.7B [14,14] | 122 / 143 (+17 %) | - | 617 / 708 (+15 %) / 1040 | 912 / 1010 (+11 %) / 1769 |
| S2a lassie local | 9.7 / 11.4 (+17 %) | - | cal | - |
| S2a 3 card | 10.1 / 12.6 (+25 %) | - | 24.1 / 28.9 (+20 %) / 125 | - |
| S3 bench | 26.8 / 30.8 (+15 %) | - | 58.5 / 66.2 (+13 %) / 165 | 61.1 / 96.9 UNCALIBRATED (MI25 mul_mm) |
| S3 bench 4k | 25.6 / 30.6 (+20 %) | - | 57.6 / 65.6 (+14 %) / 162 | - |
| S4a bench | 23.0 / 33.6 (+46 %)* | 61.6 / 75.6 (+23 %) | 87.9 / 88.4 (+1 %) / 201 (frozen 198.7) | 111 / 98.9 (-11 %) |
| S4b bench | 23.3 / 34.6 (+49 %)* | 63.7 / 75.9 (+19 %) | cal | 118 / 103 (-12 %) / 197 |
| S4b server | cal | 58.4 / 75.9 (+30 %) | 72.5 / 93.2 (+29 %) | 90.2 / 103 (+15 %) |

\* The bench ran its B=1 row after the B=8/16 passes; the fresh-server smoke gives 34.6. Scored, flagged.

Hold-out summary: n 27, median |err| 14.8 % (B=1: 17.4 %, n 9; B > 1: 14.2 %, n 18), 22 of 27 within 25 %. The
bandwidth line's median |err| on the same B > 1 points is over 100 %. The curve is still optimistic by a median +14.8 %;
named, not modelled: the RPC per-step tensor traffic (S1 split runs 11-17 % below), the server-vs-bench gap on the same
layout (S4b B=8: 72.5 vs 93.2), and the 27B's recurrent state (UNKNOWN, priced 0).

**The served engine, two terms fixed** (`agentEngineSim`):
- **Each slot group reads its own prompt at once** (read busy measured 111-117 % at B=33/40). A long read still stops
  every decoding slot (TRACED: "every decoding slot waits"; the measured stall/busy ratio 0.79-0.85 says all, not one
  group, which would give ~0.5). `read_blocks: 'group'` is the arm.
- **A cold first turn reads its prompt less the shared prefix** once any read has put that prefix in the cache.

R8 rescored, useful tok/s (`agentPointRescore`). The decode basis is the frozen synthetic curve; one factor
`decode_step_increment` lambda scales only its batched step increment (the one-agent point matches the served one, 68.8 vs
70.6), solved on ONE named point, no-door B=8:

| point | measured | evict branch (err) | sim fixed, no refit (err) | lambda 2.77 (err) | B_eff pred / meas | stall % pred / meas |
|---|---|---|---|---|---|---|
| no door B=1 | 30.3 | 34.2 (+13 %) | 34.2 (+13 %) | 34.2 (+13 %) | 0.5 / 0.4 | 0 / 0 |
| no door B=8 | 61.0 | 91.4 (+50 %) | 95.4 (+56 %) | cal | 5.8 / 5.0 | 42 / 38 |
| no door B=16 | 69.3 | 95.8 (+38 %) | 117 (+69 %) | 54.0 (-22 %) | 13.8 / 14.2 | 48 / 45 |
| no door B=24 | 79.1 | 73.6 (-7 %) | 133 (+68 %) | 67.1 (-15 %) | 20.9 / 21.6 | 49 / 57 |
| no door B=33 | 9.7 | 1.0 (-90 %) | 1.3 (-87 %) | 1.1 (-88 %) | 30.5 / 15.2 | 98 / 79 |
| no door B=40 | 8.0 | 0.5 (-94 %) | 0.2 (-98 %) | 0.4 (-95 %) | 30.4 / 9.3 | 97 / 81 |
| door 32, B=33 | 79.0 | 30.2 (-62 %) | 143 (+82 %) | 69.4 (-12 %) | 28.9 / 30.6 | 49 / 64 |
| door 32, B=40 | 85.3 | 37.8 (-56 %) | 128 (+50 %) | 67.9 (-20 %) | 28.7 / 31.9 | 53 / 64 |

Here err = predicted / measured - 1. Median |err|: evict branch 52.7 %, the fixed sim with no refit 68.4 %, the lambda
arm on its 7 hold-outs 20.3 %. INFERRED: with the sim terms fixed, B_eff and stall now match the measured runs, so the
remaining miss is the served build's decode rate at depth; the frozen synthetic curve's batched step
increment is ~2.8x too small for it. The served curve, measured once at depth, replaces lambda. Still missed: the no-door cliff at 33/40 is too deep
(every returning turn evicted against 43-52 % measured; B_eff 30 against 9-15), and B=1 (F 7.1 against 10.8: the one-agent
re-read).

### Correction 2026-11-11: the R8 points recounted inside the window

The R8 sweep and r8-admission counted turns that ended after the measurement window closed (homelab conv-door-live,
f3701cabc59, TRACED): R8 answers `Connection: close`, so `http.client` drops the socket after `getresponse()`; `Run.cut()`
then cannot shut a running stream, the 5 s joins let it finish, and `report()` counted any turn with an end and no error.
Recounted from each point's `replay.json` (MEASURED; only turns sent inside the window that ended by its end), the tables
above keep the as-reported numbers; `data/multibox.json` now carries the recount in `measured.useful_tps` and the old
number in `measured.as_reported` (never used). Each point names its rule, `measured.turn_counted_if: ended_by_window_end`;
a point without it prints UNVERIFIED-COUNT, and so does a lambda solved on one.

lambda re-solved on the recounted no-door B=8 (57.54, was 61.03): **2.7661, unchanged**. The sim counts whole turns, so
useful(lambda) at B=8 is a staircase: 61.03 just below lambda 2.7661, 56.82 just above. The recounted 57.54 sits inside
that jump, so the solve lands on the same edge and takes the nearer side, 56.82 (calibration residual -1.3 %). Any
measured value from 56.82 to 61.03 gives this lambda. The hold-out predictions do not move; their errors do, because the
measured values moved. The three conv-door-live points are new hold-outs: the lane arms with the served door's 60 s idle
release, and RM has no thinking-cap term, so all three share the client door's 67.9.

| point | as reported | recounted | late turns (max s past) | no refit (err) | lambda 2.7661 (err) | use |
|---|---|---|---|---|---|---|
| no door B=1 | 30.25 | 30.25 | 0 | 34.2 (+13 %) | 34.2 (+13 %) | hold-out |
| no door B=8 | 61.03 | 57.54 | 6 (17.5) | 95.4 (+66 %) | 56.8 (-1.3 %) | calibrates |
| no door B=16 | 69.35 | 64.47 | 11 (19.6) | 117 (+82 %) | 54.0 (-16 %) | hold-out |
| no door B=24 | 79.09 | 64.19 | 21 (28.0) | 133 (+107 %) | 67.1 (+4.5 %) | hold-out |
| no door B=33 | 9.66 | 5.68 | 9 (130.9) | 1.3 (-77 %) | 1.1 (-80 %) | hold-out |
| no door B=40 | 7.99 | 3.52 | 12 (149.3) | 0.2 (-95 %) | 0.4 (-89 %) | hold-out |
| door 32, B=33 | 79.01 | 63.02 | 29 (22.8) | 143 (+128 %) | 69.4 (+10 %) | hold-out |
| door 32, B=40 | 85.28 | 66.82 | 30 (41.9) | 128 (+91 %) | 67.9 (+1.7 %) | hold-out |
| lane door, B=40 (new) | 59.48 | 59.48 | 0 | 128 (+115 %) | 67.9 (+14 %) | hold-out |
| lane door nocap, B=40 (new) | 65.45 | 65.45 | 0 | 128 (+95 %) | 67.9 (+3.8 %) | hold-out |
| client door repeat, B=40 (new) | 55.98 | 55.98 | 0 | 128 (+128 %) | 67.9 (+21 %) | hold-out |

err = predicted / measured - 1. lambda arm hold-out median |err|: 13.2 % on the seven old hold-outs (was 20.3 %), 13.7 %
with the three new ones (n 10). The evict branch: 48.6 % (n 9; it has no lane-door value). The no-door cliff is still
too deep (B=33/40 -80 / -89 %). The frozen record scores move with the recount too (`measured.value`; old value and score
kept in `measured.as_reported`): `r8-agents-b8` -33.5 % -> -37.3 %, `r8-agents-b16` -22.4 % -> -27.9 %, `r8-door32-b33`
+162 % -> +109 %, `r8-door32-b40` +126 % -> +77 % (err = measured / predicted - 1 there), all still FALSIFIED.
Not recounted: B_eff, F (server), stall and prefill busy come from server /metrics deltas and client stalls; the
after-snapshot was read after the joins, so the old server deltas include up to the late turns' post-window work
(INFERRED).

### 2026-11-11: the batch serving mode's step term (the GSQ-RCO Coder screen)

The screen served the ISTA GSQ-RCO Coder (IQ1_M) as one XeStrata copy per B70 and replayed agent-long-16 against each
card. Its prediction was frozen before any run: `agentLandscape` with rates, the upstream prefill arm and the lambda
decode arm. The records are `data/predictions-roofline-fnc-coder-*.json`, transcribed unchanged. The no-batch record is
the screen's own arithmetic, labelled as such.

What it missed was the serving mode. With `--batch 4` a lone stream decodes 40.6 tok/s; without `--batch` it decodes
69.3 tok/s on the same card and replay. The curve's one-row point (72.7) stands for the single-session decode.

The new term is `workload.engines.strata.batch_mode_step` (`batchModeStep`). It is MEASURED per store from ONE named
pair: card-b1 (`--batch 4`) against card-b1-nobatch. The value is the ratio of their window decode counters, 69.30 /
40.62 = **1.706**. The single-stream probe agrees per forward step: 43.3-43.6 ms against 26.1 ms.

Served in a batch mode, every forward step pays the fixed extra. So a step of n rows takes n / agg(n) + (m - 1) /
agg(1), and a lone stream steps m times slower. A store with no row is never charged. R8 serves `--batch 32`, and its
served one-agent decode already matches its curve. A curve measured in the batch mode (`decode_curve.serving_mode:
'batch'`) is not charged either. card-b4 and both cards at once (a card at a time) are the hold-outs. both1-nobatch is
left out: the replay client's network failed during it.

| point | measured | frozen prediction (err) | with the term (err) | use |
|---|---|---|---|---|
| card-b1, 1 agent | 24.93 | 36.2 (+45 %) | 27.2 (+9.1 %) | calibrates |
| card-b1-nobatch, 1 agent | 37.06 | 36.2 (-2.3 %) | 36.2 (-2.3 %), not charged | calibrates |
| card-b4, 4 agents | 32.72 | 33.8 (+3.3 %) | 30.8 (-5.8 %) | hold-out |
| both1, card a | 25.52 | 36.2 (+42 %) | 27.2 (+6.6 %) | hold-out |
| both1, card b | 26.34 | 36.2 (+38 %) | 27.2 (+3.3 %) | hold-out |
| both4, card a | 30.47 | 33.8 (+11 %) | 30.8 (+1.2 %) | hold-out |
| both4, card b | 30.60 | 33.8 (+10 %) | 30.8 (+0.7 %) | hold-out |

err = predicted / measured - 1. The hold-out median |err| falls from 10.9 % to 3.3 % (n 5). Both cards, summed:

- both1 goes from 72.4 to 54.4, against 51.86 measured (+40 % to +4.9 %);
- both4 goes from 67.6 to 61.6, against 61.07 (+11 % to +0.9 %).

The guard holds: card-b1 is within 10 % of 24.93, and card-b1-nobatch is within 10 % of 37.06.

**Form.** The fixed extra is charged on every step, not on the lone stream only. The cause is a per-step cost of the
mode. This form also matches the measured per-stream decode at 4 agents (10.6 predicted at B_eff 3.6, against
10.3-10.8 measured), where the lone-only form gives 12.0. The lone-only form was run as well, so this choice saw the
hold-outs: it scores card-b4 at +3.3 % and both4 at +10.4 / +10.9 %.

**Turns per agent-minute still run over.**

- card-b1: 6.2 against 5.0.
- card-b4: 1.82 against 1.5.
- both4: 1.82 against 1.35 and 1.63.

The re-read term is not the cause. The sim reads 177 re-read tokens a turn at 4 agents (R8's one-agent 708 / 4). The
measured card-b4 re-read is 1,819 tokens a turn. Set to that, F matches (11.9 against 11.5 measured), but the predicted
turns rise to 2.25, so it is not fitted.

The output length a turn does explain the miss. The Coder writes 0.68-0.70 of the recorded output a turn, and the sim
uses R8's 0.591 (`turn_model.models.fnx` at one agent). With 0.69, card-b4 reads 1.58 turns (+5 %), but card-b1's useful
rises to 27.8 (+12 %). The two are entangled with the batch-mode prompt read, which is the open term: 759 tok/s with
`--batch` against 1,141 without, and the upstream 1,003 is what the sim uses.

**The batch-mode prompt read is a named open term, not kept.** The same calibration pair reads its prompts at 759 tok/s
with `--batch` and at 1,141 without (server prompt counters over each window): 1,141 / 759 = 1.50. Charging 1.50 on
the sim's read rate moves card-b1 from +9.1 % to -3.6 %. It breaks hold-outs past their errors, though, so it is not
kept:

- on every point served in batch mode: card-b4 -5.8 % -> -15.1 %, both4 +1.2 / +0.7 % -> -8.9 / -9.3 %;
- on lone-agent points only: both1 card b +3.3 % -> -8.8 %.

At 4 agents the batch-mode read is not slower: 1,039 and 1,012-1,125 tok/s measured, against the 1,003 the sim uses.
So the term is not a fixed factor of the mode. Its shape (by read size, by what else the engine holds) needs its own
measurement.

Open:

- a Coder model row with its own output lengths;
- the batch-mode prompt read above;
- the lambda-scaled curve's two-row aggregate (35 tok/s), which is below its one-row point (72.7).

## Larger models (analytic, not ranked)

- **GLM-5.3 Flash:** 304 B expert parameters; the UD-Q3_K_XL routed experts alone are 134.4 GB (READ).
- **DeepSeek V4.1:** 543 B expert parameters; a 197.7 GB store (READ).

Neither fits the 113.6 GB of free IB-side VRAM, or 131.7 GB with snoopy (ESTIMATE). Both need cold experts in CPU RAM,
which is the expert-offload recipe, not a VRAM pipeline. They are out of scope for this plan, and 3-bit QFN already fails
the tool-argument bar.

## Not modelled (named)

- Attention over earlier keys during prefill.
- llama-server's unified KV pool, which holds actual depths instead of the mean.
- A second rpc-server device on lassie: both lassie cards are priced as separate round trips. That is conservative if
  one server hosts both.
- The 1080 Ti and 3090 Ti lines are borrowed CUDA etas (ESTIMATE).
- The prefill etas are ESTIMATEs.
- QSA's top-k key cap: `keys_read: all` is INFERRED (llama.cpp applies QSA as a mask). The `cap` arm raises #1 to 103
  useful at B=8.
