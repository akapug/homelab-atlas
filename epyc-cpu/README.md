# CPU only: 2x AMD EPYC 7601 with 1 TB of DDR4

**Box:** lassie, a two-socket server from 2017 (Supermicro H11DSi): 2x EPYC 7601 (32 Zen 1 cores each, AVX2 only, no
AVX-512), 8 NUMA nodes, 1 TB DDR4-2400 as 16 x 64 GB LRDIMMs on 16 channels, 257.7 GB/s measured NUMA-local read.
The same box holds a [GTX 1080 Ti](../1080-ti/) and an [MI25](../mi25/); this page is what the CPUs do with no GPU.

The question: with no GPU at all, how fast does a big MoE model run from DRAM on second-hand server parts? 1 TB holds
DeepSeek V4.1 Flash with its experts at release precision. The answer turned out to be mostly about **where each
byte sits** among the 8 NUMA nodes, not about how many cores compute.

## At a glance

ik_llama.cpp with our changes, greedy, one request at a time, no GPU, weights in a RAM disk, unless the row says
otherwise. Decode and prefill are tokens per second. Terms: [GLOSSARY](../GLOSSARY.md).

| model | weights | result | conditions | details |
|---|---|---|---|---|
| DeepSeek V4.1 Flash | routed experts at release precision (MXFP4), other tensors Q8_0 | **7.02 decode** with the DSpark draft, **5.16** without; prefill 36.2 | experts and dense weights row-split over all 8 nodes, 64 threads, 8,192-token window, 3 coding-agent prompts x 300 tokens, 2026-10-20, 15 of 16 DIMMs (below) | [moe-offload points 9, 15, 16](../moe-offload/#what-we-learned-measured) |
| Qwen3-30B-A3B | Q4_K_S | **23.81 decode**; 19.26 with pages interleaved | 64 threads, 128 tokens; experts row-split (21.94 in that session), then attention and output head row-split too (21.26 to 23.81 in a second session) | [our fork's PATCHES.md](https://github.com/akapug/ik_llama.cpp/blob/mixed-cuda-vulkan-expert-banks/PATCHES.md) |
| GLM-5.3-Flash | unsloth UD-Q3_K_XL | 4.64 decode, 43.0 prompt | ik_llama.cpp `5f89bfc`, builds running on the same box: a floor, 2026-10-02 | older run |
| GLM-5.3 (744B, 40B active) | IQ4_XS | 2.4 decode, 16.5 prompt | 2026-09-24: too slow to judge | older run |

With the 1080 Ti holding attention and the dense weights, the same V4.1 store decodes 7.13 / 6.97 ([the 1080 Ti
page](../1080-ti/)). With the 1080 Ti and the MI25 holding every expert, Ornith-1.5-35B-A3B decodes 59.13 against 47.34
with its experts on these CPUs ([the MI25 entry](../mi25/)).

**One DIMM short.** From 2026-10-20 03:18Z to 2026-10-21 19:14Z one of the 16 DIMMs was slightly unseated: 15 of 16
worked, and NUMA node 4 ran on one channel (63 GiB, half its read bandwidth). Every run on this page dated in that
window ran that way. They have not been re-run since the reseat.

## Where the bytes sit is the biggest lever

Each step changes one thing, same model (V4.1, release-precision store), same prompts. The two steps ran in two
sessions, so each is shown against its own baseline:

| session | layout | draft | no draft |
|---|---|---|---|
| 1 | experts read in place from a RAM disk interleaved over the nodes | 5.31 | 4.12 |
| 1 | each expert's rows split across all 8 nodes, each node's threads computing their own slice | 6.64 | 4.45 |
| 2 | experts row-split (the same layout, a later load) | 6.45 | 4.42 |
| 2 | + the dense weights (attention, shared experts, output head) split the same way | **7.02** | **5.16** |

- Row-splitting the experts: +25 % with the draft (+27 % on the 2-bit split store).
- Row-splitting the dense weights too: +9 % with the draft, +17 % without. Both are our engine changes
  (`LLAMA_EXPERT_PLACEMENT` rules with `GGML_MOE_GROUPS=8`, `-cr` listing cores in node order); see
  [moe-offload, engine changes](../moe-offload/#engine-changes).
- **More threads do not help.** 32 threads beat 64 (point 15): no draft 4.73 against 4.42 (+7 %), at 406 W instead of 532 W.
  We keep 64 for agent turns, because prefill is 55 % faster.
- **The low-bit store buys no speed here.** Experts at 2 bits for the cold ones run at the speed of the release
  store (6.53 against 6.64, point 10). Our reading, not settled: decode on this box is not bound by DRAM bytes, but by
  a fixed cost per layer across 8 small NUMA nodes and the links between them. Without the draft a token takes about
  227 ms here against about 114 ms on the [2019 workstation](../rtx-6000/).
- Wall power: 406-590 W in these runs.

## Reproduce

- Build ik_llama.cpp from our fork's `mixed-cuda-vulkan-expert-banks` branch, CPU only, with
  `-DGGML_EXPERT_CHUNKING=OFF` ([why](../moe-offload/#what-we-learned-measured), point 13).
- Load without mmap (`--no-mmap`), 64 threads, `-cr` in node order, a placement file that row-splits every routed
  expert and the dense weights over nodes 0-7, `GGML_MOE_GROUPS=8`.
- The layout as data: [`recipe-maker/recipes/lassie-v41-q8-rows8.json`](../recipe-maker/recipes/lassie-v41-q8-rows8.json)
  is the expert-row step (6.64 / 4.45). The 7.02 / 5.16 run (ledger id `lassie-v41-q8-rows8-dense-t64` in
  [RESULTS.md](../recipe-maker/results/RESULTS.md)) uses the same files and adds row rules over nodes 0-7 for the
  attention weights (q_a, q_b, kv, output_b, and output_a in 8 groups), the shared experts and the output head, about
  0.97 GiB per node. Its full recipe is the `recipe` field of that row in the run ledger
  ([`recipe-maker/runs/`](../recipe-maker/runs/), the numbered segments).

## What did not work

- **More threads**: 96 and 128 (SMT siblings) were slower than 64; 32 beat 64 in decode by 7 % without the draft ([point 15](../moe-offload/#what-we-learned-measured)), and lost in prefill (23.7 against 36.8).
- **A smaller store**: 2-bit cold experts gave no speed on this box (above).
- **A block drafter** would not pay: a pass over 8 tokens costs 2.7x a pass over 1
  ([point 12](../moe-offload/#what-we-learned-measured)).

## Not measured yet

- Agent tasks with a CPU-only model on this box.
- A near-lossless (Q8_0) GLM reference, which this box's 1 TB could hold
  ([the quality reference problem](../moe-offload/#the-quality-reference-problem)).

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [arc](https://github.com/akapug/arc) + [fab](https://github.com/akapug/fabric) for [@akapug](https://github.com/akapug)
