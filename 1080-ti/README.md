# NVIDIA GTX 1080 Ti (11 GB, Pascal)

**Box:** lassie, a two-socket server from 2017 (Supermicro H11DSi): 2x EPYC 7601 (32 cores each, Zen 1, AVX2, 8 NUMA
nodes), 1 TB DDR4-2400 on 16 channels, the card on PCIe 3. The box also holds an [AMD MI25](../mi25/). The
[MoE offload entry](../moe-offload/) calls this box D. The box's CPUs alone have [their own page](../epyc-cpu/).

An 11 GB card from 2017 cannot hold any model on this page. What it can do is hold the parts of a model that every
token touches (attention, the KV cache, the dense and shared-expert weights) while the routed experts stay in the
server's 1 TB of DRAM. That took one engine fix first: on this card DeepSeek V4.1's attention produced garbage text.

## At a glance

All measured on this box, one request at a time, greedy, 3 coding-agent prompts of 1.5-2.2k tokens x 300 tokens,
64 threads, unless the row says otherwise. Decode and prefill are tokens per second. Terms: [GLOSSARY](../GLOSSARY.md).

| model | engine and weights | result | conditions | quality check | details |
|---|---|---|---|---|---|
| Ornith-1.5-35B-A3B | ik_llama.cpp with our changes, Q4_K_M; every routed expert on the GPUs: 10 layers on this card, 30 on the MI25 | **59.13 decode** with MTP 3 (60.01 with MTP 2), 44.62 without; prefill 331 | the model's own MTP head, 8,192-token window, 2026-10-21 | code-review screens: 1 of 2 plants found, 1 false alarm on the clean control | [the MI25 entry](../mi25/#ornith-15-35b-a3b-on-the-mi25-and-a-gtx-1080-ti-about-60-toks) |
| gpt-oss-120b | ik_llama.cpp, MXFP4; experts of layers 0-2 on this card, the rest on the CPUs | **34.6 decode** (31.0 with every expert on the CPUs), no draft | 2026-10-21 | code-review screens: 1 of 2 plants, right on the clean control | [the MI25 entry](../mi25/#the-comparison-on-the-same-box-gpt-oss-120b) |
| DeepSeek V4.1 Flash | ik_llama.cpp with our changes; routed experts at release precision (MXFP4) in DRAM, rows split over all 8 nodes; this card holds attention, the KV cache and the dense weights | **7.13 decode** with the DSpark draft, **6.97** without; prefill 48.2 | 8,192-token window, 2026-10-20, 15 of 16 DIMMs (below) | the store is the release-precision reference the droopy runs are scored against | [moe-offload point 17](../moe-offload/#what-we-learned-measured) |
| GLM-5.3-Flash | ik_llama.cpp `5f89bfc`, unsloth UD-Q3_K_XL; this card holds the 8.4 GB of non-expert weights, the experts stay in RAM | 9.65 decode with MTP 1, 8.63 without; prompt 60.6 | builds ran on the same box: a floor, not a clean number; 2026-10-02 | five of five easy prompts right (impression, no scored test) | (older run, before the engine changes on this page) |

**One DIMM short.** From 2026-10-20 03:18Z to 2026-10-21 19:14Z one of the 16 DIMMs was slightly unseated: 15 of 16
worked, and NUMA node 4 ran on one channel (63 GiB, half its read bandwidth). Every run on this page dated in that
window ran that way. They have not been re-run since the reseat.

## The engine fix: V4.1 attention on a card without a kernel for it

Moving V4.1's attention and KV cache onto this card first gave garbage text, while a small Qwen3 model ran correctly
on the same build and card. The cause, traced op by op ([moe-offload point 11](../moe-offload/#what-we-learned-measured)):
the card has no CUDA kernel for V4's flash attention (head size 512 with sinks), so that op ran on the CPU and wrote
into host memory; the in-place rotation after it was scheduled on the GPU, and the GPU had already copied its input
before the CPU's write landed.

The fix runs an in-place op on the backend that owns the tensor it writes into. It is on by default in our fork
(`GGML_SCHED_INPLACE_FOLLOW`, branch [`sched-inplace-follow`](https://github.com/akapug/ik_llama.cpp/tree/sched-inplace-follow)).
With it the per-op sums agree with a CPU-only run to within rounding, and then the card pays:

| layout | draft | no draft | prefill | card memory |
|---|---|---|---|---|
| CPUs only | 7.02 | 5.16 | 36.2 | - |
| card holds the dense weights, not attention | 6.59 | 5.60 | 37.2 | 6,385 MiB peak |
| **card holds attention and the KV cache too** | **7.13** | **6.97** | **48.2** | 11,129 of 11,264 MiB peak |

- +35 % without the draft and +33 % prefill against the CPUs alone. Wall power with attention on the card: median 574
  W, peak 590 W.
- Our reading: any device that cannot run this attention kernel would hit the same bug, not only Pascal.
- A second fault, a GPU memory fault at warm-up when weights were pinned, went away with weights kept in plain host
  memory (`LLAMA_NO_HOST_WEIGHTS=1`).

## Reproduce

- The Ornith-1.5-35B-A3B layout, with every command and switch: [the MI25 entry, To run it](../mi25/#to-run-it).
- The V4.1 layout: the engine changes and switches are in [moe-offload, engine changes](../moe-offload/#engine-changes);
  build for sm_61 with `-DGGML_EXPERT_CHUNKING=OFF`, `-ngl 99 --cpu-moe`, micro-batch 1024, experts row-split over the 8
  nodes with `LLAMA_EXPERT_PLACEMENT` and `GGML_MOE_GROUPS=8`, `-cr` listing the cores in node order.
- The layouts as data: [`recipe-maker/recipes/lassie-orn35-q4km-o6.json`](../recipe-maker/recipes/lassie-orn35-q4km-o6.json)
  (the Ornith run), and [`lassie-v41-q8-rows8.json`](../recipe-maker/recipes/lassie-v41-q8-rows8.json), the CPU-only
  expert-row step the V4.1 layout grew from. The 7.13 run itself is ledger id `lassie-v41-q8-fs3` in
  [RESULTS.md](../recipe-maker/results/RESULTS.md); its full recipe is the `recipe` field of that row in the run ledger
  ([`recipe-maker/runs/`](../recipe-maker/runs/)).

## What did not work

- **Hot V4.1 experts on this card** cost 3-4 %: the 502 MiB left over held 28 experts, 4.0 % of routed picks
  ([point 17](../moe-offload/#what-we-learned-measured)).
- **The V4.1 draft with attention on the card** buys only 1.02x (7.13 against 6.97). Putting the draft's own weights on
  the card made it slower (5.57). On this layout we would serve without a draft.
- **V4 attention on this card before the fix**: garbage text, and an earlier V4.0 run with attention on the card
  decoded 3.83 tok/s against 6.44 on the CPUs alone (2026-10-02).
- **V4.1's hot experts on the MI25 beside this card**: correct text, but slower in every version, bound by the hops
  between devices ([point 18](../moe-offload/#what-we-learned-measured)).

## Not measured yet

- gpt-oss-120b's experts in GPU banks (it needs bias and `SWIGLU_OAI` support in our bank code).
- Agent tasks on this box's models.

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [arc](https://github.com/akapug/arc) + [fab](https://github.com/akapug/fab) for [@akapug](https://github.com/akapug)
