# NVIDIA Quadro RTX 6000 (24 GB, Turing) with Optane PMem

**Box:** droopy, an HP Z8 G4 workstation from 2019: 2x Xeon Gold 5215 (10 cores each, Cascade Lake, AVX-512 VNNI),
192 GB DDR4-2666 on 12 channels, plus 256 GB of Intel Optane PMem 100 (2 x 128 GB in App Direct mode, onlined as
a CPU-less NUMA node), the card on PCIe 3 x16. The [MoE offload entry](../moe-offload/) calls this box B.

This is our box for the biggest models: weights that do not fit in VRAM or DRAM spill to Optane, which is slower
than DRAM but far faster than a disk. The question it answers: how close does a 2019 workstation with a 2018 card get
to a frontier-class MoE at usable speed and full context?

## At a glance

All measured on this box, one request at a time, greedy, on coding prompts unless the row says otherwise. Decode and
prefill are tokens per second. "n4" = speculative decoding with 4 drafted tokens a step; "n0" = no draft.
Terms: [GLOSSARY](../GLOSSARY.md).

| model | engine and weights | result | conditions | quality check | details |
|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | ik_llama.cpp with our changes; hot experts MXFP4, cold experts q2_K; 7.0 GiB of hot experts on the GPU, the rest in DRAM and PMem | **12.69 decode** (n4), 10.29 (n0), prefill 69.0 (this phase; the ledger's run summary gives 67.9) | DSpark draft model, n_max 4; 3 coding-agent prompts x 300 tokens; 2026-10-22. The build adds two changes not yet in our public fork (expert pre-binding, the tier split) and one upstream change | KL 0.0169, top-1 0.975 against the release-precision store (1,376 positions): passes our bar. Measured on the same store and layout with the earlier 10.6 tok/s build, not re-measured on this one. Upstream #2712 changes the indexer's arithmetic (draft acceptance moved from 0.890 to 0.908), so read this as the earlier build's check | [V4.1, the best run](#deepseek-v41-flash-the-best-run) |
| DeepSeek V4.1 Flash, same recipe on the public fork | as above, on the public fork's last branch | **11.88** (n4), 9.72 (n0), prefill 64.9 | as above, 2026-10-21 | as above | [Reproduce](#reproduce-the-best-v41-run) |
| DeepSeek V4.1 Flash at 189k tokens | as above, 2.2 GiB of hot experts, 212,992-token window | **16 of 16 facts recalled** from a 189,454-token prompt; decode 7.18 at that depth (n0), prefill 57.7 at 110-190k | `--swa-compress`, micro-batch 2048, greedy, 2026-10-21 | the recall itself | [Long context](#v41-at-189k-tokens-every-fact-recalled) |
| Ornith-1.5-122B (Qwen3.5-122B-A10B fine-tune) | ik_llama.cpp with our changes, Q4_K_M (70.6 GiB); experts of layers 0-8 on the GPU, the rest in DRAM | **31.6 decode** at 4,000 tokens of context (31.0 at 600), 27.1-27.3 without the draft; prefill 165.5 | its own MTP layer, n_max 3; 49,152-token window; 2026-10-20 | code-review screen: both planted defects found with an 8k thinking budget; agent tasks 0 of 3 in a configuration that limited it | [moe-offload points 19-20](../moe-offload/#what-we-learned-measured) |
| GLM-5.3-Flash (321B) | ik_llama.cpp with our changes, unsloth UD-Q4_K_XL; hot experts on the GPU, CPU experts row-split over both sockets | **16.06 decode** (mean of 4 prompts x 3 rounds), 12.30 without the draft | MTP 3 with a reduced draft vocabulary, p_min 0.75, expert chunking off; 2026-10-20 | KL 0.030, top-1 0.960 against a 3-bit reference: a measured miss of that reference; fidelity to BF16 unmeasured | [moe-offload points 2, 4-6, 13-14](../moe-offload/#what-we-learned-measured) |
| Qwen3.6-35B-A3B | llama.cpp b11175 CUDA, UD-Q4_K_XL (22.9 GB), experts of 19 layers in RAM | 38 decode fresh, 28.5 at 165k depth; 81-119 on edit output with n-gram drafting | 262,144-token window, 2026-09-29 | - | served as a coding agent's model in late September 2026 |

## DeepSeek V4.1 Flash: the best run

**The layout.** The RTX 6000 holds attention, the dense and shared-expert weights, the output head, the draft
model's dense weights and 7.0 GiB of the hottest routed experts (420 experts). Each CPU expert's rows are split
between the two sockets, so each socket's threads read their own memory. DRAM is filled with the hottest experts the
GPU does not hold; the coldest go to PMem. The weights are a split store: hot experts at MXFP4 (as DeepSeek released
them), cold experts at q2_K, 22.5 % of routed reads at 2 bits. Every engine change and its switch is in
[moe-offload, engine changes](../moe-offload/#engine-changes).

**The tier split** (not yet in our public fork). PMem reads are slower than DRAM reads, so a CPU thread that draws
mostly PMem-resident experts finishes last and every other thread waits for it. The tier split starts some threads
of each MoE group on the slow (PMem) list. Measured in one load with fresh prefill per phase, so every phase decoded
the same text (2026-10-22):

| phase | n4 decode (3 prompts) | n0 decode | prefill |
|---|---|---|---|
| off | 12.35 (13.19, 12.99, 10.87) | 10.14 | 65.3 |
| split 2 threads | 12.53 | - | 67.6 |
| split 4 | 12.62 | - | 68.6 |
| **split 8** | **12.69** (13.59, 13.31, 11.17) | **10.29** | 69.0 |
| off, repeated | 12.44 | - | 67.7 |

- +2.4 % over the mean of the two "off" phases, higher on every prompt, rising with the split. The output text and
  draft acceptance are identical in every phase: the split changes only who reads which expert, not the arithmetic.
- PMem-resident experts are 11-15 % of expert uses in decode.
- A **prefetch** that copies the next step's slow experts from PMem to a DRAM ring **lost** in every matched
  comparison: its predictor hit 21 % of the time, and the copying (about 1.5 GB/s) competed with the compute.
- The build under these numbers also carries our expert pre-binding (not yet public: 11.88 to 12.13) and upstream
  ik_llama.cpp's F32 indexer change (#2712), which on its own moved the same recipe from 12.13 to 12.40 (n4) and
  10.00 to 10.16 (n0), with draft acceptance 0.890 to 0.908. Run-to-run
  noise on this recipe, same build, two loads: 0.2-0.6 %.

## V4.1 at 189k tokens: every fact recalled

A recall probe: 8 facts ("The access code for the <name> vault is <7 digits>.") per segment, placed at evenly spaced
depths in real technical text, then a question asking for all of them. Greedy, raw `/completion`, no draft, 240
tokens out. The 189k prompt extends the 112k one, so the server restores its 110,592-token checkpoint and prefills the
rest. Layout: 2.2 GiB of hot experts, plain row placement, micro-batch 2048, a 212,992-token window with
`--swa-compress` (the sliding-window K cache stays at the window size). 2026-10-21.

| prompt | recalled | prefill | decode at that depth |
|---|---|---|---|
| 112,154 tokens | 8 of 8 | 65.4 tok/s | 7.77 |
| 189,454 tokens | **16 of 16**, from 4 % to 98 % depth | 57.7 tok/s (the 80,910 new tokens, at 110-190k depth) | **7.18** |

- The same layout decodes 11.24 tok/s (n4) and 9.25 (n0) on the short prompts: the big-context graph costs about 5 %
  of short decode.
- VRAM: 16,856 MiB after warm-up, 17,810 at the end of the 189k run. With micro-batch 4096 instead, the compute
  buffer grew with depth and the server **ran out of VRAM** at 122,880 tokens. Use 2048 for long context.
- A build without the F32 indexer change recalled the same facts with byte-identical answers, so on this box that
  change is a speed gain, not a long-context fix.

## Reproduce the best V4.1 run

What is public and what is not:

| piece | where |
|---|---|
| the engine changes behind 11.88 tok/s | [akapug/ik_llama.cpp](https://github.com/akapug/ik_llama.cpp), branch `mixed-cuda-vulkan-expert-banks` (it contains the others); [the fork's PATCHES.md](https://github.com/akapug/ik_llama.cpp/blob/mixed-cuda-vulkan-expert-banks/PATCHES.md) lists every switch |
| the changes between 11.88 and 12.69 | expert pre-binding (+2 %, binds each expert weight to its NUMA node before the load reads it) and the tier split (+2.4 %): not yet published. Upstream ik_llama.cpp's F32 indexer change (#2712, +2.2-2.5 %) is public upstream |
| the model | DeepSeek V4.1 Flash 1017; we requantized the cold experts to q2_K with an importance matrix. The split store is not uploaded; the planner's [quant recipes](../recipe-maker/QUANT-RECIPES.md) list published stores |
| the draft model | the DSpark draft that DeepSeek's V4.1 Flash 1017 release declares in its `config.json` (128 routed experts, top-3, block 5), converted to GGUF with its experts at Q2_K; our GGUF is not uploaded |
| hot-expert lists and placement | [Recipe Maker](../recipe-maker/), [HOT-EXPERT-MAPS.md](../recipe-maker/HOT-EXPERT-MAPS.md) (how to capture a map for your own prompts) |
| the layout as data | [`recipe-maker/recipes/droopy-v41-fs1c.json`](../recipe-maker/recipes/droopy-v41-fs1c.json) is the layout the 12.69 run (ledger id `v41fs-ts2-0127`) used with the tier-split build; [`droopy-v41-fs1p.json`](../recipe-maker/recipes/droopy-v41-fs1p.json) is the same layout's pre-binding run (12.19). Their `"verify": "fail"` means the launcher checks each node's placement after the load and stops the run if it is off target; it is not a failed run. Rows in [RESULTS.md](../recipe-maker/results/RESULTS.md) |
| PMem as memory | the standard Linux route, as on this box: `ipmctl create -goal PersistentMemoryType=AppDirect` and a reboot (a full power cycle after a firmware update), `ndctl create-namespace --mode=devdax` on each region, then `daxctl reconfigure-device --mode=system-ram <daxX.Y>` onlines it as a NUMA node with no CPUs. `numactl -H` should show it |

**Public or not.** Built from the public branch, this command runs the configuration behind the public-fork figure
(11.88). The 12.69 figure also needs upstream #2712 and our two unpublished changes.

The launch, as our kit runs it (paths shortened; the placement and hot-expert file names are ours):

```bash
# system state for the run: no automatic page migration, swap off
echo 0 | sudo tee /proc/sys/kernel/numa_balancing
echo false | sudo tee /sys/kernel/mm/numa/demotion_enabled
sudo swapoff -a

env GGML_CUDA_GRAPHS_PRE_AMPERE=1 GGML_AFFINITY_ONCE=1 LLAMA_MOE_OVERLAP=1 GGML_SCHED_OVERLAP=1 \
    GGML_SCHED_ASYNC_COPY=1 GGML_OMP_FAST_ENTRY=1 GGML_BARRIER_SPIN=1 GGML_MOE_GROUPS=2 GGML_MAIN_NO_MOE=1 \
    GGML_CUDA_PIN_STAGING=1 GGML_CUDA_PIN_MAX_MIB=4096 GGML_SCHED_INPLACE_FOLLOW=1 LLAMA_NO_HOST_WEIGHTS=1 \
    LLAMA_EXPERT_PLACEMENT=v41-rows-hot7p0.placement.tsv LLAMA_HOT_EXPERTS=v41-hot7p0.tsv \
  numactl --interleave=0,1 --physcpubind=0-9,20-29,10-19,30-38 \
  llama-server -m DeepSeek-V4.1-Flash-1017-hotmxfp4-coldq2k-00001-of-00010.gguf \
    -ngl 99 -fa 1 --no-mmap -t 39 -c 4096 -cr 0-9,20-29,10-19,30-38 --numa numactl -dple --cpu-moe \
    -md DeepSeek-V4.1-Flash-1017-DSpark-exps-Q2_K.gguf -ngld 99 -draft "-ncmoe 3 --no-mmap" -cd 4096 \
    --spec-type dspark:n_max=4,conf_cum_min=0.5 -ub 4096 -b 4096 -cuda offload-batch-size=100000
```

- The placement file splits each CPU expert's rows over nodes 0 and 1 and adds one rule that keeps the draft
  model's 3 layers of experts on node 1; the format is in the fork's PATCHES.md.
- `-cr` lists the cores socket by socket, so MoE thread group g is socket g.
- `-c 4096` is what our launch script passes; the run ledger's condition text says "-c unset", which is wrong. The
  prompts are under 2.2k tokens, so the window does not bind.
- For the 189k recall layout: 2.2 GiB of hot experts, `-ub 2048 -b 2048 -c 212992 --swa-compress`.
- `GGML_SCHED_ASYNC_COPY` is in the public branch's `ggml/src/ggml-backend.cpp` but not yet described in the fork's
  PATCHES.md; without it the engine runs its default copy path.
- The client: 3 coding-agent prompts of 1.5-2.2k tokens, 300 greedy tokens each through `/completion`, after a
  64-token warm-up. Each figure is the mean of the 3 prompts.

What you cannot get from us yet, so your number will differ:

- **The three prompts.** They come from our own coding-agent sessions and quote private code, so they are not
  published. The prompt matters: on the public-fork build the three gave 12.88, 12.62 and 10.13 tok/s (mean 11.88).
  Any coding prompt of 1.5-2.2k tokens tests the same thing; expect a spread of that size.
- **The split store and the draft file.** Neither is uploaded. Build the store from the release weights, requantizing
  the cold experts to q2_K with an importance matrix; the [quant recipes](../recipe-maker/QUANT-RECIPES.md) give our
  quantization rules and the published stores to start from.
- **Expert pre-binding and the tier split.** Not published yet; without them, expect the public-fork figure
  (11.88).
- **Your hot-expert list.** Capture one on your own prompts ([HOT-EXPERT-MAPS.md](../recipe-maker/HOT-EXPERT-MAPS.md));
  ours fits our prompts.

## What did not work

- **Qwen3.8 Flash-Next on this card with llama.cpp**: IQ3_S with 44 of its 48 expert layers in RAM decoded 11.7 tok/s
  (0.3 when its experts paged in from the disk), and passed 6 of 12 agent tasks, all 6 misses out of time
  (2026-09-29). The same model on one B70 with XeStrata runs at 80-87 tok/s.
- **PMem prefetch** for V4.1: a loss (above).
- **CPU weight repacking (`-rtr`)** for GLM: 15.03 to 10.04 tok/s ([point 5](../moe-offload/#what-we-learned-measured)).
- **The whole V4.1 draft model on the CPU**: 2.44-2.69 tok/s, because a CPU draft round costs more than a main-model
  token ([point 21](../moe-offload/#what-we-learned-measured)).
- **A block drafter** (verifying 8+ tokens a pass) would not pay on this box: a pass over 8 tokens costs 2.7-2.9x a
  pass over 1 ([point 12](../moe-offload/#what-we-learned-measured)).
- **More Optane** would add capacity, not speed: the planner predicts +15-18 % for V4.1 from 4 more sticks
  ([point 7](../moe-offload/#what-we-learned-measured), a prediction).
- **Ornith-1.5-122B as an agent**: 0 of 3 tasks in a configuration that limited it (a 262k window that does not fit
  beside its weights on a 24 GB card; a 45-minute wall at 18 tok/s). Not yet a verdict on the model.

## Not measured yet

- The tier split at more than 8 threads per group (the gain was still rising at 8).
- V4.1 agent work on this box.
- GLM-5.3-Flash with the route trim that passes on the [3090 Ti](../3090-ti/).

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [arc](https://github.com/akapug/arc) + [fab](https://github.com/akapug/fab) for [@akapug](https://github.com/akapug)
