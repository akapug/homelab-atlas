# AMD Radeon Instinct MI25 (16 GB HBM2): a $50 card under Vulkan

The MI25 is a 2017 AMD datacenter card: Vega 10 (gfx900), 16 GB of HBM2 on a 2048-bit bus, about 484 GB/s on
paper, passive cooling. We paid about $50 for ours. AMD's current ROCm releases dropped gfx900, so this entry runs
it on Vulkan, with no ROCm installed. The question: what does a card this cheap add to an old server, alone, as a
hot-expert store for a MoE model, and beside a GTX 1080 Ti?

It sits in box D of [the MoE offload entry](../moe-offload/): 2x EPYC 7601, 1 TB DDR4, a GTX 1080 Ti.

**The best result so far: a 35B MoE model at about 60 tok/s on two retired GPUs that nobody would pair.** The MI25
(Vega 10, Vulkan only) and the GTX 1080 Ti (Pascal, CUDA) hold every routed expert of Ornith-1.5-35B-A3B between
them, in one ik_llama.cpp process, in a 2017 server. With the model's own MTP draft head it decodes 59.1-60.0 tok/s
on coding prompts, against 42.0 with the experts on the CPUs.
[Details, commands and the quality screens below](#ornith-15-35b-a3b-on-the-mi25-and-a-gtx-1080-ti-about-60-toks).
The same hardware does not help DeepSeek V4.1 Flash: [that result is negative](#as-a-hot-expert-store-for-deepseek-v41-stopped-bound-by-hop-latency),
and we explain why.

## Ornith-1.5-35B-A3B on the MI25 and a GTX 1080 Ti: about 60 tok/s

[Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) is a fine-tune of Qwen3.6-35B-A3B: 35B
parameters, about 3B active per token, 40 layers (30 Gated DeltaNet, 10 attention), 256 routed experts per layer,
and one MTP layer that can draft tokens for the model. We ran ornith-ai's Q4_K_M GGUF
([`ornith-ai/Ornith-1.5-35B-A3B-GGUF`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF/tree/f60452672689e0040be350560267a614cc172a18),
20.2 GiB). Its routed experts total about 18.2 GiB. That does not fit on either card alone, but it fits on the two
together.

**The layout.**
- The GTX 1080 Ti (11 GB) holds attention, the DeltaNet layers, the KV cache, the router, the shared expert, the
  norms, the output head and the MTP layer (`-ngl 99 --cpu-moe`).
- Every routed expert of the 40 main layers sits in a GPU bank. Banks hold whole layers: 30 layers (13.5 GiB) on the
  MI25 and 10 layers (4.7 GiB) on the 1080 Ti. The engine reports "10240 hot experts in 40 layers, 18600.04 MiB on
  GPU".
- The host copy of the experts stays in DRAM, each expert's rows split over the 8 NUMA nodes. Prefill uses it,
  because the banks serve only batches of 8 tokens or fewer.
- In each of the 30 layers banked on the MI25, a token crosses CUDA, Vulkan and CUDA again; the other 10 layers
  stay on the 1080 Ti (CUDA banks). All of it runs in one process. That needs our engine changes
  ([below](#the-engine-changes-and-their-switches)).

**Conditions for the table.** Box D (2x EPYC 7601, 8 NUMA nodes, DDR4-2400; GTX 1080 Ti on driver 580; MI25 on Mesa
RADV, clocks forced to the high DPM state). Our ik_llama.cpp fork at the commits linked below, built for sm_61 with
Vulkan. One server load per arm. 64 threads (`-cr 0-63`), `-c 8192`, `-ub 1024`, f16 KV cache, `--no-mmap`.
The workload is 3 coding-agent prompts of about 1.5-2.2k tokens, 300 greedy tokens each, through the raw
`/completion` endpoint. Each figure is the mean of the 3 prompts. `m3` / `m2` = the model's MTP head drafting
3 / 2 tokens (`--spec-type mtp:n_max=3,p_min=0.75`, with `n_max` set per request). `n0` = no draft. Wall power is
the median for the whole box during the run. All MEASURED, 2026-10-21.

| arm | where the routed experts are | decode tok/s | prefill tok/s | 1080 Ti memory (MiB, of 11,264) | wall power |
|---|---|---|---|---|---|
| o1 | CPUs (rows over 8 nodes) | n0 41.99 (42.79, 41.14, 42.05) | 370 | 2,935 | 542 W |
| o2 | GPU banks: MI25 + 1080 Ti | n0 44.62 (45.11, 44.51, 44.24) | 368 | 7,719 | 484 W |
| o5 | CPUs, + MTP draft | m3 47.34, m2 46.96, n0 41.95; acceptance 85 % | 320 | 3,531 | 598 W |
| **o6** | **GPU banks, + MTP draft** | **m3 59.13 (56.11, 58.62, 62.67), m2 60.01 (59.27, 56.85, 63.91), n0 44.59**; acceptance 85 % | 331 | 8,315 | **508 W** |

- **The banks give +6 % without a draft and +25 % with one** (o6 against o5, m3). Our reading, not measured
  separately: when MTP verifies 4 tokens at once, the CPU path must read the union of the experts that those
  tokens route to, from DDR4. The banks compute those tokens on the GPUs, from HBM2 and GDDR5X.
- **The box draws less power when it runs faster.** o6 at 508 W median against o5 at 598 W. Our reading: the
  CPUs no longer compute the experts during decode. The MI25 peaked at 59 C junction and 57 C HBM, far below the limits in [Heat and power](#heat-and-power).
- Same-binary text check: the n0 texts of o6 and o2 are byte-identical, and so are those of o5 and o1. The bank
  arms and the CPU arms differ from each other. Our reading: rounding, because the experts are computed by other
  kernels. All texts read coherently.
- For scale: the same model, as `sym_int4` under vLLM on one current Intel Arc Pro B70, decodes about 198 tok/s at
  one stream ([the B70 entry](../b70/ornith-35b-a3b.md)). This entry is about what old parts can do, not a claim that
  they beat a new card.

**Real chat requests at the model card's sampling.** The o6 layout served three long code-review requests through
`/v1/chat/completions` with `--jinja`, temperature 0.6, top_p 0.95, top_k 20, min_p 0.0, thinking on with an 8,192-token budget
(`--reasoning-budget 8192` and a message telling the model to commit), and `-c 32768`. Decode 45.7, 50.3 and
53.3 tok/s over 9.0-9.5k generated tokens each, at 13-21k tokens of context. MTP acceptance 85.7 %. Prefill 318-354
tok/s. MEASURED from the server log.

### Quality: three code-review screens

These are the same three private screens as for Ornith-1.5-122B in [moe-offload point 20](../moe-offload/#what-we-learned-measured):
three diffs of our own code, two with one planted defect each and one clean control. One sample per screen, graded
by hand against answer keys written before the runs. They are a smoke test of judgement, not a benchmark.

| model and box | planted defect 1 | planted defect 2 | clean control |
|---|---|---|---|
| Ornith-1.5-35B-A3B Q4_K_M, box D, o6 layout, conditions above | PASS: names the defect, a scenario, a fix and a test that fails before the fix | PARTIAL: names another issue as THE defect; the plant is its first "other concern", worded correctly, with no fix or test | FALSE ALARM: it promotes a wording note to THE defect, and its scenario contradicts its own quote of the code |
| Ornith-1.5-122B Q4_K_M, box B, same budget and sampling | PASS | PASS | FALSE ALARM (the same overclaim) |
| gpt-oss-120b MXFP4, box D, medium reasoning effort, card sampling 1.0 / 1.0 | FAIL: a wrong defect | PASS: the plant, a fix and a test | PASS: "no defect" |

- The 35B finds the easy plant, ranks the second plant below a wrong primary, and claims a defect in clean code,
  like its 122B sibling. For a reviewer role that false alarm matters: a reviewer that always finds something costs
  the reader time.
- gpt-oss-120b is the one model here that did not overclaim on the clean code, but it missed the first plant.
- n = 1 per screen. None of these numbers show that one model is better at review in general.

### The comparison on the same box: gpt-oss-120b

[`ggml-org/gpt-oss-120b-GGUF`](https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/tree/7200b74bea78b0a2964ef9f3897eefa401e890ff),
native MXFP4 experts (a 59.0 GiB file), same build, same box. The 1080 Ti holds attention, KV cache and the dense weights;
the routed experts are on the CPUs with their rows split over 8 nodes. No draft: this engine's EAGLE-3 path is not
implemented yet. MEASURED, 2026-10-21.

| arm | decode tok/s (3 prompts, 300 greedy tokens) | prefill tok/s | 1080 Ti memory (MiB) | wall power |
|---|---|---|---|---|
| experts on the CPUs, `-c 32768` | 31.0 (31.97, 30.56, 30.46) | 163.2 | about 5,000 | 595 W median, 644 W peak |
| + the experts of layers 0-2 on the 1080 Ti (`-ot`), `-c 8192` | 34.57 (34.64, 34.59, 34.47; +11.5 %) | 177.1 | 8,169 | 591 W median, 618 W peak |

- Real chat requests (Harmony template, medium effort): 26.4-28.2 tok/s at 3.6-10.6k tokens of context, prefill
  155-163 tok/s.
- Each gpt-oss layer reads about 53 MB of experts per token from the CPU side, against about 14 MB for Ornith-35B
  (our arithmetic from the files). So each layer moved to a GPU saves more on gpt-oss.
- **The MI25 cannot hold gpt-oss experts yet.** gpt-oss experts carry biases and use the `SWIGLU_OAI` activation.
  Our bank partitions do not handle biases, and the Vulkan fused MoE op in this engine handles neither. A repair
  needs both. Our estimate of the gain, not measured: 0-10 %, because Ornith gained 6 % from the banks without a
  draft.
- The greedy speed check is weak for this model: through raw `/completion` with `ignore_eos`, it keeps generating
  empty analysis turns after its end token. The chat figures above use the real template.

### The engine changes and their switches

All of them are in our public fork [akapug/ik_llama.cpp](https://github.com/akapug/ik_llama.cpp), branch
[`mixed-cuda-vulkan-expert-banks`](https://github.com/akapug/ik_llama.cpp/tree/mixed-cuda-vulkan-expert-banks)
(tip [`63bf5755`](https://github.com/akapug/ik_llama.cpp/commit/63bf575558e8ff5aa220bfdc0161f1dc4ca3b828)). The
measured binary was built from the same source; the published tree differs only in comments, test-harness option
names and internal build scripts. The fork's
[PATCHES.md](https://github.com/akapug/ik_llama.cpp/blob/mixed-cuda-vulkan-expert-banks/PATCHES.md) lists every switch.
Its box A is our box D.

| switch | what it does here |
|---|---|
| `LLAMA_EXTRA_VULKAN=1`, `GGML_VK_VISIBLE_DEVICES=<n>` | a CUDA build also opens the MI25 as `Vulkan0` |
| `LLAMA_HOT_EXPERTS=<file>` | the heat file: which experts go in banks |
| heat-file third column ([`b4e5f5d7`](https://github.com/akapug/ik_llama.cpp/commit/b4e5f5d7ea8b0bbf52342624fb3ddbef21c32d01)) | `blk.N <expert ids> <backend>` puts the bank of that layer on that backend, so one file splits layers between `Vulkan0` and `CUDA0`; a layer whose experts are all banked gets no CPU split |
| `LLAMA_HOT_EXPERTS_DEVICE=Vulkan0` | the default bank device for lines without a third column |
| `LLAMA_HOT_BANK_FIRST=1` ([`1aa46479`](https://github.com/akapug/ik_llama.cpp/commit/1aa464798f0406722ff485f05ea36a3bc7c0e003)) | the bank work goes first in the graph. Before this fix the switch did not reorder anything |
| `GGML_VK_ASYNC_GRAPH=1` | the Vulkan graph returns after its last submit |
| `GGML_VK_STAGED_INPUTS=1` | inputs and the last output of the Vulkan step go through a host-visible staging buffer |
| `GGML_VK_MMVID_MAX_TOKENS` (default 8) | Vulkan `mul_mat_id` batches of up to 8 tokens use the mat-vec kernel per token. On the MI25, a 5-token bank chain went from 11.1 to 1.30 ms |
| `GGML_VK_OFFLOAD_MIN_BATCH=1000000` | prefill does not copy host experts to the MI25 |
| `LLAMA_EXPERT_PLACEMENT=<file>`, `GGML_MOE_GROUPS=8` | the host copy of each expert is split by rows over the 8 NUMA nodes |
| `LLAMA_NO_HOST_WEIGHTS=1`, `GGML_CUDA_NO_PINNED=1`, `GGML_CUDA_PIN_STAGING=1` | weights are never pinned; only staging buffers are |
| `GGML_SCHED_INPLACE_FOLLOW` (on by default) | in-place ops run on the backend that owns their memory |

The test `tests/test-expert-banks.cpp` covers mixed, non-host and whole-layer banks
([`be71aea2`](https://github.com/akapug/ik_llama.cpp/commit/be71aea26d2f7a4b35226750f35897403408dac8) fixes its
whole-layer case). It passed on box D with the measured package.

### To run it

**Build** (we built in `nvidia/cuda:12.6.3-devel-ubuntu24.04` with `libvulkan-dev` and `glslc`):

```sh
git clone -b mixed-cuda-vulkan-expert-banks https://github.com/akapug/ik_llama.cpp && cd ik_llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_VULKAN=ON -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DCMAKE_C_FLAGS=-march=znver1 -DCMAKE_CXX_FLAGS=-march=znver1 \
  -DGGML_EXPERT_CHUNKING=OFF -DGGML_SCHED_MAX_COPIES=1 -DGGML_IQK_FA_ALL_QUANTS=ON -DBUILD_SHARED_LIBS=OFF
cmake --build build --target llama-server -j
```

**Placement rules.**
1. Fill VRAM with whole layers of experts. A layer split between a bank and the CPU still pays for a CPU pass and a
   device hop. With whole layers, that split goes away.
2. Put the MI25 first in the fill order: it holds 16 GB and is free of other work. The 1080 Ti takes what is left
   after attention, KV cache and the compute buffers (8,315 of 11,264 MiB in use here).
3. Leave the MTP layer out of the heat file when the MTP draft is on.
   [`whole-layer-banks.py`](whole-layer-banks.py) does this.
4. Force the MI25 to its high DPM state before a run: `echo high > /sys/class/drm/card<N>/device/power_dpm_force_performance_level`
   (undo: `echo auto`). Under `auto` it stayed at 300 / 167 MHz through bursty decode, and a 1-token bank chain took
   3.3x longer (1,387 against 423 microseconds).
5. Pick the Vulkan device by its name, not by its index: on this box the indexes changed across a reboot
   ([the card and the stack](#the-card-and-the-stack)).
6. Watch the HBM temperature during the run: see [Heat and power](#heat-and-power).

**The heat file** (needs `gguf-py` from the same tree; it reads only the GGUF header):

```sh
PYTHONPATH=gguf-py python3 whole-layer-banks.py Ornith-1.5-35B-Q4_K_M.gguf orn35-banks.tsv Vulkan0=14 CUDA0=5
# Vulkan0: 30 layers [9, 11, 12, ..., 39], 13.49 of 14 GiB
# CUDA0: 10 layers [0, 1, ..., 8, 10], 4.67 of 5 GiB
# CPU: []; MTP layers not written: [40]
```

**The rows placement** for the host copy: one line per routed-expert tensor, all experts, rows over nodes 0-7:

```sh
PYTHONPATH=gguf-py python3 -c 'import gguf,re,sys; r=gguf.GGUFReader(sys.argv[1])
for t in r.tensors:
    if re.search(r"ffn_(up|gate|down|gate_up)_exps", t.name): print(f"{t.name}\t*\trows:0,1,2,3,4,5,6,7")' \
  Ornith-1.5-35B-Q4_K_M.gguf > orn35-rows8.tsv
```

**The server** (arm o6):

```sh
echo 0 | sudo tee /proc/sys/kernel/numa_balancing
env GGML_AFFINITY_ONCE=1 LLAMA_MOE_OVERLAP=1 GGML_SCHED_OVERLAP=1 GGML_OMP_FAST_ENTRY=1 GGML_BARRIER_SPIN=1 \
  GGML_MAIN_NO_MOE=1 GGML_CUDA_NO_PINNED=1 GGML_CUDA_PIN_STAGING=1 LLAMA_NO_HOST_WEIGHTS=1 \
  GGML_MOE_GROUPS=8 LLAMA_EXPERT_PLACEMENT=orn35-rows8.tsv LLAMA_EXPERT_PLACEMENT_VERIFY=1 \
  LLAMA_HOT_EXPERTS=orn35-banks.tsv LLAMA_EXTRA_VULKAN=1 GGML_VK_VISIBLE_DEVICES=0 LLAMA_HOT_EXPERTS_DEVICE=Vulkan0 \
  GGML_VK_OFFLOAD_MIN_BATCH=1000000 GGML_VK_ASYNC_GRAPH=1 GGML_VK_STAGED_INPUTS=1 LLAMA_HOT_BANK_FIRST=1 \
  GGML_VK_MMVID_MAX_TOKENS=8 \
numactl --interleave=0-7 build/bin/llama-server -m Ornith-1.5-35B-Q4_K_M.gguf --no-mmap \
  -t 64 -tb 64 -cr 0-63 -c 8192 -ngl 99 --cpu-moe -cuda offload-batch-size=1000000 -ub 1024 -b 1024 \
  --spec-type mtp:n_max=3,p_min=0.75 -fa 1 -np 1 --host 127.0.0.1 --port 8080
```

- For chat, add `--jinja --reasoning-format deepseek --reasoning-budget 8192` and a larger `-c`. We ran `-c 32768`.
- The CPU-side switches (`GGML_AFFINITY_ONCE` to `GGML_MAIN_NO_MOE`) are research switches. Each was worth 0-3 % on
  another box and model; we did not measure them one by one here.
- Arm o5 = the same without `LLAMA_HOT_EXPERTS` and the Vulkan switches (MTP still on). Arm o1 = o5 without
  `--spec-type` as well, so no draft at all. Arm o2 = o6 without `--spec-type`. The measured o1 and o2 loads also
  carried an external DFlash draft model (`-md`, on the CPU; see [What did not work](#what-did-not-work)); their n0
  figures come from requests that did not draft.

### What did not work

- **An external DFlash draft model** for Ornith-1.5-35B-A3B (a Q8_0 GGUF, `--spec-type dflash:n_max=4`) was accepted for
  0.9 % of drafted tokens with the draft on the CPU, 1.3 % on the 1080 Ti, and 6 % with the scheduler overlap
  switches off. Decode with it was 11.5-19.9 tok/s, below the no-draft speed. The texts were coherent. Our reading: a fault in this engine's
  DFlash path for this draft, not yet investigated. Use the model's own MTP head.
- **The MTP layer in the heat file.** If the heat file also names the MTP layer, the load aborts with "already
  device-resident".

## The card and the stack

| item | value |
|---|---|
| device | PCI id 1002:6860, vBIOS `113-D0513500-N00`, an MI25 vBIOS (not the WX9100 one some owners flash) |
| power | 110 W cap, and 110 W is also the maximum the driver allows: it cannot be raised without a vBIOS change. Other MI25 reports run at 140-220 W |
| clocks | core 300-1500 MHz, HBM up to 945 MHz (the full ~484 GB/s state) |
| link | PCIe 3.0 x16 |
| cooling | passive heatsink, fed by a 3D-printed end shroud with a 120 mm fan |
| kernel | Ubuntu kernel 6.8.0-139 |
| userspace | Mesa 25.2.8 RADV. It reports fp16 yes, integer dot product no, no matrix cores |
| engine | llama.cpp `1d88cb0` with the Vulkan backend, built in an `ubuntu:24.04` container (glslc from shaderc 2023.8), static, AVX2 for the host CPUs |

**What did not work as shipped.**
- The stock `amdgpu` module of that kernel failed to start the card on 3 cold boots ("PSP load sos failed").
- What fixed it: rebuilding `amdgpu` from the same Ubuntu kernel source with the two bootloader wait times in
  `psp_v3_1.c` raised from 20 ms to 500 ms. With that, the card initializes on a cold boot.
- Vulkan device numbers changed across one reboot: `Vulkan0` was the 1080 Ti before it and the MI25 after it.
  One of our runs landed on the wrong card that way. Pick devices by name (`llama-bench --list-devices`), not by
  index.

We did not try ROCm.

## Results

All figures are `llama-bench` on box D, one card with `-ngl 99` unless a row says otherwise, flash attention on
unless a row says otherwise. Each figure is the mean over the reps named. Every call started only once the HBM was
back under 50 C.

### Dense models (pp512 / tg128, 3 reps)

| model | file GB | prefill tok/s | decode tok/s | file size x decode |
|---|---|---|---|---|
| Qwen3-1.7B Q8_0 | 1.83 | 1547 | 112.0 | 205 GB/s |
| Llama 2 7B Q4_0 | 3.75 | 424 | 45.4 | 171 GB/s |

For scale, the llama.cpp ROCm scoreboard (discussion #15021) lists this card at 410 / 64.0 on Llama 2 7B Q4_0, at a
power its row does not state. Our Vulkan decode is about 30 % under that.

### Quant ladder (Qwen3-1.7B, pp512 / tg128, 3 reps)

Every rung was requantized from the Q8_0 file with an importance matrix: a speed test, not a quality test. The same
ladder also ran on the GTX 1080 Ti in the same box, under the same Vulkan build.

| quant | file GB | MI25 prefill | MI25 decode | 1080 Ti prefill | 1080 Ti decode |
|---|---|---|---|---|---|
| Q8_0 | 1.83 | 1581 | 114.2 | 699 | 133.9 |
| Q6_K | 1.42 | 1273 | 123.6 | 585 | 152.8 |
| Q5_K_M | 1.26 | 1371 | 120.1 | 609 | 136.0 |
| Q4_K_M | 1.11 | 1520 | 140.6 | 635 | 178.9 |
| Q4_1 | 1.14 | 1709 | 128.3 | 725 | 174.1 |
| Q4_0 | 1.06 | 1709 | 145.9 | 706 | 179.5 |
| IQ4_NL | 1.05 | 1693 | 141.9 | 693 | 171.0 |
| IQ4_XS | 1.01 | 1636 | 152.1 | 690 | 170.3 |
| IQ3_XXS | 0.75 | 1526 | 131.9 | 664 | 154.1 |

- **A small model is bound by fixed cost per token, not bandwidth.** A least-squares line over five rungs gives the
  MI25 4.15 ms per token plus 2.59 ms per GB read: about 386 GB/s at the margin, 80 % of the 484 GB/s plate. The
  1080 Ti gives 3.30 ms plus 2.27 ms per GB. Halving the bytes (Q8_0 to IQ4_XS) bought only +33 % decode on the
  MI25.
- **Best decode: IQ4_XS, then Q4_0.** Q4_1, Q5_K_M and IQ3_XXS lose to smaller or equal rungs. IQ3_XXS is 25 %
  smaller than IQ4_XS and 13 % slower. Do not use IQ3_XXS on this card for speed.
- **Best prefill: Q4_0 and Q4_1** (1709). K-quants and Q6_K cost 11-25 %.
- **The 1080 Ti decodes this model 15-28 % faster; the MI25 prefills 2.2-2.4x faster.** Vega has fast fp16 under
  Vulkan; Pascal does not.

### Context depth (Llama 2 7B Q4_0, f16 KV cache, pp512 / tg128, 2 reps)

| depth | flash attention | prefill tok/s | decode tok/s |
|---|---|---|---|
| 0 | off | 410 | 42.0 |
| 0 | on | 421 | 45.5 |
| 4,096 | off | 297 | 15.9 |
| 4,096 | on | 284 | 30.8 |
| 16,384 | off | 141 | 1.4 |
| 16,384 | on | 142 | 3.4 |

- Decode collapses with depth: 45.5, 30.8, 3.4 tok/s with flash attention on.
- Flash attention doubles decode at 4k and 16k, and costs about 4 % prefill at 4k.
- Llama 2 7B has no grouped-query attention, so at 16k each token reads about 8.6 GB of KV cache. At the ~147 GB/s
  effective rate fitted to our dense runs, that would allow about 12 tok/s (our arithmetic); we got 3.4. So the Vulkan attention path on gfx900
  runs well under the card's bandwidth at depth.
- 16k is past Llama 2's trained context: this row measures speed only.

Batch and micro-batch (pp2048): any `-ub` of 512 or more is within 1 % of the best (386 tok/s at `-b 2048 -ub 1024`).
The default `-b 2048 -ub 512` (384) is fine.

### MoE models, alone and as a hot-expert store (pp512 / tg128, 2 reps)

Box D's CPUs had no other jobs during the CPU-only and two-card rows.

| model | where the weights are | prefill tok/s | decode tok/s |
|---|---|---|---|
| gpt-oss-20b MXFP4 (12.1 GB) | MI25 only | 634 | 67.8 |
| Qwen3-30B-A3B Q4_K_S (17.5 GB) | CPU only, 2x EPYC 7601, 32 threads | 80 | 10.2 |
| same | MI25, experts of 2 layers on the CPU (`-ncmoe 2`, 16 threads) | 392 | 36.7 |
| same | `-ncmoe 3` | 353 | 28.5 |
| same | `-ncmoe 4` | 320 | 21.3 |
| same | `-ncmoe 8` | 239 | 15.3 |
| same | `-ncmoe 16` | 166 | 10.3 |
| same | 1080 Ti + MI25, `-sm layer -ts 2/15` | 365 | 71.4 |
| same | `-ts 3/14` | 327 | 51.8 (one rep disturbed: ±29.8) |
| same | `-ts 4/13` | 296 | 74.4 |

- **One MI25 lifts this MoE 3.6x in decode and 4.9x in prefill over the server's CPUs alone,** when it holds all
  but 2 layers of experts.
- **Every layer of experts left on the CPU costs a lot:** 36.7, 28.5, 21.3 tok/s for 2, 3, 4 layers. At 16 layers
  the GPU run is no faster than CPU-only. On this box the CPU expert path is the bottleneck: fill the VRAM first.
- **Two old cards hold the whole model and decode at 71-74 tok/s, 7x CPU-only.** The 1080 Ti also held a ~5 GB
  process of its own. Prefill falls as more layers go to the 1080 Ti (no fast fp16 on Pascal): 365 to 296.

## As a hot-expert store for DeepSeek V4.1: stopped, bound by hop latency

Status: **a negative result.** The text is correct, but every version was slower than the same box without the
MI25, and the profile says why ([the outcome](#the-outcome-hop-latency-not-bandwidth)). Everything here is box D
running DeepSeek V4.1 Flash as in [moe-offload point 17](../moe-offload/#what-we-learned-measured): the GTX 1080 Ti
holds attention, KV cache and the dense weights, the CPUs hold the routed experts. The MI25 is added to hold the
most-routed experts, in the same process. That needed an engine change: a CUDA build of ik_llama.cpp that also opens
Vulkan devices and puts expert banks on one of them ([engine changes](../moe-offload/#engine-changes)).

**Device checks first.** The engine's op tests on the MI25 under Vulkan: 35 of 36 required ops and 43 of 44 V4.1
cases pass, all 8 full-bank cases among them. The one failure is sorting the router's scores over 384 experts
(ARGSORT, normalized error 0.554), so expert routing must stay off this card. The banks never route; they only
compute the experts they hold.

**V4.1 with 14 GiB of hot experts on the MI25.** The 799 most-routed experts (14,326 MiB, 31.6 % of the routed picks
on held-out coding text) sit in banks on the MI25; the other routed experts stay on the CPUs. One coding-agent prompt,
300 greedy tokens, the draft first, then no draft on the cached prompt.

| layout | draft tok/s | no draft tok/s |
|---|---|---|
| 1080 Ti + CPUs, no MI25 (same prompt) | 7.24 | 6.95 |
| + 799 hot experts on the MI25 | 2.66 | 4.08 |

- The greedy text is identical to the run without the MI25, with and without the draft.
- The card hardly warmed: HBM and junction peaked at 39 C. Wall power peaked at 578 W. No GPU faults.
- We stopped after one prompt to profile it.

**Where the time goes.** Measured on a smaller model, so a profile takes minutes: Qwen3-30B-A3B Q4_K_S with 16 of
its 128 experts per layer in MI25 banks, the engine's op profiler, milliseconds per decode step.
- Experts only on the CPU: 86 ms. With the banks as first built: 141 ms.
- Each layer runs four steps in turn: CUDA (attention, router), Vulkan (the bank), the CPU (the other experts), CUDA
  again. The MI25's own GPU time per bank is about 0.39 ms; the rest of the cost is waiting and copying.
- Our reading of the profile and the code: the Vulkan step waits on its fence, so it cannot overlap the CPU's
  experts; one fused expert path allocated a fresh device buffer on every step; and summing the partial expert
  outputs on the MI25 meant 5 inputs in and a synchronous read back out.

**Fixes so far, each a runtime switch.**

| change | measured |
|---|---|
| Vulkan steps submitted without waiting, banks scheduled first, a persistent buffer, the sum moved to the CUDA card | 141 → 107 ms a step |
| inputs to the Vulkan step through one staging buffer, recorded into its first command buffer | its input copy 4.3 → 3.0 ms a step |
| the bank's result copied to a staging buffer inside the Vulkan step, read after its fence | the next CUDA step's input copy 14.2 → 3.5 ms a step |

- 107 ms is still slower than the CPU alone (86 ms) on this model.
- The smoke's own tok/s swings from run to run (7.9-13 tok/s with the CPU alone), because it has no NUMA placement.
  So the profile is the evidence here, not the smoke's speed. The greedy text matches the CPU-only run on one prompt;
  on the other it differs from character 412 on, in every bank arm. We read that as rounding, not yet proven.
- On V4.1, moving the sum to the CUDA card ran the 1080 Ti out of memory at warm-up (+1.7 GiB of compute buffer), so
  there the sum stays on the MI25. The next V4.1 run has the other fixes on.
- A model whose experts carry biases (gpt-oss-20b) skips the bank path in this build, so it is no test of the banks.
- Our ceiling estimate before the runs below, not measured: the MI25 holds the experts behind 31.6 % of CPU expert
  reads. If its work hid completely behind the CPU's, that would save about 21 ms of the 143 ms a token takes without
  the draft on V4.1 without the MI25, so about 8 tok/s at most.

### The outcome: hop latency, not bandwidth

Two more causes, found and fixed (MEASURED with the engine's `--hotbank-perf` timing of the V4.1 bank chain):
- **Clocks.** Under the driver's `auto` DPM level the MI25 sat at 300 MHz core and 167 MHz HBM through bursty decode.
  Forced to `high` (1500 / 945 MHz), a 1-token bank chain took 423 instead of 1,387 microseconds (3.3x).
- **The Vulkan MoE kernel.** For batches of a few tokens, `mul_mat_id` ran the tiled matmul path: 11.1 ms for a
  5-token chain. The mat-vec kernel per token (`GGML_VK_MMVID_MAX_TOKENS`, default 8) takes 1.30 ms.
- **Bank-first did not reorder.** `LLAMA_HOT_BANK_FIRST=1` was meant to start the MI25 before the CPU experts. The
  graph's split map showed CPU, then Vulkan, then CUDA in every layer: the bank output was pulled in only by the
  final sum. The fix ([`1aa46479`](https://github.com/akapug/ik_llama.cpp/commit/1aa464798f0406722ff485f05ea36a3bc7c0e003))
  adds every partition to the graph in order.

V4.1 on box D with the 14 GiB of hot experts on the MI25, the 1080 Ti layout of
[moe-offload point 17](../moe-offload/#what-we-learned-measured), 3 coding-agent prompts of 300 greedy tokens, one
load per row. Draft = the DSpark draft model at n_max 4.

| arm | draft tok/s | no draft tok/s |
|---|---|---|
| no MI25 (point 17) | **7.13** (7.24, 7.46, 6.70) | **6.97** (6.95, 7.00, 6.95) |
| MI25 banks, `auto` clocks, before the kernel change | 2.52 (2.71, 2.76, 2.10) | 4.41 (4.55, 4.45, 4.24) |
| + `high` clocks + the mat-vec kernel | 6.05 (6.31, 6.52, 5.31) | 5.46 (5.62, 5.10, 5.65) |
| + bank-first that reorders | 6.19 (6.27, 6.81, 5.48) | 5.61 (5.61, 5.58, 5.64) |

The engine's op profiler, per decode step without the draft, in the last arm: the banks save about 13.5 ms of CPU
expert time (31.6 % coverage cuts the CPU expert time by about 14 %). They add about 40 ms of waiting across devices:
the Vulkan steps take 20.5 ms, the bank's down projection 11.6 ms as its own step, and the sum on the CUDA card 20 ms
against 11.8 ms without banks. So the MI25 adds more latency per token than it removes.

**Why the same card wins on Ornith-35B and loses on V4.1.** Our reading of the two profiles: on V4.1 the MI25 can
hold only 31.6 % of the routed traffic, so every layer still runs its CPU expert pass, and the bank adds hops in
series with it. On Ornith-35B the two cards hold every expert, so the CPU expert pass and its hop go away. A bank
store pays when it removes a whole step per layer, not when it shortens a step that still runs. We stopped the
MI25 lever on V4.1; the best V4.1 recipe on box D stays at 7.13 / 6.97 without it.

## Heat and power

Sensors read once a second from the driver: edge, junction and HBM temperature, card power, core clock.

| phase | length | edge max | junction max | HBM max | card power (median) | core clock |
|---|---|---|---|---|---|---|
| idle | 5 min | 43 C | 43 C | 44 C | 3 W | 300 MHz |
| Llama 2 7B Q4_0 decode, with gaps for model loads | 60 s | 66 C | 73 C | 76 C | 110 W | 991-1138 MHz |
| same, sustained | 143 s, then stopped by our guard at HBM 88 C | 79 C | 90 C | 88 C | 109 W | 991-1138 MHz |

A second run kept the card decoding continuously (`tg2048` calls) and stopped at HBM 93 C (the HBM's critical mark
is 95 C, the junction's 105 C):

| time under load | 0:30 | 1:00 | 1:30 | 2:00 | 2:30 | 3:00 | 3:30 | stop 3:50 |
|---|---|---|---|---|---|---|---|---|
| junction C | 63 | 72 | 76 | 80 | 84 | 88 | 90 | 89 |
| HBM C | 68 | 75 | 82 | 85 | 87 | 90 | 91 | 93 |

- The rise slows every 30 s, so the shroud's airflow is working and the card heads for a plateau.
- Estimate, from an exponential fit over 200 s of samples (fit error under 0.7 C): HBM would level at about 94.5 C,
  junction about 98 C. That sits at the HBM's critical mark. With this cooling, minutes of full load are safe;
  sustained full load is marginal.
- The core clock stayed at 991-1138 MHz of 1500 throughout, at the 110 W cap. The clock is limited by power, not
  heat: no thermal clock drop was seen.
- Once idle, the card cooled from 88 C to 45 C in about 100 s.
- The short-burst benchmarks above peaked at 51-61 C junction.

## To reproduce

- A kernel whose `amdgpu` starts the card. If yours logs "PSP load sos failed", the `psp_v3_1.c` wait-time change
  above is what worked for us.
- Mesa RADV (we ran 25.2.8) and llama.cpp built with `-DGGML_VULKAN=ON`. We used `1d88cb0`.
- Airflow through the heatsink, and a temperature log while you run. Start each benchmark with the HBM cool, and
  stop at about 90 C HBM: hot and cool runs of the same benchmark differ.
- Models: `Qwen3-1.7B-Q8_0`, `llama-2-7b.Q4_0`, `gpt-oss-20b-MXFP4`, `Qwen3-30B-A3B-Q4_K_S`, all public GGUFs.

## Not measured yet

- Banks for experts with biases (gpt-oss), and the Vulkan `SWIGLU_OAI` path they need.
- Why the external DFlash draft for Ornith-1.5-35B-A3B is almost never accepted in this engine.
- Ornith-1.5-35B-A3B on the two cards as an agent on real repository work, and at depths beyond 21k.
- The ROCm path on this card, for a same-build comparison with Vulkan.
- Decode at 32k depth and beyond.
- Whether a WX9100 vBIOS (a 170 W cap) helps. Decode mostly reads memory, so the 110 W cap may cost little there,
  but we have not measured it.

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [arc](https://github.com/akapug/arc) + [fab](https://github.com/akapug/fab) for [@akapug](https://github.com/akapug)
