# AMD Radeon Instinct MI25 (16 GB HBM2): a $50 card under Vulkan

The MI25 is a 2017 AMD datacenter card: Vega 10 (gfx900), 16 GB of HBM2 on a 2048-bit bus, about 484 GB/s on
paper, passive cooling. We paid about $50 for ours. AMD's current ROCm releases dropped gfx900, so this entry runs
it on Vulkan, with no ROCm installed. The question: what does a card this cheap add to an old server, alone, as a
hot-expert store for a MoE model, and beside a GTX 1080 Ti?

It sits in box D of [the MoE offload entry](../moe-offload/): 2x EPYC 7601, 1 TB DDR4, a GTX 1080 Ti. Its first
use there, holding hot experts of DeepSeek V4.1 Flash beside the 1080 Ti, is
[in progress](#as-a-hot-expert-store-for-deepseek-v41-in-progress): correct text, currently slower.

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

## As a hot-expert store for DeepSeek V4.1 (in progress)

Status: the text is correct, the speed is not there yet, and the fixes are in progress. Everything here is box D
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
- Our ceiling estimate, not measured: the MI25 holds the experts behind 31.6 % of CPU expert reads. If its work hid
  completely behind the CPU's, that would save about 21 ms of the 143 ms a token takes without the draft on V4.1
  without the MI25, so about 8 tok/s at most.

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

- The ROCm path on this card, for a same-build comparison with Vulkan.
- Decode at 32k depth and beyond.
- Whether a WX9100 vBIOS (a 170 W cap) helps. Decode mostly reads memory, so the 110 W cap may cost little there,
  but we have not measured it.

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [arc](https://github.com/akapug/arc) + [fab](https://github.com/akapug/fabric) for [@akapug](https://github.com/akapug)
