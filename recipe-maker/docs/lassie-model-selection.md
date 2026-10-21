# Lassie: fast middle-sized MoEs before capacity-sized giants

Research snapshot: 2026-10-20 owner date (collection continued on 2026-10-21 UTC).
This is a model-selection report, not a new benchmark. No box, service or model state was changed for this research.

**Outcome, measured after this report (2026-10-21):** both candidates then ran on lassie, with 3 coding-agent prompts
of about 1.5-2.2k tokens, 300 greedy tokens each, an 8,192-token window and 64 threads, no draft unless named.
- Ornith-1.5-35B-A3B Q4_K_M: 41.99 tok/s with every routed expert on the CPUs (the table below put it at 29.4,
  range 18.4-42.5); 44.62 with all of them in banks on the MI25 and the GTX 1080 Ti; 59.13 with that layout and the
  model's own MTP head drafting 3 tokens (acceptance 85 %).
- gpt-oss-120b MXFP4: 31.0 tok/s with experts on the CPUs (the table: 14.2, range 9.4-19.1), 34.6 with expert
  layers 0-2 on the 1080 Ti.
- Layouts, switches and conditions: the ik_llama.cpp fork's
  [PATCHES.md](https://github.com/akapug/ik_llama.cpp/blob/mixed-cuda-vulkan-expert-banks/PATCHES.md). Both rates
  beat this report's scenarios, so the bandwidth-efficiency assumption below is too low for lassie.

## Recommendation

The next two bounded qualification candidates are **Ornith-1.5-35B-A3B Q4_K_M** and
**gpt-oss-120b native MXFP4**. The first tests the speed/quality knee with only 0.610 GB of routed
expert reads per token; the second spends about three times that traffic for a potentially stronger
reasoning model without stepping up to the giant-model latency class. This is a test priority, not
proof either outperforms DeepSeek V4.1 Flash on agentic coding. Confidence in that experimental
ordering: 85% engineering judgment, not a statistical probability of winning.

Flash-Next unpruned is the close alternative to the second candidate if coding, rather than broad
reasoning, is the sole selection criterion. It is already in Recipe Maker. Its predicted speed depends
strongly on the actual quant/kernel; the pruned Coder's smaller store did not make its existing planner
frontier faster. Keep Qwen3-30B-A3B as the measured fast control, not as a newly proven strong coder.

Lassie's likely useful niche is **moderate active-byte MoEs with GPU-resident dense work and DRAM
experts**, not merely the largest model that fits in RAM. Large capacity remains valuable for occasional
quality-first work, but 1 TB does not make a large active set interactive. Actual task completion time
also includes prefill, repeated reasoning tokens, tool latency and retries, none of which a decode-only
ranking captures. The same two runs can test NUMA placement, the speed/quality knee and household
versus coding suitability, rather than buying hardware before measuring the bottleneck.

## Measurement anchors and calibration

Retained local evidence, not publisher benchmarks:

- V4.1 fs3, T64, routed experts split by rows across eight NUMA nodes, dense/shared/head/KV on the
  GTX 1080 Ti: **6.97 tok/s without draft**, three 300-token prompt rates 6.95/7.00/6.95;
  **7.13 with draft**, rates 7.24/7.46/6.70; prefill 48.2 tok/s. One load, not a stability campaign.
  The sm_61 flash-attention operation falls back to CPU; tensor placement does not prove every
  attention operation runs on the GPU. No OOM/Xid was recorded in that cell.
- Qwen3-30B-A3B Q4_K_S smoke: CPU interleave8 19.26/19.09, CPU rows8 21.94, GPU dense plus
  interleaved experts 31.72, **GPU dense plus rows8 39.53 tok/s**. One prompt, 128 generated
  tokens, T64; identical greedy text. This is not an agent-quality qualification.
- GLM-5.3-Flash UD-Q3_K_XL: existing Recipe Maker Lassie anchor **8.63**, or 9.65 with MTP1.
  Conditions were not recorded; do not relabel it fs3/rows8 or compare it as a matched arm.
- LLaDA2.2 Flash Q4_0: actual CPU-only fixed4x256 rows **3.95 cache / 4.08 fuse tok/s**;
  different block/schedule work from autoregressive decode. No GPU/rows8 claim.

V4.1's Q8_0-named store has **native MXFP4 routed experts**, not eight-bit experts. Its routed
store is 268.95 GiB, top6 of 384. Approximate routed traffic is
`268.95 * 2^30 * 6 / 384 = 4.512e9 bytes/token`.
At 6.97 tok/s that is a **31.45 GB/s whole-path expert-rate proxy**, 12.2% of the retained
257.7 GB/s NUMA-local raw read result. It is not a measured hardware bandwidth efficiency.
MXFP4 weight packing is 4.25 bits/weight, not 4.5.

For a transparent sensitivity calculation, let `T = F + active_GB / B`:

- Nominal `F = 16.90 ms`, borrowed from Recipe Maker's estimated Lassie V4.1 host term.
- Solving against fs3 gives nominal `B = 35.65 GB/s`.
- Scenario endpoints below use `(F=30 ms, B=25 GB/s)` and `(F=10 ms, B=45 GB/s)`.

Neither F nor B is independently identified by one anchor. These are **INFERRED scenarios, not
confidence intervals**. Applying one kernel's efficiency across architectures is particularly weak.
CPU-only source-MXFP4 and split-store evidence already shows compute/synchronization can outweigh
byte savings. The nominal point should not be used as an admission or purchasing gate. Scenario
confidence is at most 65% for the small known-architecture candidates, lower for new architectures.

## Fixed-reference candidate table

All GB below are decimal routed expert bytes per generated token, excluding GPU-resident dense
work, lookup rows and KV traffic. A header-backed quant is distinguished from an assumed uniform
Q4_K expert packing of 4.5 bits/weight. A range is an arithmetic scenario, **not a measured rate**.
Coding and household labels are evidence tiers, not normalized benchmark scores.

| Priority / family | Reference precision | Routed GB/token | Nominal n0; scenario tok/s | Coding evidence | Household evidence / qualification | Confidence in placement of candidate |
|---|---|---:|---|---|---|---:|
| Fast control: Qwen3-30B-A3B | Q4_K_S | not header-read here | **MEASURED 39.53 smoke** | Local coding UNKNOWN | Fast short decode demonstrated, task quality UNKNOWN | 95% smoke fact; not a quality rank |
| 1: Ornith-1.5-35B-A3B | exact Q4_K_M, mixed Q4/Q6 down | 0.610 | **INFERRED 29.4; 18.4–42.5** | B70 local agent evidence exists; not a Lassie run | B70 household evidence exists; text/vision paths differ on Pascal | 85% next-test priority; 65% rate scenario |
| Alternate: Qwen3.8 Flash-Next | existing approximate 4.25-bpw reference | 1.253 | **INFERRED 19.2; 12.5–26.4** | Local B70 coding cells and publisher claims, not Lassie qualification | UNKNOWN on Lassie | 80% priority; 60% rate |
| Alternate: Flash-Next Coder | GSQ-RCO IQ3_S, pruned 256/512 experts | 1.032 | **INFERRED 21.8; 14.0–30.4 before kernel penalty** | Pruning and low-bit quant both affect quality; no quality-equality proof from equal n=1 outcomes | UNKNOWN | 65%; kernel penalty not identified on Lassie |
| 2: gpt-oss-120b | exact native MXFP4 + F32 expert biases | 1.909 | **INFERRED 14.2; 9.4–19.1** | Publisher reasoning/tool evidence; Lassie agent quality UNKNOWN | Potential stronger general reasoning; Harmony template/parser required | 85% next-test priority; 60% rate |
| Ornith-1.5-122B | exact Q4_K_M mixed Q4/Q6 | 2.194 | **INFERRED 12.8; 8.5–17.0** | Local three FAIL/no-edit, two VOID/OOM cells are **configuration-confounded**, not intrinsic model failure or a proven coder | UNKNOWN on Lassie | 75% worth testing after config cure; 60% rate |
| gpt-oss-20b | exact native MXFP4 + F32 expert biases | 1.272 | **INFERRED 19.0; 12.4–26.1** | Less reasoning capacity than 120b by publisher positioning; local agent quality UNKNOWN | Useful smaller control, but not the smallest expert traffic | 75%; 60% rate |
| GLM-5.3-Flash | exact UD-Q4_K_XL reference | 5.152 | **INFERRED 6.2; 4.2–8.0** | Stronger local evidence on other boxes; specific low-bit quality gates matter | Existing **MEASURED 8.63 Q3** anchor is a different quant with missing conditions | 90% retain as known candidate; 55% Q4 transfer |
| DeepSeek V4.1 Flash | native expert MXFP4, other weights Q8_0 | 4.512 | **MEASURED 6.97 n0 / 7.13 n4** | Reference model; Lassie agent-quality sweep not performed here | Quality-first baseline, not the fast floor | 95% retained speed fact |
| MiniMax-M2.7 | assumed Q4_K expert store | 3.950 | **INFERRED 7.8; 5.3–10.2** | Publisher agent benchmarks; local UNKNOWN | No clear speed win over reference after dense/attention cost | 55%; 45% runnable-rate confidence |
| MiniMax-M3 | assumed Q4_K, 57 routed layers | 7.262 | **INFERRED 4.5; 3.1–5.8 only if dense placement works** | Publisher claims; local UNKNOWN | Experimental MSA branch changes attention semantics/performance; not qualified | 45%; do not schedule before engine/fit probe |
| Kimi K2.5 | assumed Q4_K routed experts | 11.891 | **INFERRED 2.9; 2.0–3.6 only if dense placement works** | Publisher coding evidence; local UNKNOWN | Capacity/research candidate, not interactive speed target | 55% low priority; 40% runnable rate |
| Kimi K3 | native MXFP4 assumed routed weights | 25.830 | **No in-DRAM admission**; arithmetic 1.35 before spill | Publisher strong agent boards with harness/reasoning caveats | Routed store alone about 1.45 decimal TB, above nominal 1 TB before other tensors; lower bits/spill required | 90% not the speed candidate at release precision |
| Mixtral 8x22B | assumed Q4_K experts | 19.025 | **INFERRED 1.8; 1.3–2.3 before dense constraints** | Older publisher code benchmarks, no local qualification | High top2/8 read share defeats low-active-byte objective | 85% low priority |
| Mistral Large 3 | BF16 release; hypothetical Q4_K | about 11.494 at Q4 | **No numeric runnable prediction** | Publisher coding use case, no matched local test | Large dense/shared/attention fit and engine path unqualified; not obviously faster | 80% low speed priority |
| Llama 4 Scout / Maverick | no exact GGUF read in this survey | UNKNOWN | **UNMODELLED** | Publisher benchmarks; gated config unavailable anonymously | 109B/17B Scout claim does not reveal routed DRAM bytes or prove Pascal engine fit | 60% deprioritize until exact metadata |
| Kolibri-1 | assumed Q4_K experts | 0.664 | **INFERRED 28.2; 17.7–40.4 arithmetic only** | Local UNKNOWN | Attractive traffic, engine/kernel support not qualified; do not promote to runnable GO | 50% research lead, below next-two bar |

Confidence below 50% means **do not act on a runnable-speed claim**; inspect metadata/engine first.
It does not mean the model is bad. For Kimi K3 the latent routed width is **3584**, not attention
width 7168: `92 * 16 * 3 * 3584 * 3072 * 4.25/8` bytes/token. Mistral's pinned vendor `params.json`
provides 61 layers, three initial dense, 128 experts top4, hidden 7168 and expert FF 4096;
`58 * 4 * 3 * 7168 * 4096 * 4.5/8` is only hypothetical Q4 routed traffic. Its source uses
`params.json`, not `config.json`; the initial config 404 was not absence of the model.

## Every pre-existing Recipe Maker family, including diffusion

The actual solver was run on the pre-UX source at f42cebc with `ik`, no draft, no moved B70.
It returned frontier maxima: V4.1 8.55, GLM 10.29, Flash-Next 17.33, pruned Coder 13.38,
Ornith122 14.26 tok/s. **These select differing quantizations/layouts, not the table's fixed references.**
The Lassie calibration had one GLM n0 anchor for two parameters and **no Lassie V4.1 anchor**;
V4.1 and Flash-Next host terms were borrowed. No frontier quality percentage is a measured coding
score. The pruned Coder's GSQ kernel efficiency illustrates why active-byte ranking alone can invert.
V4.0 (`show=False`) is a retired calibration-only family, not a deployment recommendation.

Diffusion intake is separate from autoregressive `models`:

| Existing entry | Reference expert GiB / dense GiB | Lassie outcome | Selection consequence |
|---|---:|---|---|
| llada21mini | 8.016 / 0.440, Q4_0 | no Lassie row inspected | UNKNOWN, no AR speed substitution |
| llada21mini_q5km | 10.096 / 0.486, Q5_K_M | no Lassie row inspected | UNKNOWN, same architecture/different quant |
| llada22flash | 52.312 / 1.511, Q4_0 | **MEASURED CPU cache 3.95, fuse 4.08** | Existing measured cells do not beat the interactive AR control |
| lingmini2 | 8.016 / 0.440, Q4_0 | no Lassie row inspected | UNKNOWN; do not infer identical quality from matching shape |
| dreamcoder7b | no routed experts / 3.836, Q4_0 | no Lassie row inspected | Dense diffusion; block forwards/reuse differ from AR |

A diffusion token can require multiple forward passes, while a block may produce several tokens per
forward. `bandwidth / active bytes` alone cannot produce its generation rate. These five entries are
covered explicitly rather than silently excluded from “every family.”

## Ternary and other low-bit paths

- **Ternary Bonsai 27B** is dense, derived from Qwen3.6-27B, not a routed-expert MoE.
  The publisher's 1.71 bpw / 5.9 GB are theoretical coding density; its currently described
  group128 file is about **7.17 GB at 2.125 bpw**. Most weights are read each token. Whole-model
  GPU placement could change the bottleneck, but custom PQ2_0 kernels need the Prism fork;
  upstream group64 is a different file. Neither H100 numbers nor a CUDA build proves sm_61.
  Publisher HE+/LCB retention is not long-horizon repository-agent qualification; the card names
  multi-file/test-repair work as a weakness.
- **Ternary Bonsai 8B**: about **2.18 GB / 2.03 GiB** group128 file. The card describes CPU/Metal
  kernels in its fork, not demonstrated Pascal Q2_0 support. Publisher M4 Pro 76 GPU / 32 CPU tok/s
  is not a Lassie estimate. A CPU implementation may be the next compatibility probe.
- **BitNet b1.58 2B 4T**: verified official configuration, tiny dense native low-bit research/control,
  not evidence of parity with the smart MoEs. Native kernels and template support matter; ternary
  training does not make an arbitrary Q2 GGUF quality-equivalent.
- **Bonsai 2** listings exist, but no pinned architecture/header or local agent evidence was collected
  here. They remain explicit follow-up leads, not made-up ranked measurements.
- **Nemotron 3.5 Bolt** metadata exceeded the bounded configuration reader; no larger download or
  engine conclusion followed. It remains UNMODELLED, not nonexistent.
- Two-bit versions of a large MoE trade quality and kernel efficiency, not just capacity. No new KLD
  measurements were performed; never assign zero quantization loss to admit a candidate.

## What the source extension admits

- `families/orn35.py` reuses the existing qwen35moe family contract with its own pinned, sanitized
  header, exact mixed quant store, 40 main layers plus nextn block 40, GDN state and context terms.
  Host/compute/KLD analogs and synthetic routing stay **ESTIMATE**; local quality **UNKNOWN**.
  Unknown speculative acceptance stays null; the neutral draft multiplier is not a measured gain.
- `data/models/gptoss20.json` and `gptoss120.json` retain native weight **and F32 bias** bytes.
  They are metadata intake only, without a runnable planner family, fake KLD or borrowed “GO.”
  Existing `harvest.roofline_rows` exposes them only as **UNMODELLED**, engine=null, decode_tps=null.
  GPT-OSS expert biases are incompatible with the current bias-skipping hot-bank path; no MI25
  acceleration is assumed. Harmony formatting must be qualified at the tool boundary.
- No generic family alias maps the base Qwen model onto an Ornith fine-tune. Shared shapes are not
  shared quality evidence. No raw tokenizer templates or upstream local imatrix paths were retained
  in the new sanitized header fixture.

## Next qualification, owned by the box operator

Run each candidate without speculative decoding first, on one fixed launch/context/prompt set,
then measure matched rows8 versus interleave and optional draft separately. Record exact source,
quant, template, engine commit/build/ISA, dense and attention operation placement, context/KV,
threads, NUMA placement, prefill, decode, TTFT, peak memory and cleanup/restore. Before a large
load, prove native-format engine compatibility and the real dense/KV memory budget.

Use the same bounded coding tasks and the same household tasks, separate scoreboards, useful
answer/completion time, capped reasoning policy and failure reasons. Ornith122's previous 45-minute
walls, 8192-token thinking cap at about 18–21 tok/s, 262k Q8 KV context on a 24 GB card, and
66–76k prefill OOMs confound its no-edit cells. A corrected-configuration rerun is already separate
work; neither hide the failed cells nor call them intrinsic model failure. None of these future runs
was launched by this report.

## Sources

Local evidence: retained research STATE sections for V4.1 fs3, Qwen30 smoke and configuration-limited
Ornith122; Recipe Maker `build.py`, `src/model.js`, families, intake tables and `runs/dllm-raw/lassie/`.

Pinned header sources:

- [Ornith35 Q4_K_M, f6045267](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF/tree/f60452672689e0040be350560267a614cc172a18)
- [GPT-OSS120 native MXFP4, 7200b74b](https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/tree/7200b74bea78b0a2964ef9f3897eefa401e890ff)
- [GPT-OSS20 native MXFP4, 15dcfc7e](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/tree/15dcfc7e7c608c6161507211ed43c5b590801b4b)
- [Mistral Large 3 vendor params, d40f4a01](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512-BF16/blob/d40f4a01762df1a55ad689849582b3a3d2b21a7b/params.json)

Publisher cards and engine discussions (mutable; claims attributed, not local qualification):

- [Ornith35](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), [Ornith122](https://huggingface.co/ornith-ai/Ornith-1.5-122B)
- [Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), [GLM](https://huggingface.co/zai-org/GLM-5.3-Flash)
- [GPT-OSS120 and Harmony](https://huggingface.co/openai/gpt-oss-120b), [ik GPT-OSS support/evidence](https://github.com/ikawrakow/ik_llama.cpp/discussions/758)
- [MiniMax M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7), [experimental M3 MSA](https://github.com/ikawrakow/ik_llama.cpp/discussions/2046)
- [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3), [Kimi K2.5](https://huggingface.co/moonshotai/Kimi-K2.5)
- [Mistral Large 3](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512-BF16), [Mixtral8x22](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)
- [Llama4 Scout](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct), [Kolibri1](https://huggingface.co/Aleph-Alpha/Kolibri-1)
- [Bonsai27](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf), [Bonsai8](https://huggingface.co/prism-ml/Ternary-Bonsai-8B-gguf), [Bonsai listings](https://huggingface.co/prism-ml)
- [BitNet2B](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T), [Flash-Next ik discussion](https://github.com/ikawrakow/ik_llama.cpp/discussions/2615)
