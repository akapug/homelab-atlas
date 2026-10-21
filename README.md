# Homelab Atlas

Measurements, patches and tools for running local LLMs on hardware people can actually get:
older workstations, used server parts, single prosumer GPUs. Not a rack of H100s.

The bet behind it: as models lean on sparsity, state-space layers, speculative decoding and low-bit
weights, memory bandwidth and software matter more than raw GPU compute, and modest hardware often
runs well below what it can do because few people tune for it. Each entry here is a measured case
of that, with the patches that closed the gap and the commands to check our numbers.

**Try Recipe Maker in your browser:** [akapug.github.io/homelab-atlas](https://akapug.github.io/homelab-atlas/).
Pick one of our four boxes and see which layouts of a big MoE model fit it, how fast each should run, and how
the predictions compare with our measured runs. Nothing to install.

## Highlights

Each line is one measured configuration; the linked entry gives its conditions, its correctness check and the
commands.

- Qwen3.8-27B on one Intel Arc Pro B70: 61 tok/s at one stream and 351 at eight (vLLM, AutoRound int4, MTP 3).
  [B70](b70/)
- Qwen3.8 Flash-Next on one B70 with about half its experts computed on the CPU: 87.1 tok/s, passing 11 of our
  12 agent tasks. [MoE offload](moe-offload/)
- GLM-5.3-Flash on an RTX 3090 Ti plus 192 GB of DDR5: 23.08 tok/s. [MoE offload](moe-offload/)
- Ornith-1.5-35B-A3B with every routed expert on a $50 AMD MI25 and a GTX 1080 Ti, in a 2017 server:
  59-60 tok/s with the model's own draft head, against 47 with the same draft and the experts on the CPUs.
  [MI25](mi25/)
- DeepSeek V4.1 Flash on a 2019 dual-Xeon workstation (one RTX 6000, DDR4 and Optane): 11.9 tok/s. On a
  two-socket EPYC server with a GTX 1080 Ti and its experts as released: 7.13. [MoE offload](moe-offload/)

## Find your patch

[PATCHES.md](PATCHES.md) lists every patch and tool here by the hardware it needs, then the software,
with the symptom it fixes and how to apply it, check it and undo it. Start from your hardware:

| your hardware | what applies | where |
|---|---|---|
| Intel Arc Pro B70 (our card), with vLLM | two Intel-only fixes, four fixes that are not tied to a GPU, two settings for Intel's `xe` driver | [B70 with vLLM](PATCHES.md#2-intel-arc-pro-b70-with-vllm), [any GPU](PATCHES.md#1-any-gpu-serving-software-and-chat-template-fixes) |
| Intel Arc Pro B70, with llama.cpp | 18 patches and a build script | [B70 with llama.cpp](PATCHES.md#3-intel-arc-pro-b70-with-llamacpp) |
| Other Intel Arc (B580, Pro B60, A-series) | untested, with what may carry over and why | [other Intel Arc](PATCHES.md#other-intel-arc-cards) |
| NVIDIA: our Quadro RTX 6000 (Turing) and GTX 1080 Ti (Pascal), or others | the fixes not tied to a GPU (untested on NVIDIA), a Turing fix sent to another project, the tools | [NVIDIA](PATCHES.md#nvidia-quadro-rtx-6000-turing-24-gb) |
| AMD or Apple GPUs | the fixes not tied to a GPU (untested there), the tools | [other GPUs](PATCHES.md#amd-apple-and-other-nvidia-gpus) |
| A Ryzen with two CCDs that serves a model and also runs builds | a CPU guard | [host CPU](PATCHES.md#4-the-host-cpu-a-ryzen-with-two-ccds) |
| A used AMD Radeon Instinct MI25 (Vega 10, 16 GB HBM2) | a cold-boot fix for `amdgpu`, llama.cpp Vulkan numbers, heat; a MoE model's experts split between it and a CUDA card | [MI25](mi25/) |
| A MoE model too big for your VRAM, on any box | a planner that places it across VRAM, DRAM, PMem and disk before you run it | [Recipe Maker](PATCHES.md#recipe-maker-plan-a-moe-layout-before-you-run-it) |
| Any, or none: you want to test a local model on real work | the agent eval, benchmarks and probes | [tools](PATCHES.md#6-tools-that-need-no-particular-gpu) |

Before you start:
- A Linux machine where the GPU driver works. For the B70 that is the kernel's `xe` driver: `lspci -k`
  shows `Kernel driver in use: xe` for each card.
- For vLLM on the B70: Docker and Intel's `intel/llm-scaler-vllm:0.26.0-b2` image. The vLLM patches are
  written for exactly this version. For llama.cpp on the B70: git and Intel's oneAPI Base Toolkit.
- A model from Hugging Face, for example `Frozenlock/Qwen3.8-27B-int4-AutoRound` (18 GB) for vLLM, or
  `Qwen3.8-27B-UD-Q4_K_XL.gguf` from `unsloth/Qwen3.8-27B-GGUF` for llama.cpp.
- To drive it from Claude Code: a proxy that translates Anthropic's API to OpenAI's, set up as in
  [Claude Code on it](b70/qwen36-35b-a3b.md#claude-code-on-it).

## Entries

| entry | what is in it |
|---|---|
| [**Intel Arc Pro B70**](b70/): Qwen3.8-27B, Qwen3.6-35B-A3B and Ornith-1.5-35B-A3B | vLLM on Intel's llm-scaler image with [five changes](b70/vllm-xpu/): compiled mode (llm-scaler #699), AutoRound int4 checkpoints on the fast int4 kernels (61 tok/s at one stream, 351 at eight), a smaller vocabulary for the speculative drafter (+25-29 % at one stream), and two tool-call fixes for agent clients. [18 llama.cpp SYCL patches](b70/llama.cpp-sycl/), 39 → 101 tok/s at four streams. [The MoE](b70/qwen36-35b-a3b.md) at 2.1-5.1x llama.cpp Vulkan with its full 262k window. [Ornith-1.5-35B-A3B](b70/ornith-35b-a3b.md) serving a household assistant (web search, photos, voice) and a coding agent from one card, and a speculative-decoding bug that garbled 3 % of replies. [An agent eval](b70/agent-eval/): can a local model finish real repository work? [The benchmarks](b70/bench/) behind all of it. |
| [**Big MoE models on one GPU plus system memory**](moe-offload/): GLM-5.3-Flash, DeepSeek V4.1 Flash, Qwen3.8 Flash-Next, Ornith-1.5-122B | GLM-5.3-Flash at 23.08 tok/s on an RTX 3090 Ti plus 192 GB DDR5, and at 16.06 on a 2019 dual-Xeon workstation with an RTX 6000 and Optane; DeepSeek V4.1 Flash at 11.9 on the same workstation, and at 7.13 on a 1 TB two-socket EPYC server whose GTX 1080 Ti holds attention and the dense weights while its experts, as released, stay in DRAM; Qwen3.8 Flash-Next at 87.1 on one B70 (with half its experts computed on the CPU), passing 11 of our 12 agent tasks; Ornith-1.5-122B at 31.6 on the workstation (its first agent runs failed in a configuration that limited it; a re-run is queued). Twenty-three measured lessons: what binds, when speculative decoding loses, which layers tolerate fewer bits, splitting experts and dense weights across sockets and NUMA nodes, a build option worth +25-31 %, why verifying many drafted tokens at once does not pay on these boxes, an engine fix that lets a Pascal card run V4.1's attention, and the engine changes behind it all. |
| [**Recipe Maker**](recipe-maker/): plan a MoE layout for your box | The planner behind the moe-offload layouts: it probes a box read-only, reads a model's layout from its headers, places experts across VRAM, DRAM per NUMA node, PMem and NVMe, predicts decode speed and quality, and refuses layouts that would fail before you launch them. Every number is marked measured, inferred or unknown, and it scores its own predictions against the runs it records. Python 3 and node, nothing to install. |
| [**AMD Radeon Instinct MI25**](mi25/): a $50 16 GB card under Vulkan | **Ornith-1.5-35B-A3B at 59-60 tok/s with every routed expert on this card and a GTX 1080 Ti, two retired GPUs from rival vendors in one process in a 2017 server**, against 42-47 with the experts on the server's CPUs. A 2017 HBM2 card on llama.cpp Vulkan with no ROCm: a cold-boot driver fix, a quant ladder against a GTX 1080 Ti, context-depth limits, a 30B MoE at 36.7 tok/s with all but two layers of experts on the card (3.6x the server's CPUs) and 71-74 tok/s split with the 1080 Ti, and its heat at the 110 W cap. A negative result: as a store for 14 GiB of DeepSeek V4.1's hot experts it stays slower than the box without it, because the device hops cost more than they save. |

## How numbers are reported here

- Every number names its conditions: card, software version, model and quantization, context
  depth, stream count, speculation settings.
- Speed claims come with a correctness check on the same configuration: a new kernel path or a
  compiled mode that is faster and wrong is worse than useless.
- Measured means measured. Estimates are labelled as such, and so are the limits of what we tried.
- Repeated runs are all reported, as ranges.

Issues and corrections are welcome, especially numbers from other cards.

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [arc](https://github.com/akapug/arc) + [fab](https://github.com/akapug/fab) for [@akapug](https://github.com/akapug)
