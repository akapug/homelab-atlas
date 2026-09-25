# Homelab Atlas

Measurements, patches and tools for running local LLMs on hardware people can actually get:
older workstations, used server parts, single prosumer GPUs. Not a rack of H100s.

The bet behind it: as models lean on sparsity, state-space layers, speculative decoding and low-bit
weights, memory bandwidth and software matter more than raw GPU compute, and a lot of capability
is left on the table on modest hardware because nobody tuned for it. Each entry here is a
measured case of that, with the patches that closed the gap and the commands to check our numbers.

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
| [**Intel Arc Pro B70**](b70/): Qwen3.8-27B and Qwen3.6-35B-A3B | vLLM on Intel's llm-scaler image with [five changes](b70/vllm-xpu/): compiled mode (llm-scaler #699), AutoRound int4 checkpoints on the fast int4 kernels (61 tok/s at one stream, 351 at eight), a smaller vocabulary for the speculative drafter (+25-29 % at one stream), and two tool-call fixes for agent clients. [18 llama.cpp SYCL patches](b70/llama.cpp-sycl/), 39 → 101 tok/s at four streams. [The MoE](b70/qwen36-35b-a3b.md) at 2.1-5.1x llama.cpp Vulkan with its full 262k window. [An agent eval](b70/agent-eval/): can a local model finish real repository work? [The benchmarks](b70/bench/) behind all of it. |

## How numbers are reported here

- Every number names its conditions: card, software version, model and quantization, context
  depth, stream count, speculation settings.
- Speed claims come with a correctness check on the same configuration: a new kernel path or a
  compiled mode that is faster and wrong is worse than useless.
- Measured means measured. Estimates are labelled as such, and so are the limits of what we tried.
- Repeated runs are all reported, as ranges.

Issues and corrections are welcome, especially numbers from other cards.

---

Measured and written by Claude Opus 5.5 agents working in [helm](https://github.com/akapug/helm)
for [@akapug](https://github.com/akapug).
