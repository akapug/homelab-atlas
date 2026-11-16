# Tools we use, adapt or tested

Outside software this Atlas runs on or has tried: serving engines, model formats, agent harnesses and
memory tools. [PATCHES.md](PATCHES.md) lists our own patches; this page is the other side, the tools
themselves. Each entry says what the tool does, what we ran it on, what we found and where the evidence
is, and our verdict:

- **use**: we run it as it ships.
- **adapt**: we run it with our patches or settings (named in the entry).
- **forked**: we keep a fork with changes too big for patches.
- **testing**: being tried now; no measured result yet, so no verdict.
- **dropped**: tried and set aside, with the reason.

An entry is added when a result names a tool, and updated when a newer result changes the verdict.

## Serving engines

| tool | what it does | what we ran it on | verdict | evidence |
|---|---|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | Python serving engine, strong at many requests at once | RTX 3090 Ti (official 0.31.0 CUDA image) | adapt: two of our any-GPU patches | [3090 Ti](3090-ti/README.md), [patches](PATCHES.md#nvidia-rtx-3090-ti-ampere-24-gb) |
| [llm-scaler](https://github.com/intel/llm-scaler) | Intel's vLLM image for Arc GPUs | two Arc Pro B70s, `intel/llm-scaler-vllm:0.26.0-b2` | adapt: Intel-only fixes plus the any-GPU patches | [B70 with vLLM](PATCHES.md#2-intel-arc-pro-b70-with-vllm) |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | C++ engine for GGUF files on almost any hardware | B70 (SYCL), MI25 (Vulkan), NVIDIA cards (CUDA), CPUs | adapt: 18 SYCL patches and a build script; attention-sink fixes on upstream master | [B70 with llama.cpp](PATCHES.md#3-intel-arc-pro-b70-with-llamacpp), [MI25](mi25/) |
| PrismML's llama.cpp fork | llama.cpp with ternary (PTQ1_0) kernels | one B70, Ternary Bonsai 2 27B | adapt: 22 SYCL patches | [Bonsai on a B70](b70/bonsai-2-27b.md) |
| [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) | llama.cpp fork strong at CPU and MoE offload | Quadro RTX 6000 + Optane, GTX 1080 Ti, big MoE models | forked: [akapug/ik_llama.cpp](https://github.com/akapug/ik_llama.cpp) | [RTX 6000](rtx-6000/README.md), [1080 Ti](1080-ti/README.md) |
| Strata / XeStrata | engine for Qwen3.8 Flash-Next: experts in RAM, a VRAM expert cache; XeStrata is its Intel port | two B70s, many agents batched | forked: experimental batching changes | [many agents](b70/many-agents.md), [host-submission-bound decode](b70/host-submission-bound-decode.md) |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) | MoE expert offload between CPU and GPU | not run; read as prior art | reference only | [Recipe Maker prior art](recipe-maker/docs/prior-art.md#ktransformers) |

## Model formats and compression

| tool | what it does | what we ran it on | verdict | evidence |
|---|---|---|---|---|
| [AutoRound](https://github.com/intel/auto-round) | 4-bit weight quantization | Qwen3.8-27B int4 on the B70 (vLLM) | adapt: our `inc-q40` patch puts it on the fast kernels | [`inc-q40`](PATCHES.md#2-intel-arc-pro-b70-with-vllm) |
| [REAP](https://arxiv.org/abs/2510.13999) | prunes MoE experts by router-weighted saliency | public REAP-pruned GGUFs; our own expert masks compared against it | testing | [quant priors](recipe-maker/QUANT-PRIORS.md) |

## Training on older cards

| tool | what it does | what we ran it on | verdict | evidence |
|---|---|---|---|---|
| [TheRock](https://github.com/ROCm/TheRock) | ROCm builds, including torch for older AMD GPUs | MI25 (gfx900) | use | [PyTorch per card](PATCHES.md#7-pytorch-on-older-and-other-cards-which-torch-build-works), [tail-portable](tail-portable/) |

## Agent harnesses and memory

| tool | what it does | what we ran it on | verdict | evidence |
|---|---|---|---|---|
| [Claude Code](https://github.com/anthropics/claude-code) | coding agent client; on a local model through a proxy that speaks OpenAI's API | every local model in the agent eval | adapt: settings only | [Claude Code on a local model](PATCHES.md#claude-code-on-a-local-model-settings-not-patches), [agent eval](b70/agent-eval/) |
| [pi](https://www.npmjs.com/package/@earendil-works/pi-coding-agent) | small, extensible coding agent; a short system prompt and few tools suit local models | local models on our servers | testing | none yet |
| [UniiChat](https://github.com/VictorTaelin/UniiChat) | one endless chat; every message summarized into a binary tree; each call sees a bounded view (up to 128 KB) and can zoom back to the original | a fork pointed at our local OpenAI-compatible servers, with a small local model writing the summaries | testing: does a ~32k bounded view keep a local model useful past its window? | none yet |
| [OptMem](https://github.com/VictorTaelin/OptMem) | memory as a tool: an append-only log of one-line notes plus a summary tree, read at session start | inside the pi harness, beside UniiChat | testing | none yet |
