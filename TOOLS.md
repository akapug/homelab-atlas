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
| [vLLM](https://github.com/vllm-project/vllm) | Python serving engine, strong at many requests at once | RTX 3090 Ti (official 0.31.0 and 0.33.0 CUDA images) | adapt: two of our any-GPU patches | [3090 Ti](3090-ti/README.md), [Ornith reference](3090-ti/README.md#ornith-15-35b-a3b-on-vllm-the-batched-decode-reference), [patches](PATCHES.md#nvidia-rtx-3090-ti-ampere-24-gb) |
| [llm-scaler](https://github.com/intel/llm-scaler) | Intel's vLLM image for Arc GPUs | two Arc Pro B70s, `intel/llm-scaler-vllm:0.26.0-b2` | adapt: Intel-only fixes plus the any-GPU patches | [B70 with vLLM](PATCHES.md#2-intel-arc-pro-b70-with-vllm) |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | C++ engine for GGUF files on almost any hardware | B70 (SYCL), MI25 (Vulkan), NVIDIA cards (CUDA), CPUs | adapt: 18 SYCL patches and a build script; attention-sink fixes on upstream master | [B70 with llama.cpp](PATCHES.md#3-intel-arc-pro-b70-with-llamacpp), [MI25](mi25/) |
| PrismML's llama.cpp fork | llama.cpp with ternary (PTQ1_0) kernels | one B70, Ternary Bonsai 2 27B | adapt: 22 SYCL patches | [Bonsai on a B70](b70/bonsai-2-27b.md) |
| [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) | llama.cpp fork strong at CPU and MoE offload | Quadro RTX 6000 + Optane, GTX 1080 Ti, big MoE models | forked: [akapug/ik_llama.cpp](https://github.com/akapug/ik_llama.cpp) | [RTX 6000](rtx-6000/README.md), [1080 Ti](1080-ti/README.md) |
| Strata / XeStrata | engine for Qwen3.8 Flash-Next: experts in RAM, a VRAM expert cache; XeStrata is its Intel port | two B70s, many agents batched | forked: experimental batching changes | [many agents](b70/many-agents.md), [host-submission-bound decode](b70/host-submission-bound-decode.md) |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) | MoE expert offload between CPU and GPU | not run; read as prior art | reference only | [Recipe Maker prior art](recipe-maker/docs/prior-art.md#ktransformers) |
| [SGLang](https://github.com/sgl-project/sglang) | Python serving engine, strong at prefix reuse and structured output | not run on our cards; read as the engine a GLM-5.3-Flash XPU image names | reference only: an XPU stack for Flash-Next is named in one image's `build.sh:5` and nothing here has tested it | [vLLM on the B70](b70/vllm-xpu/README.md), [hot-expert maps](recipe-maker/HOT-EXPERT-MAPS.md#capture-per-engine) |
| ExLlama (exllamav3, EXL3) | quantization format and CUDA kernels for it, with an expert-offload server on top | one 3090 Ti: GLM-5.3-Flash EXL3 3.05 bpw, experts in a VRAM cache, the rest on the CPU | adapt: a CPU-lane image with our patches mounted over it (not published) | [GLM route trim](3090-ti/README.md#glm-53-flash-a-route-trim-clears-25-toks), [moe-offload](moe-offload/README.md) |
| Basalt (the Strata fork) | Strata with a reworked prefill path, aimed at real agent sessions | not run here; its author's RTX 5080 + 64 GB result is cited as outside evidence | reference only: the one CUDA prefill number for Flash-Next that is not ours | [split modes](b70/split-modes-and-backends.md), [quant priors](recipe-maker/QUANT-PRIORS.md) |
| [arcint](https://github.com/marfrit/arcint) | about 100 patches to llama.cpp's OpenCL backend: integer DPAS for k- and i-quants, Intel flash attention, fusions | one B70, dense Qwen3.8-27B: +26-27 % prefill and +9 % decode against stock SYCL | adapt: built as a bench binary from its main; its quality gates were not re-run | [split modes and backends](b70/split-modes-and-backends.md#arcint-opencl-against-stock-sycl-dense-27b) |

## Model formats and compression

| tool | what it does | what we ran it on | verdict | evidence |
|---|---|---|---|---|
| [AutoRound](https://github.com/intel/auto-round) | 4-bit weight quantization | Qwen3.8-27B int4 on the B70 (vLLM) | adapt: our `inc-q40` patch puts it on the fast kernels | [`inc-q40`](PATCHES.md#2-intel-arc-pro-b70-with-vllm) |
| [REAP](https://arxiv.org/abs/2510.13999) | prunes MoE experts by router-weighted saliency | public REAP-pruned GGUFs; our own expert masks compared against it | testing | [quant priors](recipe-maker/QUANT-PRIORS.md) |
| [Unsloth](https://unsloth.ai) | GGUF quantizer and a GGUF release series (the "UD" dynamic quants) | every UD-Q4_K_XL and UD-IQ3_XXS checkpoint in our panels: Qwen3.8-27B, Qwen3.8 Flash-Next, GLM-5.3-Flash, Ornith, DeepSeek V4.1 | use as weights; their method read, not run: the per-layer bumps are not published as code, and a plain uniform mix lands on about the same KLD curve | [quant recipes](recipe-maker/QUANT-RECIPES.md#2-what-unsloth-actually-does-from-their-docs--headers), [quant priors](recipe-maker/QUANT-PRIORS.md) |
| [MLX](https://github.com/ml-explore/mlx) | Apple's array framework and its model runtime | not run on our cards; it appears only as a format other people's pruned checkpoints ship in | reference only | [quant priors](recipe-maker/QUANT-PRIORS.md) |
| [MACaQ](https://github.com/huawei-csl/MACaQ) | calibrated low-bit quantization for weights and activations: a rotation, learned exponents, Hessian-aware flips; a weight-only "MACaQ-R" path refines selected GGUF codes | source-audited at a pinned commit; no quantization, conversion or eval has run | testing: the release audit only. Its main path needs CUDA and its Flash-Next path quantizes experts, not the IQ tensors we care about | [quant priors](recipe-maker/QUANT-PRIORS.md) |
| Basalt and Lamina draft checkpoints (a different Basalt from the fork above) | replacement dense + expert parts for Ornith's native MTP head: Basalt self-distils it, Lamina trains it further on 3-bit output | both staged beside R8 for a `--mtp` swap, on odie's two B70s, and neither was run | testing: byte-identical layout to the stock part, but the MTP arm missed its gate, so the ~3-point acceptance gain could not be weighed | [many agents](b70/many-agents.md#what-the-switches-do-and-what-did-not-work) |
| [Scry](https://scry.io) | read-only SQL over public corpora (Reddit, Hacker News, arXiv, X) | the research sweeps behind our engine and quant surveys | use: a research source, not a serving tool | none yet (the sweeps are our own notes) |

## Training on older cards

| tool | what it does | what we ran it on | verdict | evidence |
|---|---|---|---|---|
| [TheRock](https://github.com/ROCm/TheRock) | ROCm builds, including torch for older AMD GPUs | MI25 (gfx900) | use | [PyTorch per card](PATCHES.md#7-pytorch-on-older-and-other-cards-which-torch-build-works), [tail-portable](tail-portable/) |

## Agent harnesses and memory

| tool | what it does | what we ran it on | verdict | evidence |
|---|---|---|---|---|
| [Claude Code](https://github.com/anthropics/claude-code) | coding agent client; on a local model through a proxy that speaks OpenAI's API | every local model in the agent eval | adapt: settings only | [Claude Code on a local model](PATCHES.md#claude-code-on-a-local-model-settings-not-patches), [agent eval](b70/agent-eval/) |
| [Open WebUI](https://github.com/open-webui/open-webui) | chat front end for an OpenAI-compatible server: the house's own client, with web search, tools, photos and voice | the household questions in the Ornith entry, driven the way a phone drives it; the banner that says a server is off | use | [Ornith on a B70](b70/ornith-35b-a3b.md#thinking-photos-and-voice) |
| [DeepSeek's chat-demo](https://github.com/deepseek-ai/chat-demo) | static browser chat UI for the DeepSeek API: branching conversations, attachments, Markdown and KaTeX, themes | read, not run: its search, link reading, titles and voice are DeepSeek server endpoints, so against our local servers it is plain chat | reviewed, not adopted: a UX reference for a local front end, not a client we can use | none yet |
| [pi](https://www.npmjs.com/package/@earendil-works/pi-coding-agent) | small, extensible coding agent; a short system prompt and few tools suit local models | pi 1.2.0 against Claude Code on our own GLM-5.3-Flash server (EXL3 3.05 bpw on one RTX 3090 Ti, one request at a time, no prefix cache): three Terminal-Bench 2.1 tasks, two attempts each | testing, and far ahead on cost here: solved 6/6 against 5/6, 7.3x fewer prompt tokens (0.86M against 6.27M) and 1.7x less wall time. Claude Code sends about 18.8k prompt tokens a request (20 tool schemas and its system prompt), pi about 2k. Six trials say nothing about quality rates. On the hosted, prefix-cached GLM in the public [harness leaderboard](https://huggingface.co/spaces/joelniklaus/harness-leaderboard) (v0.1.7), not our run, the gap is 32 % fewer tokens at 65.9 against 63.2 macro quality | none yet: our run is not written up here |
| [UniiChat](https://github.com/VictorTaelin/UniiChat) | one endless chat; every message summarized into a binary tree; each call sees a bounded view (up to 128 KB) and can zoom back to the original | a fork pointed at our local OpenAI-compatible servers, with a small local model writing the summaries | testing: does a ~32k bounded view keep a local model useful past its window? | none yet |
| [OptMem](https://github.com/VictorTaelin/OptMem) | memory as a tool: an append-only log of one-line notes plus a summary tree, read at session start | inside the pi harness, beside UniiChat | testing | none yet |
