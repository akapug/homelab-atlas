# Intel Arc Pro B70 (32 GB): serving Qwen3.8-27B

The Arc Pro B70 is a 32 GB Xe2 (Battlemage) workstation card. One card holds a 27B model at
4 bits with room for a useful KV cache. This entry is what we measured serving
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) on it (a hybrid model: 48 Gated DeltaNet
layers and 16 attention layers, plus a multi-token-prediction head usable for speculative
decoding), and the patches that got us there.

## Results

Aggregate decode tok/s, one B70, prompts at ~1.5k and ~16k tokens, 4-bit weights. "MTP 3" is
speculative decoding with the model's own MTP head, 3 draft tokens.

| stack | 1 stream, 1.5k / 16k | 4 streams, 1.5k / 16k | 8 streams, 1.5k |
|---|---|---|---|
| llama.cpp SYCL, upstream `94256114c2`, no speculation | | 39.3 / | |
| llama.cpp SYCL + [our patches](llama.cpp-sycl/) 0001-0017, no speculation | 24.4 / 22.7 | 64 / 57 | |
| llama.cpp SYCL + all 18 patches, MTP 3 | 43.1-44.2 / 36.1 | 100.7-101.6 / 91.6-91.7 | |
| vLLM ([llm-scaler 0.26.0-b2](vllm-xpu/)), BF16 quantized at load, compiled, MTP 3 | 60.4 / 56.1 | 194.4 / 178.9 | |
| vLLM, AutoRound int4 + [inc-q40](vllm-xpu/), compiled, MTP 3 | 61.2-61.5 / 58.6 | 197.0-201.0 / 192.1 | 351.4 |

The card's limit for one stream without speculation is ~35 tok/s: each step reads ~16.5 GB of
weights, and we measured ~590 GB/s sustained (608 nameplate). vLLM compiled reaches 32.8
without speculation, ~94 % of that; our llama.cpp build reaches 24.4.

**Which to use.** vLLM with an AutoRound int4 checkpoint and both fixes in
[`vllm-xpu/`](vllm-xpu/) is the fastest at every stream count we measured, and it matched the
llama.cpp GGUF on our correctness eval (41 of 41 exact; 186 of 187 reasons quoted verbatim against
187). llama.cpp is the simpler install and keeps working without a patched image; its patches are
in [`llama.cpp-sycl/`](llama.cpp-sycl/). The speed-up from speculation depends on the text (see
[`bench/`](bench/)).

## Contents

- [`vllm-xpu/`](vllm-xpu/): compiled mode for llm-scaler (issue #699), AutoRound int4 on the fast
  kernels (`inc-q40`), a smaller vocabulary for the MTP drafter, two tool-call fixes for agent
  clients, a chat-template fix for `reasoning_effort`, a guard that keeps other work off the server's
  CPUs, our run of the DeltaNet mixed-batch test ([`gdn-mixed-batch/`](vllm-xpu/gdn-mixed-batch/)),
  and a CPU check for quantized checkpoints.
- [`llama.cpp-sycl/`](llama.cpp-sycl/): 18 patches, each with its measured effect, and a build
  script that verifies the patched source tree.
- [`bench/`](bench/): the decode benchmarks, the speculative step-time and CPU-contention probes, the
  structured-output probe and the tool-call length check.
- [`qwen36-35b-a3b.md`](qwen36-35b-a3b.md): the MoE model our agents run on (Qwen3.6-35B-A3B): vLLM
  at 2.1-5.1x llama.cpp Vulkan on one card, its full 262k window, a silent load hang to watch for,
  and running Claude Code on it.
- [`agent-eval/`](agent-eval/): can a local model finish real repository work? Twelve tasks with
  hidden checks, a sandboxed runner, and 72 runs: on all twelve, the dense 27B finished 22 of 24
  and the MoE 15 of 24 (p = 0.036), at little more speed once the tasks got hard.
- [`../PATCHES.md`](../PATCHES.md): every patch and tool above, sorted by the hardware and software it
  needs, with how to apply, check and undo each.

## Related work

- [CySpiegel/vllm-intel](https://github.com/CySpiegel/vllm-intel): upstream vLLM tuned on two B70s
  (tensor parallel 2) for Qwen3.8-27B, with XPU fixes proposed to vLLM and to
  vllm-xpu-kernels. One matters to anyone serving this model with speculative decoding:
  [vllm-xpu-kernels#552](https://github.com/vllm-project/vllm-xpu-kernels/pull/552), an
  out-of-bounds write in the DeltaNet kernel for batches that mix speculative and
  non-speculative sequences. Our run of its test on this image is in
  [`vllm-xpu/gdn-mixed-batch/`](vllm-xpu/gdn-mixed-batch/).
- intel/llm-scaler issues we have added data to:
  [#699](https://github.com/intel/llm-scaler/issues/699) (compiled mode),
  [#698](https://github.com/intel/llm-scaler/issues/698) (XPU graphs; Intel: not supported yet),
  [#636](https://github.com/intel/llm-scaler/issues/636) (AutoRound checkpoints).
- For NVIDIA cards, [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen) serves the same model
  family on vLLM with many patches; we measured it on a Turing Quadro RTX 6000 and sent the one
  fix it needed there ([HyperQwen#188](https://github.com/syv-ai/HyperQwen/pull/188)).
