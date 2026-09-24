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
  kernels (`inc-q40`), and a CPU check for quantized checkpoints.
- [`llama.cpp-sycl/`](llama.cpp-sycl/): 18 patches, each with its measured effect, and a build
  script that verifies the patched source tree.
- [`bench/`](bench/): the decode benchmarks and the structured-output probe.
