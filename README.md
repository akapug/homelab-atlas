# Homelab Atlas

Measurements, patches and tools for running local LLMs on hardware people can actually get:
older workstations, used server parts, single prosumer GPUs. Not a rack of H100s.

The bet behind it: as models lean on sparsity, state-space layers, speculative decoding and low-bit
weights, memory bandwidth and software matter more than raw GPU compute, and a lot of capability
is left on the table on modest hardware because nobody tuned for it. Each entry here is a
measured case of that, with the patches that closed the gap and the commands to check our numbers.

## Entries

| entry | what is in it |
|---|---|
| [**Intel Arc Pro B70** serving Qwen3.8-27B](b70/) | vLLM on Intel's llm-scaler image with two fixes: compiled mode (llm-scaler #699) and AutoRound int4 checkpoints on the fast int4 kernels, 61 tok/s at one stream and 351 at eight. 18 llama.cpp SYCL patches, 39 → 101 tok/s at four streams. The benchmarks behind both. |

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
