# Qwen3.6-35B-A3B on one Arc Pro B70: vLLM vs llama.cpp, a 262k window, and Claude Code on top

[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) is a mixture-of-experts model (35B
total, about 3B active per token) with 40 layers: 30 Gated DeltaNet (linear attention) and 10 full
attention layers with 2 KV heads of 256. It is the model our coding agents run on. Measured
2026-09-24 on one B70.

## Decode

512-token streams at ~1.5k and ~16k tokens of context. vLLM figures are the steady-state rate (the
span in which every stream is decoding; see [`bench/`](bench/)); llama.cpp's are sums of per-stream
rates with each stream pinned to its own slot and prefilled first.

| decode tok/s | llama.cpp Vulkan | vLLM, MTP 3 | vLLM, no speculation | vLLM MTP vs llama.cpp |
|---|---|---|---|---|
| 1 stream, ~1.5k | 70.3 | 149.0 | 116.1 | 2.1x |
| 1 stream, ~16k | 43.5 | 145.4 | 110.2 | 3.3x |
| 4 streams, ~1.5k | 144.4 | 347.7 | 320.3 | 2.4x |
| 4 streams, ~16k | 62.4 | 319.0 | 299.6 | 5.1x |
| 8 streams, ~1.5k | - | 420.7 | 467.4 | - |
| 8 streams, ~16k | - | 386.4 | 419.1 | - |

- **vLLM**: `intel/llm-scaler-vllm:0.26.0-b2`, the BF16 checkpoint quantized to `sym_int4` at load,
  fp8 KV cache, eager mode, `--max-num-seqs 8`, `--block-size 64`, prefix caching, speculative
  decoding with the model's own MTP head (`qwen3_5_mtp`, 3 tokens).
- **llama.cpp**: release b10545, Vulkan (Mesa 26.1.8), unsloth's `UD-Q4_K_XL` GGUF, q8_0 KV, 4 slots
  of 131,072 tokens. We ran this in production until the numbers above.

Speculation pays at 1 to 4 streams and costs about 10 % at 8. On our 41-log test-triage eval
(structured JSON out of real test logs, 4 requests at once) the vLLM server scored 41 of 41, with no
false alarms on 8 passing logs.

## The full 262,144-token window

With 10 attention layers and 2 KV heads, a token of fp8 KV cache is about 10 KB, so the model's
native window (262,144, no rope scaling) fits beside the weights: vLLM's pool holds 423,080 tokens
with MTP on. We planted three 6-digit codes at 10 %, 50 % and 90 % depth in a 239,227-token prompt
of real repository text and asked for them back, thinking off:

- all three came back right;
- the first read took 159.5 s (about 1,500 tokens/s of prefill);
- the same question again took 5.4 s, with 235,520 of the tokens served from the prefix cache.

An agent with a huge context pays for the read once per new context, not once per turn. This is
retrieval, the easiest thing to ask of a long prompt; finding a defect in one is harder, and on
Qwen3.8-27B recall of planted defects fell sharply past ~23k tokens ([`vllm-xpu/`](vllm-xpu/)).

## A load can hang silently: watch the kernel log

In one of roughly twenty vLLM loads on one card in one night, the `xe` driver reset the card's copy
engine partway through loading the weights (`xe 0000:03:00.0: [drm] GT0: Engine reset:
engine_class=bcs` in the kernel log, and a devcoredump with reason "LR job cleanup"). The reset kills the queue vLLM was using; vLLM gets
no error and never exits, so `/health` simply never answers. Our fix is in the launcher: it counts
engine resets for its own card's PCI address in `journalctl -k -b` (readable without root for the
`adm` group) and exits when a new one appears, so systemd restarts it. A health-check timeout alone
catches it too, only much later.

## Claude Code on it

Claude Code runs unchanged on this model through an Anthropic-to-OpenAI proxy (we use a fork of
[CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI)). What we had to fix on the way:

- **Thinking.** vLLM returns the model's reasoning as `reasoning`, not `reasoning_content`. A
  translator that reads only the older name drops the thinking, so the stream carries nothing while
  the model thinks, and Claude Code gave up after about 3 minutes and retried.
- **Thinking length.** Uncapped, the model thought for minutes per agent turn (about 3 minutes at
  ~95 tok/s, some 17,000 tokens). vLLM accepts a per-request `thinking_token_budget`; on one
  question, 512 kept the answer right and cut 45.7 s to 6.7 s, but at 2048 a count-only question
  came back wrong, and a model cut off mid-thought keeps reasoning in its visible answer. We set 8192 as a runaway guard, not as a speed setting.
- **Model names.** Claude Code asks for its small model for titles and summaries; map its model names
  to the local one in the proxy, or those calls fail.
- **Context.** Read the window from vLLM's `/v1/models` (`max_model_len`); llama.cpp reports it on
  `/props`. A harness that believes it has more room than the server never auto-compacts.
- **Reasoning effort (Qwen3.8-27B only).** Its chat template accepts `reasoning_effort` xhigh,
  medium or low and rejects anything else with HTTP 400, including `high`, which OpenAI-style clients
  and Claude Code send. [`vllm-xpu/effort-template.py`](vllm-xpu/effort-template.py) writes a copy
  of the template that maps `high` and `max` to `xhigh`, for vLLM's `--chat-template`.

Whether a local model finishes real repository work is a separate question from whether the harness
runs. On a small suite of repo-scale tasks with hidden checks ([`agent-eval/`](agent-eval/)), this
model finished 15 of 24 runs with thinking capped (about 2 minutes a task), and the dense
Qwen3.8-27B 22 of 24 with thinking off (about 3 minutes); capping or disabling thinking made no
detectable difference for either.

-- Claude Opus 5.5, working in [helm](https://github.com/akapug/helm) for @akapug
