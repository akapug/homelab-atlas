# vLLM on the Arc Pro B70: two fixes to llm-scaler 0.26.0-b2

Intel's [llm-scaler](https://github.com/intel/llm-scaler) image `intel/llm-scaler-vllm:0.26.0-b2`
(vLLM 0.26.1.dev0+g568afb3a1 with Intel's XPU kernels) is the fastest way we have found to serve
Qwen3.8-27B on one Arc Pro B70. We had to make two changes to it. Both are small and are here as
files you can apply.

Everything below was measured on one B70 (32 GB) with Qwen3.8-27B, `--dtype float16`,
`--max-model-len 40960`, `--max-num-seqs 4` unless stated otherwise, and speculative decoding
with the model's own MTP head (`{"method":"qwen3_5_mtp","num_speculative_tokens":3}`) where a
table says "MTP 3". Decode rates are aggregate tok/s over all streams, prompts prefilled first
so only decode is timed ([`../bench/decode-vllm.py`](../bench/decode-vllm.py)), at ~1.5k and
~16k tokens of context.

## 1. Compiled mode: remove two log calls (llm-scaler #699)

On 0.26.0-b2 every compiled mode fails at startup with `sym_int4`:
`SymInt4LinearMethod.apply()` calls `logger.info_once()` on the path `torch.compile` traces, and
Dynamo refuses a `logging.Logger` method. That is
[intel/llm-scaler#699](https://github.com/intel/llm-scaler/issues/699), whose reporter found the
cause. [`fix-699.py`](fix-699.py) removes exactly those two calls (the two at weight-load time are
not traced and stay):

```
python3 fix-699.py <site-packages>/vllm/model_executor/layers/quantization/sym_int4.py
```

With it, `VLLM_XPU_ENABLE_XPU_GRAPH=1` and no `--enforce-eager` start and serve correctly, and
faster. Qwen3.8-27B BF16 quantized at load (`--quantization sym_int4`):

| mode | 1 stream, 1.5k / 16k | 4 streams, 1.5k / 16k |
|---|---|---|
| eager, MTP 3 | 57.1 / 47.9 | 188.2 / 158.9 |
| compiled + XPU graphs, MTP 3 | 60.4 / 56.1 | 194.4 / 178.9 |
| eager, no speculation | 26.0 / 25.1 | 115.0 / 94.5 |
| compiled + XPU graphs, no speculation | 32.8 / 31.2 | 119.1 / 97.0 |

One stream without speculation at 32.8 tok/s is ~94 % of what the card's bandwidth allows for
this model (~35 tok/s: 16.5 GB of weights read per step at a measured ~590 GB/s).

## 2. AutoRound int4 checkpoints on the fast kernels (`inc-q40`)

AutoRound int4 checkpoints are a better int4 than rounding BF16 at load, and 18 GB to download
instead of 56. On 0.26.0-b2, with
[`Frozenlock/Qwen3.8-27B-int4-AutoRound`](https://huggingface.co/Frozenlock/Qwen3.8-27B-int4-AutoRound)
(also the checkpoint in [intel/llm-scaler#636](https://github.com/intel/llm-scaler/issues/636)):

- eager mode serves it correctly but slowly, 14-17 tok/s at one stream (vLLM's own throughput
  log, MTP 3). With `auto_round_kernel` installed, as it is in this image, the XPU AutoRound path
  is `INCARKLinearMethod` (`torch.ops.vllm.inc_ark_woq_linear`), where `sym_int4` sends decode to
  ESIMD kernels;
- compiled mode returns garbage from the first token, but only with XPU graphs and MTP
  speculation together: eager, compiled without XPU graphs, and XPU graphs without speculation
  all give correct output;
- the checkpoint itself is sound: [`autoround-dequant-check.py`](autoround-dequant-check.py)
  dequantizes its layers on the CPU to cosine 0.994 (relative error 10-11 %) of the BF16 weights;
- `Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound` does not load: 17 of its layers are 8-bit, and the
  XPU AutoRound path accepts 2 and 4 bits only.

The observation that fixes it: a symmetric 4-bit AutoRound checkpoint with group size 128 and
`auto_round:auto_gptq` packing already holds GGML Q4_0 with 128-wide blocks, the format
`sym_int4` produces at load: codes 0-15 with the zero point at 8, dequantized as
`(q - 8) * scale`, eight codes per int32 in order along the input dimension, fp16 scales. It is
only stored K-major (`[K/8, N]`, `[K/128, N]`) where `sym_int4` keeps it N-major. So the
conversion is two transposes after load.

[`inc-q40.diff`](inc-q40.diff) adds `INCXPUQ40LinearMethod`: it loads the checkpoint through the
AutoRound parameters, transposes `qweight` and `scales` after load, and is otherwise
`SymInt4LinearMethod`, so the weights take the ESIMD decode kernels and every fused path that
checks `is_sym_int4`. Only layers that are 4-bit, symmetric, group 128 and GPTQ-packed take it;
anything else keeps the stock path. `VLLM_INC_XPU_Q40=0` turns it off.
[`inc-q40.py`](inc-q40.py) makes the same change as a script, with `.orig` backups.

Frozenlock's checkpoint, compiled + XPU graphs, MTP 3, with both fixes:

| | 1 stream, 1.5k / 16k | 4 streams, 1.5k / 16k | 8 streams, 1.5k |
|---|---|---|---|
| AutoRound int4 + inc-q40 | 61.2-61.5 / 58.6 | 197.0-201.0 / 192.1 | 351.4 (`--max-num-seqs 16`) |
| BF16 + `--quantization sym_int4` | 60.4 / 56.1 | 194.4 / 178.9 | |

Quality, on a task eval of ours (41 labelled CI test logs; the model must return the failing
tests as JSON, with a reason quoted from the log), greedy:

| | exact failing set | malformed JSON | reasons quoted verbatim |
|---|---|---|---|
| AutoRound int4 + inc-q40, sequential | 41 / 41 | 0 | 186 / 187 |
| the same, 4 requests at a time | 41 / 41 | 0 | 185 / 187 |
| BF16 + `sym_int4` at load, sequential | 39 / 39 parsed | 2 of 41 | 169 / 170 |

A structured-output probe ([`../bench/constrained-probe.py`](../bench/constrained-probe.py):
JSON schema, JSON mode, a tool call, a thinking request) passes on both.

Limits: one model, one card, tensor parallel 1. Not tried: AWQ-packed checkpoints, other group
sizes (those keep the stock path), other models.

## Apply

Inside the image, site-packages is `/opt/venv/lib/python3.12/site-packages`:

```
cd /opt/venv/lib/python3.12/site-packages
patch -p1 < inc-q40.diff
python3 fix-699.py vllm/model_executor/layers/quantization/sym_int4.py
find vllm/model_executor/layers/quantization -name '*.pyc' -delete
```

Then serve the checkpoint with its own quantization method (no `--quantization` flag):

```
VLLM_XPU_ENABLE_XPU_GRAPH=1 vllm serve Frozenlock/Qwen3.8-27B-int4-AutoRound \
  --served-model-name qwen3.8-27b --dtype float16 --mamba-ssm-cache-dtype float16 \
  --max-model-len 40960 --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 --block-size 64 --enable-prefix-caching --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
```

Compiled mode takes ~3-4 minutes to capture graphs before `/health` answers.

## If you are coming from llama-server

Two differences callers see: vLLM returns parsed thinking in `message.reasoning` (and
`delta.reasoning` when streaming), not `reasoning_content`, and `/v1/models` reports the window
as `max_model_len` rather than `meta.n_ctx`. Answers are in `content` either way.
