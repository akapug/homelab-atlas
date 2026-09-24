# vLLM on the Arc Pro B70: three changes to llm-scaler 0.26.0-b2

Intel's [llm-scaler](https://github.com/intel/llm-scaler) image `intel/llm-scaler-vllm:0.26.0-b2`
(vLLM 0.26.1.dev0+g568afb3a1 with Intel's XPU kernels) is the fastest way we have found to serve
Qwen3.8-27B on one Arc Pro B70. We made three changes to it. All are small and are here as files
you can apply: two fixes, and a speed-up for speculative decoding (section 3). A note on the host
follows them: other work on the server's CCD can halve a MoE's decode ("The host's CPU").

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
| AutoRound int4 + inc-q40, `--enforce-eager` | 54.4 / 55.3 | 186.7 / | |
| BF16 + `--quantization sym_int4` | 60.4 / 56.1 | 194.4 / 178.9 | |

Multi-stream figures above are sums of per-stream rates from 128-token runs, which overstate
concurrent throughput when streams start staggered. The wall-clock rate of those runs was 5-12 %
lower for the AutoRound rows, 18 % lower for the two BF16 + `sym_int4` rows at 1.5k with MTP, and
31 % lower for eager `sym_int4` at 16k (158.9 vs 110.3 tok/s): read those as upper bounds.
Measured later with the steady-state rate and 512-token streams on the serving card (AutoRound +
inc-q40, eager, fp8 KV): 4 streams 197.1 tok/s at ~1.5k and 188.0 at ~16k.

Intel notes that XPU graphs are not supported yet
([#698](https://github.com/intel/llm-scaler/issues/698)); the eager row is the configuration
without them, 5-12 % slower here and ~3.5x the stock AutoRound path.

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

## 3. A smaller vocabulary for the MTP drafter (`draft-vocab.py`)

With speculative decoding on, each extra draft token cost a Qwen3.8-27B verify step ~5.5 ms on a
B70, and most of it was not the MTP layer itself. The MTP head has no lm_head of its own: vLLM gives
it the target's, which the AutoRound checkpoint keeps unquantized, 248,320 x 5,120 fp16 = 2.54 GB,
read whole for every draft token (~4.3 ms at the card's ~590 GB/s). Drafting is greedy, so only the
argmax of those logits is used, and nearly all of the 248k rows are tokens our output never
contains.

[`draft-vocab.py`](draft-vocab.py) patches the drafter (`qwen3_5_mtp.py`) to score only the token
ids listed in a file named by `VLLM_DRAFT_VOCAB`, from a copy of those rows taken once, and to give
every other token -inf. The target still verifies every draft with its full lm_head, so the output
is unchanged; a token missing from the list is a rejected draft, which costs acceptance, not
correctness. Unset, the file behaves as before. This is the idea of FR-Spec (Zhao et al., 2025,
"frequency-ranked speculative sampling"), which SGLang ships as `--speculative-token-map`; upstream
vLLM has no equivalent today.

The lists ([`draft-vocab/`](draft-vocab/)) come from [`draft-vocab-build.py`](draft-vocab-build.py):
every token seen in about 10 million tokens of our own source code, docs and CI logs (37,435
distinct), plus ids below 32,768 and the tokenizer's added tokens, 50,521 ids for Qwen3.8-27B
(50,528 for Qwen3.6-35B-A3B, whose tokenizer differs). On text the model itself wrote and the list
never saw, they cover 99.4 % and 99.2 % of tokens.

Verify-step time at one stream, on an idle server ([`../bench/spec-step-cost.sh`](../bench/spec-step-cost.sh):
1,024 tokens of a log summary, the draft counters for tokens a step; eager, fp8 KV, a 131,072-token
window, 8 sequences; two runs each, averaged):

| Qwen3.8-27B drafter vocabulary | 3 drafts | 5 drafts | 7 drafts | each extra draft |
|---|---:|---:|---:|---:|
| all 248,320 | 50.7 ms | 61.6 ms | | 5.5 ms |
| ids below 100,000 | 43.0 ms | 49.0 ms | | 3.0 ms |
| the 50,521-id list | 40.2 ms | 44.7 ms | 49.6 ms | 2.3 ms |

At 3 drafts the tokens a step did not move (2.99-3.04 with the full vocabulary, 2.96-3.00 with the
list), so one stream went from 59.5 to 74.6 tok/s, +25 %. On real coding-agent traffic (Claude Code
through a proxy, four repository tasks) the drafter accepted 2.37 tokens a step with the list
against 2.25 without, in separate runs: no loss we can measure. Qwen3.6-35B-A3B (the MoE, BF16
quantized to `sym_int4` at load), whose drafter read a 1 GB lm_head: 19.3 to 15.2 ms a step at 3
drafts, 147.4 to 189.8 tok/s, same acceptance (2.84 and 2.88 tokens a step). That MoE's short eager
steps are sensitive to other work on the host's CPUs: with two cores busy the same measurement read
15.7-19.9 ms.

Checks. Greedy output: for the 27B with the ids-below-100k list, 6 of 14 outputs are identical to
the no-speculation output, as with the full-vocabulary drafter, and the rest part at the same
near-ties; for the MoE with its list, 9 of 14 are identical to its full-vocabulary drafter's, the
rest parting late. In production with the lists, for both models: streamed tool calls arrive
identical to non-streamed ones (16 of 16), a JSON-schema, JSON-mode, tool-call and thinking probe
passes, and our 41-log CI-triage ruler scores 41 of 41 exact at 4-way concurrency.

Why it pays so much here: the card is bandwidth-bound for one stream, the vocabulary is large, and
the head is unquantized, so the head read was most of a draft. vLLM shares the target's lm_head with
MTP drafters in general, so other large-vocabulary models may pay the same cost. Cheaper drafts can
make more of them pay, but only where they are accepted: on these log summaries, with the list, 5
drafts ran 74-78 tok/s and 7 drafts 68-71.

## A 131k-token window on one card

With `--kv-cache-dtype fp8 --max-model-len 131072` (same flags otherwise, compiled + XPU graphs,
MTP 3), the KV pool on one B70 is 195,233 tokens: one full-window request with room to spare.

| context | 1 stream | 4 streams |
|---|---|---|
| ~1.5k | 56.9 | 186.9 |
| ~16k | 57.8 | 191.9 |
| ~66k | 55.4 | |
| ~129k | 44.3 | |

On our task eval the fp8 cache cost one answer in 41 a malformed JSON string (a raw control
character deep in a long answer); the other 40 were exact. A bigger window is capacity, not
comprehension: planting one of three known defects in unrelated code and growing the prompt, this
model found them in 9 of 9 reads at 2-12k tokens and in 10 of 36 past ~23k. An earlier measurement
on llama.cpp with an 8-bit cache had the same shape, so the limit is the model, not the fp8 cache.

## The host's CPU: keep other work off the server's CCD

If the machine that serves the model also runs other heavy work (builds, test suites), where that
work runs matters as much as how much of it there is. vLLM on XPU in eager mode launches every
kernel from Python, so a short speculative step is largely CPU time. We measured this with
Qwen3.6-35B-A3B (the MoE, 3B active, eager, fp8 KV, MTP 3 with the section-3 draft vocabulary), on a
Ryzen 9 9950X3D. That CPU has two CCDs, each with its own L3: 96 MB (3D V-cache) on CPUs 0-7 and
16-23, 32 MB on 8-15 and 24-31. The server ran in docker, and its CPU set was changed live
(`docker update --cpuset-cpus`). Background load ran niced 19 in a user scope. Each figure is one
run of [`../bench/spec-step-cost.sh`](../bench/spec-step-cost.sh) (one stream, 1,024 tokens), in
ms per speculative step, so lower is better. Runs that another request shared are left out.

| setup | ms per step |
|---|---|
| idle | 17.4, 19.3 |
| idle, the server on one CCD (8-15, 24-31) | 15.2, 15.2 |
| compute load (`stress-ng --cpu`) on every CPU | 23.4, 23.9 |
| the same, the server on 2 cores of that CCD, the load on the other 28 CPUs | 22.1, 25.0 |
| the same, the server on one CCD, the load on the other CCD | 19.1, 19.3 |
| memory-bandwidth load (`stress-ng --stream`) on every CPU | 379 (7 tok/s) |
| the same, the server on one CCD, the load still on every CPU | 398, 402 |
| the same, the server on one CCD, the load on the other CCD | 17.8, 19.4 |
| a real mixed load (a Rust release build, JavaScript test runners, Python jobs), unpinned | 30.5, 32.0, 33.6 |
| the same, the server on one CCD | 35.0, 35.7 |
| the same, every user process on the other CCD (`user.slice` AllowedCPUs) | **15.2, 15.2** |

- **What costs the server is sharing its CCD.** Waiting for a core is not the problem. Pinning the
  server does not help while other work still runs on its CCD, and a memory-bound load spread over
  every CPU made it about 20 times slower.
- **What helps is keeping everything else on the other CCD.** That was as fast as idle.
- **The dense Qwen3.8-27B did not move.** Its 40 ms step is mostly GPU work: 40.1 unpinned under the
  real load, 40.2 and 40.4 split.
- **Limits.** One CPU and one model family. The server ran on the smaller-cache CCD; we did not try
  it on the V-cache CCD. The real load varied from run to run.

Our servers are idle most of the time, and a permanent split would take half the CPU from the other
work. So we split only while a server is busy:
- [`serving-cpu-guard.sh`](serving-cpu-guard.sh) runs as root, with
  [`serving-cpu-guard.service`](serving-cpu-guard.service).
- It keeps the server containers on one CCD.
- While any of them has a request in flight, plus 20 s after the last one, it limits `user.slice`
  (every login session and user service) to the other CCD. Otherwise that work gets every CPU.
- Stopping it restores both. Measured with it running under our normal load: 15.2 ms per step
  (190 tok/s).

## Apply

Inside the image, site-packages is `/opt/venv/lib/python3.12/site-packages`:

```
cd /opt/venv/lib/python3.12/site-packages
patch -p1 < inc-q40.diff
python3 fix-699.py vllm/model_executor/layers/quantization/sym_int4.py
find vllm/model_executor/layers/quantization -name '*.pyc' -delete
```

For the smaller draft vocabulary (section 3), patch the drafter, put a list where the server can read
it, and name it in the environment:

```
python3 draft-vocab.py vllm/model_executor/models/qwen3_5_mtp.py
find vllm/model_executor/models -name 'qwen3_5_mtp*.pyc' -delete
export VLLM_DRAFT_VOCAB=/path/to/draft-vocab/qwen38-freq32k.txt   # before vllm serve
```

The server logs `draft vocab: 50521 of 248320 tokens` on its first draft.

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

A third, from Qwen3.8's own chat template: it takes `reasoning_effort` xhigh (the default), medium
or low, and rejects anything else with HTTP 400. llama-server does not pass the field to the
template; vLLM does, so an OpenAI-style client or Claude Code sending `high` fails on every request.
[`effort-template.py`](effort-template.py) writes a copy of the template that maps `high` and `max`
to `xhigh`, `minimal` and `none` to `low`, and a null to the default, and changes nothing else:

```bash
python3 effort-template.py /path/to/Qwen3.8-27B-checkpoint qwen38-effort.jinja
vllm serve ... --chat-template qwen38-effort.jinja
```

-- Claude Opus 5.5, working in [helm](https://github.com/akapug/helm) for @akapug
