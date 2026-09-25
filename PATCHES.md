# Patches and tools, by hardware

Every patch and tool in this repo, grouped by the hardware it needs, then by the software you run.
Each entry says what goes wrong without it (the symptom you would see), who it is for, what we
measured and where this repo shows it, and how to apply it, check it and undo it.

Everything here was measured on [our machine](#our-machine). Where an entry says **untested**, we
have not run it on that hardware, and the entry says why it may or may not carry over.

## Words used here

- **vLLM** and **llama.cpp**: the two programs ("engines") that load a model onto the GPU and answer
  requests over HTTP. vLLM is written in Python; llama.cpp is C++ and runs GGUF model files.
- **llm-scaler**: Intel's Docker image of vLLM for Arc GPUs. The vLLM patches here are written for its
  version `intel/llm-scaler-vllm:0.26.0-b2` (vLLM 0.26.1.dev0+g568afb3a1).
- **XPU**: what PyTorch and vLLM call an Intel GPU. **SYCL**: the programming layer that llama.cpp's
  Intel backend is written in. **Xe2**: the GPU generation of the Arc Pro B70 (Battlemage).
- **Stream**: one request being answered; "4 streams" means four at once. **tok/s**: tokens (word
  pieces) generated per second, summed over all streams.
- **Speculative decoding, MTP**: the model cheaply guesses a few next tokens ("drafts") and checks them
  all in one pass. Qwen3.5-family models carry a small guessing layer of their own, the MTP
  (multi-token prediction) head. "MTP 3" means three drafts a step.
- **Tool call**: the model asking the client to run something (read a file, run a command). An **agent
  client** such as Claude Code works through tool calls.
- **Proxy**: Claude Code speaks Anthropic's API; vLLM and llama.cpp speak OpenAI's. A proxy
  translates between the two.
- **KV cache, window**: the GPU memory that holds the conversation so far; the window is how many
  tokens of it one request may use.

## Four kinds of change

| kind | what it means for you | entries |
|---|---|---|
| **Any GPU** | It changes the serving software's own code or a model's chat template, not GPU code, so your GPU does not decide whether it applies: your software version does. We measured every one on Intel only. | `strict-tools`, `length-finish`, `effort-template`, `draft-vocab`, llama.cpp `0015` and `0016` |
| **Intel only** | It changes Intel GPU kernels, an Intel-only code path, or works around Intel's `xe` kernel driver. | `fix-699`, `inc-q40`, llama.cpp `0001`-`0014`, `0017`, `0018`, two driver settings, the DeltaNet check |
| **Host CPU** | It depends on the CPU the model server runs beside, not on the GPU. | `serving-cpu-guard` |
| **No GPU needed** | Tools that measure or check: they talk to a server over HTTP or read files. | `agent-eval`, `bench/`, `autoround-dequant-check`, `draft-vocab-build` |

## Index

| entry | kind | software it needs | measured on | details |
|---|---|---|---|---|
| [`strict-tools.py`](b70/vllm-xpu/strict-tools.py) | any GPU | llm-scaler 0.26.0-b2's vLLM, the `qwen3_coder` tool parser | B70 | [section 1](#strict-toolspy-tool-names-the-model-cannot-get-wrong) |
| [`length-finish.py`](b70/vllm-xpu/length-finish.py) | any GPU | llm-scaler 0.26.0-b2's vLLM | B70 | [section 1](#length-finishpy-a-cut-off-tool-call-says-it-was-cut-off) |
| [`effort-template.py`](b70/vllm-xpu/effort-template.py) | any GPU | Qwen3.8-27B served by vLLM | B70 | [section 1](#effort-templatepy-qwen38-accepts-the-usual-effort-words) |
| [`draft-vocab.py`](b70/vllm-xpu/draft-vocab.py), its lists and [`draft-vocab-build.py`](b70/vllm-xpu/draft-vocab-build.py) | any GPU | llm-scaler 0.26.0-b2's vLLM, a Qwen MTP head | B70 | [section 1](#draft-vocabpy-cheaper-drafts-for-speculative-decoding) |
| llama.cpp [`0015`, `0016`](b70/llama.cpp-sycl/patches/) | any GPU | llama.cpp `94256114c2`, MTP speculation | B70 (SYCL) | [section 1](#llamacpp-0015-and-0016-speculative-decoding-that-keeps-to-its-limits) |
| [`fix-699.py`](b70/vllm-xpu/fix-699.py) | Intel only | llm-scaler 0.26.0-b2, int4, compiled mode | B70 | [section 2](#fix-699py-compiled-mode-starts) |
| [`inc-q40.diff`](b70/vllm-xpu/inc-q40.diff) or [`inc-q40.py`](b70/vllm-xpu/inc-q40.py) | Intel only | llm-scaler 0.26.0-b2, an AutoRound int4 checkpoint | B70 | [section 2](#inc-q40-autoround-int4-checkpoints-on-the-fast-kernels) |
| `--max-num-batched-tokens 2048` | Intel only (`xe` driver) | vLLM, very long prompts | B70 | [section 2](#setting-long-prompts-without-an-engine-reset) |
| engine-reset watch | Intel only (`xe` driver) | vLLM | B70 | [section 2](#setting-a-model-load-that-hangs-with-no-error) |
| [`gdn-mixed-batch/`](b70/vllm-xpu/gdn-mixed-batch/) | Intel only | llm-scaler 0.26.0-b2's kernels, a DeltaNet model with MTP | B70 | [section 2](#gdn-mixed-batch-a-check-for-the-deltanet-kernel-bug) |
| llama.cpp [`0001`-`0014`, `0017`, `0018`](b70/llama.cpp-sycl/patches/) (`0009` adds tests only) | Intel only | llama.cpp `94256114c2` with SYCL, a GGUF model | B70 | [section 3](#3-intel-arc-pro-b70-with-llamacpp) |
| [`serving-cpu-guard.sh`](b70/vllm-xpu/serving-cpu-guard.sh) | host CPU | a CPU with two CCDs, vLLM in Docker, systemd | Ryzen 9 9950X3D | [section 4](#serving-cpu-guard-keep-other-work-off-the-servers-ccd) |
| [`agent-eval/`](b70/agent-eval/) | no GPU needed | an OpenAI-compatible server, a proxy, Docker | two B70s | [section 6](#agent-eval-can-a-local-model-finish-real-repository-work) |
| [`bench/`](b70/bench/) | no GPU needed | llama-server or an OpenAI-compatible server | B70 | [section 6](#bench-the-measurements-behind-the-numbers) |
| [`autoround-dequant-check.py`](b70/vllm-xpu/autoround-dequant-check.py) | no GPU needed | Python with numpy | CPU | [section 6](#autoround-dequant-checkpy-is-a-4-bit-checkpoint-sound) |

## Our machine

If yours matches, this is the configuration every number below comes from.

- **The model server**: an AMD Ryzen 9 9950X3D (AM5) with two Intel Arc Pro B70 (32 GB each), on Linux
  with the kernel's `xe` driver. vLLM runs from `intel/llm-scaler-vllm:0.26.0-b2` in Docker, one model
  per card, both in eager mode (no compiled graphs), with an fp8 KV cache, 8 requests at a time and
  MTP 3:
  - Qwen3.8-27B (dense), from the AutoRound int4 checkpoint through `inc-q40`, a 212,992-token window,
    with `draft-vocab`, `strict-tools`, `length-finish` and `effort-template`;
  - Qwen3.6-35B-A3B (mixture of experts, about 3B active per token), BF16 quantized to int4 at load,
    its full 262,144-token window, with `draft-vocab`, `strict-tools` and `length-finish`;
  - `serving-cpu-guard` on the host, because the same machine also runs builds and tests.
- **An NVIDIA Quadro RTX 6000** (24 GB, Turing) in an HP Z8 G4, running llama.cpp CUDA builds. Nothing
  in this repo was measured on it yet, apart from one fix to another project
  ([below](#nvidia-quadro-rtx-6000-turing-24-gb)).
- **An NVIDIA GTX 1080 Ti** (11 GB, Pascal). Nothing measured on it yet.
- **The client**: Claude Code, reaching the local models through an Anthropic-to-OpenAI proxy (a fork
  of [CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI)).

## 1. Any GPU: serving-software and chat-template fixes

These change vLLM's or llama.cpp's own code, or a model's chat template, so the GPU underneath does
not decide whether they apply. We ran them only on the B70.

### Before you patch vLLM

The vLLM patches are Python scripts that edit files inside the vLLM install, in the image's
`site-packages` folder (`/opt/venv/lib/python3.12/site-packages` in llm-scaler 0.26.0-b2). Every
script:

- checks that each file's code is exactly what it expects before writing it, and otherwise stops with
  an error, so where another vLLM version's code differs it refuses rather than guesses;
- saves the original as `<file>.orig` the first time, and always patches from that copy, so running it
  twice is safe;
- is undone by copying the `.orig` file back.

After patching, delete the stale compiled copies (`find vllm -name '*.pyc' -delete`), then start
`vllm serve`. Edits made inside a running container are lost when the container is removed, so make
them in the image you serve from. The usual Docker way is a Dockerfile in your copy of this repo:

```dockerfile
FROM intel/llm-scaler-vllm:0.26.0-b2
COPY b70/vllm-xpu /atlas
RUN cd /opt/venv/lib/python3.12/site-packages \
 && python3 /atlas/length-finish.py vllm/entrypoints/openai/chat_completion/serving.py \
 && python3 /atlas/strict-tools.py vllm/tool_parsers/structural_tag_registry.py \
 && python3 /atlas/draft-vocab.py vllm/model_executor/models/qwen3_5_mtp.py \
 && python3 /atlas/inc-q40.py vllm/model_executor/layers/quantization/inc/schemes \
 && python3 /atlas/fix-699.py vllm/model_executor/layers/quantization/sym_int4.py \
 && find vllm -name '*.pyc' -delete
```

```bash
docker build -t llm-scaler-vllm:0.26.0-b2-atlas .
```

Our own servers run a copy of the image's files rather than a rebuilt image, so this Dockerfile is the
standard route, not the one our numbers came from. Leave out the lines you do not want: each patch
is also listed on its own below. How to start the image on your GPU is in Intel's
[llm-scaler](https://github.com/intel/llm-scaler) documentation; the `vllm serve` flags we use are in
[the vLLM notes](b70/vllm-xpu/README.md#apply).

### `strict-tools.py`: tool names the model cannot get wrong

- **Symptom.** Now and then a tool call names a tool that does not exist, and the client rejects it.
  We saw a Qwen model call a tool named ``Bash` `` (the real name plus a stray backtick and a newline).
- **For.** Any GPU: the file it edits is vLLM's tool-call code. vLLM as in llm-scaler 0.26.0-b2; other
  vLLM versions are untested. Models served with `--enable-auto-tool-choice --tool-call-parser
  qwen3_coder`, as both of ours are; other tool parsers are untested. Clients that send tools with
  `tool_choice: "auto"` and never mark a tool `"strict": true`, as Claude Code through a proxy does.
- **What it does.** With `VLLM_STRICT_TOOLS_AUTO=1`, once the model starts a tool call, a grammar lets
  it write only a declared tool's name, and arguments that fit that tool's schema. Ordinary text and
  thinking stay free.
- **Measured.** Claude Code's ~120 tools compile into the grammar in 0.8 s; applying it costs under
  1 ms a step on the B70. A tool whose schema the grammar library cannot read keeps its name
  constrained and its arguments free, so one odd schema never fails a request.
  ([vLLM notes, section 4](b70/vllm-xpu/README.md#4-tool-calls-from-agent-clients-strict-toolspy-length-finishpy))
- **Apply.** In site-packages: `python3 strict-tools.py vllm/tool_parsers/structural_tag_registry.py`,
  then `export VLLM_STRICT_TOOLS_AUTO=1` before `vllm serve`.
- **Check.** `grep -c 'strict-tools patch' vllm/tool_parsers/structural_tag_registry.py` prints `2`;
  [`constrained-probe.py`](b70/bench/constrained-probe.py) still gets a tool call back. The server log
  names any tool whose schema was refused (`strict-tools: tool ... keeps its name constrained`).
- **Undo.** Unset `VLLM_STRICT_TOOLS_AUTO` (the patched file then behaves as before), or copy
  `structural_tag_registry.py.orig` back.

### `length-finish.py`: a cut-off tool call says it was cut off

- **Symptom.** When a reply runs out of its output limit (`max_tokens`) in the middle of a tool call,
  the client gets what looks like a finished tool call with broken JSON, and Claude Code fails the
  call instead of continuing. We hit it with Qwen3.8-27B on a turn that used exactly its 16,000 output
  tokens.
- **For.** Any GPU, any model that makes tool calls. vLLM as in llm-scaler 0.26.0-b2. Upstream vLLM's
  main branch already reports `length` for streamed replies, but not for non-streamed ones when we
  checked.
- **What it does.** Reports `finish_reason: "length"` whenever the output limit ended the reply, as
  OpenAI's API does, even if a tool call was under way. It has no switch: it changes only what a
  length stop is called.
- **Measured.** [`length-cut-check.py`](b70/bench/length-cut-check.py) forces the cut: `tool_calls`
  before the patch, `length` after.
  ([vLLM notes, section 4](b70/vllm-xpu/README.md#4-tool-calls-from-agent-clients-strict-toolspy-length-finishpy))
- **Also needed.** A proxy that turns OpenAI finish reasons into Anthropic stop reasons must let
  `length` win over a tool call it has already seen, or it hides the stop again.
- **Apply.** In site-packages:
  `python3 length-finish.py vllm/entrypoints/openai/chat_completion/serving.py`.
- **Check.** `python3 length-cut-check.py 8000 <served model name>` prints `finish_reason=length`.
- **Undo.** Copy `serving.py.orig` back.

### `effort-template.py`: Qwen3.8 accepts the usual effort words

- **Symptom.** Every request from Claude Code or an OpenAI-style client fails with HTTP 400,
  "Unexpected reasoning effort high". Qwen3.8-27B's chat template (the file that turns a conversation
  into the model's input text) accepts `reasoning_effort` only as `xhigh`, `medium` or `low`, and
  those clients send `high`.
- **For.** Any GPU. Qwen3.8-27B (Qwen3.6-35B-A3B's template has no such check). Engines that pass
  `reasoning_effort` into the template: vLLM does. The llama-server builds measured in this repo did
  not, so they never showed the error; the script reads a Hugging Face model folder's
  `chat_template.jinja`, not a template inside a GGUF file.
- **What it does.** Writes a copy of the template that maps `high` and `max` to `xhigh`, `minimal` and
  `none` to `low`, and an empty value to the default. Nothing else changes.
- **Result.** Requests that send `high` are answered; in use on our Qwen3.8-27B server since vLLM
  replaced llama.cpp there.
  ([vLLM notes, "If you are coming from llama-server"](b70/vllm-xpu/README.md#if-you-are-coming-from-llama-server))
- **Apply.** `python3 effort-template.py /path/to/Qwen3.8-27B-checkpoint qwen38-effort.jinja`, then
  `vllm serve ... --chat-template qwen38-effort.jinja`.
- **Check.** A request that sends `high` is answered:

  ```bash
  curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model": "qwen3.8-27b", "reasoning_effort": "high", "max_tokens": 50,
         "messages": [{"role": "user", "content": "Say hi."}]}'
  ```
- **Undo.** Serve without `--chat-template`.

### `draft-vocab.py`: cheaper drafts for speculative decoding

- **Symptom.** No error: speculative decoding is just slower than it needs to be. Each extra draft cost
  Qwen3.8-27B ~5.5 ms on a B70, most of it spent reading the full 248,320-token output layer
  (2.54 GB, unquantized) to pick one token.
- **For.** Any GPU in principle (it edits vLLM's model code, not GPU code); measured on the B70 only.
  vLLM as in llm-scaler 0.26.0-b2. Qwen3.5-family models drafting with their own MTP head
  (`"method": "qwen3_5_mtp"`): Qwen3.8-27B and Qwen3.6-35B-A3B here. It pays most where one stream is
  limited by memory bandwidth and the output layer is large and unquantized, as on the B70. vLLM shares
  the target's output layer with MTP drafters in general, so other large-vocabulary models may carry
  the same cost; untested.
- **What it does.** With `VLLM_DRAFT_VOCAB` naming a list of token ids, the drafter scores only those
  tokens. The model still checks every draft against its full vocabulary, so the output does not
  change; a token missing from the list costs a rejected draft, not a wrong answer.
- **Measured.** One stream: Qwen3.8-27B 59.5 → 74.6 tok/s (+25 %), Qwen3.6-35B-A3B 147.4 → 189.8
  (+29 %), with the same tokens accepted per step.
  ([vLLM notes, section 3](b70/vllm-xpu/README.md#3-a-smaller-vocabulary-for-the-mtp-drafter-draft-vocabpy))
- **The lists.** [`qwen38-freq32k.txt`](b70/vllm-xpu/draft-vocab/qwen38-freq32k.txt) (50,521 ids, for
  Qwen3.8-27B) and [`qwen36moe-freq32k.txt`](b70/vllm-xpu/draft-vocab/qwen36moe-freq32k.txt) (50,528
  ids, for Qwen3.6-35B-A3B). A list belongs to one tokenizer. For another model, build one with
  [`draft-vocab-build.py`](b70/vllm-xpu/draft-vocab-build.py) from text like what the model will write
  (ours: about 10 million tokens of our own code, docs and CI logs); it prints how much of the model's
  own output each list size covers.
- **Apply.** In site-packages: `python3 draft-vocab.py vllm/model_executor/models/qwen3_5_mtp.py`, then
  `export VLLM_DRAFT_VOCAB=/path/to/qwen38-freq32k.txt` before `vllm serve`.
- **Check.** The server logs `draft vocab: 50521 of 248320 tokens` at its first draft.
  [`spec-step-cost.sh`](b70/bench/spec-step-cost.sh) shows the shorter step.
- **Undo.** Unset `VLLM_DRAFT_VOCAB` (the patched file then behaves as before), or copy
  `qwen3_5_mtp.py.orig` back.

### llama.cpp `0015` and `0016`: speculative decoding that keeps to its limits

- **Symptom.** With MTP speculation in llama-server, more drafts help one stream and sink several: at
  3 drafts, 4 streams fell to 2.8 tok/s on the B70. Separately, the MTP drafter drafted up to
  `--spec-draft-n-max` tokens for every request, ignored the smaller per-request limit the server sets,
  and threw the rest away.
- **For.** Any GPU backend: they change llama.cpp's shared speculative-decoding and server code
  (`common/`, `tools/server/`), not a GPU backend. They apply on their own to the same llama.cpp commit,
  `94256114c2` (checked with `git apply --check`). Measured only with SYCL on the B70, whose fast
  small-batch kernels stop at a fixed width; other backends change speed at other batch sizes, so the
  best cap may differ there; untested.
- **What they do.** `0015` makes the MTP drafter stop at either limit, as the simple drafter already
  does. `0016` adds `--spec-batch-max N`: each active request drafts at most N / requests - 1 tokens,
  so a step never checks more than N tokens. 0, the default, keeps the old behaviour.
- **Measured.** `0015`: 4 streams 80.0-80.6 → 87.0-87.5 tok/s. `0016` with N = 8: one stream drafts 3
  (43.7 tok/s, against 34.6 with 1 draft) while 4 streams keep 87.3.
  ([llama.cpp notes](b70/llama.cpp-sycl/README.md))
- **Upstream.** `0015` fixes the same bug as the open draft
  [ggml-org/llama.cpp#28473](https://github.com/ggml-org/llama.cpp/pull/28473).
- **Apply.** On the B70, [`build.sh`](b70/llama.cpp-sycl/build.sh) applies them with the rest. On
  another backend, apply just these two and build as you normally would:

  ```bash
  git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
  git checkout 94256114c229674ef96e76eb2dea596e65b43818
  git apply /path/to/homelab-atlas/b70/llama.cpp-sycl/patches/001[56]-*.patch
  ```

  Serve with `--spec-type draft-mtp --spec-draft-n-max 3 --spec-batch-max N`. On the B70 we use N = 16
  (8 before patch `0018` widened the fast kernels); the right N on another backend is untested.
- **Check.** `llama-server --help` lists `--spec-batch-max`;
  [`decode-streams.py`](b70/bench/decode-streams.py) at 1 and 4 streams.
- **Undo.** Leave out `--spec-batch-max`, or build without the patches.

### Claude Code on a local model: settings, not patches

What we had to set for Claude Code to run on a local model through a proxy, on any GPU:

- the proxy must read the model's thinking from `reasoning` (vLLM's name) as well as
  `reasoning_content`, or the stream goes silent while the model thinks and Claude Code gives up;
- cap thinking with vLLM's per-request `thinking_token_budget` (we use 8192 as a runaway guard);
- map Claude Code's own model names to the local model in the proxy;
- read the window from vLLM's `/v1/models` (`max_model_len`), so Claude Code compacts in time.

Details: [Claude Code on it](b70/qwen36-35b-a3b.md#claude-code-on-it) and
[coming from llama-server](b70/vllm-xpu/README.md#if-you-are-coming-from-llama-server).

## 2. Intel Arc Pro B70 with vLLM

For Intel GPUs only, with `intel/llm-scaler-vllm:0.26.0-b2`. Read
[Before you patch vLLM](#before-you-patch-vllm) first. Full notes:
[`b70/vllm-xpu/`](b70/vllm-xpu/README.md).

### `fix-699.py`: compiled mode starts

- **Symptom.** vLLM stops at startup in compiled mode (without `--enforce-eager`) whenever int4 is on
  (`--quantization sym_int4`, or AutoRound through `inc-q40`): `torch.compile` refuses a logging call
  inside the traced code. Only eager mode starts.
  ([intel/llm-scaler#699](https://github.com/intel/llm-scaler/issues/699))
- **For.** Intel only; llm-scaler 0.26.0-b2; any model. Only if you want compiled mode: Intel says XPU
  graphs are not supported yet ([#698](https://github.com/intel/llm-scaler/issues/698)), and our own
  servers run eager.
- **Measured.** Qwen3.8-27B, one stream at ~1.5k tokens of context, compiled with XPU graphs against
  eager: 32.8 against 26.0 tok/s without speculation, 60.4 against 57.1 with MTP 3; at ~16k with
  MTP 3, 56.1 against 47.9.
  ([vLLM notes, section 1](b70/vllm-xpu/README.md#1-compiled-mode-remove-two-log-calls-llm-scaler-699))
- **Apply.** In site-packages: `python3 fix-699.py vllm/model_executor/layers/quantization/sym_int4.py`.
  Serve with `VLLM_XPU_ENABLE_XPU_GRAPH=1` and without `--enforce-eager`; graph capture takes 3-4
  minutes before `/health` answers.
- **Check.** `grep -c 'issue 699' vllm/model_executor/layers/quantization/sym_int4.py` prints `2`, and
  the server starts in compiled mode.
- **Undo.** Copy `sym_int4.py.orig` back, or keep it and serve with `--enforce-eager` (the patch only
  removes two log lines).

### `inc-q40`: AutoRound int4 checkpoints on the fast kernels

- **Symptom.** An AutoRound int4 checkpoint (a model downloaded already quantized to 4 bits) answers
  correctly but slowly, 14-17 tok/s at one stream; in compiled mode with XPU graphs and MTP together it
  answers with garbage from the first token.
- **For.** Intel only; llm-scaler 0.26.0-b2. Checkpoints whose `config.json` has, under
  `quantization_config`, `"bits": 4`, `"group_size": 128`, `"sym": true` and
  `"packing_format": "auto_round:auto_gptq"`, such as
  [`Frozenlock/Qwen3.8-27B-int4-AutoRound`](https://huggingface.co/Frozenlock/Qwen3.8-27B-int4-AutoRound).
  Layers stored any other way keep the stock path. Other models, other group sizes and AWQ packing are
  untested. A checkpoint with 8-bit layers (`Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound`) does not load
  on this image, patched or not.
- **What it does.** Such a checkpoint already holds the 4-bit format of the image's fast `sym_int4`
  kernels, stored in a different order. The patch reorders two tensors after load and runs the layers
  on those kernels.
- **Measured.** Qwen3.8-27B, one stream: eager 54.4 tok/s (~3.5x the stock path); compiled with XPU
  graphs and MTP 3, 61.2-61.5 at ~1.5k and 58.6 at ~16k; 351.4 at 8 streams. 41 of 41 exact on our
  test-log eval. An 18 GB download instead of 56 GB for BF16.
  ([vLLM notes, section 2](b70/vllm-xpu/README.md#2-autoround-int4-checkpoints-on-the-fast-kernels-inc-q40))
- **Apply.** In site-packages: `patch -p1 < /path/to/inc-q40.diff`, or
  `python3 inc-q40.py vllm/model_executor/layers/quantization/inc/schemes` (which keeps `.orig`
  copies). Serve the checkpoint with no `--quantization` flag. Compiled mode also needs `fix-699.py`,
  since the layers now run through the same int4 code.
- **Check.** `grep -l INCXPUQ40LinearMethod vllm/model_executor/layers/quantization/inc/schemes/*.py`
  lists two files, and one-stream decode ([`decode-vllm.py`](b70/bench/decode-vllm.py)) is in the 50s
  or 60s of tok/s, not the teens.
- **Undo.** `VLLM_INC_XPU_Q40=0` restores the stock path without unpatching; `patch -R -p1 <
  inc-q40.diff` or the `.orig` copies remove it.

### Setting: long prompts without an engine reset

- **Symptom.** A very long prompt kills the server (we saw it at ~180k tokens of context), and the
  kernel log shows `GT0: Engine reset: engine_class=ccs`.
- **Cause.** The `xe` driver kills any GPU job that runs longer than 5 s on these cards, and one vLLM
  step is one job. An 8,192-token chunk of prompt over ~180k tokens of context took longer.
- **For.** Intel GPUs on the `xe` driver, with vLLM; any model with a long window.
- **The setting.** `--max-num-batched-tokens 2048`. Measured: prompts at 104k, 146k, 188k and 229k
  tokens answered with no reset, and our Qwen3.8-27B has served a 212,992-token window with it since.
  Whether 2,048 is slower to read a prompt than 8,192 has not been timed. The limit itself is a root
  setting per boot, up to 10,000 ms:
  `/sys/class/drm/card<N>/device/tile0/gt0/engines/ccs/job_timeout_ms`; we keep the default.
  ([vLLM notes](b70/vllm-xpu/README.md#a-213k-token-window-the-xe-drivers-job-time-limit-sets-the-prefill-chunk))
- **Check.** `journalctl -k -b | grep 'Engine reset'` (as root or a member of `adm`) prints nothing new
  after a long request.
- **Undo.** Raise the value again.

### Setting: a model load that hangs with no error

- **Symptom.** vLLM never finishes loading and `/health` never answers, with nothing in vLLM's own
  log. The kernel log shows `Engine reset: engine_class=bcs` for the card. We saw it in about 1 of 20
  loads.
- **For.** Intel GPUs on the `xe` driver, with vLLM.
- **What we do.** Our launcher counts engine resets for its own card's PCI address in
  `journalctl -k -b` (readable without root by the `adm` group) and exits when a new one appears, so
  systemd restarts it. A health-check timeout catches it too, only much later. This is a method, not a
  file in this repo. ([The MoE notes](b70/qwen36-35b-a3b.md#a-load-can-hang-silently-watch-the-kernel-log))
- **Check.** `journalctl -k -b | grep 'Engine reset'` during a load that seems stuck.

### `gdn-mixed-batch/`: a check for the DeltaNet kernel bug

- **What it is.** A test, not a fix.
  [vllm-xpu-kernels#552](https://github.com/vllm-project/vllm-xpu-kernels/pull/552) (open) fixes an
  out-of-bounds write in Intel's DeltaNet kernel for batches that mix speculative and ordinary requests.
  This runs that pull request's test against the kernels in your image.
- **For.** Intel only; the kernels in llm-scaler 0.26.0-b2; models with DeltaNet layers
  (Qwen3.8-27B, Qwen3.6-35B-A3B) served with MTP.
- **Measured.** With a request's tokens scattered through the batch, 0 of 8 cases matched. With each
  request's tokens together, which is how vLLM builds a batch, 8 of 8 matched bit for bit. So on this
  image vLLM's own batches are not affected, and our 4-way concurrent eval stayed at 41 of 41.
  ([its notes](b70/vllm-xpu/gdn-mixed-batch/README.md))
- **Run.** Inside the image with one GPU visible:
  `python3 run-gdn-mixed.py test_gdn_attn_mixed_adapted.py`. It prints PASS or FAIL per case.

## 3. Intel Arc Pro B70 with llama.cpp

Eighteen patches to llama.cpp, applied together to upstream commit `94256114c2` (2026-09-23) by
[`build.sh`](b70/llama.cpp-sycl/build.sh), which checks that the patched source is exactly the one we
measured. All measured on one B70 with Qwen3.8-27B `UD-Q4_K_XL` (unsloth's GGUF). Together, 4 streams
went from 39.3 tok/s to 64 without speculation and 101 with MTP 3; one stream decodes at 24.4 without
speculation and 44 with it. Full notes: [`b70/llama.cpp-sycl/`](b70/llama.cpp-sycl/README.md).

**For.** Intel GPUs on llama.cpp's SYCL backend, measured on Xe2 only; the XMX patches also need a GPU
with XMX matrix engines. `0015` and `0016` are the exceptions: they change no GPU code and apply on
their own ([section 1](#llamacpp-0015-and-0016-speculative-decoding-that-keeps-to-its-limits)). Other llama.cpp
versions need the patches rebased. None is upstream yet.

### Which patches your model file uses

A GGUF file mixes quantization types; llama-server lists them as it loads (`llama_model_loader: -
type q5_K: ...`). The patches speed up the types and situations below. Numbers are kernel timings
(µs) or decode tok/s on the B70.

| patch | what goes faster | when it matters | measured |
|---|---|---|---|
| `0001` | Q5_K mat-vec | several streams | 1.46x at 4 streams |
| `0002` | IQ4_NL mat-vec | several streams | 1.91x / 3.46x / 6.18x at 2 / 4 / 8 |
| `0003` | flash-attention decode | long context, or 2+ streams | 1.07-1.55x (0.91x at 2k, one stream) |
| `0004` | IQ4_NL, IQ4_XS decode of 4-bit codes | always | IQ4_XS 1.08-1.16x, IQ4_NL 1.24x |
| `0005` | one token from each of several streams, e.g. a DeltaNet layer's output | several streams | Q5_K 320 → 73 µs |
| `0006` | copying a recurrent model's per-stream state | hybrid models (DeltaNet) | one row 20.2 → 3.3 µs |
| `0007` | joining short rows (a DeltaNet convolution state) | hybrid models | 5.7 → 3.2 µs |
| `0008` | IQ4_NL in a faster memory layout (needs `0004`) | IQ4_NL weights | about 3x |
| `0009` | nothing: adds test cases | - | coverage only |
| `0010` | attention when query heads per KV head is 3, 5, 6 or 7 (Qwen3.8-27B: 6) | decode at depth | 1.8-2.4x |
| `0011` | Q5_K on the XMX engines | 3-8 tokens a step (streams or drafts) | 235 → 147 µs at 4 |
| `0012` | Q4_K, Q6_K, IQ4_XS on XMX | 4-8 tokens a step | Q4_K 194 → 115 µs at 4 |
| `0013` | large IQ4_XS weights in an XMX layout | 1-8 tokens a step | 185 → 129 µs at 4 |
| `0014` | Q3_K, IQ3_S on XMX | 3-bit GGUFs | IQ3_S 1,314 → 253 µs at 4 |
| `0015`, `0016` | speculative decoding limits (any backend) | MTP speculation | [section 1](#llamacpp-0015-and-0016-speculative-decoding-that-keeps-to-its-limits) |
| `0017` | IQ4_NL and large IQ4_XS gate/up pairs on XMX | speculation, several streams | 4 streams 86.6-87.3 → 91.2-91.5 tok/s |
| `0018` | XMX for 9-32 tokens a step | speculation at several streams | 4 streams 91.2-91.5 → 100.7-101.6 tok/s |

The XMX patches (`0011`-`0014`, `0017`, `0018`) need `0008` and `0009`. Correctness is checked
against the CPU with `test-backend-ops`; the counts are in the notes.

### Build, check, undo

- **Needs.** git and Intel's oneAPI Base Toolkit in `/opt/intel/oneapi`.
- **Apply.** `./build.sh ~/llama-sycl`: it clones llama.cpp, applies the patches, verifies the source
  and builds `llama-server`. How we serve it:

  ```bash
  llama-server -m Qwen3.8-27B-UD-Q4_K_XL.gguf --device SYCL0 -ngl 99 -c 163840 --parallel 4 \
    --cont-batching -fa on --ubatch-size 2048 --jinja \
    --spec-type draft-mtp --spec-draft-n-max 3 --spec-batch-max 16
  ```

  With two cards, pin the server to its card by PCI address:
  `ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 ONEAPI_DEVICE_SELECTOR=level_zero:<n>`.
- **Check.** `build.sh` prints `patched: ... (tree verified)` and then the path of `llama-server`.
  `test-backend-ops test -b SYCL0 -o MUL_MAT` passes, and
  [`decode-streams.py`](b70/bench/decode-streams.py) gives your own before and after.
- **Undo.** Build upstream llama.cpp without the patches. At run time, `GGML_SYCL_XMX_MMVQ=0` turns the
  XMX path off.

## 4. The host CPU: a Ryzen with two CCDs

### `serving-cpu-guard`: keep other work off the server's CCD

- **Symptom.** The model slows down whenever the same machine runs builds or tests. On our Ryzen a
  speculative step took 17-19 ms idle and 30-34 ms under a real mixed load, and a memory-heavy load
  made it about 20 times slower (7 tok/s).
- **For.** A CPU with two or more CCDs (chiplets, each with its own L3 cache); measured on a Ryzen 9
  9950X3D. vLLM in Docker, answering on `/metrics`; systemd. It matters for a model with short steps:
  the Qwen3.6-35B-A3B MoE (~15 ms steps) in eager mode on the B70, where Python launches every GPU
  kernel. The dense Qwen3.8-27B (40 ms steps, mostly GPU work) did not move.
- **What it does.** Keeps the model's containers on one CCD. While any request is in flight, and for
  20 s after, it limits every login session and user service (`user.slice`) to the other CCD; when the
  servers are idle, it gives them every CPU back.
- **Measured.** Under our normal load with the guard running: 15.2 ms per step (190 tok/s), as fast as
  idle. Pinning only the server did not help (35 ms): what costs is sharing its CCD.
  ([vLLM notes, "The host's CPU"](b70/vllm-xpu/README.md#the-hosts-cpu-keep-other-work-off-the-servers-ccd))
- **Apply.** Find your CCDs' CPU lists (`lscpu -e`, or
  `/sys/devices/system/cpu/cpu*/cache/index3/shared_cpu_list`). Copy
  [`serving-cpu-guard.sh`](b70/vllm-xpu/serving-cpu-guard.sh) to `/usr/local/bin/` (executable), set
  container names, ports and CPU lists in the `Environment=` line of
  [`serving-cpu-guard.service`](b70/vllm-xpu/serving-cpu-guard.service), copy it to
  `/etc/systemd/system/`, and run `systemctl enable --now serving-cpu-guard`.
- **Check.** `journalctl -u serving-cpu-guard -f` prints `split: a model is serving` when a request
  arrives and `open: idle 20s` after; `systemctl show user.slice -p AllowedCPUs` shows the other CCD
  while a model serves.
- **Undo.** `systemctl disable --now serving-cpu-guard` gives every CPU back to both.
- **Untested.** CPUs with one CCD (one L3, nothing to split), Intel CPUs, and servers in compiled mode,
  which launch fewer kernels from Python and may be less sensitive. We did not try the server on the
  larger-cache CCD.

## 5. Hardware nothing here was measured on

### NVIDIA Quadro RTX 6000 (Turing, 24 GB)

One of our cards, not yet measured for this repo.

- **Usable as is** (nothing GPU-specific in them): the
  [Claude Code settings](#claude-code-on-a-local-model-settings-not-patches), the
  [tools](#6-tools-that-need-no-particular-gpu), and
  [`effort-template.py`](#effort-templatepy-qwen38-accepts-the-usual-effort-words) if you serve
  Qwen3.8-27B with an engine that passes `reasoning_effort` to the template (vLLM on the B70 did).
- **Untested:** llama.cpp [`0015` and `0016`](#llamacpp-0015-and-0016-speculative-decoding-that-keeps-to-its-limits)
  with CUDA (the code is not GPU code; not built or measured there), and the other
  [section 1](#1-any-gpu-serving-software-and-chat-template-fixes) vLLM patches: they are written
  against the llm-scaler image's vLLM, and where your vLLM's code differs they stop rather than edit.
- **Does not apply:** everything in sections 2 and 3 (Intel only).
- **Another project:** we ran [HyperQwen](https://github.com/syv-ai/HyperQwen) (vLLM 0.29 with its own
  patches) on this card. On Turing its server died at warm-up with Triton's `OutOfResources` for
  shared memory; our fix is an open pull request,
  [HyperQwen#188](https://github.com/syv-ai/HyperQwen/pull/188), which reports Qwen3.8-27B W4A16 at
  31.6 tok/s for one stream and 104.4 for four. Turing has no bf16 and HyperQwen's speculative path
  needs it, so there was no MTP for `draft-vocab.py` to speed up.

### NVIDIA GTX 1080 Ti (Pascal, 11 GB)

One of our cards; nothing measured on it. It cannot hold the models above on its own: Qwen3.8-27B at
4 bits is 18 GB of weights before any cache. The [section 1](#1-any-gpu-serving-software-and-chat-template-fixes)
entries are untested on it, and the [tools](#6-tools-that-need-no-particular-gpu) apply.

### Other Intel Arc cards

Untested. The Arc B580 (12 GB) and Arc Pro B60 (24 GB) are Xe2 like the B70, so the patches in
section 3 change code those cards also run, but their memory bandwidth and core counts differ, and the
gains may too; memory decides which models fit (18 GB of weights for the 27B at 4 bits). Older Arc
A-series cards (Alchemist) are an earlier generation; nothing here was tried on them. Which cards
llm-scaler supports is in Intel's documentation.

### AMD, Apple and other NVIDIA GPUs

Untested. The [section 1](#1-any-gpu-serving-software-and-chat-template-fixes) entries are not tied to
a GPU, with the version limits each entry states, and the
[tools](#6-tools-that-need-no-particular-gpu) apply.

## 6. Tools that need no particular GPU

### `agent-eval/`: can a local model finish real repository work?

- **What it is.** Twelve small repositories, each with a request as a developer would type it, a
  hidden check the agent never sees, and a reference fix. The agent runs in a container that sees only
  the task's copy.
- **For.** Any model behind an OpenAI-compatible server, driven by Claude Code through a proxy, or any
  agent command. Needs Docker and Python.
- **Measured.** Claude Code on our two B70s, 24 runs each: Qwen3.8-27B finished 22, Qwen3.6-35B-A3B 15
  (p = 0.036). ([its notes](b70/agent-eval/README.md#results-2026-09-24),
  [`results.csv`](b70/agent-eval/results.csv))
- **Run.** See [Running it](b70/agent-eval/README.md#running-it).
- **Check.** `python3 run.py --verify` prints `OK` for all twelve tasks: each check fails on the
  untouched repository and passes with the reference fix.

### `bench/`: the measurements behind the numbers

| tool | works against | use it to |
|---|---|---|
| [`decode-streams.py`](b70/bench/decode-streams.py) | llama-server, any backend | measure decode tok/s at N streams and set prompt depths |
| [`decode-vllm.py`](b70/bench/decode-vllm.py) | vLLM or another OpenAI-compatible server | the same, plus a steady-state rate for several streams |
| [`spec-step-cost.sh`](b70/bench/spec-step-cost.sh) | vLLM with speculative decoding | time one speculative step, e.g. before and after `draft-vocab` |
| [`cpu-contention-probe.sh`](b70/bench/cpu-contention-probe.sh) | vLLM in Docker | see whether CPU load on the host slows the model, and whether pinning fixes it |
| [`constrained-probe.py`](b70/bench/constrained-probe.py) | any OpenAI-compatible chat server | check JSON schema, JSON mode, a tool call and a thinking request after any change |
| [`length-cut-check.py`](b70/bench/length-cut-check.py) | vLLM | see whether a cut-off tool call is reported as `length` |

How we measured, and the rules we kept to: [`bench/`](b70/bench/README.md). Our prompt corpus (our
own CI logs) is not published; any JSONL file with a `"text"` field works, and speculative speeds
depend on the text.

### `autoround-dequant-check.py`: is a 4-bit checkpoint sound?

- **What it does.** Reads an AutoRound GPTQ-packed symmetric int4 checkpoint and its BF16 original on
  the CPU (numpy only), dequantizes the layers you name, and prints how close they are.
- **For.** Any machine; any checkpoint in that format with its BF16 original at hand.
- **Measured.** `Frozenlock/Qwen3.8-27B-int4-AutoRound` against the BF16 weights: cosine 0.994,
  relative error 10-11 %. That showed its garbage output in compiled mode was the serving path, not the
  checkpoint.
- **Run.** `python3 autoround-dequant-check.py <quantized dir> <bf16 dir> <layer name> ...`
