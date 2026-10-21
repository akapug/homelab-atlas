# Ornith-1.5-35B-A3B on one Arc Pro B70: a household assistant and a coding agent on one card

[Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) is a fine-tune of Qwen3.6-35B-A3B,
with the same configuration (35B total, about 3B active per token; 30 Gated DeltaNet layers and 10
attention layers). Since 2026-10-05 one B70 runs it for two jobs at once. The first is a household assistant:
Open WebUI with web search, tools, photos and voice. The second is a coding agent's model. The house goes
first.

This entry is what made that work on one card, and the one bug that cost us speed. It builds on
[Qwen3.6-35B-A3B on one B70](qwen36-35b-a3b.md) and the [vLLM changes](vllm-xpu/).

Unless a line says otherwise, the conditions are:
- one B70, in the Ryzen 9 9950X3D host of the other B70 entries (the cards moved to a Threadripper box later);
- Intel's `llm-scaler-vllm:0.26.0-b2` with the changes in [`vllm-xpu/`](vllm-xpu/);
- the BF16 checkpoint quantized to `sym_int4` at load, fp8 KV cache, eager mode, the model's full
  262,144-token window.

## Why this model: it codes better and answers the house as well

All three models ran on the same card configuration and the same two harnesses, 2026-10-05:

| model | coding: 12 repository tasks x 2 | household: 40 questions |
|---|---|---|
| Ornith-1.5-35B-A3B | **17/24** | 31/40 |
| Qwen3.6-35B-A3B | 13/24 | 32/40 |
| Qwen3.8-35B-A3B-Distill (empero-ai) | 9/24 | 33/40 |

- **Coding** is our [agent eval](agent-eval/): Claude Code in a sandbox, hidden checks, 1,200 s per task.
  - Ornith against Qwen3.6 is p = 0.37: a lead, not a proven gap.
  - Ornith against the distill is p = 0.04.
  - Five of the distill's failures ended its turn right after it announced its next step.
  - Qwen3.6 scored 15/24 in the agent-eval entry's earlier round and 13/24 here: expect run-to-run spread of that size.
- **Household** means 40 synthetic questions put through a throwaway Open WebUI the way a phone drives it.
  - Web search and tools are on.
  - A judge model (Qwen3.8-27B, thinking off) scores each answer against a stored key, without knowing
    which model wrote it.
  - The three scores are a tie in practice.
  - The distill was the fastest of the three (184-188 tok/s short, against Qwen3.6's 169-174), but it made
    more factual errors.

## Speed

One stream, MTP 3 with the [smaller drafter vocabulary](vllm-xpu/), 1,200-token replies. Measured
2026-10-04 in the same few minutes, with the host's load average near 50 from build jobs:

| host cores for the server | where the build jobs ran | decode tok/s |
|---|---|---|
| 2 | on other cores of the same CCD | 102-121 |
| 2 | on the other CCD only | 149-155 |
| 4 | on the other CCD only | 164-167 |

- Short MoE steps are CPU-bound, so the host matters as much as the card. The
  [CPU guard](vllm-xpu/#the-hosts-cpu-keep-other-work-off-the-servers-ccd) keeps other work off the server's CCD.
- For scale: Qwen3.6-35B-A3B with the same speculation measured 189.8 tok/s on this card (2026-09-24, before the
  CPU guard existed).
- MTP is off on the live server today; see the next section for why and what it cost.

## A bug: MTP garbled about 3 % of replies from the first token

Household users saw some replies that were garbage from the first token: runs of digits, CJK fragments,
or text that looked like the prompt. A follow-up such as "huh?" got a normal answer.

- **What triggers it.** A short request must prefill beside a house request. Open WebUI sends one: a small
  task call about 0.2 s before every chat request. A burst of classifier calls does the same.
- **Reproduced** on an identical server on the other B70, with a load generator that sends those short
  requests beside house-style requests:

| arm | garbled house replies |
|---|---|
| MTP 3 and prefix caching on (as served) | 3 of 106 |
| MTP off | 0 of 403, and 0 of 348 in a second run |
| prefix caching off, MTP on | 0 of 157 |
| the same load without the short requests beside it | 0 of ~450 |

- **Trigger.** The switch tests isolate it to speculative decoding and prefix caching together: turning off
  either one removed it. Every garbled reply had 0 cached tokens, so it is not a stale cache hit.
  - Our guess, not shown: vLLM's prefix-cache mode for this hybrid model's DeltaNet state (the "align" mode),
    where upstream vLLM has reports of the same class.
- **Our fix: MTP off, prefix caching kept.** The coding agent needs the prefix cache: its prompts are 50-160k tokens.
- **What it cost.**
  - One stream fell from ~190 to ~118 tok/s. Measured on Qwen3.6-35B-A3B, the same architecture, on this
    card: 8.5 ms a step without speculation.
  - Under the replica's concurrent load, the median house reply got faster: 3.5 s without MTP, against
    6.7 s with MTP and prefix caching off.
- **Testing a fix.** A clean single-stream test proves nothing here: a repro needs the short requests
  beside the long one. We turn MTP back on only after a newer vLLM passes this load.

## Sharing the card: the house first

The coding agent and the household share one server:
- vLLM's priority scheduling: the household runs at -10 and the agent at 10;
- at most 4 sequences;
- a proxy in front of the agent lets 2 of its requests in at once, so the house always has 2 slots.

One setting mattered most: vLLM's cap on one request's prefill per step (`--long-prefill-token-threshold`). We
measured it with Qwen3.6-35B-A3B on this card, under a nonstop stream of 19k-token agent prompts:
- At 4,096, the household's median answer went from 1.9 to 3.0 s, and p90 from 6.6 to 10.4 s.
- At 2,048, each step that carries an agent prompt is shorter, so the house waits less between words.
- 2,048 is the floor. vLLM requires the cap to be at least the KV block size, which this hybrid model gets
  on XPU (2,048 tokens). 1,024 failed every start.

## Thinking, photos and voice

- **Thinking with a budget.** Thinking is on with a 1,024-token budget (temperature 0.6, top_p 0.95,
  top_k 20, presence penalty 1.5).
  - The household set scored 33/40, against 31/40 with thinking off.
  - An easy question took 1.0 s.
  - A hard bill-split took 12.7 s and 2,000 tokens, against 6,223 tokens with no cap.
  - A no-thinking preset (temperature 0.7, top_p 0.8) answers its first word in 0.2 s through Open WebUI.
- **Photos.** The checkpoint carries the vision tower, and vLLM loads it, so photos need no separately
  deployed vision model. The tower and image-processing buffers still use VRAM; their incremental footprint
  was not measured here. Time to the first word, measured at the server:

  | photo size | first word |
  |---|---|
  | 3840x2160 | 5.7 s (the encoder holds the card's step meanwhile) |
  | 1600x900 | 0.49 s |
  | 1280x720 | 0.30 s |

  So we have the browser shrink photos to 1600 px before upload. A 4K phone photo then arrived as
  1600x900 (137 KB) and was answered correctly in 3.6 s.
- **Voice, on CPUs.** No GPU: both run on another box's CPUs, a 2019 dual Xeon Gold 5215.
  - Speech to text: NVIDIA Parakeet TDT 0.6B v3, int8, on 8 CPUs, turns a 6.8 s question into text in 0.9 s.
    Open WebUI's built-in whisper took 3.1 s (base) and 14 s (large-v3-turbo) on 4 CPUs.
  - Text to speech: Kokoro-82M v0.9.0 on 6 CPUs speaks a short sentence in 0.6 s.
  - Call mode, from the end of speech to the first spoken word, with a headless browser and a recorded
    7.5 s weather question: 12.35 and 13.86 s at first (a third run had not spoken after 22 s, still
    thinking).
  - Two changes brought that to 9.49, 8.41 and 8.67 s:
    1. pass the browser's recording to Parakeet as it is, with no re-encode (2.15 s to 1.29 s of
       transcription);
    2. give call-mode requests the no-thinking settings.
  - Where the ~8.7 s goes now: voice-activity wait ~2.0 s, transcription ~1.3 s, the weather tool call and
    first sentence ~2.4 s, and speaking the first sentence 2.5-3 s. Speech is the next lever.

## The checkpoint served today

Since 2026-10-19 the card serves a prequantized copy instead of quantizing at load:
[`wiesion/Ornith-1.5-35B-A3B-int4-AutoRound-GPTQ-XPU`](https://huggingface.co/wiesion/Ornith-1.5-35B-A3B-int4-AutoRound-GPTQ-XPU),
revision `15c67bd6f8bb`, with `--quantization gptq`.

- **One trap.** Pass the original model's tokenizer (`--tokenizer <the BF16 Ornith directory>`). The GPTQ
  repository's tokenizer differs in its regex and special tokens.
- **Measured with it.**
  - Process start to first health: 145 s (model loading 50.5 s), not from a cold disk cache.
  - Our checks passed: model list, fast reply, thinking, tool call and vision.
- We have not re-measured decode speed or the two evals on this copy.

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [arc](https://github.com/akapug/arc) + [fab](https://github.com/akapug/fab) for [@akapug](https://github.com/akapug)
