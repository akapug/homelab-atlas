# How the B70 numbers were measured

| tool | measures |
|---|---|
| [`decode-streams.py`](decode-streams.py) | aggregate decode tok/s of a llama-server at N simultaneous streams and set prompt depths, from the server's own per-request timings |
| [`decode-vllm.py`](decode-vllm.py) | the same against an OpenAI-compatible server that returns no timings (vLLM): each stream's decode is timed from its first to its last streamed chunk, tokens from `usage` |
| [`spec-step-cost.sh`](spec-step-cost.sh) | the time of one speculative verify step at the server's current draft count: one stream of 1,024 tokens, tokens a step from vLLM's draft counters; it waits for an idle server and flags a run other requests shared |
| [`constrained-probe.py`](constrained-probe.py) | JSON schema, JSON mode, a tool call and a thinking request, the request shapes speculative decoding or a new kernel path can break |

```
decode-streams.py <label> <corpus.jsonl> --port 8080 --streams 1,4 --chars 6000,60000
STREAMS=1,4,8,16 CHARS=6000 decode-vllm.py <label> <corpus.jsonl> 8000          # port; model from /v1/models
constrained-probe.py http://127.0.0.1:8000/v1/chat/completions
```

Both decode tools prefill every stream's prompt first (one-token requests), so the timed pass
reuses the prompt cache and measures decode only. `decode-vllm.py` also prints a **steady** rate:
tokens produced only while every stream is decoding, from the per-chunk token counts vLLM streams
with `continuous_usage_stats`. Trust that one for concurrency. With a hybrid model's prefix cache
some prompts are partly re-prefilled in the timed pass, their streams start late, and at 128
tokens per stream the stagger alone moved one server's 4-stream figure between 72 and 194 tok/s.
Use `TOKENS=512` for concurrency runs. Without that, one stream's decode steps carry
the others' prefill and the "decode" rate is partly prefill. Each stream starts from a different
document, so no two prompts share a prefix. Both print the uncached prompt tokens of the timed
pass: a large value means the cache was lost and that row is not a clean decode measurement.

`--chars` is prompt size in characters: with our text, 6,000 is ~1.5k tokens and 60,000 ~16k.

**The corpus.** Any JSONL file with a `"text"` field per line. Ours is 42 CI test logs (~1 MB) from
our own projects, which we do not publish. Speculative decoding's speed depends on how
predictable the output is, and the prompt asks for a summary of test logs, so expect different
numbers with other text, most of all with MTP enabled. Without speculation the rates depend on
context length, not content.

Rules we kept to: measure on an otherwise idle card; run each configuration at least twice and
report both; report the uncached-prompt column.
