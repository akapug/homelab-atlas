"""decode-streams.py for a vLLM OpenAI server: same prompts, depths, stream counts and output
format, so the numbers line up with the llama-server runs.

decode-streams.py reads llama-server's `timings` (predicted_per_second, prompt_n, cache_n) and
pins slots with `id_slot`; vLLM returns none of that. Here each stream is a streamed
/v1/completions request, and its decode rate is (completion_tokens - 1) / (last chunk - first
chunk): the prompt's prefill (time to first token) is excluded, as it is from
predicted_per_second. The aggregate is the sum over streams, as in decode-streams.py; a wall-clock
aggregate (all tokens after each stream's first / span from the earliest first chunk to the latest
last chunk) is printed beside it. Prompts are prefilled first with max_tokens 1, so the timed
pass reuses vLLM's prefix cache (serve with --enable-prefix-caching; with
--enable-prompt-tokens-details the uncached prompt tokens are reported). A text sample is printed
per row so garbage output (a broken quant path) is visible.
Usage: decode-vllm.py <label> <corpus.jsonl> [port, default 8093] [model, default: the first
model the server's /v1/models lists]
Env: STREAMS (default "1,4") and CHARS (prompt characters per depth, default "6000,60000"), for
the many-stream curve, e.g. STREAMS=1,4,8,16 CHARS=6000 (the KV pool must hold streams x depth);
TOKENS (default 128) per stream.
"steady" is the aggregate over only the span in which every stream was decoding, from the
per-chunk token counts vLLM streams with continuous_usage_stats. It is the concurrency figure to
trust: with a hybrid model's prefix cache some prompts are partly re-prefilled in the timed pass,
their streams start late, and both "aggregate" (a late stream decodes alone for a while) and
"wall" (the stagger counts as idle) then misread concurrent throughput."""
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request

port = sys.argv[3] if len(sys.argv) > 3 else "8093"
EP = f"http://127.0.0.1:{port}/v1/completions"
model = sys.argv[4] if len(sys.argv) > 4 else json.loads(urllib.request.urlopen(
    EP.replace("/completions", "/models"), timeout=30).read())["data"][0]["id"]
label = sys.argv[1]
logs = [json.loads(l)["text"] for l in open(sys.argv[2])]


def blob(offset):
    # A different starting log per stream, so the prompts share no prefix.
    return "\n\n".join(logs[offset:] + logs[:offset])


def one(args):
    chars, i, n = args
    body = {"model": model, "prompt": f"Summarise the failures in these logs.\n\n{blob((7 * i + i // 6) % len(logs))[:chars]}\n\nSummary:",
            "max_tokens": n, "temperature": 0, "ignore_eos": True, "seed": 1,
            "stream": True, "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
    r = urllib.request.Request(EP, json.dumps(body).encode(), {"Content-Type": "application/json"})
    first = last = usage = None
    text, marks = [], []  # (time, completion tokens so far) per chunk, for the overlap window
    with urllib.request.urlopen(r, timeout=1800) as resp:
        for raw in resp:
            line = raw.strip()
            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            if ev.get("choices"):
                last = time.perf_counter()
                first = first or last
                text.append(ev["choices"][0].get("text") or "")
                if ev.get("usage"):
                    marks.append((last, ev["usage"]["completion_tokens"]))
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return {"n": usage["completion_tokens"], "prompt": usage["prompt_tokens"], "cached": cached, "marks": marks,
            "first": first, "last": last, "text": "".join(text)}


def run(chars, streams, n):
    with cf.ThreadPoolExecutor(streams) as ex:
        return list(ex.map(one, [(chars, i, n) for i in range(streams)]))


w = run(2000, 1, 8)  # warm-up
print(f"{label:<8} warm-up sample: {w[0]['text']!r}", flush=True)
counts = [int(x) for x in os.environ.get("STREAMS", "1,4").split(",")]
for chars in [int(x) for x in os.environ.get("CHARS", "6000,60000").split(",")]:
    run(chars, max(counts), 1)  # prefill every stream's prompt into the prefix cache
    for streams in counts:
        ts = run(chars, streams, int(os.environ.get("TOKENS", "128")))
        dec = sum((t["n"] - 1) / (t["last"] - t["first"]) for t in ts)
        # Steady state: only the span in which every stream was decoding, so a stream that started
        # late (its prompt re-prefilled) or ran alone at the end distorts nothing.
        lo, hi = max(t["first"] for t in ts), min(t["last"] for t in ts)
        span = [[c for tm, c in t["marks"] if lo <= tm <= hi] for t in ts]
        overlap = (sum(m[-1] - m[0] for m in span if len(m) > 1) / (hi - lo)) if hi > lo and all(len(m) > 1 for m in span) else float("nan")
        wall = sum(t["n"] - 1 for t in ts) / (max(t["last"] for t in ts) - min(t["first"] for t in ts))
        depth = sum(t["prompt"] for t in ts) // streams
        fresh = "?" if any(t["cached"] is None for t in ts) else max(t["prompt"] - t["cached"] for t in ts)
        print(f"{label:<8} depth~{depth:6d} streams={streams}  decode aggregate {dec:6.1f} tok/s"
              f"  (steady {overlap:6.1f}; wall {wall:6.1f}; tokens {min(t['n'] for t in ts)}-{max(t['n'] for t in ts)};"
              f" uncached prompt tokens <= {fresh})  sample: {ts[0]['text'][:60]!r}", flush=True)
