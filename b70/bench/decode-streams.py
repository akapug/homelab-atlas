#!/usr/bin/env python3
"""Aggregate decode rate of a llama-server at N simultaneous streams and set prompt depths,
from the server's own per-request timings.

Each stream is pinned to its own slot and prefilled first (n_predict 1), so the timed pass
reuses the cache and measures decode only: otherwise one stream's decode steps carry the
others' prefill chunks and the rate measures prefill. Streams start from different logs,
so no two prompts share a prefix. The server needs at least max(streams) slots.

    decode-streams.py <label> <corpus.jsonl> [--port 8083] [--streams 1,4] [--chars 6000,60000]
                      [--tokens 128]

--chars is prompt size in characters of log text: 6000 is ~1.5k tokens, 60000 ~16k.
The corpus is JSONL with a "text" field per line (CI test logs in our runs).
Output, one line per (depth, streams):
    <label> depth~<tokens> streams=<n>  decode aggregate <tok/s>  (uncached prompt tokens <= k)
A large "uncached" number means the cache was lost and the rate includes prefill.
"""
import argparse
import concurrent.futures as cf
import json
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("corpus")
    ap.add_argument("--port", default="8083")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--streams", default="1,4")
    ap.add_argument("--chars", default="6000,60000")
    ap.add_argument("--tokens", type=int, default=128)
    a = ap.parse_args()
    ep = f"http://{a.host}:{a.port}/completion"
    logs = [json.loads(l)["text"] for l in open(a.corpus)]
    streams = [int(s) for s in a.streams.split(",")]

    def one(args):
        chars, i, n = args
        text = "\n\n".join(logs[7 * i % len(logs):] + logs[:7 * i % len(logs)])[:chars]
        body = {"prompt": f"Summarise the failures in these logs.\n\n{text}\n\nSummary:",
                "n_predict": n, "temperature": 0, "ignore_eos": True, "cache_prompt": True,
                "seed": 1, "id_slot": i}
        r = urllib.request.Request(ep, json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(r, timeout=3600).read())["timings"]

    def run(chars, n_streams, n):
        with cf.ThreadPoolExecutor(n_streams) as ex:
            return list(ex.map(one, [(chars, i, n) for i in range(n_streams)]))

    run(2000, 1, 8)                               # warm-up
    for chars in [int(c) for c in a.chars.split(",")]:
        run(chars, max(streams), 1)               # prefill every slot the largest run uses
        for n_streams in streams:
            ts = run(chars, n_streams, a.tokens)
            fresh = max(t["prompt_n"] for t in ts)
            depth = sum(t["cache_n"] + t["prompt_n"] for t in ts) // n_streams
            dec = sum(t["predicted_per_second"] for t in ts)
            print(f"{a.label:<8} depth~{depth:6d} streams={n_streams:<3d} decode aggregate {dec:7.1f} tok/s"
                  f"  (uncached prompt tokens <= {fresh})", flush=True)


if __name__ == "__main__":
    main()
