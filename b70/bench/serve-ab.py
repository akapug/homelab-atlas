#!/usr/bin/env python3
"""serve-ab.py BASE_URL MODEL [--priority N] [--idle S]: the same short measurement set against any OpenAI-compatible
vLLM server, for comparing one model across accelerators (first use 2026-10-07: Qwen3.8-27B on a B70, XPU, against
the same int4 weights on an RTX 3090 Ti, CUDA). Stdlib only, one JSON line per measurement.
The deeper per-dimension tool is decode-vllm.py (depth x streams); this is the short set, with a gate for a server
that other users share.

  decode   1 stream, short prompt, 512 tokens out (x3)
  prefill  a ~16k-token prompt, 1 token out: prompt tokens / time to first token (x2)
  depth    1 stream at ~32k tokens of context, 512 out (x2)
  streams3 3 concurrent streams, 512 out each: aggregate tok/s (x1)

Every prompt starts with a random salt, so the prefix cache never hits. Sampling: temperature 0, thinking off,
ignore_eos (every request decodes its full length). --priority sends vLLM's request priority (with
--scheduling-policy priority, higher runs later: 100 lets other users go first). --idle S waits before each measurement
until the server's /metrics has shown no running or waiting request for S seconds (a shared server);
the load right after each measurement is printed as load_after, so a reading taken beside other traffic shows.
  python3 serve-ab.py http://127.0.0.1:8000/v1 qwen27
  python3 serve-ab.py http://127.0.0.1:8000/v1 qwen27 --priority 100 --idle 20"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import random
import time
import urllib.request

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec "
         "romeo sierra tango uniform victor whiskey xray yankee zulu").split()
TOK_PER_WORD = 3.44  # Qwen3.8's tokenizer on "word<0-99>" filler (measured: 16,000 asked -> 16,037-16,058 prompt tokens)
TASK = "Write a long, detailed technical essay about building a distributed key-value store in Rust."

a = argparse.ArgumentParser()
a.add_argument("base")
a.add_argument("model")
a.add_argument("--priority", type=int)
a.add_argument("--idle", type=float, default=0)
o = a.parse_args()
BASE = o.base.rstrip("/")
METRICS = BASE.rsplit("/v1", 1)[0] + "/metrics"


def filler(n):
    r = random.Random()
    return " ".join(r.choice(WORDS) + str(r.randint(0, 99)) for _ in range(int(n / TOK_PER_WORD)))


def load():
    with urllib.request.urlopen(METRICS, timeout=5) as r:
        return sum(float(l.split()[-1]) for l in r.read().decode().splitlines()
                   if l.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{")))


def wait_idle():
    quiet = time.time()
    while time.time() - quiet < o.idle:
        if load():
            quiet = time.time()
        time.sleep(2)


def run(prompt, max_tokens, out):
    body = {"model": o.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    if o.priority is not None:
        body["priority"] = o.priority
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, first, usage = time.time(), None, None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            delta = d["choices"][0]["delta"] if d.get("choices") else {}
            if first is None and (delta.get("content") or delta.get("reasoning_content")):
                first = time.time()
    if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] <= 0
                                          for k in ("prompt_tokens", "completion_tokens")):
        raise ValueError("missing or invalid usage token counts")
    if first is None:
        raise ValueError("stream produced no content")
    out.append({"ttft": first - t0, "decode_s": time.time() - first, "prompt": usage["prompt_tokens"],
                "out": usage["completion_tokens"]})


def salt():
    return f"[{random.random()}] "


def one(prompt, n):
    r = []
    run(prompt, n, r)
    return r[0]


def decode():
    x = one(salt() + TASK, 512)
    return {"prompt_tok": x["prompt"], "out_tok": x["out"], "tok_s": round((x["out"] - 1) / x["decode_s"], 1)}


def prefill():
    x = one(salt() + filler(16000) + "\n\nSay ok.", 1)
    return {"prompt_tok": x["prompt"], "ttft_s": round(x["ttft"], 2), "tok_s": round(x["prompt"] / x["ttft"], 1)}


def depth():
    x = one(salt() + "Background notes (ignore):\n" + filler(32000) + "\n\n" + TASK, 512)
    return {"prompt_tok": x["prompt"], "ttft_s": round(x["ttft"], 2), "tok_s": round((x["out"] - 1) / x["decode_s"], 1)}


def streams3():
    r = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(run, salt() + TASK, 512, r) for _ in range(3)]
        for job in jobs:
            job.result()
    wall = time.time() - t0
    return {"out_tok": sum(x["out"] for x in r), "tok_s": round(sum(x["out"] for x in r) / wall, 1),
            "per_stream": [round((x["out"] - 1) / x["decode_s"], 1) for x in r]}


for name, f, n in (("decode", decode, 3), ("prefill", prefill, 2), ("depth", depth, 2), ("streams3", streams3, 1)):
    for _ in range(n):
        if o.idle:
            wait_idle()
        row = {"m": name, "at": time.strftime("%H:%M:%SZ", time.gmtime()), **f()}
        row["load_after"] = load()
        print(json.dumps(row), flush=True)
