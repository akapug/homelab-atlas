#!/usr/bin/env python3
"""Per-expert routing counts of a workload (split-store P0: rank coding-hot experts). Launched like decode_kl_nv.py
(bind-mount it over /opt/glm53/bench/decode_kl_nv.py and run the image's `decode-kl` entrypoint, GLM53_MODE=nvme),
so the model loads with the serving config. Counts the router's picks BEFORE nv2 lane masking (wraps Nv2.layer), split
into prefill (staged forwards) and decode calls.
Env: GLM53_RS_TEXT     raw text file, prefilled in GLM53_RS_CHUNK-token chunks (4096), one generated token each
     GLM53_RS_PROMPTS  JSON list of raw-text prompts, each decoded greedily for GLM53_RS_DECODE tokens (192)
     GLM53_RS_OUT      output dir (/out): route_stats.json = {"prefill"|"decode"|"total": {layer key: [288 counts]},
                       "meta": {...}}, same per-layer shape as data/stats_own_dec.json
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "glm53"))
import torch
import serve as SX
import nv2
from exllamav3 import Generator, Job
from exllamav3.generator.sampler import ArgmaxSampler

TEXT = os.environ.get("GLM53_RS_TEXT", "")
PROMPTS = os.environ.get("GLM53_RS_PROMPTS", "")
CHUNK = int(os.environ.get("GLM53_RS_CHUNK", "4096"))
NDEC = int(os.environ.get("GLM53_RS_DECODE", "192"))
OUT = os.environ.get("GLM53_RS_OUT", "/out")


def run(gen, ids, n):
    gen.enqueue(Job(input_ids=ids, max_new_tokens=n, sampler=ArgmaxSampler(), stop_conditions=SX.EOS))
    k = 0
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("token_ids") is not None:
                k += r["token_ids"].numel()
    return k


def main():
    SX.load()
    NV = nv2.NV
    L, E = NV.L, NV.E
    cnt = {m: torch.zeros((L, E), dtype=torch.int64, device=NV.dev) for m in ("prefill", "decode")}
    calls = {"prefill": 0, "decode": 0}
    orig = NV.layer

    def layer(li, sel, w, z, bsz):
        s = sel.reshape(-1).long()
        m = "prefill" if NV.pool.stage_n and s.numel() >= NV.pool.stage_min else "decode"
        cnt[m][li] += torch.bincount(s[s >= 0], minlength=E)[:E]
        calls[m] += 1
        return orig(li, sel, w, z, bsz)

    NV.layer = layer
    gen = Generator(model=SX.model, cache=SX.cache, tokenizer=SX.tok, max_chunk_size=SX.args.chunk_size)
    t0 = time.time()
    meta = {"prefill_tokens": 0, "decode_tokens": 0, "chunks": 0, "prompts": 0}
    if TEXT:
        ids = SX.tok.encode(open(TEXT).read(), encode_special_tokens=False)
        for a in range(0, ids.shape[-1], CHUNK):
            c = ids[:, a:a + CHUNK]
            if c.shape[-1] < 64:
                break
            run(gen, c, 1)
            meta["prefill_tokens"] += c.shape[-1]; meta["chunks"] += 1
            print(json.dumps({"chunk": meta["chunks"], "tokens": meta["prefill_tokens"], "s": round(time.time() - t0, 1)}), flush=True)
    if PROMPTS:
        for p in json.load(open(PROMPTS)):
            meta["decode_tokens"] += run(gen, SX.tok.encode(p, encode_special_tokens=False), NDEC)
            meta["prompts"] += 1
            print(json.dumps({"prompt": meta["prompts"], "decode_tokens": meta["decode_tokens"], "s": round(time.time() - t0, 1)}), flush=True)
    torch.cuda.synchronize(NV.dev)
    keys = [m.key for m in NV.pool.mods]
    out = {m: {k: cnt[m][li].tolist() for li, k in enumerate(keys)} for m in cnt}
    out["total"] = {k: [a + b for a, b in zip(out["prefill"][k], out["decode"][k])] for k in keys}
    meta.update({"calls": calls, "seconds": round(time.time() - t0, 1), "text": TEXT, "prompts_file": PROMPTS,
                 "picks": {m: int(cnt[m].sum()) for m in cnt}})
    out["meta"] = meta
    json.dump(out, open(os.path.join(OUT, "route_stats.json"), "w"))
    print("ROUTE_STATS " + json.dumps(meta), flush=True)
    print("DECODE_KL_DONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
