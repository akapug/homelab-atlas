#!/usr/bin/env python3
"""Train a LoRA on the last blocks of a MoE GGUF, from the residual stream llama.cpp itself computed.

    tail_trainer.py info    MODEL.gguf
    tail_trainer.py weights MODEL.gguf --first L --out WEIGHTS
    (tail-dump, the C++ tool in dump/, writes DUMP: the input of block L and the engine's logits, per text window)
    tail_trainer.py gate    --dump DUMP --weights WEIGHTS --out gate.json  [--device D] [--match q8|none] [--seqs 8]
                            [--lora A.gguf ...]
    tail_trainer.py train   --dump DUMP --weights WEIGHTS --out RUN [--steps 20] [--rank 8] [--lr 5e-4]
                            [--targets attn|o|attn+shexp] [--blocks A-B]       -> RUN/adapter.gguf, RUN/train.json
    tail_trainer.py eval    --dump DUMP --weights WEIGHTS --out eval.json [--lora A.gguf ...] [--seqs N]

Why: the whole model never has to fit a card. llama.cpp runs blocks 0..L-1 once, quantized, on the CPU (tail-dump);
torch holds only blocks L..n-1 and the head (dequantized from the same GGUF, routed experts in host RAM when the card
is small), so a 16 GB card trains a 30B MoE's tail. The gate first checks that the torch tail reproduces the engine's
own logits; the adapter is written as a llama.cpp LoRA GGUF (llama-server --lora RUN/adapter.gguf).

--device: cuda (NVIDIA), rocm (AMD; an alias of cuda: ROCm torch serves HIP through torch.cuda), xpu (Intel), cpu;
cuda:1 etc. picks a card. Per-card defaults (from tail-portable/): TF32 only on NVIDIA sm_80+ and only for the
optimizer's steps (every reported number is exact f32); rope angles on the CPU when the device has no f64; experts in
pinned host memory when the tail does not fit (--offload-experts auto); the MoE recompute reuses the forward's routing
(non-deterministic atomics on ROCm and XPU otherwise flip near-tied experts); on NVIDIA the allocator defaults to
expandable_segments:True (fragmentation on 11 GB cards), never on ROCm (gfx900: every allocation fails with it). Every file is read once and dropped from the page cache.

gate (thresholds fixed in this file before any run): on the logit rows of the first --seqs windows of DUMP,
  top-1 agreement >= 0.97 and mean KL(engine || torch) <= 0.02 nats; mean |delta log p| over the engine's top 20
  tokens is reported beside them. Exit 0 on a pass, 3 on a fail.
"""
import argparse
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GATE = {"top1": 0.97, "kl": 0.02}
# arch -> what the metadata does not say (llama.cpp hard-codes it per arch: llama_model_rope_type, build_moe_ffn's
# norm_w argument in src/models/<arch>.cpp)
ARCHS = {
    "qwen3moe": {"rope": "neox", "norm_w": True},
    "qwen2moe": {"rope": "neox", "norm_w": False},
    "olmoe": {"rope": "neox", "norm_w": False},
    "llama": {"rope": "norm", "norm_w": True},       # Mixtral-style MoE GGUFs use arch llama with expert_count > 0
}
MEASURED_ARCHS = ("qwen3moe",)                         # the gate has passed on these (README, results)


# ------------------------------------------------------------------------------------------------ files
def drop_cache(path):
    """POSIX_FADV_DONTNEED on a file read once: a big expert stack otherwise sits in the page cache beside its copy in
    process memory and pushes the box into reclaim."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except (OSError, AttributeError):
        pass


def write_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


# ------------------------------------------------------------------------------------------------ GGUF metadata
def field(r, k, default=None):
    """A GGUF metadata value (scalar, string or array) from a gguf.GGUFReader, or default."""
    f = r.fields.get(k)
    if f is None:
        return default
    import gguf
    t = f.types[0] if f.types else None
    if t == gguf.GGUFValueType.STRING:
        return bytes(f.parts[f.data[0]]).decode()
    if t == gguf.GGUFValueType.ARRAY:
        if len(f.types) > 1 and f.types[1] == gguf.GGUFValueType.STRING:
            return [bytes(f.parts[i]).decode() for i in f.data]
        return [f.parts[i].tolist()[0] for i in f.data]
    return f.parts[f.data[0]].tolist()[0]


def spec_of(r, first):
    """tail_arch.Spec from a GGUFReader; refuses what the torch tail does not implement."""
    arch = field(r, "general.architecture")
    if arch not in ARCHS:
        raise SystemExit(f"architecture {arch!r} is not implemented (implemented: {', '.join(sorted(ARCHS))})")
    g = lambda k, d=None: field(r, f"{arch}.{k}", d)
    n_expert = int(g("expert_count", 0) or 0)
    if n_expert == 0:
        raise SystemExit(f"{arch}: expert_count is 0 (a dense model); this tool trains MoE tails")
    if g("expert_gating_func", 1) not in (1, None):
        raise SystemExit(f"{arch}: expert_gating_func {g('expert_gating_func')} (only softmax gating is implemented)")
    if (g("rope.scaling.type", "none") or "none") not in ("none", "linear") or float(g("rope.scaling.factor", 1.0) or 1.0) != 1.0:
        raise SystemExit(f"{arch}: rope scaling {g('rope.scaling.type')} x{g('rope.scaling.factor')} is not implemented")
    n_head = int(g("attention.head_count"))
    n_kv = g("attention.head_count_kv", n_head)
    if isinstance(n_kv, list):
        if len(set(n_kv)) != 1:
            raise SystemExit(f"{arch}: per-layer head_count_kv is not implemented")
        n_kv = n_kv[0]
    n_embd = int(g("embedding_length"))
    head_dim = int(g("attention.key_length", n_embd // n_head))
    if int(g("attention.value_length", head_dim)) != head_dim:
        raise SystemExit(f"{arch}: key_length != value_length is not implemented")
    n_layer = int(g("block_count"))
    if not 1 <= first < n_layer:
        raise SystemExit(f"--first {first}: the model has blocks 0..{n_layer - 1} and the tail starts at 1 or later")
    vocab = field(r, "tokenizer.ggml.tokens")
    norm_w = g("expert_weights_norm")
    return {"arch": arch, "n_layer": n_layer, "n_embd": n_embd, "n_head": n_head, "n_head_kv": int(n_kv),
            "head_dim": head_dim, "n_rot": int(g("rope.dimension_count", head_dim)),
            "rope_base": float(g("rope.freq_base", 10000.0)), "rope": ARCHS[arch]["rope"],
            "eps": float(g("attention.layer_norm_rms_epsilon")), "n_expert": n_expert,
            "n_expert_used": int(g("expert_used_count")),
            "norm_w": bool(ARCHS[arch]["norm_w"] if norm_w is None else norm_w),
            "w_scale": float(g("expert_weights_scale", 0.0) or 0.0), "n_vocab": len(vocab) if vocab else None,
            "first": first}


def wanted(name, first):
    if name in ("output_norm.weight", "output.weight"):
        return True
    if name.startswith("blk."):
        return int(name.split(".")[1]) >= first
    return False


def cmd_info(a):
    import gguf
    r = gguf.GGUFReader(a.gguf)
    s = spec_of(r, a.first or int(field(r, f"{field(r, 'general.architecture')}.block_count")) - 1)
    types = {}
    for t in r.tensors:
        if t.name.startswith(f"blk.{s['n_layer'] - 1}.") or t.name.startswith("output"):
            types[t.name] = t.tensor_type.name
    print(json.dumps({"spec": s, "gate_measured_on_this_arch": s["arch"] in MEASURED_ARCHS,
                      "last_block_and_head_types": types}, indent=1))
    return 0


def cmd_weights(a):
    """Dequantize blocks --first..n-1 and the head: one OUT/<name>.npy per tensor (numpy order, [n_out, n_in] and
    [n_expert, n_out, n_in]); quantized tensors as float16 (the dequantized value rounded to f16), F32/F16/BF16 as
    float32 (exact). OUT/index.json: type and shape per tensor; OUT/spec.json: the metadata the tail needs.
    Resumable: a tensor whose .npy exists is skipped."""
    import gguf
    import numpy as np
    from gguf import quants
    r = gguf.GGUFReader(a.gguf)
    spec = spec_of(r, a.first)
    os.makedirs(a.out, exist_ok=True)
    names = {t.name for t in r.tensors}
    tie = "output.weight" not in names
    index = {}
    t0 = time.time()
    for t in r.tensors:
        name = t.name
        if not (wanted(name, a.first) or (tie and name == "token_embd.weight")):
            continue
        qt = t.tensor_type
        shape = [int(x) for x in reversed([int(n) for n in t.shape])]
        path = os.path.join(a.out, name + ".npy")
        if not os.path.exists(path):
            exact = qt in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16)
            dt = np.float32 if exact else np.float16
            if len(shape) == 3:
                arr = np.empty(shape, dtype=dt)
                for e in range(shape[0]):          # one expert at a time: the f32 peak is one expert
                    arr[e] = np.asarray(quants.dequantize(t.data[e], qt)).reshape(shape[1:])
            else:
                arr = np.asarray(quants.dequantize(t.data, qt)).reshape(shape).astype(dt)
            np.save(path + ".tmp.npy", arr)
            os.replace(path + ".tmp.npy", path)
            drop_cache(path)
            print(f"{name} {qt.name} {tuple(arr.shape)} {arr.dtype} {time.time() - t0:.0f}s", flush=True)
            del arr
        index[name] = {"type": qt.name, "shape": shape}
    missing = [n for n in ("output_norm.weight",) if n not in index]
    if missing:
        raise SystemExit(f"tensors missing from the GGUF: {missing}")
    write_json(os.path.join(a.out, "index.json"), index)
    write_json(os.path.join(a.out, "spec.json"), spec)
    drop_cache(a.gguf)
    print(json.dumps({"out": a.out, "tensors": len(index), "spec": spec, "s": round(time.time() - t0)}))
    return 0


# ------------------------------------------------------------------------------------------------ dump
class Dump:
    """A tail-dump directory: per window, its tokens, the residual entering the tail and the engine's logit rows."""

    def __init__(self, d):
        self.dir = d
        lines = open(os.path.join(d, "index.tsv")).read().splitlines()[1:]
        self.seqs = [ln.split("\t")[0] for ln in lines if ln.strip()]
        if not self.seqs:
            raise SystemExit(f"{d}: no windows in index.tsv")

    def meta(self, s):
        out = {}
        for ln in open(os.path.join(self.dir, s, "meta.tsv")):
            f = ln.rstrip("\n").split("\t")
            out[f[0]] = (int(f[1]), int(f[2])) if len(f) == 3 else int(f[1])
        return out

    def _read(self, s, name, dtype):
        import numpy as np
        p = os.path.join(self.dir, s, name)
        a = np.fromfile(p, dtype)
        drop_cache(p)
        return a

    def tokens(self, s):
        import numpy as np
        return self._read(s, "tokens.i32", np.int32)

    def resid(self, s, first):
        import numpy as np
        m = self.meta(s)
        name = f"l_out-{first - 1}"
        if name not in m:
            raise SystemExit(f"{self.dir}/{s}: no {name} (dump it with --layers {first})")
        rows, w = m[name]
        return self._read(s, name + ".f32", np.float32).reshape(rows, w)

    def logits(self, s):
        import numpy as np
        m = self.meta(s)
        return (self._read(s, "logits.f32", np.float32).reshape(m["logit_rows"], m["n_vocab"]),
                self._read(s, "logit_pos.i32", np.int32))


# ------------------------------------------------------------------------------------------------ torch side
def norm_device(device):
    return "cuda" + device[4:] if device.startswith("rocm") else device


def accel(torch, device):
    return {"cuda": torch.cuda, "xpu": getattr(torch, "xpu", None)}.get(torch.device(device).type)


def device_info(torch, device):
    info = {"device": device, "torch": torch.__version__}
    m = accel(torch, device)
    if m is None:
        return info | {"backend": "cpu"}
    p = m.get_device_properties(torch.device(device))
    info |= {"name": p.name, "total_gb": round(p.total_memory / 2**30, 2)}
    if torch.device(device).type == "xpu":
        return info | {"backend": "xpu", "driver": getattr(p, "driver_version", None)}
    if torch.version.hip:
        return info | {"backend": "hip", "hip": torch.version.hip, "arch": getattr(p, "gcnArchName", None)}
    return info | {"backend": "cuda", "cuda": torch.version.cuda, "sm": f"{p.major}{p.minor}"}


def tf32_ok(torch, device):
    if torch.device(device).type != "cuda" or torch.version.hip:
        return False
    return torch.cuda.get_device_capability(torch.device(device))[0] >= 8


def set_tf32(torch, on):
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


def sync(torch, device):
    m = accel(torch, device)
    if m is not None:
        m.synchronize()


def peak_gb(torch, device, reset=False):
    m = accel(torch, device)
    if m is None:
        return 0.0
    if reset:
        m.reset_peak_memory_stats()
        return 0.0
    return m.max_memory_allocated() / 2**30


def want_offload(torch, mode, weights, device, headroom_gb=3.0):
    if mode in ("on", "off"):
        return mode == "on"
    m = accel(torch, device)
    if m is None:
        return False
    try:
        free = m.mem_get_info(torch.device(device))[0]
    except (AttributeError, RuntimeError):
        free = m.get_device_properties(torch.device(device)).total_memory
    size = sum(os.path.getsize(os.path.join(weights, k + ".npy")) for k in json.load(open(os.path.join(weights, "index.json"))))
    return size + headroom_gb * 2**30 > free


def load_tail(a, lora_targets=(), rank=8):
    import numpy as np
    import torch
    import tail_arch as ta
    spec = ta.Spec.from_json(json.load(open(os.path.join(a.weights, "spec.json"))))
    index = json.load(open(os.path.join(a.weights, "index.json")))
    offload = want_offload(torch, a.offload_experts, a.weights, a.device)
    W = {}
    for k in index:
        p = os.path.join(a.weights, k + ".npy")
        W[k] = np.load(p)
        drop_cache(p)
    targets = resolve_targets(ta, spec, index, lora_targets, getattr(a, "blocks", None))
    tail = ta.Tail(spec, W, {k: v["type"] for k, v in index.items()}, device=a.device, match=a.match, lora=targets,
                   rank=rank, offload_experts=offload)
    del W
    for path in getattr(a, "lora", None) or []:
        tail.add_adapter(read_adapter(path))
    tail.notes["offload_experts"] = offload
    return tail, spec, targets


def resolve_targets(ta, spec, index, which, blocks):
    if not which:
        return []
    t = ta.default_targets(spec, set(index), which)
    if blocks:
        lo, hi = (int(x) for x in blocks.split("-"))
        t = [n for n in t if lo <= int(n.split(".")[1]) <= hi]
    if not t:
        raise SystemExit("no LoRA target in this tail for --targets/--blocks")
    return t


def compare(eng, tor):
    """Agreement of two log-prob matrices [R, V] (numpy): top-1, KL(engine || torch), top-20 |delta log p|."""
    import numpy as np
    eng, tor = eng.astype(np.float64), tor.astype(np.float64)
    kl = (np.exp(eng) * (eng - tor)).sum(1)
    top = np.argsort(-eng, 1)[:, :20]
    d20 = np.abs(np.take_along_axis(eng, top, 1) - np.take_along_axis(tor, top, 1))
    return {"rows": int(len(kl)), "top1": float((eng.argmax(1) == tor.argmax(1)).mean()),
            "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
            "top20_abs_mean": float(d20.mean()), "top20_abs_max": float(d20.max())}


def log_softmax_np(x):
    import numpy as np
    x = x.astype(np.float64)
    m = x.max(1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))


def cmd_gate(a):
    import numpy as np
    import torch
    set_tf32(torch, False)
    d = Dump(a.dump)
    seqs = d.seqs[:a.seqs]
    t0 = time.time()
    tail, spec, _ = load_tail(a)
    load_s = time.time() - t0
    peak_gb(torch, a.device, reset=True)
    E, T, per = [], [], {}
    t0 = time.time()
    with torch.no_grad():
        for s in seqs:
            x = torch.from_numpy(d.resid(s, spec.first))
            lg, pos = d.logits(s)
            lp = tail(x, pos.astype(np.int64)).double().cpu().numpy()
            e = log_softmax_np(lg)
            per[s] = compare(e, lp)
            E.append(e)
            T.append(lp)
            print(s, json.dumps(per[s]), flush=True)
    sync(torch, a.device)
    m = compare(np.concatenate(E), np.concatenate(T))
    ok = bool(m["top1"] >= GATE["top1"] and m["kl_mean"] <= GATE["kl"])
    out = {"pass": ok, "thresholds": GATE, "all": m, "match": a.match, "first": spec.first, "arch": spec.arch,
           "adapters": [os.path.basename(p) for p in a.lora or []], "windows": len(seqs), "per_window": per,
           "load_s": round(load_s, 1), "forward_s": round(time.time() - t0, 1),
           "card": device_info(torch, a.device) | {"peak_gb": round(peak_gb(torch, a.device), 2), "notes": tail.notes,
                                                    "f64_on_card": tail.f64}}
    write_json(a.out, out)
    print(json.dumps({k: v for k, v in out.items() if k != "per_window"}))
    return 0 if ok else 3


def seq_loss(torch, tail, d, s, first, chunk):
    """Mean next-token cross-entropy over every position of window s."""
    tok = torch.from_numpy(d.tokens(s).astype("int64")).to(tail.device)
    x = torch.from_numpy(d.resid(s, first))
    hn = tail.hidden(x, None, checkpoint=torch.is_grad_enabled())
    n = hn.shape[0] - 1
    total = 0.0
    for c in range(0, n, chunk):
        def ce(h, t):
            return torch.nn.functional.cross_entropy(tail.logits(h), t, reduction="sum")
        h, t = hn[c:min(n, c + chunk)], tok[c + 1:min(n, c + chunk) + 1]
        total = total + (torch.utils.checkpoint.checkpoint(ce, h, t, use_reentrant=False)
                         if torch.is_grad_enabled() else ce(h, t))
    return total / n, n


def cmd_train(a):
    import torch
    d = Dump(a.dump)
    t0 = time.time()
    tail, spec, targets = load_tail(a, a.targets, a.rank)
    load_s = time.time() - t0
    params = list(tail.lora.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    rng = random.Random(a.seed)
    order = []
    while len(order) < a.steps:
        ep = d.seqs[:]
        rng.shuffle(ep)
        order += ep
    tf32 = {"auto": tf32_ok(torch, a.device), "on": True, "off": False}[a.tf32]
    set_tf32(torch, tf32)
    os.makedirs(a.out, exist_ok=True)
    peak_gb(torch, a.device, reset=True)
    steps = []
    for i, s in enumerate(order[:a.steps]):
        t1 = time.time()
        loss, n = seq_loss(torch, tail, d, s, spec.first, a.chunk)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sync(torch, a.device)
        row = {"step": i, "window": s, "tokens": n + 1, "s": round(time.time() - t1, 3), "loss": float(loss.detach()),
               "peak_gb": round(peak_gb(torch, a.device), 2)}
        steps.append(row)
        print(json.dumps(row), flush=True)
    set_tf32(torch, False)
    state = {}
    for name in targets:
        k = name.replace(".", "_")
        state[name] = (tail.lora[k + "_A"].detach().cpu().numpy(), tail.lora[k + "_B"].detach().cpu().numpy())
    write_adapter(state, os.path.join(a.out, "adapter.gguf"), spec.arch)
    sec = [r["s"] for r in steps]
    warm = sec[1:] or sec
    out = {"arch": spec.arch, "first": spec.first, "targets": targets, "rank": a.rank, "lr": a.lr, "steps": len(steps),
           "match": a.match, "tf32": tf32, "load_s": round(load_s, 1),
           "step_s_mean_after_first": round(sum(warm) / len(warm), 3), "step_s_median": sorted(sec)[len(sec) // 2],
           "step_s_first": sec[0], "peak_gb": max(r["peak_gb"] for r in steps),
           "loss_first": steps[0]["loss"], "loss_last": steps[-1]["loss"],
           "card": device_info(torch, a.device) | {"notes": tail.notes, "f64_on_card": tail.f64},
           "adapter": "adapter.gguf", "per_step": steps}
    write_json(os.path.join(a.out, "train.json"), out)
    print(json.dumps({k: v for k, v in out.items() if k != "per_step"}))
    return 0


def cmd_eval(a):
    """Mean next-token cross-entropy of the torch tail (with any --lora adapters) over every position of the first
    --seqs windows, and at the dump's logit rows the same for the engine (whatever adapters the dump ran with) and the
    torch tail, plus their agreement."""
    import numpy as np
    import torch
    set_tf32(torch, False)
    d = Dump(a.dump)
    seqs = d.seqs[:a.seqs] if a.seqs else d.seqs
    tail, spec, _ = load_tail(a)
    tot = cnt = 0.0
    e_ce, t_ce, E, T = [], [], [], []
    with torch.no_grad():
        for s in seqs:
            loss, n = seq_loss(torch, tail, d, s, spec.first, a.chunk)
            tot += float(loss) * n
            cnt += n
            lg, pos = d.logits(s)
            tok = d.tokens(s)
            keep = pos < len(tok) - 1
            pos = pos[keep].astype(np.int64)
            e = log_softmax_np(lg[keep])
            t = tail(torch.from_numpy(d.resid(s, spec.first)), pos).double().cpu().numpy()
            nxt = tok[pos + 1]
            e_ce += (-e[np.arange(len(pos)), nxt]).tolist()
            t_ce += (-t[np.arange(len(pos)), nxt]).tolist()
            E.append(e)
            T.append(t)
    out = {"adapters": [os.path.basename(p) for p in a.lora or []], "first": spec.first, "windows": len(seqs),
           "match": a.match, "torch_ce_all_positions": tot / cnt, "positions": int(cnt),
           "logit_rows": len(e_ce), "engine_ce_logit_rows": float(np.mean(e_ce)),
           "torch_ce_logit_rows": float(np.mean(t_ce)),
           "engine_vs_torch": compare(np.concatenate(E), np.concatenate(T)),
           "card": device_info(torch, a.device)}
    write_json(a.out, out)
    print(json.dumps(out))
    return 0


# ------------------------------------------------------------------------------------------------ adapter GGUF
def write_adapter(state, out, arch):
    """{GGUF tensor name: (A [r, n_in], B [n_out, r])} -> a llama.cpp LoRA adapter GGUF (no alpha: scale 1 = B A)."""
    import gguf
    import numpy as np
    w = gguf.GGUFWriter(out + ".tmp", arch=arch)
    w.add_type("adapter")
    w.add_string("adapter.type", "lora")
    for name, (A, B) in sorted(state.items()):
        w.add_tensor(name + ".lora_a", np.ascontiguousarray(np.asarray(A, np.float32)))   # [r, n_in] -> ne (n_in, r)
        w.add_tensor(name + ".lora_b", np.ascontiguousarray(np.asarray(B, np.float32)))   # [n_out, r] -> ne (r, n_out)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    os.replace(out + ".tmp", out)


def read_adapter(path):
    """An adapter GGUF -> {name: (A, B)}; refuses one with an alpha (this tool applies scale 1 = B A)."""
    import gguf
    import numpy as np
    r = gguf.GGUFReader(path)
    if field(r, "adapter.lora.alpha") not in (None, 0, 0.0):
        raise SystemExit(f"{path}: has adapter.lora.alpha; only alpha-free adapters (scale = B A) are read")
    parts = {}
    for t in r.tensors:
        if t.name.endswith(".lora_a") or t.name.endswith(".lora_b"):
            arr = np.asarray(gguf.quants.dequantize(t.data, t.tensor_type), np.float32).reshape(
                [int(x) for x in reversed([int(n) for n in t.shape])])
            parts.setdefault(t.name[:-7], {})[t.name[-1]] = arr
    return {n: (p["a"], p["b"]) for n, p in parts.items()}


# ------------------------------------------------------------------------------------------------ CLI
def alloc_defaults():
    """expandable_segments on NVIDIA (fragmentation ran an 11 GB Pascal card out of memory without it), never on
    ROCm: on TheRock's gfx900 build every allocation fails with it. Set after importing torch (the allocator takes its
    configuration only through its own call once torch is imported: an environment variable set at this point is
    not read), and only when the user set no allocator variable."""
    import torch
    if torch.version.hip or not torch.cuda.is_available():
        return
    if "PYTORCH_CUDA_ALLOC_CONF" in os.environ or "PYTORCH_ALLOC_CONF" in os.environ:
        return
    try:
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    except (AttributeError, RuntimeError):
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info")
    p.add_argument("gguf")
    p.add_argument("--first", type=int)
    p = sub.add_parser("weights")
    p.add_argument("gguf")
    p.add_argument("--first", type=int, required=True)
    p.add_argument("--out", required=True)
    for name in ("gate", "train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--dump", required=True)
        p.add_argument("--weights", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--device", default="cuda", type=norm_device)
        p.add_argument("--match", choices=("q8", "none"), default="q8",
                       help="q8: the CPU engine's 8-bit activation rounding; none: plain f32")
        p.add_argument("--offload-experts", choices=("auto", "on", "off"), default="auto")
        p.add_argument("--lora", action="append", help="a frozen adapter GGUF applied at scale 1 (repeatable)")
        p.add_argument("--chunk", type=int, default=256, help="head rows per cross-entropy chunk")
        if name == "gate":
            p.add_argument("--seqs", type=int, default=8)
        if name == "eval":
            p.add_argument("--seqs", type=int, default=0, help="the first N windows (0: all)")
        if name == "train":
            p.add_argument("--steps", type=int, default=20)
            p.add_argument("--rank", type=int, default=8)
            p.add_argument("--lr", type=float, default=5e-4)
            p.add_argument("--targets", choices=("attn", "o", "attn+shexp"), default="attn")
            p.add_argument("--blocks", help="train only the targets in blocks A-B (inclusive)")
            p.add_argument("--tf32", choices=("auto", "on", "off"), default="auto")
            p.add_argument("--seed", type=int, default=20261115)
    a = ap.parse_args(argv)
    sys.path.insert(0, HERE)
    if a.cmd in ("gate", "train", "eval"):
        alloc_defaults()
    return {"info": cmd_info, "weights": cmd_weights, "gate": cmd_gate, "train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
