#!/usr/bin/env python3
"""Probe every torch op the QFN-Coder tail trainer uses on one device, against the CPU, before a long run.

    python3 probe_ops.py --device cuda|xpu|cpu [--json out.json]

Each op runs forward (and backward where training needs it) on small inputs on the device and on the CPU in f32;
the line says ok with the max abs difference, or the error. Exit 0 when every op ran, 1 otherwise. Covers: f32
matmul and batched matmul (BLAS), f16 -> f32 weight upcast, depthwise conv1d (the gated delta net's causal conv),
torch.linalg.solve_triangular (the chunked delta rule), topk / argsort / bincount / index_add_ (MoE routing),
scatter_ (tensor and scalar source), nonzero and boolean-mask indexing, masked softmax, rope in f64, pinned host memory and a non-blocking host-to-device copy, checkpointing
and the autograd save-on-cpu hook, and peak-memory counters.
"""
import argparse
import json
import sys
import traceback

import torch
import torch.nn.functional as F


def run(device):
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g)
    out = {}

    def case(name, fn, exact=False):
        try:
            g.manual_seed(1)
            a = fn("cpu")
            g.manual_seed(1)              # the same inputs on both sides
            b = fn(device)
            diff = max(float((x.detach().cpu().float() - y.detach().float()).abs().max()) for x, y in zip(_t(b), _t(a)))
            out[name] = {"ok": diff == 0 or not exact, "max_abs": diff}
            if exact and diff:
                out[name]["error"] = "WRONG: an index/bool result differs from the CPU"
        except Exception as e:   # noqa: BLE001 - a probe reports every failure
            out[name] = {"ok": False, "error": f"{type(e).__name__}: {str(e).splitlines()[0][:300]}",
                         "where": traceback.format_exc().splitlines()[-3][:200]}
        print(name, json.dumps(out[name]), flush=True)

    A, B = r(257, 513), r(513, 129)
    case("matmul_f32", lambda d: A.to(d) @ B.to(d))
    H = r(8, 64, 128)
    case("bmm_f32", lambda d: H.to(d) @ H.to(d).transpose(-1, -2))
    W16 = r(129, 513).half()
    case("f16_weight_upcast_mm", lambda d: A.to(d) @ W16.to(d, non_blocking=True).float().t())
    X, K = r(1, 96, 70), r(96, 1, 4)
    case("conv1d_depthwise", lambda d: F.conv1d(F.pad(X.to(d), (3, 0)), K.to(d), groups=96))

    def tri(d):
        L = torch.tril(r(4, 3, 64, 64) * 0.05, -1).to(d)
        eye = torch.eye(64, device=d)
        L.requires_grad_()
        T = torch.linalg.solve_triangular(L + eye, eye.expand_as(L), upper=False)
        T.sum().backward()
        return T, L.grad
    case("solve_triangular_fwd_bwd", tri)
    logits = r(300, 256)

    def route(d):
        p = torch.softmax(logits.to(d), -1)
        wt, ex = p.topk(10, dim=-1)
        flat = ex.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=256)
        out = torch.zeros(300, 16, device=d)
        out.index_add_(0, order // 10, r(3000, 16).to(d))
        return wt, counts.float(), out
    case("topk_argsort_bincount_index_add", route)

    def scat(d):
        v = torch.zeros(5, 40, dtype=torch.bool, device=d)
        v.scatter_(1, torch.arange(0, 40, 3, device=d)[None].expand(5, -1).contiguous(), True)
        return (v.float(),)
    case("scatter_bool", scat, exact=True)

    def msoft(d):
        s = r(4, 33, 77).to(d)
        keep = torch.rand(33, 77, generator=g).to(d) > 0.3
        keep[:, 0] = True
        return (torch.softmax(s.masked_fill(~keep[None], float("-inf")), -1),)
    case("masked_softmax", msoft)
    # exactness probes at the trainer's sizes: index results must be equal, not close
    def nz(d):
        m = torch.rand(4099, generator=g) < 0.5
        return (m.to(d).nonzero().float(), torch.arange(4099, device=d)[m.to(d)].float())
    case("nonzero_and_mask_index", nz, exact=True)

    def scal(d):
        idx = torch.stack([torch.randperm(100, generator=g)[:50] for _ in range(8)]).to(d)
        return (torch.zeros(8, 100, device=d).scatter_(1, idx, 1.0),)
    case("scatter_scalar_value", scal, exact=True)

    def big_route(d):
        flat = torch.randint(0, 256, (14088 * 10,), generator=g).to(d)
        order = torch.argsort(flat, stable=True)
        sc = r(512, 3522).to(d)
        top = sc.topk(512, dim=1).indices.sort(1).values      # a set: tie order may differ, the set may not
        return (order.float(), torch.bincount(flat, minlength=256).float(), top.float())
    case("argsort_bincount_topk_at_tail_sizes", big_route, exact=True)
    case("rope_f64", lambda d: (torch.arange(64, dtype=torch.float64, device=d)[:, None] * 1e-3).cos().float())
    case("half_round_trip", lambda d: A.to(d).half().float())
    case("bf16_round_trip", lambda d: A.to(d).to(torch.bfloat16).float())

    def pinned(d):
        t = r(1000, 1000)
        try:
            t = t.pin_memory()
            pinned_ok = True
        except Exception:   # noqa: BLE001
            pinned_ok = False
        y = t.to(d, non_blocking=True) * 2
        if d != "cpu":
            out["pin_memory_available"] = pinned_ok
        return (y,)
    case("pinned_nonblocking_copy", pinned)

    def ck(d):
        x = r(64, 64).to(d).requires_grad_()
        with torch.autograd.graph.save_on_cpu(pin_memory=d != "cpu"):
            y = torch.utils.checkpoint.checkpoint(lambda z: torch.tanh(z @ z), x, use_reentrant=False)
        y.sum().backward()
        return y, x.grad
    case("checkpoint_save_on_cpu_bwd", ck)
    m = {"cuda": torch.cuda, "xpu": getattr(torch, "xpu", None)}.get(torch.device(device).type)
    if m is not None:
        try:
            m.reset_peak_memory_stats()
            torch.ones(1 << 20, device=device)
            out["peak_counter_mib"] = round(m.max_memory_allocated() / 2**20, 1)
        except Exception as e:   # noqa: BLE001
            out["peak_counter_mib"] = f"{type(e).__name__}: {e}"
    return out


def _t(x):
    return x if isinstance(x, tuple) else (x,)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = {"torch": torch.__version__, "hip": torch.version.hip, "cuda": torch.version.cuda, "ops": run(a.device)}
    bad = [k for k, v in res["ops"].items() if isinstance(v, dict) and not v.get("ok")]
    res["failed"] = bad
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    print(json.dumps({"failed": bad, "torch": res["torch"]}))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
