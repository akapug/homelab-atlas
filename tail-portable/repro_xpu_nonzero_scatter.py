#!/usr/bin/env python3
"""Repro: torch.nonzero, boolean-mask indexing and Tensor.scatter_(dim, index, <python scalar>) return wrong
results on an Intel Arc Pro B70 (Battlemage, PCI id 0xe223) with torch 2.13.0..2.15.0+xpu on compute-runtime 26.05.37020;
torch 2.10.0..2.12.1+xpu are correct there. On compute-runtime 26.40 the trainer's model check passes on 2.15.0;
this repro is not yet rerun there.

    python3 repro_xpu_nonzero_scatter.py            # exit 0 = every case matches the CPU, 1 = a wrong result

Needs only torch with an XPU device. Prints the torch version, the device, and every case that differs.
"""
import sys

import torch


def main():
    dev = "xpu"
    p = torch.xpu.get_device_properties(0)
    print(torch.__version__, p.name, hex(getattr(p, "device_id", 0)), p.driver_version)
    wrong = []
    m = torch.tensor([True, True, True, False] * 128)
    a, b = m.nonzero().squeeze(1), m.to(dev).nonzero().squeeze(1).cpu()
    if not torch.equal(a, b):
        wrong.append(f"nonzero of [T,T,T,F]*128: max index {int(b.max())} (want {int(a.max())})")
    g = torch.Generator().manual_seed(0)
    for n in (64, 100, 256, 1000, 4096, 1 << 20):
        for frac in (0.1, 0.5, 0.9):
            m = torch.rand(n, generator=g) < frac
            if not torch.equal(m.nonzero(), m.to(dev).nonzero().cpu()):
                wrong.append(f"nonzero n={n} p={frac}")
            if not torch.equal(torch.arange(n)[m], torch.arange(n, device=dev)[m.to(dev)].cpu()):
                wrong.append(f"mask index n={n} p={frac}")
    for R in (1, 2, 5, 16, 64):
        for C, k in ((40, 14), (100, 50), (1000, 512)):
            idx = torch.stack([torch.randperm(C, generator=torch.Generator().manual_seed(i))[:k] for i in range(R)])
            want = torch.zeros(R, C).scatter_(1, idx, 1.0)
            got = torch.zeros(R, C, device=dev).scatter_(1, idx.to(dev), 1.0).cpu()
            src = torch.zeros(R, C, device=dev).scatter_(1, idx.to(dev), torch.ones(R, k, device=dev)).cpu()
            if not torch.equal(want, got):
                wrong.append(f"scatter_ scalar R={R} C={C} k={k}: {int((want != got).sum())} wrong"
                             f" (tensor source {'ok' if torch.equal(want, src) else 'also wrong'})")
    for w in wrong:
        print("WRONG", w)
    print(f"{len(wrong)} wrong cases")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
