#!/usr/bin/env python3
"""Write an LLAMA_HOT_EXPERTS heat file that banks every routed expert of whole MoE layers on GPUs.

Usage: whole-layer-banks.py MODEL.gguf OUT.tsv DEVICE=GiB [DEVICE=GiB ...] [--keep-mtp]

Layers are taken from the last one down. Each layer goes to the first device, in the order given, whose GiB
budget still holds all of its expert bytes; a layer that fits nowhere stays on the CPU (no line). A model's
MTP (nextn) layers are planned (they use budget) but not written unless --keep-mtp: with the MTP draft on,
ik_llama.cpp keeps their experts on the device itself, and banking them as well aborts the load ("already
device-resident"). This is how the measured Ornith-1.5-35B-A3B file was made.

Needs the gguf-py package of the ik_llama.cpp tree (pip install ./gguf-py, or PYTHONPATH=<tree>/gguf-py).
"""
import collections
import re
import sys

import gguf


def field(reader, suffix):
    for name, f in reader.fields.items():
        if name.endswith(suffix):
            return int(f.parts[f.data[0]][0])
    return 0


def main(argv):
    args = [a for a in argv if a != "--keep-mtp"]
    if len(args) < 3:
        sys.exit(__doc__)
    model, out, budgets = args[0], args[1], [b.split("=") for b in args[2:]]
    reader = gguf.GGUFReader(model)
    size, experts = collections.Counter(), {}
    for t in reader.tensors:
        m = re.match(r"blk\.(\d+)\.ffn_(up|gate|down|gate_up)_exps", t.name)
        if m:
            size[int(m.group(1))] += int(t.n_bytes)
            experts[int(m.group(1))] = int(t.shape[2])
    blocks, nextn = field(reader, ".block_count"), field(reader, ".nextn_predict_layers")
    mtp = set(range(blocks - nextn, blocks))
    left = {dev: float(gib) * 2**30 for dev, gib in budgets}
    plan = {}
    for il in sorted(size, reverse=True):
        dev = next((d for d, _ in budgets if left[d] >= size[il]), None)
        if dev:
            left[dev] -= size[il]
            plan[il] = dev
    keep = "--keep-mtp" in argv
    with open(out, "w") as f:
        for il in sorted(i for i in plan if keep or i not in mtp):
            f.write(f"blk.{il}\t{','.join(map(str, range(experts[il])))}\t{plan[il]}\n")
    for dev, gib in budgets:
        held = [il for il in plan if plan[il] == dev and (keep or il not in mtp)]
        print(f"{dev}: {len(held)} layers {held[::-1]}, {sum(size[il] for il in held) / 2**30:.2f} of {gib} GiB")
    print(f"CPU: {sorted(set(size) - set(plan))}; MTP layers not written: {[] if keep else sorted(mtp & set(plan))}")


if __name__ == "__main__":
    main(sys.argv[1:])
