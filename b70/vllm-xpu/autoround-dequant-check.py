"""Does an AutoRound GPTQ-packed layer dequantize to the BF16 original? Reads safetensors
by hand (numpy only), dequantizes sym int4 as w = (q - 8) * scale per group along the input
dim, and compares with the BF16 weight. Usage: autoround-dequant-check.py <quant dir> <bf16 dir> <layer>..."""
import json
import os
import struct
import sys

import numpy as np

DT = {"I32": np.int32, "F16": np.float16, "F32": np.float32, "BF16": np.uint16, "I8": np.int8, "U8": np.uint8}


def tensor(d, name):
    f = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"][name]
    with open(os.path.join(d, f), "rb") as h:
        n = struct.unpack("<Q", h.read(8))[0]
        meta = json.loads(h.read(n))[name]
        a, b = meta["data_offsets"]
        h.seek(8 + n + a)
        x = np.frombuffer(h.read(b - a), DT[meta["dtype"]]).reshape(meta["shape"])
    if meta["dtype"] == "BF16":
        x = (x.astype(np.uint32) << 16).view(np.float32)
    return x.astype(np.float32) if x.dtype != np.int32 else x, meta["dtype"]


qd, bd = sys.argv[1], sys.argv[2]
for layer in sys.argv[3:]:
    qw, _ = tensor(qd, layer + ".qweight")          # [K / 8, N] int32, 8 nibbles along K
    sc, sdt = tensor(qd, layer + ".scales")          # [K / g, N]
    qz, _ = tensor(qd, layer + ".qzeros")            # [K / g, N / 8]
    ref, rdt = tensor(bd, layer + ".weight")         # [N, K]
    K, N = qw.shape[0] * 8, qw.shape[1]
    g = K // sc.shape[0]
    q = np.stack([(qw >> (4 * i)) & 15 for i in range(8)], 1).reshape(K, N)
    z = np.stack([(qz >> (4 * i)) & 15 for i in range(8)], 2).reshape(qz.shape[0], N)
    w = (q - 8).astype(np.float32) * np.repeat(sc, g, 0)
    w, r = w.astype(np.float64), ref.T.astype(np.float64)
    rel = np.linalg.norm(w - r) / np.linalg.norm(r)
    cos = float((w * r).sum() / np.linalg.norm(w) / np.linalg.norm(r))
    print(f"{layer}: K={K} N={N} group={g} scales {sdt} ref {rdt}; zeros seen {np.unique(z)[:6]}; "
          f"rel err {rel:.4f} cos {cos:.5f}; scale range {sc.min():.3g}..{sc.max():.3g}")
