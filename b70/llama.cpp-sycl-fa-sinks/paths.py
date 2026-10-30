#!/usr/bin/env python3
"""paths.py STOCK_LOG PATCHED_LOG: per test-backend-ops case, the kernel path ([FA-DISP] trace, GGML_SYCL_MKL_FA_DEBUG=1)
and the result in each log; prints totals and every case whose path or result differs between the two logs."""
import re, sys
from collections import Counter


def parse(path):
    cases, kernel = {}, None
    for line in open(path, errors="replace"):
        m = re.match(r"\[FA-DISP\] #\d+ (\w+) ", line)
        if m:
            kernel = m.group(1)
            continue
        m = re.match(r"\s+(FLASH_ATTN_EXT\(.*\)): (OK|FAIL|not supported.*)$", line.rstrip())
        if m:
            res = "unsupported" if m.group(2).startswith("not supported") else m.group(2)
            cases[m.group(1)] = (kernel if res != "unsupported" else "-", res)
            kernel = None
    return cases


a, b = parse(sys.argv[1]), parse(sys.argv[2])
for name, c in (("stock", a), ("patched", b)):
    print(f"{name}: {len(c)} cases; results {dict(Counter(r for _, r in c.values()))}; "
          f"paths {dict(Counter(k for k, r in c.values() if r != 'unsupported'))}")
diff = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
print(f"cases that differ: {len(diff)}")
for k in diff:
    short = re.sub(r",max_bias=.*?,prec=f32", "", k).replace("FLASH_ATTN_EXT", "")
    print(f"  {a.get(k)} -> {b.get(k)}  {short}")
