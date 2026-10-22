#!/usr/bin/env python3
"""Patch ik_llama.cpp's examples/imatrix/imatrix.cpp so that IK_IMATRIX_KEEP_PARTIAL=1 stores every MoE expert tensor,
even when some experts saw no tokens. Stock ik skips a whole expert tensor when 5 % or more of its experts are unseen
(one of our short passes on V4.1 lost the expert entries of layers 34-39 that way, 12 % unseen). Unseen experts are
written as 0; build-store.sh's split (split_experts.py) then gives them the mean of the tensor's other experts.
Without the variable the output is unchanged: the writer change only touches values whose count is 0, which stock ik
never stores. Tested on the fork's tier-split branch at 6d9992e2.
  usage: patch-imatrix-keep-partial.py examples/imatrix/imatrix.cpp, rebuild, run llama-imatrix with
  IK_IMATRIX_KEEP_PARTIAL=1"""
import sys

p = sys.argv[1]
s = open(p).read()
if 'IK_IMATRIX_KEEP_PARTIAL' in s:
    print('already patched'); sys.exit(0)
a1 = "            bool store_it = false;\n            if (kv.second.n_as > 1) {\n"
b1 = ("            bool store_it = false;\n"
      "            // IK_IMATRIX_KEEP_PARTIAL=1 keeps every MoE entry; unseen experts are written as 0\n"
      "            if (kv.second.n_as > 1 && getenv(\"IK_IMATRIX_KEEP_PARTIAL\")) {\n"
      "                fprintf(stderr, \" - stored with zero rows for unseen experts (IK_IMATRIX_KEEP_PARTIAL)\\n\");\n"
      "                store_it = true;\n"
      "            } else if (kv.second.n_as > 1) {\n")
a2 = "                tmp[i] = (stat.values[i] / static_cast<float>(stat.counts[i])) * static_cast<float>(stat.ncall);\n"
b2 = ("                tmp[i] = stat.counts[i] ? (stat.values[i] / static_cast<float>(stat.counts[i])) * static_cast<float>(stat.ncall)\n"
      "                                        : 0.0f;\n")
a3 = "#include <cmath>\n"
for a in (a1, a2, a3):
    if s.count(a) != 1:
        raise SystemExit(f'anchor not found exactly once: {a[:60]!r}')
s = s.replace(a1, b1).replace(a2, b2).replace(a3, "#include <cmath>\n#include <cstdlib>\n")
open(p, 'w').write(s)
print('patched')
