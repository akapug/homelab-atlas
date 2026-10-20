#!/usr/bin/env python3
# Adds a per-expert routed-token count dump to ik_llama.cpp's imatrix.
import sys
p = sys.argv[1]
s = open(p).read()
anchor = "    std::ofstream out(fname, std::ios::binary);\n"
add = '''    {
        // per-expert routed-token counts, one row per MoE tensor
        std::ofstream ec(fname + ".expert-counts.tsv");
        for (const auto & kv : m_stats) {
            const auto & st = kv.second;
            if (st.n_as < 2 || st.counts.empty()) continue;
            const size_t ne0 = st.counts.size() / st.n_as;
            ec << kv.first;
            for (int e = 0; e < st.n_as; ++e) ec << "\\t" << st.counts[e*ne0];
            ec << "\\n";
        }
    }
'''
assert s.count(anchor) == 1 and "expert-counts" not in s
open(p, "w").write(s.replace(anchor, add + anchor))
print("patched")
