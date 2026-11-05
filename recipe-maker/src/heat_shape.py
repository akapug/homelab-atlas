"""Synthetic routing heat for a MoE model with no routing counts: the cumulative share of routed reads C(x) = x^a over the
expert fraction x, a fitted to GLM-5.3-Flash LIVE coding routing on snoopy (rs-1340: top2000 0.391 / top3000 0.501 /
top4000 0.597 of 12,096 experts). Expert ids are permuted per layer (deterministic LCG) so layers differ.

One owner for the shape: build.py (Flash-Next, Ornith families) and tools/harvest.py (the generic MoE path) both import it.
"""
import math

SHAPE_POINTS = [(2000 / 12096, 0.391), (3000 / 12096, 0.501), (4000 / 12096, 0.597)]
SHAPE_A = sum(math.log(y) / math.log(x) for x, y in SHAPE_POINTS) / len(SHAPE_POINTS)
SHAPE_SRC = ('ESTIMATE: synthetic heat, cumulative share = x^%.3f over the expert fraction x, shaped like GLM-5.3-Flash LIVE coding '
             'routing on snoopy (lab notes 13:47Z rs-1340: top2000 0.391 / top3000 0.501 / top4000 0.597 of 12,096)' % SHAPE_A)


def synth_heat(L, E, frac_of=1.0, seed=17):
    out, st = [], seed
    for l in range(L):
        perm = list(range(E))
        for i in range(E - 1, 0, -1):  # Fisher-Yates with an LCG (deterministic)
            st = (st * 1103515245 + 12345) % 2 ** 31
            j = st % (i + 1)
            perm[i], perm[j] = perm[j], perm[i]
        C = lambda x: (x * frac_of) ** SHAPE_A / frac_of ** SHAPE_A
        row = [0.0] * E
        for r in range(E):
            row[perm[r]] = round(1e6 * (C((r + 1) / E) - C(r / E)), 3)
        out.append(row)
    return out
