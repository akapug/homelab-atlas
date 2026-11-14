"""Legacy shape metadata; no measured routing means no per-expert counts.

The rank-law coefficients remain for interpreting historical calibration records.
They are not telemetry for another model, workload, or checkpoint.
"""
import math

SHAPE_POINTS = [(2000 / 12096, 0.391), (3000 / 12096, 0.501), (4000 / 12096, 0.597)]
SHAPE_A = sum(math.log(y) / math.log(x) for x, y in SHAPE_POINTS) / len(SHAPE_POINTS)
SHAPE_SRC = ('ESTIMATE: synthetic heat, cumulative share = x^%.3f over the expert fraction x, shaped like GLM-5.3-Flash LIVE coding '
             'routing on snoopy (lab notes 13:47Z rs-1340: top2000 0.391 / top3000 0.501 / top4000 0.597 of 12,096)' % SHAPE_A)


def synth_heat(L, E, frac_of=1.0, seed=17):
    if type(L) is not int or type(E) is not int or L <= 0 or E <= 0:
        raise ValueError("layers and experts must be positive integers")
    if not isinstance(frac_of, (int, float)) or isinstance(frac_of, bool) or not math.isfinite(frac_of) or not 0 < frac_of <= 1:
        raise ValueError("frac_of must be finite and in (0, 1]")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    # Compatibility entry point: absence is data, never pseudo-random counts.
    # Callers stamp status/trace_provenance and may use a uniform scenario prior.
    return None
