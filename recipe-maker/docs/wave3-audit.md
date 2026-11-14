# Wave 3 planner and script audit

2026-11-14. Baseline: `e3119f7`. This records source fixes and dispositions of the condensed P1/P2 review;
it does not add model measurements. Author: GPT-6 Codex.

## Agent recommendations

Pass-2 correction (Wave 3.5): local seats are **bounded authors under an executed Judge** (`role:
"author_under_judge"`). The executed authoring results (27B **22/24**, Flash-Next **11/12**) qualify that
role, not autonomous diff review. `diff_review` is advisory/excluded unless `review_screen` explicitly
records `executed`, `planted`, and `clean` as true, false-alarm rate <= 0.10 and recall >= 0.50.

Admission prefers measured `battery_pass_rate` or an `authoring_probe` with `executed: true` and
`passed`/`total` (or `pass_rate`). The default outcome bar is 0.90; `battery_min_pass_rate` can declare a
panel-specific bar. Otherwise EARf is compared to recipe/model/family `perturbation_floor - 0.015`.
The QFN arg_values panel measured 0.955–0.968 for >=4-bit quants: the old absolute 0.98 bar rejected
all of them. Without a perturbation floor, only EARf < 0.90 is flagged UNKNOWN/degraded; higher measured
values pass the severe-degradation screen without claiming outcome qualification. Missing and incomplete
mixed-layout measurements remain UNKNOWN. No quant-width or KL shortcut turns missing EARf into a pass.
The advisor retains UNKNOWN candidates with a badge and prefers outcome-probed seats over faster unknowns;
actual failed quality gates remain exclusions. Runtime measurement custody must still match the served build.

Only **Qwen3.8-27B int4** has the cited depth-recall evidence: **9/9 at <=12k**, **3/9 at ~23k**.
Above **12288** tokens warn (12–23k is unmeasured); above **23000** flag degraded. Other models return
UNKNOWN unless they declare `measured_recall_budget_tokens`; there is no generic Qwen or Flash-Next cliff.
Startup ceilings must be explicitly configured as `serving.max_startup_tokens`. An agent count exceeding
`serving.batch` still requires a conversation door capped at the available slots.

## Finding dispositions

| Finding | Disposition |
|---|---|
| Capacity-incompatible roofline attribution; compute-ceiling false positives | Already fixed at baseline; retained regression coverage. |
| Null selection in `choose`; nonfinite network identity; calibration mutation; date comparisons | Already fixed at baseline. |
| Order-dependent card-curve fitting | Joint bounded coordinate fit over calibration points, deterministic order, finite inputs, residual/convergence reporting. Reversed anchor order produces identical coefficients. |
| Repeated expensive calibration | Content-keyed cache includes relevant model, hardware, anchor, engine and knob inputs; cached results are copied before return. |
| Quadratic advisor KL lookup | Indexed at baseline; explicit role fidelity filtering added here. |
| Synthetic heat assumptions | `heat_shape.py` now rejects invalid dimensions/seed/scale and documents that common `frac_of` scaling cancels under normalization. Output remains explicitly synthetic; no measurement was substituted. |
| Schema-blind priors | Explicit schema at baseline; negative, nonfinite and boolean values now rejected. |
| Cross-arm run attribution and masked benchmark exits | Fixed at baseline. |
| Screen ledger atomic write and concurrent lost updates | Baseline atomic replacement retained; directory lock now covers the whole read/merge/write transaction. Eight concurrent writers preserve all rows. |
| Stale `check_atlas.py --accuracy` input | Baseline forces rebuild; full final check passed. |
| UI storage/deep-link edge cases | Wrong-shaped storage falls back; unsupported engine and nonpositive/nonfinite deep-link speed refused in `app.js` (there is no `ui.js`). |
| Patch guards disappear with `python -O` | Five standalone XPU patchers use explicit source-drift errors; an optimized-Python regression confirms the target stays unchanged. |
| Spec-step readiness, metrics and division | Readiness exhaustion and missing/invalid counters fail before measurement; pipe/curl failures propagate. |
| Sandbox host networking and key exposure | Host network refused; key mounted read-only and loaded inside the container. Bounded runtime, dropped capabilities and PID limit. A container-reachable proxy is required. |

## Wave 3.5 changed-file summary

- `src/model.js`, `src/advisor.js`: outcome-first admission, model-relative EARf margin, UNKNOWN ranking,
  bounded author/review capabilities, and model-specific depth-recall warnings.
- `tests/test_wave3_model.js`, `tests/test_advisor.js`: empirical boundary and recommendation regression assertions.
- This audit, root `CHANGELOG.md`, and `docs/audits/2026-11-14-codebase-rake/README.md`: corrected policy,
  change summary and the one-line agent orientation link.

## Verification and limits

Wave 3.5: source inspection and syntax checks only; regression assertions updated for coordinator convergence.
The suite results below belong to the original Wave 3 run, not this correction pass.


`python3 recipe-maker/build.py`, all 15 `node tests/test_*.js` files from `recipe-maker/`,
`python3 -m unittest discover -s recipe-maker/tests`, and `python3 tools/check_atlas.py --accuracy` passed.
Python ran 252 tests with 27 existing skips for optional/private fixtures and unavailable browser tooling.
The full check still reports missing private overlay/ledger inputs as unavailable, not measured passes.

No GPU server, live browser, container network or deployed patch build was exercised. Calibration fits existing
measurements and is not independent validation. Existing raw model logs and private quality panels remain
outside this public repository. Runtime deployment and exact-build EARf qualification belong to the serving repo.

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [fleet](https://github.com/akapug/fleet) for [@akapug](https://github.com/akapug)
