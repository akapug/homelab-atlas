# Wave 3 planner and script audit

2026-11-14. Baseline: `e3119f7`. This records source fixes and dispositions of the condensed P1/P2 review;
it does not add model measurements. Author: GPT-6 Codex.

## Agent recommendations

Explicit `coding`, `agent`, `agentic`, `agentic-coding`, `tool-calling` and `tool-use` roles require
measured EARf >= 0.98. Missing EARf remains UNKNOWN and is excluded from these recommendations even if KL
passes; a caller cannot lower or disable the floor for these roles. The measured IQ3_XXS value 0.894 fails.
Mixed quant layouts use the lowest available catalog score; missing coverage remains UNKNOWN unless the
whole named layout or exact recipe/run was measured. A score is a planner filter, not proof that a
different engine/build preserves the model; runtime admission must bind its own measurement to the deployment.

Small Qwen models on XPU/SYCL receive an UNKNOWN degradation warning above 23,000 context tokens. Explicit
agent profiles with `startup_tokens` above 8,000 fail. Absence of that field does not certify an 8,000-token
startup: the runtime must count the full request. An agent count exceeding `serving.batch` fails unless
`admission.policy` is `door` and its positive integral conversation limit is at most the batch slots. This
captures the measured [B=33 failure](multibox-plan.md) without claiming every backend has the same boundary.

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

## Verification and limits

`python3 recipe-maker/build.py`, all 15 `node tests/test_*.js` files from `recipe-maker/`,
`python3 -m unittest discover -s recipe-maker/tests`, and `python3 tools/check_atlas.py --accuracy` passed.
Python ran 252 tests with 27 existing skips for optional/private fixtures and unavailable browser tooling.
The full check still reports missing private overlay/ledger inputs as unavailable, not measured passes.

No GPU server, live browser, container network or deployed patch build was exercised. Calibration fits existing
measurements and is not independent validation. Existing raw model logs and private quality panels remain
outside this public repository. Runtime deployment and exact-build EARf qualification belong to the serving repo.

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [fleet](https://github.com/akapug/fleet) for [@akapug](https://github.com/akapug)
