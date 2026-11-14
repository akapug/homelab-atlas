# `homelab-atlas` Hierarchical Synthesis & Ground-Truth Audit

- **Audit Scope**: All **10 `homelab-atlas` shards** (`s177`–`s186`, `100%` `status == "ok"`) covering **62 tracked files**, **18,322 LoC**, **62 module verdicts**, and **105 findings** (`1 CRITICAL`, `22 HIGH`, `64 MEDIUM`, `12 LOW`, `6 COSPLAY`).
- **Baseline**: `main` @ `0deaf22`, cross-verified against Wave-1 (`7b7b7ff` on `ember/atlas-recipe-quality-gates`) and Wave-2 (`e3119f7` on `ember/wave2-fixes`).

---

## 1. Wave-1 & Wave-2 Resolution Ledger

| Wave / Branch | Scope Landed | Status |
| :--- | :--- | :--- |
| **Wave-1 (`7b7b7ff`)** | Tightened `klEquivKey` arithmetic provenance borrowing & enforced `EARf >= 0.98` quality gate across `advisor.js`, `model.js`, and `report.js`. | **LANDED (PR #1)** |
| **Wave-2 (`5033862`, `e3119f7`)** | Fixed `rlAttribute` `cap.fits` capacity separation (`CAPACITY_EXCEEDED` vs `UNMODELLED BOTTLENECK`), guarded `choose()` `bsel === null` deref on `Infinity`/`NaN` shadow prices, rejected `NaN` identity collision in `networkCheck`, memoized `klEquivKey` in `advisor.js`, removed `record_run.py` `or js[-1:]` cross-arm poisoning, added `pipefail` + safe JSON formatting to `bench_gpu.sh`, made `record_screen.py` atomic, and forced fresh `build.py` in `check_atlas.py --accuracy`. | **LANDED (Wave-2 PR)** |

---

## 2. Cluster Telemetry & Bloat Summary

| Cluster | Shards | Files | LoC | Findings (`CRIT / COSP / HIGH / MED / LOW`) | Verdicts (`RIP / SALV / REPL / FIX`) | Direct `loc_removable` | Assessment |
| :--- | ---: | ---: | ---: | :--- | :--- | ---: | :--- |
| **`recipe-maker` & `tools` & GPU notes** (`src/*`, `probe/*`, `tools/*`, `build.py`, `b70/*`, `rtx-6000/*`) | 10 | 62 | 18,322 | **105** (`1 / 6 / 22 / 64 / 12`) | `3 / 5 / 3 / 51` | **539 LoC** (2.9%) | **High signal-to-noise.** Core analytical physics (`model.js`, `heat_shape.py`) is genuine; removable bloat is concentrated in schema-guessing adapters (`priors_adapter.py`, `vitals_adapter.py`, `window_inputs.py`) and runtime vLLM string-surgery scripts (`b70/vllm-xpu/*.py`). |

---

## 3. Verified P0/P1 Planner Math, Recall & Tooling Findings

### 2.1 `homelab-atlas` PR #2 Candidates (Planner Math, Recall False-Positives & Ledger Integrity)

1. **[P0 / CRITICAL — Silent Recall False-Positives] `rlAttribute`, `rooflineFit`, and `agentPointRescore` misclassify capacity-incompatible runs and collapse distinct failure modes** — `recipe-maker/src/model.js:L5744-L5855` + `model.js` (`agentPointRescore`) (`recipe-maker/src/model.js:L4300-L4410`)
   - In `rlAttribute` (`recipe-maker/src/model.js:L5795-L5812`), `res.filter(x => x.err < -0.20)` fires **`UNMODELLED BOTTLENECK` on every point more than 20% below ceiling without checking `x.fits`**, labeling expected OOM / KV-spill / host-paging runs as mysterious kernel bottlenecks (`rooflineAttribute` at L5833 even checks `const cap = rlCapacity(...); fits: !cap || cap.fits`, then drops `fits` on the floor before calling `rlAttribute(data, rows)` at L5844!).
   - Simultaneously, `rooflineFit` (`recipe-maker/src/model.js:L5748-L5751`) restricts fitting to `p.fits && p.bound_tps > 0`, fitting a single scalar `eta = exp(mean(log(meas/bound)))` (geometric mean with zero outlier resistance) while excluding non-fitting runs that are then fed unfiltered into `rlAttribute`.
   - When `mTp >= 0.90 && eMed <= -0.15`, `rlAttribute` emits `COMPUTE CEILING MISSPECIFIED` solely from aggregate ratios without verifying compute-bound vs bandwidth-bound share (`shares.compute` vs `shares.vram`/`stream`).
2. **[P1 / HIGH — Crash in Layer-Scoped Quant Solver] `choose()` dereferences `null` (`bsel[i]`) when tier shadow prices push candidate cost to `Infinity`/`NaN`** — `recipe-maker/src/model.js:L436-L452`
   - In `!ctx.perExpert` mode, `let bc = Infinity, bsel = null`. If any group has empty valid options or `base[row + o] + n * price[o]` hits `Infinity`/`NaN` (`c < bc` is `false` for all `q`), `bsel` remains `null` and `for (...) add(gis[i], bsel[i])` at L451 throws `TypeError: Cannot read properties of null`.
3. **[P1 / HIGH — Identity Gate Bypass] `networkCanonical` + `JSON.stringify` equates `NaN`/`Infinity` with `null` in `networkCheck`** — `recipe-maker/src/model.js:L2871-L2889`
   - `JSON.stringify` serializes `NaN`, `Infinity`, and `-Infinity` to `"null"`. In `networkCheck` (L2887), `JSON.stringify(networkCanonical(r.network_prediction_identity)) === JSON.stringify(networkCanonical(candidate.prediction_identity))` returns `true` (`PASS`) when an uncomputed `NaN` field in `r` collides with a `null` field in `candidate`.
4. **[P1 / HIGH — Order-Dependent Greedy Calibration & Shared-State Mutation] `cardCurveFit` & `calibrate` mutate `data` in place and clamp residuals sequentially** — `recipe-maker/src/model.js:L3701-L3777`, `recipe-maker/src/model.js:L5337-L5355`
   - `cardCurveFit` solves constants one-by-one in `CCs.anchors` list order via 1D bisection (`consts[p.constant] = x`), clamping `AT_FLOOR`/`AT_CEILING` values into `consts` so every downstream anchor absorbs upstream model error without joint refit.
   - `calibrate(data)` mutates `data.boxes[bk].pmem.bw` (L3704, L3719, L3733) and `data.types[q].kappa` (L3776) directly on the caller's `data` graph; `unlockState` (`recipe-maker/src/model.js:L4703`) compares dates lexicographically (`String(q.at) < String(u.since)`), breaking on timezone offsets (`Z` vs `+00:00`).
5. **[P1 / HIGH — Silent Cross-Arm Ledger Poisoning & Masked Exit Codes in `recipe-maker/tools`]**
   - `recipe-maker/tools/record_run.py:L235-L241`: `js = [j for j in js if j.get('arm') == r.group(1)] or js[-1:]` — if an arm crashes before printing its JSON summary while `pre` holds an earlier `{...}` line, `or js[-1:]` silently attaches the wrong JSON record to the arm! Lines 653–683 rewrite `engine_flags`/`label`/`src` via fragile regexes on free-form strings.
   - `recipe-maker/tools/bench_gpu.sh:L8-L19`: `set -u` without `pipefail` runs `out=$("$@" ... | tail -n 1); r=$?` — `r` captures `tail`'s exit status (`0`), masking `llama-bench` segfaults/OOMs whenever at least one line was printed before death, and L18 formats `$arm`/`$model` into JSON via raw `printf '%s'`.
   - `recipe-maker/tools/record_screen.py:L218-L220`: `open(path, 'w')` truncates `screen_ledger.jsonl` in place without tempfile + `os.replace`.
   - `recipe-maker/src/priors_adapter.py:L24-L65`: Schema-blind recursive walk treats any dict key matching `_KLD_KEYS` (including generic `'err'`) under a quant name as KLD, defaulting unknown paths to **both** `['v41', 'glm']` (L30).
   - `tools/check_atlas.py:L97-L98`: `--accuracy` skips `build.py` whenever `recipe-maker/data.json` already exists, certifying README tables against stale build artifacts.

---

---

## 4. Planner Performance Findings

1. **`recipe-maker/src/advisor.js` (`runQuality` / `bestPassing`)**: Executes `(runs || []).find(x => ... && klEquivKey(x) === key)` (`JSON.stringify` per candidate) inside `runs.filter(...)`, turning `bestPassing(data, runs)` into an `O(N^2)` serialization loop invoked twice by `all()` (`trimProposals` + `upgradeAdvice`). *(Memoized in Wave-2 `5033862`.)*
2. **`recipe-maker/src/model.js` (`calibrate`)**: Evaluates a brute-force 311-point `fGrid` x `pGrid` x 36-iteration golden search on every build/report invocation and mutates `data.boxes[bk].pmem.bw` and `data.types[q].kappa` in place on the caller's graph. *(Remaining backlog candidate: input-digest memoization and pure return of calibrated parameters.)*

---

## 5. Prune & Boundary-Replacement Backlog (`RIP_OUT` / `SALVAGE_THEN_RIP`)

| Target Path(s) | LoC | Verdict | Rationale & Recommended Replacement |
| :--- | ---: | :--- | :--- |
| `recipe-maker/src/priors_adapter.py`, `recipe-maker/probe/vitals_adapter.py`, `recipe-maker/probe/window_inputs.py` | 356 | `RIP_OUT` | Schema-guessing recursive `walk()` adapters (`priors_adapter.py` guesses KLD from generic `'err'` keys across arbitrary JSON shapes). Replace with explicit versioned JSON schema readers (<40 LoC). |
| `b70/vllm-xpu/*.py` (`patch_vllm_qwen35_xpu.py`, `mtp-EagleProposer-patch.py`, `q4_0-moe-xpu.py`) | ~520 | `SALVAGE_THEN_RIP` | Valuable `_xpu_C::fp8_gemm_w8a16`, `Q4_0` MoE kernel bindings, and Qwen3.5 Eagle proposer fixes applied via fragile `str.replace` against installed `site-packages/vllm`. Consolidate into a clean `git apply --check` patch series. |
