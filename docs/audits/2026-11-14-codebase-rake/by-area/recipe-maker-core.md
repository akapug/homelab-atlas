# Recipe Maker Analytical Roofline & Planner (`recipe-maker/src/`, `report.js`)

Part of the [2026-11-14 Full-Codebase Audit Rake](../README.md) for `homelab-atlas`.

- **Shards in chapter**: `6`
- **File slices covered**: `18` (`12,487` lines)
- **Findings**: `63` total (`1` CRITICAL, `14` HIGH, `38` MEDIUM, `7` LOW; `10` COSPLAY-tagged)
- **Module-verdict removable LoC estimate**: `1,207`

## Chapter Shard Index

| Shard ID | Scope / Domain | Files | Lines | Findings (`C/H/M/L`) | Removable LoC |
| :--- | :--- | ---: | ---: | :--- | ---: |
| [`s178_homelab-atlas__recipe-maker`](#s178_homelab-atlas__recipe-maker) | `recipe-maker` | 1 | 2,100 | `0/3/6/1` (10) | 380 |
| [`s179_homelab-atlas__recipe-maker`](#s179_homelab-atlas__recipe-maker) | `recipe-maker` | 4 | 1,744 | `0/3/5/3` (11) | 55 |
| [`s181_homelab-atlas__recipe-maker__src`](#s181_homelab-atlas__recipe-maker__src) | `recipe-maker/src` | 7 | 1,943 | `0/1/8/1` (11) | 7 |
| [`s182_homelab-atlas__recipe-maker__src`](#s182_homelab-atlas__recipe-maker__src) | `recipe-maker/src` | 1 | 2,150 | `0/1/8/2` (11) | 300 |
| [`s183_homelab-atlas__recipe-maker__src`](#s183_homelab-atlas__recipe-maker__src) | `recipe-maker/src` | 1 | 2,150 | `0/2/7/0` (10) | 0 |
| [`s184_homelab-atlas__recipe-maker__src`](#s184_homelab-atlas__recipe-maker__src) | `recipe-maker/src` | 4 | 2,400 | `1/4/4/0` (10) | 465 |

---

## s178_homelab-atlas__recipe-maker

- **Scope**: `recipe-maker` | **Model**: `glm-5.3` | **Status**: `ok` | **Files**: `1` | **Lines**: `2,100`
- **Covered slices**: `recipe-maker/build.py` (`L1-L2100` of `2668`)

### Summary

A 2,668-line monolithic build script that hand-folds ~15 untyped dict 'tables' (types, models, boxes, engines, checks, bench, batch, window) into data.json for a viz page, executing nearly everything at module scope. The measurement-provenance discipline (tag/src/evidence on every value) is genuinely good, but it is undermined by exec()-based plugin loading, regex-against-prose 'verification' of engine behavior and sanity, post-hoc retrojection of maintained intent onto recorded runs, and silent subsystem vanishing when ledger rows are absent.

### Module & Bloat Verdicts

| Module / File | Verdict | Removable LoC | Rationale & Salvage Notes |
| :--- | :--- | ---: | :--- |
| `recipe-maker/build.py` | `FIX_IN_PLACE` | 380 | The core evidence-table pipeline (n(), evidence_of, hdr_summary, heat loading, bench least-squares fits) earns its keep and is structurally load-bearing for the whole page. But the plugin exec(), the regex sanity/flag theater, and the maintained-intent retrojection are negative-value subsystems that must be ripped or replaced with structural boundaries, and the module-scope side-effect sprawl makes the file untestable as constructed. |

### Findings (10)

1. **[HIGH | `FLAW_SECURITY_OVERSIGHT` | `REPLACE_BOUNDARY` | `-6 LoC`]** `recipe-maker/build.py:838-842` — **exec() of families/*.py into build globals()**
   - **Mechanism**: Every .py file in families/ is compiled and exec'd in the module globals with zero isolation: any file there gets arbitrary code execution, full write access to models/engines/checks/FAMILY_QUALITY, and can shadow any name the build later uses. A typo'd family file corrupts unrelated tables silently because it mutates shared global dicts.
   - **Fix / Action**: Replace with a typed entry point: each family file defines build_family(ctx) returning a plain dataclass/dict; build.py calls it explicitly and merges under validation (same overlay.py rules as data/*.json). No exec, no globals injection.
2. **[HIGH | `COSPLAY_THEATER` | `RIP_OUT` | `-4 LoC`]** `recipe-maker/build.py:2091-2094` — **Engine sanity 'verified' by regexing the number 391 in a prose note**
   - **Mechanism**: window_attempts decides an attempt passed engine sanity via re.search(r'sanity[^;]*391', layout_note): a hardcoded magic token count scraped from free-text notes. Anything else in the note (or a different count) reads as False/None; the note author controls the 'verification'. This is a rubber stamp dressed as a READ-tagged measurement.
   - **Fix / Action**: Delete the sanity field entirely, or have record_* tools write an explicit structured boolean (sanity_pass: true, count) at capture time; never re-derive ground truth from prose.
3. **[HIGH | `ARCH_DEAD_END` | `FIX_IN_PLACE`]** `recipe-maker/build.py:192-208, 739-742, 1018-1029, 1408-1412, 2044-2046` — **Module-scope side-effect sprawl: sys.path mutation, importlib exec, file reads at import**
   - **Mechanism**: The module imports heatmap via sys.path.insert, executes heat_shape.py/overlay.py/window_inputs.py via spec/loader, reads a dozen JSON files, mutates HEAT_USED/models/checks, and defines CLI behavior (--calibration-data-sha256) all at top level in interleaved order. Nothing is importable for testing, import order is load-bearing (heat_sets before extra_heat_sets), and one missing file mid-file leaves half-built globals.
   - **Fix / Action**: Wrap in a main() that constructs tables in explicit stages via plain imports of the sibling modules (they are local packages; make tools/ and probe/ importable properly), so a failure is atomic and each stage is unit-testable.
4. **[MEDIUM | `COSPLAY_THEATER` | `SALVAGE_THEN_RIP` | `-30 LoC`]** `recipe-maker/build.py:1192-1203` — **fallback/expert_residency 'controls' are regexes over engine-flag strings**
   - **Mechanism**: checks.fallback.controls and expert_residency.controls 'enforce' engine behavior by matching patterns like --dspark[ =]+auto, --ssd-cache 0, layer split (\d[\d,]*) against recipe.conditions.engine_flags strings. A recipe that sets the flag differently, wraps it in a script, or lies in conditions passes; nothing observes the live engine. The src even admits 'no SHEXP/HC positive dispatch receipt exists here'.
   - **Fix / Action**: Keep the pattern tables only as ledger-descriptive metadata; move enforcement to the run-capture path where the launcher emits a machine-readable flag struct (parsed argv, not regex on a string) recorded before launch.
5. **[MEDIUM | `ARCH_DEAD_END` | `SALVAGE_THEN_RIP` | `-20 LoC`]** `recipe-maker/build.py:1413-1436` — **maintained_recipe_checks retrojects today's intent onto recorded runs**
   - **Mechanism**: Historical ledger rows get a fallback_plan injected from current recipes/*.json spec files, keyed only on id/box/model/engine equality, producing a hybrid row that is neither the captured launch nor the maintained recipe. Downstream check evaluation cannot distinguish 'the run was safe' from 'we wish it had been', inviting accuracy-report contamination even with the basis label.
   - **Fix / Action**: Evaluate maintained checks only in a separate prospective table joined at report time by recipe id; never mutate the run rows. The compatibility-guard idea (lines 1428-1429) is worth keeping as a join predicate.
6. **[MEDIUM | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `recipe-maker/build.py:1440-1449` — **Per-card expert placement parsed from run-label prose**
   - **Mechanism**: placements for lassie-orn35 runs are reconstructed by a human reading ledger labels ('10 on the GTX 1080 Ti (blk 0-8 + 10...)') and hardcoding layer lists, with o2 marked ESTIMATE by copying o6's split. Any future run whose placement lives in its label is invisible to predictRecipe; the o2 guess can be wrong by the blk.40 difference the comment itself flags.
   - **Fix / Action**: Add a structured placement field to record_run.py intake and backfill these three rows once; delete the prose-derived table.
7. **[MEDIUM | `PERF_PATHOLOGY` | `FIX_IN_PLACE`]** `recipe-maker/build.py:1745-1759, 1766-1771` — **O(N^2)+ row rescans in gpu_stacks/gpu_lines**
   - **Mechanism**: gpu_stacks re-filters all rows per backend, then per stack calls gpu_lines which re-filters ok rows again per backend and per (box,gpu) inside nested comprehensions; row_stack is recomputed per comparison (json.dumps per row per stack key). With a growing bench ledger this is quadratic-plus in row count on every build.
   - **Fix / Action**: Group rows once into a dict keyed by (backend, box, gpu_id, stack-json) before fitting; compute row_stack one time per row and cache it.
8. **[MEDIUM | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `recipe-maker/build.py:823-833` — **fnxc = shallow dict(models['fnx']) shares all nested mutable state**
   - **Mechanism**: models['fnxc'] = dict(models['fnx']) copies one level: lookups, draft, row_limits, host_engine, and the later-added kv_host are the same dict objects as fnx. A subsequent in-place mutation of either model's shared subtree (the file's dominant idiom) silently corrupts the other; only the explicitly .update()d keys diverge.
   - **Fix / Action**: Use copy.deepcopy (already imported) for the base, or define fnxc from a shared build_model() constructor taking the divergent fields.
9. **[MEDIUM | `FLAW_SECURITY_OVERSIGHT` | `FIX_IN_PLACE`]** `recipe-maker/build.py:1963-1966, 1977, 2031-2032` — **batch term depends on overlay-injected 'hand' key and silently vanishes on missing rows**
   - **Mechanism**: card_vram_gib reads _bgpu['vram_gib']['hand'], a key that only exists after probe/overlay.py rewrites the n() dict; under FITVIZ_OVERLAYS=0 (the documented hand-tables-only mode) this KeyErrors mid-build. Conversely, any missing BATCH_ROWS ledger entry only prints and then sets batch=None, so the whole batched-decode section disappears from data.json without a non-zero exit.
   - **Fix / Action**: Read vram_gib['v'] with a fallback to the hand value; make missing mandatory ledger rows a hard build error (or an explicit --allow-missing flag) instead of print-plus-None.
10. **[LOW | `FLAW_SECURITY_OVERSIGHT` | `FIX_IN_PLACE`]** `recipe-maker/build.py:69-83` — **Non-atomic snapshot rewrite into tracked src/ during build**
   - **Mechanism**: input_path copies live sources over src/inputs snapshots with shutil.copyfile directly at the destination: an interrupted build leaves a truncated snapshot that a later commit carries as truth, and a concurrent build reading the same snapshot sees a torn file. Path components come from fitviz.local.json dirs config with no confinement.
   - **Fix / Action**: Write to snap.tmp then os.replace for atomicity, and validate that resolved live paths stay inside the configured source roots.

---

## s179_homelab-atlas__recipe-maker

- **Scope**: `recipe-maker` | **Model**: `glm-5.3` | **Status**: `ok` | **Files**: `4` | **Lines**: `1,744`
- **Covered slices**: `recipe-maker/build.py` (`L2101-L2668` of `2668`), `recipe-maker/report.js` (`L1-L1005` of `1005`), `recipe-maker/families/orn122.py` (`L1-L101` of `101`), `recipe-maker/families/orn35.py` (`L1-L70` of `70`)

### Summary

A 2,668-line single-namespace Python build script, a 1,005-line monolithic Node CLI, and two family-plugin scripts executed inside the build's global namespace. Genuine, unusually rigorous validation work (cross-checking specs against headers, honest ESTIMATE/UNKNOWN tagging) is undermined by regex-archaeology over free-text notes, substring-based evidence attribution, exec() of sibling code, and a dual-language (Python/Node) calibration-hash ritual that spawns a synchronous python3 process per check.

### Module & Bloat Verdicts

| Module / File | Verdict | Removable LoC | Rationale & Salvage Notes |
| :--- | :--- | ---: | :--- |
| `recipe-maker/build.py` | `FIX_IN_PLACE` | 15 | The validation core (window_config split coverage, ubench/spec layer-kind clash checks, priors completeness) is real invariant enforcement, not theater. But exec() of priors_adapter.py, double data.json serialization, substring run-id attribution, and regex extraction of 'MEASURED' numbers from layout_note prose are structural defects needing surgical fixes. — page_assignment_parts (lines 2597-2622): clean recursive size-bounded statement splitting with an indivisible-scalar guard; keep as-is, it is the best code in the shard. |
| `recipe-maker/report.js` | `FIX_IN_PLACE` | 40 | Fifteen subcommands in one early-exit chain is ugly but functional and each branch does real work with honest labeling (post-hoc vs launch, replayed vs frozen). Fix the synchronous python3 spawn, hostname-gated symlink resolution, and the unbounded knob-combo cartesian solve; do not rewrite. — Lines 96-103 and 2508 (EVIDENCE_CITE_MIN) honesty framing ('not judged, so not passed', 'GO never covers them') is a discipline worth preserving verbatim in any refactor. |
| `recipe-maker/families/orn122.py` | `FIX_IN_PLACE` |  | Every borrowed number is explicitly tagged ESTIMATE with the analog run cited (acceptance 0.85 'the low end', GLM-scaled compute buffer with the derivation shown) -- the opposite of cosplay. Defect is architectural: it runs in build.py's bare namespace with ~20 implicit globals (n, R, E, models, kldglm, KQM_TY, FNXH, ...) and fabricates one provenance string via FNXH.replace(). — Lines 56-62 (Gated DeltaNet state-size arithmetic: full-attention KV + f32 recurrent/conv state derivation) are a clean, reusable VRAM model for hybrid-attention families; extract into a shared helper both family files call. |
| `recipe-maker/families/orn35.py` | `FIX_IN_PLACE` |  | Honest UNKNOWN/ESTIMATE tagging throughout (mult_default 1.0 'neutral placeholder, not an acceptance claim'), but copy.deepcopy(models['orn122']) then dict-patch inheritance creates a silent load-order dependency on orn122.py having executed first (uses GG, ORN_CTX, GLM_CB4096_MIB, models['orn122']) with zero enforcement -- a KeyError away from a confusing build failure. — Lines 34-38 (explicitly refusing to inherit the 122B's shapes/evidence while borrowing only numerical priors) document the right inheritance boundary; keep the pattern, enforce it with an explicit context object. |

### Findings (11)

1. **[HIGH | `FLAW_SECURITY_OVERSIGHT` | `FIX_IN_PLACE` | `-2 LoC`]** `recipe-maker/build.py:2495-2496` — **exec() of sibling file in build namespace**
   - **Mechanism**: exec(open('src/priors_adapter.py').read()) runs arbitrary code with full access to every build global (models, data, ledger); any planted or corrupted sibling file silently mutates the entire planner dataset before it is hashed and frozen, and the calibration identity then certifies the tampered output as FROZEN.
   - **Fix / Action**: importlib.util.spec_from_file_location + a module exposing apply_priors(data); the same pattern build.py already uses for tps/build_tps.py at line 2662.
2. **[HIGH | `COSPLAY_THEATER` | `FIX_IN_PLACE`]** `recipe-maker/build.py:2286-2330` — **'MEASURED' stage-window numbers regex-harvested from free-text notes**
   - **Mechanism**: curve_configs parses engine flags with per-field user-supplied regexes over a concatenated flag string, then win_re.findall() scrapes stage timings out of recipe.layout_note prose (line 2310). A note reading 'stage run n=8 ... 41.02 ms' yields a number tagged MEASURED and fed to the fit; a reworded note silently yields no window term at all -- no error, just absent data.
   - **Fix / Action**: Move stage timings into a structured measured.stage_ms field on the ledger row (record_run.py already writes measured dicts); keep the regex only as a loud deprecation path that prints 'row X: layout_note timings no longer parsed'.
3. **[HIGH | `ARCH_DEAD_END` | `REPLACE_BOUNDARY` | `-12 LoC`]** `recipe-maker/report.js:39-51` — **Dual-language canonicalization: python3 subprocess per --check**
   - **Mechanism**: Because JS and Python serialize floats/JSON differently, the calibration check spawns execFileSync('python3', ['build.py', '--calibration-data-sha256']) on every --check invocation, shipping the entire data.json over stdin to recompute a hash Python could have written into meta at build time. A 300-line frozen-manifest ritual exists to paper over the fact that there is no single canonicalization owner across the language boundary.
   - **Fix / Action**: build.py writes data_sha256 into data.json meta.calibration at build time (it already computes it at line 2590); report.js hashes the raw data.json bytes it already read (storedData) and compares -- zero subprocesses, one owner.
4. **[MEDIUM | `COSPLAY_THEATER` | `FIX_IN_PLACE` | `-3 LoC`]** `recipe-maker/build.py:2532-2534` — **Evidence attribution by substring run-id match**
   - **Mechanism**: stamp_evidence links a value to ledger runs via `i in src or any(s in src for s in ss)` -- plain substring containment between run ids and source prose, gated only by EVIDENCE_CITE_MIN=12 (line 2508), a magic length below which a source 'names no one run'. A run id appearing incidentally in any source string fabricates a citation; a cited run whose id is short is silently dropped.
   - **Fix / Action**: Have ledger rows cite run ids in an explicit runs: [...] array field and match on exact set membership; drop the 12-char threshold entirely.
5. **[MEDIUM | `COSPLAY_THEATER` | `FIX_IN_PLACE` | `-2 LoC`]** `recipe-maker/families/orn122.py:94-97` — **Provenance string fabricated via .replace() on another family's source text**
   - **Mechanism**: The heat-set src is built as FNXH.replace('Flash-Next routing not counted yet (scout measurement 3 counts it)', 'Ornith routing not counted yet: UNKNOWN routing') -- string surgery on the Flash-Next family's evidence citation produces a src that cites Flash-Next's provenance chain for Ornith's synthetic heat, misattributing where the shape came from.
   - **Fix / Action**: Write the Ornith src string directly ('synthetic GLM shape, seed=122, no Ornith routing counts'); the FNXH replace saves 3 lines at the cost of a false citation trail.
6. **[MEDIUM | `ARCH_DEAD_END` | `REPLACE_BOUNDARY`]** `recipe-maker/families/orn35.py:39-68` — **deepcopy-then-patch inheritance with implicit load-order coupling**
   - **Mechanism**: o35 = copy.deepcopy(models['orn122']) then .update()/draft.update() patches: orn35 silently requires orn122.py to have exec'd first (GG module, ORN_CTX, GLM_CB4096_MIB, GLM_VOCAB, models['orn122'], KQM_TY all implicit build globals). Any field orn122 later adds or renames (e.g. draft structure change) half-propagates: stale fields survive the patch as orphaned 122B numbers in the 35B entry.
   - **Fix / Action**: Replace both family files' run-in-build-namespace contract with an explicit loader: build.py calls family.register(ctx) where ctx exposes models/types/helpers as an object; orn35 composes from orn122's register output via an explicit field list, not deepcopy.
7. **[MEDIUM | `PERF_PATHOLOGY` | `FIX_IN_PLACE` | `-10 LoC`]** `recipe-maker/report.js:954-1001` — **Exhaustive 384-combo cartesian knob search with full solver per cell**
   - **Mechanism**: rec() enumerates 2*3*2*2*2*2*2 = 384 knob combos, then for each of 2 models x N boxes x M engines runs a full FM.prepare/frontier/bestAt solve; combined with the frontier block at lines 918-933 (boxes x 3 knobsets x 4 models x 2 modes x engines) this is hundreds to thousands of independent solver invocations, each reconstructing FM.context from scratch with no memoization across combos sharing a prefix.
   - **Fix / Action**: Prune in rec() itself (the skip conditions at lines 972-976 are evaluated only after full enumeration) and memoize FM.context per (bk,mk,eng) since knobs like addOptane/freeDesktop only alter capacities -- or cache prepare() results keyed on the effective context inputs.
8. **[MEDIUM | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `recipe-maker/report.js:61-63` — **Hostname string match gates symlink resolution**
   - **Mechanism**: os.hostname().split('.')[0] === r.box decides whether paths are realpathSync'd; a box whose hostname differs from the recipe's box key (containers, aliases, FQDN with different short name) silently skips resolution and the check runs on unexpanded paths, and realpathSync failures are swallowed to return the raw path with no note.
   - **Fix / Action**: Pass an explicit --on-box flag or per-box fact from data.boxes (box hostname recorded at intake) instead of inferring locality from hostname; on resolve failure emit a res.checks note, not silence.
9. **[LOW | `PERF_PATHOLOGY` | `FIX_IN_PLACE` | `-1 LoC`]** `recipe-maker/build.py:2551-2591` — **data.json serialized and written twice**
   - **Mechanism**: json.dump(data...) at line 2551, then planner_outputs() runs node (up to 300 s), three blocks are grafted on, and the entire data tree is re-serialized and rewritten at line 2591. For a multi-MB document this doubles serialization cost and leaves a window where a crash leaves the first, calibration-less data.json on disk looking valid.
   - **Fix / Action**: Write once after planner_outputs grafts; if the pre-advisor dump is deliberately sacrificial, delete line 2551 and document why.
10. **[LOW | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `recipe-maker/build.py:2660-2665` — **Dynamic import of tps/build_tps.py shares build globals by side effect**
   - **Mechanism**: exec_module of build_tps runs with no arguments contract beyond write(HERE, data, PAGE_DIR); like the family files, it executes against whatever module state exists at line 2664, and a failure inside it aborts the build after data.json and page/ were already written -- a partially published tree with no rollback.
   - **Fix / Action**: Wrap in try/except that logs and exits nonzero BEFORE any writes complete, or write all outputs to a staging dir and rename atomically at the end; the latter also fixes the double-write window.
11. **[LOW | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `recipe-maker/report.js:581-588` — **Sync git subprocess with null-swallowing in --predict provenance capture**
   - **Mechanism**: git(...) wraps execFileSync in try/catch returning null, so a repo without git (or a git failure) records rm_commit: null and rm_dirty: false -- the false reads as a clean tree in the prediction record, the exact opposite of the unknown state. The record is then frozen as pre-run provenance.
   - **Fix / Action**: Distinguish the three states: return {commit, dirty: true|false|undefined} and refuse --predict outright when commit is unavailable, since a provenance record with no provenance is theater.

---

## s181_homelab-atlas__recipe-maker__src

- **Scope**: `recipe-maker/src` | **Model**: `glm-5.3` | **Status**: `ok` | **Files**: `7` | **Lines**: `1,943`
- **Covered slices**: `recipe-maker/src/accuracy.js` (`L1-L372` of `372`), `recipe-maker/src/advisor.js` (`L1-L201` of `201`), `recipe-maker/src/app.js` (`L1-L1100` of `1100`), `recipe-maker/src/gguf-header.py` (`L1-L7` of `7`), `recipe-maker/src/heat_shape.py` (`L1-L28` of `28`), `recipe-maker/src/ledger.js` (`L1-L58` of `58`), `recipe-maker/src/ledger.py` (`L1-L177` of `177`)

### Summary

A homelab MoE layout planner: a pure-function accuracy/validation core (accuracy.js), a trim/upgrade advisor, a 1100-line no-dependency DOM viz, and a dual Python/JS JSONL run-ledger. Mostly honest, unusually well-commented engineering with real measured-data discipline, but it carries duplicated ledger implementations, O(N^2) stringify/predict recomputation on render paths, a non-atomic ledger seal that duplicates rows on crash, and a synthetic-heat generator whose only tuning parameter is mathematically dead.

### Module & Bloat Verdicts

| Module / File | Verdict | Removable LoC | Rationale & Salvage Notes |
| :--- | :--- | ---: | :--- |
| `recipe-maker/src/accuracy.js` | `FIX_IN_PLACE` |  | Dense but coherent pure-function core with real hash-identity, prospective-capture and hindsight-rejection logic; the strongest file in the shard. Needs perf surgery (repeated canonical-JSON digests) and de-brittling of phase-name regexes, not a rewrite. |
| `recipe-maker/src/advisor.js` | `FIX_IN_PLACE` |  | Structurally sound offline advisor, but bestPassing is executed twice per all() call with full predictRecipe sweeps, and runQuality does per-pair JSON.stringify store comparison over the whole runs array. |
| `recipe-maker/src/app.js` | `FIX_IN_PLACE` |  | A 1100-line single-closure monolith, but a working, accessible, dependency-free UI with consistent escaping. Its disease is recomputation (measuredHere called 3x per render, each re-predicting every run) and full innerHTML teardown per knob toggle — fix with memoization, not a framework rewrite. |
| `recipe-maker/src/gguf-header.py` | `RIP_OUT` | 7 | A 7-line shim whose only justification is old commands quoted in NOTES.md; the real reader lives in probe/gguf.py. Update the notes, delete the shim. |
| `recipe-maker/src/heat_shape.py` | `FIX_IN_PLACE` |  | Small, honestly labeled (ESTIMATE src string) synthetic router whose shape is fitted to three real observations. One real correctness bug: the frac_of parameter algebraically cancels, so it silently does nothing. |
| `recipe-maker/src/ledger.js` | `FIX_IN_PLACE` |  | Correct but uncached: every rows() call re-reads and re-parses every segment (including gunzip) synchronously; callers invoke it repeatedly per report. Add a mtime-keyed cache. |
| `recipe-maker/src/ledger.py` | `FIX_IN_PLACE` |  | The authoritative ledger; the pack() seal is non-atomic across the segment-write/replace boundary (duplicate rows on crash), and the whole concept is duplicated in ledger.js — a documented but real split-brain risk. |

### Findings (11)

1. **[HIGH | `PERF_PATHOLOGY` | `FIX_IN_PLACE`]** `recipe-maker/src/app.js:211-227` — **measuredHere() re-executed 3x per render, each predicting every run**
   - **Mechanism**: frontierSVG (line 238), chartNotes (line 317) and frontierTable (line 884) each call measuredHere(), which filters data.recipes and invokes FM.predictRecipe per row. Every render therefore runs 3x the recipes count of full model predictions; render fires on every knob toggle, theme click and box change.
   - **Fix / Action**: Compute measuredHere() once in compute() (it only depends on st.box and data.overrides), store on R, and read R.anc from all three call sites. Predictions per run can be memoized by canonical recipe hash.
2. **[COSPLAY | `COSPLAY_THEATER` | `RIP_OUT` | `-7 LoC`]** `recipe-maker/src/gguf-header.py:1-7` — **Back-compat shim kept only for commands quoted in NOTES.md**
   - **Mechanism**: The file imports the real reader from probe/gguf.py and dumps it; its sole reason to exist is that old NOTES.md command lines reference it — dead indirection that will drift from the real reader's CLI.
   - **Fix / Action**: Update NOTES.md command examples to probe/gguf.py and delete this file.
3. **[MEDIUM | `PERF_PATHOLOGY` | `FIX_IN_PLACE`]** `recipe-maker/src/accuracy.js:317-318` — **decisions() recomputes canonical-JSON sha256 digests per member row**
   - **Mechanism**: Inside the per-member loop condition rs.some(...) evaluates sha and digest(core(r.pred.validation)) for every row of the cohort on every member iteration — each digest is a full networkCanonical + JSON.stringify + sha256 of a large cohort object, O(members^2) serializations.
   - **Fix / Action**: Hoist per-row digests once before the member loop: const digs = rs.map(r => digest(core(r.pred.validation))), then compare digs[i] against v.cohort_sha256 and each other by string equality.
4. **[MEDIUM | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `recipe-maker/src/accuracy.js:33-49` — **Draft/no-draft phase pairing rides on regexes over phase-name strings**
   - **Mechanism**: DRAFT_RX/OFF_RX match substrings like n4/m3/n0 inside free-form phase keys (std-n4, d600_m3_mean); any kit that names a workload containing 'm3' or a width token differently mis-pairs prediction with measurement, silently corrupting every accuracy median built on top.
   - **Fix / Action**: Make run rows carry explicit machine-readable phase metadata (draft_n per phase) at record time in record_run.py, and have primaryPhase/offOf read that field, keeping the regexes only as a fallback for legacy rows.
5. **[MEDIUM | `PERF_PATHOLOGY` | `FIX_IN_PLACE`]** `recipe-maker/src/advisor.js:24-26` — **runQuality does O(N^2) JSON.stringify store comparison**
   - **Mechanism**: For every run without an own gate, the runs array is filtered and JSON.stringify((x.recipe||{}).store) is recomputed against every same-model candidate, inside bestPassing which itself runs per run — quadratic stringify churn over the whole ledger on each advisor invocation.
   - **Fix / Action**: Precompute a Map keyed by model+engine+JSON.stringify(store) -> newest gated run once per bestPassing call (one stringify per run), then do O(1) lookups.
6. **[MEDIUM | `PERF_PATHOLOGY` | `FIX_IN_PLACE`]** `recipe-maker/src/advisor.js:184-188` — **all() runs bestPassing twice: duplicate full predict sweeps**
   - **Mechanism**: trimProposals and upgradeAdvice each call bestPassing(data, runs), which re-scans all runs, re-runs runQuality and FM.predictRecipe for each candidate — the entire qualification sweep executes twice per build/report.
   - **Fix / Action**: Compute bestPassing once in all() and pass its {best, skipped} result into both trimProposals and upgradeAdvice.
7. **[MEDIUM | `PERF_PATHOLOGY` | `FIX_IN_PLACE`]** `recipe-maker/src/app.js:714-721` — **Full innerHTML teardown and rebinding on every interaction**
   - **Mechanism**: render() rebuilds the entire multi-tab DOM (including both SVG charts, the network table, trust tables with a details-per-run) and re-attaches all listeners via bind() for a single knob change, theme click, or curve select; the 30ms debounce only coalesces knob bursts. All tabs (including hidden ones) pay the cost.
   - **Fix / Action**: Split render into per-region updaters keyed by what changed (knobs -> chart+layout only; network state -> network section only), or at minimum skip string-building hidden pages and render them lazily on tab switch.
8. **[MEDIUM | `FLAW_SECURITY_OVERSIGHT` | `FIX_IN_PLACE`]** `recipe-maker/src/heat_shape.py:23-23` — **frac_of parameter is mathematically dead in synth_heat**
   - **Mechanism**: C = lambda x: (x * frac_of) ** SHAPE_A / frac_of ** SHAPE_A simplifies to x ** SHAPE_A; frac_of cancels exactly, so callers passing frac_of != 1.0 silently get identical heat to frac_of=1.0. The synthetic distribution never scales with expert availability.
   - **Fix / Action**: Either delete the parameter (and its callers' arguments) or implement the intended semantics explicitly, e.g. renormalize C over [0, 1/frac_of]: C(x) = min(1, (x*frac_of)**SHAPE_A), with a test asserting frac_of=0.5 differs from 1.0.
9. **[MEDIUM | `PERF_PATHOLOGY` | `FIX_IN_PLACE`]** `recipe-maker/src/ledger.js:50-54` — **rows() re-reads and re-gunzips the entire ledger on every call**
   - **Mechanism**: No caching: each rows(file) call synchronously reads all sealed segments (gunzipping the .gz ones), parses every JSONL line, and re-applies the overlay. advisor.js and report tooling call it multiple times per invocation; cost grows linearly with ledger history on every command.
   - **Fix / Action**: Cache by [file, mtime-of-newest-segment] in module scope (a small Map), invalidating when the open file's mtime changes; the ledger is append-only so this is safe.
10. **[MEDIUM | `FLAW_SECURITY_OVERSIGHT` | `FIX_IN_PLACE`]** `recipe-maker/src/ledger.py:147-155` — **pack() seal is non-atomic: crash duplicates ledger rows**
   - **Mechanism**: Segments are written and closed (lines 148-151) before the open file is truncated via os.replace (152-155). A crash or power loss between the two leaves the sealed rows present in both a segment and the open file; the next pack re-seals them, and every rows() consumer sees duplicated run ids — poisoning best-run picks, fits and accuracy medians silently.
   - **Fix / Action**: Write the truncated open file first as path + '.open-tmp', then fsync segments and os.replace the open file in one final step; or record a seal manifest (sealed-through row count) that pack() consults to deduplicate on recovery. Also fsync the segment files and their directory before replace.
11. **[LOW | `ARCH_DEAD_END` | `REPLACE_BOUNDARY`]** `recipe-maker/src/ledger.js:1-58` — **Two hand-maintained implementations of the same ledger format**
   - **Mechanism**: ledger.py and ledger.js independently implement segment discovery, gzip handling, the caveats overlay (including the dotted-path recipe patching with subtly different mutation styles) and caveated(); any format evolution (e.g. a new caveat field) must be mirrored in both or JS and Python consumers see different ledgers.
   - **Fix / Action**: Keep ledger.py as the single owner and have node tooling shell out to `python3 src/ledger.py` once for a JSON dump (offline tooling, latency acceptable), or generate a packed data/ledger.json at build time that ledger.js reads — one parser, one truth.

---

## s182_homelab-atlas__recipe-maker__src

- **Scope**: `recipe-maker/src` | **Model**: `glm-5.3` | **Status**: `ok` | **Files**: `1` | **Lines**: `2,150`
- **Covered slices**: `recipe-maker/src/model.js` (`L1-L2150` of `5976`)

### Summary

A shared browser/node MoE quantization-placement planner: heat normalization, a Lagrangian tier/type solver, and a sprawling 'hard checks' layer that grades recipes by tag-based evidence classes and regex matches over config prose. The solver core (heatSet/prepare/choose/solveLambda/evaluate) is genuinely engineered, but it is buried under a 1500-line checks/prediction tower built on string-key conventions, regex flag sniffing, and duplicated value/provenance lookup pairs that will silently diverge.

### Module & Bloat Verdicts

| Module / File | Verdict | Removable LoC | Rationale & Salvage Notes |
| :--- | :--- | ---: | :--- |
| `recipe-maker/src/model.js` | `FIX_IN_PLACE` | 300 | The solver core (lines 43-670) is structurally sound typed-array math earning its keep; the checks layer (672-1625) is a regex/string-convention band-aid tower that needs a typed facts boundary, not deletion. |

### Findings (11)

1. **[HIGH | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `recipe-maker/src/model.js:426-461` — **choose() crashes on empty/infinite option sets (bsel null, option index -1)**
   - **Mechanism**: In the layer-scope branch, if ctx.qList is empty or every base+price cost is Infinity (all tiers capped), `bsel` stays null and line 451 dereferences `bsel[i]`; similarly the per-expert branch can pick `best = -1` when all costs are Infinity, so `add(g, -1)` reads `opts[-1]` -> undefined -> TypeError on O.ti. Capped tiers with cap=0 and price=0 make this reachable, not theoretical.
   - **Fix / Action**: Guard both loops: if no finite option exists for a group, mark the solution infeasible and skip add() (or fall through to the repair pass); make add() throw on o<0 defensively.
2. **[MEDIUM | `COSPLAY_THEATER` | `FIX_IN_PLACE`]** `recipe-maker/src/model.js:11-17` — **Evidence class (seen/inferred/unknown) is tag-table + src-prefix string matching, not verification**
   - **Mechanism**: evidenceOf trusts any object carrying a self-declared `evidence` field outright, else looks up a tag in data.meta.evidence.by_tag, and upgrades a READ tag to 'seen' purely because the src string starts with a listed prefix. Any catalog row tagged MEASURED with a plausibly-prefixed src passes as 'seen'; the whole downstream gateRepair/verdictReady trust chain rests on this string-match rubber stamp.
   - **Fix / Action**: Make evidence an immutable field stamped at ingest (build.py) from run receipts keyed by run_id, and delete the prefix-matching upgrade path; anything lacking a receipt link is 'unknown' unconditionally.
3. **[MEDIUM | `PERF_PATHOLOGY` | `FIX_IN_PLACE`]** `recipe-maker/src/model.js:826-878` — **gateRepair is O(N^2) in layer scope and re-sorts ctx.order per pass**
   - **Mechanism**: The while loop (line 856) runs up to N guard iterations; each layer-scope iteration scans all N experts to find the worst 2-bit unit (line 870) and upgrades within it, giving O(L^2 * E) worst case; the expert-scope branch re-sorts a copy of ctx.order (line 860) inside the loop body. For per-expert models with N in the tens of thousands this is a multi-second stall on every render.
   - **Fix / Action**: Precompute per-layer 2-bit heat once, iterate layers in sorted order of (plan+score)*w heat upgrading until the budget is met — one O(N log N) pass; hoist the expert-scope sort outside the loop (it is order-invariant).
4. **[MEDIUM | `ARCH_DEAD_END` | `REPLACE_BOUNDARY` | `-80 LoC`]** `recipe-maker/src/model.js:1546-1575` — **fallbackCheck control enforcement is a regex catalog over engine flag strings**
   - **Mechanism**: Controls are applied by new RegExp(c.pattern) against a concatenation of engine_flags/engine_extra/draft_n/isolation prose, then layer-split stage counts are derived by regex-capturing comma lists from the same blob and cross-checked against gpu_split array lengths. The code's own comments concede 'flags are configuration evidence, never proof of executed dispatch', yet the whole control apparatus is built on exactly that string matching — a band-aid tower with no structural mechanism underneath.
   - **Fix / Action**: Replace the pattern catalog with a single typed `dispatch_manifest` field the engine emits (class -> tier -> proof receipt id); checkRecipe then does exact lookups and the regex catalog, layer_split_pattern, and stage-count heuristics all delete.
5. **[MEDIUM | `ARCH_DEAD_END` | `REPLACE_BOUNDARY` | `-40 LoC`]** `recipe-maker/src/model.js:905-913` — **numaCheck load-extra lookup parses compound string keys ('box|model|engine|banks<=X')**
   - **Mechanism**: Load-extra selection builds prefix strings and parses them back: filter keys startsWith(base + '|banks<='), then +k.split('<=')[1], sorted numerically to find the smallest covering bank size. Key-string encoding of a 4-dimensional lookup is unindexable, unvalidatable, and silently picks the engine fallback when the string arithmetic misses.
   - **Fix / Action**: Store load_extra as a typed nested structure {box, model, engine, split, banks_max_gib}[] and do an explicit candidate query; delete the string-prefix/split parsing entirely.
6. **[MEDIUM | `ARCH_DEAD_END` | `FIX_IN_PLACE`]** `recipe-maker/src/model.js:269-275, 347-352, 292-298, 353-357, 311-320, 358-363` — **Triplicated value/provenance lookup pairs (kappaOf/kappaSrc, denseGiBpre/denseSrc, kvGiBpre/kvSrc)**
   - **Mechanism**: Each concept has two hand-maintained implementations — one returning the bare number, one the tagged value — with subtly different fallback orders (kappaOf returns override as raw number, kappaSrc wraps it with a synthetic src; denseSrc drops the bpw scaling branch denseGiBpre applies). Any edit to one ladder silently diverges the planner's arithmetic from the report's provenance, which is exactly how the evidence ledger lies.
   - **Fix / Action**: Collapse each pair to one function returning the tagged value; derive the plain number via v() at call sites. Three deletions, three survivors.
7. **[MEDIUM | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `recipe-maker/src/model.js:1158-1164` — **VRAM unit detection by regex against a prose units string, silently defaulting to GiB**
   - **Mechanism**: vramGiB decides MiB-vs-GiB by /MiB\s*\/\s*1000/i on vram.units prose; any other phrasing of the same convention (or a missing units field on MiB data) silently treats MiB figures as GiB, a ~4.8% error that flips borderline vram_peak PASS/FAIL verdicts with no warning emitted.
   - **Fix / Action**: Require an explicit enum field vram.units: 'gib' | 'mib_1000' and FAIL the inputs check when absent but figures are given; delete the prose regex.
8. **[MEDIUM | `PERF_PATHOLOGY` | `FIX_IN_PLACE`]** `recipe-maker/src/model.js:496-520` — **solveLambda residual-repair loop is O(groups^2 * opts) worst case**
   - **Mechanism**: The repair while-loop runs up to groups.length times per tier; each iteration rescans all groups from the end to find a candidate (line 500) then scans all opts with a groupHW lookup (line 510). With per-expert grouping (G ~ N/160, hundreds of groups) and multiple overflowing tiers this is a quadratic hot spot inside every lambda solve, which itself runs per plan/score pair per knob change.
   - **Fix / Action**: Precompute per-tier group lists sorted coldest-first once; repair by popping from the sorted list instead of rescanning, turning the loop into O(groups * opts).
9. **[MEDIUM | `ARCH_DEAD_END` | `FIX_IN_PLACE`]** `recipe-maker/src/model.js:727-741` — **layerMap memoization via data._lm mutation with a hand-rolled re-entrancy guard**
   - **Mechanism**: layerMap pre-seeds data._lm[mk] = null 'so a recipe layout below must not read a half-built map', i.e. it defends against its own indirect recursion (layerMap -> recipeDamage -> recipeContext -> prepare -> evaluate -> layerWeights -> layerMap) by publishing a deliberately wrong null mid-computation. Any new call path into this cycle reads a permanently-wrong null that is then cached as final.
   - **Fix / Action**: Break the cycle structurally: pass the layer weights explicitly into evaluate/prepare as a parameter resolved once at context build, and make layerMap a pure memoized function of (data, mk) with a building flag that throws on re-entry instead of lying.
10. **[LOW | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `recipe-maker/src/model.js:526-535` — **expand() packs tier/type indices into Int8Array — silent wrapabove 127**
   - **Mechanism**: tier and type are Int8Array; any model whose planner type set exceeds 127 entries (or a catalog with >127 tiers/mounts) wraps to negative indices, corrupting every downstream evaluate/fill/gateRepair read with no error — the layout simply computes garbage tok/s.
   - **Fix / Action**: Use Int16Array (or Int32Array) for both arrays; the memory delta is negligible at N sizes in play.
11. **[LOW | `PERF_PATHOLOGY` | `FIX_IN_PLACE`]** `recipe-maker/src/model.js:1486-1489` — **fallbackCheck rebuilds a full planner context() per recipe just to list tier families**
   - **Mechanism**: Line 1489 constructs the entire context (GPU cloning via JSON.parse(JSON.stringify(g)), tier building, fixed-allocation solving inputs) only to extract Set(tier families); checkRecipes calls this per ledger row, and checkRecipe already built an equivalent context at line 1243 — duplicated heavy construction on the report path.
   - **Fix / Action**: Accept the existing ctx from checkRecipe as a parameter, or extract a cheap tierFamilyList(data, box, eng) helper that reads the catalog without building tiers.

---

## s183_homelab-atlas__recipe-maker__src

- **Scope**: `recipe-maker/src` | **Model**: `glm-5.3` | **Status**: `ok` | **Files**: `1` | **Lines**: `2,150`
- **Covered slices**: `recipe-maker/src/model.js` (`L2151-L4300` of `5976`)

### Summary

model.js lines 2151-4300 are the physics core of a homelab LLM serving planner: dLLM/batched-decode/window cost models, a network recipe gate, predictRecipe tier packing, multi-stage calibration, and discrete-event agent simulators. The modeling is genuinely evidence-tagged and largely non-cosplay, but it is a 5976-line single-file monolith of hand-rolled state folds over untyped dicts, with brute-force grid calibration, triple full predictRecipe passes during fitting, global mutable catalog state mutated mid-fit, and JSON.stringify used as an identity gate (NaN === null passes).

### Module & Bloat Verdicts

| Module / File | Verdict | Removable LoC | Rationale & Salvage Notes |
| :--- | :--- | ---: | :--- |
| `recipe-maker/src/model.js` | `FIX_IN_PLACE` |  | The physics, evidence-tagging and validation logic (dllmTerms, winFit/winNnls, agentEngineSim) are real and worth keeping; the rot is architectural density, perf pathologies in the calibration path, and string-based identity gates, all fixable surgically. |

### Findings (10)

1. **[HIGH | `FLAW_SECURITY_OVERSIGHT` | `FIX_IN_PLACE`]** `recipe-maker/src/model.js:2808-2809` — **Identity gate compares canonicalized JSON.stringify; NaN and null are indistinguishable**
   - **Mechanism**: networkCheck and winNetwork establish priced-identity equality via JSON.stringify(networkCanonical(x)). JSON.stringify maps NaN, Infinity and (in arrays) undefined to null, so an identity carrying a NaN rate or a null placeholder stringifies identically to one carrying explicit null; key-order is handled but value-semantics are not. A corrupted or attacker-edited assumption object can pass the identity check and flip prediction from UNMODELLED to PREDICTED.
   - **Fix / Action**: Write a 10-line typed deepEqual(a,b) that rejects on type mismatch, NaN, and undefined-vs-null; use it for identity comparison and delete networkCanonical/sameValue/ordered JSON plumbing.
2. **[HIGH | `PERF_PATHOLOGY` | `FIX_IN_PLACE`]** `recipe-maker/src/model.js:3626-3666` — **calibrate brute-force grid: ~311 f-values x PMem grid x models x 36-iter golden, each evaluating every anchor**
   - **Mechanism**: fGrid builds 311 points (0.05..1.6 step 0.005), pGrid up to dozens, and for each combination runs a 36-iteration golden host search per host group, and each host trial recomputes anchorTime over all anchors with fresh anchorParts loops. That is O(311 x |pGrid| x models x 36 x anchors) full model evaluations per box per calibration, and applyCal re-runs calibrate wholesale on every addModels call.
   - **Fix / Action**: For fixed hosts the f-objective is 1-D and unimodal in log space: nest golden on f (outer) and host h (inner), or do 3-5 rounds of coordinate descent starting from f_default; this is ~1000 evaluations instead of millions with identical optima.
3. **[COSPLAY | `COSPLAY_THEATER` | `FIX_IN_PLACE`]** `recipe-maker/src/model.js:2779-2791` — **winComparison parses frontier agents out of a ledger string with /^c[1-9]\d*$/**
   - **Mechanism**: The eligibility gate extracts the captured agent count by regex-matching the prose-shaped marker inside W.frontier.ledger.tps (a string like c16). Any ledger formatting drift (c016, C16, c16x) silently yields captured = null, which degrades every network lever to UNKNOWN for a cosmetic reason, while a matching-but-wrong marker passes the identity check.
   - **Fix / Action**: Store the captured agent count as a typed integer field on the frontier object at build time (frontier.captured_agents); keep the string only for display. The regex then dies.
4. **[MEDIUM | `PERF_PATHOLOGY` | `FIX_IN_PLACE`]** `recipe-maker/src/model.js:3804-3895` — **applyCal fitting pipeline re-runs predictRecipe over the full recipe ledger three separate times**
   - **Mechanism**: ledgerHosts calls predictRecipe for every clean ledger row; ledgerHops then calls predictRecipe again for the same rows (only differing in the cross_layers filter); ledgerDrafts calls it a third time per draft twin. predictRecipe is a ~260-line tier-packing O(N log N) routine with prefillModel and numaCheck side models, so the calibration path is 3x the necessary cost and the three fits see subtly different data states (host set between passes).
   - **Fix / Action**: Compute the predictions once into a map keyed by run_id at the top of applyCal, pass it to ledgerHosts/ledgerHops/ledgerDrafts; also merge the three row filters into one pass. Removes two full ledger sweeps and the inter-pass state drift.
5. **[MEDIUM | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `recipe-maker/src/model.js:2165-2175` — **addModels mutates shared data.models in place and throws mid-loop, leaving half-applied state**
   - **Mechanism**: keys.forEach adds models[k] and injects KLD rows per type as it iterates; if a later key collides, the throw leaves earlier keys permanently in data.models and data.types with UNKNOWN KLD, but applyCal is never run, so subsequent consumers see a catalog with unfitted new models and no error marker.
   - **Fix / Action**: Validate all keys against data.models first, build a staged copy (models/types patches), then commit atomically, then applyCal; or snapshot data.models keys and roll back in a catch.
6. **[MEDIUM | `ARCH_DEAD_END` | `FIX_IN_PLACE`]** `recipe-maker/src/model.js:2182-2189` — **externalSummary hidden _ext cache via defineProperty is never invalidated**
   - **Mechanism**: Object.defineProperty(data, '_ext', ...) memoizes the external validation summary, but addModels mutates data.models and applyCal refits cal afterward; any predictExternal call path that reads model or calibration state (host terms, borrowHostLists results) after mutation serves the stale summary, silently skewing the external-check percentages printed in evidence rollups.
   - **Fix / Action**: Make the cache explicit and versioned: store _ext alongside a data._calVersion stamp set by applyCal/addModels, and recompute when the stamp differs; delete the defineProperty trick.
7. **[MEDIUM | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `recipe-maker/src/model.js:2404-2410` — **batchFit least-squares slope divides by zero variance, propagating NaN into K.row**
   - **Mechanism**: The denominator x.reduce((a,xi)=>a+(xi-mx)**2,0) is 0 when the kernel points share one N (or a single point exists); slope becomes NaN, all three K assignment branches then compare NaN (< 0 false, >= 0 false) and fall to the third branch producing NaN row/fixed, which silently poisons every batchStep prediction built on the fit with no guard or why-message.
   - **Fix / Action**: Guard: if the denominator is 0 or kp.length < 2, set K = { fixed: my, row: 0 } and push a confounded note into the fit result so downstream tables can show UNKNOWN instead of NaN totals.
8. **[MEDIUM | `FLAW_SECURITY_OVERSIGHT` | `FIX_IN_PLACE`]** `recipe-maker/src/model.js:3639-3653` — **calibrate temporarily overwrites shared box.pmem.bw; an exception leaves the catalog corrupted**
   - **Mechanism**: The PMem fit loop assigns box.pmem.bw = { v: pb } on the live data.boxes object and restores pmOrig only after both loops complete; any throw inside (bad anchor, NaN in anchorTime) skips the restore, so every later prediction on that build uses a grid-scratch bandwidth value tagged with no evidence, fail-open.
   - **Fix / Action**: Pass the candidate PMem rate as an explicit parameter through anchorTime (like f and h already are) instead of mutating shared state; the restore block and pmOrig juggling disappear.
9. **[MEDIUM | `PERF_PATHOLOGY` | `FIX_IN_PLACE`]** `recipe-maker/src/model.js:2308-2329` — **dllmValidation leave-one-out refits dllmTerms from scratch per row, O(rows^2) over cells**
   - **Mechanism**: Each loo row calls dllmTerms(data, [r.id]) which re-filters, re-groups and re-medians every cell and every kernel-gap group; dllmTerms itself is O(rows x cells). With R fit rows the validation is O(R^2), and each dllmTerms call also re-reads D.fit_modes.v/D.fit_blocks.v per row inside the filter. Tolerable at tiny R, but it is the report hot path and grows quadratically with ledger growth.
   - **Fix / Action**: Precompute the per-cell and per-group row lists once (grouped index), then each LOO refit is a median-without-one over the affected group only (O(log n) with sorted arrays or a precomputed median pair), turning validation into O(R log R).
10. **[MEDIUM | `PERF_PATHOLOGY` | `FIX_IN_PLACE`]** `recipe-maker/src/model.js:4250-4272` — **agentEngineSim event loop rescans every agent and recomputes slot statistics per dt step**
   - **Mechanism**: Each simulation step runs ag.forEach for readiness, slots.filter(s=>s.busy).length inside admit, decs/live filters, and heads.map, all O(n + S) per event; event count scales with turns x agents, so the sim is O(turns x n^2) with a 5e6 guard that converts a large run into a thrown error rather than a slower answer. dt computation also scans all gap agents each step.
   - **Fix / Action**: Keep a min-heap (or sorted set) of agent ready-times and a busy-slot counter updated incrementally; maintain the decoding set as a counted structure so rate/Dk lookups are O(1). Same sim, O(turns x log n).

---

## s184_homelab-atlas__recipe-maker__src

- **Scope**: `recipe-maker/src` | **Model**: `glm-5.3` | **Status**: `ok` | **Files**: `4` | **Lines**: `2,400`
- **Covered slices**: `recipe-maker/src/model.js` (`L4301-L5976` of `5976`), `recipe-maker/src/notes.js` (`L1-L392` of `392`), `recipe-maker/src/priors_adapter.py` (`L1-L65` of `65`), `recipe-maker/src/runs_tables.py` (`L1-L267` of `267`)

### Summary

A 6k-line hand-rolled performance-prediction monolith (roofline/prefill/decode/agent-sim/replica-gate) plus three data-plumbing sidecars. The physics engine is unusually honest about evidence classes and pre-registration, but it is riddled with bisection loops that re-execute the full model thousands of times per report, two competing date-comparison regimes, regex parsing of engine flag strings, and three parallel decode-rate models. priors_adapter.py is pure schema-guessing cosplay.

### Module & Bloat Verdicts

| Module / File | Verdict | Removable LoC | Rationale & Salvage Notes |
| :--- | :--- | ---: | :--- |
| `recipe-maker/src/model.js` | `FIX_IN_PLACE` | 300 | The core (roofline terms, pre-registration records, evidence rollups, replica gate) earns its keep and is honestly tagged, but needs surgical perf and correctness fixes: O(70x-full-model) bisections, string date compare, regex flag parsing, and the hidden __cardFit cache — rooflinePredict/rooflineScore + rlInstant/rlAfterRecord (5828-5916) is a genuinely good pre-registration boundary worth extracting into its own module |
| `recipe-maker/src/notes.js` | `FIX_IN_PLACE` | 60 | Offline report generator, structurally fine, but embeds frozen hardcoded measurements in prose guarded by dead ternaries and re-predicts every recipe at notes time |
| `recipe-maker/src/priors_adapter.py` | `RIP_OUT` | 65 | Walks a self-declared unknown schema, guesses family attribution by substring match on JSON key paths, and rubber-stamps whatever number it finds as PRIOR data — a schema contract, not a walker, is the boundary |
| `recipe-maker/src/runs_tables.py` | `FIX_IN_PLACE` | 40 | Ledger-to-table plumbing is sound (caveat overlay, holdout exclusion enforced), but the served()/draft_twins phase selection is a regex token heuristic tower over free-form phase names that should be a declared field on the ledger row — decode_fits (228-243) train/holdout overlap guard is worth keeping verbatim |

### Findings (10)

1. **[CRITICAL | `PERF_PATHOLOGY` | `FIX_IN_PLACE`]** `recipe-maker/src/model.js:5716-5732` — **rlAttribute bisection re-runs full prefillRoof 70x per term**
   - **Mechanism**: at(id,k) calls prefillRoof (the entire term/pricing pipeline) inside a 70-iteration sqrt bisection, per term, per check row; rooflineChecks triggers this for every non-fit check, so one report is terms x 70 x full-model executions on the hot path
   - **Fix / Action**: Prefactor prefillRoof into term sums once per query and bisect on the closed-form composition (rlComposedTime already exists and does exactly this on frozen records — reuse it for live attribution instead of re-running the model)
2. **[HIGH | `PERF_PATHOLOGY` | `FIX_IN_PLACE`]** `recipe-maker/src/model.js:5658-5671` — **rooflineFit: 80-iteration bisection, each pass re-pricing every fit cell**
   - **Mechanism**: at(e) sums prefillRoof over all cells per iteration; 80 iterations per engine per device, and rlEta/rlEtaFor call rooflineFit on every decodeRoof/prefillRoof caller path with no memoization, so batchLandscape and agentLandscape multiply this by 8 batch rows
   - **Fix / Action**: Cache {engine,device}->eta on the data object like cardConsts does (or an LRU keyed on the fits blob), collapsing repeated fits to one bisection per report run
3. **[HIGH | `PERF_PATHOLOGY` | `FIX_IN_PLACE`]** `recipe-maker/src/model.js:5258-5276` — **cardCurveFit: 100-iteration bisection x full pipelineRoof per anchor point**
   - **Mechanism**: Each calibration point bisects over pipelineRoof (full multi-stage plan pricing incl. mbHop interpolation) 100 times; with multiple anchors and the rescore pass re-running pipelineRoof twice per point, this is the dominant cost of any multibox report
   - **Fix / Action**: cardStage is a closed-form of the constants; derive dc/dconstant analytically (or bisect on the per-stage term only) instead of re-walking hops and ceilings 100 times
4. **[HIGH | `FLAW_SECURITY_OVERSIGHT` | `FIX_IN_PLACE`]** `recipe-maker/src/model.js:4620-4626` — **unlockState compares ISO timestamps as raw strings while rlInstant exists**
   - **Mechanism**: String(q.at) < String(u.since) is lexicographic: 2026-10-17T23:00Z vs 2026-10-18T01:00+05:00 compares wrong across offsets, and a zoneless at silently passes; the codebase already has a strict zoned-instant parser (rlInstant, 5829) it refuses to use here — split-brain date semantics in the same file
   - **Fix / Action**: Use rlInstant on both sides and refuse (fall back to stock) when either lacks a zone, matching the discipline of rlAfterRecord
5. **[HIGH | `COSPLAY_THEATER` | `RIP_OUT` | `-65 LoC`]** `recipe-maker/src/priors_adapter.py:24-30` — **Family attribution by substring match on unknown-schema JSON paths**
   - **Mechanism**: fam_of checks whether the strings glm / v4 / deepseek / ds4 appear anywhere in the key path; a sibling file key like glm_compat or v4_notes silently mis-bins community KLD priors into the wrong model family, and ambiguous paths get assigned to BOTH families — guessed data flows into the quality gate as PRIOR without any schema verification
   - **Fix / Action**: Require quant-priors.json to declare an explicit schema version and family field per row; refuse (do not apply) on mismatch, deleting the walker
6. **[COSPLAY | `COSPLAY_THEATER` | `FIX_IN_PLACE`]** `recipe-maker/src/model.js:5619-5631` — **kappa bisection over a discontinuous staircase the comments admit fails**
   - **Mechanism**: agentPointRescore bisects decode_step_increment against agentTurnPredict, which counts whole turns — useful(lambda) is a step function, the bisection lands on an edge, the target can sit inside a jump, and the whole fit rests on ONE calibration point which may itself be UNVERIFIED-COUNT; the code papers over this with a nearest-side residual field
   - **Fix / Action**: Fit lambda by minimizing turn-time error (continuous per-turn timing) instead of useful_tps (staircase), or report the interval [lo,hi] bracket rather than picking one side and calling it FIT
7. **[MEDIUM | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `recipe-maker/src/model.js:4325-4327` — **Engine flags parsed by per-call regexes, lastNum evaluated twice**
   - **Mechanism**: cpu_share_re/chunk_re/split_re regexes are re-compiled and re-executed against the flag string inside passTerms (and cpu_share lastNum runs twice back-to-back at 4326), and the same regex matching is re-implemented independently in priorFacts (4555) and cpuPoolCheck (4521) — three regex towers over one flag string
   - **Fix / Action**: Parse flags once into a typed object (single split on whitespace into a key->value map) at the query boundary and pass the parsed struct to passTerms/cpuPoolCheck/priorFacts
8. **[MEDIUM | `ARCH_DEAD_END` | `FIX_IN_PLACE`]** `recipe-maker/src/model.js:5220-5225` — **Hidden non-enumerable cache bolted onto the shared data object**
   - **Mechanism**: cardConsts defines __cardFit via defineProperty on data so JSON serialisation silently drops it and any clone refits; callers reach the cache only through the mutation side effect, and cardCurveRescore (5283-5284) pokes data.__cardFit directly
   - **Fix / Action**: Lift the fit cache into an explicit module-level WeakMap keyed on the card_curves object reference — no hidden state on shared data
9. **[MEDIUM | `ARCH_DEAD_END` | `FIX_IN_PLACE` | `-40 LoC`]** `recipe-maker/src/notes.js:109-112` — **Frozen hardcoded measurements inside narrative ternaries**
   - **Mechanism**: The glm-lite GPU-cache paragraph hardcodes 37.7 %, 126.57, 22.22, 22.2 as literals inside conditionals that print ? — the prose drifts from data.json on every rebuild while LOOKING data-driven; same pattern for the droopy f refit at 101 and k2e constants at 107-108
   - **Fix / Action**: Route every number through the tagged-value tables (data.cal / data.bench) that the rest of the file already renders, or move the whole dated narrative block into a versioned changelog data file
10. **[MEDIUM | `UNDERENGINEERED` | `FIX_IN_PLACE` | `-30 LoC`]** `recipe-maker/src/runs_tables.py:112-129` — **Served-phase detection is a regex token heuristic tower over phase names**
   - **Mechanism**: MTP_TOKEN/DSPARK_TOKEN regexes plus _groups means-or-groups fallback plus the lone-token scan in served() (188-203) guess which decode phase was the served width from free-form keys like std-m3/d600_m3_mean; a phase named m3x or m3-final is mis-detected or silently dropped, and the guess=True fallback marks rows as comparisons with no comparator
   - **Fix / Action**: Add an explicit served_phase (or draft: on/off) field to the ledger row at record time and read it directly; keep the regex path only as a migration shim with a loud warning

---
