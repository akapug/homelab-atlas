# `homelab-atlas` Full-Codebase Audit Rake (`2026-11-14`)

Start with [AGENT-ORIENTATION.md](AGENT-ORIENTATION.md) for current dispositions and navigation.

- **Repository**: `homelab-atlas` (`main` baseline @ `0deaf22`)
- **Coverage**: **`10 / 10` shards (`100%`)** covering **`62` unique files** (`65` file slices, `18,322` lines)
- **Total Findings**: **`105`** (`1` `CRITICAL`, `22` `HIGH`, `64` `MEDIUM`, `9` `LOW`; `15` `COSPLAY`-tagged)
- **Module-Verdict Removable LoC Sum**: **`1,779` LoC**

> [!IMPORTANT]
> **Wave-1 (`PR #1` / `7b7b7ff`) & Wave-2 (`e3119f7` / `ember/wave2-fixes`) Status Note**
> This rake audited `main` (`0deaf22`) across all `10/10` `homelab-atlas` shards (`62` files, `18,322 LoC`, `105` findings). High-priority findings already resolved in Wave-1 & Wave-2:
> - **Already Landed in Wave-1 (`7b7b7ff`)**: Tightened `klEquivKey` arithmetic provenance borrowing and added the coding-fidelity gate (Wave 3.5 corrects its unattainable 0.98 floor to outcome-first/model-relative admission) across `advisor.js`, `model.js`, and `report.js`.
> - **Already Landed in Wave-2 (`5033862`, `e3119f7`)**: Separated `CAPACITY_EXCEEDED` (`!x.fits`) from `UNMODELLED BOTTLENECK` in `rlAttribute` + gated `COMPUTE CEILING MISSPECIFIED` on compute share; guarded `choose()` against `bsel === null` on `Infinity`/`NaN` shadow prices; fixed `networkCheck` `NaN`/`Infinity` -> `null` identity collision; memoized `klEquivKey` in `advisor.js`; removed `record_run.py` `or js[-1:]` cross-arm fallback; added `set -o pipefail` and safe JSON encoding to `bench_gpu.sh`; made `record_screen.py` writes atomic; forced fresh `build.py` execution in `check_atlas.py --accuracy`.
> - **Remaining Open Backlog in This Pack**: Replacing schema-guessing `walk()` adapters (`priors_adapter.py`, `vitals_adapter.py`, `window_inputs.py`) with explicit versioned JSON readers, converting `b70/vllm-xpu/*.py` runtime `str.replace` monkey-patchers into a checked git patch series, and shard-level `MEDIUM`/`LOW` numerical/calibration refinements in `shards.jsonl`.

---

## 1. Directory Layout & Condensed Syntheses

| Artifact | Purpose |
| :--- | :--- |
| [`shards.jsonl`](shards.jsonl) | Deterministic, compact UTF-8 JSONL (`1` shard per line, sorted by `shard_id`), **100% lossless** across all findings and module verdicts. Built for fast `jq` / `rg` queries. |
| [`synthesis/01-homelab-atlas-synthesis.md`](synthesis/01-homelab-atlas-synthesis.md) | `homelab-atlas` Roofline Model, Planner & Probe Tooling Synthesis |
| `by-area/*.md` | Human- and agent-readable chapters rendering every shard's covered file/line ranges, summary, module verdicts, and severity-sorted findings. |

---

## 2. Context-Safe `jq` & `rg` Recipes for Agents

Never read all of `shards.jsonl` into an LLM context window at once. Filter by repo-relative file path, severity, action, or category with `jq` or `rg`:

```bash
AUDIT=docs/audits/2026-11-14-codebase-rake/shards.jsonl

# 1. All findings touching a specific file or directory prefix (e.g. b70/agent-eval/run.py)
jq -c '.result.findings[]? | select(.file | startswith("b70"))' "$AUDIT"

# 2. All CRITICAL and HIGH findings across the repo (compact TSV: severity, file:lines, action, title)
jq -r '.result.findings[]? | select(.severity == "CRITICAL" or .severity == "HIGH") | [.severity, "\(.file):\(.lines // "?")", .action, .title] | @tsv' "$AUDIT"

# 3. All RIP_OUT / SALVAGE_THEN_RIP / REPLACE_BOUNDARY module verdicts sorted by removable LoC
[ -f "$AUDIT" ] && jq -r '[.result.module_verdicts[]? | select(.verdict != "FIX_IN_PLACE")] | .[] | [.loc_removable // 0, .verdict, (.file // .module), .rationale] | @tsv' "$AUDIT" | sort -rn | head -n 40

# 4. Findings by category (FLAW_SECURITY_OVERSIGHT, PERF_PATHOLOGY, COSPLAY, ARCHITECTURAL_MISMATCH, UNDERENGINEERED)
jq -c '.result.findings[]? | select(.category | test("PERF_PATHOLOGY|SECURITY"))' "$AUDIT"

# 5. Inspect the exact shard covering a file
rg -n '"path":"b70/agent-eval/run.py"' "$AUDIT" | head -n 5
```

---

## 3. Chapters by Subsystem (`by-area/`)

| Chapter | Shards | Slices | Lines | Findings (`C / H / M / L`) | Cosplay | Removable LoC |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: |
| [`by-area/atlas-root-and-gpu-notes.md`](by-area/atlas-root-and-gpu-notes.md) — Atlas Root Build, Checkers & GPU Kernel Notes (`build.py`, `tools/`, `b70/`, `rtx-6000/`) | 2 | 28 | 1,540 | `0 / 3 / 12 / 2` (22) | 3 | 255 |
| [`by-area/recipe-maker-core.md`](by-area/recipe-maker-core.md) — Recipe Maker Analytical Roofline & Planner (`recipe-maker/src/`, `report.js`) | 6 | 18 | 12,487 | `1 / 14 / 38 / 7` (63) | 10 | 1,207 |
| [`by-area/recipe-maker-probes-and-tools.md`](by-area/recipe-maker-probes-and-tools.md) — Recipe Maker Probes & Ledger Tooling (`recipe-maker/probe/`, `recipe-maker/tools/`) | 2 | 19 | 4,295 | `0 / 5 / 14 / 0` (20) | 2 | 317 |

---

## 4. Complete Shard Index (`10` Shards)

| Shard ID | Chapter | Scope / Domain | Files | Lines | Findings (`C/H/M/L`) |
| :--- | :--- | :--- | ---: | ---: | :--- |
| [`s177_homelab-atlas`](by-area/atlas-root-and-gpu-notes.md#s177_homelab-atlas) | `atlas-root-and-gpu-notes` | `.` | 21 | 1,124 | `0/1/6/0` (11) |
| [`s178_homelab-atlas__recipe-maker`](by-area/recipe-maker-core.md#s178_homelab-atlas__recipe-maker) | `recipe-maker-core` | `recipe-maker` | 1 | 2,100 | `0/3/6/1` (10) |
| [`s179_homelab-atlas__recipe-maker`](by-area/recipe-maker-core.md#s179_homelab-atlas__recipe-maker) | `recipe-maker-core` | `recipe-maker` | 4 | 1,744 | `0/3/5/3` (11) |
| [`s180_homelab-atlas__recipe-maker__probe`](by-area/recipe-maker-probes-and-tools.md#s180_homelab-atlas__recipe-maker__probe) | `recipe-maker-probes-and-tools` | `recipe-maker/probe` | 7 | 1,813 | `0/2/7/0` (10) |
| [`s181_homelab-atlas__recipe-maker__src`](by-area/recipe-maker-core.md#s181_homelab-atlas__recipe-maker__src) | `recipe-maker-core` | `recipe-maker/src` | 7 | 1,943 | `0/1/8/1` (11) |
| [`s182_homelab-atlas__recipe-maker__src`](by-area/recipe-maker-core.md#s182_homelab-atlas__recipe-maker__src) | `recipe-maker-core` | `recipe-maker/src` | 1 | 2,150 | `0/1/8/2` (11) |
| [`s183_homelab-atlas__recipe-maker__src`](by-area/recipe-maker-core.md#s183_homelab-atlas__recipe-maker__src) | `recipe-maker-core` | `recipe-maker/src` | 1 | 2,150 | `0/2/7/0` (10) |
| [`s184_homelab-atlas__recipe-maker__src`](by-area/recipe-maker-core.md#s184_homelab-atlas__recipe-maker__src) | `recipe-maker-core` | `recipe-maker/src` | 4 | 2,400 | `1/4/4/0` (10) |
| [`s185_homelab-atlas__recipe-maker__tools`](by-area/recipe-maker-probes-and-tools.md#s185_homelab-atlas__recipe-maker__tools) | `recipe-maker-probes-and-tools` | `recipe-maker/tools` | 12 | 2,482 | `0/3/7/0` (10) |
| [`s186_homelab-atlas`](by-area/atlas-root-and-gpu-notes.md#s186_homelab-atlas) | `atlas-root-and-gpu-notes` | `.` | 7 | 416 | `0/2/6/2` (11) |
