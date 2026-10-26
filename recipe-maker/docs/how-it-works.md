# How Recipe Maker works

The reference for the planner: how it tags every number, the four parts (probe, model intake, run ledger, accuracy
report), the pre-launch gate, how to add a box, a model or a recipe, the block-diffusion and route-trim models, the
upgrade advisor, and where it is still weak. Start with the [README](../README.md).

## Which numbers to trust: `evidence`

Every number carries one trust field, `evidence`, with three values:

| evidence | meaning | tags |
|---|---|---|
| **seen** | our own measurement: a run, probe or bench on our boxes, or a read of our own file (a GGUF header, a live kit setting) | MEASURED (also bounded / confounded), READ of our own file |
| **inferred** | fitted, derived or estimated by us, or taken from a vendor spec, a doc or a community claim | FIT, ESTIMATE, PRIOR, READ (docs, specs, HF configs), INFERRED-BY-US, MEASURED-BY-THEM, M-ext, CLAIMED |
| **unknown** | not modelled: no number is given, or none is trusted | UNKNOWN, MEASURED-UNKNOWN |

`src` says which run, file, spec or URL the number came from. A seen value whose `src` names a run-ledger row also lists
it in `runs`. The tag keeps the finer kind: a FIT is inferred, but it was fitted on seen points. One table in `build.py`
(`EVIDENCE`) maps tags to classes. The build stops on a tag with no class, or on a `src` whose first word ("MEASURED",
"ESTIMATE", "UNKNOWN") contradicts its tag. `data.json` `meta.evidence.counts` totals the catalog.

`--check` prints one `trust` line under each predicted number (decode, decode without the draft, prefill, VRAM, node
MemFree). Each line counts the terms in each class, lists every UNKNOWN term, and names every INFERRED term with its source:

```
trust   decode 11.34 tok/s: 5 seen, 6 inferred, 0 unknown terms; rests on INFERRED terms: host overhead ms/token 36.69
        (FIT: ...); GPU nameplate bandwidth GB/s 672 (READ: NVIDIA spec sheet); ...; draft speed-up 1.20 (FIT: ...);
        external check (nvidia_consumer/hybrid, 4 community rows): our physics is off by median 17 % (bias -17 %) there
```

A line ends "every term SEEN" only when nothing in it is inferred or unknown. A decode line also carries the external gap
for the box's hardware class (below).

## Accidental fallback checks

The existing recipe gate also accepts `fallback_plan`, and the existing run writer accepts
`--from-log placement:<local-retained-log>`. There is no second solver or run ledger. A plan's sourced `classes`
name intended paths for dense/attention, hot/cold experts, KV, drafter and host pins, plus dense/shared/HC kernels,
batched execution, drafter state and batch-MTP. Memory follows each box's engine-supported tiers; compute follows
operation-specific sourced `capabilities`, not the presence of an XMX-capable GPU. `unknown` is an explicit unproved
path. An intentionally lower tier has a sourced `decision.reason`; that does not excuse a later downgrade below it.

A lower-than-capacity failure requires a sourced `unit_gib` and `headroom[tier]` certificate: `live_verified: true`,
`idle_gib` tagged READ/MEASURED with canonical seen provenance (vendor/documentation READ is inferred), a source, and
non-UNKNOWN `accounting` sources for KV, staging, services, safety, CMA, GTT and
planned fills. This is allocatable residual idle AFTER all those fills, not total RAM or raw MemAvailable. The
certificate's `fills` must exactly match all memory classes as `{class: {tier, gib}}` (unknown GiB is null), so a
changed placement invalidates old headroom. DRAM idle must also respect `hostPinnedBudget` containment. No certificate
means UNKNOWN, never a recommendation to fill RAM. A physically absent PMem tier is never invented. Engine-default
hazards are sourced catalog controls with model/build applicability, not flags copied between engines.

The parser stores line-cited events in `measured.fallback_ledger`. The normal writer binds their box/model/engine/build
identity only when recording a captured launch; a post-hoc parse remains unbound unless independent retained identity
is supplied. Repeated placement logs concatenate, never silently replace prior events. Recipe checks, the existing
hard-check table and `--accuracy` consume these receipts; historical rows and launch verdicts are not rewritten.
`--accuracy` exposes `launch_fallback` separately from `post_run_fallback`, and prints a separate post-run receipt
section: an UNKNOWN at launch must not hide a later observed FAIL. Today's table projects maintained annotations only
onto matching box/model/engine/recipe identities, skips a conflicting known build, and never replaces a captured plan.
The projection is labelled post-hoc in the row and details; a missing historical build stays missing and build-sensitive
checks stay UNKNOWN. The existing expert-RAM and fallback columns expose those results rather than dropping them.

Each path has its own result. Explicit lower placement, cache slot loss, RAM KV reads beyond a declared limit, read-loaned
expert slots or excess GTT pins FAIL against the plan. Missing paths remain UNKNOWN even if another path passes. Dense
XMX setup reports only its row threshold; it is not proof that the workload executed that path, and says nothing about
SHEXP/HC. An offered draft proves the drafter ran; zero offers or absence of a warning does not prove off. Cache-hit
counters exclude PCIe/remote offload and cannot prove all experts resident. KV counters are cumulative and cannot be
joined greedily to individual requests. No throughput result enables a quality-pending optimization.

Prior art and decision: `context` owns tiers/fills, `checkRecipe` owns verdicts, `record_run.py` owns the append-only
run and parser, `runs_tables.observed` owns table projection. Extend those seams rather than add a renderer/system.
One measurement hop, no daemon or recurring work: parsing is linear in receipt lines; gate work is bounded by declared
classes and those events. At rest there are zero processes or probes; ten times the lines costs ten times parsing work,
not ten times process starts. Reboots/upgrades invalidate live headroom and unmatched build receipts until re-qualified.

## Expert residency and measured lever limits

A recipe may declare `expert_residency`. The normal `--check` consumer then checks host experts against
`min(hostPinnedBudget.ceiling_gib, pre-load MemAvailable - explicit additional reserve_gib) - host KV - staging/other`.
The two reserve domains are independent. The absolute total containment ceiling keeps the existing
`hostPinnedBudget` law: total RAM minus household floor and pinned margin, or total RAM minus an explicit
`serving.host_reserve_gib` replacing that combined reserve. Its semantics do not change.

`available_gib` means free plus reclaimable memory after services, before loading this model. The explicitly sourced
`expert_residency.reserve_gib` is additional headroom against that measurement, including any allocatability discount
it did not already exclude. It is not a replacement for the absolute total reserve. A household floor may combine
current allocations with unavailable CMA and future protection; neither subtract that whole floor again from
MemAvailable nor guess that only the pinned margin remains. Without explicit additional headroom the result is UNKNOWN.

`other_gib` includes future host shadows, staging and non-expert allocations not already charged by the pre-load
measurement; do not charge existing services again. Values are nonnegative GiB with sources. Host KV reuses
`kvHostPlan` unless `kv_gib` is explicit. A negative remaining budget stays negative, so zero host experts cannot hide
an existing KV, staging or reserve deficit. Missing or historical inputs remain UNKNOWN. Dated inventory capacity and
a mapped or locked file are not current residency proof; source-only annotations keep `live_verified` false.
Even a declared live budget is an input, not a host probe.

Only a fully declared fit can flag avoidable SSD/USB expert streaming. Engine controls are catalog entries, scoped to
matching model/build when required: dsv41 RAM-copy flags do not apply to XeStrata. Non-expert on-file lookup tables remain
separate. A `ram_tier_gib` capacity is not an expert-byte measurement. GLM, Flash-Next and V4.1 source annotations leave
current live RAM/residency unverified until a passive audit supplies them.

A whole-request rate from another engine is not an identified CPU expert-kernel rate. Recipes can retain
`cpu_expert_evidence` in their prediction output as reference-only, with its model/quant/host/build/workload scope.
The V4.1 RAM-copy observations do not alter the CPU factor, other models, or the old ik prediction.

The existing window attempt graph retains controls, losses and quality-pending bundles. XMX dense/shared and HC paths
need concurrent-row quality coverage; serial PPL/LP cannot clear batch-only arithmetic. The batch-MTP crossover is
inferred sensitivity, not a measured threshold. Nothing in an attempt enables a serving flag.

`data/curves.json` contention annotations keep client TTFT, native processing and per-arrival wait correlation separate.
The curve report prints these even without a fit. CHG2's invalid polled queue/read times and unmatched native phases never
train the window law; a requeued request needs a stable identity join, not a nearest-time or prompt-length match.

## External validation: `runs/external.jsonl`

Each row is a published measurement on hardware we do not own, such as a llama.cpp scoreboard, a GitHub issue, an
r/LocalLLaMA post or a blog. It records the URL, the date (and what the date is), hardware with spec bandwidths, engine and
build, model parameters, quant, file size, placement (`all_gpu`, `cpu_only`, `experts_cpu`, `n_cpu_moe:N`, `ngl:N`),
settings, and the measured decode and prefill. The settings use the conditions fields (`ctx`, `kv_type`, `draft_n`,
`sampling`, `reasoning_effort`, `workload`). The build adds `missing`, the fields the source did not declare. Every row is
MEASURED-BY-THEM, so inferred.

`node report.js --external` predicts each row from its hardware spec using only our terms. Those terms are GPU nameplate x
the reach of the backend's measured line on that vendor's cards (`bench.gpu`; else `assume.gpu_eta`), the per-token MoE
term per GPU MoE layer, DRAM spec of one NUMA node x the spec-to-read ratio of our own boxes x the measured placement gain
(`bench.numa`), the CPU kernel table x the median fitted `f`, and the line's host term (else a borrowed one). The report prints each gap and the median gap per hardware class and placement. These rows are a
validation set: they **never set a term**. A prediction on one of our boxes prints its class gap, which lowers trust where our
physics is known to miss. Rows the physics cannot map are UNMODELLED, with the reason. Examples are a missing DRAM spec, a
draft run, or a placement it cannot express.

## The four parts

The tool has four parts: a hardware probe, a model intake, a run ledger and an accuracy report.

### 1. Probe

`probe/probe_box.py` reads a box without installing or writing anything on it. It runs `probe/collect.sh` locally, or
as one `ssh HOST bash -s` call. It reads:

- the CPU, its ISA flags and threads
- each NUMA node and its free memory
- memory-only NUMA nodes (Optane PMem or CXL)
- DIMMs, with `sudo -n` only
- GPUs and their PCIe links
- every mount and the kind of disk beneath it (rotational if any member is)
- the mount that holds the home directory
- swap, swappiness, NUMA balancing and THP

The output is `data/boxes/NAME.json`. `build.py` lays it over the hand table for that box (rules in
`probe/overlay.py`). `--diff data.json` prints every place where the probe and the table disagree.

### 2. Model intake

`probe/model_intake.py` reads a model's layout from headers only, never the weights. It accepts a local GGUF, a GGUF
on another box over ssh, or a Hugging Face repo (config.json plus safetensors or GGUF headers, fetched with HTTP range
requests). It reports:

- layers, experts, active experts and the bytes per expert
- the bytes of each tensor class (expert gate/up/down, shared experts, attention, embeddings, MTP, n-gram tables, ...)
  split by quant type

The output is `data/models/KEY.json`.

### 3. Run ledger

The ledger holds one row per real run. New rows are appended to `runs/runs.jsonl`; older rows are sealed into
segments `runs/runs.0001.jsonl.gz`, `runs/runs.0002.jsonl.gz`, ... once the open file passes 512 KiB (`src/ledger.py`;
settings `MAX_OPEN_BYTES`, `KEEP_DAYS`), so no file in the repo passes 1 MiB. A public copy seals into plain-text
segments of at most 512 KiB instead (`runs.0001.jsonl`, ...; `python3 src/ledger.py pack --plain`). Every reader takes the segments in order and
then the open file. Each row has three parts:

- the exact recipe the gate checked
- the gate's `--check --json` output captured at launch, including its prediction
- what the run measured, with the log and line range behind every number

`tools/record_run.py` writes rows and parses run logs (run-kit logs, bench.py output, decode-KL results).
`tools/kit_record.sh RECIPE.json LOG...` is the wrapper a launcher calls after a run. Rows recorded after the fact are
marked `post-hoc`: their prediction is computed with today's planner, not taken at launch.

### 4. Predicted vs measured

`node report.js --accuracy` replays the ledger and reports:

- the decode-speed error per box, model and engine, scored only on runs that went fine
- which gate checks caught the runs that went bad (true positives)
- which runs went bad with no check failing (misses), which are the next checks to add
- which runs a check refused but which ran fine (false alarms)
- which true positives are cause-matched: `runs/labels.json` labels a run's cause from its notes and logs (never
  guessed), and `causes` in `build.py` names the checks that decide each cause before launch. Our own copy also has
  `tools/propose_causes.py`, which asks a local classifier to PROPOSE a cause for each unlabelled bad run for a person
  to confirm (0.64 accuracy, macro F1 0.54 against a 0.32 / 0.05 majority baseline, 28 rows, 2026-10-19); it needs our
  classifier service, so the public copy leaves it out
- runs disturbed by interference (a label names the phases) count as went bad, so they leave the error stats
- checks added after a launch capture (`accuracy.replay_checks`) are replayed post hoc on launch rows, labelled, and the
  report lists the misses they turn into true positives
- the trend: those errors and gate counts per time bucket (`--bucket day|week`, default `accuracy.trend_bucket` in
  `build.py`), overall, for launch-time predictions only, and per box/model/engine, each with a verdict on whether the
  gap between prediction and result is shrinking (last bucket below the first and a falling fitted slope)
- the cause taxonomy: each labelled cause of a run that went bad, the checks that decide it (`causes[].decided_by`), their
  verdict on the run's own checks and on today's gate (run as `--check` runs it, with the prediction), and the cause
  classes no pre-launch check covers. A FAIL that appears only today is hindsight when the term was measured from that
  very crash.

**Prefill** has a CPU-path rate only inside a measured cell (`prefill.cpu_tps` in `build.py`), keyed
`box|model|engine|ub`, with the PMem expert GiB of the runs that read it as its envelope (and, for V4.1, a banks-on rate
FIT on three rows with one held out). `-rtr` on a split store takes `checks.rtr.prefill_mult`. Any other `-ub`, PMem
share, banks size, or `-rtr` on a one-chain store prints `prefill UNMODELLED (why)` in the gate line: the ledger has only
one `-ub` per model, so no `-ub` or PMem term can be fitted yet. GPU expert offload keeps its upload-time model.

## The pre-launch gate: `--check`

```
node report.js --check RECIPE.json [--strict] [--json]
```

| exit | verdict | meaning |
|---|---|---|
| 0 | GO | no check FAILs (with `--strict`, nothing is unjudged either) |
| 1 | REFUSE | a hard check FAILs: do not launch |
| 2 | BAD INPUT | unreadable recipe, or an unknown box, model or engine |
| 3 | HOLD | `--strict` only: a check is UNKNOWN, or the planner has no decode or prefill number for the layout (UNMODELLED) |

The recipe passes through these checks in order:

| check | what it tests |
|---|---|
| `inputs` | the box, engine, quant types, row limits and thread count agree with each other |
| `storage` | each tensor class sits on the tier the recipe claims; tensors read while serving must not sit on a rotational disk |
| `vram_peak` | the predicted VRAM peak, not the post-warm-up reading, fits the card with a measured margin |
| `numa_free` | each NUMA node keeps free DRAM after the experts and the engine's load overhead (the overhead depends on the load style) |
| `system` | the settings a box needs during a timed run (droopy: swap off, NUMA balancing 0, no PMem demotion) |
| `rtr` | CPU repack (`-rtr`) is off when the model is split across tiers |
| `quality` | the share of routed reads at 2 bits stays under the KL and top-1 bars, on a line fitted to measured points |
| `conditions` | the serving conditions are declared (below) |
| `kv_host` | an engine with a serving model (XeStrata): with KV streaming (`serving.kv_resident`) every session pins a host-RAM copy of its whole KV capacity, about bytes per cell x (`max_context` x (1 + `big_slots`) + `slot_context` x (`batch` - `big_slots`)) plus the drafters' own layer; FAIL when that leaves less than the engine's reserve (`checks.kv_host.reserve_gib`, STRATA_KV_HOST_RESERVE_GIB, default 16) of MemAvailable (`serving.mem_available_gib`, else the box DRAM less its `host_floor_gib`). Bytes per cell come from `models.<model>.kv_host` by `--kv` format; each copy adds the engine's `copy_overhead_mib` |
| `host_pinned` | a recipe with a `serving` block on an engine with a serving model: every host-pinned buffer (driver GTT, which no cgroup or OOM kill sees) = the KV host copies + the terms of `checks.host_pinned.terms[model/engine]` (the expert complement, the stage-lend duplicate of a layer split unless `serving.stage_lend_drop`, prefill staging, prompt hand-off, embeddings, the rest); FAIL when it passes DRAM total - household floor - pinned margin (`serving.host_reserve_gib` replaces floor + margin). The detail names the levers (`slot_context` near the VRAM ring with `big_slots`, `stage_lend_drop`) |
| `cpu_pool_rows` | where a stage computes experts on the CPU (a layer split on XeStrata, whose second stage has no PCIe share of the experts it misses, or `serving.cpu_experts`), every row of a batch window goes through the CPU expert pool, so ceil(`batch` / `batch_groups`) must not pass the pool's row cap (`serving.cpu_pool_max_rows`, default 16; `engines.strata.serving.cpu_pool_rows_by_build` gives a build's own cap). FAIL past it: 64 agents in 2 groups (32-row windows) did not load on a 16-row build. UNKNOWN when `serving.max_t` lets window-row prefill fill a window past the cap |
| `agent_workload` | a recipe for many agents (`recipe.conditions.workload.agents`, else `serving.batch`, at or above `many_agents_min`) needs a real-agent ledger row of the same engine build. Without one it is flagged (UNKNOWN, so `--strict` holds it): its numbers are decode capacity from a synthetic short-prompt load; it still carries the binding term with a predicted prefill rate (MODEL, below) where the box and store have a prefill pass model. With one it PASSes and names the binding term (below, "Agent workload") |
| `cpu_reserve` | a CPU-decode recipe leaves the box's reserved CPUs free of decode threads (`cpu.reserve`, default `checks.cpu_reserve.default_reserve`); UNKNOWN until `recipe.conditions` declares `threads` and `cpu_reserve` as numbers |

`--json` prints the checks, the notes, the prediction and the resolved facts. It also carries back the recipe fields the
planner keys on (`engine_build`, `banks_experts`, `ram_tier_gib`), so a ledger row read later shows what was predicted.
Run the check on the box itself if you want symlinks resolved; anywhere else, paths are taken as given and `~/` resolves
through the box's probed home mount.

A launcher saves that output beside the recipe as `RECIPE.json.check.json` before it starts the run, and calls
`tools/kit_record.sh` when the run ends.

## Screening a model: card conditions and `verdict-ready`

A quality round (planted defects, clean controls) judges a model only under the conditions its maker measured it under.
`data/cards/KEY.json` holds those per model: the recommended sampling (temperature, top_p, top_k, min_p, presence
penalty), the thinking mode with its control and effort, the chat template (sha256 of the template text) and reasoning
parser, the context and the max output. Every value is `{v, tag, src}`: `READ` with the URL and line it was read from, or
`UNKNOWN` with `v` null when the card does not say. The build refuses a value with no URL.

The public copy ships the recorder and the model cards, but not `tools/screen_sources.json` or `runs/screen.jsonl`:
bring your own source specification and quality rounds. A missing screen ledger is not a passed screen.

`tools/record_screen.py --spec tools/screen_sources.json --write` reads a round into `runs/screen.jsonl`, one row per
model arm: the settings the arm actually ran with (the request body over the server defaults from `/props`, the context
from `server.log`, the template the server ran), and for every request its failure shapes: a cap hit (`finish` length),
an empty answer, a loop (a line repeated in the output tail; `LOOP_*` settings, kept in the row), template tags in the
visible answer; plus the items answered only on retry, the grades and the verdict the round gave.

```
node report.js --screen [--json]
```

The `verdict_ready` check (the Trust tab shows it beside the other hard checks) refuses a keep or delete verdict when:

- more than `checks.verdict_ready.shape_rate_max` (default 20 %) of the requests end in a failure shape; or
- a setting departs from the card (`checks.verdict_ready.fields`: same value for sampling, thinking and template; at
  least the card max output; a context that holds the longest prompt plus the card max output) and no recorded
  best-config sweep (a `purpose: sweep` row naming that value in `sweep.best`) chose it.

It prints each departure with the source of the value the run used. An unknown card value makes it UNKNOWN, not PASS.

## Adding a box

1. Probe it with `python3 probe/probe_box.py --ssh HOST --name NAME --write data/boxes/NAME.json`.
2. Add a hand entry for NAME to `boxes` in `build.py`, using an existing box as the template. A probe file with no hand
   entry is skipped, because the planner needs fields a probe cannot measure: DRAM and PMem read bandwidth, the usable
   DRAM budget, and the CPU efficiency factor. Until you measure those, tag them ESTIMATE.
3. Run `python3 build.py` and check the overlay lines it prints.
4. To see where the probe and the hand table disagree:
   - build without overlays: `FITVIZ_OVERLAYS=0 python3 build.py`
   - run `python3 probe/probe_box.py --ssh HOST --name NAME --diff data.json`
   - build again normally

## Adding a model

Read the model with `python3 probe/model_intake.py --hf ORG/REPO --name KEY --write data/models/KEY.json`, or point it at
a local GGUF. A new model family goes in its own file, `families/KEY.py`: `build.py` runs every file there after its
built-in models, so one file holds the family's header read, KLD priors, `models[KEY]` entry and quality-gate family, and
removing the file removes the family. The entry needs the fields a header cannot give:

- expert heat (how often each expert is routed, from a routing trace)
- the per-type KL table
- the engines that can run the model

## Adding a recipe

A recipe is a JSON file in `recipes/`. The fields are:

- `id`, `label`, `box`, `model`, `engine`, and `engine_build` where the engine has calibrated builds
- `store`: the quant types and the share of routed reads per type
- `threads`, `rtr`, `draft`, and `knobs` (box changes such as freeing a GPU)
- `serving` (engines with a serving model, XeStrata): the engine's batch and KV flags by name: `max_context` (default:
  the recipe `ctx`), `batch`, `batch_groups`, `slot_context`, `big_slots`, `kv_resident`, `kv`, `batch_mtp`, `mtp_window`;
  each left out takes the engine's own default from `engines.<engine>.serving.defaults`. `kv_host_reserve_gib` sets the
  engine's STRATA_KV_HOST_RESERVE_GIB and `mem_available_gib` (with `mem_available_src`) the MemAvailable the engine
  starts with; the `kv_host` check reads them. `stage_lend_drop` (STRATA_STAGE_LEND_DROP), `host_experts_gib` (the
  experts this placement keeps off the cards) and `host_reserve_gib` feed the `host_pinned` check
- `files`: each with a `path`, what it `holds`, and how it is accessed (`mmap`, `loaded`, `on_file`)
- per-node expert placement for NUMA boxes, and `banks_gib` and `banks_experts` for VRAM expert caches
- `conditions`: the serving conditions the numbers depend on: `ctx`, `kv_type`, `draft_n` (with a draft), `sampling`,
  `reasoning_effort` and who set it, and `workload` (prompt kind, cached depth, prompt and generation lengths). The
  `conditions` check is UNKNOWN until each one is declared. Two runs that differ in an undeclared condition do not
  compare. An agent eval that hits its wall limit measures speed as much as capability, so record the limit beside the
  score.

Start from the recipe closest to your run. Check it with `node report.js --check recipes/NEW.json --strict` and fix every
FAIL. Then decide whether each UNKNOWN is acceptable for your run.

Ledger rows are never rewritten. A run found confounded after it was recorded (its phases are no clean measure of the
recipe: identical switches decoded different texts, for example), or a field the recorder got wrong, goes into
`runs/run_caveats.json` by run id: `kind: "confounded"` with `why` and `superseded_by`, and `correct` with a field's value
and source. Both ledger readers (`src/ledger.py`, `src/ledger.js`) apply it. A confounded run stays in every run list with
its caveat. It drops out of the fastest-run tables, the host, hop and draft fits, and the accuracy scores.

## dLLM (block-diffusion) recipes

A recipe with `"kind": "dllm"` describes one `llama-dllm-gen` run (LLaDA2.x on a fork of llama.cpp with block-diffusion decode):
`box`, `model` (a `data/models` intake), `backend` (`sycl` / `cpu`), `moe_path` (`stock`, `per-pair`, `grouped`), `cards`,
`quant`, `schedule` (`name` + `mode`: `cache`, `fuse`, `spec`, `full`, `ar`), `block`, `workload`, `placement` (`all_gpu`,
`cpu_only`, `n_cpu_moe:N`) and the usual `conditions`. Decode is TPF x 1000 / ms per forward:

- TPF and B (tokens submitted per forward) are measured per model | schedule | block | workload cell from the ledger.
- ms per forward = host + non-expert bytes / line + expert bytes x coverage(B) / (line x kernel gap). The line is the box's
  measured backend line; coverage(B) = 1 - (1 - k/E)^(rho B) with the measured routing locality rho (`dllm.routing`, from
  `runs/dllm-raw/coverage-2026-10-19.jsonl`); the kernel gap is fitted per box | backend |
  kernel path from the clean cache/fuse rows (settings `dllm.*` in build.py).
- `--check` prints the gate (inputs, weights per card, conditions) and the `decode` and `ms per forward` trust lines.

Record runs with `tools/record_dllm.py --from odie=DIR` (catalog `tools/dllm_catalog.json` maps GGUFs, stages, cards and
workloads); it writes ledger rows tagged `dllm` and stripped copies of the source files to `runs/dllm-raw/`. NOTES "dLLM"
holds the fitted tables, the held-out run (`dllm.holdout`) and leave-one-out.

## Layer map and route trim

`data/layer-maps.json` holds a per-model, per-layer sensitivity map. Each band names model layer ids with a value and its
provenance (CLAIMED with a url, MEASURED, INFERRED-BY-US) or `fit: true`. A layer in no band is UNKNOWN: it weighs as
the most sensitive known layer, and a trim on it has no KL. `layerMap` (src/model.js) fits a scale on the CLAIMED values
plus one coefficient per fitted band, by least squares on our measured KL points (the quality family's points with run
ids). The fit gives per-layer weights (mean 1). The quality check then reads the map-weighted 2-bit share and labels it
`PREDICTED (layer map, N calibration points)`. P3's whole-middle-layer cut predicts a FAIL. 15 middle layers at 2 bits
FAIL, while the last 15 pass. A map that cannot calibrate (too few points, or a coefficient <= 0) falls back to the plain
share line.

A recipe may state `trim: [{layers: [lo, hi], keep: n}]`: keep the top n of the k routed picks that miss VRAM. The
prediction drops those reads, and the KL is per layer map value x (k - n) / (k - ref_keep) x miss share / ref_miss_share
x transfer, all INFERRED. The route-trim proposer (`src/advisor.js proposeTrim`) starts from each box's best
quality-passing run. It trims the least sensitive layers first, down to `settings.trim_min_keep`, while KL and top-1 stay
inside the bars by the map's largest residual (`settings.plan_margin`). On snoopy GLM it proposes keep 4 on layers 27-44
plus keep 7 on 26. Engine support is per engine (glmlite: needs code).

## Upgrade advisor

`data/upgrades.json` holds the settings and the part catalog: each part has dated used prices and a bracket. For each
box, `upgradeAdvice` takes the best quality-passing run for each model (excluding `exclude_tags`). It probes each lane
(VRAM, DRAM and PMem bandwidth and capacity, NVMe, CPU kernel rate) at `probe_scale`, plus the read-ceiling + CPU pairs
that co-bind. The lane with the largest gain is the binding lane. A part is SUGGESTED only when it serves the binding
lane, its low-end gain is >= `suggest_min_gain` and the run passes quality. A binding lane with no priced part is
reported as UNPRICED. `node report.js --advice [--json]` prints the advice. The full and `--quick` reports carry it too,
in `layer_map`, `trim_proposals` and `upgrades`. build.py stores the same in data.json
(`layer_maps.calibration`, `layer_maps.trim_proposals`, `upgrade.advice`).

## Batch window: what a kernel is worth before you write it

`node report.js --batch` fits a batched-decode step to measured tok/s. The batch window term builds the step from its
kernels instead, so it can answer "if this kernel went from 7x to 1.2x its bandwidth floor, what would 32 agents get?".
One config per `data/window/*.json`. Its inputs:

| Input | What | Where it comes from |
| --- | --- | --- |
| spec | the model's dims and each layer's tensor types; a layer's kind (e.g. GDN or QSA) by the config's `kind_rule` | XeStrata `tools/batch_window_spec.py SHARD1.gguf [--layers A-B]`; layers the spec does not cover take their kind from the model's layer list in `data/models/` |
| ubench | per kernel class: ms summed over the measured layers at each row count, bytes and floor at the floor's rows, x floor; the TOTAL lines (random routing, rows that share experts); the method floor; the copy-kernel GB/s | the card's batch-window microbenchmark log (its `STAGE ROLL-UP`), parsed by `probe/window_inputs.py` |
| layout | stages (contiguous layer ranges, one card each), row groups, the engine's caps | the engine flags of the measured runs |
| points | measured stage ms (and tok/s, from the ledger row when one exists) at known agent counts | the engine's stage timer |
| profile (optional) | one point's window, ms per term per stage (a term = a few kernels of one layer kind, or a stage's own work such as the head) | the engine's own per-stage profile (XeStrata `STRATA_VERIFY_PROFILE=1`) |
| variants (optional) | a rewritten kernel's measured time against the current one | the kernel lane's bench |

The config's fields: `spec` (`file`, `kind_rule`: the tensor whose presence marks a layer's kind), `layers_beyond_spec`
(the model-list field for layers the spec does not cover), `ubench` (`file`, `card`: the box and GPU id it ran on,
`routing_phi` and its range), `stages` (`layers: [first, last]` each), `groups`, `max_agents`, `max_window_rows`,
`min_batched_agents`, `extras` (`id`, `stage`: first or last, `per`: row or fixed; `kernel_scale_alt` marks the one the
gamma-free refit drops), `settings` (`gamma`, `gamma_range`, `omega_max_us`, `omega_step_us`, `graph_launch_us`,
`lever_xfloor`, `lever_xfloor_grid`, `extra_lever_share`, `plan_agents`, `agents_grid`, `whatif_stages`) and `points`
(`id`, `use`: fit, check or law, `agents`, `stage_ms`, `split` when it ran on another layout than `stages`, and `tps`
or a `ledger` row id and its `aggregate_tps` key). The build checks every split covers the layers in order. The
optional `profile` has `point` (the point it was taken at), `split` (its layout, when not the stages'), `curve_anchor`
(the point whose kernels it measured, for its row curve), `totals_ms`, `rows_basis`, `gap` (`label`, `rows_basis`),
`checks` (point ids) and `terms`. Each term has `kind` or `stage` (first/last), `name`, `ms` (one per stage), `classes`
(`[ubench class, share]` pairs; a share below 1 where a class splits over two terms), `lever` (the terms one lever moves
together) and optionally `rows_basis` and `bw` (false for a wait). `variants` have `class`, `old_us`, `new_us` and
`launches` (per layer). The build checks every class name, kind and stage count. Every number is `{v, tag, src}`.

The build freezes effective built data, actual training/check membership and evaluator source hashes in
`data.meta.calibration`. This is experiment identity, not proof that two measurements match. Prospective recipes
predeclare exact alternatives, workload, isolation, objective and quality/memory constraints through `validation`;
`--check` verifies the loaded effective-data digest before `applyCal` mutates it, and hashes the actual local files
named in that manifest before capturing immutable candidate predictions. One cached verification per CLI invocation
passes the original stored JSON to the build's hash-only Python mode (`python3 -I build.py --calibration-data-sha256`),
which shares the build's exact exclusions and canonicalization without importing local inputs or writing outputs.
JS-reserialized numeric spellings are not assumed equivalent to Python's. Stale effective data or stale/missing
evaluator bytes fail capture with a rebuild instruction: run `python3 build.py`, then redeclare the cohort against the
rebuilt identity. Ordinary nonvalidation checks retain their existing behavior. The recorder
preserves the launch capture, and `--accuracy` scores only complete qualified cohorts; historical, incomplete,
confounded or failed observations remain UNKNOWN for decision rank/regret.

Single-seat decision validation uses per-stream decode (named default target 25 tok/s), plus declared quality and
memory constraints. Batch-fleet uses total accepted tok/s per measured all-participant CPU+GPU watt and a measured
per-agent floor (named default 10 tok/s). Current real predictors lack qualified `accepted_tps`, `gpu_cpu_watts` and
`min_agent_tps`, so real batch decision scoring remains UNKNOWN even with complete measured receipts. Synthetic
scoring fixtures test the contract, not real batch predictor coverage. Missing power is not zero, and aggregate
throughput divided by agent count is not a prediction of the slowest agent. TTFT is diagnostic unless declared.

The model, all in `src/model.js` (`win*`):

- A class runs in every layer of its kind (a class whose layer count is the measured total runs in every layer), at the
  measured per-layer average. Its in-graph time = gamma x max(launch floor, isolated ms - layers x launches x omega).
- A stage = the sum over its layers + the unmeasured extras of the first and last stage (fixed or per row; the config
  lists them).
- G row groups take turns through S stages: tok/s = eta x rows x 1000 x min(G / sum of stage ms, 1 / slowest stage ms),
  rows = agents / G.
- `winFit` fits omega (grid + golden section) and the extras (non-negative least squares on the `fit` points' stage ms,
  relative error). Eta is the median measured throughput / pipeline law over only `fit` points with finite, positive
  throughput and stage times, retaining only finite, positive ratios. Its contributing ids are frozen as `eta_train`.
  With no such training operand eta, predicted throughput and throughput ranking are UNKNOWN, not efficiency one.
  `check` points never fit eta or stage terms; `law` points (kernels the ubench did not measure) score only the pipeline
  law. The report also refits at both ends of the routing range and with gamma free (the reading where the per-row cost
  the kernels miss lives in them, so every kernel lever pays more).
- A lever takes one class to k x its floor (the ubench's own ratio; never under `graph_launch_us` a launch) and scales
  its whole row curve by the same factor. The table ranks by tok/s at the plan's agents on today's split, and also prints
  each lever with the split rebalanced (the least slowest stage over contiguous splits). The rebalance is a lever of its
  own; each unmeasured extra is priced at `extra_lever_share` off, so it gets measured.
- With a `profile`, the profile is the source (`--source ubench` turns it off). A layer's cost is the sum of its kind's
  terms on the stage it was profiled on, split evenly over that stage's layers of the kind. Stage terms go on their
  stage position. The gap (the point's measured stage run minus the term sum: launch gaps, syncs, work the profile does
  not time) is a term of its own, spread over the stage's layers. At other row counts each term follows its
  `rows_basis`: `stage` (the calibrated ubench model's stage curve, anchored when the report anchors), `classes` (its
  mapped classes' in-graph curve, noisy below 4 rows), `row` or `fixed`. The profile's `checks` score this at other
  agent counts and on the run without profiling. The profile point and curve/plan anchors condition this reading and
  are listed separately in the frozen split; they are not untouched end-to-end checks, even when a conditioning point
  has `use: check`. A lever takes each term of a group to k x its mapped classes' floor (the
  `x floor now` column is the profiled time over that floor; at or under 1, the lever pays nothing). Groups without a
  floor, and the gap, are priced at `extra_lever_share` off. A variant subtracts (old - new) x launches per layer from
  the terms that map its class, scaled along the class's row curve.
- The ubench levers stay in the table as variants too (the class at new/old), and the report prints the ubench reading
  of the levers next to the profile's.
- Evidence, frontier and proposals (optional config fields `attempts`, `frontier`, `gate`, `proposals`). An attempt
  has a mechanism key, an action, the ubench classes it acts on (`*` = every kernel, `layout: true` = the split), a
  parent, the attempt it `repairs`, an outcome (win, loss, neutral, mixed, pending, control) and a scope (isolated or
  bundle_only). Engine attempts read their tok/s (MEASURED) and their sanity line (READ) from their ledger row. Kernel-lane
  attempts carry their result in words. A lever lists every attempt on its classes, each loss with the attempt that
  repaired it (`tried` column, `attempts` block). The frontier ranking applies each lever's modelled gain on its
  rebalanced split to the frontier's measured tok/s, and flags a gain under the frontier's noise. Nothing is accepted
  before the `gate` steps pass. A proposal (mechanism key and text, action, classes, reuse or exploration, the cheapest
  test, a falsifier) gets its expected effect from the model. It is a duplicate when its mechanism key and action
  (tokens in any order) match an attempt or an earlier proposal, and related when only the key matches. Related
  proposals are never merged. `--propose LEVER` prints a blank one for a lever. The design follows the RSI-top review's
  harvest list (H1 evidence subgraph, H4/H5 proposal and dedup, H8 frontier value with a correctness gate).
- What-ifs: every GPU in `data/boxes` (kernel time x reference GB/s / that card's GB/s, efficiency carried over, extras
  unchanged), and 1..n stages (rebalanced, as many row groups as the config's ratio, hand-offs between extra stages and
  VRAM not counted).

Limits: the ubench measures one card and the layers the spec covers; the other layers take the per-layer average of
their kind (their tensor types may differ). The extras are fitted, so with as many fit stage times as free constants the
fit is exact in sample and only `check` points say how good it is. Attention classes are taken as measured at every
depth (the ubench's context is printed on its device line). A profile prices today's kernels as they run, but each
term's floor still comes from the ubench's bytes. A kernel rewritten since the ubench (a new form with other bytes)
can show at or under 1x its floor, and then the floor understates what it could still save. The pipeline law takes the
slowest stage's mean as the cycle. With the stages level (split 25: within about 1 ms), each window waits for the slower
of two jittering stages, so the law overpredicts by 2 % (V4 at split 25) to 6 % (the all-on image). Moving a layer to
another card also moves the expert cache each card holds, which the profile does not price. Its checks at split 25 are
off by up to 4 % on stage 0 at two agents, and the rebalance lever reads high for both reasons (split 25 measured
+0.3 % where the model said +3.6 %). The term-to-class mapping
is the config's reading of the engine's term names.

## Agents vs speed: `--curve`

These curves come from a synthetic load: each agent reads one short prompt once, then only its writing is timed. They
measure decode capacity, the most an engine can write, not what working coding agents get. Real coding agents read 4
to 5 new prompt tokens for each token they write, so prompt reading sets their speed: on odie's two B70s, 16 replayed
real agents got 79.6-86.0 tok/s in total (the build served on 2026-10-26, mean 83.4 over three runs) and one agent
alone 32.6. [`--workload`](#agent-workload---workload) models that.

A batched engine splits its agents into G groups (`--batch-groups`) that take turns through S pipeline stages (one per
card, from the layer split). One stage window for r agents in a group takes t(r) = a + b x r ms: a fixed part (launches,
the dense weights read once) plus a part per agent. With G = min(n, groups) groups in use, a cycle on the slowest stage
takes max(G, S) windows and yields one token per agent, so

    total(n) = eff x 1000 x n / (max(G, S) x t(n / G))

Groups of uneven size give the same sum, because t is linear in r. The fit uses only the measured (n, total) pairs:
n / total = A x max(G, S) + B x max(G, S) x n / G, with A = a / (1000 eff) and B = b / (1000 eff), fitted by least
squares. Two agent counts fix both constants. More counts give a residual, flagged when it passes `fit_warn_pct`. When a run
note prints its stage times (`stage run n=32 45.87/45.03 ms`, slowest stage = the window), the report also gives eff for
each point and the line a + b x r through the windows.

`build.py` pools ledger rows by box, model, engine, build, engine flags and context. It reads the groups, stages, slots
and serving fields (`--kv-resident`, `--kv`, `--slot-context`, `MAX_CONTEXT`, `STRATA_STAGE_LEND_DROP`) from the flags,
and from the run's isolation and KV notes when the flags omit them. Some points stay out of the fit and are marked:

- n = 1 with a drafter and batch MTP off, which runs the single-stream path
- counts above `--batch`, where agents queue
- caveated rows

The readouts:

- the most agents with each at `min_each_tps` or more, and the total there (the headline)
- the same at `second_each_tps`
- the total the setup approaches with unlimited agents: G / (max(G, S) x B)
- the host-memory ceiling: the largest slot count whose host-pinned set (the `--check` host_pinned check, with
  `--batch` set to that count) still fits. A layer split that keeps its stage-lend duplicate also shows the ceiling
  with it dropped.
- tok/s per watt at each count whose run logged `measured.power_w`, either one number or `{cN: W}`, with a straight
  line between logged counts. The watts are whatever that run measured (its row states the boundary, for example the
  cards plus the CPU package without the rest of the box); they are not wall power unless the row says so

A headline past the ceiling also gives the best count within it.

Limits:

- The window is linear in r only while the kernels scale smoothly. The profiled run (2026-10-23) shows a step between 4
  and 8 agents per group, so a two-point fit through n = 2 and n = 32 smooths over it.
- Counts past the largest measured one are projections and are drawn dashed.
- Context length, prompt mix and thinking level are whatever the runs used, and they move the window.
- The totals are decode capacity, not agent throughput: `--workload` adds prompt reading (below).

## Agent workload: `--workload`

```
node report.js --workload [RUN_ID | all] [--profile NAME] [--fresh F] [--prefill P] [--n 16,32,64] [--json]
```

The agents-vs-speed curves come from runs that give each agent one short prompt and then time only the writing. That
measures **decode capacity** D: how many tokens per second the agents can write together. Real coding agents do more.
Every turn they send the whole conversation back, and the engine has to read the new part: tool results, file contents,
and any part it dropped from its prompt cache. Call F the fresh prompt tokens read for each token written, and P the
rate the engine reads prompts at (prefill). Then useful output is at most

    min(D, P / F)

An engine that stops every decoding agent while it reads a long prompt runs the two one after the other, and gets

    1 / (F / P + 1 / D)

The smaller of D and P / F is the **binding term**. A setup is prefill-bound when P / F is smaller: adding agents
raises D but not the output, and only a faster prefill (or a smaller F) helps.

`data/workload.json` holds the inputs:

- `profiles`: workload profiles. Each gives F and the shape of a turn: the new prompt tokens per turn, the first
  (cold) prompt, the whole prompt, the output and the tool gap. `coding-agent` is measured from 16 recorded
  coding-agent sessions (923 turns) replayed with their own prompts, outputs and tool gaps: F = 4.88, the server's count
  over the 600 s windows of three replays at 16 agents on the build odie served on 2026-10-26, each after a 600 s
  warm-up (`odie-fnx-iq3xxs-2xb70-real-comp1-no-gdn-ss16-n16-1026`, `-r2`, `-r3`: 4.78, 4.87, 4.98). The replay client,
  counting only its turns inside the window, reads 3.9-4.9 across all 17 such steady-state replays, and the recorded
  sessions need 4.33 with a perfect cache. The profile also gives a median of 963 new prompt tokens per turn (p90
  3,294), cold first prompts of 20.7k-32.8k tokens and whole prompts of 82k (median) to 120k (token counts of the
  recorded sessions). Replays timed from a cold start read 17.8-67.7: every agent's cold first prompt and the re-reads
  after cache evictions fell inside their window, so the warm-up is part of the measurement. `synthetic-short` (F = 0)
  is the bench load behind the curves. `settings.default_profile` picks one; `--profile` and `--fresh` override it.
- `engines.<engine>.reads_stall_decode`: whether a read stops decoding (XeStrata: reads past
  `STRATA_BATCH_PREFILL_MAX_TOKENS` do).
- `engines.<engine>.prefill_levers`: estimated (INFERRED) gains on P from engine changes; a prefill-bound ceiling
  rises by the same factor.
- `settings`: `many_agents_min` (when the `agent_workload` check applies), `build_key_re` (what makes two rows the same
  engine build), `real_kind`, `table_n`, `real_err_warn_pct`.

`build.py` collects every ledger row whose `measured.workload_kind` is `real-agent` (a replay of recorded agent
traffic) into `data.workload.measured`, keyed by agent count like the curve rows: `agent_tps`, `prefill_tps`,
`fresh_per_output`, `prefill_busy_pct`. These rows never feed the decode curves.

For each curve config, the report takes D from its fitted curve, P from the real-agent row of the same build with the
highest output (rows of other builds are listed but never borrowed), and F from the profile. Per agent count it prints
D, P / F, the ceiling, the expected output, the binding term, the predicted prefill busy share and the `cpu_pool_rows`
window check. Every real-agent row at that count is judged again with its own P and F, so a miss is the model's
error, not a workload difference. A miss over `real_err_warn_pct` is marked.

When the build has no real-agent row yet, P is **predicted** from the box and the recipe (`prefillPass`, tag MODEL)
instead of left unknown, so a many-agent recipe names its binding term before its first real-agent run. The
`agent_workload` check still flags the recipe (no real-agent row), but its prediction carries the MODEL P, and
`prediction_first` passes on it when the build has a decode curve. The prefill model (`data/workload.json`
`prefill_pass`):

    t(T) = ceil(T / C) x t_pass + T x m        per read of T fresh tokens, chunk C
    t_pass = sum over stages of  nonresident expert bytes x (1 - cpu_share) / link  +  resident expert bytes / VRAM bandwidth
    m = sum over stages of  layer share x 2 x active params / (eta x card int8 ops)

Each chunk makes one pass over the experts: with C x k much larger than E, nearly every expert of every MoE layer is
routed to, so the experts a card does not hold cross its PCIe link once per chunk (the CPU computes `cpu_share` of
them instead) and the ones it holds are read once from VRAM. The chunk comes from `--prefill N` in the engine flags
(else the engine's `chunk_default`), the split from `layer split N`, the CPU share from
`STRATA_PREFILL_STREAM_CPU_SHARE`, the link from the box probe's PCIe generation and width x `link_eff`. P over an
agent workload is the mean read over (chunks per read x t_pass + mean read x m), with chunks per read taken over the
measured read-size mix (`read_mix`: agent reads are short, median ~780 tokens). `shared_tps` = 1 / (t_pass / C + m) is
what an engine that batched several agents' reads into one pass would approach. eta is the one fitted factor per
engine: for strata it comes from the server prompt time of the three 16-agent replays of the build served on 2026-10-26
(ledger rows `odie-fnx-iq3xxs-2xb70-real-comp1-no-gdn-ss16-n16-1026`, `-r2`, `-r3`; `engines.strata.fit`), which
the model reproduces exactly by construction (866 tok/s; eta 0.043 of the B70's inferred int8 peak, m 0.757 ms a
token, t_pass 0.39 s a 2048-token chunk). Held out (`checks`), the CUDA build on droopy's one RTX 6000 (PCIe 3) is
predicted at 424-674 tok/s for an 11,565-token prompt over its CPU-share range 0.5-0.84, measured 630-640. On that build the
per-token term is 2.6x the pass term at 2048 tokens, so a bigger chunk buys little (predictions written before the
chunk arms ran: `data/predictions-prefill-chunk-1026.json`). Left out: attention and DeltaNet costs that grow with depth.

Example, Flash-Next on odie's two B70 cards (layer split, 2 groups), the build served on 2026-10-25:

| agents | D (decode capacity) | P / F | expected (reads stall decode) | binding | window rows |
|---|---|---|---|---|---|
| 16 | 271.0 (measured) | 156.9 | 99.4 | prefill | 8, fits the 16-row CPU pool |
| 32 | 356.1 (measured) | 156.9 | 108.9 | prefill | 16, fits |
| 64 | 422.4 (projected) | 156.9 | 114.4 | prefill | 32: FAIL on this build |

P = 765.9 prompt tok/s, from this build's steady-state 16-agent replay with the highest output
(`odie-fnx-iq3xxs-2xb70-real-sp-ss16-n16-1026`, 78.6 tok/s, run with the shared-prefix setting on), and F = 4.88 from
the profile. The three steady-state 16-agent replays of this build made 70.9-78.6 tok/s. With each run's own P and F
the model expects 93.4-97.9 (24.5-36.3 % high), and its ceiling P / F is 1.9-2.1 times what they made. The model puts
prefill busy at 64-66 %; the runs measured 47-51 %. The gap includes time when no agent was waiting on the server: the
recorded tool time between turns (median 1.0 s, p90 11.9 s in these sessions), which the formula does not carry.
So a synthetic total of 356 tok/s at 32 agents stands against 71-79 tok/s that real agents got at 16 on the same build:
more agents raise decode capacity, not the output, and faster prompt reading is what helps.

Limits:

- D comes from the synthetic runs' context depth. Real agents decode at their whole prompt's depth (82k median in the
  profile's sessions), which is not measured separately.
- P is measured at one agent count. It moves with the read sizes, the cache and the engine settings: on the build
  served on 2026-10-26, two settings read prompts 25-35 % faster (1,085 and 1,170 tok/s against 866) but wrote no
  faster (72.5 and 69.3 tok/s against 83.4 at 16 agents), because they read more per token written (F 6.5 and 8.0).
- No tool-time term. The formula assumes every agent is always waiting on the server, reading or decoding. Real agents
  spend time in their tools between turns, and a lone agent leaves the server idle then. On a one-agent replay
  (`odie-fnx-iq3xxs-2xb70-real-comp1-no-gdn-ss16-n1-1026`, 32.6 tok/s) the formula read 33 % high; with the replay's
  recorded tool time added by hand, 4.5 % high. At 16 agents other agents' work fills part of that time, and the formula
  reads 24.5-36.3 % high on the example build above.

## Blind spots we hit and the checks that now catch them

The planner predicts well only what it models. Each time a run surprised us, the missing factor became a standing
check, so the next run gets it right the first time.

- **Synthetic decode-only tests.** Our many-agent curves gave each agent one short prompt and timed only the writing.
  That measures decode capacity, not what real agents get: 356 tok/s at 32 agents on the synthetic load stood against
  71-79 tok/s for 16 replayed real agents on the same build. Check `agent_workload` flags a many-agent recipe with no
  real-agent row of its own build.
- **Prefill-bound agents.** Real agents read far more prompt than they write, so reading prompts, not writing, set
  the limit, and adding agents added nothing. Check `prediction_first`: before launch, a many-agent recipe must show
  its predicted useful tok/s, which term binds (decode or prefill) and the fresh-tokens-per-output F it assumed.
  `--check` prints it on an `agents` line and `--check --json` carries it as `agent_prediction`. A recipe the model
  cannot bind (no fitted decode curve or no prefill rate for its build) is flagged unless it declares
  `conditions.workload.prediction {useful_tps, binding, fresh, src}`.
- **Replay-inflated F.** Our first replays read 17.8-67.7 fresh prompt tokens per output token. The recorded sessions
  need only 4.33-4.5 with a perfect cache (cold first prompts plus prompt growth, over the recorded outputs). The extra
  came from the replay and the engine, not from the agents: the timed window held every agent's cold first prompt and
  the re-reads after cache evictions. With a 600 s warm-up before the window, F fell to 4.5-8.0 by the server's count
  (3.9-4.9 by the replay client's). Check `replay_artifact` flags a real-agent row whose F is
  more than K times its reference (K = `settings.replay_artifact_k`, default 2). The reference is the trace-ideal
  ratio when the row names its trace (`profiles.<name>.trace_ideal_fresh_per_output`), else the profile's
  `fresh_reference_prior`. The report prints the excess, the fresh tokens above the reference and the suspected
  causes: cold restarts, evictions, template drift. Check `steady_state` asks each real-agent row for its warm-up
  (`warmup_s`, or `none`) and, when the replay gives one, its fresh breakdown (cold / growth / re-read). A row without
  a warm-up timed its window from a cold start, so its F is an upper bound. `tools/record_run.py --from-log replay:`
  takes both fields when the result has them.
- **Ungrounded expectations.** We set targets before reading what published serving work already knew.
  `data/priors-serving.json` now holds those expectations, each with a value, a unit, a tag (MEASURED-ELSEWHERE with a
  URL, MEASURED-HERE with our file and commit, or INFERRED) and `applies_when`, the recipe facts it needs. `--check`
  lists the priors that apply as `prior` lines; `node report.js --priors [recipe.json]` lists all of them or a recipe's;
  the Agents tab shows them under the workload callout.

`node report.js --workload` ends with both replay checks over every real-agent row in the ledger.

## Honest limits

From the ledger as of 2026-10-20 (`node report.js --accuracy`): 140 runs on all four boxes. Errors are scored only on
runs that went fine and were not disturbed, so most cells hold 2-15 runs. Error = (predicted - measured) / measured; a
negative bias means the planner predicted slower than the run.

- **Decode speed**, median |error| per box / model / engine:

  | box / model / engine | decode with draft | decode without draft |
  |---|---|---|
  | snoopy GLM (glmlite) | 2.5 % (n = 8, bias -0.8 %) | |
  | droopy V4.1 (ik) | 13.9 % (n = 7, bias -7.2 %) | 13.1 % (n = 4, bias +1.4 %) |
  | droopy GLM (ik) | 15.6 % (n = 4, bias -10.7 %) | 19.8 % (n = 8, bias -19.8 %: always pessimistic) |
  | odie LLaDA2.1-mini (block diffusion) | 4.1 % (n = 15) | |
  | odie / lassie, other block-diffusion models | 1.6-43.8 % (n = 2-4 each) | |

  By 2026-10-22 (185 runs) droopy V4.1 had grown worse, not better: 23.4 % (n = 20, bias -23.4 %; the newest builds
  run faster than it predicts). lassie V4.1 reads 19.4 % (n = 19, bias +4.8 %), with the decode-attention kernel's runs
  in. `node report.js
  --accuracy` prints the current figures and, in its trend table, whether the gap is shrinking.

  The in-load draft speed-up (1.198 GLM, 1.201 V4.1) matched the planner's multiplier (1.203, 1.259) then. Since
  2026-10-21 the speed-up depends on where the experts sit: decode with the draft = tau / (T + beta x CPU expert time),
  tau per box x model and one beta fitted on the in-load ledger twins (beta 0.37: lassie Ornith-35B x1.13 with 41 % of
  the token on CPU experts, x1.33 with every expert on the two GPUs). lassie's CPU-only V4.1 runs have no decode
  prediction yet.
- **Trend** (as of 2026-10-19): the gap was not shrinking for decode with drafts on (10-17 9.1 % n 6, 10-18 17.3 % n 1);
  no-draft decode and the launch-time gate miss share were falling. Most trend points hold one run.
- **Prefill** has a rate only inside its measured cells (V4.1 -ub 4096 at 62-70 GiB PMem, GLM -ub 2048 at 0-16 GiB);
  inside them 0.6 % (n = 2) and 8.4 % (n = 1). Everything else is UNMODELLED.
- **VRAM after warm-up** is within 0.4 % on droopy (n = 4). The VRAM crash terms came from the crashes they now catch.
- **Gate** (2026-10-20): 32 true positives (10 cause-matched), 13 misses, 15 false alarms (five of them lassie V4.1
  runs that failed the VRAM check and ran fine). Causes with no pre-launch check: a long
  prompt's VRAM (q4c), another program's VRAM growth (snoopy arm A), experts placed off plan after load (e1), harness bugs
  (p4), an unexplained slowdown (q4c). An interference case on odie (a build job beside the run) stays UNKNOWN because the strata engine's CPU
  decode is unknown.
- **Quality** is a straight line in the 2-bit share of routed reads, weighted per layer by the layer map when it
  calibrates (GLM: three measured mixes, one degree of freedom, largest miss 0.0034 KL; V4.1: two points, exactly
  determined). Treat a quality PASS near the bar as a guess.
- **No model yet** (UNMODELLED, so no number is given) for kernel-spread pages (numactl interleave), rotational-disk
  reads, and engine builds with no calibration.
- **Trust shares.** The catalog holds 930 tagged values: 308 seen (33 %), 616 inferred (66 %), 6 unknown. Only 49 seen
  values cite a ledger row; the rest are probe reads, kernel benches and routing captures. Across the 22 recipes that
  predict, the terms are 56 % seen, 42 % inferred and 2 % unknown. GPU nameplate bandwidth, `gpu_eta` and borrowed host
  terms are the inferred terms that recur most.
- **External gap** (41 community rows, 38 predicted; median |error| per class): Apple 20 %, Nvidia all-GPU 17 %, Nvidia
  hybrid 17 %, AMD 28 %, CPU-only 23 %, Intel Arc 56 %. Three terms are now measured on our boxes (`runs/bench.jsonl`, NOTES
  "Bench terms"): a GPU line per backend and card vendor (CUDA on droopy, SYCL and Vulkan on odie B70), a per-token MoE term
  per MoE layer, and a NUMA placement gain (lassie). The 2x EPYC 9474F rows fell from +132-240 % to +21-96 %. Arc stays the
  worst class. Vulkan reach on Intel is a property of the driver stack: the 2026-10-19 Mesa A/B on odie B70 #1 (NOTES
  "Bench terms") took our Vulkan line from 26 % of nameplate (image Mesa 26.0.8) to 65 % (Mesa 26.2.5, gate clean:
  MUL_MAT 5/5 per arm, perplexity bit-identical), llama2-7B Q4_0 from 86 to 101-104 tok/s (Linux Mesa 26.0.3: 80; the
  scoreboard's 112 is Windows, seen), Q8_0 2.4-2.6x; MMVQ off does not help on 26.2.5. The line is now the 26.2.5 stack
  (`bench.line_stack`); each stack keeps its own line in `bench.stacks`. The B70 Gemma4 SYCL row (+803 %) is that build's
  Gemma4 path, not a MoE term. Apple gpt-oss-120b (+103 %) and the AMD Vulkan MoE row (UNMODELLED: no Vulkan line on AMD
  cards) stay open.
- **dLLM decode** (block diffusion, 41 rows on odie B70 and lassie CPU): the held-out LLaDA2.2-flash run is within 0.1 %
  (TPF 1.2 %, ms per forward 1.0 %); leave-one-out 5.0 % median on decode (n = 12) and 3.2 % on ms per forward (n = 21).
  Expert coverage is MEASURED (fixed probe, 2026-10-19): a 32-token LLaDA2.1-mini block forward touches ~24 % of the
  experts per layer (rho 0.265), not 64 % (uniform) and not "nearly all" (the old probe bug); a Ling-mini-2.0 AR
  decode covers ~30 % per 32 tokens. With it the fitted kernel gaps fall to 3-7 % of the line. The stock
  SYCL kernel's gap differs by expert shape, so flash on the stock path is UNMODELLED. Spec decoding, full-sequence
  forwards, CPU experts beside a GPU and dense diffusion models (Dream-Coder) have no term.

## Exporting a public copy

```
tools/export_atlas.sh TARGET_DIR [--terms FILE]
```

What ships is written down in `publish/manifest.json`, so the copy repeats as results grow. `tools/curate.py` copies the
tracked files the manifest lists (never the build outputs, our notebook `NOTES.md`, raw logs, or tools that need our
private services) and curates the run ledger: rows of held-back models are left out, free-text notes are dropped,
per-number log paths become a placeholder, each degraded note becomes its cause class plus the phases it names, and each row's source list
becomes one placeholder. Pointers into our lab notebook become the words "lab notes"; real home paths become `~/`. It
copies the recipe of the fastest clean run per box and model plus every recipe a shipped test names, builds the copy,
writes `results/RESULTS.md` (measured runs with their conditions, and `node report.js --accuracy` on the curated
ledger), and removes the build outputs again. The ledger ships as plain text: the copy holds no archives. The script
then scans every file name and line, gzip files decompressed, for:

- private network addresses
- home paths
- session or presence wording
- credentials
- internal task, chat or job ids, and agent names
- pointers into our lab notebook that no rewrite rule caught
- e-mail addresses
- every name in a private-terms list that is never exported
- any file over 1 MiB
- any archive (gzip, zip, xz, zstd, bzip2, 7z, tar, SQLite)

Any hit refuses the export with exit 1 and marks the copy `status=REFUSED` in `.fitviz-export`. A new kind of note
pointer needs a rewrite rule in the manifest. A public URL that a shipped file must cite and that trips a rule gets an
`admit` line in the manifest; the copy carries it as `.scan-admit`. Run `tools/export_atlas.sh --scan-only DIR` again before you publish.

## Tests

```
python3 build.py
python3 -W error::ResourceWarning -m unittest discover -s tests
node tests/test_accuracy.js
node tests/test_advisor.js
node tests/test_decode_fit.js
node tests/test_batch.js
node tests/test_window.js
node tests/test_curve.js
node tests/test_workload.js
node tests/test_kv_host.js
node tests/test_screen.js          # needs the private quality-round ledger; not in the public copy
```
