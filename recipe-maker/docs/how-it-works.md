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
| 0 | GO | no check FAILs (with `--strict`, none is UNKNOWN either) |
| 1 | REFUSE | a hard check FAILs: do not launch |
| 2 | BAD INPUT | unreadable recipe, or an unknown box, model or engine |
| 3 | HOLD | `--strict` only: a check is UNKNOWN |

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

  By 2026-10-22 (175 runs) the V4.1 cells had grown worse, not better: droopy V4.1 21.9 % (n = 19, bias -21.9 %; the
  newest builds run faster than it predicts) and lassie V4.1 24.5 % (n = 10, bias +24.5 %). `node report.js
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
pointer needs a rewrite rule in the manifest. Run `tools/export_atlas.sh --scan-only DIR` again before you publish.

## Tests

```
python3 build.py
python3 -W error::ResourceWarning -m unittest discover -s tests
node tests/test_accuracy.js
node tests/test_advisor.js
node tests/test_screen.js
```
