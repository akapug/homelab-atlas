# Hot-expert maps

A mixture-of-experts model picks k of E experts per layer for every token. Some experts are picked far more often
than others. A **hot-expert map** records how often the router picked each expert, per MoE layer, on a named prompt
set. It also records the model, checkpoint, quant, tool, engine and box that produced the counts.

Most offload recipes need this map as an input:

- **GPU banks.** The hottest experts go to VRAM.
- **Placement.** DRAM holds the warm experts. PMem or NVMe holds the coldest ones.
- **Per-expert quant mixes.** Hot experts get more bits than cold ones.
- **Cache sizing.** The planner models how much routed traffic a cache of N slots catches.

Recipe Maker reads every routing count from these maps. A recipe names the map it was planned on, and the
pre-launch gate refuses the recipe when the map has changed since.

Labels below follow the README: MEASURED, READ, INFERRED, ESTIMATE.

## The format

There is one JSON file per capture, named `<id>.heatmap.json`; a map whose text passes 1 MiB is written gzipped as
`<id>.heatmap.json.gz`, and every reader takes either. The maps live in `data/heat/`. The code is
`tools/heatmap.py`, which uses the Python standard library only.

| field | content |
|---|---|
| `format` | `"hot-expert-map/1"` |
| `id` | `model__workload__engine__date`, unique per capture |
| `title` | one line |
| `model` | `id` (vendor/name), `family`, `checkpoint` (which weights), `checkpoint_hash`, `quant`, `layers` (the MoE layer ids, in the order of the count rows), `experts` (E), `k` |
| `workload` | `name`, `description`, `prompt_set_sha256`, `prompt_set_files`, `decoding` (for example "greedy, 48 tokens") |
| `producer` | `tool`, `tool_version` (a commit or a sha256 of the script), `engine`, `engine_version`, `box`, `date` |
| `provenance` | `tag` (MEASURED, PRIOR or ESTIMATE), `src` (the run behind the counts), `converted_from` (`path`, `sha256` and converter of the source file), `notes` |
| `slices` | `{name: {phase, category?, half?, tokens, tokens_tag, picks, counts}}`. `phase` is `decode`, `prefill` or `mixed`. `counts` is one row of E integers per layer, in `model.layers` order |
| `default_slice` | the slice a planner uses when none is named |
| `summary` | derived per slice: `picks`, `tokens`, `top10_cover` and `top25_cover` (the share of picks taken by the hottest 10 % and 25 % of experts, averaged over layers), `never_picked`, `effective_experts_median` (exp of the per-layer routing entropy) |
| `digest` | `"sha256:"` + sha256 of the canonical JSON of `{model, workload, slices}` (sorted keys, no spaces) |

### Rules

- **Counts are integers.** Shares are derived from them. If a slice's token count is not in the source, it is
  inferred as picks per layer / k and tagged `INFERRED`.
- **Unknowns are stated, never guessed.** `checkpoint_hash` may be null only when `checkpoint_hash_note` says why.
  The same holds for `prompt_set_sha256` with `prompt_set_note`.
- **`checkpoint_hash` is a manifest digest.** It is the sha256 of the checkpoint's shard manifest, normalized to one
  `sha256  basename` line per shard, sorted by name. `heatmap.py manifest-hash FILE` computes it from `sha256sum`
  output or from a directory of `NAME.sha256` files. The manifests live in `data/heat/manifests/`.
- **Validation is strict.** `heatmap.py validate` checks shapes, recomputes the summary and recomputes the digest.
  A changed count or a changed metadata field fails the check.

### Selection files

A selection (`*.selection.json`, `"hot-expert-selection/1"`) records a hot set that was chosen from a map, for
example the experts placed in the GPU banks:

- `id`, `model`, `n_experts`
- `rule`: how the set was chosen
- `experts`: `{layer: [expert ids]}`
- `cites`: `{map_id, digest, slice}` of the map the set was ranked by
- `provenance`

## Measuring a map

### Choose the prompt set

- **Match the workload the recipe serves.** Heat depends on the task. On GLM-5.3-Flash, a hot set learned on code
  prompts catches 28 % of routed picks on chat prompts, against 45 % inside its own category (top 10 % of experts,
  MEASURED, lassie 2026-10-09).
- **Measure decode and prefill separately.** Decode is what an offload cache serves token by token. On the same GLM
  capture, the prefill hot set catches 19.2 % of decode picks, against 37.8 % for the decode set itself (top 10 %,
  MEASURED). An imatrix pass counts prefill only.
- **Split the set into halves.** Plan the layout on one half and score it on the other, so a layout is never graded on
  its own data. Interleave the categories so both halves have the same mix.
- **Collect enough tokens.** With few tokens per layer, coverage looks better than it is. Compare against a uniform
  router with the same number of picks: at 480 decode tokens per layer the uniform top-10 % floor is 15.1 %, at 3,456
  it is 11.9 % (MEASURED, GLM).
- **Record the prompt set's sha256.** Use the model's own chat template and greedy decoding.

Load on the box changes timings, not routing. Routing counts taken beside other work stay valid. Speed numbers from
the same run do not.

### Capture, per engine

| engine | how | output | converter |
|---|---|---|---|
| llama.cpp (mainline) | `tools/heat-capture/moe-trace.cpp`, a `cb_eval` hook on `ffn_moe_topk-<layer>`. Build it in a llama.cpp tree as `tools/moe-trace` (with the CMakeLists.txt there). Run `llama-moe-trace -m MODEL -t T -c 8192 -b 8192 -ub 512 -n 48` with `MOE_TRACE_WORKLOAD=prompts.jsonl` (`{"id", "cat", "messages"}` per line) and `MOE_TRACE_OUT=trace.bin`. On CPU-only runs add `-nr`: the CPU repack doubles RAM at load | binary trace: magic, prompt id, phase, layer, position, k, n, then the ids | `from-moe-trace` (`opts.experts`, `opts.categories_from` = the workload JSONL) |
| ik_llama.cpp | `llama-imatrix` with `tools/heat-capture/patch-imatrix-expert-counts.py` applied to `examples/imatrix/imatrix.cpp`. It writes `<imatrix>.expert-counts.tsv` beside each save; `--save-frequency 100` gives counts after 100 chunks. Our runs: `-c 512 -b 512 -ub 512 -ngl 99 -ot exps=CPU -fa 1 --no-mmap`. ik fuses gate and up, so only `ffn_up_exps` rows appear, and the converter takes up, then gate, then down | one TSV row per expert tensor | `from-imatrix-tsv` (phase `prefill`) |
| glm53 offload (exllamav3 + nv2) | `tools/heat-capture/route_stats.py`, bind-mounted over the image's `decode_kl_nv.py` and run through its `decode-kl` entrypoint. `GLM53_RS_TEXT` is prefilled in `GLM53_RS_CHUNK`-token chunks; `GLM53_RS_PROMPTS` is decoded `GLM53_RS_DECODE` tokens each. It counts the router's picks before lane masking | `route_stats.json` | `from-route-stats` |
| exllamav3 DeepSeek V4.1 recipe | the recipe's `EXL3_EXPERT_STATS` dump, with the capture-safe patch so it never dumps while a CUDA graph captures, then `snap2json.py` | `{layer: [[decode], [prefill]]}` | `from-exl3-stats` |
| vLLM, SGLang | not done yet. A forward hook on the router's top-k gives the same counts | | |

### Convert, check, publish

1. Add a row to `tools/heat_sources.json`. The row gives the source path, the converter, its options, and the facts
   the source file lacks: model, checkpoint, workload, producer and provenance. A `manifest` entry fills
   `checkpoint_hash`. A shared `model_ref` keeps model blocks consistent.
2. Run `python3 tools/heatmap.py convert-all tools/heat_sources.json data/heat`. It converts every source present on
   the machine and names the rest.
3. Run `python3 tools/heatmap.py validate data/heat/*.json data/heat/*.json.gz` and `python3 tools/heatmap.py summary MAP`.
4. Sanity checks before a recipe uses a map:
   - The coverage is far above the uniform floor for the same token count.
   - The hot set from one half covers the other half close to that half's own oracle. On GLM decode it reached 0.98
     of oracle (MEASURED).
   - After a checkpoint change, compare the new map with the old one per layer. For DeepSeek V4.1 0731 to 1017 on
     the A5 corpus: Spearman 0.991, layer hot-set overlap 0.960. That justified keeping the old hot set (MEASURED).
5. Run `python3 build.py`. The planner loads `data/heat` through one loader (`heatmap.load_dir`). If a map's tracked
   source file has changed since conversion, the build prints `STALE`.

## How Recipe Maker uses a map

- **Heat sets.** Each heat set in `data.json` is one slice of a map, and carries `map: {map_id, digest, slice}`.
  `build.py` names the sets the planner defaults to (for example V4.1 `coding_odd` = slice `coding_odd` of
  `dsv41-flash__skew187__dsv41-exl3__2026-10-08`). Every other slice of a map for the same model family becomes a
  selectable set named `map id:slice`.
- **Plan heat and score heat.** Plan heat orders the experts hottest first. That decides the banks, the DRAM/PMem
  split and the per-expert types. Score heat projects the reads per tier, the decode speed and the 2-bit share of
  routed reads. Use different halves for the two.
- **The gate.** `report.js --check` fails the recipe inputs when a heat set does not exist, or when the recipe cites
  a map digest that differs from the data's map. A recipe that cites nothing gets a note that names the map behind
  its sets.
- **The prediction.** `predictRecipe` returns `heat_maps: {plan, score}`, the citations it planned and scored on.

## How a recipe cites the map it used

```json
"heat": {
  "plan": "coding_odd",
  "score": "coding_even",
  "maps": {
    "plan":  {"map_id": "dsv41-flash__skew187__dsv41-exl3__2026-10-08", "digest": "sha256:c0a8...", "slice": "coding_odd"},
    "score": {"map_id": "dsv41-flash__skew187__dsv41-exl3__2026-10-08", "digest": "sha256:c0a8...", "slice": "coding_even"}
  }
}
```

- A sum set (for example `coding` = `coding_odd` + `coding_even`) cites a list, one entry per part.
- A recipe whose GPU banks come from a bank file also names the selection file. The selection file cites its map.
- A run ledger row keeps the recipe as checked, so the map behind every measured run stays known.

## Maps in this repo (2026-10-18)

All counts are MEASURED. Tokens marked * are INFERRED as picks per layer / k. Cover = the share of picks taken by
the hottest 10 % / 25 % of experts, averaged over layers.

| map | slice | tokens | cover 10 % / 25 % |
|---|---|---|---|
| `glm53-flash__skew72__llamacpp-cpu__2026-10-09` (UD-Q4_K_XL, llama.cpp CPU, lassie) | decode | 3,456 | 0.378 / 0.627 |
| | decode_code | 864 | 0.449 / 0.703 |
| | prefill | 32,693 | 0.281 / 0.506 |
| `glm53-flash__coding-kld-agent__glm53-nv2__2026-10-17` (EXL3 3.05, glm53 offload, snoopy) | prefill | 86,770 | 0.272 / 0.485 |
| | decode | 1,536 | 0.268 / 0.498 |
| `dsv41-flash__skew187__dsv41-exl3__2026-10-08` (V4.1 0731, EXL3 3.02, snoopy) | coding_all | 81,788* | 0.469 / 0.709 |
| | general_all | 110,850* | 0.370 / 0.610 |
| `dsv41-flash__a5-docs-51k__ik-imatrix__2026-10-16` (V4.1 0731 Q8_0, ik imatrix, droopy) | prefill | 51,202* | 0.500 / 0.729 |
| `dsv41-flash__a5b-heldout__ik-imatrix__2026-10-16` | prefill | 51,202* | 0.485 / 0.723 |
| `dsv41-flash-1017__a5-docs-51k__ik-imatrix__2026-10-17` (V4.1 1017 Q8_0) | prefill | 50,688* | 0.511 / 0.739 |
| `dsv41-flash-1017__v41q-corpus__ik-imatrix__2026-10-17` | prefill | 131,072* | 0.483 / 0.720 |
| `dsv4-flash__a5-docs__ik-imatrix__2026-10-14` (V4 0731, calibration only) | prefill | 51,201* | 0.469 / 0.694 |
| `kimi-k3__skew72-prompts0-9__llamacpp-cpu__2026-10-09` (UD-Q2_K_XL, PRELIMINARY, 10 prompts) | decode | 480 | 0.526 / 0.756 |

Two checks back the converters:

- The GLM trace converted from the raw `moe-trace` binary reproduces the planner's original per-category count file
  exactly (`tests/test_heatmap.py`).
- The V4.1 decode map's token count, 192,638 decode rows, equals the capture's own accounting.

## Gaps

- **The prompt sets are not public yet.** skew72, A5 and the coding sets contain our own code and documents. Before a
  map ships as a reproducible Atlas recipe input, it needs a public prompt set of the same shape, re-captured.
- **No V4.1 1017 decode map.** The 1017 maps are imatrix prefill counts. A decode capture of 1017 on a coding
  workload is owed.
- **The GLM coding decode slice is small.** It holds 1,536 tokens, so the GLM recipes plan on the 3,456-token skew72
  decode map.
- **Missing checkpoint hashes.** The EXL3 packs and V4 0731 have no shard manifest recorded. Their maps say so.
- **Not converted.** The Qwen3.8-Flash-Next expert profiles from Strata (`expert-profile-*.bin`) are not converted:
  their format is not read yet. The Flash-Next planner heat is still a synthetic ESTIMATE.
