# Recipe Maker

**Plan where every part of a big mixture-of-experts model lives before you load it.** Recipe Maker takes your boxes
(GPUs, DRAM per NUMA node, Optane PMem, disks) and a model's layout, places each expert on a tier, predicts the decode
speed and quality of that layout, and refuses a launch that would run out of memory, read from a spinning disk or miss
a quality bar.

- **Predicts before you run:** decode tokens per second with and without a draft model, VRAM peak, free memory per
  NUMA node, and the share of routed reads at low bit widths, for each layout it proposes or each recipe you write.
- **Refuses bad launches:** `node report.js --check RECIPE.json` is a pre-launch gate. On the 201 runs in the shipped
  ledger (2026-10-24) it failed 31 of the 58 runs that later went bad (10 of those 58 are many-agent rate probes on an
  engine it does not model, which the ledger marks degraded because the client cancels its streams after the timed
  window); `node report.js --accuracy` prints the current
  count.
- **Says what each number rests on:** every number is tagged `seen` (we measured it), `inferred` (fitted, estimated or
  read from a spec) or `unknown`, with its source. A prediction prints how many of its terms are each kind.
- **Measures itself:** every real run goes into a run ledger with the prediction made at launch, and
  `node report.js --accuracy` scores the planner against what happened.
- **Nothing to install:** Python 3 and Node.js, standard library only. The page opens from disk; nothing is fetched.

## Best measured result per model (2026-10-26)

Read this first. One row per model, with the box behind the fastest result that has a quality check beside it. Decode
is tokens per second for one request; "many agents" is the total over all agents decoding at once. Every number is
`seen` (measured), except where a cell says `not measured`. The run id is the ledger row (`node report.js --accuracy`
and the [Atlas](../README.md) card pages carry the full conditions); "card page" marks a hand-aggregated headline
whose ledger fastest phase can be higher.

| model | best one-request decode | box | many agents at once | tok/s per W | quality check | date | source |
|---|---|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | **27.39** (2-bit IQ2_XS experts) | snoopy, RTX 3090 Ti + DDR5 | not measured | not measured | 13 of 15 agent tasks; no KL check | 2026-10-24 | `snoopy-v41-gsq-iq2xs-strata-ramcopy-eval` |
| DeepSeek V4.1 Flash | 12.69 (release-precision hot experts) | droopy, RTX 6000 + DDR4 + PMem | not measured | not measured | KL 0.0169, top-1 0.975 on the earlier build of the same store | 2026-10-22 | card page (ledger fastest phase 13.06, `droopy-v41-v41fs-ts1-0026`) |
| DeepSeek V4.1 Flash | 12.09 | lassie, GTX 1080 Ti + EPYC | not measured | not measured | release-precision experts | 2026-10-21 | card page (ledger fastest phase 13.22, `lassie-v41-q8-fs3dec`) |
| GLM-5.3-Flash | **26.69** (route trim) | snoopy, RTX 3090 Ti + DDR5 | not measured | not measured | KL 0.0154, top-1 0.975: passes | 2026-10-20 | card page (ledger fastest phase 27.6, `snoopy-glm-glm-coder-s1w-trim-a`) |
| GLM-5.3-Flash | 16.06 (mean of 4 prompts x 3 rounds) | droopy, RTX 6000 + DDR4 | not measured | not measured | KL 0.030, top-1 0.960 against a 3-bit reference: a miss | 2026-10-20 | card page (one prompt's fastest phase 17.56, `droopy-glm-glm-be1r1-1140`) |
| Qwen3.8 Flash-Next | **94.6** (one request on both cards, layers split between them, every expert in VRAM at 262,144 context, int8 KV, MTP 4; median of 3 prompts x 800 tokens, thinking off; measured 2026-10-22. A one-request layout, not the batched build served for many agents, which keeps 12,800 experts a card in VRAM and the rest in RAM. The Atlas's 87.1 is another run: one B70, IQ3_S, about half the experts computed on the CPU) | odie, 2x Arc Pro B70 | **377.9** over 32 agents x 131,072 slots, 11.81 each (synthetic decode-only load, 60 s; see below) | **0.729** (cards + CPU package, 518.6 W) | code-review screen passed on the one-request build; no quality run on the batched build | 2026-10-24 | `odie-fnx-iq3xxs-2xb70-split-1022`; `odie-fnx-iq3xxs-2xb70-sweep-served-c8-base-1024` |
| Qwen3.8 Flash-Next, real coding-agent traffic | **32.6** for one agent alone; about 5.2 each at 16 agents (whole turns: prompt reading, writing and tool time) | odie, 2x B70 | **83.4** over 16 agents (mean of three runs: 86.0, 84.6, 79.6; recorded coding-agent sessions replayed after a 600 s warm-up, 600 s window); limited by prompt reading: about 0.87k prompt tok/s, 4.8-5.0 new prompt tokens per token written (server count; the replay client counts 3.9-4.9) | 0.287 (the two cards only, 291 W; not comparable with the cards + CPU figures above) | not graded (replayed sessions) | 2026-10-26 | `odie-fnx-iq3xxs-2xb70-real-comp1-no-gdn-ss16-n16-1026` (and `-r2`, `-r3`); `odie-fnx-iq3xxs-2xb70-real-comp1-no-gdn-ss16-n1-1026` |
| Qwen3.8 Flash-Next | 49.13 | snoopy, RTX 3090 Ti | not measured | not measured | 23 of 24 agent tasks | 2026-10-18 | `snoopy-fnx-qfn-snoopy-strata-q8-0517` |
| Qwen3.8-27B | **103-105** | snoopy, RTX 3090 Ti | 351 over 8 streams (B70, compiled mode) | not measured | 41 of 41 correctness eval | 2026-10-07 | card page |
| Ornith-1.5-35B-A3B | **198** (with MTP 3 drafting; served with MTP off, about 118) | one Arc Pro B70 | 290 over 2 streams | not measured | 17 of 24 agent tasks | 2026-10-04 | card page |
| Ornith-1.5-35B-A3B | 60.01 | lassie, GTX 1080 Ti + MI25 | not measured | not measured | code-review screen 1 of 2 | 2026-10-21 | `lassie-orn35-q4km-o6` |
| Ornith-1.5-35B-A3B | 59.6 (short reply, thinking off) | droopy, RTX 6000, CPU part bound to DRAM | not measured | not measured | none on this box | 2026-10-24 | lab notes; placement fixture `runs/assistant-droopy-fallback-20261024.jsonl` |
| Ornith-1.5-122B | 31.6 | droopy, RTX 6000 + DDR4 | not measured | not measured | both planted defects found | 2026-10-20 | card page (ledger fastest phase 33.78, `droopy-orn122-orn122-r0g9-1557`) |

The "many agents" totals from the rate probe are decode capacity on a synthetic load: each agent reads one short
prompt once, and only the writing after it is timed. Real coding agents read 4 to 5 new prompt tokens for every token
they write (tool results, file contents, each turn's new text), so prompt reading, not writing, sets their speed. On
the two-B70 box, 16 replayed real agents got 79.6 to 86.0 tok/s in total on the build served on 2026-10-26 (the batched engine is our
experimental XeStrata fork, branch [`homelab-batched`](https://github.com/akapug/XeStrata/tree/homelab-batched); mean 83.4
over three runs; 77.8 to 89.7 with the prompt chunk size changed), and one agent alone got 32.6. The
[B70 many-agent page](../b70/many-agents.md#real-coding-agent-traffic-the-number-to-plan-with) has those runs;
`node report.js --workload` shows which of the two limits a setup.
Batch rows above come from the rate probe and real serving; the planner does not yet predict batch tok/s per watt or
the slowest agent's speed (see below), so it cannot score them.

**Checking a choice, not only a number.** Before a set of runs, a recipe can name in its `validation` field the
alternatives you are choosing between, the workload, and what counts as best (decode speed for one request, default
target 25 tok/s, within the quality and memory limits you declare). `--check` then records the planner's prediction for
every alternative before any of them runs, and once all have run, `--accuracy` reports whether the planner ranked them
right and how much its pick would have given up. If the data changed since the last build, `--check` refuses to record:
run `python3 build.py` and declare the set again. Runs made before a set was declared are not scored this way. Choices
about many-agent serving cannot be scored yet: the planner does not predict total tok/s per watt or the slowest
agent's speed, and those stay unknown rather than guessed.

`runs/assistant-droopy-fallback-20261024.jsonl` keeps one placement failure apart from the main ledger: a model meant
to sit in DRAM node 0 on droopy had 3,862,511,616 bytes (3.6 GiB) of its file on the Optane PMem node instead, with no
error. After the fix its buffers sat on node 0. `node report.js --accuracy runs/assistant-droopy-fallback-20261024.jsonl`
shows it; it does not count in the ledger figures below. Context, offload, batching, threads and load mode changed in
the same fix, so it shows where the memory went, not how much speed the placement alone cost.

It started as the research notebook of a homelab that runs DeepSeek V4.1 Flash and GLM-5.3-Flash on four mixed boxes.
The numbers in it are ours; the method and the tools work for any Linux box.

## Our boxes

| Box | Hardware (installed) | In the Atlas's [moe-offload entry](../moe-offload/) |
| --- | --- | --- |
| **snoopy** | AM5: Ryzen 9 9950X3D, 192 GB DDR5, RTX 3090 Ti 24 GB | box A |
| **droopy** | HP Z8 G4: 2x Xeon Gold 5215, 192 GB DDR4 + 256 GB Optane PMem 100, Quadro RTX 6000 24 GB | box B |
| **odie** | TRX40: Threadripper 3960X, 128 GB DDR4, 2x Intel Arc Pro B70 32 GB | box C |
| **lassie** | Supermicro H11DSi: 2x EPYC 7601 (8 NUMA nodes), 1 TB DDR4, GTX 1080 Ti 11 GB, Radeon Instinct MI25 16 GB | box D |

Every measured run, the fastest clean run per box and model with its conditions, and the planner's accuracy are in
[results/RESULTS.md](results/RESULTS.md), generated from the run ledger.

## How close are its predictions?

From `node report.js --accuracy` on the shipped ledger (201 runs, 2026-10-24). Error = (predicted - measured) / measured, scored
only on runs that finished cleanly and were not disturbed; a negative bias means the planner predicted slower. The first
two columns score what the planner said when each run launched; the last replays every run through today's planner.

| Box / model / engine | Decode (draft on) | Decode (no draft) | Decode (draft on), today's planner |
| --- | ---: | ---: | ---: |
| snoopy, GLM-5.3-Flash, glmlite | 1.8 % (9 runs, bias -1.2 %) | | 1.8 % (9 runs, bias -1.2 %) |
| droopy, DeepSeek V4.1 Flash, ik_llama.cpp | 24.8 % (21 runs, bias -24.8 %) | 26.3 % (15 runs, bias -26.3 %) | 25.3 % (21 runs, bias -25.3 %) |
| droopy, GLM-5.3-Flash, ik_llama.cpp | 15.6 % (4 runs, bias -10.6 %) | 19.8 % (8 runs, bias -19.8 %) | 15.8 % (4 runs, bias -10.5 %) |
| droopy, Ornith-1.5-122B, ik_llama.cpp (in sample: its host term is fitted on this one run) | 1.3 % (1 run, bias +1.3 %) | 0.4 % (1 run, bias -0.4 %) | 1.3 % (1 run, bias +1.3 %) |
| lassie, DeepSeek V4.1 Flash, ik_llama.cpp | 15.7 % (22 runs, bias +3.7 %) | 9.9 % (21 runs, bias +6.0 %) | 19.4 % (23 runs, bias +3.8 %) |
| lassie, Ornith-1.5-35B-A3B, ik_llama.cpp | 16.1 % (4 runs, bias +16.1 %) | 9.7 % (4 runs, bias -9.7 %) | 1.2 % (3 runs, bias +1.2 %) |
| odie, LLaDA2.1-mini (block diffusion), one Arc Pro B70 | 4.1 % (15 runs, bias -1.5 %) | | 4.1 % (15 runs, bias -1.5 %) |
| odie / lassie, other block-diffusion models | 1.6-43.8 % (2-4 runs each) | | 1.6-43.8 % (2-4 runs each) |

The pre-launch gate, over the same ledger (201 runs, 2026-10-24): 31 runs it failed went bad (10 of them for the
cause the failing check names), 27 runs went bad with no check failing (10 of them the many-agent rate probes above), and 24 runs it failed ran fine. On a box it has never seen, expect the
gap it shows on published community measurements (17-59 % by hardware class, `node report.js --external`) until your
own runs are in the ledger. Where it is still weak: [docs/how-it-works.md#honest-limits](docs/how-it-works.md#honest-limits).

**Prompt reading, predicted before the runs.** On 2026-10-26, before three runs on the two-B70 box with 16 real agents
that changed only the prompt chunk size (4,096, 8,192 and 16,384 tokens against the default 2,048), the planner wrote
down its predictions from the hardware and one fitted per-token cost (`data/predictions-prefill-chunk-1026.json`):
prompt reading at 913, 921 and 922 tok/s, with anything outside 780 to 1,000 counting as a miss. The runs read at 814,
844 and 858. All three landed inside that band, but the predictions read 7.5 to 12.2 % high, and the small gain they
predicted from bigger chunks did not appear: all three read slower than the 866 of the default chunk. Its agent
formula, applied after the fact to a one-agent run without refitting, read 33 % high (43.3 against a measured 32.6
tok/s): it has no term for the tool time between turns, when a lone agent leaves the server idle. With that time added
by hand it read 4.5 % high (34.1). The formula still has no tool-time term, so it overstates setups with few agents.

## What you need

| | |
| --- | --- |
| **Software** | Python 3.10 or newer and Node.js 18 or newer. Both use their standard library only |
| **A browser** | for the planner page; it opens `index.html` from disk |
| **Your boxes** | Linux. The probe reads them over ssh (or locally) and never writes on them |
| **Your model** | a GGUF file or a Hugging Face repo: the intake reads headers only, never the weights |

## Use it

```
python3 build.py                     # data.json, index.html and page/*.js; generated, never in git
node report.js                       # best layouts at 20 and 25 tok/s per model, box and engine
node report.js --check RECIPE.json   # the pre-launch gate
node report.js --accuracy            # predicted vs measured over the run ledger
node report.js --external            # our physics on published community measurements
node report.js --screen              # your quality rounds against card conditions (no screen ledger ships here)
node report.js --window              # batched decode from its kernels: which kernel to write first (lever table)
node report.js --curve               # agents vs decode speed on a synthetic load (not real agent traffic: see --workload)
node report.js --workload            # real agents: decode capacity vs prompt reading, which one limits the output
```

Open `index.html` after the build. It shows our boxes, models and runs. The same page, built from this folder, is at
https://akapug.github.io/homelab-atlas/ (nothing to install).

### With your own boxes

1. Probe each box: `python3 probe/probe_box.py --ssh HOST --name NAME --write data/boxes/NAME.json` (or run it on the
   box without `--ssh`). Then add a hand entry for NAME to `boxes` in `build.py`, starting from the closest of ours
   ([Adding a box](docs/how-it-works.md#adding-a-box)). Until you measure the DRAM and PMem read rates and the CPU
   factor, tag them ESTIMATE; every prediction says it rests on them.
2. Read your model's layout from its headers: `python3 probe/model_intake.py --hf ORG/REPO --name KEY --write
   data/models/KEY.json` ([Adding a model](docs/how-it-works.md#adding-a-model)). A new model family goes in its own
   file, `families/KEY.py`.
3. Copy the recipe in `recipes/` closest to your plan, point it at your box, model and files, and run
   `node report.js --check recipes/NEW.json --strict`. Fix every FAIL before you launch; read each `trust` line.
   GO means no hard check failed; whatever the verdict line lists as "not judged" (an UNKNOWN check, an UNMODELLED
   speed) has no answer either way.
4. After the run, `tools/kit_record.sh RECIPE.json LOG...` adds it to the run ledger, and `node report.js --accuracy`
   shows how far the prediction was off. Each recorded run on your box narrows the next prediction for it.

### Which kernel to write first (batched decode)

`node report.js --window` builds one batched-decode window per pipeline stage from the card's measured kernels and
ranks each kernel class by the tok/s it would pay at N agents if it reached k x its bandwidth floor (default 1.2x), next
to the layer-split rebalance and the unmeasured terms. To get your own ranking on another model or card:

1. Read the model's batch-window spec from its GGUF header: in an XeStrata tree, `tools/batch_window_spec.py
   SHARD1.gguf [--layers A-B] > spec.txt`. Copy it to `data/window/`.
2. Run the batch-window microbenchmark on your card for the same layers and put its log in `data/window/`. Only the
   `STAGE ROLL-UP` table, the copy-kernel lines and the method-floor line are read, so drop wrapper lines that name your
   hosts or jobs. `python3 probe/window_inputs.py --spec spec.txt --ubench run.log` shows what will be read.
3. Write `data/window/NAME.json`: start from our `odie-fnx-2xb70.json` where your copy has it, else from the toy config
   in `tests/test_window.js`. Point it at your two files and give your stages (layer ranges, one card each), row groups,
   engine caps and your spec's kind rule. Add the stage times your engine prints as `fit` points (two agent counts, a few
   and many, fit the default extras on two stages). Include finite, positive measured throughput and stage times on
   training points to fit pipeline efficiency eta; without those operands, throughput and its ranking are UNKNOWN,
   not an assumed efficiency of one. Add independent observations as `check` points: they never fit eta or stage terms.
   If your engine can profile a window per kernel and per stage, add that as the config's `profile`: each term's ms a
   window per stage, its layer kind (or `stage: last` for the head) and the ubench classes it runs. The profile then
   prices the plan and the levers. The ubench gives each term its floor and remains the fallback. A kernel you have already
   rewritten and timed goes in `variants` (its class, the old and new us). The table prices it as measured. Profile points
   and curve/plan anchors condition that reading; they are not untouched end-to-end checks, even if marked `check`.
4. `python3 build.py`, then `node report.js --window NAME` (options `--agents N`, `--xfloor K`, `--bw GBS` for a card
   not in `data/boxes`, `--stages S`, `--source ubench` to ignore the profile, `--json`).
5. Record what you tried in the config's `attempts`, including the losses and the attempts that repaired them, so the
   lever table's `tried` column shows them next to each lever. Name your best measured run as the `frontier`, with its
   run-to-run noise. Before building a kernel, print a blank proposal with `--propose LEVER`, fill it in and add it to
   `proposals`. The report then dedups it against what was tried (same mechanism and action), prices it against the
   frontier and prints the correctness gate it still owes. Read the calibration error first (the `check` lines), then the lever
   table: its top rows are the kernels to write first, and the rebalanced column says what each pays once the split
   follows it.

The page's **Kernel + network** tab uses the same report to price plans that span boxes: pipeline stages on two
boxes, reading prompts on another box, drafting on a side card, keeping idle KV cache on another box. It needs a
catalog of measured links between your boxes (`data/interconnects.json`: each link's direction, transport and measured
rate). This copy ships none, so the tab ranks nothing until you add yours. A plan that rests on an unmeasured link rate,
round-trip time or engine feature is listed but not ranked. Tok/s per watt there counts the lending box's idle draw
plus the extra CPU and GPU power, not an assumed wall figure. Edits on the tab stay in your browser.

[How it works](docs/how-it-works.md#batch-window-what-a-kernel-is-worth-before-you-write-it) has the model and its limits.

### How many agents at once

`node report.js --curve` reads every ledger run that measured total tok/s at two or more agent counts and draws total and
per-agent tok/s from 1 to 64 agents. Those runs use a synthetic load: each agent reads one short prompt once, then only
its writing is timed. The curve is therefore decode capacity, the most the engine can write, not what working agents
get. From it the report prints the most agents that each still decode 10 tok/s or more and the total there (`--min X`),
the same at 20 each (`--second X`), the total the setup approaches with unlimited agents, how many agent slots host
memory holds (the `--check` host-pinned check run per slot count), and tok/s per watt when a run logs watts
(`measured.power_w`). `--curve all` lists every config; `--curve RUN_ID` picks one. The page shows the same on its
**Agents at once** tab, in plain words. The settings live in `data/curves.json`.

Real coding agents get less. Every turn they send the conversation back and the engine reads the new part: tool
results, file contents, and whatever it dropped from its cache. In our steady-state replays that was 3.9 to 4.9 new
prompt tokens per token written (the recorded sessions need 4.33 with a perfect cache), so prompt reading sets their
speed. On the two-B70 box, whose synthetic load decodes up to 377.9 tok/s over 32 agents (build served on 2026-10-24),
16 replayed real agents got 79.6 to 86.0 tok/s in total and one agent alone 32.6 (build served on 2026-10-26; see the
[B70 many-agent page](../b70/many-agents.md#real-coding-agent-traffic-the-number-to-plan-with)).
`node report.js --workload` adds this: with F new prompt tokens read per token written and a prompt reading rate P,
agents write at most min(decode capacity, P / F) tokens per second in total. It takes P from runs that replayed
recorded agent sessions on the same engine build (or, with none, predicts it from the cards, link and expert
placement), says which term limits each setup (prefill-bound or decode-bound), and checks that each batch window fits
the engine's CPU expert pool. F and the shape of a turn are a workload profile in `data/workload.json`; the default is
measured coding-agent traffic. A recipe for many agents with no real-agent run of its build is flagged by `--check`.
[How it works](docs/how-it-works.md#agent-workload---workload) has the model and its limits.

### Hot-expert maps

The planner keeps the experts a model routes to most in the fastest tier. It reads that from hot-expert maps
(`data/heat/`): per-layer routing counts on a named prompt set. [HOT-EXPERT-MAPS.md](HOT-EXPERT-MAPS.md) says how to
capture one for your model with `tools/heat-capture/` and convert it with `tools/heatmap.py`.

## What is in this folder

| Path | What |
| --- | --- |
| `build.py`, `families/` | the planner's tables: boxes, models, quant types, fitted terms; one file per extra model family |
| `src/model.js`, `src/app.js` | the solver and the gate (shared by the page and `report.js`), and the page |
| `report.js` | the command line: report, `--check`, `--accuracy`, `--external`, `--advice`, `--batch`, `--window`, `--curve`, `--workload` |
| `probe/` | the box probe, the model intake, and the batch-window spec and microbenchmark readers |
| `data/` | probed boxes, model layouts, hot-expert maps, layer maps, the upgrade catalog, batch-window configs (`data/window/`) |
| `runs/` | the run ledger (`runs.jsonl`, older rows in numbered segments), bench terms, cause labels, community measurements |
| `data/cards/` | each model's card conditions (sampling, thinking, template, context, output) with sources. The quality-round ledger the `verdict-ready` check reads (`runs/screen.jsonl`, `node report.js --screen`) is kept private and not shipped; without it that check reports unknown |
| `recipes/` | example recipes: the fastest clean run per box and model, and the ones the tests use |
| `results/RESULTS.md` | every measured run with its conditions, and the accuracy report |
| `QUANT-PRIORS.md`, `QUANT-RECIPES.md` | desk research on published quants, with links; the build reads their JSON twins |
| `tools/` | run recording, bench sweeps, routing capture, placement check |
| `tests/` | `python3 build.py` first (several tests read the `data.json` it writes), then `python3 -m unittest discover -s tests`, `node tests/test_accuracy.js`, `node tests/test_advisor.js`, `node tests/test_decode_fit.js`, `node tests/test_batch.js`, `node tests/test_window.js`, `node tests/test_curve.js`, `node tests/test_workload.js` and `node tests/test_kv_host.js`; `node tests/test_screen.js` needs the private quality-round ledger and is not in the public copy |
| `docs/how-it-works.md` | the reference: evidence tags, the gate's checks, recipe fields, the models behind each number, honest limits |

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [arc](https://github.com/akapug/arc) + [fab](https://github.com/akapug/fabric) for [@akapug](https://github.com/akapug)
