# Recipe Maker

**Plan where every part of a big mixture-of-experts model lives before you load it.** Recipe Maker takes your boxes
(GPUs, DRAM per NUMA node, Optane PMem, disks) and a model's layout, places each expert on a tier, predicts the decode
speed and quality of that layout, and refuses a launch that would run out of memory, read from a spinning disk or miss
a quality bar.

- **Predicts before you run:** decode tokens per second with and without a draft model, VRAM peak, free memory per
  NUMA node, and the share of routed reads at low bit widths, for each layout it proposes or each recipe you write.
- **Refuses bad launches:** `node report.js --check RECIPE.json` is a pre-launch gate. On the 175 runs in the shipped
  ledger (2026-10-22) it failed 31 of the 47 runs that later went bad; `node report.js --accuracy` prints the current
  count.
- **Says what each number rests on:** every number is tagged `seen` (we measured it), `inferred` (fitted, estimated or
  read from a spec) or `unknown`, with its source. A prediction prints how many of its terms are each kind.
- **Measures itself:** every real run goes into a run ledger with the prediction made at launch, and
  `node report.js --accuracy` scores the planner against what happened.
- **Nothing to install:** Python 3 and Node.js, standard library only. The page opens from disk; nothing is fetched.

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

From `node report.js --accuracy` on the shipped ledger (2026-10-21). Error = (predicted - measured) / measured, scored
only on runs that finished cleanly and were not disturbed; a negative bias means the planner predicted slower. The first
two columns score what the planner said when each run launched; the last replays every run through today's planner.

| Box / model / engine | Decode (draft on) | Decode (no draft) | Decode (draft on), today's planner |
| --- | ---: | ---: | ---: |
| snoopy, GLM-5.3-Flash, glmlite | 2.5 % (8 runs, bias -0.8 %) | | 2.5 % (8 runs, bias -0.8 %) |
| droopy, DeepSeek V4.1 Flash, ik_llama.cpp | 20.2 % (14 runs, bias -19.3 %) | 23.3 % (11 runs, bias -23.3 %) | 19.5 % (14 runs, bias -18.5 %) |
| droopy, GLM-5.3-Flash, ik_llama.cpp | 15.6 % (4 runs, bias -10.6 %) | 19.8 % (8 runs, bias -19.8 %) | 16.1 % (4 runs, bias -10.6 %) |
| droopy, Ornith-1.5-122B, ik_llama.cpp (in sample: its host term is fitted on this one run) | 1.3 % (1 run, bias +1.3 %) | 0.4 % (1 run, bias -0.4 %) | 1.3 % (1 run, bias +1.3 %) |
| lassie, DeepSeek V4.1 Flash, ik_llama.cpp | 24.5 % (10 runs, bias +24.5 %) | 10.8 % (9 runs, bias +10.8 %) | 26.9 % (11 runs, bias +26.9 %) |
| lassie, Ornith-1.5-35B-A3B, ik_llama.cpp | 14.6 % (4 runs, bias +14.6 %) | 9.7 % (4 runs, bias -9.7 %) | 1.1 % (3 runs, bias +1.1 %) |
| odie, LLaDA2.1-mini (block diffusion), one Arc Pro B70 | 4.1 % (15 runs, bias -1.5 %) | | 4.1 % (15 runs, bias -1.5 %) |
| odie / lassie, other block-diffusion models | 1.6-43.8 % (2-4 runs each) | | 1.6-43.8 % (2-4 runs each) |

The pre-launch gate, over the same ledger (175 runs, 2026-10-22): 31 runs it failed went bad (10 of them for the
cause the failing check names), 16 runs went bad with no check failing, and 24 runs it failed ran fine. On a box it has never seen, expect the
gap it shows on published community measurements (17-59 % by hardware class, `node report.js --external`) until your
own runs are in the ledger. Where it is still weak: [docs/how-it-works.md#honest-limits](docs/how-it-works.md#honest-limits).

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
node report.js --screen              # quality rounds against each model's card conditions (verdict-ready)
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
4. After the run, `tools/kit_record.sh RECIPE.json LOG...` adds it to the run ledger, and `node report.js --accuracy`
   shows how far the prediction was off. Each recorded run on your box narrows the next prediction for it.

### Hot-expert maps

The planner keeps the experts a model routes to most in the fastest tier. It reads that from hot-expert maps
(`data/heat/`): per-layer routing counts on a named prompt set. [HOT-EXPERT-MAPS.md](HOT-EXPERT-MAPS.md) says how to
capture one for your model with `tools/heat-capture/` and convert it with `tools/heatmap.py`.

## What is in this folder

| Path | What |
| --- | --- |
| `build.py`, `families/` | the planner's tables: boxes, models, quant types, fitted terms; one file per extra model family |
| `src/model.js`, `src/app.js` | the solver and the gate (shared by the page and `report.js`), and the page |
| `report.js` | the command line: report, `--check`, `--accuracy`, `--external`, `--advice` |
| `probe/` | the box probe and the model intake |
| `data/` | probed boxes, model layouts, hot-expert maps, layer maps, the upgrade catalog |
| `runs/` | the run ledger (`runs.jsonl`, older rows in numbered segments), bench terms, cause labels, community measurements |
| `data/cards/`, `runs/screen.jsonl` | each model's card conditions (sampling, thinking, template, context, output) with sources, and the quality-round ledger the `verdict-ready` check reads (`node report.js --screen`) |
| `recipes/` | example recipes: the fastest clean run per box and model, and the ones the tests use |
| `results/RESULTS.md` | every measured run with its conditions, and the accuracy report |
| `QUANT-PRIORS.md`, `QUANT-RECIPES.md` | desk research on published quants, with links; the build reads their JSON twins |
| `tools/` | run recording, bench sweeps, routing capture, placement check |
| `tests/` | `python3 -m unittest discover -s tests`, then `node tests/test_accuracy.js`, `node tests/test_advisor.js` and `node tests/test_screen.js` |
| `docs/how-it-works.md` | the reference: evidence tags, the gate's checks, recipe fields, the models behind each number, honest limits |

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [arc](https://github.com/akapug/arc) + [fab](https://github.com/akapug/fab) for [@akapug](https://github.com/akapug)
