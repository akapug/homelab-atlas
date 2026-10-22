# AGENTS.md

Dense map for coding agents reading or extending this repo. Humans: start at [README.md](README.md).

## What this repo is

Measured local-LLM serving on four owned boxes (6 accelerators + CPUs), with the patches, commands and failed
attempts behind each number. Static Markdown plus one planner (`recipe-maker/`, Python 3 + node, stdlib only) and a
GitHub Pages site built from `docs/`. No build step for the Markdown.

## File map

| path | holds | read it for |
|---|---|---|
| `README.md` | hardware table, best result per card, pick-a-path, highlights | the overview |
| `atlas.json` | boxes, cards, headline results with conditions, status and source page | machine-readable answers; keep in sync with README |
| `3090-ti/README.md` | RTX 3090 Ti (box A): Qwen3.8-27B vLLM CUDA, GLM-5.3-Flash route trim, Flash-Next agent runs, code-review rounds | |
| `3090-ti/vllm-cuda/Dockerfile` | vLLM 0.31.0 CUDA image + `draft-vocab.py` + `length-finish.py` | building the 3090 Ti server |
| `rtx-6000/README.md` | Quadro RTX 6000 + Optane (box B): V4.1 best run with full command, tier split, 189k recall, GLM, Ornith-122B | reproducing the best V4.1 run |
| `b70/` | Arc Pro B70 x2 (box C): `README.md` at a glance, `vllm-xpu/` patches, `llama.cpp-sycl/` patches, `bench/`, `agent-eval/`, model pages | Intel serving |
| `1080-ti/README.md` | GTX 1080 Ti (box D): V4.1 attention fix, Ornith-35B banks with the MI25 | Pascal + big MoE |
| `mi25/README.md` | Radeon Instinct MI25 (box D): Vulkan, cold-boot fix, expert banks, heat, the hop-latency negative | old AMD cards |
| `epyc-cpu/README.md` | 2x EPYC 7601 alone (box D): NUMA row-split ladder | CPU-only MoE |
| `moe-offload/README.md` | cross-box MoE story: boxes, best per model, 23 numbered lessons ("points"), quality reference problem, engine-change branches | why, and the engine changes |
| `recipe-maker/` | planner: `report.js`, `build.py`, `probe/`, `recipes/*.json` (layouts as data), `runs/` (ledger), `results/RESULTS.md` (generated) | planning a layout; run history |
| `PATCHES.md` | every patch/tool by hardware: symptom, apply, check, undo | applying a fix |
| `GLOSSARY.md` | terms | decoding shorthand |
| `CHANGELOG.md`, `CONTRIBUTING.md` | history; what a result needs | adding results |
| `docs/` | GitHub Pages output for Recipe Maker (generated; do not hand-edit) | |

## Naming crosswalk

| host name | moe-offload letter | Recipe Maker | ik_llama.cpp fork PATCHES.md letter | card pages |
|---|---|---|---|---|
| snoopy | A | snoopy | - | `3090-ti/` (B70s here until 2026-10-07) |
| droopy | B | droopy | B | `rtx-6000/` |
| odie | C | odie | - | `b70/` |
| lassie | D | lassie | A | `1080-ti/`, `mi25/`, `epyc-cpu/` |

The fork's letters are its own (its A is our D). B70 "#1"/"#2" or "card 1/2" = odie's two cards; card 2 serves
Ornith-1.5-35B-A3B, card 1 Qwen3.8 Flash-Next (2026-10-22).

## Conventions

- Speeds are tok/s, one request unless "summed over N streams". Decode vs prefill always named.
- `n4` = 4 drafted tokens/step (draft model or MTP), `n0` = no draft, `c1` = one client, `MTP 3` = model's own head,
  3 drafted. `keepK Lx-y` = route trim keeping K picks on layers x-y.
- Quality bar for speed changes: KL <= 0.03 and top-1 >= 0.97 vs a reference of the same model.
- Evidence words: "measured" (default), "our reading" / "inferred" (interpretation or fit), "not measured".
  Recipe Maker tags: `seen`, `inferred`, `unknown`.
- "Not yet in our public fork" / "unpublished" = the number used code not in public branches; the public-fork number
  is given beside it where measured.
- Dates are ISO, UTC.

## Which number is authoritative

- Card pages and `moe-offload` best table: hand-written, aggregated as the row states (mean/median of prompts).
- `recipe-maker/results/RESULTS.md`: generated from the ledger; its "fastest clean run" column takes the fastest
  **phase**, which can exceed a hand table's mean, and can include a run that failed a quality check later. Prefer
  the card page for a headline; use RESULTS.md for per-run detail.
- `atlas.json` mirrors the card pages; if they disagree, the card page wins and `atlas.json` is the bug.

## Verifying a claim

1. Find the row in `atlas.json` (`source` field) or the card page's "At a glance".
2. Follow its details link for conditions and quality check.
3. Commands: the card page's "Reproduce"; layouts as data in `recipe-maker/recipes/`; engine branches in
   `moe-offload/README.md#engine-changes`.
4. `cd recipe-maker && node report.js --check recipes/<file>.json` checks a layout without a GPU.

## Changing the repo

- Every push that changes results or structure adds a CHANGELOG line under Unreleased; cut a dated tag
  (`vYYYY.MM.DD`, `.N` same day) when a batch lands. Never move or delete a tag.
- Keep old paths working: external pages link into `b70/`, `mi25/`, `moe-offload/`, `PATCHES.md` anchors.
- New card: new folder with the card-page sections (box, at a glance, details, reproduce, what did not work, not
  measured yet), a README hardware row, an `atlas.json` entry.
- Every page ends with the sign-off line used on the other pages.
- See [CONTRIBUTING.md](CONTRIBUTING.md) for what a result needs.

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [arc](https://github.com/akapug/arc) + [fab](https://github.com/akapug/fab) for [@akapug](https://github.com/akapug)
