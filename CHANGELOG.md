# Changelog

What changed in this atlas, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions are dates (CalVer), and each release is an annotated git tag `vYYYY.MM.DD` (`.N` for a second one the same
day). Results that change a headline number say so.

## [Unreleased]

### Added
- GTX 1080 Ti: a split-K decode-attention kernel for DeepSeek V4's head size 512 with sinks (`LLAMA_FA_DEC=1`, our
  fork's `fa-dec-sinks`) takes V4.1 Flash from 7.96 to 11.74 tok/s without a draft and 8.28 to 12.09 with one, in one
  binary on the same prompts. Headline: V4.1 on the 2017 server now reads 12.09 (was 7.13 on 15 of 16 DIMMs).
- `rtx-6000/v41-recipe/`: every input of the 12.69 tok/s V4.1 run except the prompts and the importance matrix: the
  engine build, scripts that build the split store and the draft from the public weights (with checksums), the
  hot-expert list and placement file the run loaded, the launch and the benchmark client, plus three public stand-in
  prompts. Its README says what each file did and where a rebuild differs.
- The engine changes behind 12.69 are public: our ik_llama.cpp fork's branches `expert-prebind`,
  `prebind-f32-indexer` and `tier-split`, plus `numa-experts` and `fa-tile-sinks`. The RTX 6000 page, the MoE offload
  page, PATCHES.md and `atlas.json` (the 12.69 entry is now `measured`) say so.

### Fixed
- From devindsflash at 0491930 (86/80): `check_atlas.py` now finds each atlas number as a whole number inside the
  section its anchor names (it caught two: the Flash-Next entry now cites the B70 page, and a 69.0 that reads as 69),
  and walks `docs/` for links too (only the generated `docs/page/` is skipped); Recipe Maker `--check --strict` holds
  (exit 3) on an UNMODELLED speed as well as an UNKNOWN check, with a test; the recipe scripts no longer leave an
  empty entry (the current directory) on `LD_LIBRARY_PATH`; the front page says the folder holds scripts that build
  the store from the public weights, not the weights.
- From the round-3 readers (Claude at 0491930; Qwen3.8 Flash-Next 86/72 and Ornith 88/85 at 9e05692): the 189k recall
  run names its build (`prebind-f32-indexer`) and ledger id, and its hot list and placement ship as
  `recall-hot-experts.tsv` and `recall-placement.tsv`; `build-store.sh` checks for PyYAML too, fails when
  `split.rules` differs, and has a `FETCH_ONLY=1` step so the imatrix command runs on a downloaded source;
  `build-engine.sh` says why it pins public `6d9992e2` while the run used private `cde5fc20`, lists the host needs and
  packages libgomp; `launch.sh` sets weighted interleave only where the kernel has it and says CPU 39 sits idle; the
  card reads the window as 4,096 and the layout's `ctx` as a label; the B70 row says its 27B numbers predate the move;
  the GLM 23.08 note carries its conditions.
- From a fresh Claude reader at dc8a266: the README no longer calls the 12.69 changes unpublished; `build-engine.sh`
  pins the measured commit and also builds `llama-imatrix`, with the imatrix command beside the others; the store and
  draft scripts find the packaged CUDA libraries, use `hf` (or the older `huggingface-cli`), and stop with a hint
  instead of a system-wide pip install; the store size reads 233 GB; the card's command names the published hot-list
  and placement files, the warm-up prefill, `-dple`'s part, #2712's gain with and without the draft (+2.2 / +1.6 %),
  and the wider tier-split sweep (12 threads ties 8, 16 is lower).
- From devindsflash and the two B70 readers at dc8a266: the V4.1 layouts keep the ledger's `ctx` (rewriting it had
  turned `--check` UNMODELLED and failed `tests/test_decode_fit.js`) and carry the real `-c 4096` in `ctx_note`;
  `SHA256SUMS` says how to feed it to `sha256sum -c`; the card says shard 10 is the source's too and claims every
  input "we can publish"; the recipe says why 12.69 and not 12.71 heads it and what `hot` means in the placement
  file; the Flash-Next headline no longer puts 87.1 tok/s and 11 of 12 on one build.
- Recipe Maker `--check`: the verdict line names what it could not judge (an UNKNOWN check, an UNMODELLED speed), so a
  GO never reads as a pass for them (a cloud reader saw GO under an UNMODELLED prefill); `test_decode_fit.js` covers it.

### Changed
- The publish scan now runs over the whole repository and comes back CLEAN. The few public strings it used to flag
  (heading anchors and commands that name Claude Code, the `noreply` author line in the SYCL patches, the eval's
  `example.com` address) are listed with their reasons in `.scan-admit`; anything else still refuses a push.
  CONTRIBUTING and AGENTS say so.
- README: speed claims name their correctness evidence and its limits; a check from an earlier build or another backend
  says so (from a cloud reader's diff).
- The V4.1 layout files `droopy-v41-fs1c.json` / `fs1p.json` say in a `ctx_note` that the launch really used `-c 4096`;
  `ctx` keeps the run ledger's wrong "-c unset", because the planner's decode fit matches layouts on it (rewriting it
  turned `--check` UNMODELLED and failed `tests/test_decode_fit.js`, found by a cloud reader). The RTX 6000 page says
  which is right.
- Recipe Maker README and how-it-works: no screen ledger or screen source list ships here, and a missing screen is not a
  passed one.

### Removed
- `.gitattributes`: its one line was a waiver for an internal commit guard, which no longer needs it.

## [2026.10.22]

The atlas gets a shape: one page per card, a map up front, and a way to grow.

### Added
- README: "Reviewed by its own hardware", what two local models on the two B70s found when they read the repo.
- `tools/check_atlas.py`: checks every `atlas.json` number against the page it cites, and every Markdown link and
  anchor. `atlas.json` rows now carry the run ledger id where one exists.
- Card pages for the hardware that had none: [RTX 3090 Ti](3090-ti/), [Quadro RTX 6000 + Optane](rtx-6000/),
  [GTX 1080 Ti](1080-ti/) and [the EPYC CPUs alone](epyc-cpu/), each with its best results first, the commands to
  reproduce them, what did not work and what is not measured yet.
- RTX 3090 Ti results: Qwen3.8-27B on vLLM 0.31.0 CUDA at 103-105 tok/s (MTP 3, draft vocabulary +28 %);
  GLM-5.3-Flash at 26.69 tok/s with a route trim that passes the quality bar; Flash-Next 23 of 24 agent tasks;
  four models in code-review rounds.
- [`3090-ti/vllm-cuda/Dockerfile`](3090-ti/vllm-cuda/Dockerfile): vLLM's official 0.31.0 CUDA image with
  `draft-vocab` and `length-finish`.
- RTX 6000 + Optane: DeepSeek V4.1 Flash at 12.69 tok/s (was 11.88), the tier split measured step by step, the full
  reproduce command, 16 of 16 facts recalled at 189k tokens.
- README hardware table with every box and card, a "pick a path" table, and newer highlights.
- [GLOSSARY.md](GLOSSARY.md), [AGENTS.md](AGENTS.md), [atlas.json](atlas.json), [CONTRIBUTING.md](CONTRIBUTING.md)
  and this changelog.
- B70 entry: an at-a-glance table for both cards and what each serves today.
- Agent eval: later results (2026-10-05 to 10-18), local and hosted.

### Changed
- `length-finish.py` skips the streaming half when vLLM already has it upstream (vLLM 0.31.0).
- PATCHES.md: the machine description now matches where the cards are (the B70s moved to a Threadripper box on
  2026-10-07); the RTX 6000, GTX 1080 Ti and RTX 3090 Ti are measured, no longer listed as untested hardware.
- MoE offload: the best-result table adds GLM-5.3-Flash with the route trim on box A and V4.1 at 12.69 on box B, and
  links each box to its card page.
- MI25: the B70 comparison figure now names its conditions and source.

### Fixed
- Recipe Maker's gate counts now match what `node report.js --accuracy` prints on the shipped ledger (175 runs: 31 of
  47 bad runs caught, 16 missed, 24 false alarms); three pages had three older counts.
- The B70's 61 / 351 tok/s headline names its condition: compiled mode with XPU graphs (54.4 eager).
- The agent eval's main table says how its 24 runs per config split over the two rounds.
- Three pages still signed off with an old workspace link.
- The V4.1 12.69 run: its quality check is marked as measured on the earlier build; the layout file is the one that
  run used; `-c 4096` explained; a list of what you cannot get from us yet (prompts, store, draft, two changes) and
  the PMem setup steps.
- GLM's 26.69 says its route trim is not published yet, everywhere it appears.
- The README names the boxes (snoopy, droopy, odie, lassie; A-D) where a human meets them.
- AGENTS.md explains how to read the run ledger (segments, null checks, what is private).
- Flash-Next on the 3090 Ti: the speed window (32,768) and the agent-run window (131,072) are told apart; Sage-A1.5's
  MTP figure says it is a speed-only probe; the RTX 6000 reproduce section says which figure the public branch gives.
- The EPYC and 1080 Ti pages name the recipe each V4.1 figure really used (the shipped recipe is the expert-row step;
  the 7.02 and 7.13 runs are named by ledger id), and date the lassie runs to 2026-10-20/21 with the one-DIMM-short
  condition they ran under; the MI25 page notes it too.
- The Recipe Maker page carries a plain-text introduction and repository links for readers without JavaScript.

- Readers' round two: the README table says which headline needs unpublished code, which quant, and which ran one DIMM
  short; atlas.json windows, threads and dates match the card pages; the B70's 61 and 74.6 tok/s are told apart;
  the 11.88 row no longer claims the unpublished changes; the intro is less pleased with itself.
- Readers' round three: the B70's headline is its fastest checked result, 74.6 tok/s at one stream with the reduced
  draft vocabulary (61 was the compiled figure without it); the MI25 compares its 59-60 with the same draft on the
  CPUs (47.3), not with no draft; the V4.1 ledger id is spelled as the ledger has it; the RTX 6000 page says where the
  engram shards sit and which quality figures Recipe Maker predicts; AGENTS.md says to run `build.py` before
  `report.js`; the intro no longer quotes an untold story or an unpriced price.
- The local readers' second round: the 12.69 caveat leads its quality cell; the RTX 6000 page names the hot-expert and
  placement files you cannot get from us; the GLM trim says the public image gives 23.08; the README shows both rounds.
- Readers' round four: the README hardware table keeps one clean headline per card and moves its fine print to a
  list right under it; the review-by-hardware section moves down and shrinks; Flash-Next's speed and agent runs name
  their two builds and windows; the EPYC page agrees with itself on 32 against 64 threads; atlas.json files the
  Ornith result under the MI25 too (`cards`), and `check_atlas.py` fails if any card has no result.
- The planner's V4.1 error figures in PATCHES.md and its how-it-works page are the tool's current ones (21.9 % and
  24.5 %, not the older 13.9 %), with the note that the gap is not closing yet.
- Readers' round five: Recipe Maker's accuracy table carries the tool's current figures (175 runs), and
  `check_atlas.py --accuracy` now fails when it drifts from `report.js --accuracy`; the B70 headline says which box it
  was measured in; the two B70 eager figures say they come from different workloads; "more Optane" no longer
  contradicts its own +15-18 %; every atlas.json source has an anchor; the test list names only tests that ship.
- Recipe Maker named GLM's quality reference (EXL3-3.05) on V4.1 checks too; each model family now names its own, and
  a measured KL beside a predicted top-1 says which is which.

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [arc](https://github.com/akapug/arc) + [fab](https://github.com/akapug/fab) for [@akapug](https://github.com/akapug)
