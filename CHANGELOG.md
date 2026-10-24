# Changelog

What changed in this atlas, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions are dates (CalVer), and each release is an annotated git tag `vYYYY.MM.DD` (`.N` for a second one the same
day). Results that change a headline number say so.

## [Unreleased]

## [2026.10.24]

Two Arc Pro B70s serve 32 agents at once, and three $10 network cards link three boxes at 56 Gb/s.

### Added
- `b70/many-agents.md`: Qwen3.8 Flash-Next (IQ3_XXS) across both B70s on our batched XeStrata branches (not published
  yet) serves 32 agents at once in a coding team's shape, 2 top-level agents with 262,144-token slots and 30 subagents
  with 131,072, a 248k-token conversation among them: 329.6 tok/s in total, every agent 10.27-10.32 tok/s, 0.623 tok/s
  per watt of card plus CPU package power. With every slot 131,072 and short prompts it peaks at 349.6 (10.92 each,
  0.653 per watt). One 60 s window per run, synthetic coding-agent prompts. The page has the ladder from 24
  agents, k8v4 against int8 KV, a matched control for the new switches (+3.7 %), every switch we tried with its
  verdict, why the losers lost, the server slot fix that took the mixed shape from 31 decoding agents to 32, and what is still running. Headline: the B70
  row in the README table now leads with it; `atlas.json` has the result (`unpublished-code`).
- `network/`: three used ConnectX-3 cards at $10.19 each, cross-flashed from QCBT to FCBT firmware, as a switchless
  triangle: 52.6 Gb/s RDMA per link in InfiniBand mode, 52.3-52.6 RoCE and about 50 Gb/s parallel TCP in 56GbE
  Ethernet mode; flashing over the cable when a box cannot flash its own card.
- Recipe Maker export at fit-viz 50ad5c1: the **Agents at once** view and `node report.js --curve` (total and
  per-agent tok/s by agent count, the host-memory ceiling, tok/s per watt), the batch-window kernel planner
  (`--window`), the KV host-memory check for streamed KV, and the many-agent runs above in the ledger (201 rows).
  The accuracy table and gate counts in its README are recomputed on the new ledger.
  Its tok/s per watt uses the watts each run measured, within the boundary the run row states; it no longer calls them
  wall watts.
- GLOSSARY: k8v4, agent slot, RDMA / RoCE / IPoIB.

## [2026.10.22.1]

The 12.69 tok/s V4.1 run becomes rebuildable from public code and weights, and the 1080 Ti gets a decode-attention
kernel of its own.

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
- From the round-8 readers (Claude 78/72 at f4b3012; Flash-Next 77/63 and Ornith 88/82 at 4ccfd53): the Recipe Maker's
  "fastest clean run" table no longer lists a run that failed its own quality gate (the GLM 27.60 trim arm; the
  26.69 arm that passes now leads); `launch.sh` writes the tier-split control file to `./tier.ctl` (or `$TIER_CTL`)
  and prints its path, so one load can time several settings as the page says; the 3090 Ti 27B headline gives its
  10-run range (95-107) beside the one-pass 103-105; the 12.69 quality cell links the earlier build it came from.
- From a fresh Claude reader (76/66 at 4ccfd53): the measured `tier-split` commit was named wrong. The 12.69 run used
  our private `945fa93b`; the public `cde5fc20` is the same compiled code with comments and one test flag renamed, and
  `6d9992e2` adds only PATCHES.md (`build-engine.sh`, the recipe README and the MoE offload page now say so). The RTX
  6000 page's launch block sets `LD_LIBRARY_PATH` and runs `$ENGINE/bin/llama-server` as `launch.sh` does; its
  placement note says cold experts also go to the Optane node (330 of 811 rules); the PMem share names its two counts;
  the GLM 16.06 row gives its window, output length and prompts; the switch list points at the PATCHES.md that has
  all of `launch.sh`'s switches. The recipe adds a placement check (`place_verify.py`) after the load, our imatrix
  settings and timing, and a key to the lab names in `placement.tsv`. The GLM trim headline compares with the
  same-session 22.77, not the 23.08 base run.
- From the GPT-6.1 reader at 4ccfd53 (its diff): `serve-ab.py` waits for all three streams and fails on a worker
  error or missing token counts instead of reporting a short `streams3` row; `atlas.json` marks the Flash-Next 49.13
  agent eval as run at another window.
- From the DeepSeek V4.1 Flash reader (trust 85, repro 83 at 12aa2d5): `tools/check_atlas.py --accuracy` now also
  fails when a launch-time decode line of `report.js` has no row in the Recipe Maker accuracy table, so a deleted row
  no longer passes (each of four deleted rows tried fails the check); the Recipe Maker README says to run `build.py`
  before the Python tests (the suite reads the built data) and drops `test_screen.js`, which does not exist; the V4.1
  context correction names its real source (the rows' `lab notes`; the caveats file and launcher stay home).
- From the GPT-6.1 reader at 0491930 (its diff): `build-engine.sh` refuses a dirty source checkout instead of stamping a
  `SOURCE_SHA` that does not describe what it built.
- From the round-6 local readers (Flash-Next 86/62, Ornith 87/88 at 6323216): the 26.69 says it is the coding
  prompts' figure and gives the same arm's agent-brief figure (24.65) beside it, on the 3090 Ti page and in
  `atlas.json`; the B70 351 names its vocabulary and prompts; the recipe README explains the tier split's control file
  (`split=S prefetch=0|1 steal=0|1`, re-read while the server runs); `bench.py` says when it runs the public
  stand-in prompts.
- From the round-6 Claude reader (80/76 at 6323216): the RTX 6000 page prints `launch.sh`'s exact command (the
  tier split's three settings and the interleave weights were missing, so the page's command was the split-off arm);
  the recipe ships our `llama-imatrix` keep-partial patch and says the stock tool keeps every tensor at our pass size
  (at most 2.3 % of experts unseen, under its 5 % limit); the 3090 Ti 103-105 names its three requests, and the
  Flash-Next 49.1 its six (upper median, 42.3-55.4); the 1080 Ti highlight explains 6.97 to 7.96; the GLM highlight
  names its paired base; the B70 highlight says the 54.4 eager figure is another workload.
- From the round-5 local readers (Qwen3.8 Flash-Next 72/57, Ornith 88/86 at 12aa2d5): `atlas.json` gains
  `quality_on_this_build: false` on the three results whose quality check ran on another build or card (the 12.69
  V4.1, the 3090 Ti 27B, the B70 Flash-Next); the front page says the 189k recall ran on the build before the tier
  split, and that the recipe folder holds every input the run loaded but the prompts and the importance matrix (our
  quality harness and the lab launcher's checks are not in it either). Recipe Maker's publishing test covers the
  manifest's keep list.
- GLM-5.3-Flash on the 3090 Ti: the pages said the untrimmed 23.08 ran on the public image. It ran the public image
  with our unpublished CPU-lane changes mounted over it (the layer-ahead warm-up among them), as did every trim arm;
  the README, the 3090 Ti and MoE offload pages and `atlas.json` now say so. The base and trim runs used the same CPU
  lane (14 cores), not 14 against 16. The trim table gives each arm's two rounds and the untrimmed arm run beside it,
  which its percent compares against (22.68, 22.77, 23.11).
- From the round-5 Claude reader: `serve-ab.py`, the client behind the 3090 Ti and B70 comparison, is now in
  `b70/bench/` (the page linked a folder without it); the box D sentence under the MoE offload table says about half
  the bar (12.09), not under a third; the README's EPYC fine print says 15 of 16 DIMMs (the 12.09 had 16); the recipe
  says to keep the source on NVMe (the server reads shards 8 and 9 while it decodes) and what `"verify": "fail"` means.
- From the GPT-6.1 reader at 12aa2d5 (its diff): `build-draft.sh` can be re-run: a link it already made is kept, and a
  different file in its place still stops it.
- `build-store.sh` counts the routed-expert tensors in your importance matrix before the split and stops if any are
  missing (stock `llama-imatrix` drops a whole expert tensor when 5 % of its experts saw no tokens; gate may come from
  up). Tested on the Q8_0 source: our matrix passes 120 of 120, and the same matrix with one tensor removed is refused.
  The recipe README gives our matrix's size as collected, 789 MB (it said 1.56 GB).
- The droopy V4.1 window: 13 runs (the 12.69 run among them) were recorded as "server default (-c unset)"; our
  launcher sets `-c 4096` when no window is given, and every server log of them reads `ctx_size = 4096`. The Recipe
  Maker ledger now corrects the field through our caveats file (`runs/run_caveats.json`, which stays home because
  its sources are lab notes; the published rows carry the corrected value, and our rows on disk stay as recorded), its layout files read 4096, and the RTX 6000 page says so. Recipe Maker can now correct any recipe field
  that way (`recipe.<dotted path>`).
- From the round-4 readers: the 3090 Ti 3-stream figure is 279-288 and says what it is (all tokens over the wall
  time, prefill included; each stream decodes at 96-100); the 22.20 GLM figure says it predates layer-ahead warm-up;
  the 3090 Ti page says why its headline is the 27B and not the faster Sage-A1.5 (failed our code-review screen);
  the GLM conditions in `atlas.json` add the two-round agent-brief arm; the 1080 Ti highlight gives both 35 % pairs
  (6.97 against 5.16 without a draft, 7.13 against 7.02 with one); the B70 351 says compiled mode; the store highlight
  says 233 GB are new shards and three are the source's own; the moe-offload table has the 12.09 decode-attention row
  and says the 7.13 ran on 15 of 16 DIMMs; the recipe lists `numactl` and says why `test-expert-place` was not counted
  on a one-node box; AGENTS.md points at the recipe's list of what you cannot get from us; the build outputs are in
  `.gitignore`.
- Recipe Maker export and its live page: 185 runs (the 1080 Ti decode-attention rows added), so the accuracy tables in its README,
  how-it-works, AGENTS.md and PATCHES.md (V4.1: 23.4 % on the Optane box, 19.4 % on the 2017 server) are regenerated;
  the export keeps `lassie-v41-q8-rows8.json`, which two pages link to.
- `build-engine.sh` as published (with libgomp packaged) rebuilt `6d9992e2` with rc 0, and its `test-tier-split` passed
  all 16 arms on a host with the NVIDIA driver; the recipe README says so.
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
  `build-engine.sh` says why it pins public `6d9992e2` while the run used a private commit, lists the host needs and
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
