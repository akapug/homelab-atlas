# Changelog

What changed in this atlas, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions are dates (CalVer), and each release is an annotated git tag `vYYYY.MM.DD` (`.N` for a second one the same
day). Results that change a headline number say so.

## [Unreleased]

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

### Added
- `tools/check_atlas.py`: checks every `atlas.json` number against the page it cites, and every Markdown link and
  anchor. `atlas.json` rows now carry the run ledger id where one exists.

## [2026.10.22]

The atlas gets a shape: one page per card, a map up front, and a way to grow.

### Added
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

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [arc](https://github.com/akapug/arc) + [fab](https://github.com/akapug/fab) for [@akapug](https://github.com/akapug)
