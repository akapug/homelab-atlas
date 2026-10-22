# Changelog

What changed in this atlas, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions are dates (CalVer), and each release is an annotated git tag `vYYYY.MM.DD` (`.N` for a second one the same
day). Results that change a headline number say so.

## [Unreleased]

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
