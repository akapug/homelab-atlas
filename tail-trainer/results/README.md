# tail-trainer: stranger smoke on two cards and a two-card run, 2026-11-15

Tool: `tail-trainer/` (its README is one level up). Question: does the packaged tool, run from a clean directory with only
its README's steps, pass its preregistered gate and train on a public MoE GGUF on the Quadro RTX 6000 and the
Instinct MI25; and can the two cards train two different layer tails at the same time whose adapters compose?

Every number is MEASURED from the JSON in this directory unless tagged otherwise.

## Setup

- **Model:** unsloth `Qwen3-30B-A3B-Q4_K_S.gguf` (17.46 GB, sha256 `fa0c96be...390b4a`, the Hugging Face file):
  48 blocks, n_embd 2048, 32 q heads / 4 kv heads of 128, 128 experts with 8 used, vocab 151,936. Tail types: blocks
  Q4_K (attention and experts), head Q6_K (`*/info.json`). Why this one: the most used small public MoE, under 20 GB,
  a real 128-way router, and arch `qwen3moe` (the same family as most current local MoEs); we already had it, so no download.
- **Text:** llama.cpp's own docs from the clone (every stranger has them): train = `docs/build.md`,
  `docs/backend/SYCL.md`, `docs/backend/OPENVINO.md`, `tools/completion/README.md`, `tools/cli/README.md`,
  `docs/speculative.md`, `docs/function-calling.md` (282 KB; the first 32 windows of 1024 tokens are used);
  held-out = `tools/server/README.md` (8 windows of 1024). 32 logit rows per window, spread over its second half.
- **Engine:** stock llama.cpp `e574a17` (ggml-org master, 2026-11-15), CPU build of `tail-dump` against it,
  `--no-mmap -t 16`, one ubatch of 1024 per window.
- **Tool:** the RTX 6000 smoke ran the tool before the allocator fix below; the MI25 smoke's GPU half, the two-card
  run and the MI25 adapter check ran it after. Only the allocator default differs between them.
- **Boxes:** both dumps on the CPU of the box that trained (RTX 6000 box: 2x Xeon Gold 5215, the dump under
  `numactl --interleave=0,1`; MI25 box: 2x EPYC 7601, `--interleave=all`). Both boxes ran other jobs (load average
  about 20-40), so seconds are upper bounds.
- **Gate (fixed in `tail_trainer.py` before any run):** top-1 agreement >= 0.97 and mean KL(engine || torch) <= 0.02
  nats over the 256 logit rows of the 8 held-out windows. Mean |delta log p| over the engine's top 20 is reported.

## Stranger smoke (README steps 1-7, clean directory per box)

| step | Quadro RTX 6000 24 GB | Instinct MI25 16 GB |
|---|---|---|
| torch | 2.9.1+cu126 (`install-cuda-legacy.sh`) | 2.11.0+rocm7.13 TheRock gfx900 (`install-mi25-gfx900.sh`) |
| clone + build `tail-dump` | 88 s | 65 s |
| dump train, 32 windows | 425 s | 738 s |
| dump held-out, 8 windows | 120 s | 215 s |
| weights `--first 44` (blocks 44..47 + head) | 89 s | 115 s |
| **gate, `--match q8`** | **PASS: top-1 0.992 (254/256), KL 0.00085** (max 0.019), top-20 0.092 | **PASS: top-1 0.992 (254/256), KL 0.00080** (max 0.028), top-20 0.090 |
| gate, `--match none` | PASS: top-1 0.988 (253/256), KL 0.00140, top-20 0.115 | PASS: top-1 0.984 (252/256), KL 0.00130, top-20 0.116 |
| gate seconds (load / forward of 8 windows), card peak | 33 / 7.5, 6.0 GB | 24 / 11.8, 6.0 GB |
| **train, 20 steps (rank 8, 16 targets: attn q/k/v/o of 44..47, 1024 tokens a step)** | **1.79 s a step** (median 1.85, first 2.65), 6.33 GB | **2.46 s a step** (median 2.46, first 4.77), 6.39 GB |
| train loss, step 0 / step 19 | 2.041 / 1.210 | 2.041 / 1.204 |
| held-out CE, every position (8,184), torch tail: base / with the adapter | 1.528 / 1.403 | 1.526 / 1.403 |
| held-out CE at the 248 logit rows, the engine: base / with `--lora` | 1.144 / 1.091 | 1.145 / 1.090 |
| **adapter gate: torch with the adapter vs the engine with `--lora run/adapter.gguf`** | **PASS: top-1 0.988 (253/256), KL 0.00083** | **PASS: top-1 0.988 (253/256), KL 0.00089** |

Files: `rtx6000/` and `mi25/` (`gate*.json`, `train.json`, `eval-*.json`, `info.json`, `smoke.steps` with the step
times, `llama.sha`). `eval-lora.json` compares torch-with-adapter against the base engine dump (KL 0.094: the adapter
moved the distribution); `eval-lora-engine.json` against the engine with the adapter (KL 0.0008).

Reading:
- The same GGUF dumped by two different CPU engines (AVX-512 Xeon, AVX2 EPYC) and run by two torch backends (CUDA
  on Turing, HIP on gfx900) gives the same gate to 3 decimals.
- 20 steps on 32 windows of llama.cpp docs lower the held-out CE of the server README by 0.125 nats in torch and
  0.053 nats in the engine's own logit rows: the tail learns the domain, and the engine with the exported GGUF
  adapter computes what torch trained (adapter gate PASS on both cards). This is a smoke, not a quality claim: one
  seed, one text, 20 steps.

## What the smoke found (fixed in the tool)

- **MI25: `expandable_segments` breaks every allocation on ROCm gfx900.** The first MI25 gate died loading 6 GB onto
  an empty 16 GB card: `hipErrorOutOfMemory` at the first `t.to(device)`. A 4 GB allocation fails with
  `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` and succeeds without it (probe on the card). The tool had set it
  by default for every cuda device (the 1080 Ti needed it, `tail-portable/`). Now it is set only on NVIDIA,
  and through `torch.cuda.memory._set_allocator_settings`: measured on the RTX 6000 with torch 2.9.1, the environment
  variable (set before or after import) left `is_expandable` False, the call sets it True.
- **Dump speed: the page cache sat on the Optane node.** With mmap and `numactl --interleave=all`, the first dev dump
  ran 80 s a window; with `--no-mmap` under `--interleave=0,1` (the two DRAM nodes) 16 s. The tool gained
  `--no-mmap` and the README says to use it.
- Stock llama.cpp has no tool that writes full intermediate tensors: `llama-debug` / `llama-eval-callback` print a
  few values per tensor as text and save the final logits only, so `tail-dump` (public API only: `cb_eval`,
  `llama_set_adapters_lora`) is needed.

## Two cards, two tails, one dump

Both cards trained at the same time (started at the same second, 04:34:55Z) on the same 32 windows (the RTX 6000
box's dump, copied to the MI25 box), 40 steps each, rank 8, attn q/k/v/o:

| card | tail | trained blocks | step s (mean after first) | peak GB | loss step 0 / 39 | wall |
|---|---|---|---|---|---|---|
| Quadro RTX 6000 | 44..47 + head (`--first 44`) | 44..47 (adapter A) | 1.33 | 6.33 | 2.041 / 1.809 | 94 s |
| Instinct MI25 | 40..47 + head (`--first 40`) | 40..43 (adapter B; 44..47 frozen) | 4.46 | 11.04 | 2.038 / 1.798 | 259 s |

(`two-card/train-a-rtx6000.json`, `two-card/train-b-mi25.json`.) The MI25 runs twice the blocks, and its 8-block tail
(11 GB) fits the 16 GB card without expert offload.

### Do the two adapters compose?

All evals on the RTX 6000 over the held-out text (8 windows, the `--first 40` tail so both adapters apply). "engine"
is a llama.cpp dump made with the same `--lora` set; the gate compares that dump with torch carrying the same set.

| adapters | torch CE, all 8,184 positions | engine CE, 248 logit rows | torch CE, same rows | engine vs torch gate |
|---|---|---|---|---|
| none (base) | 1.5284 | 1.1439 | 1.1460 | top-1 0.992, KL 0.0013 |
| A (blocks 44..47) | 1.3874 | 1.0852 | 1.0854 | PASS, top-1 0.996, KL 0.00141 |
| B (blocks 40..43) | 1.3883 | 1.0843 | 1.0818 | PASS, top-1 0.984, KL 0.00124 |
| A + B, trained in parallel | **1.4579** | **1.1635** | 1.1643 | PASS, top-1 0.980, KL 0.00159 |
| A + B', B' trained with A frozen (`--lora A`) | **1.3855** | not dumped | 1.0800 | not run |

(`two-card/mc-eval-{base,a,b,ab}.json`, `two-card/mc-gate-{a,b,ab}.json`, `two-card/mc-eval-ab-seq.json`,
`two-card/train-b-after-a-rtx6000.json`.)

- **The engine side needs no work.** llama.cpp loads both GGUF adapters at once (`--lora A --lora B`) and computes
  exactly what torch computes with both: the A+B gate passes at the same KL as one adapter.
- **Two tails trained at the same time on the same objective do not compose.** Each one alone cuts held-out CE by
  0.14; together they are worse than either (1.458), and in the engine worse than no adapter (1.164 vs 1.144). Both
  learned to correct the same error, so stacking them corrects it twice.
- **Training the second tail with the first frozen does compose**, measured in torch only: B' (40 steps, blocks
  40..43, `--lora A` frozen, 2.70 s a step on the RTX 6000, loss 1.855 to 1.733) gives 1.3855 with A, a little better
  than either adapter alone. The engine dump for A + B' was not made. B' also adds 40 more steps of training, and no
  80-step single-adapter control was run, so this shows that sequential training does not break composition. It
  does not show that two tails beat one.
- So multi-card training is fast as a sweep (two tails, two cards, one dump, each adapter usable alone), and for one
  combined model the protocol is sequential: train A, then B with `--lora A`. Parallel training of composable
  tails would need a shared objective (for example cards exchanging their adapters every k steps), which this tool
  does not do.
