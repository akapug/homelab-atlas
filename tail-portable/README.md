# A PyTorch trainer on every card we own: B70, RTX 6000, MI25, 1080 Ti

We had a small PyTorch trainer that ran only on CUDA with TF32 (an RTX 3090 Ti). It is a torch reimplementation of the
last five blocks and the output head of Qwen3.8 Flash-Next (llama.cpp's `qwen4exp` architecture: hyper-connections,
full-attention blocks, gated delta nets, a 256-expert MoE), checked row by row against the log-probabilities that
llama.cpp itself computes, and it trains a rank-8 LoRA on those blocks. On 2026-11-14 we ported it to the other cards in
the house: an Intel Arc Pro B70, an NVIDIA Quadro RTX 6000 (Turing), an AMD Radeon Instinct MI25 (Vega 10) and a GTX
1080 Ti (Pascal). This folder is what that took, so someone with one of these cards can skip the dead ends.

What carries over to any torch model on these cards: the install recipes (which torch build works where, and which do
not), the op probe, and the list of changes below. The trainer itself is research code for one model and is not in
this repository, so the patch is a record of the changes, not something you can apply as is.

Every number here is measured on 2026-11-14 unless it says otherwise.

## What is here

| file | what it does |
|---|---|
| [`install-b70-xpu.sh`](install-b70-xpu.sh) | torch 2.10.0+xpu in a user-space venv for an Arc Pro B70. Needs Intel's compute runtime (Level Zero) and the render group, nothing else |
| [`install-cuda-legacy.sh`](install-cuda-legacy.sh) | torch 2.9.1+cu126 for Turing (sm_75) and Pascal (sm_61) cards. The cu128 and later wheels dropped Pascal |
| [`install-mi25-gfx900.sh`](install-mi25-gfx900.sh) | AMD's TheRock multi-arch torch wheels with the gfx900 device package, for an MI25 or other Vega 10 card, on the stock kernel `amdgpu` driver with no ROCm install on the host |
| [`probe_ops.py`](probe_ops.py) | runs every op the trainer uses on a device against the CPU and says which fail or give wrong results (index ops are checked exactly). Run it first on a new card: `python3 probe_ops.py --device xpu` |
| [`repro_xpu_nonzero_scatter.py`](repro_xpu_nonzero_scatter.py) | the torch-only check for the XPU wrong-result bug below; exit 1 when any case differs from the CPU |
| [`patches/0001-tail-trainer-device-portable.patch`](patches/0001-tail-trainer-device-portable.patch) | the whole trainer change as one diff against the CUDA-only version (unpublished base; read it for how each change below was made) |

Each install is one venv directory (`$VENV`); rollback is deleting it. Nothing is installed outside it.

## Results

The check is the same on every card and was fixed before the port: a forward pass over 413 rows (8 sequences)
compared with llama.cpp's own log-probabilities; it passes when top-1 agreement is at least 0.97, mean KL at most 0.02
nats, mean |delta log p| over the top 20 tokens at most 0.10, and over the paragraph-break token at most 0.10. Then 24
optimizer steps, one sequence each (8.1k to 14.1k tokens, mean 10.1k). Every number of the check is exact f32 (TF32
off).

| card | torch | top-1 | mean KL | top-20 | break | check | step s (mean after the first) | peak GB | training flags |
|---|---|---|---|---|---|---|---|---|---|
| RTX 3090 Ti 24 GB (reference) | CUDA | 0.998 (412/413; this check ran on the CPU) | 0.00036 | 0.046 | 0.014 | PASS | about 5 (TF32) | 20.6 | default |
| Arc Pro B70 32 GB | 2.10.0+xpu | 0.995 (411/413) | 0.00035 | 0.046 | 0.013 | PASS | 9.8 (card 1), 9.3 (card 2), f32 | 21.1 | default |
| Arc Pro B70 32 GB, driver 26.05 | 2.15.0+xpu | 0.978 (404/413) | 0.0065 | **0.223** | 0.033 | **FAIL** | not run | - | - |
| Arc Pro B70 32 GB, driver 26.40 | 2.15.0+xpu | | | | | PASS (2026-11-15) | not run | - | - |
| Quadro RTX 6000 24 GB | 2.9.1+cu126 | 0.993 (410/413) | 0.00036 | 0.046 | 0.013 | PASS | 18.2 (f32) | 15.4 | `--fine 2 --save-on-cpu` (5 GB of the card held by another server) |
| Instinct MI25 16 GB | 2.11.0+rocm7.13 (TheRock, gfx900) | 0.993 (410/413) | 0.00035 | 0.046 | 0.013 | PASS | 77.2 (f32; 49 to 154) | 9.9 | `--offload-activations` |
| GTX 1080 Ti 11 GB | 2.9.1+cu126 | not measured yet | | | | | | | `--offload-activations` (planned) |

- Step times are wall clock on shared boxes. The MI25's box ran other jobs at load average 100 to 123 on 64 cores during
  its run, so its step time is an upper bound.
- The 1080 Ti has its venv installed; the run waits for the card, which was busy with another experiment.
- The MI25's other memory settings, partial runs at 8.1k to 11.3k tokens: default checkpoints 13.7 GB at 8.8k and out of
  memory at 11.3k; `--fine 1 --save-on-cpu` and `--fine 2 --save-on-cpu` fit 11.3k (12.8-13.7 GB, about 39 s a step)
  and run out at 14.1k on the 15.98 GB card. Only `--offload-activations` (no recompute, every saved tensor in pinned
  host memory) fits the 14.1k sequences: 6.4 to 9.9 GB.

## What the trainer needed to run everywhere

| change | why | cards |
|---|---|---|
| `--device rocm\|cuda\|xpu\|cpu` (`rocm` is `cuda`: HIP is `torch.cuda`) | device-agnostic memory stats, sync and pinning | all |
| TF32 only where it exists (`--tf32 auto`: NVIDIA sm_80 and up) | Turing, Pascal and Vega have none | all |
| `--offload-experts auto` for the check | 14 GB of tail weights do not fit a 16 GB card beside the activations | MI25, 1080 Ti |
| rope angles on the CPU when the device has no f64 | some Arc parts lack native f64; the B70 reports `has_fp64=1`, so they stay on the card there | Intel without f64 |
| a boolean mask by comparison, not `nonzero` or mask indexing; `scatter_` with a tensor source | wrong results on torch 2.13 to 2.15+xpu (below) | B70 |
| `(I + L)^-1` of the chunked delta rule by recursive 2x2 blocks on XPU | `torch.linalg.solve_triangular` falls back to the CPU on XPU: 26 s a call at the tail's size, against 0.24 s on the card (relative error 1e-7 against f64, the same as the CPU f32 solve) | B70 |
| the MoE reuses its forward's routing in a checkpoint recompute | atomic sums (rocBLAS split-K GEMMs, oneDNN, `index_add_`) differ in the last bits from run to run; a near-tied router then picks another expert in the recompute and torch raises `CheckpointError` | MI25, B70 |
| `bench --fine 1\|2`, `--save-on-cpu`, `--offload-activations` | 14k-token sequences need about 21 GB of activations at the default checkpointing | MI25, 1080 Ti |

A first try at the B70 inverse, the power series `prod (I + (-L)^(2^j))`, took 0.006 s on random inputs, but its
partial powers overflow f32 on the real tail (NaN in the check), so it was replaced by the 2x2 blocks.

## Per card

### Intel Arc Pro B70 (Battlemage): use the latest driver with the latest torch

On compute-runtime 26.05.37020 (the Level Zero driver), `torch.nonzero`, boolean-mask indexing and
`Tensor.scatter_` with a Python scalar return silently wrong results on this card in torch 2.13.0, 2.14.1 and
2.15.0+xpu; 2.10.0, 2.11.0 and 2.12.1+xpu are correct on the same machine and driver. Measured with the pip wheels from
download.pytorch.org/whl/xpu, one venv per version:

| torch | `nonzero` and mask indexing (60 random-mask cases) | `scatter_` with a scalar (sweep) |
|---|---|---|
| 2.10.0+xpu | correct | correct |
| 2.11.0+xpu | correct | correct |
| 2.12.1+xpu | correct | correct |
| 2.13.0+xpu | 60 wrong | 9 wrong shapes |
| 2.14.1+xpu | 60 wrong | 9 wrong shapes |
| 2.15.0+xpu | 60 wrong | 9 wrong shapes |

The wrong cases are small to medium (64 to 4,096 elements for `nonzero`; [2..64, 40..100] for `scatter_`); large
shapes pass. With those three ops avoided, 2.15.0 still fails the check above (top-20 0.223 against 0.046), so
another op is likely affected; we have not isolated it. A wrong `nonzero` also trips the "index out of bounds" device
assertion when its output is used as an index.

**On a newer driver the check passes.** After these runs the box moved to compute-runtime 26.40.40196.15 (IGC 2.42.14,
Level Zero 1.34), and there torch 2.15.0+xpu passes the 413-row check above (2026-11-15). The op-level rerun of
`repro_xpu_nonzero_scatter.py` on that driver is pending. So read the table as "driver 26.05 with torch 2.13 to 2.15
gives wrong results", not as a torch bug on every driver: use the latest compute-runtime with the latest torch, and run
the repro on yours before you trust an upgrade. On driver 26.05, pin torch 2.10.0 to 2.12.1+xpu (the install script
pins 2.10.0).

Run one process per card with `ZE_AFFINITY_MASK=N`. Two early runs on one card ended in `UR_RESULT_ERROR_DEVICE_LOST`
at the 14.1k step (kernel log: an `xe` VM worker error -12, then exec queue resets), both with the CPU-fallback solve;
with the on-card inverse both cards ran all 24 steps clean. Not repeated enough to name the cause.

### NVIDIA Quadro RTX 6000 (Turing, sm_75): torch 2.9.1+cu126, f32

Nothing else was needed. Turing has no TF32, so the trainer runs f32. With 5 GB of the card used by another server, it
trained with `--fine 2 --save-on-cpu` at 15.4 GB peak.

### NVIDIA GTX 1080 Ti (Pascal, sm_61): torch 2.9.1+cu126, untested so far

The cu126 wheels are the last with Pascal kernels. The venv installs and sees the card; the check and the training run
have not run on it yet. On 11 GB it will need `--offload-activations`.

### AMD Radeon Instinct MI25 (Vega 10, gfx900): TheRock wheels, not the stable ROCm wheel

The stable `torch==2.9.1+rocm6.4` wheel imports, reports the GPU as available and lists `gfx900` first in
`torch.cuda.get_arch_list()`, but its bundled rocBLAS has no gfx900 Tensile kernels, so the first f32 matmul fails:

```
rocBLAS error: Cannot read .../torch/lib/rocblas/library/TensileLibrary.dat: Illegal seek for GPU arch : gfx900
```

The bundled library set is gfx908, gfx90a, gfx942, gfx1030, gfx110x and gfx120x (no gfx906 either). AMD's own rocBLAS
4.4.1 package for ROCm 6.4.2 ships the same set, so copying it in does not help. AMD's TheRock multi-arch staging
wheels do work: `torch==2.11.0+rocm7.13.0a20260515` with `amd-torch-device-gfx900==2.11.0+rocm7.13.0a20260515` from
`rocm.nightlies.amd.com/whl-staging-multi-arch/`, on the stock Ubuntu 6.8 kernel driver. They are nightlies; the pin
in the install script is the one we measured. We checked the 2.9.1 stable wheel only, not newer stable ones.

rocBLAS on gfx900 is not bitwise repeatable (the hyper-connection mixer's K = 10,240 GEMM differs from run to run; with
`torch.use_deterministic_algorithms` it repeats), which is what made the routing reuse above necessary.

## Reproduce

You need the trainer's inputs (block-42 residuals dumped from llama.cpp, its log-probabilities, the dequantized tail
weights), which come from our unpublished tools. Without them, the parts that stand alone are the install scripts,
`probe_ops.py` and the repro:

```bash
VENV=~/venvs/torch-xpu ./install-b70-xpu.sh        # or install-cuda-legacy.sh, install-mi25-gfx900.sh
. ~/venvs/torch-xpu/bin/activate
python3 probe_ops.py --device xpu --json probe.json   # rocm on AMD, cuda on NVIDIA
python3 repro_xpu_nonzero_scatter.py                  # XPU only
```

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [fleet](https://github.com/akapug/fleet) for [@akapug](https://github.com/akapug)
