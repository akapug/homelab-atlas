# tail-trainer: a LoRA on the last blocks of a MoE GGUF, on any card torch runs on

You have a quantized MoE model as a GGUF (say Qwen3-30B-A3B at 4 bits, 17 GB) and an old or small GPU (an 11 GB
GTX 1080 Ti, a 16 GB Radeon Instinct MI25, a 24 GB Quadro RTX 6000, an Intel Arc). The whole model does not fit the card
in any trainable form. This tool trains a LoRA adapter on the model's **last few blocks only**:

1. **llama.cpp runs blocks 0..L-1 on the CPU**, quantized, exactly as it serves the model, and writes the residual
   stream that enters block L for every token of your text (`dump/tail-dump`, ~250 lines of C++ on llama.cpp's public
   API).
2. **torch holds blocks L..n-1 and the output head**, dequantized from the same GGUF (routed experts in host RAM if the
   card is small), and trains a LoRA on them from that residual.
3. **A gate checks the torch tail against the engine** before you train: on held-out text, the torch tail's next-token
   distribution must agree with llama.cpp's own logits (top-1 agreement >= 0.97, mean KL <= 0.02 nats).
4. **The adapter is a llama.cpp LoRA GGUF**: `llama-server -m MODEL.gguf --lora run/adapter.gguf`. The dump tool
   runs with the adapter too, so the gate also checks that the engine applies the adapter the way torch trained it.

Two cards can train two different tails at the same time from one dump (one card blocks 44..47, another blocks
40..43), and llama.cpp loads both adapters at once (`--lora a.gguf --lora b.gguf`), computing them exactly as
torch does. But two tails trained in parallel on the same text do not add up: measured, the pair is worse than
either one alone. For one combined model, train the second tail with the first frozen (`--lora a.gguf`); see below.

## How this differs from what exists (checked 2026-11-15)

| tool | trains | from a quantized GGUF | memory | cards |
|---|---|---|---|---|
| llama.cpp `llama-finetune` (examples/training) | full fine-tune | no: F32 models; "technically functional ... very much WIP" | whole model | CPU, CUDA |
| Tether QVAC Fabric (llama.cpp fork) | LoRA, whole model | Q4/Q8 bases; dense models tested (Qwen3 0.6-1.7B, Gemma 1B) | whole model | Vulkan, Metal, CUDA, CPU |
| transformers `from_pretrained(gguf_file=...)` + PEFT / Unsloth | LoRA, whole model | dequantizes the whole model to f16/f32 first; some quant types (IQ) unsupported | whole model at f16 | CUDA (ROCm/XPU partly) |
| **tail-trainer** | LoRA, last blocks only | any type gguf-py dequantizes; blocks before the tail stay quantized in llama.cpp | the tail only (experts may stay in RAM) | anything torch runs on: CUDA sm_61+, ROCm gfx900, XPU, CPU |

What you give up: only the tail learns. A tail adapter cannot change what the first L blocks compute, so it suits
changes the last blocks can express (output style, vocabulary, a format, recalibrating the last layers of a quant),
not new knowledge spread over the whole stack. One measured case where a tail was the wrong place: in our quant-repair
read on a 48-block model, 90 % of a 3-bit build's gap to its Q8_0 was already in the residual entering block 43.

## Supported models

The torch tail is built from the GGUF metadata (block count, widths, heads, experts, rope base, norm epsilon) and the
tensors present (biases, q/k norms per head or over the row, a shared expert). Architectures implemented from the
llama.cpp source: `qwen3moe` (gate measured, below), `qwen2moe`, `olmoe`, and `llama` with experts (Mixtral-style).
Only `qwen3moe` has passed the gate on a real model; for the others run the gate first: it is the check. Not
implemented (the tool refuses): dense models, sigmoid or grouped routing (DeepSeek), rope scaling (YaRN), per-layer
head counts, sliding-window attention.

## Steps

You need: a llama.cpp checkout (only to build the dump tool, CPU), CMake and a C++17 compiler, Python 3.10+, and a torch
for your card. Below, `TT` is this directory.

```sh
# 1. the dump tool, against stock llama.cpp (CPU build; nothing in the llama.cpp tree changes)
git clone https://github.com/ggml-org/llama.cpp
LLAMA_CPP=$PWD/llama.cpp bash $TT/dump/build.sh              # -> $TT/dump/bin/tail-dump

# 2. torch for your card + numpy + gguf-py
python3 -m venv venv && . venv/bin/activate
pip install torch numpy gguf                                 # recent NVIDIA cards; "Which torch" for the others
# the model of our runs: unsloth's Qwen3-30B-A3B Q4_K_S (17.5 GB; 48 blocks, 128 experts, 8 used)
hf download unsloth/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q4_K_S.gguf --local-dir . && ln -s Qwen3-30B-A3B-Q4_K_S.gguf MODEL.gguf
python3 $TT/tail_trainer.py info MODEL.gguf                  # what the tail will be, and the tensor types

# 3. text -> windows of 1024 tokens: the residual entering blocks 40 and 44 (44 is this tail; 40 is for the two-card
#    section below), and 32 rows of the engine's logits per window. Any text works; a first try with llama.cpp's own docs:
L=llama.cpp; cat $L/docs/build.md $L/docs/backend/SYCL.md $L/docs/backend/OPENVINO.md $L/tools/completion/README.md \
    $L/tools/cli/README.md $L/docs/speculative.md $L/docs/function-calling.md > train.txt
cp $L/tools/server/README.md heldout.txt
$TT/dump/bin/tail-dump -m MODEL.gguf --out dump-train --layers 40,44 --max-seqs 32 -t 16 --no-mmap train.txt
$TT/dump/bin/tail-dump -m MODEL.gguf --out dump-held  --layers 40,44 --max-seqs 8  -t 16 --no-mmap heldout.txt

# 4. the tail's weights (blocks 44..47 and the head), dequantized
python3 $TT/tail_trainer.py weights MODEL.gguf --first 44 --out w44

# 5. the gate: does the torch tail reproduce llama.cpp? (exit 0 = pass)
python3 $TT/tail_trainer.py gate --dump dump-held --weights w44 --out gate.json --device cuda

# 6. train (one window per step), then compare held-out loss with and without the adapter
python3 $TT/tail_trainer.py train --dump dump-train --weights w44 --out run --steps 20 --device cuda
python3 $TT/tail_trainer.py eval --dump dump-held --weights w44 --out eval-base.json --device cuda
python3 $TT/tail_trainer.py eval --dump dump-held --weights w44 --out eval-lora.json --device cuda --lora run/adapter.gguf

# 7. the engine with the adapter: dump again with --lora, and gate torch-with-adapter against it
$TT/dump/bin/tail-dump -m MODEL.gguf --out dump-held-lora --layers 44 --max-seqs 8 -t 16 --no-mmap \
    --lora run/adapter.gguf heldout.txt
python3 $TT/tail_trainer.py gate --dump dump-held-lora --weights w44 --out gate-lora.json --device cuda --lora run/adapter.gguf

# 8. serve it
llama-server -m MODEL.gguf --lora run/adapter.gguf
```

Notes:
- `--layers` takes several blocks (`--layers 40,44`): one dump then feeds tails of different lengths.
- `--no-mmap` reads the model into process memory. On a box with a slow NUMA node (Optane PMem, a far socket), run
  the dump under `numactl --interleave=<the DRAM nodes>`; with the default mmap the pages sit wherever the page cache
  put them (5x slower in our run).
- `--device`: `cuda`, `rocm` (= cuda: ROCm torch serves AMD cards through `torch.cuda`), `xpu`, `cpu`; `cuda:1` picks
  a card. `--match q8` (default) rounds activations as llama.cpp's CPU kernels do (8-bit blocks for quantized weights,
  f16 q/k/v); `--match none` is plain f32. Both are reported by the gate; use the one that passes with the lower KL.
- `train` writes `run/adapter.gguf` (f32 LoRA tensors, no alpha: llama.cpp applies it at `--lora` scale 1 exactly as
  trained) and `run/train.json` (per-step loss, seconds, peak memory). Defaults: rank 8, AdamW lr 5e-4, LoRA on
  attn_q/k/v/output of every tail block (`--targets o` for attn_output only, `attn+shexp` adds a shared expert),
  `--blocks A-B` to train only some of the tail's blocks.
- On NVIDIA cards the tool sets the torch allocator to `expandable_segments` (a 1080 Ti needs it to fit); it
  never does on ROCm, where on gfx900 that setting makes every allocation fail. Your own `PYTORCH_CUDA_ALLOC_CONF`
  wins.
- Every file is read once and dropped from the page cache (`posix_fadvise DONTNEED`): a box that keeps a dequantized
  expert stack both in the page cache and in process memory goes into reclaim.

### Two cards, two tails, one dump

```sh
# the dumps of step 3 hold both blocks' inputs (--layers 40,44)
# card 1 (any box): blocks 44..47
tail_trainer.py weights MODEL.gguf --first 44 --out w44
tail_trainer.py train --dump dump-train --weights w44 --out run44 --device cuda
# card 2 (another box, at the same time): blocks 40..43, with 44..47 frozen
tail_trainer.py weights MODEL.gguf --first 40 --out w40
tail_trainer.py train --dump dump-train --weights w40 --out run40 --device rocm --blocks 40-43
# both together, in torch and in the engine
tail_trainer.py eval --dump dump-held --weights w40 --out both.json --lora run44/adapter.gguf --lora run40/adapter.gguf
tail-dump ... --lora run44/adapter.gguf --lora run40/adapter.gguf heldout.txt
```

Each card trains its own tail on the base model's residual; neither sees the other's adapter while it trains.
Measured (Qwen3-30B-A3B, results README): each adapter alone cuts held-out CE by 0.14 nats; the two together are
worse than either (in the engine, worse than no adapter). Both learned to correct the same error, so loading both
corrects it twice. The engine is not at fault: its A+B dump matches torch with A+B (gate PASS).

So use parallel tails as a sweep (each adapter on its own, e.g. which blocks to train), and build one combined
adapter set sequentially, the second tail trained with the first frozen:

```sh
tail_trainer.py train --dump dump-train --weights w40 --out run40 --device cuda --blocks 40-43 \
    --lora run44/adapter.gguf
llama-server -m MODEL.gguf --lora run44/adapter.gguf --lora run40/adapter.gguf
```

Measured in torch and in llama.cpp: the pair matches the first adapter alone in torch (1.3870 against 1.3866) and is
slightly better in the engine (1.0747 against 1.0819), and it beats one adapter trained for as many steps (results
README, "Follow-up").

### One tail split across two cards

When the tail does not fit one card, train it jointly over two: the last stage (blocks B..n-1 and the head) listens,
the first stage (blocks L..B-1) connects; each box needs only its own blocks' weight files and the same dump.

```sh
# box 1 (card 1): blocks 44..47 + head
tail_trainer.py train --dump dump-train --weights w44 --out split-last --steps 80 --device cuda \
    --stage last --split-at 44 --peer 0.0.0.0:29561
# box 2 (card 2): blocks 40..43 (weights --first 40; blk.44+ and the head need not be copied there)
tail_trainer.py train --dump dump-train --weights w40 --out split-first --steps 80 --device rocm \
    --stage first --split-at 44 --peer BOX1:29561
llama-server -m MODEL.gguf --lora split-first/adapter.gguf --lora split-last/adapter.gguf
```

It computes what the whole tail on one card computes (measured: step losses within 0.0064 over 80 steps). One window
is in flight at a time, so a step costs both cards' time plus the hand-off: the second card adds memory, not speed.

## Which torch, per card

From `../tail-portable/` (install scripts, `probe_ops.py`; every op this tool uses is in the probe):

| card | torch | install | notes |
|---|---|---|---|
| NVIDIA Ampere and newer (sm_80+) | stable CUDA wheels | `pip install torch` | TF32 used for the optimizer steps only |
| NVIDIA Pascal (1080 Ti) / Turing (RTX 6000) | 2.9.1+cu126 | `install-cuda-legacy.sh` | cu128+ dropped sm_61; f32 |
| AMD MI25 / Vega 10 (gfx900) | TheRock 2.11.0+rocm7.13 multi-arch | `install-mi25-gfx900.sh` | the stable ROCm wheel has no gfx900 rocBLAS kernels |
| Intel Arc Pro B70 | 2.15.0+xpu with compute-runtime 26.40 | `install-b70-xpu.sh` | older runtime 26.05 + torch 2.13-2.15 gives wrong index ops |

Then `pip install gguf` in the same venv.

## Results

See [results/README.md](results/README.md): the stranger smoke on both cards, the two-card run and the compose
table, with the JSON of every run.

## Files

- `tail_trainer.py`: the CLI (info, weights, gate, train, eval) and the adapter GGUF reader/writer.
- `tail_arch.py`: the torch tail, built from the GGUF metadata; its docstring is the arithmetic, read from llama.cpp.
- `dump/`: `tail-dump.cpp`, `CMakeLists.txt` (builds against a llama.cpp checkout without changing it), `build.sh`.
- `test_tail_trainer.py`, `run_tests.sh`: tests on a tiny random GGUF, including a token-by-token reference of the
  arithmetic (`bash run_tests.sh` makes a CPU-torch venv and runs them).

## Rollback

Everything is in the directories you made (the venv, the dumps, the weights, the run). Delete them.
