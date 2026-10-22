# V4.1 on the RTX 6000 + Optane: every input of the 12.69 tok/s run

Our fastest DeepSeek V4.1 Flash run on this box decoded **12.69 tok/s** with a 4-token draft (n4) and 10.29 without
one (n0). A later sweep on the same build, layout and switches gave 12.71 / 10.29. This folder holds the inputs for
that run, so you can rebuild it. Each file below is the exact one the run used, unless its line says otherwise.

| file | what it is | how the run used it |
|---|---|---|
| [`build-engine.sh`](build-engine.sh) | builds [akapug/ik_llama.cpp](https://github.com/akapug/ik_llama.cpp) branch `tier-split` with our CMake flags, in the CUDA 12.6.3 container | the engine: `llama-server`, `llama-quantize` and the two tests |
| [`build-draft.sh`](build-draft.sh) | converts DeepSeek's draft head (DSpark, release shards 44-46) to GGUF, then makes its routed experts q2_K | `-md DeepSeek-V4.1-Flash-1017-DSpark-exps-Q2_K.gguf` |
| [`build-store.sh`](build-store.sh) | downloads the public Q8_0 GGUF, checks it, and splits it into hot MXFP4 and cold q2_K experts | `-m DeepSeek-V4.1-Flash-1017-hotmxfp4-coldq2k-00001-of-00010.gguf` |
| [`hot-alloc.json`](hot-alloc.json) | which experts of each layer stay MXFP4 (hot); about 3 bits per weight overall | input of `build-store.sh` |
| [`split.rules`](split.rules) | the quantize rule list the split script writes from those choices | `build-store.sh` checks that yours is the same |
| [`hot-experts.tsv`](hot-experts.tsv) | 420 experts (7.0 GiB) kept on the GPU, per layer | `LLAMA_HOT_EXPERTS` |
| [`placement.tsv`](placement.tsv) | where every CPU expert lives: hot ones split by rows over the two sockets, cold ones on the sockets or the Optane node, the draft's experts on socket 1 | `LLAMA_EXPERT_PLACEMENT` (the same bytes as the run's file) |
| [`launch.sh`](launch.sh) | the system settings, the environment and the `llama-server` command line | the launch, with `GGML_TIER_SPLIT` 8 |
| [`bench.py`](bench.py) | the benchmark client: warm-up, then each prompt once per phase, greedy, 300 tokens | the timing |
| [`prompts.json`](prompts.json) | three public coding-agent prompts | **not** the run's prompts, see below |
| [`SHA256SUMS`](SHA256SUMS) | hashes and sizes of the public sources, our store shards and our draft | checks |

## Rebuild it

```bash
OUT=$HOME/build/ik-tier-split ./build-engine.sh                 # about 30-60 min on 16 cores
QUANTIZE=$HOME/build/ik-tier-split/pkg/bin/llama-quantize OUT=$PWD/draft ./build-draft.sh
IK=$HOME/build/ik-tier-split/src QUANTIZE=$HOME/build/ik-tier-split/pkg/bin/llama-quantize \
  IMATRIX=<your imatrix .dat> OUT=$PWD/v41-1017-split ./build-store.sh
ENGINE=$HOME/build/ik-tier-split/pkg STORE=$PWD/v41-1017-split \
  DRAFT=$PWD/draft/DeepSeek-V4.1-Flash-1017-DSpark-exps-Q2_K.gguf ./launch.sh &
# after "server is listening" (the load takes about 45 min from Optane):
PORT=8790 N=300 PHASES=n4:4,n0:0,n4b:4 python3 bench.py
```

`launch.sh` is written for this box: two 10-core Xeons with 20 hardware threads each (NUMA nodes 0 and 1; CPU 39 runs
the tier split's copy thread, outside the 39 compute threads) and Optane as a CPU-less node 2. On another box, change `CPUS` and `T` and the node numbers in `placement.tsv`, and see
[the planner](../../recipe-maker/) for a layout that fits.

## Where your rebuild will differ from ours

- **The prompts.** Our three prompts came from our own coding-agent sessions and quote private code, so they stay
  private; the figures on this page were measured with them. `prompts.json` holds three public stand-ins of the same
  kind (a system line, a task, a few tool turns, cut where the agent starts writing code), 4.5-5.1k characters each
  against our 5.6-8.1k. We have not timed the public set on this box yet. On the public-fork build our three prompts
  gave 12.88, 12.62 and 10.13 tok/s: decode moves with the prompt, so expect a spread of that size.
- **The importance matrix, so the cold experts' bytes.** Ours (1.56 GB) came from one `llama-imatrix` pass of
  512 x 512 tokens over a text that includes private code, with a small local patch that keeps expert entries even
  when some experts saw no tokens. Neither the text nor the matrix is published. Any code-heavy imatrix of the 1017
  weights works with `build-store.sh`; your shards will have our sizes (in `SHA256SUMS`) but not our hashes, and the
  q2_K quality will be close but not equal. Lucebox publishes one
  ([`v41-1017-262144.dat`](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF/tree/30faf4aae57edae24420cec466fb9863f27feef1/imatrix));
  we did not use it, because its importance values correlated poorly with ours (median log correlation 0.17).
- **The quantizer build.** Our store and draft were quantized by an earlier build of the same fork (the same
  `split_experts.py`, byte for byte, but a `llama-quantize` from before the fork caught up with upstream). Expect the
  same sizes, not the same hashes.
- **Our harness.** Our launcher also checked each node's pages after the load (`"verify"` in the layout file) and
  stopped a run whose placement was off; it reported PLACEMENT PASS for this run. `launch.sh` leaves that out.
- **The engine.** `tier-split` holds the measured source with two kinds of edit: comments and a test flag that named
  our internal tracker are renamed, and a few lab build scripts are left out. No compiled code path changed.

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [arc](https://github.com/akapug/arc) + [fab](https://github.com/akapug/fab) for [@akapug](https://github.com/akapug)
