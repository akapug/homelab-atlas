# V4.1 on the RTX 6000 + Optane: every input of the 12.69 tok/s run

Our fastest DeepSeek V4.1 Flash run on this box decoded **12.69 tok/s** with a 4-token draft (n4) and 10.29 without
one (n0), in a sweep whose two "off" arms framed it on the same load. A second load of the same build, layout and
switches gave 12.71 / 10.29; we keep 12.69 as the headline because its sweep is the one the RTX 6000 page tables. This folder holds the inputs for
that run, so you can rebuild it. The data files are the exact ones the run loaded; the scripts do what our lab kit
did, written out on their own.

| file | what it is | how the run used it |
|---|---|---|
| [`build-engine.sh`](build-engine.sh) | builds [akapug/ik_llama.cpp](https://github.com/akapug/ik_llama.cpp) branch `tier-split` at the pinned public commit `6d9992e2` (our measured private commit `945fa93b` with comments and one test flag renamed, published as `cde5fc20`, plus PATCHES.md; the same compiled code) with our CMake flags, in the CUDA 12.6.3 container | the engine: `llama-server`, `llama-quantize`, `llama-imatrix` and the two tests |
| [`build-draft.sh`](build-draft.sh) | converts DeepSeek's draft head (DSpark, release shards 44-46) to GGUF, then makes its routed experts q2_K | `-md DeepSeek-V4.1-Flash-1017-DSpark-exps-Q2_K.gguf` |
| [`build-store.sh`](build-store.sh) | downloads the public Q8_0 GGUF, checks it, and splits it into hot MXFP4 and cold q2_K experts | `-m DeepSeek-V4.1-Flash-1017-hotmxfp4-coldq2k-00001-of-00010.gguf` |
| [`hot-alloc.json`](hot-alloc.json) | which experts of each layer stay MXFP4 (hot); about 3 bits per weight overall | input of `build-store.sh` |
| [`split.rules`](split.rules) | the quantize rule list the split script writes from those choices | `build-store.sh` checks that yours is the same |
| [`hot-experts.tsv`](hot-experts.tsv) | 420 experts (7.0 GiB) kept on the GPU, per layer: 365 from the store's hot (MXFP4) partition and 55 cold (q2_K, or q3_K for down projections of layers 0-9) ones | `LLAMA_HOT_EXPERTS` |
| [`placement.tsv`](placement.tsv) | where every CPU expert lives (`hot` and `cold` in its tensor names are the store's MXFP4 and q2_K partitions, not GPU and CPU): hot ones split by rows over the two sockets, cold ones on the sockets or the Optane node, the draft's experts on socket 1 | `LLAMA_EXPERT_PLACEMENT` (the same bytes as the run's file) |
| [`launch.sh`](launch.sh) | the system settings, the environment and the `llama-server` command line | the launch, with the tier split at 8 threads (`split=8` in the `GGML_TIER_CTL` file) |
| [`bench.py`](bench.py) | a client that does what our lab client did: a 64-token warm-up and a 13,000-character prefill to touch every expert page, then each prompt once per phase, greedy, 300 tokens, no prompt cache | the timing (ours ran its phases from the lab kit, and its warm-up text was different) |
| [`recall-hot-experts.tsv`](recall-hot-experts.tsv), [`recall-placement.tsv`](recall-placement.tsv) | the 189k-token recall run's GPU hot list (129 experts, 2.2 GiB) and CPU placement, on the `prebind-f32-indexer` build | that run's `LLAMA_HOT_EXPERTS` and `LLAMA_EXPERT_PLACEMENT`, exact bytes; its haystack is our own notes and is not published |
| [`patch-imatrix-keep-partial.py`](patch-imatrix-keep-partial.py) | our patch to `llama-imatrix`: with `IK_IMATRIX_KEEP_PARTIAL=1` it keeps an expert tensor that stock ik would drop at 5 % unseen experts | our pass ran with it set (see below) |
| [`prompts.json`](prompts.json) | three public coding-agent prompts | **not** the run's prompts, see below |
| [`SHA256SUMS`](SHA256SUMS) | hashes and sizes of the public sources, our store shards and our draft | checks. Its middle column is the size, so `sha256sum -c` needs the other two: `awk 'length($1)==64 {print $1"  "$3}' SHA256SUMS \| sha256sum -c --ignore-missing` in the download folder (the build scripts do this) |

## Rebuild it

```bash
OUT=$HOME/build/ik-tier-split ./build-engine.sh                 # about 15 min on 16 cores (see below)
QUANTIZE=$HOME/build/ik-tier-split/pkg/bin/llama-quantize OUT=$PWD/draft ./build-draft.sh
# the Q8_0 source first, then an importance matrix of it (ours: 512 chunks of 512 tokens of code-heavy text, see below)
IK=$HOME/build/ik-tier-split/src QUANTIZE=$HOME/build/ik-tier-split/pkg/bin/llama-quantize FETCH_ONLY=1 ./build-store.sh
LD_LIBRARY_PATH=$HOME/build/ik-tier-split/pkg/lib $HOME/build/ik-tier-split/pkg/bin/llama-imatrix \
  -m v41-1017-q8/Q8_0/DeepSeek-V4.1-Flash-1017-Q8_0-00001-of-00010.gguf -f <your text> -o imatrix.dat \
  -c 512 -b 512 -ub 512 --chunks 512 -t 20 --no-mmap -dple -ngl 99 -ot exps=CPU -fa 1 --no-ppl   # our pass's settings
IK=$HOME/build/ik-tier-split/src QUANTIZE=$HOME/build/ik-tier-split/pkg/bin/llama-quantize \
  IMATRIX=$PWD/imatrix.dat OUT=$PWD/v41-1017-split ./build-store.sh
ENGINE=$HOME/build/ik-tier-split/pkg STORE=$PWD/v41-1017-split \
  DRAFT=$PWD/draft/DeepSeek-V4.1-Flash-1017-DSpark-exps-Q2_K.gguf ./launch.sh > server.log 2>&1 &
# after "server is listening" (about 45 min: the store is read from disk and placed over DRAM and PMem),
# check the placement landed where placement.tsv put it (exit 0 = on target), with the server's log:
python3 ../../recipe-maker/tools/place_verify.py server.log
PORT=8790 N=300 PHASES=n4:4,n0:0,n4b:4 python3 bench.py
```

**The host needs** an NVIDIA driver (every binary links `libcuda.so.1`, `llama-quantize` too), glibc 2.39 or newer
(Ubuntu 24.04 or later; the build container's), Docker for the two builds, Python 3 with numpy and PyYAML, the Hugging
Face CLI, `numactl`, about 1 TB of disk, and for the run itself the RAM and Optane of this box (or a layout from the planner).
`pkg/lib` carries the CUDA runtime, cuBLAS and libgomp.

Put the source (`SRCDIR`, default `./v41-1017-q8`) on an NVMe drive: the store's shards 8-10 are links to the
source's own, and the server reads the engram tables in 8 and 9 from the file while it decodes (our copy sat on a PCIe NVMe
drive). The other shards are read once at the load.

The tier split is switched by a file: `GGML_TIER_CTL` names it, and the server re-reads its one line
(`split=S prefetch=0|1 steal=0|1`) while it runs, so one load can time several settings: `launch.sh` writes it to
`./tier.ctl` (or `$TIER_CTL`), and you rewrite that line between `bench.py` runs. The 12.69 run had `split=8
prefetch=0`; `GGML_TIER_RING_MIB` and `GGML_TIER_PREFETCH_CPUS` size and place the prefetch ring, which that setting
leaves idle. The fork's [PATCHES.md on `tier-split`](https://github.com/akapug/ik_llama.cpp/blob/tier-split/PATCHES.md)
lists the switches that branch adds; the one on [`fa-tile-sinks`](https://github.com/akapug/ik_llama.cpp/blob/fa-tile-sinks/PATCHES.md),
a later branch with the same code underneath, lists every switch `launch.sh` sets (`GGML_MOE_GROUPS`,
`LLAMA_NO_HOST_WEIGHTS` and `LLAMA_HOT_EXPERTS` among them).

The `#` lines in `placement.tsv` are our generator's notes, kept because the file is the run's own bytes; the
server skips them. Their names are ours: `v41r1` is this placement, `b1017heal` an earlier load whose free memory
per node set one move, `DNODE` the draft's node (1), and the `.tsv`/`.json` files they cite are the generator's inputs
(the rules they produced are the file). In the layout file's label, `v41r1b` is this placement again and
`droopy-stack (stack cf2b2b86)` the engine switch set we first ran it with, on fork commit `cf2b2b86` (where
`expert-prebind` starts); the 12.69 run used the same switches on the `tier-split` build.

`launch.sh` is written for this box: two 10-core Xeons with 20 hardware threads each (NUMA nodes 0 and 1; the 39
compute threads leave CPU 39 for the tier split's prefetch, which this run had off) and Optane as a CPU-less node 2. On another box, change `CPUS` and `T` and the node numbers in `placement.tsv`, and see
[the planner](../../recipe-maker/) for a layout that fits.

We ran `build-engine.sh` as published on 2026-10-22 (16 cores, about 15 min): it built `6d9992e2` with rc 0. On a
host with the NVIDIA driver (our RTX 3090 Ti box: driver 595, glibc 2.42, a Zen 5 CPU that runs the Cascade Lake code)
`test-tier-split` passed all 16 arms with `LD_LIBRARY_PATH=pkg/lib`, and `llama-server` starts. `test-expert-place`
places pages on NUMA nodes 0-2, so it means something only on a box with those nodes, like the Optane box; the 3090 Ti box
has one node, so we did not count it. The other
scripts have not been run as written (they repeat steps our lab kit ran, which needs the 500 GB download and the box).

## Where your rebuild will differ from ours

- **The prompts.** Our three prompts came from our own coding-agent sessions and quote private code, so they stay
  private; the figures on this page were measured with them. `prompts.json` holds three public stand-ins of the same
  kind (a system line, a task, a few tool turns, cut where the agent starts writing code), 4.5-5.1k characters each
  against our 5.6-8.1k. We have not timed the public set on this box yet. On the public-fork build our three prompts
  gave 12.88, 12.62 and 10.13 tok/s: decode moves with the prompt, so expect a spread of that size.
- **The importance matrix, so the cold experts' bytes.** Ours (789 MB) came from one `llama-imatrix` pass of
  512 x 512 tokens over a text that includes private code, with a small local patch that keeps expert entries even
  when some experts saw no tokens (stock `llama-imatrix` drops a whole expert tensor when 5 % or more of its experts
  saw none). Neither the text nor the matrix is published. `build-store.sh` counts the routed-expert tensors in yours
  before it splits and stops if any are missing; a gate entry may be absent, because ik fuses gate with up and the
  split uses up's. At our size the stock tool keeps every tensor: after 512 x 512 tokens no tensor had more than 2.3 %
  of its 384 experts unseen (layer 39), under the 5 % limit. A shorter pass can lose tensors (one of ours lost layers
  34-39 at 12 % unseen). [`patch-imatrix-keep-partial.py`](patch-imatrix-keep-partial.py) is our patch: with
  `IK_IMATRIX_KEEP_PARTIAL=1` it keeps them, and it writes 0 for an unseen expert where stock ik writes a flat
  value, so the split gives that expert its tensor's mean. Our pass ran with it set. On this box a 256-chunk pass with these settings (the whole Q8_0 held in DRAM and PMem) took about 3 h, an 11-minute load included, so 512 chunks should take about
  twice that. Any code-heavy imatrix of the 1017 weights that passes the count
  works; your shards will have our sizes (in `SHA256SUMS`) but not our hashes, and the
  q2_K quality will be close but not equal. Lucebox publishes one
  ([`v41-1017-262144.dat`](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF/tree/30faf4aae57edae24420cec466fb9863f27feef1/imatrix));
  we did not use it, because its importance values correlated poorly with ours (median log correlation 0.17).
- **The quantizer build.** Our store and draft were quantized by an earlier build of the same fork (the same
  `split_experts.py`, byte for byte, but a `llama-quantize` from before the fork caught up with upstream). Expect the
  same sizes, not the same hashes.
- **Our harness.** Our launcher also checked each node's pages after the load and stopped a run whose placement was
  off; it reported PLACEMENT PASS for this run. That is what `"verify": "fail"` in the layout file means: on a failed
  check, fail the run. It is a setting, not a failed run. It also pinned every other task on
  the box to CPU 39 and paused maintenance timers for the load. `launch.sh` does neither.
- **A quality check.** Ours (KL against the release-precision store) ran on a harness that is not in this repo, so you
  cannot check a rebuilt store's q2_K experts against ours here.
- **The CPU.** `build-engine.sh` targets Cascade Lake (AVX-512 VNNI) and the RTX 6000 (sm_75). On another CPU or card,
  change `-march` and `CMAKE_CUDA_ARCHITECTURES`, or `llama-quantize` itself will not run.
- **The engine.** `tier-split` holds the measured source with two kinds of edit: comments and a test flag that named
  our internal tracker are renamed, and a few lab build scripts are left out. No compiled code path changed.

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [arc](https://github.com/akapug/arc) + [fab](https://github.com/akapug/fab) for [@akapug](https://github.com/akapug)
