# On Intel GPUs, decode can be limited by the CPU sending work, not by the GPU

On two Arc Pro B70s, llama.cpp's SYCL tensor split decoded a MoE model at half the speed of layer split. The cards
were not the limit. One CPU thread was building and submitting about 7,300 GPU kernels a token, one at a time, and
the cards sat idle about two thirds of each token waiting for it. Recording each card's work as a SYCL graph and
replaying it removes that cost: **16.9 -> 39.0 tok/s (2.3x), past layer split (32.2)**, with identical output. The
patch is [`sycl-graph-multi-device.patch`](sycl-graph-multi-device.patch). A different engine on the same box hit the
same wall a few days earlier ([below](#the-same-wall-in-another-engine)).

Every number here was measured on our machine unless it says CLAIMED (someone else's) or not measured. Numbers
marked "our reading" are interpretation.

## At a glance

| question | answer | details |
|---|---|---|
| Why does `-sm tensor` decode a MoE at half of `-sm layer` on two B70s? | The host submits 7,270 kernels a token (layer: 3,456) on one thread; each card's kernels fill 34 % of the token; the cross-card reduce is only ~10 % | [the profile](#the-profile-what-a-token-is-made-of) |
| Is it the CPU or the GPU's own per-command overhead? | The CPU: about 57 % of a token was host submission. With graphs the token is kernel time plus the reduce | [host against GPU](#host-against-gpu-how-the-two-were-separated) |
| Does a runtime setting fix it (immediate command lists, batched submission, the older Level Zero adapter)? | No: every one tried is neutral or slower | [settings](#settings-that-do-not-help) |
| What fixes it? | Per-device SYCL graphs: llama.cpp refused graphs whenever two GPUs were active; the patch lifts that. Tensor decode 16.93 -> 38.96 tok/s | [the patch](#the-patch-sycl-graphs-on-more-than-one-gpu) |
| Is the output the same? | Yes: KL divergence 0.00000, same top token 63 of 63, under both split modes | [output check](#output-check) |
| Does it help several agents at once (batched decode)? | Not yet: graphs are only recorded for one token at a time. At 2-8 streams tensor split with graphs equals stock tensor split, and layer split is faster | [batching](#batching-several-streams-at-once) |
| How do I tell if my own setup is host-bound? | Count kernels a token against the cards' busy share, and run a CPU-load A/B | [recipe](#how-to-tell-if-you-are-host-bound) |

## Our machine

Box C: AMD Threadripper 3960X (24 cores), 121 GiB RAM, two Intel Arc Pro B70 (32 GB, Xe2) on PCIe 4 x16, Linux 7.0
with the `xe` driver and Intel's 26.40 GPU packages on the host. Every run is in llama.cpp's own Docker image
`ghcr.io/ggml-org/llama.cpp:full-intel` (digest `sha256:77faff7d38b7...`), so the user-space GPU stack is the image's:
DPC++ 2026.1.1, the Level Zero GPU runtime `libze_intel_gpu` 1.15.38308, loader 1.32, and the Level Zero v2 adapter by
default, with llama.cpp built from upstream master
`e574a17b0` (2026-11-15) in that image (`-DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_F16=OFF
-DGGML_SYCL_DNN=ON -DGGML_SYCL_GRAPH=ON`). The model is Qwen3.8 Flash-Next UD-IQ3_XXS (a hybrid MoE, 512 experts,
48 blocks; about 27 GB on each card), all on the GPUs, no `-ot`. Decode is `llama-bench -ngl 99 -fa 1 -t 21 -p 0 -n
128`, one stream, f16 KV.

## The profile: what a token is made of

Measured before any patch, on the same box and build ([earlier notes](split-modes-and-backends.md) had read the source
and guessed the reduce):

| | layer split | tensor split |
|---|---|---|
| decode, tok/s (quiet box) | 32.1 | 15.8-16.9 |
| GPU kernels a token, both cards | 3,456 | **7,270** (2.1x) |
| kernel time a token (GPU timestamps) | 28.6 ms (the cards take turns) | 19.6 ms a card (in parallel) |
| token time | 31.2 ms | 57-59 ms |
| share of the token a card spends running kernels | ~92 % | **~34 %** |
| the thread that submits for both cards | 85 % of a core | **100 %** |
| decode lost when other jobs load the CPU (load ~5 -> 34-43) | -7 % (32.1 -> 29.8) | **-25 %** (16.9 -> 12.6) |

- **Why twice the kernels.** Tensor split cuts every matrix across both cards and adds a reduce after each split layer,
  about 96 a token. Mirrored work (the shared experts of this architecture, the norms and elementwise ops) runs on
  both cards. The routed-expert kernels themselves split cleanly (4.94 ms against 4.74 ms a token in layer split).
- **The reduce is not the cost.** A build that skips the cross-card exchange altogether (wrong output, a timing
  floor) decodes 17.51 against 15.75 for the stock host-staged exchange: the reduce is about 10 % of the token.
- **Peer-to-peer works on these cards.** Both B70s share one SYCL context with peer access both ways (no atomics).
  16 MB copies: pushed by the source card 28.4 GB/s, pulled by the destination 6.4 GB/s, through pinned host memory
  10.7 GB/s. 16 KB (one decode all-reduce): push 2.4 us, pull 5.4 us, host bounce 8.8-10.7 us. The exchange as llama.cpp
  does it (both directions plus the add) took 35.2 us through the host and 23.2 us by direct peer copy. A direct peer
  reduce matches the host-staged output exactly but changes the token time within noise, because the reduce was never
  the limit; we do not ship it.
- How it was measured: Intel's `unitrace` (device timing, started paused and resumed for 8 s of steady decode, so the
  model load is not traced; tracing slows decode ~4x but kernel durations are GPU timestamps), the `xe` driver's
  per-process engine counters, and the busiest thread's CPU ticks sampled once a second.

## Host against GPU: how the two were separated

Two separate reads, both untraced:

1. **Remove the host's per-kernel work and see what is left.** With per-device graphs (below) the token falls from
   59.1 ms (16.93 tok/s) to 25.7 ms (38.96). What remains is the 19.6 ms of kernel time a card plus about 6 ms, which is
   about what the reduce costs (6.4 ms). So about 33 ms, **57 % of a stock tensor-split token, was the host
   submitting kernels**, and the GPU's own per-command overhead is small.
2. **Profile the submitting thread** (`perf record -F 1999` for 10 s on the busiest thread during steady decode; its
   on-CPU share from `/proc/<pid>/task/<tid>/schedstat`):

| share of the thread's samples | stock tensor split | tensor split with graphs |
|---|---|---|
| on-CPU share of wall time | 99 % | 96 % |
| Level Zero GPU driver (`libze_intel_gpu`) | 36.9 % | 18.2 % |
| SYCL runtime, Unified Runtime adapter and loader | 27.6 % | 5.6 % |
| ggml and the app, oneMKL | 13.2 % | 2.3 % |
| libc, libstdc++ | 13.2 % | 3.6 % |
| Linux kernel | 8.0 % | **68.6 %** (syscall entry 42.8 %, `sched_yield`, the scheduler) |

Stock, the thread spends nearly all its time preparing and submitting work. With graphs it still runs 96 % of the
time, but mostly inside the driver's wait loop (`sched_yield` while it waits for the GPU). The host now waits on the
GPU, not the other way round.

A trap when you profile this: our first stock sample, taken 140 s after launch on a busy box, caught the model still
JIT-compiling its GPU kernels (Intel's graphics compiler `libigc` was 78 % of samples). Check the DSO list before you
trust the window; the rerun at 210 s is the one in the table.

## Settings that do not help

QFN, tensor split, `llama-bench -n 128 -r 5`, one container per setting, run interleaved:

| setting | decode tok/s (median of 5) |
|---|---|
| none (stock) | **16.93** |
| `UR_L0_V2_FORCE_BATCHED=1` (batched, not immediate, submission) | 14.74 |
| `SYCL_UR_USE_LEVEL_ZERO_V2=0` (the older v1 adapter) | 13.87 |
| v1 adapter + `UR_L0_USE_IMMEDIATE_COMMANDLISTS=0` | 13.80 |
| `UR_L0_SINGLE_THREAD_MODE=1` | 16.26 |
| `GGML_SYCL_ENABLE_GRAPH=1` without the patch (graphs refused) | 16.71 |
| `GGML_SYCL_ENABLE_GRAPH=1` with the patch | **38.96** |

`SYCL_CACHE_PERSISTENT=1` with a mounted cache directory crashed `llama-bench` in this image (exit 139, a segfault),
so we did not use it; it only caches compiled kernels between runs (load time), not decode.

## The patch: SYCL graphs on more than one GPU

llama.cpp's SYCL backend can record a computation as a SYCL graph and replay it in one submission (`-DGGML_SYCL_GRAPH=ON`
at build, `GGML_SYCL_ENABLE_GRAPH=1` at run time; off by default, since on one B70 its author measured a negligible gain,
CLAIMED in [PR 28725](https://github.com/ggml-org/llama.cpp/pull/28725)). But `check_graph_compatibility()`
(`ggml/src/ggml-sycl/ggml-sycl.cpp:6936` at `e574a17b0`) refused graphs whenever more than one GPU was active, with the
comment "A sycl_ex::command_graph object can only be created for a single device". So with two cards no graph was ever
recorded, under either split mode.

No graph here needs to span two devices. Each GPU's backend records its own graphs, on its own queue, from the part of
the computation it is given: the scheduler's per-GPU split under `-sm layer`, the meta backend's per-GPU subgraphs
under `-sm tensor` (`ggml/src/ggml-backend-meta.cpp:2532-2539`). The same function already refuses any node whose data
is not on that GPU. The patch drops the blanket refusal and keeps the old behaviour behind
`GGML_SYCL_GRAPH_SINGLE_DEVICE=1`. llama.cpp's CUDA backend has no such refusal. Under tensor split a decode token
becomes about 97 graph launches a card instead of about 3,600 kernel launches; the cross-card reduce stays outside the
graphs, between them.

- **For.** llama.cpp with the SYCL backend on two or more Intel GPUs. Measured on two B70s with `e574a17b0`.
- **Apply.** From the llama.cpp root at `e574a17b0`: `git apply sycl-graph-multi-device.patch` (or `patch -p1 <`).
  Build with `-DGGML_SYCL_GRAPH=ON`; run with `GGML_SYCL_ENABLE_GRAPH=1`.
- **Check.** With `-v`, the log says `enabling SYCL graphs` once a GPU and, during decode, no `disabling SYCL graphs ...
  multiple devices`. Decode tok/s against the same build with `GGML_SYCL_ENABLE_GRAPH=0`.
- **Undo.** Unset `GGML_SYCL_ENABLE_GRAPH`, or set `GGML_SYCL_GRAPH_SINGLE_DEVICE=1`, or build without the patch.

### Output check

`llama-perplexity --kl-divergence` over 128 tokens of llama.cpp's README (63 scored), `-ub 1` so every token is a
decode step (the path graphs replay), the same build with graphs off as the reference:

| split | reference PPL | with graphs PPL | KL divergence | same top token | time a pass, off -> on |
|---|---|---|---|---|---|
| `-sm tensor` | 1.4899 | 1.4899 | 0.00000 | 63 of 63 | 8.69 -> 4.50 s |
| `-sm layer` | 1.5288 | 1.5288 | 0.00000 | 63 of 63 | 4.52 -> 4.31 s |

### Speed

| arm | decode tg128, quiet | decode tg128, box at load 35-46 | prefill pp4096 (busy box) |
|---|---|---|---|
| layer split | 32.20 | 31.55 | 368.4 |
| layer split, graphs | 33.18 (+3 %) | 32.95 | 383.6 |
| tensor split | 16.93 | 15.34 | 214.4 |
| tensor split, graphs | **38.96** (+130 %) | **37.75** | 218.5 |

- Tensor split with graphs is now the fastest single-stream decode of this model on the two cards: +21 % over layer
  split.
- Graphs also remove most of tensor split's sensitivity to other CPU load: under load 35-46 the graph arm lost 3 %,
  stock tensor split 9 %.
- Prefill does not change: prompt matmuls go through oneDNN, whose calls cannot be recorded, and the log says so
  (`disabling SYCL graphs due to MUL_MAT using oneDNN`). Layer split keeps the faster prefill.
- The first repetition with graphs is slower (31.6 against ~39): graphs are recorded after two identical calls.

### A second model: Ornith-1.5-122B with experts on the CPU

Ornith-1.5-122B (a Qwen3.5-122B-A10B fine-tune), Q4_K_M, with the routed experts of 14 of its 49 layers (18-24 and
42-48) on the host CPU, so the model fits two 32 GB cards. Same build and settings as above, run on a quiet box (no
builds or other engines); median of three after one warm-up repetition, which is slow (8-10 tok/s) while the CPU
experts page in.

| arm | decode tg128 | prefill pp4096 |
|---|---|---|
| layer split | 36.00 | 249.1 |
| layer split, graphs | **37.23** (+3 %) | 261.3 |
| tensor split | 21.61 | 244.9 |
| tensor split, graphs | 34.46 (+59 %) | 238.8 |

- Graphs lift tensor split by 59 % here too, but layer split stays ahead. With experts on the CPU, part of every
  token is CPU work that graphs do not touch, so tensor split's head start on the GPU side is smaller.
- **Rule of thumb:** with experts on the host, use layer split plus graphs; with every expert on the cards, use
  tensor split plus graphs.
- We did not repeat the output check for this model; on Flash-Next the graph arms gave identical output.
- **These numbers are bound by the host's memory, not the B70s.** The 14 CPU-side layers live in 4-channel DDR4
  (a Threadripper 3960X with four DIMMs). A rough estimate: about 6 GB read per token at Q4_K_M with 10B active,
  roughly 1.5 GB of it from host RAM at about 80 GB/s, which is about 19 ms. The B70 side is about 4.5 GB at
  608 GB/s, about 7 ms. That puts most of each token on the CPU side, and 37 tok/s sits near that ceiling. This
  estimate is INFERRED, not profiled. A quant small enough to put every expert on the two cards (about 3 bits)
  is the test that shows what the B70s themselves can do.

## Batching: several streams at once

`llama-batched-bench -ngl 99 -fa on -kvu -c 8192 -npp 512 -ntg 128 -npl 1,2,4,8`, the box under other load (22-34).
Decode tok/s summed over the B streams:

| arm | B=1 | B=2 | B=4 | B=8 | a stream at B=8 |
|---|---|---|---|---|---|
| layer split | 30.86 | 32.27 | 44.86 | **52.47** | 6.6 |
| layer split, graphs | 28.95 | 29.60 | 46.07 | **53.59** | 6.7 |
| tensor split | 15.54 | 15.34 | 22.74 | 32.12 | 4.0 |
| tensor split, graphs | **30.38** | 15.82 | 23.82 | 33.04 | 4.1 |

- **The gain is for one stream only.** From two streams up, tensor split with graphs equals stock tensor split. Our
  reading: with more than one token a matmul leaves the matrix-vector kernels for oneDNN, which graphs cannot record
  (the same refusal the prefill runs log), so each step falls back to one launch a kernel. Recording graphs for small
  batches is the next fix for many agents; until then, layer split is faster from two streams up.
- Recording cost: no separate stall shows. B=1 with graphs (30.4) sits below the 39.0 of `llama-bench`; our reading is
  that its 128 steps carry the recording and the two warm-up calls, and the box was busier.
- The first batch-2 prompt took 46-54 s in every arm (new kernel shapes being compiled), the same with and without
  graphs.

**KV room.** Both split modes already divide the KV cache between the cards (layer split by layer, tensor split by
head), about 12-14 KB a token a card on this model. Tensor split starts higher on each card (by 0.2 and 1.1 GB at an
8k context: mirrored shared experts and the meta backend's buffers), so from per-card VRAM at 8k and 131k contexts,
extended in a straight line (our reading): about 413k KV tokens fit under layer split and 326k under tensor split. One
full 262k window fits either way; as 64k slots, 6 against 5.

## The same wall in another engine

Days earlier, the [XeStrata](https://github.com/MistVVK/XeStrata) engine we run for many agents
([many-agents.md](many-agents.md)) showed the same limit from a different angle. One engine per B70, a coder model, a
batched mode that left the expert cache slightly short of holding every expert:

- With every expert resident, the engine captures one GPU graph a decode step that asks the host nothing. With the
  cache short, each step runs a host-planned graph: for every layer the host publishes a plan, a PCIe share and a CPU
  share, and the GPU waits for each, **48 host handshakes a decode window**.
- The engine's host thread spins on those flags, pinned to the first core of its allowed CPU set. Both engines ran in
  containers with the same CPU set, so **both pinned their host thread to CPU 0**.
- Alone, a step took 24.6-25.0 ms. With both engines decoding at once it took 39.8-42.6 ms, nearly all of the extra in
  waiting for the host to publish. Either cure brought it back: every expert resident (no handshakes) 24.7-25.2 ms, or
  disjoint CPU sets (each host thread on its own core) 24.9-25.3 ms. A setting that pins each engine's host thread to
  its own core by GPU index fixed it in place (24.8-25.2 ms with both decoding).

Two engines, two code bases, one lesson: on this box, decode speed depends on one host thread getting a core to itself
and on how often the GPU has to wait for it.

## What it implies

- **Graph capture and fewer launches first.** On Intel GPUs, before tuning kernels, count launches. A decode step of
  thousands of small kernels is limited by the submitting thread, and runtime knobs do not change that; recording and
  replaying does.
- **Give the submitting thread its own fast core.** Isolate it from build jobs and other engines, and do not let two
  engines pin to the same core. A box that also runs builds needs this most.
- **Single-thread CPU speed matters on Intel hosts.** While submission dominates, a faster core is faster decode; the
  host-bound mode lost 25 % to background CPU load where the GPU-bound mode lost 7 %.
- **MoE expert dispatch multiplies launches.** Routing, gathering and per-expert kernels add many small launches a
  layer; a MoE is more exposed than a dense model of the same speed.
- **Layer split hides it, tensor split exposes it.** Layer split runs each layer once on one card (half the kernels,
  cards taking turns); tensor split runs every layer on both cards, so the host submits twice as much in the same time.
  A tensor-split result that loses to layer split may be measuring the host, not the interconnect.

## How to tell if you are host-bound

1. **Count kernels against busy time.** Trace a few seconds of steady decode with `unitrace -d --start-paused
   --session s1` and resume it once the model is loaded (`unitrace --resume s1`, then `--pause s1`). Divide the
   kernel count by the tokens in the window, and the summed kernel time by the window. Thousands of kernels a token with
   the cards busy well under the token time (here 34 %) point at the host.
2. **Run a CPU-load A/B.** Decode once on a quiet box and once with other jobs loading the CPU. A host-bound setup loses
   far more (here -25 % against -7 %).
3. **Look at the submitting thread.** Its CPU share near 100 %, with samples in the GPU driver, the SYCL runtime and
   the Unified Runtime adapter, means submission. Samples in `sched_yield` and syscall entry mean it is waiting on the
   GPU (not host-bound). Make sure the window is past the kernel JIT (`libigc` in the samples).
4. **Try graphs.** If your engine can record graphs, compare decode with them on and off at the same settings; the
   difference is your host share.

## Open questions

- **CUDA.** llama.cpp's CUDA backend has no multi-GPU refusal, so it should record per-GPU graphs under tensor split
  (not checked). Is tensor split on NVIDIA cards host-bound without them, and how close is the B70 with graphs to a CUDA card of the same bandwidth? Not measured.
- **vLLM on Intel (XPU).** vLLM's compiled mode records graphs; we have not profiled its submitting thread on the B70,
  or checked whether its eager mode shows the same limit.
- **oneDNN graph recording.** Batched decode and prefill fall back because oneDNN calls cannot be recorded. Whether
  oneDNN's own graph API, or routing small batches through the matrix-vector kernels, lets B=2-8 replay is the next
  question for many agents.
- **Fewer launches without graphs.** Fused kernels (for example llama.cpp's open fused add + norm work for SYCL) cut
  the kernel count directly; how much they add on top of graphs is not measured.

## Files

- [`sycl-graph-multi-device.patch`](sycl-graph-multi-device.patch): the patch, on llama.cpp `e574a17b0`.

---

-- Claude Opus 5.5, working in [fleet](https://github.com/akapug/fleet) for [@akapug](https://github.com/akapug)
