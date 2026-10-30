# llama.cpp SYCL: attention sinks on the MKL flash-attention path, and a GPU fault on broadcast masks

Two commits to llama.cpp's SYCL backend, on upstream master `a34de8618b25` (build b12015, 2026-10-30), tested on one
Intel Arc Pro B70:

1. **A fix.** The MKL flash-attention path reads a mask out of bounds when the mask is broadcast over heads or
   batches. On master this faults the GPU (`UR_RESULT_ERROR_DEVICE_LOST`) with f16 K/V and returns NaN with q8_0 K/V.
2. **A feature.** Flash attention with attention sinks (gpt-oss, Flint) can now take the MKL path. Before, the kernel
   choice sent it to the TILE kernel. One attention op at a Flint prompt shape is 1.87x faster, and at a gpt-oss
   shape 1.32x.

- Branch: [akapug/llama.cpp `sycl-mkl-fa-sinks`](https://github.com/akapug/llama.cpp/tree/sycl-mkl-fa-sinks)
- Compare against upstream master: [ggml-org/llama.cpp `master...akapug:sycl-mkl-fa-sinks`](https://github.com/ggml-org/llama.cpp/compare/master...akapug:llama.cpp:sycl-mkl-fa-sinks)
- The same two commits as patch files: [`patches/`](patches/). They apply with `git am` on `a34de8618b25`.
- 3 files, +60/-5: `fattn-mkl.cpp` +14/-4, `fattn.cpp` +1/-1, `tests/test-backend-ops.cpp` +45.

Below the patch notes is the model that found both: [Flint on two B70s and on an RTX 3090 Ti](#flint-on-two-b70s-and-on-an-rtx-3090-ti).

## Conditions

| | |
|---|---|
| card | one Intel Arc Pro B70 (32 GB, Xe2), Linux `xe` driver, Level Zero |
| build and run | llama.cpp's own `full-intel` image, `ghcr.io/ggml-org/llama.cpp@sha256:77faff7d38b7...` (IntelLLVM 2026.1), [`build.sh`](build.sh): `GGML_SYCL=ON`, `GGML_SYCL_F16=ON`, Release |
| tests | `test-backend-ops test -b SYCL0 -o FLASH_ATTN_EXT`: the SYCL result against the CPU backend, through [`tbo.sh`](tbo.sh) |
| kernel path | llama.cpp's own `GGML_SYCL_MKL_FA_DEBUG=1` trace (`[FA-DISP]` lines), read by [`paths.py`](paths.py) |
| date | 2026-10-30 |

## The fix: a mask broadcast over heads or batches

`ggml_flash_attn_ext` accepts a mask whose `ne[2]` and `ne[3]` only divide the number of Q heads and batches. The CPU
backend reads it at `iq2 % mask->ne[2]` and `iq3 % mask->ne[3]`. The MKL path read mask head `kvh_head + gqa_group`
(when `ne[2] > 1`) and mask batch `ib` (when `ne[3] > 1`) directly. So a mask with `1 < ne[2] < n_head` or
`1 < ne[3] < n_batch` is read out of bounds. The fix takes both indices modulo the mask dims:

```diff
-                        ? (kvh_head + gqa_group) : 0;
+                        ? (kvh_head + gqa_group) % mask_n_heads : 0;
...
-            int m_batch = (mask->ne[3] > 1) ? ib : 0;
+            int m_batch = ib % (int)mask->ne[3];
```

**Reproduce on master.** Commit 1 adds `test_flash_attn_ext_mask_bcast`: mask `[1024, 32, 2, 2]`, Q of 32 heads (4 KV
heads x GQA 8) and 4 batches, kv 1024, 32 query rows, f16 and q8_0 K/V, sinks off and on. Build master with only this
branch's test file:

```bash
A=$PWD/homelab-atlas/b70/llama.cpp-sycl-fa-sinks   # this folder, cloned
git clone --depth 3 -b sycl-mkl-fa-sinks https://github.com/akapug/llama.cpp && cd llama.cpp
git checkout a34de8618b25 && git checkout sycl-mkl-fa-sinks -- tests/test-backend-ops.cpp
$A/build.sh .
export CARD=0000:03:00.0          # your GPU's PCI address, from ls /dev/dri/by-path
DBG=1 $A/tbo.sh build-pr/bin test 'sinks=0.*type_K=f16.*mask_ne23'  f16.log    # DEVICE_LOST, rc 134
DBG=1 $A/tbo.sh build-pr/bin test 'sinks=0.*type_K=q8_0.*mask_ne23' q8_0.log   # FAIL, NaN
git checkout -f sycl-mkl-fa-sinks && $A/build.sh .
DBG=1 $A/tbo.sh build-pr/bin test 'mask_ne23' bcast.log                        # 4/4 OK, all MKL
```

The first test faults the GPU context. On our card the other processes on it kept running, but use a card you can
spare.

| case | master | commit 1 | both commits |
|---|---|---|---|
| broadcast, sinks off, f16 K/V | MKL path, GPU page fault, `UR_RESULT_ERROR_DEVICE_LOST`, abort (rc 134) | OK (MKL) | OK (MKL) |
| broadcast, sinks off, q8_0 K/V | MKL path, FAIL: `NaN at index 512 (SYCL0=-nan CPU=0.005835)` | OK (MKL) | OK (MKL) |
| broadcast, sinks on, f16 and q8_0 | not supported | not supported | 2 of 2 OK (MKL) |

Logs: [`stock-bcast-sinks0-f16.log`](logs/stock-bcast-sinks0-f16.log),
[`stock-bcast-sinks0-q8_0.log`](logs/stock-bcast-sinks0-q8_0.log), [`c1-bcast.log`](logs/c1-bcast.log),
[`patched-bcast.log`](logs/patched-bcast.log). The kernel log of the f16 run shows `xe` page faults on the compute
engine, all from one address space. A llama-server on the same card kept serving through the fault.

**Who is affected.** Not llama-server, as far as we can tell from the source: `llama-graph.cpp` builds the KQ mask with
`ne[2] = 1` and `ne[3]` equal to the Q batch, and for those masks the modulo gives the same address as before. Other
ggml users that pass a broadcast mask can hit it. We did not check a server trace.

## The feature: attention sinks on the MKL path

A sink is one more key with a logit and no value. It adds `exp(sink - row max)` to the softmax denominator and
nothing to the numerator. The MKL path now adds that term in its final normalization (`mkl_fa_normalize_head`, which
already has the row max), and the kernel choice in `fattn.cpp` admits sinks when `sinks->nb[0] == sizeof(float)`
(ggml asserts F32 sinks). The TILE kernel adds the same term after its KV loop. Commit 2 needs commit 1: with sinks
supported, the broadcast cases above take the MKL path.

Before, any flash attention with sinks ran on TILE, at every depth: oneDNN SDPA excludes sinks too. With Flint,
setting `GGML_SYCL_FA_ONEDNN=0` and `GGML_SYCL_ENABLE_MKL_FA=0` left prompt reading at depth 48k unchanged
(483.1 / 483.3 tok/s against 482.6), because TILE ran either way.

**One attention op** (`test-backend-ops perf`, a prompt of 1,024 tokens after 49,152 cached ones, sinks on; 3 runs
each, master and the branch alternated; medians):

| shape | master (TILE) | branch (MKL) | speedup |
|---|---|---|---|
| Flint: head size 128, 32 Q / 4 KV heads, q8_0 K/V | 138.1 ms (140.3 / 137.9 / 138.1), 6.1 TFLOPS | 73.8 ms (74.5 / 73.8 / 73.7), 11.4 TFLOPS | **1.87x** |
| gpt-oss: head size 64, 64 Q / 8 KV heads, f16 K/V | 168.2 ms (168.2 / 191.6 / 165.8), 5.0 TFLOPS | 127.1 ms (129.2 / 126.5 / 127.1), 6.6 TFLOPS | **1.32x** |

During these runs a llama-server on the same card was decoding 2 streams (about 30 tok/s each), so the absolute times
are high. On a quieter card the Flint op took 54.2 ms on MKL (build b12004, same kernel code). Logs:
[`perf-stock-*.log`](logs/perf-stock-1.log), [`perf-patched-*.log`](logs/perf-patched-1.log); the path trace in
[`perf-dbg-stock.log`](logs/perf-dbg-stock.log) (TILE) and [`perf-dbg-patched.log`](logs/perf-dbg-patched.log) (MKL).

**End to end.** llama-bench with Flint Q4_K_M, FA on, ubatch 2,048, q8_0 K/V unless marked, one run each, tok/s.
This is **build b12004** (`3c809f443`) with the same sinks change, before the two modulo lines were added. Those lines cost no
measurable time: one Flint op at depth 48k took 54,230 / 54,255 / 54,254 µs with them and 54,217 / 54,228 / 54,213
without (+0.05 %). We did not run llama-bench on the new base: each card was serving a model.

| test | b12004 | b12004 + sinks on MKL |
|---|---|---|
| pp4096 at depth 0 | 1,922 (f16 K/V, an earlier pass) | 2,087 |
| pp4096 at depth 16k | 945 (batch 4,096) | 1,417 |
| pp4096 at depth 64k | 387 (batch 4,096) | 737 |
| pp1024 at depth 48k | **483** | **857** |
| tg64 at depth 48k | 51.4 | 51.7 |

## Tests

All on one B70, SYCL against the CPU backend. The master and branch test binaries are byte-identical (same test
source); only the SYCL library differs.

| cases | master `a34de86` | branch (both commits) |
|---|---|---|
| whole op, `-o FLASH_ATTN_EXT` (5,399 cases) | 4,120 / 4,120 OK, 1,275 not supported (the 4 broadcast cases left out: the first aborts the process) | **4,124 / 4,124 OK**, 1,275 not supported; the only changed results are the 4 broadcast cases |
| every case with sinks, `-p sinks=1` (2,621 cases) | 1,986 / 1,986 OK, 635 not supported | 1,988 / 1,988 OK, 633 not supported |
| kernel paths in that sinks sweep | TILE 1,770, DPAS 120, VEC 96 | TILE 1,754, DPAS 120, VEC 96, **MKL 18** |

- **Only 18 sinks cases change path:** the 16 new sinks cases (TILE -> MKL) and the 2 broadcast cases with sinks (not
  supported -> MKL). Every other sinks case misses one of the MKL path's other conditions (Q batch >= 32, kv >= 1024,
  GQA >= 2, head size, no ALiBi or softcap) and keeps its kernel. So the sinks sweep mostly re-checks TILE, DPAS and
  VEC; the new sink term on MKL is checked by those 18 cases plus the 2 sinks-off broadcast cases (MKL on both builds).
- The 16 new sinks cases: kv 1,024 and 16,384 (two of the MKL path's 8,192-row KV chunks), 32 and 512 query rows,
  f16 and q8_0 K/V, in two shapes: gpt-oss (head size 64, 8 KV heads, GQA 8) and Flint (head size 128, 4 KV heads,
  GQA 8). `init_tensor_uniform(-10, 10)` draws the sinks, so a missing sink term is an O(1) error.
- Summaries: [`full-op-summary.txt`](logs/full-op-summary.txt), [`sinks-sweep-paths.txt`](logs/sinks-sweep-paths.txt)
  (the 18 cases by name). The full sweep logs are not published (sha256 in those files); `tbo.sh` reruns them.
- 2 perf cases are added: the two shapes in the table above.

## Limits

- **One GPU family.** Tested on the Intel Arc Pro B70 (Xe2) only. The MKL path is not gated by architecture, so other
  SYCL devices take it too; they are not tested.
- **A sink far above the row max.** The term is `exp(sink - row max)` in fp32. If a sink exceeds the row max by more
  than about 88, it overflows to inf and the output row becomes 0, where the exact value is very small but not 0. The
  tests draw sinks from U(-10, 10) and do not reach that. The TILE kernel folds the sink into its running max and has
  no such limit.
- The end-to-end llama-bench numbers are from b12004 with the same kernel code, not from the branch's base. The
  op-level numbers are from the branch.
- gpt-oss: tested through the test-backend-ops shapes only. No gpt-oss model was run.

## Flint on two B70s and on an RTX 3090 Ti

[Flint](https://huggingface.co/bartowski/RecursiveSI_flint-GGUF) (RecursiveSI flint v1.0, a MoE of 39.8B parameters
with about 2.5B active; llama.cpp support merged in [#30928](https://github.com/ggml-org/llama.cpp/pull/30928)) has attention sinks in all 49 layers, 14 of them full
attention. We measured it as a coding-agent model on 2026-10-30. **Its coding quality is not measured.** On the B70,
three short prompts (a Python function, a bash pipeline, a planted off-by-one) were answered correctly. On the 3090 Ti,
two prompts were, and a third carried a false premise by mistake, which the IQ4_XS file named when given 8k tokens.

### Speed

| card | engine | weights, K/V | decode, tg128 | prompt reading, pp4096 |
|---|---|---|---|---|
| one Arc Pro B70 | llama.cpp b12004 SYCL, FA on, ubatch 2,048 | Q4_K_M, f16 | 92.3 (92.8 at ubatch 512) | 1,922; **2,087** with the sinks change and q8_0 K/V |
| one RTX 3090 Ti | llama.cpp b12004 CUDA (`server-cuda` image), FA on, 3 runs | UD-Q3_K_XL, f16 | **148.9** | 4,052 |
| one RTX 3090 Ti | the same | IQ4_XS, f16 | 146.1 | 4,398 |

On the B70, FA on is required: it gives +44 % decode, q8_0 K/V needs it, and with FA off a 64k-depth bench did not
finish in 15 min. Splitting one model over both B70s (`--split-mode layer`) gained nothing, so we run one server per
card.

### Serving recipe

**Two B70s** (Threadripper 3960X box): one llama-server per card, each pinned to its card, and a small proxy in front
that keeps each conversation on one card (our own, not published). Weights `bartowski/RecursiveSI_flint-GGUF`
`RecursiveSI_flint-Q4_K_M.gguf` (sha256 `ceec8347ea35...`). Per card, build b12004 with the sinks change on MKL:

```bash
llama-server -m RecursiveSI_flint-Q4_K_M.gguf -ngl 99 -fa on -c 262144 -np 2 \
  --cache-type-k q8_0 --cache-type-v q8_0 -ub 2048 -b 2048 --jinja --reasoning-budget 1024
```

That is 2 slots of 131,072 tokens and 29,616 of 32,656 MiB in use under load. f16 K/V decoded 10 % slower than q8_0
in the replay below and filled the card. `--cache-reuse` is turned off by llama-server for this model (SWA layers); prompts
are still reused from the slot's own cache and its SWA checkpoints.

**One RTX 3090 Ti** (Ryzen 9 9950X3D desktop), stock build b12004, for 2 agents:

```bash
llama-server -m flint-UD-Q3_K_XL.gguf -ngl 99 -fa on -ctk f16 -ctv f16 --jinja \
  -np 2 -c 163840 -no-kvu -cram 32768
```

On CUDA, f16 K/V matters: at depth 32k with 4 sequences, decode summed to 271.4 tok/s with f16 K/V against 170.1 with
q8_0 (llama-batched-bench). Keep the KV split per slot: unified KV (`-kvu`) at 4 agents stalled the other streams
33 % of the time against 3.5 % split.

### Replayed coding-agent sessions

The load: 16 recorded Claude Code subagent sessions of our own (long coding work, median prompt depth about 50k
tokens), replayed with their own prompts, their recorded tool time, and output capped at the recorded length; one
600 s window after a warm-up (B70: 120 s; 3090 Ti: 600 s). Flint ends its turns at about a quarter of the recorded
Claude length, so **turns per agent-minute** is the number to compare, with decode per request beside it.

| setup | agents | turns per agent-minute | decode per request (median) | time to first token (median) |
|---|---|---|---|---|
| one B70, master build | 1 | 8.1 | 65.0 | 2.2 s |
| one B70, sinks on MKL | 1 | **10.8** | 66.1 | 1.5 s |
| one B70, master build (2 per card, on both cards) | 2 | 4.6 | 31.0 | 3.4 s |
| one B70, sinks on MKL | 2 | **7.6** | 30.5 | 2.4 s |
| two B70s behind the proxy, sinks on MKL | 2 (one per card) | **10.45** (9.2 on master) | 60.8 | 1.8 s |
| one RTX 3090 Ti, IQ4_XS, q8_0 K/V, 1 x 128k | 1 | **15.9** | 87.3 | 0.6 s |
| one RTX 3090 Ti, UD-Q3_K_XL, f16 K/V, 2 x 80k | 2 | **12.9** | 89.2 | 1.4 s |
| one RTX 3090 Ti, UD-IQ3_XXS, f16 K/V, 4 x 80k | 4 | 6.55 | 42.2 | 1.65 s |

- **On the B70, prompt reading at depth is the limit, and the sinks change is the lever:** +33 % turns for one agent,
  +65 % for two on one card. The 3090 Ti reads prompts at 2.1k-2.6k tok/s at agent depth, about 4x the B70 before
  the change (564-600), and one agent there runs 15.9 turns a minute against 10.8 on a B70 with the change.
- **A second agent on a card adds no total work.** Each card's decode total stays near 60 tok/s on the B70 with one
  agent or two; the 3090 Ti runs 25.8 total turns per minute at 2 agents and 26.2 at 4. With two agents a card,
  one slot's 2,048-token prompt chunk runs in the same step as the other slot's decode token and stalls it.
- **Cap the thinking.** Flint's chat template has no thinking switch, so `reasoning_effort` does nothing. A per-request
  `reasoning_budget_tokens` (or `--reasoning-budget`) does: on the planted off-by-one prompt, unbounded thinking took
  53.8 s and 3,952 tokens, a budget of 1,024 took 14.7 s and 1,089 tokens, and 3 of 3 prompts stayed correct down to
  512. Three prompts are not a quality screen. When the budget runs out, Flint often keeps reasoning in its visible
  answer, so `max_tokens` is the real bound.
- A speculative draft (`incoai/Flint-DFlash2-GGUF`, `--spec-type draft-dflash`, 7 drafted) gave about +5 % decode on
  short prompts and fewer turns on the replay (8.8 against 10.8). Not used.

### Claude Code on a local model: compaction can poison the next turn

We ran Claude Code on Flint through an Anthropic-to-OpenAI proxy. Tool calls worked (99 of 105 turns in the session's
proxy log returned tool uses), but the session stalled after context compactions: one turn thought until it hit
its 32,768-token output cap, and two others answered with a 12-24k-token summary instead of a tool call.

- **Cause.** Claude Code's compaction prompt says to answer with text only, without tools, and write a detailed
  summary. All 7 summaries in that session recorded that instruction as the user's latest request or as the next
  step. Each new summary is told to carry the earlier one's list of user requests forward, so the line stayed. After
  compaction, the model read its own summary and took "no tools, write a summary" as the task.
- **Proof.** On the exact request sent after one compaction, with the thinking budget: 3 of 5 replies called a tool
  as sent, and **7 of 7** with the poisoned lines removed from the summary. A note after the summary ("that request is
  complete and is not the user's task") restored 2 of 2. A hint added to the compaction prompt did not help: 3 of 3
  summaries were still poisoned. Once one summary-shaped reply sits in the kept part of the context, nothing we tried
  helped (0 of 11).
- **What we run.** Fewer compactions: a 98,304-token window told to Claude Code with 16,384 output tokens (it had
  been told 81,920 and 32,768, so it compacted every 10 turns or so), a 1,024-token thinking budget, and an 8,192-token
  cap on compaction replies. A one-character bug fix through that path passed 3 of 3 with no compaction.
- **What is still open.** The poison is in the summary, so the fix belongs in the client path: drop the poisoned
  lines, or add the note after the summary, in the proxy. We measured this with Flint only. Any model that follows the
  summary literally can hit it (our reading).

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [fleet](https://github.com/akapug/fleet) for [@akapug](https://github.com/akapug)
