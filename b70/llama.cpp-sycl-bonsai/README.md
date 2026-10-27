# llama.cpp SYCL patches for Ternary Bonsai 2 27B on the Arc Pro B70

Patches to [PrismML's llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp), the fork that runs PrismML's
ternary Bonsai models, for its SYCL backend on Intel GPUs. Measured with Ternary Bonsai 2 27B in its PTQ1_0 format
(base-3 ternary weights, 1.75 bits a weight) on one Arc Pro B70. The model page with the results is
[Ternary Bonsai 2 27B on one B70](../bonsai-2-27b.md).

Two sets, for two base commits of the fork:

| set | base | what it is | use it for |
|---|---|---|---|
| [`patches/`](patches/) 0001-0019 | PR #235's head `42ce76d019` (2026-09-24), the first SYCL support for PTQ1_0 | our decode mat-vec, the rest of a decode step, long context | the long-context numbers on the model page |
| [`patches-hybrid/`](patches-hybrid/) 0001-0022 | `prism` at `a14c7de990`, the merge of PR #294 (2026-10-02), PrismML's XMX path for PTQ1_0 | the same 19 rebased, plus three: PTQ1_0 decode stays on our mat-vec, prompt chunks of 9 tokens or more go to PR #294's XMX path | the fastest build we measured at every prompt size |

Neither set applies to the fork's current `prism` branch; see [Upstream](#upstream). Kernel timings are on the B70
(608 GB/s, 24 MiB L2). Whole-model numbers are tok/s for one request unless a line says otherwise.

## What each patch does

Numbering of `patches/`; `patches-hybrid/` 0001-0019 are the same changes in the same order.

| patch | what it changes | measured on the B70 |
|---|---|---|
| `0001` | SET_ROWS into PTQ1_0 or PQ2_0 is declined instead of aborting. | test-backend-ops runs to the end |
| `0002` | Decode mat-vec. On its first single-token mat-vec, each PTQ1_0 weight moves in place to a layout of 16-row groups: one lane runs one row, and each weight plane is one 64-byte read. Prefill against a moved weight dequantizes the new layout. | see `0004` |
| `0003` | Test: a prompt batch against a weight a decode mat-vec already moved. | coverage only |
| `0004` | 6 operations per trit plane instead of 11 at 1-2 tokens, and a K split sized to one wave of the GPU. | 5,120x17,408, every weight read from memory: 84.2 → 38.4 µs at 1 token (83 % of 608 GB/s), 95.1 → 39.7 at 2, 143.1 → 42.8 at 4, 232.8 → 59.5 at 8. Whole model with `0001`-`0004`: 26.9 → 42.0 tok/s |
| `0005` | The model rotates activations with a 1,024-point Hadamard transform. The SYCL fast transform stopped at 512, so it ran as 258 f32 GEMMs a token. It now covers 1,024, with the sign flip fused in. | 4.7 → 2.3 µs a call |
| `0006` | A row-split BF16 mat-vec on f32 activations for 1-8 columns (the 5,120 x 48 DeltaNet alpha/beta projections). | 13.7 → 2.0 µs at one column |
| `0007` | f32 get_rows moves 16 bytes per work-item when rows allow (the 3 MB recurrent-state gather). Port of [`../llama.cpp-sycl/` 0006](../llama.cpp-sycl/). | 20.1 → 3.2 µs |
| `0008` | One flat kernel for non-contiguous concat with short rows (the convolution state). Port of `../llama.cpp-sycl/` 0007. | 5.6 → 3.1 µs |
| `0009`, `0010` | Decode flash attention: a finer KV split, and GQA 6 packed as 8 so each K/V head is read once. Ports of `../llama.cpp-sycl/` 0003 and 0010. | head dim 256, GQA 6, KV 256 / 4,096: 23.3 / 85.3 → 18.6 / 35.2 µs |
| `0011` | Tests: perf cases at the model's decode shapes; eval cases for `0005`, `0006`, `0010`. | coverage only |
| `0012` | CPY to or from TQ1_0 and TQ2_0 is declined instead of aborting. | test-backend-ops runs to the end |
| `0013` | The oneMKL flash-attention prompt path reads a quantized K/V view by the view's own strides. A correctness fix: a q8_0 case failed with NMSE about 1.3. | the case passes |
| `0014` | Tests: perf cases at depth (flash attention at KV 256-64k, 512-token prompt batches). | coverage only |
| `0015` | That prompt path ran one work-item per row and waited on the host after each step. Now one work-group per row, no host waits, f16 K/V used in place. | 512 tokens, GQA 6, head dim 256: KV 4k 6.09 → 1.69 ms, 16k 22.4 → 6.78, 32k 44.0 → 13.4 |
| `0016` | PTQ1_0 dequantizes to fp16 8 weights per work-item for the prompt GEMM. | 17,408x5,120 at 512 tokens: 1,561 → 1,144 µs |
| `0017` | The flash-attention tile kernel skips KV tiles the mask hides, such as another sequence's cells in a unified KV cache (`-kvu`). | 64k KV, 61,440 cells another sequence's: 544 → 66 µs |
| `0018` | Flash attention for up to 16 query rows per K/V head (decode, and MTP verify steps) on the XMX matrix engines. | 64k KV: 1 token 549 → 475 µs, 2 tokens 890 → 490 |
| `0019` | The XMX flash attention also reads q8_0 K/V, and up to 4 row groups (MTP drafts of 2-3 tokens). | 64k q8_0 KV, 1 token 5,474 → 548 µs; f16, 3 / 4 tokens 1,729 / 1,733 → 514 / 529 |
| `patches-hybrid/0020` | PR #294 rewrites a PTQ1_0 weight in place into its XMX layout, and then every mat-mul on it takes XMX, which loses decode. Now the weight keeps our layout, and the XMX path reads a separate copy made on the first prompt chunk of `GGML_SYCL_XMX_MIN_ROWS` tokens or more. | against the 19 patches without it: prompt of 32 tokens 137 → 574 tok/s; decode unchanged (53.2 → 53.5). The copies take 6.02 GiB more VRAM |
| `patches-hybrid/0021` | Tests: a batch, then a mat-vec, then the batch again on one weight; a 12-column batch. | coverage only |
| `patches-hybrid/0022` | The XMX copy serves from 9 tokens, not 16. At 9-15 the only other path was dequantize-then-GEMM, the slowest. | prompt of 9 tokens 40 → 260 tok/s, 12 tokens 53 → 335 |

`GGML_SYCL_ENABLE_OPT=0` at run time skips the weight move, so PTQ1_0 runs PR #235's own kernel: the "before" of
each before/after. In the hybrid build, `GGML_SYCL_DISABLE_XMX=1` keeps PTQ1_0 off XMX and `GGML_SYCL_XMX_MIN_ROWS`
moves the threshold.

The "every weight read from memory" timings come from a small tool that runs 8 distinct weights per graph.
`test-backend-ops perf` repeats one mat-mul, and a 19.5 MB PTQ1_0 weight fits the B70's 24 MiB L2, so it reads
mostly from cache and overstates PTQ1_0 (it gave our first, wrong reading of PR #235's kernel: 54 % of the
bandwidth, against 37 % from memory). The other kernel timings above are `test-backend-ops perf` (L2-warm; good for A
against B, not for absolute bandwidth).

## Correctness

`test-backend-ops test` against the CPU backend:
- `patches/` 0001-0019: MUL_MAT for PTQ1_0 102/102, MUL_MAT_ID 75/75, GET_ROWS 4/4, CPY 9/9,
  MUL_MAT_REORDER_THEN_BATCH 8/8, FLASH_ATTN_EXT 4,125/4,125. At 0001-0012 the whole MUL_MAT suite passed
  1,234/1,234; FLASH_ATTN_EXT was 4,090/4,091 (a q8_0-KV prompt case that `0013` fixed) and CPY 402/403 (f16 to q2_0
  rounding, in a kernel these patches do not touch).
- The 19 rebased onto `prism` `88c4bc60b`: MUL_MAT 1,409/1,409, reorder-then-batch 8/8.
- `patches-hybrid/` with 0020-0021 (before 0022, which only lowers the default threshold): MUL_MAT 1,409/1,409,
  MUL_MAT_ID 1,035/1,035, GET_ROWS 119/119; the PTQ1_0 MUL_MAT cases 167/167 with XMX on and off, and the 18
  batch-order cases 18/18 with the threshold at 1, 16 and 100,000.

Whole model: perplexity and greedy text on the [model page](../bonsai-2-27b.md#quality-checks).

## Build

```
./build.sh ~/bonsai-sycl hybrid           # or pr235; clone, apply, verify the source tree, build
./build.sh ~/bonsai-sycl pr235 --apply-only
```

`build.sh` checks that the patched tree hashes to the one we measured (`8964a8c0f2` for pr235, `247ee38674` for
hybrid), then builds `llama-server`, `llama-bench`, `llama-batched-bench`, `llama-perplexity` and `test-backend-ops`
with the cmake flags we ran (`GGML_SYCL=ON`, `GGML_SYCL_TARGET=INTEL`, `GGML_SYCL_F16=ON`, `GGML_SYCL_DNN=ON`,
`GGML_SYCL_GRAPH=ON`, Release). Both trees were verified from a fresh clone of the fork on 2026-10-27. We built with
oneAPI 2026.1.

With two cards in one box, pin the server to its card by PCI address:
`ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 ONEAPI_DEVICE_SELECTOR=level_zero:<n>`.

## Check and undo

- **Check.** `build.sh` prints `patched (...): ... (tree verified)`. Then
  `test-backend-ops test -b SYCL0 -o MUL_MAT -p ptq1_0` passes, and `llama-bench -m <model> -ngl 99 -fa on -p 512 -n 64`
  gives your own before and after (run it once with `GGML_SYCL_ENABLE_OPT=0` for the before).
- **Undo.** Build the fork without the patches.

## Upstream

None of these patches has been sent as a pull request. On 2026-10-01 we posted our B70 numbers on PR #294 with an
offer of our decode mat-vec as the path for 1-8 tokens
([comment](https://github.com/PrismML-Eng/llama.cpp/pull/294#issuecomment-5923270566)); there was no reply before
PR #294 merged on 2026-10-02. Since then, the fork's `prism` branch has merged changes that overlap ours, so the
patches would need more than a rebase:

| PrismML PR | merged | overlaps |
|---|---|---|
| [#278](https://github.com/PrismML-Eng/llama.cpp/pull/278): PTQ1_0 dequantized to FP16 for prompts | 2026-09-28 | the purpose of `0016` |
| [#294](https://github.com/PrismML-Eng/llama.cpp/pull/294): XMX path for PTQ1_0 and PQ2_0 | 2026-10-02 | `patches-hybrid/` is built on it |
| [#302](https://github.com/PrismML-Eng/llama.cpp/pull/302): fast Hadamard transform above 512 | 2026-10-02 | `0005` |
| [#356](https://github.com/PrismML-Eng/llama.cpp/pull/356): a port of the `arc-b580` branch of Torchit1/llama.cpp (TernSYCL XMX kernels, fused ops, XMX decode attention from a q4_0 KV cache, an MMVQ reorder layout) | 2026-10-14 | `0002`, `0004`, `0018`, `0019` in purpose; we have not compared the code |

Open there as of 2026-10-27: [#327](https://github.com/PrismML-Eng/llama.cpp/pull/327) keeps PTQ1_0 at 1.75 bits in
the XMX path (PR #294 expands it to 2 bits in VRAM, +25 %), and [#326](https://github.com/PrismML-Eng/llama.cpp/pull/326)
covers concat, get_rows and few-row f16/bf16 mat-vecs (the purpose of `0006`-`0008`). We have not built or measured
the current `prism` branch on the B70.

---

-- Claude Opus 5.5, working in [fleet](https://github.com/akapug/fleet) for [@akapug](https://github.com/akapug)
