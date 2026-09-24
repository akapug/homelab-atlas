# llama.cpp SYCL decode patches for the Arc Pro B70

Eighteen patches to llama.cpp (fifteen to the SYCL backend, one of tests, two to speculative
decoding) that make decode faster on Intel Xe2: single stream, several streams, and at long
context. Measured with Qwen3.8-27B `UD-Q4_K_XL` (unsloth's GGUF, a mix of Q4_K, Q5_K, Q6_K,
IQ4_XS and IQ4_NL) on one Arc Pro B70. Together they take four-stream decode from 39.3 tok/s on
the unpatched base to 64 without speculation and 101 with speculation from the model's MTP head.
One stream decodes at 24.4 tok/s without speculation (~70 % of what the card's bandwidth allows)
and 44 with it.

Most patches are small. They are not all independent (0008 calls a helper 0004 adds, and the XMX
patches need 0008 and 0009). Kernel timings are `test-backend-ops perf` on the B70; end-to-end
numbers are aggregate decode tok/s from [`../bench/decode-streams.py`](../bench/decode-streams.py)
at ~1.5k ("short") and ~16k tokens of context.

| patch | what it changes | measured on the B70 |
|---|---|---|
| `0001` Q5_K | The multi-column mat-vec re-unpacked each Q5_K weight block once per stream. It now unpacks once and applies the result to every stream, the shape upstream merged for Q4_K in #27062. | kernel 1.46× at 4 streams, unchanged at 1 |
| `0002` IQ4_NL | IQ4_NL had no multi-column kernel, so a batch of n tokens ran the single-column kernel n times. This adds one that decodes each block once. | kernel 1.91× / 3.46× / 6.18× at 2 / 4 / 8 streams |
| `0003` flash attention | For decode, the KV range is split into far more work-groups: at least `min(KV tiles, 128)` pieces instead of a split sized to 192 work-groups in total. | attention 1.07–1.55× at depth or with 2+ streams; 0.91× at 2k with one stream |
| `0004` IQ4 lookup | IQ4_NL and IQ4_XS decode each 4-bit code through a 16-entry table. The table lookups were eight byte gathers per 32-bit word, and those gathers, not DRAM, capped the kernels (IQ4_XS at 325 GB/s; the same kernel without a lookup runs at 585). The table now lives in four 32-bit constants and each lookup is two selects, a shift and a mask. | IQ4_XS 1.08–1.16× at one token, 1.27× at four; IQ4_NL 1.24× |
| `0005` batch fold | Decode over several sequences hands mul_mat one token per sequence (`[k, 1, n_seq]`), e.g. a DeltaNet layer's output projection. The backend ran that as n_seq single-column mat-vecs. For a contiguous 2-D quantized weight it is now folded into one multi-column mat-vec. | `[6144, 1, 4]` against 5120×6144: Q5_K 320 → 73 µs, Q6_K 848 → 76 µs (equal to `[6144, 4]`) |
| `0006` get_rows f32 | f32 get_rows moved one float per work-item, recomputing the row index and pointers for each; a recurrent model's per-sequence state (3 MB per layer) copied at a fraction of bandwidth. With aligned rows it now moves 16 bytes per work-item. | 786,432-float rows: 1 row 20.2 → 3.3 µs, 4 rows 78.2 → 10.3 µs |
| `0007` concat | Non-contiguous concat launched one work-group per row; for rows shorter than a sub-group (a conv state of 4 floats per channel) most lanes idled. One flat kernel over all elements instead. | `[3, 10240, n]` + `[1, 10240, n]`: n=1 5.7 → 3.2 µs, n=4 18.3 → 9.6 µs |
| `0008` IQ4_NL layout | IQ4_NL had no reordered layout. Its block is byte-for-byte Q4_0's, so Q4_0's reorder moves it, and the reordered mat-vec and dequantize read it with the table in place of `- 8`. IQ4_NL also joins the tensor-extra list; without that, the reorder silently never engages. | 17,408×5,120 one token 420 → 144 µs; 4,096×14,336 at 1/4/8 tokens about 3× |
| `0009` tests | Two eval gaps: the fused FFN mat-vec at IQ4_XS and Q5_K (base_types lacks both), and a batch against a weight an earlier single-token mat-vec reordered (prefill after decode). | coverage only |
| `0010` GQA packing | The flash-attention tile kernel packs the Q heads sharing a K/V head by the largest power of two dividing the GQA ratio, so GQA 6 was packed 2 at a time and read every K/V head 3 times. It now rounds up (3 → 4; 5–7 → 8), indexes blocks by (sequence, K/V head, head group) and skips the idle columns. Adds 96 eval cases at GQA 3, 5, 6, 7. | GQA 6 decode attention 1.8–2.4× (16k: 372 → 187 µs; 4 sequences 1,181 → 588 µs); GQA 2, 4, 8 unchanged |
| `0011` XMX mat-mul | At 3-8 streams the dp4a mat-vec kernels turn compute bound (Q5_K 5,120×17,408: 108 µs at 1 column, 235 at 4, 415 at 8, against a ~104 µs read). Q5_K weights at 3-8 columns now run on the XMX matrix engines: each lane dequantizes its own row to fp16 straight into the B operand, against fp16 activations padded to 8 rows, with K split over work-groups when rows are few. The fused gate/up + GLU dp4a path hands Q5_K pairs over from 4 columns. `GGML_SYCL_XMX_MMVQ=0` turns it off. | that weight 188 / 235 / 415 → 146 / 147 / 150 µs at 3 / 4 / 8 columns; 17,408×5,120 at 4 / 8: 186 / 373 → 157 / 162 |
| `0012` XMX types | The XMX path extends to Q4_K and Q6_K (reorder layouts; Q6_K includes the 248,320-row output head) and to IQ4_XS (plain blocks; codes decoded through a 256-entry fp16 pair table in shared memory). At most 1,280 row groups run at once and the sub-groups walk the rest. The path now starts at 4 columns (at 3 a whole model decoded 3% slower). IQ4_XS gate/up pairs keep the fused dp4a kernel. | at 4 / 8 columns: Q4_K 5,120×17,408 194 / 389 → 115 / 117 µs; Q6_K the same shape 237 / 398 → 151 / 153; Q6_K output head 2,882 / 4,794 → 2,317 / 2,343; IQ4_XS down 190 / 259 → 175 / 177 |
| `0013` IQ4_XS layout | IQ4_XS gets a reorder layout ([qs 128][d, scales_h, scales_l]), which leaves the codes line-aligned. Only the XMX mat-mul reads it, so a weight moves to it only where XMX will serve it at every column count (a large 2-D weight, rows in groups of 16, a device with XMX); a reordered IQ4_XS reaching the dp4a dispatch asserts. Prefill dequantizes the new layout. Weights of 32M values or more only: Qwen3-4B's 25M-value ones decoded 0.8% slower at one stream. IQ4_XS gate/up pairs keep the fused dp4a kernel. | 5,120×17,408 at 1 / 4 / 8 columns 145 / 185 / 259 → 127 / 129 / 131 µs; 17,408×5,120 137 / 177 / 245 → 137 / 139 / 141 |
| `0014` Q3_K, IQ3_S | XMX takes Q3_K (reorder layout; 3-bit codes through the fp16 trick) and IQ3_S (plain 110-byte blocks, 2-byte aligned; its 512-entry grid in shared memory as fp16 pairs, signs by xor). The fewest columns now depend on the type: IQ3_S from 1 (its dp4a kernel runs at a fifth of the read rate), Q3_K from 2 (its multi-column dp4a kernel costs 3× its single column at 2). | Q3_K 17,408×5,120 at 2 / 4 / 8 columns 321 / 366 / 483 → 123 / 123 / 125 µs; IQ3_S 5,120×17,408 at 1 / 4 / 8: 329 / 1,314 / 2,648 → 250 / 253 / 254 |
| `0015` MTP draft limit | The MTP drafter stopped a sequence only at the global `--spec-draft-n-max` and ignored the per-sequence limit the server sets (context space, tokens left, and now the cap below): it drafted up to n_max for every slot and discarded the rest. It now stops at either, as the simple drafter does. | with 1 draft per slot allowed and n_max 3, 4 streams 80.0–80.6 → 87.0–87.5 tok/s |
| `0016` `--spec-batch-max` | A server flag capping the tokens a speculative step verifies: each active slot drafts at most N / slots − 1. With N = 8, the model drafts 3 at one stream and 1 at four, so every step stays within the ≤ 8-column XMX kernels; uncapped, 3 drafts put 4 streams at 16 tokens a step, onto dequantize-then-GEMM. | 1 stream 34.6 (1 draft) → 43.7 tok/s; 4 streams 88.0 → 87.3 (uncapped 3 drafts: 2.8) |
| `0017` XMX for IQ4_NL, and large IQ4_XS gate/up pairs | IQ4_NL joins the XMX mat-mul in its reorder layout, with IQ4_XS's pair table; IQ4_XS gate/up pairs large enough for their reorder layout leave the fused dp4a GLU kernel from 4 columns, since speculation now makes even one stream verify 4-8 tokens. IQ4_NL 17408x5120 at 4 / 8 columns: 184 / 228 → 140 / 142 µs. | 4 streams 86.6–87.3 → 91.2–91.5 tok/s short, 77.4–77.7 → 81.5–82.1 at ~16k; 1 stream unchanged (43–45 / 36) |
| `0018` XMX for 9-32 columns | The XMX mat-mul multiplies each dequantized weight tile against 2-4 activation tiles of 8 rows, so a step of 9-32 tokens (speculative verify at several streams, or many streams) no longer falls to MMQ or dequantize-then-GEMM; the reorder bootstrap now allows up to 32 columns. 17408x5120 at 16 / 32 columns: Q5_K 893 / 903 → 190 / 316 µs, Q4_K 1191 / 1200 → 147 / 198, IQ4_XS 2052 / 2064 → 164 / 199. | with the MTP step cap raised 8 → 16: 4 streams 91.2–91.5 → 100.7–101.6 tok/s short, 81.5–82.1 → 91.6–91.7 at ~16k; 1 stream unchanged (43–44 / 36) |

End to end, 4-stream decode on the model is 45.4 tok/s with the first two. The
unpatched base commit gave 39.3, and an older upstream build
36.1. The third pays at depth: at ~16k, 4 streams 36.3 → 39.3 tok/s.
The fourth: single stream 21.8 → 22.8 tok/s shallow and 19.4 → 20.1 at ~16k;
4 streams 47.5 → 48.3 shallow and 39.3 → 40.3 at ~16k. The fifth, at 4 streams:
48.3 → 50.8–51.8 shallow and 40.3 → 41.2–41.7 at ~16k; one stream unchanged. The
sixth and seventh together: 4 streams → 53.2–53.6 shallow and 43.0–43.1 at ~16k; one
stream → 23.0–23.2 shallow and 20.4 at ~16k. The eighth: one stream → 23.8–24.0
shallow and 21.0 at ~16k; 4 streams → 54.5–54.9 and 43.8. The tenth: one stream → 24.1–24.4
shallow and 22.5 at ~16k; 4 streams → 55.3–56.0 and 49.1–49.2. The eleventh: 4 streams → 59.6–60.1
shallow and 53.0–53.3 at ~16k; one stream 24.0 and 22.5–22.6. The twelfth: 4 streams → 61.2–62.3
and 54.8–54.9; one stream 24.1 and 22.3–22.6. The thirteenth: 4 streams → 63.4 and 54.9–55.8;
one stream 24.3–24.4 and 22.7. The fourteenth: 4 streams → 63.9–64.5 and 57.2–57.3; one stream
unchanged. With speculation from the model's own MTP head (`--spec-type draft-mtp
--spec-draft-n-max 3 --spec-batch-max 8`, needing 0015 and 0016): one stream
42.7–43.7 shallow and 37.0–39.2 at ~16k; 4 streams 86.6–87.3 and 77.4–77.7.

**Correctness.** `test-backend-ops test` against the CPU. MUL_MAT and
MUL_MAT_ID pass for Q5_K, IQ4_NL and IQ4_XS at n = 1 to 8. The fused FFN
gate/up + SWIGLU path passes 16 of 16 cases added for IQ4_XS and Q5_K (1–8
tokens, even and odd block counts); upstream's fusion tests do not cover those types.
With the batch fold, MUL_MAT passes 1,126 of 1,126; GET_ROWS 216 of 216 and
CONCAT 192 of 192 with patches 6 and 7. With all ten: MUL_MAT 1,126/1,126, fused FFN
1,281/1,281, reorder-then-batch 12/12, FLASH_ATTN_EXT 4,121/4,123 (the 2 are upstream's).
With all eleven: MUL_MAT 1,134/1,134 and fused FFN 1,293/1,293. With all twelve: MUL_MAT 1,158/1,158, fused FFN
1,329/1,329, MUL_MAT_ID 933/935 (the 2 are Q8_0 at activations of 1e5, NaN on the build before
the XMX patches too). With all thirteen: MUL_MAT 1,161/1,161, fused FFN 1,329/1,329,
reorder-then-batch 14/14. With all fourteen: MUL_MAT 1,189/1,189, fused FFN 1,353/1,353,
reorder-then-batch 14/14, MUL_MAT_ID 932/935 (the 3 are the Q8_0 1e5-activation cases, which pass
0-1 of 3 on the build before any XMX patch too). FLASH_ATTN_EXT passes 4,025 of
4,027. The 2 failures are prefill cases that also fail on unmodified upstream at
the base commit, so the patch did not cause them. Each deploy was also gated
on a task score under 4-way concurrency, because only concurrent requests run
the batched kernels.

## Build

```
./build.sh ~/llama-sycl              # clone, apply, verify the source tree, build llama-server
./build.sh ~/llama-sycl --apply-only
```

The patches apply to upstream `94256114c2` (2026-09-23). `build.sh` checks that the patched source
tree hashes to `35581c4e65`, the tree we measured, and builds with the cmake flags we ran. How we
served it (one B70, four slots of 40,960 tokens, MTP speculation capped at 16 tokens a step):

```
llama-server -m Qwen3.8-27B-UD-Q4_K_XL.gguf --device SYCL0 -ngl 99 -c 163840 --parallel 4 \
  --cont-batching -fa on --ubatch-size 2048 --jinja \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-batch-max 16
```

With two cards in one box, pin the server to its card by PCI address rather than by device index
or free memory: `ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 ONEAPI_DEVICE_SELECTOR=level_zero:<n>` makes the
Level Zero index follow the PCI bus, and `sycl-ls --verbose` shows the card's UUID, which encodes
the bus.

## Reproduce the kernel measurements

```
test-backend-ops perf -b SYCL0 -o MUL_MAT -p "type_a=(q5_K|iq4_nl),type_b=f32,m=4096,n=(1|2|4|8),k=14336"
test-backend-ops test -b SYCL0 -o FLASH_ATTN_EXT
```

The flash-attention decode timings use Qwen3.8-27B's attention shape (head dim 256, 4 KV heads,
GQA 6, f16 KV, one query token), which needs perf cases added to `tests/test-backend-ops.cpp`;
they are not in these patches. `test-backend-ops perf` re-runs one small tensor that stays in
cache, so its timings compare A with B well but are not absolute bandwidth.

## Upstream

None of these is upstream yet. 0015 fixes the same bug as the open draft ggml-org/llama.cpp#28473.
Until they land, a llama.cpp upgrade means rebasing them.

## What did not help

An IQ4_XS version of the Q5_K load-once change (0.98-1.05x), SYCL graphs (no gain on this model,
16 % slower when forced on), an int8 XMX path through shared memory, prefetching, and a fused
gate/up XMX kernel (slower than two launches plus the GLU).
