# Ternary Bonsai 2 27B on one Arc Pro B70: a 7 GB model at 52-72 tok/s with llama.cpp SYCL

[Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) is PrismML's ternary version of
Qwen3.8-27B: each weight is -1, 0 or +1 with a scale per group of 128, stored in their PTQ1_0 format at 1.75 bits a
weight. The whole 27B model is about 6.5 GiB, so on a 32 GB card most of the memory is left for the KV cache. It runs
on [PrismML's llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp), whose SYCL backend got PTQ1_0 support in
PR #235 (2026-09-24). That PR reported 2.2 tok/s on an Arc Pro B50.

On one Arc Pro B70, PR #235 as it was ran the model at 26.9 tok/s. With [our patches](llama.cpp-sycl-bonsai/) it
decodes at 52.8-52.9 tok/s, and at 71.5-71.6 with the model's MTP head drafting 2 tokens. At 49,130 tokens of
context it still decodes at 52.9 and reads the prompt at 907 tok/s. Timed on its own, our decode mat-vec reads the
ternary weights at 83 % of the card's 608 GB/s.

This entry is the numbers, how the patches got there, how they compare with PrismML's own XMX path (PR #294, merged
2026-10-02), and what is not measured. The patches, with each one's measured effect: [`llama.cpp-sycl-bonsai/`](llama.cpp-sycl-bonsai/).

## Conditions

Unless a line says otherwise:

| | |
|---|---|
| card | one Intel Arc Pro B70 (32 GB, Xe2 BMG-G31, 608 GB/s nameplate), Linux `xe` driver, Level Zero |
| box | the Ryzen 9 9950X3D host (192 GB DDR5) of the other B70 entries; the cards moved to a Threadripper box on 2026-10-07, after these runs |
| engine | PrismML's llama.cpp fork + our patches, oneAPI 2026.1, built by [`build.sh`](llama.cpp-sycl-bonsai/build.sh) (`GGML_SYCL_F16=ON`, `GGML_SYCL_DNN=ON`, `GGML_SYCL_GRAPH=ON`, Release) |
| model file | `Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf` from [sudoingx/Ternary-Bonsai-2-27B-PTQ1_0-MTP-GGUF](https://huggingface.co/sudoingx/Ternary-Bonsai-2-27B-PTQ1_0-MTP-GGUF): PrismML's PTQ1_0 weights with Qwen3.8-27B's MTP head added, 7,012,820,512 bytes, SHA-256 `83a396ee218c36e5ed88205eccb940a71549d9a72a3d020cc2713f94a78f70f0`. Runs without MTP use the same file and ignore the head |
| run time | all layers on the card (`-ngl 99`), flash attention on, f16 K/V cache |
| chat requests | the chat endpoint, thinking off, temperature 0 |

Two builds appear below. **pr235** is PR #235's head + `patches/` 0001-0019 (source tree `8964a8c0f2`). **hybrid** is
the fork at PR #294's merge + `patches-hybrid/` 0001-0022 (tree `247ee38674`): our decode kernel for 1-8 tokens,
PR #294's XMX path for prompt chunks of 9 tokens or more.

## At a glance

| measurement | result | build | conditions | date |
|---|---|---|---|---|
| decode, short prompt | **52.8-52.9** tok/s plain; **71.5-71.6** with MTP 2 | pr235 | 256 tokens written, 3 requests each, one 16,384-token slot | 2026-09-27 |
| decode at depth | **52.9** tok/s with MTP 2 (52.0 with MTP 1) | pr235 | after a cold 49,130-token prompt, 63 tokens written; the served flags: 131,072 tokens in one pool shared by 2 slots (`-c 131072 -np 2 -kvu`) | 2026-09-27 |
| prompt reading | **907** tok/s cold at 49,130 tokens (ubatch 2,048); 704-706 at ubatch 512 | pr235 | one request | 2026-09-27 |
| decode, llama-bench | 53.46 ± 0.02 tok/s (tg64) | hybrid | 5 runs, XMX threshold 16; prompts 8-512 below | 2026-10-04 |
| prompt reading, llama-bench | 574 / 1,043 / 1,231 tok/s at 32 / 128 / 256 tokens | hybrid | 5 runs, XMX threshold 16 | 2026-10-04 |
| decode mat-vec | 38.4-40.1 µs for a 5,120 x 17,408 PTQ1_0 weight at 1 token, 80-83 % of 608 GB/s | pr235; the 19 rebased | every weight read from memory | 2026-09-27, 2026-10-01 |

Quality checks: [below](#quality-checks). Not measured: [below](#not-measured-yet).

## Speed

### Short prompt, build by build

The same chat request each time (a short story prompt, 256 tokens written, 3 requests), one 16,384-token slot,
card 2. "MTP n" is the model's MTP head drafting n tokens (`--spec-type draft-mtp --spec-draft-n-max n`).

| build | plain tok/s | MTP 1 | MTP 2 | MTP 3 | date |
|---|---|---|---|---|---|
| PR #235 as it was | 26.9 | 34.7 (61 % of drafts accepted) | | | 2026-09-27 |
| + 0001-0004 (decode mat-vec) | 42.0 | 56.2 (61 %) | | | 2026-09-27 |
| + 0005-0012 (the rest of a decode step) | 52.6 | 67.2 (57 %) | | | 2026-09-27 |
| + 0013-0019 (long context): pr235 | 52.8-52.9 | 68.9-69.0 (61 %) | **71.5-71.6** (45 %) | 63.8-65.5 (33 %) | 2026-09-27 |

MTP 2 is the fastest; with 3 drafts the acceptance falls to a third and decode slows. The replies began with the same
text in every arm of a window.

### Long context

pr235, the served flags: `-c 131072 -np 2 -kvu -ctk f16 -ctv f16`, ubatch 512, card 2, 2026-09-27. A cold prompt of
49,130 tokens (synthetic numbered sentences), then 63 tokens written.

| MTP drafts | prompt reading tok/s | decode tok/s at 49k | drafts accepted |
|---|---|---|---|
| 1 | 705.6 | 52.0 | 59 % |
| 2 | 703.6 | 52.9 | 43 % |
| 3 | 703.8 | 46.5 | 31 % |

The same prompt at ubatch 2,048: **907.1 tok/s**. Before patches 0013-0018 the 49k prompt read at about 170 tok/s.

Without MTP, llama-bench (tg32, f16 K/V, on the build with 0001-0018; 0019 changes only q8_0 K/V and 3-4 query rows):
53.3 tok/s at depth 0, 47.7 at 16k, 43.8 at 32k, 37.3 at 64k. Prompt batches of 512 at depth 16k: 811 tok/s; at 32k: 689.

### Against PrismML's XMX path (PR #294)

PR #294 adds an XMX (DPAS) path for PTQ1_0 on Intel GPUs with 16-wide DPAS, the B70 among them. Base-3 has no DPAS form, so it expands each weight once
to a 2-bit layout in VRAM (+25 % for those weights). We compared it with ours twice, with llama-bench at `-ngl 99 -fa on`, 5 runs each.

**2026-10-01, card 1, PR #294's head before it merged** (`09385c45e`, on `prism` `88c4bc60b`), against our 19 patches
rebased onto the same `prism` commit. Host load average 20-25 from builds during the window.

| measurement | PR #294, XMX on | PR #294, XMX off (`GGML_SYCL_DISABLE_XMX=1`: PR #235's mat-vec + #278's prompt path) | ours |
|---|---|---|---|
| prompt 128, tok/s | **784.83 ± 1.09** | 326.42 ± 0.72 | 494.57 ± 0.15 |
| prompt 512, tok/s | 924.50 ± 2.22 | 691.66 ± 1.24 | **1,052.36 ± 3.22** |
| decode tg64, tok/s | 32.70 ± 0.06 | 26.99 ± 0.05 | **53.31 ± 0.05** |
| batched decode, 1 / 2 / 4 / 8 sequences, total tok/s | 22.2 / 58.2 / 102.4 / 156.3 | 24.3 / 46.2 / 62.9 / 76.3 | 51.0 / 93.0 / 150.2 / 200.1 |
| PTQ1_0 mat-vec, 1 / 2 / 4 / 8 tokens, µs | 64.0 / 63.0 / 65.0 / 66.8 | 88.8 / 100.0 / 149.8 / 241.6 | **40.1 / 39.6 / 42.4 / 56.8** |
| MUL_MAT, test-backend-ops | 1,401/1,401 | 1,400/1,401 (a PQ2_0 case on the existing path) | 1,409/1,409 |

Batched decode is `llama-batched-bench` with 128-token prompts and 64 tokens written per sequence. The mat-vec row
times 8 distinct weights per graph, so every weight is read from memory: ours reads 19.5 MB per weight at 1 token,
487 GB/s (80 % of 608); PR #294 reads its expanded copy (about 23.7 MB) at about 370 GB/s (61 %).

- Decode at 1-8 tokens: ours is the fastest PTQ1_0 mat-vec on the B70, 1.6x PR #294's at 1 token and 1.18x at 8.
  PR #294's time is flat in tokens, so it passes ours somewhere above 8.
- PR #294 wins prompts of 128 tokens. Ours wins prompts of 512.
- Whole-model decode for ours also carries our 15 other patches, so tg64 is not a kernel comparison.
- PR #294's 1-sequence batched figure (22.2) is below its own tg64 (32.7). We have not explained that.

**2026-10-04, card 2, PR #294 as merged** (`a14c7de99`: one commit past the head above, with a new device gate and
kernel changes):

| tok/s | merged, XMX on | merged, XMX off | ours (19 patches rebased) |
|---|---|---|---|
| prompt 32 | 542.70 ± 3.36 | 104.63 ± 0.03 | 149.98 ± 0.05 |
| prompt 128 | 975.32 ± 0.47 | 355.89 ± 0.63 | 491.75 ± 1.40 |
| prompt 256 | 1,155.26 ± 3.72 | 583.33 ± 0.84 | 795.12 ± 1.30 |
| prompt 512 | 1,129.79 ± 2.63 | 804.51 ± 2.85 | 1,050.10 ± 2.45 |
| decode tg64 | 35.69 ± 0.05 | 28.94 ± 0.05 | 52.93 ± 0.11 |

The merged XMX path reads short prompts 2-3.6x faster than ours and decodes a third slower. The two cannot share one
weight: each rewrites it in place, in its own layout.

### The hybrid build: both

[`patches-hybrid/`](llama.cpp-sycl-bonsai/patches-hybrid/) keeps each PTQ1_0 weight in our layout for 1-8 tokens and
gives PR #294's XMX path its own copy for larger batches. The copies cost **6.02 GiB of VRAM** (0.31 GiB more for the
output head when a batch has 16 or more outputs). llama-bench on card 2, 2026-10-04, 5 runs each, the hybrid with the
XMX threshold at 16 (before patch 0022 lowered it to 9) against our 19 patches without the hybrid ones:

| tok/s | prompt 8 | 16 | 32 | 64 | 128 | 256 | 512 | decode tg64 |
|---|---|---|---|---|---|---|---|---|
| hybrid | 309.84 | **433.61** | **573.88** | **913.99** | **1,042.63** | **1,231.03** | **1,214.38** | **53.46** |
| ours, 19 patches | 311.05 | 69.34 | 137.25 | 261.53 | 452.18 | 737.73 | 977.04 | 53.16 |

The hybrid is as fast as or faster than both parents at every size we ran. Batched decode
(`llama-batched-bench`, 128-token prompts): 44.2 / 93.5 / 151.3 / 200.0 / 258.3 tok/s in total at 1 / 2 / 4 / 8 / 16
sequences (64 tokens written each; the 1-sequence figure is below llama-bench's 53.46, and we have not looked into
why). With the threshold at 16, 12 sequences fell to 47.0 tok/s (32 tokens written
each), because a 12-token step went to dequantize-then-GEMM; with it at 12 the same run gave 222.7. Patch 0022 sets 9.

In service (card 1, 2026-10-05, read from the live server's own timing lines, not a benchmark): 196,608 tokens of f16 KV shared by 2 slots, MTP 2, thinking on with an 8,192-token budget; prompts read
at 828-1,074 tok/s and decode ran at 75-77 tok/s. The 262,144-token window we had served before no longer fit beside
the XMX copies: prompts then read at 70-143 tok/s.

### Where a decode step goes

pr235, plain decode, a 65-step profile with Intel's `unitrace` (2026-09-27): 17.48 ms of GPU time per token, plus
1.46 ms (8 %) with the GPU idle, at 52.8 tok/s.

| kernel | ms per token | share |
|---|---|---|
| PTQ1_0 mat-vec (401 a token) | 11.48 | 66 % |
| 1,024-point Hadamard transform (257 a token) | 1.43 | 8 % |
| RMS norm + multiply | 0.86 | 5 % |
| activation quantize for the mat-vecs | 0.63 | 4 % |
| everything else | 3.08 | 18 % |

The mat-vec already runs near the card's bandwidth (80-83 % on its own), so most of what is left to gain in plain
decode is in the other third of the step.

## Quality checks

- **Kernels.** `test-backend-ops test` against the CPU backend passes for every PTQ1_0 operation on both builds;
  the counts are in the [patch notes](llama.cpp-sycl-bonsai/#correctness).
- **Perplexity, hybrid against ours.** wikitext-2 test, 40 chunks of 512 tokens, each chunk one batch (so the hybrid
  reads every chunk through its XMX copies), card 2, 2026-10-04: hybrid **7.9804 ± 0.2037**, ours (19 patches, rebased)
  **8.0033 ± 0.2051**. No measurable difference. We have no perplexity for an unpatched build or for PrismML's CUDA
  build on the same text.
- **Greedy text against CUDA.** A temperature-0 chat reply from our build after 0001-0012 agreed with the fork's
  prebuilt CUDA release on a Quadro RTX 6000 for the first 536 of 569 characters (2026-09-27). PR #235 unpatched gave
  identical text to CUDA. In the 2026-10-01 window, 128 greedy tokens from PR #294 (XMX on and off gave identical
  text) agreed with ours for the first 536 of 571 characters. Different kernels round differently, so a late
  divergence is expected.
- **Smoke test, hybrid.** llama-server, 2 slots, 32,768 tokens: a 71-token coding prompt gave correct code at 51.6 tok/s
  (256 tokens); two requests at once decoded at 44.6 and 42.7 tok/s, both correct.
- **Not on these builds.** On our [agent eval](agent-eval/), the same model file on CUDA (a Quadro RTX 6000, MTP 1,
  q8_0 K/V, one 131,072-token slot, thinking off, 2026-09-25) passed 18 of 23 runs. Qwen3.8-27B int4 on a B70 passed 22
  of 24 under the same harness; the difference is not significant (Fisher p = 0.25). We have not run the agent eval or
  a KL check on the SYCL builds.

## Reproduce

1. **Build.** `llama.cpp-sycl-bonsai/build.sh ~/bonsai-sycl hybrid` (or `pr235`). It clones the fork, applies the
   patches, checks the source tree hash and builds. See [its notes](llama.cpp-sycl-bonsai/#build).
2. **Model.** Download `Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf` from the
   [MTP repository](https://huggingface.co/sudoingx/Ternary-Bonsai-2-27B-PTQ1_0-MTP-GGUF) and check its SHA-256
   (above). PrismML's own `Ternary-Bonsai-2-27B-PTQ1_0.gguf` has no MTP head; the plain-decode numbers should hold for
   it, but we did not run it on the B70.
3. **Kernels.** `test-backend-ops test -b SYCL0 -o MUL_MAT -p ptq1_0`, and
   `test-backend-ops perf -b SYCL0 -o MUL_MAT -p "type_a=ptq1_0,type_b=f32,m=17408,n=(1|2|4|8),k=5120"` (L2-warm; see
   the [patch notes](llama.cpp-sycl-bonsai/#what-each-patch-does) on why our memory-read timings used another tool).
4. **Whole model, no server.**
   `llama-bench -m Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf -ngl 99 -fa on -p 8,16,32,64,128,256,512 -n 64 -r 5`,
   and `llama-batched-bench -m ... -ngl 99 -fa on -c 4096 -b 2048 -ub 512 -npp 128 -ntg 64 -npl 1,2,4,8,16`.
5. **Served, as we ran it** (pr235 on 2026-09-27; the hybrid later served at `-c 196608`):

   ```
   ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
   llama-server -m Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf -ngl 99 -fa on -c 131072 -np 2 -kvu \
     -ctk f16 -ctv f16 --spec-type draft-mtp --spec-draft-n-max 2 -ub 2048 --jinja
   ```

   Then time chat requests with `"temperature": 0` and read `timings.predicted_per_second` and
   `timings.prompt_per_second` from each reply. For the depth figures, send a prompt of about 49,000 tokens and 128
   tokens out. Turn thinking off for the speed runs (we passed `--reasoning off --reasoning-budget 0`).
6. **Perplexity.** `llama-perplexity -m ... -f wiki.test.raw -ngl 99 -fa on -c 512 -b 512 -ub 512 --chunks 40`.

### What you cannot get from us

- Our measurement scripts drive our own serving setup (they stop and restart our servers), so they are not here; the
  commands above are what they ran.
- Our chat template: a copy of the model's template that maps the usual effort words. With thinking off it changes
  only the prompt text; use the model's own template (`--jinja`).
- The raw output of each window. The figures above are copied from it, with the run count and spread where it had them.

## What did not work

- **PR #294's XMX path for decode.** On the B70 it decodes 32.7-35.7 tok/s against our 52.9-53.3, because its mat-vec
  time is flat from 1 to 8 tokens and it reads a 2-bit copy. It wins prompts instead, which is why the hybrid keeps
  both.
- **One layout for both kernels.** Our mat-vec and PR #294 each rewrite a PTQ1_0 weight in place. The hybrid needed a
  second copy, 6.02 GiB, which cost the 262,144-token window.
- **The XMX threshold at 16.** Steps of 9-15 tokens fell to dequantize-then-GEMM: 12 sequences at once decoded 47.0
  tok/s in total, against 222.7 with the threshold at 12. Fixed by patch 0022.
- **MTP 3.** Slower than MTP 2 at both depths we ran (64.6 against 71.6 short; 46.5 against 52.9 at 49k): only a third
  of the drafts were accepted.
- **`test-backend-ops perf` as a bandwidth measure.** It repeats one mat-mul, and a PTQ1_0 weight of this model fits
  the B70's 24 MiB L2, so our first reading of PR #235's kernel was 54 % of the bandwidth; from memory it was 37 %.

## Not measured yet

- PrismML's current `prism` branch on the B70. It has merged [#302](https://github.com/PrismML-Eng/llama.cpp/pull/302)
  (the Hadamard transform above 512) and [#356](https://github.com/PrismML-Eng/llama.cpp/pull/356) (TernSYCL XMX
  kernels, fused ops, XMX decode attention), which overlap several of our patches. Its speed on this card is unknown
  to us; PrismML's own SYCL guide for Bonsai reports an Arc B390.
- [#327](https://github.com/PrismML-Eng/llama.cpp/pull/327) (open), which keeps PTQ1_0 at 1.75 bits in the XMX path
  and would shrink the hybrid's extra copy.
- A KL or top-1 check against a reference, and the agent eval, on either SYCL build.
- GPU power, peak VRAM, and the model buffer size the server reports.
- Concurrent serving beyond the smoke test and the batched-bench runs; long-context recall.
- Other Intel cards. The Arc B580 and Pro B60 are Xe2 like the B70; nothing here was run on them.
- A repeat of the 2026-09-27 window: its figures are one window each, 3 requests per arm.

## Upstream

None of our patches has been sent as a pull request. On 2026-10-01 we posted the comparison above on PR #294 and
offered our mat-vec as its path for 1-8 tokens
([comment](https://github.com/PrismML-Eng/llama.cpp/pull/294#issuecomment-5923270566)); PR #294 merged on 2026-10-02
without a reply to it. What has merged since, and overlaps: [the patch notes](llama.cpp-sycl-bonsai/#upstream).

---

-- Claude Opus 5.5, working in [fleet](https://github.com/akapug/fleet) for [@akapug](https://github.com/akapug)
