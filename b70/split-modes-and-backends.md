# Two B70s, one model: tensor split loses decode on MoE; three outside claims checked

Three claims about Arc Pro B70 serving, each checked on our two B70s against its source on 2026-11-14, and a fourth run
on the models we actually serve. Box C: Threadripper 3960X (24 cores), 121 GiB RAM, two Arc Pro B70 32 GB on PCIe 4 x16,
kernel 7.0 with the `xe` driver, compute-runtime 26.05.37020. Our server was stopped for each window and restored after.
Every number is measured here unless it says CLAIMED (someone else's) or not measured.

## At a glance

| question | answer | details |
|---|---|---|
| Does llama.cpp SYCL's `-sm tensor` (PR 24152) beat `-sm layer` on a MoE model that needs both cards? | **No: decode -41 % to -54 %.** Prefill -21 % (all on the GPUs) to +23 % (experts partly on the CPU) | [tensor split on MoE](#tensor-split-on-moe-models-that-need-both-cards) |
| Does the split cover the expert matmuls (`MUL_MAT_ID`)? | Yes, already: every card holds a slice of every expert. The cost is the cross-card reduce, staged through host memory | [why decode loses](#why-decode-loses) |
| Tensor split on a dense 27B that fits one card? | Nothing gained: prefill within 3 %, decode -11 % | [dense 27B](#tensor-split-on-a-dense-27b-that-fits-one-card) |
| vLLM XPU: is `FLASH_ATTN` 8-11x faster to first token than `TRITON_ATTN`? | Larger here (9.8x at 4k, 61x at 64k), but `FLASH_ATTN` is already the default on llm-scaler 0.26.0-b2. Never force `TRITON_ATTN` | [attention backend](#vllm-xpu-flash_attn-against-triton_attn) |
| arcint's OpenCL kernels against stock SYCL on a dense 27B? | +26 % prefill, +9 % decode on one B70 (its README's 5.5x is another comparison) | [arcint](#arcint-opencl-against-stock-sycl-dense-27b) |
| Layer split with `-ot` that moves the last layers' experts to the CPU? | Overcommits card 1 and stalls: balance the CPU layers per card | [the -ot rule](#a-rule-for-layer-split-with--ot-balance-the-cpu-layers-per-card) |

## Tensor split on MoE models that need both cards

Engine: stock llama.cpp SYCL, `ghcr.io/ggml-org/llama.cpp:full-intel` b12004 (3c809f443, 2026-10-29), which carries
PR 24152. `llama-bench -ngl 99 -fa 1 -t 21 [-ot ...] -sm layer,tensor,row -p 4096 -n 128 -r 2`, f16 KV, ubatch 512,
batch 2048, one stream, the container on CPUs 0-20,24-44:

```bash
docker run --rm --device /dev/dri/card0 --device /dev/dri/renderD128 --device /dev/dri/card1 --device /dev/dri/renderD129 \
  --cpuset-cpus 0-20,24-44 -v /models:/models:ro -e ONEAPI_DEVICE_SELECTOR=level_zero:0,1 \
  --entrypoint /app/llama-bench \
  ghcr.io/ggml-org/llama.cpp@sha256:77faff7d38b78820e5b7718f43a4284e27e58f83a643e4cb4bbe7653967f4691 \
  -m /models/MODEL.gguf -ngl 99 -fa 1 -t 21 -sm layer,tensor -p 4096 -n 128 -r 2 -o jsonl
```

One prompt length (4k) and n = 2; the Ornith decode was rerun at `-r 5` because its pair read 9.3 and 35.1.

| model | `-ot` | test | layer | tensor | tensor vs layer | row |
|---|---|---|---|---|---|---|
| Qwen3.8 Flash-Next UD-IQ3_XXS (512 experts) | none (26.6 / 27.4 GB on the two cards) | pp4096 | 638.0, 664.7 -> 651.4 | 514.6, 518.2 -> 516.4 | **-21 %** | aborts |
| | | tg128 | 30.32, 30.23 -> 30.3 | 13.65, 14.03 -> 13.8 | **-54 %** | aborts |
| Ornith-1.5-122B Q4_K_M (256 experts) | experts of layers 18-24 and 42-48 on the CPU (14 of 49) | pp4096 | 209.6, 211.8 -> 210.7 | 257.3, 261.4 -> 259.4 | **+23 %** | does not load |
| | | tg128 (`-r 5`) | 9.6, 35.42, 35.34, 35.31, 35.32 -> 35.3 (reps 2-5) | 20.66, 21.75, 21.73, 19.81, 20.63 -> 20.9 | **-41 %** | does not load |
| GLM-5.3-Flash UD-Q4_K_XL | experts on the CPU | - | not measured | refused | - | not measured |

Layer split stays our serving default on this box. The PR's own numbers (CLAIMED: dual B70, Llama-3.3-70B Q4_K_M tg128
17.40 tensor against 9.74 layer, +78.6 %; Qwen3-Coder-Next-80B-A3B tg 17.60 against 14.31, measured against another
split, not layer) do not reproduce on these MoE models. The PR author's later dual-B70 A/B on a dense 27B (tg 15.7 ->
7.4, CLAIMED) points the same way as ours. The dense-70B case the PR measured is not measured here: we hold no dense
model over 32 GB.

Side findings:

- **Row split runs neither model.** Flash-Next aborts at `ggml-sycl.cpp:1431` (`get_row_rounding` has no case for BF16;
  the model carries 24 BF16 `indexer.k_proj` tensors). Ornith-122B fails to load with
  `ggml_backend_sycl_split_buffer_init_tensor: can't allocate 0 Bytes of memory on device`.
- **The SYCL first-repetition decode stall.** With layer split Ornith decoded 9.3 and 9.6 tok/s on repetition 1, then
  35.3. Tensor split showed no such stall. Count repetition 1 out, or warm up with a full run first.
- **GLM-5.3-Flash refuses tensor split** at model creation: "LLAMA_SPLIT_MODE_TENSOR not implemented for architecture
  'glm5-next'". `src/llama-arch.cpp:1390` lists it beside DEEPSEEK2, DEEPSEEK32 and GLM_DSA: its attention (MLA with a
  sparse indexer, plus linear-attention layers) is not covered by the split rules. That is an attention gap, not an
  expert one. Its layer arm was not measured: the 199 GB GGUF is about all of the box's RAM plus VRAM.

Not measured: other prompt lengths, more than one stream, tensor split with `-ub` above 512, Flash-Next with `-ot`.

### Why decode loses

Read from the source at b12004 (not profiled):

- PR 24152 adds only the SYCL all-reduce. Which tensor goes where comes from the backend-agnostic meta device and
  llama's split rules (`src/llama-model.cpp:606-623`): `ffn_up_exps` and `ffn_gate_exps` split on axis 1 (each expert's
  FF width), `ffn_down_exps` on axis 0, and `GGML_OP_MUL_MAT_ID` goes through the same handler as `MUL_MAT`
  (`ggml/src/ggml-backend-meta.cpp:929-932`). So each routed expert is computed half on each card and then reduced:
  tensor parallelism inside each expert, not expert parallelism. No `MUL_MAT_ID` patch is needed.
- One gap: shared-expert tensors (`ffn_*_shexp`) are split only for DeepSeek V4 (`src/llama-model.cpp:531-537`) and
  mirrored for every other architecture, so both cards hold and compute the whole shared expert. That costs VRAM and
  duplicate work, not correctness; it is small next to the routed experts on these models.
- Every split layer closes with an all-reduce, about two a layer, so about 96 a token on Flash-Next's 48 blocks. The
  SYCL all-reduce stages each one through pinned host memory: a device-to-host copy on both cards, a host wait, two host
  `memcpy`s, host-to-device (`ggml/src/ggml-sycl/ggml-sycl.cpp:7846-7869`; the comment at `:8002` says a direct peer
  copy between the two SYCL contexts is a silent no-op). Flash-Next's gap is 72 - 33 = 39 ms a token, about 0.4 ms a
  hop over 96 hops (our reading). Layer split crosses the cards once a token.

So the patch that could change this verdict is a device-to-device all-reduce (Level Zero IPC, or one shared SYCL
context with peer access). We have not built it.

### A rule for layer split with `-ot`: balance the CPU layers per card

Layer split assigns layers to cards by count, not by bytes. Our first Ornith-122B run put the experts of layers 37-48 on
the CPU, so card 1 got 25 full-weight layers (about 34 GB on a 32 GB card) and card 2 the light half. Prefill was 23.4
and 25.5 tok/s, decode had not finished one 128-token repetition after 5 minutes, card 1 was 100 % busy and card 2
idle, and the host's memory compaction ran at 100 %; we stopped it. Spreading the CPU layers over both halves (18-24
and 42-48, `-ot 'blk\.(1[89]|2[0-4]|4[2-8])\.ffn_(up|gate|down)_exps\.weight=CPU'`) gave 28.1 and 26.3 GB on the two cards and 9x the prefill (210.7 tok/s). The cause, `xe` VRAM overcommit
and eviction, is our reading. Any `-ot` recipe with layer split should move the same number of layers' experts off
each card's half.

## Tensor split on a dense 27B that fits one card

Qwen3.8-27B UD-Q4_K_XL (16.34 GiB), the same image, `llama-bench -ngl 99 -fa 1 -ctk f16 -ctv f16 -sm none,layer,tensor
-p 512,4096 -n 128 -r 4`. `none` is card 1 alone. The summary is the mean of the 4 samples, or of reps 2-4 where rep 1
is the first-repetition stall.

| test | none (1 card) | layer (2 cards) | tensor (2 cards) | tensor vs layer |
|---|---|---|---|---|
| pp512 | 885.1 | 860.5 | 883.7 | +2.7 % |
| pp4096 | 873.5 | 853.4 | 824.8 | -3.4 % |
| tg128 | 22.4 (reps 2-4; rep 1 7.37) | 22.4 | 20.0 | **-11 %** |

On a dense model that fits one B70, tensor split buys nothing, and its samples are the noisiest.

## vLLM XPU: FLASH_ATTN against TRITON_ATTN

The claim (CLAIMED, a tester on Reddit r/LocalLLM, 2026-10-03, on `intel/llm-scaler-vllm:0.26.0-b2` with Qwen3.8-27B
GPTQ-Int4 and MTP): switching from `TRITON_ATTN` to `--attention-backend=FLASH_ATTN` gave 8-11x faster time to first
token at 9.6-19k tokens (69.9 -> 8.2 s, 246 -> 22.3 s).

Ours: the same image (vLLM 0.26.1.dev0+g568afb3a1), unpatched, one container per card, Ornith-1.5-35B-A3B BF16
quantized at load to int4, eager, fp8 KV, block 64, `--max-num-batched-tokens 8192`, `--max-model-len 81920`, prefix
caching on, MTP off. Streamed `/v1/completions` with token-id prompts cut from one corpus, each request from a distinct
offset so none shares a prefix (the prefix-cache hit counter did not move), greedy, 128 output tokens, n = 2 each,
after one discarded warm-up per length.

| prompt | default (= FLASH_ATTN) time to first token | TRITON_ATTN | ratio | decode, default | decode, Triton |
|---|---|---|---|---|---|
| 4,096 | 0.413 / 0.413 s | 4.047 / 4.044 s | **9.8x** | 47.5 / 45.2 | 43.0 / 46.8 |
| 16,384 | 2.014 / 2.009 s | 57.74 / 57.65 s | **28.7x** | 46.1 / 48.3 | 21.8 / 21.7 |
| 65,536 | 15.06 / 15.09 s | 943.2 / 903.9 s | **61.3x** | 54.2 / 55.1 | 6.3 / 6.3 |

- `FLASH_ATTN` is already the default here: the engine log says "Using Flash Attention backend" with no flag, fp8 KV
  included, and an explicit `--attention-backend FLASH_ATTN` on the other card gave the same outputs bit for bit (6 of
  6 hashes) and the same times. So there is nothing to switch on; the rule is never to force `TRITON_ATTN` on this image.
- Triton's time to first token grows about 4x per doubling of the prompt, and its decode falls with depth.
- Default-arm decode rose from about 46 to 55 tok/s over the run as other CPU load on the box changed; the Triton cells
  ran under the same load, so the ratios hold but the decode values carry that noise.
- The two backends' greedy outputs differ (0 of 6 identical at 4k and 16k). At 4k they part only at near-ties. At 16k
  one prompt showed a confident flip; the Triton server asked again from its cached prefix gave the `FLASH_ATTN` answer,
  so the cold Triton long prefill is the outlier (our reading, one prompt).

## arcint OpenCL against stock SYCL, dense 27B

The claim (CLAIMED, [marfrit/arcint](https://github.com/marfrit/arcint) README, main fa9eb55e46b7, on an Arc Pro B60 or
A770): about 100 patches to ggml's OpenCL backend. Its "5.5x" is the patch series' own start-to-end on OpenCL with
Llama-2-7B Q4_0 (426 -> 2,333), not a comparison with SYCL; against SYCL on a dense Qwen3.8-27B Q4_K_M it claims +37 %
prefill and -4.4 % decode on a B60.

Ours: Qwen3.8-27B UD-Q4_K_XL on one B70, `llama-bench -ngl 99 -fa 1,0 -p 512,4096 -n 128 -r 3`. arcint built from its
main with its patch files on llama.cpp 3e4dfa8 (OpenCL backend); SYCL is the b12004 image above.

| test | arcint, FA on | arcint, FA off | SYCL, FA on | SYCL, FA off | arcint vs best SYCL |
|---|---|---|---|---|---|
| pp512 | 1,104.5 +- 4.5 | 1,083.0 +- 4.1 | 872.0 +- 4.4 | 869.9 +- 3.4 | **+27 %** |
| pp4096 | 1,099.2 +- 1.6 | 992.6 +- 1.6 | 874.1 +- 2.5 | 837.9 +- 0.9 | **+26 %** |
| tg128 | 24.23 +- 0.02 | 24.14 +- 0.04 | 22.3 (steady; see below) | 22.31 +- 0.03 | **+9 %** |

SYCL with FA on reports tg128 17.44 +- 8.53 because its first repetition runs at 7.5 tok/s and the next four at
22.26-22.36 (a separate `-r 5` run); the steady value is 22.3. arcint's quality gates were not rerun here.

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [fleet](https://github.com/akapug/fleet) for [@akapug](https://github.com/akapug)
