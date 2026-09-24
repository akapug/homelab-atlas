# DeltaNet mixed-batch check (vllm-xpu-kernels #552)

[vllm-project/vllm-xpu-kernels#552](https://github.com/vllm-project/vllm-xpu-kernels/pull/552)
(open) fixes an out-of-bounds write in the XE2 gated-delta-rule prefill epilogue: it assumes each
sequence's token indices are contiguous. Batches that mix speculative and non-speculative
sequences are where it shows. This model (Qwen3.8-27B, 48 DeltaNet layers) served with MTP
produces exactly such batches, so we ran the PR's bitwise differential against the kernels in
`intel/llm-scaler-vllm:0.26.0-b2` (vllm_xpu_kernels 0.1.11.2.dev0+ga692986) on one Arc Pro B70.

```
python3 run-gdn-mixed.py test_gdn_attn_mixed_adapted.py      # inside the image, one XPU visible
```

Result (bf16, 24 cases: 3 batch shapes x 2 speculative shapes x reorder on/off x 2 layouts):

| token layout | batch has a prefill | result |
|---|---|---|
| fully shuffled (the PR's test) | yes | 0 of 8 bitwise identical; relative error 0.76-1.01, 29-48 of 34-51 rows differ |
| each request contiguous (how vLLM lays out a batch) | yes | 8 of 8 bitwise identical |
| fully shuffled | no (decodes and speculative decodes) | 4 of 4 bitwise identical |
| each request contiguous | no | 4 of 4 bitwise identical |

So on this build the fault is real and gross when a request's tokens are scattered, and absent for
contiguous per-request layouts, which is what vLLM builds; that is also why our 4-way concurrent
serving eval stayed at 41 of 41. The adaptation is described at the top of the test file: this
build rejects a single population over the full token count, so the reference runs each
population on compactly gathered rows.
