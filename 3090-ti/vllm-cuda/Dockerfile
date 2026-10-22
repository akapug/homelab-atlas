# vLLM's official CUDA image, v0.31.0 pinned by digest, with the two patches from b70/vllm-xpu/ that it still needs.
# Build from the repo root:  docker build -f 3090-ti/vllm-cuda/Dockerfile -t vllm-cuda:0.31.0-atlas .
# - draft-vocab.py: the MTP drafter scores only the token ids in VLLM_DRAFT_VOCAB (+28 % at one stream on an RTX 3090 Ti).
# - length-finish.py: a reply cut by max_tokens inside a tool call reports "length", not "tool_calls". 0.31.0 already has
#   the streaming half upstream; the script skips it and patches the non-streaming half.
# strict-tools is upstream in 0.31.0 as --tool-strict-level parameter, so it is not applied here.
FROM vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
COPY b70/vllm-xpu/draft-vocab.py b70/vllm-xpu/length-finish.py /opt/atlas-patches/
RUN V=/usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /opt/atlas-patches/draft-vocab.py $V/model_executor/models/qwen3_5_mtp.py \
 && python3 /opt/atlas-patches/length-finish.py $V/entrypoints/openai/chat_completion/serving.py \
 && grep -q "draft-vocab patch" $V/model_executor/models/qwen3_5_mtp.py \
 && grep -q "length-finish patch" $V/entrypoints/openai/chat_completion/serving.py \
 && find $V -name '*.pyc' -delete
