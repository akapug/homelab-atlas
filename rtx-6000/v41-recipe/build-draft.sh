#!/usr/bin/env bash
# Build the DSpark draft model our V4.1 runs use, from DeepSeek's own release.
# The draft head (mtp.* tensors) lives in safetensors shards 44-46 of deepseek-ai/DeepSeek-V4.1-Flash-1017; the
# converter is JigSawPT's llama.cpp fork at 8ce09685 (DeepseekV41DSparkModel).
#   QUANTIZE=<llama-quantize from build-engine.sh> OUT=<dir> ./build-draft.sh
# Needs about 25 GB free, docker, and the Hugging Face CLI (`hf`, from pip install -U huggingface_hub).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
QUANTIZE=${QUANTIZE:?QUANTIZE=<llama-quantize>}; OUT=$(realpath -m "${OUT:-$PWD/draft}")
REV=1bae0e1088993b9b82de08f59de5be5a34e42170; JIG=8ce0968539b1a6795f3413c2783dfa5646b4688a
HF=$(command -v hf || command -v huggingface-cli) || { echo "needs the Hugging Face CLI: pip install -U huggingface_hub"; exit 1; }
export LD_LIBRARY_PATH="$(dirname "$QUANTIZE")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"   # the CUDA libraries build-engine.sh packaged
mkdir -p "$OUT/st" "$OUT/in"
"$HF" download deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision $REV --local-dir "$OUT/st" \
  --include 'model-0004[4-6]-of-00048.safetensors' '*.json' '*.py' 'tokenizer*'
(cd "$OUT/st" && grep 'safetensors$' "$here/SHA256SUMS" | awk '{print $1"  "$3}' | sha256sum -c)
# the converter refuses an index that names absent shards: give it an index trimmed to the mtp.* tensors
python3 - "$OUT/st" "$OUT/in" <<'PY'
import json, os, sys
src, dst = sys.argv[1:]
idx = json.load(open(f'{src}/model.safetensors.index.json'))
wm = {k: v for k, v in idx['weight_map'].items() if k.startswith('mtp.')}
json.dump({'metadata': idx.get('metadata', {}), 'weight_map': wm}, open(f'{dst}/model.safetensors.index.json', 'w'), indent=1)
for f in os.listdir(src):
    if f != 'model.safetensors.index.json' and not f.startswith('.'):
        target = os.path.abspath(f'{src}/{f}')
        link = f'{dst}/{f}'
        if os.path.islink(link) and os.readlink(link) == target:
            continue
        os.symlink(target, link)
print(len(wm), 'mtp.* tensors in', sorted(set(wm.values())))
PY
[ -d "$OUT/jig/.git" ] || git clone -q https://github.com/JigSawPT/llama.cpp "$OUT/jig"
git -C "$OUT/jig" checkout -q $JIG
# the same path inside the container, so the links in $OUT/in resolve
docker run --rm -u "$(id -u):$(id -g)" -e HOME="$OUT/home" -e O="$OUT" -v "$OUT":"$OUT" -w "$OUT" python:3.12-slim bash -euc '
  mkdir -p "$O/home"
  pip install -q --no-cache-dir --target "$O/py" -r "$O/jig/requirements/requirements-convert_hf_to_gguf.txt" \
    --extra-index-url https://download.pytorch.org/whl/cpu
  PYTHONPATH="$O/py" python3 "$O/jig/convert_hf_to_gguf.py" "$O/in" --dspark --target-model-dir "$O/in" --outtype q8_0 \
    --outfile "$O/DeepSeek-V4.1-Flash-1017-DSpark.gguf"'
"$QUANTIZE" --allow-requantize --custom-q 'ffn_(up|gate|down)_exps=q2_K' "$OUT/DeepSeek-V4.1-Flash-1017-DSpark.gguf" \
  "$OUT/DeepSeek-V4.1-Flash-1017-DSpark-exps-Q2_K.gguf" Q8_0 16
ls -l "$OUT"/*.gguf
grep DSpark "$here/SHA256SUMS"
