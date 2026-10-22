#!/usr/bin/env bash
# Build the hot/cold split store of DeepSeek V4.1 Flash 1017 that our RTX 6000 + Optane runs load.
# Hot experts (hot-alloc.json, 3 bits per weight budget) keep the release's native MXFP4 bytes; the other experts are
# requantized to q2_K (down projections of layers 0-9 to q3_K) with an importance matrix; everything else stays as the
# source has it. split.rules is the llama-quantize rule list the script writes from those choices; the build checks
# that yours is the same. The tools are scripts/split_experts.py and llama-quantize from the tier-split branch.
#   IK=<ik_llama.cpp checkout, branch tier-split>  QUANTIZE=<llama-quantize>  IMATRIX=<imatrix .dat>  OUT=<dir>  ./build-store.sh
# Space: the Q8_0 source (about 508 GB) plus about 289 GB of intermediate split and 222 GB of output.
# Time on a 16-core box: about 55 min of split and quantization, then 40 min of verification.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
IK=${IK:?IK=<ik_llama.cpp checkout>}; QUANTIZE=${QUANTIZE:?QUANTIZE=<llama-quantize>}; IMATRIX=${IMATRIX:?IMATRIX=<imatrix .dat>}
OUT=$(realpath -m "${OUT:-$PWD/v41-1017-split}"); SRCDIR=$(realpath -m "${SRCDIR:-$PWD/v41-1017-q8}"); J=${J:-16}
REV=2c97dba3934f27a99767a90830e7519cc7b8b2d9

# 1. the source: the Q8_0 GGUF of the 1017 release, checked against our hashes
huggingface-cli download mrfakename/DeepSeek-V4.1-Flash-1017-GGUF --revision $REV --local-dir "$SRCDIR" --include 'Q8_0/*'
(cd "$SRCDIR/Q8_0" && grep 'Q8_0-000' "$here/SHA256SUMS" | awk '{print $1"  "$3}' | sha256sum -c)
SRC=$SRCDIR/Q8_0/DeepSeek-V4.1-Flash-1017-Q8_0-00001-of-00010.gguf

# 2. split + quantize (the imatrix is sliced the same way as the experts; never-routed experts get the mean row)
mkdir -p "$OUT/tmp"
python3 -c "import numpy" || pip install numpy   # split_experts.py imports gguf-py from the checkout
python3 "$IK/scripts/split_experts.py" "$SRC" "$OUT/tmp/V41-1017-split-00001-of-00010.gguf" --shard-search "$SRCDIR/Q8_0" \
  --part hot="$here/hot-alloc.json" --rest cold \
  --imatrix "$IMATRIX" --imatrix-out "$OUT/v41-1017-split.imatrix.dat" \
  --type hot=keep --type cold.down@0-9=q3_K --type cold=q2_K --quant-rules "$OUT/split.rules" --quantize-bin "$(dirname "$QUANTIZE")" \
  --quant-out "$OUT/DeepSeek-V4.1-Flash-1017-hotmxfp4-coldq2k.gguf" --threads "$J"

cmp "$OUT/split.rules" "$here/split.rules" && echo "quantize rules: the same as ours"

# 3. verify: sampled bytes and types against the source and the rules (hot bytes must equal the source MXFP4)
F=$OUT/DeepSeek-V4.1-Flash-1017-hotmxfp4-coldq2k-00001-of-00010.gguf
python3 "$IK/scripts/split_experts_verify.py" "$SRC" "$OUT/tmp/V41-1017-split-00001-of-00010.gguf" --bytes --sample 4 --shard-search "$SRCDIR/Q8_0"
python3 "$IK/scripts/split_experts_verify.py" "$SRC" "$F" --bytes --sample 4 --expect-rules "$OUT/split.rules" --shard-search "$SRCDIR/Q8_0"

# 4. shards 8-10 hold only tensors the split leaves alone (the engram tables): use the source shards
for i in 8 9 10; do
  n=$(printf %05d $i)
  ln -sf "$SRCDIR/Q8_0/DeepSeek-V4.1-Flash-1017-Q8_0-$n-of-00010.gguf" "$OUT/DeepSeek-V4.1-Flash-1017-hotmxfp4-coldq2k-$n-of-00010.gguf"
done
rm -rf "$OUT/tmp"
# sizes must equal ours; hashes match only with our imatrix and quantizer build
(cd "$OUT" && for f in DeepSeek-V4.1-Flash-1017-hotmxfp4-coldq2k-0000[1-7]-of-00010.gguf; do echo "$(stat -c %s "$f") $f"; done)
grep coldq2k "$here/SHA256SUMS" | awk '{print "ours: " $2 " " $3}'
