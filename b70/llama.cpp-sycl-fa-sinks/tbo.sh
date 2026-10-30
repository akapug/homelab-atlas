#!/bin/bash
# tbo.sh BIN MODE FILTER OUT: run BIN/test-backend-ops MODE (test or perf) -b SYCL0 -o FLASH_ATTN_EXT -p FILTER on one
# Intel GPU, chosen by its PCI address in CARD (for example CARD=0000:03:00.0; see ls /dev/dri/by-path). The backends
# load from BIN only, inside the same image build.sh uses. DBG=1 turns on llama.cpp's own GGML_SYCL_MKL_FA_DEBUG trace,
# which prints the kernel path of each case ([FA-DISP] lines); paths.py reads it.
# Writes OUT with colour codes stripped and rc=N as its last line.
set -u
BIN=$(cd "$1" && pwd) MODE=$2 FILT=$3 OUT=$4
P=/dev/dri/by-path/pci-${CARD:?set CARD to the GPU PCI address, e.g. CARD=0000:03:00.0}
CR=$(readlink -f $P-card) RR=$(readlink -f $P-render)
IMAGE=${IMAGE:-ghcr.io/ggml-org/llama.cpp@sha256:77faff7d38b78820e5b7718f43a4284e27e58f83a643e4cb4bbe7653967f4691}
docker run --rm --init --device $CR --device $RR --group-add "$(stat -c %g $RR)" \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:* -e GGML_SYCL_MKL_FA_DEBUG=${DBG:-0} -v $BIN:/t:ro -w /t \
  --entrypoint /t/test-backend-ops "$IMAGE" $MODE -b SYCL0 -o FLASH_ATTN_EXT -p "$FILT" > $OUT.raw 2>&1
rc=$?
sed -r "s/\x1b\[[0-9;]*m//g" $OUT.raw > $OUT; rm -f $OUT.raw
echo "rc=$rc" >> $OUT
exit $rc
