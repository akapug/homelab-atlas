#!/usr/bin/env bash
# PyTorch for an AMD Instinct MI25 / Vega 10 (gfx900) in a user-space venv. No ROCm install on the host: the wheels
# carry the HIP runtime and the math libraries; the host needs only the in-kernel amdgpu driver (/dev/kfd, user in
# the render group). Rollback: rm -rf "$VENV".
#
# Why not the stable wheel: torch 2.9.1+rocm6.4 lists gfx900 in torch.cuda.get_arch_list(), but its bundled rocBLAS
# has no gfx900 Tensile kernels, so the first f32 matmul fails ("rocBLAS error: Cannot read .../TensileLibrary.dat:
# Illegal seek for GPU arch : gfx900"); AMD's own ROCm 6.4.x rocBLAS packages ship none for gfx900 either.
# AMD's TheRock multi-arch builds do: a base torch wheel plus amd-torch-device-gfx900 (device code and rocBLAS,
# MIOpen, rocSOLVER for gfx900). They are nightlies from a staging index; the pin below is the one measured.
set -euo pipefail
VENV=${VENV:-$HOME/tail-portable/venv-therock}
V=${TORCH_THEROCK:-2.11.0+rocm7.13.0a20260515}
python3 -m venv "$VENV"
. "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q --index-url https://rocm.nightlies.amd.com/whl-staging-multi-arch/ \
    "torch==$V" "amd-torch-device-gfx900==$V" numpy
python - <<'PY'
import torch
assert torch.cuda.is_available(), "no HIP device: check /dev/kfd and the render group"
p = torch.cuda.get_device_properties(0)
print(torch.__version__, torch.version.hip, p.name, p.gcnArchName, round(p.total_memory / 2**30, 1), "GiB")
x = torch.randn(512, 512, device="cuda")
print("f32 matmul max |gpu - cpu|:", float((x @ x - (x.cpu() @ x.cpu()).cuda()).abs().max()))
PY
