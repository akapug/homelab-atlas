#!/usr/bin/env bash
# PyTorch CUDA for older NVIDIA cards (GTX 1080 Ti = Pascal sm_61, Quadro RTX 6000 = Turing sm_75) in a user-space
# venv. The cu126 wheels still carry sm_50..sm_61 kernels; the cu128+ wheels dropped Maxwell and Pascal.
# Neither card has TF32 or fast bf16: the trainer runs f32 there (--tf32 auto stays off below sm_80).
# Rollback: rm -rf "$VENV".
set -euo pipefail
VENV=${VENV:-$HOME/tail-portable/venv-cu126}
V=${TORCH_CUDA:-2.9.1}
python3 -m venv "$VENV"
. "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q "torch==$V" numpy --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, f"sm_{p.major}{p.minor}", round(p.total_memory / 2**30, 1), "GiB")
PY
