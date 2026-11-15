#!/usr/bin/env bash
# PyTorch XPU for an Intel Arc Pro B70 (Battlemage, 0xe223) in a user-space venv. The host needs the Intel GPU
# compute runtime (Level Zero; measured with compute-runtime 26.05.37020) and the user in the render group.
# Rollback: rm -rf "$VENV".
#
# Pinned to 2.10.0+xpu: on compute-runtime 26.05.37020, 2.13.0..2.15.0+xpu return wrong torch.nonzero /
# boolean-mask-indexing / scalar scatter_ results on this card (repro_xpu_nonzero_scatter.py; 2.10.0..2.12.1 correct)
# and, with those three worked around in the trainer, 2.15.0 still fails the tail's forward-match gate (top-20
# |delta log p| 0.22 against a 0.10 bar); 2.10.0+xpu passes it. On compute-runtime 26.40.40196.15, 2.15.0+xpu passes
# that gate (the op-level repro there is pending): with the latest driver, TORCH_XPU=2.15.0+xpu is the one to try.
set -euo pipefail
VENV=${VENV:-$HOME/tail-portable/venv-xpu}
V=${TORCH_XPU:-2.10.0+xpu}
python3 -m venv "$VENV"
. "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q "torch==$V" numpy --index-url https://download.pytorch.org/whl/xpu --extra-index-url https://pypi.org/simple
python - <<'PY'
import torch
assert torch.xpu.is_available(), "no XPU device: check the compute runtime and the render group"
p = torch.xpu.get_device_properties(0)
print(torch.__version__, p.name, p.driver_version, round(p.total_memory / 2**30, 1), "GiB", "fp64", p.has_fp64)
PY
