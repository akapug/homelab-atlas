"""Serve AutoRound int4 checkpoints on llm-scaler's sym_int4 kernels (0.26.0-b2).

An AutoRound checkpoint packed as auto_gptq, symmetric, 4 bits, group 128 already holds
GGML Q4_0 with 128-wide blocks: q in [0, 15] with the zero point at 8, eight nibbles per
int32 in order along K, fp16 scales. It stores them K-major ([K/8, N], [K/128, N]); the
sym_int4 method wants them N-major. vLLM's stock XPU AutoRound path (with auto_round_kernel
installed, as in this image: INCARKLinearMethod, torch.ops.vllm.inc_ark_woq_linear) decodes
Qwen3.8-27B at 14-17 tok/s against ~57 for sym_int4 in eager mode, and returns garbage with XPU
graphs and MTP speculation together.

This adds INCXPUQ40LinearMethod: it loads the checkpoint's tensors as inc does, transposes
them into sym_int4's layout after load, and is otherwise the sym_int4 method, so the tuned
weights take the ESIMD decode kernels and the fused paths that check is_sym_int4.
VLLM_INC_XPU_Q40=0 restores the stock inc path.

Usage: inc-q40.py <site-packages>/vllm/model_executor/layers/quantization/inc/schemes
Writes a .orig backup of each file once and always patches from it."""
import os
import shutil
import sys

d = sys.argv[1]


def patch(name, pairs):
    p = os.path.join(d, name)
    if not os.path.exists(p + ".orig"):
        shutil.copy2(p, p + ".orig")
    s = open(p + ".orig").read()
    for old, new in pairs:
        assert s.count(old) == 1, (name, old[:60])
        s = s.replace(old, new)
    open(p, "w").write(s)
    print("patched", p)


patch("inc_wna16_scheme.py", [
    ("from typing import TYPE_CHECKING\n", "import os\nfrom typing import TYPE_CHECKING\n"),
    ("""            if layer_config.bits in XPU_WNA16_SUPPORTED_BITS and layer_config.sym:
""", """            if layer_config.bits in XPU_WNA16_SUPPORTED_BITS and layer_config.sym:
                if (
                    layer_config.bits == 4
                    and layer_config.group_size == 128
                    and layer_config.is_gptq
                    and os.environ.get("VLLM_INC_XPU_Q40", "1") != "0"
                ):
                    # inc-q40: the checkpoint is Q4_0 already; serve it on sym_int4's kernels
                    from .inc_wna16_linear import INCXPUQ40LinearMethod

                    return INCXPUQ40LinearMethod(layer_config)
""")])

patch("inc_wna16_linear.py", [("""class INCXPUW4A16LinearScheme(INCXPULinearMethod):""", '''from vllm.model_executor.layers.quantization.sym_int4 import (  # noqa: E402
    SymInt4LinearMethod,
    _register_linear_int4_layouts,
)


class INCXPUQ40LinearMethod(SymInt4LinearMethod):
    """AutoRound int4 (auto_gptq packing, symmetric, group 128) on the sym_int4 kernels.

    Such a checkpoint is GGML Q4_0 with 128-wide blocks stored K-major. Loading uses the inc
    parameters; after load the two tensors are transposed into sym_int4's N-major layout and
    everything else is sym_int4's, including apply() and is_sym_int4.
    """

    uses_meta_device = False

    def __init__(self, layer_config: "INCLayerConfig") -> None:
        self.quant_config = None
        self.weight_bits = layer_config.bits
        self.group_size = layer_config.group_size
        self.sym = layer_config.sym
        self.pack_factor = 32 // self.weight_bits
        self.is_awq_packed = False

    create_weights = INCXPULinearBase.create_weights
    _create_inc_weights = INCXPULinearBase._create_inc_weights

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        qweight = layer.qweight.data.t().contiguous()  # [N, K/8] int32
        scales = layer.scales.data.t().contiguous().to(torch.float16)  # [N, K/128]
        _register_linear_int4_layouts(layer, qweight, scales)
        del layer.g_idx
        in_features = qweight.shape[1] * self.pack_factor
        layer._int4_in_features = in_features
        layer._int4_in_features_padded = in_features
        layer._already_called_process_weights_after_loading = True


class INCXPUW4A16LinearScheme(INCXPULinearMethod):''')])
