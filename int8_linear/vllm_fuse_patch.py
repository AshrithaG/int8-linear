"""Route Qwen3's RMSNorms and SiLU-and-mul through the fused quantizing kernels.

vLLM 0.28 has the plumbing for this, a pre-quantized activation a linear kernel
can consume (vllm/model_executor/layers/fusion/quant_activation.py), but the int8
kernels do not consume one and Qwen3 does not produce one. This patch does both for int8,
in the process that builds the model (VLLM_ENABLE_V1_MULTIPROCESSING=0):

  * the int8 linear kernels (Triton and CUTLASS) accept a PreQuant input and go
    straight to the matmul, skipping their own scaled_int8_quant;
  * Qwen3's decoder layer feeds qkv_proj and gate_up_proj from add_rmsnorm_quant,
    and its MLP feeds down_proj from silu_mul_quant.

o_proj still quantizes its own input: attention's output has no kernel before it
to fuse into. The residual stream, the norms' weights and everything else are
unchanged. PREQUANT_CALLS counts matmuls that received a PreQuant, so a patch
that silently never fired shows up as zero.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from int8_linear import fused_quant
from int8_linear.fused_quant import add_rmsnorm_quant_op, rmsnorm_quant_op, silu_mul_quant_op

PREQUANT_CALLS = 0


@dataclass
class PreQuant:
    """An int8 activation quantized per token, with the dtype the matmul should output."""
    data: torch.Tensor
    scale: torch.Tensor
    dtype: torch.dtype


def _decoder_forward(self, positions, hidden_states, residual):
    ln = self.input_layernorm
    if residual is None:
        residual = hidden_states
        q, s = rmsnorm_quant_op(hidden_states, ln.weight, ln.variance_epsilon)
    else:
        q, s, residual = add_rmsnorm_quant_op(hidden_states, residual, ln.weight, ln.variance_epsilon)
    hidden_states = self.self_attn(positions=positions, hidden_states=PreQuant(q, s, residual.dtype))
    ln = self.post_attention_layernorm
    q, s, residual = add_rmsnorm_quant_op(hidden_states, residual, ln.weight, ln.variance_epsilon)
    hidden_states = self.mlp(PreQuant(q, s, residual.dtype))
    return hidden_states, residual


def _mlp_forward(self, x):
    gate_up, _ = self.gate_up_proj(x)
    q, s = silu_mul_quant_op(gate_up)
    out, _ = self.down_proj(PreQuant(q, s, gate_up.dtype))
    return out


def _wrap_apply(cls, matmul):
    original = cls.apply_weights

    def apply_weights(self, layer, x, bias=None):
        global PREQUANT_CALLS
        if not isinstance(x, PreQuant):
            return original(self, layer, x, bias)
        w_q, w_s, i_s, i_zp, azp_adj = self._get_layer_params(layer)
        if i_s is not None or azp_adj is not None:
            raise NotImplementedError("PreQuant covers symmetric dynamic per-token int8 only")
        PREQUANT_CALLS += 1
        return matmul(x.data, w_q, x.scale, w_s, x.dtype, bias)

    apply_weights._fuse_patched = True
    cls.apply_weights = apply_weights


def use_measured_numerics(path: Path = Path("results/fuse_numerics.json")) -> None:
    """Set the fused kernels' division and SiLU rounding to whichever choice
    bench/fuse_quant.py found closest to vLLM's unfused kernels on this machine."""
    if path.exists():
        num = json.loads(path.read_text())
        fused_quant.PRECISE_DIV = num["best_precise_div"]
        fused_quant.ROUND_SILU = num["best_round_silu"]


def patch_vllm() -> None:
    use_measured_numerics()
    from vllm import _custom_ops as ops
    from vllm.model_executor.kernels.linear.scaled_mm import cutlass, triton as vllm_triton
    from vllm.model_executor.models import qwen2, qwen3

    if not getattr(vllm_triton.TritonInt8ScaledMMLinearKernel.apply_weights, "_fuse_patched", False):
        # Through the module attribute, so int8_linear.vllm_patch's kernel swap is honoured.
        _wrap_apply(vllm_triton.TritonInt8ScaledMMLinearKernel,
                    lambda a, w, sa, sb, dt, b: vllm_triton.triton_scaled_mm(
                        a, w, scale_a=sa, scale_b=sb, out_dtype=dt, bias=b))
        _wrap_apply(cutlass.CutlassInt8ScaledMMLinearKernel,
                    lambda a, w, sa, sb, dt, b: ops.cutlass_scaled_mm(
                        a, w, scale_a=sa, scale_b=sb, out_dtype=dt, bias=b))
    qwen3.Qwen3DecoderLayer.forward = _decoder_forward
    qwen2.Qwen2MLP.forward = _mlp_forward
