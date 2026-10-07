"""The activation quantizer fused into the kernel before it, in Triton.

A W8A8 int8 layer quantizes its input per token right before the matmul. In
vLLM 0.28 that is its own kernel, dynamic_scaled_int8_quant, launched after the
RMSNorm (or SiLU-and-mul) that produced the input: one more launch, and one more
write and read of the activation. vLLM fuses this pair for FP8 (its RMSNorm-quant
compile pass), but that pass only matches FP8 quantizers, so int8 models still
run both kernels, even though vLLM's own fused CUDA kernel,
rms_norm_dynamic_per_token_quant, accepts int8 output.

Two fused kernels, one Triton program per token row:

    add_rmsnorm_quant(x, residual, w, eps) -> (x_q, scale, residual_out)
    rmsnorm_quant(x, w, eps)               -> (x_q, scale)
    silu_mul_quant(gate_up)                -> (x_q, scale)

They follow vLLM's arithmetic step by step, so they can be compared bit for bit:
the residual add and the norm in fp32 (vllm/ir/ops/layernorm.py), the normalized
value rounded to the activation dtype before and after the weight multiply, then
dynamic_scaled_int8_quant's quantizer (csrc/.../w8a8/int8/scaled_quant.cu):

    scale = absmax / 127,   x_q = round_half_even(x * (127 / absmax)), saturated

PRECISE_DIV picks IEEE division for those two divides (CUDA's default without
fast-math) or Triton's default approximate one; bench/fuse_quant.py reports which
matches vLLM. ROUND_SILU rounds silu(gate) to the activation dtype before the
multiply, as vLLM's silu_and_mul CUDA kernel does; Inductor's compiled native
SiLU-and-mul keeps it in fp32.

Each is also a torch custom op, so torch.compile treats it as one opaque kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _quantize_row(y, mask, q_ptr, s_ptr, row, cols, PRECISE_DIV: tl.constexpr):
    absmax = tl.max(tl.where(mask, tl.abs(y), 0.0), axis=0)
    if PRECISE_DIV:
        scale = tl.math.div_rn(absmax, 127.0)
        inv = tl.where(absmax == 0.0, 0.0, tl.math.div_rn(127.0, absmax))
    else:
        scale = absmax / 127.0
        inv = tl.where(absmax == 0.0, 0.0, 127.0 / absmax)
    tl.store(s_ptr + row, scale)
    q = libdevice.rint(y * inv)
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(q_ptr + cols, q.to(tl.int8), mask=mask)   # q_ptr already points at this row


@triton.jit
def _add_rmsnorm_quant_kernel(x_ptr, r_ptr, w_ptr, q_ptr, s_ptr, rout_ptr, K, eps,
                              HAS_RES: tl.constexpr, PRECISE_DIV: tl.constexpr,
                              BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < K
    base = row * K
    x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RES:
        x = x + tl.load(r_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(rout_ptr + base + cols, x.to(rout_ptr.dtype.element_ty), mask=mask)
    var = tl.sum(x * x, axis=0) / K
    xn = x * libdevice.rsqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    # vLLM: x.to(weight.dtype) * weight, then .to(orig_dtype)
    y = (xn.to(w.dtype).to(tl.float32) * w.to(tl.float32)).to(x_ptr.dtype.element_ty).to(tl.float32)
    _quantize_row(y, mask, q_ptr + base, s_ptr, row, cols, PRECISE_DIV)


@triton.jit
def _silu_mul_quant_kernel(x_ptr, q_ptr, s_ptr, I, PRECISE_DIV: tl.constexpr,
                           ROUND_SILU: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < I
    g = tl.load(x_ptr + row * 2 * I + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(x_ptr + row * 2 * I + I + cols, mask=mask, other=0.0).to(tl.float32)
    a = g / (1.0 + libdevice.exp(-g))
    if ROUND_SILU:
        a = a.to(x_ptr.dtype.element_ty).to(tl.float32)
    y = (a * u).to(x_ptr.dtype.element_ty).to(tl.float32)
    _quantize_row(y, mask, q_ptr + row * I, s_ptr, row, cols, PRECISE_DIV)


def _warps(block: int) -> int:
    return 4 if block <= 1024 else 8 if block <= 4096 else 16


def add_rmsnorm_quant(x: torch.Tensor, residual: torch.Tensor | None, weight: torch.Tensor,
                      eps: float, precise_div: bool = True):
    """(x_q int8 [M, K], scale fp32 [M, 1], residual_out [M, K] or None)."""
    assert x.is_contiguous() and x.dim() == 2
    M, K = x.shape
    q = torch.empty((M, K), dtype=torch.int8, device=x.device)
    s = torch.empty((M, 1), dtype=torch.float32, device=x.device)
    rout = torch.empty_like(x) if residual is not None else None
    block = triton.next_power_of_2(K)
    if M:
        _add_rmsnorm_quant_kernel[(M,)](
            x, residual if residual is not None else x, weight, q, s,
            rout if rout is not None else x, K, eps,
            HAS_RES=residual is not None, PRECISE_DIV=precise_div, BLOCK=block,
            num_warps=_warps(block))
    return q, s, rout


def silu_mul_quant(gate_up: torch.Tensor, precise_div: bool = True, round_silu: bool = False):
    """silu(gate) * up, quantized: (x_q int8 [M, I], scale fp32 [M, 1])."""
    assert gate_up.is_contiguous() and gate_up.dim() == 2
    M, I2 = gate_up.shape
    I = I2 // 2
    q = torch.empty((M, I), dtype=torch.int8, device=gate_up.device)
    s = torch.empty((M, 1), dtype=torch.float32, device=gate_up.device)
    block = triton.next_power_of_2(I)
    if M:
        _silu_mul_quant_kernel[(M,)](gate_up, q, s, I, PRECISE_DIV=precise_div,
                                     ROUND_SILU=round_silu, BLOCK=block, num_warps=_warps(block))
    return q, s


# Numerics the custom ops use; bench/fuse_quant.py measures which choice matches vLLM.
PRECISE_DIV = True
ROUND_SILU = False


@torch.library.custom_op("int8_linear::rmsnorm_quant", mutates_args=())
def rmsnorm_quant_op(x: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    q, s, _ = add_rmsnorm_quant(x.contiguous(), None, weight, eps, PRECISE_DIV)
    return q, s


@rmsnorm_quant_op.register_fake
def _(x, weight, eps):
    return (x.new_empty(x.shape, dtype=torch.int8),
            x.new_empty((x.shape[0], 1), dtype=torch.float32))


@torch.library.custom_op("int8_linear::add_rmsnorm_quant", mutates_args=())
def add_rmsnorm_quant_op(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                         eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return add_rmsnorm_quant(x.contiguous(), residual.contiguous(), weight, eps, PRECISE_DIV)


@add_rmsnorm_quant_op.register_fake
def _(x, residual, weight, eps):
    return (x.new_empty(x.shape, dtype=torch.int8),
            x.new_empty((x.shape[0], 1), dtype=torch.float32),
            torch.empty_like(x))


@torch.library.custom_op("int8_linear::silu_mul_quant", mutates_args=())
def silu_mul_quant_op(gate_up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return silu_mul_quant(gate_up.contiguous(), PRECISE_DIV, ROUND_SILU)


@silu_mul_quant_op.register_fake
def _(gate_up):
    return (gate_up.new_empty((gate_up.shape[0], gate_up.shape[1] // 2), dtype=torch.int8),
            gate_up.new_empty((gate_up.shape[0], 1), dtype=torch.float32))
