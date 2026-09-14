"""A W8A8 int8 matmul with dequantization fused in, written in Triton.

    C[M, N] = (A_q[M, K] @ W_q[K, N]) * scale_a[M] * scale_b[N] + bias[N]

The int8 product accumulates exactly in int32 on tensor cores, and the two scales
and the bias are applied in the same kernel's epilogue, so nothing is
materialised between the product and the output.

W_q is taken as the transposed [K, N] view of an [N, K] nn.Linear weight, which
is also how vLLM passes it. That view is K-contiguous, the layout int8
tensor-core GEMMs are built for: in nanoinfer's GEMM benchmark the other layout
cost Triton 1.3x and cuBLAS 3.4x.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import triton
import triton.language as tl

TUNED_PATH = Path(__file__).with_name("tuned_configs.json")


class NotSupported(Exception):
    """The kernel does not take this shape or dtype, so the caller should fall back."""


@dataclass(frozen=True)
class Config:
    bm: int
    bn: int
    bk: int
    warps: int = 4
    stages: int = 3
    group_m: int = 1

    @property
    def name(self) -> str:
        return f"{self.bm}x{self.bn}x{self.bk} w{self.warps} s{self.stages} g{self.group_m}"


@triton.jit
def _w8a8_mm(
    a_ptr, b_ptr, sa_ptr, sb_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    SA_PER_ROW: tl.constexpr, SB_PER_COL: tl.constexpr, HAS_BIAS: tl.constexpr,
    EVEN_M: tl.constexpr, RAW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Which output tile this program owns. GROUP_M=1 is row-major tile order; a
    # larger GROUP_M walks GROUP_M rows of tiles together so concurrent programs
    # share the same A tiles in L2.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    first_pid_m = (pid // num_pid_in_group) * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 64-bit offsets: M * K passes 2**31 for long prefills at the wider layers.
    offs_m = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        # The launcher requires N and K to divide their tiles, so only the batch
        # axis can be ragged, and only then does a load need a mask.
        if EVEN_M:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0)
        acc += tl.dot(a, tl.load(b_ptrs), out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    if RAW:
        out = acc
    else:
        out = acc.to(tl.float32)
        if SA_PER_ROW:
            out = out * tl.load(sa_ptr + offs_m, mask=mask_m, other=0.0)[:, None]
        else:
            out = out * tl.load(sa_ptr)
        if SB_PER_COL:
            out = out * tl.load(sb_ptr + offs_n)[None, :]
        else:
            out = out * tl.load(sb_ptr)
        if HAS_BIAS:
            out = out + tl.load(bias_ptr + offs_n).to(tl.float32)[None, :]
        out = out.to(c_ptr.type.element_ty)
    if EVEN_M:
        tl.store(c_ptrs, out)
    else:
        tl.store(c_ptrs, out, mask=mask_m[:, None])


def default_config(m: int, n: int, k: int) -> Config:
    """The untuned choice. From the square-GEMM study: large tiles win once there
    are enough blocks to occupy the SMs, and lose below that."""
    if m >= 512:
        return Config(128, 128, 64, 4, 3, 8)
    if m >= 64:
        return Config(64, 128, 64, 4, 3, 1)
    return Config(16, 128, 128, 4, 3, 1)


_tuned: dict | None = None


def tuned_config(m: int, n: int, k: int) -> Config | None:
    """The fastest measured config for this GPU and shape, from the largest tuned
    batch bucket at or below M, or None if nothing has been tuned."""
    global _tuned
    if _tuned is None:
        _tuned = json.loads(TUNED_PATH.read_text()) if TUNED_PATH.exists() else {}
    table = _tuned.get(torch.cuda.get_device_name(), {}).get(f"{k}x{n}", {})
    fits = [int(bucket) for bucket in table if int(bucket) <= m]
    return Config(**table[str(max(fits))]["config"]) if fits else None


def w8a8_mm(
    a_q: torch.Tensor,
    b_q: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: torch.Tensor | None = None,
    *,
    config: Config | None = None,
    raw: bool = False,
) -> torch.Tensor:
    """(A_q @ B_q) * scale_a * scale_b + bias, for int8 A_q [M, K] and B_q [K, N].

    scale_a is per row ([M] or [M, 1]) or a scalar, scale_b per column ([N] or
    [N, 1]) or a scalar, both float32. With raw=True the exact int32 product is
    returned and the scales and bias are ignored.
    """
    if a_q.dtype != torch.int8 or b_q.dtype != torch.int8:
        raise NotSupported(f"int8 operands only, got {a_q.dtype} and {b_q.dtype}")
    if a_q.dim() != 2 or b_q.dim() != 2 or a_q.shape[1] != b_q.shape[0]:
        raise ValueError(f"cannot multiply {tuple(a_q.shape)} by {tuple(b_q.shape)}")
    m, k = a_q.shape
    n = b_q.shape[1]
    cfg = config or tuned_config(m, n, k) or default_config(m, n, k)
    if n % cfg.bn or k % cfg.bk:
        raise NotSupported(f"N={n} and K={k} must divide the {cfg.name} tile")
    # The kernel indexes scales and bias as flat stride-1 arrays; contiguous() is
    # free when they already are, and a copy only for a slice or a transpose.
    sa, sb = scale_a.reshape(-1).contiguous(), scale_b.reshape(-1).contiguous()
    if sa.dtype != torch.float32 or sb.dtype != torch.float32:
        raise NotSupported(f"float32 scales only, got {sa.dtype} and {sb.dtype}")
    if sa.numel() not in (1, m) or sb.numel() not in (1, n):
        raise ValueError(f"scale sizes {sa.numel()} and {sb.numel()} fit neither {m} nor {n}")
    if bias is not None and tuple(bias.shape) != (n,):
        raise ValueError(f"bias must have shape ({n},), got {tuple(bias.shape)}")
    if bias is not None:
        bias = bias.contiguous()

    out = torch.empty((m, n), dtype=torch.int32 if raw else out_dtype, device=a_q.device)
    grid = (triton.cdiv(m, cfg.bm) * (n // cfg.bn),)
    _w8a8_mm[grid](
        a_q, b_q, sa, sb, bias, out,
        m, n, k,
        a_q.stride(0), a_q.stride(1), b_q.stride(0), b_q.stride(1), out.stride(0), out.stride(1),
        SA_PER_ROW=sa.numel() > 1, SB_PER_COL=sb.numel() > 1, HAS_BIAS=bias is not None,
        EVEN_M=m % cfg.bm == 0, RAW=raw,
        BLOCK_M=cfg.bm, BLOCK_N=cfg.bn, BLOCK_K=cfg.bk, GROUP_M=cfg.group_m,
        num_warps=cfg.warps, num_stages=cfg.stages,
    )
    return out
