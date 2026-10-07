"""The fused norm/SiLU + int8 quantizer kernels and the vLLM patch's matmul routing.

python -m pytest -q tests/test_fused_quant.py   (needs a CUDA GPU)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

if torch.cuda.is_available():
    from int8_linear import fused_quant as fq
    from int8_linear import vllm_fuse_patch as vfp

EPS = 1e-6


def rand(shape, seed, scale=1.0, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(shape, generator=g, device="cuda") * scale).to(dtype)


def quant(y):
    """vLLM's dynamic_scaled_int8_quant, in torch: IEEE divides, round half to even."""
    yf = y.float()
    absmax = yf.abs().amax(-1, keepdim=True)
    inv = torch.where(absmax == 0, torch.zeros_like(absmax), 127.0 / absmax)
    return (torch.round(yf * inv).clamp(-128, 127).to(torch.int8), absmax / 127.0)


def ref_norm(x, r, w):
    xf = x.float()
    rout = None
    if r is not None:
        xf = xf + r.float()
        rout = xf.to(x.dtype)
    var = xf.pow(2).mean(-1, keepdim=True)
    y = ((xf * torch.rsqrt(var + EPS)).to(w.dtype) * w).to(x.dtype)
    return (*quant(y), rout)


def ref_silu(gu, round_silu):
    i = gu.shape[1] // 2
    g, u = gu[:, :i].float(), gu[:, i:].float()
    a = g / (1.0 + torch.exp(-g))
    if round_silu:
        a = a.to(gu.dtype).float()
    return quant((a * u).to(gu.dtype))


def assert_close_int8(q, s, q0, s0):
    d = (q.int() - q0.int()).abs()
    assert d.max().item() <= 1                              # never more than one step
    assert d.ne(0).float().mean().item() < 1e-3             # and almost always exact
    assert ((s - s0).abs() / s0.clamp_min(1e-30)).max().item() < 1e-5


@pytest.mark.parametrize("m,k", [(1, 2048), (7, 2048), (64, 1024), (300, 2048), (5, 1000)])
@pytest.mark.parametrize("has_res", [False, True])
def test_add_rmsnorm_quant(m, k, has_res):
    x = rand((m, k), m)
    r = rand((m, k), m + 1) if has_res else None
    w = rand((k,), 99, 0.1) + 1.0
    q, s, rout = fq.add_rmsnorm_quant(x, r, w, EPS, precise_div=True)
    q0, s0, r0 = ref_norm(x, r, w)
    assert_close_int8(q, s, q0, s0)
    if has_res:
        assert torch.equal(rout, r0)                        # the residual stream must not drift
    assert q.dtype == torch.int8 and s.shape == (m, 1)


@pytest.mark.parametrize("m,i", [(1, 6144), (33, 6144), (8, 1000)])
@pytest.mark.parametrize("round_silu", [False, True])
def test_silu_mul_quant(m, i, round_silu):
    gu = rand((m, 2 * i), m, 2.0)
    q, s = fq.silu_mul_quant(gu, precise_div=True, round_silu=round_silu)
    assert_close_int8(q, s, *ref_silu(gu, round_silu))


def test_zero_row_gives_zero_scale_and_values():
    x = torch.zeros((3, 2048), device="cuda", dtype=torch.bfloat16)
    w = torch.ones(2048, device="cuda", dtype=torch.bfloat16)
    q, s, _ = fq.add_rmsnorm_quant(x, None, w, EPS)
    assert q.abs().max().item() == 0 and s.abs().max().item() == 0


def test_saturates_at_127():
    """The largest-magnitude element of every row must land on +-127 exactly."""
    gu = rand((16, 2 * 512), 5, 3.0)
    q, _ = fq.silu_mul_quant(gu)
    assert q.int().abs().amax(-1).eq(127).all()


def test_custom_ops_opcheck():
    x, r = rand((9, 2048), 1), rand((9, 2048), 2)
    w = rand((2048,), 3) + 1.0
    torch.library.opcheck(fq.rmsnorm_quant_op, (x, w, EPS))
    torch.library.opcheck(fq.add_rmsnorm_quant_op, (x, r, w, EPS))
    torch.library.opcheck(fq.silu_mul_quant_op, (rand((9, 2 * 6144), 4),))


def test_prequant_routing():
    """A PreQuant goes straight to the matmul; a plain tensor takes the original path."""
    calls = []

    class Kernel:
        def _get_layer_params(self, layer):
            return "w_q", "w_s", None, None, None

        def apply_weights(self, layer, x, bias=None):
            calls.append("original")
            return "orig"

    vfp._wrap_apply(Kernel, lambda a, w, sa, sb, dt, b: calls.append((a, w, sa, sb, dt, b)) or "fast")
    before = vfp.PREQUANT_CALLS
    k = Kernel()
    assert k.apply_weights(None, vfp.PreQuant("q", "s", torch.bfloat16), "b") == "fast"
    assert calls[-1] == ("q", "w_q", "s", "w_s", torch.bfloat16, "b")
    assert vfp.PREQUANT_CALLS == before + 1
    assert k.apply_weights(None, torch.zeros(1)) == "orig"


def test_prequant_rejects_static_or_asymmetric():
    class Kernel:
        def _get_layer_params(self, layer):
            return "w_q", "w_s", "static_scale", None, None

        def apply_weights(self, layer, x, bias=None):
            return "orig"

    vfp._wrap_apply(Kernel, lambda *a: "fast")
    with pytest.raises(NotImplementedError):
        Kernel().apply_weights(None, vfp.PreQuant("q", "s", torch.bfloat16))
