"""Correctness of int8_linear.kernel on a CUDA GPU: python -m pytest -q tests"""

from __future__ import annotations

import pytest
import torch

import int8_linear.kernel as kernel_module
from int8_linear.kernel import (
    Config,
    NotSupported,
    _launch_args,
    _launch_key,
    _w8a8_mm,
    choose_config,
    w8a8_mm,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

# (M, K, N): ragged and even batch sizes over Qwen3-1.7B's layer widths.
SHAPES = [(1, 2048, 2048), (3, 2048, 1024), (17, 2048, 6144), (64, 2048, 2048), (100, 6144, 2048)]


def ints(shape, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(-127, 128, shape, dtype=torch.int8, device="cuda", generator=g)


def offset_ints(shape, seed):
    """int8 values stored one byte into their buffer, so the data pointer is odd."""
    flat = ints((shape[0] * shape[1] + 1,), seed)
    return flat[1:].view(shape)


def exact(a, w):
    # float64 matmul of int8 values is exact: every partial sum is far below 2**53
    return (a.double() @ w.double().t()).to(torch.int32)


@pytest.mark.parametrize("m,k,n", SHAPES)
def test_int32_product_is_exact(m, k, n):
    a, w = ints((m, k), 1), ints((n, k), 2)
    one = torch.ones(1, device="cuda")
    assert torch.equal(w8a8_mm(a, w.t(), one, one, raw=True), exact(a, w))


@pytest.mark.parametrize("config", [Config(16, 64, 64, 4, 1, 1), Config(64, 128, 128, 8, 3, 8)])
def test_exact_under_other_tilings(config):
    m, k, n = 256, 2048, 1024
    a, w = ints((m, k), 3), ints((n, k), 4)
    one = torch.ones(1, device="cuda")
    assert torch.equal(w8a8_mm(a, w.t(), one, one, raw=True, config=config), exact(a, w))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("per_row", [True, False])
@pytest.mark.parametrize("per_col", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
def test_scaled_output_matches_float64(dtype, per_row, per_col, use_bias):
    m, k, n = 37, 2048, 1024
    a, w = ints((m, k), 5), ints((n, k), 6)
    g = torch.Generator(device="cuda").manual_seed(7)
    sa = torch.rand((m, 1) if per_row else (1,), device="cuda", generator=g) * 1e-2 + 1e-3
    sb = torch.rand((n, 1) if per_col else (1,), device="cuda", generator=g) * 1e-2 + 1e-3
    bias = torch.randn(n, device="cuda", generator=g).to(dtype) if use_bias else None
    got = w8a8_mm(a, w.t(), sa, sb, dtype, bias)
    term = exact(a, w).double() * sa.double().reshape(-1, 1) * sb.double().reshape(1, -1)
    ref = term + bias.double() if use_bias else term
    # Error relative to the larger of the output and its matmul term. Where the bias
    # nearly cancels the term, float32 rounding of the term is large next to the tiny
    # sum without anything being wrong. One bf16 rounding step is about 0.8% of a value.
    scale = torch.maximum(ref.abs(), term.abs()).clamp_min(1e-6)
    rel = ((got.double() - ref).abs() / scale).max().item()
    assert got.dtype == dtype and rel < 1e-2


def test_declines_shapes_the_tiles_do_not_divide():
    a, w = ints((4, 100), 8), ints((64, 100), 9)
    one = torch.ones(1, device="cuda")
    with pytest.raises(NotSupported):
        w8a8_mm(a, w.t(), one, one)


def test_replays_inside_a_cuda_graph():
    m, k, n = 16, 2048, 2048
    a, w = ints((m, k), 10), ints((n, k), 11)
    sa = torch.full((m, 1), 1e-2, device="cuda")
    sb = torch.full((n, 1), 1e-2, device="cuda")
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        w8a8_mm(a, w.t(), sa, sb)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = w8a8_mm(a, w.t(), sa, sb)
    a.copy_(ints((m, k), 12))  # new input in the same buffer
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(static_out, w8a8_mm(a, w.t(), sa, sb))


# Each case changes something Triton specializes on: M equal to 1 or not divisible
# by 16, an input pointer not divisible by 16, a scalar or per-row scale, a bias and
# its dtype, the output dtype.
LAUNCH_CASES = [
    (1, 2048, 2048, False, True, None, torch.bfloat16),
    (1, 2048, 1024, True, False, None, torch.bfloat16),
    (16, 2048, 2048, False, True, torch.bfloat16, torch.bfloat16),
    (17, 2048, 6144, True, True, None, torch.float16),
    (64, 6144, 2048, False, False, torch.float32, torch.bfloat16),
]
needs_cached_launch = pytest.mark.skipif(
    not kernel_module.CACHED_LAUNCH, reason="the launch cache is off for this Triton version")


@needs_cached_launch
@pytest.mark.parametrize("m,k,n,odd_ptr,per_row,bias_dtype,dtype", LAUNCH_CASES)
def test_cached_launch_runs_the_kernel_triton_picks(m, k, n, odd_ptr, per_row, bias_dtype, dtype):
    a = offset_ints((m, k), 20) if odd_ptr else ints((m, k), 20)
    w = ints((n, k), 21)
    g = torch.Generator(device="cuda").manual_seed(22)
    sa = torch.rand((m, 1) if per_row else (1,), device="cuda", generator=g) * 1e-2 + 1e-3
    sb = torch.rand((n, 1), device="cuda", generator=g) * 1e-2 + 1e-3
    bias = torch.randn(n, device="cuda", generator=g).to(bias_dtype) if bias_dtype else None
    through_jit = w8a8_mm(a, w.t(), sa, sb, dtype, bias, cached_launch=False)
    w8a8_mm(a, w.t(), sa, sb, dtype, bias)  # fills the cache if this key is new
    cached = w8a8_mm(a, w.t(), sa, sb, dtype, bias)
    assert torch.equal(cached, through_jit)

    cfg = choose_config(m, n, k)
    out = torch.empty((m, n), dtype=dtype, device="cuda")
    args, grid = _launch_args(a, w.t(), sa.reshape(-1), sb.reshape(-1), bias, out, cfg, False)
    picked = _w8a8_mm.run(*args, grid=grid, warmup=True, num_warps=cfg.warps,
                          num_stages=cfg.stages)
    assert kernel_module._launches[_launch_key(args, cfg)] is picked


@needs_cached_launch
def test_cached_launch_is_exact_on_an_odd_pointer():
    m, k, n = 5, 2048, 1024
    a, w = offset_ints((m, k), 23), ints((n, k), 24)
    one = torch.ones(1, device="cuda")
    for _ in range(3):  # later calls launch from the cache
        assert torch.equal(w8a8_mm(a, w.t(), one, one, raw=True), exact(a, w))
