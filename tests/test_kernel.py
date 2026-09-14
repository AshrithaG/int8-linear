"""Correctness of int8_linear.kernel on a CUDA GPU: python -m pytest -q tests"""

from __future__ import annotations

import pytest
import torch

from int8_linear.kernel import Config, NotSupported, w8a8_mm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

# (M, K, N): ragged and even batch sizes over Qwen3-1.7B's layer widths.
SHAPES = [(1, 2048, 2048), (3, 2048, 1024), (17, 2048, 6144), (64, 2048, 2048), (100, 6144, 2048)]


def ints(shape, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(-127, 128, shape, dtype=torch.int8, device="cuda", generator=g)


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
