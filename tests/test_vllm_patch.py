"""The custom ops vLLM calls: identical to their kernels, configured from the runtime
batch size under torch.compile, and falling back on declined shapes.
python -m pytest -q tests"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def operands(m, k, n, seed, low=-127, high=128):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randint(low, high, (m, k), dtype=torch.int8, device="cuda", generator=g)
    w = torch.randint(low, high, (n, k), dtype=torch.int8, device="cuda", generator=g)
    sa = torch.rand((m, 1), device="cuda", generator=g) * 1e-2 + 1e-3
    sb = torch.rand((n, 1), device="cuda", generator=g) * 1e-2 + 1e-3
    return a, w.t(), sa, sb


def test_op_matches_kernel():
    from int8_linear.kernel import w8a8_mm
    from int8_linear.vllm_patch import scaled_mm

    a, w, sa, sb = operands(64, 2048, 4096, 1)
    got = scaled_mm(a, w, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    assert torch.equal(got, w8a8_mm(a, w, sa, sb, torch.bfloat16))


def test_compiled_op_configures_from_runtime_batch():
    # If the configuration were chosen at trace time, both batch sizes would share it.
    from int8_linear import vllm_patch

    def layer(a, w, sa, sb):
        return vllm_patch.scaled_mm(a, w, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)

    compiled = torch.compile(layer, fullgraph=True, dynamic=True)
    vllm_patch.CONFIGS.clear()
    for m in (3, 700):
        compiled(*operands(m, 2048, 2048, m))
    chosen = {m: name for (m, _, _), name in vllm_patch.CONFIGS.items()}
    assert chosen[3] != chosen[700], chosen


def test_eager_calls_skip_the_custom_op(monkeypatch):
    from int8_linear import vllm_patch
    from int8_linear.kernel import w8a8_mm

    def custom_op(*args):
        raise AssertionError("an eager call went through the custom op")

    monkeypatch.setattr(vllm_patch, "w8a8_mm_op", custom_op)
    a, w, sa, sb = operands(16, 2048, 2048, 4)
    got = vllm_patch.scaled_mm(a, w, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    assert torch.equal(got, w8a8_mm(a, w, sa, sb, torch.bfloat16))


def test_declined_shapes_go_to_the_replaced_kernel(monkeypatch):
    from int8_linear import vllm_patch

    seen = []

    def replaced(a, w, sa, sb, out_dtype, bias):
        seen.append(tuple(a.shape))
        return torch.zeros((a.shape[0], w.shape[1]), dtype=out_dtype, device=a.device)

    monkeypatch.setattr(vllm_patch, "_vllm_triton_scaled_mm", replaced)
    before = vllm_patch.FALLBACKS
    a, w, sa, sb = operands(4, 100, 64, 2)  # K=100 divides no tile
    out = vllm_patch.scaled_mm(a, w, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    assert seen == [(4, 100)] and vllm_patch.FALLBACKS == before + 1
    assert tuple(out.shape) == (4, 64)


@pytest.mark.parametrize("m", [1, 33, 64, 300])
def test_vendored_pr45126_kernel_is_exact(m):
    # Small integers keep every accumulated sum below 2**24, where float32 is exact, so
    # with unit scales the output must equal the int8 product.
    from int8_linear import pr45126

    a, w, _, _ = operands(m, 2048, 1024, 3, low=-8, high=9)
    one = torch.ones(1, device="cuda")
    got = pr45126.triton_scaled_mm(a, w, one, one, torch.float32)
    assert torch.equal(got.double(), a.double() @ w.double())
