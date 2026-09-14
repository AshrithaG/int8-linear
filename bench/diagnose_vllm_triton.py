"""Is vLLM's triton_scaled_mm wrong on this GPU, or is the benchmark wrong?

bench_linear.py found vLLM's Triton int8 kernel failing a float64 check at
nearly every shape, while CUTLASS and this repo's kernel passed the same check
on the same operands. A benchmark that calls a vendor kernel wrong is usually a
wrong benchmark, so this isolates the kernel: fresh operands, no output
poisoning, the weight passed both ways (the transposed view vLLM's linear layers
pass, and the contiguous [K, N] tensor vLLM's unit test passes), and its
tensor-descriptor loads left at the default, forced off, and forced on.

    python bench/diagnose_vllm_triton.py
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from int8_linear.kernel import w8a8_mm  # noqa: E402

# (M, K, N): one ragged and three even batch sizes over Qwen3-1.7B's layer widths
CASES = [(1, 2048, 2048), (64, 2048, 2048), (64, 2048, 1024), (256, 6144, 2048)]
LAYOUTS = ("transposed view", "contiguous [K,N]")


def summarize(out: torch.Tensor, ref: torch.Tensor, rtol: float) -> str:
    out, ref = out.double(), ref.double()
    rel = (out - ref).abs() / ref.abs().clamp_min(1e-6)
    bad = rel > rtol
    exact = (out == ref).double().mean().item() * 100
    line = (f"exact {exact:6.2f}%  bad {bad.double().mean().item() * 100:6.2f}%  "
            f"bad rows {int(bad.any(1).sum())}/{out.shape[0]}  "
            f"bad cols {int(bad.any(0).sum())}/{out.shape[1]}  "
            f"zeros out/ref {int((out == 0).sum())}/{int((ref == 0).sum())}")
    if bad.any():
        i, j = (int(x) for x in bad.nonzero()[0])
        line += f"  first bad ({i},{j}) got {out[i, j].item():.6g} want {ref[i, j].item():.6g}"
    return line


def main() -> None:
    import vllm
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.compressed_tensors import (
        triton_scaled_mm as tsm_module,
    )

    tsm = tsm_module.triton_scaled_mm
    utd = tsm_module.use_tensor_descriptor
    print(f"vLLM {vllm.__version__}, torch {torch.__version__}, Triton {triton.__version__}, "
          f"{torch.cuda.get_device_name()}")
    print(f"use_tensor_descriptor(None, False, True) = {utd(None)}, {utd(False)}, {utd(True)}")
    try:
        print("\n" + inspect.getsource(utd))
    except (OSError, TypeError):
        pass

    g = torch.Generator(device="cuda").manual_seed(0)
    for m, k, n in CASES:
        print(f"\n===== M={m} K={k} N={n}")
        # Small integers, so an exact product is still exact after the kernel's
        # float32 cast: the largest |value| is K * 32 * 32, far below 2**24.
        a = torch.randint(-32, 32, (m, k), dtype=torch.int8, device="cuda", generator=g)
        w = torch.randint(-32, 32, (n, k), dtype=torch.int8, device="cuda", generator=g)
        product = a.double() @ w.double().t()
        sa = torch.rand((m, 1), device="cuda", generator=g) * 1e-2 + 1e-3
        sb = torch.rand((n, 1), device="cuda", generator=g) * 1e-2 + 1e-3
        bias = (torch.randn(n, device="cuda", generator=g) * 1e-2).to(torch.bfloat16)
        scaled = product * sa.double() * sb.double().t() + bias.double()
        ones_a, ones_b = torch.ones((m, 1), device="cuda"), torch.ones((n, 1), device="cuda")

        print("  sanity, this repo, int32 product:",
              summarize(w8a8_mm(a, w.t(), ones_a, ones_b, raw=True), product, 0.0))
        cutlass = ops.cutlass_scaled_mm(a, w.t(), sa, sb, torch.bfloat16, bias)
        print("  sanity, CUTLASS, scaled bf16:     ", summarize(cutlass, scaled, 1e-2))
        for layout in LAYOUTS:
            b = w.t() if layout == LAYOUTS[0] else w.t().contiguous()
            for td in (None, False, True):
                tag = f"  {layout:<17} use_td={str(td):<5}"
                try:
                    exact_out = tsm(a, b, ones_a, ones_b, torch.float32, None, use_td=td)
                    scaled_out = tsm(a, b, sa, sb, torch.bfloat16, bias, use_td=td)
                    torch.cuda.synchronize()
                except Exception as e:  # a variant that cannot run is still an answer
                    print(f"{tag} error: {str(e).strip().splitlines()[0][:150]}")
                    continue
                print(f"{tag} product: {summarize(exact_out, product, 0.0)}")
                print(f"{'':<31} scaled:  {summarize(scaled_out, scaled, 1e-2)}")


if __name__ == "__main__":
    main()
