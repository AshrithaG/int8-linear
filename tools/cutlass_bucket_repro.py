"""Standalone repro: vLLM's int8 CUTLASS kernel on Ada slows down past M=16 and M=32.

vLLM's sm_89 int8 dispatch (csrc/libtorch_stable/quantization/w8a8/cutlass/
scaled_mm_c2x_sm89_int8_dispatch.cuh) picks a compiled GEMM configuration by
next_pow_2(M), floored at 16. This times cutlass_scaled_mm under CUDA-graph replay on
both sides of the 16 and 32 edges for Qwen3-1.7B's linear layers as vLLM runs them,
next to bf16 F.linear. It needs only torch and vLLM, on an sm_89 GPU:

    python tools/cutlass_bucket_repro.py
"""

import statistics

import torch
import torch.nn.functional as F
import vllm
from vllm import _custom_ops as ops

# (layer, input K, output N) for Qwen3-1.7B with q, k, v and gate, up merged.
LAYERS = [("qkv_proj", 2048, 4096), ("o_proj", 2048, 2048),
          ("gate_up_proj", 2048, 12288), ("down_proj", 6144, 2048)]
BATCH = [16, 17, 32, 33, 64]


def replay_us(fn, iters: int = 100, windows: int = 7) -> float:
    """Median microseconds per call, with `iters` calls captured in one CUDA graph."""
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    graph.replay()
    times = []
    for _ in range(windows):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop) * 1e3 / iters)
    return statistics.median(times)


def main() -> None:
    torch.manual_seed(0)
    major, minor = torch.cuda.get_device_capability()
    print(f"{torch.cuda.get_device_name()} (sm_{major}{minor}), torch {torch.__version__}, "
          f"vLLM {vllm.__version__}")
    print("microseconds per call, CUDA-graph replay, median of 7 windows of 100 calls")
    print(f"{'layer':<14}{'M':>4}{'CUTLASS':>10}{'bf16':>8}{'CUTLASS step':>14}")
    for name, k, n in LAYERS:
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
        w_s = w.float().abs().amax(dim=1, keepdim=True) / 127
        w_q = (w.float() / w_s).round().clamp(-127, 127).to(torch.int8).t()
        previous = None
        for m in BATCH:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            x_q, x_s, _ = ops.scaled_int8_quant(x)
            cutlass = replay_us(lambda: ops.cutlass_scaled_mm(x_q, w_q, x_s, w_s, torch.bfloat16))
            bf16 = replay_us(lambda: F.linear(x, w))
            step = f"{cutlass / previous:.2f}x" if previous and m in (17, 33) else ""
            print(f"{name:<14}{m:>4}{cutlass:>10.2f}{bf16:>8.2f}{step:>14}")
            previous = cutlass


if __name__ == "__main__":
    main()
