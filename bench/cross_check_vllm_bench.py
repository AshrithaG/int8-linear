"""Cross-check this repo's CUTLASS and bf16 timings with vLLM's own benchmark.

bench_linear.py measured vLLM's CUTLASS int8 kernel at close to bf16 speed on an
RTX 4090, and this repo's Triton kernel well ahead of both. Before the second
result means anything, the first has to survive a harness that is not this
repo's: vLLM's benchmarks/kernels/benchmark_int8_gemm.py, which builds its own
operands and times with triton.testing.do_bench_cudagraph. This calls that
script's benchmark function unmodified, at the shapes and batch sizes
bench_linear.py measured, and sets the two harnesses side by side.

vLLM's benchmark adds no bias; bench_linear.py's CUTLASS rows include one.

    python bench/cross_check_vllm_bench.py --vllm-bench DIR --ours results/<run>.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch

# vLLM benchmark provider -> bench_linear.py provider timing the same thing
PROVIDERS = {
    "torch-bf16": "bf16 F.linear (cuBLAS)",
    "int8-channel-w-token-a-noquant": "int8 CUTLASS (vLLM)",
    "int8-channel-w-token-a": "int8 CUTLASS (vLLM) + quant",
}
OURS_TUNED = "int8 Triton (this repo, tuned)"
BATCH = [1, 16, 64, 256, 1024, 4096]


def load_vllm_benchmark(directory: Path):
    sys.path.insert(0, str(directory))  # the script imports weight_shapes from its own folder
    spec = importlib.util.spec_from_file_location(
        "vllm_benchmark_int8_gemm", directory / "benchmark_int8_gemm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # triton.testing.perf_report wraps the function in a Mark; .fn is the original.
    return getattr(module.benchmark, "fn", module.benchmark)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vllm-bench", type=Path, required=True,
                    help="folder holding vLLM's benchmark_int8_gemm.py and weight_shapes.py")
    ap.add_argument("--ours", type=Path, required=True, help="a bench_linear.py results JSON")
    args = ap.parse_args()

    bench = load_vllm_benchmark(args.vllm_bench)
    ours = json.loads(args.ours.read_text())
    rows = {(r["k"], r["n"], r["m"], r["provider"]): r for r in ours["rows"]}

    def mine(k: int, n: int, m: int, provider: str) -> float | None:
        r = rows.get((k, n, m, provider))
        return r.get("graph_us") if r and r["status"] == "ok" else None

    records, ratios = [], {p: [] for p in PROVIDERS}
    print(f"{torch.cuda.get_device_name()}. Microseconds, both harnesses under CUDA-graph replay.")
    print(f"{'shape':<20} {'M':>5} {'provider':<32} {'vLLM bench':>11} {'bench_linear':>13} "
          f"{'ratio':>6} {'ours tuned':>11}")
    for label, k, n in ours["shapes"]:
        for m in BATCH:
            for provider, our_name in PROVIDERS.items():
                tflops, _, _ = bench(batch_size=m, provider=provider, N=n, K=k)
                theirs = 2 * m * n * k / (tflops * 1e12) * 1e6
                here = mine(k, n, m, our_name)
                tuned = mine(k, n, m, OURS_TUNED)
                ratio = here / theirs if here else None
                if ratio:
                    ratios[provider].append(ratio)
                records.append({"shape": label, "k": k, "n": n, "m": m, "provider": provider,
                                "vllm_bench_us": theirs, "bench_linear_us": here,
                                "ours_tuned_us": tuned})
                print(f"{label:<20} {m:>5} {provider:<32} {theirs:>11.2f} "
                      f"{here if here is None else f'{here:.2f}':>13} "
                      f"{'n/a' if ratio is None else f'{ratio:.2f}':>6} "
                      f"{tuned if tuned is None else f'{tuned:.2f}':>11}")

    print("\nbench_linear / vLLM bench, median and range per provider "
          "(near 1.00 means the harnesses agree):")
    for provider, rs in ratios.items():
        if rs:
            print(f"  {provider:<32} median {statistics.median(rs):.2f}  "
                  f"range {min(rs):.2f} to {max(rs):.2f}  over {len(rs)} points")
    out = args.ours.with_name(args.ours.stem + "_cross_check.json")
    out.write_text(json.dumps({"device": torch.cuda.get_device_name(), "rows": records}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
