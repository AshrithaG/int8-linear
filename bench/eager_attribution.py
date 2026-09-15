"""Where an eager decode step's host time goes, CUTLASS against this repo's kernel.

Without CUDA graphs this kernel decodes at 0.85x CUTLASS. Per call, its host cost through
vLLM's entry point is 5.2 microseconds above CUTLASS's (bench/launch_overhead.py), which
over 112 linear layers covers about 0.6 ms of a 3.0 ms longer step at batch 1. This times,
inside the served model, the host time per generated token spent in the linear layers'
apply_weights, in the activation quantizer and the matmul call inside it, and outside the
linear layers. The timers themselves add a few hundred nanoseconds per wrapped call, the
same for both backends.

Run once per backend, then --report:

    VLLM_ENABLE_V1_MULTIPROCESSING=0 python bench/eager_attribution.py --backend cutlass
    VLLM_ENABLE_V1_MULTIPROCESSING=0 python bench/eager_attribution.py --backend ours
    python bench/eager_attribution.py --report
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2e_vllm import W8A8_MODEL, W8A8_REVISION, gpu_processes, random_prompt  # noqa: E402

RESULTS = Path("results")
PARTS = ["apply_weights", "activation quantizer", "matmul call"]
TOTALS: dict[str, float] = defaultdict(float)
COUNTS: dict[str, int] = defaultdict(int)


def timed(name: str, fn):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            TOTALS[name] += time.perf_counter() - start
            COUNTS[name] += 1
    return wrapper


def build(backend: str):
    from vllm import LLM
    from vllm import _custom_ops as ops
    from vllm.model_executor.kernels.linear.scaled_mm import cutlass as cutlass_kernel
    from vllm.model_executor.kernels.linear.scaled_mm import triton as triton_kernel

    ops.scaled_int8_quant = timed("activation quantizer", ops.scaled_int8_quant)
    ops.cutlass_scaled_mm = timed("matmul call", ops.cutlass_scaled_mm)
    for cls in (cutlass_kernel.CutlassInt8ScaledMMLinearKernel,
                triton_kernel.TritonInt8ScaledMMLinearKernel):
        cls.apply_weights = timed("apply_weights", cls.apply_weights)
    if backend == "ours":
        from int8_linear.vllm_patch import patch_vllm

        patch_vllm()
        triton_kernel.triton_scaled_mm = timed("matmul call", triton_kernel.triton_scaled_mm)
    return LLM(model=W8A8_MODEL, revision=W8A8_REVISION,
               linear_backend="cutlass" if backend == "cutlass" else "triton",
               max_model_len=1024, gpu_memory_utilization=0.85, seed=0, dtype="bfloat16",
               enable_prefix_caching=False, enforce_eager=True)


def measure(llm, tokens: int, repeats: int) -> list[dict]:
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=tokens, ignore_eos=True)
    prompt = {"prompt_token_ids": random_prompt(llm, 64, seed=0)}
    llm.generate([prompt], params, use_tqdm=False)  # warmup
    runs = []
    for _ in range(repeats):
        TOTALS.clear()
        COUNTS.clear()
        torch.cuda.synchronize()
        start = time.perf_counter()
        llm.generate([prompt], params, use_tqdm=False)
        torch.cuda.synchronize()
        runs.append({"wall_s": time.perf_counter() - start,
                     "totals_s": dict(TOTALS), "counts": dict(COUNTS)})
    return runs


def per_token_ms(run: dict, tokens: int) -> dict[str, float]:
    t = run["totals_s"]
    apply = t.get("apply_weights", 0.0)
    inner = t.get("activation quantizer", 0.0) + t.get("matmul call", 0.0)
    out = {"whole step": run["wall_s"], "outside the linear layers": run["wall_s"] - apply,
           **{p: t.get(p, 0.0) for p in PARTS}, "rest of apply_weights": apply - inner}
    return {k: v / tokens * 1e3 for k, v in out.items()}


def report() -> None:
    runs = {b: json.loads((RESULTS / f"eager_attribution_{b}.json").read_text())
            for b in ("cutlass", "ours")}
    rows = {}
    for b, r in runs.items():
        per = [per_token_ms(x, r["tokens"]) for x in r["runs"]]
        rows[b] = {k: statistics.median(p[k] for p in per) for k in per[0]}
    env = runs["ours"]["env"]
    out = [f"# Host time of an eager decode step on {env['device']}", "",
           f"vLLM {env['vllm']}, torch {env['torch']}. Batch 1, {runs['ours']['tokens']} "
           f"generated tokens, median of {len(runs['ours']['runs'])} runs, milliseconds per "
           "token. Linear-layer parts are host time: kernels launched from them run on the GPU "
           "asynchronously.", "",
           "| | CUTLASS | this repo | difference |", "|---|---|---|---|"]
    for k in rows["ours"]:
        out.append(f"| {k} | {rows['cutlass'][k]:.3f} | {rows['ours'][k]:.3f} | "
                   f"{rows['ours'][k] - rows['cutlass'][k]:+.3f} |")
    calls = {b: {k: v / r["tokens"] for k, v in r["runs"][0]["counts"].items()}
             for b, r in runs.items()}
    out += ["", f"Calls per token: CUTLASS {calls['cutlass']}; this repo {calls['ours']}."]
    (RESULTS / "eager_attribution.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=("cutlass", "ours"))
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--report", action="store_true", help="combine the two backends' results")
    args = ap.parse_args()
    if args.report:
        report()
        return
    if not args.backend:
        raise SystemExit("give --backend or --report")
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise SystemExit("set VLLM_ENABLE_V1_MULTIPROCESSING=0 so the timers reach the model")
    import vllm

    llm = build(args.backend)
    res = {"backend": args.backend, "tokens": args.tokens,
           "env": {"device": torch.cuda.get_device_name(), "vllm": vllm.__version__,
                   "torch": torch.__version__},
           "interference": {"before": gpu_processes()},
           "runs": measure(llm, args.tokens, args.repeats)}
    res["interference"]["after"] = gpu_processes()
    path = RESULTS / f"eager_attribution_{args.backend}.json"
    path.write_text(json.dumps(res, indent=1))
    print({k: round(v, 3) for k, v in per_token_ms(res["runs"][-1], args.tokens).items()})
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
