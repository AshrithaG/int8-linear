"""W8A8 int8 linear layers on one GPU: this repo's Triton kernel against vLLM's
CUTLASS and Triton kernels, the Triton tables proposed in vLLM #45126, an unfused
torch._int_mm path, and bf16 cuBLAS.

For every linear-layer shape and number of tokens M, each provider computes the
same layer, A @ W^T (plus a bias with --bias), from the same quantized operands,
and must agree with a float64 reference before it is timed. Time is CUDA-graph
replay, which is how vLLM runs decode and how vLLM's own int8 GEMM benchmark
times kernels; back-to-back launches are reported next to it because that is how
most kernel benchmarks are still run.

Bias is off by default. The Qwen3 and Llama linear layers measured here have
none, and in the first run a bias slowed CUTLASS by a median 1.33x against vLLM's
own measurement of the same kernel.

On a CUDA machine, from the repo root:

    python bench/bench_linear.py --shapes qwen3-1.7b --tune
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
import triton

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from int8_linear import pr45126  # noqa: E402
from int8_linear.kernel import (  # noqa: E402
    TUNED_PATH,
    Config,
    NotSupported,
    default_config,
    tuned_config,
    w8a8_mm,
)
from int8_linear.shapes import SHAPES  # noqa: E402

BATCH = [1, 4, 16, 64, 256, 1024, 4096]
# Launches per timing window, so each window spans at least a few milliseconds.
ITERS = {1: 200, 4: 200, 16: 100, 64: 50, 256: 20, 1024: 10, 4096: 5}
OUT_DTYPE = torch.bfloat16
# Error is measured relative to the larger of the output and its matmul term, not
# the output alone. With a bias the two can nearly cancel, and a kernel that rounds
# the product to bf16 before adding the bias (vLLM's Triton kernel does) is then a
# whole rounding step of the term away from a near-zero sum without being wrong.
# bf16 keeps 7 fraction bits, about 0.8% per step; an indexing or scaling bug is
# off by far more than 1%.
RTOL = 1e-2
SMEM_LIMIT = 99 * 1024  # the most shared memory one block can opt into on sm_89

BF16 = "bf16 F.linear (cuBLAS)"
CUTLASS = "int8 CUTLASS (vLLM)"
VLLM_TRITON = "int8 Triton (vLLM)"
# vLLM's Triton kernel with the tuned tables from vllm-project/vllm#45126, vendored in
# int8_linear/pr45126.py so it runs next to the shipped one.
PR45126 = "int8 Triton (vLLM #45126 tables)"
INT_MM = "int8 torch._int_mm, unfused"
OURS_DEFAULT = "int8 Triton (this repo, default)"
OURS_TUNED = "int8 Triton (this repo, tuned)"
# Configurations read from int8_linear/tuned_configs.json rather than chosen in this
# run, so a run without --tune measures them without the optimism of picking the
# fastest of a sweep on the same timings it then reports.
OURS_TABLE = "int8 Triton (this repo, stored table)"
CUTLASS_Q = "int8 CUTLASS (vLLM) + quant"
OURS_Q = "int8 Triton (this repo) + quant"
INT8 = [CUTLASS, VLLM_TRITON, PR45126, INT_MM, OURS_DEFAULT, OURS_TUNED, OURS_TABLE]


def first_line(e: BaseException) -> str:
    text = str(e).strip()
    return (text.splitlines()[0] if text else type(e).__name__)[:200]


def load_vllm():
    try:
        import vllm
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
            triton_scaled_mm,
        )
    except Exception as e:  # the benchmark still runs without vLLM, minus its columns
        return None, None, f"unavailable ({first_line(e)})"
    return ops, triton_scaled_mm, vllm.__version__


# ------------------------------------------------------------------------- timing
def time_eager(fn: Callable[[], object], iters: int, windows: int) -> list[float]:
    """Back-to-back launches between two CUDA events, per launch, in milliseconds."""
    fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(windows):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        stop.record()
        stop.synchronize()
        out.append(start.elapsed_time(stop) / iters)
    return out


def time_graph(fn: Callable[[], object], iters: int, windows: int) -> list[float]:
    """The same launches captured once into a CUDA graph and replayed, so the CPU
    work of each launch happens at capture and not inside the timed window."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(windows):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        out.append(start.elapsed_time(stop) / iters)
    del graph
    return out


# ---------------------------------------------------------------------- operands
def quant_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int8 with one float32 scale per row: per-token for activations,
    per-channel for an [N, K] weight."""
    scale = x.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
    return (x.float() / scale).round().clamp(-127, 127).to(torch.int8), scale


def operands(m: int, k: int, n: int, seed: int, with_bias: bool) -> dict:
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn((m, k), generator=g, device="cuda").to(OUT_DTYPE)
    w = (torch.randn((n, k), generator=g, device="cuda") * 0.02).to(OUT_DTYPE)
    bias = None
    if with_bias:
        bias = (torch.randn((n,), generator=g, device="cuda") * 0.01).to(OUT_DTYPE)
    a_q, a_s = quant_rows(a)
    w_q, w_s = quant_rows(w)
    # float64 matmul of int8 values is exact here, every partial sum far below
    # 2**53, so this is the quantized layer computed without rounding.
    term = (a_q.double() @ w_q.double().t()) * a_s.double() * w_s.double().t()
    ref = term + bias.double() if bias is not None else term
    return {"a": a, "w": w, "bias": bias, "a_q": a_q, "a_s": a_s,
            "w_q": w_q, "w_qt": w_q.t(), "w_s": w_s, "term": term, "ref": ref}


def poison(shape) -> None:
    """Fill and free a block of the output's size, so a kernel that writes nothing
    cannot pass on memory still holding another provider's answer."""
    torch.full(shape, float("nan"), dtype=OUT_DTYPE, device="cuda")


def check(fn: Callable[[], torch.Tensor], ref: torch.Tensor | None,
          term: torch.Tensor | None) -> str:
    if ref is not None:
        poison(ref.shape)
    try:
        out = fn()
        torch.cuda.synchronize()
    except NotSupported as e:
        return f"declined: {first_line(e)}"
    except Exception as e:  # a provider failing is a result, not a crash
        return f"error: {first_line(e)}"
    if ref is None:
        return "ok"
    if tuple(out.shape) != tuple(ref.shape):
        return f"FAIL: shape {tuple(out.shape)} against {tuple(ref.shape)}"
    scale = torch.maximum(ref.abs(), term.abs()).clamp_min(1e-6)
    rel = ((out.double() - ref).abs() / scale).max().item()
    return "ok" if rel <= RTOL else f"FAIL: max relative error {rel:.3g}"


def measure(fn, ref, term, iters: int, windows: int, eager: bool = True) -> dict:
    status = check(fn, ref, term)
    if status != "ok":
        return {"status": status}
    row: dict = {"status": "ok"}
    if eager:
        e = time_eager(fn, iters, windows)
        row.update(eager_us=statistics.median(e) * 1e3, eager_windows_us=[x * 1e3 for x in e])
    try:
        g = time_graph(fn, iters, windows)
        row.update(graph_us=statistics.median(g) * 1e3, graph_windows_us=[x * 1e3 for x in g])
    except Exception as ex:
        row["graph_status"] = f"capture failed: {first_line(ex)}"
    return row


# --------------------------------------------------------------------- providers
def candidate_configs(m: int, k: int, n: int) -> list[Config]:
    """The tuning sweep. Batch tiles grow with M; for each, every N and K tile that
    divides the shape, one or three stages, and 8 warps only for tiles big enough."""
    bms = [16] if m <= 16 else [32, 64] if m <= 128 else [64, 128]
    configs = []
    for bm in bms:
        for bn in (64, 128, 256):
            for bk in (64, 128, 256):
                if n % bn or k % bk:
                    continue
                for warps in ((4, 8) if bm * bn >= 128 * 128 else (4,)):
                    for stages in (1, 3):
                        if stages * (bm * bk + bk * bn) <= SMEM_LIMIT:
                            configs.append(Config(bm, bn, bk, warps, stages, 8 if m >= 256 else 1))
    return configs


def providers(d: dict, m: int, k: int, n: int, ops, vllm_triton) -> dict:
    """name -> (callable, reference, matmul term); reference None means unchecked."""
    a_q, w_qt, a_s, w_s, bias = d["a_q"], d["w_qt"], d["a_s"], d["w_s"], d["bias"]
    ref, term = d["ref"], d["term"]

    def int_mm_unfused():
        # torch._int_mm is the cuBLASLt product; the scales and bias follow as
        # separate kernels, in float32 so the only rounding is the last cast.
        out = torch._int_mm(a_q, w_qt).float() * (a_s * w_s.t())
        if bias is not None:
            out = out + bias.float()
        return out.to(OUT_DTYPE)

    ours = partial(w8a8_mm, a_q, w_qt, a_s, w_s, OUT_DTYPE, bias, config=default_config(m, n, k))
    p = {
        BF16: (partial(F.linear, d["a"], d["w"], bias), None, None),
        INT_MM: (int_mm_unfused, ref, term),
        OURS_DEFAULT: (ours, ref, term),
        OURS_TABLE: (partial(w8a8_mm, a_q, w_qt, a_s, w_s, OUT_DTYPE, bias), ref, term),
        PR45126: (partial(pr45126.triton_scaled_mm, a_q, w_qt, a_s, w_s, OUT_DTYPE, bias),
                  ref, term),
    }
    if ops is not None:
        cutlass = partial(ops.cutlass_scaled_mm, a_q, w_qt, a_s, w_s, OUT_DTYPE, bias)
        p[CUTLASS] = (cutlass, ref, term)
        p[VLLM_TRITON] = (partial(vllm_triton, a_q, w_qt, a_s, w_s, OUT_DTYPE, bias), ref, term)
    return p


def quant_providers(d: dict, ops, cfg: Config) -> dict:
    """The same layers with vLLM's per-token activation quantizer inside the timing,
    checked against a reference built from that quantizer's own output."""
    a, w_qt, w_s, bias = d["a"], d["w_qt"], d["w_s"], d["bias"]
    q, s, _ = ops.scaled_int8_quant(a)
    term = (q.double() @ w_qt.double()) * s.double() * w_s.double().t()
    ref = term + bias.double() if bias is not None else term

    def cutlass_q():
        q, s, _ = ops.scaled_int8_quant(a)
        return ops.cutlass_scaled_mm(q, w_qt, s, w_s, OUT_DTYPE, bias)

    def ours_q():
        q, s, _ = ops.scaled_int8_quant(a)
        return w8a8_mm(q, w_qt, s, w_s, OUT_DTYPE, bias, config=cfg)

    return {CUTLASS_Q: (cutlass_q, ref, term), OURS_Q: (ours_q, ref, term)}


# --------------------------------------------------------------------- reporting
def gpu_processes() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return ["nvidia-smi unavailable"]
    me = f"{os.getpid()},"
    return [ln.strip() for ln in out.splitlines() if ln.strip() and not ln.startswith(me)]


def environment(vllm_version: str, windows: int, bias: bool) -> dict:
    p = torch.cuda.get_device_properties(0)
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.split()
        driver = smi[0] if smi else "unknown"
    except (OSError, subprocess.SubprocessError):
        driver = "unknown"
    return {"device": p.name, "sm": f"sm_{p.major}{p.minor}", "sms": p.multi_processor_count,
            "driver": driver, "torch": torch.__version__, "triton": triton.__version__,
            "vllm": vllm_version, "python": platform.python_version(),
            "vllm_triton_use_td": os.environ.get("VLLM_TRITON_USE_TD", "unset"),
            "windows": windows, "bias": bias, "pr45126": pr45126.table_label()}


def fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f}" if v < 10 else f"{v:.1f}"


def speedup(slow: float | None, fast: float | None) -> str:
    return "n/a" if not slow or not fast else f"{slow / fast:.2f}x"


def markdown(res: dict, batch: list[int]) -> str:
    env = res["env"]
    rows = {(r["shape"], r["m"], r["provider"]): r for r in res["rows"]}

    def us(shape: str, m: int, prov: str) -> float | None:
        r = rows.get((shape, m, prov))
        return r.get("graph_us") if r and r["status"] == "ok" else None

    def table(header: list[str], body: list[list[str]]) -> list[str]:
        return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
                *("| " + " | ".join(r) + " |" for r in body), ""]

    out = [
        f"# W8A8 int8 linear layers on {env['device']} ({env['sm']})",
        "",
        f"torch {env['torch']}, Triton {env['triton']}, vLLM {env['vllm']}, driver "
        f"{env['driver']}, VLLM_TRITON_USE_TD={env['vllm_triton_use_td']}, #45126 on its "
        f"{env.get('pr45126', 'n/a')}.",
        "",
        f"Microseconds per layer, bf16 output, {'with' if env.get('bias') else 'no'} bias, "
        f"CUDA-graph replay, median of {env['windows']} windows. Every number passed a "
        "float64 check at its own shape first. A speedup is the second provider's time "
        "over the first's, so above 1.00x the first is faster. \"ours\" in the speedup "
        "columns is the tuned kernel when this run tuned, and the stored table otherwise.",
        "",
    ]
    for label, k, n in res["shapes"]:
        out += [f"## {label}: K={k}, N={n}", ""]
        body = []
        for m in batch:
            t = {p: us(label, m, p) for p in [BF16, *INT8]}
            int8_times = [v for p, v in t.items() if p in INT8 and v]
            mine = t[OURS_TUNED] or t[OURS_TABLE]
            body.append([str(m), *(fmt(t[p]) for p in [BF16, *INT8]),
                         speedup(t[CUTLASS], mine), speedup(t[VLLM_TRITON], mine),
                         speedup(t[PR45126], mine),
                         speedup(t[BF16], min(int8_times) if int8_times else None)])
        out += table(["M", "bf16", "CUTLASS", "vLLM Triton", "#45126", "_int_mm",
                      "ours default", "ours tuned", "ours table", "ours over CUTLASS",
                      "ours over vLLM Triton", "ours over #45126", "best int8 over bf16"], body)
        body = []
        for m in batch:
            t = {p: us(label, m, p) for p in (BF16, CUTLASS_Q, OURS_Q)}
            body.append([str(m), fmt(t[BF16]), fmt(t[CUTLASS_Q]), fmt(t[OURS_Q]),
                         speedup(t[BF16], t[CUTLASS_Q]), speedup(t[BF16], t[OURS_Q])])
        out += ["With per-token activation quantization inside the timing:", ""]
        out += table(["M", "bf16", "CUTLASS + quant", "ours + quant", "CUTLASS + quant over bf16",
                      "ours + quant over bf16"], body)
        tuned = res["tuned"].get(f"{k}x{n}", {})
        if tuned:
            out += ["Tuned configurations:", ""]
            out += table(["M", "config", "microseconds"],
                         [[m, Config(**v["config"]).name, fmt(v["graph_us"])]
                          for m, v in sorted(tuned.items(), key=lambda kv: int(kv[0]))])
    bad = [r for r in res["rows"] if r["status"] != "ok"]
    out += ["## Not timed", ""]
    out += [f"- {r['shape']}, M={r['m']}, {r['provider']}: {r['status']}" for r in bad] or ["None."]
    sweep_bad = [r for r in res["sweep"] if r["status"].startswith(("FAIL", "error"))]
    out += ["", f"Tuning sweep: {len(res['sweep'])} configurations, "
                f"{len(sweep_bad)} failed or errored."]
    seen = res["interference"]
    others = sorted(set(seen.get("before", []) + seen.get("after", [])))
    out += ["", "Other processes on the GPU during the run: "
            + ("none." if not others else "; ".join(others))]
    return "\n".join(out) + "\n"


def save(path: Path, res: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1, default=str))


def merge_tuned(tuned: dict, device: str) -> None:
    table = json.loads(TUNED_PATH.read_text()) if TUNED_PATH.exists() else {}
    for shape, buckets in tuned.items():
        table.setdefault(device, {}).setdefault(shape, {}).update(buckets)
    TUNED_PATH.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")


# -------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shapes", choices=sorted(SHAPES), default="qwen3-1.7b")
    ap.add_argument("--batch", default=",".join(map(str, BATCH)), help="comma-separated M values")
    ap.add_argument("--windows", type=int, default=5, help="timing windows per measurement")
    ap.add_argument("--tune", action="store_true", help="sweep this repo's kernel configurations")
    ap.add_argument("--bias", action="store_true", help="give every layer a bias")
    ap.add_argument("--tag", default="", help="suffix for the result file names")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")

    batch = [int(x) for x in args.batch.split(",")]
    ops, vllm_triton, vllm_version = load_vllm()
    device = torch.cuda.get_device_name()
    stem = f"{args.shapes}_{device.replace(' ', '_')}" + ("_bias" if args.bias else "")
    stem += f"_{args.tag}" if args.tag else ""
    json_path = args.out / f"{stem}.json"
    res: dict = {"env": environment(vllm_version, args.windows, args.bias),
                 "shapes": SHAPES[args.shapes], "interference": {"before": gpu_processes()},
                 "rows": [], "sweep": [], "tuned": {}}
    print(json.dumps(res["env"], indent=1))
    if res["interference"]["before"]:
        print("WARNING: other processes are using the GPU:", res["interference"]["before"])

    for label, k, n in SHAPES[args.shapes]:
        for m in batch:
            iters = ITERS.get(m) or max(5, min(200, 3200 // m))
            d = operands(m, k, n, seed=m * 7919 + k * 31 + n, with_bias=args.bias)
            best: tuple[Config, float] | None = None
            if args.tune:
                for cfg in candidate_configs(m, k, n):
                    fn = partial(w8a8_mm, d["a_q"], d["w_qt"], d["a_s"], d["w_s"], OUT_DTYPE,
                                 d["bias"], config=cfg)
                    r = measure(fn, d["ref"], d["term"], iters, args.windows, eager=False)
                    res["sweep"].append({"shape": label, "k": k, "n": n, "m": m,
                                         "config": asdict(cfg), **r})
                    if r.get("graph_us") and (best is None or r["graph_us"] < best[1]):
                        best = (cfg, r["graph_us"])
                if best:
                    res["tuned"].setdefault(f"{k}x{n}", {})[str(m)] = {
                        "config": asdict(best[0]), "graph_us": best[1]}
            provs = providers(d, m, k, n, ops, vllm_triton)
            if best:
                tuned = partial(w8a8_mm, d["a_q"], d["w_qt"], d["a_s"], d["w_s"], OUT_DTYPE,
                                d["bias"], config=best[0])
                provs[OURS_TUNED] = (tuned, d["ref"], d["term"])
            if ops is not None:
                quant_cfg = best[0] if best else tuned_config(m, n, k) or default_config(m, n, k)
                provs.update(quant_providers(d, ops, quant_cfg))
            for name, (fn, ref, term) in provs.items():
                r = measure(fn, ref, term, iters, args.windows)
                res["rows"].append({"shape": label, "k": k, "n": n, "m": m, "provider": name, **r})
                shown = r["status"] if r["status"] != "ok" else (
                    f"eager {fmt(r.get('eager_us'))}  graph {fmt(r.get('graph_us'))} us")
                print(f"{label:<20} M={m:<5} {name:<34} {shown}")
            del d
            torch.cuda.empty_cache()
        # Written after every shape, so a later failure cannot cost earlier timings.
        save(json_path, res)

    res["interference"]["after"] = gpu_processes()
    save(json_path, res)
    if res["tuned"] and not args.bias:
        merge_tuned(res["tuned"], device)
    md = markdown(res, batch)
    json_path.with_suffix(".md").write_text(md)
    print("\n" + md)
    print(f"wrote {json_path} and {json_path.with_suffix('.md')}")


if __name__ == "__main__":
    main()
