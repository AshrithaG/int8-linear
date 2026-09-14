"""Host cost of launching one int8 linear layer, and what skipping Triton's JIT dispatch saves.

In the kernel benchmark this repo's kernel was the slowest int8 option without CUDA
graphs at M=1 on q_proj: 37.0 microseconds per back-to-back launch, against 25.0 for
CUTLASS and 10.6 for bf16, though its GPU work under graph replay was the smallest.
This times small-M layers through each launch path:

    bf16 F.linear          cuBLAS through PyTorch's dispatcher
    CUTLASS                vLLM's cutlass_scaled_mm
    vLLM Triton            vLLM's triton_scaled_mm, through Triton's JIT dispatch
    ours, JIT dispatch     w8a8_mm(cached_launch=False), as in the runs so far
    ours, cached launch    w8a8_mm as it now runs by default
    ours, custom op        the cached launch behind torch.ops.int8_linear.w8a8_mm,
                           which is what a torch.compile graph calls
    ours, vLLM entry       vllm_patch.scaled_mm, which the patched model calls: the
                           custom op under torch.compile, the kernel directly otherwise

Three times for each: CPU time per call with nothing synchronized inside the window
(the host cost), back-to-back time between CUDA events (what a step without CUDA
graphs pays), and CUDA-graph replay (the GPU work alone).

    python bench/launch_overhead.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_linear as bl  # noqa: E402  (also puts the repo root on sys.path)

from int8_linear import vllm_patch  # noqa: E402  (also registers torch.ops.int8_linear.w8a8_mm)
from int8_linear.kernel import CACHED_LAUNCH, choose_config, w8a8_mm  # noqa: E402

SHAPES = [("q_proj, o_proj", 2048, 2048), ("gate_up_proj", 2048, 12288)]
M_VALUES = [1, 4, 16]
JIT = "ours, JIT dispatch"
CACHED = "ours, cached launch"
OP = "ours, custom op"
ENTRY = "ours, vLLM entry"
ORDER = [bl.BF16, bl.CUTLASS, bl.VLLM_TRITON, JIT, CACHED, OP, ENTRY]


def time_host(fn, iters: int, windows: int) -> list[float]:
    """CPU microseconds per call. The GPU catches up between windows, not inside them."""
    fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(windows):
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        out.append((time.perf_counter() - start) / iters * 1e6)
        torch.cuda.synchronize()
    return out


def markdown(res: dict) -> str:
    env = res["env"]
    out = [f"# Launch cost of one int8 linear layer on {env['device']}", "",
           f"torch {env['torch']}, Triton {env['triton']}, vLLM {env['vllm']}, launch cache "
           f"{'on' if env['cached_launch'] else 'off (untested Triton version)'}. Microseconds "
           f"per call, median of {env['windows']} windows, no bias. Host is CPU time with "
           "nothing synchronized inside the window; back-to-back is time between CUDA events; "
           "graph is CUDA-graph replay.", ""]
    for label, k, n in res["shapes"]:
        out += [f"## {label}: K={k}, N={n}", "",
                "| M | provider | host | back-to-back | graph | back-to-back minus graph |",
                "|---|---|---|---|---|---|"]
        for r in res["rows"]:
            if r["shape"] != label:
                continue
            if r["status"] != "ok":
                out.append(f"| {r['m']} | {r['provider']} | {r['status']} | | | |")
                continue
            gap = (r["eager_us"] - r["graph_us"]) if r.get("graph_us") else None
            out.append(f"| {r['m']} | {r['provider']} | {bl.fmt(r['host_us'])} | "
                       f"{bl.fmt(r['eager_us'])} | {bl.fmt(r.get('graph_us'))} | {bl.fmt(gap)} |")
        out.append("")
    seen = res["interference"]
    others = sorted(set(seen.get("before", []) + seen.get("after", [])))
    out += ["Other processes on the GPU during the run: "
            + ("none." if not others else "; ".join(others))]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iters", type=int, default=1000, help="calls per timing window")
    ap.add_argument("--windows", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    ops, vllm_triton, vllm_version = bl.load_vllm()
    path = args.out / f"launch_{torch.cuda.get_device_name().replace(' ', '_')}.json"
    res: dict = {"env": {**bl.environment(vllm_version, args.windows, False),
                         "cached_launch": CACHED_LAUNCH, "iters": args.iters},
                 "shapes": SHAPES, "interference": {"before": bl.gpu_processes()}, "rows": []}
    for label, k, n in SHAPES:
        for m in M_VALUES:
            d = bl.operands(m, k, n, seed=m * 7919 + k * 31 + n, with_bias=False)
            q = (d["a_q"], d["w_qt"], d["a_s"], d["w_s"], bl.OUT_DTYPE, None)
            checked = (d["ref"], d["term"])
            cfg = choose_config(m, n, k)
            provs = {
                bl.BF16: (partial(F.linear, d["a"], d["w"]), None, None),
                JIT: (partial(w8a8_mm, *q, config=cfg, cached_launch=False), *checked),
                CACHED: (partial(w8a8_mm, *q, config=cfg), *checked),
                OP: (partial(torch.ops.int8_linear.w8a8_mm, *q), *checked),
                ENTRY: (partial(vllm_patch.scaled_mm, *q), *checked),
            }
            if ops is not None:
                provs[bl.CUTLASS] = (partial(ops.cutlass_scaled_mm, *q), *checked)
                provs[bl.VLLM_TRITON] = (partial(vllm_triton, *q), *checked)
            for name in [p for p in ORDER if p in provs]:
                fn, ref, term = provs[name]
                row: dict = {"shape": label, "k": k, "n": n, "m": m, "provider": name,
                             "status": bl.check(fn, ref, term)}
                if row["status"] == "ok":
                    h = time_host(fn, args.iters, args.windows)
                    e = bl.time_eager(fn, args.iters, args.windows)
                    row.update(host_us=statistics.median(h), host_windows_us=h,
                               eager_us=statistics.median(e) * 1e3,
                               eager_windows_us=[x * 1e3 for x in e])
                    try:
                        g = bl.time_graph(fn, min(args.iters, 200), args.windows)
                        row.update(graph_us=statistics.median(g) * 1e3,
                                   graph_windows_us=[x * 1e3 for x in g])
                    except Exception as ex:
                        row["graph_status"] = f"capture failed: {bl.first_line(ex)}"
                    shown = (f"host {bl.fmt(row['host_us'])}  back-to-back "
                             f"{bl.fmt(row['eager_us'])}  graph {bl.fmt(row.get('graph_us'))} us")
                else:
                    shown = row["status"]
                res["rows"].append(row)
                print(f"{label:<16} M={m:<3} {name:<24} {shown}")
            del d, q, checked, provs
            torch.cuda.empty_cache()
    res["interference"]["after"] = bl.gpu_processes()
    bl.save(path, res)
    md = markdown(res)
    path.with_suffix(".md").write_text(md)
    print("\n" + md)
    print(f"wrote {path} and {path.with_suffix('.md')}")


if __name__ == "__main__":
    main()
