"""Where vLLM's int8 CUTLASS kernel slows down with batch size, set against its dispatch.

vLLM's sm_89 int8 CUTLASS dispatch (csrc/libtorch_stable/quantization/w8a8/cutlass/
scaled_mm_c2x_sm89_int8_dispatch.cuh at v0.28.0) holds one compiled GEMM
configuration per bucket of next_pow_2(M), floored at 16: [1, 16], (16, 32],
(32, 64], (64, 128], (128, 256] and above 256. Inside a bucket it picks by
next_pow_2(N). The kernel benchmark found CUTLASS's layer time doubling between M=16
and M=64 at every Qwen3-1.7B shape, while bf16 and the Triton kernels barely moved.

The prediction, written before this ran: if a bucket's configuration causes that,
CUTLASS's time jumps between two adjacent M at a bucket edge (16 to 17, or 32 to 33)
and stays flat inside the bucket, while bf16 and the Triton kernels cross the same
edges smoothly. For gate_up_proj, whose next_pow_2(N) is 16384, the (32, 64] bucket
compiles a different configuration, so that layer need not jump where the others do.

    python bench/cutlass_cliff.py
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_linear as bl  # noqa: E402  (also puts the repo root on sys.path)

from int8_linear import pr45126  # noqa: E402
from int8_linear.kernel import w8a8_mm  # noqa: E402

M_VALUES = [8, 16, 17, 24, 32, 33, 40, 48, 56, 64, 65, 80, 96, 128, 129, 192, 256, 257, 512, 513]
EDGES = [16, 32, 64, 128, 256, 512]
# Every distinct linear-layer shape of Qwen3-1.7B, separate and merged as vLLM runs them.
SHAPES = [
    ("q_proj, o_proj", 2048, 2048),
    ("k_proj, v_proj", 2048, 1024),
    ("qkv_proj", 2048, 4096),
    ("gate_proj, up_proj", 2048, 6144),
    ("gate_up_proj", 2048, 12288),
    ("down_proj", 6144, 2048),
]
NAMES = {bl.BF16: "bf16", bl.CUTLASS: "CUTLASS", bl.VLLM_TRITON: "vLLM Triton",
         bl.PR45126: "#45126", bl.OURS_TABLE: "this repo"}


def next_pow_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def bucket(m: int, n: int) -> str:
    """The M bucket and N key CUTLASS's sm_89 int8 dispatch picks a configuration by."""
    mp2 = max(16, next_pow_2(m))
    span = "[1, 16]" if mp2 == 16 else "(256, inf)" if mp2 > 256 else f"({mp2 // 2}, {mp2}]"
    return f"M in {span}, next_pow_2(N)={next_pow_2(n)}"


def markdown(res: dict) -> str:
    env, ms = res["env"], res["m_values"]
    t = {(r["shape"], r["m"], r["provider"]): r.get("graph_us")
         for r in res["rows"] if r["status"] == "ok"}
    out = [f"# CUTLASS int8 across its dispatch buckets on {env['device']}", "",
           f"torch {env['torch']}, Triton {env['triton']}, vLLM {env['vllm']}. Microseconds "
           f"per layer, no bias, CUDA-graph replay, median of {env['windows']} windows. Every "
           "int8 output passed the float64 check first. \"this repo\" reads "
           "int8_linear/tuned_configs.json.", "",
           "## Steps at the bucket edges", "",
           "Time at M = edge + 1 over time at M = edge. One more token adds at most 1/edge "
           "to the work, so a kernel whose configuration does not change sits near 1.00x.", "",
           "| layer | N | edge | " + " | ".join(NAMES.values()) + " |",
           "|---" * (3 + len(NAMES)) + "|"]
    for label, _, n in res["shapes"]:
        for e in EDGES:
            if e not in ms or e + 1 not in ms:
                continue
            cells = []
            for p in NAMES:
                a, b = t.get((label, e, p)), t.get((label, e + 1, p))
                cells.append(f"{b / a:.2f}x" if a and b else "n/a")
            out.append(f"| {label} | {n} | {e} to {e + 1} | " + " | ".join(cells) + " |")
    for label, k, n in res["shapes"]:
        out += ["", f"## {label}: K={k}, N={n}", "",
                "| M | CUTLASS picks by | " + " | ".join(NAMES.values())
                + " | CUTLASS / bf16 time | CUTLASS / this repo time |",
                "|---" * (4 + len(NAMES)) + "|"]
        for m in ms:
            v = {p: t.get((label, m, p)) for p in NAMES}
            out.append(f"| {m} | {bucket(m, n)} | " + " | ".join(bl.fmt(v[p]) for p in NAMES)
                       + f" | {bl.speedup(v[bl.CUTLASS], v[bl.BF16])}"
                       + f" | {bl.speedup(v[bl.CUTLASS], v[bl.OURS_TABLE])} |")
    bad = [r for r in res["rows"] if r["status"] != "ok"]
    out += ["", "## Not timed", ""]
    out += [f"- {r['shape']}, M={r['m']}, {r['provider']}: {r['status']}" for r in bad] or ["None."]
    seen = res["interference"]
    others = sorted(set(seen.get("before", []) + seen.get("after", [])))
    out += ["", "Other processes on the GPU during the run: "
            + ("none." if not others else "; ".join(others))]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--m", default=",".join(map(str, M_VALUES)), help="comma-separated M values")
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    ops, vllm_triton, vllm_version = bl.load_vllm()
    if ops is None:
        raise SystemExit(f"needs vLLM for its CUTLASS kernel: {vllm_version}")

    ms = [int(x) for x in args.m.split(",")]
    path = args.out / f"cliff_{torch.cuda.get_device_name().replace(' ', '_')}.json"
    res: dict = {"env": bl.environment(vllm_version, args.windows, False), "m_values": ms,
                 "shapes": SHAPES, "interference": {"before": bl.gpu_processes()}, "rows": []}
    for label, k, n in SHAPES:
        for m in ms:
            d = bl.operands(m, k, n, seed=m * 7919 + k * 31 + n, with_bias=False)
            q = (d["a_q"], d["w_qt"], d["a_s"], d["w_s"], bl.OUT_DTYPE, None)
            checked = (d["ref"], d["term"])
            provs = {
                bl.BF16: (partial(F.linear, d["a"], d["w"]), None, None),
                bl.CUTLASS: (partial(ops.cutlass_scaled_mm, *q), *checked),
                bl.VLLM_TRITON: (partial(vllm_triton, *q), *checked),
                bl.PR45126: (partial(pr45126.triton_scaled_mm, *q), *checked),
                bl.OURS_TABLE: (partial(w8a8_mm, *q), *checked),
            }
            iters = max(5, min(200, 3200 // m))
            for name, (fn, ref, term) in provs.items():
                r = bl.measure(fn, ref, term, iters, args.windows, eager=False)
                res["rows"].append({"shape": label, "k": k, "n": n, "m": m,
                                    "bucket": bucket(m, n), "provider": name, **r})
                shown = r["status"] if r["status"] != "ok" else f"{bl.fmt(r.get('graph_us'))} us"
                print(f"{label:<20} M={m:<4} {name:<34} {shown}")
            del d, q, checked, provs
            torch.cuda.empty_cache()
        bl.save(path, res)
    res["interference"]["after"] = bl.gpu_processes()
    bl.save(path, res)
    md = markdown(res)
    path.with_suffix(".md").write_text(md)
    print("\n" + md)
    print(f"wrote {path} and {path.with_suffix('.md')}")


if __name__ == "__main__":
    main()
