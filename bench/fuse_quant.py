"""Fusing the int8 activation quantizer into the kernel before it: numerics, speed, prediction.

Three parts, each written to results/ before the next runs:

1. Numerics (results/fuse_numerics.json). Each fused kernel against vLLM's unfused
   pair on the same inputs: RMSNorm sites against fused_add_rms_norm (or rms_norm)
   then scaled_int8_quant, and against vLLM's own fused CUDA kernel
   rms_norm_dynamic_per_token_quant; the SiLU site against silu_and_mul then
   scaled_int8_quant. Reported per numerics choice (IEEE or approximate division,
   SiLU rounded or not): the share of int8 values that differ, the largest
   difference, the largest relative scale difference, and whether the residual
   stream is bit-identical. The custom ops are then set to the best-matching choice.

2. Kernel time (results/fuse_kernels.json). Each site at decode and prefill batch
   sizes, unfused vLLM pair against the fused kernels, timed as CUDA-graph replays of
   100 back-to-back calls so launch cost on the GPU is counted but host cost is not.

3. Prediction (results/fuse_prediction.json), written before any end-to-end run:
   a 28-layer stand-in of Qwen3-1.7B's decode step as vLLM serves it (this repo's
   int8 matmul with the shipped table, norms, quantizers, SiLU-and-mul, residual
   adds; attention left out), with and without the fusion, timed in alternating
   fresh CUDA-graph captures. The per-layer saving, applied to the served decode
   step of the last end-to-end run, predicts the fused backend's tokens per second.

    python bench/fuse_quant.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from int8_linear import fused_quant as fq  # noqa: E402
from int8_linear.kernel import w8a8_mm  # noqa: E402

RESULTS = Path("results")
HIDDEN, INTER, EPS, LAYERS = 2048, 6144, 1e-6, 28
SHAPES = {"qkv_proj": (2048, 4096), "o_proj": (2048, 2048),
          "gate_up_proj": (2048, 12288), "down_proj": (6144, 2048)}
DECODE = [1, 4, 16, 32, 48, 64, 128]   # the batch sizes the served runs measured
PREFILL = [1024, 4096]
OPS = None


def activations(m: int, k: int, seed: int, dtype=torch.bfloat16) -> torch.Tensor:
    """Gaussian with a few large outlier channels, as LLM activations have."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, k), generator=g, device="cuda")
    x[:, torch.randperm(k, generator=g, device="cuda")[:8]] *= 30.0
    return x.to(dtype)


def compare(a, b) -> dict:
    """int8 tensors / scales / residuals from two paths."""
    (qa, sa, ra), (qb, sb, rb) = a, b
    d = (qa.int() - qb.int()).abs()
    rel = ((sa - sb).abs() / sb.abs().clamp_min(1e-30)).max().item()
    out = {"int8_diff_share": round(d.ne(0).float().mean().item(), 7),
           "int8_max_diff": int(d.max().item()), "scale_max_rel_diff": rel}
    if ra is not None and rb is not None:
        out["residual_identical"] = bool(torch.equal(ra, rb))
    return out


def vllm_norm_quant(x, r, w):
    x, r = x.clone(), None if r is None else r.clone()
    if r is None:
        y = torch.empty_like(x)
        OPS.rms_norm(y, x, w, EPS)
    else:
        OPS.fused_add_rms_norm(x, r, w, EPS)   # in place: x <- norm, r <- x + r
        y = x
    q, s, _ = OPS.scaled_int8_quant(y)
    return q, s, r


def vllm_fused_cuda(x, r, w):
    r = None if r is None else r.clone()
    q, s = OPS.rms_norm_dynamic_per_token_quant(x.clone(), w, EPS, torch.int8, residual=r)
    return q, s, r


def vllm_silu_quant(gu):
    y = torch.empty((gu.shape[0], gu.shape[1] // 2), dtype=gu.dtype, device=gu.device)
    torch.ops._C.silu_and_mul(y, gu)
    q, s, _ = OPS.scaled_int8_quant(y)
    return q, s, None


def numerics() -> dict:
    res = {"norm": [], "silu": []}
    for m in (1, 7, 64, 333, 2048):
        for has_res in (False, True):
            x = activations(m, HIDDEN, m)
            r = activations(m, HIDDEN, m + 1) if has_res else None
            w = (1.0 + 0.1 * activations(1, HIDDEN, 99)[0].float()).to(torch.bfloat16)
            ref = vllm_norm_quant(x, r, w)
            row = {"M": m, "residual": has_res, "vllm_fused_cuda": compare(vllm_fused_cuda(x, r, w), ref)}
            for precise in (True, False):
                ours = fq.add_rmsnorm_quant(x, r, w, EPS, precise)
                row[f"ours_precise_div={precise}"] = compare(ours, ref)
            res["norm"].append(row)
        gu = activations(m, 2 * INTER, m + 7)
        ref = vllm_silu_quant(gu)
        row = {"M": m}
        for precise in (True, False):
            for rnd in (True, False):
                q, s = fq.silu_mul_quant(gu, precise, rnd)
                row[f"ours_precise_div={precise}_round_silu={rnd}"] = compare((q, s, None), ref)
        res["silu"].append(row)

    def total(site, key):
        return sum(r[key]["int8_diff_share"] for r in res[site])
    res["best_precise_div"] = min((True, False), key=lambda p: total("norm", f"ours_precise_div={p}"))
    res["best_round_silu"] = min((True, False), key=lambda rnd: total(
        "silu", f"ours_precise_div={res['best_precise_div']}_round_silu={rnd}"))
    return res


def graph_us(fn, calls: int = 100, windows: int = 7) -> float:
    """Microseconds per call, as a CUDA graph of `calls` back-to-back calls."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    graph.replay()
    times = []
    for _ in range(windows):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3 / calls)
    graph.reset()
    return statistics.median(times)


def kernel_times() -> list[dict]:
    rows = []
    w = torch.ones(HIDDEN, device="cuda", dtype=torch.bfloat16)
    for m in DECODE + PREFILL:
        x, r = activations(m, HIDDEN, 1), activations(m, HIDDEN, 2)
        gu = activations(m, 2 * INTER, 3)
        y = torch.empty_like(x)
        ys = torch.empty((m, INTER), dtype=torch.bfloat16, device="cuda")

        def unfused_norm():
            OPS.fused_add_rms_norm(y.copy_(x), r, w, EPS)   # copy keeps the input fixed
            OPS.scaled_int8_quant(y)

        def copy_only():
            y.copy_(x)

        def unfused_silu():
            torch.ops._C.silu_and_mul(ys, gu)
            OPS.scaled_int8_quant(ys)

        t_copy = graph_us(copy_only)
        row = {"M": m,
               "norm_unfused_us": round(graph_us(unfused_norm) - t_copy, 2),
               "norm_vllm_fused_cuda_us": round(graph_us(
                   lambda: OPS.rms_norm_dynamic_per_token_quant(x, w, EPS, torch.int8, residual=r)), 2),
               "norm_ours_us": round(graph_us(lambda: fq.add_rmsnorm_quant(x, r, w, EPS, fq.PRECISE_DIV)), 2),
               "silu_unfused_us": round(graph_us(unfused_silu), 2),
               "silu_ours_us": round(graph_us(lambda: fq.silu_mul_quant(gu, fq.PRECISE_DIV, fq.ROUND_SILU)), 2)}
        print(row, flush=True)
        rows.append(row)
    return rows


class Stack:
    """28 layers of Qwen3-1.7B's decode step as vLLM serves it, attention left out."""

    def __init__(self, m: int, seed: int = 0):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.x = torch.randn((m, HIDDEN), generator=g, device="cuda").to(torch.bfloat16)
        self.res = torch.empty_like(self.x)
        self.h = torch.empty_like(self.x)
        self.w = torch.ones(HIDDEN, device="cuda", dtype=torch.bfloat16)
        self.layers = []
        for _ in range(LAYERS):
            layer = {}
            for name, (k, n) in SHAPES.items():
                wq = torch.randint(-127, 128, (n, k), generator=g, device="cuda", dtype=torch.int8)
                layer[name] = (wq.t(), torch.rand((n, 1), generator=g, device="cuda") * 1e-3 + 1e-4)
            self.layers.append(layer)

    def mm(self, layer, name, q, s):
        w, ws = layer[name]
        return w8a8_mm(q, w, s, ws, torch.bfloat16)

    def unfused(self):
        # The residual stream is updated in place, so it restarts from x on every replay.
        res = self.res.copy_(self.x)
        h = self.h
        OPS.rms_norm(h, self.x, self.w, EPS)
        for i, layer in enumerate(self.layers):
            if i:
                OPS.fused_add_rms_norm(h, res, self.w, EPS)
            q, s, _ = OPS.scaled_int8_quant(h)
            a = self.mm(layer, "qkv_proj", q, s)[:, :HIDDEN].contiguous()
            q, s, _ = OPS.scaled_int8_quant(a)
            h = self.mm(layer, "o_proj", q, s)
            OPS.fused_add_rms_norm(h, res, self.w, EPS)
            q, s, _ = OPS.scaled_int8_quant(h)
            gu = self.mm(layer, "gate_up_proj", q, s)
            act = torch.empty((gu.shape[0], INTER), dtype=gu.dtype, device="cuda")
            torch.ops._C.silu_and_mul(act, gu)
            q, s, _ = OPS.scaled_int8_quant(act)
            h = self.mm(layer, "down_proj", q, s)
        return h

    def fused(self):
        res = self.x
        q, s, _ = fq.add_rmsnorm_quant(self.x, None, self.w, EPS, fq.PRECISE_DIV)
        h = None
        for i, layer in enumerate(self.layers):
            if i:
                q, s, res = fq.add_rmsnorm_quant(h, res, self.w, EPS, fq.PRECISE_DIV)
            a = self.mm(layer, "qkv_proj", q, s)[:, :HIDDEN].contiguous()
            q, s, _ = OPS.scaled_int8_quant(a)
            h = self.mm(layer, "o_proj", q, s)
            q, s, res = fq.add_rmsnorm_quant(h, res, self.w, EPS, fq.PRECISE_DIV)
            gu = self.mm(layer, "gate_up_proj", q, s)
            q, s = fq.silu_mul_quant(gu, fq.PRECISE_DIV, fq.ROUND_SILU)
            h = self.mm(layer, "down_proj", q, s)
        return h


def stack_us(fn, windows: int) -> float:
    return graph_us(fn, calls=1, windows=windows) / LAYERS


def prediction(rounds: int, windows: int) -> dict:
    served = json.loads((RESULTS / "e2e_ours.json").read_text())["decode"]
    rows = []
    for m in DECODE:
        stack = Stack(m)
        t = {"unfused": [], "fused": []}
        for _ in range(rounds):
            t["unfused"].append(stack_us(stack.unfused, windows))
            t["fused"].append(stack_us(stack.fused, windows))
        un, fu = statistics.median(t["unfused"]), statistics.median(t["fused"])
        saving = un - fu
        row = {"M": m, "standin_unfused_us_per_layer": round(un, 2),
               "standin_fused_us_per_layer": round(fu, 2), "saving_us_per_layer": round(saving, 2)}
        if str(m) in served:
            tok_s = served[str(m)]["tok_s"]
            step_us = m / tok_s * 1e6
            row.update(served_ours_tok_s=tok_s,
                       predicted_ours_fused_tok_s=round(m / ((step_us - LAYERS * saving) / 1e6), 1),
                       predicted_change=round(step_us / (step_us - LAYERS * saving) - 1, 4))
        print(row, flush=True)
        rows.append(row)
        del stack
        torch.cuda.empty_cache()
    return {"note": "written before the fused end-to-end runs; compare with e2e_ours_fused.json",
            "rows": rows}


def main() -> None:
    global OPS
    from vllm import _custom_ops as ops
    OPS = ops
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--windows", type=int, default=7)
    ap.add_argument("--parts", default="numerics,kernels,prediction")
    a = ap.parse_args()
    parts = a.parts.split(",")
    dev = torch.cuda.get_device_name()
    if "numerics" in parts:
        res = numerics()
        res["device"] = dev
        (RESULTS / "fuse_numerics.json").write_text(json.dumps(res, indent=1))
        print("best precise_div", res["best_precise_div"], "best round_silu", res["best_round_silu"])
    num = json.loads((RESULTS / "fuse_numerics.json").read_text())
    fq.PRECISE_DIV, fq.ROUND_SILU = num["best_precise_div"], num["best_round_silu"]
    if "kernels" in parts:
        (RESULTS / "fuse_kernels.json").write_text(json.dumps(
            {"device": dev, "precise_div": fq.PRECISE_DIV, "round_silu": fq.ROUND_SILU,
             "rows": kernel_times()}, indent=1))
    if "prediction" in parts:
        pred = prediction(a.rounds, a.windows)
        pred.update(device=dev, precise_div=fq.PRECISE_DIV, round_silu=fq.ROUND_SILU)
        (RESULTS / "fuse_prediction.json").write_text(json.dumps(pred, indent=1))


if __name__ == "__main__":
    main()
