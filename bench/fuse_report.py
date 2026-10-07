"""Summarize the fusion study into results/fuse.md: numerics, kernel times, the
stand-in's prediction against what serving measured, and quality."""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2e_report import logprob_gap, row  # noqa: E402

RESULTS = Path("results")
E2E = RESULTS / "fuse_e2e"
PAIRS = [("cutlass", "cutlass_fused"), ("ours", "ours_fused")]


def runs(backend: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(E2E.glob(f"round*/e2e_{backend}.json"))]


def median_tok_s(rs: list[dict], batch: str) -> float | None:
    v = [r["decode"][batch]["tok_s"] for r in rs if batch in r.get("decode", {})]
    return statistics.median(v) if v else None


def main() -> None:
    out = ["# Fusing the int8 activation quantizer into RMSNorm and SiLU-and-mul", ""]
    num = json.loads((RESULTS / "fuse_numerics.json").read_text())
    out += [f"Numerics chosen: IEEE division = {num['best_precise_div']}, "
            f"SiLU rounded before the multiply = {num['best_round_silu']}.", "",
            "| site | M | residual | path | int8 values that differ | largest difference | residual identical |",
            "|---|---|---|---|---|---|---|"]
    key = f"ours_precise_div={num['best_precise_div']}"
    for r in num["norm"]:
        for path in ("vllm_fused_cuda", key):
            c = r[path]
            out.append(row(["RMSNorm", str(r["M"]), str(r["residual"]), path,
                            f"{c['int8_diff_share']:.2e}", str(c["int8_max_diff"]),
                            str(c.get("residual_identical", "n/a"))]))
    skey = f"{key}_round_silu={num['best_round_silu']}"
    for r in num["silu"]:
        c = r[skey]
        out.append(row(["SiLU-and-mul", str(r["M"]), "-", skey, f"{c['int8_diff_share']:.2e}",
                        str(c["int8_max_diff"]), "n/a"]))

    k = json.loads((RESULTS / "fuse_kernels.json").read_text())
    out += ["", "Kernel time, microseconds per call (CUDA-graph replay of 100 calls):", "",
            "| M | RMSNorm + quant, vLLM unfused | vLLM fused CUDA | this repo, Triton | "
            "SiLU-and-mul + quant, vLLM unfused | this repo, Triton |", "|---|---|---|---|---|---|"]
    for r in k["rows"]:
        out.append(row([str(r["M"]), str(r["norm_unfused_us"]), str(r["norm_vllm_fused_cuda_us"]),
                        str(r["norm_ours_us"]), str(r["silu_unfused_us"]), str(r["silu_ours_us"])]))

    pred = json.loads((RESULTS / "fuse_prediction.json").read_text())
    out += ["", "Decode, the stand-in's prediction (written first) against serving, tokens/s:", "",
            "| batch | stand-in saving, us/layer | predicted change | cutlass -> cutlass_fused | "
            "ours -> ours_fused |", "|---|---|---|---|---|"]
    for r in pred["rows"]:
        b = str(r["M"])
        cells = [b, str(r["saving_us_per_layer"]),
                 f"{100 * r['predicted_change']:+.1f}%" if "predicted_change" in r else "n/a"]
        for base, fused in PAIRS:
            t0, t1 = median_tok_s(runs(base), b), median_tok_s(runs(fused), b)
            cells.append("n/a" if not t0 or not t1 else f"{t0:,.0f} -> {t1:,.0f} ({100 * (t1 / t0 - 1):+.1f}%)")
        out.append(row(cells))

    out += ["", "Prefill (8 x 512 tokens) and quality, fused against unfused, same GEMM:", "",
            "| pair | prefill tok/s change | perplexity, unfused / fused | logprob gap mean / max | "
            "greedy samples identical | matmuls that took a PreQuant |", "|---|---|---|---|---|---|"]
    for base, fused in PAIRS:
        a, f = runs(base), runs(fused)
        if not a or not f:
            continue
        p0 = statistics.median(r["prefill"]["tok_s"] for r in a)
        p1 = statistics.median(r["prefill"]["tok_s"] for r in f)
        gap = logprob_gap(f[0], a[0])
        ppl = lambda r: f"{r['perplexity']['ppl']:.4f}" if r.get("perplexity", {}).get("status") == "ok" else "n/a"
        same = "yes" if a[0].get("samples") == f[0].get("samples") else "no"
        out.append(row([f"{base} -> {fused}", f"{100 * (p1 / p0 - 1):+.1f}%", f"{ppl(a[0])} / {ppl(f[0])}",
                        "n/a" if gap is None else f"{gap[0]:.2e} / {gap[1]:.2e}", same,
                        str(f[0].get("prequant_calls", "n/a"))]))
    # Quality as paired differences: mean log-probability per WikiText window, fused minus
    # reference, with the standard error over windows. For scale, the same statistic for two
    # runs of unchanged CUTLASS a month apart (driver update in between) and int8 vs bf16.
    def paired(a, b):
        wa = [statistics.mean(w) for w in a["perplexity"]["token_logprobs"]]
        wb = [statistics.mean(w) for w in b["perplexity"]["token_logprobs"]]
        d = [x - y for x, y in zip(wa, wb)]
        m, se = statistics.mean(d), statistics.stdev(d) / math.sqrt(len(d))
        return f"{m:+.4f} (SE {se:.4f}, z {m / se:+.2f})"

    def pub(b):
        p = RESULTS / f"e2e_{b}.json"
        return json.loads(p.read_text()) if p.exists() else None

    rows = [("cutlass_fused - cutlass", runs("cutlass_fused"), runs("cutlass")),
            ("ours_fused - ours", runs("ours_fused"), runs("ours"))]
    bf16 = pub("bf16")
    out += ["", "Paired quality, mean log-probability per token over 40 WikiText-2 windows "
            "(positive: the first assigns the text higher probability):", "",
            "| comparison | difference |", "|---|---|"]
    for label, a, b in rows:
        if a and b:
            out.append(row([label, paired(a[0], b[0])]))
    if bf16:
        for label, a in (("cutlass_fused - bf16", runs("cutlass_fused")), ("ours_fused - bf16", runs("ours_fused")),
                         ("cutlass - bf16", runs("cutlass")), ("ours - bf16", runs("ours"))):
            if a:
                out.append(row([label, paired(a[0], bf16)]))
    if pub("cutlass") and runs("cutlass"):
        out.append(row(["cutlass today - cutlass in the earlier published run (no code change)",
                        paired(runs("cutlass")[0], pub("cutlass"))]))
    (RESULTS / "fuse.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
