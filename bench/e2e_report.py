"""Combine bench/e2e_vllm.py results into results/e2e.md."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

RESULTS = Path("results")
ORDER = ["bf16", "cutlass", "vllm_triton", "pr45126", "ours"]
NAMES = {"bf16": "bf16", "cutlass": "CUTLASS", "vllm_triton": "vLLM Triton",
         "pr45126": "#45126 Triton", "ours": "this repo"}


def load(suffix: str) -> dict:
    return {b: json.loads((RESULTS / f"e2e_{b}{suffix}.json").read_text())
            for b in ORDER if (RESULTS / f"e2e_{b}{suffix}.json").exists()}


def num(v: float | None, digits: int = 1) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def rel(value: float | None, base: float | None) -> str:
    return "n/a" if not value or not base else f"{value / base:.2f}x"


def row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def throughput(runs: dict) -> list[str]:
    """Decode by batch size and prefill, per backend, with each relative to CUTLASS."""
    cols = list(runs)
    others = [b for b in cols if b != "cutlass"] if "cutlass" in runs else []
    head = ["batch", *(NAMES[b] for b in cols), *(f"{NAMES[b]} vs CUTLASS" for b in others)]
    out = ["Decode tokens per second:", "", row(head), "|---" * len(head) + "|"]
    for bs in sorted({int(k) for r in runs.values() for k in r.get("decode", {})}):
        v = {b: runs[b].get("decode", {}).get(str(bs), {}).get("tok_s") for b in cols}
        out.append(row([str(bs), *(num(v[b]) for b in cols),
                        *(rel(v[b], v.get("cutlass")) for b in others)]))
    pre = {b: runs[b].get("prefill", {}).get("tok_s") for b in cols}
    first = next(iter(runs.values())).get("prefill", {})
    out += ["", f"Prefill, {first.get('prompts', '?')} prompts of {first.get('prompt_len', '?')} "
            "tokens, prompt tokens per second:", "",
            row(["", *(NAMES[b] for b in cols)]), "|---" * (1 + len(cols)) + "|",
            row(["tokens/s", *(num(pre[b], 0) for b in cols)]),
            row(["vs CUTLASS", *(rel(pre[b], pre.get("cutlass")) for b in cols)])]
    return out


def logprob_gap(a: dict, b: dict) -> tuple[float, float] | None:
    pa, pb = a.get("perplexity", {}), b.get("perplexity", {})
    if pa.get("status") != "ok" or pb.get("status") != "ok":
        return None
    diffs = [abs(x - y) for ra, rb in zip(pa["token_logprobs"], pb["token_logprobs"])
             for x, y in zip(ra, rb)]
    return (statistics.mean(diffs), max(diffs)) if diffs else None


def quality(runs: dict) -> list[str]:
    out = ["| backend | perplexity | tokens | logprob gap to CUTLASS, mean / max | "
           "mean logprob gap to bf16 | greedy samples same as CUTLASS |",
           "|---|---|---|---|---|---|"]
    for b, r in runs.items():
        p = r.get("perplexity", {})
        ppl = f"{p['ppl']:.4f}" if p.get("status") == "ok" else p.get("status", "not run")
        to_cut = logprob_gap(r, runs["cutlass"]) if "cutlass" in runs else None
        to_bf = logprob_gap(r, runs["bf16"]) if "bf16" in runs else None
        same = ("n/a" if "samples" not in r or "samples" not in runs.get("cutlass", {})
                else "yes" if r["samples"] == runs["cutlass"]["samples"] else "no")
        out.append(row([NAMES[b], ppl, str(p.get("tokens", "n/a")),
                        "n/a" if not to_cut else f"{to_cut[0]:.4f} / {to_cut[1]:.4f}",
                        "n/a" if not to_bf else f"{to_bf[0]:.4f}", same]))
    triton = [b for b in ("vllm_triton", "pr45126", "ours")
              if runs.get(b, {}).get("perplexity", {}).get("status") == "ok"]
    if len(triton) > 1:
        first = runs[triton[0]]["perplexity"]["token_logprobs"]
        same = all(runs[b]["perplexity"]["token_logprobs"] == first for b in triton[1:])
        out += ["", f"{', '.join(NAMES[b] for b in triton)}: per-token logprobs "
                f"{'identical at every token' if same else 'not identical'}."]
    source = next((r["perplexity"]["source"] for r in runs.values()
                   if r.get("perplexity", {}).get("status") == "ok"), None)
    if source:
        out += ["", f"Text: {source}."]
    return out


def kernels(runs: dict, label: str) -> list[str]:
    out = []
    for b, r in runs.items():
        calls = r.get("kernel_calls")
        note = "" if calls is None else f"; patched kernel calls {calls}" + (
            f", fallbacks {r['fallbacks']}" if "fallbacks" in r else "") + (
            " (PATCH NOT REACHED)" if calls == 0 else "")
        out.append(f"- {label} {NAMES[b]}: {r.get('linear_kernels', 'n/a')}{note}")
    return out


def repeat(now: dict, before: dict, label: str) -> list[str]:
    """Throughput in this run over the first run's, per backend present in both."""
    common = [b for b in now if b in before]
    if not common:
        return []
    head = ["", *(NAMES[b] for b in common)]
    out = [f"{label}:", "", row(head), "|---" * len(head) + "|"]
    batches = sorted({int(k) for b in common for k in now[b].get("decode", {})
                      if k in before[b].get("decode", {})})
    for bs in batches:
        out.append(row([f"decode batch {bs}",
                        *(rel(now[b]["decode"].get(str(bs), {}).get("tok_s"),
                              before[b]["decode"].get(str(bs), {}).get("tok_s")) for b in common)]))
    out.append(row(["prefill", *(rel(now[b].get("prefill", {}).get("tok_s"),
                                     before[b].get("prefill", {}).get("tok_s")) for b in common)]))
    return out


def prior_runs() -> list[str]:
    """Suffixes of earlier runs kept as results/e2e_*_runN.json, oldest first."""
    numbers = {p.stem.rsplit("_run", 1)[1] for p in RESULTS.glob("e2e_*_run*.json")}
    return [f"_run{n}" for n in sorted((n for n in numbers if n.isdigit()), key=int)]


def main() -> None:
    graphs, eager = load(""), load("_eager")
    if not graphs and not eager:
        raise SystemExit("no results/e2e_*.json found")
    env = next(iter((graphs or eager).values()))["env"]
    out = [f"# Qwen3-1.7B end to end in vLLM on {env['device']}", "",
           f"vLLM {env['vllm']}, torch {env['torch']}, Triton {env['triton']}, max_model_len "
           f"{env['max_model_len']}, prefix caching off, each backend in its own process. "
           f"Decode: a {env['prompt_len']}-token prompt, {env['decode_tokens']} tokens generated, "
           f"median of {env['repeats']} repeats, with the first-token step timed separately "
           "and taken off. Relative columns are throughput over CUTLASS, so above 1.00x is "
           "faster.", ""]
    if graphs:
        out += ["## With torch.compile and CUDA graphs (vLLM's default)", "",
                *throughput(graphs), ""]
    if eager:
        out += ["## Without torch.compile or CUDA graphs", "", *throughput(eager), ""]
    for tag in prior_runs():
        before, before_eager = load(tag), load("_eager" + tag)
        n = tag.removeprefix("_run")
        out += [f"## Against run {n}", "",
                f"This run's throughput over run {n}'s (results/e2e_*{tag}.json). Backends "
                "whose code and configurations did not change show run-to-run variation.", ""]
        if before:
            out += [*repeat(graphs, before, "With torch.compile and CUDA graphs"), ""]
        if before_eager:
            out += [*repeat(eager, before_eager, "Without torch.compile or CUDA graphs"), ""]
    if graphs:
        out += ["## Quality", "", "Perplexity on WikiText-2 test windows, and how far each "
                "backend's per-token logprobs sit from CUTLASS's and from bf16's.", "",
                *quality(graphs), ""]
    out += ["## Kernels in use", "", *kernels(graphs, "compiled"), *kernels(eager, "eager"), ""]
    others = sorted({p for r in [*graphs.values(), *eager.values()]
                     for k in ("before", "after") for p in r.get("interference", {}).get(k, [])})
    out += ["Other processes on the GPU during the runs: " + ("none." if not others
                                                               else "; ".join(others))]
    (RESULTS / "e2e.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
