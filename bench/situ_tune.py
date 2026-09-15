"""Tuning in a stand-in for the served model, checked first against what the served model did.

In the second end-to-end run, per-layer timings mispredicted the served model: the
configurations that were faster timed alone made decode slower at batch 32 and 128, and
this kernel's lead over #45126 at batch 64, 10 microseconds per decoder layer when each
layer was timed alone, did not show up in the decode step. The per-layer benchmark replays
one kernel on one weight, back to back. The served model runs 28 decoder layers, each with
its own weights, with norms, activation quantization and an activation function between
the int8 matmuls.

This replays that structure as one CUDA graph: 28 decoder layers of Qwen3-1.7B's int8
linear layers as vLLM runs them, each with its own random int8 weights, with vLLM's
per-token activation quantizer, RMSNorm, residual adds and SiLU-and-mul in between.
Attention, the rotary embedding and the q and k norms are left out.

Before it tunes, it checks itself against the two served runs. Written before it ran, all
three of these must hold:
  1. batch 128: the second run's table is slower than the first run's, as served;
  2. batch 48: the second run's table is faster than the first run's, as served;
  3. batch 64: this kernel and #45126's are less than half as far apart as the per-layer
     benchmark put them.
Batch 32 and CUTLASS at batch 64 are reported but not required. If the check passes, it
tunes each decode batch size up to 256: one pass over the four layers, trying each layer's
current configuration, the first run's, and its best configurations from the per-layer
sweep, and keeping whichever makes the stand-in's step fastest. The chosen set is then
timed against the starting set in alternating fresh captures, and written to the table
only if it is at least 1% faster.

    python bench/situ_tune.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_linear as bl  # noqa: E402  (also puts the repo root on sys.path)

from int8_linear import pr45126  # noqa: E402
from int8_linear.kernel import TUNED_PATH, Config, w8a8_mm  # noqa: E402

RESULTS = Path("results")
LAYERS, HIDDEN, INTER, Q = 28, 2048, 6144, 2048
# Qwen3-1.7B's int8 linear layers as vLLM runs them, as (K, N).
SHAPES = {"qkv_proj": (2048, 4096), "o_proj": (2048, 2048),
          "gate_up_proj": (2048, 12288), "down_proj": (6144, 2048)}
CLIFF_LABELS = {"qkv_proj": "qkv_proj", "o_proj": "q_proj, o_proj",
                "gate_up_proj": "gate_up_proj", "down_proj": "down_proj"}
DECODE_SIZES = [1, 2, 4, *range(8, 257, 8)]
ACCEPT = 0.01
OPS = None


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def table_config(table: dict, m: int, k: int, n: int) -> Config:
    """The configuration a stored table gives this shape at batch m, by the kernel's rule:
    the largest tuned batch at or below m."""
    entries = table[f"{k}x{n}"]
    return Config(**entries[str(max(int(b) for b in entries if int(b) <= m))]["config"])


def configs(table: dict, m: int) -> dict[str, Config]:
    return {name: table_config(table, m, k, n) for name, (k, n) in SHAPES.items()}


class Stack:
    """28 decoder layers' worth of int8 linear layers with distinct weights, and the norms,
    activation quantization, residual adds and SiLU-and-mul between them."""

    def __init__(self, m: int, seed: int = 0):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.x = torch.randn((m, HIDDEN), generator=g, device="cuda").to(torch.bfloat16)
        self.norm = torch.ones(HIDDEN, device="cuda", dtype=torch.bfloat16)
        self.layers = []
        for _ in range(LAYERS):
            layer = {}
            for name, (k, n) in SHAPES.items():
                w = torch.randint(-127, 128, (n, k), generator=g, device="cuda", dtype=torch.int8)
                s = torch.rand((n, 1), generator=g, device="cuda") * 1e-3 + 1e-4
                layer[name] = (w.t(), s)
            self.layers.append(layer)

    def forward(self, mm) -> torch.Tensor:
        h = residual = self.x
        for layer in self.layers:
            h = F.rms_norm(h, (HIDDEN,), self.norm)
            # Attention is left out: the q part of qkv goes straight to o_proj.
            h = mm(layer, "o_proj", mm(layer, "qkv_proj", h)[:, :Q].contiguous())
            h = residual = h + residual
            h = F.rms_norm(h, (HIDDEN,), self.norm)
            gate, up = mm(layer, "gate_up_proj", h).split(INTER, dim=1)
            h = residual = mm(layer, "down_proj", F.silu(gate) * up) + residual
        return h


def ours(cfgs: dict[str, Config]):
    def mm(layer, name, x):
        w, ws = layer[name]
        q, s, _ = OPS.scaled_int8_quant(x)
        return w8a8_mm(q, w, s, ws, torch.bfloat16, config=cfgs[name])
    return mm


def pr45126_mm(layer, name, x):
    w, ws = layer[name]
    q, s, _ = OPS.scaled_int8_quant(x)
    return pr45126.triton_scaled_mm(q, w, s, ws, torch.bfloat16)


def cutlass_mm(layer, name, x):
    w, ws = layer[name]
    q, s, _ = OPS.scaled_int8_quant(x)
    return OPS.cutlass_scaled_mm(q, w, s, ws, torch.bfloat16)


def per_layer_us(stack: Stack, mm, windows: int) -> float:
    """Median microseconds per decoder layer, with the whole stack captured as one graph."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        stack.forward(mm)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stack.forward(mm)
    graph.replay()

    def window(reps: int) -> float:
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            graph.replay()
        stop.record()
        stop.synchronize()
        return start.elapsed_time(stop) * 1e3 / reps / LAYERS

    one = window(1) * LAYERS / 1e3  # milliseconds for one step
    reps = max(3, int(20.0 / max(one, 1e-3)))  # about 20 ms per window
    times = [window(reps) for _ in range(windows)]
    graph.reset()
    torch.cuda.empty_cache()
    return statistics.median(times)


def alternating(stack: Stack, options: dict, rounds: int, windows: int) -> dict[str, float]:
    """Each option timed in fresh captures, in rotation, and the median over rounds."""
    times: dict[str, list[float]] = {key: [] for key in options}
    for _ in range(rounds):
        for key, mm in options.items():
            times[key].append(per_layer_us(stack, mm, windows))
    return {key: statistics.median(v) for key, v in times.items()}


def served_us(backend: str, tag: str, batch: int) -> float:
    """Microseconds per decoder layer of a served decode step."""
    decode = load(f"e2e_{backend}{tag}.json")["decode"][str(batch)]
    return batch / decode["tok_s"] * 1e6 / LAYERS


def check(device: str, run1: dict, run2: dict, rounds: int, windows: int) -> dict:
    out: dict = {"points": []}
    for m in (32, 48, 128):
        stack = Stack(m)
        t = alternating(stack, {"run1": ours(configs(run1, m)), "run2": ours(configs(run2, m))},
                        rounds, windows)
        out["points"].append({
            "batch": m, "stand_in_us": t, "stand_in_change": t["run2"] - t["run1"],
            "served_change": served_us("ours", "_run2", m) - served_us("ours", "_run1", m)})
        print(f"check batch {m}: {out['points'][-1]}")
        del stack
    stack = Stack(64)
    t = alternating(stack, {"ours": ours(configs(run2, 64)), "pr45126": pr45126_mm,
                            "cutlass": cutlass_mm}, rounds, windows)
    del stack
    cliff = load(f"cliff_{device.replace(' ', '_')}.json")
    g = {(r["shape"], r["provider"]): r.get("graph_us") for r in cliff["rows"]
         if r["m"] == 64 and r["status"] == "ok"}

    def isolated(provider: str) -> float:
        return sum(g[(CLIFF_LABELS[name], provider)] for name in SHAPES)

    served = served_us("ours", "_run2", 64)
    b = out["batch64"] = {
        "stand_in_us": t,
        "stand_in_ours_minus_pr45126": t["ours"] - t["pr45126"],
        "isolated_ours_minus_pr45126": isolated(bl.OURS_TABLE) - isolated(bl.PR45126),
        "served_ours_minus_pr45126": served - served_us("pr45126", "_run2", 64),
        "stand_in_ours_minus_cutlass": t["ours"] - t["cutlass"],
        "isolated_ours_minus_cutlass": isolated(bl.OURS_TABLE) - isolated(bl.CUTLASS),
        "served_ours_minus_cutlass": served - served_us("cutlass", "_run2", 64),
    }
    print(f"check batch 64: {b}")
    p = {x["batch"]: x for x in out["points"]}

    def same(x: dict) -> bool:
        return (x["stand_in_change"] > 0) == (x["served_change"] > 0)

    out["criteria"] = {
        "batch 128: run 2 table slower than run 1, as served":
            same(p[128]) and p[128]["served_change"] > 0,
        "batch 48: run 2 table faster than run 1, as served":
            same(p[48]) and p[48]["served_change"] < 0,
        "batch 64: gap to #45126 under half the per-layer gap":
            abs(b["stand_in_ours_minus_pr45126"]) < 0.5 * abs(b["isolated_ours_minus_pr45126"]),
    }
    out["reported"] = {
        "batch 32: direction of the run 2 change matches the served model": same(p[32])}
    out["passed"] = all(out["criteria"].values())
    return out


def tune(device: str, run1: dict, run2: dict, sweep: list, sizes: list[int], rounds: int,
         windows: int, top: int) -> list[dict]:
    table = json.loads(TUNED_PATH.read_text())
    results = []
    for m in sizes:
        stack = Stack(m)
        start = configs(run2, m)
        current = dict(start)
        trials = []
        for name, (k, n) in SHAPES.items():
            rows = sorted((r for r in sweep if r["m"] == m and r["k"] == k and r["n"] == n
                           and r.get("graph_us")), key=lambda r: r["graph_us"])
            options = list(dict.fromkeys([current[name], table_config(run1, m, k, n),
                                          *(Config(**r["config"]) for r in rows[:top])]))
            best: tuple[Config, float] | None = None
            for cfg in options:
                us = per_layer_us(stack, ours({**current, name: cfg}), windows)
                trials.append({"layer": name, "config": cfg.name, "us": us})
                if best is None or us < best[1]:
                    best = (cfg, us)
            current[name] = best[0]
        entry: dict = {"batch": m, "start": {k: v.name for k, v in start.items()},
                       "chosen": {k: v.name for k, v in current.items()}, "trials": trials,
                       "kept": False}
        if current != start:
            t = alternating(stack, {"start": ours(start), "chosen": ours(current)},
                            max(rounds, 5), windows)
            entry["confirm_us"] = t
            entry["kept"] = t["chosen"] < t["start"] * (1 - ACCEPT)
            if entry["kept"]:
                for name, (k, n) in SHAPES.items():
                    table[device].setdefault(f"{k}x{n}", {})[str(m)] = {
                        "config": asdict(current[name]), "stand_in_us": t["chosen"]}
        del stack
        changed = {k: f"{entry['start'][k]} -> {entry['chosen'][k]}" for k in SHAPES
                   if entry["start"][k] != entry["chosen"][k]}
        print(f"batch {m}: {changed or 'no change'}; confirm {entry.get('confirm_us')}; "
              f"kept {entry['kept']}")
        results.append(entry)
    TUNED_PATH.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    return results


def markdown(res: dict) -> str:
    c = res["check"]
    env = res["env"]
    out = [f"# Stand-in model on {env['device']}", "",
           f"torch {env['torch']}, Triton {env['triton']}, vLLM {env['vllm']}. Microseconds per "
           "decoder layer: the stand-in's CUDA-graph replay, and the served decode step from "
           "results/e2e_*_run1.json and _run2.json.", "",
           "## Check against the served runs", "",
           "| batch | stand-in, run 1 table | stand-in, run 2 table | stand-in change "
           "| served change |",
           "|---|---|---|---|---|"]
    for p in c["points"]:
        t = p["stand_in_us"]
        out.append(f"| {p['batch']} | {t['run1']:.1f} | {t['run2']:.1f} "
                   f"| {p['stand_in_change']:+.1f} | {p['served_change']:+.1f} |")
    b = c["batch64"]
    out += ["", "At batch 64, this kernel minus the other, per decoder layer:", "",
            "| other | per-layer benchmark | stand-in | served |", "|---|---|---|---|",
            *(f"| {label} | {b['isolated_ours_minus_' + key]:+.1f} "
              f"| {b['stand_in_ours_minus_' + key]:+.1f} | {b['served_ours_minus_' + key]:+.1f} |"
              for label, key in (("#45126", "pr45126"), ("CUTLASS", "cutlass"))), "",
            "Criteria, fixed before the run:", ""]
    out += [f"- {k}: {'yes' if v else 'no'}" for k, v in c["criteria"].items()]
    out += [f"- (reported only) {k}: {'yes' if v else 'no'}" for k, v in c["reported"].items()]
    out += ["", f"**Check {'passed' if c['passed'] else 'failed'}.**", ""]
    if isinstance(res.get("tuning"), list):
        out += ["## Tuning in the stand-in", "",
                "| batch | layers changed | start | chosen | kept |", "|---|---|---|---|---|"]
        for e in res["tuning"]:
            changed = [k for k in e["start"] if e["start"][k] != e["chosen"][k]]
            conf = e.get("confirm_us")
            kept = "yes" if e["kept"] else "no"
            out.append(f"| {e['batch']} | {', '.join(changed) or 'none'} | {conf['start']:.1f} | "
                       f"{conf['chosen']:.1f} | {kept} |" if conf
                       else f"| {e['batch']} | none | | | no |")
    else:
        out += [f"Tuning: {res.get('tuning')}"]
    seen = res["interference"]
    others = sorted(set(seen.get("before", []) + seen.get("after", [])))
    out += ["", "Other processes on the GPU during the run: "
            + ("none." if not others else "; ".join(others))]
    return "\n".join(out) + "\n"


def main() -> None:
    global OPS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--top", type=int, default=5,
                    help="best per-layer configurations tried per layer")
    ap.add_argument("--batch", default=",".join(map(str, DECODE_SIZES)), help="batch sizes to tune")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    OPS, _, vllm_version = bl.load_vllm()
    if OPS is None:
        raise SystemExit(f"needs vLLM for its quantizer and CUTLASS kernel: {vllm_version}")

    device = torch.cuda.get_device_name()
    stem = device.replace(" ", "_")
    run1 = load("tuned_configs_run1.json")[device]
    run2 = load("tuned_configs_run2.json")[device]
    path = RESULTS / f"situ_{stem}.json"
    res: dict = {"env": bl.environment(vllm_version, args.windows, False),
                 "interference": {"before": bl.gpu_processes()}}
    res["check"] = check(device, run1, run2, args.rounds, args.windows)
    bl.save(path, res)
    if not res["check"]["passed"]:
        res["tuning"] = "skipped: the stand-in did not reproduce the served runs"
    elif args.check_only:
        res["tuning"] = "skipped: --check-only"
    else:
        sweep = load(f"qwen3-1.7b-vllm_{stem}_capture_sizes.json")["sweep"]
        sizes = [int(x) for x in args.batch.split(",")]
        res["tuning"] = tune(device, run1, run2, sweep, sizes, args.rounds, args.windows, args.top)
    res["interference"]["after"] = bl.gpu_processes()
    bl.save(path, res)
    md = markdown(res)
    path.with_suffix(".md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
