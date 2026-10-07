"""End-to-end decode, prefill and perplexity in vLLM with each int8 linear kernel.

The kernel study times one matmul at a time. This measures what serving sees: one
W8A8 checkpoint in vLLM with only the int8 matmul kernel changing between runs, and
the unquantized model for reference.

    bf16          Qwen/Qwen3-1.7B, unquantized
    cutlass       the W8A8 checkpoint, --linear-backend cutlass (vLLM's default on NVIDIA)
    vllm_triton   the same checkpoint, --linear-backend triton, as vLLM ships it
    pr45126       the Triton backend running the kernel and tables from vllm-project/vllm#45126
    ours          the Triton backend running this repo's kernel (int8_linear/vllm_patch.py)
    cutlass_fused cutlass, with each RMSNorm and SiLU-and-mul quantizing its own output
                  in one fused kernel instead of a separate scaled_int8_quant
                  (int8_linear/fused_quant.py, int8_linear/vllm_fuse_patch.py)
    ours_fused    ours, with the same fusion

pr45126 and ours each sit behind a custom op, so under torch.compile they choose a
configuration from every call's batch; vllm_triton is traced by torch.compile as
shipped. By default vLLM compiles the model and replays CUDA graphs for decode;
--enforce-eager turns both off.

Run one backend per process so nothing carries over, with the engine in-process (the
kernel swaps need it) and compile caching off (a graph compiled for one backend must
not be loaded by another). Prefix caching is off so repeated prompts are recomputed:

    VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_DISABLE_COMPILE_CACHE=1 \\
        python bench/e2e_vllm.py --backend ours
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BF16_MODEL = "Qwen/Qwen3-1.7B"
# Smoothing strength 0.7, then GPTQ, through llm-compressor: int8 per-channel symmetric
# weights, dynamic per-token symmetric int8 activations, lm_head left unquantized. The
# checkpoint's config.json and recipe.yaml at this revision say so.
W8A8_MODEL = "nytopop/Qwen3-1.7B.w8a8"
W8A8_REVISION = "cbf1e72c353f3b9ae24487c2dae8877ee53fe002"
BACKENDS = ("bf16", "cutlass", "vllm_triton", "pr45126", "ours", "cutlass_fused", "ours_fused")
SAMPLE_PROMPTS = ["The capital of France is", "Explain what a hash table is.", "def fibonacci(n):"]
WIKITEXT = Path.home() / ".cache" / "int8-linear" / "wikitext-2-raw-v1-test.txt"


def first_line(e: BaseException) -> str:
    text = str(e).strip()
    return (text.splitlines()[0] if text else type(e).__name__)[:200]


def gpu_processes() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return ["nvidia-smi unavailable"]
    me = f"{os.getpid()},"
    return [ln.strip() for ln in out.splitlines() if ln.strip() and not ln.startswith(me)]


def build_llm(backend: str, max_model_len: int, enforce_eager: bool):
    from vllm import LLM

    common = dict(max_model_len=max_model_len, gpu_memory_utilization=0.85, seed=0,
                  dtype="bfloat16", enable_prefix_caching=False, enforce_eager=enforce_eager)
    if backend == "bf16":
        return LLM(model=BF16_MODEL, **common)
    if backend in ("ours", "ours_fused"):
        from int8_linear.vllm_patch import patch_vllm
        patch_vllm()
    elif backend == "pr45126":
        from int8_linear.pr45126 import patch_vllm
        patch_vllm()
    if backend.endswith("_fused"):
        from int8_linear import vllm_fuse_patch
        vllm_fuse_patch.patch_vllm()
    linear_backend = "cutlass" if backend.startswith("cutlass") else "triton"
    return LLM(model=W8A8_MODEL, revision=W8A8_REVISION, linear_backend=linear_backend, **common)


def linear_kernels(llm) -> dict:
    """The linear kernel classes the built model uses, counted by layer, so a run that
    silently picked another backend shows."""
    def probe(model):
        counts: dict[str, int] = {}
        for module in model.modules():
            kernel = getattr(getattr(module, "scheme", None), "kernel", None)
            if kernel is not None:
                counts[type(kernel).__name__] = counts.get(type(kernel).__name__, 0) + 1
        return counts

    try:
        return llm.apply_model(probe)[0]
    except Exception as e:  # reporting only; the benchmark does not depend on it
        return {"unavailable": first_line(e)}


def random_prompt(llm, length: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    vocab = llm.get_tokenizer().vocab_size
    return torch.randint(100, vocab - 100, (length,), generator=g).tolist()


def timed(llm, prompts, params) -> tuple[float, list]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    outs = llm.generate(prompts, params, use_tqdm=False)
    torch.cuda.synchronize()
    return time.perf_counter() - start, outs


def decode(llm, batch_sizes: list[int], repeats: int, tokens: int, prompt_len: int) -> dict:
    """Decode tokens per second for a batch of identical short prompts. Each repeat
    generates `tokens` tokens and then, separately, only the first one; the median
    first-token time is taken off the median total, so the prompt step is not counted."""
    from vllm import SamplingParams

    full_params = SamplingParams(temperature=0.0, max_tokens=tokens, ignore_eos=True)
    first_params = SamplingParams(temperature=0.0, max_tokens=1)
    prompt = {"prompt_token_ids": random_prompt(llm, prompt_len, seed=0)}
    out = {}
    for bs in batch_sizes:
        timed(llm, [prompt] * bs, full_params)  # warmup
        full, first, generated = [], [], 0
        for _ in range(repeats):
            dt, outs = timed(llm, [prompt] * bs, full_params)
            full.append(dt)
            generated = sum(len(o.outputs[0].token_ids) for o in outs)
            first.append(timed(llm, [prompt] * bs, first_params)[0])
        decode_s = statistics.median(full) - statistics.median(first)
        out[str(bs)] = {"tok_s": bs * (tokens - 1) / decode_s,
                        "end_to_end_tok_s": generated / statistics.median(full),
                        "generated": generated, "full_s": full, "first_token_s": first}
        print(f"decode batch {bs:>3}: {out[str(bs)]['tok_s']:.1f} tok/s "
              f"({generated} tokens generated per run)")
    return out


def prefill(llm, n_prompts: int, prompt_len: int, repeats: int) -> dict:
    """Prompt tokens per second for distinct long prompts generating one token each."""
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=1)
    prompts = [{"prompt_token_ids": random_prompt(llm, prompt_len, seed=100 + i)}
               for i in range(n_prompts)]
    timed(llm, prompts, params)  # warmup
    runs = [n_prompts * prompt_len / timed(llm, prompts, params)[0] for _ in range(repeats)]
    print(f"prefill {n_prompts} x {prompt_len}: {statistics.median(runs):.0f} tok/s")
    return {"prompts": n_prompts, "prompt_len": prompt_len,
            "tok_s": statistics.median(runs), "runs": runs}


def wikitext_test() -> tuple[str, str]:
    """The WikiText-2 raw test split and where it came from. The first source that
    works is cached, so every backend in a study scores the same text."""
    source_file = WIKITEXT.with_suffix(".source")
    if WIKITEXT.exists() and source_file.exists():
        return WIKITEXT.read_text(), source_file.read_text().strip()
    errors: list[str] = []
    text = source = None
    try:
        from datasets import load_dataset

        rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
        text = "\n\n".join(rows)
        source = "datasets: Salesforce/wikitext wikitext-2-raw-v1 test, rows joined by blank lines"
    except Exception as e:  # try the next source
        errors.append(f"datasets: {first_line(e)}")
    if text is None:
        try:
            import pyarrow.parquet as pq
            from huggingface_hub import hf_hub_download

            name = "wikitext-2-raw-v1/test-00000-of-00001.parquet"
            path = hf_hub_download("Salesforce/wikitext", name, repo_type="dataset")
            text = "\n\n".join(pq.read_table(path).column("text").to_pylist())
            source = f"hub: Salesforce/wikitext {name}, rows joined by blank lines"
        except Exception as e:  # try the next source
            errors.append(f"parquet: {first_line(e)}")
    if text is None:
        url = "https://wikitext.smerity.com/wikitext-2-raw-v1.zip"
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                archive = zipfile.ZipFile(io.BytesIO(r.read()))
            member = next(n for n in archive.namelist() if n.endswith("wiki.test.raw"))
            text, source = archive.read(member).decode(), f"{url}: {member}"
        except Exception as e:
            errors.append(f"zip: {first_line(e)}")
    if text is None:
        raise RuntimeError("; ".join(errors))
    WIKITEXT.parent.mkdir(parents=True, exist_ok=True)
    WIKITEXT.write_text(text)
    source_file.write_text(source + "\n")
    return text, source


def perplexity(llm, windows: int, window_len: int) -> dict:
    """Perplexity over fixed WikiText-2 test windows, plus every token's logprob so
    backends can be compared token by token."""
    from vllm import SamplingParams

    try:
        text, source = wikitext_test()
    except Exception as e:
        return {"status": f"skipped: could not load WikiText-2 ({first_line(e)})"}
    ids = llm.get_tokenizer()(text).input_ids
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    rows, nll, count = [], 0.0, 0
    for i in range(windows):
        window = ids[i * window_len:(i + 1) * window_len]
        if len(window) < window_len:
            break
        # One window per call keeps the full-vocabulary prompt logprobs small.
        out = llm.generate([{"prompt_token_ids": window}], params, use_tqdm=False)[0]
        lps = [out.prompt_logprobs[p][window[p]].logprob for p in range(1, len(window))]
        nll -= sum(lps)
        count += len(lps)
        rows.append([round(x, 5) for x in lps])
    ppl = math.exp(nll / count)
    print(f"perplexity over {count} tokens: {ppl:.4f}")
    return {"status": "ok", "source": source, "windows": len(rows), "window_len": window_len,
            "tokens": count, "ppl": ppl, "token_logprobs": rows}


def samples(llm) -> list[str]:
    from vllm import SamplingParams

    outs = llm.generate(SAMPLE_PROMPTS, SamplingParams(temperature=0.0, max_tokens=48),
                        use_tqdm=False)
    return [o.outputs[0].text for o in outs]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=BACKENDS, required=True)
    ap.add_argument("--enforce-eager", action="store_true",
                    help="no torch.compile and no CUDA graphs")
    ap.add_argument("--quality", action=argparse.BooleanOptionalAction, default=True,
                    help="perplexity and greedy samples")
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--batch", default="1,4,16,32,48,64,128", help="decode batch sizes")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--prefill-prompts", type=int, default=8)
    ap.add_argument("--prefill-len", type=int, default=512)
    ap.add_argument("--ppl-windows", type=int, default=40)
    ap.add_argument("--ppl-window-len", type=int, default=512)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise SystemExit("set VLLM_ENABLE_V1_MULTIPROCESSING=0: the kernel swaps only reach "
                         "a model built in this process")
    if not args.enforce_eager and os.environ.get("VLLM_DISABLE_COMPILE_CACHE") != "1":
        raise SystemExit("set VLLM_DISABLE_COMPILE_CACHE=1: a graph compiled for another "
                         "backend must not be loaded from the cache")
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")

    import triton
    import vllm

    suffix = "_eager" if args.enforce_eager else ""
    path = args.out / f"e2e_{args.backend}{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    res: dict = {
        "backend": args.backend,
        "model": BF16_MODEL if args.backend == "bf16" else W8A8_MODEL,
        "revision": None if args.backend == "bf16" else W8A8_REVISION,
        "env": {"device": torch.cuda.get_device_name(), "vllm": vllm.__version__,
                "torch": torch.__version__, "triton": triton.__version__,
                "python": platform.python_version(), "max_model_len": args.max_model_len,
                "enforce_eager": args.enforce_eager, "prefix_caching": False,
                "decode_tokens": args.decode_tokens, "prompt_len": args.prompt_len,
                "repeats": args.repeats,
                "compile_cache_disabled": os.environ.get("VLLM_DISABLE_COMPILE_CACHE") == "1"},
        "interference": {"before": gpu_processes()},
    }

    def save() -> None:
        path.write_text(json.dumps(res, indent=1, default=str))

    llm = build_llm(args.backend, args.max_model_len, args.enforce_eager)
    res["linear_kernels"] = linear_kernels(llm)
    print("linear kernels:", res["linear_kernels"])
    batch = [int(x) for x in args.batch.split(",")]
    res["decode"] = decode(llm, batch, args.repeats, args.decode_tokens, args.prompt_len)
    save()
    res["prefill"] = prefill(llm, args.prefill_prompts, args.prefill_len, args.repeats)
    save()
    if args.quality:
        res["samples"] = samples(llm)
        save()
        res["perplexity"] = perplexity(llm, args.ppl_windows, args.ppl_window_len)
    if args.backend.endswith("_fused"):
        from int8_linear import fused_quant, vllm_fuse_patch
        res.update(prequant_calls=vllm_fuse_patch.PREQUANT_CALLS,
                   fused_numerics={"precise_div": fused_quant.PRECISE_DIV,
                                   "round_silu": fused_quant.ROUND_SILU})
    if args.backend in ("ours", "ours_fused"):
        from int8_linear import vllm_patch
        res.update(kernel_calls=vllm_patch.CALLS, fallbacks=vllm_patch.FALLBACKS,
                   configs={f"M={m} {k}x{n}": c for (m, k, n), c in vllm_patch.CONFIGS.items()})
    elif args.backend == "pr45126":
        from int8_linear import pr45126
        res["kernel_calls"] = pr45126.CALLS
    res["interference"]["after"] = gpu_processes()
    save()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
