#!/usr/bin/env bash
# Everything after the kernel benchmark, on the GPU machine, in one go:
#
#   bash tools/run_e2e_vm.sh 2>&1 | tee ~/e2e_int8.log
#
# STAGES picks a subset, for example STAGES="cliff e2e" bash tools/run_e2e_vm.sh.
#   tests    pytest, including the cached launch against Triton's own dispatch
#   tune     tuning for the layer shapes vLLM runs for Qwen3-1.7B (merged qkv, gate_up)
#   cliff    CUTLASS across its dispatch buckets (bench/cutlass_cliff.py)
#   launch   host launch cost per call (bench/launch_overhead.py)
#   repro    tools/cutlass_bucket_repro.py, the standalone CUTLASS repro, into results/
#   attrib   host time inside an eager decode step, CUTLASS against this kernel
#   situ     the stand-in model: checked against the served runs, then tuned in if it passes
#   e2e      decode, prefill and perplexity in vLLM per backend, with and without CUDA graphs
#   kernels  the kernel benchmark with #45126's tables on Qwen3-1.7B shapes, then Llama-3-8B
#
# The first e2e run downloads Qwen/Qwen3-1.7B and nytopop/Qwen3-1.7B.w8a8, about 6.5 GB.
# PYTHON picks the interpreter, as in run_vm.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

py="${PYTHON:-}"
if [[ -z "$py" ]]; then
  for cand in "$HOME"/*/.venv/bin/python "$HOME"/*-env/bin/python "$HOME"/.venv/bin/python; do
    [[ -x "$cand" ]] || continue
    if "$cand" -c 'import torch, triton, vllm; assert torch.cuda.is_available()' >/dev/null 2>&1; then
      py="$cand"
      break
    fi
  done
fi
[[ -n "$py" ]] || { echo "No Python with torch, triton, vllm and a visible GPU; set PYTHON." >&2; exit 1; }
echo "python: $py"

stages="${STAGES:-tests tune cliff launch repro attrib situ e2e kernels}"
want() { [[ " $stages " == *" $1 "* ]]; }
step() { echo; echo "===== $* ($(date +%H:%M:%S))"; }
# The kernel swaps only reach a model built in this process.
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# A graph compiled in one backend's run must not be loaded from cache by another's.
export VLLM_DISABLE_COMPILE_CACHE=1

echo "other processes on the GPU (timings are only clean if this is empty):"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true

if want tests; then
  step tests
  # pytest goes into a folder of its own if the environment lacks it, so the
  # borrowed vLLM environment is never modified.
  pyt="$HOME/.cache/int8-linear-pytest"
  if "$py" -m pytest --version >/dev/null 2>&1; then
    "$py" -m pytest -q tests || echo "TESTS FAILED; continuing"
  elif "$py" -m pip install --quiet --target "$pyt" pytest >/dev/null 2>&1; then
    PYTHONPATH="$pyt${PYTHONPATH:+:$PYTHONPATH}" "$py" -m pytest -q tests || echo "TESTS FAILED; continuing"
  else
    echo "pytest unavailable and could not be installed into $pyt; skipping tests/"
  fi
fi

if want tune; then
  step "tune: Qwen3-1.7B layers as vLLM merges them, at vLLM's CUDA-graph batch sizes"
  # vLLM pads each decode batch up to one of its capture sizes (1, 2, 4, then 8 to 248
  # by 8, then 256 to 512 by 16), so those are the M values decode runs at; the last
  # three are prefill sizes. Tuning only 1, 4, 16 and 64 left batches 17 to 63 on a
  # configuration tuned for 16.
  sizes="$("$py" -c 'print(",".join(map(str, [1, 2, 4, *range(8, 256, 8), *range(256, 513, 16), 1024, 2048, 4096])))')"
  "$py" bench/bench_linear.py --shapes qwen3-1.7b-vllm --tune --batch "$sizes" --tag capture_sizes \
    || echo "TUNE FAILED; continuing"
fi

if want cliff; then
  step "CUTLASS dispatch buckets"
  "$py" bench/cutlass_cliff.py || echo "CLIFF SWEEP FAILED; continuing"
fi

if want launch; then
  step "launch overhead"
  "$py" bench/launch_overhead.py || echo "LAUNCH OVERHEAD FAILED; continuing"
fi

if want repro; then
  step "standalone CUTLASS repro"
  "$py" tools/cutlass_bucket_repro.py 2>&1 | tee results/cutlass_bucket_repro.txt || echo "REPRO FAILED; continuing"
fi

if want attrib; then
  for b in cutlass ours; do
    step "eager host-time attribution: $b"
    "$py" bench/eager_attribution.py --backend "$b" || echo "ATTRIBUTION $b FAILED; continuing"
  done
  "$py" bench/eager_attribution.py --report || echo "ATTRIBUTION REPORT FAILED; continuing"
fi

if want situ; then
  step "stand-in model: check against the served runs, then tune in it if the check passes"
  "$py" bench/situ_tune.py || echo "STAND-IN FAILED; continuing"
fi

if want e2e; then
  for b in bf16 cutlass vllm_triton pr45126 ours; do
    step "e2e $b"
    "$py" bench/e2e_vllm.py --backend "$b" || echo "E2E $b FAILED; continuing"
  done
  for b in bf16 cutlass ours; do
    step "e2e $b without torch.compile or CUDA graphs"
    "$py" bench/e2e_vllm.py --backend "$b" --enforce-eager --no-quality --batch 1,16,64 \
      || echo "E2E $b EAGER FAILED; continuing"
  done
  step "e2e report"
  "$py" bench/e2e_report.py || echo "E2E REPORT FAILED"
fi

if want kernels; then
  step "kernels: Qwen3-1.7B layers with #45126, stored configurations"
  "$py" bench/bench_linear.py --shapes qwen3-1.7b --tag pr45126 || echo "KERNELS qwen3 FAILED; continuing"
  step "kernels: Llama-3-8B layers, tuned"
  "$py" bench/bench_linear.py --shapes llama-3-8b --tune || echo "KERNELS llama FAILED"
fi

step done
ls -la results
