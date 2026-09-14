#!/usr/bin/env bash
# Tests, tuning and the benchmark on a CUDA machine.
#
#   bash tools/run_vm.sh --batch 1,64 --windows 3   # a quick check first
#   bash tools/run_vm.sh                            # Qwen3-1.7B shapes, full run
#   bash tools/run_vm.sh --shapes llama-3-8b        # the shapes vLLM benchmarks
#
# PYTHON picks the interpreter. Otherwise the first venv under $HOME that imports
# torch, triton and vllm with a GPU visible is used; vLLM supplies the CUTLASS and
# vLLM-Triton columns and the activation quantizer.
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
if [[ -z "$py" ]]; then
  echo "No Python with torch, triton, vllm and a visible GPU. Set PYTHON=/path/to/python." >&2
  exit 1
fi
echo "python: $py"

echo "other processes on the GPU (timings are only clean if this is empty):"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true

# pytest goes into a cache folder of its own if the environment lacks it, so the
# vLLM environment this borrows is never modified.
pyt="$HOME/.cache/int8-linear-pytest"
if "$py" -m pytest --version >/dev/null 2>&1; then
  "$py" -m pytest -q tests || echo "TESTS FAILED; continuing so the benchmark's own checks run"
elif "$py" -m pip install --quiet --target "$pyt" pytest >/dev/null 2>&1; then
  PYTHONPATH="$pyt${PYTHONPATH:+:$PYTHONPATH}" "$py" -m pytest -q tests \
    || echo "TESTS FAILED; continuing so the benchmark's own checks run"
else
  echo "pytest unavailable and could not be installed into $pyt; skipping tests/"
fi

"$py" bench/bench_linear.py --tune "$@"
