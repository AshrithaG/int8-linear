#!/usr/bin/env bash
# The fused activation-quantizer study, on the GPU machine, in one go:
#
#   bash tools/run_fuse_vm.sh 2>&1 | tee ~/fuse_int8.log
#
# STAGES picks a subset (default: tests fuse e2e report).
#   tests   tests/test_fused_quant.py: the fused kernels against torch references
#   fuse    bench/fuse_quant.py: numerics against vLLM, kernel times, and the stand-in's
#           prediction, written to results/ BEFORE any end-to-end run
#   e2e     two rounds of cutlass, cutlass_fused, ours, ours_fused in vLLM, alternating,
#           into results/fuse_e2e/round{1,2}/ (the published e2e_*.json are untouched)
#   report  results/fuse.md
set -euo pipefail
cd "$(dirname "$0")/.."

py="${PYTHON:-}"
if [[ -z "$py" ]]; then
  for cand in "$HOME"/*/.venv/bin/python "$HOME"/*-env/bin/python "$HOME"/.venv/bin/python; do
    [[ -x "$cand" ]] || continue
    if "$cand" -c 'import torch, triton, vllm; assert torch.cuda.is_available()' >/dev/null 2>&1; then
      py="$cand"; break
    fi
  done
fi
[[ -n "$py" ]] || { echo "No Python with torch, triton, vllm and a visible GPU; set PYTHON." >&2; exit 1; }
echo "python: $py"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || { echo "nvidia-smi failed: GPU unusable (driver mismatch? needs a reboot)"; exit 1; }

stages="${STAGES:-tests fuse e2e report}"
want() { [[ " $stages " == *" $1 "* ]]; }
step() { echo; echo "===== $* ($(date +%H:%M:%S))"; }
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_DISABLE_COMPILE_CACHE=1

echo "other processes on the GPU (timings are only clean if this is empty):"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true

if want tests; then
  step tests
  pyt="$HOME/.cache/int8-linear-pytest"
  if "$py" -m pytest --version >/dev/null 2>&1; then
    "$py" -m pytest -q tests/test_fused_quant.py || echo "TESTS FAILED; continuing"
  else
    PYTHONPATH="$pyt${PYTHONPATH:+:$PYTHONPATH}" "$py" -m pytest -q tests/test_fused_quant.py || echo "TESTS FAILED; continuing"
  fi
fi

if want fuse; then
  step "fuse: numerics, kernel times, stand-in prediction"
  "$py" bench/fuse_quant.py
fi

if want e2e; then
  for round in 1 2; do
    for b in cutlass cutlass_fused ours ours_fused; do
      step "e2e round $round: $b"
      "$py" bench/e2e_vllm.py --backend "$b" --out "results/fuse_e2e/round$round" \
        || echo "E2E $b ROUND $round FAILED; continuing"
    done
  done
fi

if want report; then
  step report
  "$py" bench/fuse_report.py || echo "REPORT FAILED"
fi
