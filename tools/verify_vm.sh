#!/usr/bin/env bash
# Independent checks on the first benchmark run, on the GPU machine:
#
#   bash tools/verify_vm.sh 2>&1 | tee ~/verify_int8.log
#
# 1. whether tests/ passed, 2. whether vLLM's triton_scaled_mm is really wrong or
# the benchmark is, 3. whether vLLM's own int8 benchmark agrees with this repo's
# CUTLASS and bf16 timings. PYTHON picks the interpreter, as in run_vm.sh.
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
[[ -n "$py" ]] || { echo "No Python with torch, triton and vllm found; set PYTHON." >&2; exit 1; }
echo "python: $py"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true

echo; echo "===== 1. tests"
grep -h -E '[0-9]+ (passed|failed)|skipping tests|TESTS FAILED' ~/run_int8*.log 2>/dev/null | tail -3 \
  || echo "(no test summary in ~/run_int8*.log)"
if "$py" -m pytest --version >/dev/null 2>&1; then
  "$py" -m pytest -q tests || echo "(tests failed, continuing with the other checks)"
else
  echo "(pytest is not installed in this environment)"
fi

echo; echo "===== 2. vLLM triton_scaled_mm against exact products"
"$py" bench/diagnose_vllm_triton.py

echo; echo "===== 3. vLLM's own int8 benchmark against bench_linear.py"
src="$(mktemp -d)"
base="https://raw.githubusercontent.com/vllm-project/vllm/v0.28.0/benchmarks/kernels"
curl -fsSL "$base/benchmark_int8_gemm.py" -o "$src/benchmark_int8_gemm.py"
curl -fsSL "$base/weight_shapes.py" -o "$src/weight_shapes.py"
ours="$(ls results/qwen3-1.7b_*.json | grep -v -E 'cross_check|run1|_bias' | head -1)"
"$py" bench/cross_check_vllm_bench.py --vllm-bench "$src" --ours "$ours"
