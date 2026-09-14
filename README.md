# int8-linear

W8A8 int8 linear layers for LLM inference on a consumer GPU: a Triton kernel,
tuned and measured against vLLM's CUTLASS and Triton kernels and bf16 cuBLAS.

**Status: measured on an RTX 4090, cross-checked against vLLM's own benchmark, and
confirmed on a second run. Kernel-level only so far; end-to-end decode throughput is next.**

## The question

On an RTX 4090, where does W8A8 int8 actually beat bf16 for the linear layers of
a small LLM, and can a Triton kernel match vLLM's CUTLASS kernel at the shapes
decode and prefill really use?

## Where this comes from

[nanoinfer](https://github.com/AshrithaG/nanoinfer#against-cublass-fast-path)
compared a hand-written CUDA int8 GEMM, the same GEMM in Triton, and cuBLAS on
square matrices, in one process under CUDA-graph timing. Four results carry over:

- **Operand layout decides the vendor comparison.** At n=4096 cuBLAS was 3.4x
  faster with both operands K-contiguous than with B row-major. A transposed
  nn.Linear weight is already K-contiguous, so every provider here gets that layout.
- **Tile size was the largest lever** for Triton (1.66x), while pipelining at
  small tiles was worth 1.01x.
- **Triton matched cuBLAS** on square matrices, at 540 TOPS.
- **Back-to-back timing misranks small kernels.** At n=256 about 10 microseconds
  of every cuBLAS call was host work, enough to reverse a comparison.

Square matrices are not what an LLM runs. Its linear layers are wide and short:
at decode M is a handful of tokens against K and N in the thousands, a different
regime for tiling, for memory traffic and for the epilogue.

## Prior work this builds on

- **Tuning vLLM's Triton int8 kernel is not new.**
  [#44998](https://github.com/vllm-project/vllm/issues/44998) showed its tile
  heuristic, tuned on AMD, leaves up to 1.82x on an H800 and 1.65x on an L20, and
  [#45126](https://github.com/vllm-project/vllm/pull/45126) proposes tuned tables
  with grouped tile order. Both compare the tuned kernel with the current one and
  treat the benefit on NVIDIA as bounded, because CUTLASS is the default there.
  Neither compares against CUTLASS, and neither checks outputs against an
  independent reference.
- **The small-batch W8A8 slowdown is understood.**
  [#38697](https://github.com/vllm-project/vllm/issues/38697) reported W8A8
  decode slower than FP16, and later measurements on Hopper attributed it to
  host launch cost that disappears under CUDA graphs.
- **vLLM's Triton kernel has a newer load path.**
  [#47205](https://github.com/vllm-project/vllm/pull/47205) added
  tensor-descriptor loads, on by default only on XPU and opt-in on CUDA through
  `VLLM_TRITON_USE_TD`.

What is left open, and what this repo measures, is the comparison those threads
skip: vLLM's default int8 kernel on a consumer Ada GPU against a fused Triton
kernel and bf16, with every provider checked against a float64 reference first.

## What is measured

| provider | what it is |
|---|---|
| bf16 F.linear | cuBLAS, the unquantized baseline |
| int8 CUTLASS (vLLM) | `cutlass_scaled_mm`, vLLM's W8A8 path on NVIDIA |
| int8 Triton (vLLM) | `triton_scaled_mm`, vLLM's fallback and its ROCm path |
| int8 torch._int_mm, unfused | the cuBLASLt product, then scales and bias as separate kernels |
| int8 Triton (this repo) | [`int8_linear/kernel.py`](int8_linear/kernel.py), default and tuned |

CUTLASS and this repo's kernel are also timed with vLLM's per-token
`scaled_int8_quant` inside the measurement, since a real layer pays for both.

Shapes are Qwen3-1.7B's four distinct linear layers (2048 to 2048, 2048 to 1024,
2048 to 6144, 6144 to 2048) and, for comparison with vLLM's own benchmark,
Llama-3-8B's. M runs 1, 4, 16, 64, 256, 1024 and 4096 tokens.

## Method

The rules from the nanoinfer study:

- Every provider runs in one process, on one stream, under one timer.
- CUDA-graph replay is the primary timing, with back-to-back launches reported
  next to it.
- This repo's kernel must reproduce the int32 product exactly (`tests/`), and
  every provider's bf16 output must agree with a float64 reference to within 1% of
  the larger of the output and its matmul term before it is timed. vLLM's own test
  for its Triton kernel allows 10%.
- Output buffers are poisoned before each check, so a kernel that writes nothing
  fails instead of passing on another provider's leftover answer.
- Other GPU processes are recorded before and after, and timings are written to
  disk after every shape.

## The kernel

- int8 times int8 accumulated in int32 with `tl.dot`, so the product is exact.
- Per-token and per-channel scales and the bias applied in the same kernel and
  written straight to bf16.
- The weight taken as the transposed [K, N] view of an nn.Linear weight, which is
  K-contiguous.
- Only the batch axis may be ragged. N and K must divide their tiles, which every
  shape here does; anything else is declined rather than masked.
- Tile shape, warps, stages and tile order are tunable per GPU, shape and batch
  bucket. `bench/bench_linear.py --tune` writes them to
  `int8_linear/tuned_configs.json`, which the kernel reads.

vLLM's `triton_scaled_mm` differs in ways this is set up to test. It picks tile
shapes from M alone, from 64x64x256 for small batches to 128x128x128 for large
ones, never tunes warps or stages, and masks every axis unless its
tensor-descriptor loads are in use, which `VLLM_TRITON_USE_TD` controls.

## Results

RTX 4090, vLLM 0.28.0, Triton 3.7.1, torch 2.13.0+cu130, Qwen3-1.7B layer shapes, no bias. Microseconds per layer under CUDA-graph replay, median of five windows, with no other process on the GPU. The worst window spread was 6.1% of its median. Full output, including back-to-back timings and every tuned configuration, is in [`results/`](results/).

**The harness agrees with vLLM's own.** Timed with vLLM's `benchmarks/kernels/benchmark_int8_gemm.py` at the same shapes, this benchmark's numbers match within a median ratio of 1.00 for bf16 (range 0.95 to 1.11), 1.00 for CUTLASS (0.91 to 1.09), and 1.00 for CUTLASS with activation quantization (0.93 to 1.09).

**The tuned Triton kernel is faster than vLLM's CUTLASS kernel at all 28 points,** by a median 1.26x (1.05x to 2.05x). It is faster than vLLM's Triton kernel at all 28 (median 1.42x, 1.04x to 3.32x) and than bf16 cuBLAS at all 28 (median 2.39x, 1.27x to 3.51x). With vLLM's per-token activation quantizer inside the timing it is still ahead of CUTLASS at all 28 points (median 1.21x, 1.04x to 2.02x). At M=4096 it runs at 538 to 594 TOPS, against 278 to 489 for CUTLASS.

**Confirmed on a fresh run.** Rerun without tuning, with this repo's kernel reading its
configurations from the stored table, it is again faster than CUTLASS at all 28 points
(median 1.29x, 1.05x to 2.24x) and than bf16 at all 28 (median 2.39x). Its time with the
stored table was a median 0.999 of the time the sweep had picked, so choosing and reporting
from the same run added no measurable optimism. Between the two runs CUTLASS timings moved
by a median 2%, and bf16 by less than 0.1%.

**CUTLASS has a cliff at M=64 on this GPU.** From M=16 to M=64 its time roughly doubles at every shape, and at M=64 it is slower than bf16 for 3 of 4 shapes (q_proj 0.81x, k_proj 0.67x, down_proj 0.92x), and for all 4 with activation quantization (q_proj 0.71x, k_proj 0.59x, gate_proj 0.93x, down_proj 0.82x). vLLM's own benchmark shows the same cliff, and in the confirmation run CUTLASS alone was slower than bf16 at M=64 for all four shapes.

**Where it loses.**

- **Without CUDA graphs, at decode.** At M=1 on q_proj, back-to-back launches take 37.0 microseconds for this kernel against 25.0 for CUTLASS and 10.6 for bf16: Triton's Python launcher costs more than the kernel. The advantage exists only under CUDA graphs, which vLLM uses for decode, or at large M.
- **Untuned.** The fixed default configuration is a median 1.01x of CUTLASS and slower at 14 of 28 points. The advantage comes from per-shape tuning.
- **Narrowly at some shapes.** The smallest margin over CUTLASS is 1.05x (down_proj, M=256), in both runs.
- **No accuracy measurement.** Everything here is kernel time on random operands; what naive W8A8 costs a real model is not measured yet.

vLLM's own Triton kernel is a median 0.93x of CUTLASS. It is weakest at M=256, where its fixed tile heuristic leaves it at 0.42x to 0.54x of CUTLASS on three of the four shapes; on gate_proj it is 1.06x.

### q_proj, o_proj: K=2048, N=2048

| M | bf16 | CUTLASS | vLLM Triton | this repo, tuned | tuned over CUTLASS |
|---|---|---|---|---|---|
| 1 | 6.29 | 4.87 | 6.41 | 3.92 | 1.24x |
| 4 | 9.50 | 4.91 | 6.43 | 3.93 | 1.25x |
| 16 | 10.1 | 5.02 | 6.25 | 4.27 | 1.18x |
| 64 | 8.21 | 10.1 | 6.57 | 5.37 | 1.88x |
| 256 | 16.9 | 11.3 | 21.0 | 8.91 | 1.27x |
| 1024 | 52.0 | 26.7 | 23.3 | 18.3 | 1.46x |
| 4096 | 199.5 | 101.6 | 88.1 | 62.3 | 1.63x |

### k_proj, v_proj: K=2048, N=1024

| M | bf16 | CUTLASS | vLLM Triton | this repo, tuned | tuned over CUTLASS |
|---|---|---|---|---|---|
| 1 | 4.71 | 4.40 | 5.85 | 3.53 | 1.25x |
| 4 | 8.58 | 4.45 | 5.90 | 3.56 | 1.25x |
| 16 | 8.96 | 4.44 | 5.61 | 3.70 | 1.20x |
| 64 | 6.02 | 8.97 | 6.06 | 4.64 | 1.93x |
| 256 | 11.4 | 10.2 | 20.9 | 6.30 | 1.63x |
| 1024 | 28.1 | 21.5 | 21.7 | 13.0 | 1.65x |
| 4096 | 102.0 | 53.2 | 45.4 | 31.9 | 1.67x |

### gate_proj, up_proj: K=2048, N=6144

| M | bf16 | CUTLASS | vLLM Triton | this repo, tuned | tuned over CUTLASS |
|---|---|---|---|---|---|
| 1 | 11.3 | 5.72 | 6.09 | 4.49 | 1.28x |
| 4 | 10.2 | 5.90 | 6.30 | 4.95 | 1.19x |
| 16 | 12.3 | 6.54 | 6.15 | 5.31 | 1.23x |
| 64 | 15.3 | 15.0 | 7.82 | 7.50 | 2.01x |
| 256 | 49.3 | 23.5 | 22.2 | 17.6 | 1.33x |
| 1024 | 154.2 | 96.6 | 66.6 | 48.5 | 1.99x |
| 4096 | 605.4 | 370.3 | 257.6 | 180.4 | 2.05x |

### down_proj: K=6144, N=2048

| M | bf16 | CUTLASS | vLLM Triton | this repo, tuned | tuned over CUTLASS |
|---|---|---|---|---|---|
| 1 | 13.1 | 9.14 | 14.5 | 7.61 | 1.20x |
| 4 | 23.6 | 9.10 | 14.5 | 7.62 | 1.19x |
| 16 | 25.2 | 9.23 | 13.7 | 8.13 | 1.14x |
| 64 | 14.5 | 15.8 | 14.2 | 11.4 | 1.38x |
| 256 | 47.8 | 24.1 | 57.1 | 23.0 | 1.05x |
| 1024 | 159.4 | 55.7 | 59.9 | 47.2 | 1.18x |
| 4096 | 608.0 | 210.7 | 234.1 | 173.5 | 1.21x |

## Status

1. **Kernel, tests and benchmark: built, run once, and checked.** The first run
   (RTX 4090, Qwen3-1.7B shapes, vLLM 0.28.0, Triton 3.7.1) is kept as
   `results/*_run1_bias.*`. Checking it found two problems, both in this
   benchmark rather than in vLLM:
   - **vLLM's `triton_scaled_mm` is not wrong.** It failed the float64 check at
     nearly every shape, but `bench/diagnose_vllm_triton.py` shows its int32
     product is exact with the weight in either layout and with
     tensor-descriptor loads on or off. It rounds the scaled product to bf16 and
     then adds the bias, so where the two nearly cancel, the output lands a
     rounding step of the product away from a near-zero sum, sometimes at exactly
     zero. The check divided by that sum. It now divides by the larger of the sum
     and its matmul term.
   - **The first run slowed CUTLASS down.** Its layers carried a bias, which the
     Qwen3 layers do not have. Against vLLM's own
     `benchmarks/kernels/benchmark_int8_gemm.py` on the same GPU, bf16 timings
     agreed (median ratio 1.00) while CUTLASS measured a median 1.33x slower here,
     up to 1.71x. Bias is now off by default. Comparisons with CUTLASS wait for a
     single-process rerun, since set against vLLM's own numbers they would mix two
     runs.
2. **Rerun with bias off and the corrected check: done.** Those are the results above.
3. **Confirmation run: done.** No tuning, configurations from `int8_linear/tuned_configs.json`,
   all 25 tests passing; results in `results/*_confirm.*`.
4. End to end: Qwen3-1.7B decode throughput under CUDA graphs with these layers
   swapped in, and the perplexity cost of naive W8A8.
5. Once end-to-end numbers exist, the Ada measurements go upstream: the CUTLASS cliff at
   M=64 as a vLLM issue with data, and tuned configurations alongside
   [#45126](https://github.com/vllm-project/vllm/pull/45126).

## Reproduce

On a CUDA machine with torch, Triton and vLLM:

```bash
bash tools/run_vm.sh              # tests, tuning sweep and benchmark
bash tools/verify_vm.sh           # vLLM Triton diagnostic and vLLM's own benchmark
python bench/bench_linear.py --tag confirm   # stored configurations, no tuning
```

## License

MIT.
