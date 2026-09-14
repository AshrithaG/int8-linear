# Launch cost of one int8 linear layer on NVIDIA GeForce RTX 4090

torch 2.13.0+cu130, Triton 3.7.1, vLLM 0.28.0, launch cache on. Microseconds per call, median of 7 windows, no bias. Host is CPU time with nothing synchronized inside the window; back-to-back is time between CUDA events; graph is CUDA-graph replay.

## q_proj, o_proj: K=2048, N=2048

| M | provider | host | back-to-back | graph | back-to-back minus graph |
|---|---|---|---|---|---|
| 1 | bf16 F.linear (cuBLAS) | 9.82 | 9.80 | 5.82 | 3.98 |
| 1 | int8 CUTLASS (vLLM) | 24.2 | 24.2 | 4.48 | 19.7 |
| 1 | int8 Triton (vLLM) | 46.0 | 45.8 | 5.89 | 39.9 |
| 1 | ours, JIT dispatch | 38.6 | 38.4 | 3.93 | 34.4 |
| 1 | ours, cached launch | 29.1 | 29.1 | 3.93 | 25.2 |
| 1 | ours, custom op | 60.6 | 60.7 | 3.93 | 56.7 |
| 4 | bf16 F.linear (cuBLAS) | 11.1 | 11.3 | 9.41 | 1.93 |
| 4 | int8 CUTLASS (vLLM) | 24.3 | 24.4 | 4.47 | 19.9 |
| 4 | int8 Triton (vLLM) | 46.0 | 46.0 | 5.93 | 40.1 |
| 4 | ours, JIT dispatch | 38.4 | 38.4 | 3.62 | 34.8 |
| 4 | ours, cached launch | 28.6 | 29.1 | 3.96 | 25.2 |
| 4 | ours, custom op | 60.8 | 60.9 | 3.95 | 57.0 |
| 16 | bf16 F.linear (cuBLAS) | 10.7 | 11.2 | 10.3 | 0.92 |
| 16 | int8 CUTLASS (vLLM) | 24.3 | 24.3 | 4.55 | 19.8 |
| 16 | int8 Triton (vLLM) | 46.0 | 45.9 | 5.71 | 40.2 |
| 16 | ours, JIT dispatch | 38.2 | 38.6 | 3.86 | 34.7 |
| 16 | ours, cached launch | 28.2 | 28.9 | 4.20 | 24.7 |
| 16 | ours, custom op | 60.9 | 61.2 | 4.20 | 57.0 |

## gate_up_proj: K=2048, N=12288

| M | provider | host | back-to-back | graph | back-to-back minus graph |
|---|---|---|---|---|---|
| 1 | bf16 F.linear (cuBLAS) | 10.1 | 55.1 | 18.8 | 36.4 |
| 1 | int8 CUTLASS (vLLM) | 24.2 | 24.3 | 9.77 | 14.5 |
| 1 | int8 Triton (vLLM) | 45.7 | 46.1 | 9.62 | 36.5 |
| 1 | ours, JIT dispatch | 38.7 | 38.8 | 6.96 | 31.9 |
| 1 | ours, cached launch | 28.7 | 29.3 | 6.96 | 22.3 |
| 1 | ours, custom op | 61.0 | 61.1 | 7.55 | 53.6 |
| 4 | bf16 F.linear (cuBLAS) | 10.8 | 21.8 | 19.9 | 1.88 |
| 4 | int8 CUTLASS (vLLM) | 24.2 | 24.3 | 9.95 | 14.3 |
| 4 | int8 Triton (vLLM) | 45.8 | 46.1 | 9.79 | 36.3 |
| 4 | ours, JIT dispatch | 38.2 | 38.4 | 7.07 | 31.3 |
| 4 | ours, cached launch | 28.5 | 29.4 | 7.07 | 22.4 |
| 4 | ours, custom op | 61.1 | 61.0 | 7.67 | 53.3 |
| 16 | bf16 F.linear (cuBLAS) | 10.7 | 24.0 | 23.2 | 0.78 |
| 16 | int8 CUTLASS (vLLM) | 24.3 | 24.2 | 10.5 | 13.8 |
| 16 | int8 Triton (vLLM) | 45.4 | 45.8 | 9.94 | 35.9 |
| 16 | ours, JIT dispatch | 37.7 | 38.0 | 8.45 | 29.5 |
| 16 | ours, cached launch | 28.2 | 28.3 | 8.45 | 19.9 |
| 16 | ours, custom op | 60.6 | 60.8 | 9.16 | 51.7 |

Other processes on the GPU during the run: none.
