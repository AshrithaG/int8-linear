# Launch cost of one int8 linear layer on NVIDIA GeForce RTX 4090

torch 2.13.0+cu130, Triton 3.7.1, vLLM 0.28.0, launch cache on. Microseconds per call, median of 7 windows, no bias. Host is CPU time with nothing synchronized inside the window; back-to-back is time between CUDA events; graph is CUDA-graph replay.

## q_proj, o_proj: K=2048, N=2048

| M | provider | host | back-to-back | graph | back-to-back minus graph |
|---|---|---|---|---|---|
| 1 | bf16 F.linear (cuBLAS) | 10.5 | 10.5 | 6.31 | 4.20 |
| 1 | int8 CUTLASS (vLLM) | 24.3 | 24.4 | 4.49 | 19.9 |
| 1 | int8 Triton (vLLM) | 45.3 | 45.4 | 5.89 | 39.5 |
| 1 | ours, JIT dispatch | 36.7 | 36.8 | 3.93 | 32.9 |
| 1 | ours, cached launch | 27.4 | 27.5 | 3.93 | 23.6 |
| 1 | ours, custom op | 57.9 | 57.9 | 3.93 | 54.0 |
| 1 | ours, vLLM entry | 29.5 | 29.4 | 3.93 | 25.5 |
| 4 | bf16 F.linear (cuBLAS) | 11.1 | 11.1 | 8.64 | 2.49 |
| 4 | int8 CUTLASS (vLLM) | 24.0 | 24.0 | 4.47 | 19.5 |
| 4 | int8 Triton (vLLM) | 43.7 | 43.8 | 5.93 | 37.8 |
| 4 | ours, JIT dispatch | 35.3 | 35.4 | 3.62 | 31.8 |
| 4 | ours, cached launch | 27.0 | 26.9 | 3.96 | 23.0 |
| 4 | ours, custom op | 57.9 | 57.9 | 3.95 | 53.9 |
| 4 | ours, vLLM entry | 29.8 | 29.9 | 3.95 | 25.9 |
| 16 | bf16 F.linear (cuBLAS) | 11.2 | 11.3 | 10.3 | 0.91 |
| 16 | int8 CUTLASS (vLLM) | 23.1 | 23.2 | 4.54 | 18.7 |
| 16 | int8 Triton (vLLM) | 44.6 | 45.0 | 5.71 | 39.3 |
| 16 | ours, JIT dispatch | 36.6 | 36.5 | 3.86 | 32.6 |
| 16 | ours, cached launch | 26.5 | 26.6 | 4.22 | 22.4 |
| 16 | ours, custom op | 57.6 | 57.8 | 4.20 | 53.6 |
| 16 | ours, vLLM entry | 29.7 | 29.8 | 4.20 | 25.6 |

## gate_up_proj: K=2048, N=12288

| M | provider | host | back-to-back | graph | back-to-back minus graph |
|---|---|---|---|---|---|
| 1 | bf16 F.linear (cuBLAS) | 10.8 | 55.1 | 18.8 | 36.4 |
| 1 | int8 CUTLASS (vLLM) | 23.3 | 23.2 | 9.78 | 13.4 |
| 1 | int8 Triton (vLLM) | 44.8 | 44.9 | 9.63 | 35.3 |
| 1 | ours, JIT dispatch | 36.8 | 36.6 | 6.96 | 29.7 |
| 1 | ours, cached launch | 26.8 | 27.1 | 6.96 | 20.1 |
| 1 | ours, custom op | 58.2 | 58.2 | 7.55 | 50.6 |
| 1 | ours, vLLM entry | 29.6 | 30.0 | 7.56 | 22.5 |
| 4 | bf16 F.linear (cuBLAS) | 11.2 | 20.5 | 19.9 | 0.64 |
| 4 | int8 CUTLASS (vLLM) | 24.5 | 24.5 | 9.94 | 14.6 |
| 4 | int8 Triton (vLLM) | 45.2 | 45.4 | 9.79 | 35.6 |
| 4 | ours, JIT dispatch | 37.0 | 36.9 | 7.06 | 29.9 |
| 4 | ours, cached launch | 27.6 | 27.6 | 7.06 | 20.5 |
| 4 | ours, custom op | 58.0 | 58.1 | 7.67 | 50.4 |
| 4 | ours, vLLM entry | 30.7 | 30.9 | 7.66 | 23.2 |
| 16 | bf16 F.linear (cuBLAS) | 11.0 | 24.0 | 23.2 | 0.78 |
| 16 | int8 CUTLASS (vLLM) | 24.3 | 24.5 | 10.5 | 14.0 |
| 16 | int8 Triton (vLLM) | 45.2 | 44.7 | 9.95 | 34.8 |
| 16 | ours, JIT dispatch | 36.5 | 35.8 | 8.44 | 27.3 |
| 16 | ours, cached launch | 27.3 | 27.6 | 8.44 | 19.2 |
| 16 | ours, custom op | 57.7 | 57.7 | 9.17 | 48.6 |
| 16 | ours, vLLM entry | 29.8 | 30.6 | 9.18 | 21.4 |

Other processes on the GPU during the run: none.
