# W8A8 int8 linear layers on NVIDIA GeForce RTX 4090 (sm_89)

torch 2.13.0+cu130, Triton 3.7.1, vLLM 0.28.0, driver 580.178.04, VLLM_TRITON_USE_TD=unset, #45126 on its tuned table for sm_89.

Microseconds per layer, bf16 output, no bias, CUDA-graph replay, median of 5 windows. Every number passed a float64 check at its own shape first. A speedup is the second provider's time over the first's, so above 1.00x the first is faster. "ours" in the speedup columns is the tuned kernel when this run tuned, and the stored table otherwise.

## q_proj, o_proj: K=2048, N=2048

| M | bf16 | CUTLASS | vLLM Triton | #45126 | _int_mm | ours default | ours tuned | ours table | ours over CUTLASS | ours over vLLM Triton | ours over #45126 | best int8 over bf16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6.31 | 4.87 | 6.41 | 6.16 | n/a | 5.42 | n/a | 3.93 | 1.24x | 1.63x | 1.57x | 1.61x |
| 4 | 9.50 | 4.90 | 6.43 | 6.24 | n/a | 5.44 | n/a | 3.93 | 1.25x | 1.64x | 1.59x | 2.42x |
| 16 | 10.2 | 5.05 | 6.26 | 6.38 | n/a | 5.73 | n/a | 4.27 | 1.18x | 1.47x | 1.49x | 2.38x |
| 64 | 8.21 | 10.1 | 6.62 | 6.98 | 14.6 | 11.3 | n/a | 5.45 | 1.85x | 1.21x | 1.28x | 1.51x |
| 256 | 18.4 | 12.4 | 22.9 | 9.62 | 18.7 | 11.8 | n/a | 9.63 | 1.29x | 2.38x | 1.00x | 1.92x |
| 1024 | 56.8 | 31.0 | 25.6 | 38.9 | 48.3 | 21.2 | n/a | 20.4 | 1.52x | 1.26x | 1.91x | 2.79x |
| 4096 | 198.7 | 106.3 | 87.7 | 111.8 | 202.3 | 62.7 | n/a | 62.3 | 1.71x | 1.41x | 1.80x | 3.19x |

With per-token activation quantization inside the timing:

| M | bf16 | CUTLASS + quant | ours + quant | CUTLASS + quant over bf16 | ours + quant over bf16 |
|---|---|---|---|---|---|
| 1 | 6.31 | 6.18 | 5.23 | 1.02x | 1.21x |
| 4 | 9.50 | 6.20 | 5.27 | 1.53x | 1.80x |
| 16 | 10.2 | 6.28 | 5.60 | 1.62x | 1.82x |
| 64 | 8.21 | 11.6 | 6.74 | 0.71x | 1.22x |
| 256 | 18.4 | 14.1 | 11.5 | 1.31x | 1.60x |
| 1024 | 56.8 | 34.0 | 23.4 | 1.67x | 2.42x |
| 4096 | 198.7 | 115.7 | 69.4 | 1.72x | 2.86x |

## k_proj, v_proj: K=2048, N=1024

| M | bf16 | CUTLASS | vLLM Triton | #45126 | _int_mm | ours default | ours tuned | ours table | ours over CUTLASS | ours over vLLM Triton | ours over #45126 | best int8 over bf16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4.72 | 4.40 | 5.84 | 5.57 | n/a | 5.00 | n/a | 3.52 | 1.25x | 1.66x | 1.58x | 1.34x |
| 4 | 8.59 | 4.43 | 5.89 | 5.71 | n/a | 5.01 | n/a | 3.56 | 1.25x | 1.65x | 1.60x | 2.41x |
| 16 | 8.97 | 4.47 | 5.64 | 5.74 | n/a | 5.18 | n/a | 3.76 | 1.19x | 1.50x | 1.53x | 2.39x |
| 64 | 6.06 | 8.87 | 6.08 | 6.39 | 13.1 | 10.4 | n/a | 4.63 | 1.92x | 1.31x | 1.38x | 1.31x |
| 256 | 11.4 | 10.3 | 20.9 | 6.20 | 14.6 | 10.7 | n/a | 6.30 | 1.63x | 3.32x | 0.98x | 1.84x |
| 1024 | 28.1 | 22.1 | 21.7 | 16.0 | 25.6 | 17.8 | n/a | 13.3 | 1.66x | 1.63x | 1.20x | 2.11x |
| 4096 | 101.8 | 55.5 | 45.3 | 58.6 | 84.2 | 33.2 | n/a | 31.5 | 1.76x | 1.44x | 1.86x | 3.23x |

With per-token activation quantization inside the timing:

| M | bf16 | CUTLASS + quant | ours + quant | CUTLASS + quant over bf16 | ours + quant over bf16 |
|---|---|---|---|---|---|
| 1 | 4.72 | 5.61 | 4.74 | 0.84x | 0.99x |
| 4 | 8.59 | 5.59 | 4.78 | 1.54x | 1.80x |
| 16 | 8.97 | 5.69 | 4.96 | 1.58x | 1.81x |
| 64 | 6.06 | 10.2 | 5.94 | 0.60x | 1.02x |
| 256 | 11.4 | 11.9 | 7.83 | 0.96x | 1.46x |
| 1024 | 28.1 | 24.8 | 15.8 | 1.13x | 1.78x |
| 4096 | 101.8 | 63.1 | 38.1 | 1.61x | 2.67x |

## gate_proj, up_proj: K=2048, N=6144

| M | bf16 | CUTLASS | vLLM Triton | #45126 | _int_mm | ours default | ours tuned | ours table | ours over CUTLASS | ours over vLLM Triton | ours over #45126 | best int8 over bf16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 11.5 | 6.15 | 6.09 | 5.84 | n/a | 5.23 | n/a | 4.52 | 1.36x | 1.35x | 1.29x | 2.55x |
| 4 | 10.2 | 6.24 | 6.23 | 6.03 | n/a | 5.35 | n/a | 4.86 | 1.28x | 1.28x | 1.24x | 2.11x |
| 16 | 12.1 | 6.85 | 6.17 | 6.26 | n/a | 5.85 | n/a | 5.43 | 1.26x | 1.14x | 1.15x | 2.24x |
| 64 | 15.4 | 16.5 | 8.07 | 7.76 | 15.7 | 10.6 | n/a | 7.78 | 2.12x | 1.04x | 1.00x | 1.99x |
| 256 | 48.9 | 24.5 | 22.3 | 22.5 | 37.4 | 19.0 | n/a | 17.6 | 1.39x | 1.27x | 1.28x | 2.79x |
| 1024 | 154.0 | 104.6 | 66.0 | 83.6 | 139.7 | 54.9 | n/a | 48.1 | 2.17x | 1.37x | 1.74x | 3.20x |
| 4096 | 604.6 | 403.9 | 258.0 | 318.3 | 1026.3 | 180.4 | n/a | 179.4 | 2.25x | 1.44x | 1.77x | 3.37x |

With per-token activation quantization inside the timing:

| M | bf16 | CUTLASS + quant | ours + quant | CUTLASS + quant over bf16 | ours + quant over bf16 |
|---|---|---|---|---|---|
| 1 | 11.5 | 7.28 | 5.84 | 1.58x | 1.97x |
| 4 | 10.2 | 7.42 | 5.79 | 1.38x | 1.77x |
| 16 | 12.1 | 8.01 | 6.78 | 1.52x | 1.79x |
| 64 | 15.4 | 17.8 | 9.03 | 0.86x | 1.71x |
| 256 | 48.9 | 25.8 | 19.0 | 1.89x | 2.57x |
| 1024 | 154.0 | 110.7 | 51.0 | 1.39x | 3.02x |
| 4096 | 604.6 | 433.4 | 206.6 | 1.40x | 2.93x |

## down_proj: K=6144, N=2048

| M | bf16 | CUTLASS | vLLM Triton | #45126 | _int_mm | ours default | ours tuned | ours table | ours over CUTLASS | ours over vLLM Triton | ours over #45126 | best int8 over bf16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 13.0 | 9.08 | 14.4 | 13.6 | n/a | 12.2 | n/a | 7.59 | 1.20x | 1.90x | 1.79x | 1.71x |
| 4 | 23.5 | 9.09 | 14.5 | 13.8 | n/a | 12.2 | n/a | 7.64 | 1.19x | 1.89x | 1.81x | 3.07x |
| 16 | 24.5 | 9.21 | 13.7 | 13.9 | n/a | 12.7 | n/a | 8.23 | 1.12x | 1.66x | 1.69x | 2.98x |
| 64 | 14.5 | 16.4 | 14.0 | 14.7 | 25.8 | 27.7 | n/a | 11.7 | 1.40x | 1.20x | 1.26x | 1.24x |
| 256 | 46.7 | 24.6 | 57.1 | 23.3 | 29.7 | 28.1 | n/a | 23.3 | 1.05x | 2.45x | 1.00x | 2.01x |
| 1024 | 158.1 | 57.3 | 59.6 | 105.1 | 86.4 | 48.6 | n/a | 46.7 | 1.23x | 1.28x | 2.25x | 3.39x |
| 4096 | 605.6 | 213.0 | 233.9 | 328.9 | 302.7 | 172.9 | n/a | 175.5 | 1.21x | 1.33x | 1.87x | 3.50x |

With per-token activation quantization inside the timing:

| M | bf16 | CUTLASS + quant | ours + quant | CUTLASS + quant over bf16 | ours + quant over bf16 |
|---|---|---|---|---|---|
| 1 | 13.0 | 10.7 | 9.17 | 1.21x | 1.41x |
| 4 | 23.5 | 10.7 | 9.24 | 2.19x | 2.54x |
| 16 | 24.5 | 10.9 | 9.81 | 2.26x | 2.50x |
| 64 | 14.5 | 18.1 | 13.2 | 0.80x | 1.10x |
| 256 | 46.7 | 26.7 | 25.2 | 1.75x | 1.86x |
| 1024 | 158.1 | 62.0 | 53.0 | 2.55x | 2.98x |
| 4096 | 605.6 | 311.7 | 277.3 | 1.94x | 2.18x |

## Not timed

- q_proj, o_proj, M=1, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 1
- q_proj, o_proj, M=4, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 4
- q_proj, o_proj, M=16, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 16
- k_proj, v_proj, M=1, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 1
- k_proj, v_proj, M=4, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 4
- k_proj, v_proj, M=16, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 16
- gate_proj, up_proj, M=1, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 1
- gate_proj, up_proj, M=4, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 4
- gate_proj, up_proj, M=16, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 16
- down_proj, M=1, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 1
- down_proj, M=4, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 4
- down_proj, M=16, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 16

Tuning sweep: 0 configurations, 0 failed or errored.

Other processes on the GPU during the run: none.
