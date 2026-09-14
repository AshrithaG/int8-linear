# W8A8 int8 linear layers on NVIDIA GeForce RTX 4090 (sm_89)

torch 2.13.0+cu130, Triton 3.7.1, vLLM 0.28.0, driver 580.178.04, VLLM_TRITON_USE_TD=unset.

Microseconds per layer, bf16 output, no bias, CUDA-graph replay, median of 5 windows. Every number passed a float64 check at its own shape first. A speedup is the second provider's time over the first's, so above 1.00x the first is faster. "ours" in the speedup columns is the tuned kernel when this run tuned, and the stored table otherwise.

## q_proj, o_proj: K=2048, N=2048

| M | bf16 | CUTLASS | vLLM Triton | _int_mm | ours default | ours tuned | ours table | ours over CUTLASS | ours over vLLM Triton | best int8 over bf16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6.30 | 4.90 | 6.41 | n/a | 5.42 | n/a | 3.92 | 1.25x | 1.63x | 1.61x |
| 4 | 9.50 | 4.89 | 6.43 | n/a | 5.44 | n/a | 3.93 | 1.25x | 1.64x | 2.42x |
| 16 | 10.2 | 5.00 | 6.25 | n/a | 5.73 | n/a | 4.26 | 1.17x | 1.47x | 2.39x |
| 64 | 8.23 | 10.1 | 6.62 | 14.6 | 11.3 | n/a | 5.43 | 1.86x | 1.22x | 1.52x |
| 256 | 18.3 | 12.4 | 22.9 | 18.6 | 11.8 | n/a | 9.63 | 1.29x | 2.38x | 1.90x |
| 1024 | 56.8 | 31.0 | 25.5 | 48.5 | 21.1 | n/a | 20.4 | 1.52x | 1.25x | 2.79x |
| 4096 | 198.9 | 105.7 | 87.4 | 202.3 | 62.1 | n/a | 62.1 | 1.70x | 1.41x | 3.20x |

With per-token activation quantization inside the timing:

| M | bf16 | CUTLASS + quant | ours + quant | CUTLASS + quant over bf16 | ours + quant over bf16 |
|---|---|---|---|---|---|
| 1 | 6.30 | 6.20 | 5.23 | 1.02x | 1.20x |
| 4 | 9.50 | 6.24 | 5.27 | 1.52x | 1.80x |
| 16 | 10.2 | 6.29 | 5.60 | 1.62x | 1.82x |
| 64 | 8.23 | 11.6 | 6.74 | 0.71x | 1.22x |
| 256 | 18.3 | 14.1 | 11.5 | 1.30x | 1.59x |
| 1024 | 56.8 | 33.9 | 23.2 | 1.68x | 2.44x |
| 4096 | 198.9 | 115.5 | 69.0 | 1.72x | 2.88x |

## k_proj, v_proj: K=2048, N=1024

| M | bf16 | CUTLASS | vLLM Triton | _int_mm | ours default | ours tuned | ours table | ours over CUTLASS | ours over vLLM Triton | best int8 over bf16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4.72 | 4.36 | 5.82 | n/a | 5.00 | n/a | 3.50 | 1.25x | 1.66x | 1.35x |
| 4 | 8.59 | 4.41 | 5.86 | n/a | 4.99 | n/a | 3.55 | 1.24x | 1.65x | 2.42x |
| 16 | 8.95 | 4.52 | 5.61 | n/a | 5.17 | n/a | 3.74 | 1.21x | 1.50x | 2.39x |
| 64 | 6.04 | 8.83 | 6.04 | 13.1 | 10.3 | n/a | 4.59 | 1.92x | 1.32x | 1.32x |
| 256 | 11.4 | 10.2 | 20.8 | 14.6 | 10.6 | n/a | 6.25 | 1.64x | 3.34x | 1.83x |
| 1024 | 28.0 | 22.0 | 21.7 | 25.5 | 17.9 | n/a | 13.1 | 1.68x | 1.66x | 2.14x |
| 4096 | 101.4 | 55.1 | 45.3 | 84.0 | 33.0 | n/a | 31.9 | 1.73x | 1.42x | 3.18x |

With per-token activation quantization inside the timing:

| M | bf16 | CUTLASS + quant | ours + quant | CUTLASS + quant over bf16 | ours + quant over bf16 |
|---|---|---|---|---|---|
| 1 | 4.72 | 5.59 | 4.72 | 0.84x | 1.00x |
| 4 | 8.59 | 5.60 | 4.76 | 1.53x | 1.81x |
| 16 | 8.95 | 5.66 | 4.94 | 1.58x | 1.81x |
| 64 | 6.04 | 10.1 | 5.90 | 0.60x | 1.02x |
| 256 | 11.4 | 11.9 | 7.78 | 0.96x | 1.47x |
| 1024 | 28.0 | 24.6 | 15.8 | 1.14x | 1.78x |
| 4096 | 101.4 | 62.9 | 38.1 | 1.61x | 2.66x |

## gate_proj, up_proj: K=2048, N=6144

| M | bf16 | CUTLASS | vLLM Triton | _int_mm | ours default | ours tuned | ours table | ours over CUTLASS | ours over vLLM Triton | best int8 over bf16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 11.2 | 6.18 | 6.08 | n/a | 5.22 | n/a | 4.50 | 1.37x | 1.35x | 2.48x |
| 4 | 10.2 | 6.23 | 6.24 | n/a | 5.29 | n/a | 4.82 | 1.29x | 1.29x | 2.11x |
| 16 | 12.2 | 6.91 | 6.21 | n/a | 5.80 | n/a | 5.44 | 1.27x | 1.14x | 2.24x |
| 64 | 15.3 | 16.5 | 7.86 | 15.6 | 10.5 | n/a | 7.66 | 2.16x | 1.03x | 2.00x |
| 256 | 48.9 | 24.4 | 22.2 | 37.1 | 18.9 | n/a | 17.5 | 1.39x | 1.27x | 2.80x |
| 1024 | 153.7 | 104.8 | 66.0 | 139.8 | 58.0 | n/a | 48.3 | 2.17x | 1.37x | 3.18x |
| 4096 | 604.4 | 403.5 | 257.2 | 1027.3 | 180.4 | n/a | 180.0 | 2.24x | 1.43x | 3.36x |

With per-token activation quantization inside the timing:

| M | bf16 | CUTLASS + quant | ours + quant | CUTLASS + quant over bf16 | ours + quant over bf16 |
|---|---|---|---|---|---|
| 1 | 11.2 | 7.22 | 5.82 | 1.54x | 1.92x |
| 4 | 10.2 | 7.42 | 5.79 | 1.37x | 1.75x |
| 16 | 12.2 | 7.90 | 6.79 | 1.55x | 1.80x |
| 64 | 15.3 | 17.8 | 8.99 | 0.86x | 1.71x |
| 256 | 48.9 | 25.9 | 19.1 | 1.89x | 2.56x |
| 1024 | 153.7 | 111.7 | 51.0 | 1.38x | 3.02x |
| 4096 | 604.4 | 436.0 | 206.2 | 1.39x | 2.93x |

## down_proj: K=6144, N=2048

| M | bf16 | CUTLASS | vLLM Triton | _int_mm | ours default | ours tuned | ours table | ours over CUTLASS | ours over vLLM Triton | best int8 over bf16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 13.0 | 9.07 | 14.4 | n/a | 12.2 | n/a | 7.60 | 1.19x | 1.90x | 1.71x |
| 4 | 23.4 | 9.10 | 14.4 | n/a | 12.3 | n/a | 7.63 | 1.19x | 1.89x | 3.07x |
| 16 | 24.5 | 9.22 | 13.7 | n/a | 12.6 | n/a | 8.09 | 1.14x | 1.69x | 3.03x |
| 64 | 14.4 | 16.4 | 14.0 | 25.9 | 27.8 | n/a | 11.5 | 1.42x | 1.22x | 1.25x |
| 256 | 47.7 | 24.5 | 57.1 | 29.5 | 28.1 | n/a | 23.3 | 1.05x | 2.45x | 2.04x |
| 1024 | 159.0 | 56.5 | 59.5 | 86.9 | 48.4 | n/a | 46.6 | 1.21x | 1.28x | 3.41x |
| 4096 | 610.1 | 217.3 | 233.7 | 303.9 | 179.4 | n/a | 176.7 | 1.23x | 1.32x | 3.45x |

With per-token activation quantization inside the timing:

| M | bf16 | CUTLASS + quant | ours + quant | CUTLASS + quant over bf16 | ours + quant over bf16 |
|---|---|---|---|---|---|
| 1 | 13.0 | 10.6 | 9.18 | 1.22x | 1.41x |
| 4 | 23.4 | 10.7 | 9.24 | 2.19x | 2.54x |
| 16 | 24.5 | 10.8 | 9.74 | 2.26x | 2.51x |
| 64 | 14.4 | 18.1 | 13.3 | 0.80x | 1.09x |
| 256 | 47.7 | 26.8 | 26.0 | 1.78x | 1.84x |
| 1024 | 159.0 | 62.2 | 52.9 | 2.56x | 3.00x |
| 4096 | 610.1 | 310.4 | 257.6 | 1.97x | 2.37x |

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
