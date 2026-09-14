# W8A8 int8 linear layers on NVIDIA GeForce RTX 4090 (sm_89)

torch 2.13.0+cu130, Triton 3.7.1, vLLM 0.28.0, driver 580.178.04, VLLM_TRITON_USE_TD=unset, #45126 on its tuned table for sm_89.

Microseconds per layer, bf16 output, no bias, CUDA-graph replay, median of 5 windows. Every number passed a float64 check at its own shape first. A speedup is the second provider's time over the first's, so above 1.00x the first is faster. "ours" in the speedup columns is the tuned kernel when this run tuned, and the stored table otherwise.

## qkv_proj: K=2048, N=4096

| M | bf16 | CUTLASS | vLLM Triton | #45126 | _int_mm | ours default | ours tuned | ours table | ours over CUTLASS | ours over vLLM Triton | ours over #45126 | best int8 over bf16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 9.58 | 5.47 | 6.58 | 6.34 | n/a | 5.52 | 4.08 | 5.53 | 1.34x | 1.61x | 1.55x | 2.35x |
| 4 | 10.3 | 5.51 | 6.57 | 6.36 | n/a | 5.54 | 4.20 | 5.54 | 1.31x | 1.56x | 1.52x | 2.44x |
| 16 | 11.9 | 5.66 | 6.30 | 6.44 | n/a | 5.81 | 4.73 | 5.81 | 1.20x | 1.33x | 1.36x | 2.51x |
| 64 | 13.0 | 12.9 | 7.57 | 7.60 | 15.8 | 11.4 | 7.51 | 11.4 | 1.72x | 1.01x | 1.01x | 1.73x |
| 256 | 30.6 | 17.7 | 23.5 | 17.2 | 27.5 | 13.7 | 13.6 | 13.7 | 1.30x | 1.73x | 1.26x | 2.25x |
| 1024 | 101.5 | 52.3 | 44.9 | 58.0 | 83.4 | 32.4 | 31.1 | 32.6 | 1.68x | 1.44x | 1.86x | 3.26x |
| 4096 | 405.9 | 197.6 | 173.5 | 212.8 | 644.1 | 120.2 | 120.4 | 120.2 | 1.64x | 1.44x | 1.77x | 3.38x |

With per-token activation quantization inside the timing:

| M | bf16 | CUTLASS + quant | ours + quant | CUTLASS + quant over bf16 | ours + quant over bf16 |
|---|---|---|---|---|---|
| 1 | 9.58 | 6.79 | 5.36 | 1.41x | 1.79x |
| 4 | 10.3 | 6.81 | 5.48 | 1.50x | 1.87x |
| 16 | 11.9 | 6.96 | 6.37 | 1.71x | 1.86x |
| 64 | 13.0 | 13.9 | 8.09 | 0.94x | 1.61x |
| 256 | 30.6 | 19.4 | 15.4 | 1.58x | 1.99x |
| 1024 | 101.5 | 55.2 | 33.9 | 1.84x | 2.99x |
| 4096 | 405.9 | 217.7 | 127.8 | 1.86x | 3.18x |

Tuned configurations:

| M | config | microseconds |
|---|---|---|
| 1 | 16x64x256 w4 s3 g1 | 4.08 |
| 4 | 16x64x256 w4 s3 g1 | 4.19 |
| 16 | 16x64x256 w4 s3 g1 | 4.73 |
| 64 | 64x64x128 w4 s3 g1 | 7.52 |
| 256 | 64x128x64 w4 s3 g8 | 13.8 |
| 1024 | 128x128x64 w4 s1 g8 | 31.1 |
| 4096 | 128x128x64 w4 s3 g8 | 120.0 |

## o_proj: K=2048, N=2048

| M | bf16 | CUTLASS | vLLM Triton | #45126 | _int_mm | ours default | ours tuned | ours table | ours over CUTLASS | ours over vLLM Triton | ours over #45126 | best int8 over bf16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5.70 | 4.46 | 5.89 | 5.63 | n/a | 5.00 | 3.57 | 3.57 | 1.25x | 1.65x | 1.58x | 1.60x |
| 4 | 8.81 | 4.50 | 5.94 | 5.75 | n/a | 5.02 | 3.62 | 3.62 | 1.24x | 1.64x | 1.59x | 2.43x |
| 16 | 9.66 | 4.58 | 5.75 | 5.87 | n/a | 5.23 | 3.88 | 3.87 | 1.18x | 1.48x | 1.51x | 2.49x |
| 64 | 7.64 | 9.34 | 6.12 | 6.47 | 13.5 | 10.4 | 4.89 | 4.87 | 1.91x | 1.25x | 1.32x | 1.57x |
| 256 | 16.7 | 11.4 | 21.1 | 8.91 | 17.0 | 10.8 | 8.86 | 8.86 | 1.28x | 2.38x | 1.01x | 1.88x |
| 1024 | 52.2 | 26.9 | 23.2 | 36.2 | 44.6 | 19.0 | 18.4 | 18.4 | 1.46x | 1.26x | 1.97x | 2.83x |
| 4096 | 199.5 | 102.0 | 88.1 | 111.2 | 204.2 | 62.1 | 62.5 | 62.1 | 1.63x | 1.41x | 1.78x | 3.21x |

With per-token activation quantization inside the timing:

| M | bf16 | CUTLASS + quant | ours + quant | CUTLASS + quant over bf16 | ours + quant over bf16 |
|---|---|---|---|---|---|
| 1 | 5.70 | 5.68 | 4.80 | 1.00x | 1.19x |
| 4 | 8.81 | 5.74 | 4.84 | 1.54x | 1.82x |
| 16 | 9.66 | 5.82 | 5.07 | 1.66x | 1.91x |
| 64 | 7.64 | 10.6 | 6.27 | 0.72x | 1.22x |
| 256 | 16.7 | 13.0 | 10.5 | 1.29x | 1.58x |
| 1024 | 52.2 | 29.7 | 21.1 | 1.76x | 2.48x |
| 4096 | 199.5 | 111.8 | 69.2 | 1.78x | 2.88x |

Tuned configurations:

| M | config | microseconds |
|---|---|---|
| 1 | 16x64x256 w4 s3 g1 | 3.59 |
| 4 | 16x64x256 w4 s3 g1 | 3.62 |
| 16 | 16x64x256 w4 s3 g1 | 3.87 |
| 64 | 32x64x128 w4 s3 g1 | 4.87 |
| 256 | 64x64x128 w4 s3 g8 | 8.90 |
| 1024 | 128x128x64 w8 s3 g8 | 18.4 |
| 4096 | 128x128x64 w4 s3 g8 | 62.2 |

## gate_up_proj: K=2048, N=12288

| M | bf16 | CUTLASS | vLLM Triton | #45126 | _int_mm | ours default | ours tuned | ours table | ours over CUTLASS | ours over vLLM Triton | ours over #45126 | best int8 over bf16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 18.5 | 9.41 | 9.57 | 11.0 | n/a | 6.90 | 6.88 | 6.89 | 1.37x | 1.39x | 1.60x | 2.68x |
| 4 | 19.8 | 9.55 | 9.85 | 11.2 | n/a | 7.08 | 7.06 | 7.07 | 1.35x | 1.40x | 1.59x | 2.81x |
| 16 | 24.2 | 10.2 | 9.92 | 13.0 | n/a | 8.20 | 8.21 | 8.20 | 1.24x | 1.21x | 1.58x | 2.95x |
| 64 | 27.5 | 14.1 | 16.5 | 17.4 | 21.7 | 11.6 | 11.2 | 11.6 | 1.26x | 1.47x | 1.55x | 2.45x |
| 256 | 79.1 | 43.8 | 43.0 | 43.5 | 64.4 | 33.6 | 28.3 | 33.6 | 1.55x | 1.52x | 1.54x | 2.80x |
| 1024 | 297.6 | 150.3 | 130.0 | 125.0 | 426.7 | 91.8 | 91.6 | 91.6 | 1.64x | 1.42x | 1.36x | 3.25x |
| 4096 | 1214.1 | 564.8 | 510.6 | 485.0 | 2070.7 | 354.9 | 356.1 | 354.9 | 1.59x | 1.43x | 1.36x | 3.42x |

With per-token activation quantization inside the timing:

| M | bf16 | CUTLASS + quant | ours + quant | CUTLASS + quant over bf16 | ours + quant over bf16 |
|---|---|---|---|---|---|
| 1 | 18.5 | 10.7 | 8.23 | 1.73x | 2.24x |
| 4 | 19.8 | 10.7 | 8.28 | 1.85x | 2.39x |
| 16 | 24.2 | 11.4 | 9.44 | 2.12x | 2.56x |
| 64 | 27.5 | 14.5 | 12.0 | 1.89x | 2.29x |
| 256 | 79.1 | 46.6 | 30.4 | 1.70x | 2.60x |
| 1024 | 297.6 | 156.0 | 94.4 | 1.91x | 3.15x |
| 4096 | 1214.1 | 592.9 | 387.9 | 2.05x | 3.13x |

Tuned configurations:

| M | config | microseconds |
|---|---|---|
| 1 | 16x128x128 w4 s3 g1 | 6.91 |
| 4 | 16x128x128 w4 s3 g1 | 7.07 |
| 16 | 16x128x128 w4 s3 g1 | 8.21 |
| 64 | 64x128x128 w4 s3 g1 | 11.2 |
| 256 | 64x128x64 w4 s1 g8 | 28.1 |
| 1024 | 128x128x64 w4 s3 g8 | 91.9 |
| 4096 | 128x128x64 w4 s3 g8 | 355.5 |

## down_proj: K=6144, N=2048

| M | bf16 | CUTLASS | vLLM Triton | #45126 | _int_mm | ours default | ours tuned | ours table | ours over CUTLASS | ours over vLLM Triton | ours over #45126 | best int8 over bf16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 13.8 | 9.13 | 14.4 | 13.6 | n/a | 12.2 | 7.59 | 7.59 | 1.20x | 1.90x | 1.79x | 1.82x |
| 4 | 23.5 | 9.10 | 14.5 | 13.8 | n/a | 12.3 | 7.65 | 7.65 | 1.19x | 1.89x | 1.80x | 3.08x |
| 16 | 25.0 | 9.24 | 13.7 | 13.9 | n/a | 12.7 | 8.04 | 8.04 | 1.15x | 1.70x | 1.72x | 3.11x |
| 64 | 14.5 | 15.9 | 14.0 | 14.7 | 25.9 | 27.7 | 11.6 | 11.6 | 1.37x | 1.21x | 1.27x | 1.26x |
| 256 | 48.1 | 24.2 | 57.2 | 23.7 | 29.5 | 28.2 | 23.7 | 23.8 | 1.02x | 2.41x | 1.00x | 2.03x |
| 1024 | 159.4 | 56.0 | 59.8 | 104.9 | 86.7 | 48.7 | 46.8 | 46.8 | 1.20x | 1.28x | 2.24x | 3.41x |
| 4096 | 607.8 | 210.9 | 234.3 | 330.8 | 302.9 | 175.3 | 174.5 | 175.3 | 1.21x | 1.34x | 1.90x | 3.48x |

With per-token activation quantization inside the timing:

| M | bf16 | CUTLASS + quant | ours + quant | CUTLASS + quant over bf16 | ours + quant over bf16 |
|---|---|---|---|---|---|
| 1 | 13.8 | 10.7 | 9.19 | 1.29x | 1.50x |
| 4 | 23.5 | 10.7 | 9.24 | 2.20x | 2.55x |
| 16 | 25.0 | 10.8 | 9.71 | 2.31x | 2.58x |
| 64 | 14.5 | 17.6 | 13.6 | 0.82x | 1.06x |
| 256 | 48.1 | 26.4 | 26.7 | 1.82x | 1.80x |
| 1024 | 159.4 | 60.9 | 52.7 | 2.62x | 3.02x |
| 4096 | 607.8 | 307.0 | 275.0 | 1.98x | 2.21x |

Tuned configurations:

| M | config | microseconds |
|---|---|---|
| 1 | 16x64x256 w4 s3 g1 | 7.60 |
| 4 | 16x64x256 w4 s3 g1 | 7.65 |
| 16 | 16x64x256 w4 s3 g1 | 8.04 |
| 64 | 32x64x256 w4 s3 g1 | 11.6 |
| 256 | 64x64x256 w4 s3 g8 | 23.8 |
| 1024 | 128x128x128 w4 s3 g8 | 46.9 |
| 4096 | 128x128x64 w4 s1 g8 | 174.7 |

## Not timed

- qkv_proj, M=1, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 1
- qkv_proj, M=4, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 4
- qkv_proj, M=16, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 16
- o_proj, M=1, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 1
- o_proj, M=4, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 4
- o_proj, M=16, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 16
- gate_up_proj, M=1, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 1
- gate_up_proj, M=4, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 4
- gate_up_proj, M=16, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 16
- down_proj, M=1, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 1
- down_proj, M=4, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 4
- down_proj, M=16, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 16

Tuning sweep: 820 configurations, 0 failed or errored.

Other processes on the GPU during the run: none.
