# W8A8 int8 linear layers on NVIDIA GeForce RTX 4090 (sm_89)

torch 2.13.0+cu130, Triton 3.7.1, vLLM 0.28.0, driver 580.178.04, VLLM_TRITON_USE_TD=unset.

Microseconds per layer, bf16 output, no bias, CUDA-graph replay, median of 5 windows. Every number passed a float64 check at its own shape first. A speedup is the second provider's time over the first's, so above 1.00x the first is faster.

## q_proj, o_proj: K=2048, N=2048

| M | bf16 | CUTLASS | vLLM Triton | _int_mm | ours default | ours tuned | ours tuned over CUTLASS | ours tuned over vLLM Triton | best int8 over bf16 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 6.29 | 4.87 | 6.41 | n/a | 5.41 | 3.92 | 1.24x | 1.63x | 1.60x |
| 4 | 9.50 | 4.91 | 6.43 | n/a | 5.44 | 3.93 | 1.25x | 1.64x | 2.42x |
| 16 | 10.1 | 5.02 | 6.25 | n/a | 5.73 | 4.27 | 1.18x | 1.46x | 2.38x |
| 64 | 8.21 | 10.1 | 6.57 | 14.7 | 11.3 | 5.37 | 1.88x | 1.23x | 1.53x |
| 256 | 16.9 | 11.3 | 21.0 | 17.0 | 10.8 | 8.91 | 1.27x | 2.36x | 1.90x |
| 1024 | 52.0 | 26.7 | 23.3 | 44.2 | 19.0 | 18.3 | 1.46x | 1.27x | 2.84x |
| 4096 | 199.5 | 101.6 | 88.1 | 204.0 | 62.7 | 62.3 | 1.63x | 1.41x | 3.20x |

With per-token activation quantization inside the timing:

| M | bf16 | CUTLASS + quant | ours + quant | CUTLASS + quant over bf16 | ours + quant over bf16 |
|---|---|---|---|---|---|
| 1 | 6.29 | 6.19 | 5.22 | 1.02x | 1.20x |
| 4 | 9.50 | 6.23 | 5.26 | 1.53x | 1.80x |
| 16 | 10.1 | 6.30 | 5.59 | 1.61x | 1.82x |
| 64 | 8.21 | 11.6 | 6.76 | 0.71x | 1.22x |
| 256 | 16.9 | 13.0 | 10.6 | 1.30x | 1.59x |
| 1024 | 52.0 | 29.7 | 21.2 | 1.75x | 2.45x |
| 4096 | 199.5 | 111.2 | 69.2 | 1.79x | 2.88x |

Tuned configurations:

| M | config | microseconds |
|---|---|---|
| 1 | 16x64x256 w4 s3 g1 | 3.92 |
| 4 | 16x64x256 w4 s3 g1 | 3.93 |
| 16 | 16x64x256 w4 s3 g1 | 4.26 |
| 64 | 32x64x128 w4 s3 g1 | 5.37 |
| 256 | 64x64x128 w4 s3 g8 | 9.68 |
| 1024 | 128x128x64 w8 s3 g8 | 18.5 |
| 4096 | 128x128x64 w4 s3 g8 | 62.5 |

## k_proj, v_proj: K=2048, N=1024

| M | bf16 | CUTLASS | vLLM Triton | _int_mm | ours default | ours tuned | ours tuned over CUTLASS | ours tuned over vLLM Triton | best int8 over bf16 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4.71 | 4.40 | 5.85 | n/a | 5.00 | 3.53 | 1.25x | 1.66x | 1.34x |
| 4 | 8.58 | 4.45 | 5.90 | n/a | 5.00 | 3.56 | 1.25x | 1.66x | 2.41x |
| 16 | 8.96 | 4.44 | 5.61 | n/a | 5.16 | 3.70 | 1.20x | 1.51x | 2.42x |
| 64 | 6.02 | 8.97 | 6.06 | 13.1 | 10.3 | 4.64 | 1.93x | 1.31x | 1.30x |
| 256 | 11.4 | 10.2 | 20.9 | 14.6 | 10.6 | 6.30 | 1.63x | 3.32x | 1.81x |
| 1024 | 28.1 | 21.5 | 21.7 | 25.4 | 17.9 | 13.0 | 1.65x | 1.67x | 2.16x |
| 4096 | 102.0 | 53.2 | 45.4 | 83.8 | 33.2 | 31.9 | 1.67x | 1.42x | 3.19x |

With per-token activation quantization inside the timing:

| M | bf16 | CUTLASS + quant | ours + quant | CUTLASS + quant over bf16 | ours + quant over bf16 |
|---|---|---|---|---|---|
| 1 | 4.71 | 5.62 | 4.75 | 0.84x | 0.99x |
| 4 | 8.58 | 5.60 | 4.77 | 1.53x | 1.80x |
| 16 | 8.96 | 5.67 | 4.95 | 1.58x | 1.81x |
| 64 | 6.02 | 10.3 | 5.94 | 0.59x | 1.01x |
| 256 | 11.4 | 11.8 | 7.83 | 0.97x | 1.46x |
| 1024 | 28.1 | 24.2 | 15.7 | 1.16x | 1.79x |
| 4096 | 102.0 | 60.2 | 38.5 | 1.69x | 2.65x |

Tuned configurations:

| M | config | microseconds |
|---|---|---|
| 1 | 16x64x256 w4 s3 g1 | 3.53 |
| 4 | 16x64x256 w4 s3 g1 | 3.56 |
| 16 | 16x64x256 w4 s3 g1 | 3.70 |
| 64 | 32x64x128 w4 s3 g1 | 4.63 |
| 256 | 64x64x128 w4 s3 g8 | 6.30 |
| 1024 | 64x128x64 w4 s3 g8 | 13.0 |
| 4096 | 128x128x64 w4 s1 g8 | 32.2 |

## gate_proj, up_proj: K=2048, N=6144

| M | bf16 | CUTLASS | vLLM Triton | _int_mm | ours default | ours tuned | ours tuned over CUTLASS | ours tuned over vLLM Triton | best int8 over bf16 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 11.3 | 5.72 | 6.09 | n/a | 5.23 | 4.49 | 1.28x | 1.36x | 2.52x |
| 4 | 10.2 | 5.90 | 6.30 | n/a | 5.28 | 4.95 | 1.19x | 1.27x | 2.07x |
| 16 | 12.3 | 6.54 | 6.15 | n/a | 5.85 | 5.31 | 1.23x | 1.16x | 2.31x |
| 64 | 15.3 | 15.0 | 7.82 | 15.6 | 10.5 | 7.50 | 2.01x | 1.04x | 2.04x |
| 256 | 49.3 | 23.5 | 22.2 | 37.1 | 19.2 | 17.6 | 1.33x | 1.26x | 2.80x |
| 1024 | 154.2 | 96.6 | 66.6 | 138.7 | 56.8 | 48.5 | 1.99x | 1.37x | 3.18x |
| 4096 | 605.4 | 370.3 | 257.6 | 1028.9 | 180.8 | 180.4 | 2.05x | 1.43x | 3.36x |

With per-token activation quantization inside the timing:

| M | bf16 | CUTLASS + quant | ours + quant | CUTLASS + quant over bf16 | ours + quant over bf16 |
|---|---|---|---|---|---|
| 1 | 11.3 | 6.93 | 5.79 | 1.63x | 1.95x |
| 4 | 10.2 | 7.05 | 5.80 | 1.45x | 1.77x |
| 16 | 12.3 | 7.73 | 6.85 | 1.59x | 1.80x |
| 64 | 15.3 | 16.4 | 9.20 | 0.93x | 1.66x |
| 256 | 49.3 | 25.1 | 19.1 | 1.96x | 2.58x |
| 1024 | 154.2 | 103.2 | 51.2 | 1.49x | 3.01x |
| 4096 | 605.4 | 406.5 | 207.5 | 1.49x | 2.92x |

Tuned configurations:

| M | config | microseconds |
|---|---|---|
| 1 | 16x64x128 w4 s3 g1 | 4.48 |
| 4 | 16x64x256 w4 s3 g1 | 4.95 |
| 16 | 16x64x256 w4 s3 g1 | 5.31 |
| 64 | 64x64x128 w4 s3 g1 | 7.51 |
| 256 | 128x128x64 w8 s3 g8 | 17.7 |
| 1024 | 128x128x64 w8 s3 g8 | 48.5 |
| 4096 | 128x128x64 w4 s3 g8 | 180.6 |

## down_proj: K=6144, N=2048

| M | bf16 | CUTLASS | vLLM Triton | _int_mm | ours default | ours tuned | ours tuned over CUTLASS | ours tuned over vLLM Triton | best int8 over bf16 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 13.1 | 9.14 | 14.5 | n/a | 12.2 | 7.61 | 1.20x | 1.90x | 1.71x |
| 4 | 23.6 | 9.10 | 14.5 | n/a | 12.2 | 7.62 | 1.19x | 1.90x | 3.09x |
| 16 | 25.2 | 9.23 | 13.7 | n/a | 12.6 | 8.13 | 1.14x | 1.69x | 3.10x |
| 64 | 14.5 | 15.8 | 14.2 | 25.8 | 27.7 | 11.4 | 1.38x | 1.24x | 1.27x |
| 256 | 47.8 | 24.1 | 57.1 | 29.7 | 28.2 | 23.0 | 1.05x | 2.49x | 2.08x |
| 1024 | 159.4 | 55.7 | 59.9 | 87.3 | 48.9 | 47.2 | 1.18x | 1.27x | 3.38x |
| 4096 | 608.0 | 210.7 | 234.1 | 303.9 | 173.0 | 173.5 | 1.21x | 1.35x | 3.51x |

With per-token activation quantization inside the timing:

| M | bf16 | CUTLASS + quant | ours + quant | CUTLASS + quant over bf16 | ours + quant over bf16 |
|---|---|---|---|---|---|
| 1 | 13.1 | 10.7 | 9.24 | 1.21x | 1.41x |
| 4 | 23.6 | 10.7 | 9.22 | 2.20x | 2.56x |
| 16 | 25.2 | 10.8 | 9.78 | 2.33x | 2.58x |
| 64 | 14.5 | 17.6 | 13.4 | 0.82x | 1.08x |
| 256 | 47.8 | 26.4 | 25.4 | 1.81x | 1.88x |
| 1024 | 159.4 | 61.4 | 52.7 | 2.59x | 3.02x |
| 4096 | 608.0 | 306.8 | 258.3 | 1.98x | 2.35x |

Tuned configurations:

| M | config | microseconds |
|---|---|---|
| 1 | 16x64x256 w4 s3 g1 | 7.61 |
| 4 | 16x64x256 w4 s3 g1 | 7.62 |
| 16 | 16x64x256 w4 s3 g1 | 8.13 |
| 64 | 32x64x256 w4 s3 g1 | 11.4 |
| 256 | 64x64x256 w4 s3 g8 | 23.0 |
| 1024 | 128x128x128 w4 s3 g8 | 47.3 |
| 4096 | 128x128x64 w8 s3 g8 | 173.5 |

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

Tuning sweep: 820 configurations, 0 failed or errored.

Other processes on the GPU during the run: none.
