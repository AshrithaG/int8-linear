# W8A8 int8 linear layers on NVIDIA GeForce RTX 4090 (sm_89)

torch 2.13.0+cu130, Triton 3.7.1, vLLM 0.28.0, driver 580.178.04, VLLM_TRITON_USE_TD=unset.

Microseconds per layer, bf16 output, CUDA-graph replay, median of 5 windows. Every number passed a float64 check at its own shape first. A speedup is the second provider's time over the first's, so above 1.00x the first is faster.

## q_proj, o_proj: K=2048, N=2048

| M | bf16 | CUTLASS | vLLM Triton | _int_mm | ours default | ours tuned | ours tuned over CUTLASS | ours tuned over vLLM Triton | best int8 over bf16 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5.89 | 5.98 | n/a | n/a | 5.09 | 3.60 | 1.66x | n/a | 1.63x |
| 4 | 9.51 | 6.50 | n/a | n/a | 5.40 | 3.93 | 1.65x | n/a | 2.42x |
| 16 | 10.3 | 6.66 | n/a | n/a | 6.05 | 4.15 | 1.60x | n/a | 2.49x |
| 64 | 7.99 | 10.3 | n/a | 17.7 | 10.4 | 5.00 | 2.07x | n/a | 1.60x |
| 256 | 16.8 | 15.3 | n/a | 20.8 | 10.2 | 9.01 | 1.69x | n/a | 1.87x |
| 1024 | 54.5 | 43.8 | n/a | 51.6 | 19.0 | 18.6 | 2.35x | n/a | 2.92x |
| 4096 | 199.7 | 169.6 | n/a | 231.6 | 62.3 | 62.5 | 2.71x | n/a | 3.21x |

With per-token activation quantization inside the timing:

| M | bf16 | CUTLASS + quant | ours + quant | CUTLASS + quant over bf16 | ours + quant over bf16 |
|---|---|---|---|---|---|
| 1 | 5.89 | 7.16 | 4.81 | 0.82x | 1.22x |
| 4 | 9.51 | 7.82 | 5.29 | 1.22x | 1.80x |
| 16 | 10.3 | 7.92 | 5.61 | 1.30x | 1.84x |
| 64 | 7.99 | 11.0 | 6.21 | 0.73x | 1.29x |
| 256 | 16.8 | 16.8 | 10.4 | 1.00x | 1.61x |
| 1024 | 54.5 | 46.8 | 21.2 | 1.16x | 2.57x |
| 4096 | 199.7 | 182.7 | 68.8 | 1.09x | 2.90x |

Tuned configurations:

| M | config | microseconds |
|---|---|---|
| 1 | 16x64x256 w4 s3 g1 | 3.93 |
| 4 | 16x64x256 w4 s3 g1 | 3.60 |
| 16 | 16x64x256 w4 s3 g1 | 4.16 |
| 64 | 32x64x128 w4 s3 g1 | 5.45 |
| 256 | 64x64x128 w4 s3 g8 | 9.01 |
| 1024 | 128x128x64 w8 s3 g8 | 18.6 |
| 4096 | 128x128x64 w4 s3 g8 | 62.7 |

## k_proj, v_proj: K=2048, N=1024

| M | bf16 | CUTLASS | vLLM Triton | _int_mm | ours default | ours tuned | ours tuned over CUTLASS | ours tuned over vLLM Triton | best int8 over bf16 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4.78 | 5.88 | n/a | n/a | 5.11 | 3.54 | 1.66x | n/a | 1.35x |
| 4 | 8.64 | 5.88 | n/a | n/a | 4.95 | 3.58 | 1.64x | n/a | 2.41x |
| 16 | 9.09 | 5.98 | n/a | n/a | 5.43 | 3.72 | 1.61x | n/a | 2.45x |
| 64 | 6.12 | 9.18 | n/a | 16.0 | 9.83 | 4.65 | 1.97x | n/a | 1.32x |
| 256 | 12.7 | 13.2 | n/a | 18.1 | 10.0 | 6.35 | 2.08x | n/a | 2.01x |
| 1024 | 31.2 | 37.5 | n/a | 30.4 | 17.8 | 13.1 | 2.86x | n/a | 2.38x |
| 4096 | 102.0 | 86.6 | n/a | 94.8 | 33.0 | 31.5 | 2.75x | n/a | 3.23x |

With per-token activation quantization inside the timing:

| M | bf16 | CUTLASS + quant | ours + quant | CUTLASS + quant over bf16 | ours + quant over bf16 |
|---|---|---|---|---|---|
| 1 | 4.78 | 7.05 | 4.77 | 0.68x | 1.00x |
| 4 | 8.64 | 7.10 | 4.80 | 1.22x | 1.80x |
| 16 | 9.09 | 7.13 | 4.97 | 1.28x | 1.83x |
| 64 | 6.12 | 10.4 | 5.94 | 0.59x | 1.03x |
| 256 | 12.7 | 14.8 | 7.88 | 0.86x | 1.62x |
| 1024 | 31.2 | 40.8 | 15.6 | 0.77x | 2.01x |
| 4096 | 102.0 | 95.8 | 38.5 | 1.06x | 2.65x |

Tuned configurations:

| M | config | microseconds |
|---|---|---|
| 1 | 16x64x256 w4 s3 g1 | 3.54 |
| 4 | 16x64x256 w4 s3 g1 | 3.58 |
| 16 | 16x64x256 w4 s3 g1 | 3.72 |
| 64 | 32x64x128 w4 s3 g1 | 4.65 |
| 256 | 64x64x128 w4 s3 g8 | 6.35 |
| 1024 | 64x128x64 w4 s3 g8 | 13.0 |
| 4096 | 128x128x64 w4 s1 g8 | 31.9 |

## gate_proj, up_proj: K=2048, N=6144

| M | bf16 | CUTLASS | vLLM Triton | _int_mm | ours default | ours tuned | ours tuned over CUTLASS | ours tuned over vLLM Triton | best int8 over bf16 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 11.3 | 8.22 | n/a | n/a | 5.34 | 4.44 | 1.85x | n/a | 2.55x |
| 4 | 10.3 | 8.65 | n/a | n/a | 5.41 | 4.85 | 1.78x | n/a | 2.13x |
| 16 | 12.0 | 9.05 | n/a | n/a | 6.05 | 5.57 | 1.62x | n/a | 2.15x |
| 64 | 15.6 | 15.4 | n/a | 19.4 | 10.0 | 7.74 | 1.99x | n/a | 2.01x |
| 256 | 50.1 | 39.6 | n/a | 43.4 | 18.6 | 17.8 | 2.23x | n/a | 2.82x |
| 1024 | 150.9 | 151.9 | n/a | 153.9 | 55.9 | 48.4 | 3.14x | n/a | 3.12x |
| 4096 | 607.4 | 576.1 | n/a | 1244.0 | 180.2 | 178.8 | 3.22x | n/a | 3.40x |

With per-token activation quantization inside the timing:

| M | bf16 | CUTLASS + quant | ours + quant | CUTLASS + quant over bf16 | ours + quant over bf16 |
|---|---|---|---|---|---|
| 1 | 11.3 | 9.55 | 5.78 | 1.18x | 1.96x |
| 4 | 10.3 | 9.55 | 5.84 | 1.08x | 1.77x |
| 16 | 12.0 | 10.4 | 6.78 | 1.15x | 1.76x |
| 64 | 15.6 | 16.8 | 9.50 | 0.93x | 1.64x |
| 256 | 50.1 | 41.2 | 19.3 | 1.22x | 2.60x |
| 1024 | 150.9 | 159.7 | 51.2 | 0.94x | 2.95x |
| 4096 | 607.4 | 609.9 | 206.8 | 1.00x | 2.94x |

Tuned configurations:

| M | config | microseconds |
|---|---|---|
| 1 | 16x64x256 w4 s3 g1 | 4.46 |
| 4 | 16x64x256 w4 s3 g1 | 4.85 |
| 16 | 16x64x256 w4 s3 g1 | 5.57 |
| 64 | 64x64x256 w4 s3 g1 | 7.78 |
| 256 | 128x128x64 w8 s3 g8 | 17.7 |
| 1024 | 128x128x64 w8 s3 g8 | 48.5 |
| 4096 | 128x128x64 w4 s3 g8 | 178.8 |

## down_proj: K=6144, N=2048

| M | bf16 | CUTLASS | vLLM Triton | _int_mm | ours default | ours tuned | ours tuned over CUTLASS | ours tuned over vLLM Triton | best int8 over bf16 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 13.4 | 10.8 | 14.4 | n/a | 12.3 | 7.62 | 1.42x | 1.89x | 1.76x |
| 4 | 23.6 | 10.8 | n/a | n/a | 12.0 | 7.65 | 1.41x | n/a | 3.08x |
| 16 | 25.3 | 11.0 | n/a | n/a | 13.5 | 8.16 | 1.35x | n/a | 3.10x |
| 64 | 14.8 | 15.8 | n/a | 28.9 | 26.1 | 11.6 | 1.36x | n/a | 1.28x |
| 256 | 45.5 | 28.3 | n/a | 33.4 | 26.5 | 23.3 | 1.21x | n/a | 1.95x |
| 1024 | 159.8 | 72.0 | n/a | 93.5 | 48.7 | 46.8 | 1.54x | n/a | 3.42x |
| 4096 | 597.0 | 279.6 | n/a | 337.1 | 178.6 | 173.7 | 1.61x | n/a | 3.44x |

With per-token activation quantization inside the timing:

| M | bf16 | CUTLASS + quant | ours + quant | CUTLASS + quant over bf16 | ours + quant over bf16 |
|---|---|---|---|---|---|
| 1 | 13.4 | 12.4 | 9.22 | 1.08x | 1.45x |
| 4 | 23.6 | 12.5 | 9.25 | 1.89x | 2.55x |
| 16 | 25.3 | 12.5 | 9.81 | 2.02x | 2.58x |
| 64 | 14.8 | 17.6 | 13.5 | 0.84x | 1.10x |
| 256 | 45.5 | 30.6 | 25.5 | 1.49x | 1.78x |
| 1024 | 159.8 | 79.5 | 52.8 | 2.01x | 3.02x |
| 4096 | 597.0 | 376.0 | 277.5 | 1.59x | 2.15x |

Tuned configurations:

| M | config | microseconds |
|---|---|---|
| 1 | 16x64x256 w4 s3 g1 | 7.62 |
| 4 | 16x64x256 w4 s3 g1 | 7.65 |
| 16 | 16x64x256 w4 s3 g1 | 8.16 |
| 64 | 32x64x256 w4 s3 g1 | 11.6 |
| 256 | 64x64x256 w4 s3 g8 | 23.3 |
| 1024 | 128x128x128 w4 s3 g8 | 46.8 |
| 4096 | 128x128x64 w4 s1 g8 | 173.7 |

## Not timed

- q_proj, o_proj, M=1, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 1
- q_proj, o_proj, M=1, int8 Triton (vLLM): FAIL: max relative error 0.168
- q_proj, o_proj, M=4, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 4
- q_proj, o_proj, M=4, int8 Triton (vLLM): FAIL: max relative error 0.0339
- q_proj, o_proj, M=16, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 16
- q_proj, o_proj, M=16, int8 Triton (vLLM): FAIL: max relative error 0.579
- q_proj, o_proj, M=64, int8 Triton (vLLM): FAIL: max relative error 1
- q_proj, o_proj, M=256, int8 Triton (vLLM): FAIL: max relative error 1
- q_proj, o_proj, M=1024, int8 Triton (vLLM): FAIL: max relative error 1
- q_proj, o_proj, M=4096, int8 Triton (vLLM): FAIL: max relative error 1
- k_proj, v_proj, M=1, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 1
- k_proj, v_proj, M=1, int8 Triton (vLLM): FAIL: max relative error 0.018
- k_proj, v_proj, M=4, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 4
- k_proj, v_proj, M=4, int8 Triton (vLLM): FAIL: max relative error 0.052
- k_proj, v_proj, M=16, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 16
- k_proj, v_proj, M=16, int8 Triton (vLLM): FAIL: max relative error 0.0669
- k_proj, v_proj, M=64, int8 Triton (vLLM): FAIL: max relative error 1
- k_proj, v_proj, M=256, int8 Triton (vLLM): FAIL: max relative error 1
- k_proj, v_proj, M=1024, int8 Triton (vLLM): FAIL: max relative error 1
- k_proj, v_proj, M=4096, int8 Triton (vLLM): FAIL: max relative error 1
- gate_proj, up_proj, M=1, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 1
- gate_proj, up_proj, M=1, int8 Triton (vLLM): FAIL: max relative error 1
- gate_proj, up_proj, M=4, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 4
- gate_proj, up_proj, M=4, int8 Triton (vLLM): FAIL: max relative error 1
- gate_proj, up_proj, M=16, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 16
- gate_proj, up_proj, M=16, int8 Triton (vLLM): FAIL: max relative error 1
- gate_proj, up_proj, M=64, int8 Triton (vLLM): FAIL: max relative error 1
- gate_proj, up_proj, M=256, int8 Triton (vLLM): FAIL: max relative error 1
- gate_proj, up_proj, M=1024, int8 Triton (vLLM): FAIL: max relative error 1
- gate_proj, up_proj, M=4096, int8 Triton (vLLM): FAIL: max relative error 1
- down_proj, M=1, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 1
- down_proj, M=4, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 4
- down_proj, M=4, int8 Triton (vLLM): FAIL: max relative error 0.141
- down_proj, M=16, int8 torch._int_mm, unfused: error: self.size(0) needs to be greater than 16, but got 16
- down_proj, M=16, int8 Triton (vLLM): FAIL: max relative error 0.0253
- down_proj, M=64, int8 Triton (vLLM): FAIL: max relative error 1
- down_proj, M=256, int8 Triton (vLLM): FAIL: max relative error 1
- down_proj, M=1024, int8 Triton (vLLM): FAIL: max relative error 1
- down_proj, M=4096, int8 Triton (vLLM): FAIL: max relative error 1

Tuning sweep: 820 configurations, 0 failed or errored.

Other processes on the GPU during the run: none.
