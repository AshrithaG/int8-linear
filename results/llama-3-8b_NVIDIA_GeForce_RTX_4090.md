# W8A8 int8 linear layers on NVIDIA GeForce RTX 4090 (sm_89)

torch 2.13.0+cu130, Triton 3.7.1, vLLM 0.28.0, driver 580.178.04, VLLM_TRITON_USE_TD=unset, #45126 on its tuned table for sm_89.

Microseconds per layer, bf16 output, no bias, CUDA-graph replay, median of 5 windows. Every number passed a float64 check at its own shape first. A speedup is the second provider's time over the first's, so above 1.00x the first is faster. "ours" in the speedup columns is the tuned kernel when this run tuned, and the stored table otherwise.

## qkv_proj: K=4096, N=6144

| M | bf16 | CUTLASS | vLLM Triton | #45126 | _int_mm | ours default | ours tuned | ours table | ours over CUTLASS | ours over vLLM Triton | ours over #45126 | best int8 over bf16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 13.9 | 8.37 | 10.3 | 9.79 | n/a | 8.91 | 7.17 | 8.91 | 1.17x | 1.44x | 1.37x | 1.93x |
| 4 | 18.5 | 8.70 | 10.4 | 10.0 | n/a | 9.07 | 7.34 | 9.05 | 1.19x | 1.42x | 1.37x | 2.52x |
| 16 | 22.0 | 10.0 | 10.5 | 10.6 | n/a | 10.2 | 9.20 | 10.2 | 1.09x | 1.14x | 1.15x | 2.39x |
| 64 | 27.9 | 20.2 | 15.4 | 14.0 | 22.0 | 19.2 | 15.5 | 19.2 | 1.30x | 0.99x | 0.90x | 2.00x |
| 256 | 86.4 | 36.4 | 40.4 | 44.4 | 55.8 | 35.6 | 31.9 | 35.6 | 1.14x | 1.27x | 1.39x | 2.70x |
| 1024 | 299.5 | 146.1 | 121.0 | 164.9 | 202.1 | 107.3 | 90.0 | 109.2 | 1.62x | 1.34x | 1.83x | 3.33x |
| 4096 | 1203.8 | 522.0 | 478.6 | 632.2 | 1183.1 | 344.7 | 352.1 | 344.5 | 1.48x | 1.36x | 1.80x | 3.49x |

With per-token activation quantization inside the timing:

| M | bf16 | CUTLASS + quant | ours + quant | CUTLASS + quant over bf16 | ours + quant over bf16 |
|---|---|---|---|---|---|
| 1 | 13.9 | 9.74 | 8.51 | 1.42x | 1.63x |
| 4 | 18.5 | 10.0 | 8.79 | 1.84x | 2.10x |
| 16 | 22.0 | 11.3 | 10.5 | 1.95x | 2.10x |
| 64 | 27.9 | 21.3 | 15.4 | 1.31x | 1.81x |
| 256 | 86.4 | 38.3 | 33.8 | 2.25x | 2.55x |
| 1024 | 299.5 | 162.9 | 93.5 | 1.84x | 3.20x |
| 4096 | 1203.8 | 585.9 | 404.9 | 2.05x | 2.97x |

Tuned configurations:

| M | config | microseconds |
|---|---|---|
| 1 | 16x64x256 w4 s3 g1 | 7.81 |
| 4 | 16x64x256 w4 s3 g1 | 7.31 |
| 16 | 16x64x256 w4 s3 g1 | 9.22 |
| 64 | 64x64x256 w4 s3 g1 | 15.6 |
| 256 | 128x128x64 w8 s3 g8 | 31.9 |
| 1024 | 128x128x64 w8 s3 g8 | 90.1 |
| 4096 | 128x128x64 w4 s3 g8 | 344.7 |

## o_proj: K=4096, N=4096

| M | bf16 | CUTLASS | vLLM Triton | #45126 | _int_mm | ours default | ours tuned | ours table | ours over CUTLASS | ours over vLLM Triton | ours over #45126 | best int8 over bf16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 15.4 | 7.31 | 10.3 | 9.77 | n/a | 8.68 | 5.80 | 8.68 | 1.26x | 1.77x | 1.68x | 2.65x |
| 4 | 17.1 | 7.34 | 10.3 | 9.88 | n/a | 8.69 | 5.96 | 8.69 | 1.23x | 1.73x | 1.66x | 2.87x |
| 16 | 18.7 | 7.58 | 9.91 | 10.1 | n/a | 9.27 | 7.37 | 9.26 | 1.03x | 1.34x | 1.37x | 2.54x |
| 64 | 18.4 | 15.2 | 11.7 | 10.6 | 20.7 | 19.1 | 11.4 | 19.1 | 1.33x | 1.02x | 0.93x | 1.73x |
| 256 | 52.4 | 26.8 | 39.7 | 30.3 | 35.7 | 24.7 | 23.9 | 24.7 | 1.12x | 1.67x | 1.27x | 2.20x |
| 1024 | 197.9 | 80.1 | 81.2 | 110.8 | 128.5 | 59.7 | 57.0 | 59.7 | 1.40x | 1.42x | 1.94x | 3.47x |
| 4096 | 796.5 | 308.6 | 319.9 | 423.3 | 748.7 | 230.6 | 230.6 | 233.1 | 1.34x | 1.39x | 1.84x | 3.45x |

With per-token activation quantization inside the timing:

| M | bf16 | CUTLASS + quant | ours + quant | CUTLASS + quant over bf16 | ours + quant over bf16 |
|---|---|---|---|---|---|
| 1 | 15.4 | 8.66 | 7.12 | 1.78x | 2.16x |
| 4 | 17.1 | 8.69 | 7.28 | 1.97x | 2.35x |
| 16 | 18.7 | 8.82 | 8.55 | 2.12x | 2.19x |
| 64 | 18.4 | 16.4 | 12.4 | 1.12x | 1.49x |
| 256 | 52.4 | 28.8 | 25.7 | 1.82x | 2.04x |
| 1024 | 197.9 | 85.3 | 61.2 | 2.32x | 3.23x |
| 4096 | 796.5 | 367.8 | 289.0 | 2.17x | 2.76x |

Tuned configurations:

| M | config | microseconds |
|---|---|---|
| 1 | 16x64x256 w4 s3 g1 | 5.81 |
| 4 | 16x64x256 w4 s3 g1 | 5.95 |
| 16 | 16x64x256 w4 s3 g1 | 7.36 |
| 64 | 64x64x128 w4 s3 g1 | 11.4 |
| 256 | 64x128x128 w4 s3 g8 | 23.9 |
| 1024 | 128x128x64 w4 s1 g8 | 56.8 |
| 4096 | 128x128x64 w4 s3 g8 | 230.0 |

## gate_up_proj: K=4096, N=28672

| M | bf16 | CUTLASS | vLLM Triton | #45126 | _int_mm | ours default | ours tuned | ours table | ours over CUTLASS | ours over vLLM Triton | ours over #45126 | best int8 over bf16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 246.6 | 140.3 | 127.3 | 126.5 | n/a | 131.5 | 125.4 | 131.4 | 1.12x | 1.02x | 1.01x | 1.97x |
| 4 | 249.1 | 140.6 | 128.2 | 127.5 | n/a | 130.8 | 125.7 | 130.8 | 1.12x | 1.02x | 1.01x | 1.98x |
| 16 | 252.8 | 142.3 | 129.9 | 129.6 | n/a | 134.4 | 127.3 | 134.4 | 1.12x | 1.02x | 1.02x | 1.99x |
| 64 | 278.2 | 147.4 | 133.6 | 135.0 | 176.4 | 248.8 | 133.8 | 248.8 | 1.10x | 1.00x | 1.01x | 2.08x |
| 256 | 376.0 | 189.7 | 280.8 | 204.2 | 286.5 | 510.8 | 163.5 | 510.8 | 1.16x | 1.72x | 1.25x | 2.30x |
| 1024 | 1468.3 | 605.0 | 1125.2 | 594.2 | 1368.3 | 410.3 | 421.5 | 424.1 | 1.44x | 2.67x | 1.41x | 3.58x |
| 4096 | 5844.2 | 2241.7 | 4526.1 | 2472.6 | 5485.2 | 1862.0 | 1800.4 | 1722.1 | 1.25x | 2.51x | 1.37x | 3.39x |

With per-token activation quantization inside the timing:

| M | bf16 | CUTLASS + quant | ours + quant | CUTLASS + quant over bf16 | ours + quant over bf16 |
|---|---|---|---|---|---|
| 1 | 246.6 | 141.7 | 126.8 | 1.74x | 1.94x |
| 4 | 249.1 | 142.1 | 127.3 | 1.75x | 1.96x |
| 16 | 252.8 | 143.7 | 129.0 | 1.76x | 1.96x |
| 64 | 278.2 | 149.9 | 135.9 | 1.86x | 2.05x |
| 256 | 376.0 | 192.7 | 170.0 | 1.95x | 2.21x |
| 1024 | 1468.3 | 625.9 | 432.4 | 2.35x | 3.40x |
| 4096 | 5844.2 | 2286.8 | 1890.9 | 2.56x | 3.09x |

Tuned configurations:

| M | config | microseconds |
|---|---|---|
| 1 | 16x64x256 w4 s3 g1 | 125.4 |
| 4 | 16x128x256 w4 s1 g1 | 125.7 |
| 16 | 16x128x256 w4 s1 g1 | 127.3 |
| 64 | 64x64x256 w4 s3 g1 | 133.8 |
| 256 | 64x128x128 w4 s3 g8 | 163.8 |
| 1024 | 128x128x64 w8 s3 g8 | 414.2 |
| 4096 | 128x128x64 w8 s3 g8 | 1661.9 |

## down_proj: K=14336, N=4096

| M | bf16 | CUTLASS | vLLM Triton | #45126 | _int_mm | ours default | ours tuned | ours table | ours over CUTLASS | ours over vLLM Triton | ours over #45126 | best int8 over bf16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 125.7 | 19.0 | 31.7 | 29.7 | n/a | 26.7 | 16.7 | 26.7 | 1.14x | 1.90x | 1.78x | 7.52x |
| 4 | 127.9 | 19.0 | 31.8 | 30.1 | n/a | 26.8 | 17.3 | 26.8 | 1.10x | 1.83x | 1.73x | 7.38x |
| 16 | 127.7 | 20.7 | 29.8 | 30.2 | n/a | 27.8 | 21.8 | 27.8 | 0.95x | 1.37x | 1.39x | 6.17x |
| 64 | 143.7 | 37.3 | 38.7 | 34.9 | 51.9 | 62.3 | 36.7 | 62.3 | 1.02x | 1.05x | 0.95x | 4.12x |
| 256 | 193.0 | 81.4 | 131.4 | 101.1 | 96.5 | 81.3 | 77.1 | 81.3 | 1.06x | 1.70x | 1.31x | 2.50x |
| 1024 | 744.3 | 225.9 | 280.3 | 426.1 | 354.8 | 215.8 | 206.7 | 222.2 | 1.09x | 1.36x | 2.06x | 3.60x |
| 4096 | 2925.0 | 906.9 | 1055.3 | 1537.6 | 1248.7 | 857.7 | 887.4 | 881.7 | 1.02x | 1.19x | 1.73x | 3.41x |

With per-token activation quantization inside the timing:

| M | bf16 | CUTLASS + quant | ours + quant | CUTLASS + quant over bf16 | ours + quant over bf16 |
|---|---|---|---|---|---|
| 1 | 125.7 | 21.2 | 18.8 | 5.92x | 6.67x |
| 4 | 127.9 | 21.3 | 20.0 | 6.01x | 6.41x |
| 16 | 127.7 | 21.9 | 22.9 | 5.84x | 5.58x |
| 64 | 143.7 | 40.7 | 37.8 | 3.53x | 3.80x |
| 256 | 193.0 | 93.9 | 84.3 | 2.06x | 2.29x |
| 1024 | 744.3 | 278.9 | 252.5 | 2.67x | 2.95x |
| 4096 | 2925.0 | 1084.6 | 1022.2 | 2.70x | 2.86x |

Tuned configurations:

| M | config | microseconds |
|---|---|---|
| 1 | 16x64x256 w4 s3 g1 | 16.8 |
| 4 | 16x64x256 w4 s3 g1 | 17.3 |
| 16 | 16x64x256 w4 s3 g1 | 21.7 |
| 64 | 64x64x128 w4 s3 g1 | 36.9 |
| 256 | 64x128x128 w4 s3 g8 | 77.1 |
| 1024 | 128x256x64 w8 s3 g8 | 200.8 |
| 4096 | 128x256x64 w8 s3 g8 | 811.0 |

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
