# Fusing the int8 activation quantizer into RMSNorm and SiLU-and-mul

Numerics chosen: IEEE division = True, SiLU rounded before the multiply = True.

| site | M | residual | path | int8 values that differ | largest difference | residual identical |
|---|---|---|---|---|---|---|
| RMSNorm | 1 | False | vllm_fused_cuda | 2.44e-03 | 1 | n/a |
| RMSNorm | 1 | False | ours_precise_div=True | 2.44e-03 | 1 | n/a |
| RMSNorm | 1 | True | vllm_fused_cuda | 6.84e-03 | 1 | True |
| RMSNorm | 1 | True | ours_precise_div=True | 6.84e-03 | 1 | True |
| RMSNorm | 7 | False | vllm_fused_cuda | 3.49e-03 | 1 | n/a |
| RMSNorm | 7 | False | ours_precise_div=True | 3.49e-03 | 1 | n/a |
| RMSNorm | 7 | True | vllm_fused_cuda | 1.13e-02 | 2 | True |
| RMSNorm | 7 | True | ours_precise_div=True | 1.13e-02 | 2 | True |
| RMSNorm | 64 | False | vllm_fused_cuda | 5.52e-03 | 1 | n/a |
| RMSNorm | 64 | False | ours_precise_div=True | 5.52e-03 | 1 | n/a |
| RMSNorm | 64 | True | vllm_fused_cuda | 1.05e-02 | 2 | True |
| RMSNorm | 64 | True | ours_precise_div=True | 1.05e-02 | 2 | True |
| RMSNorm | 333 | False | vllm_fused_cuda | 5.57e-03 | 2 | n/a |
| RMSNorm | 333 | False | ours_precise_div=True | 5.56e-03 | 2 | n/a |
| RMSNorm | 333 | True | vllm_fused_cuda | 9.10e-03 | 2 | True |
| RMSNorm | 333 | True | ours_precise_div=True | 9.09e-03 | 2 | True |
| RMSNorm | 2048 | False | vllm_fused_cuda | 5.48e-03 | 2 | n/a |
| RMSNorm | 2048 | False | ours_precise_div=True | 5.48e-03 | 2 | n/a |
| RMSNorm | 2048 | True | vllm_fused_cuda | 8.65e-03 | 2 | True |
| RMSNorm | 2048 | True | ours_precise_div=True | 8.65e-03 | 2 | True |
| SiLU-and-mul | 1 | - | ours_precise_div=True_round_silu=True | 0.00e+00 | 0 | n/a |
| SiLU-and-mul | 7 | - | ours_precise_div=True_round_silu=True | 0.00e+00 | 0 | n/a |
| SiLU-and-mul | 64 | - | ours_precise_div=True_round_silu=True | 0.00e+00 | 0 | n/a |
| SiLU-and-mul | 333 | - | ours_precise_div=True_round_silu=True | 0.00e+00 | 0 | n/a |
| SiLU-and-mul | 2048 | - | ours_precise_div=True_round_silu=True | 0.00e+00 | 0 | n/a |

Kernel time, microseconds per call (CUDA-graph replay of 100 calls):

| M | RMSNorm + quant, vLLM unfused | vLLM fused CUDA | this repo, Triton | SiLU-and-mul + quant, vLLM unfused | this repo, Triton |
|---|---|---|---|---|---|
| 1 | 2.9 | 2.23 | 1.66 | 3.51 | 2.4 |
| 4 | 2.88 | 2.24 | 1.65 | 3.5 | 2.39 |
| 16 | 2.95 | 2.26 | 1.7 | 3.55 | 2.47 |
| 32 | 3.02 | 2.31 | 1.76 | 3.67 | 2.51 |
| 48 | 3.13 | 2.39 | 1.82 | 3.85 | 2.58 |
| 64 | 3.21 | 2.46 | 1.91 | 3.9 | 2.63 |
| 128 | 3.56 | 2.71 | 2.18 | 4.45 | 2.8 |
| 1024 | 8.79 | 14.09 | 5.59 | 15.33 | 10.17 |
| 4096 | 24.59 | 48.58 | 13.88 | 211.76 | 136.98 |

Decode, the stand-in's prediction (written first) against serving, tokens/s:

| batch | stand-in saving, us/layer | predicted change | cutlass -> cutlass_fused | ours -> ours_fused |
|---|---|---|---|---|
| 1 | 3.44 | +3.2% | 312 -> 318 (+1.8%) | 325 -> 337 (+3.5%) |
| 4 | 3.58 | +3.2% | 1,201 -> 1,226 (+2.1%) | 1,254 -> 1,302 (+3.8%) |
| 16 | 3.69 | +3.1% | 4,459 -> 4,558 (+2.2%) | 4,670 -> 4,865 (+4.2%) |
| 32 | 3.55 | +2.8% | 7,692 -> 7,782 (+1.2%) | 8,752 -> 9,063 (+3.5%) |
| 48 | 3.99 | +2.9% | 10,357 -> 10,531 (+1.7%) | 12,028 -> 12,385 (+3.0%) |
| 64 | 4.68 | +3.1% | 12,901 -> 13,122 (+1.7%) | 14,794 -> 15,201 (+2.8%) |
| 128 | 5.59 | +2.9% | 19,120 -> 19,449 (+1.7%) | 22,652 -> 23,231 (+2.6%) |

Prefill (8 x 512 tokens) and quality, fused against unfused, same GEMM:

| pair | prefill tok/s change | perplexity, unfused / fused | logprob gap mean / max | greedy samples identical | matmuls that took a PreQuant |
|---|---|---|---|---|---|
| cutlass -> cutlass_fused | +6.8% | 20.5800 / 20.4323 | 1.85e-01 / 8.60e+00 | no | 84 |
| ours -> ours_fused | +10.1% | 20.5208 / 20.4037 | 1.86e-01 / 8.60e+00 | no | 84 |

Paired quality, mean log-probability per token over 40 WikiText-2 windows (positive: the first assigns the text higher probability):

| comparison | difference |
|---|---|
| cutlass_fused - cutlass | +0.0072 (SE 0.0025, z +2.85) |
| ours_fused - ours | +0.0057 (SE 0.0029, z +1.97) |
| cutlass_fused - bf16 | +0.0005 (SE 0.0036, z +0.13) |
| ours_fused - bf16 | +0.0019 (SE 0.0034, z +0.55) |
| cutlass - bf16 | -0.0067 (SE 0.0037, z -1.83) |
| ours - bf16 | -0.0038 (SE 0.0033, z -1.15) |
| cutlass today - cutlass in the earlier published run (no code change) | -0.0046 (SE 0.0018, z -2.53) |
