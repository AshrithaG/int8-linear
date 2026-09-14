# Qwen3-1.7B end to end in vLLM on NVIDIA GeForce RTX 4090

vLLM 0.28.0, torch 2.13.0+cu130, Triton 3.7.1, max_model_len 1024, prefix caching off, each backend in its own process. Decode: a 64-token prompt, 128 tokens generated, median of 5 repeats, with the first-token step timed separately and taken off. Relative columns are throughput over CUTLASS, so above 1.00x is faster.

## With torch.compile and CUDA graphs (vLLM's default)

Decode tokens per second:

| batch | bf16 | CUTLASS | vLLM Triton | #45126 Triton | this repo | bf16 vs CUTLASS | vLLM Triton vs CUTLASS | #45126 Triton vs CUTLASS | this repo vs CUTLASS |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 227.1 | 307.4 | 218.8 | 315.4 | 325.3 | 0.74x | 0.71x | 1.03x | 1.06x |
| 4 | 810.3 | 1181.3 | 854.5 | 1216.0 | 1251.8 | 0.69x | 0.72x | 1.03x | 1.06x |
| 16 | 3094.9 | 4393.2 | 3272.2 | 4554.6 | 4671.6 | 0.70x | 0.74x | 1.04x | 1.06x |
| 32 | 5985.1 | 7607.6 | 6261.5 | 8569.6 | 8745.8 | 0.79x | 0.82x | 1.13x | 1.15x |
| 48 | 8361.9 | 10272.2 | 8812.9 | 11891.9 | 11466.3 | 0.81x | 0.86x | 1.16x | 1.12x |
| 64 | 10543.9 | 12798.3 | 11117.6 | 14663.2 | 14659.2 | 0.82x | 0.87x | 1.15x | 1.15x |
| 128 | 16968.0 | 19049.2 | 18202.8 | 21079.5 | 22416.6 | 0.89x | 0.96x | 1.11x | 1.18x |

Prefill, 8 prompts of 512 tokens, prompt tokens per second:

|  | bf16 | CUTLASS | vLLM Triton | #45126 Triton | this repo |
|---|---|---|---|---|---|
| tokens/s | 47475 | 79278 | 95122 | 80188 | 102584 |
| vs CUTLASS | 0.60x | 1.00x | 1.20x | 1.01x | 1.29x |

## Without torch.compile or CUDA graphs

Decode tokens per second:

| batch | bf16 | CUTLASS | this repo | bf16 vs CUTLASS | this repo vs CUTLASS |
|---|---|---|---|---|---|
| 1 | 72.1 | 60.1 | 48.6 | 1.20x | 0.81x |
| 16 | 1128.7 | 950.5 | 768.8 | 1.19x | 0.81x |
| 64 | 4257.3 | 3723.6 | 3005.7 | 1.14x | 0.81x |

Prefill, 8 prompts of 512 tokens, prompt tokens per second:

|  | bf16 | CUTLASS | this repo |
|---|---|---|---|
| tokens/s | 47365 | 78238 | 98450 |
| vs CUTLASS | 0.61x | 1.00x | 1.26x |

## Quality

Perplexity on WikiText-2 test windows, and how far each backend's per-token logprobs sit from CUTLASS's and from bf16's.

| backend | perplexity | tokens | logprob gap to CUTLASS, mean / max | mean logprob gap to bf16 | greedy samples same as CUTLASS |
|---|---|---|---|---|---|
| bf16 | 20.4419 | 20440 | 0.2280 / 12.3591 | 0.0000 | no |
| CUTLASS | 20.4857 | 20440 | 0.0000 / 0.0000 | 0.2280 | yes |
| vLLM Triton | 20.5208 | 20440 | 0.1563 / 7.7321 | 0.2274 | no |
| #45126 Triton | 20.5208 | 20440 | 0.1563 / 7.7321 | 0.2274 | no |
| this repo | 20.5208 | 20440 | 0.1563 / 7.7321 | 0.2274 | no |

Text: datasets: Salesforce/wikitext wikitext-2-raw-v1 test, rows joined by blank lines.

## Kernels in use

- compiled bf16: {}
- compiled CUTLASS: {'CutlassInt8ScaledMMLinearKernel': 112}
- compiled vLLM Triton: {'TritonInt8ScaledMMLinearKernel': 112}
- compiled #45126 Triton: {'TritonInt8ScaledMMLinearKernel': 112}; patched kernel calls 26208
- compiled this repo: {'TritonInt8ScaledMMLinearKernel': 112}; patched kernel calls 26208, fallbacks 0
- eager bf16: {}
- eager CUTLASS: {'CutlassInt8ScaledMMLinearKernel': 112}
- eager this repo: {'TritonInt8ScaledMMLinearKernel': 112}; patched kernel calls 260960, fallbacks 0

Other processes on the GPU during the runs: none.
