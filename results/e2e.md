# Qwen3-1.7B end to end in vLLM on NVIDIA GeForce RTX 4090

vLLM 0.28.0, torch 2.13.0+cu130, Triton 3.7.1, max_model_len 1024, prefix caching off, each backend in its own process. Decode: a 64-token prompt, 128 tokens generated, median of 5 repeats, with the first-token step timed separately and taken off. Relative columns are throughput over CUTLASS, so above 1.00x is faster.

## With torch.compile and CUDA graphs (vLLM's default)

Decode tokens per second:

| batch | bf16 | CUTLASS | vLLM Triton | #45126 Triton | this repo | bf16 vs CUTLASS | vLLM Triton vs CUTLASS | #45126 Triton vs CUTLASS | this repo vs CUTLASS |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 227.0 | 307.5 | 218.7 | 315.3 | 325.3 | 0.74x | 0.71x | 1.03x | 1.06x |
| 4 | 809.1 | 1184.3 | 853.8 | 1216.4 | 1252.3 | 0.68x | 0.72x | 1.03x | 1.06x |
| 16 | 3093.1 | 4412.0 | 3267.7 | 4557.9 | 4671.3 | 0.70x | 0.74x | 1.03x | 1.06x |
| 32 | 5992.9 | 7603.1 | 6256.6 | 8564.0 | 8663.7 | 0.79x | 0.82x | 1.13x | 1.14x |
| 48 | 8346.9 | 10288.8 | 8792.2 | 11868.3 | 11877.1 | 0.81x | 0.85x | 1.15x | 1.15x |
| 64 | 10536.7 | 12819.5 | 11079.9 | 14638.5 | 14655.2 | 0.82x | 0.86x | 1.14x | 1.14x |
| 128 | 16960.1 | 19035.2 | 18190.6 | 21080.3 | 20970.6 | 0.89x | 0.96x | 1.11x | 1.10x |

Prefill, 8 prompts of 512 tokens, prompt tokens per second:

|  | bf16 | CUTLASS | vLLM Triton | #45126 Triton | this repo |
|---|---|---|---|---|---|
| tokens/s | 47490 | 79218 | 94941 | 80013 | 101919 |
| vs CUTLASS | 0.60x | 1.00x | 1.20x | 1.01x | 1.29x |

## Without torch.compile or CUDA graphs

Decode tokens per second:

| batch | bf16 | CUTLASS | this repo | bf16 vs CUTLASS | this repo vs CUTLASS |
|---|---|---|---|---|---|
| 1 | 70.6 | 60.9 | 51.5 | 1.16x | 0.85x |
| 16 | 1102.4 | 959.8 | 814.6 | 1.15x | 0.85x |
| 64 | 4171.5 | 3747.3 | 3194.2 | 1.11x | 0.85x |

Prefill, 8 prompts of 512 tokens, prompt tokens per second:

|  | bf16 | CUTLASS | this repo |
|---|---|---|---|
| tokens/s | 47302 | 79715 | 98502 |
| vs CUTLASS | 0.59x | 1.00x | 1.24x |

## Against the first run

This run's throughput over the first run's (results/e2e_*_run1.json). Backends whose code did not change between the runs show run-to-run variation.

With torch.compile and CUDA graphs:

|  | bf16 | CUTLASS | vLLM Triton | #45126 Triton | this repo |
|---|---|---|---|---|---|
| decode batch 1 | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |
| decode batch 4 | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |
| decode batch 16 | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |
| decode batch 32 | 1.00x | 1.00x | 1.00x | 1.00x | 0.99x |
| decode batch 48 | 1.00x | 1.00x | 1.00x | 1.00x | 1.04x |
| decode batch 64 | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |
| decode batch 128 | 1.00x | 1.00x | 1.00x | 1.00x | 0.94x |
| prefill | 1.00x | 1.00x | 1.00x | 1.00x | 0.99x |

Without torch.compile or CUDA graphs:

|  | bf16 | CUTLASS | this repo |
|---|---|---|---|
| decode batch 1 | 0.98x | 1.01x | 1.06x |
| decode batch 16 | 0.98x | 1.01x | 1.06x |
| decode batch 64 | 0.98x | 1.01x | 1.06x |
| prefill | 1.00x | 1.02x | 1.00x |

## Quality

Perplexity on WikiText-2 test windows, and how far each backend's per-token logprobs sit from CUTLASS's and from bf16's.

| backend | perplexity | tokens | logprob gap to CUTLASS, mean / max | mean logprob gap to bf16 | greedy samples same as CUTLASS |
|---|---|---|---|---|---|
| bf16 | 20.4419 | 20440 | 0.2280 / 12.3591 | 0.0000 | no |
| CUTLASS | 20.4857 | 20440 | 0.0000 / 0.0000 | 0.2280 | yes |
| vLLM Triton | 20.5208 | 20440 | 0.1563 / 7.7321 | 0.2274 | no |
| #45126 Triton | 20.5208 | 20440 | 0.1563 / 7.7321 | 0.2274 | no |
| this repo | 20.5208 | 20440 | 0.1563 / 7.7321 | 0.2274 | no |

vLLM Triton, #45126 Triton, this repo: per-token logprobs identical at every token.

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
