# int8-linear

W8A8 int8 linear layers for LLM inference on a consumer GPU: a Triton kernel,
measured against vLLM's CUTLASS and Triton kernels, the tuned Triton tables proposed
in vLLM #45126, and bf16 cuBLAS, first one layer at a time and then end to end in vLLM.

**Status: measured on one RTX 4090 (vLLM 0.28.0, Triton 3.7.1, torch 2.13.0+cu130).**
Per layer on Qwen3-1.7B and Llama-3-8B shapes, cross-checked against vLLM's own
benchmark and repeated; Qwen3-1.7B served end to end in vLLM, three times, with WikiText-2
perplexity; a sweep of CUTLASS's dispatch buckets; host launch cost per call, alone and inside
the served model; and a stand-in for the model that predicts its decode step well enough to
tune in.

## Results in brief

- **End to end it is faster than vLLM's default int8 kernel, by less than per layer.**
  Serving a W8A8 Qwen3-1.7B in vLLM with CUDA graphs, decode runs at 1.06x CUTLASS's
  tokens per second at batch 1 to 16 and 1.15x to 1.19x at 32 to 128. Prefill runs at
  1.29x. Per layer, at the batch sizes vLLM decodes at, the median is 1.42x.
- **Against #45126's tables the end-to-end lead is small.** Decode is 1.01x to 1.03x at
  batch 1 to 64 and 1.07x at 128, although per layer this kernel is a median 1.28x faster
  at those batch sizes. Prefill is 1.28x.
- **Timing layers alone mispredicts the served model; a stand-in does not.** Tables tuned
  per layer in isolation made decode slower at two batch sizes. A 28-layer stand-in, each
  layer with its own weights and with the norms and quantizer in between, reproduced the
  served model's changes between tables, and its tie with #45126, within 0.5 microseconds
  per decoder layer. Retuned in it, decode is at least as fast as with either earlier table
  at every batch size measured, and 8.1% faster than the second table at batch 128.
- **No measurable accuracy cost.** WikiText-2 perplexity is 20.44 for bf16, 20.49 with
  CUTLASS and 20.52 with the Triton kernels, and neither int8 result is distinguishable
  from bf16 on this sample. vLLM's Triton kernel, #45126's and this one produced identical
  logprobs at all 20,440 tokens, in all three runs.
- **CUTLASS's slowdown starts at M=17, at a dispatch bucket edge.** Its time steps up
  1.31x to 1.99x from M=16 to M=17 on five of six layer shapes, where vLLM's source
  switches CUTLASS configuration. The sixth, which gets a different configuration in
  that bucket, steps 1.06x. Both were predicted before the sweep ran.
- **Fusing the activation quantizer into the kernel before it adds 2.6% to 4.2% decode
  and 10.1% prefill.** vLLM fuses RMSNorm with the quantizer for FP8 models but not for
  int8 ones. Two Triton kernels, add+RMSNorm+quantize and SiLU-and-mul+quantize, wired
  into the served model, raise decode 2.6% to 4.2% with this kernel and 1.2% to 2.2% with
  CUTLASS; a stand-in, written down first, predicted 2.8% to 3.2%. Prefill rises 10.1% and
  6.8%. vLLM's own fused CUDA kernel is slower than the two unfused kernels from 1,024
  tokens up (48.6 against 24.6 microseconds at 4,096).
- **Where it loses.** Without CUDA graphs, decode is 0.85x to 0.90x CUTLASS and 0.73x to
  0.77x bf16 across runs. Inside the served model its matmul call costs 45.6 microseconds
  of host time against CUTLASS's 29.5, three times the gap measured alone, and that is
  most of the loss.

## The question

On an RTX 4090, where does W8A8 int8 actually beat bf16 for the linear layers of
a small LLM, can a Triton kernel beat vLLM's CUTLASS kernel at the shapes decode and
prefill really use, and does a per-layer win survive in a served model?

## Where this comes from

[nanoinfer](https://github.com/AshrithaG/nanoinfer#against-cublass-fast-path)
compared a hand-written CUDA int8 GEMM, the same GEMM in Triton, and cuBLAS on
square matrices, in one process under CUDA-graph timing. Four results carry over:

- **Operand layout decides the vendor comparison.** At n=4096 cuBLAS was 3.4x
  faster with both operands K-contiguous than with B row-major. A transposed
  nn.Linear weight is already K-contiguous, so every provider here gets that layout.
- **Tile size was the largest lever** for Triton (1.66x), while pipelining at
  small tiles was worth 1.01x.
- **Triton matched cuBLAS** on square matrices, at 540 TOPS.
- **Back-to-back timing misranks small kernels.** At n=256 about 10 microseconds
  of every cuBLAS call was host work, enough to reverse a comparison.

Square matrices are not what an LLM runs. Its linear layers are wide and short:
at decode M is a handful of tokens against K and N in the thousands, a different
regime for tiling, for memory traffic and for the epilogue.

## Prior work this builds on

- **Tuning vLLM's Triton int8 kernel is not new.**
  [#44998](https://github.com/vllm-project/vllm/issues/44998) showed its tile
  heuristic, tuned on AMD, leaves up to 1.82x on an H800 and 1.65x on an L20, and
  [#45126](https://github.com/vllm-project/vllm/pull/45126) proposes tuned tables
  with grouped tile order. Both compare the tuned kernel with the current one and
  treat the benefit on NVIDIA as bounded, because CUTLASS is the default there.
  Neither compares against CUTLASS, and neither checks outputs against an
  independent reference.
- **The small-batch W8A8 slowdown is understood.**
  [#38697](https://github.com/vllm-project/vllm/issues/38697) reported W8A8
  decode slower than FP16, and later measurements on Hopper attributed it to
  host launch cost that disappears under CUDA graphs.
- **vLLM's Triton kernel has a newer load path.**
  [#47205](https://github.com/vllm-project/vllm/pull/47205) added
  tensor-descriptor loads, on by default only on XPU and opt-in on CUDA through
  `VLLM_TRITON_USE_TD`.

What those threads skip, and this repo measures, is vLLM's default int8 kernel on a
consumer Ada GPU against a fused Triton kernel, #45126's tables and bf16: per layer,
with every output checked against a float64 reference, and then in a served model.

## What is measured

| provider | what it is |
|---|---|
| bf16 F.linear | cuBLAS, the unquantized baseline |
| int8 CUTLASS (vLLM) | `cutlass_scaled_mm`, vLLM's W8A8 path on NVIDIA |
| int8 Triton (vLLM) | `triton_scaled_mm`, vLLM's fallback and its ROCm path |
| int8 Triton (#45126 tables) | vLLM's Triton kernel with the NVIDIA tables proposed in #45126, vendored in [`int8_linear/pr45126.py`](int8_linear/pr45126.py) |
| int8 torch._int_mm, unfused | the cuBLASLt product, then scales and bias as separate kernels |
| int8 Triton (this repo) | [`int8_linear/kernel.py`](int8_linear/kernel.py), default and tuned |

**Per layer** ([`bench/bench_linear.py`](bench/bench_linear.py)): Qwen3-1.7B's four
distinct linear layers (2048 to 2048, 2048 to 1024, 2048 to 6144, 6144 to 2048); the
four vLLM actually runs for it after merging q, k and v and gate and up (2048 to 4096,
2048 to 2048, 2048 to 12288, 6144 to 2048); and the merged Llama-3-8B layers from vLLM's
own kernel benchmark. M is 1, 4, 16, 64, 256, 1024 and 4096 tokens, and for vLLM's
Qwen3-1.7B layers also the 51 batch sizes vLLM captures CUDA graphs at plus 1024, 2048
and 4096. CUTLASS and this repo's kernel are also timed with vLLM's per-token
`scaled_int8_quant` inside the measurement, since a real layer pays for both.

**End to end** ([`bench/e2e_vllm.py`](bench/e2e_vllm.py)): Qwen3-1.7B served by vLLM from
one W8A8 checkpoint, with only the int8 matmul changing between backends, plus the
unquantized model. Decode and prefill tokens per second, with torch.compile and CUDA
graphs and without, WikiText-2 perplexity, and per-token logprobs to compare backends.

## Method

The rules from the nanoinfer study:

- Every provider runs in one process, on one stream, under one timer.
- CUDA-graph replay is the primary timing, with back-to-back launches reported
  next to it.
- This repo's kernel must reproduce the int32 product exactly (`tests/`), and
  every provider's bf16 output must agree with a float64 reference to within 1% of
  the larger of the output and its matmul term before it is timed. vLLM's own test
  for its Triton kernel allows 10%.
- Output buffers are poisoned before each check, so a kernel that writes nothing
  fails instead of passing on another provider's leftover answer.
- Other GPU processes are recorded before and after, and timings are written to
  disk after every shape.

End to end:

- One checkpoint for every int8 backend:
  [nytopop/Qwen3-1.7B.w8a8](https://huggingface.co/nytopop/Qwen3-1.7B.w8a8) at a pinned
  revision, SmoothQuant then GPTQ, with int8 per-channel weights and dynamic per-token
  int8 activations. Its config and recipe were checked, not assumed.
- Each backend runs in its own process, with prefix caching off and vLLM's compile
  cache disabled. CUTLASS and vLLM's Triton kernel are selected with `--linear-backend`;
  #45126's kernel and this one replace the function vLLM's Triton path calls. Each run
  records which kernel class its 112 linear layers use and how many calls reached the
  patched kernel.
- Decode generates 128 tokens from a 64-token prompt, with the first-token step timed
  separately and subtracted; the median of five repeats is reported. Repeats varied by
  at most 0.3% with CUDA graphs and 1.3% without.
- Perplexity uses 40 windows of 512 tokens from the WikiText-2 test set. Differences
  between backends are paired token by token.

## The kernel

- int8 times int8 accumulated in int32 with `tl.dot`, so the product is exact.
- Per-token and per-channel scales and the bias applied in the same kernel and
  written straight to bf16.
- The weight taken as the transposed [K, N] view of an nn.Linear weight, which is
  K-contiguous.
- Only the batch axis may be ragged. N and K must divide their tiles, which every
  shape here does; anything else is declined rather than masked.
- Tile shape, warps, stages and tile order are tunable per GPU, shape and batch
  bucket. `bench/bench_linear.py --tune` writes them to
  `int8_linear/tuned_configs.json`, which the kernel reads, and `bench/situ_tune.py`
  retunes them in a stand-in for the served model.
- After the first call for a given specialization, launches skip Triton's JIT dispatch
  and relaunch the kernel that dispatch returned, keyed on everything Triton 3.7
  specializes on. `tests/` check that the kept kernel is the one Triton picks.
- [`int8_linear/vllm_patch.py`](int8_linear/vllm_patch.py) puts the kernel into vLLM.
  Under torch.compile it sits behind a custom op, so its configuration comes from each
  call's batch size rather than the one seen while tracing. Run eagerly, vLLM calls it
  directly, because the custom op's dispatch cost more than the launch itself.

vLLM's `triton_scaled_mm` differs in ways this is set up to test. It picks tile
shapes from M alone, from 64x64x256 for small batches to 128x128x128 for large
ones, never tunes warps or stages, and masks every axis unless its
tensor-descriptor loads are in use, which `VLLM_TRITON_USE_TD` controls.

## End to end in vLLM

Qwen3-1.7B W8A8 on the RTX 4090, tokens per second, from the third end-to-end run
([`results/e2e.md`](results/e2e.md)). No other process was on the GPU.

### With torch.compile and CUDA graphs (vLLM's default)

| batch | bf16 | CUTLASS | vLLM Triton | #45126 | this repo | this repo / CUTLASS | this repo / #45126 | this repo / bf16 |
|---|---|---|---|---|---|---|---|---|
| 1 | 227 | 307 | 219 | 315 | 325 | 1.06x | 1.03x | 1.43x |
| 4 | 809 | 1,183 | 854 | 1,216 | 1,252 | 1.06x | 1.03x | 1.55x |
| 16 | 3,089 | 4,407 | 3,267 | 4,558 | 4,671 | 1.06x | 1.02x | 1.51x |
| 32 | 5,988 | 7,612 | 6,243 | 8,564 | 8,756 | 1.15x | 1.02x | 1.46x |
| 48 | 8,348 | 10,298 | 8,794 | 11,865 | 12,019 | 1.17x | 1.01x | 1.44x |
| 64 | 10,541 | 12,810 | 11,082 | 14,646 | 14,774 | 1.15x | 1.01x | 1.40x |
| 128 | 16,940 | 19,042 | 18,187 | 21,088 | 22,668 | 1.19x | 1.07x | 1.34x |
| prefill, 8 x 512 | 47,335 | 79,103 | 94,666 | 80,040 | 102,392 | 1.29x | 1.28x | 2.16x |

- W8A8 pays off here: bf16 decodes at 0.68x to 0.89x of CUTLASS.
- vLLM's Triton kernel as shipped decodes at 0.71x to 0.96x of CUTLASS, but prefills
  at 1.20x.
- This kernel's lead over CUTLASS jumps from 1.06x at batch 16 to 1.15x at batch 32,
  where CUTLASS's slow dispatch buckets begin ([below](#what-causes-the-cutlass-slowdown)).

### Without torch.compile or CUDA graphs

| batch | bf16 | CUTLASS | this repo | this repo / CUTLASS | this repo / bf16 |
|---|---|---|---|---|---|
| 1 | 72.0 | 59.4 | 52.9 | 0.89x | 0.73x |
| 16 | 1,100.6 | 929.0 | 840.7 | 0.90x | 0.76x |
| 64 | 4,274.4 | 3,636.6 | 3,280.2 | 0.90x | 0.77x |
| prefill, 8 x 512 | 47,390 | 79,634 | 101,133 | 1.27x | 2.13x |

Without graphs bf16 decodes fastest and this kernel slowest. Eager timings move more
between runs than compiled ones, by up to 3.2% for backends that did not change, and across
the second and third runs this kernel decoded at 0.85x to 0.90x of CUTLASS.

Where the time goes inside the served model at batch 1
([`bench/eager_attribution.py`](bench/eager_attribution.py)), in milliseconds of host time
per generated token:

| | CUTLASS | this repo | difference |
|---|---|---|---|
| whole step | 17.262 | 20.138 | +2.876 |
| outside the linear layers | 10.777 | 11.566 | +0.789 |
| apply_weights | 6.485 | 8.573 | +2.089 |
| activation quantizer | 2.510 | 2.697 | +0.187 |
| matmul call | 3.307 | 5.107 | +1.800 |
| rest of apply_weights | 0.667 | 0.768 | +0.101 |

The linear layers account for 2.09 ms of the 2.88 ms longer step, and the matmul call alone
for 1.80 ms: 45.6 microseconds per call against CUTLASS's 29.5. Timed alone through the same
entry point the two differ by 5.2 microseconds ([launch cost](#launch-cost)), so most of the
in-model gap is something the launch benchmark does not reproduce, and it has not been
identified. Another 0.79 ms of the gap is outside the linear layers.

### Quality

| backend | perplexity | mean logprob change vs bf16 (nats), 95% interval | mean absolute per-token gap to CUTLASS |
|---|---|---|---|
| bf16 | 20.44 | | 0.228 |
| CUTLASS | 20.49 | -0.0021 [-0.0093, +0.0050] | 0 |
| vLLM Triton, #45126, this repo | 20.52 | -0.0038 [-0.0109, +0.0032] | 0.156 |

- **The three Triton kernels agree bit for bit.** vLLM's Triton kernel, #45126's and
  this one produced identical logprobs at all 20,440 tokens and identical greedy
  samples, in all three runs. All three accumulate the int8 product exactly in int32 and
  apply the scales in the same order, so their tile choices cannot change the result.
- **CUTLASS's results differ, but not in one direction.** Its per-token logprobs differ
  from the Triton kernels' by a mean 0.156 nats, with a mean difference of -0.0017
  (95% interval -0.0059 to +0.0025).
- **No measurable cost against bf16.** Neither int8 backend's change is distinguishable
  from zero on these 20,440 tokens. The upper ends of the intervals correspond to 0.9%
  (CUTLASS) and 1.1% (Triton) higher perplexity.

### Three runs: what isolated tuning missed, and tuning in a stand-in

The first end-to-end run ([`results/e2e_run1.md`](results/e2e_run1.md)) found two
losses, and both were addressed before the second
([`results/e2e_run2.md`](results/e2e_run2.md)):

- **A gap in the tuned table.** It covered M = 1, 4, 16, 64 and up, so batches of 17 to
  63 used a configuration tuned for 16. At M=33 to 56, gate_up_proj ran at 0.50x to 0.53x
  of CUTLASS. The second run tuned at every batch size vLLM captures CUDA graphs at.
- **The custom op in eager runs.** vLLM called the kernel through its custom op even when
  nothing was being compiled, at 57.9 microseconds per call against 29.5 directly. The
  second run calls it directly unless torch.compile is tracing.

Backends that did not change moved by at most 0.4% between consecutive runs with CUDA
graphs. With CUDA graphs, this kernel's decode in each run:

| batch | run 1: isolated, M = 1, 4, 16, 64 and up | run 2: isolated, vLLM's batch sizes | run 3: stand-in | run 3 vs run 2 | run 3 vs run 1 | largest change of the other four backends, run 2 to 3 |
|---|---|---|---|---|---|---|
| 1 | 325 | 325 | 325 | +0.0% | +0.1% | 0.1% |
| 4 | 1,252 | 1,252 | 1,252 | -0.1% | -0.0% | 0.1% |
| 16 | 4,672 | 4,671 | 4,671 | -0.0% | -0.0% | 0.1% |
| 32 | 8,746 | 8,664 | 8,756 | +1.1% | +0.1% | 0.2% |
| 48 | 11,466 | 11,877 | 12,019 | +1.2% | +4.8% | 0.1% |
| 64 | 14,659 | 14,655 | 14,774 | +0.8% | +0.8% | 0.1% |
| 128 | 22,417 | 20,971 | 22,668 | +8.1% | +1.1% | 0.1% |

In the second run batch 48 improved as the per-layer numbers predicted, but batches 32 and
128 got slower, although in isolation the new configurations were faster at both.
Microseconds per decoder layer, where the first three columns sum the four linear layers as
timed alone by the dispatch sweep in each run:

| batch | four layers, run 1 table | run 2 table | predicted change | measured change in the decode step |
|---|---|---|---|---|
| 32 | 33.4 | 29.2 | -4.1 | +1.2 |
| 48 | 51.6 | 32.3 | -19.3 | -5.2 |
| 128 | 53.7 | 50.3 | -3.5 | +14.1 |

The same gap shows against #45126. At batch 64 this kernel's four layers take 34.1
microseconds per decoder layer alone and #45126's take 44.1, yet both decode steps take
4.37 ms. The per-layer benchmark replays each kernel alone and back to back; inside the
model each layer runs between other work.

**A stand-in for the model.** [`bench/situ_tune.py`](bench/situ_tune.py) replays 28 decoder
layers of the same four int8 linear layers as one CUDA graph, each layer with its own random
weights and with vLLM's activation quantizer, RMSNorm, residual adds and SiLU-and-mul between
them; attention is left out. Before tuning anything it had to reproduce the served runs, on
criteria written into the script before it ran: the direction of the change at batch 128
and at batch 48, and a gap to #45126 at batch 64 under half the per-layer benchmark's.
Microseconds per decoder layer:

| batch | run 1: isolated, M = 1, 4, 16, 64 and up | run 2: isolated, vLLM's batch sizes | run 3: stand-in | run 3 vs run 2 | run 3 vs run 1 | largest change of the other four backends, run 2 to 3 |
|---|---|---|---|---|---|---|
| 1 | 325 | 325 | 325 | +0.0% | +0.1% | 0.1% |
| 4 | 1,252 | 1,252 | 1,252 | -0.1% | -0.0% | 0.1% |
| 16 | 4,672 | 4,671 | 4,671 | -0.0% | -0.0% | 0.1% |
| 32 | 8,746 | 8,664 | 8,756 | +1.1% | +0.1% | 0.2% |
| 48 | 11,466 | 11,877 | 12,019 | +1.2% | +4.8% | 0.1% |
| 64 | 14,659 | 14,655 | 14,774 | +0.8% | +0.8% | 0.1% |
| 128 | 22,417 | 20,971 | 22,668 | +8.1% | +1.1% | 0.1% |

At batch 64 it put this kernel 0.2 microseconds per decoder layer ahead of #45126, as served,
where the per-layer benchmark had said 10.0. It did less well against CUTLASS, putting this
kernel 15.4 ahead against 22.3 served; that comparison was reported but not required.

**Tuning in the stand-in.** At each decode batch size up to 256 it tried, per layer, the
current configuration, the first run's and the five best from the per-layer sweep, then
re-timed the chosen set against the starting one and kept it only if it was at least 1%
faster. It kept new sets at 20 of 35 batch sizes, 1.0% to 16.4% faster in the stand-in, and
none at batch 1 to 16. The third end-to-end run used that table, and the changes the
stand-in predicted held in the served model:

| batch | stand-in, run 2 table | stand-in, run 3 table | stand-in change | served change |
|---|---|---|---|---|
| 32 | 78.9 | 77.4 | -1.5 | -1.4 |
| 48 | 81.7 | 80.0 | -1.7 | -1.7 |
| 64 | 83.0 | 81.6 | -1.4 | -1.3 |
| 128 | 101.3 | 85.4 | -15.8 | -16.3 |

The third run's table is the one in `int8_linear/tuned_configs.json`; the first two are
kept as `results/tuned_configs_run1.json` and `results/tuned_configs_run2.json`.

## What causes the CUTLASS slowdown

The per-layer benchmark found CUTLASS's time roughly doubling from M=16 to M=64 at every
Qwen3-1.7B shape. On Ada, vLLM's source (`csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c2x_sm89_int8_dispatch.cuh`)
compiles one CUTLASS configuration per bucket of next_pow_2(M), floored at 16: [1, 16],
(16, 32], (32, 64], (64, 128], (128, 256] and above. Within a bucket it picks by
next_pow_2(N).

[`bench/cutlass_cliff.py`](bench/cutlass_cliff.py) states its prediction in its
docstring, written before it ran: if those configurations cause the slowdown, CUTLASS's
time jumps between adjacent M at a bucket edge and stays flat inside buckets, while the
Triton kernels cross the same edges smoothly. gate_up_proj, whose next_pow_2(N) is 16384,
gets a different configuration in those buckets, so it need not jump where the others do.

That held. Microseconds per layer under CUDA-graph replay, second run; the first run gave
the same steps to two decimals:

| layer | N | CUTLASS M=16 | M=17 | M=32 | M=33 | M=64 | CUTLASS 16 to 17 | CUTLASS 32 to 33 | vLLM Triton 16 to 17 | #45126 16 to 17 |
|---|---|---|---|---|---|---|---|---|---|---|
| q_proj, o_proj | 2048 | 4.95 | 7.82 | 7.32 | 8.99 | 9.26 | 1.58x | 1.23x | 1.06x | 1.01x |
| k_proj, v_proj | 1024 | 4.51 | 6.92 | 7.07 | 8.75 | 9.09 | 1.54x | 1.24x | 1.05x | 1.00x |
| qkv_proj | 4096 | 5.35 | 10.27 | 10.53 | 12.31 | 12.92 | 1.92x | 1.17x | 1.06x | 1.00x |
| gate_proj, up_proj | 6144 | 6.71 | 13.36 | 13.78 | 15.04 | 16.24 | 1.99x | 1.09x | 1.01x | 0.99x |
| gate_up_proj | 12288 | 10.73 | 11.41 | 12.07 | 13.22 | 14.52 | 1.06x | 1.09x | 1.04x | 0.99x |
| down_proj | 2048 | 9.18 | 12.05 | 12.28 | 16.03 | 16.28 | 1.31x | 1.31x | 1.07x | 1.00x |

- Inside a bucket CUTLASS barely moves: from M=17 to 32, and from 33 to 64, its time
  changes by at most 10% at any shape while M nearly doubles.
- vLLM's own Triton kernel steps where its tile heuristic changes instead: 1.13x to 1.94x
  from M=64 to 65 and 1.61x to 1.86x from 128 to 129.
- **One part of the prediction did not hold.** bf16 was meant as a second smooth
  control, but its time also jumps at those sizes (0.53x to 1.24x from M=16 to 17).
  Only the Triton kernels were smooth.
- **The practical effect.** CUTLASS is slower than bf16 at 43 of the sweep's 120 points,
  all at M=17 or above. vLLM pads each decode batch up to a captured batch size, so
  every batch of 17 to 64 lands in those buckets.
- **Not established here:** why the (16, 32] and (32, 64] configurations are slow. The
  sweep ties the slowdown to them, not to a mechanism inside them.
- [`tools/cutlass_bucket_repro.py`](tools/cutlass_bucket_repro.py) reproduces the step with
  only torch and vLLM ([its output](results/cutlass_bucket_repro.txt)). Reported upstream as
  [vllm-project/vllm#56924](https://github.com/vllm-project/vllm/issues/56924).

## Launch cost

q_proj at M=1, microseconds per call ([`bench/launch_overhead.py`](bench/launch_overhead.py)).
Host is CPU time with nothing synchronized, back-to-back is what a step without CUDA
graphs pays, and graph is the GPU work alone.

| path | host | back-to-back | graph |
|---|---|---|---|
| bf16 F.linear (cuBLAS) | 10.5 | 10.5 | 6.31 |
| int8 CUTLASS (vLLM) | 24.3 | 24.4 | 4.49 |
| int8 Triton (vLLM) | 45.3 | 45.4 | 5.89 |
| this repo, through Triton's JIT dispatch | 36.7 | 36.8 | 3.93 |
| this repo, cached launch | 27.4 | 27.5 | 3.93 |
| this repo, behind the custom op | 57.9 | 57.9 | 3.93 |
| this repo, through vLLM's entry point | 29.5 | 29.4 | 3.93 |

Skipping Triton's JIT dispatch saves 9.3 microseconds per call. That leaves this kernel
3.1 microseconds behind CUTLASS on host cost, although its GPU work is the smallest of
the four.

## Per-layer results

RTX 4090, vLLM 0.28.0, Triton 3.7.1, torch 2.13.0+cu130, Qwen3-1.7B layer shapes, no bias. Microseconds per layer under CUDA-graph replay, median of five windows, with no other process on the GPU. The worst window spread was 6.1% of its median. Full output, including back-to-back timings and every tuned configuration, is in [`results/`](results/).

**The harness agrees with vLLM's own.** Timed with vLLM's `benchmarks/kernels/benchmark_int8_gemm.py` at the same shapes, this benchmark's numbers match within a median ratio of 1.00 for bf16 (range 0.95 to 1.11), 1.00 for CUTLASS (0.91 to 1.09), and 1.00 for CUTLASS with activation quantization (0.93 to 1.09).

**The tuned Triton kernel is faster than vLLM's CUTLASS kernel at all 28 points,** by a median 1.26x (1.05x to 2.05x). It is faster than vLLM's Triton kernel at all 28 (median 1.42x, 1.04x to 3.32x) and than bf16 cuBLAS at all 28 (median 2.39x, 1.27x to 3.51x). With vLLM's per-token activation quantizer inside the timing it is still ahead of CUTLASS at all 28 points (median 1.21x, 1.04x to 2.02x). At M=4096 it runs at 538 to 594 TOPS, against 278 to 489 for CUTLASS.

**Confirmed twice more.** Rerun without tuning, with this repo's kernel reading its
configurations from the stored table, it was again faster than CUTLASS at all 28 points
(median 1.29x, 1.05x to 2.24x) and than bf16 at all 28 (median 2.39x). Its time with the
stored table was a median 0.999 of the time the sweep had picked, so choosing and reporting
from the same run added no measurable optimism. Between those two runs CUTLASS timings moved
by a median 2%, and bf16 by less than 0.1%. A third run, which added the #45126 column, gave
28 of 28 again (median 1.29x, 1.05x to 2.25x).

**CUTLASS at M=64.** At M=64 it is slower than bf16 for 3 of 4 shapes (q_proj 0.81x, k_proj 0.67x, down_proj 0.92x), and for all 4 with activation quantization (q_proj 0.71x, k_proj 0.59x, gate_proj 0.93x, down_proj 0.82x). vLLM's own benchmark shows the same. The slowdown starts at M=17 ([above](#what-causes-the-cutlass-slowdown)).

**#45126's tables do not beat CUTLASS at these batch sizes.** Its kernel is a median 0.99x
of CUTLASS on these shapes (faster at 14 of 28), 0.86x on the layers as vLLM merges them
(9 of 28) and 0.86x on Llama-3-8B (8 of 28). It is weakest at small M, at 0.66x to 0.79x of
CUTLASS for M=1 to 16 on q_proj, k_proj and down_proj, and ahead at M=64 and 256 on all four
shapes. This kernel is faster than #45126's at 24 of 28 points (median 1.55x), and within
2% at the other four.

**At vLLM's batch sizes.** Tuned on the layers as vLLM merges them, at vLLM's
51 CUDA-graph batch sizes plus 1024, 2048 and 4096 (216 points), this kernel is faster than
CUTLASS at all 216 (median 1.44x, 1.02x to 1.99x), than vLLM's Triton kernel at all 216
(median 1.61x) and than bf16 at all 216 (median 2.43x). With activation quantization it is
ahead of CUTLASS at all 216 (median 1.39x). It is faster than #45126's at 201 (median 1.31x),
and within 2% at the other 15. Across these batch sizes #45126 is a median 1.08x of CUTLASS
(faster at 143), because more of them fall where CUTLASS is slow. The worst window spread in
this run was 13.7%.

**Llama-3-8B.** Tuned on its merged layer shapes, this kernel is faster than CUTLASS at 27 of
28 points (median 1.14x, 0.95x to 1.62x), losing at down_proj M=16. It is faster than bf16 at
all 28 (median 2.68x), and with activation quantization ahead of CUTLASS at 27 of 28 (median
1.13x). Against #45126 it is faster at 25 of 28 (median 1.37x), losing at M=64 on qkv_proj,
o_proj and down_proj (0.90x to 0.95x).

vLLM's own Triton kernel is a median 0.93x of CUTLASS on the Qwen3-1.7B shapes. It is weakest at M=256, where its fixed tile heuristic leaves it at 0.42x to 0.54x of CUTLASS on three of the four shapes; on gate_proj it is 1.06x.

### q_proj, o_proj: K=2048, N=2048

| M | bf16 | CUTLASS | vLLM Triton | this repo, tuned | tuned over CUTLASS |
|---|---|---|---|---|---|
| 1 | 6.29 | 4.87 | 6.41 | 3.92 | 1.24x |
| 4 | 9.50 | 4.91 | 6.43 | 3.93 | 1.25x |
| 16 | 10.1 | 5.02 | 6.25 | 4.27 | 1.18x |
| 64 | 8.21 | 10.1 | 6.57 | 5.37 | 1.88x |
| 256 | 16.9 | 11.3 | 21.0 | 8.91 | 1.27x |
| 1024 | 52.0 | 26.7 | 23.3 | 18.3 | 1.46x |
| 4096 | 199.5 | 101.6 | 88.1 | 62.3 | 1.63x |

### k_proj, v_proj: K=2048, N=1024

| M | bf16 | CUTLASS | vLLM Triton | this repo, tuned | tuned over CUTLASS |
|---|---|---|---|---|---|
| 1 | 4.71 | 4.40 | 5.85 | 3.53 | 1.25x |
| 4 | 8.58 | 4.45 | 5.90 | 3.56 | 1.25x |
| 16 | 8.96 | 4.44 | 5.61 | 3.70 | 1.20x |
| 64 | 6.02 | 8.97 | 6.06 | 4.64 | 1.93x |
| 256 | 11.4 | 10.2 | 20.9 | 6.30 | 1.63x |
| 1024 | 28.1 | 21.5 | 21.7 | 13.0 | 1.65x |
| 4096 | 102.0 | 53.2 | 45.4 | 31.9 | 1.67x |

### gate_proj, up_proj: K=2048, N=6144

| M | bf16 | CUTLASS | vLLM Triton | this repo, tuned | tuned over CUTLASS |
|---|---|---|---|---|---|
| 1 | 11.3 | 5.72 | 6.09 | 4.49 | 1.28x |
| 4 | 10.2 | 5.90 | 6.30 | 4.95 | 1.19x |
| 16 | 12.3 | 6.54 | 6.15 | 5.31 | 1.23x |
| 64 | 15.3 | 15.0 | 7.82 | 7.50 | 2.01x |
| 256 | 49.3 | 23.5 | 22.2 | 17.6 | 1.33x |
| 1024 | 154.2 | 96.6 | 66.6 | 48.5 | 1.99x |
| 4096 | 605.4 | 370.3 | 257.6 | 180.4 | 2.05x |

### down_proj: K=6144, N=2048

| M | bf16 | CUTLASS | vLLM Triton | this repo, tuned | tuned over CUTLASS |
|---|---|---|---|---|---|
| 1 | 13.1 | 9.14 | 14.5 | 7.61 | 1.20x |
| 4 | 23.6 | 9.10 | 14.5 | 7.62 | 1.19x |
| 16 | 25.2 | 9.23 | 13.7 | 8.13 | 1.14x |
| 64 | 14.5 | 15.8 | 14.2 | 11.4 | 1.38x |
| 256 | 47.8 | 24.1 | 57.1 | 23.0 | 1.05x |
| 1024 | 159.4 | 55.7 | 59.9 | 47.2 | 1.18x |
| 4096 | 608.0 | 210.7 | 234.1 | 173.5 | 1.21x |

## Fusing the activation quantizer into the kernel before it

Every int8 linear layer quantizes its input per token right before the matmul. In vLLM
0.28 that is a kernel of its own, `dynamic_scaled_int8_quant`, launched after the RMSNorm
or SiLU-and-mul that produced the input: one more launch, and one more write and read of
the activation. vLLM fuses this pair for FP8 models, but its RMSNorm-quant compile pass
matches FP8 and NVFP4 quantizers and no int8 one, so int8 models run both kernels even though vLLM's own fused
CUDA kernel, `rms_norm_dynamic_per_token_quant`, accepts int8 output. vLLM's main branch
is the same; [vllm-project/vllm#38026](https://github.com/vllm-project/vllm/pull/38026)
tried int8 by another route and was closed unmerged.

`int8_linear/fused_quant.py` has two Triton kernels, add+RMSNorm+quantize and
SiLU-and-mul+quantize, one program per token row, that follow vLLM's arithmetic step by
step. `int8_linear/vllm_fuse_patch.py` lets vLLM's int8 kernels (Triton and CUTLASS) take
a pre-quantized activation and routes Qwen3's decoder layer through the fused kernels,
for 3 of each layer's 4 matmuls (o_proj still quantizes attention's output itself): 84
matmuls per forward pass.

**Numerics.** The SiLU-and-mul kernel's int8 output is bit-identical to vLLM's
`silu_and_mul` followed by `dynamic_scaled_int8_quant`. The RMSNorm kernel differs from
vLLM's unfused pair on 0.2% to 1.1% of int8 values, by at most two steps, which is the
same share vLLM's own fused CUDA kernel differs by, at every size measured; the residual
stream is bit-identical. `bench/fuse_quant.py` picked the matching division and SiLU
rounding before anything was timed.

**Kernel time,** microseconds per call, as CUDA-graph replays of 100 calls:

| tokens | RMSNorm + quantize: vLLM, two kernels | vLLM's fused CUDA kernel | this repo | SiLU-and-mul + quantize: vLLM, two kernels | this repo |
|---|---|---|---|---|---|
| 1 | 2.90 | 2.23 | **1.66** | 3.51 | **2.40** |
| 64 | 3.21 | 2.46 | **1.91** | 3.90 | **2.63** |
| 128 | 3.56 | 2.71 | **2.18** | 4.45 | **2.80** |
| 1,024 | 8.79 | 14.09 | **5.59** | 15.33 | **10.17** |
| 4,096 | 24.59 | 48.58 | **13.88** | 211.76 | **136.98** |

vLLM's fused CUDA kernel is faster than its two unfused kernels at decode sizes and slower
from 1,024 tokens up.

**Served,** with a prediction written down first. The 28-layer stand-in, rebuilt to match
the served path (vLLM's fused add-RMSNorm, quantizer and SiLU-and-mul, this repo's matmul),
saved 3.4 to 5.6 microseconds per decoder layer, which applied to the last run's decode
step predicted 2.8% to 3.2% more tokens per second. Then each backend was served twice,
alternating, with and without the fusion (medians of the two rounds, which agreed within
0.55% on decode and 0.71% on prefill, with identical logprobs):

| batch | predicted | CUTLASS: unfused -> fused | this repo: unfused -> fused |
|---|---|---|---|
| 1 | +3.2% | 312 -> 318 (+1.8%) | 325 -> 337 (+3.5%) |
| 4 | +3.2% | 1,201 -> 1,226 (+2.1%) | 1,254 -> 1,302 (+3.8%) |
| 16 | +3.1% | 4,459 -> 4,558 (+2.2%) | 4,670 -> 4,865 (+4.2%) |
| 32 | +2.8% | 7,692 -> 7,782 (+1.2%) | 8,752 -> 9,063 (+3.5%) |
| 48 | +2.9% | 10,357 -> 10,531 (+1.7%) | 12,028 -> 12,385 (+3.0%) |
| 64 | +3.1% | 12,901 -> 13,122 (+1.7%) | 14,794 -> 15,201 (+2.8%) |
| 128 | +2.9% | 19,120 -> 19,449 (+1.7%) | 22,652 -> 23,231 (+2.6%) |

Prefill, 8 prompts of 512 tokens: +6.8% with CUTLASS and +10.1% with this kernel.

**Quality.** The fused runs do not reproduce the unfused runs' logprobs token for token:
with per-token int8 quantization, a one-step difference in a few values moves later
layers, and the per-token logprob gap is 0.19 on average. The same happened to CUTLASS by
itself between the earlier published run and this one, with no code change (a driver
update came in between): its per-token gap is 0.14 and its perplexity moved from 20.49 to
20.58. Over the 40 WikiText-2 windows, as paired differences in mean logprob per token:

| comparison | difference |
|---|---|
| fused minus unfused, CUTLASS | +0.0072 (SE 0.0025) |
| fused minus unfused, this kernel | +0.0057 (SE 0.0029) |
| fused minus bf16, CUTLASS | +0.0005 (SE 0.0036) |
| fused minus bf16, this kernel | +0.0019 (SE 0.0034) |
| CUTLASS today minus CUTLASS in the earlier run, no code change | -0.0046 (SE 0.0018) |

The fused models are indistinguishable from bf16. They score slightly better than their
unfused counterparts, by about as much as unfused CUTLASS moved between two runs on its
own, so I read that as numerical reshuffling rather than a quality gain.

Full tables: `results/fuse.md`. Run it all with `bash tools/run_fuse_vm.sh`.

## Where it loses

- **Without CUDA graphs.** End to end, decode is 0.85x to 0.90x CUTLASS and 0.73x to 0.77x
  bf16. Most of the gap is the matmul call's host time inside the model, three times the gap
  measured alone, for a reason not yet found.
- **Against #45126, end to end.** The lead is 1.01x to 1.03x at batch 1 to 64 and 1.07x at
  128, despite a 1.28x median per layer at decode batch sizes. Most of the per-layer
  advantage does not reach the served model.
- **The stand-in is not the model.** It leaves out attention, and it predicted the gap to
  CUTLASS at batch 64 less well than the rest (15.4 against 22.3 microseconds per decoder
  layer served).
- **Untuned.** The fixed default configuration is a median 1.01x of CUTLASS in the main run
  and slower at 14 of 28 points. The advantage comes from per-shape tuning.
- **Narrow margins.** The smallest per-layer margin over CUTLASS is 1.02x, and on Llama-3-8B
  down_proj at M=16 it loses (0.95x).
- **Scope.** One GPU, one model served end to end, one checkpoint, and 20,440 tokens of
  perplexity, which cannot resolve differences much below 1%.

## How the runs went

1. **First per-layer run** (`results/*_run1_bias.*`). Checking it found two problems, both
   in this benchmark rather than in vLLM:
   - **vLLM's `triton_scaled_mm` is not wrong.** It failed the float64 check at nearly
     every shape, but `bench/diagnose_vllm_triton.py` shows its int32 product is exact with
     the weight in either layout and with tensor-descriptor loads on or off. It rounds the
     scaled product to bf16 and then adds the bias, so where the two nearly cancel, the
     output lands a rounding step of the product away from a near-zero sum, sometimes at
     exactly zero. The check divided by that sum. It now divides by the larger of the sum
     and its matmul term.
   - **The first run slowed CUTLASS down.** Its layers carried a bias, which the Qwen3
     layers do not have. Against vLLM's own `benchmarks/kernels/benchmark_int8_gemm.py` on
     the same GPU, bf16 timings agreed (median ratio 1.00) while CUTLASS measured a median
     1.33x slower here, up to 1.71x. Bias is now off by default.
2. **Rerun with bias off and the corrected check.** The per-layer tables above.
3. **Confirmation run** with stored configurations (`results/*_confirm.*`).
4. **First end-to-end run** (`results/*_run1.*`), with the dispatch sweep, launch cost, and
   #45126 and Llama-3-8B per layer. It found the two losses described above.
5. **Second end-to-end run** (`results/*_run2.*`) after both changes, with tuning at vLLM's
   batch sizes. All 39 tests passed. The dispatch sweep and launch numbers above are from
   this run.
6. **Third end-to-end run** with the table tuned in the stand-in, after the stand-in passed
   its check, plus the in-model host timing and the standalone CUTLASS repro. All 39 tests
   passed. These are the end-to-end numbers above.

## Next

- Find why the matmul call costs three times more host time inside the served model than
  alone.
- Add attention to the stand-in, and check it against CUTLASS as well.
- A second GPU.
- Follow [vllm-project/vllm#56924](https://github.com/vllm-project/vllm/issues/56924), the
  CUTLASS bucket report, and time any patch it leads to on this GPU.
- Share the Ada measurements alongside [#45126](https://github.com/vllm-project/vllm/pull/45126).
- Report the int8 fusion gap and the fused CUDA kernel's prefill slowdown to vLLM.

## Reproduce

On a CUDA machine with torch, Triton and vLLM:

```bash
bash tools/run_vm.sh              # tests, tuning sweep and per-layer benchmark
bash tools/verify_vm.sh           # vLLM Triton diagnostic and vLLM's own benchmark
python bench/bench_linear.py --tag confirm   # stored configurations, no tuning
bash tools/run_e2e_vm.sh          # tuning at vLLM's batch sizes, dispatch sweep, launch cost, end to end, #45126 and Llama per layer
python tools/cutlass_bucket_repro.py         # the CUTLASS step with only torch and vLLM
python bench/situ_tune.py                    # stand-in check, then tuning in it
bash tools/run_fuse_vm.sh         # fused quantizer: numerics, kernel times, prediction, end to end
```

## License

MIT, except [`int8_linear/pr45126.py`](int8_linear/pr45126.py), which is vendored from
vLLM and keeps its Apache-2.0 header.
