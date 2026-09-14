"""Swap this repo's kernel into vLLM's Triton int8 linear path.

vLLM's TritonInt8ScaledMMLinearKernel calls ``triton_scaled_mm(input, weight,
scale_a=..., scale_b=..., out_dtype=..., bias=...)`` through a name in its own
module. Replacing that one name changes the matmul kernel and nothing else: weight
processing, the activation quantizer and the dispatch around it stay vLLM's.

Under torch.compile the kernel sits behind a torch custom op. vLLM traces models
with torch.compile, and Python that picks a tile configuration from the batch size
would run once at trace time and freeze, the trap vLLM's own batch-invariant matmul
fell into (vllm-project/vllm#54243). Behind an opaque op the configuration is chosen
from the real batch size on every call. Run eagerly, the model calls the kernel
directly instead: dispatching a Python custom op roughly doubled the host cost of
each call (bench/launch_overhead.py). A shape the kernel declines goes to the
triton_scaled_mm the patch replaced, and FALLBACKS counts it.

The patch must be applied in the process that builds the model, so vLLM's engine
has to run in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0).
"""

from typing import Optional

import torch

from int8_linear.kernel import NotSupported, choose_config, tuned_config, w8a8_mm

# Calls that reached this kernel, calls it declined, and the configuration each
# (M, K, N) used. Under CUDA graphs only warmup and capture run Python, so a patched
# model that never reached the kernel shows up as CALLS == 0.
CALLS = 0
FALLBACKS = 0
CONFIGS: dict[tuple[int, int, int], str] = {}
_vllm_triton_scaled_mm = None


def _run(a_q: torch.Tensor, b_q: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor,
         out_dtype: torch.dtype, bias: Optional[torch.Tensor]) -> torch.Tensor:
    global CALLS, FALLBACKS
    m, k = a_q.shape
    n = b_q.shape[1]
    cfg = choose_config(m, n, k)
    CALLS += 1
    CONFIGS.setdefault((m, k, n), cfg.name)
    try:
        return w8a8_mm(a_q, b_q, scale_a, scale_b, out_dtype, bias, config=cfg)
    except NotSupported:
        if _vllm_triton_scaled_mm is None:
            raise
        FALLBACKS += 1
        return _vllm_triton_scaled_mm(a_q, b_q, scale_a, scale_b, out_dtype, bias)


@torch.library.custom_op("int8_linear::w8a8_mm", mutates_args=())
def w8a8_mm_op(a_q: torch.Tensor, b_q: torch.Tensor, scale_a: torch.Tensor,
               scale_b: torch.Tensor, out_dtype: torch.dtype,
               bias: Optional[torch.Tensor]) -> torch.Tensor:
    return _run(a_q, b_q, scale_a, scale_b, out_dtype, bias)


@w8a8_mm_op.register_fake
def _(a_q, b_q, scale_a, scale_b, out_dtype, bias):
    return a_q.new_empty((a_q.shape[0], b_q.shape[1]), dtype=out_dtype)


def scaled_mm(input: torch.Tensor, weight: torch.Tensor, scale_a: torch.Tensor,
              scale_b: torch.Tensor, out_dtype: torch.dtype,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Drop-in replacement with vLLM's triton_scaled_mm signature: the custom op
    while torch.compile traces, the kernel directly otherwise."""
    if torch.compiler.is_compiling():
        return w8a8_mm_op(input, weight, scale_a, scale_b, out_dtype, bias)
    return _run(input, weight, scale_a, scale_b, out_dtype, bias)


def patch_vllm() -> None:
    global _vllm_triton_scaled_mm
    import vllm.model_executor.kernels.linear.scaled_mm.triton as vllm_triton_kernel

    if vllm_triton_kernel.triton_scaled_mm is not scaled_mm:
        _vllm_triton_scaled_mm = vllm_triton_kernel.triton_scaled_mm
    # Load the stored table now, so no file is read inside a traced or captured region.
    tuned_config(1, 1, 1)
    vllm_triton_kernel.triton_scaled_mm = scaled_mm
