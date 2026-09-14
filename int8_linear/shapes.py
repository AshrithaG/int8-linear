"""Linear-layer shapes, as (layers, K, N) for C[M, N] = A[M, K] @ W[N, K]^T."""

SHAPES = {
    # Qwen3-1.7B: hidden 2048, intermediate 6144, 16 query and 8 KV heads of 128.
    "qwen3-1.7b": [
        ("q_proj, o_proj", 2048, 2048),
        ("k_proj, v_proj", 2048, 1024),
        ("gate_proj, up_proj", 2048, 6144),
        ("down_proj", 6144, 2048),
    ],
    # The TP1 shapes vLLM's own benchmarks/kernels/weight_shapes.py lists for
    # Llama-3-8B, with QKV and gate/up merged the way vLLM runs them.
    "llama-3-8b": [
        ("qkv_proj", 4096, 6144),
        ("o_proj", 4096, 4096),
        ("gate_up_proj", 4096, 28672),
        ("down_proj", 14336, 4096),
    ],
}
