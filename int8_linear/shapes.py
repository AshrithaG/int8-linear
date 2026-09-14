"""Linear-layer shapes, as (layers, K, N) for C[M, N] = A[M, K] @ W[N, K]^T."""

SHAPES = {
    # Qwen3-1.7B: hidden 2048, intermediate 6144, 16 query and 8 KV heads of 128.
    "qwen3-1.7b": [
        ("q_proj, o_proj", 2048, 2048),
        ("k_proj, v_proj", 2048, 1024),
        ("gate_proj, up_proj", 2048, 6144),
        ("down_proj", 6144, 2048),
    ],
    # What vLLM actually runs for Qwen3-1.7B. It merges q, k and v into one
    # projection and gate with up, so two of these differ from the view above.
    "qwen3-1.7b-vllm": [
        ("qkv_proj", 2048, 4096),
        ("o_proj", 2048, 2048),
        ("gate_up_proj", 2048, 12288),
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
