# Host time of an eager decode step on NVIDIA GeForce RTX 4090

vLLM 0.28.0, torch 2.13.0+cu130. Batch 1, 128 generated tokens, median of 5 runs, milliseconds per token. Linear-layer parts are host time: kernels launched from them run on the GPU asynchronously.

| | CUTLASS | this repo | difference |
|---|---|---|---|
| whole step | 17.262 | 20.138 | +2.876 |
| outside the linear layers | 10.777 | 11.566 | +0.789 |
| apply_weights | 6.485 | 8.573 | +2.089 |
| activation quantizer | 2.510 | 2.697 | +0.187 |
| matmul call | 3.307 | 5.107 | +1.800 |
| rest of apply_weights | 0.667 | 0.768 | +0.101 |

Calls per token: CUTLASS {'activation quantizer': 112.0, 'matmul call': 112.0, 'apply_weights': 112.0}; this repo {'activation quantizer': 112.0, 'matmul call': 112.0, 'apply_weights': 112.0}.
