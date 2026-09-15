# Stand-in model on NVIDIA GeForce RTX 4090

torch 2.13.0+cu130, Triton 3.7.1, vLLM 0.28.0. Microseconds per decoder layer: the stand-in's CUDA-graph replay, and the served decode step from results/e2e_*_run1.json and _run2.json.

## Check against the served runs

| batch | stand-in, run 1 table | stand-in, run 2 table | stand-in change | served change |
|---|---|---|---|---|
| 32 | 77.4 | 79.0 | +1.6 | +1.2 |
| 48 | 86.8 | 81.9 | -4.9 | -5.2 |
| 128 | 87.5 | 101.4 | +13.9 | +14.1 |

At batch 64, this kernel minus the other, per decoder layer:

| other | per-layer benchmark | stand-in | served |
|---|---|---|---|
| #45126 | -10.0 | -0.2 | -0.2 |
| CUTLASS | -18.9 | -15.4 | -22.3 |

Criteria, fixed before the run:

- batch 128: run 2 table slower than run 1, as served: yes
- batch 48: run 2 table faster than run 1, as served: yes
- batch 64: gap to #45126 under half the per-layer gap: yes
- (reported only) batch 32: direction of the run 2 change matches the served model: yes

**Check passed.**

## Tuning in the stand-in

| batch | layers changed | start | chosen | kept |
|---|---|---|---|---|
| 1 | gate_up_proj | 73.1 | 73.0 | no |
| 2 | gate_up_proj | 75.7 | 75.6 | no |
| 4 | gate_up_proj | 75.8 | 75.7 | no |
| 8 | gate_up_proj | 76.1 | 76.0 | no |
| 16 | none | | | no |
| 24 | qkv_proj, o_proj, gate_up_proj, down_proj | 79.1 | 77.5 | yes |
| 32 | qkv_proj, o_proj, gate_up_proj, down_proj | 78.9 | 77.4 | yes |
| 40 | qkv_proj, o_proj, gate_up_proj, down_proj | 81.6 | 80.1 | yes |
| 48 | qkv_proj, o_proj, gate_up_proj, down_proj | 81.7 | 80.0 | yes |
| 56 | qkv_proj, o_proj, gate_up_proj, down_proj | 82.5 | 81.1 | yes |
| 64 | qkv_proj, o_proj, gate_up_proj | 83.0 | 81.6 | yes |
| 72 | o_proj | 86.0 | 85.2 | no |
| 80 | qkv_proj, gate_up_proj | 97.2 | 83.7 | yes |
| 88 | qkv_proj, o_proj | 82.5 | 81.7 | no |
| 96 | o_proj, gate_up_proj | 95.4 | 82.0 | yes |
| 104 | o_proj, down_proj | 87.7 | 83.7 | yes |
| 112 | o_proj, gate_up_proj, down_proj | 100.9 | 84.4 | yes |
| 120 | o_proj, gate_up_proj, down_proj | 101.5 | 85.0 | yes |
| 128 | o_proj, gate_up_proj, down_proj | 101.3 | 85.4 | yes |
| 136 | o_proj, gate_up_proj, down_proj | 102.7 | 88.5 | yes |
| 144 | qkv_proj, o_proj, gate_up_proj, down_proj | 103.1 | 99.4 | yes |
| 152 | o_proj, gate_up_proj | 101.4 | 89.6 | yes |
| 160 | o_proj, gate_up_proj | 102.2 | 91.0 | yes |
| 168 | gate_up_proj, down_proj | 104.5 | 101.5 | yes |
| 176 | o_proj, gate_up_proj | 103.7 | 102.4 | yes |
| 184 | o_proj, gate_up_proj | 104.7 | 103.6 | yes |
| 192 | qkv_proj, gate_up_proj | 104.2 | 103.9 | no |
| 200 | gate_up_proj, down_proj | 108.9 | 107.9 | no |
| 208 | down_proj | 110.1 | 109.9 | no |
| 216 | qkv_proj | 110.8 | 110.8 | no |
| 224 | gate_up_proj, down_proj | 112.8 | 112.0 | no |
| 232 | gate_up_proj | 113.6 | 112.7 | no |
| 240 | qkv_proj, gate_up_proj | 114.9 | 113.7 | no |
| 248 | gate_up_proj, down_proj | 117.1 | 114.7 | yes |
| 256 | down_proj | 115.5 | 115.2 | no |

Other processes on the GPU during the run: none.
