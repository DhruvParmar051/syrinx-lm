# Feasibility probe — 2026-09-22 (pre-Phase-0, proxy measurements)

**Status: PROXY.** These numbers use HF `LlamaForCausalLM` (SDPA attention) at Syrinx shapes,
not Syrinx code. They bound what is achievable; Gate 6a must re-measure with our own model.
Burst measurements (15 timed steps after 4 warmup), NOT sustained — thermal behaviour over hours is unmeasured.

## Machine (command output)
- `sysctl`: Apple M5 Pro, 18 CPUs, hw.memsize 51539607552 (48 GiB); macOS 27.0 (26A428)
- `df -h .`: 482 GiB free
- torch 2.14.0, MPS available; `torch.mps.recommended_max_memory()` = 37.44 GiB

## Matmul (4096², 30 iters, synchronized)
| dtype | TFLOPS |
|---|---|
| fp32 | 7.43 |
| fp16 | 30.68 |
| bf16 | 30.83 |

bf16 autocast probe: SDPA(causal) + 768→32768 matmul + CE, fwd+bwd on MPS → OK, loss and grads finite.

## Training step throughput (seq 1024, AdamW, grad clip; script: `feasibility_tput_proxy.py`)
| model | params | micro-batch | autocast bf16 | tok/s | ms/step | driver mem |
|---|---|---|---|---|---|---|
| 125m | 124,275,456 | 8  | no  | 4,699  | 1743 | 21.3 GiB |
| 125m | 124,275,456 | 8  | yes | 6,674  | 1228 | 20.1 GiB |
| 125m | 124,275,456 | 16 | yes | 7,165  | 2287 | 35.2 GiB |
| 50m  | 50,866,688  | 16 | yes | 13,362 | 1226 | 24.2 GiB |

Proxy param count equals the PLAN.md pinned count for 125m and 50m → shapes confirmed:
10m d=256 L=4 ff=768 · 50m d=512 L=10 ff=1536 (8 heads) · 125m d=768 L=12 ff=2560 (12 heads).

## Derived (arithmetic on the rows above)
- 125m, 2.5B tokens: bf16 mb8 → 2.5e9/6,674 = 4.3 days; mb16 → 4.0 days; fp32 → 6.2 days.
- 50m, 1.0B tokens: 1e9/13,362 = 20.8 h.
- Model-FLOPs utilisation at 125m mb8 bf16: 6·124.3M·6,674 ≈ 5.0 TFLOPS ≈ 16% of the bf16 matmul peak.
- Memory: plan estimate for 125m mb8 was ~16 GiB; proxy driver-allocated is 20.1 GiB (+26%).
  mb16 (35.2 GiB) is within 2.3 GiB of the recommended MPS ceiling → do not use mb16 for the long run.
