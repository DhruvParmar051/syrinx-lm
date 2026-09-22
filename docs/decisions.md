# Decisions

Append-only. Newest at the bottom. Format:

```
## YYYY-MM-DD — <decision>
**Context:** what forced the choice
**Decision:** what we chose
**Why:** the reasoning (link evidence in experiments/ where it exists)
**Revisit if:** the observation that would reopen it
```

---

## 2026-09-22 — Architecture
**Context:** We need a small decoder-only LM whose weights load into HF `LlamaForCausalLM` without `trust_remote_code`.
**Decision:** Pre-norm blocks with RMSNorm, RoPE, SwiGLU MLP, causal MHA (n_kv_heads = n_heads), no biases,
tied input/output embeddings, vocab V = 32,768 (byte-level BPE), context 1024.
**Why:** It matches Llama's layout one-to-one, so the Phase 9 export is a rename and not a conversion.
Tying the embeddings saves V·d params, which is a large share of the budget at small d.
**Revisit if:** the Phase 9 parity test shows a mismatch that can't be fixed by renaming.

## 2026-09-22 — RoPE convention
**Context:** There are two RoPE pairing conventions: interleaved (GPT-NeoX/GPT-J style) and split-halves (`rotate_half`).
**Decision:** Split-halves (`rotate_half`), matching HF Llama.
**Why:** With interleaved pairing, the export would need a permutation of q/k weights, which is a silent-bug risk.
**Revisit if:** never, unless the export target changes.

## 2026-09-22 — Model shapes
**Context:** PLAN.md pins the total param counts (11,798,784 / 50,866,688 / 124,275,456).
Solving `N = V·d + L·(4d² + 3·d·ff + 2d) + d` gives a unique fit with ff a multiple of 64.
**Decision:**

| Config | d_model | n_layers | d_ff | n_heads | head_dim |
|---|---|---|---|---|---|
| syrinx-10m  | 256 | 4  | 768  | **open** (4 → 64) | — |
| syrinx-50m  | 512 | 10 | 1536 | 8  | 64 |
| syrinx-125m | 768 | 12 | 2560 | 12 | 64 |

**Why:** The 50m and 125m shapes were confirmed by building HF Llama at these shapes, which produced
exactly the pinned counts (`experiments/feasibility-2026-09-22.md`). Head count doesn't change the
param count, so it's a free choice for 10m.
**Revisit if:** the 10m head count is decided (record it here).

## 2026-09-22 — Micro-batch 8 for the 125M run
**Context:** The proxy benchmark measured 20.1 GiB at mb 8 and 35.2 GiB at mb 16, against a
`torch.mps.recommended_max_memory()` of 37.44 GiB.
**Decision:** Micro-batch 8, and reach the global batch through gradient accumulation.
**Why:** mb 16 leaves ~2 GiB of headroom on a machine that also runs the OS and other apps. That's
only +7% tok/s (7,165 vs 6,674) for a real risk of OOM or swapping partway through a multi-day run.
**Revisit if:** our own model's measured peak at mb 8 is well below the proxy's.

## 2026-09-22 — License
**Decision:** Apache-2.0 for code and weights.
**Revisit if:** a training-data license requires something different (check in `docs/data.md`).

---

## Open (not yet decided — decide before Phase 5)
- Global batch in tokens/step (starting point ~0.25M for 50M, ~0.5M for 125M).
- Peak LR, min LR, warmup steps, weight decay, AdamW betas/eps.
- Validation token count and eval interval.
- 10m head count.
- Whether `<|pad|>` is actually needed (random-window packing doesn't pad).
