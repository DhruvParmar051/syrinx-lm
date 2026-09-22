# Syrinx — Execution Plan (local-only, human-written code, AI as reviewer)

## 0. Operating model

**You** write every line in `src/`, `tests/`, `scripts/`, `configs/`.
**The AI reviewer** explains, reviews, designs tests, debugs Socratically, and audits experiments.
This is enforced by a local, untracked reviewer config that also blocks its write access to those directories.

### The loop for every component
```
/explain <component>   → reviewer briefing: math, shapes, pitfalls, test list
write tests first      → from the test list, in your own words
implement              → until `uv run pytest -q` is green on CPU, then on MPS
/review                → fix the findings (bugs → design → tests → style)
git commit             → one component per commit, message says what + why
/end-session           → decisions go to docs/decisions.md
```
Session hygiene: one reviewer session per component, `/clear` between them. The reviewer
re-reads its working agreement, PLAN.md, decisions.md and log.md at the start, so context lives in those files, not in chat history.
Before a phase ends, run `/gate <N>`. NO-GO means you do not move on.

## 1. The compute constraint (read this before anything else)

The only machine is an M5 Pro with 48 GB. Memory is not the binding constraint. **Compute is.**

| Config | Params (total / non-emb) | Train mem est.* | Tokens planned | Train FLOPs (≈6N+attn per tok) |
|---|---|---|---|---|
| syrinx-10m  | 11.8M / 3.4M  | < 6 GiB  | overfit + ~0.3B (TinyStories) | ~0.03e18 |
| syrinx-50m  | 50.9M / 34.1M | ~10 GiB  | ~1.0B  | ~0.37e18 |
| syrinx-125m | 124.3M / 99.1M | ~16 GiB (micro-batch 8) | ~2.5B (≈20 tok/param) | ~2.15e18 |

\*Analytic: fp32 weights + grads + AdamW (16 B/param) + bf16 autocast copies (2 B/param)
+ activations assuming unfused attention + logits, all ×1.25 overhead. These are estimates. Phase 4 measures the real peak.

**Wall-clock for the 125M run depends on sustained throughput.**
2.15e18 FLOPs at 2 TFLOPS takes about 12 days, at 3.5 TFLOPS about 7 days, and at 5 TFLOPS about 5 days.
To finish 2.5B tokens in 7 days you need about **4,100 tokens/sec** sustained.

**Proxy measurement (2026-09-22, `experiments/feasibility-2026-09-22.md`):** HF `LlamaForCausalLM`
at our exact shapes, SDPA, seq 1024, 15-step burst:

| Model | micro-batch | precision | tok/s | driver mem | → wall-clock for planned tokens |
|---|---|---|---|---|---|
| 125M | 8  | fp32        | 4,699  | 21.3 GiB | 2.5B → 6.2 days |
| 125M | 8  | bf16 autocast | 6,674 | 20.1 GiB | 2.5B → 4.3 days |
| 125M | 16 | bf16 autocast | 7,165 | 35.2 GiB | too close to the 37.4 GiB MPS ceiling |
| 50M  | 16 | bf16 autocast | 13,362 | 24.2 GiB | 1.0B → 20.8 h |

Matmul peak: fp32 7.4, bf16 30.8 TFLOPS. The 125M proxy sits at ≈16% MFU, so there is headroom.
These are **upper-ish bounds from a burst**, not our code and not sustained. Expect our first
(unfused) implementation to be slower and a multi-day run to lose some throughput to heat.
Gate 6a re-measures with our code, sustained. Until then, plan on **~5–7 days** for the 125M run.
Memory: the analytic 16 GiB estimate for 125M mb8 came in at 20.1 GiB driver-allocated (+26%).
Use micro-batch 8 for the long run; the headroom matters when other apps are open.

Out of reach locally (don't pretend otherwise): 350M/1B models, multi-seed runs at 125M, and large
hyperparameter sweeps at 125M. Research experiments (Phase 11) run at 10M–50M, where
local compute allows several seeds.

## 2. Phases

Effort figures are rough focused-hours for a solo builder. Treat them as a sanity check, not a schedule.

### Phase 0 — Environment + foundations (≈6–10 h)
**Scaffold (done 2026-09-22):** `pyproject.toml` (hatchling, `src/` layout, extras `export` =
transformers and `eval` = lm-eval, dev group = pytest + ruff), `uv.lock` refreshed, LICENSE (Apache-2.0),
README, `docs/` (decisions, data, reference/config-spec), `experiments/log.md`, first commit.
**Still to do before anything else:** create `src/syrinx/__init__.py` (the editable build needs the package
to exist), then run `uv sync --all-extras`.
Build (you): `src/syrinx/utils/device.py` (device resolution, bf16 *probe*,
synchronize, memory stats); `src/syrinx/config.py` (typed dataclasses, reject unknown YAML keys,
analytic `num_parameters`); `src/syrinx/estimate.py` + `scripts/estimate_memory.py`;
`scripts/check_env.py` (reports MPS/bf16/SDPA status and *measured* matmul TFLOPS).
Learn: why the MPS fallback env var is dangerous; unified memory; autocast vs full bf16.
A config spec (fields, validation rules, param-count formula, test list) is in
`docs/reference/config-spec.md`. There's no reference code. Write your own, then check it against the spec.
**Gate 0:** `check_env.py` output saved to `experiments/env.txt`; the param-count tests pin
11,798,784 / 50,866,688 / 124,275,456; `uv run pytest` is green.

### Phase 1 — Tokenizer (≈8–12 h)
Build: a byte-level BPE with 32,768 vocab (HF `tokenizers`), trained on ~1–2 GB drawn **only from the
train split**. Special tokens: `<|endoftext|>` (plus `<|pad|>` for batching later).
Evaluate: exact roundtrip on held-out text, zero UNK (byte-level guarantees this, so test it),
bytes/token on held-out FineWeb-Edu and TinyStories, and a comparison with the GPT-2 tokenizer on the same text.
**Gate 1:** 100% roundtrip on held-out docs; compression numbers are in the log; tokenizer files are versioned.

### Phase 2 — Data pipeline (≈12–18 h)
Sources: TinyStories (for 10M) and a FineWeb-Edu sample (for 50M/125M). Check each license on its
dataset card and record it in `docs/data.md`.
Pipeline: stream → normalize (NFC, strip control chars) → exact dedup (hash) → quality/length
filter → **split at the document level** → tokenize → `uint16` memmap shards with EOS separators
→ random-window dataloader. Disk: ~2 bytes/token, so 2.5B tokens ≈ 5 GB tokenized. Raw text costs more.
Learn: leakage (splitting after packing leaks docs across splits), contamination, dedup tradeoffs.
**Gate 2:** hash check confirms zero train/val doc overlap; decoding a shard reproduces the
source text; token counts are logged per split.

### Phase 3 — Model components (≈15–25 h)
Order: RMSNorm → RoPE (HF split-halves convention) → causal attention → SwiGLU → block → model.
Tests you must write: shapes; RMSNorm vs a manual formula; RoPE preserves norms and relative
position (q·k depends only on offset); attention matches `F.scaled_dot_product_attention`;
**causality** (perturbing token t leaves outputs < t unchanged); the model's `sum(p.numel())`
matches the analytic count; initial loss ≈ ln(32768) ≈ 10.40; CPU vs MPS outputs agree within tolerance.
**Gate 3:** all of the above pass on CPU and MPS.

### Phase 4 — Training loop + 10M model (≈15–20 h)
Build: AdamW (no weight decay on norms/embeddings, and justify that choice), warmup + cosine, grad accumulation
(loss / accum_steps), clipping, bf16 autocast if the probe passes, checkpoint save/**resume**
(model + optimizer + scheduler + RNG + data position), JSONL logging of train/val loss,
lr, grad-norm, tokens/sec, and memory.
Correctness ladder: (a) one batch → loss near 0; (b) ~1 MB of text → memorized;
(c) resume test: a run interrupted and resumed reproduces the uninterrupted loss curve (CPU,
fixed seed); (d) TinyStories run → coherent-ish samples.
**Gate 4 (blocking):** (a)–(c) pass. Measured peak memory is logged next to the estimate.

**Decide and record in `docs/decisions.md` before Phase 5 (the plan left these open):**
- Global batch in tokens/step. Starting point: ~0.25M for 50M, ~0.5M for 125M (mb 8 × 1024 × accum 64 =
  524,288). At the proxy's 6.7k tok/s that is ~78 s per optimizer step and ~4,800 steps for 2.5B tokens,
  so warmup should be a few hundred steps, not thousands.
- Peak LR, min LR (e.g. 10% of peak), weight decay value, AdamW betas (0.9, 0.95), eps.
- Validation: fixed val token count (e.g. 10–20M tokens) and the eval interval. Eval is forward-only
  but still costs wall-clock, so put it in the time budget.
- Checkpoint size at 125M: fp32 weights + 2 AdamW moments ≈ 1.5 GB each. Keep last 3 + milestones (disk has 482 GiB free).

### Phase 5 — 50M model (≈8 h build + ~1–2 days compute)
First, a short LR sweep at 10M/50M (3 LRs, a few hundred steps each). Then the full run of about 1B tokens.
Record measured tokens/sec. This number predicts the 125M wall-clock.
**Gate 5:** smooth val curve with no unexplained spikes; the 125M time projection is recomputed from measured throughput.

### Phase 6 — 125M model (≈6 h build + days of compute)
**Gate 6a (decision):** a *sustained* benchmark of our own model, at least 30–45 min of real training
steps (200 steps at ~1.2 s is only ~4 min, which is shorter than the laptop takes to heat-soak).
Report the median tok/s over the last 15 min, and compare against the proxy (6,674 tok/s bf16 mb8).
If we are more than 30% below the proxy, find out why before starting the long run (manual attention? fp32 logits? a CPU fallback?). Then:
≥ 4,000 tok/s → train 2.5B tokens. 2,000–4,000 → choose between 2.5B tokens (longer run) and ~1.5B
(say it's under-trained in the card). < 2,000 → run Phase 7 optimizations first, then re-measure.
This is your decision. It gets documented, never changed silently.
Long-run operations: see §3.
**Gate 6b:** training is complete and the loss curves and all checkpoints are archived.

### Phase 7 — Optimization (only what measurements justify, ≈8–15 h)
Candidates: SDPA vs manual attention on MPS; chunked cross-entropy (the logits take ~2.5 GiB per
micro-batch at 125M); activation checkpointing if memory-bound; `torch.compile` on MPS
(experimental, so benchmark it and drop it if it's flaky); data-loader stalls. Profile with
`torch.profiler` and Instruments. For every change, record a before/after in tokens/sec.

### Phase 8 — Evaluation (≈8–12 h)
Held-out val loss/ppl, plus **bits-per-byte** (the only metric comparable across tokenizers).
Benchmarks via lm-evaluation-harness on the exported model (HellaSwag, ARC-Easy, PIQA, LAMBADA),
run locally. Expect near-chance results on some tasks at 125M, and report them anyway. Contamination check:
13-gram overlap between benchmark test sets and the training shards. Keep "our metrics",
"benchmarks" and "comparisons to other models" separate, with caveats on data and token count.

### Phase 9 — HF export (≈8 h)
Map weights to `LlamaForCausalLM` (no `trust_remote_code`); tokenizer → `PreTrainedTokenizerFast`;
`generation_config.json`. **Parity test:** fp32 CPU logits of ours vs HF, max abs diff < 1e-4.

### Phase 10 — Release (≈6 h)
Model card with measured fields only; safetensors; upload with the `hf` CLI from `huggingface_hub`
(check the current commands at release time); milestone checkpoints go up as revisions.

### Phase 11 — Research experiments (open-ended, 10M–50M scale)
A mini scaling law (5 sizes, fit on non-embedding params); vocab 8K/16K/32K (compared in bits/byte);
context 256/512/1024; LR × batch; MHA vs GQA vs NoPE. Use ≥2 seeds where differences are
small. Caveat: conclusions at small scale may not transfer.

## 3. Runbook for multi-day training on a laptop
- Power adapter always; `caffeinate -dimsu` wrapping the run; run inside `tmux` so the session survives a closed terminal.
- Keep the lid open (a closed lid without an external display sleeps the machine regardless). Check
  System Settings → Battery → Energy Mode on AC, and record which mode was used, since it affects tok/s.
- Micro-batch 8 at 125M. mb 16 measured 35.2 GiB against a 37.4 GiB MPS recommended ceiling.
- Turn off automatic macOS updates for the duration. Keep the machine on a hard surface with the vents clear.
- Checkpoint every ~30–60 min (keep the last 3, plus milestones). Assume the run *will* be interrupted.
- Keep other heavy apps closed: GPU memory comes out of the same 48 GB pool.
- Watch for throttling: if tokens/sec in the log drifts down over hours, that's thermal. Record it.
- Keep a written "known-good resume" command in `experiments/log.md`.

## 4. Risk register
| Risk | Early signal | Mitigation |
|---|---|---|
| Throughput too low for 125M | Phase 5 tok/s | Gate 6a decision; Phase 7 |
| bf16 unstable/unsupported on some MPS op | NaNs, probe fails | fp32 fallback (≈2× time); isolate the op |
| Silent MPS→CPU fallback | tok/s far below matmul bound | never set the fallback env var |
| Resume not exact | Phase 4c test | save RNG + data position |
| Train/val leakage | Gate 2 hash check | split before packing |
| Over-reading single-seed diffs | small deltas | multiple seeds at 10M |
| Disk fills with raw data | `df` | stream and tokenize, delete raw |
| Own model much slower than HF proxy | Gate 6a tok/s ≪ 6.7k | SDPA, bf16 logits + fp32 CE, chunked CE (Phase 7) |
| Thermal throttling over days | tok/s drifts down in the JSONL log | hard surface, Energy Mode, accept a longer run and record it |
| Memory creep toward the MPS ceiling | driver mem grows across steps | mb 8, `empty_cache` only at eval boundaries, watch the logged memory |
