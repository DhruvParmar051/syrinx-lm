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

## 2026-09-22 — Package layout: `src/syrinx/`
**Context:** Considered `src/__init__.py` (package named `src`) and a flat root-level `syrinx/`.
**Decision:** `src/` layout with the package at `src/syrinx/`; import name `syrinx`.
**Why:** A package named `src` collides with any other project using the same pattern and gives
`from src.x import` imports. The `src/` layout makes tests import the installed package, so packaging
bugs surface instead of being masked by the working directory.
**Revisit if:** never, for this project.

## 2026-09-22 — Config values fixed by `configs/*.yaml`
**Context:** The config files need concrete values before Phase 4, including a few PLAN.md left open.
**Decision:**
- syrinx-10m uses **4 heads** (head_dim 64, same as 50m/125m).
- `rope_theta = 10000.0`, `norm_eps = 1.0e-5`. Both must be copied unchanged into the exported `LlamaConfig`.
- `precision: bf16-autocast` for all sizes (`probe_bf16` passed on MPS and CPU, loss rel_err ~1e-6).
- The `train:` sections are **provisional placeholders**, marked as such in each file:
  10m: mb 32 × accum 1 (32,768 tok/step), lr 1e-3; 50m: mb 16 × 16 (262,144 tok/step), lr 6e-4;
  125m: mb 8 × 64 (524,288 tok/step), lr 6e-4. min_lr = lr/10, wd 0.1, betas (0.9, 0.95), clip 1.0, seed 1337.
  `max_steps` is set so steps × tokens/step reaches the PLAN.md budgets (1.0B / 2.5B); a test pins that.
**Why:** head_dim 64 everywhere keeps attention kernels comparable across sizes. The train values are
common GPT-2/3-small-style starting points, not tuned; the Phase 5 LR sweep replaces them.
**Revisit if:** the LR sweep or Gate 6a throughput changes the batch or LR (record the new values here).

## 2026-09-22 — Document-level train/val split by content hash
**Context:** PLAN.md requires the tokenizer to train on the train split only (Phase 1), but the split was
defined in Phase 2. If the two phases chose validation documents differently, validation text would leak
into tokenizer training.
**Decision:** One function decides the split for every stage: `assign_split(document_hash(normalize_text(doc)))`
in `src/syrinx/data/documents.py`. The hash is SHA-256 of the normalized text; the split re-hashes it with
the salt `syrinx-split-v1` and sends a document to val iff the first 8 bytes / 2^64 < `val_fraction`.
The same `val_fraction` must be used in Phase 1 and Phase 2 (0.005, see the 2026-10-11 entry).
**Why:** Deterministic on any machine and in any order, needs no stored state, and identical documents
always share a split. Python's `hash()` is randomized per process (verified: two runs gave different
values for the same string), so it is never used.
**Revisit if:** near-duplicate (not identical) documents turn out to straddle the split in Gate 2's check;
then add near-dup detection before the split.

## 2026-09-22 — Tokenizer versioning
**Context:** Gate 1 requires versioned tokenizer files, but `.gitignore` excludes `tokenizer/artifacts/`.
**Decision:** Commit the final `tokenizer.json` at `tokenizer/syrinx-bpe-32k/tokenizer.json` (a tracked path),
and record its SHA-256, the training-data description (dataset, subset, bytes, document counts,
`val_fraction`) and the command in `experiments/log.md`. Scratch tokenizers go in `tokenizer/artifacts/`.
**Why:** Every shard and checkpoint depends on the exact merges; the file in git plus its hash makes that
dependency checkable. The HF release ships the same file.
**Revisit if:** the file grows too large for comfortable git use.

## 2026-10-11 — Split digits individually in the tokenizer
**Context:** The GPT-2 split pattern keeps digit runs together, so BPE learns arbitrary number chunks
(on a number-heavy test corpus it learned space + "19" + "9" + "9" for " 1999").
**Decision:** Pre-tokenizer = `Sequence[Digits(individual_digits=True), ByteLevel(add_prefix_space=False,
use_regex=True)]`. Every digit is one token; no merge contains a digit.
**Why:** Numbers tokenize the same way everywhere, which helps arithmetic and copying numbers; Llama and
PaLM do the same. It loads in HF `PreTrainedTokenizerFast` like any `tokenizer.json`. Cost: number-heavy
text needs more tokens, and the space before a number becomes its own token (" 2026" is 5 tokens).
Must be fixed before the 32K tokenizer is trained; changing it later means retraining it and every shard.
**Revisit if:** Phase 1's bytes/token on held-out text is clearly worse than GPT-2 because of numbers.

## 2026-10-11 — val_fraction 0.005; tokenizer trained on 1.5 GB of FineWeb-Edu sample-10BT
**Context:** Phase 1 needs the split fraction (shared with Phase 2) and a tokenizer corpus.
**Decision:** `val_fraction = 0.005`. The tokenizer is trained on the first ~1.5 GB (UTF-8, after
normalization) of TRAIN-split, exact-deduplicated documents streamed from `HuggingFaceFW/fineweb-edu`,
subset `sample-10BT`. These are the defaults in `scripts/train_tokenizer.py`.
**Why:** 0.5% of a multi-million-document sample is tens of thousands of validation documents, enough
for a stable val loss, while losing little training data. FineWeb-Edu is the 50M/125M corpus; 1.5 GB is
in the PLAN's 1-2 GB range. TinyStories is left out to keep one source; its bytes/token is still
measured in Gate 1.
**Revisit if:** the license check on the dataset card fails, or Gate 1 shows TinyStories compresses
much worse than FineWeb-Edu.

---

## 2026-10-11 — Phase 2 data pipeline design
**Context:** Phase 2 turns streamed text into training shards; Gate 1 requires that the shards use
exactly the document path the tokenizer was trained with.
**Decision:**
- One document path, `syrinx.data.selection.iter_documents`: normalize -> skip empty -> exact dedup
  (content hash) -> drop documents shorter than **200 characters** after normalization -> content-hash
  split (`val_fraction` 0.005, salt `syrinx-split-v1`). With the length filter off it selects exactly
  the tokenizer script's train documents (tested).
- **FineWeb-Edu** (`sample-10BT`, pinned revision): stop once the train shards hold **2.5B tokens**
  (EOS included); val is whatever the split assigns along the way (~0.5%, ~12.5M tokens expected).
- **TinyStories:** **all** of it. The official `train` and `validation` splits are pooled and
  re-split with the same content hash, so every source follows one leakage rule.
- Shards: flat `uint16` files, `ids(doc) + EOS` per document, ~100M tokens per file, a document never
  spans two files, written to a temp name and renamed; a JSON sidecar records counts and provenance.
  Layout `data_cache/tokens/<source>/<split>/<split>_NNNN.bin`.
- Loader: train = uniform random windows of T+1 tokens over all valid starts (never crossing a
  shard), from a seeded `np.random.Generator` whose state is checkpointed; val = fixed,
  non-overlapping windows.
- No attention mask at document boundaries (EOS separates documents); no BOS token (the HF export
  will set `bos_token_id` to the EOS id).
**Why:** one selection rule means Gate 2's overlap check and Gate 1's train-only guarantee refer to
the same sets. FineWeb-Edu is already quality-filtered upstream, so only very short fragments are
dropped. `uint16` halves disk since V = 32,768 < 65,536. A per-window document mask would force a
non-fused attention path on MPS. Random windows need no epoch bookkeeping; resume needs only the RNG
state.
**Revisit if:** the 200-character filter drops a surprising share (check the counts in the build
summary), or Phase 5 shows boundary effects that a document mask would fix.

---

## Open (not yet decided — decide before Phase 5)
- Global batch in tokens/step (starting point ~0.25M for 50M, ~0.5M for 125M).
- Peak LR, min LR, warmup steps, weight decay, AdamW betas/eps.
- Validation token count and eval interval.
- Whether `<|pad|>` is actually needed (random-window packing doesn't pad; only batched eval might).
