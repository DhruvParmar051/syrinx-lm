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
| syrinx-10m  | 256 | 4  | 768  | 4  | 64 |
| syrinx-50m  | 512 | 10 | 1536 | 8  | 64 |
| syrinx-125m | 768 | 12 | 2560 | 12 | 64 |

**Why:** The 50m and 125m shapes were confirmed by building HF Llama at these shapes, which produced
exactly the pinned counts (`experiments/feasibility-2026-09-22.md`). Head count doesn't change the
param count, so it's a free choice for 10m.
**Revisit if:** a config's measured throughput or loss suggests a different head count (it does not
change the param count).

## 2026-10-12 — 10m model uses 4 heads (head_dim 64)
**Context:** Head count was the one open shape for syrinx-10m (d_model 256); any divisor with an even
head_dim keeps the pinned 11,798,784 params.
**Decision:** n_heads = 4, head_dim = 64, as already set in `configs/syrinx-10m.yaml`.
**Why:** head_dim 64 matches the 50m and 125m models, so the 10m runs exercise the same attention
shapes (and the same SDPA/MPS kernel path) as the larger models. The Gate 3 tests ran on this config
(EXP-006).
**Revisit if:** the 10m model is used for an experiment where head count is the variable.

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

## 2026-10-12 — Download source parquet files once and read them locally
**Context:** the first TinyStories build streamed with `load_dataset(..., streaming=True)` and stalled:
one HTTP range read of a parquet file trickled at about 3 KB/s (30,744 bytes in 10 s) with 0% CPU,
and no timeout fired because bytes kept arriving. The connection to Hugging Face from this network
measured 70-170 KB/s (`curl`) while general download speed was 16.2 Mbps (`networkQuality`), so
any long streamed build risks a stall that throws away the whole run. Streaming builds also hung
after finishing (twice in Phase 1/2 smoke runs).
**Decision:** Phase 2 shard building and checking read through `syrinx.data.sources.iter_hub_texts`:
list the split's parquet files at the pinned revision, then for each file in sorted order
`hf_hub_download` it (cached, resumable, retried) and read the text column locally with `pyarrow`.
Files are fetched lazily, so a FineWeb-Edu build that stops at 2.5B tokens never downloads the
remaining files. Provenance (dataset, subset, revision) is unchanged. The Phase 1 tokenizer scripts
keep streaming; they are finished and their results are recorded.
**Why:** downloads can resume and are kept, so a stall costs only a restart of the transfer, not the
build, and the build time measures the pipeline rather than the network. Same rows in the same
order as streaming: on TinyStories `validation` the local read returned the identical 21,990 rows
in the same order (0.4 s locally vs 9.9 s streamed). The TinyStories build then took 2:40.90 and
exited normally (EXP-004).
**Revisit if:** disk space for the FineWeb-Edu `sample-10BT` files becomes a problem (14 files,
0.54-2.15 GB each, 28.5 GB in total per the Hub file listing at the pinned revision, 2026-10-12), or a source is not published as parquet.

---

## 2026-10-12 — AdamW groups and an LR schedule computed from the step
**Context:** PLAN.md asks for no weight decay on norms and embeddings, with a justification, and a
warmup + cosine schedule that survives checkpoint/resume.
**Decision:** Two AdamW groups: weight decay on every 2-D weight (q/k/v/o, gate/up/down, an untied
`lm_head`); none on the RMSNorm gains or the (tied) embedding. The LR is `lr_at(step)` (linear warmup
from `lr/warmup` at step 0, half-cosine to `min_lr` at `max_steps`, then `min_lr`), set on every group
before each step; there is no `LRScheduler` object.
**Why:** RMSNorm gains act as per-channel scales whose neutral value is 1; decoupled decay pulls them
toward 0 for no regularization benefit. Embedding rows receive gradient only when their token occurs,
but decay shrinks every row every step, so rare tokens would decay toward zero between updates; with
tied weights the same matrix is the output layer, so one rule covers both. A pure function of the step
means a checkpoint restores the exact LR from the step number alone.
**Revisit if:** the 50M/125M runs show embedding or norm-gain growth that decay would have prevented.

## 2026-10-12 — Phase 5 (50M) training settings
**Context:** PLAN.md leaves batch, optimizer values, validation and checkpointing open until Phase 5.
Times below assume the EXP-000 proxy's 13,362 tok/s for 50M (a 15-step burst on a stand-in model);
Phase 5 measures the real number.
**Decision:**
- **Tokens per optimizer step: 262,144** (micro-batch 16 x accumulation 16 x 1024); 3,815 steps for
  1.0B tokens. Micro-batch 16: estimate 16.0 GiB (20.0 GiB with margin), proxy 24.2 GiB, ceiling 37.44 GiB.
- **Peak LR: chosen by a sweep** over 3e-4, 6e-4, 1.2e-3. Each sweep run is the full-run config with only
  `lr` and `min_lr` changed, stopped with `--stop-at-step 300`, so it sees the real warmup and schedule.
  Pick rule (`scripts/lr_sweep.py`): drop runs with a non-finite loss or grad norm, no val loss at step
  300, or a train loss more than 1.0 nat above its running minimum after warmup; take the lowest val
  loss at step 300; if lower-LR runs are within 0.01 of it, take the lowest such LR.
- **Unchanged from the placeholders:** min LR = peak/10, 200 warmup steps (52M tokens), weight decay 0.1
  (matrices only), betas (0.9, 0.95), eps 1e-8, grad clip 1.0.
- **Validation:** the same first 100 val batches (1.64M tokens) every 250 steps; the reported final
  number is one full-split evaluation (12.2M tokens) of the last checkpoint.
- **Checkpoints:** every 250 steps, keep the last 3, plus a milestone every 1,000 steps
  (about 0.61 GB each, estimated as 50.87M parameters x 12 bytes).
- **`<|pad|>`:** stays in the vocabulary (removing it would change token ids) but is reserved and
  unused: random-window training and the fixed val windows never pad.
- The sweep, the pick and the full run are chained by `scripts/run_phase5.sh`, so the full run starts
  without a manual step; the chosen config is written to `runs/50m-a.yaml`.
**Why:** batch size does not change wall-clock (micro-batch sets throughput), so it stays at the planned
value and the sweep tunes only the LR. Short sweeps favour high LRs, hence the tie-break toward the
lower LR. 20 val batches (the default) are too noisy to judge Gate 5's "no unexplained spikes"; 100
cost an estimated ~41 s per eval, about 1% of the run.
**Revisit if:** the sweep's best LR is the highest tried (extend the sweep upward before 125M), the
measured 50M throughput differs from the proxy by more than 30%, or the val curve is too noisy at 100
batches.

## Open (not yet decided)
- 125M batch and LR: to be set from the 50M sweep and the Gate 6a throughput measurement.
