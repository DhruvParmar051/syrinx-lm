# Data

Every license and count on this page must come from the dataset card or a command output,
with the date it was checked. Leave a field blank rather than filling it from memory.

## Sources

| Dataset | HF repo id | Subset / config | License (from card) | Checked on | Used for | Notes |
|---|---|---|---|---|---|---|
| TinyStories | `roneneldan/TinyStories` | — (splits `train`, `validation`) | CDLA-Sharing-1.0 | 2026-10-11 | 10M model; tokenizer eval | not gated; dataset revision (commit) `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` (last modified 2024-08-12); `validation` has 21,990 rows, `train` 2,119,719; text column `text` |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | `sample-10BT` | ODC-By v1.0; also subject to the Common Crawl Terms of Use | 2026-10-11 | tokenizer training; 50M/125M | not gated; dataset revision (commit) `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` (last modified 2025-07-11); `sample-10BT` train split has 9,672,101 rows; text column `text` |

## License obligations
- **FineWeb-Edu (ODC-By v1.0):** attribution is required. The model card and README must credit
  FineWeb-Edu (Hugging Face) with a link to the dataset and to the license, and note that the data comes
  from Common Crawl and is subject to the Common Crawl Terms of Use.
  Source: dataset card, "Licensing Information" section, checked 2026-10-11.
- **TinyStories (CDLA-Sharing-1.0):** a share-alike data license. Using the data to train a model
  is allowed; redistributing the DATA (or modified data) must keep the same license. We do not
  redistribute it. Credit it in the model card. Source: dataset card metadata, checked 2026-10-11.
- **GPT-2 tokenizer (`openai-community/gpt2`, MIT, commit `607a30d783dfa663caf39e06633721c8d4cfcd7e`):**
  used only as a comparison in Gate 1, never shipped.

## Token counts (from the build summaries)

Tokens include one EOS per document. Raw UTF-8 bytes were not measured by the build.

| Dataset | Split | Documents | Tokens | Shards | Source log |
|---|---|---|---|---|---|
| TinyStories | train | 1,805,471 | 411,455,118 | 5 | EXP-004, [`shards-tinystories-build-2026-10-12.txt`](../experiments/shards-tinystories-build-2026-10-12.txt) |
| TinyStories | val | 9,122 | 2,074,361 | 1 | EXP-004 (same file) |
| FineWeb-Edu (`sample-10BT`, files 000-003) | train | 2,334,471 | 2,500,678,395 | 26 | EXP-005, [`shards-fineweb-edu-build-2026-10-12.txt`](../experiments/shards-fineweb-edu-build-2026-10-12.txt) |
| FineWeb-Edu (`sample-10BT`, files 000-003) | val | 11,513 | 12,215,490 | 1 | EXP-005 (same file) |

## Pipeline rules
1. **Download once, read locally** (Phase 2 shard builds and checks, `syrinx.data.sources`): each
   parquet file of a split is fetched at the pinned revision with `hf_hub_download` into the local
   HF cache (outside the repo), one file at a time and only when the reader reaches it, then read
   with `pyarrow`. File order is the sorted repo paths, the same order as Hub streaming. Only the
   files actually read are downloaded, not the whole corpus. (Phase 1 tokenizer training and
   evaluation streamed from the Hub; see docs/decisions.md, 2026-10-12.)
2. **Normalize** (`syrinx.data.documents.normalize_text`): `\r\n` and lone `\r` become `\n`; Unicode
   NFC; remove every character in Unicode category `Cc` except `\n` and `\t`. No lowercasing,
   whitespace collapsing or stripping.
3. **Exact dedup**: SHA-256 of the UTF-8 bytes of the normalized text; later copies are dropped.
   TinyStories: 326,842 of 2,141,709 documents dropped (15.3%, EXP-004).
   FineWeb-Edu: 21,877 of 2,367,869 dropped (0.92%, EXP-005).
4. **Length filter**: drop documents under 200 characters after normalization. Empty documents are
   counted separately. TinyStories: 44 too short, 230 empty (EXP-004). FineWeb-Edu: 8 too short,
   0 empty (EXP-005).
5. **Split at the document level**, **before** tokenization and packing. The split has to be
   deterministic, e.g. from a hash of the document, so it's reproducible without keeping state.
6. **Tokenize** → `uint16` memmap shards, with an EOS separator (`<|endoftext|>`) between documents.
7. **Raw text** stays only in the HF cache, never in the repo or `data_cache/tokens`. It can be
   deleted once the counts are logged, but `check_shards.py`'s source round-trip reads it again
   (it would re-download).

## Tokenizer training data
- Drawn **only** from the train split: the first ~1.5 GB of normalized, exact-deduplicated train-split
  documents from FineWeb-Edu `sample-10BT`, with `val_fraction = 0.005` (docs/decisions.md, 2026-10-11).
  Record the exact byte count and document count from the script's JSON summary.
- Held-out evaluation text (roundtrip, bytes/token) comes from the val split.

## Leakage and contamination checks
| Check | Method | Result | Evidence |
|---|---|---|---|
| Train/val doc overlap | SHA-256 of each document's token ids in the train and val shards (equivalent to the text, since the tokenizer round-trips exactly); intersection must be empty | TinyStories: 0 of 1,805,471 train / 9,122 val; FineWeb-Edu: 0 of 2,334,471 train / 11,513 val | EXP-004, [`shards-tinystories-check-2026-10-12.txt`](../experiments/shards-tinystories-check-2026-10-12.txt); EXP-005, [`shards-fineweb-edu-check-2026-10-12.txt`](../experiments/shards-fineweb-edu-check-2026-10-12.txt) |
| Tokenizer trained on train only | the list of shards used by the tokenizer script | | |
| Benchmark contamination (Phase 8) | 13-gram overlap between benchmark test sets and train shards | | |
