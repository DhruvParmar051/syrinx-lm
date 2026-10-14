# Data

Every license and count on this page must come from the dataset card or a command output,
with the date it was checked. Leave a field blank rather than filling it from memory.

## Sources

| Dataset | HF repo id | Subset / config | License (from card) | Checked on | Used for | Notes |
|---|---|---|---|---|---|---|
| TinyStories | `roneneldan/TinyStories` | — (splits `train`, `validation`) | CDLA-Sharing-1.0 | 2026-10-11 | 10M model; tokenizer eval | not gated; dataset revision (commit) `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` (last modified 2024-08-12); `validation` has 21,990 rows, `train` 2,119,719; text column `text` |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | `sample-10BT` | ODC-By v1.0; also subject to the Common Crawl Terms of Use | 2026-10-11 | tokenizer training; 50M/125M | not gated; dataset revision (commit) `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` (last modified 2025-07-11); `sample-10BT` train split has 9,672,101 rows; text column `text` |

## Evaluation benchmarks (not used for training)

Fetched 2026-10-14 at pinned revisions into `data_cache/benchmarks/` (git-ignored) and exported as
JSONL; sources, file SHA-256 and row counts in
[`benchmarks-fetch-2026-10-14.txt`](../experiments/benchmarks-fetch-2026-10-14.txt).

| Benchmark | HF repo id | Revision | File (split) | Rows | License (from card) | Checked on |
|---|---|---|---|---|---|---|
| HellaSwag | `Rowan/hellaswag` | `218ec52e09a7e7462a5400043bb9a69a41d06b76` | `data/validation-00000-of-00001.parquet` | 10,042 | MIT (card text, "Licensing Information"; not in the card metadata) | 2026-10-14 |
| ARC-Easy | `allenai/ai2_arc` | `210d026faf9955653af8916fad021475a3f00453` | `ARC-Easy/test-00000-of-00001.parquet` | 2,376 | CC-BY-SA-4.0 (card metadata) | 2026-10-14 |
| PIQA | `ybisk/piqa`, HF parquet conversion `refs/convert/parquet` | `142c51238b3ca2bc61e9a075913871b8b600e8e1` | `plain_text/validation/0000.parquet` | 1,838 | unknown (card metadata "unknown"; the card's licensing section is empty) | 2026-10-14 |
| LAMBADA (OpenAI version) | `EleutherAI/lambada_openai` | `900124bf3b8235c6daf21033af9948b3f07346c4` | `default/test/default.parquet` | 5,153 | MIT (card metadata) | 2026-10-14 |

PIQA's main branch holds only a loading script that downloads from an external site; the HF parquet
conversion was used instead so that no repository code is executed. The benchmarks are used only for
evaluation and the contamination check below; they are not redistributed.

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
| Tokenizer trained on train only | `train_tokenizer.py` skips every document that `assign_split` puts in val, using the same `normalize_text`, `document_hash`, salt `syrinx-split-v1` and `val_fraction` 0.005 as the Phase 2 shards (`test_selection.py::test_matches_tokenizer_script_selection` pins this). The split depends only on the document hash, so the 200-character filter cannot move a document across splits | FineWeb-Edu: 1,553 val documents skipped, 316,160 train documents used; no Phase 2 val document can be among them. TinyStories: not used for tokenizer training. This holds by construction: no check compared the tokenizer's input with the val shards directly | EXP-002, [`tokenizer-train-2026-10-11.txt`](../experiments/tokenizer-train-2026-10-11.txt) |
| Benchmark contamination (Phase 8) | Word 13-grams (NFKC, lowercase, alphanumeric runs) of every benchmark example vs every train document of TinyStories and FineWeb-Edu; hash hits confirmed word by word; examples under 13 words reported as unchecked | HellaSwag 40 / 10,042 (0.40%); ARC-Easy 22 / 2,300 checked (0.96%, 76 unchecked); PIQA 3 / 1,685 checked (0.18%, 153 unchecked); LAMBADA 0 / 5,153. Matches are mostly common phrases; an upper bound on leakage, not confirmed leaks. Scores are reported on all and on clean examples | EXP-016, [`contamination-2026-10-14.json`](../experiments/contamination-2026-10-14.json) |
