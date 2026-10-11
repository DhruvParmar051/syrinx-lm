# Data

Every license and count on this page must come from the dataset card or a command output,
with the date it was checked. Leave a field blank rather than filling it from memory.

## Sources

| Dataset | HF repo id | Subset / config | License (from card) | Checked on | Used for | Notes |
|---|---|---|---|---|---|---|
| TinyStories | _verify_ | — | _verify on card_ | | 10M model; tokenizer eval | |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | `sample-10BT` | ODC-By v1.0; also subject to the Common Crawl Terms of Use | 2026-10-11 | tokenizer training; 50M/125M | not gated; dataset revision (commit) `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` (last modified 2025-07-11); `sample-10BT` train split has 9,672,101 rows; text column `text` |

## License obligations
- **FineWeb-Edu (ODC-By v1.0):** attribution is required. The model card and README must credit
  FineWeb-Edu (Hugging Face) with a link to the dataset and to the license, and note that the data comes
  from Common Crawl and is subject to the Common Crawl Terms of Use.
  Source: dataset card, "Licensing Information" section, checked 2026-10-11.

## Token counts (fill in during Phase 2, from the pipeline log)

| Dataset | Split | Documents | Tokens | Bytes (raw UTF-8) | Source log |
|---|---|---|---|---|---|
| | train | | | | |
| | val | | | | |

## Pipeline rules
1. **Stream**: don't download the full raw corpus to disk.
2. **Normalize**: Unicode NFC; strip control characters except `\n` and `\t`. _(Record the exact set.)_
3. **Exact dedup**: hash the normalized document text. _(Record the hash function and how many documents were dropped.)_
4. **Quality/length filter**: _(record thresholds and drop counts)_.
5. **Split at the document level**, **before** tokenization and packing. The split has to be
   deterministic, e.g. from a hash of the document, so it's reproducible without keeping state.
6. **Tokenize** → `uint16` memmap shards, with an EOS separator (`<|endoftext|>`) between documents.
7. **Delete raw text** after tokenizing, once the counts are logged.

## Tokenizer training data
- Drawn **only** from the train split: the first ~1.5 GB of normalized, exact-deduplicated train-split
  documents from FineWeb-Edu `sample-10BT`, with `val_fraction = 0.005` (docs/decisions.md, 2026-10-11).
  Record the exact byte count and document count from the script's JSON summary.
- Held-out evaluation text (roundtrip, bytes/token) comes from the val split.

## Leakage and contamination checks
| Check | Method | Result | Evidence |
|---|---|---|---|
| Train/val doc overlap | hash sets of normalized docs; intersection must be empty | | |
| Tokenizer trained on train only | the list of shards used by the tokenizer script | | |
| Benchmark contamination (Phase 8) | 13-gram overlap between benchmark test sets and train shards | | |
