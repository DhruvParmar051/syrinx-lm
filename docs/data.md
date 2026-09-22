# Data

Every license and count on this page must come from the dataset card or a command output,
with the date it was checked. Leave a field blank rather than filling it from memory.

## Sources

| Dataset | HF repo id | Subset / config | License (from card) | Checked on | Used for | Notes |
|---|---|---|---|---|---|---|
| TinyStories | _verify_ | — | _verify on card_ | | 10M model; tokenizer eval | |
| FineWeb-Edu | _verify_ | _e.g. a sample-NBT subset_ | _verify on card_ | | tokenizer training; 50M/125M | |

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
- Drawn **only** from the train split (~1–2 GB). Record the exact byte count and document count.
- Held-out evaluation text (roundtrip, bytes/token) comes from the val split.

## Leakage and contamination checks
| Check | Method | Result | Evidence |
|---|---|---|---|
| Train/val doc overlap | hash sets of normalized docs; intersection must be empty | | |
| Tokenizer trained on train only | the list of shards used by the tokenizer script | | |
| Benchmark contamination (Phase 8) | 13-gram overlap between benchmark test sets and train shards | | |
