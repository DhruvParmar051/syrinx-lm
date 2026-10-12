# Experiment log

Rule: every number here traces to a file in `experiments/` or `runs/`, or to a command output
quoted in the entry. If it can't be traced, it doesn't go in.

## Entry template
```
## EXP-NNN — <short title>
- Date:            YYYY-MM-DD
- Git SHA:         <commit>  (dirty? yes/no)
- Config:          configs/<file>.yaml
- Seed(s):         
- Command:         
- Hardware state:  AC power? Energy Mode? other apps open? ambient notes
- Tokens / steps:  
- Metrics:         <metric> = <value>   (source: <file>:<line or key>)
- Peak memory:     measured <x> GiB vs estimate <y> GiB  (gap: …)
- Throughput:      <tok/s> median over <window>  (source: …)
- Observations:    only claims supported by the metrics above
- Caveats:         single seed? proxy? burst vs sustained? tokenizer differences?
```

---

## EXP-000 — Feasibility probe (PROXY)
- Date: 2026-09-22
- Git SHA: none (pre-first-commit)
- Command: `experiments/feasibility_tput_proxy.py` (HF `LlamaForCausalLM` stand-in, not Syrinx code)
- Details and raw numbers: [`feasibility-2026-09-22.md`](feasibility-2026-09-22.md)
- Headline: 125M bf16 mb8 at 6,674 tok/s (burst) → 2.5B tokens in ~4.3 days; 50M at 13,362 tok/s.
- Caveats: **proxy**, a 15-step burst, not sustained, not our implementation. Gate 6a re-measures.

## EXP-001 — Matmul throughput depends on power source (battery vs AC)
- Date: 2026-09-22
- Git SHA: pre-commit of `scripts/check_env.py` power fields (same session as EXP-000)
- Command: `uv run python scripts/check_env.py --out experiments/env-battery-2026-09-22.txt`
- Hardware state: **Battery Power**, `pmset_powermode: 1` (the AC profile on this
  machine is `powermode 2`, per `pmset -g custom`)
- Metrics (source: `env-battery-2026-09-22.txt`): fp32 3.83, fp16 23.97,
  bf16 22.13 TFLOPS (n=4096, 30 iters)
- Comparison: EXP-000 measured fp32 7.43 / bf16 30.83 TFLOPS the same morning **on AC**
  (`experiments/feasibility-2026-09-22.md`).
- Observations: on battery, matmul throughput is roughly 2x lower (fp32) and substantially lower (bf16).
  Same code, same shapes; only the power state differed.
- Caveats: one run per condition; thermal state not controlled. The AC baseline `experiments/env.txt`
  (Gate 0) must be regenerated on AC with the new power fields.
- Confirmation (same script, AC): `experiments/env.txt` (generated 2026-09-22T13:06, `power_source: AC Power`,
  `pmset_powermode: 2`) measured fp32 7.53 / fp16 31.76 / bf16 32.1 TFLOPS, matching EXP-000's AC numbers.
  The drop is the power state, not the code.
- Consequence: every throughput measurement (Gate 6a, Phase 7 before/after) must record `power_source`
  and `pmset_powermode`; comparisons across different power states are invalid.

---

## Known-good resume command
_(fill in once Phase 4 resume works: the exact command, checkpoint path pattern, and the date it was verified)_

## EXP-002 — Train the 32K byte-level BPE tokenizer (Phase 1)
- Date: 2026-10-11
- Git SHA: f3523f6 (clean)
- Command: `time uv run python scripts/train_tokenizer.py` (all defaults)
- Raw output: [`tokenizer-train-2026-10-11.txt`](tokenizer-train-2026-10-11.txt)
- Data: `HuggingFaceFW/fineweb-edu`, subset `sample-10BT`, revision
  `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, `val_fraction` 0.005, split salt `syrinx-split-v1`
- Settings: vocab 32,768, `min_frequency` 2, digits split individually, no prefix space
- Hardware state: AC power (stated by Dhruv; not captured in the output)
- Metrics (source: raw output above):
  - documents seen 317,847 = 0 empty + 134 exact duplicates + 1,553 val + 316,160 train
  - val share 1,553 / 317,713 unique = 0.489% (target 0.5%; expected ~1,589 +/- 40)
  - train text 1,500,059,862 UTF-8 bytes; 1,733,679 distinct pre-tokens; 32,510 merges
  - wall-clock 5:19 total, of which ~5:02 streaming/selection and ~14 s BPE training
- Output: `tokenizer/syrinx-bpe-32k/tokenizer.json`, 2,328,481 bytes,
  SHA-256 `0620753285f04d8302e725b5b1b7f43b9fe7a63ed063126aa8688f20073f07e8`
- Checks after loading the file (2026-10-11): SHA-256 matches; vocab exactly 32,768;
  `<|endoftext|>` = 0, `<|pad|>` = 1; the only vocab entries containing a digit are `0`-`9`;
  exact round-trip on four probe strings (English, numbers, accents/CJK/emoji, a literal
  `<|endoftext|>`).
- Observations: the 100 MB smoke run took ~40 s; this run scaled roughly linearly with data and
  is dominated by streaming, not training.
- Caveats: peak memory not measured. Round-trip so far only on probe strings; Gate 1 still needs
  round-trip on held-out val documents, bytes/token on held-out FineWeb-Edu and TinyStories, and
  the GPT-2 comparison. One timed 100 MB smoke run (not this one) did not exit after finishing
  (a download retry was in flight); cause unconfirmed.

## EXP-003 — Gate 1 tokenizer evaluation on held-out text
- Date: 2026-10-11
- Git SHA: 9ea43f8 (clean)
- Command: `time uv run python scripts/eval_tokenizer.py` (all defaults)
- Raw output: [`tokenizer-eval-2026-10-11.txt`](tokenizer-eval-2026-10-11.txt)
- Tokenizer: `tokenizer/syrinx-bpe-32k/tokenizer.json`, SHA-256
  `0620753285f04d8302e725b5b1b7f43b9fe7a63ed063126aa8688f20073f07e8` (EXP-002)
- Reference: GPT-2 tokenizer, `openai-community/gpt2` at commit
  `607a30d783dfa663caf39e06633721c8d4cfcd7e` (50,257 tokens)
- Held-out corpora:
  - `fineweb_edu_val`: the first 1,000 FineWeb-Edu `sample-10BT` documents (revision
    `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`) that the content-hash split assigns to val
    (`val_fraction` 0.005), i.e. never used for tokenizer training
  - `tinystories_val`: the first 1,000 stories of the TinyStories `validation` split (revision
    `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`), a corpus the tokenizer never saw
- Hardware state: AC power (stated by Dhruv; not captured in the output)
- Metrics (source: raw output above):

  | corpus | docs | UTF-8 bytes | Syrinx tokens | GPT-2 tokens | Syrinx B/tok | GPT-2 B/tok | Syrinx / GPT-2 tokens | round-trip failures |
  |---|---|---|---|---|---|---|---|---|
  | fineweb_edu_val | 1,000 | 4,560,949 | 1,021,215 | 981,362 | 4.466 | 4.648 | 1.041 | 0 |
  | tinystories_val | 1,000 | 792,375 | 198,686 | 193,559 | 3.988 | 4.094 | 1.026 | 0 |

  Wall-clock 3:29 total (mostly streaming).
- Observations: 100% exact round-trip (with and without EOS) on all 2,000 held-out documents.
  Syrinx needs 4.1% more tokens than GPT-2 on FineWeb-Edu val and 2.6% more on TinyStories val.
  (The prediction before the run was "about equal or fewer" on FineWeb-Edu; that was wrong.)
- Caveats: 1,000 documents per corpus, single run. The cause of the gap is not measured; candidate
  explanations are the smaller vocabulary (32,768 vs 50,257) and digit splitting. Per-token losses
  of Syrinx and GPT-2 are not comparable; use bits per byte with these bytes/token values.

## EXP-004 — TinyStories token shards and Gate 2 checks (Phase 2)
- Date: 2026-10-12
- Git SHA: 643fc25 (the build started from the uncommitted working tree; its code was committed
  unchanged as b7cfd66 + 643fc25 while the build ran)
- Commands: `time uv run python scripts/build_shards.py --source tinystories`, then
  `time uv run python scripts/check_shards.py --data-dir data_cache/tokens/tinystories` (all defaults)
- Raw output: [`shards-tinystories-build-2026-10-12.txt`](shards-tinystories-build-2026-10-12.txt)
  (the JSON line is the summary), [`shards-tinystories-check-2026-10-12.txt`](shards-tinystories-check-2026-10-12.txt)
- Input: TinyStories revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, `train` then
  `validation` (both official splits re-split by content hash), read from parquet files
  downloaded once to the local HF cache (`syrinx.data.sources`)
- Settings: tokenizer SHA-256 `0620753285f04d8302e725b5b1b7f43b9fe7a63ed063126aa8688f20073f07e8`,
  `val_fraction` 0.005, split salt `syrinx-split-v1`, `min_chars` 200, shard target 100M tokens
- Hardware state: AC power (`pmset -g batt` showed 'AC Power' before and after the build)
- Metrics (source: raw output above):

  | split | docs | tokens (EOS incl.) | shards | tokens / doc |
  |---|---|---|---|---|
  | train | 1,805,471 | 411,455,118 | 5 | 227.9 |
  | val | 9,122 | 2,074,361 | 1 | 227.4 |

  Selection: 2,141,709 source documents seen (2,119,719 train + 21,990 validation rows),
  326,842 exact duplicates after normalization (15.3%), 230 empty, 44 under 200 characters;
  kept 1,814,593, of which 0.503% went to val.
  Build wall-clock 2:40.90 (656 s user, 415% CPU), about 2.56M tokens/s end to end.
  Check wall-clock 12.5 s; exit 0, `passed: true`, no problems.
- Gate 2 checks (from the check output): train/val overlap 0 documents; 0 duplicate documents
  within each split; counts, shard names and provenance match `summary.json`; the first 20,000
  selected source documents (19,902 train, 98 val) decode from the shards to exactly the source
  text.
- Observations: 15.3% of TinyStories documents are exact duplicates of earlier ones. Copies of
  `validation` stories that also occur in `train` were dropped as duplicates before the hash split,
  so they cannot leak into our val. The process exited on its own; the post-run hang seen with
  remote streaming (Phase 1 and the Phase 2 smoke builds) did not happen when reading local files.
- Caveats: the decode check covers the first 20,000 documents, not all 1.8M; the overlap and
  count checks cover every document. Throughput is not comparable with the Phase 2 smoke builds,
  which were on battery and streamed over the network. The 2:40.90 excludes the one-time
  1,000.8 MB download (about 40 minutes on 2026-10-11, network-bound).

## EXP-005 — FineWeb-Edu token shards and Gate 2 checks (Phase 2)
- Date: 2026-10-12
- Git SHA: 1c3964d (clean)
- Commands: `nohup caffeinate -i zsh -c 'time uv run python scripts/build_shards.py --source fineweb-edu'`,
  then `time uv run python scripts/check_shards.py --data-dir data_cache/tokens/fineweb-edu`
  (all defaults)
- Raw output: [`shards-fineweb-edu-build-2026-10-12.txt`](shards-fineweb-edu-build-2026-10-12.txt)
  (the JSON line is the summary), [`shards-fineweb-edu-check-2026-10-12.txt`](shards-fineweb-edu-check-2026-10-12.txt),
  [`shards-fineweb-edu-download-times-2026-10-12.txt`](shards-fineweb-edu-download-times-2026-10-12.txt)
- Input: FineWeb-Edu `sample-10BT`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`; files
  `000`-`003` of 14 under `sample/10BT/`, downloaded during the build (none were cached before) and
  read locally; the build stopped partway through file `003` at the train-token target
- Settings: tokenizer SHA-256 `0620753285f04d8302e725b5b1b7f43b9fe7a63ed063126aa8688f20073f07e8`,
  `val_fraction` 0.005, split salt `syrinx-split-v1`, `min_chars` 200, train target 2,500,000,000
  tokens, shard target 100M tokens
- Hardware state: AC power (`pmset -g batt` showed 'AC Power' before and after the build)
- Metrics (source: raw output above):

  | split | docs | tokens (EOS incl.) | shards | tokens / doc |
  |---|---|---|---|---|
  | train | 2,334,471 | 2,500,678,395 | 26 | 1,071.2 |
  | val | 11,513 | 12,215,490 | 1 | 1,061.0 |

  Selection: 2,367,869 source documents seen, 21,877 exact duplicates after normalization (0.92%),
  0 empty, 8 under 200 characters; kept 2,345,984, of which 0.491% went to val. Train overshoots
  the target by 678,395 tokens because the stop check runs after each batch of 1,024 documents.
  Build wall-clock 59:17.45 (4,006 s user, 114% CPU), including the downloads. No retries or
  timeouts in the log. Check wall-clock 20.3 s; exit 0, `passed: true`, no problems.
- Download vs processing (from the build log's per-file "Reading" timestamps and the files'
  download-completion times; minute-level, not timed in code):

  | file | download | processing |
  |---|---|---|
  | 000 | 12:50:38 -> 13:02:25 (11:47) | -> 13:07:16 (4:51) |
  | 001 | 13:07:16 -> 13:19:06 (11:50) | -> 13:23:46 (4:40) |
  | 002 | 13:23:46 -> 13:34:26 (10:40) | -> 13:38:46 (4:20) |
  | 003 (partial) | 13:38:46 -> 13:48:46 (10:00) | -> 13:49:53 (about 1:07) |

  About 44 min downloading 8.6 GB (about 3.2 MB/s) and about 15 min processing (about 2.8M
  tokens/s).
- Gate 2 checks (from the check output): train/val overlap 0 documents; 0 duplicate documents
  within each split; counts, shard names and provenance match `summary.json`; the first 20,000
  selected source documents (19,903 train, 97 val) decode from the shards to exactly the source
  text.
- Observations: the 200-character filter dropped 8 documents, as expected for a corpus that is
  quality-filtered upstream (docs/decisions.md, 2026-10-11 Phase 2 design). Exact duplicates are
  0.92%, against 15.3% for TinyStories (EXP-004). The val split holds 12.2M tokens, close to the
  ~12.5M expected in the design entry.
- Caveats: the decode check covers the first 20,000 documents; the overlap and count checks cover
  every document. Only the first 4 of 14 files are used, so the train set is the start of
  `sample-10BT` in file order, not a random sample of it. The download/processing split comes from
  file timestamps, not from timers in the code.
