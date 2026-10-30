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
Verified 2026-10-12 on CPU (EXP-010): rerun the same command with `--resume` added. It continues from the
newest `<out-dir>/checkpoints/step_NNNNNNN.pt` and appends to `<out-dir>/metrics.jsonl`; the config must
be unchanged (a different config is refused).
```
uv run python scripts/train.py --config configs/syrinx-10m.yaml --out-dir runs/<name> --resume
```

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

## EXP-006 — Gate 3 model-component tests on CPU and MPS (Phase 3)
- Date: 2026-10-12
- Git SHA: cad5fdf plus the uncommitted test changes that move the property tests onto a
  CPU/MPS device fixture (committed next, unchanged)
- Commands: `uv run pytest -v -rs tests/test_norm.py tests/test_rope.py tests/test_attention.py
  tests/test_mlp.py tests/test_block.py tests/test_model.py`, then the same with `SYRINX_DEVICE=cpu`
- Raw output: [`model-tests-mps-2026-10-12.txt`](model-tests-mps-2026-10-12.txt),
  [`model-tests-cpu-2026-10-12.txt`](model-tests-cpu-2026-10-12.txt)
- Hardware state: AC power (`pmset -g batt` in the output header); torch 2.14.0, Python 3.12.13
- Results (source: raw output above): MPS available, 85 passed (61 CPU cases + 23 `[mps]`
  parametrized cases + `test_cpu_and_mps_agree`), exit 0. `SYRINX_DEVICE=cpu`: 61 passed, 24
  skipped (the MPS cases), exit 0.
- Gate 3 criteria, each passing as `[cpu]` and `[mps]`:

  | Criterion | Tests |
  |---|---|
  | Shapes | `test_forward_shape_and_dtype`, shape asserts in the component tests |
  | RMSNorm vs manual formula | `test_norm.py::test_matches_manual_formula` (+ `test_eps_used`, `test_bf16_input_statistic_in_float32`) |
  | RoPE preserves norms | `test_rope.py::test_preserves_norm` |
  | RoPE q.k depends only on offset | `test_rope.py::test_dot_product_depends_only_on_offset` |
  | Attention matches SDPA | `test_attention.py::test_reference_matches_sdpa`, `test_module_matches_manual_pipeline` |
  | Causality | `test_attention.py::test_module_is_causal`, `test_block.py::test_is_causal`, `test_model.py::test_model_is_causal` |
  | Initial loss near ln(32768) | `test_model.py::test_initial_loss_near_log_vocab` (within 0.15, 10m config) |
  | CPU vs MPS agree | `test_model.py::test_cpu_and_mps_agree` (10m logits within 1e-3, loss within 1e-4) |
  | Parameter count | `test_parameter_count_matches_analytic` (meta device; 11,798,784 / 50,866,688 / 124,275,456) |

  Also: `test_bf16_autocast_forward` passes on both devices (bf16 finite logits, float32 loss
  within 0.05 of the float32 loss), and `test_logits_match_hf_llama` passes on CPU.
- Observations: before this run only `test_cpu_and_mps_agree` touched MPS; the property tests
  built every tensor on CPU. They now take an `each_device` fixture.
- Caveats: the tests assert thresholds, not values, so the measured initial loss and the
  CPU/MPS gap are not recorded here. HF parity runs on CPU only. MPS cases use the
  default float32 tolerances of `torch.testing.assert_close`, and all pass with them.

## EXP-007 — 10m memory estimate vs a 30-step measured probe (Phase 4)
- Date: 2026-10-12
- Git SHA: 0023509 (clean)
- Config: `configs/syrinx-10m.yaml` (micro-batch 32, accum 1, seq 1024, bf16-autocast, seed 1337)
- Commands: `uv run python scripts/estimate_memory.py configs/syrinx-10m.yaml`;
  `uv run python scripts/train.py --config configs/syrinx-10m.yaml --out-dir <scratch> --stop-at-step 30
  --log-interval 1 --eval-interval 100000 --ckpt-interval 100000` (MPS)
- Raw output: [`estimate-10m-2026-10-12.txt`](estimate-10m-2026-10-12.txt),
  [`probe-10m-metrics-2026-10-12.jsonl`](probe-10m-metrics-2026-10-12.jsonl) (the probe's `metrics.jsonl`)
- Hardware state: AC power
- Metrics (source: raw output above):

  | | estimate | measured |
  |---|---|---|
  | peak memory | 17.48 GiB subtotal (21.85 GiB with the x1.25 margin); logits 14.00 GiB | 18.24 GiB (`mem_peak`, max over 30 steps) |
  | throughput | — | median 28,601 tok/s over steps 6-30 (min 15,759, max 42,680) |
  | step time | — | step 1 6.96 s; median 1.146 s over steps 6-30 |
  | loss | ln(32768) = 10.397 at init | 10.451 at step 1, 9.015 at step 30 |

- Gap: measured peak is 4.3% above the unmargined estimate and 17% below the margined one.
- Time projection (not measured): 2.949e8 tokens at 28.6k tok/s is about 2.9 h for the full 10m run,
  1.9-5.2 h across the observed min/max throughput, excluding evaluation.
- Observations: a second probe with micro-batch 16 x accum 2 (same tokens per step; scratch config,
  output not kept) gave the same step-30 loss to 4 decimals and similar throughput at about half the
  peak memory; the config stays at micro-batch 32 (memory is not a constraint). Step times vary about
  3x; system time was high, consistent with cold reads of the memory-mapped shards.
  **Correction (EXP-008):** the cold-read explanation is not supported. Ladder (a) with
  `--overfit-batch` reads one batch and still spent 47% of wall time in system time. The cause is
  unexplained.
- Caveats: 30 steps is a short sample, all in warmup; throughput may change once shards are cached.
  The micro-batch 16 numbers are not backed by a saved file and are not used elsewhere.

## EXP-008 — Ladder (a): overfit one batch on MPS (Phase 4)
- Date: 2026-10-12
- Git SHA: 5048c38 (clean for tracked files)
- Config: `configs/syrinx-10m.yaml` (micro-batch 32, accum 1, seq 1024, bf16-autocast, seed 1337)
- Command: `uv run python scripts/train.py --config configs/syrinx-10m.yaml --out-dir runs/ladder-a-overfit
  --overfit-batch --stop-at-step 300 --log-interval 10 --eval-interval 100000 --ckpt-interval 100000` (MPS)
- Raw output: [`ladder-a-overfit-2026-10-12.txt`](ladder-a-overfit-2026-10-12.txt); per-step records in
  `runs/ladder-a-overfit/metrics.jsonl` (not tracked)
- Hardware state: AC power
- Tokens / steps: one fixed batch of 32 x 1024 tokens, 300 optimizer steps
- Metrics (source: raw output; `metrics.jsonl` for peak memory, grad norm and throughput):

  | step | 10 | 160 | 220 | 300 |
  |---|---|---|---|---|
  | train loss | 9.9541 | 1.1318 | 0.0259 | 0.0030 |

  No NaN or inf loss; max logged grad norm 3.50. Peak memory 18.24 GiB (`mem_peak`, same as EXP-007).
  Median 27,990 tok/s over logged steps 10-300 (min 12,979, max 42,785).
- Wall clock: real 372.42 s, user 4.31 s, sys 174.96 s (`/usr/bin/time -p`).
- Result: PASS. The loss on a single batch goes to 0.003, so the forward pass, loss, backward pass and
  optimizer update can fit data.
- Observations: system time is 47% of wall time although `--overfit-batch` reads the shards only once.
  This contradicts the EXP-007 explanation (cold memmap reads); see the correction there.
- Caveats: single seed; the loss is logged every 10 steps only.

## EXP-009 — Ladder (b): memorize ~1 MB of TinyStories on MPS (Phase 4)
- Date: 2026-10-12
- Git SHA: 5048c38 (clean for tracked files)
- Config: `configs/syrinx-10m.yaml` (as EXP-008)
- Command: `uv run python scripts/train.py --config configs/syrinx-10m.yaml --out-dir runs/ladder-b-memorize
  --train-token-limit 2.5e5 --stop-at-step 1000 --log-interval 10 --eval-interval 100 --eval-batches 20
  --ckpt-interval 100000` (MPS)
- Raw output: [`ladder-b-memorize-2026-10-12.txt`](ladder-b-memorize-2026-10-12.txt); per-step records in
  `runs/ladder-b-memorize/metrics.jsonl` (not tracked)
- Hardware state: AC power
- Tokens / steps: training restricted to the first 250,000 tokens of the TinyStories train shards
  (`limit_tokens`); about 1.0 MB of text at the 3.988 bytes/token measured on TinyStories val in EXP-003
  (an estimate, not measured on these tokens). 1,000 steps x 32,768 tokens = 32.8M tokens seen, about
  131 passes over the subset. Val: 20 batches of the full TinyStories val shard.
- Metrics (source: raw output):

  | step | 100 | 200 | 300 | 500 | 700 | 900 | 1000 |
  |---|---|---|---|---|---|---|---|
  | train loss | 4.4171 | 2.7251 | — | 0.3219 | — | — | 0.1052 |
  | val loss | 4.5580 | **3.4814** | 3.7862 | 4.8976 | 5.5255 | 6.0212 | 6.2379 |

  Lowest logged train loss 0.0901 at step 960 (`metrics.jsonl`). No NaN or inf; max logged grad norm 2.99.
  Peak memory 18.24 GiB. Median 45,535 tok/s over logged steps 10-1000 (min 9,947, max 49,098).
- Wall clock: real 1011.98 s, user 11.48 s, sys 370.40 s.
- Result: PASS. Train loss falls to about 0.1 while val loss reaches its minimum at step 200 and then
  rises every evaluation to 6.24: the model memorizes the subset instead of generalizing, as intended.
- Observations: median throughput is 1.6x EXP-008's on the same config and device; the cause is not
  measured.
- Caveats: single seed; train loss is a single-batch value at each logged step, val loss a 20-batch mean.

## EXP-010 — Ladder (c): interrupted and resumed run reproduces the uninterrupted one on CPU (Phase 4)
- Date: 2026-10-12
- Git SHA: 067a73e (clean for tracked files)
- Config: `configs/syrinx-10m.yaml` with `micro_batch_size: 4` and `precision: fp32` (scratch copy; the
  diff is printed at the top of the raw output). The unmodified config (micro-batch 32, bf16-autocast)
  did not complete one CPU step in 5.5 min, so that attempt was stopped and not kept.
- Commands (all with `--device cpu --log-interval 1 --eval-interval 100000 --ckpt-interval 100000`):
  - straight: `--out-dir runs/ladder-c-straight --stop-at-step 20`
  - split, part 1: `--out-dir runs/ladder-c-split --stop-at-step 10`
  - split, part 2: `--out-dir runs/ladder-c-split --stop-at-step 20 --resume` (from `step_0000010.pt`)
- Raw output: [`ladder-c-resume-cpu-2026-10-12.txt`](ladder-c-resume-cpu-2026-10-12.txt) (training logs,
  timings, and the comparison output)
- Hardware state: AC power
- Metrics (source: raw output):
  - Per-step train loss and grad norm, steps 1-20: identical in both runs (exact float equality,
    0 differing steps). Step 1 loss 10.470919609069824, step 11 (first step after resume)
    10.002915382385254, step 20 9.619067192077637.
  - Step-20 checkpoints: all 38 model tensors and all optimizer state tensors equal (`torch.equal`).
  - Wall clock: straight 9.82 s; split 5.39 s + 5.31 s. sys time 3.70 s / 1.90 s / 1.86 s.
- Result: PASS. Resume restores model, optimizer, LR schedule (computed from the step), RNG and data
  position exactly on CPU.
- Observations: system time is 35-38% of wall time here, similar to EXP-008/009 on MPS (47% and 37%),
  so these runs do not isolate the cause of the high system time.
- Caveats: CPU only, fp32, micro-batch 4; 10 steps after resume. MPS with the real config: EXP-011.

## EXP-011 — Ladder (c) on MPS with the real 10m config: resume is within MPS run-to-run noise (Phase 4)
- Date: 2026-10-12
- Git SHA: 2e04f62 (clean for tracked files)
- Config: `configs/syrinx-10m.yaml`, unmodified (micro-batch 32, seq 1024, bf16-autocast, seed 1337)
- Commands (all `uv run python scripts/train.py --config configs/syrinx-10m.yaml --device mps
  --log-interval 1 --eval-interval 100000 --ckpt-interval 100000` plus):
  - straight: `--out-dir runs/ladder-c-mps-straight --stop-at-step 20`
  - split: `--out-dir runs/ladder-c-mps-split --stop-at-step 10`, then the same with
    `--stop-at-step 20 --resume`
  - straight repeat (noise floor): `--out-dir runs/ladder-c-mps-straight2 --stop-at-step 20`
- Raw output: [`ladder-c-resume-mps-2026-10-12.txt`](ladder-c-resume-mps-2026-10-12.txt) (training logs,
  timings, per-step comparisons)
- Hardware state: AC power
- Metrics (source: raw output):

  | max \|train loss difference\| | steps 1-10 | steps 11-20 (after the resume) |
  |---|---|---|
  | straight vs straight repeat (noise floor) | 1.72e-05 | 2.48e-05 |
  | straight vs split (resume) | 2.29e-05 | 2.57e-05 |

  Train loss is bit-identical at steps 1-3 in all three runs. Straight vs split: grad norm first differs
  at step 3, loss at step 4, both before the resume point.
  Step-20 checkpoints of straight vs split: weights not bit-equal, max abs difference 4.68e-05.
- Wall clock: straight 15.65 s and 15.50 s; split 8.69 s + 8.56 s. sys time 0.93-1.03 s per run.
- Result: PASS within tolerance. Two uninterrupted MPS runs with the same seed are not bitwise
  identical, so bit-exact resume cannot be shown on MPS. The resumed run differs from the straight run
  by the same amount as two straight runs do, and there is no jump at step 11, so resume adds no error
  beyond MPS run-to-run noise. Bit-exactness of resume itself is shown on CPU (EXP-010).
- Observations: here system time is about 1 s per run, against 175 s in EXP-008 (300 steps) and 370 s in
  EXP-009 (1,000 steps). The high system time grows with run length; the cause is still not measured.
- Caveats: 20 steps, all in warmup; one repeat for the noise floor. The source of MPS nondeterminism
  (kernel reduction order, bf16 autocast) is not identified.

## EXP-012 — Ladder (d): full 10m run on TinyStories, with samples (Phase 4)
- Date: 2026-10-12 21:59 to 2026-10-13 00:28
- Git SHA: 6c8739d (clean for tracked files); samples written by `scripts/run_phase5.sh` at 3acca9f
- Config: `configs/syrinx-10m.yaml` (micro-batch 32, accum 1, seq 1024, bf16-autocast, lr 1e-3 -> 1e-4,
  200 warmup steps, 9,000 steps, seed 1337)
- Command: `caffeinate -i uv run python scripts/train.py --config configs/syrinx-10m.yaml --out-dir
  runs/10m-a` (defaults: log every 10 steps, eval every 250 steps on 20 val batches, checkpoint every 500)
- Raw output: [`10m-a-train-2026-10-12.txt`](10m-a-train-2026-10-12.txt); per-step records in
  `runs/10m-a/metrics.jsonl` (not tracked); samples
  [`samples-10m-a-step9000.jsonl`](samples-10m-a-step9000.jsonl) (temperature 0.8, top-k 50, seeds 0-2)
  and [`samples-10m-a-step9000-greedy.jsonl`](samples-10m-a-step9000-greedy.jsonl) (temperature 0)
- Hardware state: AC power throughout. From 22:50:56 an Ollama `llama-server` process (another
  application) was loaded and shared the GPU until the end of the run.
- Tokens / steps: 9,000 steps x 32,768 = 294,912,000 tokens (`tokens_seen`), about 0.72 of the
  411,455,118-token TinyStories train split (EXP-004), drawn as random windows.
- Metrics (source: raw output and `metrics.jsonl`):

  | step | 250 | 500 | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | 7000 | 8000 | 9000 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | val loss | 2.9117 | 2.3607 | 2.0436 | 1.8545 | 1.7681 | 1.7145 | 1.6666 | 1.6312 | 1.6010 | 1.5827 | **1.5725** |

  Final train loss 1.5620 (single batch at step 9000). The val loss fell at every one of the 36
  evaluations; the lowest is the last. No non-finite loss or grad norm in 900 logged steps; max grad
  norm 3.02 (during warmup), 1.03 after step 200.
- Peak memory: 18.24 GiB measured (`mem_peak`) vs 17.48 GiB estimated (EXP-007), +4.3%.
- Throughput (`tokens_per_sec`, logged steps): median 48,501 tok/s over steps 20-4300 (min 41,149);
  median 26,212 tok/s over steps 4500-9000 (min 10,958, max 46,572). The drop is at step 4500
  (22:51:07), 11 s after `llama-server` started; 4,490 logged 44,898 tok/s and 4,500 logged 22,335.
- Wall clock: real 8,940.55 s (2.48 h), user 84.51 s, sys 3,548.52 s. EXP-007 projected 2.9 h
  (1.9-5.2 h) from a 28.6k tok/s warmup median; the uncontended part of this run ran 1.7x faster than
  that median.
- Checkpoints: 141,646,859 bytes each; `step_0008000.pt`, `step_0008500.pt`, `step_0009000.pt` kept.
- Samples (all three prompts, both settings, in the files above): at temperature 0.8 the continuations
  are TinyStories-style stories with consistent characters, quoted dialogue and an ending, and each
  stops at EOS. Errors: logical drift (a sign that addresses the character by name), a repeated list
  item ("the swings ... and the swings"), and one broken word ("C cones", "C cream"). Greedy decoding
  starts fluently and then loops ("He bit the bread. He bit the bread.").
- Result: rung (d) PASS. The loss curve is smooth and still falling at the end, and the samples are
  coherent at the level PLAN.md asks for ("coherent-ish").
- Observations: system time is again about 40% of wall time (EXP-008 to EXP-011), with user time under
  1%; the cause is still not measured.
- Caveats: one seed; the val loss is a 20-batch mean (655,360 tokens), not the full split; the sample
  judgement is qualitative, from 3 prompts. Half the run shared the GPU with another process, so its
  wall clock and throughput are not a clean measurement of this config.

## EXP-013 — 50M learning-rate sweep (Phase 5)
- Date: 2026-10-13 00:28 to 09:01
- Git SHA: 3acca9f (clean for tracked files)
- Config: `configs/syrinx-50m.yaml` (micro-batch 16 x accum 16 = 262,144 tokens/step, seq 1024,
  bf16-autocast, warmup 200, max_steps 3815, seed 1337) with only `lr` and `min_lr = lr/10` changed per
  run (`scripts/lr_sweep.py make-configs`); data FineWeb-Edu `sample-10BT` shards (EXP-005)
- Commands: `scripts/run_phase5.sh` stage 3, i.e. per LR `uv run python scripts/train.py --out-dir
  runs/50m-sweep/<lr-tag> --config runs/50m-sweep/<lr-tag>.yaml --stop-at-step 300 --log-interval 10
  --eval-interval 100 --eval-batches 100 --ckpt-interval 100 --keep-last 1`; then
  `scripts/lr_sweep.py pick --step 300` (rule in docs/decisions.md, 2026-10-12)
- Raw output: [`50m-sweep-lr-3.0e-04.txt`](50m-sweep-lr-3.0e-04.txt),
  [`50m-sweep-lr-6.0e-04.txt`](50m-sweep-lr-6.0e-04.txt),
  [`50m-sweep-lr-1.2e-03.txt`](50m-sweep-lr-1.2e-03.txt), pick output
  [`50m-lr-sweep-summary.json`](50m-lr-sweep-summary.json), pipeline log
  [`phase5-pipeline.txt`](phase5-pipeline.txt); per-step records in `runs/50m-sweep/*/metrics.jsonl`
  (not tracked)
- Hardware state: AC power throughout. The Ollama `llama-server` process from EXP-012 stayed loaded on
  the GPU for the whole sweep.
- Tokens / steps: 300 steps x 262,144 = 78,643,200 tokens per run, all inside the 200-step warmup and
  the start of the cosine decay. Val: the first 100 val batches (1,638,400 tokens), same windows for
  every run.
- Metrics (source: `metrics.jsonl` and the summary file):

  | peak LR | val@100 | val@200 | val@300 | train@300 | max grad norm after step 200 | stable |
  |---|---|---|---|---|---|---|
  | 3e-4 | 6.7321 | 5.7158 | 5.1935 | 5.2262 | 1.017 | yes |
  | 6e-4 | 6.2522 | 5.4332 | 4.8629 | 4.8934 | 0.861 | yes |
  | **1.2e-3** | **5.9779** | **5.1977** | **4.6752** | 4.7014 | 0.617 | yes |

  No non-finite loss or grad norm in any run; no loss spike above the 1.0-nat threshold.
- Pick: **1.2e-3** (`runs/50m-a.yaml`): lowest val loss at step 300, 0.188 below 6e-4 and 0.518 below
  3e-4; no lower LR within the 0.01 tolerance.
- Peak memory: 17.94 GiB in all three runs vs 16.01 GiB estimated (20.01 GiB with the x1.25 margin):
  +12% over the unmargined estimate, inside the margined one. Proxy (EXP-000): 24.2 GiB driver memory.
- Throughput: median 7,775 / 7,874 / 7,766 tok/s (3e-4 / 6e-4 / 1.2e-3, logged steps 20-300; overall
  range 7,437-8,140); median step time 33.3-33.8 s. The EXP-000 proxy measured 13,362 tok/s for 50M
  mb16 bf16: this is 42% lower, measured while another process shared the GPU (EXP-012 showed that
  process halving the 10m throughput). Not a clean measurement of the 50M config.
- Eval cost: 41-43 s per 100-batch evaluation (log timestamps), against the 41 s estimate.
- Wall clock: 10,255 / 10,224 / 10,245 s per run (2.85 h each). Checkpoint size 610,527,323 bytes
  (estimate 0.61 GB).
- Observations: val loss is ordered by LR at every evaluation (higher LR lower), with the gap between
  1.2e-3 and 6e-4 shrinking from 0.274 at step 100 to 0.188 at step 300. The full 50M run was not
  started: the pipeline controller was stopped at 08:28 by request, the 6e-4 run finished on its own
  (its log therefore has no `exit` line), and the pick was run separately at 09:01 (pipeline log).
- Caveats: the best LR is the highest tried, so the optimum may be above 1.2e-3; per docs/decisions.md
  the sweep should be extended upward (e.g. 2.4e-3) before choosing the 125M LR. 300 steps cover only
  warmup and the start of decay; short runs favour high LRs, so the long-run behaviour of 1.2e-3 is
  untested. One seed per LR. Throughput is contended (see above).

## EXP-014 — Full-split evaluation of the 10m model: bits per byte (Phase 8 tooling)
- Date: 2026-10-13
- Git SHA: 865c60b (clean for tracked files)
- Checkpoint: `runs/10m-a/checkpoints/step_0009000.pt` (EXP-012)
- Commands: `uv run python scripts/eval_checkpoint.py --checkpoint runs/10m-a/checkpoints/step_0009000.pt
  --precision <p> --out experiments/eval-10m-a-step9000-<p>-2026-10-13.json` for `<p>` = `bf16-autocast`
  (the checkpoint's training precision) and `fp32`; batch size 32 and val path from the checkpoint config
- Raw output: [`eval-10m-a-step9000-2026-10-13.txt`](eval-10m-a-step9000-2026-10-13.txt),
  [`eval-10m-a-step9000-bf16-autocast-2026-10-13.json`](eval-10m-a-step9000-bf16-autocast-2026-10-13.json),
  [`eval-10m-a-step9000-fp32-2026-10-13.json`](eval-10m-a-step9000-fp32-2026-10-13.json)
- Hardware state: AC power; the Ollama `llama-server` process was still loaded (timing only).
- Data: TinyStories val shard (2,074,361 tokens, EXP-004), 2,025 non-overlapping 1,024-token windows =
  2,073,600 target tokens (the last 760 targets of the shard do not fill a window), 8,110,848 bytes,
  9,118 EOS targets (in the loss, 0 bytes).
- Metrics (source: the JSON files):

  | precision | loss (nats/token) | perplexity | bits per byte | seconds |
  |---|---|---|---|---|
  | bf16-autocast | 1.56407 | 4.7782 | **0.57688** | 25.0 |
  | fp32 | 1.56390 | 4.7774 | 0.57682 | 23.9 |

- Observations: the full-split loss (1.5641) is 0.008 below the 20-batch training-time val loss at
  step 9000 (1.5725, EXP-012), so the 20-batch estimate was close. bf16 and fp32 differ by 1.7e-4 nats
  per token. Bytes per target token here are 3.911 (8,110,848 / 2,073,600), lower than the 3.988 of
  EXP-003 because EOS targets count 0 bytes.
- Caveats: non-overlapping windows, so early positions in each window have short context; a
  sliding-window evaluation would give a somewhat lower number. Bits per byte is comparable across
  tokenizers only for the same text and the same window convention. Single checkpoint, single split;
  timings were measured with another process on the GPU.

## EXP-015 — Hugging Face export of the 10m model: parity and HF-only generation (Phase 9)
- Date: 2026-10-13
- Git SHA: 83a7cc9 (clean for tracked files)
- Checkpoint: `runs/10m-a/checkpoints/step_0009000.pt` (EXP-012)
- Command: `uv run python scripts/export_hf.py --checkpoint runs/10m-a/checkpoints/step_0009000.pt
  --out-dir exports/syrinx-10m` (defaults: parity batch 2, seq len = max_seq_len 1024, seed 0); then an
  HF-only generation check with `AutoModelForCausalLM` / `AutoTokenizer` and the exported
  `generation_config.json` (`torch.manual_seed(0)`, prompt `<|endoftext|>Once upon a time`, 200 new tokens max)
- Raw output: [`export-10m-a-step9000-2026-10-13.txt`](export-10m-a-step9000-2026-10-13.txt) (script
  output, file sizes, SHA-256 of every exported file, the generation record) and
  [`export-10m-a-step9000-parity-2026-10-13.json`](export-10m-a-step9000-parity-2026-10-13.json)
  (copy of `parity.json`); the export itself is in `exports/syrinx-10m/` (not tracked)
- Hardware state: AC power
- Metrics (source: the parity file):
  - Logit parity, Syrinx vs the reloaded `LlamaForCausalLM`, fp32 on the CPU, 2 x 1,024 random ids:
    max abs difference **0.0** (tolerance 1e-4). PASS.
  - Tokenizer parity on the 4 `PARITY_TEXTS` (accents, emoji, CJK, digits, tabs, `\r\n`): HF ids equal
    ours and decoding is lossless. PASS.
  - Export: `model.safetensors` 47,199,272 bytes (fp32, tied head not stored), `tokenizer.json`
    2,328,862 bytes, plus `config.json`, `tokenizer_config.json`, `generation_config.json`; transformers 5.17.0.
  - HF-only generation: sampling with the exported defaults (do_sample, temperature 0.8, top-k 50) produced
    175 new tokens and stopped at `<|endoftext|>`; the text is a coherent TinyStories-style story.
- Result: Gate for Phase 9 on the 10m model met (parity at most 1e-4). The difference is exactly 0, as
  expected: both sides run the same fp32 operations in the same order on the CPU.
- Caveats: parity uses random ids, not natural text; one checkpoint (10m). The 50M and 125M exports need
  their own parity runs. Generation was checked on one prompt and one seed.

## EXP-016 — Benchmark contamination: 13-gram overlap with all training shards (Phase 8)
- Date: 2026-10-14
- Git SHA: c0e8045 (clean for tracked files)
- Benchmarks: HellaSwag validation, ARC-Easy test, PIQA validation, LAMBADA (OpenAI) test, fetched at pinned
  revisions (docs/data.md, "Evaluation benchmarks"; [`benchmarks-fetch-2026-10-14.txt`](benchmarks-fetch-2026-10-14.txt))
- Training text scanned: every train shard any model used: TinyStories train (5 shards) and FineWeb-Edu
  train (26 shards); the tokenizer's training text is a subset of the FineWeb-Edu train split
- Command: `uv run python scripts/check_contamination.py --benchmark hellaswag=... --benchmark arc_easy=...
  --benchmark piqa=... --benchmark lambada=... --workers 16 --out experiments/contamination-2026-10-14.json`
- Method: word 13-grams after NFKC + lowercase + alphanumeric-run splitting; an example is contaminated if
  any of its 13-grams occurs inside one training document (hash candidates confirmed word by word);
  examples with fewer than 13 words are "unchecked". Text checked per example: HellaSwag context + all
  endings, ARC question + all choices, PIQA goal + both solutions, LAMBADA passage.
- Raw output: [`contamination-2026-10-14.json`](contamination-2026-10-14.json) (all contaminated ids, first
  5 matched n-grams per benchmark, per-shard timings), [`contamination-2026-10-14.txt`](contamination-2026-10-14.txt)
- Hardware state: battery power (72% at start), CPU only (16 worker processes)
- Scale: 4,139,942 documents, 2,154,745,552 words, 1,665,559 distinct benchmark 13-grams (index built from
  19,409 examples, 229 of them unchecked).
- Metrics (source: the JSON file):

  | benchmark | examples | unchecked | checked | contaminated | % of checked |
  |---|---|---|---|---|---|
  | HellaSwag | 10,042 | 0 | 10,042 | 40 | 0.40 |
  | ARC-Easy | 2,376 | 76 | 2,300 | 22 | 0.96 |
  | PIQA | 1,838 | 153 | 1,685 | 3 | 0.18 |
  | LAMBADA | 5,153 | 0 | 5,153 | 0 | 0.00 |

- Time: 368 s wall (`/usr/bin/time`), against an estimate of 3-5 min made from one shard scanned alone
  (51.6 s); with 16 workers in parallel a full FineWeb-Edu shard took up to 180 s (the small last shard
  1.2 s) and the four large TinyStories shards 195-197 s each, so per-shard throughput was about 3.5x lower
  than the single-shard estimate (battery power, 16 concurrent workers).
- Observations: most matched n-grams in the report are common phrases or facts that also occur in
  ordinary web text (e.g. "the most common eating disorders are anorexia nervosa bulimia nervosa and binge
  eating", the "20 20 20" eye-strain rule); some ARC-Easy matches contain question wording ("which of the
  following elements is best able to combine with itself and"), consistent with quiz pages in FineWeb-Edu.
  A flagged example therefore shares 13 words with training text; it is an upper bound on leakage, not a
  confirmed leak. Benchmark scores will be reported on all examples and on the clean subset.
- Process note: a first attempt on the same day used `np.isin` against the index, which re-sorts the
  1.67M-entry index per document (measured 861 ms per document, about 60 h projected); it was stopped
  after 31 min with no output and replaced by a binary search (commit c9955fe, 0.024 ms per document).
- Caveats: exact 13-gram overlap after normalization misses paraphrases and shorter overlaps; unchecked
  examples (229) are not known to be clean; only the matched n-grams listed in the JSON were inspected by eye.

## EXP-017 — Benchmarks for the 10m model, overall and on clean examples (Phase 8)
- Date: 2026-10-14
- Git SHA: 9ef7d6f (clean for tracked files)
- Model: `exports/syrinx-10m` (EXP-015; the 10m TinyStories model of EXP-012, step 9000)
- Command: `uv run python scripts/run_benchmarks.py --model-dir exports/syrinx-10m --out
  experiments/benchmarks-10m-a-2026-10-14.json` (lm-eval 0.4.13, local task configs on the pinned files,
  0-shot, fp32, `add_bos_token=True`, batch size 32, MPS, offline)
- Raw output: [`benchmarks-10m-a-2026-10-14.json`](benchmarks-10m-a-2026-10-14.json) (all metrics, every
  subset, hashes and versions), [`benchmarks-10m-a-2026-10-14.txt`](benchmarks-10m-a-2026-10-14.txt)
- Hardware state: battery power
- Metrics (source: the JSON file; ± is the binomial standard error; chance for ARC-Easy computed from the
  file: 2,365 four-choice, 7 three-choice, 4 five-choice questions):

  | benchmark | metric | all (n) | clean (n) | chance | all vs chance |
  |---|---|---|---|---|---|
  | HellaSwag | acc_norm | 0.2586 ± 0.0044 (10,042) | 0.2587 ± 0.0044 (10,002) | 0.25 | +2.0 SE |
  | ARC-Easy | acc | 0.2698 ± 0.0091 (2,376) | 0.2722 ± 0.0093 (2,278) | 0.2502 | +2.2 SE |
  | PIQA | acc_norm | 0.5305 ± 0.0116 (1,838) | 0.5351 ± 0.0122 (1,682) | 0.5 | +2.6 SE |
  | LAMBADA | acc | 0.0778 ± 0.0037 (5,153) | 0.0778 ± 0.0037 (5,153) | ~0 | |
  | LAMBADA | perplexity | 75,330 | 75,330 | | |

  Also recorded: HellaSwag acc 0.2614, ARC-Easy acc_norm 0.2698, PIQA acc 0.5419. Contaminated subsets:
  HellaSwag 40, ARC-Easy 22, PIQA 3 examples (too few for any conclusion). Unchecked: ARC-Easy 76, PIQA 153.
  Unmatched: 0 in every benchmark. The "all" value equals lm-eval's own aggregate for every metric.
- Time: 111.1 s harness time (113.6 s wall), against an estimate of about 6 min scaled from a
  200-example-per-task run (13.3 s); the estimate scaled the fixed start-up cost along with the work.
- Observations: on the three multiple-choice benchmarks the 10m model is 2.0-2.6 standard errors above
  chance; LAMBADA accuracy is 7.8% with perplexity 75,330. This model was trained only on TinyStories
  (children's stories), and these benchmarks are general-domain text, so near-chance results are the
  expected outcome, not a defect. Clean-subset scores are within one standard error of the full-set
  scores, so the flagged contamination (EXP-016) does not change any number here.
- Caveats: one model, one run, 0-shot; scores near chance are sensitive to prompt format, so they are only
  comparable to runs with the same task configs (`task_config_sha256` in the JSON). Not comparable with
  published scores of other models without matching data, prompts and normalization.

## EXP-018 — Full 50M run on FineWeb-Edu, 1.0B tokens (Phase 5)
- Date: 2026-10-14 15:07 to 2026-10-30 13:43, in eight segments (below)
- Git SHA: d98304c, 4b05e9a, 37cd55f, 050cc05, ee9be1e (one per segment; `git diff d98304c ee9be1e`
  touches no model, training or data code, only `src/syrinx/model_card.py` and `src/syrinx/utils/throughput.py`)
- Config: [`50m-a-config.yaml`](50m-a-config.yaml) (copy of `runs/50m-a.yaml`: `configs/syrinx-50m.yaml`
  with the EXP-013 pick, lr 1.2e-3 -> 1.2e-4; micro-batch 16 x accum 16 = 262,144 tokens/step, seq 1024,
  bf16-autocast, warmup 200, 3,815 steps, seed 1337)
- Command: `scripts/run_phase5.sh` stage 5, i.e. `uv run python scripts/train.py --out-dir runs/50m-a
  [--resume] --config runs/50m-a.yaml --log-interval 10 --eval-interval 250 --eval-batches 100
  --ckpt-interval 250 --keep-last 3 --milestone-every 1000`, launched with `nohup caffeinate -is` from Terminal.app
- Raw output: [`50m-a-train.txt`](50m-a-train.txt) (every segment), pipeline log
  [`phase5-pipeline.txt`](phase5-pipeline.txt) (start, stop and resume notes); per-step records in
  `runs/50m-a/metrics.jsonl` (not tracked)
- Segments (from the log headers and the last log line of each; every stop was by request):

  | # | start | end | steps run | power | resumed from |
  |---|---|---|---|---|---|
  | 1 | 10-14 15:07 | 10-14 18:57 | 1-750 | AC | |
  | 2 | 10-15 10:05 | 10-15 11:04 | 751-950 | AC | step 750 |
  | 3 | 10-15 11:18 | 10-15 13:44 | 751-1250 | AC | step 750 |
  | 4 | 10-16 09:35 | 10-16 12:21 | 1251-1820 | AC | step 1250 |
  | 5 | 10-28 20:17 | 10-28 20:28 | 1751-1770 | battery | step 1750 |
  | 6 | 10-29 09:45 | 10-29 16:32 | 1751-3150 | AC | step 1750 |
  | 7 | 10-30 09:42 | 10-30 12:08 | 3001-3500 | AC | step 3000 |
  | 8 | 10-30 12:10 | 10-30 13:43 | 3501-3815 | AC (battery at the end) | step 3500 |

  Segment 2 has no stop note in the pipeline log; segment 3 restarted from the step-750 checkpoint
  under a new commit. 440 steps were run twice (steps past the last checkpoint at each stop).
- Hardware state: AC power except segment 5 and the final minutes of segment 8. Other GPU processes
  were not checked; the AC throughput is steady across segments (below).
- Tokens / steps: 3,815 steps x 262,144 = 1,000,079,360 tokens (`tokens_seen` at step 3815), about 0.40
  of the 2,500,678,395-token FineWeb-Edu train split (EXP-005), drawn as random windows.
- Metrics (source: `metrics.jsonl`, val = mean over the first 100 val batches, 1,638,400 tokens):

  | step | 250 | 500 | 750 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 3750 | 3815 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | val loss | 4.8854 | 4.1354 | 3.8626 | 3.7318 | 3.5781 | 3.4851 | 3.4142 | 3.3638 | 3.3319 | 3.3231 | **3.3217** |

  Final train loss 3.3581 (single batch at step 3815). The val loss fell at every one of the 16
  evaluations. No non-finite loss or grad norm in 426 logged train records; max grad norm 4.41 (step 20,
  warmup), 0.737 after step 200.
- Resume: steps 1760-1820 logged the same losses to 4 decimals in segments 4, 5 and 6, and steps 1760-1770
  in segments 5 and 6 (one of them on battery). Step 760 logged 3.8234 (segment 2) and 3.8235 (segment 3),
  a 1e-4 difference within the MPS run-to-run noise of EXP-011.
- Peak memory: 17.94 GiB measured (`mem_peak`, every segment) vs 16.01 GiB estimated (20.01 GiB with the
  x1.25 margin, EXP-013): +12% over the unmargined estimate, inside the margined one.
- Throughput (`tokens_per_sec`, logged steps >= 20): median 15,091-15,192 tok/s per AC segment (median
  step time 17.26-17.37 s); segment 5 on battery 7,715 tok/s (33.98 s/step), half the AC rate, as in
  EXP-001. The 7,830 tok/s minimum in segments 7-8 is one logged step; its cause is not identified.
- Wall clock: 20.97 h summed over the eight segments (header to last log line), including 440 repeated
  steps and the 16 evaluations; 3,815 steps at the 17.27 s median would be 18.3 h. Only segment 8 has a
  `/usr/bin/time` line (real 5,578.07 s).
- Checkpoints: 610,527,643 bytes each; kept `step_0001000.pt`, `step_0002000.pt`, `step_0003000.pt`
  (milestones) and `step_0003500.pt`, `step_0003750.pt`, `step_0003815.pt`.
- Result: the 50M model trained to its full 1.0B-token budget at the swept LR without instability.
- Caveats: one seed; the val losses here are 100-batch estimates (the full split is EXP-019). The run
  was stopped and resumed seven times; resume reproduced the logged losses (above), but a run without
  interruptions was not done for comparison.

## EXP-019 — Full-split evaluation of the 50M model (Phase 5 wrap-up)
- Date: 2026-10-30
- Git SHA: ee9be1e (tracked files clean apart from run logs)
- Checkpoint: `runs/50m-a/checkpoints/step_0003815.pt` (EXP-018)
- Commands: `uv run python scripts/eval_checkpoint.py --checkpoint runs/50m-a/checkpoints/step_0003815.pt
  --precision <p> --out experiments/eval-50m-a-step3815-<p>.json` for `<p>` = `bf16-autocast` (the
  training precision; the command `scripts/run_phase6.sh` uses) and `fp32`; batch size 16 and val path
  from the checkpoint config
- Raw output: [`eval-50m-a-step3815.txt`](eval-50m-a-step3815.txt),
  [`eval-50m-a-step3815-bf16-autocast.json`](eval-50m-a-step3815-bf16-autocast.json),
  [`eval-50m-a-step3815-fp32.json`](eval-50m-a-step3815-fp32.json)
- Hardware state: AC power, MPS
- Data: FineWeb-Edu val shard, 11,929 non-overlapping 1,024-token windows = 12,215,296 target tokens,
  54,310,934 bytes, 11,512 EOS targets (in the loss, 0 bytes).
- Metrics (source: the JSON files):

  | precision | loss (nats/token) | perplexity | bits per byte | seconds |
  |---|---|---|---|---|
  | bf16-autocast | 3.32063 | 27.678 | **1.07749** | 140.1 |
  | fp32 | 3.32049 | 27.674 | 1.07744 | 282.3 |

- Time: bf16 140 s against an estimate of ~2.5 min, scaled from the 100-batch training-time evaluation
  (19 s, EXP-018 log timestamps) to 746 batches.
- Observations: the full-split loss (3.3206) is 0.001 below the 100-batch value at step 3815 (3.3217,
  EXP-018). bf16 and fp32 differ by 1.4e-4 nats per token (EXP-014: 1.7e-4). fp32 took 2.0x as long.
  Bytes per target token: 4.446 (54,310,934 / 12,215,296).
- Caveats: same window convention as EXP-014 (non-overlapping, so early positions have short context).
  Not comparable with the 10m numbers of EXP-014: different text (TinyStories vs FineWeb-Edu).

## EXP-020 — Hugging Face export of the 50M model: parity and HF-only generation (Phase 9)
- Date: 2026-10-30
- Git SHA: ee9be1e (tracked files clean apart from run logs)
- Checkpoint: `runs/50m-a/checkpoints/step_0003815.pt` (EXP-018)
- Command: `uv run python scripts/export_hf.py --checkpoint runs/50m-a/checkpoints/step_0003815.pt
  --out-dir exports/syrinx-50m` (defaults as in EXP-015); then the EXP-015 HF-only generation check with
  prompt `<|endoftext|>The water cycle is`, 150 new tokens max, `torch.manual_seed(0)`
- Raw output: [`export-50m-a-step3815.txt`](export-50m-a-step3815.txt) (script output, file sizes,
  SHA-256 of every exported file, the generation record) and
  [`export-50m-a-step3815-parity.json`](export-50m-a-step3815-parity.json) (copy of `parity.json`);
  the export itself is in `exports/syrinx-50m/` (not tracked)
- Hardware state: AC power
- Metrics (source: the parity file and the raw output):
  - Logit parity, fp32 on the CPU, 2 x 1,024 random ids: max abs difference **0.0** (tolerance 1e-4). PASS.
  - Tokenizer parity on the 4 `PARITY_TEXTS`: PASS. `tokenizer.json` has the same SHA-256 as the 10m export.
  - Export: `model.safetensors` 203,476,920 bytes (fp32, tied head not stored), `LlamaForCausalLM`
    config hidden 512, 10 layers, 8 heads (8 KV heads), intermediate 1536, vocab 32,768, context 1,024,
    RoPE theta 10,000, tied embeddings; transformers 5.17.0. Export took 6.6 s wall.
  - HF-only generation: sampling with the exported defaults (temperature 0.8, top-k 50) produced 150 new
    tokens without reaching EOS. The text is fluent English on the prompt's topic but factually confused
    and repetitive ("the process of changing the water cycle due to changes in the temperature").
- Result: Phase 9 parity gate met for the 50M model.
- Caveats: parity uses random ids; generation was checked on one prompt and one seed. The first command
  attempt failed before running (a shell quoting error, exit 127) and wrote nothing; its log lines were removed.

