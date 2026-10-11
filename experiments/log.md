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
