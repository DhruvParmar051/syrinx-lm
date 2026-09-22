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
