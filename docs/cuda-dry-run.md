# CUDA dry run (before any training on a rented GPU)

Syrinx is written device-agnostic (`.to(device)`, `resolve_device()`), but until this checklist
passes it has only ever run on MPS and CPU. Work through it top to bottom on the CUDA machine.
Each step has a pass condition; stop at the first failure and record it. Nothing here is a
result yet: every number below that is not marked as a threshold is an estimate to compare
against.

Target: one NVIDIA GPU with at least 80 GB (an H100 80 GB or an RTX PRO 6000 96 GB), Linux,
the repo at the same commit as the Mac. Blackwell cards (RTX PRO 6000) need a torch build
with CUDA 12.8 or newer; step 1 checks that the GPU actually runs a kernel.

## 0. Before renting

- [ ] Add a `docs/decisions.md` entry: which models train where, and why. The README, model
      cards and posts say "trained on a single local device"; that wording must change for any
      model trained on the GPU.
- [ ] Decide what moves: the token shards are 4.7 GB (`data_cache/tokens/fineweb-edu/train`),
      plus 23 MB val and the 2.2 MB tokenizer (`tokenizer/syrinx-bpe-32k`). Copying them is
      faster than rebuilding them; rebuilding needs `scripts/build_shards.py` and a dataset download.
- [ ] Note the commit: `git rev-parse --short HEAD`.

## 1. Environment

```bash
git clone https://github.com/DhruvParmar051/syrinx-lm.git && cd syrinx-lm
uv sync --all-extras
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
- [ ] Pass: torch 2.14.0 (the version in `uv.lock`), a CUDA version printed (not `None`),
      `True`, and the GPU you rented. A `None` CUDA version means a CPU-only wheel.

```bash
uv run python scripts/check_env.py --device cuda --out experiments/env-cuda-$(date +%F).txt
```
- [ ] Pass: the report runs, the bf16 probe says supported, and the matmul benchmark prints a number.

## 2. Tests

```bash
SYRINX_DEVICE=cuda uv run pytest -q -rs
uv run pytest tests/test_accelerators.py -q -rs
```
- [ ] Pass: no failures. On the GPU machine the `cuda`-marked tests must show as **passed**,
      not skipped (`-rs` lists every skip and why). MPS tests skip there; that is expected.

What `tests/test_accelerators.py` checks on CUDA (the same tests already pass on MPS):

| Test | Catches |
|---|---|
| `test_bf16_probe_passes` | bf16 autocast forward/backward broken or inaccurate |
| `test_real_model_matches_cpu_fp32` | a kernel or a buffer (RoPE tables) that differs from CPU |
| `test_bf16_train_step_tracks_cpu_fp32` | bf16 training step that is non-finite or drifts from fp32 |
| `test_checkpoint_from_cpu_resumes_on_accelerator` | a checkpoint from another device that resumes with lost or misplaced AdamW state |
| `test_timer_and_memory_stats` | timing without synchronize; the untested CUDA branch of `memory_stats` |

## 3. Memory plan

The Mac config (micro-batch 8 x 64 accumulation = 524,288 tokens per step) is sized for 48 GB.
On 80-96 GB, keep the same tokens per step with a larger micro-batch, so results stay comparable:

```bash
uv run python scripts/estimate_memory.py configs/syrinx-125m.yaml --micro-batch 32
```
- [ ] Estimates (2026-10-16, `estimate_memory.py`, x1.25 margin): micro-batch 8 -> 16.4 GiB,
      32 -> 57.7 GiB, 64 -> 112.8 GiB. So micro-batch 32 with `grad_accum_steps: 16`
      (32 x 16 x 1024 = 524,288) is the candidate on both 80 GB and 96 GB; 64 does not fit either.
- [ ] Make a scratch copy of the config with those two values. Do not edit
      `configs/syrinx-125m.yaml`: the Mac runs still use it.

## 4. Short run: does it train?

```bash
uv run python scripts/train.py --config <scratch-125m-mb32.yaml> --out-dir runs/cuda-dryrun \
    --device cuda --stop-at-step 30 --log-interval 1 --eval-interval 30 --eval-batches 20 \
    --ckpt-interval 15 --keep-last 2
```
- [ ] Step-1 loss within 0.15 of ln(32768) = 10.40 (the `test_initial_loss_near_log_vocab`
      tolerance).
- [ ] Loss falls over the 30 steps; grad norm finite on every step.
- [ ] `mem_peak` in `metrics.jsonl` is at or under the 57.7 GiB estimate plus 25%; record the gap.
- [ ] The log says `Using device cuda`.

## 5. Resume on CUDA

```bash
uv run python scripts/train.py --config <scratch-125m-mb32.yaml> --out-dir runs/cuda-dryrun \
    --device cuda --resume --stop-at-step 30 --log-interval 1 --eval-interval 30 \
    --eval-batches 20 --ckpt-interval 15 --keep-last 2
```
Run step 4 with `--stop-at-step 15` first, then this to finish at 30, in a fresh out-dir.
- [ ] Pass: steps 16-30 match a straight 30-step run within run-to-run noise (EXP-011 is
      the MPS comparison for how much noise to expect).

## 6. Same checkpoint, same answer on both machines

Evaluate one existing Mac checkpoint on both devices with the same settings:

```bash
uv run python scripts/eval_checkpoint.py --checkpoint <50m step_0001250.pt> \
    --precision bf16-autocast --max-batches 50 --device cuda --out experiments/parity-cuda.json
```
Run the same command on the Mac with `--device mps`.
- [ ] Pass: val loss agrees to within 1e-2 (bf16 tolerance). A larger gap means the two
      backends compute different things, and nothing trained on one can be compared with the other.

## 7. Throughput

```bash
uv run python scripts/train.py --config <scratch-125m-mb32.yaml> --out-dir runs/cuda-bench \
    --device cuda --stop-at-step 300 --log-interval 10 --eval-interval 300 --eval-batches 20 \
    --ckpt-interval 300 --keep-last 1
uv run python scripts/throughput_report.py runs/cuda-bench/metrics.jsonl --window-minutes 5
```
- [ ] Record the median tok/s. Back-of-envelope for an H100 is roughly 0.3-0.6M tok/s
      (about 1-2.5 h for 2.5B tokens); expect an RTX PRO 6000 to be slower. These are
      estimates, not targets. The report's
      `decision` field uses the Mac Gate 6a thresholds and does not apply here.

## 8. Known gaps (fix only if a step above needs it)

- **CUDA RNG is not checkpointed** (`capture_rng_state` saves CPU, NumPy, Python and MPS
  state). Today nothing draws random numbers on the GPU (no dropout; the loader uses NumPy),
  so resume is unaffected. If GPU randomness is ever added, add `torch.cuda` state too.
- **No TF32 setting.** Only matters for fp32 runs; training uses bf16 autocast.
- **No `torch.compile` or pinned-memory loading.** Throughput only. Try them after steps 1-7
  pass, one at a time, and re-run step 6 after each change.

## 9. Record

- [ ] An `experiments/log.md` entry for the dry run: commit, GPU, driver and CUDA versions,
      each step's pass/fail, peak memory against the estimate, and measured tok/s.
- [ ] Only after every box above is ticked: plan the real run.
