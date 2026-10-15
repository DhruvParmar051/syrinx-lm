#!/usr/bin/env bash
# Unattended Phase 5 wrap-up -> Phase 6 pipeline (docs/decisions.md, 2026-10-15):
#   1. wait for the 50M run (scripts/run_phase5.sh) to write its final checkpoint
#   2. 50M wrap-up: full-split eval (bf16 autocast), HF export + parity, benchmarks; a failure
#      here is logged and does not block the 125M run
#   3. 125M LR sweep: 1.2e-3, 6e-4, 2.4e-3, 300 steps each on the full-run schedule
#   4. Gate 6a on the first sweep run: median tok/s over its last 15 min against the proxy;
#      anything but "full-budget" without a >30% shortfall STOPS here (the decision is Dhruv's;
#      rerun with GATE6A_ACCEPT=1 once it is made; that is recorded in experiments/ and remembered)
#   5. pick the LR (scripts/lr_sweep.py rule) -> runs/125m-a.yaml
#   6. full 125M run (2.5B tokens) with eval every 250 steps (100 batches), checkpoints every 250,
#      keep 3, milestones every 1000
#
# Every stage skips itself when its output exists and resumes from the newest checkpoint when
# one exists, so rerunning this script after a crash, sleep or Gate 6a stop continues the
# pipeline. Run from the repo root, under caffeinate so the Mac does not idle-sleep:
#
#   nohup caffeinate -is scripts/run_phase6.sh >> experiments/phase6-pipeline.txt 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."

RUN50=runs/50m-a
RUN50_FINAL=$RUN50/checkpoints/step_0003815.pt
EXPORT50=exports/syrinx-50m
EVAL50=experiments/eval-50m-a-step3815-bf16-autocast.json
PARITY50=experiments/export-50m-a-step3815-parity.json
BENCH50=experiments/benchmarks-50m-a.json

SWEEP_LRS=(1.2e-3 6e-4 2.4e-3)   # the first one is also the Gate 6a benchmark
SWEEP_STEPS=300
SWEEP_DIR=runs/125m-sweep
GATE=experiments/gate6a-125m.json
GATE_ACCEPTED=experiments/gate6a-125m-accepted.txt   # written once by GATE6A_ACCEPT=1
RUN125=runs/125m-a
RUN125_CONFIG=runs/125m-a.yaml
RUN125_FINAL=$RUN125/checkpoints/step_0004769.pt
MAX_ATTEMPTS=3

log() { echo "# $(date '+%Y-%m-%d %H:%M:%S') | git $(git rev-parse --short HEAD) | $(pmset -g batt | head -1) | $*"; }
die() { log "STOP: $*"; exit 1; }
has_checkpoint() { compgen -G "$1/checkpoints/step_*.pt" > /dev/null; }

# train_stage <out_dir> <final_ckpt> <log_file> <train.py args...>
# Fresh start, or --resume from the newest checkpoint; retried up to MAX_ATTEMPTS times.
train_stage() {
    local out=$1 final=$2 logfile=$3
    shift 3
    local attempt
    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
        [[ -f $final ]] && return 0
        local mode=()
        if has_checkpoint "$out"; then
            mode=(--resume)
        elif [[ -e $out ]]; then
            local aside="$out.incomplete-$(date +%Y%m%d-%H%M%S)-$$-$attempt"
            log "moving $out (no checkpoint) to $aside"
            mv "$out" "$aside"
        fi
        log "train attempt $attempt: --out-dir $out ${mode[*]-} $*" | tee -a "$logfile"
        /usr/bin/time -p uv run python scripts/train.py --out-dir "$out" ${mode[@]+"${mode[@]}"} "$@" >> "$logfile" 2>&1
        local status=$?
        echo "exit $status" >> "$logfile"
        log "train attempt $attempt for $out exited $status"
    done
    [[ -f $final ]]
}

log "pipeline start"

# 1. wait for the 50M run ----------------------------------------------------------------
while [[ ! -f $RUN50_FINAL ]]; do
    if pgrep -f "scripts/run_phase5.sh" > /dev/null || pgrep -f "scripts/train.py .*--out-dir $RUN50( |$)" > /dev/null; then
        sleep 60
    else
        die "50M run is not running and $RUN50_FINAL is missing; resume it with scripts/run_phase5.sh"
    fi
done
log "50M run complete: $RUN50_FINAL"

# 2. 50M wrap-up (failures are logged, not fatal) -----------------------------------------
if [[ ! -f $EVAL50 ]]; then
    uv run python scripts/eval_checkpoint.py --checkpoint "$RUN50_FINAL" --precision bf16-autocast \
        --out "$EVAL50" >> experiments/eval-50m-a-step3815.txt 2>&1 \
        || log "50M full-split eval FAILED (see experiments/eval-50m-a-step3815.txt)"
fi
if [[ ! -f $EXPORT50/parity.json ]]; then
    if [[ -e $EXPORT50 ]]; then
        aside="$EXPORT50.incomplete-$(date +%Y%m%d-%H%M%S)-$$"
        log "moving $EXPORT50 (no parity.json) to $aside"
        mv "$EXPORT50" "$aside"
    fi
    uv run python scripts/export_hf.py --checkpoint "$RUN50_FINAL" --out-dir "$EXPORT50" \
        >> experiments/export-50m-a-step3815.txt 2>&1 \
        || log "50M export or parity FAILED (see experiments/export-50m-a-step3815.txt)"
fi
[[ -f $EXPORT50/parity.json && ! -f $PARITY50 ]] && cp "$EXPORT50/parity.json" "$PARITY50"
if [[ -f $PARITY50 ]] && grep -q '"passed": true' "$PARITY50" && [[ ! -f $BENCH50 ]]; then
    uv run python scripts/run_benchmarks.py --model-dir "$EXPORT50" --out "$BENCH50" \
        >> experiments/benchmarks-50m-a.txt 2>&1 \
        || log "50M benchmarks FAILED (see experiments/benchmarks-50m-a.txt)"
fi
log "50M wrap-up: eval $([[ -f $EVAL50 ]] && echo ok || echo MISSING)," \
    "parity $([[ -f $PARITY50 ]] && grep -q '"passed": true' "$PARITY50" && echo passed || echo MISSING/FAILED)," \
    "benchmarks $([[ -f $BENCH50 ]] && echo ok || echo MISSING)"

# 3-4. 125M LR sweep, with Gate 6a after the first run ------------------------------------
uv run python scripts/lr_sweep.py make-configs --base configs/syrinx-125m.yaml \
    --lrs "${SWEEP_LRS[@]}" --out-dir "$SWEEP_DIR" || die "making sweep configs failed"
sweep_runs=()
for lr in "${SWEEP_LRS[@]}"; do
    tag=$(printf 'lr-%.1e' "$lr")   # same name as scripts/lr_sweep.py lr_tag
    run=$SWEEP_DIR/$tag
    final=$run/checkpoints/$(printf 'step_%07d.pt' "$SWEEP_STEPS")
    train_stage "$run" "$final" "experiments/125m-sweep-$tag.txt" \
        --config "$run.yaml" --stop-at-step "$SWEEP_STEPS" --log-interval 10 \
        --eval-interval 100 --eval-batches 100 --ckpt-interval 100 --keep-last 1 \
        || log "sweep run $run did not reach step $SWEEP_STEPS; it will count as unstable"
    [[ -f $run/config.yaml ]] && sweep_runs+=("$run")

    if [[ ! -f $GATE ]]; then
        [[ -f $final ]] || die "Gate 6a needs a finished sweep run; $run did not finish"
        uv run python scripts/throughput_report.py "$run/metrics.jsonl" --window-minutes 15 \
            --out "$GATE" || die "Gate 6a report failed"
        log "Gate 6a: $(cat "$GATE")"
    fi
    if [[ ${GATE6A_ACCEPT:-0} == 1 && ! -f $GATE_ACCEPTED ]]; then
        log "Gate 6a accepted by hand (GATE6A_ACCEPT=1): $(cat "$GATE")" | tee "$GATE_ACCEPTED"
    fi
    if [[ ! -f $GATE_ACCEPTED ]] \
        && { ! grep -q '"decision": "full-budget"' "$GATE" || grep -q '"investigate": true' "$GATE"; }; then
        die "Gate 6a did not pass automatically ($GATE). Decide the token budget (edit
configs/syrinx-125m.yaml max_steps if it changes), then rerun with GATE6A_ACCEPT=1"
    fi
done
[[ ${#sweep_runs[@]} -gt 0 ]] || die "no sweep run started"

# 5. pick the LR ----------------------------------------------------------------------------
if [[ ! -f $RUN125_CONFIG ]]; then
    uv run python scripts/lr_sweep.py pick --runs "${sweep_runs[@]}" --step "$SWEEP_STEPS" \
        --base configs/syrinx-125m.yaml --config-out "$RUN125_CONFIG" \
        --summary-out experiments/125m-lr-sweep-summary.json \
        || die "no stable sweep run; not starting the 125M run (see experiments/125m-lr-sweep-summary.json)"
fi
log "125M config: $RUN125_CONFIG (lr $(grep -E '^  lr:' "$RUN125_CONFIG"))"

# 6. full 125M run --------------------------------------------------------------------------
train_stage "$RUN125" "$RUN125_FINAL" experiments/125m-a-train.txt \
    --config "$RUN125_CONFIG" --log-interval 10 --eval-interval 250 --eval-batches 100 \
    --ckpt-interval 250 --keep-last 3 --milestone-every 1000 \
    || die "125M run did not finish"
log "pipeline complete: $RUN125_FINAL"
