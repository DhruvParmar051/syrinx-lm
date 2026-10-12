#!/usr/bin/env bash
# Unattended Phase 4 -> Phase 5 pipeline (docs/decisions.md, 2026-10-12):
#   1. finish the 10m run (wait for it, or resume it if its process is gone)
#   2. 10m samples (temperature 0.8 / top-k 50, and greedy)
#   3. 50M LR sweep: 3e-4, 6e-4, 1.2e-3, 300 steps each on the full-run schedule
#   4. pick the LR (scripts/lr_sweep.py rule) -> runs/50m-a.yaml
#   5. full 50M run (1B tokens) with eval every 250 steps (100 batches), checkpoints
#      every 250, keep 3, milestones every 1000
#
# Every stage skips itself when its output exists and resumes from the newest checkpoint
# when one exists, so rerunning this script after a crash or sleep continues the pipeline.
# A stage that keeps failing stops the pipeline (a failed sweep LR only marks that LR
# unstable). Run from the repo root, under caffeinate so the Mac does not idle-sleep:
#
#   nohup caffeinate -is scripts/run_phase5.sh >> experiments/phase5-pipeline.txt 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."

SWEEP_LRS=(3e-4 6e-4 1.2e-3)
SWEEP_STEPS=300
RUN10=runs/10m-a
RUN10_FINAL=$RUN10/checkpoints/step_0009000.pt
SWEEP_DIR=runs/50m-sweep
RUN50=runs/50m-a
RUN50_CONFIG=runs/50m-a.yaml
RUN50_FINAL=$RUN50/checkpoints/step_0003815.pt
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

# 1. 10m run --------------------------------------------------------------------------
while [[ ! -f $RUN10_FINAL ]]; do
    if pgrep -f "scripts/train.py .*--out-dir $RUN10( |$)" > /dev/null; then
        sleep 60
    else
        log "10m process not running and $RUN10_FINAL missing: resuming"
        train_stage "$RUN10" "$RUN10_FINAL" experiments/10m-a-train-2026-10-12.txt \
            --config configs/syrinx-10m.yaml || die "10m run did not finish"
    fi
done
log "10m run complete: $RUN10_FINAL"

# 2. 10m samples ----------------------------------------------------------------------
SAMPLES=experiments/samples-10m-a-step9000.jsonl
GREEDY=experiments/samples-10m-a-step9000-greedy.jsonl
[[ -f $SAMPLES ]] || uv run python scripts/sample.py --checkpoint "$RUN10_FINAL" --out "$SAMPLES" \
    || die "10m sampling failed"
[[ -f $GREEDY ]] || uv run python scripts/sample.py --checkpoint "$RUN10_FINAL" --out "$GREEDY" \
    --temperature 0 || die "10m greedy sampling failed"
log "10m samples written: $SAMPLES $GREEDY"

# 3. 50M LR sweep ---------------------------------------------------------------------
uv run python scripts/lr_sweep.py make-configs --base configs/syrinx-50m.yaml \
    --lrs "${SWEEP_LRS[@]}" --out-dir "$SWEEP_DIR" || die "making sweep configs failed"
sweep_runs=()
for cfg in "$SWEEP_DIR"/lr-*.yaml; do
    run=${cfg%.yaml}
    final=$run/checkpoints/$(printf 'step_%07d.pt' "$SWEEP_STEPS")
    train_stage "$run" "$final" "experiments/50m-sweep-$(basename "$run").txt" \
        --config "$cfg" --stop-at-step "$SWEEP_STEPS" --log-interval 10 \
        --eval-interval 100 --eval-batches 100 --ckpt-interval 100 --keep-last 1 \
        || log "sweep run $run did not reach step $SWEEP_STEPS; it will count as unstable"
    [[ -f $run/config.yaml ]] && sweep_runs+=("$run")
done
[[ ${#sweep_runs[@]} -gt 0 ]] || die "no sweep run started"

# 4. pick the LR ----------------------------------------------------------------------
if [[ ! -f $RUN50_CONFIG ]]; then
    uv run python scripts/lr_sweep.py pick --runs "${sweep_runs[@]}" --step "$SWEEP_STEPS" \
        --base configs/syrinx-50m.yaml --config-out "$RUN50_CONFIG" \
        --summary-out experiments/50m-lr-sweep-summary.json \
        || die "no stable sweep run; not starting the 50M run (see experiments/50m-lr-sweep-summary.json)"
fi
log "50M config: $RUN50_CONFIG (lr $(grep -E '^  lr:' "$RUN50_CONFIG"))"

# 5. full 50M run ---------------------------------------------------------------------
train_stage "$RUN50" "$RUN50_FINAL" experiments/50m-a-train.txt \
    --config "$RUN50_CONFIG" --log-interval 10 --eval-interval 250 --eval-batches 100 \
    --ckpt-interval 250 --keep-last 3 --milestone-every 1000 \
    || die "50M run did not finish"
log "pipeline complete: $RUN50_FINAL"
