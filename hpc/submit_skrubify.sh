#!/bin/bash
# Submit one SLURM array task per line of hpc/batches.txt. Each task converts
# one family of pipelines (train0, train1, ...) in a single skrubify process.
#
# On the cluster, from the repo root:
#   bash hpc/submit_skrubify.sh
#   CONCURRENCY=2 FORCE=1 bash hpc/submit_skrubify.sh
set -euo pipefail

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HPC_DIR/env.sh"

[ -f "$SIF" ] || {
    echo "ERROR: image not found: $SIF" >&2
    echo "  build from the repo root:  singularity build --fakeroot hpc/mle_trajectories.sif hpc/mle_trajectories.def" >&2
    exit 1
}
[ -f "$BATCHES" ] || { echo "ERROR: missing $BATCHES" >&2; exit 1; }
[ -f "$REPO/$RUN_IN/input/train.csv" ] || {
    echo "ERROR: no $REPO/$RUN_IN/input/train.csv" >&2
    echo "  push.sh sends titanic/input from the laptop when that file exists there." >&2
    exit 1
}
case "$PROVIDER" in
    openai)    : "${OPENAI_API_KEY:?ERROR: set OPENAI_API_KEY in hpc/secrets.env}" ;;
    gemini)    : "${GEMINI_API_KEY:?ERROR: set GEMINI_API_KEY or GOOGLE_API_KEY in hpc/secrets.env}" ;;
    anthropic) : "${ANTHROPIC_API_KEY:?ERROR: set ANTHROPIC_API_KEY in hpc/secrets.env}" ;;
esac
command -v sbatch >/dev/null || {
    echo "ERROR: sbatch not on PATH (not a SLURM login node?)" >&2
    exit 1
}
module load singularity 2>/dev/null || true

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/done" "$RUN_DIR/reports"

NTASKS=$(grep -cEv '^[[:space:]]*(#|$)' "$BATCHES")
[ "$NTASKS" -gt 0 ] || { echo "ERROR: no batches in $BATCHES" >&2; exit 1; }
ARRAY_MAX=$((NTASKS - 1))

echo "Skrubify: ${NTASKS} batches, ${CONCURRENCY} at a time"
echo "  PROVIDER=${PROVIDER}  MODEL=${MODEL}  ENGINE=${ENGINE}"
echo "  RUN_IN=${RUN_IN}  PIPELINE_DIR=${PIPELINE_DIR}  COMPARE_SOURCE=${COMPARE_SOURCE}"
export FORCE="${FORCE:-0}"
echo "  FORCE=${FORCE}"

export REPO SIF BATCHES RUN_DIR PIPELINE_DIR RUN_IN PROVIDER MODEL ENGINE \
    COMPARE_SOURCE FORCE OPENAI_API_KEY GEMINI_API_KEY GOOGLE_API_KEY \
    ANTHROPIC_API_KEY

EXTRA=()
[ -n "$ACCOUNT" ]   && EXTRA+=(--account="$ACCOUNT")
[ -n "$PARTITION" ] && EXTRA+=(--partition="$PARTITION")
[ -n "$EXCLUDE" ]   && EXTRA+=(--exclude="$EXCLUDE")

JOB=$(sbatch --parsable "${EXTRA[@]}" \
    --array="0-${ARRAY_MAX}%${CONCURRENCY}" \
    --time="$TIME_WORKER" \
    --cpus-per-task="$WORKER_CPUS" \
    --mem="$WORKER_MEM" \
    "$HPC_DIR/skrubify_batch.sbatch")
echo "Array job: $JOB"
echo "Monitor:   squeue --me    |    tail -f $RUN_DIR/logs/skrubify-${JOB}_*.out"
echo "Outputs:   $REPO/$PIPELINE_DIR/../../skrubify/1/"
echo "Reports:   $RUN_DIR/reports/<batch>.json"
echo "A finished batch is skipped on resubmit (hpc/_run/done/<batch>). FORCE=1 reruns it."
