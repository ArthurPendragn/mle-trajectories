#!/bin/bash
#SBATCH --job-name=skrubify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=hpc/_run/logs/%x-%A_%a.out
#SBATCH --error=hpc/_run/logs/%x-%A_%a.err
# One array element converts one family of pipelines (one line of hpc/batches.txt).
# Submit from the repo root:  bash hpc/submit_skrubify.sh
set -euo pipefail

module load singularity 2>/dev/null || true
if ! command -v singularity >/dev/null 2>&1; then
    for _cand in \
        /cluster/singularity/4.4.1/bin \
        /cluster/singularity/4.4.0/bin \
        /cluster/singularity/4.3.7/bin
    do
        if [ -x "$_cand/singularity" ]; then
            export PATH="$_cand:${PATH:-/usr/bin:/bin}"
            break
        fi
    done
fi
command -v singularity >/dev/null 2>&1 || {
    echo "ERROR: singularity not on PATH on $(hostname)" >&2
    exit 127
}

REPO="${REPO:-${SLURM_SUBMIT_DIR:?}}"
SIF="${SIF:-$REPO/hpc/mle_trajectories.sif}"
BATCHES="${BATCHES:-$REPO/hpc/batches.txt}"
PIPELINE_DIR="${PIPELINE_DIR:-titanic/mle_star_flash_run_1/pipelines/1}"
RUN_IN="${RUN_IN:-titanic}"
PROVIDER="${PROVIDER:-openai}"
MODEL="${MODEL:-gpt-5.6-sol}"
ENGINE="${ENGINE:-stratum}"
COMPARE_SOURCE="${COMPARE_SOURCE:-1}"
cd "$REPO"
[ -f "$SIF" ] || { echo "ERROR: missing image $SIF" >&2; exit 1; }
[ -f "$RUN_IN/input/train.csv" ] || {
    echo "ERROR: no $RUN_IN/input/train.csv — push.sh did not send the data" >&2
    exit 1
}

# hpc/env.sh maps GOOGLE_API_KEY onto GEMINI_API_KEY. Repeat it here so a job
# started with --export=ALL still sees the name skrubify checks.
if [ -z "${GEMINI_API_KEY:-}" ] && [ -n "${GOOGLE_API_KEY:-}" ]; then
    export GEMINI_API_KEY="$GOOGLE_API_KEY"
fi

# Skip blank lines and comments, matching submit_skrubify.sh.
mapfile -t LINES < <(grep -vE '^[[:space:]]*(#|$)' "$BATCHES")
LINE="${LINES[$SLURM_ARRAY_TASK_ID]:-}"
if [ -z "$LINE" ]; then
    echo "[skrubify] no batch at index ${SLURM_ARRAY_TASK_ID} in $BATCHES" >&2
    exit 1
fi
# shellcheck disable=SC2086
set -- $LINE
NAME="$1"
shift
SOURCES=()
for script in "$@"; do
    SOURCES+=("$PIPELINE_DIR/$script")
    [ -f "$PIPELINE_DIR/$script" ] || {
        echo "[skrubify] missing $PIPELINE_DIR/$script" >&2
        exit 1
    }
done

DONE="$REPO/hpc/_run/done/${NAME}"
if [ -f "$DONE" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[skrubify] SKIP ${NAME}: $DONE exists"
    exit 0
fi

mkdir -p "$REPO/hpc/_run/reports" "$REPO/hpc/_run/done"
echo "[skrubify] idx=${SLURM_ARRAY_TASK_ID} batch=${NAME} node=$(hostname)"
echo "[skrubify] ${PROVIDER}/${MODEL} engine=${ENGINE} run-in=${RUN_IN}"
printf '  %s\n' "${SOURCES[@]}"

export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_PYTHONPATH="$REPO/tools"
# Keys the container process actually sees. Empty values are fine; skrubify
# reports a missing key itself.
export SINGULARITYENV_OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export SINGULARITYENV_GEMINI_API_KEY="${GEMINI_API_KEY:-}"
export SINGULARITYENV_ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"

ARGS=(
    python -m skrubify
    "${SOURCES[@]}"
    --provider "$PROVIDER"
    --model "$MODEL"
    --engine "$ENGINE"
    --run-in "$RUN_IN"
    --json-report "$REPO/hpc/_run/reports/${NAME}.json"
)
if [ "$COMPARE_SOURCE" = "1" ]; then
    ARGS+=(--compare-source)
fi

singularity exec --bind "$REPO" --pwd "$REPO" "$SIF" "${ARGS[@]}"

# Compare-source prints a score gap and still exits 0. The sentinel means the
# conversion finished, not that the scores matched.
date -Is > "$DONE"
echo "[skrubify] done ${NAME}"
