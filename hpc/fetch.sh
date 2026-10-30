#!/bin/bash
# Pull finished conversions back from the cluster.
#
#   bash hpc/fetch.sh              # skrubify/1/*.py and the json reports
#   bash hpc/fetch.sh --logs       # also SLURM logs
#   bash hpc/fetch.sh --dry-run
set -euo pipefail

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HPC_DIR/env.sh"

DO_LOGS=0
DRY_RUN=()
while [ $# -gt 0 ]; do
    case "$1" in
        --logs) DO_LOGS=1 ;;
        --dry-run) DRY_RUN=(--dry-run) ;;
        *) echo "unknown arg: $1 (want --logs|--dry-run)" >&2; exit 1 ;;
    esac
    shift
done

: "${SSH_HOST:?set SSH_HOST in hpc/config.env}"
: "${CLUSTER_PROJECT_DIR:?set CLUSTER_PROJECT_DIR in hpc/config.env}"

REMOTE="${SSH_HOST}:${CLUSTER_PROJECT_DIR}"
DEST="$REPO/titanic/mle_star_flash_run_1/skrubify"
mkdir -p "$DEST" "$RUN_DIR/reports"

echo "[fetch] skrubify <- ${REMOTE}/titanic/mle_star_flash_run_1/skrubify/"
rsync -avz "${DRY_RUN[@]}" \
    "${REMOTE}/titanic/mle_star_flash_run_1/skrubify/" \
    "$DEST/"

echo "[fetch] reports <- ${REMOTE}/hpc/_run/reports/"
rsync -avz "${DRY_RUN[@]}" \
    "${REMOTE}/hpc/_run/reports/" \
    "$RUN_DIR/reports/"

if [ "$DO_LOGS" = 1 ]; then
    mkdir -p "$RUN_DIR/logs"
    echo "[fetch] logs <- ${REMOTE}/hpc/_run/logs/"
    rsync -avz "${DRY_RUN[@]}" \
        "${REMOTE}/hpc/_run/logs/" \
        "$RUN_DIR/logs/"
fi

if [ ${#DRY_RUN[@]} -eq 0 ]; then
    echo "[fetch] done."
    echo "  pipelines: $DEST/"
    echo "  reports:   $RUN_DIR/reports/"
else
    echo "[fetch] dry-run only; nothing written."
fi
