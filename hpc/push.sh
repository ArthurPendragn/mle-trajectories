#!/bin/bash
# Push this checkout to CLUSTER_PROJECT_DIR. Run on the laptop, from anywhere.
#
#   bash hpc/push.sh              # source tree + titanic/input if present
#   bash hpc/push.sh --image      # also scp the .sif
#   bash hpc/push.sh --mkdir      # create the remote directory first
set -euo pipefail

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HPC_DIR/env.sh"

DO_IMAGE=0
DO_MKDIR=0
while [ $# -gt 0 ]; do
    case "$1" in
        --image) DO_IMAGE=1 ;;
        --mkdir) DO_MKDIR=1 ;;
        *) echo "unknown arg: $1 (want --image|--mkdir)" >&2; exit 1 ;;
    esac
    shift
done

: "${SSH_HOST:?set SSH_HOST in hpc/config.env}"
: "${CLUSTER_PROJECT_DIR:?set CLUSTER_PROJECT_DIR in hpc/config.env}"

if [ "$DO_MKDIR" = 1 ]; then
    echo "[push] mkdir -p ${CLUSTER_PROJECT_DIR}/hpc on ${SSH_HOST}"
    ssh "$SSH_HOST" "mkdir -p '${CLUSTER_PROJECT_DIR}/hpc'"
fi

echo "[push] rsync -> ${SSH_HOST}:${CLUSTER_PROJECT_DIR}/"
# P filters keep remote-only files when --delete would remove them: the image,
# SLURM logs, and conversions already written on the cluster.
rsync -avz --delete \
    --exclude='.git/' --exclude='.venv/' \
    --exclude='__pycache__/' --exclude='.pytest_cache/' \
    --exclude='**/input/**' \
    --exclude='*.sif' --filter='P *.sif' --filter='P **/*.sif' \
    --exclude='/hpc/_run/' --filter='P hpc/_run/***' \
    --filter='P titanic/mle_star_flash_run_1/skrubify/***' \
    --exclude='/hpc/config.env' \
    --exclude='/hpc/secrets.env' \
    "$REPO/" "${SSH_HOST}:${CLUSTER_PROJECT_DIR}/"

if [ -d "$REPO/titanic/input" ]; then
    echo "[push] titanic/input"
    rsync -avz "$REPO/titanic/input/" \
        "${SSH_HOST}:${CLUSTER_PROJECT_DIR}/titanic/input/"
else
    echo "[push] WARNING: no titanic/input locally; the job needs train.csv on the cluster." >&2
fi

if [ -f "$HPC_DIR/config.env" ]; then
    echo "[push] scp hpc/config.env"
    scp "$HPC_DIR/config.env" "${SSH_HOST}:${CLUSTER_PROJECT_DIR}/hpc/config.env"
fi
if [ -f "$HPC_DIR/secrets.env" ]; then
    echo "[push] scp hpc/secrets.env"
    scp "$HPC_DIR/secrets.env" "${SSH_HOST}:${CLUSTER_PROJECT_DIR}/hpc/secrets.env"
    ssh "$SSH_HOST" "chmod 600 '${CLUSTER_PROJECT_DIR}/hpc/secrets.env'"
fi

if [ "$DO_IMAGE" = 1 ]; then
    [ -f "$SIF" ] || { echo "ERROR: missing $SIF — build it first" >&2; exit 1; }
    echo "[push] scp image"
    scp "$SIF" "${SSH_HOST}:${CLUSTER_PROJECT_DIR}/hpc/$(basename "$SIF")"
fi

echo "[push] done. On the cluster:  cd ${CLUSTER_PROJECT_DIR} && bash hpc/submit_skrubify.sh"
