#!/usr/bin/env bash
# No deletion: preserve remote data, images, and benchmark artifacts.
set -euo pipefail
source "$(dirname "$0")/env.sh"
: "${CLUSTER_PROJECT_DIR:?set CLUSTER_PROJECT_DIR in hpc/config.env}"
args=()
image=0
for option in "$@"; do
    case "$option" in
        --dry-run) args+=(--dry-run) ;;
        --image) image=1 ;;
        *) echo "Usage: bash hpc/push.sh [--image] [--dry-run]" >&2; exit 2 ;;
    esac
done
# --mkpath requires rsync >=3.2.3 on both ends. Config, data, and ADC are excluded.
rsync -avz --mkpath "${args[@]}" \
    --exclude='.git/' --exclude='.venv/' --exclude='node_modules/' \
    --exclude='__pycache__/' --exclude='.pytest_cache/' --exclude='.stratum/' \
    --exclude='.aws/' --exclude='.env*' --exclude='*.sif' --exclude='*.sif.sha256' \
    --exclude='input/' --exclude='raw/' --exclude='runtime_stats_*' \
    --exclude='/hpc/_run/' --exclude='/hpc/config.env' --exclude='/hpc/secrets.env' \
    --exclude='/website/.cache/' --exclude='/website/frontend/.next/' \
    "$HPC_REPO/" "${SSH_HOST}:${CLUSTER_PROJECT_DIR}/"
if [[ $image == 1 ]]; then
    [[ -f "$SIF" ]] || { echo "Missing image: $SIF" >&2; exit 1; }
    rsync -av --mkpath "${args[@]}" "$SIF" "${SSH_HOST}:${CLUSTER_PROJECT_DIR}/hpc/benchmarks.sif"
fi
echo "Code pushed without deleting remote files. Configure hpc/config.env on the cluster."
echo "Data and credentials are staged separately; neither was transferred."
