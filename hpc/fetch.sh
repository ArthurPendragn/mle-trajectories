#!/usr/bin/env bash
# Fetch raw campaigns, preserving local results. Publication is a separate step.
set -euo pipefail
source "$(dirname "$0")/env.sh"
: "${CLUSTER_PROJECT_DIR:?set CLUSTER_PROJECT_DIR in hpc/config.env}"
args=()
[[ $# -eq 0 ]] || { [[ $# -eq 1 && $1 == --dry-run ]] && args+=(--dry-run); } || {
    echo "Usage: bash hpc/fetch.sh [--dry-run]" >&2; exit 2;
}
mkdir -p "$HPC_DIR/_run"
rsync -avz --update "${args[@]}" \
    "${SSH_HOST}:${CLUSTER_PROJECT_DIR}/hpc/_run/" "$HPC_DIR/_run/"
echo "Fetched new/updated raw campaign files, preserving newer local files. Publish completed cells separately."
