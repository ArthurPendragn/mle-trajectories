#!/usr/bin/env bash
# bash hpc/submit.sh hpc/pilot.toml [--dry-run]
set -euo pipefail
source "$(dirname "$0")/env.sh"
[[ $# -ge 1 && $# -le 2 ]] || { echo "Usage: bash hpc/submit.sh CAMPAIGN.toml [--dry-run]" >&2; exit 2; }
dry=0
if [[ $# -eq 2 ]]; then
    [[ $2 == --dry-run ]] || { echo "Unknown option: $2" >&2; exit 2; }
    dry=1
fi
cd "$HPC_REPO"
for value in "$WORKER_CPUS" "$CONCURRENCY"; do
    [[ $value =~ ^[1-9][0-9]*$ ]] || { echo "CPU count and concurrency must be positive integers." >&2; exit 2; }
done
[[ $EXCLUSIVE == 0 || $EXCLUSIVE == 1 ]] || { echo "EXCLUSIVE must be 0 or 1." >&2; exit 2; }
if [[ $dry == 0 ]]; then
    command -v sbatch >/dev/null || { echo "sbatch unavailable; submit from a cluster frontend." >&2; exit 1; }
    [[ -f "$SIF" ]] || { echo "Missing image: $SIF" >&2; exit 1; }
fi
mkdir -p "$HPC_REPO/hpc/_run"
campaign_dir=$(mktemp -d "$HPC_REPO/hpc/_run/$(date -u +%Y%m%dT%H%M%S)-XXXXXX")
mkdir -p "$campaign_dir/logs"
export HPC_MANIFEST="$campaign_dir/manifest.json"
# Dry-run planning needs only Python >=3.11, no container/data/stratum imports.
if [[ $dry == 1 ]]; then
    planner="${HPC_PYTHON:-$HPC_REPO/.venv/bin/python}"
    [[ -x "$planner" ]] || planner=python3
    "$planner" hpc/campaign.py plan "$1" --out "$HPC_MANIFEST" --image "$SIF" --allow-missing-image
else
    bash hpc/exec.sh python hpc/campaign.py plan "$1" --out "$HPC_MANIFEST" --image "$SIF"
fi
count=$(cat "$campaign_dir/task_count")
args=(--parsable --chdir="$HPC_REPO" --nodes=1 --ntasks=1
    --array="0-$((count - 1))%$CONCURRENCY" --cpus-per-task="$WORKER_CPUS"
    --mem="$WORKER_MEM" --time="$TIME_WORKER"
    --output="$campaign_dir/logs/%x-%A_%a.out" --error="$campaign_dir/logs/%x-%A_%a.err"
    --export=ALL)
[[ -z "$ACCOUNT" ]] || args+=(--account="$ACCOUNT")
[[ -z "$PARTITION" ]] || args+=(--partition="$PARTITION")
[[ $EXCLUSIVE == 0 ]] || args+=(--exclusive)
printf 'Command: '; printf '%q ' sbatch "${args[@]}" "$HPC_REPO/hpc/run_cell.sbatch"; printf '\n'
if [[ $dry == 1 ]]; then
    echo "Dry run only. Re-plan on the cluster when ready; manifest: $HPC_MANIFEST"
else
    sbatch "${args[@]}" "$HPC_REPO/hpc/run_cell.sbatch" | tee "$campaign_dir/job_id"
    echo "Manifest: $HPC_MANIFEST"
fi
