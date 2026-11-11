#!/usr/bin/env bash
# Source from helpers. This file intentionally never loads MLE-STAR secrets.
HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HPC_REPO="$(cd "$HPC_DIR/.." && pwd)"
[[ ! -f "$HPC_DIR/config.env" ]] || source "$HPC_DIR/config.env"
export SSH_HOST="${SSH_HOST:-hpc}"
export CLUSTER_PROJECT_DIR="${CLUSTER_PROJECT_DIR:-}"
export SIF="${SIF:-$HPC_DIR/benchmarks.sif}"
export ACCOUNT="${ACCOUNT:-}"
export PARTITION="${PARTITION:-standard}"
export WORKER_CPUS="${WORKER_CPUS:-16}"
export WORKER_MEM="${WORKER_MEM:-60G}"
export TIME_WORKER="${TIME_WORKER:-02:00:00}"
export CONCURRENCY="${CONCURRENCY:-1}"
export EXCLUSIVE="${EXCLUSIVE:-1}"
export CONTAINER_MODULE="${CONTAINER_MODULE:-singularity}"
export GCS_CREDENTIALS="${GCS_CREDENTIALS:-}"
declare -p HPC_EXTRA_BINDS >/dev/null 2>&1 || HPC_EXTRA_BINDS=()

hpc_container() {
    if [[ -n "${HPC_CONTAINER:-}" ]]; then
        command -v "$HPC_CONTAINER" >/dev/null || {
            echo "HPC_CONTAINER executable not found: $HPC_CONTAINER" >&2; return 1;
        }
    else
        if ! command -v singularity >/dev/null && ! command -v apptainer >/dev/null; then
            if type module >/dev/null 2>&1; then module load "$CONTAINER_MODULE"; fi
        fi
        if command -v singularity >/dev/null; then HPC_CONTAINER=singularity;
        elif command -v apptainer >/dev/null; then HPC_CONTAINER=apptainer;
        else echo "Load Singularity/Apptainer, or set HPC_CONTAINER to its executable." >&2; return 1; fi
    fi
    export HPC_CONTAINER
}
