#!/usr/bin/env bash
# Run a command in the pinned image. Used both interactively and under srun.
set -euo pipefail
source "$(dirname "$0")/env.sh"
hpc_container
[[ -f "$SIF" ]] || { echo "Missing $SIF; run bash hpc/build.sh first." >&2; exit 1; }
binds=(--bind "$HPC_REPO" --bind "$SIF:$SIF:ro")
for mount in "${HPC_EXTRA_BINDS[@]}"; do binds+=(--bind "$mount"); done
threads="${SLURM_CPUS_PER_TASK:-$WORKER_CPUS}"
container_env=("PYTHONPATH=$HPC_REPO:$HPC_REPO/tools" "PYTHONUNBUFFERED=1"
    "HPC_IMAGE=$SIF"
    "OMP_NUM_THREADS=$threads" "OPENBLAS_NUM_THREADS=$threads"
    "MKL_NUM_THREADS=$threads" "BLIS_NUM_THREADS=$threads"
    "NUMEXPR_NUM_THREADS=$threads" "VECLIB_MAXIMUM_THREADS=$threads"
    "SKRUB_RUST_THREADS=$threads" "LOKY_MAX_CPU_COUNT=$threads")
for name in SLURM_JOB_ID SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURM_CPUS_PER_TASK \
    SLURM_MEM_PER_NODE SLURM_JOB_PARTITION SLURM_JOB_NODELIST; do
    [[ -z "${!name:-}" ]] || container_env+=("$name=${!name}")
done
if [[ -n "${SLURM_TMPDIR:-}" && -d "$SLURM_TMPDIR" ]]; then
    binds+=(--bind "$SLURM_TMPDIR")
    container_env+=("TMPDIR=$SLURM_TMPDIR")
fi
if [[ -n "$GCS_CREDENTIALS" ]]; then
    [[ -f "$GCS_CREDENTIALS" ]] || { echo "Missing GCS_CREDENTIALS file." >&2; exit 1; }
    binds+=(--bind "$GCS_CREDENTIALS:/run/benchmark-reader.json:ro")
    container_env+=("GOOGLE_APPLICATION_CREDENTIALS=/run/benchmark-reader.json")
fi
exec "$HPC_CONTAINER" exec --cleanenv --no-home "${binds[@]}" --pwd "$HPC_REPO" \
    "$SIF" env "${container_env[@]}" "$@"
