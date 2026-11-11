#!/usr/bin/env bash
#SBATCH --job-name=mle-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
: "${HPC_REPO:?submit using hpc/submit.sh}" "${HPC_MANIFEST:?}" "${SLURM_ARRAY_TASK_ID:?}"
exec srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:?}" --cpu-bind=cores \
    bash "$HPC_REPO/hpc/exec.sh" python "$HPC_REPO/hpc/campaign.py" \
    run "$HPC_MANIFEST" "$SLURM_ARRAY_TASK_ID"
