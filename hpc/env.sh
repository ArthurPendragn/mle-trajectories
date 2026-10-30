#!/bin/bash
# Shared configuration. Sourced by submit_skrubify.sh / push.sh / fetch.sh.
# Per-user paths live in hpc/config.env. Keys live in hpc/secrets.env.

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HPC_DIR
export REPO="$(cd "$HPC_DIR/.." && pwd)"
export PROJECT="$REPO"

# Caller exports win over config.env.
_ENV_PROVIDER="${PROVIDER-}"
_ENV_MODEL="${MODEL-}"
_ENV_ENGINE="${ENGINE-}"
_ENV_RUN_IN="${RUN_IN-}"
_ENV_PIPELINE_DIR="${PIPELINE_DIR-}"
_ENV_COMPARE_SOURCE="${COMPARE_SOURCE-}"
_ENV_CONCURRENCY="${CONCURRENCY-}"
_ENV_SIF="${SIF-}"

if [ -f "$HPC_DIR/config.env" ]; then
    # shellcheck disable=SC1091
    source "$HPC_DIR/config.env"
fi

[ -n "${_ENV_PROVIDER}" ] && PROVIDER="$_ENV_PROVIDER"
[ -n "${_ENV_MODEL}" ] && MODEL="$_ENV_MODEL"
[ -n "${_ENV_ENGINE}" ] && ENGINE="$_ENV_ENGINE"
[ -n "${_ENV_RUN_IN}" ] && RUN_IN="$_ENV_RUN_IN"
[ -n "${_ENV_PIPELINE_DIR}" ] && PIPELINE_DIR="$_ENV_PIPELINE_DIR"
[ -n "${_ENV_COMPARE_SOURCE}" ] && COMPARE_SOURCE="$_ENV_COMPARE_SOURCE"
[ -n "${_ENV_CONCURRENCY}" ] && CONCURRENCY="$_ENV_CONCURRENCY"
[ -n "${_ENV_SIF}" ] && SIF="$_ENV_SIF"

export RUN_DIR="$HPC_DIR/_run"
export BATCHES="${BATCHES:-$HPC_DIR/batches.txt}"
export SIF="${SIF:-$HPC_DIR/mle_trajectories.sif}"

export PROVIDER="${PROVIDER:-openai}"
export MODEL="${MODEL:-gpt-5.6-sol}"
export ENGINE="${ENGINE:-stratum}"
export RUN_IN="${RUN_IN:-titanic}"
export PIPELINE_DIR="${PIPELINE_DIR:-titanic/mle_star_flash_run_1/pipelines/1}"
export COMPARE_SOURCE="${COMPARE_SOURCE:-1}"

export CONCURRENCY="${CONCURRENCY:-1}"
export ACCOUNT="${ACCOUNT:-}"
export PARTITION="${PARTITION:-standard}"
export TIME_WORKER="${TIME_WORKER:-24:00:00}"
export WORKER_CPUS="${WORKER_CPUS:-8}"
export WORKER_MEM="${WORKER_MEM:-32G}"
export EXCLUDE="${EXCLUDE:-gpu[028-042]}"

export SSH_HOST="${SSH_HOST:-hpc}"
export CLUSTER_PROJECT_DIR="${CLUSTER_PROJECT_DIR:-}"

if [ -f "$HPC_DIR/secrets.env" ]; then
    # shellcheck disable=SC1091
    source "$HPC_DIR/secrets.env"
fi
# sbatch --export=ALL only forwards exported variables.
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export GEMINI_API_KEY="${GEMINI_API_KEY:-}"
export GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"

# mle-star-stratum stores the Gemini key as GOOGLE_API_KEY. skrubify asks for
# GEMINI_API_KEY when --provider gemini.
if [ -z "${GEMINI_API_KEY:-}" ] && [ -n "${GOOGLE_API_KEY:-}" ]; then
    export GEMINI_API_KEY="$GOOGLE_API_KEY"
fi
