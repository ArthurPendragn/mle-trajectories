#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
hpc_container
cd "$HPC_REPO"
[[ ! -e "$SIF" ]] || { echo "Image already exists: $SIF. Set SIF to a new filename to rebuild." >&2; exit 1; }
# The sandbox holds the rootfs, the Rust toolchain, and uv's download cache at
# once. WSL's /tmp is an 8G tmpfs and fills up during `uv sync`.
if [[ -z "${SINGULARITY_TMPDIR:-}" ]]; then
    tmp_avail_kb="$(df -k --output=avail /tmp | awk 'NR==2 { print $1 }')"
    if [[ "${tmp_avail_kb}" -lt 20971520 ]]; then
        export TMPDIR="${HPC_DIR}/_buildtmp"
        export SINGULARITY_TMPDIR="$TMPDIR"
        mkdir -p "$TMPDIR"
        echo "Using $TMPDIR for the build sandbox (/tmp has ${tmp_avail_kb}K free)."
    fi
fi
# Extra build flags replace the default --fakeroot (e.g. bash hpc/build.sh --remote).
flags=(--fakeroot)
[[ $# -eq 0 ]] || flags=("$@")
"$HPC_CONTAINER" build "${flags[@]}" "$SIF" hpc/benchmarks.def
sha256sum "$SIF" > "$SIF.sha256"
echo "Built $SIF. Next: bash hpc/exec.sh python hpc/campaign.py smoke"
