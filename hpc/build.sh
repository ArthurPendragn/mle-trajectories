#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
hpc_container
cd "$HPC_REPO"
[[ ! -e "$SIF" ]] || { echo "Image already exists: $SIF. Set SIF to a new filename to rebuild." >&2; exit 1; }
# Extra build flags replace the default --fakeroot (e.g. bash hpc/build.sh --remote).
flags=(--fakeroot)
[[ $# -eq 0 ]] || flags=("$@")
"$HPC_CONTAINER" build "${flags[@]}" "$SIF" hpc/benchmarks.def
sha256sum "$SIF" > "$SIF.sha256"
echo "Built $SIF. Next: bash hpc/exec.sh python hpc/campaign.py smoke"
