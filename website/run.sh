#!/usr/bin/env bash
# Start the explorer: the Python API on a private unix socket, and the Next.js
# frontend on 127.0.0.1:${PORT:-3000} (reach it through an SSH tunnel).
#
#   website/run.sh          # development: both reload on code changes
#   website/run.sh prod     # production build, then serve it
set -euo pipefail
cd "$(dirname "$0")/.."

mode="${1:-dev}"
front=website/frontend
export NEXT_TELEMETRY_DISABLED=1

if [[ ! -f $front/.env.local ]]; then
    echo "no login configured yet: (cd $front && npm run set-password)" >&2
    exit 1
fi
[[ -d $front/node_modules ]] || (cd $front && npm ci)

api_args=()
[[ $mode == dev ]] && api_args+=(--reload)
uv run --group website python -m website.backend "${api_args[@]}" &
api=$!
trap 'kill "$api" 2>/dev/null || true' EXIT INT TERM

cd $front
case $mode in
    dev)  npm run dev ;;
    prod) npm run build && npm run start ;;
    *)    echo "usage: $0 [dev|prod]" >&2; exit 2 ;;
esac
