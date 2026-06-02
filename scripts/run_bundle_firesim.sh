#!/usr/bin/env bash
# Drive a candidate bundle through FireSim under a single FIRESIM_QUEUE
# session. Thin wrapper around scripts/run_xpurt_bundle.py that
# guarantees the build/SDK env is sourced and the venv is active before
# any work runs. This is the entry point the /realize-and-run skill
# invokes, and the entry point for repro from a fresh shell.
#
# Usage:
#   bash scripts/run_bundle_firesim.sh \
#       --batch /scratch2/agustin/XPU-RT/artifacts/iterate/firesim_batch.json \
#       --out-dir artifacts/bundle \
#       [--include baseline,A2]      # restrict to specific candidate ids
#       [--runner firesim]           # firesim (default) | spike
#
# All other args pass through to run_xpurt_bundle.py.
set -o pipefail

cd "$(dirname "$0")/.."

# Defang conda's strict-mode incompatibilities (same pattern the
# in-repo capture scripts use): conda init scripts reference
# unset variables, so `set -u` trips them. The verify below
# checks west/spike actually landed on PATH; we don't trust the
# source rc.
set +u
source scripts/setup_benchmark_env.sh >/dev/null 2>&1 || true
set -u
command -v west >/dev/null 2>&1 || {
    echo "ERROR: west not on PATH after sourcing scripts/setup_benchmark_env.sh" >&2
    exit 1
}

export FIRESIM_QUEUE=1

if [[ -x .venv/bin/python ]]; then
    PY=.venv/bin/python
elif command -v uv >/dev/null 2>&1; then
    PY="uv run python3"
else
    PY=python3
fi

# shellcheck disable=SC2086
exec $PY scripts/run_xpurt_bundle.py "$@"
