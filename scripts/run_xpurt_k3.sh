#!/usr/bin/env bash
# SpaceMiT K3 entry point: the K1 runner, generalised. See the "Board
# selection" note at the top of run_xpurt_k1.sh for what MODELBLASTER_BOARD=k3 changes.
set -euo pipefail
export MODELBLASTER_BOARD=k3
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_xpurt_k1.sh" "$@"
