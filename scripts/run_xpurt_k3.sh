#!/usr/bin/env bash
# SpaceMiT K3 entry point: the K1 runner, generalised. See the "Board
# selection" note at the top of run_xpurt_k1.sh for what MODELBLASTER_BOARD=k3 changes.
#
# Defaults here: --backends rvv_k3,rvv_k3_a100 (parallel to core kinds x100,a100;
# the A100 build is zvl1024b and must only run on CPUs 8-15), and the walker
# gets the pre-timing warm-up pass (generate_xpurt_main --warmup).
#
# SCHEDULING (XPU-RT side, not done by this script): solve K3 schedules with
#   XPURT_CODEGEN_CONTRACT=<ModelBlaster>/cores/codegen_contract.json \
#   XPURT_UNIFORM_PACKED_WIDTH=1 <your XPU-RT solve command>
# XPU-RT's default contract path (XPU-RT/ModelBlaster/cores/...) is an empty
# submodule placeholder, so without XPURT_CODEGEN_CONTRACT the packed-weight
# width rules are not applied. The contract is copied into this repo's cores/.
set -euo pipefail
export MODELBLASTER_BOARD=k3
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${XPURT_CODEGEN_CONTRACT:-}" || "${XPURT_UNIFORM_PACKED_WIDTH:-}" != "1" ]]; then
    echo "hint: solve the schedule with XPURT_CODEGEN_CONTRACT=${_here}/../cores/codegen_contract.json XPURT_UNIFORM_PACKED_WIDTH=1" >&2
fi
exec "${_here}/run_xpurt_k1.sh" "$@"
