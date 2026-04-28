#!/usr/bin/env bash
# Sweep per-dispatch profile data across (HW backend) x (pool size).
#
# For each combination, this rebuilds the multi-model harness with
# AGENTS_POOL_THREADS pinned to the sweep's pool size, runs spike with
# matching `-p<N>`, and lets spike_runner emit one IREE-shape
# results.csv per model under PROFILE_OUT_ROOT.
#
# The output tree mirrors the IREE scheduler's expected layout:
#   <out_root>/<backend>/<cpu>/<model>/<model>.<quant>/<spec>/topo_<cores>/results.csv
#
# Env knobs:
#   MODELS=mlp_generic,mlp_control       comma list of constituent models
#   BACKENDS=scalar[,rvv]                HW backends to sweep
#   POOL_SIZES=1,2,3,4                   pool worker counts to sweep
#   QUANT=fp32                           shared quant (per multi_demo run.sh)
#   PROFILE_OUT_ROOT=$REPO/gen/profile   where the IREE-shape tree lands
#   PROFILE_SOURCE=spike                 provenance tag (firesim / rtlsim later)
#   PROFILE_CPU=<src>                    CPU label (defaults to PROFILE_SOURCE)
#   PROFILE_CLOCK_MHZ=1000               cycles->ns conversion (1 GHz default)
#
# Prereqs (run once per shell):
#   source tools/miniforge3/etc/profile.d/conda.sh && conda activate zephyr
#   source scripts/set_envvars_sdk.sh
#
# Run from repo root:
#   bash agents/scripts/sweep_profile.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODELS="${MODELS:-mlp_generic,mlp_control}"
BACKENDS="${BACKENDS:-scalar}"
POOL_SIZES="${POOL_SIZES:-1,2,4}"
QUANT="${QUANT:-fp32}"
PROFILE_OUT_ROOT="${PROFILE_OUT_ROOT:-${REPO_ROOT}/gen/profile}"

IFS=',' read -ra BACKEND_LIST <<< "${BACKENDS}"
IFS=',' read -ra POOL_LIST   <<< "${POOL_SIZES}"

mkdir -p "${PROFILE_OUT_ROOT}"
echo "[sweep] models=${MODELS}"
echo "[sweep] backends=${BACKENDS}"
echo "[sweep] pool_sizes=${POOL_SIZES}"
echo "[sweep] out_root=${PROFILE_OUT_ROOT}"

for backend in "${BACKEND_LIST[@]}"; do
    for pool in "${POOL_LIST[@]}"; do
        cores=""
        for i in $(seq 0 $((pool - 1))); do
            cores="${cores:+${cores},}${i}"
        done
        echo
        echo "===> sweep: backend=${backend} pool=${pool} cores=${cores}"
        # Spike is always launched with -p4 (the harness's
        # CONFIG_MP_MAX_NUM_CPUS=4 — Zephyr's SMP boot stalls if fewer
        # harts respond). Only AGENTS_POOL_THREADS varies, which
        # controls how many of those harts pthreadpool actually spawns
        # workers on. Idle harts cost nothing in the sim.
        TARGET="${backend}" \
        QUANT="${QUANT}" \
        SPIKE_HARTS=4 \
        AGENTS_POOL_THREADS="${pool}" \
        FORCE_REGEN=1 \
        MODELS="${MODELS}" \
        BACKEND="${KERNEL_BACKEND:-reference}" \
        OPTIMIZE="${OPTIMIZE:-0}" \
        BEAM="${BEAM:-2}" \
        EXPANSIONS="${EXPANSIONS:-3}" \
        ITERATIONS="${ITERATIONS:-2}" \
        PROFILE_OUT_ROOT="${PROFILE_OUT_ROOT}" \
        PROFILE_SOURCE="${PROFILE_SOURCE:-spike}" \
        PROFILE_CPU="${PROFILE_CPU:-${PROFILE_SOURCE:-spike}}" \
        PROFILE_CORES="${cores}" \
        PROFILE_CLOCK_MHZ="${PROFILE_CLOCK_MHZ:-1000.0}" \
            bash "${REPO_ROOT}/agents/examples/multi_demo/run.sh"
    done
done

echo
echo "[sweep] complete. results CSVs:"
find "${PROFILE_OUT_ROOT}" -name 'results.csv' | sort
