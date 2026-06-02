#!/usr/bin/env bash
# Source me to make the benchmark harness work from a fresh shell:
#
#   source scripts/setup_benchmark_env.sh
#
# Sets up everything required by:
#   - examples/<model>/run.sh        (needs west, spike, ZEPHYR_BASE, ZEPHYR_SDK_INSTALL_DIR)
#   - pipeline.bedrock_client        (needs AWS_BEARER_TOKEN_BEDROCK)
#   - benchmarks/arms/arm_*          (uses both)
#
# Designed to be idempotent: re-sourcing in an already-activated shell is
# harmless. Paths are pinned to the host this repo lives on; if they move,
# edit the constants below.

# ---- pinned paths (host-specific) -----------------------------------------

_MB_CHIPYARD_CONDA="/scratch2/agustin/chipyard/.conda-env"
_MB_ZEPHYR_BASE="/scratch2/agustin/zephyr-chipyard-sw/zephyr_ws/zephyr"
_MB_ZEPHYR_SDK="/scratch2/dima/zephyr-chipyard-sw-fresh/tools-manual/zephyr-sdk-1.0.0-beta1"

# ---- resolve repo root (no matter where you sourced from) -----------------

_MB_SCRIPT="${BASH_SOURCE[0]:-$0}"
_MB_REPO_ROOT="$(cd "$(dirname "${_MB_SCRIPT}")/.." && pwd)"

# ---- 1) conda env (gives us west + spike + riscv-tools) -------------------

if [ -f "${_MB_CHIPYARD_CONDA}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${_MB_CHIPYARD_CONDA}/etc/profile.d/conda.sh"
    conda activate "${_MB_CHIPYARD_CONDA}" >/dev/null
else
    echo "warning: chipyard conda env not found at ${_MB_CHIPYARD_CONDA}" >&2
fi

# riscv-tools/bin lives inside the conda env but isn't on PATH after activate.
if [ -d "${_MB_CHIPYARD_CONDA}/riscv-tools/bin" ]; then
    export PATH="${_MB_CHIPYARD_CONDA}/riscv-tools/bin:${PATH}"
fi

# ---- 2) Zephyr workspace + SDK --------------------------------------------

export ZEPHYR_BASE="${_MB_ZEPHYR_BASE}"
export ZEPHYR_SDK_INSTALL_DIR="${_MB_ZEPHYR_SDK}"
export ZEPHYR_TOOLCHAIN_VARIANT="zephyr"

if [ -d "${ZEPHYR_SDK_INSTALL_DIR}/gnu/riscv64-zephyr-elf/bin" ]; then
    export PATH="${ZEPHYR_SDK_INSTALL_DIR}/gnu/riscv64-zephyr-elf/bin:${PATH}"
fi

# ---- 3) Bedrock creds from .env ------------------------------------------

if [ -f "${_MB_REPO_ROOT}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${_MB_REPO_ROOT}/.env"
    set +a
fi

# ---- 4) FireSim env -------------------------------------------------------
# FireSim needs (a) the firesim CLI on PATH and (b) ssh-agent holding the
# key that authenticates to localhost (the FPGA host on this machine).
# Both are skipped silently when their prerequisites aren't present so
# spike-only sessions don't hit avoidable warnings.

_MB_FIRESIM_DIR="/scratch2/agustin/chipyard/sims/firesim"
_MB_FIRESIM_SSH_KEY="${HOME}/.ssh/firesim"

# Point validation/firesim_runner.py at OUR install (agustin's) -- the
# bitstream there is alveo_u250_firesim_shuttle_gemmini_opu (the
# GemminiAndOPUShuttleConfig hetero, tile 0 Gemmini RoCC + tile 1 Saturn
# OPU). dima's install has dual-rocket-saturn-gemmini-q31 but the logs
# dir under that tree is not writable by us, so firesim CLI invocations
# from this user can't complete there. Override via FIRESIM_ROOT /
# FIRESIM_ENV if pointing at a different chipyard install.
export FIRESIM_ROOT="${FIRESIM_ROOT:-${_MB_FIRESIM_DIR}}"
export FIRESIM_ENV="${FIRESIM_ENV:-/scratch2/agustin/chipyard/env.sh}"

if [ -f "${_MB_FIRESIM_DIR}/sourceme-manager.sh" ]; then
    # sourceme-manager appends ${dir}/deploy to PATH so the `firesim`
    # CLI becomes available. It also expects to be sourced FROM the
    # firesim dir, so cd-then-back.
    _mb_orig_cwd="$(pwd)"
    cd "${_MB_FIRESIM_DIR}"
    set +u
    # shellcheck disable=SC1091
    source ./sourceme-manager.sh --skip-ssh-setup >/dev/null
    set -u 2>/dev/null || true
    cd "${_mb_orig_cwd}"
    unset _mb_orig_cwd
fi

# ssh-agent: load the firesim key if (a) the key file exists and
# (b) it's not already loaded. ssh-add -l returns 1 when no keys.
if [ -r "${_MB_FIRESIM_SSH_KEY}" ]; then
    if [ -z "${SSH_AUTH_SOCK:-}" ] || ! ssh-add -l >/dev/null 2>&1; then
        eval "$(ssh-agent -s)" >/dev/null 2>&1 || true
    fi
    if ssh-add -l 2>/dev/null | grep -q "${_MB_FIRESIM_SSH_KEY}"; then
        :   # already loaded
    else
        ssh-add "${_MB_FIRESIM_SSH_KEY}" >/dev/null 2>&1 || true
    fi
fi

# ---- 5) verify (defers to check_benchmark_env.sh for the assertions) ------

if [ -x "${_MB_REPO_ROOT}/scripts/check_benchmark_env.sh" ]; then
    "${_MB_REPO_ROOT}/scripts/check_benchmark_env.sh"
else
    echo "note: scripts/check_benchmark_env.sh not found or not executable; skipping env check" >&2
fi
