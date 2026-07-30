#!/usr/bin/env bash
# Run one KernelBench level1 benchmark through ModelBlaster (one op per ELF).
# Thin wrapper around _run_lib.sh: parses a level1 .py via extract_graph's
# --bench-file loader and drives the rest of the pipeline. Per-bench example
# dir (examples/kernelbench/<kb_name>/) keeps generated/build/cache isolated.
#
# Env: BENCH_FILE (required, level1 .py path); TARGET={scalar,rvv} (default rvv);
#      BACKEND={reference,llm} (default reference); QUANT={fp32} (default fp32);
#      RUNNER={spike,firesim,native} (default spike);
#      MB_KB_DATA_ROOT (default /scratch/dima/mb_kb; set empty to disable) —
#        the per-bench generated/build data (which reaches many GB at stock
#        dims) is placed here via a symlink so it lands on a roomy/fast disk
#        instead of filling the repo's partition; parallel-safe (per-bench dir).
set -euo pipefail
: "${BENCH_FILE:?BENCH_FILE must be a KernelBench level1 .py path}"
[[ -f "${BENCH_FILE}" ]] || { echo "ERROR: BENCH_FILE=${BENCH_FILE} not found" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
_BASE="$(basename "${BENCH_FILE}" .py)"
# Mirror extract_graph._load_kernelbench's C-id sanitization.
KB_NAME="kb_$(echo "${_BASE}" | sed 's/[^A-Za-z0-9_]/_/g; s/__*/_/g; s/^_//; s/_$//')"
MODEL_NAME="kernelbench/${KB_NAME}"

# Redirect the heavy per-bench data dir onto a roomy disk (default: /scratch).
: "${MB_KB_DATA_ROOT:=/scratch/dima/mb_kb}"
if [[ -n "${MB_KB_DATA_ROOT}" ]]; then
    _kb_local="${REPO_ROOT}/modelblaster/examples/kernelbench/${KB_NAME}"
    _kb_data="${MB_KB_DATA_ROOT}/${KB_NAME}"
    mkdir -p "${_kb_data}"
    # Swap any pre-existing real dir for a symlink (FORCE_EXTRACT regenerates).
    if [[ -e "${_kb_local}" && ! -L "${_kb_local}" ]]; then rm -rf "${_kb_local}"; fi
    ln -sfn "${_kb_data}" "${_kb_local}"
fi
: "${TARGET:=rvv}"; : "${BACKEND:=reference}"; : "${QUANT:=fp32}"; : "${RUNNER:=spike}"
export BENCH_FILE MODEL_NAME REPO_ROOT TARGET BACKEND QUANT RUNNER
source "${REPO_ROOT}/modelblaster/examples/_run_lib.sh"
