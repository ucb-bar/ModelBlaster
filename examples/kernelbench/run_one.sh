#!/usr/bin/env bash
# Run one KernelBench level1 benchmark through ModelBlaster (one op per ELF).
# Thin wrapper around _run_lib.sh: parses a level1 .py via extract_graph's
# --bench-file loader and drives the rest of the pipeline. Per-bench example
# dir (examples/kernelbench/<kb_name>/) keeps generated/build/cache isolated.
#
# Env: BENCH_FILE (required, level1 .py path); TARGET={scalar,rvv} (default rvv);
#      BACKEND={reference,llm} (default reference); QUANT={fp32} (default fp32);
#      RUNNER={spike,firesim} (default spike).
set -euo pipefail
: "${BENCH_FILE:?BENCH_FILE must be a KernelBench level1 .py path}"
[[ -f "${BENCH_FILE}" ]] || { echo "ERROR: BENCH_FILE=${BENCH_FILE} not found" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
_BASE="$(basename "${BENCH_FILE}" .py)"
# Mirror extract_graph._load_kernelbench's C-id sanitization.
KB_NAME="kb_$(echo "${_BASE}" | sed 's/[^A-Za-z0-9_]/_/g; s/__*/_/g; s/^_//; s/_$//')"
MODEL_NAME="kernelbench/${KB_NAME}"
: "${TARGET:=rvv}"; : "${BACKEND:=reference}"; : "${QUANT:=fp32}"; : "${RUNNER:=spike}"
export BENCH_FILE MODEL_NAME REPO_ROOT TARGET BACKEND QUANT RUNNER
source "${REPO_ROOT}/modelblaster/examples/_run_lib.sh"
