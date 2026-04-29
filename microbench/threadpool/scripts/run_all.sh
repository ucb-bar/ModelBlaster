#!/usr/bin/env bash
# Run every threadpool microbench variant and merge the results.
#
# Builds + runs:
#   pthreadpool   (variant=default)  — condvar-based, 1M spin iters
#   pthreadpool   (variant=spin)     — 100M spin iters, no condvar fallback
#   pthreads_raw  (variant=k_sem)    — bare POSIX threads + k_sem
#   k_thread      (variant=k_sem)    — bare Zephyr threads + k_sem
#
# Outputs a merged CSV under
#   agents/microbench/threadpool/results/<runner>_overhead.csv
#
# Env:
#   RUNNER=<spike|firesim>          default firesim
#   FIRESIM_TIMEOUT=900             per-bench timeout
set -euo pipefail

RUNNER="${RUNNER:-firesim}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

OUT_DIR="${REPO_ROOT}/agents/microbench/threadpool/results"
mkdir -p "${OUT_DIR}"

case "${RUNNER}" in
    spike)   OUT_CSV="${OUT_DIR}/spike_overhead.csv" ;;
    firesim) OUT_CSV="${OUT_DIR}/firesim_overhead.csv" ;;
    *) echo "RUNNER=${RUNNER} not supported" >&2; exit 2 ;;
esac

# Run order: native (cheapest) first; pthreadpool variants last because
# they take longer per iteration (event-driven fall-back path).
declare -a CONFIGS=(
    "zephyr_threads:k_sem"
    "pthreads_raw:k_sem"
    "pthreadpool:default"
    "pthreadpool:spin"
)

# Reset header.
HEADER=""
TMP_BODY="$(mktemp)"
trap 'rm -f "${TMP_BODY}"' EXIT

for cfg in "${CONFIGS[@]}"; do
    BENCH="${cfg%%:*}"
    VARIANT="${cfg##*:}"
    echo
    echo "========================================="
    echo " RUNNING: BENCH=${BENCH} VARIANT=${VARIANT} RUNNER=${RUNNER}"
    echo "========================================="
    BENCH="${BENCH}" VARIANT="${VARIANT}" RUNNER="${RUNNER}" \
        SKIP_BUILD="${SKIP_BUILD:-0}" \
        bash "${REPO_ROOT}/agents/microbench/threadpool/scripts/run_bench.sh"
    SUB_CSV="${REPO_ROOT}/agents/microbench/threadpool/build/${BENCH}_${VARIANT}_${RUNNER}/bench.csv"
    if [[ ! -s "${SUB_CSV}" ]]; then
        echo "FAIL: ${SUB_CSV} not produced" >&2
        exit 3
    fi
    if [[ -z "${HEADER}" ]]; then
        HEADER="$(head -n 1 "${SUB_CSV}")"
    fi
    tail -n +2 "${SUB_CSV}" >> "${TMP_BODY}"
done

{ echo "${HEADER}"; cat "${TMP_BODY}"; } > "${OUT_CSV}"
echo
echo "[microbench] merged CSV -> ${OUT_CSV}"
wc -l "${OUT_CSV}"
