#!/usr/bin/env bash
# One-command demo of the full XPU-RT ⇄ ModelBlaster agentic loop.
#
#   1. XPU-RT side: run iterate_firesim.py + granularity_loop.py to
#      produce a candidate bundle (firesim_batch.json) + fusion hint
#      (granularity_hint.json) + a predicted before/after Gantt.
#   2. ModelBlaster side: build each candidate's harness_xpurt ELF and
#      run it on FireSim under FIRESIM_QUEUE=1.
#   3. Close the loop: emit measured SchedulerReports, render
#      predicted-vs-actual Gantts, run XPU-RT's advisor on the
#      measured numbers, and produce a single round1_report.md.
#
# Usage:
#   bash scripts/run_agentic_loop_demo.sh                # all xpurt-realizable candidates
#   bash scripts/run_agentic_loop_demo.sh baseline,A2    # restrict to two
#
# Output: artifacts/bundle/round1_report.md (the final demo report),
# artifacts/iterate/before_after_gantt.png (predicted-only, XPU-RT side),
# artifacts/bundle/<id>/predicted_vs_actual.png (per candidate, ModelBlaster side).
set -o pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

INCLUDE="${1:-baseline,A1,A2,A3,A4}"
DEADLINE_US="${DEADLINE_US:-65}"

XPURT_ROOT="${XPURT_ROOT:-/scratch2/agustin/XPU-RT}"
BATCH="${XPURT_ROOT}/artifacts/iterate/firesim_batch.json"
HINT="${XPURT_ROOT}/artifacts/iterate/granularity_hint.json"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/artifacts/bundle}"

echo "=== [1/3] XPU-RT iterate (predicted bundle + Gantt + hint) ==="
if [[ ! -f "${BATCH}" || "${FORCE_REITERATE:-0}" == "1" ]]; then
    (cd "${XPURT_ROOT}" && bash scripts/demo_iterate_firesim.sh)
else
    echo "  using existing ${BATCH}"
fi

echo
echo "=== [2/3] ModelBlaster bundle on FireSim ==="
mkdir -p "${OUT_DIR}"
bash scripts/run_bundle_firesim.sh \
    --batch "${BATCH}" \
    --out-dir "${OUT_DIR}" \
    --include "${INCLUDE}"

echo
echo "=== [2.5/3] Per-step Gantt PNGs (predicted + measured side by side) ==="
python3 scripts/render_per_step.py --manifest "${OUT_DIR}/manifest.json" || true

echo
echo "=== [3/3] Close the loop (measured reports + predicted-vs-actual Gantts + re-advise) ==="
set +u
source scripts/setup_benchmark_env.sh >/dev/null 2>&1 || true
set -u
python3 scripts/close_xpurt_loop.py \
    --manifest "${OUT_DIR}/manifest.json" \
    --deadline-us "${DEADLINE_US}"

echo
echo "demo done — see:"
echo "  predicted before/after: ${XPURT_ROOT}/artifacts/iterate/before_after_gantt.png"
echo "  per-candidate measured: ${OUT_DIR}/<id>/predicted_vs_actual.png"
echo "  round-1 verdict:        ${OUT_DIR}/round1_report.md"
