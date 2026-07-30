#!/usr/bin/env bash
# Run a set of KernelBench level1 benchmarks through ModelBlaster, one op per
# ELF (cleanest per-op cycle measurement). Default: fp32 reference on RVV/spike.
# Env: BENCHES=comma-list of level1 basenames (default: 6 drop-in benches);
#      TARGET (default rvv), BACKEND (default reference), QUANT (default fp32),
#      RUNNER (default spike); JOBS=N runs N benches concurrently (default 1).
# Results -> examples/kernelbench/results/.
#
# Parallelism: each bench has its own examples/kernelbench/<kb>/ tree (generated
# IR + build dir + spike run are all isolated), so benches fan out safely. spike
# is single-threaded per run, so JOBS≈min(cores/4, nbenches) keeps the box busy
# without oversubscribing the parallel ninja builds. Do NOT use JOBS>1 on the
# firesim RUNNER — that serializes on the shared FPGA and must stay JOBS=1.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BENCH_DIR="${REPO_ROOT}/modelblaster/bench/level1"
: "${TARGET:=rvv}"; : "${BACKEND:=reference}"; : "${QUANT:=fp32}"; : "${RUNNER:=spike}"
: "${JOBS:=1}"
export TARGET BACKEND QUANT RUNNER FORCE_EXTRACT="${FORCE_EXTRACT:-1}"
if [[ "${RUNNER}" == "firesim" && "${JOBS}" != "1" ]]; then
  echo "refusing JOBS=${JOBS} on firesim (shared FPGA); forcing JOBS=1" >&2; JOBS=1
fi

DEFAULT=(19_ReLU 21_Sigmoid 31_ELU 33_BatchNorm 42_Max_Pooling_2D \
         63_conv_standard_2D__square_input__square_kernel)
if [[ -n "${BENCHES:-}" ]]; then IFS=',' read -ra LIST <<< "${BENCHES}"; else LIST=("${DEFAULT[@]}"); fi

# One bench end-to-end; prints a single "STATUS<TAB>bench<TAB>err<TAB>note" line
# so the parallel driver can collect results race-free (each writes one line).
run_bench() {
  local b="$1"
  local f="${BENCH_DIR}/${b}.py"
  [[ -f "$f" ]] || f="$(ls ${BENCH_DIR}/${b}*.py 2>/dev/null | head -1)"
  if [[ -z "$f" || ! -f "$f" ]]; then printf 'SKIP\t%s\t-\t(not found)\n' "$b"; return; fi
  local log="/tmp/kb_${b}.log"
  if BENCH_FILE="$f" bash "${REPO_ROOT}/modelblaster/examples/kernelbench/run_one.sh" > "$log" 2>&1; then
    local err; err=$(grep -aoE "max_abs_err=[0-9.eE+-]+" "$log" | tail -1)
    if grep -qaE "^PASS$" "$log"; then printf 'PASS\t%s\t%s\tok\n' "$b" "$err"
    else printf 'FAIL-verify\t%s\t%s\t-\n' "$b" "$err"; fi
  else
    local why; why=$(grep -aE "NotImplementedError|Error:|error:|does not exist" "$log" | tail -1 | cut -c1-90)
    printf 'FAIL-run\t%s\t-\t%s\n' "$b" "$why"
  fi
}
export -f run_bench
export BENCH_DIR REPO_ROOT

# Fan out across JOBS workers; each bench emits exactly one status line, which
# xargs streams to a temp file we then parse (order is completion-order).
res_tmp="$(mktemp)"
printf '%s\n' "${LIST[@]}" | xargs -d '\n' -P "${JOBS}" -I{} bash -c 'run_bench "$@"' _ {} > "$res_tmp"

pass=0; fail=0; declare -a rows
# Re-emit in the original bench order for a stable report.
for b in "${LIST[@]}"; do
  line=$(grep -P "\t${b}\t" "$res_tmp" | head -1)
  [[ -z "$line" ]] && continue
  IFS=$'\t' read -r s bb e n <<< "$line"
  case "$s" in
    PASS)        echo "PASS  $b   (max_abs_err=${e#max_abs_err=})"; pass=$((pass+1)); rows+=("$b|PASS|$e|ok");;
    SKIP)        echo "SKIP  $b ($n)"; rows+=("$b|SKIP|-|-");;
    FAIL-verify) echo "FAIL  $b (verify)  ($e)"; fail=$((fail+1)); rows+=("$b|FAIL-verify|$e|-");;
    *)           echo "FAIL  $b (build/run): $n"; fail=$((fail+1)); rows+=("$b|FAIL-run|-|$n");;
  esac
done
rm -f "$res_tmp"
echo "=== ${pass} PASS / ${fail} FAIL (TARGET=${TARGET} BACKEND=${BACKEND} QUANT=${QUANT}) ==="
out="${REPO_ROOT}/modelblaster/examples/kernelbench/results/${TARGET}_${QUANT}.md"
{ echo "# KernelBench level1 on ${TARGET}/${QUANT} (${BACKEND}, ${RUNNER})"; echo;
  echo "| bench | status | err | note |"; echo "|---|---|---|---|";
  for r in "${rows[@]}"; do IFS='|' read -r a s e n <<< "$r"; echo "| $a | $s | $e | $n |"; done;
  echo; echo "_${pass} PASS / ${fail} FAIL_"; } > "$out"
echo "results -> $out"
