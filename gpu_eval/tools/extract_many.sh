#!/bin/bash
# extract_many.sh OUT_ROOT TAG FILE...   (env: MB_FX_TRACER, MB_TARGET_MB, MB_WEIGHTS_MB, MB_GFLOPS, MB_QUANT, JOBS, KEEP_NPZ)
# Runs extract_graph on each KernelBench file in parallel; writes OUT_ROOT/<stem>/ and
# OUT_ROOT/status_<TAG>.txt (stem<TAB>OK|FAIL<TAB>last error line).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
OUT=$1; TAG=$2; shift 2; mkdir -p "$OUT/logs"
# one torch thread pool per job, not 48 threads x JOBS
export OMP_NUM_THREADS=$(( $(nproc) / ${JOBS:-16} > 0 ? $(nproc) / ${JOBS:-16} : 1 )); export MKL_NUM_THREADS=$OMP_NUM_THREADS
one() {
  f=$1; stem=$(basename "$f" .py); d="$OUT/$stem"
  if timeout 900 python -m modelblaster.pipeline.extract_graph --bench-file "$f" \
       --bench-target-mb "${MB_TARGET_MB:-64}" --bench-target-weights-mb "${MB_WEIGHTS_MB:-0}" --bench-target-gflops "${MB_GFLOPS:-0}" --out-dir "$d" --quant "${MB_QUANT:-fp32}" > "$OUT/logs/$stem.$TAG.log" 2>&1; then
    printf "%s\tOK\t\n" "$stem"
  else
    printf "%s\tFAIL\t%s\n" "$stem" "$(grep -E 'Error|error' "$OUT/logs/$stem.$TAG.log" | tail -1 | cut -c1-200)"
  fi
  [ -n "$KEEP_NPZ" ] || rm -f "$d"/*.npz
}
export -f one; export OUT TAG
printf "%s\n" "$@" | xargs -P "${JOBS:-16}" -I{} bash -c 'one {}' | sort -V > "$OUT/status_$TAG.txt"
grep -c OK "$OUT/status_$TAG.txt"
