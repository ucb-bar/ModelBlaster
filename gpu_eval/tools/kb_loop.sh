#!/bin/bash
# One ModelBlaster <-> KernelBlaster CUDA optimisation round for an IR:
#   1. export the IR's partitions as multi-slot problems      (pipeline/kb_export)
#   2. KernelBlaster optimises them, one slot per step         (scripts/run_multislot.py)
#   3. for each problem with fusion proposals: build the fused variant
#      (pipeline/kb_proposals variant), optimise it, accept/reject it (decide)
#   4. ingest the winners into the CUDA kernel cache and re-verify the whole
#      IR with them                                            (pipeline/kb_ingest)
#
#   [PY=python] [KB=KernelBlaster checkout] [MODEL=claude-code:sonnet] [REFLECT_MODEL=...]
#   [ARCH=75] [STRATEGY=whole|window|per_op] [MAX_OPS=12] [ROLLOUTS=3] [STEPS=5]
#   [CACHE=dir] [MB_PATH_EXCLUDE=pattern] \
#       bash gpu_eval/tools/kb_loop.sh IR_DIR WORK_DIR
#
# WORK_DIR/knowledge is the shared KernelBlaster knowledge dir (impl store,
# outcomes, claims): reuse WORK_DIR, or point KNOWLEDGE at one, to carry
# winners and lessons across IRs.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
MB_ROOT=${MB_ROOT:-$(cd "$here/../.." && pwd -P)}
IR=$(realpath "$1"); W=$(realpath -m "$2")
PY=${PY:-python}; KB=${KB:-$MB_ROOT/third_party/KernelBlaster}
MODEL=${MODEL:-claude-code:sonnet}; ARCH=${ARCH:-75}
STRATEGY=${STRATEGY:-whole}; MAX_OPS=${MAX_OPS:-12}; ROLLOUTS=${ROLLOUTS:-3}; STEPS=${STEPS:-5}
KNOWLEDGE=${KNOWLEDGE:-$W/knowledge}; CACHE=${CACHE:-$W/cache}
[ -z "${MB_PATH_EXCLUDE:-}" ] || export PATH=$(echo "$PATH" | tr : '\n' | grep -Ev "$MB_PATH_EXCLUDE" | paste -sd:)
export PYTHONPATH="$(dirname "$MB_ROOT")${PYTHONPATH:+:$PYTHONPATH}"
step() { echo "[$(date +%H:%M:%S)] $*"; }
run_kb() {  # problem(s) out [winners]
  local out=$1; shift
  $PY "$KB/scripts/run_multislot.py" "$@" --out "$out" --knowledge-dir "$KNOWLEDGE" --model "$MODEL" \
      ${REFLECT_MODEL:+--reflect-model "$REFLECT_MODEL"} --arch "$ARCH" --rollouts "$ROLLOUTS" \
      --steps "$STEPS" --export-winners "$W/winners" > "$out.log" 2>&1
}
mkdir -p "$W"

step "export ($STRATEGY)"
rm -rf "$W/problems"
$PY -m modelblaster.pipeline.kb_export "$IR" "$W/problems" --strategy "$STRATEGY" --max-ops "$MAX_OPS" > "$W/export.jsonl"

step "optimise $(grep -c problem "$W/export.jsonl") problem(s)"
run_kb "$W/run" "$W/problems"

for props in "$W"/run/*/proposals.json; do
  [ -f "$props" ] || continue
  n=$($PY -c "import json,sys; print(len(json.load(open(sys.argv[1]))['proposals']))" "$props")
  [ "$n" -gt 0 ] || continue
  name=$(basename "$(dirname "$props")"); V="$W/variants/$name"
  step "$name: $n fusion proposal(s) -> variant"
  rm -rf "$V"; mkdir -p "$W/variants"
  $PY -m modelblaster.pipeline.kb_proposals variant "$props" \
      "$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['problem_dir'])" "$props")" "$V" > "$V.variant.json"
  run_kb "$V/run" "$V/problem"
  $PY -m modelblaster.pipeline.kb_proposals decide "$(dirname "$props")/result.json" \
      "$V/run/$name"__fused/result.json --variant-json "$V/variant.json" \
      --registry "$W/fusion_registry.json" | tee "$V.decision.json"
done

step "ingest winners + model-level verify"
$PY -m modelblaster.pipeline.kb_ingest "$W/winners/winners.json" --cache "$CACHE" --verify "$IR" --arch "$ARCH" \
    --fusion-registry "$W/fusion_registry.json" | tee "$W/ingest.jsonl"
step "done: $W"
