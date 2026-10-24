#!/bin/bash
# One ModelBlaster <-> KernelBlaster CUDA optimisation round for an IR:
#   1. export the IR's partitions as multi-slot problems      (pipeline/kb_export)
#      -- STRATEGY=plan: profile the IR and send the cost-ranked selection
#      (pipeline/kb_plan; COVERAGE, TOP_K)
#   2. KernelBlaster optimises them, one slot per step         (scripts/run_multislot.py)
#   3. for each problem with proposals (fuse, split): build the variant seeded
#      with the original's best kernels (pipeline/kb_proposals variant),
#      optimise it, accept/reject it on re-measured times (decide); the
#      decision lands in WORK_DIR/graph_hints.json (modelblaster.graph_hints/v1)
#   4. ingest the winners into the CUDA kernel cache and re-verify the whole
#      IR with them and the accepted hints                     (pipeline/kb_ingest)
#   5. the model's result: every partition's best slots composed into one
#      whole-model runtime, checked and timed by CUDA-graph replay against the
#      hand-seeded model                                      (pipeline/kb_model_run)
# Accepted hints in WORK_DIR/graph_hints.json are applied before step 1, so
# rerunning the script continues where the last round stopped. Rounds to a
# fixpoint over several IRs: python -m modelblaster.pipeline.kb_graph_opt.
#
#   [PY=python] [KB=KernelBlaster checkout] [MODEL=claude-code:sonnet] [REFLECT_MODEL=...]
#   [ARCH=75] [STRATEGY=whole|window|per_op|plan] [MAX_OPS=12] [COVERAGE=0.8] [TOP_K=6]
#   [ROLLOUTS=3] [STEPS=5] [REMEASURE=5] [CACHE=dir] [PLUGIN=stubs.py] [MB_PATH_EXCLUDE=pattern] \
#   [GPUS=auto|all|0,2] [SSH_HOST="user@h1 user@h2" REMOTE_DIR=dir [CVD=auto|ids] [SSH_KEY=file]] \
#   [RUNTIME=auto|host|process] [GPU_MODEL="NVIDIA L40S"] \
#       bash gpu_eval/tools/kb_loop.sh IR_DIR WORK_DIR
#
# GPUs: KernelBlaster's runs, the re-measurement of proposals (with SSH_HOST)
# and the model's result (step 5) use the same evaluator -- local GPU pool
# (GPUS) or the SSH_HOST GPUs -- and step 5 times on the GPU model the runs
# recorded (or GPU_MODEL), so the model and its partitions share the hardware.
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
COVERAGE=${COVERAGE:-0.8}; TOP_K=${TOP_K:-6}; REMEASURE=${REMEASURE:-5}
KNOWLEDGE=${KNOWLEDGE:-$W/knowledge}; CACHE=${CACHE:-$W/cache}; HINTS=$W/graph_hints.json
[ -z "${MB_PATH_EXCLUDE:-}" ] || export PATH=$(echo "$PATH" | tr : '\n' | grep -Ev "$MB_PATH_EXCLUDE" | paste -sd:)
export PYTHONPATH="$(dirname "$MB_ROOT")${PYTHONPATH:+:$PYTHONPATH}"
step() { echo "[$(date +%H:%M:%S)] $*"; }
# the evaluator flags (run_multislot.py, kb_model_run and kb_proposals decide take the same names)
EVAL=(--runtime "${RUNTIME:-auto}")
if [ -n "${SSH_HOST:-}" ]; then
  EVAL+=(--evaluator ssh --remote-dir "${REMOTE_DIR:?SSH_HOST needs REMOTE_DIR}")
  for h in $SSH_HOST; do EVAL+=(--ssh-host "$h"); done
  for c in ${CVD:-}; do EVAL+=(--cuda-visible-devices "$c"); done
  [ -z "${SSH_KEY:-}" ] || EVAL+=(--ssh-key "$SSH_KEY")
else
  EVAL+=(--evaluator local --gpus "${GPUS:-auto}")
fi
GPU=(); [ -z "${GPU_MODEL:-}" ] || GPU=(--gpu-model "$GPU_MODEL")
run_kb() {  # problem(s) out [winners]
  local out=$1; shift
  $PY "$KB/scripts/run_multislot.py" "$@" --out "$out" --knowledge-dir "$KNOWLEDGE" --model "$MODEL" \
      ${REFLECT_MODEL:+--reflect-model "$REFLECT_MODEL"} --arch "$ARCH" --rollouts "$ROLLOUTS" \
      --steps "$STEPS" --export-winners "$W/winners" ${PLUGIN:+--plugin "$PLUGIN"} "${EVAL[@]}" "${GPU[@]}" \
      > "$out.log" 2>&1
}
mkdir -p "$W"

step "export ($STRATEGY)"
rm -rf "$W/problems"
if [ "$STRATEGY" = plan ]; then
  $PY -m modelblaster.pipeline.kb_plan profile "$IR" --out "$W/profile" --hints "$HINTS" --cache "$CACHE" \
      --arch "$ARCH" > "$W/profile.jsonl"
  $PY -m modelblaster.pipeline.kb_plan select "$W"/profile/*/profile.json --out "$W/plan.json" \
      --coverage "$COVERAGE" --top-k "$TOP_K"
  $PY -m modelblaster.pipeline.kb_export --plan "$W/plan.json" "$W/problems" --cache "$CACHE" \
      --arch "$ARCH" > "$W/export.jsonl"
else
  $PY -m modelblaster.pipeline.kb_export "$IR" "$W/problems" --strategy "$STRATEGY" --max-ops "$MAX_OPS" \
      --hints "$HINTS" > "$W/export.jsonl"
fi

step "optimise $(grep -c problem "$W/export.jsonl") problem(s)"
run_kb "$W/run" "$W/problems"

for props in "$W"/run/*/proposals.json; do
  [ -f "$props" ] || continue
  n=$($PY -c "import json,sys; print(len(json.load(open(sys.argv[1]))['proposals']))" "$props")
  [ "$n" -gt 0 ] || continue
  name=$(basename "$(dirname "$props")"); V="$W/variants/$name"
  step "$name: $n proposal(s) -> variant"
  rm -rf "$V"; mkdir -p "$W/variants"
  $PY -m modelblaster.pipeline.kb_proposals variant "$props" \
      "$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['problem_dir'])" "$props")" "$V" \
      --seed-from "$(dirname "$props")" > "$V.variant.json"
  run_kb "$V/run" "$V/problem"
  $PY -m modelblaster.pipeline.kb_proposals decide "$(dirname "$props")/result.json" \
      "$V/run/$name"__variant/result.json --variant-json "$V/variant.json" \
      --hints "$HINTS" --remeasure "$REMEASURE" --arch "$ARCH" ${SSH_HOST:+"${EVAL[@]}" --kb "$KB" --kb-python "$PY"} \
      | tee "$V.decision.json"
done

step "ingest winners + model-level verify"
$PY -m modelblaster.pipeline.kb_ingest "$W/winners/winners.json" --cache "$CACHE" --verify "$IR" --arch "$ARCH" \
    --hints "$HINTS" | tee "$W/ingest.jsonl"
step "model result: every partition's winners in one whole-model runtime (CUDA graph replay)"
$PY -m modelblaster.pipeline.kb_model_run "$IR" --kb-runs "$W/run" --cache "$CACHE" --hints "$HINTS" \
    --arch "$ARCH" "${EVAL[@]}" "${GPU[@]}" --kb "$KB" --kb-python "$PY" --work "$W/model_run" \
    --out "$W/model_result.json" | tee "$W/model_result.line.json"
step "done: $W"
