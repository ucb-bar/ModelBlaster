#!/bin/bash
# Level3 GPU gates on any CUDA box: build the hand kernels, walk every IR on
# them (H3: golden + per-intermediate check), and build/run one KernelBlaster
# problem per op type through KB's own split + cuda_env CMake (H4).
#
#   [PY=python] [KB=KernelBlaster checkout] [NCU=1] [SPECS=specs.json] [MB_PATH_EXCLUDE=pattern] \
#       bash harness/run_level3_gpu.sh IR_ROOT OUT_DIR [SM_ARCH=89]
set -u
H=$(cd "$(dirname "$0")" && pwd -P); IRR=$(realpath "$1"); OUT=$(realpath -m "$2"); ARCH=${3:-89}
PY=${PY:-python}; KB=${KB:-$H/../../third_party/KernelBlaster}
mkdir -p "$OUT"
SPECS="${SPECS:-$OUT/specs.json}"; [ -f "$SPECS" ] || $PY "$H/../tools/dump_specs.py" "$SPECS" > /dev/null || exit 1
[ -z "${MB_PATH_EXCLUDE:-}" ] || export PATH=$(echo "$PATH" | tr : '\n' | grep -Ev "$MB_PATH_EXCLUDE" | paste -sd:)
nvcc -O3 -arch=sm_$ARCH -Xcompiler -fPIC -shared -o "$OUT/lib_hand.so" "$H/cuda/mb_cuda_kernels.cu" \
    > "$OUT/hand_build.log" 2>&1 || { cat "$OUT/hand_build.log"; exit 1; }
IRS=$(ls -d "$IRR"/*/graph.json | xargs -n1 dirname | sort -V)
(cd "$H" && $PY walk_cuda.py $IRS --specs "$SPECS" --hand "$OUT/lib_hand.so" --impl hand \
    --reps 3 --json "$OUT/h3_hand.json" > "$OUT/h3_hand.log" 2>&1)
$PY "$H/kb_targets.py" "$IRR" > "$OUT/kb_targets.txt"
(cd "$H" && $PY kb_bridge.py --kb "$KB" --specs "$SPECS" --hand-cu cuda/mb_cuda_kernels.cu \
    --out "$OUT/kb_problems" --arch "$ARCH" ${NCU:+--ncu} $(cat "$OUT/kb_targets.txt") \
    > "$OUT/h4_kb.log" 2>&1)
$PY - "$OUT" <<'PYEOF'
import json, sys
out = sys.argv[1]
h3 = json.load(open(f"{out}/h3_hand.json"))
h4 = json.load(open(f"{out}/kb_problems/kb_bridge_hand.json"))
ok3 = [r for r in h3 if r.get("ok") and not r.get("first_divergent_op")]
print(f"H3 hand-CUDA walk: {len(ok3)}/{len(h3)} IRs pass with no divergent intermediate")
for r in h3:
    if r not in ok3:
        print("  FAIL", r["ir"], r.get("error"), r.get("missing"), r.get("first_divergent_op"))
print(f"H4 KernelBlaster problems: {sum(r.get('ok', False) for r in h4)}/{len(h4)} op instances pass")
for r in h4:
    if not r.get("ok"):
        print("  FAIL", r["problem"], r.get("stage"), (r.get("run_stdout") or r.get("error") or "")[-200:])
PYEOF
