#!/usr/bin/env bash
# Profile all v3 vision slices on QRB5165 across CPU, DSP, and HTA backends.
#
# For each segment:
#   - Push DLC + input data to board
#   - Run qnn-net-run with --profiling_level detailed
#   - Pull profiling log, convert to CSV via qnn-profile-viewer
#
# Output: boards/qrb5165_v66/profiles/smolvlm_vision_v3/
#   dsp_seg_00__CPU.csv  dsp_seg_00__DSP.csv  dsp_seg_00__HTA.csv
#   cpu_seg_00__CPU.csv  ...
#
# Usage:
#   ./profile_vision_v3.sh                     # profile all segments on all backends
#   ./profile_vision_v3.sh --backend CPU       # only CPU
#   ./profile_vision_v3.sh --backend HTA       # only HTA (dsp_seg only, conv1x1)
#   ./profile_vision_v3.sh --seg dsp_seg_03    # only one segment
#
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
SLICES_DIR="$HERE/vision_slices_v3"
HTA_CONVS_DIR="$SLICES_DIR/hta_convs/dlc"
DLC_DIR="$SLICES_DIR/dlc"
BOARD="${QNN_BOARD_HOST:-root@<BOARD_IP>}"
QNN_SDK="<PATH_TO_QAIRT_SDK>"
PYTHON="<PATH_TO_CONDA_ENV>/bin/python"
N_ITERS=10

OUT_DIR="$REPO_ROOT/qnn_models/boards/qrb5165_v66/profiles/smolvlm_vision_v3"
mkdir -p "$OUT_DIR"

# Parse args
BACKEND_FILTER=""
SEG_FILTER=""
while [ $# -gt 0 ]; do
    case "$1" in
        --backend) BACKEND_FILTER="$2"; shift 2 ;;
        --backend=*) BACKEND_FILTER="${1#*=}"; shift ;;
        --seg) SEG_FILTER="$2"; shift 2 ;;
        --seg=*) SEG_FILTER="${1#*=}"; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

REMOTE_DIR="/root/models/smolvlm_vision_v3"

lib_for() {
    case "$1" in
        CPU) echo libQnnCpu.so ;;
        DSP) echo libQnnDsp.so ;;
        HTA) echo libQnnHta.so ;;
    esac
}

echo "=== SmolVLA Vision v3 Profile Sweep ==="
echo "Board: $BOARD  Output: $OUT_DIR"
echo ""

# Ensure remote dir exists
ssh "$BOARD" "mkdir -p $REMOTE_DIR/dlc $REMOTE_DIR/hta_dlc $REMOTE_DIR/inputs" 2>/dev/null

TOTAL=0; OK=0; FAIL=0; SKIP=0

# Generate random input data for each segment (if not already done)
echo "--- Generating input data ---"
INPUT_DIR="$SLICES_DIR/profile_inputs"
mkdir -p "$INPUT_DIR"
$PYTHON - "$SLICES_DIR" "$INPUT_DIR" <<'PYEOF'
import sys, os
import numpy as np
from pathlib import Path

slices_dir = Path(sys.argv[1])
input_dir = Path(sys.argv[2])

try:
    import onnx
except ImportError:
    print("ERROR: need onnx. run: pip install onnx")
    sys.exit(1)

for onnx_file in sorted(slices_dir.glob("*_seg_*.onnx")):
    seg_name = onnx_file.stem
    list_file = input_dir / f"{seg_name}_input_list.txt"
    if list_file.exists():
        continue
    model = onnx.load(str(onnx_file))
    input_files = []
    for inp in model.graph.input:
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        # Generate random fp32 input
        raw_path = input_dir / f"{seg_name}_{inp.name}.raw"
        if not raw_path.exists():
            data = np.random.randn(*dims).astype(np.float32) * 0.1
            data.tofile(str(raw_path))
        input_files.append(f"{inp.name}:=/root/models/smolvlm_vision_v3/inputs/{raw_path.name}")
    with open(list_file, "w") as f:
        line = " ".join(input_files)
        for _ in range(10):
            f.write(line + "\n")
    print(f"  {seg_name}: {len(model.graph.input)} inputs, dims={[d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]}")
print("Input data ready.")
PYEOF

# Push all input data to board
echo "--- Pushing inputs to board ---"
rsync -az --info=progress2 "$INPUT_DIR/" "$BOARD:$REMOTE_DIR/inputs/" 2>/dev/null
echo "  Done."

# Push DLCs to board
echo "--- Pushing DLCs to board ---"
if [ -d "$DLC_DIR" ]; then
    rsync -az --info=progress2 "$DLC_DIR/" "$BOARD:$REMOTE_DIR/dlc/" 2>/dev/null
fi
if [ -d "$HTA_CONVS_DIR" ]; then
    rsync -az --info=progress2 "$HTA_CONVS_DIR/" "$BOARD:$REMOTE_DIR/hta_dlc/" 2>/dev/null
fi
echo "  Done."
echo ""

# Profile each segment on each backend
for seg_onnx in "$SLICES_DIR"/*_seg_*.onnx; do
    seg_name="$(basename "$seg_onnx" .onnx)"

    # Filter
    [ -n "$SEG_FILTER" ] && [ "$seg_name" != "$SEG_FILTER" ] && continue

    # Determine which backends to try for this segment
    if [[ "$seg_name" == cpu_seg_* ]]; then
        BACKENDS=(CPU)
    else
        BACKENDS=(CPU DSP HTA)
    fi

    for BE in "${BACKENDS[@]}"; do
        [ -n "$BACKEND_FILTER" ] && [ "$BE" != "$BACKEND_FILTER" ] && continue

        OUTCSV="$OUT_DIR/${seg_name}__${BE}.csv"
        if [ -f "$OUTCSV" ] && [ -s "$OUTCSV" ]; then
            SKIP=$((SKIP+1))
            continue
        fi
        TOTAL=$((TOTAL+1))

        LIB=$(lib_for "$BE")
        INPUT_LIST="$INPUT_DIR/${seg_name}_input_list.txt"
        [ -f "$INPUT_LIST" ] || { echo "  SKIP $seg_name/$BE: no input list"; continue; }

        # Determine which DLC to use
        if [ "$BE" = "HTA" ]; then
            # For HTA, use the pure conv1x1 DLCs (one per conv op in the segment)
            # For now, run the quantized DSP DLC on HTA (it will use what it can)
            # Actually: HTA needs the special standalone conv DLCs
            # Use the first matching hta conv DLC for this segment
            HTA_DLC=$(ssh "$BOARD" "ls $REMOTE_DIR/hta_dlc/${seg_name}_*_quantized.dlc 2>/dev/null | head -1" 2>/dev/null)
            if [ -z "$HTA_DLC" ]; then
                # Try the segment's quantized DLC directly
                DLC_PATH="$REMOTE_DIR/dlc/${seg_name}_quantized.dlc"
            else
                DLC_PATH="$HTA_DLC"
            fi
        elif [ "$BE" = "DSP" ]; then
            DLC_PATH="$REMOTE_DIR/dlc/${seg_name}_quantized.dlc"
        else
            DLC_PATH="$REMOTE_DIR/dlc/${seg_name}.dlc"
        fi

        echo -n "  $seg_name / $BE ... "

        # Run profiling on board
        PROF_OK=$(ssh "$BOARD" bash -s <<REMOTE 2>/dev/null
set -e
cd $REMOTE_DIR
DLC="$DLC_PATH"
[ -f "\$DLC" ] || exit 3
export LD_LIBRARY_PATH=/root/qairt/lib/target:\${LD_LIBRARY_PATH:-}
export ADSP_LIBRARY_PATH="/root/qairt/lib/hexagon-v66;/dsp/cdsp;/dsp"
rm -rf /tmp/_v3_prof
/root/qairt/bin/target/qnn-net-run \
    --dlc_path "\$DLC" \
    --backend "/root/qairt/lib/target/$LIB" \
    --input_list "$REMOTE_DIR/inputs/${seg_name}_input_list.txt" \
    --output_dir /tmp/_v3_prof \
    --profiling_level detailed >/dev/null 2>&1 || exit 5
[ -f /tmp/_v3_prof/qnn-profiling-data_0.log ] || exit 6
cp /tmp/_v3_prof/qnn-profiling-data_0.log /tmp/_v3_prof_log.bin
echo OK
REMOTE
        )

        if [ "$PROF_OK" != "OK" ]; then
            echo "FAIL (board run)"
            FAIL=$((FAIL+1))
            continue
        fi

        # Pull log
        TMP_LOG="/tmp/_v3_${seg_name}__${BE}.log"
        scp -q "$BOARD:/tmp/_v3_prof_log.bin" "$TMP_LOG" 2>/dev/null
        if [ ! -s "$TMP_LOG" ]; then
            echo "FAIL (empty log)"
            FAIL=$((FAIL+1))
            continue
        fi

        # Convert to CSV via qnn-profile-viewer in docker
        TMP_CSV="/tmp/_v3_${seg_name}__${BE}.csv"
        sudo docker run --rm \
            -v "$QNN_SDK":/qnn:ro \
            -v /tmp:/work \
            qnn-convert \
            /qnn/bin/x86_64-linux-clang/qnn-profile-viewer \
                --input_log "/work/$(basename "$TMP_LOG")" \
                --output "/work/$(basename "$TMP_CSV")" >/dev/null 2>&1

        if [ -f "$TMP_CSV" ] && [ -s "$TMP_CSV" ]; then
            sudo cp "$TMP_CSV" "$OUTCSV"
            sudo chown "$(id -u):$(id -g)" "$OUTCSV"
            ROWS=$(wc -l < "$OUTCSV")
            echo "OK ($ROWS rows)"
            OK=$((OK+1))
        else
            echo "FAIL (viewer)"
            FAIL=$((FAIL+1))
        fi
        rm -f "$TMP_LOG" "$TMP_CSV"
    done
done

echo ""
echo "=== Summary ==="
echo "  OK: $OK  FAIL: $FAIL  SKIP: $SKIP  Total attempted: $TOTAL"
echo "  Output: $OUT_DIR/"
