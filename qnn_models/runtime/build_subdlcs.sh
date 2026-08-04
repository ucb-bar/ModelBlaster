#!/usr/bin/env bash
#
# Take per-segment sub-ONNXes (from slice_to_subonnx.py's manifest.json)
# all the way through to per-segment per-backend context binaries the
# generated runtime can load. Three stages, each driven from a single
# manifest entry:
#
#   1. ONNX → DLC                  (qnn-onnx-converter, in qnn-convert docker)
#   2. DLC + calib → quantized DLC (qnn-quantizer, same docker)
#   3. quantized DLC + lib<X>.so → ctx_<net>_<label>_seg<id>.bin
#                                  (qnn-context-binary-generator, on board)
#
# The output bins are scp'd to the board's $CTX_DIR so the existing
# runtime (qnn_runtime built by build_and_run.sh) finds them on the next
# launch. The runtime's lookup key needs to grow a seg-id dimension once
# per-segment contexts are in play — see TODO at the bottom of this
# script.
#
# Usage:
#   bash build_subdlcs.sh <gen_dir>            # all segments
#   bash build_subdlcs.sh <gen_dir> --network dronet   # only one
#
# Requires (host):  docker + qnn-convert image (existing pipeline)
# Requires (board): the same QAIRT 2.45 install used by qnn_runtime.

set -euo pipefail
GEN_DIR="${1:?usage: build_subdlcs.sh <runtime_gen_dir> [--network NET]}"
shift || true

NET_FILTER=""
while [ $# -gt 0 ]; do
    case "$1" in
        --network) NET_FILTER="$2"; shift 2;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

BOARD_USER="${BOARD_USER:-root}"
BOARD_IP="${BOARD_IP:-<BOARD_IP>}"
BOARD_CTX_DIR="${BOARD_CTX_DIR:-/root/qnn_runtime_ctx}"
QNN_SDK="${QNN_SDK:-<PATH_TO_QAIRT_SDK>}"
DOCKER_IMAGE="${DOCKER_IMAGE:-qnn-convert}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SUB_ONNX_DIR="$GEN_DIR/sub_onnx"
SUB_DLC_DIR="$GEN_DIR/sub_dlc"
SUB_CTX_DIR="$GEN_DIR/sub_ctx"
mkdir -p "$SUB_DLC_DIR" "$SUB_CTX_DIR"

MANIFEST="$SUB_ONNX_DIR/manifest.json"
[ -f "$MANIFEST" ] || { echo "no $MANIFEST — run slice_to_subonnx.py first" >&2; exit 1; }

# Per-network calibration data path. The deploy.sh pipeline staged these
# under boards/<target>/calibration_data/<net>/ during the original DLC
# generation. We assume the same convention here; override
# CALIB_DIR_<NET> per network if your layout differs.
CALIB_BASE="${CALIB_BASE:-$REPO_DIR/qnn_models/boards/qrb5165_v66/calibration_data}"

# Walk the manifest. We process each unique sub-ONNX once (entries with
# `alias_of` already point at a previously-built one).
mapfile -t segs < <(<PATH_TO_CONDA_ENV>/bin/python3 -c "
import json, sys
m = json.load(open('$MANIFEST'))
for s in m['segments']:
    if 'alias_of' in s: continue
    if '$NET_FILTER' and s['network'] != '$NET_FILTER': continue
    src = s.get('source_onnx', '')   # absent in older manifests; we still need it for layout flags
    print(f\"{s['seg_id']}\t{s['network']}\t{s['label']}\t{s['sub_onnx_path']}\t{src}\")
")

echo "==> processing ${#segs[@]} unique sub-ONNXes (filter: '${NET_FILTER:-all}')"

# ---- Stage 1+2: ONNX → DLC → quantized DLC inside the qnn-convert container.
# Mount the repo at /work so paths line up with what we already use in
# deploy.sh / docker_run_qnn.
host_relpath() { realpath --relative-to="$REPO_DIR" "$1"; }

for entry in "${segs[@]}"; do
    seg_id="$(echo "$entry" | cut -f1)"
    net="$(echo "$entry" | cut -f2)"
    label="$(echo "$entry" | cut -f3)"
    sub_onnx="$(echo "$entry" | cut -f4)"
    src_onnx="$(echo "$entry" | cut -f5)"
    base="${net}_${label}_seg${seg_id}"

    sub_onnx_rel="$(host_relpath "$sub_onnx")"
    dlc_out="$SUB_DLC_DIR/$base.dlc"
    qdlc_out="$SUB_DLC_DIR/${base}_quantized.dlc"
    dlc_rel="$(host_relpath "$dlc_out")"
    qdlc_rel="$(host_relpath "$qdlc_out")"

    echo
    echo "=== seg $seg_id  $net/$label  (src=$(basename "$src_onnx")) ==="
    echo "    onnx → dlc"
    # snpe-onnx-to-dlc emits a real .dlc that qnn-context-binary-generator
    # consumes (the newer qnn-onnx-converter emits a .cpp+.bin model-lib
    # pair which context-binary-generator rejects with code 1002).
    docker run --rm -v "$REPO_DIR":/work "$DOCKER_IMAGE" bash -c "
        python3.10 /qnn/bin/x86_64-linux-clang/snpe-onnx-to-dlc \\
            --input_network /work/$sub_onnx_rel \\
            --output_path  /work/$dlc_rel 2>&1 | tail -3
    "

    # Per-segment calibration list emitted by capture_boundary_calibration.py.
    # The list contains absolute /work/... paths if the script was run
    # before the docker mount existed, so we patch them at quantize-time
    # to point inside /work.
    calib_list_host="$GEN_DIR/sub_onnx/calib/${net}/seg_${seg_id}/input_list.txt"
    if [ ! -f "$calib_list_host" ]; then
        echo "    (skip quantize — no per-segment calib at $(basename $calib_list_host))"
        continue
    fi
    # Rewrite absolute paths in input_list.txt → /work/... relative
    # paths so the docker container can read them. Also reorder per the
    # converted DLC's actual input order — snpe-onnx-to-dlc reorders
    # graph inputs by first-consumer position rather than ONNX
    # graph.input order, so an alphabetically-sorted input_list (which
    # is what slice_to_subonnx.py emits) sends e.g. yolov8n's head
    # 80×80 feature map to the slot expecting the 20×20 one. The
    # snpe-dlc-info output marks each input with `[NW Input]`; we parse
    # that to recover the right order, then read the captured tensor
    # files in DLC-input-name order rather than slicer-sorted order.
    docker_calib_list="${calib_list_host%.txt}.docker.txt"
    <PATH_TO_CONDA_ENV>/bin/python3 - \
        "$calib_list_host" "$docker_calib_list" "$REPO_DIR" "$dlc_out" <<'PY'
import sys, os, re, subprocess
inp, out, repo, dlc_path = sys.argv[1:5]

def safe(t: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", t)

# Discover the DLC's input order by parsing snpe-dlc-info output.
# Each "NW Input" line names the input tensor; tensors appear in DLC
# input-slot order. Run snpe-dlc-info inside the docker image so the
# version matches the converter that produced the DLC.
ABS = os.path.abspath(dlc_path)
REPO_ABS = os.path.abspath(repo)
rel = os.path.relpath(ABS, REPO_ABS)
info = subprocess.check_output([
    "docker", "run", "--rm", "-v", f"{REPO_ABS}:/work", "qnn-convert",
    "/qnn/bin/x86_64-linux-clang/snpe-dlc-info",
    "-i", f"/work/{rel}",
], stderr=subprocess.DEVNULL).decode()

dlc_input_names = []
for m in re.finditer(r"\b([\w/.]+)\s*\(data type:[^)]+\)\s*\[NW Input\]", info):
    nm = m.group(1)
    if nm not in dlc_input_names:
        dlc_input_names.append(nm)

# Read the per-tensor captured paths from the slicer-emitted input_list,
# index them by safe(tensor_name), then re-emit in DLC-input order.
with open(inp) as f:
    sample_rows = [ln.strip().split() for ln in f if ln.strip()]
# Each row's tokens are alphabetically ordered tensor paths from the
# slicer. Rebuild a (safe_tensor_name → path) map per row by reading
# the path basename (the parent dir is the safe-tensor-name).
rows_by_tensor: list[dict[str, str]] = []
for row in sample_rows:
    m = {}
    for p in row:
        # parent-of-parent is .../tensors/<safe_name>/sample_NNNN.raw
        safe_t = os.path.basename(os.path.dirname(p))
        m[safe_t] = p
    rows_by_tensor.append(m)

with open(out, "w") as fo:
    for row in rows_by_tensor:
        ordered = []
        for nm in dlc_input_names:
            sn = safe(nm)
            if sn not in row:
                # Tensor name in DLC may have a leading "_" not present
                # in our captured folder name (snpe-dlc-info reports
                # "/model.9/..." which our safe() turns into
                # "_model_9_..." — the leading slash → leading "_").
                # Fall back: lookup by stripped name.
                sn = sn.lstrip("_")
                # try common variants
                cands = [k for k in row if k.lstrip("_") == sn]
                if cands:
                    ordered.append(row[cands[0]])
                    continue
                # Last-resort fallback: keep original order; print a warn.
                sys.stderr.write(f"WARN: no captured tensor matches DLC input "
                                  f"'{nm}' (safe={safe(nm)}); falling back\n")
                ordered = list(row.values())
                break
            ordered.append(row[sn])
        rel_paths = [os.path.relpath(p, REPO_ABS) for p in ordered]
        fo.write(" ".join(f"/work/{r}" for r in rel_paths) + "\n")
PY
    calib_list_rel="$(host_relpath "$docker_calib_list")"

    echo "    dlc → quantized dlc  (per-segment calib: $(basename $docker_calib_list))"
    # qairt-quantizer is the maintained Python 3 path; the older
    # snpe-dlc-quantize shipped in QAIRT 2.45's docker is a bash wrapper
    # that fails to parse correctly when invoked with a python3.10
    # prefix (its ELF backend `snpe-dlc-quant` is what actually runs).
    docker run --rm -v "$REPO_DIR":/work "$DOCKER_IMAGE" bash -c "
        /qnn/bin/x86_64-linux-clang/qairt-quantizer \\
            --input_dlc /work/$dlc_rel \\
            --output_dlc /work/$qdlc_rel \\
            --input_list /work/$calib_list_rel 2>&1 | tail -5
    " || echo "    (quantize failed for seg $seg_id — see above)"
done

# ---- Stage 3: on-board context binary generation.
# Push the sub-DLCs to the board, run qnn-context-binary-generator per
# (segment, backend), pull the .bin files, stage as
# ctx_<net>_<label>_seg<id>.bin under $CTX_DIR.
echo
echo "==> staging quantized sub-DLCs to board for context binary gen"
# docker writes the converted/quantized DLCs as root inside the container,
# so they land on the host owned by root. Fix permissions before scp so
# we don't silently lose files (scp -q swallows the EACCES).
sudo chown -R "$(id -u):$(id -g)" "$SUB_DLC_DIR/" 2>/dev/null || true
ssh "$BOARD_USER@$BOARD_IP" "rm -rf $BOARD_CTX_DIR/sub_dlc && mkdir -p $BOARD_CTX_DIR/sub_dlc"
scp "$SUB_DLC_DIR"/*.dlc \
    "$BOARD_USER@$BOARD_IP:$BOARD_CTX_DIR/sub_dlc/" 2>&1 | tail -2 || true

echo
echo "==> generating per-segment context binaries on board"
# Look up which lib to use per (network, label) — for QRB5165
# yolov8n+HTA_split → DSP, dronet+HTA_split → HTA, *+CPU → CPU.
for entry in "${segs[@]}"; do
    seg_id="$(echo "$entry" | cut -f1)"
    net="$(echo "$entry" | cut -f2)"
    label="$(echo "$entry" | cut -f3)"
    base="${net}_${label}_seg${seg_id}"   # match conversion stage's naming

    # Pick the lib based on (net, label). dronet HTA_split now routes
    # to real HTA because we sliced from dronet_bnfree.onnx — BN was
    # the blocker, the rewrite removes it. yolov8n HTA_split still
    # needs DSP (broader op coverage; HTA can't run yolov8 ops at all).
    # Mirrors generate_runtime.py's PER_NET_LIB_OVERRIDE.
    case "$net:$label" in
        dronet:HTA_split)   lib=libQnnHta.so ;;
        yolov8n:HTA_split)  lib=libQnnDsp.so ;;
        *:CPU)              lib=libQnnCpu.so ;;
        *:GPU_fp16)         lib=libQnnGpu.so ;;
        *)                  echo "    (skip $base — unknown net:label='$net:$label')" >&2; continue ;;
    esac

    out_bin="ctx_${net}_${label}_seg${seg_id}.bin"
    ssh "$BOARD_USER@$BOARD_IP" bash <<EOF
set +e   # don't abort on a single seg's compose failure — let the rest run
cd $BOARD_CTX_DIR
QNN=/root/qairt
export LD_LIBRARY_PATH=\$QNN/lib/target
export ADSP_LIBRARY_PATH="\$QNN/lib/hexagon-v66;/dsp/cdsp;/dsp"
GEN=\$QNN/bin/target/qnn-context-binary-generator
# Prefer quantized; fall back to fp DLC.
src=sub_dlc/${base}_quantized.dlc
[ -f "\$src" ] || src=sub_dlc/${base}.dlc
echo "  seg $seg_id  $base  lib=$lib  src=\$src"
out_log=$BOARD_CTX_DIR/_compose_${base}.log
"\$GEN" --backend \$QNN/lib/target/$lib \\
       --model \$QNN/lib/target/libQnnModelDlc.so \\
       --dlc_path "\$src" \\
       --binary_file ${out_bin%.bin} --output_dir . > "\$out_log" 2>&1
rc=\$?
if [ "\$rc" -ne 0 ]; then
    # Surface the actual compose error rather than just "failed":
    grep -E '\\[ ERROR \\]|incorrect Datatype|unsupported|validation' "\$out_log" | head -3
    echo "    -> compose FAILED (rc=\$rc); full log: \$out_log"
else
    sz=\$(stat -c%s ${out_bin})
    echo "    -> compose OK (\${sz} B)"
fi
EOF
done

echo
echo "==> done. seg-level context binaries on board at $BOARD_CTX_DIR/"
ssh "$BOARD_USER@$BOARD_IP" "ls -lh $BOARD_CTX_DIR/ctx_*_seg*.bin 2>/dev/null | head -20" || true

# TODO (future): once seg-level contexts exist, the runtime's g_ctx
# lookup needs a third key (seg_id). Switch generate_runtime.py to
# emit per-segment `LoadedCtx` lookup keyed by seg_id, and have the
# walker pick the right ctx by entry seg_id rather than by (network,
# kind). The walker shape, sync, and timing logic don't change — just
# the dispatch-context lookup.
