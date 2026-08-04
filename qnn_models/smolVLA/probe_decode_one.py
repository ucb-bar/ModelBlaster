"""Probe one decode segment: build DLC for CPU and DSP, profile on board.

A fast feasibility check before committing to the full decode pipeline.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx


HERE = Path(__file__).parent
PYTHON = "<PATH_TO_CONDA_ENV>/bin/python"
QNN_SDK = "<PATH_TO_QAIRT_SDK>"
DOCKER_IMAGE = "qnn-convert"
BOARD = "root@<BOARD_IP>"
REMOTE_BASE = "/root/decode_probe"


def gen_calibration(onnx_path: Path, cal_dir: Path, seg_name: str, n_samples: int = 8) -> Path:
    cal_dir.mkdir(parents=True, exist_ok=True)
    m = onnx.load(str(onnx_path))
    rng = np.random.default_rng(hash(seg_name) % 2**32)
    list_path = cal_dir / f"{seg_name}_cal_list.txt"
    with open(list_path, "w") as f:
        for i in range(n_samples):
            tokens = []
            for inp in m.graph.input:
                dims = [d.dim_value if d.dim_value > 0 else 1
                        for d in inp.type.tensor_type.shape.dim]
                elem = inp.type.tensor_type.elem_type
                if elem == onnx.TensorProto.FLOAT:
                    data = rng.standard_normal(size=dims).astype(np.float32) * 0.3
                elif elem == onnx.TensorProto.INT32:
                    data = rng.integers(0, 50, size=dims).astype(np.int32)
                elif elem == onnx.TensorProto.INT64:
                    data = rng.integers(0, 50, size=dims).astype(np.int64)
                else:
                    data = rng.standard_normal(size=dims).astype(np.float32) * 0.3
                safe = inp.name.replace("/", "_").replace(".", "_").replace(":", "_")
                raw = cal_dir / f"{seg_name}_s{i:02d}_{safe}.raw"
                data.tofile(str(raw))
                tokens.append(f"{inp.name}:={raw.absolute()}")
            f.write(" ".join(tokens) + "\n")
    return list_path


def build_input_flags(onnx_path: Path) -> str:
    m = onnx.load(str(onnx_path))
    flags = []
    for inp in m.graph.input:
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        flags.append(f"-d {inp.name} {','.join(str(d) for d in dims)}")
        if len(dims) == 4:
            flags.append(f"--input_layout {inp.name} NCHW")
        else:
            flags.append(f"--input_layout {inp.name} NONTRIVIAL")
    return " ".join(flags)


def build_dlc_via_docker(onnx_path: Path, dlc_dir: Path, cal_dir: Path,
                           seg_name: str, work_dir: Path):
    dlc_dir.mkdir(parents=True, exist_ok=True)
    # Copy ONNX into work_dir so docker can see it
    target_onnx = work_dir / f"{seg_name}.onnx"
    if not target_onnx.exists() or target_onnx.stat().st_mtime < onnx_path.stat().st_mtime:
        import shutil
        shutil.copy2(str(onnx_path), str(target_onnx))

    cal_list = cal_dir / f"{seg_name}_cal_list.txt"
    cal_list_docker = cal_dir / f"{seg_name}_cal_list_docker.txt"
    # Rewrite cal list paths
    text = cal_list.read_text().replace(str(cal_dir), "/cal")
    cal_list_docker.write_text(text)

    input_flags = build_input_flags(onnx_path)

    cmd = [
        "sudo", "docker", "run", "--rm",
        "-v", f"{QNN_SDK}:/qnn:ro",
        "-v", f"{work_dir}:/workspace",
        "-v", f"{cal_dir}:/cal",
        DOCKER_IMAGE, "bash", "-c",
        f"pip install -q 'numpy<2' && "
        f"python3.10 /qnn/bin/x86_64-linux-clang/snpe-onnx-to-dlc "
        f"  --input_network /workspace/{seg_name}.onnx {input_flags} "
        f"  --output_path /workspace/dlc/{seg_name}.dlc && "
        f"python3.10 /qnn/bin/x86_64-linux-clang/qairt-quantizer "
        f"  --input_dlc /workspace/dlc/{seg_name}.dlc "
        f"  --output_dlc /workspace/dlc/{seg_name}_q.dlc "
        f"  --input_list /cal/{seg_name}_cal_list_docker.txt "
        f"  --act_bitwidth 8 --weights_bitwidth 8 --bias_bitwidth 8"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        print("  Docker conversion stderr (tail):")
        print(result.stderr[-800:])
        print(result.stdout[-800:])

    # Fix ownership: use current UID, not hardcoded
    import os
    uid, gid = os.getuid(), os.getgid()
    subprocess.run(["sudo", "chown", "-R", f"{uid}:{gid}", str(dlc_dir)], check=False)
    return dlc_dir / f"{seg_name}_q.dlc"


def push_and_probe(dlc_path: Path, seg_name: str):
    subprocess.run(["ssh", BOARD, f"mkdir -p {REMOTE_BASE}"], check=True)
    subprocess.run(["scp", "-q", str(dlc_path), f"{BOARD}:{REMOTE_BASE}/"], check=True)

    bash = f"""
set +e
cd {REMOTE_BASE}
QNN=/root/qairt
export LD_LIBRARY_PATH=$QNN/lib/target
export ADSP_LIBRARY_PATH="$QNN/lib/hexagon-v66;/dsp/cdsp;/dsp"
for BE in Cpu Dsp; do
    BIN=ctx_{seg_name}__$BE.bin
    rm -f "$BIN"
    $QNN/bin/target/qnn-context-binary-generator \\
        --backend $QNN/lib/target/libQnn$BE.so \\
        --model $QNN/lib/target/libQnnModelDlc.so \\
        --dlc_path "{seg_name}_q.dlc" \\
        --binary_file "ctx_{seg_name}__$BE" --output_dir . > /tmp/_ctxgen_{seg_name}_$BE.log 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -f "$BIN" ]; then
        echo "  ctx {seg_name}/$BE: OK ($(stat -c%s $BIN) B)"
        /root/models/smolvlm_vision_v3/profile_seg "$BIN" $QNN/lib/target/libQnn$BE.so 50 2>/dev/null | grep '^{{'
    else
        echo "  ctx {seg_name}/$BE: FAIL"
        head -8 /tmp/_ctxgen_{seg_name}_$BE.log
    fi
done
"""
    subprocess.run(["ssh", BOARD, bash], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default="dsp_seg_00")
    ap.add_argument("--use-conv1x1", action="store_true", default=True,
                    help="use the conv1x1-rewritten variant")
    args = ap.parse_args()

    src = HERE / "decode_slices_v1" / ("conv1x1" if args.use_conv1x1 else ".") / f"{args.seg}.onnx"
    if not src.exists():
        print(f"missing {src}"); sys.exit(1)
    work = HERE / "decode_slices_v1" / "probe"
    cal = work / "cal"
    dlc = work / "dlc"
    print(f"--- {args.seg} ---")
    print(f"  ONNX: {src} ({src.stat().st_size:,} B)")
    gen_calibration(src, cal, args.seg)
    out_dlc = build_dlc_via_docker(src, dlc, cal, args.seg, work)
    if not out_dlc.exists():
        print(f"  ERROR: DLC build failed")
        sys.exit(2)
    print(f"  DLC: {out_dlc} ({out_dlc.stat().st_size:,} B)")
    push_and_probe(out_dlc, args.seg)


if __name__ == "__main__":
    main()
