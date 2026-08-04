"""Validate the per-segment runtime's numeric output against an
onnxruntime golden.

Pipeline:
  1. Pick a real input sample (e.g. the calibration data we already have).
  2. Run the source ONNX through onnxruntime → golden output(s).
  3. Run the QNN runtime (per-segment) on the same input via
     `--input <name>=<file>` and `--output-dir <dir>`.
  4. For each (output_tensor, sample) the runtime emitted, dequantise
     to fp32 if needed and compute (max_abs_err, mean_abs_err, cosine
     similarity) against the golden.

We expect *some* divergence because the per-segment chain went through
`qairt-quantizer` (int8) and the runtime decodes outputs without applying
the reverse-quantize step end-to-end. The bar is "errors are bounded and
look like normal int8-quant noise (max ~1e-1 in the activation range,
cosine ≥ 0.99)", not bit-exact.

CLI:
    python3 validate_numeric.py \\
        --runtime-gen qnn_models/runtime/gen/qrb5165_dronet_only_validate \\
        --network dronet \\
        --onnx qnn_models/dronet.onnx \\
        --input-name input \\
        --input qnn_models/boards/qrb5165_v66/calibration_data/calibration_data_dronet/input_0.raw \\
        --input-shape 1,3,112,112 --input-dtype float32 \\
        --board-output-dir /root/qnn_runtime_outputs
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np


def _run_onnxruntime(onnx_path: str, in_name: str,
                      raw_path: str, shape: list[int], dtype: str
                      ) -> dict[str, np.ndarray]:
    import onnx
    import onnxruntime as ort
    inp = np.fromfile(raw_path, dtype=np.dtype(dtype)).reshape(shape)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(out_names, {in_name: inp})
    return {n: a for n, a in zip(out_names, outs)}


def _read_runtime_output(out_dir: str, tensor_name: str) -> np.ndarray | None:
    """The runtime mangled '/' → '_' in filenames. Match accordingly.
    Returns the raw bytes interpreted as int8 (since our sub-DLCs are
    quantized) AND as float32; caller picks whichever matches by size."""
    safe = tensor_name.replace("/", "_")
    cand = os.path.join(out_dir, safe + ".raw")
    if not os.path.exists(cand):
        # Try variants — the runtime's path-mangler keeps '/' inside
        # output_dir but escapes them later in the name.
        for f in os.listdir(out_dir):
            if f.endswith(".raw") and f.replace("_", "/").startswith(tensor_name.lstrip("/")):
                cand = os.path.join(out_dir, f)
                break
        else:
            return None
    return np.fromfile(cand, dtype=np.uint8)   # raw bytes; caller reshapes


def _dequant_int8(qbytes: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    qint = qbytes.view(np.int8).astype(np.int32)
    return (qint - zero_point).astype(np.float32) * scale


def _stats(golden: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    g = golden.astype(np.float64).ravel()
    c = candidate.astype(np.float64).ravel()
    n = min(g.size, c.size)
    g = g[:n]; c = c[:n]
    diff = np.abs(g - c)
    cos = (np.dot(g, c) / (np.linalg.norm(g) * np.linalg.norm(c) + 1e-12)
           if g.size else 0.0)
    return {
        "n":              int(n),
        "golden_min":     float(g.min()) if g.size else 0,
        "golden_max":     float(g.max()) if g.size else 0,
        "max_abs_err":    float(diff.max()) if diff.size else 0,
        "mean_abs_err":   float(diff.mean()) if diff.size else 0,
        "rel_max_err":    float(diff.max() / (abs(g).max() + 1e-12)) if g.size else 0,
        "cosine":         float(cos),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network",     required=True)
    ap.add_argument("--onnx",        required=True)
    ap.add_argument("--input-name",  required=True,
                    help="graph input tensor name in the ONNX (e.g. 'input' for dronet, 'images' for yolov8n)")
    ap.add_argument("--input",       required=True,
                    help="path to a single .raw calibration sample")
    ap.add_argument("--input-shape", required=True,
                    help="comma-sep dims, e.g. 1,3,112,112")
    ap.add_argument("--input-dtype", default="float32")
    ap.add_argument("--board-output-dir", required=True,
                    help="directory the runtime wrote --output-dir contents into "
                         "(host path after scp from board)")
    ap.add_argument("--graph-json",  default=None,
                    help="optional graph.json (output of qnn_models/export_graph_json.py) "
                         "for the dequant scale/zero_point of each output tensor — "
                         "without this, we compare raw int8 bytes vs golden float32, "
                         "which is meaningless. Most validation runs should pass this.")
    args = ap.parse_args()

    print(f"==> running onnxruntime on {args.onnx}")
    shape = [int(s) for s in args.input_shape.split(",")]
    golden = _run_onnxruntime(args.onnx, args.input_name,
                                args.input, shape, args.input_dtype)
    print(f"   golden outputs: {list(golden)}")
    for n, a in golden.items():
        print(f"     {n}: shape={a.shape} dtype={a.dtype} range=[{a.min():.4g}, {a.max():.4g}]")

    quant = {}
    if args.graph_json:
        gj = json.load(open(args.graph_json))
        for name, t in gj.get("tensors", {}).items():
            q = t.get("quant")
            if q:
                quant[name] = (q["scale"], q.get("zero_point", 0))

    print(f"\n==> reading runtime outputs from {args.board_output_dir}")
    if not os.path.isdir(args.board_output_dir):
        sys.exit(f"missing dir {args.board_output_dir}")
    files = sorted(os.listdir(args.board_output_dir))
    print(f"   {len(files)} file(s): {files[:10]}{'...' if len(files)>10 else ''}")

    print(f"\n==> per-output comparison")
    print(f"    {'tensor':<30s}  {'n':>8s}  {'g_range':>22s}  {'max_abs':>10s}  {'mean_abs':>10s}  {'rel_max':>8s}  {'cosine':>8s}")
    for gn, garr in golden.items():
        runtime_bytes = _read_runtime_output(args.board_output_dir, gn)
        if runtime_bytes is None:
            print(f"    {gn:<30s}  MISSING runtime output")
            continue
        # Decide candidate dtype + reshape. Most sub-DLCs emit int8 +
        # quant; without quant info we attempt fp32 then int8.
        cand = None
        if gn in quant or any(k.startswith(gn) for k in quant):
            # Find the matching sanitized key (graph.json sanitizes
            # e.g. "/steer" → "steer").
            key = next((k for k in quant if k == gn or k == gn.lstrip("/")), None)
            if key:
                scale, zp = quant[key]
                cand = _dequant_int8(runtime_bytes, scale, zp)
        if cand is None:
            # Fall back: try fp32 first, then int8 raw (no dequant).
            try:
                cand = runtime_bytes.view(np.float32).reshape(garr.shape)
            except Exception:
                cand = runtime_bytes.view(np.int8).astype(np.float32)
        s = _stats(garr, cand)
        print(f"    {gn:<30s}  "
              f"{s['n']:>8d}  "
              f"[{s['golden_min']:>9.4g},{s['golden_max']:>9.4g}]  "
              f"{s['max_abs_err']:>10.4g}  "
              f"{s['mean_abs_err']:>10.4g}  "
              f"{s['rel_max_err']:>8.4g}  "
              f"{s['cosine']:>8.4f}")


if __name__ == "__main__":
    main()
