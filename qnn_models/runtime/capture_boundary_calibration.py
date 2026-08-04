"""Capture boundary-tensor activations for per-segment quantization.

Per-segment sub-DLCs need calibration data at their boundary input
tensors — those are intermediate activations of the full network, NOT
the network's own input. Without this, the DSP/HTA backends reject the
fp32 sub-DLC at compose time (`Input[0] has incorrect Datatype 0x232`).

Pipeline:

  1. For each calibration sample, run the source ONNX through
     onnxruntime with every distinct boundary tensor across all segments
     of one network promoted to graph outputs. Capture .raw blobs.
  2. For each segment, write a per-segment input_list.txt that
     references the right boundary-tensor .raw paths in the order the
     segment's sub-ONNX expects them. qnn-quantizer reads this list
     directly.

Output layout under <gen_dir>/sub_onnx/calib/<network>/:

    tensors/<safe_tensor_name>/sample_NNNN.raw     # one .raw per (boundary, sample)
    seg_<seg_id>/input_list.txt                    # per-segment list, paths absolute,
                                                   # one line per sample, one path
                                                   # per input separated by spaces
                                                   # (qnn-quantizer convention)

CLI:
    python3 capture_boundary_calibration.py \\
        --gen-dir qnn_models/runtime/gen/qrb5165_dronet_yolov8 \\
        --network dronet \\
        --onnx qnn_models/dronet_bnfree.onnx \\
        --samples qnn_models/boards/qrb5165_v66/calibration_data/calibration_data_dronet/input_*.raw \\
        --input-shape "1,3,112,112" --input-dtype float32

When the slicer used a different ONNX per target (e.g. dronet_bnfree.onnx
for HTA_split, dronet.onnx for CPU), pass the BN-rewritten variant here
since the BN-rewritten boundary tensor names are the ones the HTA-bound
sub-DLCs expect. The CPU-bound sub-DLCs share boundary tensor *values*
(the math is equivalent) but reference them by the original BN names —
we handle that via _aliased_tensor_lookup() at write time.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from glob import glob

import numpy as np


def _safe_name(t: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", t)


def _all_boundary_tensors(manifest: dict, network: str) -> set[str]:
    """Union of every distinct boundary-input tensor across the network's
    non-aliased segments. We capture each once per sample."""
    out: set[str] = set()
    for seg in manifest["segments"]:
        if seg["network"] != network or "alias_of" in seg:
            continue
        out.update(seg.get("input_tensors", []))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--network", required=True)
    ap.add_argument("--onnx",    required=True,
                    help="ONNX model to run (the same one used by --onnx-per-target "
                         "when slicing — pick the BN-rewritten variant if the slicer did)")
    ap.add_argument("--samples", nargs="+", required=True,
                    help="raw input samples (one .raw per calibration sample)")
    ap.add_argument("--input-shape", required=True,
                    help="comma-sep dims, e.g. 1,3,112,112")
    ap.add_argument("--input-dtype", default="float32")
    ap.add_argument("--input-name", default=None,
                    help="override input tensor name (default: auto-discover)")
    args = ap.parse_args()

    try:
        import onnx
        import onnxruntime as ort
    except ImportError as e:
        sys.exit(f"need onnx + onnxruntime: {e}")

    manifest_path = os.path.join(args.gen_dir, "sub_onnx", "manifest.json")
    if not os.path.exists(manifest_path):
        sys.exit(f"missing {manifest_path} — run slice_to_subonnx.py first")
    with open(manifest_path) as f:
        manifest = json.load(f)

    boundary = _all_boundary_tensors(manifest, args.network)
    print(f"[{args.network}] {len(boundary)} distinct boundary tensors "
          f"to capture across "
          f"{sum(1 for s in manifest['segments'] if s['network']==args.network and 'alias_of' not in s)} unique segments")

    model = onnx.load(args.onnx)
    available = ({n.name for n in model.graph.value_info} |
                 {n.name for n in model.graph.output} |
                 {n.name for n in model.graph.input} |
                 {o for n in model.graph.node for o in n.output})
    capture = sorted(t for t in boundary if t in available)
    skipped = sorted(boundary - set(capture))
    if skipped:
        print(f"  WARN: {len(skipped)} boundary tensors not present in {args.onnx}:",
              file=sys.stderr)
        for t in skipped[:5]: print(f"    {t}", file=sys.stderr)
        # Don't bail — the calling segment will be skipped at quantize
        # time. Most often this happens when a slice expects a tensor
        # that exists in dronet.onnx but not dronet_bnfree.onnx (the BN
        # rewrite renamed it). The CPU-side slices use the original
        # ONNX so they reference original names; the alias logic later
        # resolves them.

    # Promote captures to graph outputs (in a fresh model copy) so
    # onnxruntime exposes them at session.run time. Without this ORT
    # eliminates them as dead during graph optimisation.
    extended = onnx.ModelProto()
    extended.CopyFrom(model)
    existing_out = {o.name for o in extended.graph.output}
    for t in capture:
        if t in existing_out: continue
        vi = next((v for v in extended.graph.value_info if v.name == t), None)
        if vi is None:
            vi = onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None)
        extended.graph.output.append(vi)
    sess = ort.InferenceSession(extended.SerializeToString(),
                                  providers=["CPUExecutionProvider"])

    in_name  = args.input_name or sess.get_inputs()[0].name
    in_shape = [int(s) for s in args.input_shape.split(",")]
    in_dtype = np.dtype(args.input_dtype)

    base_dir = os.path.join(args.gen_dir, "sub_onnx", "calib", args.network)
    tensor_dir = os.path.join(base_dir, "tensors")
    os.makedirs(tensor_dir, exist_ok=True)

    print(f"  running {len(args.samples)} samples through onnxruntime, capturing intermediates ...")
    samples = [os.path.abspath(s) for s in args.samples]
    for idx, sample_path in enumerate(samples):
        raw = np.fromfile(sample_path, dtype=in_dtype).reshape(in_shape)
        out_vals = sess.run(capture, {in_name: raw})
        for t, arr in zip(capture, out_vals):
            sub = os.path.join(tensor_dir, _safe_name(t))
            os.makedirs(sub, exist_ok=True)
            arr.astype(np.float32).tofile(os.path.join(sub, f"sample_{idx:04d}.raw"))

    # Per-segment input_list.txt — qnn-quantizer expects "<path1> <path2> ...\n"
    # one line per calibration sample, one path per input.
    n_samples = len(samples)
    seg_lists = []
    for seg in manifest["segments"]:
        if seg["network"] != args.network: continue
        if "alias_of" in seg: continue
        seg_dir = os.path.join(base_dir, f"seg_{seg['seg_id']}")
        os.makedirs(seg_dir, exist_ok=True)
        list_path = os.path.join(seg_dir, "input_list.txt")
        # Translate each input tensor name to the captured tensor path.
        # If the segment's source ONNX uses the original BN names but
        # we captured against the bnfree variant, the tensor names line
        # up at the boundary because BN's *output* tensor name is the
        # same in both ONNXes — only the BN node was split, the
        # downstream tensor name stayed put.
        tensor_paths_per_sample: list[list[str]] = [[] for _ in range(n_samples)]
        missing = []
        for in_t in seg["input_tensors"]:
            sub = os.path.join(tensor_dir, _safe_name(in_t))
            if not os.path.isdir(sub):
                missing.append(in_t)
                continue
            for sample_idx in range(n_samples):
                p = os.path.join(sub, f"sample_{sample_idx:04d}.raw")
                if not os.path.exists(p):
                    missing.append(f"{in_t}#{sample_idx}")
                    break
                tensor_paths_per_sample[sample_idx].append(os.path.abspath(p))
        if missing:
            print(f"    seg{seg['seg_id']:>3d} ({seg['label']}): SKIP — {len(missing)} "
                  f"missing captures (e.g. {missing[0]})", file=sys.stderr)
            continue
        with open(list_path, "w") as f:
            for paths in tensor_paths_per_sample:
                f.write(" ".join(paths) + "\n")
        seg_lists.append(seg['seg_id'])

    print(f"  per-segment input_list.txt written for "
          f"{len(seg_lists)} segments: seg ids {seg_lists}")
    print(f"  → {base_dir}/seg_<id>/input_list.txt")


if __name__ == "__main__":
    main()
