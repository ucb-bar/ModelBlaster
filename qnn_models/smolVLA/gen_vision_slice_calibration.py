"""Generate realistic calibration data for vision slice quantization.

Runs the original smolvlm_vision.onnx through onnxruntime, captures
intermediate activations at all boundary tensors between segments, and
saves them as .raw files suitable for qairt-quantizer's --input_list.

This gives the quantizer realistic activation distributions instead of
random noise, which is critical for int8 quality on the DSP segments.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def get_all_boundary_tensors(slices_dir: Path) -> set[str]:
    """Collect all tensor names that appear as inputs to any segment."""
    names = set()
    for onnx_file in sorted(slices_dir.glob("dsp_seg_*.onnx")):
        model = onnx.load(str(onnx_file))
        for inp in model.graph.input:
            names.add(inp.name)
    return names


def get_segment_inputs(slices_dir: Path) -> dict[str, list[str]]:
    """Map segment name → list of input tensor names."""
    result = {}
    for onnx_file in sorted(slices_dir.glob("dsp_seg_*.onnx")):
        model = onnx.load(str(onnx_file))
        result[onnx_file.stem] = [inp.name for inp in model.graph.input]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="Path to smolvlm_vision.onnx")
    ap.add_argument("--slices-dir", required=True, help="Directory with sub-ONNXes")
    ap.add_argument("--out-dir", required=True, help="Output directory for .raw files")
    ap.add_argument("--num-samples", type=int, default=10)
    args = ap.parse_args()

    slices_dir = Path(args.slices_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    boundary_tensors = get_all_boundary_tensors(slices_dir)
    seg_inputs = get_segment_inputs(slices_dir)

    print(f"Boundary tensors to capture: {len(boundary_tensors)}")
    print(f"Segments: {len(seg_inputs)}")

    # Load original model and add boundary tensors as outputs so we can
    # capture them in a single inference pass.
    print(f"Loading {args.src}...")
    model = onnx.load(args.src)
    graph = model.graph

    # Existing outputs
    existing_outputs = {out.name for out in graph.output}

    # Find value_info for boundary tensors and add as graph outputs
    vi_map = {vi.name: vi for vi in graph.value_info}
    input_map = {inp.name: inp for inp in graph.input}

    added = 0
    for name in boundary_tensors:
        if name in existing_outputs or name in input_map:
            continue
        if name in vi_map:
            graph.output.append(vi_map[name])
            added += 1
        else:
            print(f"  WARNING: {name} not in value_info — skipping")

    print(f"  Added {added} intermediate outputs for capture")

    # Save modified model to temp file
    tmp_model_path = str(out_dir / "_vision_with_intermediates.onnx")
    onnx.save(model, tmp_model_path)

    # Create session
    print("Creating onnxruntime session...")
    sess = ort.InferenceSession(tmp_model_path, providers=["CPUExecutionProvider"])
    output_names = [out.name for out in sess.get_outputs()]

    # Generate calibration samples
    np.random.seed(42)
    print(f"Running {args.num_samples} inference passes for calibration...")
    for sample_idx in range(args.num_samples):
        # Use varied but bounded inputs (simulating normalized RGB images)
        image = np.random.randn(1, 3, 512, 512).astype(np.float32) * 0.5
        results = sess.run(output_names, {"image": image})
        result_map = dict(zip(output_names, results))

        # Also store the graph input itself
        result_map["image"] = image

        # Save each segment's input tensors as .raw files
        for seg_name, input_names in seg_inputs.items():
            for tensor_name in input_names:
                if tensor_name in result_map:
                    data = result_map[tensor_name]
                elif tensor_name == "image":
                    data = image
                else:
                    print(f"  WARNING: {tensor_name} not captured for {seg_name}")
                    continue

                raw_path = out_dir / f"{seg_name}_{tensor_name}_{sample_idx}.raw"
                data.astype(np.float32).tofile(str(raw_path))

        if (sample_idx + 1) % 5 == 0:
            print(f"  {sample_idx + 1}/{args.num_samples} done")

    # Clean up temp model
    os.remove(tmp_model_path)

    # Write calibration lists per segment
    for seg_name, input_names in seg_inputs.items():
        cal_list_path = out_dir / f"{seg_name}_cal_list.txt"
        lines = []
        for i in range(args.num_samples):
            parts = []
            for name in input_names:
                raw_path = f"/workspace/calibration/{seg_name}_{name}_{i}.raw"
                parts.append(f"{name}:={raw_path}")
            lines.append(" ".join(parts))
        cal_list_path.write_text("\n".join(lines) + "\n")

    print(f"\nCalibration data saved to {out_dir}/")
    print(f"  {args.num_samples} samples × {len(seg_inputs)} segments")
    total_raw = len(list(out_dir.glob("*.raw")))
    print(f"  Total .raw files: {total_raw}")


if __name__ == "__main__":
    main()
