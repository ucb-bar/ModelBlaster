"""Slice smolvlm_vision.onnx into DSP-compatible segments and CPU
trampolines (single Tanh ops) for mixed DSP+CPU execution on QRB5165 v66.

The SigLIP ViT has 12 Tanh activations (one per MLP layer's GELU
approximation) that block whole-graph DSP compilation. This script
extracts:
  - 13 DSP segments: all nodes between consecutive Tanh ops
  - 12 CPU segments: single Tanh op each

Output: qnn_models/smolVLA/vision_slices/
  dsp_seg_00.onnx .. dsp_seg_12.onnx
  cpu_seg_00.onnx .. cpu_seg_11.onnx
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, numpy_helper

_HERE = Path(__file__).parent
_VISION_ONNX = _HERE / "smolvlm_vision.onnx"
_OUT_DIR = _HERE / "vision_slices"


def find_tanh_indices(graph) -> list[int]:
    """Return sorted list of node indices for Tanh ops."""
    return [i for i, n in enumerate(graph.node) if n.op_type == "Tanh"]


def compute_segment_io(graph, start: int, end: int, init_names: set[str]):
    """Compute external inputs and outputs for nodes [start, end)."""
    seg_produced = set()
    for i in range(start, end):
        for out in graph.node[i].output:
            seg_produced.add(out)

    external_inputs = []
    seen_inputs = set()
    for i in range(start, end):
        for inp in graph.node[i].input:
            if inp in init_names or inp in seg_produced or inp in seen_inputs:
                continue
            if inp == "":
                continue
            external_inputs.append(inp)
            seen_inputs.add(inp)

    graph_output_names = {out.name for out in graph.output}
    seg_node_indices = set(range(start, end))

    external_outputs = []
    seen_outputs = set()
    for i in range(start, end):
        for out in graph.node[i].output:
            if out in seen_outputs:
                continue
            is_external = out in graph_output_names
            if not is_external:
                for j, node in enumerate(graph.node):
                    if j in seg_node_indices:
                        continue
                    if out in node.input:
                        is_external = True
                        break
            if is_external:
                external_outputs.append(out)
                seen_outputs.add(out)

    return external_inputs, external_outputs


def slice_model(src_path: str, out_path: str, input_names: list[str],
                output_names: list[str]):
    """Extract a sub-graph using onnx.utils.extract_model."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    onnx.utils.extract_model(str(src_path), str(out_path),
                             input_names, output_names)
    model = onnx.load(str(out_path))
    onnx.checker.check_model(model, full_check=False)
    return model


def main():
    print(f"Loading {_VISION_ONNX} ...")
    model = onnx.load(str(_VISION_ONNX))
    graph = model.graph
    print(f"  {len(graph.node)} nodes, {len(graph.initializer)} initializers")

    init_names = {init.name for init in graph.initializer}

    tanh_indices = find_tanh_indices(graph)
    print(f"  Found {len(tanh_indices)} Tanh ops at indices: {tanh_indices}")
    assert len(tanh_indices) == 12, f"Expected 12 Tanh ops, got {len(tanh_indices)}"

    if _OUT_DIR.exists():
        shutil.rmtree(_OUT_DIR)
    _OUT_DIR.mkdir(parents=True)

    # DSP segments: between consecutive Tanh boundaries
    # seg 0: [0, tanh[0])
    # seg k: [tanh[k-1]+1, tanh[k])  for k in 1..11
    # seg 12: [tanh[11]+1, end)
    boundaries = [-1] + tanh_indices + [len(graph.node)]
    total_dsp_nodes = 0
    total_cpu_nodes = 0

    print("\n=== Extracting 13 DSP segments ===")
    for seg_idx in range(13):
        start = boundaries[seg_idx] + 1
        end = boundaries[seg_idx + 1]
        n_nodes = end - start
        total_dsp_nodes += n_nodes

        inputs, outputs = compute_segment_io(graph, start, end, init_names)
        out_path = _OUT_DIR / f"dsp_seg_{seg_idx:02d}.onnx"

        print(f"  dsp_seg_{seg_idx:02d}: nodes [{start}, {end}) = {n_nodes} nodes, "
              f"{len(inputs)} inputs, {len(outputs)} outputs")

        try:
            sub_model = slice_model(_VISION_ONNX, str(out_path), inputs, outputs)
            print(f"    -> {out_path.name}: {len(sub_model.graph.node)} nodes, OK")
        except Exception as e:
            print(f"    -> FAILED: {e}")
            sys.exit(1)

    print("\n=== Extracting 12 CPU segments (single Tanh each) ===")
    for cpu_idx, tanh_idx in enumerate(tanh_indices):
        start = tanh_idx
        end = tanh_idx + 1
        total_cpu_nodes += 1

        inputs, outputs = compute_segment_io(graph, start, end, init_names)
        out_path = _OUT_DIR / f"cpu_seg_{cpu_idx:02d}.onnx"

        print(f"  cpu_seg_{cpu_idx:02d}: Tanh @ node {tanh_idx}, "
              f"input={inputs}, output={outputs}")

        try:
            sub_model = slice_model(_VISION_ONNX, str(out_path), inputs, outputs)
            print(f"    -> {out_path.name}: {len(sub_model.graph.node)} nodes, OK")
        except Exception as e:
            print(f"    -> FAILED: {e}")
            sys.exit(1)

    print(f"\n=== Summary ===")
    print(f"  Total DSP nodes: {total_dsp_nodes}")
    print(f"  Total CPU nodes: {total_cpu_nodes}")
    print(f"  Total: {total_dsp_nodes + total_cpu_nodes} (original: {len(graph.node)})")
    assert total_dsp_nodes + total_cpu_nodes == len(graph.node), \
        "Node count mismatch — ops were dropped or duplicated!"
    print(f"  All 25 sub-ONNXes extracted and validated successfully.")
    print(f"  Output directory: {_OUT_DIR}")


if __name__ == "__main__":
    main()
