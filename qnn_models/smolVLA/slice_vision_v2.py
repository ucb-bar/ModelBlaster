"""Slice smolvlm_vision.onnx into DSP/HTA-compatible segments by cutting at
ALL blocker ops (Tanh + Softmax) for v66 Hexagon.

Output: qnn_models/smolVLA/vision_slices_v2/
  dsp_seg_00.onnx .. dsp_seg_24.onnx  (25 accelerator segments)
  cpu_seg_00.onnx .. cpu_seg_23.onnx  (24 CPU trampolines: 12 Softmax + 12 Tanh)

Pipeline shape per inference:
  DSP[0] → CPU[Softmax_0] → DSP[1] → CPU[Tanh_0] → DSP[2] → CPU[Softmax_1] → ...
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

_HERE = Path(__file__).parent
_VISION_ONNX = _HERE / "smolvlm_vision.onnx"
_OUT_DIR = _HERE / "vision_slices_v2"


def find_blocker_indices(graph) -> list[tuple[int, str]]:
    """Return sorted list of (node_index, op_type) for Tanh and Softmax ops."""
    blockers = []
    for i, node in enumerate(graph.node):
        if node.op_type in ("Tanh", "Softmax"):
            blockers.append((i, node.op_type))
    return blockers


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

    blockers = find_blocker_indices(graph)
    print(f"  Found {len(blockers)} blocker ops:")
    print(f"    Softmax: {sum(1 for _, t in blockers if t == 'Softmax')}")
    print(f"    Tanh:    {sum(1 for _, t in blockers if t == 'Tanh')}")

    if _OUT_DIR.exists():
        shutil.rmtree(_OUT_DIR)
    _OUT_DIR.mkdir(parents=True)

    # DSP/HTA segments: between consecutive blocker ops
    # seg 0: [0, blockers[0])
    # seg k: [blockers[k-1]+1, blockers[k])  for k=1..23
    # seg 24: [blockers[23]+1, end)
    blocker_indices = [idx for idx, _ in blockers]
    boundaries = [-1] + blocker_indices + [len(graph.node)]
    n_dsp_segs = len(boundaries) - 1
    n_cpu_segs = len(blockers)

    total_dsp_nodes = 0
    total_cpu_nodes = 0

    print(f"\n=== Extracting {n_dsp_segs} DSP/HTA segments ===")
    for seg_idx in range(n_dsp_segs):
        start = boundaries[seg_idx] + 1
        end = boundaries[seg_idx + 1]
        n_nodes = end - start
        total_dsp_nodes += n_nodes

        inputs, outputs = compute_segment_io(graph, start, end, init_names)
        out_path = _OUT_DIR / f"dsp_seg_{seg_idx:02d}.onnx"

        print(f"  dsp_seg_{seg_idx:02d}: nodes [{start}, {end}) = {n_nodes} nodes, "
              f"{len(inputs)} in, {len(outputs)} out")

        try:
            sub_model = slice_model(_VISION_ONNX, str(out_path), inputs, outputs)
            print(f"    -> {out_path.name}: {len(sub_model.graph.node)} nodes, OK")
        except Exception as e:
            print(f"    -> FAILED: {e}")
            sys.exit(1)

    print(f"\n=== Extracting {n_cpu_segs} CPU segments (single blocker op each) ===")
    for cpu_idx, (blocker_idx, op_type) in enumerate(blockers):
        start = blocker_idx
        end = blocker_idx + 1
        total_cpu_nodes += 1

        inputs, outputs = compute_segment_io(graph, start, end, init_names)
        out_path = _OUT_DIR / f"cpu_seg_{cpu_idx:02d}.onnx"

        print(f"  cpu_seg_{cpu_idx:02d}: {op_type} @ node {blocker_idx}, "
              f"input={inputs}, output={outputs}")

        try:
            sub_model = slice_model(_VISION_ONNX, str(out_path), inputs, outputs)
            print(f"    -> {out_path.name}: {len(sub_model.graph.node)} nodes, OK")
        except Exception as e:
            print(f"    -> FAILED: {e}")
            sys.exit(1)

    print(f"\n=== Summary ===")
    print(f"  Total DSP/HTA nodes: {total_dsp_nodes}")
    print(f"  Total CPU nodes:     {total_cpu_nodes}")
    print(f"  Total:               {total_dsp_nodes + total_cpu_nodes} (original: {len(graph.node)})")
    assert total_dsp_nodes + total_cpu_nodes == len(graph.node), \
        "Node count mismatch!"
    print(f"  {n_dsp_segs} DSP/HTA segments + {n_cpu_segs} CPU trampolines = {n_dsp_segs + n_cpu_segs} total")
    print(f"  Output directory: {_OUT_DIR}")


if __name__ == "__main__":
    main()
