"""Shared ONNX-slicing mechanics extracted from slice_vision_v3.py and
slice_decode_v1.py. Each per-model slicer still names the *cut-points*
itself (since they're intrinsically model-specific — finding Softmax
boundaries in a ViT is different from finding ScatterND clusters in a
decoder), but the surrounding machinery (compute segment I/O, run
onnx.utils.extract_model, write per-segment files) is shared here.

Typical usage from a per-model slicer:

    from onnx_slice_lib import (
        compute_segment_io, slice_model,
        ranges_complement, write_segments,
    )

    model = onnx.load("smolvlm_vision.onnx")
    cpu_ranges = my_find_cpu_ranges(model.graph)        # (start, end, label)
    dsp_ranges = ranges_complement(cpu_ranges, len(model.graph.node))

    write_segments(model, "smolvlm_vision.onnx", out_dir,
                   {"dsp_seg": dsp_ranges, "cpu_seg": cpu_ranges})
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import onnx


def compute_segment_io(graph, start: int, end: int, init_names: set[str]
                       ) -> tuple[list[str], list[str]]:
    """Compute external (input_names, output_names) for nodes [start, end).

    External inputs:  any node-input not produced inside the range and not
                      an initializer.
    External outputs: any node-output consumed outside the range OR listed
                      as a graph output.
    """
    seg_produced = set()
    for i in range(start, end):
        for out in graph.node[i].output:
            seg_produced.add(out)

    external_inputs: list[str] = []
    seen_inputs: set[str] = set()
    for i in range(start, end):
        for inp in graph.node[i].input:
            if inp in init_names or inp in seg_produced or inp in seen_inputs or inp == "":
                continue
            external_inputs.append(inp)
            seen_inputs.add(inp)

    graph_output_names = {out.name for out in graph.output}
    seg_node_indices = set(range(start, end))

    external_outputs: list[str] = []
    seen_outputs: set[str] = set()
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


def slice_model(src_path: str | os.PathLike, out_path: str | os.PathLike,
                input_names: list[str], output_names: list[str]):
    """Run onnx.utils.extract_model with input/output sanity-check."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    onnx.utils.extract_model(str(src_path), str(out_path),
                              input_names, output_names)
    sub = onnx.load(str(out_path))
    onnx.checker.check_model(sub, full_check=False)
    return sub


def ranges_complement(cpu_ranges: list[tuple[int, int, str]],
                       total_nodes: int) -> list[tuple[int, int]]:
    """Given sorted, non-overlapping (start, end, label) CPU ranges,
    return the (start, end) pairs for the DSP ranges that fill the gaps."""
    dsp_ranges: list[tuple[int, int]] = []
    prev_end = 0
    for cpu_start, cpu_end, _ in cpu_ranges:
        if cpu_start > prev_end:
            dsp_ranges.append((prev_end, cpu_start))
        prev_end = cpu_end
    if prev_end < total_nodes:
        dsp_ranges.append((prev_end, total_nodes))
    return dsp_ranges


def write_segments(model, src_path, out_dir,
                    range_groups: dict, fresh: bool = True,
                    verbose: bool = True) -> int:
    """Write per-segment ONNX files to `out_dir`.

    range_groups: {prefix: ranges} where ranges is a list of
        (start, end) or (start, end, label).
    Returns the total node count covered, for sanity-checking.
    """
    out_dir = Path(out_dir)
    if fresh and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_names = {init.name for init in model.graph.initializer}
    total = 0
    for prefix, ranges in range_groups.items():
        if verbose:
            print(f"\n=== Extracting {len(ranges)} {prefix} segments ===")
        for idx, r in enumerate(ranges):
            if len(r) == 3:
                start, end, label = r
            else:
                start, end = r
                label = ""
            n_nodes = end - start
            total += n_nodes
            inputs, outputs = compute_segment_io(model.graph, start, end, init_names)
            out_path = out_dir / f"{prefix}_{idx:02d}.onnx"
            if verbose:
                tag = f" {label}" if label else ""
                print(f"  {prefix}_{idx:02d}:{tag} [{start},{end}) = {n_nodes} nodes, "
                       f"{len(inputs)} in, {len(outputs)} out")
            try:
                slice_model(src_path, str(out_path), inputs, outputs)
            except Exception as e:
                print(f"    -> FAILED: {e}")
                sys.exit(1)
    return total


def assert_full_coverage(model, range_groups: dict):
    """Verify that the union of all ranges across groups exactly covers
    the original graph's nodes (no gaps, no overlaps)."""
    covered: list[tuple[int, int]] = []
    for ranges in range_groups.values():
        for r in ranges:
            covered.append((r[0], r[1]))
    covered.sort()
    expected = 0
    for s, e in covered:
        if s != expected:
            raise AssertionError(f"gap between {expected} and {s}")
        expected = e
    if expected != len(model.graph.node):
        raise AssertionError(
            f"covered {expected} nodes, expected {len(model.graph.node)}")
