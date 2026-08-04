"""Slice smolvlm_vision.onnx v3: cut at Tanh AND (Softmax+V_MatMul) boundaries.

The v2 approach of cutting exactly at single Softmax ops causes a 307ms layout
conversion penalty on DSP because the 4D attention tensor [1,12,1024,1024] needs
expensive requantize+transpose when entering the DSP graph.

v3 fix: include {Softmax → V_MatMul → Transpose → Reshape} as the CPU trampoline
(4 ops instead of 1). The DSP segment then starts at a 3D tensor (view_7 [1,1024,768])
which is cheap to quantize.

Structure per ViT layer:
  DSP_A: [LayerNorm → QKV split → reshape heads → Q×K^T + scale]  (pre-Softmax)
  CPU:   [Softmax → V matmul → Transpose → Reshape]               (attention core)
  DSP_B: [output proj → Add residual → LayerNorm → fc1 → GELU prep]  (post-attention)
  CPU:   [Tanh]                                                    (GELU activation)

Output: vision_slices_v3/
  dsp_seg_XX.onnx  (25 DSP segments)
  cpu_seg_XX.onnx  (24 CPU segments: 12 Softmax-blocks + 12 Tanh)
"""

from __future__ import annotations

import sys
from pathlib import Path

import onnx

from onnx_slice_lib import (
    ranges_complement,
    write_segments,
    assert_full_coverage,
)

_HERE = Path(__file__).parent
_VISION_ONNX = _HERE / "smolvlm_vision.onnx"
_OUT_DIR = _HERE / "vision_slices_v3"


def find_cpu_ranges(graph) -> list[tuple[int, int, str]]:
    """Find CPU segment ranges: (start_idx, end_idx_exclusive, label).

    For Softmax: the CPU block is [Softmax, V_MatMul, Transpose, Reshape] = 4 ops.
    For Tanh: just the single Tanh op.
    """
    ranges = []
    n_nodes = len(graph.node)

    for i, node in enumerate(graph.node):
        if node.op_type == "Softmax":
            # Softmax → MatMul(V) → Transpose → Reshape
            # Verify the expected pattern
            if i + 3 < n_nodes:
                n1 = graph.node[i + 1]
                n2 = graph.node[i + 2]
                n3 = graph.node[i + 3]
                if (n1.op_type == "MatMul" and
                    n2.op_type == "Transpose" and
                    n3.op_type == "Reshape"):
                    ranges.append((i, i + 4, "softmax_block"))
                else:
                    # Fallback: just the Softmax alone
                    print(f"  WARNING: unexpected pattern after Softmax at {i}: "
                          f"{n1.op_type}, {n2.op_type}, {n3.op_type}")
                    ranges.append((i, i + 1, "softmax"))
            else:
                ranges.append((i, i + 1, "softmax"))
        elif node.op_type == "Tanh":
            ranges.append((i, i + 1, "tanh"))

    return ranges


def main():
    print(f"Loading {_VISION_ONNX} ...")
    model = onnx.load(str(_VISION_ONNX))
    print(f"  {len(model.graph.node)} nodes")

    cpu_ranges = find_cpu_ranges(model.graph)
    softmax_blocks = sum(1 for _, _, t in cpu_ranges if t == "softmax_block")
    tanhs = sum(1 for _, _, t in cpu_ranges if t == "tanh")
    print(f"  Found {len(cpu_ranges)} CPU segments: "
           f"{softmax_blocks} softmax-blocks (4 ops each) + {tanhs} tanh")

    dsp_ranges = ranges_complement(cpu_ranges, len(model.graph.node))

    range_groups = {"dsp_seg": dsp_ranges, "cpu_seg": cpu_ranges}
    write_segments(model, _VISION_ONNX, _OUT_DIR, range_groups)
    assert_full_coverage(model, range_groups)

    n_dsp = sum(e - s for s, e in dsp_ranges)
    n_cpu = sum(e - s for s, e, _ in cpu_ranges)
    print(f"\n=== Summary ===")
    print(f"  {len(dsp_ranges)} DSP ({n_dsp} nodes) + {len(cpu_ranges)} CPU "
           f"({n_cpu} nodes) = {len(dsp_ranges) + len(cpu_ranges)} segments")
    print(f"  Output: {_OUT_DIR}")


if __name__ == "__main__":
    main()
