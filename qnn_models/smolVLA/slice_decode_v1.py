"""Slice smolvlm_expert_decode_patched3.onnx into transformer-layer segments.

Decode is a 16-layer transformer expert. Per-layer op pattern:

  S1 (DSP target):  LayerNorm-1 + Q,K,V projections + Q reshapes
                    (~13 ops; 3 heavy MatMul = Conv1x1-rewritable Linear)
  T1 (CPU only):    Rotary frequency setup (Sin/Cos in layers 0,1 only),
                    apply rotation (Split/Mul/Mul/Sub/Add), ScatterND for
                    KV cache writes (4 × per layer), Concat/Expand reshapes
                    (~30 ops; ScatterND is the dealbreaker on DSP/HTA)
  S2 (DSP target):  Attention prep + Q×K^T MatMul + scale
                    (~9 ops; the matmul has activation-on-activation so
                     can stay as raw MatMul on DSP, not Conv-rewritable)
  T2 (CPU only):    Mul/Unsqueeze/Cast/Where/Softmax (attention mask + softmax)
                    (~5 ops; Where + Softmax not supported on HTA)
  S3 (DSP target):  attn×V + output_proj + residual + LayerNorm-2 + SwiGLU FFN
                    (~20 ops; output_proj, fc_gate, fc_up, fc_down all
                     Conv1x1-rewritable; attn×V is activation×activation)

We cut at:
   - Sin / Cos prologue (only present in layers 0,1)
   - The first Cast → ScatterND cluster (start of T1)
   - The last ScatterND output (end of T1)
   - The Where → Softmax block (T2)

Output: vision_slices_decode_v1/
   dsp_seg_XX.onnx  (~3 per layer × 16 = ~48 DSP segments)
   cpu_seg_XX.onnx  (~2 per layer × 16 = ~32 CPU segments)
"""

from __future__ import annotations

from pathlib import Path

import onnx

from onnx_slice_lib import (
    ranges_complement,
    write_segments,
    compute_segment_io,
)

_HERE = Path(__file__).parent
_DECODE_ONNX = _HERE / "smolvlm_expert_decode_patched3.onnx"
_OUT_DIR = _HERE / "decode_slices_v1"


def find_cpu_ranges(graph) -> list[tuple[int, int, str]]:
    """Find CPU segment ranges (start_idx, end_idx_exclusive, label).

    Two patterns we cut at:
      (1) ROTARY+SCATTERND block per layer:
          starts at the first op after the V_proj Reshape (which precedes
          either Unsqueeze→Cast→Div→Sin→Cos rotary-freq-setup OR directly
          Split/Mul/Sub rotary-apply), ends at the last ScatterND in the
          block (4 ScatterND per layer for K/V cache writes; or sometimes
          a Concat right after that consolidates cache+new entries).
      (2) MASK+SOFTMAX block:
          Mul → Unsqueeze → Cast → Where → Softmax (5 ops).

    For (1) we identify by scanning forward for a maximal run that contains
    only "CPU-friendly" small-op types {Cast, Div, Sin, Cos, Split, Mul,
    Sub, Add, Transpose, Reshape, Unsqueeze, ScatterND, Concat, Expand,
    ReduceMin, Slice}. The run starts on a Cast/Unsqueeze/Split AFTER the
    final V-projection Reshape, ends on the last Concat or ScatterND
    before the next "heavy" op (MatMul).
    """
    nodes = graph.node
    n = len(nodes)
    ranges: list[tuple[int, int, str]] = []

    # First pass: find each Softmax → cut a 5-op block ending at it.
    softmax_blocks = []
    for i in range(n):
        if nodes[i].op_type == "Softmax":
            # Walk back to find the start: Mul or Where typically precedes.
            start = i
            for j in range(i - 1, max(-1, i - 6), -1):
                ot = nodes[j].op_type
                if ot in ("Where", "Cast", "Unsqueeze", "Mul"):
                    start = j
                else:
                    break
            softmax_blocks.append((start, i + 1, "softmax_block"))

    # Second pass: find each ScatterND cluster.
    # A "rotary+ScatterND CPU block" per layer is bounded:
    #   - on the left by the last MatMul (Q/K/V proj) preceding it,
    #     OR by the start of a Sin/Cos rotary-freq sequence.
    #   - on the right by the last ScatterND in the cluster + any
    #     immediately following Concat/Expand/Reshape/Transpose ops
    #     before the next MatMul (which would be Q×K^T).
    visited = set()
    scatter_blocks = []
    for i in range(n):
        if nodes[i].op_type != "ScatterND" or i in visited:
            continue
        # Find left boundary: walk back over "light" ops until we hit a
        # MatMul, or until we cross the last MatMul that produced V_proj.
        LIGHT = {"Cast", "Div", "Sin", "Cos", "Split", "Mul", "Sub", "Add",
                  "Transpose", "Reshape", "Unsqueeze", "ScatterND", "Concat",
                  "Expand", "ReduceMin", "Slice"}
        start = i
        for j in range(i - 1, -1, -1):
            ot = nodes[j].op_type
            if ot in LIGHT:
                start = j
            else:
                break
        # Find right boundary: walk forward until we hit a MatMul.
        end = i + 1
        for j in range(i + 1, n):
            ot = nodes[j].op_type
            if ot in LIGHT:
                end = j + 1
            else:
                break
        # Mark all ScatterND in this range visited
        for k in range(start, end):
            if nodes[k].op_type == "ScatterND":
                visited.add(k)
        scatter_blocks.append((start, end, "rotary_scatter_block"))

    ranges = scatter_blocks + softmax_blocks
    # Sort by start; merge overlaps (the softmax block sometimes falls
    # inside the broader scatter_block when both are between adjacent
    # MatMuls — softmax_blocks should take priority since they include
    # the actual Softmax/Where ops).
    ranges.sort()

    merged: list[tuple[int, int, str]] = []
    for r in ranges:
        if not merged:
            merged.append(r); continue
        last = merged[-1]
        if r[0] < last[1]:
            # Overlap: combine, keep wider span; prefer the label that
            # mentions "softmax" if present.
            combined_start = min(last[0], r[0])
            combined_end   = max(last[1], r[1])
            combined_label = "softmax_block" if "softmax" in (last[2] + r[2]) else last[2]
            merged[-1] = (combined_start, combined_end, combined_label)
        else:
            merged.append(r)

    # Split merged "rotary_scatter+softmax_block" into TWO separate ones
    # if a MatMul sits between them — that's the Q×K^T MatMul which we
    # want on DSP. We rely on the input ranges already being separate
    # in that case (left scatter ends before MatMul, softmax starts after).
    return merged


def main():
    print(f"Loading {_DECODE_ONNX} ...")
    model = onnx.load(str(_DECODE_ONNX))
    print(f"  {len(model.graph.node)} nodes")

    cpu_ranges = find_cpu_ranges(model.graph)
    sb = sum(1 for _, _, t in cpu_ranges if "softmax" in t)
    rs = sum(1 for _, _, t in cpu_ranges if "rotary" in t)
    print(f"  Found {len(cpu_ranges)} CPU segments "
           f"(softmax_block × {sb}, rotary_scatter × {rs})")

    dsp_ranges = ranges_complement(cpu_ranges, len(model.graph.node))
    print(f"  Implied {len(dsp_ranges)} DSP segments\n")

    # Filter out empty/degenerate ranges (decode can produce zero-IO sub-graphs)
    init_names = {init.name for init in model.graph.initializer}
    def keep(ranges, with_label):
        kept = []
        for r in ranges:
            s, e = r[0], r[1]
            ins, outs = compute_segment_io(model.graph, s, e, init_names)
            if not ins or not outs:
                print(f"  skip empty/degenerate [{s}..{e-1}]: in={ins} out={outs}")
                continue
            kept.append(r)
        return kept

    dsp_ranges = keep(dsp_ranges, with_label=False)
    cpu_ranges = keep(cpu_ranges, with_label=True)

    range_groups = {"dsp_seg": dsp_ranges, "cpu_seg": cpu_ranges}
    write_segments(model, _DECODE_ONNX, _OUT_DIR, range_groups)

    print()
    print(f"  Emitted: {len(dsp_ranges)} DSP + {len(cpu_ranges)} CPU = "
           f"{len(dsp_ranges) + len(cpu_ranges)} segments")
    print(f"  Output dir: {_OUT_DIR}")


if __name__ == "__main__":
    main()
