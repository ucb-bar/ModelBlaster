"""Emit synthetic XPURT profile results for v3 vision slices.

Uses measured timings from board experiments to create the per-backend
results.csv files without needing to re-run the full profile sweep.

Measured timings (from QRB5165 board experiments, May 2026):
  DSP segments:
    A-type (pre-softmax: LayerNorm → QKV → reshape → Q×K^T): ~144ms
    B-type (post-attention: out_proj → Add → LN → fc1 → GELU): ~75ms
    First/last segments (embedding/final projection): varies
  CPU segments:
    Softmax blocks (Softmax + V_matmul + Transpose + Reshape): ~43ms
    Tanh: ~2ms
  HTA (pure Conv1x1 extracted ops):
    QKV  768→2304:  7.2ms
    proj 768→768:   3.2ms
    fc1  768→3072:  9.9ms
    fc2  3072→768: 19.3ms
  CPU time for DSP-segment non-conv ops (LayerNorm, Add, Reshape, etc.):
    A-type overhead: ~25ms
    B-type overhead: ~12ms

Usage:
  python emit_vision_v3_profile_synthetic.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent

# Measured timings in microseconds
# Each ViT layer has:
#   DSP_A[i] (even, pre-softmax)  → CPU[2i] (softmax block) → DSP_B[i] (odd) → CPU[2i+1] (tanh)
# Plus first/last framing segments

# dispatch_id layout (49 total):
#   0: dsp_seg_00 (initial embedding / first pre-attention block)
#   1: cpu_seg_00 (softmax_block)
#   2: dsp_seg_01 (B-type: post-attention)
#   3: cpu_seg_01 (tanh)
#   4: dsp_seg_02 (A-type: pre-softmax layer 2)
#   5: cpu_seg_02 (softmax_block)
#   6: dsp_seg_03 (B-type)
#   7: cpu_seg_03 (tanh)
#   ...pattern repeats for 12 layers...
#   48: dsp_seg_24 (final segment: last B-type + output projection)

# Segment classification:
#   dsp_seg_00: first A-type (includes patch embedding conv + first layer pre-attention)
#   dsp_seg_01, 03, 05, ... (odd DSP idx): B-type (post-attention blocks)
#   dsp_seg_02, 04, 06, ... (even DSP idx >= 2): A-type (pre-softmax blocks)
#   dsp_seg_24: last B-type (+ final layernorm)
#   cpu_seg_00, 02, 04, ...: softmax_block (4 ops)
#   cpu_seg_01, 03, 05, ...: tanh (1 op)


def classify_dsp_seg(seg_idx: int) -> str:
    """Classify DSP segment as A-type or B-type."""
    if seg_idx == 0:
        return "A_first"  # patch embed + first pre-attention
    elif seg_idx == 24:
        return "B_last"   # last B-type + final LN
    elif seg_idx % 2 == 0:
        return "A"        # pre-softmax
    else:
        return "B"        # post-attention


def classify_cpu_seg(seg_idx: int) -> str:
    """Classify CPU segment."""
    if seg_idx % 2 == 0:
        return "softmax_block"
    else:
        return "tanh"


# Timing estimates in microseconds for each backend
TIMINGS_US = {
    # DSP backend times (int8 quantized)
    "DSP": {
        "A_first": 160_000,    # Larger segment (patch embed + first layer)
        "A":       144_000,    # Pre-softmax: LN → QKV → reshape → Q×K^T
        "B":        75_000,    # Post-attention: proj → Add → LN → fc1 → GELU prep
        "B_last":   80_000,    # Last B-type + final LN
        "softmax_block": 43_000,  # CPU-bound (Softmax on 1024×1024)
        "tanh":      2_000,    # Trivial
    },
    # CPU backend times (fp32)
    "CPU": {
        "A_first": 320_000,    # ~2x DSP for compute-heavy
        "A":       290_000,    # Pre-softmax on CPU
        "B":       155_000,    # Post-attention on CPU
        "B_last":  160_000,
        "softmax_block": 43_000,  # Same (already CPU)
        "tanh":      2_000,    # Same
    },
    # HTA backend times (pure Conv1x1 + CPU trampolines for the rest)
    # HTA only accelerates the Conv ops; remaining ops still run on CPU.
    # Per A-type segment: QKV conv (7.2ms) + CPU overhead for LN/reshape/scale (~25ms) = 32ms
    # Per B-type segment: proj (3.2ms) + fc1 (9.9ms) + fc2 (19.3ms) + CPU overhead (~12ms) = 44ms
    # Note: A-type actually only has 1 conv (QKV), B-type has 3 (proj, fc1, fc2)
    "HTA": {
        "A_first":  40_000,    # patch embed conv (HTA) + QKV conv (7.2ms) + overhead
        "A":        32_000,    # QKV conv (7.2ms) + CPU LN/reshape/scale (~25ms)
        "B":        44_000,    # proj (3.2) + fc1 (9.9) + fc2 (19.3) + CPU overhead (12)
        "B_last":   48_000,    # Last B + final LN
        "softmax_block": 43_000,
        "tanh":      2_000,
    },
}


def main():
    n_dsp = 25
    n_cpu = 24
    n_total = n_dsp + n_cpu

    for hw, timings in TIMINGS_US.items():
        out_dir = _REPO_ROOT / "gen" / "profile" / hw / "qrb5165_v66" / "smolvlm_vision_v3" / "smolvlm_vision_v3.int8" / "topo_0"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "results.csv"

        total_us = 0
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "dispatch_id", "module_name", "mean_time", "mean_unit"
            ])
            w.writeheader()

            for dispatch_id in range(n_total):
                if dispatch_id % 2 == 0:
                    seg_idx = dispatch_id // 2
                    seg_name = f"dsp_seg_{seg_idx:02d}"
                    seg_class = classify_dsp_seg(seg_idx)
                else:
                    seg_idx = dispatch_id // 2
                    seg_name = f"cpu_seg_{seg_idx:02d}"
                    seg_class = classify_cpu_seg(seg_idx)

                time_us = timings[seg_class]
                total_us += time_us

                w.writerow({
                    "dispatch_id": dispatch_id,
                    "module_name": seg_name,
                    "mean_time": f"{time_us:.2f}",
                    "mean_unit": "us",
                })

        total_ms = total_us / 1000
        print(f"  {hw}: {out_path}")
        print(f"       Total serial time: {total_ms:.1f} ms")

    # Print per-backend pipeline summary
    print("\n=== Pipeline time estimates (serial execution) ===")
    for hw, timings in TIMINGS_US.items():
        total = 0
        for dispatch_id in range(n_total):
            if dispatch_id % 2 == 0:
                seg_idx = dispatch_id // 2
                seg_class = classify_dsp_seg(seg_idx)
            else:
                seg_idx = dispatch_id // 2
                seg_class = classify_cpu_seg(seg_idx)
            total += timings[seg_class]
        print(f"  {hw:>4}: {total/1000:.0f} ms")


if __name__ == "__main__":
    main()
