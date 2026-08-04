"""Generate dispatch_graph.json for the v3-sliced vision encoder.

The v3 slicing produces 49 segments in a linear chain:
  DSP[0] → CPU[0] → DSP[1] → CPU[1] → ... → DSP[24]

Each segment becomes one dispatch. The dispatch graph encodes the sequential
dependencies and segment metadata needed by the XPURT scheduler.

Usage:
  python gen_vision_v3_dispatch_graph.py
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).parent
_SLICES_DIR = _HERE / "vision_slices_v3"
_GEN_ROOT = _HERE.parent.parent / "gen"


def main():
    n_dsp = 25
    n_cpu = 24
    n_total = n_dsp + n_cpu  # 49

    dispatches = {}
    for i in range(n_total):
        dispatch_key = f"dispatch_{i}"
        if i % 2 == 0:
            seg_idx = i // 2
            seg_name = f"dsp_seg_{seg_idx:02d}"
            seg_type = "dsp"
        else:
            seg_idx = i // 2
            seg_name = f"cpu_seg_{seg_idx:02d}"
            seg_type = "cpu"

        deps = [f"dispatch_{i-1}"] if i > 0 else []

        dispatches[dispatch_key] = {
            "id": i,
            "ordinal": 1,
            "total": 1,
            "dependencies": deps,
            "vmfb_path": f"slices/{seg_name}.dlc",
            "segment_name": seg_name,
            "segment_type": seg_type,
        }

    graph = {
        "dot_file": "",
        "dispatch_vmfb_dir": "slices",
        "_comment": f"Fine-grained smolvlm_vision v3: {n_dsp} DSP + {n_cpu} CPU segments in a linear chain.",
        "dispatches": dispatches,
    }

    # Write to all backend slots so profile_loader can find it
    for hw in ("CPU", "DSP", "HTA"):
        out_dir = _GEN_ROOT / "qnn_vmfb" / "smolvlm_vision_v3" / "qrb5165_v66" / hw / "smolvlm_vision_v3.int8"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "smolvlm_vision_v3.int8_dispatch_graph.json"
        with open(out_path, "w") as f:
            json.dump(graph, f, indent=2)
        print(f"  {out_path}")

    print(f"\nGenerated dispatch graph: {n_total} dispatches ({n_dsp} DSP + {n_cpu} CPU)")


if __name__ == "__main__":
    main()
