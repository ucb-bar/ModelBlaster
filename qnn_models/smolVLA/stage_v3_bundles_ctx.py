"""Stage context binaries on the board for the bundle-aware v3 runtime.

The runtime generator (generate_runtime.py) expects ctx files named
   ctx_<network>_<label>_seg<dispatch_id>__<Be>.bin
where:
   network    = smolvlm_vision_v3_bundles
   label      = the dispatch's segment_name (e.g. "dsp_seg_01_conv1")
   dispatch_id = ctx_seg_id (== dispatch index in the table)
   Be         = title-case backend short name (Cpu / Dsp / Hta)

We have these source binaries on the board at /root/models/smolvlm_vision_v3/ctx/:
   ctx_dsp_seg_XX__Cpu.bin / __Dsp.bin                 (full segment, CPU/DSP)
   ctx_cpu_seg_XX__Cpu.bin                              (CPU-only segment)
   ctx_dsp_seg_XX_tramp_pY__Cpu.bin                     (trampoline phase, CPU)
   ctx_dsp_seg_XX_node_<conv_name>__Hta.bin             (single HTA conv)

This script generates a Bash snippet that creates the symlinks in
$CTX_DIR (default /root/qnn_runtime_ctx_v3) so the runtime can find each
.bin under the expected name.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent

NETWORK = "smolvlm_vision_v3_bundles"


def _be_short(label: str) -> str:
    """'CPU' → 'Cpu', 'HTA' → 'Hta', 'DSP' → 'Dsp'."""
    return label[0].upper() + label[1:].lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ctx-src", default="/root/models/smolvlm_vision_v3/ctx",
                    help="board path with the existing ctx_*.bin source files")
    ap.add_argument("--ctx-dst", default="/root/qnn_runtime_ctx_v3",
                    help="board path the runtime will look in")
    ap.add_argument("--placement-plan",
                    default=str(_HERE / "v3_placement_plan.json"))
    ap.add_argument("--segment-perf",
                    default=str(_REPO / "qnn_models/boards/qrb5165_v66"
                                "/profiles/smolvlm_vision_v3/segment_perf.json"))
    ap.add_argument("--schedule",
                    default=str(_REPO / "schedules"
                                "/scheduled_networks_smolvla_vision_v3_bundles"
                                "_qrb5165_greedy_profiled.json"))
    ap.add_argument("--out-script", default=str(_HERE / "stage_v3_bundles_ctx.sh"))
    args = ap.parse_args()

    with open(args.segment_perf) as f:
        perf = json.load(f)
    with open(args.schedule) as f:
        sched = json.load(f)

    # Build conv name lookup: for each dsp_seg_XX, the two conv-op names
    # (in order). These are in perf[seg]['Hta']['convs'].
    seg_to_convs: dict[str, list[str]] = {}
    for seg, data in perf.items():
        if not seg.startswith("dsp_seg_"):
            continue
        convs = data.get("Hta", {}).get("convs", [])
        seg_to_convs[seg] = [c["name"] for c in convs]

    # Walk the schedule's dispatches and build the (target_name, source_name)
    # pairs we need to symlink.
    pairs: list[tuple[str, str]] = []
    for did_str, d in sched["dispatches"].items():
        did = d["id"]
        seg_name = d["module_name"]    # e.g. "dsp_seg_01_conv1"
        hw = d["hardware_target"].split("#")[0]  # CPU_P / CPU_E / CPU_X
        # The runtime emits actual_backend in the dispatch table by matching
        # the slot to its backend lib. The bundle table maps:
        #   CPU_P → HTA / Hta
        #   CPU_E → DSP / Dsp
        #   CPU_X → CPU / Cpu
        slot_to_be = {"CPU_P": "Hta", "CPU_E": "Dsp", "CPU_X": "Cpu"}
        be = slot_to_be[hw]

        # ctx_seg_id is parsed from the module_name's trailing _seg<digits>$.
        # None of our segment names match that pattern (e.g. "dsp_seg_01" has
        # two underscores between "seg" and the digits; "dsp_seg_01_conv1" /
        # "dsp_seg_01_tramp_p0" end in non-numeric tokens). So the runtime
        # falls back to ctx_seg_id=0 for ALL our dispatches, and looks for
        # ctx_<net>_<label>_seg0__<Be>.bin uniformly.
        target = f"ctx_{NETWORK}_{seg_name}_seg0__{be}.bin"

        # Figure out the source name on disk.
        if seg_name.startswith("cpu_seg_"):
            # CPU-only segment: ctx_<seg_name>__Cpu.bin (only Cpu exists)
            source = f"ctx_{seg_name}__Cpu.bin"
        elif seg_name.startswith("dsp_seg_") and "_tramp_p" in seg_name:
            # Trampoline phase. The backend was chosen by the scheduler:
            # CPU_X → use CPU-quantized DLC; CPU_E → use DSP-quantized DLC.
            # On-board names: ctx_<seg_name>__Cpu.bin or ctx_<seg_name>__Dsp.bin.
            source = f"ctx_{seg_name}__{be}.bin"
        elif seg_name.endswith("_conv1") or seg_name.endswith("_conv2"):
            # HTA conv: ctx_<dsp_seg_XX>_<conv_name>__Hta.bin
            parent_seg = seg_name.rsplit("_", 1)[0]   # strip "_conv1" / "_conv2"
            convs = seg_to_convs.get(parent_seg, [])
            conv_idx = 0 if seg_name.endswith("_conv1") else 1
            if conv_idx >= len(convs):
                print(f"  WARN: {parent_seg} has only {len(convs)} convs, can't map {seg_name}")
                continue
            conv_op_name = convs[conv_idx]   # e.g. "dsp_seg_01_node_MatMul_549_conv1x1"
            source = f"ctx_{conv_op_name}__Hta.bin"
        elif seg_name.startswith("dsp_seg_") and "_tramp_" not in seg_name:
            # Mono dsp segment: ctx_<seg_name>__<Cpu|Dsp>.bin
            source = f"ctx_{seg_name}__{be}.bin"
        else:
            print(f"  WARN: unrecognized seg_name '{seg_name}' (dispatch {did}); skipping")
            continue
        pairs.append((target, source))

    # Emit a Bash script that creates the symlinks.
    lines = []
    lines.append("#!/usr/bin/env bash")
    lines.append("# Auto-generated by stage_v3_bundles_ctx.py — do not edit.")
    lines.append("# Creates symlinks in CTX_DST pointing at source ctx binaries in CTX_SRC.")
    lines.append("set -euo pipefail")
    lines.append(f'CTX_SRC="${{CTX_SRC:-{args.ctx_src}}}"')
    lines.append(f'CTX_DST="${{CTX_DST:-{args.ctx_dst}}}"')
    lines.append('mkdir -p "$CTX_DST"')
    lines.append('n_ok=0; n_missing=0')
    for tgt, src in pairs:
        lines.append(f'if [ -f "$CTX_SRC/{src}" ]; then')
        lines.append(f'  ln -sf "$CTX_SRC/{src}" "$CTX_DST/{tgt}"; n_ok=$((n_ok+1))')
        lines.append(f'else')
        lines.append(f'  echo "MISSING: {src}"; n_missing=$((n_missing+1))')
        lines.append(f'fi')
    lines.append('echo "Staged $n_ok ctx symlinks; $n_missing missing source files."')
    with open(args.out_script, "w") as f:
        f.write("\n".join(lines) + "\n")
    Path(args.out_script).chmod(0o755)

    print(f"Wrote {args.out_script}")
    print(f"  total symlink pairs: {len(pairs)}")
    print(f"  sample mappings:")
    for tgt, src in pairs[:6]:
        print(f"    {src}  →  {tgt}")


if __name__ == "__main__":
    main()
