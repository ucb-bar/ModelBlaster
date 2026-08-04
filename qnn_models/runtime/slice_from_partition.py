"""Slice ONNX into per-segment sub-ONNXes from a partition annotation file.

Companion to slice_to_subonnx.py. Where that tool drives slicing from the
scheduler's per-op routing (which assumes per-op dispatch granularity is
free), this one drives slicing from a curated compose-feasibility
partition file (e.g. boards/qrb5165_v66/graphs/yolov8n_HTA_split.json)
where each op carries a `hardware_target` and the partition guarantees
each contiguous run lands on a backend that can actually compose it.

For yolov8n on QRB5165 the partition is:
  - 103-op backbone (HTA)
  - 146-op head    (DSP)
  - 3 handoff tensors at model_4/6/9 outputs

This is the right granularity for the QNN runtime: each sub-DLC can be
compiled by snpe-onnx-to-dlc + qairt-quantizer + qnn-context-binary-
generator on its target backend without per-op-routing artifacts.

Output is structurally the same as slice_to_subonnx.py's manifest.json,
so build_subdlcs.sh + capture_boundary_calibration.py + the runtime
walker consume it unchanged.

CLI:
    python3 slice_from_partition.py \\
        --partition qnn_models/boards/qrb5165_v66/graphs/yolov8n_HTA_split.json \\
        --network yolov8n \\
        --onnx qnn_models/yolov8n.onnx \\
        --label-for-target HTA=HTA_split,DSP=HTA_split \\
        --kind-for-target HTA=CPU_P,DSP=CPU_P \\
        --base-seg-id 100 \\
        --out-dir qnn_models/runtime/gen/qrb5165_dronet_yolov8/sub_onnx

`--label-for-target` and `--kind-for-target` map the partition's raw
hardware_target values onto the schedule-side label/kind taxonomy so the
emitted manifest entries align with what build_subdlcs.sh + the runtime
expect. For yolov8n we collapse both HTA and DSP onto the single
"HTA_split" label (the runtime's PER_NET_LIB_OVERRIDE for
yolov8n+HTA_split routes to libQnnDsp.so anyway, and HTA-only ops would
require a 3-way label split which the current toplevel doesn't model).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict


def _safe_name(t: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", t)


def _parse_kv_map(spec: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for kv in spec.split(","):
        if not kv.strip(): continue
        k, _, v = kv.strip().partition("=")
        if not k or not v: raise SystemExit(f"bad kv pair '{kv}'")
        out[k.strip()] = v.strip()
    return out


def _segments_from_partition(partition: dict) -> list[dict]:
    """Walk the partition's ops list and group consecutive runs of the
    same `hardware_target` into segments. Each segment carries the list
    of ONNX node names (via `_orig_name`) it covers, plus the original
    op-list-index range used as the seg's `op_ids`.

    The partition file's order *is* the network's topological order
    (it's emitted by the same export pipeline that wrote the per-op CSV
    used to drive scheduling), so contiguous ranges are well-defined.
    """
    segments: list[dict] = []
    current_target = None
    current: dict | None = None
    for idx, op in enumerate(partition["ops"]):
        tgt = op.get("hardware_target")
        if tgt is None:
            raise SystemExit(f"op[{idx}] {op.get('name')} has no hardware_target")
        if tgt != current_target:
            if current is not None:
                segments.append(current)
            current = {
                "hardware_target":  tgt,
                "op_indices":       [],
                "orig_names":       [],
            }
            current_target = tgt
        current["op_indices"].append(idx)
        nm = op.get("_orig_name") or op.get("name")
        current["orig_names"].append(nm)
    if current is not None:
        segments.append(current)
    return segments


def _slice_one(network: str, label: str, kind: str, seg_id: int,
                seg_orig_names: list[str], seg_indices: list[int],
                onnx_path: str, out_path: str) -> dict | None:
    import onnx
    import onnx.utils
    model = onnx.load(onnx_path)
    onnx_node_names = {n.name for n in model.graph.node}
    initializers   = {init.name for init in model.graph.initializer}
    graph_outputs  = {o.name for o in model.graph.output}

    # Some partition ops are synthetic — `.nchw`-suffixed Transposes that
    # QNN's NCHW→NHWC layout pass inserts during DLC conversion, but
    # never appear in the source ONNX. Drop those: snpe-onnx-to-dlc
    # re-inserts the layout conversions automatically when it converts
    # the sub-ONNX, so the resulting sub-DLC is unaffected.
    present = [n for n in seg_orig_names if n in onnx_node_names]
    dropped = [n for n in seg_orig_names if n not in onnx_node_names]
    if dropped:
        # Heuristic: these should all be converter artifacts. If not,
        # warn loudly so we don't silently produce a wrong slice.
        synthetic_ish = [n for n in dropped if ".nchw" in n or "_sel_" in n]
        unknown       = [n for n in dropped if n not in synthetic_ish]
        if unknown:
            print(f"  seg{seg_id}: WARN — dropping {len(unknown)} non-synthetic "
                  f"missing ops (e.g. {unknown[:2]})", file=sys.stderr)
        else:
            print(f"  seg{seg_id}: dropped {len(dropped)} converter-synthetic ops "
                  f"(e.g. {dropped[:1]})", file=sys.stderr)
    if not present:
        print(f"  seg{seg_id}: SKIP — no resolvable ONNX nodes left after drop",
              file=sys.stderr)
        return None
    seg_orig_names = present

    seg_set = set(seg_orig_names)
    in_seg_outputs: set[str] = set()
    in_seg_inputs:  set[str] = set()
    out_seg_consumers: dict[str, list[str]] = defaultdict(list)
    for n in model.graph.node:
        if n.name in seg_set:
            in_seg_inputs.update(t for t in n.input  if t)
            in_seg_outputs.update(t for t in n.output if t)
        else:
            for t in n.input:
                if t: out_seg_consumers[t].append(n.name)

    inputs  = sorted(t for t in in_seg_inputs
                       if t not in in_seg_outputs and t not in initializers)
    outputs = sorted(t for t in in_seg_outputs
                       if t in graph_outputs or out_seg_consumers.get(t))

    if not inputs or not outputs:
        print(f"  seg{seg_id}: SKIP — empty boundary "
              f"(in={len(inputs)}, out={len(outputs)})", file=sys.stderr)
        return None

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    onnx.utils.extract_model(onnx_path, out_path, inputs, outputs)
    sub = onnx.load(out_path)
    print(f"  seg{seg_id}  net={network:<10s}  label={label:<10s}  "
          f"ops={len(seg_orig_names):>3d}  in={len(inputs):>2d} out={len(outputs):>2d}  "
          f"sub_nodes={len(sub.graph.node):>3d}  size={os.path.getsize(out_path)//1024} KB")
    return {
        "seg_id":         seg_id,
        "network":        network,
        "label":          label,
        "kind":           kind,
        "instance":       0,
        "op_ids":         seg_indices,
        "input_tensors":  inputs,
        "output_tensors": outputs,
        "sub_onnx_path":  out_path,
        "source_onnx":    onnx_path,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partition", required=True,
                    help="path to the partition annotation JSON (yolov8n_HTA_split.json shape)")
    ap.add_argument("--network",   required=True)
    ap.add_argument("--onnx",      required=True)
    ap.add_argument("--label-for-target", required=True,
                    help="comma-sep <hw_target>=<schedule_label> "
                         "(e.g. HTA=HTA_split,DSP=HTA_split)")
    ap.add_argument("--kind-for-target",  required=True,
                    help="comma-sep <hw_target>=<schedule_kind> "
                         "(e.g. HTA=CPU_P,DSP=CPU_P)")
    ap.add_argument("--base-seg-id", type=int, default=100,
                    help="first seg_id to assign — start at 100 to avoid "
                         "colliding with a coexisting schedule-driven manifest")
    ap.add_argument("--out-dir",    required=True,
                    help="emits <network>_<label>_part<N>.onnx + a manifest.json. "
                         "If a manifest.json already exists, MERGE the new entries "
                         "into it (so this can run after slice_to_subonnx.py without "
                         "clobbering its output for other networks).")
    args = ap.parse_args()

    label_for = _parse_kv_map(args.label_for_target)
    kind_for  = _parse_kv_map(args.kind_for_target)

    with open(args.partition) as f:
        partition = json.load(f)

    segments = _segments_from_partition(partition)
    print(f"[{args.network}] partition has {len(segments)} contiguous hw_target segments:")
    for s in segments:
        print(f"  hw={s['hardware_target']:<6s}  ops={len(s['orig_names'])}  "
              f"first={s['orig_names'][0]}  last={s['orig_names'][-1]}")

    # Slice each segment.
    new_entries: list[dict] = []
    print(f"\nslicing into {args.out_dir}/")
    for i, s in enumerate(segments):
        tgt   = s["hardware_target"]
        label = label_for.get(tgt)
        kind  = kind_for.get(tgt)
        if not label or not kind:
            raise SystemExit(f"missing --label-for-target or --kind-for-target for tgt={tgt}")
        seg_id = args.base_seg_id + i
        out_path = os.path.join(args.out_dir,
                                  f"{args.network}_{label}_part{i}.onnx")
        meta = _slice_one(args.network, label, kind, seg_id,
                           s["orig_names"], s["op_indices"],
                           args.onnx, out_path)
        if meta is not None:
            meta["partition_segment"] = i
            meta["hardware_target"]   = tgt
            new_entries.append(meta)

    # Merge into the existing manifest.json if present (preserving any
    # earlier slice_to_subonnx.py output for other networks). Replace
    # our own network's entries — re-running this tool overwrites them.
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"segments": []}
    # Drop any existing entries for this network with seg_id in our
    # base_seg_id..+N range so re-running doesn't pile up duplicates.
    base_lo = args.base_seg_id
    base_hi = args.base_seg_id + len(segments)
    keep = [
        s for s in manifest.get("segments", [])
        if not (s.get("network") == args.network and base_lo <= s.get("seg_id", -1) < base_hi)
    ]
    manifest["segments"] = keep + new_entries
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {len(new_entries)} partition slices into {manifest_path}")


if __name__ == "__main__":
    main()
