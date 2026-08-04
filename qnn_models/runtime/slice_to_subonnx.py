"""Slice the source ONNX(s) into per-segment sub-ONNXes the runtime can
compile to per-backend sub-DLCs.

The runtime generator (`generate_runtime.py`) already coalesces the
scheduler's per-op routing into per-(network, instance, kind) segments
and writes the op_id list of each. To actually run those segments
through QNN's DLC pipeline we need each segment as its own ONNX
subgraph, with the boundary tensors promoted to sub-graph I/O. Once
sliced, each sub-ONNX goes through the standard:

    qnn-onnx-converter      (sub.onnx → sub.dlc)
    qnn-quantizer           (sub.dlc + calibration → sub_quantized.dlc)
    qnn-context-binary-generator  (sub_quantized.dlc + lib<X>.so → ctx.bin)

driven by `build_subdlcs.sh`.

Implementation notes:

  * The boundary ↔ ONNX node-name mapping is the only nontrivial step.
    For dronet the scheduler's `module_name` already matches the ONNX
    node name 1:1 (we see "/conv_modules.0/Conv" in both). For yolov8n
    the names were mangled by QNN's converter ("pad_0",
    "convolution_0", ...) — the user's
    boards/qrb5165_v66/graphs/yolov8n_HTA_split.json carries the
    `_orig_name → name` mapping; we consume it via --name-map.

  * `onnx.utils.extract_model` does the heavy lifting once we have the
    correct boundary tensor names. It traces backwards from each output
    to the requested inputs, materialising a valid sub-graph with all
    needed initializers (weights, biases) included.

  * Periodic-instance segments (dronet0, dronet1, ...) get the same
    sub-ONNX — only one slice per (network, kind, op_id-tuple) is
    needed because the IR is identical across instances.

CLI:
    python3 slice_to_subonnx.py \\
        --runtime-gen qnn_models/runtime/gen/qrb5165_dronet_yolov8 \\
        --onnx-map "dronet=qnn_models/dronet.onnx,yolov8n=qnn_models/yolov8n.onnx" \\
        --name-map "yolov8n=qnn_models/boards/qrb5165_v66/graphs/yolov8n_HTA_split.json" \\
        --out-dir  qnn_models/runtime/gen/qrb5165_dronet_yolov8/sub_onnx
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Iterable

import onnx
import onnx.utils


def _read_dispatch_table(runtime_gen_dir: str) -> list[dict]:
    """Re-derive segment metadata directly from dispatch_table.h.

    Could call generate_runtime's coalescer instead, but parsing the
    emitted header keeps this tool independent — anyone who already has
    a runtime gen dir can slice without re-running the generator.
    """
    path = os.path.join(runtime_gen_dir, "dispatch_table.h")
    text = open(path).read()
    # Each segment is emitted as:
    #   static const int seg<I>_ops[] = { ... };
    #   { <I>, "<network>", <inst>, "<kind>", "<label>", <n_ops>, seg<I>_ops, ... },
    seg_ops_re = re.compile(r"static const int seg(\d+)_ops\[\]\s*=\s*\{\s*([^}]*)\s*\}\s*;")
    seg_ops: dict[int, list[int]] = {}
    for m in seg_ops_re.finditer(text):
        seg_id = int(m.group(1))
        ids = [int(x) for x in m.group(2).split(",") if x.strip()]
        seg_ops[seg_id] = ids
    # Two shapes — older (no ctx_seg_id) and current (with ctx_seg_id):
    #   { <seg_id>, "<net>", <inst>, "<kind>", "<label>", <n_ops>, ...
    #   { <seg_id>, <ctx_seg_id>, "<net>", <inst>, "<kind>", "<label>", <n_ops>, ...
    entry_re = re.compile(
        r'\{\s*(\d+),\s*(?:(\d+),\s*)?"([^"]+)",\s*(\d+),'
        r'\s*"([^"]+)",\s*"([^"]+)",\s*(\d+),'
    )
    entries = []
    for m in entry_re.finditer(text):
        seg_id = int(m.group(1))
        entries.append({
            "seg_id":     seg_id,
            "ctx_seg_id": int(m.group(2)) if m.group(2) else seg_id,
            "network":    m.group(3),
            "instance":   int(m.group(4)),
            "kind":       m.group(5),
            "label":      m.group(6),
            "n_ops":      int(m.group(7)),
            "op_ids":     seg_ops.get(seg_id, []),
        })
    return entries


def _load_name_map(path: str) -> dict[str, str]:
    """Read a per-network annotation file (yolov8n_HTA_split.json shape)
    and return ``{mangled_name: orig_onnx_name}``.

    We map FROM the runtime/schedule's `module_name` (which is the
    QNN-mangled identifier for tflite-route models) TO the original ONNX
    node name that lives in the source .onnx file.
    """
    with open(path) as f:
        data = json.load(f)
    out: dict[str, str] = {}
    for op in data.get("ops", []):
        nm = op.get("name")
        orig = op.get("_orig_name")
        if nm and orig:
            out[nm] = orig
    return out


def _read_qnn_ir(qnn_json_path: str) -> dict:
    """Load the QNN JSON IR (from qairt-dlc-to-json) used to map a
    network's dispatch_id ordering onto QNN node names.
    """
    with open(qnn_json_path) as f:
        return json.load(f)


def _resolve_dispatch_id_to_onnx_nodes(
    network: str,
    runtime_gen_dir: str,
    qnn_json_path: str | None,
    name_map: dict[str, str] | None,
    onnx_node_names: set[str],
) -> dict[int, list[str]]:
    """Build dispatch_id → list-of-ONNX-node-names map for one network.

    A dispatch_id usually maps to one ONNX node (the common case) but
    can map to multiple when the source ONNX has had structural rewrites
    applied — most commonly the BatchNorm rewrite (PARTITIONING_GUIDE.md
    §3) where each `BatchNormalization` node was expanded into a
    `<name>_mul` + `<name>_add` pair. The slicer needs to pull both
    when slicing a segment that the schedule said contains the BN op.

    Source of truth for op-order is the per-op stats CSV (same exec_order
    the runtime generator used). We sourced QNN node names from there
    via the schedule's `module_name` — those are the strings that need
    to be translated to source-ONNX names.

    Args:
      network: the network identifier (e.g. "dronet")
      runtime_gen_dir: the runtime gen dir; we read schedule cache from it
      qnn_json_path: optional path to qairt-dlc-to-json output (not yet
        used; placeholder for pure-ID-based remapping when a name map
        isn't available)
      name_map: optional QNN-mangled → ONNX-orig map
      onnx_node_names: the actual node names in the source ONNX, for
        verification of the resolved mappings
    """
    # We pull the (dispatch_id, module_name) pairs from the schedule
    # cached in the runtime gen dir's meta sidecar — but generate_runtime.py
    # didn't emit one, so we re-derive from the original schedule JSON.
    sched_meta_path = os.path.join(runtime_gen_dir, "schedule_source.json")
    if not os.path.exists(sched_meta_path):
        raise SystemExit(
            f"missing {sched_meta_path}; run generate_runtime.py with the "
            f"updated --emit-source flag (or pass --schedule directly to this tool)")
    with open(sched_meta_path) as f:
        sched = json.load(f)

    out: dict[int, list[str]] = {}
    for k, v in sched["dispatches"].items():
        net = v["job_name"].rstrip("0123456789")
        if net != network:
            continue
        did = v["id"]
        mname = v["module_name"]
        # If the schedule's name matches an ONNX node directly, use it.
        if mname in onnx_node_names:
            out[did] = [mname]
            continue
        # BN rewrite: the BatchNorm op in the schedule may have been
        # expanded into a Mul+Add pair in the BN-free ONNX variant.
        # When we don't find the BN name as-is, look for `<name>_mul`
        # + `<name>_add` and pull both into this dispatch_id's slice.
        mul_n = mname + "_mul"
        add_n = mname + "_add"
        if mul_n in onnx_node_names and add_n in onnx_node_names:
            out[did] = [mul_n, add_n]
            continue
        # Optional explicit translation map (rarely needed in practice;
        # main use case is yolov8n's TFLite-route names where this is a
        # whole-graph mismatch — see README §5.2).
        if name_map and mname in name_map:
            translated = name_map[mname]
            if translated in onnx_node_names:
                out[did] = [translated]
                continue
        # Otherwise leave the mapping unresolved; the caller skips the
        # segment with a clear error rather than emit a busted slice.
        out[did] = []        # sentinel: unresolved
    return out


def _segment_node_names(seg: dict, dispatch_to_onnx: dict[int, list[str]]) -> list[str]:
    """Return the ONNX node names for a segment's dispatch_ids, dropping
    any that resolved to [] (unresolved). Each dispatch_id can map to
    multiple ONNX nodes (BN rewrite case); we flatten."""
    names: list[str] = []
    for did in seg["op_ids"]:
        ns = dispatch_to_onnx.get(did, [])
        names.extend(ns)
    return names


def _all_resolved(seg: dict, dispatch_to_onnx: dict[int, list[str]]) -> bool:
    """True iff every dispatch_id in the segment has at least one ONNX-
    side node mapped (we tolerate 1-to-many but not 1-to-zero)."""
    return all(dispatch_to_onnx.get(d) for d in seg["op_ids"])


def _compute_boundary_tensors(
    seg_node_names: list[str],
    onnx_model: onnx.ModelProto,
) -> tuple[list[str], list[str]]:
    """Determine the sub-ONNX's I/O tensors for an op-name subset.

    A tensor is a sub-graph INPUT iff:
       - it's consumed by some node in the segment AND
       - it's NOT produced by any node in the segment AND
       - it's not an initializer (weights/biases live in the slice
         automatically — they're part of the sub-ONNX's initializer list,
         not exposed as I/O).
    A tensor is a sub-graph OUTPUT iff:
       - it's produced by some node in the segment AND
         (it's a graph output of the original model OR
          some node OUTSIDE the segment consumes it).

    Tensors that are produced and only consumed inside the segment stay
    as internal edges of the sub-graph.
    """
    seg_set = set(seg_node_names)
    initializers = {init.name for init in onnx_model.graph.initializer}
    graph_outputs = {o.name for o in onnx_model.graph.output}

    in_seg_outputs:    set[str] = set()
    in_seg_inputs:     set[str] = set()
    out_seg_consumers: dict[str, list[str]] = defaultdict(list)

    for n in onnx_model.graph.node:
        if n.name in seg_set:
            in_seg_inputs.update(t for t in n.input if t)
            in_seg_outputs.update(t for t in n.output if t)
        else:
            # Track which out-of-segment nodes consume each tensor so we
            # can identify boundary outputs.
            for t in n.input:
                if t:
                    out_seg_consumers[t].append(n.name)

    inputs:  list[str] = []
    outputs: list[str] = []
    for t in sorted(in_seg_inputs):
        if t in in_seg_outputs:
            continue          # produced internally
        if t in initializers:
            continue          # weight/bias — included automatically by extractor
        inputs.append(t)
    for t in sorted(in_seg_outputs):
        if t in graph_outputs or out_seg_consumers.get(t):
            outputs.append(t)
    return inputs, outputs


def _slice_one(seg: dict, network_onnx: str, dispatch_to_onnx: dict[int, list[str]],
                out_path: str) -> dict | None:
    """Produce one sub-ONNX file for `seg` from `network_onnx`.

    Returns a metadata dict (logged into the manifest), or None if the
    segment couldn't be sliced (e.g. unresolved name mapping).
    """
    model = onnx.load(network_onnx)
    onnx_node_names = {n.name for n in model.graph.node}

    if not _all_resolved(seg, dispatch_to_onnx):
        unresolved = [d for d in seg["op_ids"] if not dispatch_to_onnx.get(d)]
        print(f"  seg{seg['seg_id']:>3d}: SKIP — {len(unresolved)} op_ids "
              f"unresolved against {network_onnx}", file=sys.stderr)
        return None
    seg_names = _segment_node_names(seg, dispatch_to_onnx)
    missing = [n for n in seg_names if n not in onnx_node_names]
    if missing:
        print(f"  seg{seg['seg_id']:>3d}: SKIP — names not in ONNX: "
              f"{missing[:3]}{'...' if len(missing)>3 else ''}", file=sys.stderr)
        return None

    inputs, outputs = _compute_boundary_tensors(seg_names, model)
    if not inputs or not outputs:
        print(f"  seg{seg['seg_id']:>3d}: SKIP — empty boundary "
              f"(in={len(inputs)}, out={len(outputs)})", file=sys.stderr)
        return None

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    onnx.utils.extract_model(network_onnx, out_path, inputs, outputs)
    sub = onnx.load(out_path)
    print(f"  seg{seg['seg_id']:>3d}  net={seg['network']:<10s}  "
          f"label={seg['label']:<10s}  ops={len(seg_names):>3d}  "
          f"in={len(inputs):>2d} out={len(outputs):>2d}  "
          f"sub_nodes={len(sub.graph.node):>3d}  size={os.path.getsize(out_path)//1024} KB")
    return {
        "seg_id":      seg["seg_id"],
        "network":     seg["network"],
        "label":       seg["label"],
        "kind":        seg["kind"],
        "instance":    seg["instance"],
        "op_ids":      seg["op_ids"],
        "input_tensors":  inputs,
        "output_tensors": outputs,
        "sub_onnx_path":  out_path,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runtime-gen", required=True,
                    help="path to the runtime gen dir produced by generate_runtime.py")
    ap.add_argument("--schedule", default=None,
                    help="path to the source schedule.json. If omitted, we look "
                         "for <runtime-gen>/schedule_source.json (the generator "
                         "now copies the schedule there for downstream tooling).")
    ap.add_argument("--onnx-map", required=True,
                    help="comma-sep <network>=<onnx_path> (e.g. "
                         "dronet=qnn_models/dronet.onnx,yolov8n=qnn_models/yolov8n.onnx). "
                         "This is the default per-network source ONNX. May be "
                         "overridden per-target-label via --onnx-per-target.")
    ap.add_argument("--onnx-per-target", default=None,
                    help="comma-sep <network>:<label>=<onnx_path> overrides — "
                         "for HTA-targeted dronet segments we want the BN-rewritten "
                         "graph (qnn_models/dronet_bnfree.onnx) so HTA's narrow op "
                         "set accepts them, while CPU segments use the original. "
                         "See PARTITIONING_GUIDE.md §3 for the rationale.")
    ap.add_argument("--name-map", default=None,
                    help="comma-sep <network>=<annotation.json> with the per-op "
                         "{name, _orig_name} mapping (for networks where the "
                         "scheduler's module_name doesn't match the ONNX node "
                         "name — e.g. yolov8n's tflite-route mangled names)")
    ap.add_argument("--out-dir", required=True,
                    help="emit per-segment sub_onnx/<seg_id>.onnx + a manifest.json")
    args = ap.parse_args()

    onnx_map: dict[str, str] = {}
    for kv in args.onnx_map.split(","):
        net, _, p = kv.strip().partition("=")
        if not net or not p: raise SystemExit(f"--onnx-map: bad entry '{kv}'")
        onnx_map[net.strip()] = p.strip()

    onnx_per_target: dict[tuple[str, str], str] = {}
    for kv in (args.onnx_per_target or "").split(","):
        if not kv.strip(): continue
        head, _, p = kv.strip().partition("=")
        net, _, label = head.partition(":")
        if not net or not label or not p:
            raise SystemExit(f"--onnx-per-target: bad entry '{kv}'")
        onnx_per_target[(net.strip(), label.strip())] = p.strip()

    name_map_files: dict[str, str] = {}
    for kv in (args.name_map or "").split(","):
        if not kv.strip(): continue
        net, _, p = kv.strip().partition("=")
        if not net or not p: raise SystemExit(f"--name-map: bad entry '{kv}'")
        name_map_files[net.strip()] = p.strip()

    # Stage the schedule next to the runtime-gen dir so downstream tools
    # (this slicer, build_subdlcs.sh) can consume it without each user
    # threading --schedule through every command.
    sched_path = args.schedule
    if not sched_path:
        sched_path = os.path.join(args.runtime_gen, "schedule_source.json")
    if not os.path.exists(sched_path):
        raise SystemExit(f"schedule not found at {sched_path}")
    if args.schedule and args.schedule != os.path.join(args.runtime_gen, "schedule_source.json"):
        # Mirror it into the runtime-gen dir so further calls don't need
        # --schedule. (Cheap; idempotent.)
        os.makedirs(args.runtime_gen, exist_ok=True)
        with open(sched_path) as fi, \
             open(os.path.join(args.runtime_gen, "schedule_source.json"), "w") as fo:
            fo.write(fi.read())

    segments = _read_dispatch_table(args.runtime_gen)
    print(f"loaded {len(segments)} segments from {args.runtime_gen}/dispatch_table.h")

    # For each (network, label), build the dispatch_id → ONNX-name(s)
    # lookup using whichever source ONNX applies. The map is per-(net,
    # label) because the BN-rewritten variant has different node names
    # for the BN regions, so the resolution is content-dependent.
    pairs = sorted({(s["network"], s["label"]) for s in segments})
    dispatch_maps: dict[tuple[str, str], dict[int, list[str]]] = {}
    onnx_for_pair: dict[tuple[str, str], str] = {}
    for (net, label) in pairs:
        path = onnx_per_target.get((net, label), onnx_map.get(net))
        if path is None:
            raise SystemExit(f"--onnx-map / --onnx-per-target missing entry for "
                              f"network='{net}' label='{label}'")
        onnx_for_pair[(net, label)] = path
        nm = _load_name_map(name_map_files[net]) if net in name_map_files else None
        m = onnx.load(path)
        onnx_names = {n.name for n in m.graph.node}
        dispatch_maps[(net, label)] = _resolve_dispatch_id_to_onnx_nodes(
            net, args.runtime_gen, qnn_json_path=None,
            name_map=nm, onnx_node_names=onnx_names)
        # When BN-rewritten variant is used, the resolution may map a
        # single dispatch_id to multiple ONNX nodes (e.g. bn → mul+add).
        # Count "fully resolved" dispatch_ids, plus how many had to be
        # expanded via the BN rewrite — useful diagnostic.
        n_resolved = sum(1 for v in dispatch_maps[(net, label)].values() if v)
        n_total    = len(dispatch_maps[(net, label)])
        n_expanded = sum(1 for v in dispatch_maps[(net, label)].values() if len(v) > 1)
        print(f"  {net}/{label}: dispatch_id→ONNX-name resolved {n_resolved}/"
              f"{n_total}  expanded={n_expanded}  ({path})")
        if n_resolved == 0 and n_total > 0:
            # Whole-network mismatch — almost certainly a name-space gap
            # (e.g. yolov8n's schedule was profiled on a TFLite-route DLC
            # whose QNN node names — pad_0/convolution_0/... — don't
            # appear in the ONNX. The HTA_split annotation file uses
            # ONNX-route names — model_0_conv_Conv — which are also
            # different. Two distinct DLC routes; no direct map.)
            print(
                f"\n  WARN: '{net}/{label}' didn't resolve any names. The schedule's "
                "module_name strings don't appear in the source ONNX, and the\n"
                "  --name-map (if provided) didn't bridge them either. This\n"
                "  usually means the schedule was profiled on a different\n"
                "  conversion route (TFLite → DLC) than the ONNX file you're\n"
                "  trying to slice. Two ways forward:\n"
                "    a) Re-profile and re-schedule using a DLC built directly\n"
                "       from the ONNX (qnn-onnx-converter route) so the\n"
                "       schedule's module_names match the ONNX node names.\n"
                "    b) Capture a per-(scheduler-name, ONNX-name) map by\n"
                "       running qairt-dlc-to-json on BOTH the TFLite-route\n"
                "       DLC and the ONNX-route DLC and matching them by\n"
                "       op-type sequence, then pass the result via --name-map.\n",
                file=sys.stderr)

    # Slice each segment. Two segments with identical (network, op_ids)
    # share one sub-ONNX file (e.g. the dronet0..4 instances all reduce
    # to the same 7 unique slices); the manifest records both seg_ids.
    # Dedup key includes the label because the BN-rewritten variant
    # produces a different sub-graph for the same op_ids on a different
    # target. Two segments with the same op_ids on the same (net, label)
    # genuinely alias; on different labels they are different slices.
    seen: dict[tuple[str, str, tuple[int, ...]], str] = {}
    manifest = []
    sub_dir = args.out_dir
    print(f"\nslicing into {sub_dir}/  (deduping identical (network, label, op_ids) tuples)")
    for s in segments:
        net   = s["network"]
        label = s["label"]
        key = (net, label, tuple(s["op_ids"]))
        if key in seen:
            manifest.append({
                "seg_id":     s["seg_id"],
                "network":    net,
                "label":      label,
                "kind":       s["kind"],
                "instance":   s["instance"],
                "op_ids":     s["op_ids"],
                "alias_of":   seen[key],
            })
            continue
        out_path = os.path.join(sub_dir, f"{net}_{label}_seg{s['seg_id']}.onnx")
        meta = _slice_one(s, onnx_for_pair[(net, label)],
                          dispatch_maps[(net, label)], out_path)
        if meta is not None:
            meta["source_onnx"] = onnx_for_pair[(net, label)]
            manifest.append(meta)
            seen[key] = out_path

    manifest_path = os.path.join(sub_dir, "manifest.json")
    os.makedirs(sub_dir, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump({"segments": manifest}, f, indent=2)
    n_unique = len(seen)
    n_emitted = sum(1 for m in manifest if "alias_of" not in m)
    print(f"\nwrote {n_emitted} unique sub-ONNXes ({n_unique} cache hits avoided), "
          f"manifest at {manifest_path}")


if __name__ == "__main__":
    main()
