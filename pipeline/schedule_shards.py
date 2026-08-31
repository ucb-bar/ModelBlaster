"""Derive per-op codegen shard widths from an XPU-RT schedule.

XPU-RT expresses an intra-op implementation as a composite
``hardware_target`` such as ``CPU_P#0+CPU_P#1+CPU_P#2+CPU_P#3``.  For packed
RVV convolution weights, ModelBlaster must know that width while emitting the
skeleton so it can materialize one correctly packed weight array per shard.
This module makes that last schedule-to-codegen edge explicit.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


_PACKED_WEIGHT_SHARD_OPS = {
    "conv2d_s8",
    "conv2d_batchnorm2d_s8",
    "conv2d_batchnorm2d_silu_s8",
    "conv2d_silu_s8",
}
_MARKER = "_xpurt_schedule_shard_factor"


def _is_instance(job_name: str, network: str) -> bool:
    if job_name == network:
        return True
    suffix = job_name[len(network):] if job_name.startswith(network) else ""
    return bool(suffix) and suffix.isdigit()


def _output_channels(op: dict[str, Any]) -> int:
    shape = op.get("shape") or {}
    if "OC" in shape:
        return int(shape["OC"] or 0)
    for sub in op.get("sub_ops") or []:
        if str(sub.get("op", "")).startswith("conv2d"):
            return int((sub.get("shape") or {}).get("OC", 0) or 0)
    return 0


def apply_schedule_shards(ir: dict[str, Any], schedule: dict[str, Any],
                          network: str) -> tuple[dict[str, Any], list[dict]]:
    """Return an IR whose packed-weight ops match scheduled core widths.

    All periodic instances of one dispatch must choose one width: generated
    weight layout is per dispatch, not per invocation.  Linear operations do
    not need annotations because their row-major weights can be sliced at
    runtime using the exact entry pool width.
    """
    out = copy.deepcopy(ir)
    widths: dict[int, set[int]] = {}
    for entry in (schedule.get("dispatches") or {}).values():
        if not _is_instance(str(entry.get("job_name", "")), network):
            continue
        did = int(entry["id"])
        width = len([x for x in str(entry["hardware_target"]).split("+")
                     if x.strip()])
        widths.setdefault(did, set()).add(width)

    varying = {did: sorted(values) for did, values in widths.items()
               if len(values) != 1}
    if varying:
        raise ValueError(
            f"{network}: periodic instances choose different widths for the "
            f"same dispatch: {varying}; one generated model cannot encode "
            "invocation-dependent packed weight layouts")

    applied: list[dict] = []
    for op in out.get("ops", []):
        # Remove only annotations this bridge wrote on a prior run. Explicit
        # user/compiler shard hints remain authoritative.
        if op.pop(_MARKER, False):
            op.pop("shard_factor", None)
        did = op.get("dispatch_id")
        if did is None or op.get("op") not in _PACKED_WEIGHT_SHARD_OPS:
            continue
        width = next(iter(widths.get(int(did), {1})))
        if width <= 1:
            continue
        oc = _output_channels(op)
        if oc <= 0 or oc % width:
            raise ValueError(
                f"{network} dispatch {did} ({op.get('op')}) has OC={oc}, "
                f"not divisible by scheduled width {width}; codegen would "
                "silently run a serial implementation")
        explicit = op.get("shard_factor")
        if explicit is not None and int(explicit) != width:
            raise ValueError(
                f"{network} dispatch {did} already requests shard_factor="
                f"{explicit}, but the schedule reserves width {width}")
        op["shard_factor"] = width
        op[_MARKER] = True
        applied.append({"dispatch_id": int(did), "op": op["op"],
                        "oc": oc, "n_shards": width})

    rewrite = out.setdefault("_rewrite", {})
    rewrite["xpurt_schedule_shards"] = {
        "network": network,
        "dispatches": applied,
    }
    return out, applied


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ir", required=True)
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--network", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ir = json.loads(Path(args.ir).read_text())
    schedule = json.loads(Path(args.schedule).read_text())
    out, applied = apply_schedule_shards(ir, schedule, args.network)
    text = json.dumps(out, indent=1) + "\n"
    destination = Path(args.out)
    if not destination.exists() or destination.read_text() != text:
        destination.write_text(text)
        changed = "updated"
    else:
        changed = "unchanged"
    print(f"schedule shards [{args.network}]: {len(applied)} packed-weight "
          f"dispatch(es), {changed} {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
