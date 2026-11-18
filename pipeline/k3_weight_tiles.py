"""Build-time matrix-tile weights for the SpaceMiT K3 IME conv kernels.

The K3 IME conv kernels (kernels/ime_k3_x100/, and kernels/ime_k3_a100/ when its
selected conv kernel reads tiles) can take their conv weights already packed in
the engine's B-operand tile order, behind a 64-byte header they check:

  * X100 (smt.vmadot 4x8x4):  "MBX100TW", tiles [k8 slab][ceil(OC/4)][4][8]
  * A100 (smt.vmadot 8x16x8): "MBIME2TW", tiles [ceil(OC/8)][k16 slab][8][16]

with k = (ic*KH + kh)*KW + kw (NCHW k order). Without a header the kernels read
the backend's plain IHWOC weights and pack them per call, so tiling is a speed
choice, never a correctness one. NHWC k order (the kernels' NHWC panel) is not
built here (k3-work TASK-015: NCHW only for now).

Only unsplit, unsharded, single-reader, symmetric convs whose SELECTED kernel
(the first file generate_kernels' curated probe would find for the backend)
looks for the header are tiled. MODELBLASTER_IME2_TILES=0 turns tiling off.

A tiled weight cannot be OC-sliced: the IHWOC parallel_conv2d_s8 wrapper sees
the header and runs the call serially (generate_skeleton._conv2d_s8_ihwoc_wrapper).
Build with MODELBLASTER_IME2_TILES=0 to let a pool shard these convs.

Ported from k3-work's TASK-014 kernels (ModelBlaster k3-r dfd02d8).
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np

#: backend -> {op: tile family}. Every other backend: no tiles.
TILE_OPS: dict[str, dict[str, str]] = {
    # conv2d_s8_pc: the per-channel X100 conv reads the same tiles (k3-work TASK-019)
    "ime_k3_x100": {"conv2d_s8": "x100_conv", "conv2d_s8_pc": "x100_conv"},
    "ime_k3_a100": {"conv2d_s8": "a100_conv"},
}
TILE_MAGIC = {"x100_conv": b"MBX100TW", "a100_conv": b"MBIME2TW"}
TILE_LABEL = {"x100_conv": "X100 conv", "a100_conv": "A100 conv"}
#: Header size = the tiles' alignment: magic, rows, K (uint32 LE), then zeros.
HEADER_BYTES = 64
#: Written next to weights.c; generate_kernels refuses a kernel that cannot read it.
WEIGHT_LAYOUTS_JSON = "weight_layouts.json"

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def selected_kernel_path(backend: str, op: str,
                         kernels_dir: Optional[str] = None) -> Optional[str]:
    """The backend's own curated file generate_kernels' probe finds first for `op`."""
    from modelblaster.pipeline.reference_kernels import KERNEL_SPECS
    spec = KERNEL_SPECS.get(op)
    if spec is None:
        return None
    root = kernels_dir or os.path.join(_REPO, "kernels")
    for algo in spec.algorithms:
        path = os.path.join(root, backend, f"{backend}_{op}_{algo.name}.c")
        if os.path.exists(path):
            return path
    return None


def _kernel_reads(backend: str, op: str, magic: bytes) -> bool:
    path = selected_kernel_path(backend, op)
    if path is None:
        return False
    with open(path) as f:
        return magic.decode() in f.read()


def _conv_weight(op: dict[str, Any]) -> Optional[str]:
    if op.get("weight"):
        return op["weight"]
    for sub in op.get("sub_ops") or ():
        if str(sub.get("op", "")).startswith("conv2d") and sub.get("weight"):
            return sub["weight"]
    return None


#: Shard arrays of an output-channel-split conv: "<weight>.shard_<t>"
#: (generate_skeleton._shard_key).
_SHARD_SUFFIX = ".shard_"


def tile_plan(ir: Optional[dict[str, Any]], backend: Optional[str],
              exclude_ids: frozenset = frozenset(),
              weights: Optional[dict[str, np.ndarray]] = None) -> dict[str, tuple[str, str, int, int]]:
    """`{weight: (op, family, rows, K)}` for the weights this build tiles.

    `exclude_ids` are the convs split by output channels (shard_conv_weights):
    their parent weight is replaced by one array per shard. With `weights`
    given, those shard arrays are tiled instead (one tile set per shard, rows =
    the shard's channels; k3-work TASK-016: 1.09-1.31x on the X100's
    channel-split layers). A conv split by output rows keeps its full weight
    and is tiled like an unsplit one. MODELBLASTER_K3_SHARD_TILES=0 leaves
    shard arrays plain (IHWOC)."""
    tile_ops = TILE_OPS.get(backend or "")
    if not tile_ops or ir is None or os.environ.get("MODELBLASTER_IME2_TILES", "1") == "0":
        return {}
    ops = ir.get("ops", [])
    readers: dict[str, int] = {}
    for op in ops:
        for w in {op.get("weight"), _conv_weight(op)} - {None}:
            readers[w] = readers.get(w, 0) + 1
    plan: dict[str, tuple[str, str, int, int]] = {}
    for op in ops:
        fam = tile_ops.get(op.get("op"))
        if fam is None or op.get("split_from"):
            continue
        quant = op.get("quant") or {}
        if int(quant.get("input_offset", 0)) or int(quant.get("filter_offset", 0)):
            continue
        w, sh = _conv_weight(op), op.get("shape") or {}
        rows = int(sh.get("OC", 0))
        k = int(sh.get("IC", 0)) * int(sh.get("KH", 0)) * int(sh.get("KW", 0))
        if not w or readers.get(w) != 1 or rows * k < 16:
            continue
        if not _kernel_reads(backend, op["op"], TILE_MAGIC[fam]):
            continue
        if op.get("dispatch_id") in exclude_ids:
            if weights is None or os.environ.get("MODELBLASTER_K3_SHARD_TILES", "1") == "0":
                continue
            t = 0
            while f"{w}{_SHARD_SUFFIX}{t}" in weights:
                sk = f"{w}{_SHARD_SUFFIX}{t}"
                plan[sk] = (op["op"], fam, int(weights[sk].shape[0]), k)
                t += 1
            continue
        plan[w] = (op["op"], fam, rows, k)
    return plan


def tile_pack(arr: np.ndarray, fam: str, rows: int, k: int) -> np.ndarray:
    """OIHW conv weight `arr` in its family's tile order behind the 64-byte header."""
    w = np.asarray(arr, dtype=np.int8).reshape(rows, k)
    if fam == "x100_conv":
        ks, panels = (k + 7) // 8, (rows + 3) // 4
        pad = np.zeros((panels * 4, ks * 8), np.int8)
        pad[:rows, :k] = w
        tiles = pad.reshape(panels, 4, ks, 8).transpose(2, 0, 1, 3)
    elif fam == "a100_conv":
        ks, otiles = (k + 15) // 16, (rows + 7) // 8
        pad = np.zeros((otiles * 8, ks * 16), np.int8)
        pad[:rows, :k] = w
        tiles = pad.reshape(otiles, 8, ks, 16).transpose(0, 2, 1, 3)
    else:
        raise ValueError(f"unknown tile family {fam!r}")
    header = np.frombuffer(TILE_MAGIC[fam] + np.array([rows, k], "<u4").tobytes()
                           + bytes(HEADER_BYTES - 16), np.int8)
    return np.concatenate([header, np.ascontiguousarray(tiles).reshape(-1)])


def pack_weights(ir: Optional[dict[str, Any]], backend: Optional[str],
                 weights: dict[str, np.ndarray], out_dir: str,
                 exclude_ids: frozenset = frozenset()) -> dict[str, tuple[np.ndarray, str]]:
    """`{weight: (tiled array, comment)}`; writes or clears weight_layouts.json."""
    plan = tile_plan(ir, backend, exclude_ids, weights)
    path = os.path.join(out_dir, WEIGHT_LAYOUTS_JSON)
    if not plan:
        if os.path.exists(path):
            os.remove(path)   # a stale manifest would fail the kernel check
        return {}
    op_magic: dict[str, set[str]] = {}
    for op, fam, _r, _k in plan.values():
        op_magic.setdefault(op, set()).add(TILE_MAGIC[fam].decode())
    with open(path, "w") as f:
        json.dump({"k3_tiles": {
            "op_magic": {op: sorted(m) for op, m in sorted(op_magic.items())},
            "weights": {w: {"op": v[0], "family": v[1], "rows": v[2], "K": v[3]}
                        for w, v in sorted(plan.items())}}}, f, indent=1)
    out = {}
    for w, (_op, fam, rows, k) in plan.items():
        if w in weights:
            out[w] = (tile_pack(weights[w], fam, rows, k),
                      f"{TILE_LABEL[fam]} tiles, rows={rows} K={k}, "
                      f"64-byte {TILE_MAGIC[fam].decode()} header")
    return out


def check_kernels_read_tiles(impls: dict[str, str], out_dir: str) -> None:
    """Refuse a kernel set in which a tiled weight's op kernel does not check its header."""
    path = os.path.join(out_dir, WEIGHT_LAYOUTS_JSON)
    if not os.path.exists(path):
        return
    with open(path) as f:
        need = json.load(f).get("k3_tiles", {}).get("op_magic", {})
    bad = sorted(f"{op} ({m})" for op, magics in need.items() for m in magics
                 if m not in (impls.get(op) or ""))
    if bad:
        raise SystemExit(
            f"{path}: weights of {bad} were packed as matrix tiles, but the kernel "
            f"chosen for them does not read that layout. Regenerate the skeleton, "
            f"or build with MODELBLASTER_IME2_TILES=0.")
