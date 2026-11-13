"""Apply XPU-RT axis-C split hints (modelblaster.split_hints/v1) to an IR.

The dual of `pipeline/apply_fusion_hint.py`. Where fusion collapses N
adjacent dispatches into one (good for too_fine workloads), splitting
decomposes one heavy dispatch into N smaller ones the scheduler can
place on separate cores in parallel (good for too_coarse workloads).

XPU-RT's `granularity_loop.py` emits a Contract-2 split hint like:

    {"contract": "modelblaster.split_hints/v1",
     "reason": "granularity 'too_coarse'; split conv2d_s8[4] across OC",
     "networks": [
        {"network": "yolov8_nano",
         "split_ops": [{"op": 4, "n_splits": 2}]}
     ]}

For `linear_s8` (Phase 1e initial scope), splitting along the output
feature dim (N) is the cleanest tile: each tile computes a disjoint
subset of output rows from the same input. No final concat needed —
each tile writes to its own output slice; downstream consumers
treat the original output tensor as the concat of the tiles by name
convention (`<orig>.tile_0`, `<orig>.tile_1`, ...).

For `conv2d_s8`, the tile dimension is OC (output channels) — same
shape, just on the conv output. Whether the weights need slicing is a
BACKEND question, and `generate_skeleton` answers it, not this module:
on a backend that keeps conv weights OIHW `[OC, IC, KH, KW]` an OC slice
is contiguous and a pointer offset suffices, while the rvv backends pack
them IHWOC `(IC, KH, KW, OC)` — OC innermost — where an OC slice is
strided and each tile is given its own re-packed array. See
`generate_skeleton.split_conv_tile_weights`.

Inserting tiles renumbers every op after the split point, so the output
graph carries an `id_remap` field mapping every input `dispatch_id` to
its id(s) in the rewritten graph (JSON object keys are strings)::

    "id_remap": {"0": [0, 1], "1": [2], "2": [3]}

Values are always lists: splitting is one-to-many, so op 0 above became
tiles 0 and 1, while untouched ops 1 and 2 (renumbered to 2 and 3) get
single-element lists. Consumers keyed on dispatch_id must translate
through this before joining a pre-rewrite profile / cost DB against a
post-rewrite graph — and must sum across the tiles when attributing a
split op's cost. (`apply_fusion_hint.py` emits the same field, but
scalar-valued — fusion is many-to-one.)
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


SPLIT_CONTRACT = "modelblaster.split_hints/v1"


class SplitHintError(ValueError):
    """Hint can't be applied — e.g. op kind not split-capable, N doesn't
    divide the tilable dim, op id missing from IR."""


def _ops_by_id(ops: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {op["dispatch_id"]: op for op in ops
            if op.get("dispatch_id") is not None}


def _split_linear_s8(op: dict[str, Any], n_splits: int,
                     network: str, tile_sizes=None) -> list[dict[str, Any]]:
    """Split a linear_s8 / linear op along the N (output features) dim.

    Dtype-agnostic: reads only shape["N"]. Registered for both the int8 and
    the fp32 linear in _SPLITTABLE.

    Generates `n_splits` new ops, each computing rows [t*N/n, (t+1)*N/n)
    of the original output. Weight and bias slices share the source
    tensor (no new buffers — the kernel call passes `weight + t*N/n*K`
    and `bias + t*N/n` as offsets) so we keep weight extraction simple.

    Output is one tensor `<orig.out>.tile_t` per tile; downstream
    consumers of the original tensor either rely on contiguous layout
    (the tiles are placed back-to-back in memory) or get a follow-up
    concat op.
    """
    shape = op["shape"]
    N = int(shape["N"])
    widths = _tile_widths(N, n_splits, tile_sizes, network, "linear_s8 N")
    out_tensor = op["outputs"][0]
    tiles: list[dict[str, Any]] = []
    n0 = 0
    for t, tile_n in enumerate(widths):
        tile = copy.deepcopy(op)
        tile["shape"] = dict(shape)
        tile["shape"]["N"] = tile_n
        # New per-tile output name; the OG tensor is reassembled by the
        # codegen via offset aliasing (downstream consumers see the tile
        # output's contiguous storage).
        tile["outputs"] = [f"{out_tensor}.tile_{t}"]
        tile["name"] = op["name"] + f".tile_{t}"
        tile["split_from"] = {"op_id": op["dispatch_id"], "tile": t,
                              "n_splits": len(widths), "tile_n": tile_n,
                              "tile_offset_N": n0}
        # `hardware_target` left as-is; the scheduler may place tiles
        # on different cores at scheduling time.
        tiles.append(tile)
        n0 += tile_n
    return tiles


def _tile_widths(total: int, n_splits: int, tile_sizes,
                 network: str, what: str) -> list[int]:
    """Resolve a split into per-tile widths along the tiled axis.

    Two ways to ask for a split, and the second is why this exists:

      * `n_splits=N` -- N EVEN tiles, the original contract. Still requires
        `total % N == 0`; an uneven remainder was rejected before and stays
        rejected, because "n_splits" names a count, not a partition.
      * `tile_sizes=[a, b, ...]` -- an explicit, possibly UNEVEN partition.
        The widths must be positive and sum to `total`; anything else would
        leave output channels unwritten or make two tiles claim the same ones.

    An uneven partition is the point: the two backends here do NOT have the
    same cost curve, so the split that balances them is generally not the
    even one. Every downstream consumer keys off `tile_offset_OC` (which the
    splitters record) rather than recomputing `tile * tile_oc`, so uneven
    widths need no further special-casing.
    """
    if tile_sizes is not None:
        widths = [int(x) for x in tile_sizes]
        if len(widths) < 2:
            raise SplitHintError(
                f"{network}: tile_sizes={tile_sizes} must name at least 2 tiles")
        if any(w <= 0 for w in widths):
            raise SplitHintError(
                f"{network}: tile_sizes={tile_sizes} has a non-positive tile")
        if sum(widths) != total:
            raise SplitHintError(
                f"{network}: tile_sizes={tile_sizes} sums to {sum(widths)}, "
                f"but {what}={total}. A partition that does not sum to the "
                f"full axis leaves outputs unwritten or double-written.")
        return widths
    if total % n_splits != 0:
        raise SplitHintError(
            f"{network}: {what}={total} doesn't divide cleanly into "
            f"{n_splits} tiles (need {what} % n_splits == 0). Pass "
            f"tile_sizes=[...] for an uneven partition.")
    return [total // n_splits] * n_splits


def _split_conv2d_s8(op: dict[str, Any], n_splits: int,
                     network: str, tile_sizes=None) -> list[dict[str, Any]]:
    """Split a conv2d_s8 op along the OC (output channel) dim.

    Splitting along OC produces n tiles that each compute a disjoint
    slice of output channels. How the tile reaches its own weights is
    left to the codegen, because it depends on the layout the backend
    packs conv weights in: OIHW `[OC, IC, KH, KW]` makes an OC slice
    contiguous, so `weight + t*(OC/n)*IC*KH*KW` and `bias + t*(OC/n)`
    are enough; the rvv backends' IHWOC packing makes it strided, so the
    tile gets its own re-packed array instead. This function records
    only `tile_offset_OC` and `axis`, and
    `generate_skeleton.split_conv_tile_weights` decides.

    The output tensor `[N, OC, OH, OW]` splits along OC: tile_t
    writes channels `[t*OC/n, (t+1)*OC/n)`. Output is named
    `<orig>.tile_<t>`. Downstream consumers see the original tensor
    as the concat of tiles along OC (or read each tile slice
    explicitly by name).
    """
    shape = op["shape"]
    OC = int(shape["OC"])
    widths = _tile_widths(OC, n_splits, tile_sizes, network, "conv2d_s8 OC")
    out_tensor = op["outputs"][0]
    tiles: list[dict[str, Any]] = []
    oc0 = 0
    for t, tile_oc in enumerate(widths):
        tile = copy.deepcopy(op)
        tile["shape"] = dict(shape)
        tile["shape"]["OC"] = tile_oc
        tile["outputs"] = [f"{out_tensor}.tile_{t}"]
        tile["name"] = op["name"] + f".tile_{t}"
        tile["split_from"] = {"op_id": op["dispatch_id"], "tile": t,
                              "n_splits": len(widths), "tile_oc": tile_oc,
                              "tile_offset_OC": oc0,
                              "axis": "OC"}
        tiles.append(tile)
        oc0 += tile_oc
    return tiles


def _split_conv2d_s8_oh(op: dict[str, Any], n_splits: int,
                        network: str, tile_sizes=None) -> list[dict[str, Any]]:
    """Split a conv2d_s8 op along the OH (output ROW) dim.

    WHY THIS AXIS EXISTS AT ALL. `_split_conv2d_s8` tiles OC, and OC is
    quantized by the backend's blocking factor -- 32 for rvv (`TILE_OC` is
    clamped to `vsetvlmax_e32m4()`, kernels/rvv/
    rvv_conv2d_s8_rvv_vsmul_vnclip.c), 16 for gemmini (systolic DIM). A conv
    whose OC IS that quantum (dronet has four at OC=32) cannot be OC-split at
    all without paying for a second full slab: `ceil(32/32) = 1` unsplit, 2 for
    any 2-way partition. OH carries no such quantum, so an OH split of the same
    conv keeps every tile a full-width slab.

    WHAT MAKES IT DIFFERENT FROM AN OC SPLIT, and every one of these is a trap:

      * **The tiles' INPUTS overlap.** Output rows [r0, r1) need input rows
        [r0*SH - PH, (r1-1)*SH + KH - 1 - PH]. Adjacent tiles therefore share
        `KH - SH` input rows -- the HALO -- so the input is read more than once
        and the duplication factor is `window_rows / tile_rows`, not 1.0. The
        OUTPUTS stay disjoint and no accumulation is reordered, which is what
        keeps this a plain output-space split rather than a reduction split.

      * **Padding is per-tile.** Only the first tile gets the conv's top
        padding and only the last gets its bottom padding. One `PH` cannot say
        that -- the kernel signature pads BOTH ends by PH -- so this splitter
        does not try. It reports the window in PARENT input-row coordinates
        (`in_row_lo`, which is NEGATIVE for the first tile) and sets the tile's
        own `PH` to 0; the codegen materialises the out-of-range rows as
        explicit zeros. A zero row contributes `(0 + input_offset) * (w +
        filter_offset)`, which is exactly what the kernel's own out-of-bounds
        branch contributes, so the two are equivalent by construction rather
        than by luck.

      * **`PW` goes to 0 with it.** Not because the width needs splitting, but
        because gemmini's curated conv refuses `PH != PW` and falls back to a
        scalar loop (kernels/gemmini/gemmini_conv2d_s8_gemmini_tiled_conv.c:122
        -- `KH != KW || SH != SW || PH != PW`). A tile with PH=0 and PW=1 would
        build, run, verify -- and quietly leave the systolic array idle. So the
        codegen materialises the column padding too and the tile is a genuine
        zero-padding conv on a pre-padded window.

    NOT SLICED: the weights. Every tile computes every output channel from the
    full filter bank, so unlike the OC path there is no per-tile weight array
    and no layout hazard (`split_conv_tile_weights` deliberately matches only
    `axis == "OC"`).

    The output tensor `[N, OC, OH, OW]` is NCHW, so a row band is NOT a
    contiguous slice of it -- it is `OC` separate runs of `tile_oh*OW`
    elements, one per channel plane. That is the one place an OH tile is
    strictly more expensive to express than an OC tile, and it is why the tile
    output aliases the parent at offset 0 and the codegen scatters, instead of
    aliasing at a per-tile offset the way OC does.
    """
    shape = op["shape"]
    IH, IW = int(shape["IH"]), int(shape["IW"])
    KH, SH, PH = int(shape["KH"]), int(shape["SH"]), int(shape["PH"])
    if int(shape.get("DH", 1)) != 1 or int(shape.get("DW", 1)) != 1:
        raise SplitHintError(
            f"{network}: {op.get('name')} has dilation "
            f"DH={shape.get('DH')} DW={shape.get('DW')}; the conv2d_s8 kernel "
            f"signature carries no dilation, so the halo this splitter derives "
            f"would understate the input window")
    OH = int(shape.get("OH") or ((IH + 2 * PH - KH) // SH + 1))
    widths = _tile_widths(OH, n_splits, tile_sizes, network, "conv2d_s8 OH")
    out_tensor = op["outputs"][0]
    tiles: list[dict[str, Any]] = []
    oh0 = 0
    for t, tile_oh in enumerate(widths):
        # Input row window this tile reads, in PARENT coordinates. `lo` is
        # negative wherever the conv's top padding falls inside the window
        # (always, for tile 0 of a padded conv), and `lo + window_rows` runs
        # past IH at the bottom of the last tile. Both are intentional: the
        # window is the padded extent, and the codegen zero-fills the parts the
        # parent tensor does not have.
        lo = oh0 * SH - PH
        window_rows = (tile_oh - 1) * SH + KH
        pad_top = max(0, -lo)
        pad_bot = max(0, lo + window_rows - IH)
        in_rows = window_rows - pad_top - pad_bot
        if in_rows <= 0:
            raise SplitHintError(
                f"{network}: {op.get('name')} OH tile {t} (rows "
                f"[{oh0}, {oh0 + tile_oh})) reads no real input row -- its "
                f"window [{lo}, {lo + window_rows}) misses [0, {IH}). The "
                f"partition is degenerate for this conv's stride/padding.")
        tile = copy.deepcopy(op)
        tile["shape"] = dict(shape)
        # The tile IS a conv over the pre-padded window: IH becomes the window
        # height and the padding becomes zero, which reproduces exactly
        # `tile_oh` output rows -- (window_rows - KH)/SH + 1 == tile_oh by the
        # definition of window_rows above. Recording the geometry this way (and
        # not as "parent shape plus a note") means a cost model that multiplies
        # the shape out gets the tile's real work without knowing about OH
        # splitting at all.
        tile["shape"]["IH"] = window_rows
        tile["shape"]["PH"] = 0
        tile["shape"]["OH"] = tile_oh
        # PW follows PH to zero for gemmini's square-padding guard; IW grows to
        # the padded width so OW is unchanged. See the docstring.
        PW = int(shape["PW"])
        tile["shape"]["IW"] = IW + 2 * PW
        tile["shape"]["PW"] = 0
        tile["outputs"] = [f"{out_tensor}.tile_{t}"]
        tile["name"] = op["name"] + f".tile_{t}"
        tile["split_from"] = {
            "op_id": op["dispatch_id"], "tile": t, "n_splits": len(widths),
            "axis": "OH",
            "tile_oh": tile_oh, "tile_offset_OH": oh0,
            # Window in parent input rows. `in_row_lo` may be negative.
            "in_row_lo": lo, "window_rows": window_rows,
            "pad_top": pad_top, "pad_bot": pad_bot, "in_rows": in_rows,
            # Parent geometry the codegen needs for the gather/scatter strides;
            # the tile's own `shape` no longer carries it.
            "parent_IH": IH, "parent_IW": IW, "parent_OH": OH,
            "parent_PH": PH, "parent_PW": int(shape["PW"]),
        }
        tiles.append(tile)
        oh0 += tile_oh
    return tiles

def _split_elementwise_n(op: dict[str, Any], n_splits: int,
                         network: str, tile_sizes=None) -> list[dict[str, Any]]:
    """Split a POINTWISE op along its flat element count `n`.

    The cheapest split in the system, and the last one to be built. A pointwise
    op's IR shape is a single `n` -- `{"n": 100352}`, not `{N, C, H, W}` --
    because output element `i` is a function of input element `i` alone. So a
    tile is a contiguous element RANGE: the same kernel, the same scalar quant
    parameters, `n` narrowed, and every pointer advanced by the tile's offset.
    There is no weight to slice, no per-channel array to keep in step, and no
    halo, which is why this splitter is ten lines where `_split_conv2d_s8_oh`
    is a hundred.

    WHY IT IS LAYOUT-SAFE, uniquely among the axes here. `OC` is a contiguous
    plane range only in NCHW and `OH` only in NHWC, so both carry a guard that
    refuses the wrong layout (see `apply_split_hint`). A flat element range is
    contiguous under EVERY layout -- permuting the axes of a tensor does not
    change which elements a `[lo, hi)` byte range covers, only which logical
    coordinates they carry, and a pointwise op does not read coordinates. That
    is the same property `act_layout.LAYOUT_AGNOSTIC_OPS` records for these
    kinds, arrived at from the other direction.

    WHAT IT IS NOT SAFE FOR, and the reason the call site guards the operands
    rather than trusting this registry alone: a kernel whose signature is
    `(in, out, n, ...)` is not necessarily pointwise. `kernel_mse_loss` and
    `kernel_frobenius_norm` have exactly that shape and REDUCE -- their output
    is one scalar, so a "tile" would have both halves writing element 0 and the
    result would be the last tile's partial sum. `_assert_flat_pointwise`
    rejects those by checking the operands' extents against `n`, which catches
    a reduction and a broadcast without either being enumerated.

    Tile boundaries are element counts, not bytes: every buffer here is a typed
    pointer (`int8_t *`, `_Float16 *`), so `buf + off` advances by `off`
    ELEMENTS. Computing a byte offset and adding it is the vint `conv2d_f16`
    bug in a new place.
    """
    shape = op["shape"]
    n = int(shape["n"])
    widths = _tile_widths(n, n_splits, tile_sizes, network, f"{op['op']} n")
    out_tensor = op["outputs"][0]
    tiles: list[dict[str, Any]] = []
    n0 = 0
    for t, tile_n in enumerate(widths):
        tile = copy.deepcopy(op)
        tile["shape"] = dict(shape)
        tile["shape"]["n"] = tile_n
        tile["outputs"] = [f"{out_tensor}.tile_{t}"]
        tile["name"] = op["name"] + f".tile_{t}"
        tile["split_from"] = {"op_id": op["dispatch_id"], "tile": t,
                              "n_splits": len(widths), "tile_n": tile_n,
                              "tile_offset_E": n0,
                              # Parent extent, so a cost model can price an
                              # UNEVEN partition proportionally instead of
                              # falling back to "every tile is 1/n" -- which is
                              # the whole point of allowing uneven tiles.
                              "parent_n": n,
                              "axis": "E"}
        tiles.append(tile)
        n0 += tile_n
    return tiles


def _split_pool2d_c(op: dict[str, Any], n_splits: int,
                    network: str, tile_sizes=None) -> list[dict[str, Any]]:
    """Split a 2-D pooling op along the channel dim `C`.

    Arithmetically this is `_split_conv2d_s8`'s OC split with the weights taken
    away: NCHW makes a channel range a whole run of planes, every output plane
    depends only on the input plane beneath it, and no accumulation is
    reordered. It is strictly SIMPLER than the conv case -- a pool has no
    filter bank, so `split_conv_tile_weights` has nothing to repack and the
    whole layout question that forces the rvv convs into per-tile arrays does
    not arise.

    WHY IT IS AXIS "C" AND NOT "OC", which matters downstream rather than here.
    A pool's shape key is `C`: input and output channel counts are the same
    number and the IR names it once. Registering it under "OC" would have made
    `profile_loader` cost it with the OUTPUT-CHANNEL SLAB model, which asks
    "how many 32-wide vector slabs is this tile" -- meaningful for a conv whose
    inner loop is blocked on `vsetvlmax_e32m4`, meaningless for a pool that
    walks channels one at a time. On DroNet's `maxpool2d_s8` (C=32, the rvv
    quantum exactly) the slab model answers that both halves of a 2-way split
    cost a full slab each, i.e. that splitting the single most expensive
    gemmini op in the network buys nothing. A pool's cost is linear in C, and a
    separate axis is how the cost model gets told that.

    The layout guard still applies and is applied to "C" by name at the call
    site: a channel range is contiguous in NCHW and strided in NHWC, exactly as
    for a conv's OC.
    """
    shape = op["shape"]
    C = int(shape["C"])
    widths = _tile_widths(C, n_splits, tile_sizes, network, f"{op['op']} C")
    out_tensor = op["outputs"][0]
    tiles: list[dict[str, Any]] = []
    c0 = 0
    for t, tile_c in enumerate(widths):
        tile = copy.deepcopy(op)
        tile["shape"] = dict(shape)
        tile["shape"]["C"] = tile_c
        tile["outputs"] = [f"{out_tensor}.tile_{t}"]
        tile["name"] = op["name"] + f".tile_{t}"
        tile["split_from"] = {"op_id": op["dispatch_id"], "tile": t,
                              "n_splits": len(widths), "tile_c": tile_c,
                              "tile_offset_C": c0,
                              "parent_C": C,
                              "axis": "C"}
        tiles.append(tile)
        c0 += tile_c
    return tiles


#: Sub-op shape keys that name an output-channel count, in the order they are
#: tried. A fused conv's constituents each describe the same tensor in their
#: own vocabulary: the conv says `OC`, the batchnorm says `C`, the elementwise
#: activation says `n` (a flat element count). Narrowing only `sub_ops[0]` and
#: leaving the rest at full width would put a graph on disk whose own
#: constituents disagree about how wide the tile is.
_CHANNEL_SHAPE_KEYS = ("OC", "C")


def _narrow_sub_shape(sub: dict[str, Any], full_oc: int, tile_oc: int) -> None:
    """Narrow one fused constituent's shape from `full_oc` channels to `tile_oc`.

    Only narrows what it can verify: a channel key must currently read
    `full_oc`, and a flat element count must divide by it. Anything else is
    left alone rather than scaled on a guess -- a shape that does not describe
    this tensor is not made truer by rewriting it.
    """
    shape = sub.get("shape")
    if not isinstance(shape, dict):
        return
    shape = dict(shape)
    for key in _CHANNEL_SHAPE_KEYS:
        if int(shape.get(key, -1)) == full_oc:
            shape[key] = tile_oc
            sub["shape"] = shape
            return
    n = shape.get("n")
    if isinstance(n, int) and n > 0 and n % full_oc == 0:
        shape["n"] = n // full_oc * tile_oc
        sub["shape"] = shape


def _split_fused_conv_s8(op: dict[str, Any], n_splits: int,
                         network: str, tile_sizes=None) -> list[dict[str, Any]]:
    """Split a fused conv->BN[->SiLU] along OC.

    The ARITHMETIC is `_split_conv2d_s8`'s, unchanged: OC is an output axis,
    so slicing it partitions output elements and reorders no accumulation, and
    the epilogue parameters are 1-D per-output-channel (`bn_scale`, `bn_bias`,
    `bias`) so their slice is a plain `+ oc0`. That is not an assumption -- it
    is the same slicing the intra-op SHARD path already performs at runtime and
    that was measured bit-exact on the board; see
    `generate_skeleton._CONV2D_BN_SILU_S8_SHARD_WRAPPER`.

    What differs is BOOKKEEPING, and each difference is a trap that the plain
    conv path never has to see:

      * **A fused op has no `shape` of its own.** The geometry lives on the
        conv sub-op. Reading `op["shape"]` here is the same mistake that wrote
        `noshape` for 57 of yolov8n's dispatches into a profile.
      * **The constituents' tensors are internal names**, and every tile
        deep-copies them. Two tiles both claiming to produce `l0_conv` is a
        graph that says one tensor has two producers. Names that a sibling
        sub-op produces are suffixed per tile; the op's real INPUT is not,
        because every tile reads the same input.
      * **The constituents describe the channel count three different ways**
        (`OC`, `C`, `n`), so narrowing only the conv leaves the graph
        internally inconsistent -- see `_narrow_sub_shape`.

    This does NOT unfuse anything: the tile is still one fused op running one
    curated fused kernel, just over `OC/n` channels. The epilogue fusion that
    makes `conv2d_batchnorm2d_silu_s8` 97% of yolov8n is preserved.
    """
    sub_ops = op.get("sub_ops") or []
    conv = next((s for s in sub_ops
                 if str(s.get("op", "")).startswith("conv2d") and s.get("shape")),
                None)
    if conv is None:
        raise SplitHintError(
            f"{network}: {op['op']} (dispatch_id={op.get('dispatch_id')}) "
            f"carries no conv sub_op with a shape. A fused op has no shape of "
            f"its own -- the geometry lives under sub_ops -- so there is no OC "
            f"to tile along")
    full_oc = int(conv["shape"]["OC"])
    widths = _tile_widths(full_oc, n_splits, tile_sizes, network,
                          f"{op['op']} OC")
    out_tensor = op["outputs"][0]
    # Produced by a sibling constituent => internal to this fused op => must
    # not be shared across tiles. The op's own input is produced elsewhere and
    # is deliberately absent from this set.
    internal = {t for s in sub_ops for t in (s.get("outputs") or [])}
    tiles: list[dict[str, Any]] = []
    oc0 = 0
    for t, tile_oc in enumerate(widths):
        tile = copy.deepcopy(op)
        suffix = f".tile_{t}"
        for sub in tile["sub_ops"]:
            for field in ("inputs", "outputs"):
                sub[field] = [nm + suffix if nm in internal else nm
                              for nm in (sub.get(field) or [])]
            _narrow_sub_shape(sub, full_oc, tile_oc)
        tile["outputs"] = [f"{out_tensor}{suffix}"]
        tile["name"] = op["name"] + suffix
        tile["split_from"] = {"op_id": op["dispatch_id"], "tile": t,
                              "n_splits": len(widths), "tile_oc": tile_oc,
                              "tile_offset_OC": oc0,
                              "axis": "OC"}
        tiles.append(tile)
        oc0 += tile_oc
    return tiles


#: Pointwise kinds an `E` (flat element) split can be built for.
#:
#: Derived mechanically, not by taste: every kind here has a branch in
#: `generate_skeleton`'s dispatch chain that reads `op["shape"]["n"]` and takes
#: each operand through `ptr_for`, which is exactly the shape the E emitter
#: needs -- narrow `n`, advance the pointers, change nothing else. That is why
#: no per-kind emitter work was needed to add all 47 of them at once, and why
#: adding the 48th is a one-line change here IF it has such a branch.
#:
#: DELIBERATELY ABSENT: `frobenius_norm`, `mse_loss`, `hinge_loss`,
#: `huber_loss`. Their kernels have the identical `(in, out, n, ...)`
#: signature and they REDUCE -- one scalar out, not `n` -- so tiling them
#: would have both halves writing element 0 and the answer would be whichever
#: tile finished last. Their exclusion is enforced twice: they are not listed,
#: and `_assert_flat_pointwise` re-derives the property from the operands'
#: extents at split time so a future addition cannot reintroduce one silently.
#:
#: ALSO ABSENT: `mul_c1_*`. Flat-looking, but the `c1` is a per-CHANNEL
#: broadcast -- one operand has `C` elements, not `n`.
_POINTWISE_KINDS: tuple[str, ...] = (
    "add", "add_f16", "add_s8", "cast_f16_to_i8",
    "cast_i8_to_f16", "elu", "elu_f16", "elu_s8",
    "gelu", "gelu_exact", "gelu_exact_f16", "gelu_f16",
    "gelu_s8", "hardsigmoid", "hardsigmoid_f16", "hardtanh",
    "hardtanh_f16", "leaky_relu", "leaky_relu_f16", "leaky_relu_s8",
    "log", "log_f16", "mul", "mul_f16",
    "mul_s8", "relu", "relu6", "relu6_f16",
    "relu6_s8", "relu_f16", "relu_s8", "selu",
    "selu_f16", "sigmoid", "sigmoid_f16", "sigmoid_s8",
    "silu", "silu_f16", "silu_s8", "softplus",
    "softplus_f16", "softsign", "softsign_f16", "swish",
    "swish_f16", "tanh", "tanh_f16",
)

#: Pooling kinds a `C` (channel) split can be built for. Only the int8 maxpool
#: today, because that is the one with a `C`-split emitter in
#: `generate_skeleton` (`_C_SLICEABLE_POOL_OPS`) -- `upsample_nearest_s8` is
#: the same shape of problem and is NOT here, because its output extent is
#: `IH*scale` and it needs its own two lines in that emitter first.
_POOL_C_KINDS: tuple[str, ...] = ("maxpool2d_s8",)


#: Op kinds an OC/N split can be built for. The fused convs are here for the
#: same reason they are in `generate_skeleton._SHARDABLE_CONV_OPS`: that is
#: where the time is. `conv2d_batchnorm2d_silu_s8` alone is 97% of yolov8n and
#: `conv2d_batchnorm2d_s8` is 29% of DroNet, so a split path that refuses them
#: can only ever tile ops too cheap for the tiling to matter.
_SPLITTABLE: dict[str, Any] = {
    "linear_s8": _split_linear_s8,
    # fp32 `linear` uses the SAME splitter: `_split_linear_s8` only reads
    # shape["N"], narrows it, renames the output and records tile metadata --
    # it touches no dtype, no quant params and no weight layout, so it is a
    # generic N-splitter that happens to be named for its first caller.
    # NOTE for anyone adding another dtype here: registering the op is NOT
    # sufficient on its own. generate_skeleton gates BOTH the output offset
    # alias and the weight/bias pointer offset on the op kind; a kind that is
    # split but missing from _N_SLICEABLE_LINEAR_OPS there builds cleanly and
    # computes garbage (every tile writes at offset 0 and reads weight row 0).
    "linear": _split_linear_s8,
    # fp16 kinds ride the same shape-only splitters. Safe for exactly the
    # reasons spelled out above _SPLITTABLE_BY_AXIS, and the codegen half of the
    # registration is done too: conv2d_f16 is in _OC_SLICEABLE_CONV_OPS and
    # linear_f16 in _N_SLICEABLE_LINEAR_OPS in generate_skeleton, which is what
    # the NOTE above warns is required and not optional.
    "linear_f16": _split_linear_s8,
    "conv2d_s8": _split_conv2d_s8,
    "conv2d_f16": _split_conv2d_s8,
    "conv2d_batchnorm2d_s8": _split_fused_conv_s8,
    "conv2d_batchnorm2d_silu_s8": _split_fused_conv_s8,
    # Pointwise and pooling, added together because they are the two kinds of
    # op that were left unsplittable once the conv work landed and that
    # DOMINATE what is left: silu_s8 is 57 of yolov8n's 155 dispatches and
    # maxpool2d_s8 is 28.9% of DroNet's gemmini time.
    **{k: _split_elementwise_n for k in _POINTWISE_KINDS},
    **{k: _split_pool2d_c for k in _POOL_C_KINDS},
}


#: The axis each kind is split along when a hint does not name one. Kept
#: separate from `_SPLITTABLE` so the DEFAULT and the SET OF CHOICES are two
#: facts, not one: a hint that says nothing still gets OC for a conv (every
#: existing hint on disk relies on that), while one that says `"axis": "OH"`
#: gets the spatial splitter.
_DEFAULT_AXIS: dict[str, str] = {
    "linear_s8": "N",
    "linear": "N",
    "linear_f16": "N",
    "conv2d_s8": "OC",
    "conv2d_f16": "OC",
    "conv2d_batchnorm2d_s8": "OC",
    "conv2d_batchnorm2d_silu_s8": "OC",
    **{k: "E" for k in _POINTWISE_KINDS},
    **{k: "C" for k in _POOL_C_KINDS},
}

#: axis -> {op kind: splitter}. Only conv2d_s8 has an OH splitter: the fused
#: convs are excluded deliberately rather than by oversight, because their
#: epilogue geometry (`bn_scale`/`bn_bias` are per-CHANNEL, and an OH tile
#: keeps every channel) is fine but their codegen path has no row-window
#: emitter -- and a kind registered here without one in generate_skeleton is
#: exactly the "builds cleanly, computes garbage" failure that
#: `_N_SLICEABLE_LINEAR_OPS` warns about.
# conv2d_f16 / linear_f16 are registered against the SAME splitters as their s8
# counterparts because the splitters are shape-only -- they set shape["OC"] (or
# ["N"]) and record split_from; no operand is touched, and
# generate_skeleton.split_conv_tile_weights selects tiles by
# `split_from.axis == "OC"` rather than by op kind. Two things make this safe
# for f16 specifically and NOT for the _pc kinds:
#
#   * conv2d_f16's signature is (input, weight, bias, output, N, IC, ...) --
#     structurally identical to conv2d_s8, with no per-channel array.
#   * _conv_weight_layout_for_op("conv2d_f16", <any backend>) is None, i.e.
#     plain OIHW, where an OC slice IS contiguous.
#
# conv2d_s8_pc and linear_s8_pc are deliberately NOT registered. They carry
# per-output-channel scale arrays, and nothing in split_conv_tile_weights slices
# a scale -- it handles weight and bias only. A _pc tile would therefore index
# the PARENT's full scale vector from 0 and apply the wrong scale to every
# channel past the first tile: numerically wrong, size-identical, and silent.
# Registering them needs per-channel slicing plus a test that catches exactly
# that, first. See the log entry `sweep3net_vint_not_shardable`.
_SPLITTABLE_BY_AXIS: dict[str, dict[str, Any]] = {
    "N": {"linear_s8": _split_linear_s8, "linear": _split_linear_s8,
          "linear_f16": _split_linear_s8},
    "OC": {"conv2d_s8": _split_conv2d_s8,
           "conv2d_f16": _split_conv2d_s8,
           "conv2d_batchnorm2d_s8": _split_fused_conv_s8,
           "conv2d_batchnorm2d_silu_s8": _split_fused_conv_s8},
    "OH": {"conv2d_s8": _split_conv2d_s8_oh},
    # Flat element range: the only axis with no layout precondition, because a
    # contiguous byte range is contiguous under every permutation of the axes.
    "E": {k: _split_elementwise_n for k in _POINTWISE_KINDS},
    # Channel range on a pool. Separate from "OC" so the cost model does not
    # apply the output-channel slab quantum to it -- see `_split_pool2d_c`.
    "C": {k: _split_pool2d_c for k in _POOL_C_KINDS},
}


def _assert_flat_pointwise(graph: dict[str, Any], op: dict[str, Any],
                           network: str) -> None:
    """Refuse an E split unless every operand really is `n` elements wide.

    `_POINTWISE_KINDS` is a list, and a list is a claim someone has to keep
    true. This re-derives the claim from the graph instead: if the op is
    pointwise then each of its activation operands holds exactly `shape["n"]`
    elements, so any operand whose declared shape multiplies out to something
    else means the op is a reduction (output smaller), a broadcast (one input
    smaller), or simply mis-shaped -- and in all three cases a tile would
    address memory the parent never owned, or two tiles would address the same
    element.

    Skips operands the IR gives no shape for rather than guessing. That makes
    it a check on what is knowable, not a completeness requirement: a graph
    with no `tensors` entry for an operand loses the check for that operand,
    which is the same position the conv splitters are in for all of theirs.
    """
    tensors = graph.get("tensors") or {}
    n = int(op["shape"]["n"])
    for role in ("inputs", "outputs"):
        for name in (op.get(role) or []):
            shape = (tensors.get(name) or {}).get("shape")
            if not shape:
                continue
            extent = 1
            for d in shape:
                extent *= int(d)
            if extent != n:
                raise SplitHintError(
                    f"{network}: {op.get('name')} ({op['op']}, "
                    f"dispatch_id={op.get('dispatch_id')}) cannot be split "
                    f"along E: its {role[:-1]} {name!r} has {extent} elements "
                    f"but the op declares n={n}. A flat split assumes every "
                    f"operand is the same {n} elements, so this op either "
                    f"reduces, broadcasts, or is mis-shaped -- and tiling it "
                    f"would read or write outside the tile's own range.")


def _resolve_splitter(op_kind: str, axis, network: str, op_id):
    """Pick the splitter for (op kind, requested axis), or say why there is none."""
    if axis is None:
        axis = _DEFAULT_AXIS.get(op_kind)
    if axis not in _SPLITTABLE_BY_AXIS:
        raise SplitHintError(
            f"{network}: split_ops[dispatch_id={op_id}] asks for axis "
            f"{axis!r}; known axes are {sorted(_SPLITTABLE_BY_AXIS)}")
    by_kind = _SPLITTABLE_BY_AXIS[axis]
    if op_kind not in by_kind:
        raise SplitHintError(
            f"{network}: op kind {op_kind!r} (dispatch_id={op_id}) cannot be "
            f"split along {axis}; that axis supports {sorted(by_kind)}. "
            f"Split-capable kinds overall: {sorted(_SPLITTABLE)}")
    return by_kind[op_kind], axis


def _register_tile_tensors(graph: dict[str, Any], op: dict[str, Any],
                           widths: list[int], axis: str) -> None:
    """When we split a linear_s8 op along N, the per-tile output names
    (`<orig>.tile_<i>`) need entries in the IR's tensors dict so the
    downstream skeleton emitter can allocate buffers for them. The
    tile shape is the parent op's output shape with the last dim
    (N) replaced by tile_n.

    Without this, generate_skeleton.py walks `graph["tensors"]` to
    emit `buf_<network>_<tensor>[size]` definitions, the tile tensors
    aren't found, and `model.c` references undeclared `buf_..._tile_0`
    at link time."""
    tensors = graph.setdefault("tensors", {})
    out_name = op["outputs"][0]
    parent = tensors.get(out_name)
    if not parent or "shape" not in parent:
        return  # nothing to do — IR has no shape info for this tensor
    parent_shape = list(parent["shape"])
    if not parent_shape:
        return
    # Which dim the tile width narrows is the SPLIT AXIS, not the op kind: an
    # OC split narrows dim 1 of NCHW, an N split narrows the last dim. Keying
    # it on the op kind is why this used to name `conv2d_s8` explicitly, which
    # silently sent every fused conv down the linear branch and narrowed W.
    if axis == "E":
        # A flat element range has no dim to narrow -- it is a run of `tile_n`
        # elements out of the parent's flattened storage, and the parent's rank
        # is irrelevant to it (that is what makes the axis layout-free). Record
        # the tile as the 1-D thing it is rather than inventing a 4-D shape
        # that would only be true when the partition happened to fall on a
        # plane boundary.
        off = 0
        for t, tile_n in enumerate(widths):
            _add_tile_tensor(tensors, parent, out_name, t, len(widths),
                             [tile_n], off)
            off += tile_n
        return
    if axis in ("OC", "C") and len(parent_shape) >= 4:
        axis_dim = 1                       # [N, C, OH, OW]
    elif axis == "OH" and len(parent_shape) >= 4:
        axis_dim = 2                       # [N, OC, OH, OW]
    else:
        axis_dim = len(parent_shape) - 1   # linear [M, N]
    # Element offset of each tile within the parent buffer. With EVEN tiles
    # this is just `t * prod(tile_shape)`, which is what the consumer used to
    # recompute; with UNEVEN tiles that is wrong for every tile after the
    # first, so the offset is recorded here where the partition is known.
    stride_per_unit = 1
    for d in parent_shape[axis_dim + 1:]:
        stride_per_unit *= int(d)
    off_units = 0
    n_splits = len(widths)
    for t, tile_n in enumerate(widths):
        tile_shape = list(parent_shape)
        tile_shape[axis_dim] = tile_n
        _add_tile_tensor(tensors, parent, out_name, t, n_splits, tile_shape,
                         0 if axis == "OH" else off_units * stride_per_unit,
                         alias_kind="row_window" if axis == "OH" else None)
        off_units += tile_n


def _add_tile_tensor(tensors, parent, out_name, t, n_splits, tile_shape,
                     elem_offset, alias_kind=None) -> None:
    """Register one `<parent>.tile_<t>` entry. Never overwrites an existing one."""
    tile_name = f"{out_name}.tile_{t}"
    if tile_name not in tensors:
        entry = {
            "shape": list(tile_shape),
            "dtype": parent.get("dtype", "i8"),
            "split_from": out_name,
            "tile": t,
            "n_splits": n_splits,
            # An OC/N tile is a CONTIGUOUS block of the parent, so it can
            # be expressed as a pointer offset and the tile simply writes
            # there. An OH tile is not: in NCHW a band of rows is `OC`
            # separate runs, one per channel plane. So it aliases the
            # parent at 0 and the codegen scatters row-band by row-band --
            # recording any other offset here would put every tile's first
            # channel plane at the wrong place.
            "elem_offset": int(elem_offset),
        }
        if alias_kind:
            entry["alias_kind"] = alias_kind
        # Carry the parent's quantization across. A tile is a slice of the
        # parent tensor -- same scale, same zero point, by construction --
        # and anything reading the tile's own metadata (the codegen's
        # inspect blocks, an accuracy comparison against a golden, a
        # downstream requantize) otherwise falls back to scale=1.0 and
        # silently reports raw int8 counts as physical values.
        if parent.get("quant") is not None:
            entry["quant"] = copy.deepcopy(parent["quant"])
        tensors[tile_name] = entry


def apply_split_hint(graph: dict[str, Any],
                     split_ops: list[dict[str, Any]]) -> dict[str, Any]:
    """Rewrite `graph` so each op listed in `split_ops` is decomposed
    into `n_splits` tile ops. Returns a new dict; input is not mutated.
    """
    out = copy.deepcopy(graph)
    if not split_ops:
        return out
    network = out.get("name", "<unknown>")
    ops = out["ops"]
    ops_by_id = _ops_by_id(ops)

    # Validate everything first; reject if any op is unsupported.
    for spec in split_ops:
        op_id = spec["op"]
        n = int(spec.get("n_splits", 2))
        if spec.get("tile_sizes"):
            n = len(spec["tile_sizes"])
        if op_id not in ops_by_id:
            raise SplitHintError(
                f"{network}: split_ops references unknown dispatch_id {op_id}")
        op = ops_by_id[op_id]
        if op["op"] not in _SPLITTABLE:
            raise SplitHintError(
                f"{network}: op kind {op['op']!r} (dispatch_id={op_id}) "
                f"not yet split-capable; supported kinds: "
                f"{sorted(_SPLITTABLE)}")
        # Resolve here as well as at rewrite time so an unsupported (kind,
        # axis) pair is rejected BEFORE any op is rewritten -- the validate
        # pass exists so a bad hint leaves the graph untouched.
        _resolve_splitter(op["op"], spec.get("axis"), network, op_id)
        if n < 2:
            raise SplitHintError(
                f"{network}: n_splits={n} must be >= 2 to split")

    # Single-pass rewrite: walk in original order, replacing split ops
    # with their tile lists. Re-assign dispatch_ids contiguously after.
    target_ids = {s["op"]: (int(s.get("n_splits", 2)), s.get("tile_sizes"),
                            s.get("axis"))
                  for s in split_ops}
    new_ops: list[dict[str, Any]] = []
    id_remap: dict[int, list[int]] = {}  # original -> [new tile ids]
    next_new_id = 0
    for op in ops:
        did = op.get("dispatch_id")
        if did is None:
            new_ops.append(copy.deepcopy(op))
            continue
        if did in target_ids:
            n, tile_sizes, want_axis = target_ids[did]
            # An OC slice is a pointer offset ONLY in NCHW, where a channel is a
            # whole plane. Under NHWC the channel is innermost, so an OC slice
            # becomes STRIDED -- and the codegen's alias for it
            # (`elem_offset = oc0 * OH * OW` in generate_skeleton) would hand the
            # tile a contiguous window that is not the data it was promised.
            # Same bytes, same size, plausible wrong answer.
            #
            # OH is the mirror image: strided in NCHW (hence the gather/scatter
            # the OH wrapper carries), contiguous in NHWC. Layout does not remove
            # the split tax, it MOVES it -- docs/IR_TENSOR_LAYOUT_DESIGN.md §6.
            #
            # This guard lands in stage 2, BEFORE any NHWC tensor can exist,
            # precisely so it can never become the thing someone debugs later.
            eff_axis = want_axis or _DEFAULT_AXIS.get(op["op"])
            if eff_axis == "E":
                _assert_flat_pointwise(out, op, network)
            # "C" (a pool's channel range) is the same contiguity claim as a
            # conv's "OC" and gets the same guard: both name a run of whole
            # NCHW planes, and both become strided the moment the tensor is
            # nhwc.
            if eff_axis in ("OC", "C"):
                _out = (op.get("outputs") or [None])[0]
                _lay = ((graph.get("tensors") or {}).get(_out) or {}).get(
                    "layout", "nchw")
                if _lay != "nchw":
                    raise SplitHintError(
                        f"{network}: {op.get('name')} cannot be "
                        f"{eff_axis}-split while "
                        f"its output declares layout={_lay!r}. A channel slice "
                        f"is a contiguous plane range only in nchw; under {_lay!r} "
                        f"the channel is innermost, so the tile's offset alias "
                        f"would select the wrong elements and return a "
                        f"plausible wrong answer. Split on OH instead "
                        f"(contiguous under nhwc), or keep this tensor nchw.")
            splitter, _axis = _resolve_splitter(op["op"], want_axis, network, did)
            tile_ops = splitter(op, n, network, tile_sizes)
            # Register the per-tile output tensors in the IR's tensors
            # dict so generate_skeleton can allocate per-tile buffers
            # (fixes "buf_<network>_<out>_tile_0 undeclared" build error).
            if tile_ops:
                # Read the tile width off the metadata the splitter RECORDED,
                # not off the tile's `shape`. A fused conv has no `shape` key
                # at all, so the old `"shape" in tile_ops[0]` gate skipped
                # tensor registration for it entirely -- silently, and the
                # failure surfaced much later as an undeclared
                # `buf_<net>_<out>_tile_0` at link time.
                sf0 = tile_ops[0].get("split_from") or {}
                axis = sf0.get("axis", "N")
                key = {"OC": "tile_oc", "OH": "tile_oh",
                       "C": "tile_c"}.get(axis, "tile_n")
                widths = [int((t.get("split_from") or {}).get(key, 0))
                          for t in tile_ops]
                if all(w > 0 for w in widths):
                    _register_tile_tensors(out, op, widths, axis)
            new_tile_ids: list[int] = []
            for tile in tile_ops:
                tile["dispatch_id"] = next_new_id
                new_tile_ids.append(next_new_id)
                new_ops.append(tile)
                next_new_id += 1
            id_remap[did] = new_tile_ids
        else:
            nop = copy.deepcopy(op)
            id_remap[did] = [next_new_id]
            nop["dispatch_id"] = next_new_id
            new_ops.append(nop)
            next_new_id += 1

    # Rewire depends_on: a downstream op that depended on original
    # split op `K` now depends on ALL of K's tiles (the original
    # output is the concat of the tiles, so the consumer can't start
    # until every tile finishes).
    for nop in new_ops:
        if nop.get("dispatch_id") is None:
            continue
        new_deps: list[int] = []
        for d in nop.get("depends_on", []):
            if d in id_remap:
                new_deps.extend(id_remap[d])
        # Deduplicate while preserving order.
        seen: set[int] = set()
        nop["depends_on"] = [x for x in new_deps if not (x in seen or seen.add(x))]

    out["ops"] = new_ops
    # Publish the renumbering. Inserting tiles shifts every id after the
    # split point, so ops that were NOT split still change identity —
    # without this, anything keyed on dispatch_id (profile CSVs, cost
    # DBs, SchedulerReports, Gantt labels) silently re-attaches to the
    # wrong op after a rewrite.
    #
    # Values are LISTS because splitting is one-to-many: a split op maps
    # to all of its tile ids, in tile order, and collapsing that to the
    # first tile would misreport the op's cost as one tile's cost.
    # Untouched ops get a single-element list so consumers never have to
    # branch on the value type. (`apply_fusion_hint.py` emits the same
    # field scalar-valued — fusion is many-to-one.)
    # `dispatches` is documented by extract_graph as "the ordered list of
    # dispatch_ids" (:57), and it was being carried through the rewrite
    # unchanged -- so every rewritten graph on disk describes the PRE-rewrite
    # dispatch set. Measured: a fused mlp_control graph with 4 ops still
    # claimed 7 dispatches; a split DroNet graph with 22 ops claimed 21.
    #
    # It is currently harmless on the main path because emit_dispatch_graph
    # walks `ops`, not this field. It is not harmless everywhere:
    # ModelBlaster/scripts/plot_frequency_sweep_v2.py:105 reads
    # `len(fx["dispatches"])` as the op count, which is wrong by exactly the
    # rewrite delta -- silently, and in a plot.
    if isinstance(out.get("dispatches"), list):
        out["dispatches"] = [o["dispatch_id"] for o in new_ops
                             if o.get("dispatch_id") is not None]
    out["id_remap"] = {str(old): list(new)
                       for old, new in sorted(id_remap.items())}
    return out


def _load_hint(path: Path) -> dict[str, Any]:
    hint = json.loads(path.read_text())
    if hint.get("contract") != SPLIT_CONTRACT:
        raise SplitHintError(
            f"hint contract is {hint.get('contract')!r}, expected "
            f"{SPLIT_CONTRACT!r}")
    return hint


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hint", required=True, type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--ir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)

    hint = _load_hint(args.hint)
    split_ops: list[dict[str, Any]] = []
    for entry in hint.get("networks", []):
        if entry.get("network") == args.model:
            split_ops = entry.get("split_ops", [])
            break
    if not split_ops:
        print(f"apply_split_hint: no split_ops for {args.model!r} in "
              f"{args.hint} — copying IR through unchanged", file=sys.stderr)

    graph = json.loads(args.ir.read_text())
    if graph.get("name") != args.model:
        graph["name"] = args.model

    rewritten = apply_split_hint(graph, split_ops)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rewritten, indent=2))
    n_split = sum(1 for op in rewritten["ops"] if "split_from" in op)
    n_total = sum(1 for op in rewritten["ops"]
                  if op.get("dispatch_id") is not None)
    print(f"apply_split_hint: wrote {args.out} "
          f"({n_split} tile ops from splits, {n_total} total dispatches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
