"""Which axis a pool-split ``conv2d_s8`` takes at a given core width.

A conv given ``w > 1`` cores is split either by OUTPUT CHANNELS (each core gets
``OC/w`` channels, its own packed weight slice, and the full input: the
original ``parallel_conv2d_s8_sharded`` path) or by OUTPUT ROWS (each core gets
``~OH/w`` rows and every channel: ``parallel_conv2d_s8_rows`` copies the core's
input band, calls the ordinary kernel on it and copies the rows back).

Measured on the K3 (k3-work TASK-016 step 3, 50 conv shapes of dronet and
yolov8_nano, widths 2/4/8, all four K3 backends): the channel split repeats the
kernels' input-side setup on every core (40-80% of a 3x3 conv on the IME
kernels; the RVV kernels vectorise over OC, so on the A100's 1024-bit vectors
an OC slice leaves lanes idle) and gained 0.7-1.9x, the row split 1.7-4.1x at
width 8. The row split loses only where there are too few output pixels for
the width: one ``min_pixels`` threshold (OH*OW) kept every backend within 10%
of the per-layer best.

The rule's parameters live in ``cores/codegen_contract.json``
(``rules.conv_split_axis``), which XPU-RT reads too, so the scheduler prices
exactly the widths codegen can build.
"""
from __future__ import annotations

import functools
import json
import os
from typing import Any

_CONTRACT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "cores", "codegen_contract.json")


@functools.lru_cache(maxsize=None)
def _rule() -> dict[str, Any]:
    with open(_CONTRACT) as f:
        r = json.load(f).get("rules", {}).get("conv_split_axis") or {}
    return {"ops": frozenset(r.get("applies_to_ops") or ()),
            "prefixes": tuple(r.get("backend_prefixes") or ()),
            "min_pixels": int(r.get("min_pixels", 0))}


def row_capable(backend: str | None) -> bool:
    """Does `backend` take the row split (a K3 backend, by name prefix)?"""
    return bool(backend) and any(str(backend).startswith(p) for p in _rule()["prefixes"])


def _out_hw(shape: dict[str, Any]) -> tuple[int, int] | None:
    """(OH, OW), or None when the shape does not say (then there is no row split)."""
    try:
        if "OH" in shape and "OW" in shape:
            return int(shape["OH"]), int(shape["OW"])
        oh = (int(shape["IH"]) + 2 * int(shape.get("PH", 0)) - int(shape["KH"])) // int(shape.get("SH", 1)) + 1
        ow = (int(shape["IW"]) + 2 * int(shape.get("PW", 0)) - int(shape["KW"])) // int(shape.get("SW", 1)) + 1
        return oh, ow
    except (KeyError, TypeError, ValueError):
        return None


def split_axis(op: str, shape: dict[str, Any], width: int,
               backend: str | None = None) -> str:
    """"row" or "oc" for `op` with `shape` (an IR conv shape dict) on `width` cores
    of `backend`. No backend, or one the rule does not cover: "oc", as before."""
    r = _rule()
    if width <= 1 or op not in r["ops"] or not row_capable(backend):
        return "oc"
    if os.environ.get("MB_CONV_SPLIT_AXIS") == "oc":
        return "oc"   # A/B measurements only: the channel split everywhere
    hw = _out_hw(shape or {})
    if hw is None:
        return "oc"
    oh, ow = hw
    return "row" if oh >= width and oh * ow >= r["min_pixels"] else "oc"


def width_buildable(op: str, shape: dict[str, Any], width: int,
                    backends: tuple[str, ...] | list[str] = ()) -> bool:
    """Can codegen split `op` over `width` cores for every backend in `backends`?
    A channel split needs width | OC; a row split needs no weight slicing. With
    no backends given only the channel split counts (the pre-row-split rule)."""
    if width <= 1:
        return True
    oc = int((shape or {}).get("OC", 0) or 0)
    if oc > 0 and oc % width == 0:
        return True
    return bool(backends) and all(split_axis(op, shape, width, b) == "row" for b in backends)
