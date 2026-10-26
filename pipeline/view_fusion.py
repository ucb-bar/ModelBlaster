"""View-transparent producer -> elementwise fusion (A2 capabilities, item 6a; FEATURE_TRACKER idea C.4).

``propose_from_profile`` / ``fuse_pair`` fuse a producer into an elementwise consumer only when the consumer reads
the producer's output directly. A pure reshape in between (an IR ``view`` with offset 0 and the same element count)
blocks it although it changes nothing in memory: 31_VisionAttention's attention scale ``bmm -> view ->
ewise_scalar`` costs 12.6 ms of the model's 38 ms as a separate pass over the 2^31-element score tensor.

A *flat* elementwise op (its shape is just ``{"n"}``: relu, ewise_scalar, ewise1, add/mul of equal sizes, ...)
indexes its operands linearly, so it may read the view's base instead of the view. ``elide`` rewires such readers to
the base and drops views nobody reads any more; the ordinary fusion machinery then sees the pair.

Hint kind ``fuse_view`` (``--view-fusion on``; default off: never proposed, no graph changes): ``{"fused_kind",
"chains": [[producer, consumer], ...], "views": [view op names]}``. Applier: elide exactly those views (refused
if a view is no longer a pure reshape read only by flat elementwise ops), then the ``fuse`` applier on the chains.
Decided like any fusion (variant = composite seed of the stage winners; paired re-timing).
"""
from __future__ import annotations

import copy
import os

ENV = "MB_VIEW_FUSION"


def enabled(v: str | None = None) -> bool:
    v = (v if v is not None else os.environ.get(ENV, "off")).strip().lower()
    if v not in ("on", "off", "1", "0"):
        raise ValueError(f"view fusion must be on|off, not {v!r}")
    return v in ("on", "1")


def _numel(graph, t):
    sh = graph["tensors"].get(t, {}).get("shape")
    if sh is None:
        return None
    n = 1
    for d in sh:
        n *= int(d)
    return n


def flat_elementwise(op: dict, specs: dict | None = None) -> bool:
    """An elementwise op that indexes every operand linearly (shape keys exactly {"n"}; not fused)."""
    from .op_classes import op_class
    if "sub_ops" in op or op["op"].startswith("fused__"):
        return False
    return op_class(op["op"]) == "elementwise" and set((op.get("shape") or {}).keys()) == {"n"}


def transparent_views(graph: dict) -> list[str]:
    """Names of ``view`` ops that are pure reshapes read only by flat elementwise ops (and not graph outputs)."""
    outs = set(graph["output"].get("tensors") or [graph["output"]["tensor"]])
    readers: dict = {}
    for o in graph["ops"]:
        for t in o["inputs"]:
            readers.setdefault(t, []).append(o)
    res = []
    for o in graph["ops"]:
        if o["op"] != "view" or int(o.get("offset", 0)) != 0:
            continue
        src, dst = o["inputs"][0], o["outputs"][0]
        if dst in outs or _numel(graph, src) is None or _numel(graph, src) != _numel(graph, dst):
            continue
        r = readers.get(dst, [])
        if r and all(flat_elementwise(x) for x in r):
            res.append(o["name"])
    return res


def elide(graph: dict, names: list[str] | None = None) -> tuple[dict, list[str]]:
    """(new graph, elided view names): readers of the given transparent views (default: all) read the views' bases."""
    g = copy.deepcopy(graph)
    ok = set(transparent_views(g))
    want = ok if names is None else set(names)
    bad = want - ok
    if bad:
        raise ValueError(f"views {sorted(bad)} are not transparent here")
    done = []
    for o in [x for x in g["ops"] if x["name"] in want]:
        src, dst = o["inputs"][0], o["outputs"][0]
        for r in g["ops"]:
            if dst in r["inputs"]:
                r["inputs"] = [src if t == dst else t for t in r["inputs"]]
        g["ops"] = [x for x in g["ops"] if x is not o]
        done.append(o["name"])
    return g, done


INT32_MAX = 2 ** 31 - 1


def _fits_int32(*ops) -> bool:
    """Every scalar of the stages fits a C int (export chunks only plain flat elementwise calls past 2^31 elements:
    31_VisionAttention's 2^31-element scale op cannot be a fused stage)."""
    for o in ops:
        for v in (o.get("shape") or {}).values():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and abs(v) > INT32_MAX:
                return False
    return True


def propose(graph: dict, op_us: dict, specs: dict, *, doc: dict | None = None, min_gain_pct: float = 2.0,
            ir: str | None = None, launch_us: float = 0.0, total_us: float | None = None) -> list[dict]:
    """``fuse_view`` hints: the fusion proposals that only exist once the transparent views are elided."""
    from .graph_hints import decided, propose_from_profile, unique_names
    g0 = unique_names(copy.deepcopy(graph))
    views = transparent_views(g0)
    if not views:
        return []
    g1, _ = elide(g0, views)
    done = decided(doc) if doc else set()
    vset = set(views)
    by_out = {o["outputs"][0]: o for o in g0["ops"] if o["name"] in vset}       # view output -> view
    by_name = {o["name"]: o for o in g0["ops"]}
    out, skipped = [], []
    for h in propose_from_profile(g1, op_us, specs, min_gain_pct=min_gain_pct, ir=ir, launch_us=launch_us,
                                  total_us=total_us):
        chains, used = [], set()
        for c in h["chains"]:
            a, b = by_name.get(c[0]), by_name.get(c[-1])
            if a is None or b is None:
                continue
            vs = [by_out[t]["name"] for t in b["inputs"] if t in by_out and by_out[t]["inputs"][0] in a["outputs"]]
            if vs and not _fits_int32(a, b):
                skipped.append({"chain": c, "why": "a stage scalar >= 2^31: a fused op cannot be chunked like a flat "
                                                   "elementwise op (C int parameters)"})
                continue
            if vs:                                       # the pair exists only through a transparent view
                chains.append(c)
                used.update(vs)
        if not chains:
            continue
        hid = f"fuse_view:{h['fused_kind']}"
        if hid in done:
            continue
        out.append({"id": hid, "kind": "fuse_view", "fused_kind": h["fused_kind"], "chains": chains,
                    "views": sorted(used), "target": "cuda", "status": "proposed", "evidence": [],
                    "source": [{**x, "by": "mb-view-fusion"} for x in h.get("source", [])]})
    return out
