"""Graph surgery for open-ended rewrites (docs/graph_rewrite.md): the edit handle a rewrite author uses, and the
replayable *patch* an accepted rewrite is stored as.

* :class:`Editor` -- works on a deep copy of an IR graph: add tensors / ops, remove ops, rewire readers, record derived
  weights (``weight_derivations.Recorder``), finish with a stable topological sort. Ported from the graph-agent pilot's
  ``gt_edit.py`` (Surgery + toposort + provenance-carrying weights); the restricted-script interpreter of the pilot is
  NOT ported (that belongs to the LLM proposer, a later phase).
* :func:`diff` / :func:`apply_patch` -- an edited graph as a patch against its base graph
  (``modelblaster.graph_rewrite.patch/v1``: ops removed / added by name, tensors and derived weights added, a digest of
  every removed op as the shape guard). ``apply_patch(base, diff(base, edited))`` reproduces ``edited`` exactly (op
  order aside, which is re-derived by the stable topological sort); a patch refuses to apply on a graph whose ops it
  removes are not the ones it was made from, with the reason. This is what a ``rewrite`` hint
  (``graph_hints._apply_rewrite_cuda``) carries, so an accepted rewrite re-applies every round without storing a graph.

Pure python (no torch, no numpy beyond ``weight_derivations``).
"""
from __future__ import annotations

import copy
import hashlib
import heapq
import json

from . import weight_derivations as wd

PATCH_FORMAT = "modelblaster.graph_rewrite.patch/v1"
# op fields that carry no semantics (renumbered / annotated by the appliers)
_VOLATILE = ("dispatch_id", "depends_on", "hint", "slot_variant")


class PatchError(ValueError):
    """A patch that does not apply (``code``: stale | exists | missing | cycle | invalid)."""

    def __init__(self, code: str, msg: str):
        super().__init__(msg)
        self.code = code


# ------------------------------------------------------------------------------ helpers on a graph dict
def io_names(g: dict) -> tuple[list[str], list[str]]:
    inp = g["input"]
    ins = [p["name"] for p in inp.get("packed_inputs") or [{"name": inp["tensor"]}]]
    outs = g["output"].get("tensors") or [g["output"]["tensor"]]
    return ins, list(outs)


def produced(g: dict) -> dict[str, str]:
    """tensor -> name of the op producing it (outputs, internal tensors and sub-op outputs)."""
    out = {}
    for o in g["ops"]:
        for t in list(o.get("outputs", [])) + list(o.get("internal", [])) + [x for s in o.get("sub_ops", []) for x in s.get("outputs", [])]:
            out.setdefault(t, o["name"])
    return out


def op_digest(o: dict) -> str:
    """Hash of an op's semantics (volatile bookkeeping fields excluded)."""
    return hashlib.sha1(json.dumps({k: v for k, v in o.items() if k not in _VOLATILE}, sort_keys=True,
                                   default=str).encode()).hexdigest()[:16]


def graph_digest(g: dict) -> str:
    """Hash of the semantic content of a graph: ops (order-independent), tensors, io, derivations."""
    ops = sorted((o["name"], op_digest(o)) for o in g["ops"])
    return hashlib.sha1(json.dumps({"ops": ops, "tensors": g["tensors"], "in": g["input"], "out": g["output"],
                                    "d": g.get(wd.KEY) or {}}, sort_keys=True, default=str).encode()).hexdigest()[:16]


def toposort(g: dict) -> None:
    """Stable topological order of ``g['ops']`` in place (the original order wins among ready ops). Raises PatchError
    on a cycle, a tensor with two producers, or an input nothing produces (not a graph input / constant)."""
    ops = g["ops"]
    ins, _ = io_names(g)
    ginputs = set(ins) | {t for t, m in g["tensors"].items() if m.get("const")}
    prod: dict[str, int] = {}
    for i, o in enumerate(ops):
        for t in list(o.get("outputs", [])) + list(o.get("internal", [])):
            if t in prod:
                raise PatchError("invalid", f"tensor {t!r} is produced by two ops ({ops[prod[t]]['name']}, {o['name']})")
            prod[t] = i
    deps = []
    for i, o in enumerate(ops):
        d = set()
        for t in o.get("inputs", []):
            if t in prod:
                if prod[t] != i:
                    d.add(prod[t])
            elif t not in ginputs:
                raise PatchError("missing", f"op {o['name']!r} reads tensor {t!r} that nothing produces "
                                            f"(not a graph input or constant)")
        deps.append(d)
    users: dict[int, list[int]] = {}
    for i, d in enumerate(deps):
        for x in d:
            users.setdefault(x, []).append(i)
    waiting = {i: set(d) for i, d in enumerate(deps) if d}
    ready = [i for i in range(len(ops)) if not deps[i]]
    heapq.heapify(ready)
    order = []
    while ready:
        i = heapq.heappop(ready)
        order.append(i)
        for j in users.get(i, []):
            waiting[j].discard(i)
            if not waiting[j]:
                heapq.heappush(ready, j)
    if len(order) != len(ops):
        raise PatchError("cycle", "the rewritten graph has a dependency cycle")
    g["ops"] = [ops[i] for i in order]


# ------------------------------------------------------------------------------ the editor
class Editor:
    """A rewrite in progress: ``Editor(graph, base_weights)`` copies the graph; mutate through the methods; ``result()``
    sorts and returns the edited graph (the original is never touched).

    ``base_weights`` = the ORIGINAL weights.npz as a mapping (values are only read to evaluate derived weights; a
    rewrite author never needs to see them). ``derive`` is the ``weight_derivations.Recorder``."""

    def __init__(self, graph: dict, base_weights, base_graph: dict | None = None):
        self.g = copy.deepcopy(graph)
        self.g0 = graph
        self.derive = wd.Recorder(self.g, base_weights, base_graph if base_graph is not None else graph)

    # -- lookup
    def op(self, name: str) -> dict:
        for o in self.g["ops"]:
            if o["name"] == name:
                return o
        raise KeyError(f"no op {name!r}")

    def producer(self, t: str) -> dict | None:
        n = produced(self.g).get(t)
        return self.op(n) if n else None

    def consumers(self, t: str) -> list[dict]:
        return [o for o in self.g["ops"] if t in o.get("inputs", [])]

    def unique(self, base: str) -> str:
        names = {o["name"] for o in self.g["ops"]} | set(self.g["tensors"])
        if base not in names:
            return base
        k = 1
        while f"{base}_{k}" in names:
            k += 1
        return f"{base}_{k}"

    # -- mutation
    def add_tensor(self, name: str, shape, dtype: str = "f32", **extra) -> str:
        if name in self.g["tensors"]:
            raise ValueError(f"tensor {name!r} exists")
        self.g["tensors"][name] = {"shape": [int(s) for s in shape], "dtype": dtype, "quant": None, **extra}
        return name

    def add_op(self, op: dict) -> dict:
        if any(o["name"] == op["name"] for o in self.g["ops"]):
            raise ValueError(f"op name {op['name']!r} exists")
        op.setdefault("inputs", [])
        op.setdefault("outputs", [])
        self.g["ops"].append(op)
        return op

    def remove_ops(self, names) -> None:
        names = set(names)
        miss = names - {o["name"] for o in self.g["ops"]}
        if miss:
            raise KeyError(f"no such ops {sorted(miss)}")
        self.g["ops"][:] = [o for o in self.g["ops"] if o["name"] not in names]

    def rewire(self, old: str, new: str, only_ops=None) -> int:
        """Readers of tensor ``old`` read ``new`` instead (all readers, or those named in ``only_ops``)."""
        only = set(only_ops) if only_ops is not None else None
        n = 0
        for o in self.g["ops"]:
            if only is not None and o["name"] not in only:
                continue
            for lst in [o["inputs"]] + [s["inputs"] for s in o.get("sub_ops", [])]:
                for i, t in enumerate(lst):
                    if t == old:
                        lst[i] = new
                        n += 1
        return n

    def result(self) -> dict:
        toposort(self.g)
        return self.g


# ------------------------------------------------------------------------------ patches
def diff(base: dict, edited: dict, *, tensor_map: dict | None = None) -> dict:
    """The patch that turns ``base`` into ``edited`` (both graph dicts; names made unique the way the flow does)."""
    from .graph_hints import unique_names
    b, e = unique_names(copy.deepcopy(base)), unique_names(copy.deepcopy(edited))
    bo, eo = {o["name"]: o for o in b["ops"]}, {o["name"]: o for o in e["ops"]}
    removed = [n for n in bo if n not in eo or op_digest(bo[n]) != op_digest(eo[n])]
    added = [n for n in eo if n not in bo or op_digest(bo[n]) != op_digest(eo[n])]
    p = {"format": PATCH_FORMAT,
         "remove_ops": removed,
         "add_ops": [copy.deepcopy(eo[n]) for n in eo if n in set(added)],
         "add_tensors": {t: m for t, m in e["tensors"].items() if t not in b["tensors"]},
         "remove_tensors": [t for t in b["tensors"] if t not in e["tensors"]],
         "changed_tensors": [t for t in e["tensors"] if t in b["tensors"] and e["tensors"][t] != b["tensors"][t]],
         "weight_derivations": {k: v for k, v in (e.get(wd.KEY) or {}).items() if k not in (b.get(wd.KEY) or {})},
         "weight_dtypes": {k: v for k, v in (e.get("weight_dtypes") or {}).items() if k not in (b.get("weight_dtypes") or {})},
         "requires": {"ops": {n: op_digest(bo[n]) for n in removed}}}
    for o in p["add_ops"]:
        for k in _VOLATILE:
            o.pop(k, None)
    if tensor_map:
        p["tensor_map"] = tensor_map
    return p


def apply_patch(graph: dict, patch: dict) -> tuple[dict, dict]:
    """``graph`` + ``patch`` -> (edited graph, info {"removed": [...], "added": [...]}). Raises PatchError with the
    reason when the base is not the graph the patch was made from (an op it removes is missing or different), when a
    name it adds is taken, or when the result is not a DAG. ``graph`` is not modified."""
    from .graph_hints import unique_names
    if patch.get("format") != PATCH_FORMAT:
        raise PatchError("invalid", f"not a {PATCH_FORMAT} patch")
    g = unique_names(copy.deepcopy(graph))
    have = {o["name"]: o for o in g["ops"]}
    for n, dg in patch.get("requires", {}).get("ops", {}).items():
        if n not in have:
            raise PatchError("stale", f"the rewrite removes op {n!r}, which is not in this graph (the graph changed since it was proposed)")
        if op_digest(have[n]) != dg:
            raise PatchError("stale", f"op {n!r} differs from the op the rewrite was made from")
    rm = set(patch["remove_ops"])
    g["ops"] = [o for o in g["ops"] if o["name"] not in rm]
    names = {o["name"] for o in g["ops"]}
    for o in patch["add_ops"]:
        if o["name"] in names:
            raise PatchError("exists", f"the rewrite adds op {o['name']!r}, whose name is taken")
        names.add(o["name"])
        g["ops"].append(copy.deepcopy(o))
    for t, m in patch["add_tensors"].items():
        if t in g["tensors"] and g["tensors"][t] != m:
            raise PatchError("exists", f"the rewrite adds tensor {t!r}, which exists with different shape/dtype")
        g["tensors"][t] = copy.deepcopy(m)
    for t in patch.get("remove_tensors", []):
        if any(t in o.get("inputs", []) or t in o.get("outputs", []) for o in g["ops"]):
            continue                                   # still used by an op the patch did not touch: keep it registered
        g["tensors"].pop(t, None)
    if patch.get("weight_derivations"):
        d = g.setdefault(wd.KEY, {})
        for k, v in patch["weight_derivations"].items():
            if k in d and d[k] != v:
                raise PatchError("exists", f"derived weight {k!r} exists with a different derivation")
            d[k] = copy.deepcopy(v)
        if patch.get("weight_dtypes"):
            g.setdefault("weight_dtypes", {}).update(patch["weight_dtypes"])
    toposort(g)
    return g, {"removed": sorted(rm), "added": [o["name"] for o in patch["add_ops"]]}


def new_op_kinds(patch: dict) -> list[str]:
    return sorted({o["op"] for o in patch.get("add_ops", [])})
