"""Derived weights: ``weights.npz`` entries that are computed from other weights by a recorded, replayable derivation.

An open-ended graph rewrite (docs/graph_rewrite.md) often needs a weight the model does not have: the zero-padded
stack of two convolutions' filters that lets one wider convolution replace two (SqueezeNet's expand1x1 + expand3x3,
37 -> 21 ops), a concatenation of Q/K/V projections, a transposed copy. Such a weight must stay *traceable*: it is an
expression over the original weights, written into ``graph.json`` so that anyone can rebuild it from the original
``weights.npz`` and compare -- the loader, the verifier (``graph_verify``), the torch-baseline mapping -- and so that a
rewrite whose weights are not derivable is rejected with a reason instead of failing late with "IR weights not mapped
to the Model".

``graph.json`` (absent = the IR has no derived weights and is byte-for-byte what it always was)::

    "weight_derivations": {
      "d.fire2.w": {"dtype": "f16", "shape": [128, 16, 3, 3],
                    "expr": {"op": "concat", "axis": 0, "args": [
                        {"op": "pad", "pads": [[0, 0], [0, 0], [1, 1], [1, 1]], "value": 0.0,
                         "args": [{"w": "fire2.expand1x1.weight"}]},
                        {"w": "fire2.expand3x3.weight"}]}}}

The language is declarative and restricted (no code, no I/O):

=========  ==================================  ============================================================
op         parameters                          semantics (``a`` = argument values)
=========  ==================================  ============================================================
``w``      (leaf) ``{"w": key}``               a base weight, or another derived key (acyclic)
``pad``    ``pads`` [[lo, hi]] per dim,        ``np.pad`` with a constant (dtype kept, exact)
           ``value`` (default 0)
``stack``  ``axis``                            ``np.stack`` (all arguments the same shape and dtype)
``concat`` ``axis``                            ``np.concatenate`` (all arguments the same dtype)
``transpose`` ``perm``                         ``a.transpose(perm)``
``reshape``   ``shape`` (one -1 allowed)       ``a.reshape(shape)``
``slice``  ``axis``, ``start``, ``stop``       ``a[start:stop]`` along ``axis`` (inside the dimension)
``scale``  ``by`` (finite number)              float32(a) * float32(by), rounded once to a's dtype
``cast``   ``dtype``                           storage conversion (RNE for f16 / bf16, truncation to ints)
``add``/``mul``                                float32 math on two floats (numpy broadcasting), rounded once
``const``  ``shape``, ``value``, ``dtype``     ``np.full`` (no arguments)
=========  ==================================  ============================================================

Dtypes are the IR's tensor dtypes (f32, f16, bf16, i64, i32, u8, b8; ``cuda_dtypes``). The structural ops keep their
argument's dtype and move values without arithmetic, so they are exact; ``concat`` / ``stack`` refuse mixed dtypes (say
``cast`` where a conversion is meant). A bf16 weight is a float32 array holding bf16-exact values (as everywhere in the
IR) and ``scale`` / ``cast`` / ``add`` / ``mul`` round to it. Replay is deterministic numpy, so the rebuilt array is
bit-identical to what the recorder stored; the verifier compares bytes.

Per-tensor dtypes (``--quant native``, docs/tensor_dtypes.md): a derived weight declares its dtype and, in a native IR,
``weight_dtypes[key]`` names it like any other weight (``declare``). A non-native IR stores every float weight in its
own storage dtype, so a derivation's float dtype must be that dtype (``validate``).

Keys of derived weights start with ``d.`` (cannot collide with a state-dict name). Loaders: ``fill_missing`` rebuilds
the derived entries a ``weights.npz`` lacks (``gpu_eval/harness/mbir.load_ir`` calls it); ``materialize`` /
``kb_proposals.write_variant_ir`` write a ``weights.npz`` holding the base weights AND the derived ones, so exporters and
walkers see ordinary weights.

This module needs only numpy (and ``cuda_dtypes`` next to it): the GPU harness loads it by path.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

PREFIX = "d."
KEY = "weight_derivations"
LOGICAL = ("f32", "f16", "bf16", "i64", "i32", "u8", "b8")
_FLOATS = ("f32", "f16", "bf16")
# ops that only move values (exact, no rounding): a rewrite made of these alone is an exact rewrite
STRUCTURAL = ("w", "pad", "stack", "concat", "transpose", "reshape", "slice", "const")
OPS = STRUCTURAL + ("scale", "cast", "add", "mul")
_ARITY = {"pad": 1, "transpose": 1, "reshape": 1, "slice": 1, "scale": 1, "cast": 1, "add": 2, "mul": 2, "const": 0}


class DerivationError(ValueError):
    """A derivation that cannot be validated or replayed; ``code`` names the class, str() the reason."""

    def __init__(self, code: str, msg: str):
        super().__init__(msg)
        self.code = code


_CD = None


def _cuda_dtypes():
    global _CD
    if _CD is None:
        spec = importlib.util.spec_from_file_location("cuda_dtypes_wd", Path(__file__).resolve().parent / "cuda_dtypes.py")
        _CD = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_CD)
    return _CD


# ------------------------------------------------------------------------------ dtypes
def storage(a, ldt: str) -> np.ndarray:
    """``a`` as ``weights.npz`` holds a tensor of logical dtype ``ldt`` (bf16: float32 with bf16-exact values)."""
    return np.ascontiguousarray(_cuda_dtypes().np_storage(np.asarray(a), ldt))


def logical_dtype(graph: dict, key: str, arr) -> str:
    """Logical (IR) dtype of a weights.npz array: ``weight_dtypes`` of a native IR, else its storage dtype
    (``cuda_dtypes.weight_dtype``: fp16 IR -> f16, bf16 IR -> bf16, fp32 -> f32; integers i32 / i64)."""
    cd = _cuda_dtypes()
    a = np.asarray(arr)
    d = (graph.get("weight_dtypes") or {}).get(key)
    if d:
        return d
    if a.dtype.kind in "iub":
        return {1: "u8", 4: "i32", 8: "i64"}.get(a.dtype.itemsize, "i32") if a.dtype.kind != "b" else "b8"
    wd = cd.weight_dtype(graph, key, a.dtype)
    return {"f16": "f16", "bf16": "bf16", "f32": "f32"}.get(wd, wd)


# ------------------------------------------------------------------------------ the expression language
def _check_node(e, where: str, leaves: set, depth: int = 0) -> None:
    if depth > 64:
        raise DerivationError("too_deep", f"{where}: expression nested deeper than 64")
    if not isinstance(e, dict):
        raise DerivationError("bad_node", f"{where}: a node is an object ({{'w': key}} or {{'op': ...}}), got {type(e).__name__}")
    if "w" in e:
        if set(e) != {"w"} or not isinstance(e["w"], str):
            raise DerivationError("bad_node", f"{where}: a leaf is exactly {{'w': 'key'}}")
        leaves.add(e["w"])
        return
    op = e.get("op")
    if op not in OPS or op == "w":
        raise DerivationError("unknown_op", f"{where}: unknown derivation op {op!r} (known: {', '.join(o for o in OPS if o != 'w')})")
    args = e.get("args", [])
    if not isinstance(args, list):
        raise DerivationError("bad_node", f"{where}: args must be a list")
    want = _ARITY.get(op)
    if want is not None and len(args) != want:
        raise DerivationError("bad_arity", f"{where}: {op} takes {want} argument(s), got {len(args)}")
    if want is None and not args:
        raise DerivationError("bad_arity", f"{where}: {op} needs at least one argument")
    ints = lambda v: isinstance(v, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in v)  # noqa: E731
    isint = lambda v: isinstance(v, int) and not isinstance(v, bool)  # noqa: E731
    if op == "pad":
        p = e.get("pads")
        if not (isinstance(p, list) and p and all(ints(x) and len(x) == 2 and min(x) >= 0 for x in p)):
            raise DerivationError("bad_param", f"{where}: pad needs pads = [[lo, hi], ...] (non-negative ints, one per dim)")
        if not _num(e.get("value", 0.0)):
            raise DerivationError("bad_param", f"{where}: pad value must be a finite number")
    elif op in ("stack", "concat"):
        if not isint(e.get("axis")):
            raise DerivationError("bad_param", f"{where}: {op} needs an integer axis")
    elif op == "transpose":
        if not ints(e.get("perm")):
            raise DerivationError("bad_param", f"{where}: transpose needs perm (list of ints)")
    elif op == "reshape":
        s = e.get("shape")
        if not (ints(s) and s and sum(1 for x in s if x == -1) <= 1 and all(x == -1 or x > 0 for x in s)):
            raise DerivationError("bad_param", f"{where}: reshape needs shape (positive ints, at most one -1)")
    elif op == "slice":
        if not (isint(e.get("axis")) and isint(e.get("start")) and isint(e.get("stop")) and 0 <= e["start"] < e["stop"]):
            raise DerivationError("bad_param", f"{where}: slice needs integer axis and 0 <= start < stop")
    elif op == "scale":
        if not _num(e.get("by")):
            raise DerivationError("bad_param", f"{where}: scale needs a finite number 'by'")
    elif op == "cast":
        if e.get("dtype") not in LOGICAL:
            raise DerivationError("bad_param", f"{where}: cast dtype must be one of {LOGICAL}")
    elif op == "const":
        s = e.get("shape")
        if not (ints(s) and s and all(x > 0 for x in s) and _num(e.get("value")) and e.get("dtype") in LOGICAL):
            raise DerivationError("bad_param", f"{where}: const needs shape, a finite value and a dtype")
    for k in e:
        if k not in ("op", "args", "pads", "value", "axis", "perm", "shape", "start", "stop", "by", "dtype"):
            raise DerivationError("bad_param", f"{where}: unknown field {k!r} on {op}")
    for i, a in enumerate(args):
        _check_node(a, f"{where}.args[{i}]", leaves, depth + 1)


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def expr_leaves(e: dict, acc: set | None = None) -> set:
    """Keys an expression reads."""
    acc = set() if acc is None else acc
    if "w" in e:
        acc.add(e["w"])
    for a in e.get("args", []):
        expr_leaves(a, acc)
    return acc


def is_exact(graph: dict, keys=None) -> bool:
    """True when every (selected) derivation is made of structural ops only: the derived weights are exact copies /
    rearrangements of base values, so a rewrite using them must reproduce the original in exact arithmetic."""
    d = graph.get(KEY) or {}

    def ok(e):
        if "w" in e:
            return True
        return e["op"] in STRUCTURAL and all(ok(a) for a in e.get("args", []))
    return all(ok(d[k]["expr"]) for k in (keys if keys is not None else d))


def _eval(e, resolve):
    """-> (array, logical dtype)."""
    if "w" in e:
        return resolve(e["w"])
    op = e["op"]
    if op == "const":
        return storage(np.full(e["shape"], e["value"], dtype=np.float64), e["dtype"]), e["dtype"]
    vals = [_eval(a, resolve) for a in e["args"]]
    arrs, dts = [v[0] for v in vals], [v[1] for v in vals]
    if op == "pad":
        a = arrs[0]
        if len(e["pads"]) != a.ndim:
            raise DerivationError("shape", f"pad: {len(e['pads'])} pad pairs for a {a.ndim}-D argument")
        return np.pad(a, [tuple(p) for p in e["pads"]], mode="constant", constant_values=e.get("value", 0.0)).astype(a.dtype), dts[0]
    if op in ("stack", "concat"):
        if len(set(dts)) != 1:
            raise DerivationError("dtype", f"{op}: arguments have different dtypes {dts} (say 'cast' where a conversion is meant)")
        ax = e["axis"]
        try:
            r = np.stack(arrs, axis=ax) if op == "stack" else np.concatenate(arrs, axis=ax)
        except (ValueError, IndexError) as ex:       # shape / axis mismatch (AxisError is both): a clear reason
            raise DerivationError("shape", f"{op} on axis {ax}: {ex}")
        return r, dts[0]
    if op == "transpose":
        a = arrs[0]
        if sorted(e["perm"]) != list(range(a.ndim)):
            raise DerivationError("shape", f"transpose perm {e['perm']} is not a permutation of {a.ndim} dims")
        return np.ascontiguousarray(a.transpose(e["perm"])), dts[0]
    if op == "reshape":
        a = arrs[0]
        try:
            return np.ascontiguousarray(a.reshape(e["shape"])), dts[0]
        except ValueError as ex:
            raise DerivationError("shape", f"reshape {list(a.shape)} -> {e['shape']}: {ex}")
    if op == "slice":
        a = arrs[0]
        ax = e["axis"]
        if not -a.ndim <= ax < a.ndim or e["stop"] > a.shape[ax]:
            raise DerivationError("shape", f"slice [{e['start']}:{e['stop']}] outside axis {ax} of shape {list(a.shape)}")
        sl = [slice(None)] * a.ndim
        sl[ax] = slice(e["start"], e["stop"])
        return np.ascontiguousarray(a[tuple(sl)]), dts[0]
    if op in ("scale", "add", "mul"):
        if any(d not in _FLOATS for d in dts):
            raise DerivationError("dtype", f"{op} needs float arguments, got {dts}")
        f = [x.astype(np.float32) for x in arrs]
        if op == "scale":
            r = f[0] * np.float32(e["by"])
        elif op == "add":
            r = f[0] + f[1]
        else:
            r = f[0] * f[1]
        return storage(r, dts[0]), dts[0]
    if op == "cast":
        a = arrs[0]
        return storage(a.astype(np.float32) if a.dtype.kind == "f" else a, e["dtype"]), e["dtype"]
    raise DerivationError("unknown_op", f"unknown op {op!r}")


# ------------------------------------------------------------------------------ whole-graph validation and replay
def derivations(graph: dict) -> dict:
    return graph.get(KEY) or {}


def order(graph: dict, base_names) -> list[str]:
    """Derived keys in dependency order. Raises DerivationError on an unknown leaf, a cycle or a key that shadows a base weight."""
    d, base = derivations(graph), set(base_names)
    out, state = [], {}

    def visit(k, path):
        if state.get(k) == 2:
            return
        if state.get(k) == 1:
            raise DerivationError("cycle", "derivation cycle: " + " -> ".join(path + [k]))
        state[k] = 1
        for leaf in sorted(expr_leaves(d[k]["expr"])):
            if leaf in d:
                visit(leaf, path + [k])
            elif leaf not in base:
                raise DerivationError("unmapped", f"derived weight {k!r} reads {leaf!r}, which is neither a weight of the "
                                                  f"original IR nor another derivation (IR weights not mapped to the Model)")
        state[k] = 2
        out.append(k)
    for k in sorted(d):
        visit(k, [])
    return out


def validate(graph: dict, base_names, base_graph: dict | None = None) -> list[dict]:
    """Structural validation of ``graph[weight_derivations]`` against the base weight names (no values).
    -> list of {"code", "key", "msg"} (empty = valid)."""
    d = derivations(graph)
    bad = []
    if not isinstance(d, dict):
        return [{"code": "bad_node", "key": None, "msg": "weight_derivations must be an object"}]
    base = set(base_names)
    cd = _cuda_dtypes()
    for k, rec in sorted(d.items()):
        def add(code, msg, k=k):
            bad.append({"code": code, "key": k, "msg": msg})
        if not k.startswith(PREFIX):
            add("bad_key", f"derived weight key {k!r} must start with {PREFIX!r}")
        if k in base:
            add("shadows_base", f"derived weight {k!r} has the name of a base weight (a derivation may not replace an original weight)")
        if not isinstance(rec, dict) or "expr" not in rec or rec.get("dtype") not in LOGICAL:
            add("bad_record", f"{k}: record needs 'expr' and a 'dtype' of {LOGICAL}")
            continue
        try:
            _check_node(rec["expr"], k, set())
        except DerivationError as e:
            add(e.code, str(e))
            continue
        sh = rec.get("shape")
        if sh is not None and not (isinstance(sh, list) and all(isinstance(x, int) and x > 0 for x in sh)):
            add("bad_record", f"{k}: shape must be a list of positive ints")
        # dtype contract: a float derived weight is stored like every float weight of this IR
        if rec["dtype"] in _FLOATS:
            if cd.is_mixed(graph):
                wd = (graph.get("weight_dtypes") or {}).get(k)
                if wd != rec["dtype"]:
                    add("dtype_contract", f"{k}: a native-dtype IR must name the weight's dtype in weight_dtypes "
                                          f"(have {wd!r}, derivation says {rec['dtype']!r}); use declare()")
            else:
                want = {"fp16": "f16", "bf16": "bf16"}.get(graph.get("quant"), "f32")
                if rec["dtype"] != want:
                    add("dtype_contract", f"{k}: this IR stores float weights as {want}, the derivation produces {rec['dtype']}")
    try:
        order(graph, base)
    except DerivationError as e:
        bad.append({"code": e.code, "key": None, "msg": str(e)})
    return bad


def replay(graph: dict, base_weights, base_graph: dict | None = None, keys=None) -> dict:
    """Rebuild the derived weights from ``base_weights`` (mapping key -> ndarray; ``base_graph`` supplies the base
    weights' dtypes, default ``graph``). -> {key: array}; raises DerivationError (code, reason) on any failure,
    including a result whose dtype / shape differs from the record."""
    d = derivations(graph)
    if not d:
        return {}
    bg = base_graph if base_graph is not None else graph
    errs = validate(graph, set(base_weights), bg)
    if errs:
        raise DerivationError(errs[0]["code"], errs[0]["msg"])
    done: dict = {}
    want = _closure(d, keys) if keys is not None else None

    def resolve(k):
        if k in done:
            return done[k]
        if k in base_weights:
            a = np.asarray(base_weights[k])
            return a, logical_dtype(bg, k, a)
        raise DerivationError("unmapped", f"weight {k!r} does not exist (IR weights not mapped to the Model)")
    for k in order(graph, set(base_weights)):
        if want is not None and k not in want:
            continue
        rec = d[k]
        try:
            a, ldt = _eval(rec["expr"], resolve)
        except DerivationError as e:
            raise DerivationError(e.code, f"{k}: {e}")
        if ldt != rec["dtype"]:
            raise DerivationError("dtype", f"{k}: the expression yields {ldt} but the record says {rec['dtype']} (add an explicit cast)")
        if rec.get("shape") is not None and list(a.shape) != list(rec["shape"]):
            raise DerivationError("shape", f"{k}: replay has shape {list(a.shape)}, record says {rec['shape']}")
        done[k] = (a, ldt)
    return {k: v[0] for k, v in done.items() if keys is None or k in set(keys)}


def _closure(d: dict, keys) -> set:
    """``keys`` plus every derived key they read, transitively."""
    out, stack = set(), list(keys)
    while stack:
        x = stack.pop()
        if x in out or x not in d:
            continue
        out.add(x)
        stack.extend(expr_leaves(d[x]["expr"]))
    return out


def fill_missing(graph: dict, weights: dict) -> dict:
    """``weights`` (a loaded weights.npz as a dict) plus every derived entry it lacks, rebuilt from its base entries.
    No-op (same object) when the graph has no derivations or nothing is missing."""
    d = derivations(graph)
    miss = [k for k in d if k not in weights]
    if not miss:
        return weights
    base = {k: v for k, v in weights.items() if k not in d}
    out = dict(weights)
    out.update(replay(graph, base, graph, keys=miss))
    return out


def materialize(graph: dict, base_weights) -> dict:
    """Base weights plus every derived weight (derived entries win over same-named stale ones)."""
    out = dict(base_weights)
    base = {k: v for k, v in base_weights.items() if k not in derivations(graph)}
    out.update(replay(graph, base, graph))
    return out


def write_weights(base_npz: Path, graph: dict, out_npz: Path) -> list[str]:
    """Write ``out_npz`` = the base npz's arrays + the graph's derived weights. -> the derived keys."""
    z = np.load(base_npz)
    base = {k: z[k] for k in z.files}
    full = materialize(graph, base)
    out_npz = Path(out_npz)
    if out_npz.is_symlink() or out_npz.exists():
        out_npz.unlink()
    with open(out_npz, "wb") as f:
        np.savez(f, **full)
    return sorted(derivations(graph))


def digest(graph: dict) -> str:
    """Short stable hash of the derivation table (cache-key material)."""
    d = derivations(graph)
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12] if d else ""


# ------------------------------------------------------------------------------ recording (builders)
class Recorder:
    """Builds derivations while evaluating them (the rewrite author's handle on weights: no raw arrays).

        rec = Recorder(graph, base_weights)              # base_weights: the ORIGINAL weights.npz as a dict
        k = rec.concat("d.fire2.w", [rec.pad("d.fire2.w1p", "fire2.expand1x1.weight", [[0,0],[0,0],[1,1],[1,1]]),
                                     "fire2.expand3x3.weight"], axis=0)

    Each call evaluates its expression at once (numpy; the value is available through ``arrays``), writes the record
    into ``graph['weight_derivations']`` and, for a native IR, ``weight_dtypes``. Arguments are keys of base weights, of
    earlier derived weights, or (for convenience) nested nodes."""

    def __init__(self, graph: dict, base_weights, base_graph: dict | None = None):
        self.graph, self.base = graph, base_weights
        self.base_graph = base_graph if base_graph is not None else graph
        self.arrays: dict = {}

    def _arg(self, a):
        return a if isinstance(a, dict) else {"w": a}

    def _put(self, new: str, expr: dict) -> str:
        d = self.graph.get(KEY) or {}
        if new in d or new in self.base:
            raise DerivationError("exists", f"weight key {new!r} already exists")
        if not new.startswith(PREFIX):
            raise DerivationError("bad_key", f"derived weight keys must start with {PREFIX!r} (got {new!r})")
        leaves = set()
        _check_node(expr, new, leaves)

        def resolve(k):
            if k in self.arrays:
                return self.arrays[k], d[k]["dtype"]
            if k in self.base:
                a = np.asarray(self.base[k])
                return a, logical_dtype(self.base_graph, k, a)
            raise DerivationError("unmapped", f"weight {k!r} does not exist (IR weights not mapped to the Model)")
        a, ldt = _eval(expr, resolve)
        self.graph.setdefault(KEY, {})[new] = {"dtype": ldt, "shape": [int(s) for s in a.shape], "expr": expr}
        self.arrays[new] = a
        self.declare(new, ldt)
        return new

    def declare(self, key: str, ldt: str) -> None:
        if _cuda_dtypes().is_mixed(self.graph) and ldt in _FLOATS:
            self.graph.setdefault("weight_dtypes", {})[key] = ldt

    def pad(self, new, k, pads, value=0.0):
        return self._put(new, {"op": "pad", "pads": [list(map(int, p)) for p in pads], "value": value, "args": [self._arg(k)]})

    def stack(self, new, keys, axis=0):
        return self._put(new, {"op": "stack", "axis": int(axis), "args": [self._arg(k) for k in keys]})

    def concat(self, new, keys, axis=0):
        return self._put(new, {"op": "concat", "axis": int(axis), "args": [self._arg(k) for k in keys]})

    def transpose(self, new, k, perm):
        return self._put(new, {"op": "transpose", "perm": [int(p) for p in perm], "args": [self._arg(k)]})

    def reshape(self, new, k, shape):
        return self._put(new, {"op": "reshape", "shape": [int(s) for s in shape], "args": [self._arg(k)]})

    def slice(self, new, k, axis, start, stop):
        return self._put(new, {"op": "slice", "axis": int(axis), "start": int(start), "stop": int(stop), "args": [self._arg(k)]})

    def scale(self, new, k, by):
        return self._put(new, {"op": "scale", "by": float(by), "args": [self._arg(k)]})

    def cast(self, new, k, dtype):
        return self._put(new, {"op": "cast", "dtype": dtype, "args": [self._arg(k)]})

    def add(self, new, a, b):
        return self._put(new, {"op": "add", "args": [self._arg(a), self._arg(b)]})

    def mul(self, new, a, b):
        return self._put(new, {"op": "mul", "args": [self._arg(a), self._arg(b)]})

    def const(self, new, shape, value, dtype):
        return self._put(new, {"op": "const", "shape": [int(s) for s in shape], "value": value, "dtype": dtype})
