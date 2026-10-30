"""The REWRITE SANDBOX: a restricted interpreter for free-form graph rewrites (offline discovery mode; docs/graph_rewrite_sandbox.md).

A proposer (an LLM through a tool loop, or a person) writes::

    def rewrite(g, W, P, T):
        ...                       # graph surgery through g, new weights ONLY through P, read-only metadata through W and T

and :func:`run_rewrite` executes it on a copy of an IR graph and returns a :class:`SandboxResult`: the edited graph, the PATCH
(``graph_edit.diff``: ops removed / added, tensors added) and the DERIVATIONS (``weight_derivations``: every new weight as a
declarative expression over the model's original parameters, replayable by the verifier and the loader). Nothing is applied to
the flow: the result goes through ``graph_verify`` (structural + numerical equivalence with sampled intermediates) and, offline,
through the discovery CLI (``graph_rewrite_discover``). Ported in spirit from the graph-agent pilot's ``gt_edit.py`` (restricted
interpreter + provenance-carrying weights); the code is new, the pilot's tree is not imported.

What the script sees
--------------------
``g``  :class:`GraphHandle` over a ``graph_edit.Editor``: ``ops() op(name) producer(t) consumers(t) tensors() has_tensor(t)
       inputs() outputs() add_tensor(name, shape, dtype) add_op(op_dict) update_op(name, **fields) remove_ops(names)
       remove_tensor(t) rewire(old, new[, only_ops]) unique(base)``. Reads return COPIES; the only way to change the graph is a
       mutating method, each of which validates its arguments and raises a clear error (never a raw ``KeyError``).
``W``  :class:`WeightCatalogue`: ``has(k) shape(k) dtype(k) keys()`` of the model's parameters and of the weights derived so far.
       Weight VALUES are not readable (``W.value`` raises): a rewrite that needs a number the model does not have is not a rewrite.
``P``  :class:`ProvenanceWeights` over ``weight_derivations.Recorder``: ``concat slice reshape transpose pad stack scale cast mul
       add zeros ones``; each takes the NEW key (must start with ``d.``) first and returns it; the expression is recorded over the
       parameters it was built from, evaluated with numpy (exact structural ops; arithmetic in float32, rounded once to the
       operand dtype) and replayed by the verifier from the original weights.npz.
``T``  :class:`TensorMeta`: ``shape(t) dtype(t) numel(t) has(t) names()`` of the graph being edited (live).
``math`` a few pure functions (``ceil floor sqrt prod gcd``).

The script may ``return {"tensor_map": {...}, "description": "..."}``: tensors the rewrite removed but that are a slice of a new
one (``{old: {"from": new, "axis": 1, "start": 0, "stop": 64}}``, compared by the gate), and a one-line hypothesis.

What is rejected (``SandboxError``, ``code`` in parentheses)
-----------------------------------------------------------
* syntax errors (``syntax``); more than 200 KB (``too_large``); no ``def rewrite(g, W, P, T)`` (``no_rewrite``);
* imports, ``while`` loops, classes, ``global`` / ``nonlocal`` / ``with`` / ``yield`` / ``await`` / ``del``, walrus, decorators, ``**``
  and ``<<`` (giant integers), huge numeric constants (``forbidden_construct``); names starting with ``_`` or naming dangerous builtins
  (``eval exec open getattr type ...``) (``forbidden_name``); attributes starting with ``_`` or not on the whitelist of container /
  string / handle methods, which also keeps ``str.format`` attribute walks out (``forbidden_attr``);
* unbounded work: a statement budget (``max_steps``, default 300,000 traced lines) and a wall-clock limit (``timeout_s``, main thread)
  stop runaway ``for`` loops; ``range`` is capped at 1,000,000 elements (``step_budget`` / ``timeout``);
* weights the model cannot reproduce: an op naming a ``weight`` / ``bias`` / ``gamma_key`` / ``beta_key`` that is neither a model parameter
  nor created by ``P`` (``weight_untraceable``); ``W.value`` (``weight_not_readable``); a bad ``P`` call (shape mismatch, existing key,
  unknown source key: ``bad_weight``);
* graph surgery errors with a message that says what to do: dangling tensors (an op reads a tensor nothing produces, a removed op's
  output still has readers: ``dangling_tensor``), unregistered output tensors, duplicate names, unknown ops (``bad_edit`` / ``bad_op``), a cycle
  (``cycle``); anything else the script raises is ``script_error`` with the exception type and the script line.

This is a robustness boundary against LLM mistakes and prompt-injected scripts (no I/O, no imports, no attribute escape, bounded work); it
is NOT a security sandbox against a determined adversary running in your process: run the proposer's tool loop in a container as well.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import signal
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

from . import graph_edit as ge
from . import weight_derivations as wd

MAX_SOURCE = 200_000
MAX_STEPS = 300_000
MAX_RANGE = 1_000_000
MAX_CONST = 10_000_000
MAX_NEW_OPS = 2000
MAX_DERIVED = 400
TIMEOUT_S = 60.0

ENTRY = "rewrite"
WEIGHT_FIELDS = ("weight", "bias", "gamma_key", "beta_key")


class SandboxError(Exception):
    """A rejected script or edit: ``code`` names the class (module docstring), ``str()`` is the message the proposer sees."""

    def __init__(self, code: str, msg: str, line: int | None = None):
        super().__init__(msg)
        self.code, self.line = code, line

    def to_dict(self) -> dict:
        return {"code": self.code, "msg": str(self), "line": self.line}


# ------------------------------------------------------------------------------ static check
_BAD_NAMES = {"eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr",
              "hasattr", "input", "breakpoint", "os", "sys", "subprocess", "importlib", "builtins", "memoryview", "type", "object",
              "super", "classmethod", "staticmethod", "property", "exit", "quit", "help", "np", "numpy", "torch", "id", "hash",
              "bytes", "bytearray", "callable", "iter", "next", "format", "ascii", "chr", "ord", "bin", "hex", "oct"}
_BANNED_NODES = (ast.Import, ast.ImportFrom, ast.While, ast.ClassDef, ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.Await,
                 ast.Global, ast.Nonlocal, ast.With, ast.Yield, ast.YieldFrom, ast.Delete, ast.NamedExpr)
_BANNED_OPS = (ast.Pow, ast.LShift)
_ALLOWED_ATTRS = {
    # containers / strings (no format, no __x__)
    "append", "extend", "insert", "remove", "pop", "index", "count", "sort", "reverse", "copy", "clear", "get", "items", "keys",
    "values", "setdefault", "update", "add", "discard", "union", "intersection", "difference", "issubset", "issuperset", "join",
    "split", "rsplit", "strip", "lstrip", "rstrip", "startswith", "endswith", "lower", "upper", "replace", "isdigit", "find",
    "rfind", "removeprefix", "removesuffix", "zfill",
    # handles
    "ops", "op", "producer", "consumers", "tensors", "has_tensor", "inputs", "outputs", "add_tensor", "add_op", "update_op",
    "remove_ops", "remove_tensor", "rewire", "unique", "has", "shape", "dtype", "keys", "value", "numel", "names",
    "concat", "slice", "reshape", "transpose", "pad", "stack", "scale", "cast", "mul", "zeros", "ones",
    # math
    "ceil", "floor", "sqrt", "prod", "gcd", "pi",
}


def check_script(src: str) -> ast.Module:
    """AST check of a rewrite script (module docstring). Returns the tree; raises SandboxError."""
    if len(src) > MAX_SOURCE:
        raise SandboxError("too_large", f"script larger than {MAX_SOURCE // 1000} KB")
    try:
        tree = ast.parse(src, filename="<rewrite>")
    except SyntaxError as e:
        raise SandboxError("syntax", f"syntax error: {e.msg} (line {e.lineno})", e.lineno) from None
    has = False
    for n in ast.walk(tree):
        ln = getattr(n, "lineno", None)
        if isinstance(n, _BANNED_NODES):
            what = {"Import": "imports (math is pre-bound)", "ImportFrom": "imports (math is pre-bound)",
                    "While": "`while` loops (use `for ... in range(n)` / a list; loops are bounded)"}.get(type(n).__name__, type(n).__name__)
            raise SandboxError("forbidden_construct", f"{what} are not allowed" if not what.endswith(")") and "loops" not in what else f"{what} not allowed", ln)
        if isinstance(n, ast.BinOp) and isinstance(n.op, _BANNED_OPS):
            raise SandboxError("forbidden_construct", f"operator {type(n.op).__name__} is not allowed (giant integers); use math.prod / repeated multiplication", ln)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool) and abs(n.value) > MAX_CONST:
            raise SandboxError("forbidden_construct", f"numeric constant {n.value!r} is larger than {MAX_CONST}", ln)
        if isinstance(n, ast.Constant) and isinstance(n.value, (str, bytes)) and len(n.value) > 10_000:
            raise SandboxError("forbidden_construct", "string constant longer than 10000 characters", ln)
        if isinstance(n, ast.Attribute):
            if n.attr.startswith("_"):
                raise SandboxError("forbidden_attr", f"attribute {n.attr!r}: underscore attributes are not allowed", ln)
            if n.attr not in _ALLOWED_ATTRS:
                raise SandboxError("forbidden_attr", f"attribute {n.attr!r} is not on the whitelist of container / string / handle methods", ln)
        if isinstance(n, ast.Name) and (n.id in _BAD_NAMES or n.id.startswith("_")):
            raise SandboxError("forbidden_name", f"name {n.id!r} is not allowed", ln)
        if isinstance(n, (ast.FunctionDef, ast.Lambda)):
            args = n.args
            for a in args.args + args.kwonlyargs + ([args.vararg] if args.vararg else []) + ([args.kwarg] if args.kwarg else []):
                if a.arg.startswith("_") or a.arg in _BAD_NAMES:
                    raise SandboxError("forbidden_name", f"parameter name {a.arg!r} is not allowed", ln)
        if isinstance(n, ast.FunctionDef):
            if n.decorator_list:
                raise SandboxError("forbidden_construct", "decorators are not allowed", ln)
            if n.name.startswith("_"):
                raise SandboxError("forbidden_name", f"function name {n.name!r} is not allowed", ln)
            if n.name == ENTRY and n.col_offset == 0 and len(n.args.args) == 4:
                has = True
    if not has:
        raise SandboxError("no_rewrite", "define `def rewrite(g, W, P, T):` at the top level of the script")
    return tree


# ------------------------------------------------------------------------------ handles given to the script
_JSON = (str, int, float, bool, type(None))


def _jsonable(x, where: str, depth: int = 0):
    if depth > 8:
        raise SandboxError("bad_op", f"{where}: nested deeper than 8")
    if isinstance(x, _JSON):
        return x
    if isinstance(x, (list, tuple)):
        return [_jsonable(v, where, depth + 1) for v in x]
    if isinstance(x, dict):
        for k in x:
            if not isinstance(k, str):
                raise SandboxError("bad_op", f"{where}: dict keys must be strings (got {k!r})")
        return {k: _jsonable(v, where, depth + 1) for k, v in x.items()}
    raise SandboxError("bad_op", f"{where}: {type(x).__name__} is not a JSON value")


class WeightCatalogue:
    """Read-only metadata of the weights a rewrite may reference: the model's parameters and the ones derived so far."""

    def __init__(self, base, rec: "wd.Recorder"):
        self._base, self._rec = base, rec

    def has(self, k) -> bool:
        return k in self._rec.arrays or k in self._base

    def _arr(self, k):
        if k in self._rec.arrays:
            return self._rec.arrays[k]
        if k in self._base:
            return self._base[k]
        raise SandboxError("bad_weight", f"weight {k!r} does not exist (model parameters, e.g. {sorted(self._base)[:4]}; derived {sorted(self._rec.arrays)[:4]})")

    def shape(self, k):
        return [int(s) for s in self._arr(k).shape]

    def dtype(self, k) -> str:
        import numpy as np
        a = np.asarray(self._arr(k))
        d = self._rec.graph.get(wd.KEY, {}).get(k, {}).get("dtype")
        return d or wd.logical_dtype(self._rec.base_graph, k, a)

    def keys(self):
        return sorted(set(self._base) | set(self._rec.arrays))

    def value(self, *a, **k):
        raise SandboxError("weight_not_readable", "weight values are not readable in a rewrite: derive new weights from the model's parameters "
                                                  "with P.concat / P.slice / P.pad / P.mul ... so that they stay traceable (provenance)")

    get = value


class ProvenanceWeights:
    """P: the only way to create a weight. Every method records a declarative derivation (weight_derivations) and returns the new key."""

    def __init__(self, rec: "wd.Recorder", cat: WeightCatalogue):
        self._rec, self._cat = rec, cat

    def _guard(self, new, *srcs):
        if not isinstance(new, str) or not new.startswith(wd.PREFIX):
            raise SandboxError("bad_weight", f"derived weight keys must be strings starting with {wd.PREFIX!r} (got {new!r})")
        if len(self._rec.arrays) >= MAX_DERIVED:
            raise SandboxError("bad_weight", f"more than {MAX_DERIVED} derived weights")
        for s in srcs:
            if not isinstance(s, str) or not self._cat.has(s):
                raise SandboxError("weight_untraceable", f"weight {s!r} is neither a model parameter nor created by P (known: {self._cat.keys()[:6]}...)")

    def _do(self, fn, new, srcs, *a):
        self._guard(new, *srcs)
        try:
            return fn(new, *a)
        except wd.DerivationError as e:
            raise SandboxError("bad_weight", f"P: {e}") from None
        except (ValueError, TypeError, IndexError) as e:
            raise SandboxError("bad_weight", f"P: {type(e).__name__}: {e}") from None

    def concat(self, new, keys, axis=0):
        keys = list(keys)
        return self._do(lambda n: self._rec.concat(n, keys, axis), new, keys)

    def stack(self, new, keys, axis=0):
        keys = list(keys)
        return self._do(lambda n: self._rec.stack(n, keys, axis), new, keys)

    def slice(self, new, k, axis, start, stop):
        return self._do(lambda n: self._rec.slice(n, k, axis, start, stop), new, [k])

    def reshape(self, new, k, shape):
        return self._do(lambda n: self._rec.reshape(n, k, shape), new, [k])

    def transpose(self, new, k, perm):
        return self._do(lambda n: self._rec.transpose(n, k, perm), new, [k])

    def pad(self, new, k, pads, value=0.0):
        return self._do(lambda n: self._rec.pad(n, k, pads, value), new, [k])

    def scale(self, new, k, by):
        return self._do(lambda n: self._rec.scale(n, k, by), new, [k])

    def cast(self, new, k, dtype):
        return self._do(lambda n: self._rec.cast(n, k, _ldt(dtype)), new, [k])

    def mul(self, new, a, b):
        return self._do(lambda n: self._rec.mul(n, a, b), new, [a, b])

    def add(self, new, a, b):
        return self._do(lambda n: self._rec.add(n, a, b), new, [a, b])

    def zeros(self, new, shape, dtype="f16"):
        return self._do(lambda n: self._rec.const(n, shape, 0.0, _ldt(dtype)), new, [])

    def ones(self, new, shape, dtype="f16"):
        return self._do(lambda n: self._rec.const(n, shape, 1.0, _ldt(dtype)), new, [])


def _ldt(d: str) -> str:
    m = {"float16": "f16", "half": "f16", "float32": "f32", "float": "f32", "bfloat16": "bf16", "int32": "i32", "int64": "i64"}
    d = m.get(d, d)
    if d not in wd.LOGICAL:
        raise SandboxError("bad_weight", f"unknown dtype {d!r} (one of {wd.LOGICAL})")
    return d


class TensorMeta:
    """T: read-only shapes / dtypes of the graph being edited."""

    def __init__(self, ed: ge.Editor):
        self._ed = ed

    def _m(self, t):
        m = self._ed.g["tensors"].get(t)
        if m is None:
            raise SandboxError("bad_edit", f"no tensor {t!r} (it was removed, or never added with g.add_tensor)")
        return m

    def has(self, t) -> bool:
        return t in self._ed.g["tensors"]

    def shape(self, t):
        return list(self._m(t)["shape"])

    def dtype(self, t) -> str:
        return self._m(t)["dtype"]

    def numel(self, t) -> int:
        return int(math.prod(self._m(t)["shape"]))

    def names(self):
        return sorted(self._ed.g["tensors"])


class GraphHandle:
    """g: the edit handle. Reads return copies; every mutation is validated here, with messages that say what to do."""

    def __init__(self, ed: ge.Editor, cat: WeightCatalogue):
        self._ed, self._cat, self._added = ed, cat, 0

    # -- reads
    def ops(self):
        return [o["name"] for o in self._ed.g["ops"]]

    def op(self, name):
        for o in self._ed.g["ops"]:
            if o["name"] == name:
                return copy.deepcopy(o)
        raise SandboxError("bad_edit", f"no op {name!r} (the graph has {len(self._ed.g['ops'])} ops, e.g. {self.ops()[:5]})")

    def producer(self, t):
        p = ge.produced(self._ed.g).get(t)
        return self.op(p) if p else None

    def consumers(self, t):
        return [copy.deepcopy(o) for o in self._ed.g["ops"] if t in o.get("inputs", [])]

    def tensors(self):
        return sorted(self._ed.g["tensors"])

    def has_tensor(self, t) -> bool:
        return t in self._ed.g["tensors"]

    def inputs(self):
        return list(ge.io_names(self._ed.g)[0])

    def outputs(self):
        return list(ge.io_names(self._ed.g)[1])

    def unique(self, base):
        return self._ed.unique(str(base))

    # -- mutations
    def add_tensor(self, name, shape, dtype="f16"):
        if not isinstance(name, str) or not name:
            raise SandboxError("bad_edit", "tensor names are non-empty strings")
        if name in self._ed.g["tensors"]:
            raise SandboxError("bad_edit", f"tensor {name!r} already exists (g.unique(base) gives a free name; to reuse a name remove the old tensor first)")
        try:
            sh = [int(s) for s in shape]
        except (TypeError, ValueError):
            raise SandboxError("bad_edit", f"tensor {name!r}: shape must be a list of ints, got {shape!r}") from None
        if not sh or any(s <= 0 for s in sh):
            raise SandboxError("bad_edit", f"tensor {name!r}: shape {sh} must be non-empty and positive")
        if dtype not in wd.LOGICAL:
            raise SandboxError("bad_edit", f"tensor {name!r}: unknown dtype {dtype!r} (one of {wd.LOGICAL})")
        self._ed.add_tensor(name, sh, dtype)
        return name

    def remove_tensor(self, t):
        if t not in self._ed.g["tensors"]:
            raise SandboxError("bad_edit", f"no tensor {t!r}")
        if t in ge.io_names(self._ed.g)[0] + ge.io_names(self._ed.g)[1]:
            raise SandboxError("bad_edit", f"{t!r} is a graph input/output: a rewrite keeps the graph's interface")
        self._ed.g["tensors"].pop(t)

    def add_op(self, op):
        op = _jsonable(op, "g.add_op(op)")
        if not isinstance(op, dict):
            raise SandboxError("bad_op", "g.add_op takes a dict: {'name', 'op', 'inputs', 'outputs', ['weight', 'bias', 'shape', ...]}")
        for k in ("name", "op"):
            if not isinstance(op.get(k), str) or not op[k]:
                raise SandboxError("bad_op", f"g.add_op: field {k!r} (a non-empty string) is required")
        for k in ("inputs", "outputs"):
            if not isinstance(op.get(k, []), list) or not all(isinstance(t, str) for t in op.get(k, [])):
                raise SandboxError("bad_op", f"g.add_op({op['name']!r}): {k!r} must be a list of tensor names")
        if op["op"].startswith(("fused__", "region__")) or "sub_ops" in op:
            raise SandboxError("bad_op", f"g.add_op({op['name']!r}): fused / region ops are made by the flow's fusion hints, not by a rewrite")
        if self._added >= MAX_NEW_OPS:
            raise SandboxError("bad_op", f"more than {MAX_NEW_OPS} added ops")
        if any(o["name"] == op["name"] for o in self._ed.g["ops"]):
            raise SandboxError("bad_edit", f"op name {op['name']!r} exists (g.unique(base) gives a free name)")
        for t in op.get("outputs", []):
            if t not in self._ed.g["tensors"]:
                raise SandboxError("bad_edit", f"op {op['name']!r}: output tensor {t!r} is not registered: call g.add_tensor({t!r}, shape, dtype) first")
        for t in op.get("inputs", []):
            if t not in self._ed.g["tensors"]:
                raise SandboxError("dangling_tensor", f"op {op['name']!r} reads tensor {t!r}, which does not exist in the graph "
                                                      f"(removed earlier, or never added: add it with g.add_tensor and an op that produces it)")
        self._check_weights(op)
        self._added += 1
        self._ed.add_op(op)
        return op["name"]

    def _check_weights(self, op):
        for f in WEIGHT_FIELDS:
            k = op.get(f)
            if k is not None and (not isinstance(k, str) or not self._cat.has(k)):
                raise SandboxError("weight_untraceable",
                                   f"op {op['name']!r}: {f} {k!r} is neither a parameter of the model nor a weight created by P. Weights can only be "
                                   f"created with P.concat / P.slice / P.pad / P.mul ... (traced back to the parameters); parameters: {self._cat.keys()[:5]}...")

    def update_op(self, name, **fields):
        fields = _jsonable(fields, f"g.update_op({name!r})")
        o = next((o for o in self._ed.g["ops"] if o["name"] == name), None)
        if o is None:
            raise SandboxError("bad_edit", f"no op {name!r}")
        if any(k in fields for k in ("name",)):
            raise SandboxError("bad_edit", "an op's name cannot change: add a new op and remove the old one")
        probe = {**o, **fields}
        self._check_weights(probe)
        for t in fields.get("inputs", []) + fields.get("outputs", []):
            if t not in self._ed.g["tensors"]:
                raise SandboxError("dangling_tensor", f"op {name!r}: tensor {t!r} does not exist (g.add_tensor first)")
        if "shape" in fields and isinstance(o.get("shape"), dict) and isinstance(fields["shape"], dict):
            fields["shape"] = {**o["shape"], **fields["shape"]}
        o.update(fields)

    def remove_ops(self, names):
        names = list(names)
        have = {o["name"] for o in self._ed.g["ops"]}
        miss = [n for n in names if n not in have]
        if miss:
            raise SandboxError("bad_edit", f"no such ops {miss} (the graph has e.g. {sorted(have)[:5]})")
        self._ed.remove_ops(names)

    def rewire(self, old, new, only_ops=None):
        for t in (old, new):
            if t not in self._ed.g["tensors"]:
                raise SandboxError("bad_edit", f"rewire: no tensor {t!r}")
        return self._ed.rewire(old, new, only_ops)


# ------------------------------------------------------------------------------ restricted execution
def _safe_builtins(log: list):
    def _range(*a):
        r = range(*[int(x) for x in a])
        if len(r) > MAX_RANGE:
            raise SandboxError("step_budget", f"range of {len(r)} elements exceeds the cap of {MAX_RANGE}")
        return r

    def _print(*a, **k):
        log.append(" ".join(str(x) for x in a)[:500])
        if len(log) > 200:
            del log[0]

    b = {k: v for k, v in {"len": len, "enumerate": enumerate, "zip": zip, "sorted": sorted, "min": min, "max": max, "sum": sum,
                           "any": any, "all": all, "list": list, "dict": dict, "set": set, "tuple": tuple, "int": int, "float": float,
                           "str": str, "bool": bool, "isinstance": isinstance, "abs": abs, "reversed": reversed, "map": map,
                           "filter": filter, "round": round, "divmod": divmod, "repr": repr, "frozenset": frozenset, "slice": slice,
                           "ValueError": ValueError, "KeyError": KeyError, "Exception": Exception, "IndexError": IndexError,
                           "TypeError": TypeError, "AssertionError": AssertionError, "None": None, "True": True, "False": False}.items()}
    b["range"], b["print"] = _range, _print
    return b


class _MathNS:
    ceil = staticmethod(math.ceil)
    floor = staticmethod(math.floor)
    sqrt = staticmethod(math.sqrt)
    prod = staticmethod(math.prod)
    gcd = staticmethod(math.gcd)
    pi = math.pi


@dataclass
class SandboxResult:
    ok: bool
    script_sha: str
    graph: dict | None = None                   # the edited graph (weight_derivations recorded in it)
    patch: dict | None = None                   # graph_edit.diff(base, edited)
    derivations: dict = field(default_factory=dict)   # {key: {dtype, shape, expr}} created by this rewrite
    provenance: dict = field(default_factory=dict)    # {key: {"leaves": [model parameters it traces back to], "exact": bool}}
    tensor_map: dict = field(default_factory=dict)
    description: str = ""
    error: dict | None = None                   # {"code", "msg", "line"} when not ok
    log: list = field(default_factory=list)     # what the script printed
    steps: int = 0

    def to_dict(self) -> dict:
        return {"ok": self.ok, "script_sha": self.script_sha, "error": self.error, "description": self.description,
                "ops_removed": (self.patch or {}).get("remove_ops"), "ops_added": [o["name"] for o in (self.patch or {}).get("add_ops", [])],
                "derivations": sorted(self.derivations), "provenance": self.provenance, "tensor_map": self.tensor_map, "log": self.log,
                "steps": self.steps}


class _Alarm:
    """Wall-clock limit (SIGALRM; only on the main thread of a POSIX process, else the step budget is the limit)."""

    def __init__(self, s: float):
        self.s = s
        self.on = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread() and s > 0

    def __enter__(self):
        if self.on:
            def h(*_):
                raise SandboxError("timeout", f"the rewrite ran longer than {self.s:g} s")
            self.old = signal.signal(signal.SIGALRM, h)
            signal.setitimer(signal.ITIMER_REAL, self.s)
        return self

    def __exit__(self, *a):
        if self.on:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self.old)


def _call_script(code, env, args, max_steps):
    """Run ``rewrite(*args)`` with a traced line budget on frames of the script only."""
    steps = [0]

    def tracer(frame, event, arg):
        if frame.f_code.co_filename != "<rewrite>":
            return None

        def local(frame, event, arg):
            if event == "line":
                steps[0] += 1
                if steps[0] > max_steps:
                    raise SandboxError("step_budget", f"the rewrite executed more than {max_steps} statements (unbounded or very long loop)", frame.f_lineno)
            return local
        return local
    old = sys.gettrace()
    exec(code, env)                                                   # defines functions only (no imports, no I/O possible)
    sys.settrace(tracer)
    try:
        r = env[ENTRY](*args)
    finally:
        sys.settrace(old)
    return r, steps[0]


def _line_of(tb) -> int | None:
    ln = None
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == "<rewrite>":
            ln = tb.tb_lineno
        tb = tb.tb_next
    return ln


def run_rewrite(src: str, graph: dict, base_weights, *, max_steps: int = MAX_STEPS, timeout_s: float = TIMEOUT_S,
                tensor_map: dict | None = None) -> SandboxResult:
    """Run the rewrite script ``src`` on a COPY of ``graph`` (``base_weights``: the original weights, a mapping; values are only read to
    evaluate derived weights). Never raises: a rejected script is ``SandboxResult(ok=False, error={...})``; ``graph`` is never modified."""
    sha = hashlib.sha1(src.encode()).hexdigest()[:12]
    res = SandboxResult(ok=False, script_sha=sha)
    log: list = []
    try:
        tree = check_script(src)
        ed = ge.Editor(graph, base_weights)
        cat = WeightCatalogue(base_weights, ed.derive)
        g, P, T = GraphHandle(ed, cat), ProvenanceWeights(ed.derive, cat), TensorMeta(ed)
        env = {"__builtins__": _safe_builtins(log), "math": _MathNS}
        code = compile(tree, "<rewrite>", "exec")
        with _Alarm(timeout_s):
            try:
                ret, res.steps = _call_script(code, env, (g, cat, P, T), max_steps)
            except SandboxError:
                raise
            except RecursionError:
                raise SandboxError("step_budget", "recursion too deep") from None
            except (KeyError, IndexError, TypeError, ValueError, AttributeError, NameError, ZeroDivisionError, AssertionError, Exception) as e:  # noqa: B014
                import traceback
                raise SandboxError("script_error", f"{type(e).__name__}: {e}", _line_of(e.__traceback__)) from None
        opts = ret if isinstance(ret, dict) else {}
        if opts.get("tensor_map"):
            res.tensor_map = _jsonable(opts["tensor_map"], "return value tensor_map")
        if tensor_map:
            res.tensor_map = {**res.tensor_map, **tensor_map}
        res.description = str(opts.get("description", ""))[:500]
        res.graph = _finish(ed, cat)
        res.patch = ge.diff(graph, res.graph, tensor_map=res.tensor_map or None)
        res.derivations = {k: v for k, v in (res.graph.get(wd.KEY) or {}).items() if k not in (graph.get(wd.KEY) or {})}
        res.provenance = {k: {"leaves": sorted(_roots(res.graph, k)),
                              "exact": wd.is_exact(res.graph, [k]), "shape": v["shape"], "dtype": v["dtype"]} for k, v in res.derivations.items()}
        res.ok = True
    except SandboxError as e:
        res.error = e.to_dict()
    except Exception as e:  # noqa: BLE001 -- the sandbox never raises
        res.error = {"code": "internal", "msg": f"{type(e).__name__}: {e}", "line": None}
    res.log = log
    return res


def _roots(graph: dict, key: str, acc: set | None = None) -> set:
    """The model parameters (not derived keys) a derived weight is computed from, through every intermediate derivation."""
    acc = set() if acc is None else acc
    d = wd.derivations(graph)
    for leaf in wd.expr_leaves(d[key]["expr"]):
        if leaf in d:
            _roots(graph, leaf, acc)
        else:
            acc.add(leaf)
    return acc


def _finish(ed: ge.Editor, cat: WeightCatalogue) -> dict:
    """Close an edit: topological order (dangling / cyclic graphs are reasons), orphan tensors pruned, weights re-checked."""
    g = ed.g
    # a removed op's output that is still read (or a graph output nobody produces) is the classic mistake: say so
    ins, outs = ge.io_names(g)
    prod = ge.produced(g)
    for o in g["ops"]:
        for t in o.get("inputs", []):
            if t not in prod and t not in ins and not g["tensors"].get(t, {}).get("const"):
                raise SandboxError("dangling_tensor", f"op {o['name']!r} reads tensor {t!r} that nothing produces any more (its producer was removed): "
                                                      f"add an op that writes {t!r}, or rewire the readers (g.rewire(old, new)) to a tensor that exists")
    for t in outs:
        if t not in prod and t not in ins:
            raise SandboxError("dangling_tensor", f"graph output {t!r} is not produced by any op after the rewrite (keep an op that writes it)")
    try:
        ge.toposort(g)
    except ge.PatchError as e:
        raise SandboxError("cycle" if e.code == "cycle" else "dangling_tensor", str(e)) from None
    for o in g["ops"]:
        for f in WEIGHT_FIELDS:
            k = o.get(f)
            if k is not None and not cat.has(k):
                raise SandboxError("weight_untraceable", f"op {o['name']!r}: {f} {k!r} is neither a model parameter nor created by P")
        for t in list(o.get("inputs", [])) + list(o.get("outputs", [])):
            if t not in g["tensors"]:
                raise SandboxError("bad_edit", f"op {o['name']!r}: tensor {t!r} is not registered (g.add_tensor)")
    used = set(ins) | set(outs)
    for o in g["ops"]:
        used.update(o.get("inputs", []))
        used.update(o.get("outputs", []))
        used.update(o.get("internal", []))
        for s in o.get("sub_ops", []):
            used.update(s.get("inputs", []))
            used.update(s.get("outputs", []))
    for t in [t for t in g["tensors"] if t not in used and not g["tensors"][t].get("const")]:
        del g["tensors"][t]
    return g


# ------------------------------------------------------------------------------ IR helpers + a small CLI
def load_ir(ir_dir) -> tuple[dict, dict]:
    import numpy as np
    d = Path(ir_dir)
    z = np.load(d / "weights.npz")
    return json.loads((d / "graph.json").read_text()), {k: z[k] for k in z.files}


def write_result_ir(res: SandboxResult, src_ir, out) -> Path:
    """The edited IR dir of an ok result (graph.json, weights.npz = source + derived, io.npz): what graph_verify and the flow consume."""
    from .kb_proposals import write_variant_ir
    if not res.ok:
        raise ValueError("the rewrite was rejected: " + json.dumps(res.error))
    return write_variant_ir(Path(src_ir), res.graph, Path(out))


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="run a rewrite script on an IR in the restricted interpreter (no flow, no GPU)")
    ap.add_argument("ir")
    ap.add_argument("script")
    ap.add_argument("--out", help="write the edited IR dir here")
    ap.add_argument("--verify", action="store_true", help="then run graph_verify (CPU)")
    a = ap.parse_args(argv)
    g, w = load_ir(a.ir)
    res = run_rewrite(Path(a.script).read_text(), g, w)
    print(json.dumps(res.to_dict(), indent=1, default=str))
    if res.ok and a.out:
        e = write_result_ir(res, a.ir, a.out)
        if a.verify:
            from . import graph_verify as gv
            v = gv.verify(a.ir, e, tensor_map=res.tensor_map or None)
            print(v.summary())
            return 0 if v.ok else 1
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
