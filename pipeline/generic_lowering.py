"""Fallback lowering for fx nodes the per-op branches in extract_graph reject.

extract() first tries its specific lowering for every node; when that
raises NotImplementedError for a node this module `can_lower`, extract
retries with the node routed here (see extract_graph._extract_fp32). So
graphs that extracted before are unchanged, and everything else lands on
the shape-generic ops in reference_kernels_ext:

data movement (view/reshape/permute/transpose/slicing/split/expand/roll/
pad/unfold/...): the op is replayed on an index-encoded copy of its input
(float64 arange + 1). The result says which input element each output
element reads (0 = constant zero), whatever the op was, which becomes

  * `view` (with `offset` for a contiguous window) — zero cost,
  * `strided_copy` when the map is affine, or
  * `gather` with a constant index tensor otherwise.

compute: broadcasting elementwise (`ewise2`, `ewise_scalar`, `ewise1`),
reductions over any dims (`reduce`, permuting first when the dims aren't
adjacent), softmax over any dim, N-way cat/stack (`catn`, tree above 8),
matmul with batch broadcasting (`matmul`/`bmm` + expand copies), einsum
(pairwise, as permute + bmm), grouped conv, functional pool / norm forms,
BatchNorm1d/3d, InstanceNorm, logsumexp, F.normalize, constant tensors
(`get_attr`).
"""
from __future__ import annotations

import math
import operator
import re
from typing import Any

import numpy as np
import torch
import torch.fx

from modelblaster.pipeline.reference_kernels_ext import (
    EWISE1_OPS, EWISE2_OPS, EWISE_SCALAR_OPS, MB_MAXR, REDUCE_OPS)

_F = torch.nn.functional

# -- classification ------------------------------------------------------------

_MOVE_NAMES = {
    "view", "reshape", "permute", "transpose", "t", "contiguous", "clone",
    "flatten", "unflatten", "squeeze", "unsqueeze", "expand", "expand_as",
    "repeat", "narrow", "select", "split", "chunk", "unbind", "tensor_split",
    "unfold", "flip", "fliplr", "flipud", "roll", "movedim", "moveaxis",
    "swapaxes", "swapdims", "detach", "view_as", "reshape_as", "diagonal",
    "float", "to", "type_as", "getitem", "pad", "pixel_shuffle",
    "pixel_unshuffle", "channel_shuffle", "dropout", "tril", "triu",
    "repeat_interleave", "tile", "rot90", "identity",
}
_IDENTITY = {"dropout", "float", "type_as", "detach", "contiguous", "clone", "identity"}
# second arg only supplies a shape/dtype
_SHAPE_ONLY_ARG = {"expand_as", "view_as", "reshape_as", "type_as"}

_UNARY = {  # name -> (ewise1 op, p0, p1)
    "neg": ("neg",), "exp": ("exp",), "log": ("log",), "sqrt": ("sqrt",),
    "rsqrt": ("rsqrt",), "abs": ("abs",), "tanh": ("tanh",),
    "sigmoid": ("sigmoid",), "relu": ("relu",), "sin": ("sin",),
    "cos": ("cos",), "erf": ("erf",), "silu": ("silu",), "mish": ("mish",),
    "hardswish": ("hardswish",), "hardsigmoid": ("hardsigmoid",),
    "log1p": ("log1p",), "expm1": ("expm1",), "sign": ("sign",),
    "floor": ("floor",), "ceil": ("ceil",), "round": ("round",),
    "logical_not": ("logical_not",), "reciprocal": ("reciprocal",),
    "square": ("square",), "softplus": ("softplus",),
}
_BINARY = {  # name -> ewise2 / ewise_scalar op, plus whether it commutes
    "add": ("add", True), "sub": ("sub", False), "subtract": ("sub", False),
    "mul": ("mul", True), "multiply": ("mul", True), "truediv": ("div", False),
    "div": ("div", False), "true_divide": ("div", False), "pow": ("pow", False),
    "maximum": ("max", True), "minimum": ("min", True), "eq": ("eq", True),
    "ne": ("ne", True), "lt": ("lt", False), "le": ("le", False),
    "gt": ("gt", False), "ge": ("ge", False), "fmax": ("max", True),
    "fmin": ("min", True),
}
_SWAPPED_CMP = {"lt": "gt", "gt": "lt", "le": "ge", "ge": "le"}
_REDUCE = {"sum": "sum", "mean": "mean", "amax": "max", "amin": "min",
           "prod": "prod", "max": "max", "min": "min", "argmax": "argmax",
           "argmin": "argmin", "var": "var", "std": "std", "norm": "norm"}
_MATMUL = {"matmul", "bmm", "mm"}


def _name(node) -> str:
    if node.op == "call_module":
        return type(node.graph.owning_module.get_submodule(node.target)).__name__
    if node.op == "get_attr":
        return "get_attr"
    t = node.target
    if isinstance(t, str):
        return t.strip("_") if t.startswith("__") else t
    return getattr(t, "__name__", str(t)).strip("_")


def _is_tensor_node(node) -> bool:
    # ShapeProp's TensorMetadata is itself a NamedTuple; tuple-valued nodes
    # (split, max(dim), LSTM) carry a plain tuple of them.
    return hasattr(node.meta.get("tensor_meta"), "shape")


def _has_tensor_index(node) -> bool:
    """getitem / mb_setitem whose index holds a tensor (advanced indexing)."""
    idx = node.args[1] if len(node.args) > 1 else None
    items = idx if isinstance(idx, (tuple, list)) else (idx,)
    return any(isinstance(it, torch.fx.Node) and _is_tensor_node(it) for it in items)


def _prod(xs) -> int:
    return int(np.prod([int(x) for x in xs])) if len(xs) else 1


def _contig_strides(shape) -> list[int]:
    st, acc = [], 1
    for s in reversed(shape):
        st.append(acc)
        acc *= int(s)
    return st[::-1]


def _merge_dims(shape, *strides):
    """Drop size-1 dims and merge adjacent dims that are contiguous in every
    stride set. Returns (shape, [strides...])."""
    keep = [i for i, s in enumerate(shape) if s != 1]
    dims = [int(shape[i]) for i in keep]
    sts = [[int(st[i]) for i in keep] for st in strides]
    i = len(dims) - 1
    while i > 0:
        if all(st[i - 1] == st[i] * dims[i] for st in sts):
            dims[i - 1] *= dims[i]
            for st in sts:
                st[i - 1] = st[i]
            del dims[i]
            for st in sts:
                del st[i]
        i -= 1
    return dims, sts


def _pad6(dims, sts):
    if len(dims) > MB_MAXR:
        raise NotImplementedError(f"rank {len(dims)} > {MB_MAXR} after merging dims")
    pad = MB_MAXR - len(dims)
    return [1] * pad + dims, [[0] * pad + st for st in sts]


class GenericLowering:
    def __init__(self, gm: torch.fx.GraphModule, tensors: dict, ops: list,
                 weights: dict, weight_dtype, skip_nodes: set):
        self.gm, self.tensors, self.ops, self.weights = gm, tensors, ops, weights
        self.wdtype = weight_dtype
        self.skip = skip_nodes
        self._tmp = 0
        # (tuple-producing node, index) -> IR tensor, for modules that return
        # tuples (MultiheadAttention, RNNs); their getitem users alias these.
        self.tuple_items: dict[tuple[str, int], str] = {}

    # -- predicates ---------------------------------------------------------
    def can_lower(self, node: torch.fx.Node) -> bool:
        if node.op == "get_attr":
            return True
        if node.op not in ("call_function", "call_method", "call_module"):
            return False
        n = _name(node)
        if node.op == "call_module":
            return n in _MODULES
        if n == "getattr":
            return len(node.args) == 2 and node.args[1] in ("T", "mT", "values", "indices")
        return (n in _MOVE_NAMES or n in _UNARY or n in _BINARY or n in _REDUCE
                or n in _MATMUL or n in _FUNCS or n in _METHODS_EXTRA)

    def obviously_generic(self, node: torch.fx.Node) -> bool:
        """Nodes no specific branch handles; route them without a retry."""
        if node.op == "get_attr":
            return True
        if node.op == "call_method":
            n = node.target
            if n == "chunk":  # extract_graph handles chunk(2, dim=1) on 4-D only
                return not (len(node.args) >= 2 and node.args[1] == 2)
            return n != "flip" and self.can_lower(node)
        return False

    # -- IR helpers ----------------------------------------------------------
    def _meta_shape(self, node) -> list[int]:
        tm = node.meta.get("tensor_meta")
        if tm is None or not hasattr(tm, "shape"):
            raise NotImplementedError(f"{node.name}: no tensor shape")
        return [int(s) for s in tm.shape]

    def _shape(self, name: str) -> list[int]:
        return [int(s) for s in self.tensors[name]["shape"]]

    def _new(self, base: str, shape) -> str:
        self._tmp += 1
        name = f"{base}__g{self._tmp}"
        self.tensors[name] = {"shape": [int(s) for s in shape], "dtype": self._dt(), "quant": None}
        return name

    def _dt(self) -> str:
        return "f16" if self.wdtype == np.float16 else "f32"

    def _out(self, node, shape=None) -> str:
        self.tensors[node.name] = {"shape": shape if shape is not None else self._meta_shape(node),
                                   "dtype": self._dt(), "quant": None}
        return node.name

    def _emit(self, name, op, inputs, outputs, shape, **extra) -> None:
        d = {"name": name, "op": op, "inputs": list(inputs), "outputs": list(outputs),
             "shape": shape}
        d.update(extra)
        self.ops.append(d)

    def _const(self, name: str, value, dtype=None) -> str:
        arr = np.asarray(value.detach().cpu().float().numpy()
                         if isinstance(value, torch.Tensor) else value)
        key = f"const.{name}"
        self.weights[key] = arr.astype(dtype or self.wdtype).reshape(arr.shape)
        self.tensors[name] = {"shape": list(arr.shape), "dtype": self._dt(), "quant": None,
                              "const": key}
        return name

    def _is_graph_output(self, node) -> bool:
        return any(u.op == "output" for u in node.users)

    def _tensor_arg(self, a) -> str:
        if not isinstance(a, torch.fx.Node):
            raise NotImplementedError(f"expected tensor arg, got {type(a).__name__}")
        if a.name not in self.tensors:
            raise NotImplementedError(f"tensor {a.name} not materialized")
        return a.name

    # -- elementary emitters -------------------------------------------------
    def copy_affine(self, base, src, out_shape, off, strides, out_name=None, force_copy=False):
        """out[i] = src[off + sum i_k*strides_k]; a view when that is contiguous."""
        out = out_name or self._new(base, out_shape)
        self.tensors.setdefault(out, {"shape": list(out_shape), "dtype": self._dt(), "quant": None})
        dims, (st,) = _merge_dims(out_shape, strides)
        contiguous = (len(dims) == 0 or (len(dims) == 1 and st[0] == 1))
        if contiguous and not force_copy:
            extra = {"offset": int(off)} if off else {}
            self._emit(base, "view", [src], [out], {"n": _prod(out_shape)}, **extra)
            return out
        dims, (st,) = _pad6(dims, [st])
        shape = {"off": int(off)}
        shape.update({f"d{i}": dims[i] for i in range(MB_MAXR)})
        shape.update({f"s{i}": st[i] for i in range(MB_MAXR)})
        self._emit(base, "strided_copy", [src], [out], shape)
        return out

    def permute(self, base, src, perm, out_name=None):
        shape = self._shape(src)
        if list(perm) == list(range(len(shape))):
            return src if out_name is None else self.copy_affine(
                base, src, shape, 0, _contig_strides(shape), out_name)
        cs = _contig_strides(shape)
        return self.copy_affine(base, src, [shape[p] for p in perm], 0,
                                [cs[p] for p in perm], out_name)

    def reshape(self, base, src, shape, out_name=None):
        return self.copy_affine(base, src, shape, 0, _contig_strides(shape), out_name)

    def expand(self, base, src, shape):
        s = self._shape(src)
        s = [1] * (len(shape) - len(s)) + s
        if s == list(shape):
            return src
        cs = _contig_strides(s)
        return self.copy_affine(base, src, shape, 0,
                                [0 if (s[i] == 1 and shape[i] != 1) else cs[i]
                                 for i in range(len(shape))])

    def ewise1(self, base, src, fn, p0=0.0, p1=0.0, out_name=None):
        out = out_name or self._new(base, self._shape(src))
        self._emit(base, "ewise1", [src], [out], {"n": _prod(self._shape(src))},
                   opc=EWISE1_OPS[fn], fn=fn, p0=float(p0), p1=float(p1))
        return out

    def ewise_scalar(self, base, src, fn, c, out_name=None):
        out = out_name or self._new(base, self._shape(src))
        self._emit(base, "ewise_scalar", [src], [out], {"n": _prod(self._shape(src))},
                   opc=EWISE_SCALAR_OPS[fn], fn=fn, c=float(c))
        return out

    def ewise2(self, base, a, b, fn, alpha=1.0, out_name=None):
        sa, sb = self._shape(a), self._shape(b)
        out_shape = list(torch.broadcast_shapes(tuple(sa), tuple(sb)))
        r = len(out_shape)

        def bstrides(s):
            s = [1] * (r - len(s)) + s
            cs = _contig_strides(s)
            return [0 if (s[i] == 1 and out_shape[i] != 1) else cs[i] for i in range(r)]
        dims, (ta, tb) = _merge_dims(out_shape, bstrides(sa), bstrides(sb))
        dims, (ta, tb) = _pad6(dims, [ta, tb])
        out = out_name or self._new(base, out_shape)
        shape = {f"d{i}": dims[i] for i in range(MB_MAXR)}
        shape.update({f"sa{i}": ta[i] for i in range(MB_MAXR)})
        shape.update({f"sb{i}": tb[i] for i in range(MB_MAXR)})
        self._emit(base, "ewise2", [a, b], [out], shape, opc=EWISE2_OPS[fn], fn=fn,
                   alpha=float(alpha))
        return out

    def reduce(self, base, src, dims, fn, keepdim=False, out_name=None):
        shape = self._shape(src)
        r = len(shape)
        dims = sorted({d % r for d in dims}) if dims is not None else list(range(r))
        keep = [d for d in range(r) if d not in dims]
        out_shape = [1 if d in dims else shape[d] for d in range(r)] if keepdim else \
            [shape[d] for d in keep]
        if fn == "std":
            v = self.reduce(base, src, dims, "var_unbiased", keepdim)
            return self.ewise1(base, v, "sqrt", out_name=out_name)
        if dims == list(range(dims[0], dims[-1] + 1)) if dims else True:
            outer = _prod(shape[:dims[0]]) if dims else _prod(shape)
            R = _prod([shape[d] for d in dims])
            inner = _prod(shape[dims[-1] + 1:]) if dims else 1
            x = src
        else:
            x = self.permute(base, src, keep + dims)
            outer, R, inner = _prod([shape[d] for d in keep]), _prod([shape[d] for d in dims]), 1
        out = out_name or self._new(base, out_shape)
        self.tensors.setdefault(out, {"shape": out_shape, "dtype": self._dt(), "quant": None})
        self._emit(base, "reduce", [x], [out], {"outer": outer, "R": R, "inner": inner},
                   opc=REDUCE_OPS[fn], fn=fn)
        return out

    def softmax(self, base, src, dim, log=False, out_name=None):
        shape = self._shape(src)
        d = dim % len(shape)
        out = out_name or self._new(base, shape)
        self._emit(base, "softmax_dim", [src], [out],
                   {"outer": _prod(shape[:d]), "R": shape[d], "inner": _prod(shape[d + 1:])},
                   log_out=int(log))
        return out

    def cat(self, base, srcs, dim, out_name=None, stack=False):
        shapes = [self._shape(s) for s in srcs]
        r = len(shapes[0]) + (1 if stack else 0)
        d = dim % r
        if stack:
            outer, inner = _prod(shapes[0][:d]), _prod(shapes[0][d:])
            cs = [1] * len(srcs)
            out_shape = shapes[0][:d] + [len(srcs)] + shapes[0][d:]
        else:
            outer, inner = _prod(shapes[0][:d]), _prod(shapes[0][d + 1:])
            cs = [s[d] for s in shapes]
            out_shape = shapes[0][:d] + [sum(cs)] + shapes[0][d + 1:]
        return self._catn(base, list(zip(srcs, cs)), outer, inner, out_shape, out_name)

    def _catn(self, base, parts, outer, inner, out_shape, out_name):
        while len(parts) > 8:
            groups = [parts[i:i + 8] for i in range(0, len(parts), 8)]
            parts = []
            for g in groups:
                c = sum(c for _, c in g)
                t = self._catn(base, g, outer, inner, [outer, c, inner], None)
                parts.append((t, c))
        out = out_name or self._new(base, out_shape)
        self.tensors.setdefault(out, {"shape": out_shape, "dtype": self._dt(), "quant": None})
        shape = {"n_in": len(parts), "outer": outer, "inner": inner}
        shape.update({f"c{k}": (parts[k][1] if k < len(parts) else 0) for k in range(8)})
        self._emit(base, "catn", [p for p, _ in parts], [out], shape)
        return out

    def matmul(self, base, a, b, out_name=None):
        sa, sb = self._shape(a), self._shape(b)
        squeeze_m = squeeze_n = False
        if len(sa) == 1:
            a, sa, squeeze_m = self.reshape(base, a, [1, sa[0]]), [1, sa[0]], True
        if len(sb) == 1:
            b, sb, squeeze_n = self.reshape(base, b, [sb[0], 1]), [sb[0], 1], True
        M, K, N = sa[-2], sa[-1], sb[-1]
        if sb[-2] != K:
            raise NotImplementedError(f"matmul K mismatch {sa} @ {sb}")
        batch = list(torch.broadcast_shapes(tuple(sa[:-2]), tuple(sb[:-2])))
        res_shape = batch + [M, N]
        final_shape = [s for i, s in enumerate(res_shape)
                       if not ((squeeze_m and i == len(res_shape) - 2) or
                               (squeeze_n and i == len(res_shape) - 1))]
        direct = out_name is not None and not squeeze_m and not squeeze_n
        out = out_name if direct else self._new(base, res_shape)
        self.tensors.setdefault(out, {"shape": res_shape, "dtype": self._dt(), "quant": None})
        if len(sb) == 2:
            # [..., M, K] @ [K, N]: rows of a are independent -> one GEMM.
            self._emit(base, "matmul", [a, b], [out], {"M": _prod(sa[:-1]), "K": K, "N": N})
        else:
            a2 = self.expand(base, a, batch + [M, K])
            b2 = self.expand(base, b, batch + [K, N])
            self._emit(base, "bmm", [a2, b2], [out], {"batch": _prod(batch), "M": M, "K": K, "N": N})
        if direct:
            return out
        return self.reshape(base, out, final_shape, out_name)

    # -- entry -----------------------------------------------------------------
    def lower(self, node: torch.fx.Node) -> None:
        if node.op == "get_attr":
            val = self.gm
            for atom in node.target.split("."):
                val = getattr(val, atom)
            if not isinstance(val, torch.Tensor):
                raise NotImplementedError(f"get_attr {node.target} is not a tensor")
            self._const(node.name, val)
            return
        n = _name(node)
        if node.op == "call_module":
            return _MODULES[n](self, node, node.graph.owning_module.get_submodule(node.target))
        if n == "getattr" and node.args[1] in ("values", "indices"):
            k = 0 if node.args[1] == "values" else 1
            if (node.args[0].name, k) in self.tuple_items:   # topk / sort
                src = self.tuple_items[(node.args[0].name, k)]
                self._emit(node.name, "view", [src], [self._out(node, self._shape(src))],
                           {"n": _prod(self._shape(src))})
                return
            return self._lower_minmax_item(node, k)
        if n == "getitem" and isinstance(node.args[0], torch.fx.Node) and \
                (node.args[0].name, node.args[1]) in self.tuple_items:
            src = self.tuple_items[(node.args[0].name, node.args[1])]
            if isinstance(src, list):  # nested tuple, e.g. LSTM (h_n, c_n)
                for i, t in enumerate(src):
                    self.tuple_items[(node.name, i)] = t
                return
            self._emit(node.name, "view", [src], [self._out(node, self._shape(src))],
                       {"n": _prod(self._shape(src))})
            return
        if n == "getitem" and isinstance(node.args[0], torch.fx.Node) and \
                _name(node.args[0]) in ("max", "min") and isinstance(node.args[1], int):
            return self._lower_minmax_item(node, node.args[1])
        if n == "getitem" and _has_tensor_index(node):
            return _f_index(self, node)
        if n in _FUNCS:
            return _FUNCS[n](self, node)
        if n in _MOVE_NAMES or (n == "getattr"):
            return self._lower_movement(node)
        if n in _METHODS_EXTRA:
            return _METHODS_EXTRA[n](self, node)
        if n in _UNARY:
            return self._lower_unary(node, n)
        if n in _BINARY:
            return self._lower_binary(node, n)
        if n in _REDUCE:
            return self._lower_reduce(node, n)
        if n in _MATMUL:
            a, b = node.args[:2]
            self.matmul(node.name, self._tensor_arg(a), self._tensor_arg(b), self._out(node))
            return
        raise NotImplementedError(f"generic lowering: no rule for {n} at {node.name}")

    # -- data movement ---------------------------------------------------------
    def _root(self, name: str, roots: dict):
        """Index-encoded stand-in for IR tensor `name`: element k of the j-th
        root reads as (sum of earlier roots' sizes) + k + 1, so one probe can
        trace data movement over several tensors (0 = constant zero)."""
        if name not in roots:
            base = sum(int(t.numel()) for t in roots.values())
            shape = self._shape(name)
            roots[name] = torch.arange(base + 1, base + _prod(shape) + 1,
                                       dtype=torch.float64).reshape(shape)
        return roots[name]

    def _probe(self, a, roots: dict):
        if isinstance(a, torch.fx.Node):
            if a.name in self.tensors:
                return self._root(a.name, roots)
            if self._is_movement(a):
                return self._apply(a, roots)
            raise NotImplementedError(f"cannot see through {a.name} ({_name(a)})")
        if isinstance(a, (list, tuple)):
            return type(a)(self._probe(x, roots) for x in a)
        if isinstance(a, slice):
            return slice(self._probe(a.start, roots), self._probe(a.stop, roots),
                         self._probe(a.step, roots))
        if isinstance(a, torch.dtype):
            return torch.float64 if a.is_floating_point else a
        return a

    def _is_movement(self, node) -> bool:
        if node.op == "call_module":
            return _name(node) in ("Dropout", "Identity", "Flatten", "Unflatten",
                                   "ZeroPad2d", "ConstantPad2d", "PixelShuffle", "ChannelShuffle")
        if node.op not in ("call_function", "call_method"):
            return False
        n = _name(node)
        if n == "getattr":
            return node.args[1] in ("T", "mT")
        if n in ("getitem", "mb_setitem") and _has_tensor_index(node):
            return False
        return n in _MOVE_NAMES or n in ("mb_setitem", "mb_copy")

    def _apply(self, node, roots):
        n = _name(node)
        if node.op == "call_module":
            mod = node.graph.owning_module.get_submodule(node.target)
            return mod(self._probe(node.args[0], roots))
        args = list(node.args)
        if n in _SHAPE_ONLY_ARG:
            other = args[1]
            args = [self._probe(args[0], roots),
                    torch.empty(self._meta_shape(other), dtype=torch.float64)]
        elif n == "mb_copy":     # x only supplies the shape
            args = [torch.empty(self._meta_shape(args[0]), dtype=torch.float64),
                    self._probe(args[1], roots)]
        elif n == "mb_setitem" and not isinstance(args[2], torch.fx.Node):
            # a scalar value: route it through a one-element constant
            c = self._const(f"{node.name}__v{len(roots)}", torch.tensor([float(args[2])]))
            args = [self._probe(args[0], roots), self._probe(args[1], roots),
                    self._root(c, roots)[0]]
        else:
            args = [self._probe(x, roots) for x in args]
        kwargs = {k: self._probe(v, roots) for k, v in node.kwargs.items()}
        if n == "getattr":
            return getattr(args[0], args[1])
        if n in _IDENTITY or (n == "to" and not any(
                isinstance(x, torch.dtype) and not x.is_floating_point
                for x in list(args[1:]) + list(kwargs.values()))):
            # value-preserving at fp32 (the probe must stay float64: indices
            # past 2**24 are not exact in fp32)
            return args[0]
        if node.op == "call_method":
            return getattr(args[0], node.target)(*args[1:], **kwargs)
        return node.target(*args, **kwargs)

    def _lower_movement(self, node) -> None:
        if not _is_tensor_node(node):
            return  # tuple producer (split/chunk/unbind): its getitems lower themselves
        roots: dict = {}
        val = self._apply(node, roots)
        if not roots:
            raise NotImplementedError(f"{node.name}: no tensor input")
        self._emit_moved(node.name, roots, val, self._meta_shape(node), self._out(node),
                         self._is_graph_output(node))

    def apply_movement(self, base, srcs, fn, out_shape=None, out_name=None):
        """Lower `fn(*srcs)` (pure data movement on IR tensors) like a movement node."""
        roots: dict = {}
        val = fn(*[self._root(s, roots) for s in srcs])
        shape = list(val.shape) if out_shape is None else list(out_shape)
        out = out_name or self._new(base, shape)
        self.tensors.setdefault(out, {"shape": shape, "dtype": self._dt(), "quant": None})
        self._emit_moved(base, roots, val, shape, out, False)
        return out

    def _emit_moved(self, base, roots, val, out_shape, out, force_copy) -> None:
        idx = torch.round(val).to(torch.int64) - 1
        if list(idx.shape) != list(out_shape):
            raise NotImplementedError(f"{base}: probe shape {list(idx.shape)} != {out_shape}")
        if len(roots) > 1:
            # data movement over several tensors: gather from their concatenation
            parts = [(r, _prod(self._shape(r))) for r in roots]
            total = sum(c for _, c in parts)
            cat = self._catn(base, [(self.reshape(base, r, [c]) if len(self._shape(r)) != 1
                                     else r, c) for r, c in parts], 1, 1, [total], None)
            key = self._const(f"{base}__idx", idx.reshape(-1).numpy(), dtype=np.int32)
            self._emit(base, "gather", [cat, key], [out], {"n": int(idx.numel())})
            return
        (src,) = roots
        flat = idx.reshape(-1)
        affine = None if (flat < 0).any() else self._affine(idx)
        if affine is not None:
            off, strides = affine
            self.copy_affine(base, src, out_shape, off, strides, out_name=out,
                             force_copy=force_copy)
            return
        key = self._const(f"{base}__idx", flat.numpy(), dtype=np.int32)
        self._emit(base, "gather", [src, key], [out], {"n": int(flat.numel())})

    @staticmethod
    def _affine(idx: torch.Tensor):
        shape = list(idx.shape)
        if idx.numel() == 0:
            return None
        off = int(idx.reshape(-1)[0])
        strides = []
        for d, s in enumerate(shape):
            if s == 1:
                strides.append(0)
                continue
            e = [0] * len(shape)
            e[d] = 1
            strides.append(int(idx[tuple(e)]) - off)
        grid = torch.full(shape, off, dtype=torch.int64)
        for d, s in enumerate(shape):
            ar = torch.arange(s, dtype=torch.int64) * strides[d]
            grid = grid + ar.reshape([s if i == d else 1 for i in range(len(shape))])
        return (off, strides) if torch.equal(grid, idx) else None

    # -- compute ------------------------------------------------------------------
    def _lower_unary(self, node, n) -> None:
        x = self._tensor_arg(node.args[0])
        self.ewise1(node.name, x, _UNARY[n][0], out_name=self._out(node))

    def _lower_binary(self, node, n) -> None:
        fn, commutes = _BINARY[n]
        a = node.args[0]
        b = node.args[1] if len(node.args) > 1 else node.kwargs.get("other")
        alpha = node.kwargs.get("alpha", 1.0)
        out = self._out(node)
        an, bn = isinstance(a, torch.fx.Node), isinstance(b, torch.fx.Node)
        if an and bn:
            self.ewise2(node.name, self._tensor_arg(a), self._tensor_arg(b), fn, alpha, out)
        elif an:
            c = float(b) * (alpha if fn in ("add", "sub") else 1.0)
            if fn == "pow" and c == 2.0:
                self.ewise1(node.name, self._tensor_arg(a), "square", out_name=out)
            else:
                self.ewise_scalar(node.name, self._tensor_arg(a), fn, c, out)
        elif bn:
            sfn = fn if commutes else {"sub": "rsub", "div": "rdiv", "pow": "rpow"}.get(
                fn, _SWAPPED_CMP.get(fn))
            if sfn is None or alpha != 1.0:
                raise NotImplementedError(f"{node.name}: scalar {n} tensor")
            self.ewise_scalar(node.name, self._tensor_arg(b), sfn, float(a), out)
        else:
            raise NotImplementedError(f"{node.name}: {n} without tensor operands")

    def _dims_arg(self, node, pos=1, key="dim"):
        d = node.args[pos] if len(node.args) > pos else node.kwargs.get(key)
        if d is None:
            return None
        return [int(d)] if isinstance(d, int) else [int(x) for x in d]

    def _lower_reduce(self, node, n) -> None:
        x = node.args[0]
        if n in ("max", "min") and len(node.args) > 1 and isinstance(node.args[1], torch.fx.Node):
            return self._lower_binary(node, "maximum" if n == "max" else "minimum")
        if not _is_tensor_node(node):
            return  # max/min(dim) -> (values, indices); the getitem users lower
        dims = self._dims_arg(node)
        keepdim = bool(node.kwargs.get("keepdim", node.args[2] if len(node.args) > 2 and
                                       isinstance(node.args[2], bool) else False))
        src = self._tensor_arg(x)
        fn = _REDUCE[n]
        if fn == "norm":
            p = node.kwargs.get("p", node.args[1] if len(node.args) > 1 and
                                not isinstance(node.args[1], (list, tuple)) and n == "norm"
                                and "dim" in node.kwargs else 2)
            dims = node.kwargs.get("dim", None)
            dims = None if dims is None else ([dims] if isinstance(dims, int) else list(dims))
            if p in (2, "fro", 2.0):
                fn = "l2"
            elif p in (1, 1.0):
                src = self.ewise1(node.name, src, "abs")
                fn = "sum"
            else:
                raise NotImplementedError(f"{node.name}: norm p={p}")
        if fn == "var":
            unbiased = node.kwargs.get("unbiased", True)
            corr = node.kwargs.get("correction", 1 if unbiased else 0)
            fn = "var_unbiased" if corr == 1 else "var"
        if fn in ("argmax", "argmin") and dims is None:
            src = self.reshape(node.name, src, [_prod(self._shape(src))])
            dims = [0]
        self.reduce(node.name, src, dims, fn, keepdim, out_name=self._out(node))

    def _lower_minmax_item(self, node, item) -> None:
        red = node.args[0]
        src = self._tensor_arg(red.args[0])
        dims = self._dims_arg(red)
        keepdim = bool(red.kwargs.get("keepdim", red.args[2] if len(red.args) > 2 else False))
        fn = _name(red) if item == 0 else ("argmax" if _name(red) == "max" else "argmin")
        self.reduce(node.name, src, dims, fn, keepdim, out_name=self._out(node))


# -- function / method rules ------------------------------------------------------

def _arg(node, pos, key, default=None):
    if len(node.args) > pos:
        return node.args[pos]
    return node.kwargs.get(key, default)


def _f_softmax(g: GenericLowering, node, log=False):
    dim = _arg(node, 1, "dim")
    if dim is None:
        raise NotImplementedError(f"{node.name}: softmax without dim")
    g.softmax(node.name, g._tensor_arg(node.args[0]), int(dim), log, g._out(node))


def _f_cat(g: GenericLowering, node, stack=False):
    srcs = [g._tensor_arg(a) for a in node.args[0]]
    dim = int(_arg(node, 1, "dim", 0))
    g.cat(node.name, srcs, dim, g._out(node), stack=stack)


def _f_masked_fill(g: GenericLowering, node):
    x, mask, value = node.args[0], node.args[1], _arg(node, 2, "value")
    if isinstance(value, torch.fx.Node):
        raise NotImplementedError(f"{node.name}: tensor fill value")
    g.ewise2(node.name, g._tensor_arg(x), g._tensor_arg(mask), "masked_fill", float(value),
             g._out(node))


def _f_where(g: GenericLowering, node):
    cond, a, b = node.args[:3]
    out = g._out(node)
    if isinstance(a, torch.fx.Node) and not isinstance(b, torch.fx.Node):
        nc = g.ewise1(node.name, g._tensor_arg(cond), "logical_not")
        g.ewise2(node.name, g._tensor_arg(a), nc, "masked_fill", float(b), out)
    elif isinstance(b, torch.fx.Node) and not isinstance(a, torch.fx.Node):
        g.ewise2(node.name, g._tensor_arg(b), g._tensor_arg(cond), "masked_fill", float(a), out)
    else:
        # b + cond * (a - b) is wrong for inf/nan operands; select explicitly:
        # masked_fill can only write a scalar, so blend via two products.
        c = g._tensor_arg(cond)
        nc = g.ewise1(node.name, c, "logical_not")
        t1 = g.ewise2(node.name, g._tensor_arg(a), c, "mul")
        t2 = g.ewise2(node.name, g._tensor_arg(b), nc, "mul")
        g.ewise2(node.name, t1, t2, "add", 1.0, out)


def _f_clamp(g: GenericLowering, node):
    lo = _arg(node, 1, "min")
    hi = _arg(node, 2, "max")
    if isinstance(lo, torch.fx.Node) or isinstance(hi, torch.fx.Node):
        raise NotImplementedError(f"{node.name}: tensor clamp bounds")
    g.ewise1(node.name, g._tensor_arg(node.args[0]), "clamp",
             -math.inf if lo is None else lo, math.inf if hi is None else hi, g._out(node))


def _f_act(fn, p0_key=None, p0_pos=None, p0_default=0.0):
    def rule(g: GenericLowering, node):
        p0 = p0_default if p0_key is None else _arg(node, p0_pos, p0_key, p0_default)
        g.ewise1(node.name, g._tensor_arg(node.args[0]), fn, p0, out_name=g._out(node))
    return rule


def _f_gelu(g: GenericLowering, node):
    approx = node.kwargs.get("approximate", "none")
    g.ewise1(node.name, g._tensor_arg(node.args[0]), "gelu_tanh" if approx == "tanh" else "gelu",
             out_name=g._out(node))


def _f_hardtanh(g: GenericLowering, node):
    g.ewise1(node.name, g._tensor_arg(node.args[0]), "clamp", _arg(node, 1, "min_val", -1.0),
             _arg(node, 2, "max_val", 1.0), g._out(node))


def _f_relu6(g: GenericLowering, node):
    g.ewise1(node.name, g._tensor_arg(node.args[0]), "clamp", 0.0, 6.0, g._out(node))


def _f_normalize(g: GenericLowering, node):
    x = g._tensor_arg(node.args[0])
    p = float(_arg(node, 1, "p", 2.0))
    dim = int(_arg(node, 2, "dim", 1))
    eps = float(_arg(node, 3, "eps", 1e-12))
    if p != 2.0:
        raise NotImplementedError(f"{node.name}: normalize p={p}")
    nrm = g.reduce(node.name, x, [dim], "l2", keepdim=True)
    nrm = g.ewise_scalar(node.name, nrm, "max", eps)
    g.ewise2(node.name, x, nrm, "div", 1.0, g._out(node))


def _f_linear(g: GenericLowering, node):
    x, w = g._tensor_arg(_arg(node, 0, "input")), _arg(node, 1, "weight")
    b = _arg(node, 2, "bias")
    wn = g._tensor_arg(w)
    out = g._out(node)
    xs, ws = g._shape(x), g._shape(wn)
    if g.tensors[wn].get("const") and (b is None or g.tensors[b.name].get("const")):
        g._emit(node.name, "linear", [x], [out],
                {"M": _prod(xs[:-1]), "K": xs[-1], "N": ws[0]},
                weight=g.tensors[wn]["const"],
                bias=g.tensors[b.name]["const"] if b is not None else None)
        return
    wt = g.permute(node.name, wn, [1, 0])
    if b is None:
        g.matmul(node.name, x, wt, out)
    else:
        y = g.matmul(node.name, x, wt)
        g.ewise2(node.name, y, g._tensor_arg(b), "add", 1.0, out)


def _f_layer_norm(g: GenericLowering, node):
    x = g._tensor_arg(node.args[0])
    ns = _arg(node, 1, "normalized_shape")
    w, b = _arg(node, 2, "weight"), _arg(node, 3, "bias")
    eps = float(_arg(node, 4, "eps", 1e-5))
    K = _prod(ns)
    M = _prod(g._shape(x)) // K
    gk = g.tensors[w.name]["const"] if w is not None else \
        g.tensors[g._const(f"{node.name}__gamma", np.ones(K, np.float32))]["const"]
    bk = g.tensors[b.name]["const"] if b is not None else \
        g.tensors[g._const(f"{node.name}__beta", np.zeros(K, np.float32))]["const"]
    g.weights[gk] = g.weights[gk].reshape(-1)
    g.weights[bk] = g.weights[bk].reshape(-1)
    g._emit(node.name, "layer_norm", [x], [g._out(node)], {"M": M, "K": K},
            gamma_key=gk, beta_key=bk, eps=eps)


def _pool_shape(g, node, x, out):
    N_, C, IH, IW = g._shape(x)
    _, _, OH, OW = g._shape(out)
    return N_, C, IH, IW, OH, OW


def _pair(v):
    return (int(v), int(v)) if isinstance(v, int) else (int(v[0]), int(v[1]))


def _emit_maxpool(g: GenericLowering, node, x, k, s, p, d, ceil_mode):
    out = g._out(node)
    N_, C, IH, IW, OH, OW = _pool_shape(g, node, x, out)
    KH, KW = _pair(k)
    SH, SW = _pair(s if s is not None and s != [] else k)
    PH, PW = _pair(p)
    DH, DW = _pair(d)
    sh = {"N": N_, "C": C, "IH": IH, "IW": IW, "OH": OH, "OW": OW, "KH": KH, "KW": KW,
          "SH": SH, "SW": SW, "PH": PH, "PW": PW, "DH": DH, "DW": DW}
    floor_oh = (IH + 2 * PH - DH * (KH - 1) - 1) // SH + 1
    floor_ow = (IW + 2 * PW - DW * (KW - 1) - 1) // SW + 1
    op = "maxpool2d" if (OH, OW) == (floor_oh, floor_ow) else "maxpool2d_ex"
    g._emit(node.name, op, [x], [out], sh)


def _f_max_pool2d(g: GenericLowering, node):
    x = g._tensor_arg(node.args[0])
    _emit_maxpool(g, node, x, _arg(node, 1, "kernel_size"), _arg(node, 2, "stride"),
                  _arg(node, 3, "padding", 0), _arg(node, 4, "dilation", 1),
                  _arg(node, 5, "ceil_mode", False))


def _f_avg_pool2d(g: GenericLowering, node):
    x = g._tensor_arg(node.args[0])
    out = g._out(node)
    N_, C, IH, IW, OH, OW = _pool_shape(g, node, x, out)
    k = _arg(node, 1, "kernel_size")
    s = _arg(node, 2, "stride")
    KH, KW = _pair(k)
    SH, SW = _pair(s if s is not None and s != [] else k)
    PH, PW = _pair(_arg(node, 3, "padding", 0))
    if _arg(node, 4, "ceil_mode", False) or (not _arg(node, 5, "count_include_pad", True)
                                              and (PH or PW)):
        raise NotImplementedError(f"{node.name}: avg_pool2d ceil_mode/count_include_pad")
    g._emit(node.name, "avgpool2d", [x], [out],
            {"N": N_, "C": C, "IH": IH, "IW": IW, "OH": OH, "OW": OW, "KH": KH, "KW": KW,
             "SH": SH, "SW": SW, "PH": PH, "PW": PW})


def _f_adaptive_avg_pool2d(g: GenericLowering, node):
    x = g._tensor_arg(node.args[0])
    out = g._out(node)
    N_, C, IH, IW, OH, OW = _pool_shape(g, node, x, out)
    if (OH, OW) == (1, 1):
        g.reduce(node.name, x, [2, 3], "mean", keepdim=True, out_name=out)
        return
    raise NotImplementedError(f"{node.name}: adaptive_avg_pool2d to {OH}x{OW}")


def _f_einsum(g: GenericLowering, node):
    eq = node.args[0].replace(" ", "")
    operands = node.args[1:]
    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        operands = operands[0]
    srcs = [g._tensor_arg(o) for o in operands]
    lhs, rhs = eq.split("->")
    terms = lhs.split(",")
    if "." in eq or any(len(set(t)) != len(t) for t in terms):
        raise NotImplementedError(f"{node.name}: einsum ellipsis/diagonal '{eq}'")
    size = {}
    for t, s in zip(terms, srcs):
        for l, n in zip(t, g._shape(s)):
            size[l] = max(size.get(l, 1), n)
    cur, cl = srcs[0], terms[0]
    for k in range(1, len(srcs) + 1):
        later = set(rhs) | set("".join(terms[k + 1:]))
        if k < len(srcs):
            b, bl = srcs[k], terms[k]
            # sum out labels private to one side and unused later
            cur, cl = _einsum_sum_private(g, node, cur, cl, set(bl) | later)
            b, bl = _einsum_sum_private(g, node, b, bl, set(cl) | later)
            cur, cl = _einsum_pair(g, node, cur, cl, b, bl, later, size)
        else:
            cur, cl = _einsum_sum_private(g, node, cur, cl, set(rhs))
    perm = [cl.index(l) for l in rhs]
    out = g._out(node)
    g.permute(node.name, cur, perm, out_name=out)


def _einsum_sum_private(g, node, src, labels, keep):
    drop = [i for i, l in enumerate(labels) if l not in keep]
    if not drop:
        return src, labels
    return (g.reduce(node.name, src, drop, "sum"),
            "".join(l for i, l in enumerate(labels) if i not in drop))


def _einsum_pair(g, node, a, al, b, bl, later, size):
    batch = [l for l in al if l in bl and l in later]
    contract = [l for l in al if l in bl and l not in later]
    afree = [l for l in al if l not in bl]
    bfree = [l for l in bl if l not in al]
    sa, sb = g._shape(a), g._shape(b)
    # broadcast size-1 batch dims by expanding before the permute
    a = g.expand(node.name, a, [size[l] if l in batch else sa[i] for i, l in enumerate(al)])
    b = g.expand(node.name, b, [size[l] if l in batch else sb[i] for i, l in enumerate(bl)])
    a = g.permute(node.name, a, [al.index(l) for l in batch + afree + contract])
    b = g.permute(node.name, b, [bl.index(l) for l in batch + contract + bfree])
    Bt = _prod([size[l] for l in batch])
    M, K, N = (_prod([size[l] for l in afree]), _prod([size[l] for l in contract]),
               _prod([size[l] for l in bfree]))
    out_labels = "".join(batch + afree + bfree)
    out = g._new(node.name, [size[l] for l in out_labels])
    g._emit(node.name, "bmm", [a, b], [out], {"batch": Bt, "M": M, "K": K, "N": N})
    return out, out_labels


def _f_conv2d(g: GenericLowering, node):
    x = g._tensor_arg(node.args[0])
    w = g._tensor_arg(node.args[1])
    b = _arg(node, 2, "bias")
    if (not g.tensors[w].get("const") or (b is not None and not g.tensors[b.name].get("const"))
            or isinstance(_arg(node, 4, "padding", 0), str)):
        return _f_convnd(g, node)    # weight is a graph input / computed
    SH, SW = _pair(_arg(node, 3, "stride", 1))
    pad = _arg(node, 4, "padding", 0)
    if isinstance(pad, str):
        raise NotImplementedError(f"{node.name}: string padding")
    PH, PW = _pair(pad)
    DH, DW = _pair(_arg(node, 5, "dilation", 1))
    G = int(_arg(node, 6, "groups", 1))
    _emit_conv(g, node, x, g.tensors[w]["const"], g.tensors[b.name]["const"] if b is not None else None,
               g._shape(w), SH, SW, PH, PW, DH, DW, G)


def _emit_conv(g, node, x, wkey, bkey, wshape, SH, SW, PH, PW, DH, DW, G):
    out = g._out(node)
    N_, IC, IH, IW = g._shape(x)
    _, OC, OH, OW = g._shape(out)
    KH, KW = int(wshape[2]), int(wshape[3])
    g._emit(node.name, "conv2d_grouped", [x], [out],
            {"N": N_, "IC": IC, "IH": IH, "IW": IW, "OC": OC, "OH": OH, "OW": OW,
             "KH": KH, "KW": KW, "SH": SH, "SW": SW, "PH": PH, "PW": PW, "DH": DH, "DW": DW,
             "G": G}, weight=wkey, bias=bkey)


def _batchnorm_affine(g: GenericLowering, node, x, mean, var, w, b, eps):
    scale = w / torch.sqrt(var + eps)
    shift = b - mean * scale
    r = len(g._shape(x))
    bshape = [1, -1] + [1] * (r - 2)
    sk = g._const(f"{node.name}__scale", scale.reshape(bshape))
    hk = g._const(f"{node.name}__shift", shift.reshape(bshape))
    t = g.ewise2(node.name, x, sk, "mul")
    g.ewise2(node.name, t, hk, "add", 1.0, g._out(node))


def _f_batch_norm(g: GenericLowering, node):
    x = g._tensor_arg(node.args[0])
    vals = []
    for pos, key in ((1, "running_mean"), (2, "running_var"), (3, "weight"), (4, "bias")):
        a = _arg(node, pos, key)
        if a is None:
            vals.append(None)
            continue
        if not g.tensors.get(a.name, {}).get("const"):
            raise NotImplementedError(f"{node.name}: batch_norm with non-constant {key}")
        vals.append(torch.from_numpy(g.weights[g.tensors[a.name]["const"]].astype(np.float32)))
    mean, var, w, b = vals
    if _arg(node, 5, "training", False):
        raise NotImplementedError(f"{node.name}: batch_norm in training mode")
    w = torch.ones_like(mean) if w is None else w
    b = torch.zeros_like(mean) if b is None else b
    _batchnorm_affine(g, node, x, mean, var, w, b, float(_arg(node, 7, "eps", 1e-5)))


def _f_logsumexp(g: GenericLowering, node):
    """max + log(sum(exp(x - max))), the stable form torch uses."""
    x = g._tensor_arg(node.args[0])
    dims = g._dims_arg(node)
    keepdim = bool(_arg(node, 2, "keepdim", False))
    m = g.reduce(node.name, x, dims, "max", keepdim=True)
    e = g.ewise1(node.name, g.ewise2(node.name, x, m, "sub"), "exp")
    s = g.ewise1(node.name, g.reduce(node.name, e, dims, "sum", keepdim=True), "log")
    out = g._out(node)
    if keepdim:
        g.ewise2(node.name, s, m, "add", 1.0, out)
    else:
        g.reshape(node.name, g.ewise2(node.name, s, m, "add"), g._shape(out), out)


def _f_filled(value):
    def rule(g: GenericLowering, node):
        v = value if value is not None else float(_arg(node, 1, "fill_value"))
        g._const(node.name, torch.full(g._meta_shape(node), v))
    return rule


# -- functionalized in-place ops (see concrete_trace) ------------------------------

def _f_mb_setitem(g: GenericLowering, node):
    x, idx, v = node.args[:3]
    if _has_tensor_index(node):
        return _index_put(g, node, g._tensor_arg(x), idx, v, accumulate=False)
    # basic indexing selects an affine box of x: copy x and write the box
    # (a gather would need a constant index the size of x per write, which a
    # per-timestep `out[:, t] = h` loop turns quadratic)
    xn = g._tensor_arg(x)
    xs = g._shape(xn)
    enc = torch.arange(_prod(xs), dtype=torch.float64).reshape(xs)
    box = enc[idx]
    aff = g._affine(box.to(torch.int64)) if box.numel() else None
    if aff is None:
        return g._lower_movement(node)
    off, strides = aff
    bshape = list(box.shape)
    if isinstance(v, torch.fx.Node):
        vn = g._tensor_arg(v)
        vs = g._shape(vn)
        if _prod(vs) == _prod(bshape):           # same data, maybe other unit dims
            src = vn
        else:
            vs = [d for d in vs]
            while len(vs) > len(bshape) and vs[0] == 1:
                vs = vs[1:]
            src = g.expand(node.name, g.reshape(node.name, vn, vs), bshape)
    else:
        src = g._const(f"{node.name}__v", torch.full(bshape, float(v)))
    dims, (st,) = _merge_dims(bshape, strides)
    dims, (st,) = _pad6(dims, [st])
    sh = {"n": _prod(xs), "off": int(off)}
    sh.update({f"d{i}": dims[i] for i in range(MB_MAXR)})
    sh.update({f"s{i}": st[i] for i in range(MB_MAXR)})
    g._emit(node.name, "strided_scatter", [xn, src], [g._out(node)], sh)


def _f_mb_fill(g: GenericLowering, node):
    v = 0.0 if _name(node) == "mb_zero" else node.args[1]
    if isinstance(v, torch.fx.Node):   # fill_(0-dim tensor)
        src = g.expand(node.name, g._tensor_arg(v), g._meta_shape(node))
        g.reshape(node.name, src, g._meta_shape(node), g._out(node))
        return
    g._const(node.name, torch.full(g._meta_shape(node), float(v)))


# -- runtime indices -----------------------------------------------------------------

def _norm_items(idx, rank):
    """Index tuple with Ellipsis expanded and trailing full slices added."""
    items = list(idx) if isinstance(idx, (tuple, list)) else [idx]
    consumed = sum(1 for it in items if it is not None and it is not Ellipsis)
    if any(it is Ellipsis for it in items):
        k = next(i for i, it in enumerate(items) if it is Ellipsis)
        items[k:k + 1] = [slice(None)] * (rank - consumed)
    else:
        items += [slice(None)] * (rank - consumed)
    return items


def _wrap_index(g, base, idx, size):
    """idx + size where idx < 0 (PyTorch's negative-index wrap)."""
    neg = g.ewise_scalar(base, idx, "lt", 0.0)
    return g.ewise2(base, idx, g.ewise_scalar(base, neg, "mul", float(size)), "add")


def _linear_index(g, base, tensors, dims):
    """Broadcast index tensors over consecutive axes of size `dims` into one
    linear index over their flattened product."""
    if len(tensors) == 1:
        return tensors[0]
    lin = None
    for k, (t, _) in enumerate(zip(tensors, dims)):
        t = _wrap_index(g, base, t, dims[k])
        stride = _prod(dims[k + 1:])
        t = g.ewise_scalar(base, t, "mul", float(stride)) if stride != 1 else t
        lin = t if lin is None else g.ewise2(base, lin, t, "add")
    return lin


def _advanced(g, node, x, idx):
    """Split x[idx] into basic indexing (movement) and one block of
    consecutive tensor indices. Returns (T, p0, m, index tensors) where T =
    x[basic] and the tensors index T's axes p0 .. p0+m-1."""
    items = _norm_items(idx, len(g._shape(x)))
    if any(isinstance(it, torch.fx.Node) and _is_tensor_node(it)
           and str(it.meta["tensor_meta"].dtype) == "torch.bool" for it in items):
        raise NotImplementedError(f"{node.name}: boolean-mask index")
    basic, tpos, out_dim = [], [], 0
    for it in items:
        if isinstance(it, torch.fx.Node):
            basic.append(slice(None))
            tpos.append(out_dim)
            out_dim += 1
        else:
            basic.append(it)
            if it is None or isinstance(it, slice):
                out_dim += 1
    if tpos != list(range(tpos[0], tpos[0] + len(tpos))):
        raise NotImplementedError(f"{node.name}: non-adjacent advanced indices")
    if any(isinstance(it, int) for it in items) and len(tpos) > 1:
        raise NotImplementedError(f"{node.name}: int mixed with several tensor indices")
    if all(isinstance(it, slice) and it == slice(None) for it in basic):
        T = x
    else:
        T = g.apply_movement(node.name, [x], lambda t: t[tuple(basic)])
    tens = [g._tensor_arg(it) for it in items if isinstance(it, torch.fx.Node)]
    return T, tpos[0], len(tpos), tens


def _f_index(g: GenericLowering, node):
    """x[idx] with tensor indices (aten.index)."""
    x, idx = g._tensor_arg(node.args[0]), node.args[1]
    T, p0, m, tens = _advanced(g, node, x, idx)
    ts = g._shape(T)
    bshape = list(torch.broadcast_shapes(*[tuple(g._shape(t)) for t in tens]))
    tens = [g.expand(node.name, t, bshape) for t in tens]
    lin = _linear_index(g, node.name, tens, ts[p0:p0 + m])
    lin = g.reshape(node.name, lin, [_prod(bshape)])
    out_shape = ts[:p0] + bshape + ts[p0 + m:]
    if out_shape != g._meta_shape(node):
        raise NotImplementedError(f"{node.name}: advanced-index result {out_shape} "
                                  f"!= traced {g._meta_shape(node)}")
    g._emit(node.name, "index_select", [T, lin], [g._out(node)],
            {"outer": _prod(ts[:p0]), "A": _prod(ts[p0:p0 + m]), "J": _prod(bshape),
             "inner": _prod(ts[p0 + m:])})


def _scatter(g, base, x, idx3, src3, outer, A, J, inner, mode, out):
    g._emit(base, "scatter_dim", [x, idx3, src3], [out],
            {"outer": outer, "A": A, "J": J, "inner": inner, "mode": int(mode)})


def _index_put(g, node, x, idx, v, accumulate):
    """x[idx] = v / x.index_put_(idx, v, accumulate) with tensor indices."""
    xs = g._shape(x)
    items = _norm_items(idx, len(xs))
    out = g._out(node)
    masks = [it for it in items if isinstance(it, torch.fx.Node)
             and str(it.meta["tensor_meta"].dtype) == "torch.bool"]
    if masks:
        # x[mask] = scalar / one-element tensor: a select, not a scatter
        if len(masks) != 1 or len([it for it in items if isinstance(it, torch.fx.Node)]) != 1 \
                or accumulate:
            raise NotImplementedError(f"{node.name}: boolean-mask assignment form")
        mask = g._tensor_arg(masks[0])
        ms = g._shape(mask)
        lead = [i for i, it in enumerate(items) if it is masks[0]][0]
        if any(not (isinstance(it, slice) and it == slice(None)) for it in items if it is not masks[0]) \
                or lead != 0:
            raise NotImplementedError(f"{node.name}: boolean mask with other indices")
        m = g.reshape(node.name, mask, ms + [1] * (len(xs) - len(ms)))
        if not isinstance(v, torch.fx.Node):
            g.ewise2(node.name, x, m, "masked_fill", float(v), out)
            return
        vt = g._tensor_arg(v)
        if _prod(g._shape(vt)) != 1:
            raise NotImplementedError(f"{node.name}: x[mask] = tensor (data-dependent count)")
        vb = g.expand(node.name, g.reshape(node.name, vt, [1] * len(xs)), xs)
        t1 = g.ewise2(node.name, vb, m, "mul")
        t2 = g.ewise2(node.name, x, g.ewise1(node.name, m, "logical_not"), "mul")
        g.ewise2(node.name, t1, t2, "add", 1.0, out)
        return
    if any(not (isinstance(it, torch.fx.Node) or (isinstance(it, slice) and it == slice(None)))
           for it in items):
        raise NotImplementedError(f"{node.name}: index_put with slices/ints next to tensors")
    tpos = [i for i, it in enumerate(items) if isinstance(it, torch.fx.Node)]
    if tpos != list(range(tpos[0], tpos[0] + len(tpos))):
        raise NotImplementedError(f"{node.name}: non-adjacent advanced indices")
    p0, m = tpos[0], len(tpos)
    tens = [g._tensor_arg(items[i]) for i in tpos]
    bshape = list(torch.broadcast_shapes(*[tuple(g._shape(t)) for t in tens]))
    tens = [g.expand(node.name, t, bshape) for t in tens]
    lin = _linear_index(g, node.name, tens, xs[p0:p0 + m])
    outer, A, J, inner = _prod(xs[:p0]), _prod(xs[p0:p0 + m]), _prod(bshape), _prod(xs[p0 + m:])
    vshape = xs[:p0] + bshape + xs[p0 + m:]
    if isinstance(v, torch.fx.Node):
        src = g.expand(node.name, g._tensor_arg(v), vshape)
    else:
        src = g._const(f"{node.name}__v", torch.full(vshape, float(v)))
    idx3 = g.expand(node.name, g.reshape(node.name, lin, [1, J, 1]), [outer, J, inner])
    _scatter(g, node.name, x, idx3, src, outer, A, J, inner, accumulate, out)


def _f_index_put(g: GenericLowering, node):
    x = g._tensor_arg(node.args[0])
    indices = _arg(node, 1, "indices")
    v = _arg(node, 2, "values")
    acc = bool(_arg(node, 3, "accumulate", False))
    items = tuple(slice(None) if it is None else it for it in indices)
    _index_put(g, node, x, items, v, acc)


def _dim_split(shape, dim):
    d = dim % len(shape)
    return d, _prod(shape[:d]), shape[d], _prod(shape[d + 1:])


def _f_index_select(g: GenericLowering, node):
    x = g._tensor_arg(node.args[0])
    dim, idx = int(_arg(node, 1, "dim")), g._tensor_arg(_arg(node, 2, "index"))
    d, outer, A, inner = _dim_split(g._shape(x), dim)
    J = _prod(g._shape(idx))
    g._emit(node.name, "index_select", [x, g.reshape(node.name, idx, [J])], [g._out(node)],
            {"outer": outer, "A": A, "J": J, "inner": inner})


def _f_embedding(g: GenericLowering, node):
    idx, w = g._tensor_arg(node.args[0]), g._tensor_arg(node.args[1])
    if _arg(node, 2, "padding_idx") is not None and node.op == "call_function" and \
            _arg(node, 3, "max_norm") is not None:
        raise NotImplementedError(f"{node.name}: embedding max_norm")
    V, D = g._shape(w)
    J = _prod(g._shape(idx))
    g._emit(node.name, "index_select", [w, g.reshape(node.name, idx, [J])], [g._out(node)],
            {"outer": 1, "A": V, "J": J, "inner": D})


def _crop_to(g, base, x, shape, keep_dim):
    """x[:shape[0], ..., :, ...] — crop every axis but keep_dim to `shape`."""
    xs = g._shape(x)
    if all(xs[i] == shape[i] or i == keep_dim for i in range(len(xs))):
        return x
    sl = tuple(slice(None) if i == keep_dim else slice(0, shape[i]) for i in range(len(xs)))
    return g.apply_movement(base, [x], lambda t: t[sl])


def _f_gather(g: GenericLowering, node):
    x = g._tensor_arg(node.args[0])
    dim, idx = int(_arg(node, 1, "dim")), g._tensor_arg(_arg(node, 2, "index"))
    ish = g._shape(idx)
    d = dim % len(ish)
    x = _crop_to(g, node.name, x, ish, d)
    _, outer, J, inner = _dim_split(ish, d)
    g._emit(node.name, "gather_dim", [x, idx], [g._out(node)],
            {"outer": outer, "A": g._shape(x)[d], "J": J, "inner": inner})


def _f_scatter(g: GenericLowering, node):
    """scatter / scatter_add (src tensor or scalar value, reduce=None|'add')."""
    x = g._tensor_arg(node.args[0])
    dim, idx = int(_arg(node, 1, "dim")), g._tensor_arg(_arg(node, 2, "index"))
    src = _arg(node, 3, "src", node.kwargs.get("value"))
    reduce = node.kwargs.get("reduce")
    mode = 1 if (_name(node) == "scatter_add" or reduce in ("add", "sum")) else 0
    if reduce not in (None, "add", "sum"):
        raise NotImplementedError(f"{node.name}: scatter reduce={reduce}")
    xs, ish = g._shape(x), g._shape(idx)
    d = dim % len(xs)
    if any(xs[i] != ish[i] for i in range(len(xs)) if i != d):
        raise NotImplementedError(f"{node.name}: scatter index smaller than self off-axis")
    if isinstance(src, torch.fx.Node):
        src = _crop_to(g, node.name, g._tensor_arg(src), ish, -1)
    else:
        src = g._const(f"{node.name}__v", torch.full(ish, float(src)))
    _, outer, J, inner = _dim_split(ish, d)
    _scatter(g, node.name, x, idx, src, outer, xs[d], J, inner, mode, g._out(node))


def _f_index_add(g: GenericLowering, node, mode=1):
    """index_add (mode 1) / index_copy (mode 0) along `dim` with a 1-D index."""
    x = g._tensor_arg(node.args[0])
    dim, idx = int(_arg(node, 1, "dim")), g._tensor_arg(_arg(node, 2, "index"))
    src = g._tensor_arg(_arg(node, 3, "source"))
    alpha = node.kwargs.get("alpha", 1.0)
    if alpha != 1.0:
        src = g.ewise_scalar(node.name, src, "mul", float(alpha))
    d, outer, A, inner = _dim_split(g._shape(x), dim)
    J = _prod(g._shape(idx))
    idx3 = g.expand(node.name, g.reshape(node.name, idx, [1, J, 1]), [outer, J, inner])
    _scatter(g, node.name, x, idx3, src, outer, A, J, inner, mode, g._out(node))


# -- topk / sort ------------------------------------------------------------------------

def _topk(g, node, x, k, dim, largest, want=(0, 1)):
    xs = g._shape(x)
    d, outer, A, inner = _dim_split(xs, dim)
    oshape = xs[:d] + [k] + xs[d + 1:]
    vals, inds = g._new(node.name, oshape), g._new(node.name, oshape)
    g._emit(node.name, "topk", [x], [vals, inds],
            {"outer": outer, "A": A, "inner": inner, "K": int(k), "largest": int(bool(largest))})
    return vals, inds


def _f_topk(g: GenericLowering, node):
    x = g._tensor_arg(node.args[0])
    k = int(_arg(node, 1, "k"))
    dim = int(_arg(node, 2, "dim", -1))
    largest = bool(_arg(node, 3, "largest", True))
    vals, inds = _topk(g, node, x, k, dim, largest)
    g.tuple_items[(node.name, 0)] = vals
    g.tuple_items[(node.name, 1)] = inds


def _f_sort(g: GenericLowering, node):
    x = g._tensor_arg(node.args[0])
    dim = _arg(node, 1, "dim", -1)
    desc = node.kwargs.get("descending", node.args[2] if len(node.args) > 2 else False)
    if isinstance(dim, bool):          # sort(x, stable=...) is keyword-only; guard anyway
        dim = -1
    A = g._shape(x)[int(dim) % len(g._shape(x))]
    vals, inds = _topk(g, node, x, A, int(dim), bool(desc))
    if _name(node) == "argsort":
        g.reshape(node.name, inds, g._shape(inds), g._out(node))
        return
    g.tuple_items[(node.name, 0)] = vals
    g.tuple_items[(node.name, 1)] = inds


def _f_aminmax(g: GenericLowering, node):
    x = g._tensor_arg(node.args[0])
    dim = node.kwargs.get("dim")
    keep = bool(node.kwargs.get("keepdim", False))
    dims = None if dim is None else [int(dim)]
    g.tuple_items[(node.name, 0)] = g.reduce(node.name, x, dims, "min", keep)
    g.tuple_items[(node.name, 1)] = g.reduce(node.name, x, dims, "max", keep)


# -- convolution with tensor weights --------------------------------------------------

def _tup(v, n):
    return tuple(int(x) for x in v) if isinstance(v, (tuple, list)) else (int(v),) * n


def _f_convnd(g: GenericLowering, node):
    """F.conv1d / conv2d / conv3d: one convnd op, lower ranks as size-1 dims."""
    x = g._tensor_arg(node.args[0])
    w = g._tensor_arg(_arg(node, 1, "weight"))
    b = _arg(node, 2, "bias")
    nd = len(g._shape(w)) - 2
    if len(g._shape(x)) != nd + 2:
        raise NotImplementedError(f"{node.name}: unbatched conv")
    S = _tup(_arg(node, 3, "stride", 1), nd)
    Dl = _tup(_arg(node, 5, "dilation", 1), nd)
    G = int(_arg(node, 6, "groups", 1))
    K = [int(k) for k in g._shape(w)[2:]]
    pad = _arg(node, 4, "padding", 0)
    if isinstance(pad, str):
        if pad == "valid":
            P = (0,) * nd
        elif pad == "same":
            tot = [Dl[i] * (K[i] - 1) for i in range(nd)]
            if any(t % 2 for t in tot):
                raise NotImplementedError(f"{node.name}: asymmetric 'same' padding")
            P = tuple(t // 2 for t in tot)
        else:
            raise NotImplementedError(f"{node.name}: padding={pad}")
    else:
        P = _tup(pad, nd)
    xs, os_ = g._shape(x), g._meta_shape(node)
    lift = lambda v, fill: [fill] * (3 - nd) + list(v)
    sp_in, sp_out = lift(xs[2:], 1), lift(os_[2:], 1)
    Kk, Ss, Pp, Dd = lift(K, 1), lift(S, 1), lift(P, 0), lift(Dl, 1)
    ins = [x, w] + ([g._tensor_arg(b)] if b is not None else [])
    sh = {"N": xs[0], "IC": xs[1], "OC": os_[1], "G": G}
    for nm, vals in (("I", sp_in), ("O", sp_out), ("K", Kk), ("S", Ss), ("P", Pp), ("D", Dd)):
        for ax, v in zip("DHW", vals):
            sh[f"{nm}{ax}"] = int(v)
    g._emit(node.name, "convnd", ins, [g._out(node)], sh)


def _f_conv_transposend(g: GenericLowering, node):
    """F.conv_transpose1d/2d/3d as a stride-1 convolution: zero-insert the
    input (stride), pad by D*(K-1)-P (+ output_padding on the far side), and
    convolve with the spatially flipped weight with in/out channels swapped
    per group. All data movement, then one convnd."""
    x = g._tensor_arg(node.args[0])
    w = g._tensor_arg(_arg(node, 1, "weight"))
    b = _arg(node, 2, "bias")
    ws = g._shape(w)
    nd = len(ws) - 2
    S = _tup(_arg(node, 3, "stride", 1), nd)
    P = _tup(_arg(node, 4, "padding", 0), nd)
    G = int(_arg(node, 6, "groups", 1))
    Dl = _tup(_arg(node, 7, "dilation", 1), nd)
    xs, os_ = g._shape(x), g._meta_shape(node)
    K = ws[2:]
    IC, OCg = ws[0], ws[1]
    for i in range(nd):
        if Dl[i] * (K[i] - 1) - P[i] < 0:
            raise NotImplementedError(f"{node.name}: conv_transpose padding > dilated kernel")

    def dilate_pad(t):
        # [N, C, *sp] -> zero-inserted and padded; index 0 of the probe is "zero"
        N_, C = t.shape[:2]
        sp = list(t.shape[2:])
        full = [(sp[i] - 1) * S[i] + 1 for i in range(nd)]
        lo = [Dl[i] * (K[i] - 1) - P[i] for i in range(nd)]
        tot = [os_[2 + i] + Dl[i] * (K[i] - 1) for i in range(nd)]
        y = torch.zeros([N_, C] + tot, dtype=t.dtype)
        sl = tuple([slice(None), slice(None)] +
                   [slice(lo[i], lo[i] + full[i], S[i]) for i in range(nd)])
        y[sl] = t
        return y
    xp = g.apply_movement(node.name, [x], dilate_pad)

    def flip_swap(t):
        # [IC, OC/G, *K] -> [OC, IC/G, *K], flipped spatially
        t = t.flip(list(range(2, 2 + nd)))
        t = t.reshape([G, IC // G, OCg] + K).transpose(1, 2)
        return t.reshape([G * OCg, IC // G] + K)
    wf = g.apply_movement(node.name, [w], flip_swap)
    lift = lambda v, fill: [fill] * (3 - nd) + list(v)
    sp_in = lift(g._shape(xp)[2:], 1)
    sp_out = lift(os_[2:], 1)
    ins = [xp, wf] + ([g._tensor_arg(b)] if b is not None else [])
    sh = {"N": xs[0], "IC": IC, "OC": os_[1], "G": G}
    for nm, vals in (("I", sp_in), ("O", sp_out), ("K", lift(K, 1)), ("S", [1, 1, 1]),
                     ("P", [0, 0, 0]), ("D", lift(Dl, 1))):
        for ax, v in zip("DHW", vals):
            sh[f"{nm}{ax}"] = int(v)
    g._emit(node.name, "convnd", ins, [g._out(node)], sh)


def _f_sdpa(g: GenericLowering, node):
    """scaled_dot_product_attention at any rank, as matmul + mask + softmax +
    matmul: softmax(q @ k^T * scale + mask) @ v. A bool mask (True = attend)
    and is_causal fill with -inf like torch."""
    q, k, v = (g._tensor_arg(_arg(node, i, nm)) for i, nm in enumerate(("query", "key", "value")))
    mask = _arg(node, 3, "attn_mask")
    dropout, causal = _arg(node, 4, "dropout_p") or 0.0, _arg(node, 5, "is_causal") or False
    scale, gqa = _arg(node, 6, "scale"), node.kwargs.get("enable_gqa", False)
    if dropout or gqa or isinstance(scale, torch.fx.Node):
        raise NotImplementedError(f"{node.name}: sdpa dropout / gqa / tensor scale")
    sq, sk = g._shape(q), g._shape(k)
    L, S, D = sq[-2], sk[-2], sq[-1]
    kt = g.permute(node.name, k, list(range(len(sk) - 2)) + [len(sk) - 1, len(sk) - 2])
    att = g.matmul(node.name, q, kt)
    att = g.ewise_scalar(node.name, att, "mul", float(scale if scale is not None else D ** -0.5))
    if causal:
        tri = g._const(f"{node.name}__causal", torch.ones(L, S).tril())
        att = g.ewise2(node.name, att, g.ewise1(node.name, tri, "logical_not"), "masked_fill",
                       float("-inf"))
    if isinstance(mask, torch.fx.Node):
        m = g._tensor_arg(mask)
        if mask.meta["tensor_meta"].dtype == torch.bool:
            att = g.ewise2(node.name, att, g.ewise1(node.name, m, "logical_not"), "masked_fill",
                           float("-inf"))
        else:
            att = g.ewise2(node.name, att, m, "add")
    elif mask is not None:
        raise NotImplementedError(f"{node.name}: non-tensor attn_mask")
    p = g.softmax(node.name, att, -1)
    g.matmul(node.name, p, v, g._out(node))


def _f_scan(g: GenericLowering, node):
    """cumsum / cumprod along one dim, viewed as [outer, axis, inner]"""
    x = g._tensor_arg(node.args[0])
    s = g._shape(x)
    d = node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else None)
    if d is None or node.kwargs.get("dtype") not in (None, torch.float32):
        raise NotImplementedError(f"{node.name}: {_name(node)} without dim / with a dtype")
    d = int(d) % len(s)
    g._emit(node.name, _name(node), [x], [g._out(node, list(s))],
            {"outer": _prod(s[:d]), "axis": s[d], "inner": _prod(s[d + 1:])})


def _f_one_hot(g: GenericLowering, node):
    idx = g._tensor_arg(node.args[0])
    C = g._meta_shape(node)[-1]
    s = g._shape(idx)
    col = g.reshape(node.name, idx, s + [1])
    ar = g._const(f"{node.name}__ar", torch.arange(C, dtype=torch.float32))
    g.ewise2(node.name, col, ar, "eq", 1.0, g._out(node))


_INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


def _f_cast(g: GenericLowering, node):
    """dtype casts. Everything is f32 in the IR, so float casts are views;
    to bool is `!= 0`; to an integer type truncates toward zero (unless the
    source already holds integers)."""
    n = _name(node)
    src = node.args[0]
    target = {"bool": torch.bool, "long": torch.int64, "int": torch.int32}.get(n)
    if target is None:   # .to(...) / .type(...)
        for a in list(node.args[1:]) + list(node.kwargs.values()):
            if isinstance(a, torch.dtype):
                target = a
    sdt = src.meta.get("tensor_meta").dtype if isinstance(src, torch.fx.Node) and \
        hasattr(src.meta.get("tensor_meta"), "dtype") else None
    x = g._tensor_arg(src)
    out = g._out(node)
    if target == torch.bool and sdt != torch.bool:
        g.ewise_scalar(node.name, x, "ne", 0.0, out)
    elif target in _INT_DTYPES and sdt is not None and sdt.is_floating_point:
        # trunc(x) = sign(x) * floor(|x|)
        fl = g.ewise1(node.name, g.ewise1(node.name, x, "abs"), "floor")
        g.ewise2(node.name, fl, g.ewise1(node.name, x, "sign"), "mul", 1.0, out)
    else:
        g.reshape(node.name, x, g._shape(x), out)


# -- norms ---------------------------------------------------------------------------------

def _affine_tail(g, node, y, w, b, bshape, out):
    """y * w + b with w / b (tensor nodes or None) broadcast as `bshape`."""
    if w is not None:
        wt = g.reshape(node.name, g._tensor_arg(w), bshape)
        y = g.ewise2(node.name, y, wt, "mul", 1.0, None if b is not None else out)
    if b is not None:
        bt = g.reshape(node.name, g._tensor_arg(b), bshape)
        y = g.ewise2(node.name, y, bt, "add", 1.0, out)
    if w is None and b is None:
        g.reshape(node.name, y, g._shape(y), out)


def _add_eps(g, base, t, eps):
    """t + eps for a Python number or a (0-d / 1-element) tensor eps."""
    if isinstance(eps, torch.fx.Node):
        return g.ewise2(base, t, g.reshape(base, g._tensor_arg(eps), [1]), "add")
    return g.ewise_scalar(base, t, "add", float(eps))


def _eps_arg(node, pos, default):
    e = _arg(node, pos, "eps", default)
    return e if isinstance(e, torch.fx.Node) else float(default if e is None else e)


def _normalize_rows(g, node, x3, eps):
    """(x - mean) / sqrt(var + eps) over the last axis of [M, R]."""
    m = g.reduce(node.name, x3, [1], "mean", keepdim=True)
    v = g.reduce(node.name, x3, [1], "var", keepdim=True)
    d = g.ewise2(node.name, x3, m, "sub")
    r = g.ewise1(node.name, _add_eps(g, node.name, v, eps), "rsqrt")
    return g.ewise2(node.name, d, r, "mul")


def _f_group_norm(g: GenericLowering, node):
    x = g._tensor_arg(node.args[0])
    G = int(_arg(node, 1, "num_groups"))
    w, b = _arg(node, 2, "weight"), _arg(node, 3, "bias")
    eps = _eps_arg(node, 4, 1e-5)
    xs = g._shape(x)
    N_, C = xs[0], xs[1]
    y = _normalize_rows(g, node, g.reshape(node.name, x, [N_ * G, _prod(xs) // (N_ * G)]), eps)
    y = g.reshape(node.name, y, xs)
    _affine_tail(g, node, y, w, b, [1, C] + [1] * (len(xs) - 2), g._out(node))


def _f_layer_norm_any(g: GenericLowering, node):
    w, b = _arg(node, 2, "weight"), _arg(node, 3, "bias")
    eps = _eps_arg(node, 4, 1e-5)
    if all(a is None or g.tensors.get(a.name, {}).get("const") for a in (w, b)) and \
            not isinstance(eps, torch.fx.Node):
        return _f_layer_norm(g, node)
    x = g._tensor_arg(node.args[0])
    ns = _arg(node, 1, "normalized_shape")
    ns = [int(ns)] if isinstance(ns, int) else [int(v) for v in ns]
    xs = g._shape(x)
    K = _prod(ns)
    y = g.reshape(node.name, _normalize_rows(g, node, g.reshape(node.name, x, [_prod(xs) // K, K]), eps), xs)
    _affine_tail(g, node, y, w, b, ns, g._out(node))


def _f_rms_norm(g: GenericLowering, node):
    x = g._tensor_arg(node.args[0])
    ns = _arg(node, 1, "normalized_shape")
    ns = [int(ns)] if isinstance(ns, int) else [int(v) for v in ns]
    w = _arg(node, 2, "weight")
    eps = _eps_arg(node, 3, torch.finfo(torch.float32).eps)
    xs = g._shape(x)
    K = _prod(ns)
    x2 = g.reshape(node.name, x, [_prod(xs) // K, K])
    ms = g.reduce(node.name, g.ewise1(node.name, x2, "square"), [1], "mean", keepdim=True)
    r = g.ewise1(node.name, _add_eps(g, node.name, ms, eps), "rsqrt")
    y = g.reshape(node.name, g.ewise2(node.name, x2, r, "mul"), xs)
    _affine_tail(g, node, y, w, None, ns, g._out(node))


def _f_vector_norm(g: GenericLowering, node):
    x = g._tensor_arg(node.args[0])
    ord_ = _arg(node, 1, "ord", 2)
    dims = _arg(node, 2, "dim")
    dims = None if dims is None else ([int(dims)] if isinstance(dims, int) else [int(d) for d in dims])
    keep = bool(_arg(node, 3, "keepdim", False))
    out = g._out(node)
    if ord_ in (2, 2.0):
        g.reduce(node.name, x, dims, "l2", keep, out)
    elif ord_ in (1, 1.0):
        g.reduce(node.name, g.ewise1(node.name, x, "abs"), dims, "sum", keep, out)
    elif ord_ in (math.inf, float("inf")):
        g.reduce(node.name, g.ewise1(node.name, x, "abs"), dims, "max", keep, out)
    else:
        raise NotImplementedError(f"{node.name}: vector_norm ord={ord_}")


# -- interpolate ---------------------------------------------------------------------------

def _f_interpolate(g: GenericLowering, node):
    """nearest modes are data movement; (bi/tri)linear and bicubic are separable,
    so each spatial axis is a matmul with a constant interpolation matrix."""
    x = g._tensor_arg(node.args[0])
    mode = node.kwargs.get("mode", _arg(node, 3, "mode", "nearest"))
    out_shape = g._meta_shape(node)
    xs = g._shape(x)
    kw = {k: v for k, v in node.kwargs.items() if k != "input"}
    if mode in ("nearest", "nearest-exact"):
        g.apply_movement(node.name, [x], lambda t: _F.interpolate(t, *node.args[1:], **kw),
                         out_shape, g._out(node))
        return
    if mode not in ("linear", "bilinear", "trilinear", "bicubic") or kw.get("antialias"):
        raise NotImplementedError(f"{node.name}: interpolate mode={mode}")
    ac = kw.get("align_corners", _arg(node, 4, "align_corners", None))
    nsp = len(xs) - 2
    sf = kw.get("scale_factor", _arg(node, 2, "scale_factor", None))
    sfs = None if sf is None else (list(sf) if isinstance(sf, (tuple, list)) else [sf] * nsp)
    cur = x
    for ax in range(nsp):
        L, Lo = xs[2 + ax], out_shape[2 + ax]
        if L == Lo and sfs is None:
            continue
        one = "linear" if mode != "bicubic" else "bicubic"
        eye = torch.eye(L, dtype=torch.float64)
        if one == "linear":
            W = _F.interpolate(eye.reshape(L, 1, L), size=None if sfs else Lo,
                               scale_factor=None if sfs is None else sfs[ax], mode="linear",
                               align_corners=ac, recompute_scale_factor=kw.get("recompute_scale_factor"))
            W = W.reshape(L, Lo)
        else:
            W = _F.interpolate(eye.reshape(L, 1, 1, L).expand(L, 1, 4, L).contiguous(),
                               size=None if sfs else (4, Lo),
                               scale_factor=None if sfs is None else (1.0, sfs[ax]),
                               mode="bicubic", align_corners=ac)[:, 0, 0, :].reshape(L, Lo)
        # move the axis last, contract with W, move it back
        cs = g._shape(cur)
        r = len(cs)
        perm = [i for i in range(r) if i != 2 + ax] + [2 + ax]
        t = g.permute(node.name, cur, perm)
        wk = g._const(f"{node.name}__w{ax}", W.float())
        t = g.matmul(node.name, t, wk)
        inv = [perm.index(i) for i in range(r)]
        cur = g.permute(node.name, t, inv)
    g.reshape(node.name, cur, out_shape, g._out(node))


# -- bool / integer arithmetic ---------------------------------------------------------------

def _is_bool(node) -> bool:
    tm = node.meta.get("tensor_meta") if isinstance(node, torch.fx.Node) else None
    return tm is not None and hasattr(tm, "dtype") and tm.dtype == torch.bool


def _f_not(g: GenericLowering, node):
    if not _is_bool(node.args[0]):
        raise NotImplementedError(f"{node.name}: bitwise not of a non-bool tensor")
    g.ewise1(node.name, g._tensor_arg(node.args[0]), "logical_not", out_name=g._out(node))


def _f_logic(fn):
    """and / or / xor on bool tensors (0/1 floats): mul / max / ne."""
    def rule(g: GenericLowering, node):
        a, b = node.args[:2]
        if not all(_is_bool(t) or not isinstance(t, torch.fx.Node) for t in (a, b)) \
                and _name(node).startswith("bitwise"):
            raise NotImplementedError(f"{node.name}: bitwise op on integers")
        out = g._out(node)
        an, bn = isinstance(a, torch.fx.Node), isinstance(b, torch.fx.Node)
        if an and bn:
            ta, tb = g._tensor_arg(a), g._tensor_arg(b)
            if not (_is_bool(a) and _is_bool(b)):   # logical_* on numbers
                ta = g.ewise_scalar(node.name, ta, "ne", 0.0)
                tb = g.ewise_scalar(node.name, tb, "ne", 0.0)
            g.ewise2(node.name, ta, tb, fn, 1.0, out)
        else:
            t, c = (a, b) if an else (b, a)
            c = float(bool(c))
            sfn = {"mul": "mul", "max": "max", "ne": "ne"}[fn]
            g.ewise_scalar(node.name, g._tensor_arg(t), sfn, c, out)
    return rule


def _f_any_all(fn):
    def rule(g: GenericLowering, node):
        x = g._tensor_arg(node.args[0])
        if not _is_bool(node.args[0]):
            x = g.ewise_scalar(node.name, x, "ne", 0.0)
        g.reduce(node.name, x, g._dims_arg(node), fn,
                 bool(_arg(node, 2, "keepdim", False)), g._out(node))
    return rule


def _f_floor_divide(g: GenericLowering, node):
    a, b = node.args[:2]
    out = g._out(node)
    if isinstance(b, torch.fx.Node):
        q = g.ewise2(node.name, g._tensor_arg(a), g._tensor_arg(b), "div") \
            if isinstance(a, torch.fx.Node) else \
            g.ewise_scalar(node.name, g._tensor_arg(b), "rdiv", float(a))
    else:
        q = g.ewise_scalar(node.name, g._tensor_arg(a), "div", float(b))
    g.ewise1(node.name, q, "floor", out_name=out)


def _f_remainder(g: GenericLowering, node):
    """a - floor(a / b) * b (Python / torch.remainder sign convention)."""
    a, b = node.args[:2]
    if not isinstance(a, torch.fx.Node):
        raise NotImplementedError(f"{node.name}: scalar % tensor")
    ta = g._tensor_arg(a)
    if isinstance(b, torch.fx.Node):
        tb = g._tensor_arg(b)
        q = g.ewise1(node.name, g.ewise2(node.name, ta, tb, "div"), "floor")
        g.ewise2(node.name, ta, g.ewise2(node.name, q, tb, "mul"), "sub", 1.0, g._out(node))
    else:
        q = g.ewise1(node.name, g.ewise_scalar(node.name, ta, "div", float(b)), "floor")
        g.ewise2(node.name, ta, g.ewise_scalar(node.name, q, "mul", float(b)), "sub", 1.0,
                 g._out(node))


_FUNCS = {
    "zeros_like": _f_filled(0.0), "ones_like": _f_filled(1.0), "full_like": _f_filled(None),
    "new_zeros": _f_filled(0.0), "new_ones": _f_filled(1.0),
    "softmax": _f_softmax,
    "log_softmax": lambda g, n: _f_softmax(g, n, log=True),
    "cat": _f_cat, "concat": _f_cat, "concatenate": _f_cat,
    "stack": lambda g, n: _f_cat(g, n, stack=True),
    "masked_fill": _f_masked_fill, "where": _f_where,
    "clamp": _f_clamp, "clip": _f_clamp,
    "gelu": _f_gelu, "hardtanh": _f_hardtanh, "relu6": _f_relu6,
    "leaky_relu": _f_act("leaky_relu", "negative_slope", 1, 0.01),
    "elu": _f_act("elu", "alpha", 1, 1.0),
    "normalize": _f_normalize, "linear": _f_linear, "layer_norm": _f_layer_norm_any,
    "max_pool2d": _f_max_pool2d, "avg_pool2d": _f_avg_pool2d,
    "adaptive_avg_pool2d": _f_adaptive_avg_pool2d,
    "einsum": _f_einsum, "conv2d": _f_conv2d, "batch_norm": _f_batch_norm,
    "logsumexp": _f_logsumexp,
    # functionalized in-place ops (concrete_trace)
    "mb_setitem": _f_mb_setitem, "mb_copy": lambda g, n: g._lower_movement(n),
    "mb_fill": _f_mb_fill, "mb_zero": _f_mb_fill,
    # runtime indices
    "index_select": _f_index_select, "embedding": _f_embedding, "gather": _f_gather,
    "scatter": _f_scatter, "scatter_add": _f_scatter, "index_add": _f_index_add,
    "index_copy": lambda g, n: _f_index_add(g, n, mode=0), "index_put": _f_index_put,
    "topk": _f_topk, "sort": _f_sort, "argsort": _f_sort, "aminmax": _f_aminmax,
    "conv1d": _f_convnd, "conv3d": _f_convnd,
    "conv_transpose1d": _f_conv_transposend, "conv_transpose2d": _f_conv_transposend,
    "conv_transpose3d": _f_conv_transposend, "one_hot": _f_one_hot,
    "bool": _f_cast, "long": _f_cast, "int": _f_cast, "to": _f_cast, "type": _f_cast,
    "group_norm": _f_group_norm, "rms_norm": _f_rms_norm,
    "linalg_vector_norm": _f_vector_norm, "vector_norm": _f_vector_norm,
    "interpolate": _f_interpolate, "upsample": _f_interpolate,
    "invert": _f_not, "bitwise_not": _f_not,
    "logical_and": _f_logic("mul"), "bitwise_and": _f_logic("mul"), "and": _f_logic("mul"),
    "logical_or": _f_logic("max"), "bitwise_or": _f_logic("max"), "or": _f_logic("max"),
    "logical_xor": _f_logic("ne"), "bitwise_xor": _f_logic("ne"), "xor": _f_logic("ne"),
    "any": _f_any_all("max"), "all": _f_any_all("min"),
    "floor_divide": _f_floor_divide, "floordiv": _f_floor_divide,
    "remainder": _f_remainder, "mod": _f_remainder,
    "cumsum": _f_scan, "cumprod": _f_scan,
    "scaled_dot_product_attention": _f_sdpa,
}
_METHODS_EXTRA = {
    "masked_fill_": _f_masked_fill,
}


# -- modules ---------------------------------------------------------------------

def _m_softmax(g, node, mod, log=False):
    if mod.dim is None:
        raise NotImplementedError(f"{node.name}: Softmax(dim=None)")
    g.softmax(node.name, g._tensor_arg(node.args[0]), int(mod.dim), log, g._out(node))


def _m_movement(g, node, mod):
    g._lower_movement(node)


def _m_conv2d(g, node, mod):
    x = g._tensor_arg(node.args[0])
    wkey = f"{node.target}.weight"
    bkey = f"{node.target}.bias" if mod.bias is not None else None
    g.weights[wkey] = mod.weight.detach().cpu().numpy().astype(g.wdtype)
    if bkey:
        g.weights[bkey] = mod.bias.detach().cpu().numpy().astype(g.wdtype)
    if isinstance(mod.padding, str):
        raise NotImplementedError(f"{node.name}: string padding")
    _emit_conv(g, node, x, wkey, bkey, list(mod.weight.shape), *_pair(mod.stride),
               *_pair(mod.padding), *_pair(mod.dilation), int(mod.groups))


def _m_maxpool2d(g, node, mod):
    _emit_maxpool(g, node, g._tensor_arg(node.args[0]), mod.kernel_size, mod.stride,
                  mod.padding, mod.dilation, mod.ceil_mode)


def _m_batchnorm(g, node, mod):  # eval-mode BatchNorm1d/2d/3d (any rank)
    x = g._tensor_arg(node.args[0])
    w = mod.weight.detach() if mod.weight is not None else torch.ones(mod.num_features)
    b = mod.bias.detach() if mod.bias is not None else torch.zeros(mod.num_features)
    _batchnorm_affine(g, node, x, mod.running_mean, mod.running_var, w, b, float(mod.eps))


def _m_instance_norm(g, node, mod):
    """InstanceNorm1d/2d/3d: per-(N, C) stats over the spatial dims."""
    if mod.track_running_stats:
        raise NotImplementedError(f"{node.name}: InstanceNorm with running stats")
    x = g._tensor_arg(node.args[0])
    r = len(g._shape(x))
    sp = list(range(2, r))
    xc = g.ewise2(node.name, x, g.reduce(node.name, x, sp, "mean", keepdim=True), "sub")
    var = g.reduce(node.name, g.ewise1(node.name, xc, "square"), sp, "mean", keepdim=True)
    rstd = g.ewise1(node.name, g.ewise_scalar(node.name, var, "add", float(mod.eps)), "rsqrt")
    if not mod.affine:
        g.ewise2(node.name, xc, rstd, "mul", 1.0, g._out(node))
        return
    bshape = [1, -1] + [1] * (r - 2)
    wk = g._const(f"{node.name}__w", mod.weight.detach().reshape(bshape))
    bk = g._const(f"{node.name}__b", mod.bias.detach().reshape(bshape))
    t = g.ewise2(node.name, g.ewise2(node.name, xc, rstd, "mul"), wk, "mul")
    g.ewise2(node.name, t, bk, "add", 1.0, g._out(node))


def _m_act(fn, p0=None, p1=None):
    def rule(g, node, mod):
        a0 = getattr(mod, p0) if isinstance(p0, str) else (p0 or 0.0)
        a1 = getattr(mod, p1) if isinstance(p1, str) else (p1 or 0.0)
        g.ewise1(node.name, g._tensor_arg(node.args[0]), fn, a0, a1, g._out(node))
    return rule


def _m_mha(g: GenericLowering, node, mod):
    """nn.MultiheadAttention (eval, no masks) as linear + bmm + softmax."""
    q_in, k_in, v_in = (node.args + (None, None, None))[:3]
    k_in = k_in if k_in is not None else node.kwargs.get("key")
    v_in = v_in if v_in is not None else node.kwargs.get("value")
    for kw in ("key_padding_mask", "attn_mask"):
        if node.kwargs.get(kw) is not None:
            raise NotImplementedError(f"{node.name}: MultiheadAttention {kw}")
    if not mod._qkv_same_embed_dim or mod.bias_k is not None or mod.add_zero_attn:
        raise NotImplementedError(f"{node.name}: MultiheadAttention variant")
    E, H = mod.embed_dim, mod.num_heads
    hd = E // H
    W = mod.in_proj_weight.detach()
    b = mod.in_proj_bias.detach() if mod.in_proj_bias is not None else torch.zeros(3 * E)
    names = [g._tensor_arg(t) for t in (q_in, k_in, v_in)]
    shape = g._shape(names[0])
    if mod.batch_first:
        N_, L = shape[0], shape[1]
    else:
        L, N_ = shape[0], shape[1]
    S = g._shape(names[1])[1 if mod.batch_first else 0]
    heads = []
    for i, (src, seq) in enumerate(zip(names, (L, S, S))):
        wk, bk = f"{node.target}.in_proj_weight.{i}", f"{node.target}.in_proj_bias.{i}"
        g.weights[wk] = W[i * E:(i + 1) * E].cpu().numpy().astype(g.wdtype)
        g.weights[bk] = b[i * E:(i + 1) * E].cpu().numpy().astype(g.wdtype)
        p = g._new(node.name, g._shape(src)[:-1] + [E])
        g._emit(node.name, "linear", [src], [p], {"M": _prod(g._shape(src)[:-1]), "K": E, "N": E},
                weight=wk, bias=bk)
        # -> [N, H, seq, hd]
        if mod.batch_first:
            p4 = g.reshape(node.name, p, [N_, seq, H, hd])
            heads.append(g.permute(node.name, p4, [0, 2, 1, 3]))
        else:
            p4 = g.reshape(node.name, p, [seq, N_, H, hd])
            heads.append(g.permute(node.name, p4, [1, 2, 0, 3]))
    q, k, v = heads
    kt = g.permute(node.name, k, [0, 1, 3, 2])
    sc = g.matmul(node.name, q, kt)
    sc = g.ewise_scalar(node.name, sc, "mul", 1.0 / math.sqrt(hd))
    att = g.softmax(node.name, sc, -1)
    ctx = g.matmul(node.name, att, v)                          # [N, H, L, hd]
    ctx = g.permute(node.name, ctx, [0, 2, 1, 3] if mod.batch_first else [2, 0, 1, 3])
    ctx = g.reshape(node.name, ctx, ([N_, L] if mod.batch_first else [L, N_]) + [E])
    wk, bk = f"{node.target}.out_proj.weight", f"{node.target}.out_proj.bias"
    g.weights[wk] = mod.out_proj.weight.detach().cpu().numpy().astype(g.wdtype)
    g.weights[bk] = (mod.out_proj.bias.detach() if mod.out_proj.bias is not None
                     else torch.zeros(E)).cpu().numpy().astype(g.wdtype)
    out = g._new(node.name, g._shape(ctx))
    g._emit(node.name, "linear", [ctx], [out], {"M": _prod(g._shape(ctx)[:-1]), "K": E, "N": E},
            weight=wk, bias=bk)
    g.tuple_items[(node.name, 0)] = out
    if any(u.op == "call_function" and u.target is operator.getitem and u.args[1] == 1
           for u in node.users):
        w = att if not node.kwargs.get("average_attn_weights", True) else \
            g.reduce(node.name, att, [1], "mean")
        g.tuple_items[(node.name, 1)] = w


def _m_conv1d(g: GenericLowering, node, mod):
    x = g._tensor_arg(node.args[0])
    N_, IC, IW = g._shape(x)
    out = g._out(node)
    _, OC, OW = g._shape(out)
    if isinstance(mod.padding, str):
        raise NotImplementedError(f"{node.name}: string padding")
    wkey = f"{node.target}.weight"
    bkey = f"{node.target}.bias" if mod.bias is not None else None
    w = mod.weight.detach().cpu().numpy().astype(g.wdtype)
    g.weights[wkey] = w.reshape(w.shape[0], w.shape[1], 1, w.shape[2])
    if bkey:
        g.weights[bkey] = mod.bias.detach().cpu().numpy().astype(g.wdtype)
    g._emit(node.name, "conv2d_grouped", [x], [out],
            {"N": N_, "IC": IC, "IH": 1, "IW": IW, "OC": OC, "OH": 1, "OW": OW,
             "KH": 1, "KW": int(mod.kernel_size[0]), "SH": 1, "SW": int(mod.stride[0]),
             "PH": 0, "PW": int(mod.padding[0]), "DH": 1, "DW": int(mod.dilation[0]),
             "G": int(mod.groups)}, weight=wkey, bias=bkey)


def _m_adaptive_avg(g: GenericLowering, node, mod):
    x = g._tensor_arg(node.args[0])
    xs, os_ = g._shape(x), g._meta_shape(node)
    nd = {torch.nn.AdaptiveAvgPool1d: 1, torch.nn.AdaptiveAvgPool3d: 3}.get(type(mod), 2)
    if any(o != 1 for o in os_[-nd:]):
        raise NotImplementedError(f"{node.name}: adaptive avg pool to {os_[-nd:]}")
    g.reduce(node.name, x, list(range(len(xs) - nd, len(xs))), "mean", keepdim=True,
             out_name=g._out(node))


def _m_rnn(g: GenericLowering, node, mod):
    """nn.LSTM / nn.GRU over a whole sequence: one lstm_seq / gru_seq op per
    layer and direction; bidirectional outputs are concatenated on the
    feature dim, h_n/c_n stacked as [layers * dirs, B, H]."""
    is_lstm = isinstance(mod, torch.nn.LSTM)
    if mod.proj_size if is_lstm else False:
        raise NotImplementedError(f"{node.name}: LSTM proj_size")
    x = g._tensor_arg(node.args[0])
    state = node.args[1] if len(node.args) > 1 else node.kwargs.get("hx")
    L, H, D = mod.num_layers, mod.hidden_size, 2 if mod.bidirectional else 1
    xs = g._shape(x)
    if len(xs) != 3:
        raise NotImplementedError(f"{node.name}: unbatched RNN input")
    if not mod.batch_first:
        x = g.permute(node.name, x, [1, 0, 2])
    B, T, _ = g._shape(x)

    def init(st):
        if st is None:
            return g._const(f"{node.name}__zeros{g._tmp}", torch.zeros(L * D, B, H))
        return g._tensor_arg(st)
    if is_lstm:
        h0, c0 = (init(None), init(None)) if state is None else (init(state[0]), init(state[1]))
    else:
        h0 = init(state)

    def layer_state(src, idx):  # [L*D, B, H] -> [B, H] at idx (contiguous window)
        return g.copy_affine(node.name, src, [B, H], idx * B * H, [H, 1])

    hns, cns = [], []
    inp = x
    for l in range(L):
        outs = []
        for d in range(D):
            sfx = f"_l{l}" + ("_reverse" if d else "")
            w_ih = getattr(mod, f"weight_ih{sfx}").detach()
            w_hh = getattr(mod, f"weight_hh{sfx}").detach()
            b_ih = getattr(mod, f"bias_ih{sfx}", None)
            b_hh = getattr(mod, f"bias_hh{sfx}", None)
            G = (4 if is_lstm else 3) * H
            b_ih = b_ih.detach() if b_ih is not None else torch.zeros(G)
            b_hh = b_hh.detach() if b_hh is not None else torch.zeros(G)
            base = f"{node.name}{sfx}"
            wi = g._const(f"{base}__w_ih", w_ih)
            wh = g._const(f"{base}__w_hh", w_hh)
            I = g._shape(inp)[2]
            y = g._new(node.name, [B, T, H])
            hn = g._new(node.name, [B, H])
            sh = {"B": B, "T": T, "I": I, "H": H, "reverse": d}
            if is_lstm:
                bg = g._const(f"{base}__b", b_ih + b_hh)
                cn = g._new(node.name, [B, H])
                g._emit(base, "lstm_seq",
                        [inp, layer_state(h0, l * D + d), layer_state(c0, l * D + d), wi, wh, bg],
                        [y, hn, cn], sh)
                cns.append(cn)
            else:
                bi = g._const(f"{base}__b_ih", b_ih)
                bh = g._const(f"{base}__b_hh", b_hh)
                g._emit(base, "gru_seq", [inp, layer_state(h0, l * D + d), wi, wh, bi, bh],
                        [y, hn], sh)
            hns.append(hn)
            outs.append(y)
        inp = outs[0] if D == 1 else g.cat(node.name, outs, 2)
    out = inp if mod.batch_first else g.permute(node.name, inp, [1, 0, 2])
    stack = lambda ts: g.cat(node.name, ts, 0, stack=True)  # noqa: E731
    g.tuple_items[(node.name, 0)] = out
    g.tuple_items[(node.name, 1)] = [stack(hns), stack(cns)] if is_lstm else stack(hns)


_MODULES = {
    "MultiheadAttention": _m_mha, "Conv1d": _m_conv1d,
    "LSTM": _m_rnn, "GRU": _m_rnn,
    "AdaptiveAvgPool1d": _m_adaptive_avg, "AdaptiveAvgPool2d": _m_adaptive_avg,
    "AdaptiveAvgPool3d": _m_adaptive_avg,
    "Softmax": _m_softmax,
    "LogSoftmax": lambda g, n, m: _m_softmax(g, n, m, log=True),
    "Identity": _m_movement, "Flatten": _m_movement, "Unflatten": _m_movement,
    "ZeroPad2d": _m_movement, "PixelShuffle": _m_movement, "ChannelShuffle": _m_movement,
    "Conv2d": _m_conv2d, "MaxPool2d": _m_maxpool2d,
    "BatchNorm1d": _m_batchnorm, "BatchNorm3d": _m_batchnorm,
    "InstanceNorm1d": _m_instance_norm, "InstanceNorm3d": _m_instance_norm,
    "Hardswish": _m_act("hardswish"), "Mish": _m_act("mish"),
    "Hardtanh": _m_act("clamp", "min_val", "max_val"),
}


def materialize_output_views(ops: list, tensors: dict, input_names: list,
                             output_names: list) -> None:
    """Make every graph output a tensor some kernel actually writes.

    Codegen resolves `view` aliases before mapping tensors to the output
    buffer, so an output that is a view (eval Dropout, flatten, a final
    reshape) would leave the output buffer unwritten. Rename the view's
    source to the output name when the source is an intermediate; copy
    otherwise (source is an input/constant/other output, or an offset view).
    """
    for out in output_names:
        prod_idx = next((i for i, op in enumerate(ops) if out in op["outputs"]), None)
        if prod_idx is None or ops[prod_idx]["op"] != "view":
            continue
        view = ops[prod_idx]
        src = view["inputs"][0]
        src_is_produced = any(src in op["outputs"] and op["op"] != "view" for op in ops)
        if (src_is_produced and not view.get("offset") and src not in input_names
                and src not in output_names and not tensors[src].get("const")):
            for op in ops:
                op["inputs"] = [out if t == src else t for t in op["inputs"]]
                op["outputs"] = [out if t == src else t for t in op["outputs"]]
                for k, v in list(op.get("state", {}).items()):
                    if v == src:
                        op["state"][k] = out
            del ops[prod_idx]
            del tensors[src]
            continue
        n = _prod(tensors[out]["shape"])
        sh = {"off": int(view.get("offset", 0))}
        sh.update({f"d{i}": (n if i == MB_MAXR - 1 else 1) for i in range(MB_MAXR)})
        sh.update({f"s{i}": (1 if i == MB_MAXR - 1 else 0) for i in range(MB_MAXR)})
        ops[prod_idx] = {"name": view["name"], "op": "strided_copy", "inputs": [src],
                         "outputs": [out], "shape": sh}
