"""Shape-specializing fx tracer.

`torch.fx.symbolic_trace` refuses Python that depends on tensor shapes —
`x.size(0)`, `B, C, H, W = x.shape`, `self.bias[:, :, :T, :T]`,
`if x.dim() == 4:`. Those are everywhere in KernelBench level3 (and in real
models), but the IR is per-shape anyway: every op already records concrete
sizes. So we trace with the sample inputs in hand and run each op eagerly
alongside the trace:

  * an op whose result is a tensor (or a tuple holding tensors) becomes a
    graph node, exactly as with symbolic_trace;
  * an op whose result is anything else (int, torch.Size, dtype, device,
    bool) is folded to that Python value, so shape arithmetic, slicing and
    shape-driven control flow just run;
  * `bool(proxy)` on a tensor uses the sample's value; that specializes
    data-dependent branches to the sample and is recorded in
    `ConcreteTracer.value_branches` (callers may warn).

Tensors created inside forward from folded values (`torch.zeros(B, H)`,
`self.h0.to(x.device)`) are not proxies; fx stores them as
`_tensor_constant*` attributes, i.e. `get_attr` nodes (a snapshot, since
forward may mutate them later).

In-place mutation is functionalized while tracing: `x.index_add_(...)`,
`x[idx] = v`, `x.copy_(y)`, `x.fill_(v)` become out-of-place nodes
(`index_add`, `mb_setitem`, `mb_copy`, `mb_fill`) and later reads of `x`
(a proxy or a plain tensor) see the new node. Writing through a view made by
static indexing (`x[:, :4].copy_(y)`, `x[2][mask] = 0`) is replayed as a
`mb_setitem` on its base. Any other tensor sharing the mutated storage (a
`view`/`transpose` of it, or its base) is marked stale and reading it later
raises, as do data-dependent shapes/values (`nonzero`, `.item()`,
boolean-mask indexing, ...): the IR is static, so those would otherwise be
baked to the sample.

The resulting GraphModule has the same node vocabulary as symbolic_trace
for code symbolic_trace could already handle, so the extractor's lowering
is unchanged.
"""
from __future__ import annotations

import operator
from typing import Any

import torch
import torch.fx
from torch.fx.proxy import Attribute, Proxy


TRACE_SEED = 0


def _has_tensor(v: Any) -> bool:
    if isinstance(v, torch.Tensor):
        return True
    if isinstance(v, (tuple, list)):
        return any(_has_tensor(x) for x in v)
    if isinstance(v, dict):
        return any(_has_tensor(x) for x in v.values())
    return False


def _fetch_attr(root: torch.nn.Module, target: str):
    obj = root
    for atom in target.split("."):
        obj = getattr(obj, atom)
    return obj


# -- functional stand-ins for in-place ops (call_function targets) -----------
def mb_setitem(x, idx, v):
    """x[idx] = v, out of place."""
    y = x.clone()
    y[idx] = v
    return y


def mb_copy(x, src):
    """x.copy_(src), out of place: src broadcast to x's shape and dtype."""
    return src.to(x.dtype).expand_as(x).clone()


def mb_fill(x, v):
    """x.fill_(v) / x.zero_(), out of place."""
    return torch.full_like(x, v)


# methods that would make the IR depend on tensor *values* for its shape or
# constants; the static IR can't represent them faithfully
_DATA_DEP = {"item", "tolist", "nonzero", "argwhere", "unique", "unique_consecutive",
             "bincount", "masked_select", "histc", "_local_scalar_dense"}
# tensor factories: inside forward they return proxies (folded back to
# constants after tracing), so later in-place writes and reads stay traced
_FACTORIES = ("zeros", "ones", "full", "empty", "arange", "eye", "linspace", "tensor",
              "as_tensor", "zeros_like", "ones_like", "full_like", "empty_like")
# in-place methods that don't change data
_INPLACE_NOOP = {"requires_grad_", "retain_grad", "share_memory_", "detach_"}
def mb_zero(x):
    """x.zero_(), out of place."""
    return torch.zeros_like(x)


_SPECIAL_INPLACE = {"copy_": mb_copy, "fill_": mb_fill, "zero_": mb_zero}


def _static_index(idx) -> bool:
    """Basic indexing (ints/slices/None/Ellipsis): the result is a view."""
    items = idx if isinstance(idx, tuple) else (idx,)
    for it in items:
        if isinstance(it, slice):
            if not all(v is None or isinstance(v, int) for v in (it.start, it.stop, it.step)):
                return False
        elif not (it is None or it is Ellipsis or (isinstance(it, int) and not isinstance(it, bool))):
            return False
    return True


def _storage(v):
    try:
        return v.untyped_storage().data_ptr() if isinstance(v, torch.Tensor) else None
    except Exception:
        return None


class ConcreteProxy(Proxy):
    def __setitem__(self, idx, v):
        self.tracer.create_proxy("call_function", operator.setitem, (self, idx, v), {})

    def __len__(self):
        return len(self.tracer.concrete(self))

    def __getattr__(self, k):
        # Data attributes that aren't tensors (shape, dtype, device, ndim) are
        # just values; methods and tensor attributes (.T) still trace.
        if not k.startswith("_"):
            v = getattr(self.tracer.concrete(self), k)
            if not callable(v) and not _has_tensor(v):
                return v
        return super().__getattr__(k)


class ConcreteTracer(torch.fx.Tracer):
    def __init__(self, sample_inputs, **kw):
        super().__init__(**kw)
        self._samples = list(sample_inputs)
        self._n_placeholders = 0
        self._values: dict[torch.fx.Node, Any] = {}
        self._eager = 0
        # plain (non-parameter) tensors overwritten in place with a traced
        # value, `self.hidden.copy_(h0)`: later reads of them are that value
        self._overwritten: dict[int, Proxy] = {}
        self.value_branches: list[str] = []
        # functionalization: node -> proxy holding its current value; nodes
        # whose storage was mutated through an alias we can't replay; the
        # storage each node's value lives in; pre-mutation snapshots
        self._alias: dict[torch.fx.Node, Proxy] = {}
        self._stale: set = set()
        self._stale_ids: set = set()
        self._storage: dict[torch.fx.Node, int] = {}
        self._pre: dict[torch.fx.Node, Any] = {}

    # -- concrete values -------------------------------------------------
    def concrete(self, a: Any) -> Any:
        if isinstance(a, Attribute):
            return getattr(self.concrete(a.root), a.attr)
        if isinstance(a, Proxy):
            return self._node_value(a.node)
        if isinstance(a, torch.fx.Node):
            return self._node_value(a)
        if isinstance(a, tuple):
            return tuple(self.concrete(x) for x in a)
        if isinstance(a, list):
            return [self.concrete(x) for x in a]
        if isinstance(a, dict):
            return {k: self.concrete(v) for k, v in a.items()}
        if isinstance(a, slice):
            return slice(self.concrete(a.start), self.concrete(a.stop), self.concrete(a.step))
        return a

    def _node_value(self, node: torch.fx.Node) -> Any:
        if node not in self._values and node.op == "get_attr":
            self._values[node] = self._run("get_attr", node.target, (), {})
        return self._values[node]

    def _fold(self, a: Any) -> Any:
        """Replace proxies whose value is not tensor-like with the value."""
        if isinstance(a, Proxy):
            v = self.concrete(a)
            return a if _has_tensor(v) else v
        if isinstance(a, tuple):
            return tuple(self._fold(x) for x in a)
        if isinstance(a, list):
            return [self._fold(x) for x in a]
        if isinstance(a, dict):
            return {k: self._fold(v) for k, v in a.items()}
        if isinstance(a, slice):
            return slice(self._fold(a.start), self._fold(a.stop), self._fold(a.step))
        return a

    def _run(self, kind: str, target, args, kwargs):
        # Everything here executes eagerly: Module.__call__/__getattr__ are
        # patched for the duration of the trace (see call_module/getattr).
        self._eager += 1
        try:
            return self._run_eager(kind, target, args, kwargs)
        finally:
            self._eager -= 1

    def _run_eager(self, kind: str, target, args, kwargs):
        cargs, ckwargs = self.concrete(args), self.concrete(kwargs)
        if kind == "placeholder":
            # fx still emits placeholders for concrete_args; they bind None
            # and are erased after tracing.
            i = self._n_placeholders
            self._n_placeholders += 1
            return self._samples[i] if i < len(self._samples) else None
        if kind == "get_attr":
            return _fetch_attr(self.root, target)
        if kind == "call_function":
            return target(*cargs, **ckwargs)
        if kind == "call_method":
            return getattr(cargs[0], target)(*cargs[1:], **ckwargs)
        if kind == "call_module":
            return self.root.get_submodule(target)(*cargs, **ckwargs)
        raise RuntimeError(f"concrete trace: unexpected node kind {kind}")

    # -- tracer hooks ----------------------------------------------------
    def call_module(self, m, forward, args, kwargs):
        # Module.__call__ is patched for the whole trace; while computing a
        # leaf module's concrete value, run it (and its children) eagerly.
        if self._eager:
            return forward(*args, **kwargs)
        return super().call_module(m, forward, args, kwargs)

    def getattr(self, attr, attr_val, parameter_proxy_cache):
        # Module.__getattr__ is patched too: parameters read during eager
        # execution must stay real tensors.
        if self._eager:
            return attr_val
        return super().getattr(attr, attr_val, parameter_proxy_cache)

    def _cur(self, a: Any) -> Any:
        """Resolve proxies to their current (post-mutation) version."""
        if isinstance(a, Proxy):
            if a.node in self._stale:
                raise NotImplementedError(
                    f"{a.node.name} is read after an in-place write through an alias "
                    "(view) that the tracer can't replay")
            while a.node in self._alias:
                a = self._alias[a.node]
            return a
        if isinstance(a, tuple):
            return tuple(self._cur(x) for x in a)
        if isinstance(a, list):
            return [self._cur(x) for x in a]
        if isinstance(a, dict):
            return {k: self._cur(v) for k, v in a.items()}
        if isinstance(a, slice):
            return slice(self._cur(a.start), self._cur(a.stop), self._cur(a.step))
        return a

    def _check_data_dependent(self, kind, target, args):
        nm = target if isinstance(target, str) else getattr(target, "__name__", "")
        if nm == "where" and len(args) == 1 and isinstance(args[0], Proxy):
            nm = "nonzero"   # torch.where(cond) is nonzero(cond, as_tuple=True)
        if kind in ("call_method", "call_function") and nm in _DATA_DEP:
            if kind == "call_function" or isinstance(args[0], Proxy):
                raise NotImplementedError(
                    f"data-dependent {nm}(): the result's shape/value depends on tensor "
                    "data, which a static IR can't represent")
        if kind in ("call_function", "call_method") and nm in ("getitem", "__getitem__", "setitem",
                                                               "__setitem__"):
            items = args[1] if isinstance(args[1], tuple) else (args[1],)
            for it in items:
                v = self.concrete(it) if isinstance(it, Proxy) else it
                if isinstance(v, torch.Tensor) and v.dtype == torch.bool and (
                        nm.endswith("getitem") or isinstance(args[2], Proxy) and
                        self.concrete(args[2]).numel() > 1 and
                        self.concrete(args[2]).dim() > 0 and
                        list(self.concrete(args[2]).shape) != list(v.shape)):
                    raise NotImplementedError(
                        "boolean-mask indexing: the result's shape depends on tensor data")

    def _inplace(self, kind, target, args, kwargs):
        """Out-of-place (kind, target) for a mutating call, or None."""
        if kind == "call_function" and target in (operator.setitem, torch.Tensor.__setitem__):
            return ("call_function", mb_setitem)
        if kind != "call_method" or not isinstance(target, str):
            return None
        if target == "__setitem__":
            return ("call_function", mb_setitem)
        if target in _SPECIAL_INPLACE:
            return ("call_function", _SPECIAL_INPLACE[target])
        if (target.endswith("_") and not target.endswith("__") and target not in _INPLACE_NOOP
                and hasattr(torch.Tensor, target[:-1])):
            return ("call_method", target[:-1])
        return None

    def _functionalize(self, kind, target, args, kwargs, fkind, ftarget):
        base = args[0]
        live = self.concrete(base)
        if not isinstance(base, Proxy) and id(base) in self._overwritten:
            base_arg = self._cur(self._overwritten[id(base)])   # written before
        elif isinstance(base, Proxy):
            self._pre.setdefault(base.node, live.clone())
            base_arg = base
        else:
            # a plain tensor: snapshot it before the write so earlier readers
            # (and the functional node's input) keep the old data
            base_arg = self.proxy(self.create_arg(live.clone()))
            self._values[base_arg.node] = self._pre[base_arg.node] = live.clone()
        # every value sharing the written storage (views of it, its base)
        # changes in place: keep their pre-write data for constant folding
        sid = _storage(live)
        if sid is not None:
            for n, st in self._storage.items():
                if st == sid and n not in self._pre and isinstance(self._values.get(n), torch.Tensor):
                    self._pre[n] = self._values[n].clone()
        self._run(kind, target, args, kwargs)        # mutate the concrete value too
        fargs = (base_arg,) + tuple(args[1:])
        p = super().create_proxy(fkind, ftarget, fargs, kwargs)
        self._values[p.node] = live.clone()
        self._storage[p.node] = _storage(self._values[p.node])
        if isinstance(base, Proxy):
            self._alias[base.node] = p
            self._propagate_view_write(base.node, p, _storage(live))
        else:
            self._overwritten[id(base)] = p
        return p

    def _propagate_view_write(self, node, newval: Proxy, sid) -> None:
        """`node` (now `newval`) may be a view; write through to its base."""
        handled = {node}
        while (node.op == "call_function" and node.target is operator.getitem
               and isinstance(node.args[0], torch.fx.Node)
               and _storage(self._values.get(node.args[0])) == sid
               and _static_index(node.args[1])):
            base = node.args[0]
            cur = self._cur(self.proxy(base))
            self._pre.setdefault(base, self.concrete(base).clone())
            newval = super().create_proxy("call_function", mb_setitem,
                                          (cur, node.args[1], newval), {})
            self._values[newval.node] = self.concrete(base).clone()
            self._alias[base] = newval
            handled.add(base)
            node = base
        if sid is None:
            return
        for n, st in self._storage.items():
            if st == sid and n not in handled and n not in self._alias and n.op != "placeholder":
                self._stale.add(n)
        for n, st in self._storage.items():
            if st == sid and n.op == "placeholder" and n not in handled:
                self._stale.add(n)

    def create_proxy(self, kind, target, args, kwargs, name=None, type_expr=None,
                     proxy_factory_fn=None):
        args, kwargs = self._fold(args), self._fold(kwargs)
        if kind not in ("placeholder", "get_attr", "output"):
            args, kwargs = self._cur(args), self._cur(kwargs)
            self._check_data_dependent(kind, target, args)
            f = self._inplace(kind, target, args, kwargs)
            if f is not None and (isinstance(args[0], Proxy) or any(
                    isinstance(x, Proxy) for x in list(args[1:]) + list(kwargs.values()))):
                return self._functionalize(kind, target, args, kwargs, *f)
            if kind == "call_method" and target in _INPLACE_NOOP:
                return args[0]
        value = self._run(kind, target, args, kwargs)
        if kind in ("call_function", "call_method") and not _has_tensor(value):
            return value
        p = super().create_proxy(kind, target, args, kwargs, name, type_expr,
                                 proxy_factory_fn)
        self._values[p.node] = value
        if isinstance(value, torch.Tensor):
            self._storage[p.node] = _storage(value)
        return p

    # Containers whose forward is plain module calls + Python control flow:
    # trace through them so their parts lower individually.
    _TRACE_THROUGH = (torch.nn.TransformerEncoderLayer, torch.nn.TransformerEncoder,
                      torch.nn.TransformerDecoderLayer, torch.nn.TransformerDecoder,
                      torch.nn.Transformer)

    def is_leaf_module(self, m, module_qualified_name):
        if isinstance(m, self._TRACE_THROUGH):
            return False
        return super().is_leaf_module(m, module_qualified_name)

    def create_arg(self, a):
        if isinstance(a, Proxy):
            a = self._cur(a)   # also resolves the graph outputs
        if isinstance(a, torch.Tensor) and id(a) in self._overwritten:
            p = self._cur(self._overwritten[id(a)])
            return p.node
        if (isinstance(a, torch.Tensor) and not isinstance(a, torch.nn.Parameter)
                and self.tensor_attrs.get(a) is None and not self._eager):
            # a plain tensor made in forward: freeze its current data (forward
            # may still mutate it in place)
            a = a.detach().clone()
        return super().create_arg(a)

    def proxy(self, node: torch.fx.Node) -> Proxy:
        return ConcreteProxy(node, self)

    def to_bool(self, obj: Proxy) -> bool:
        self.value_branches.append(
            f"{obj.root}.{obj.attr}" if isinstance(obj, Attribute) else obj.node.name)
        return bool(self.concrete(obj))

    def iter(self, obj: Proxy):
        v = self.concrete(obj)
        if isinstance(obj, Attribute) or not _has_tensor(v):
            return iter(v)
        return iter([obj[i] for i in range(len(v))])

    def keys(self, obj: Proxy):
        return self.concrete(obj).keys()


def _register_einops() -> None:
    """einops picks a backend by tensor type; give our proxies the torch one
    (its ops are plain tensor methods, which trace normally)."""
    try:
        import einops._backends as eb
    except ImportError:
        return
    eb._type2backend[ConcreteProxy] = eb.TorchBackend()


def _fold_constants(gm: torch.fx.GraphModule, values: dict) -> None:
    """Replace tensors computed only from parameters/buffers/constants (e.g.
    `clamp(self.logit_scale, max=...)`, Mamba's A/B/C preprocessing) with
    their value, stored as a new attribute (a get_attr node)."""
    const: set = set()
    for n in gm.graph.nodes:
        if n.op == "get_attr":
            const.add(n)
        elif n.op in ("call_function", "call_method", "call_module"):
            ins = n.all_input_nodes
            if (ins and all(i in const for i in ins)) or (not ins and n.op == "call_function"):
                const.add(n)   # input-less calls are tensor factories (torch.zeros)
    k = 0
    for n in list(gm.graph.nodes):
        if (n not in const or n.op == "get_attr"
                or not isinstance(values.get(n), torch.Tensor)
                or all(u in const for u in n.users)):
            continue
        attr = f"_mb_folded_{k}"
        k += 1
        gm.register_buffer(attr, values[n].detach().clone())
        with gm.graph.inserting_before(n):
            new = gm.graph.get_attr(attr)
        new.meta.update(n.meta)
        n.replace_all_uses_with(new)


def concrete_trace(model: torch.nn.Module, sample_inputs) -> torch.fx.GraphModule:
    if isinstance(sample_inputs, torch.Tensor):
        sample_inputs = [sample_inputs]
    # forward may rebind module attributes (RNN state: `self.hidden = ...`)
    # to proxies, and the caller still runs the model for the golden output:
    # snapshot every module's attributes and restore them after tracing. (A
    # deepcopy would break forwards that close over submodules, e.g. MinGPT's
    # `self.mlpf = lambda x: m.c_proj(...)`.) The RNG is seeded so tensors
    # drawn inside forward (h0 = torch.randn(...)) match a golden run made
    # after torch.manual_seed(TRACE_SEED).
    saved = {m: {k: (dict(v) if isinstance(v, dict) else v) for k, v in m.__dict__.items()}
             for m in model.modules()}
    torch.manual_seed(TRACE_SEED)
    _register_einops()
    tracer = ConcreteTracer(sample_inputs)
    # forward params past the sample inputs (`mask=None`, `initial_states=None`)
    # are bound to their defaults rather than becoming graph inputs.
    import inspect
    params = list(inspect.signature(model.forward).parameters.values())
    concrete_args = {p.name: p.default for p in params[len(sample_inputs):]
                     if p.default is not inspect.Parameter.empty}
    # nn.MultiheadAttention / TransformerEncoderLayer would otherwise take
    # the fused native fast path in eval mode.
    mha_fast = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    saved_fn = {n: getattr(torch, n) for n in _FACTORIES}

    def _traced_factory(fn):
        def inner(*args, **kwargs):
            if tracer._eager or getattr(tracer, "root", None) is None:
                return fn(*args, **kwargs)
            # `dtype=bool` (a Python type) is legal but not a graph argument
            if isinstance(kwargs.get("dtype"), type):
                kwargs = {**kwargs, "dtype": _PY_DTYPES[kwargs["dtype"]]}
            return tracer.create_proxy("call_function", fn, args, kwargs)
        return inner
    try:
        for n, fn in saved_fn.items():
            setattr(torch, n, _traced_factory(fn))
        with torch.no_grad():
            graph = tracer.trace(model, concrete_args=concrete_args or None)
    finally:
        for n, fn in saved_fn.items():
            setattr(torch, n, fn)
        torch.backends.mha.set_fastpath_enabled(mha_fast)
    gm = torch.fx.GraphModule(tracer.root, graph, model.__class__.__name__)
    for m, d in saved.items():
        m.__dict__.clear()
        m.__dict__.update(d)
    _fold_constants(gm, {**tracer._values, **tracer._pre})
    gm.graph.eliminate_dead_code()
    for i, n in enumerate([n for n in gm.graph.nodes if n.op == "placeholder"]):
        if i >= len(sample_inputs) and not n.users:
            gm.graph.erase_node(n)
    gm.recompile()
    _check_outputs_depend_on_inputs(gm)
    gm.meta["value_branches"] = tracer.value_branches
    return gm


_PY_DTYPES = {bool: torch.bool, int: torch.int64, float: torch.float32, complex: torch.complex64}


def _check_outputs_depend_on_inputs(gm: torch.fx.GraphModule) -> None:
    """A graph output computed from constants only means the trace folded the
    real computation into the sample's values (or the forward ignores its
    inputs); either way the IR would be a constant, so refuse it."""
    dep: set = set()
    for n in gm.graph.nodes:
        if n.op == "placeholder" or any(i in dep for i in n.all_input_nodes):
            dep.add(n)
        if n.op == "output":
            outs = torch.fx.node.map_arg(n.args[0], lambda x: x)
            flat = []
            torch.fx.node.map_arg(outs, flat.append)
            bad = [o.name for o in flat if o not in dep]
            if bad:
                raise NotImplementedError(
                    f"graph output(s) {bad} do not depend on any input: the forward's "
                    "result was folded to the sample's values")
