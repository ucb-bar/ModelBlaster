"""PyTorch model -> IR JSON + weights.npz + io.npz.

IR shape (v1):
  {
    "name": <model name>,
    "version": 1,
    "input":  {"tensor": <name>},
    "output": {"tensor": <name>},
    "tensors": {
      <name>: {"shape": [...], "dtype": "f32", "quant": null}
    },
    "ops": [
      {"name": <node name>, "op": "linear",
       "inputs": [<name>], "outputs": [<name>],
       "weight": <name>, "bias": <name>,
       "shape": {"M": ..., "K": ..., "N": ...},
       /* dispatch fields, post-processed by _annotate_dispatches: */
       "dispatch_id": <int|null>,        # null for view ops; else 0..N-1
       "hardware_target": "any",         # "scalar","rvv","gemmini",...; "any" = whatever the build picks
       "depends_on": [<dispatch_id>...]  # other dispatches that must complete first (data deps)
      },
      ...
    ],
    "dispatches": [<dispatch_id>...]      # ordered list of non-view dispatch_ids
  }

Quantization fields are reserved (`dtype`/`quant` per tensor) so the int8 PT2E
flow can land without changing the schema. fp32 first cut.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import torch
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp


def _annotate_dispatches(ops: list[dict]) -> list[int]:
    """Promote each non-view op to a first-class dispatch.

    Adds three fields to each op (in-place):
      * dispatch_id: 0..N-1 across non-view ops in execution order.
        view ops get None — they're zero-cost tensor aliases, not
        runnable dispatches.
      * hardware_target: forward-compat for the heterogeneous core
        registry (task 62). Defaults to "any" — the build picks.
      * depends_on: list of dispatch_ids whose outputs feed this op,
        derived from the data-flow graph. view ops propagate their
        producer transitively so dependents see the real source.

    Returns the ordered list of dispatch_ids (= ir["dispatches"])."""
    producer_of: dict[str, int] = {}
    next_id = 0
    dispatches: list[int] = []
    for op in ops:
        if op["op"] == "view":
            # view aliases input tensor; propagate producer info so
            # downstream ops see the real upstream dispatch.
            for t_in, t_out in zip(op.get("inputs", []), op.get("outputs", [])):
                if t_in in producer_of:
                    producer_of[t_out] = producer_of[t_in]
            op["dispatch_id"] = None
            op["hardware_target"] = "any"
            op["depends_on"] = []
            continue

        deps: set[int] = set()
        for t_in in op.get("inputs", []):
            if t_in in producer_of:
                deps.add(producer_of[t_in])
        op["dispatch_id"] = next_id
        op["hardware_target"] = "any"
        op["depends_on"] = sorted(deps)
        for t_out in op.get("outputs", []):
            producer_of[t_out] = next_id
        dispatches.append(next_id)
        next_id += 1
    return dispatches


# ---------------------------------------------------------------------------
# Compound-activation pattern recognizer. Walks the FX graph from the
# output node backward; if the entire forward expression matches a known
# multi-op activation (Swish, Softsign, MinGPT-style exact GELU), we
# replace the subgraph with a single sentinel call_function node so the
# downstream per-node IR emit loop sees one tidy op instead of 4-8.
# ---------------------------------------------------------------------------

import operator as _operator


def _round_half_away(x):
    """Round half away from zero, like C's round()/roundf().

    numpy's np.round is half-to-EVEN (banker's rounding), so a value landing
    exactly on .5 goes the other way from the generated C. On an int8 grid that
    is a whole LSB of disagreement between the golden simulator and the device,
    and it shows up as a verify FAIL with max_abs_err=1 that no amount of extra
    float precision fixes -- the two are computing the same real number and
    disagreeing about how to round it.
    """
    import numpy as _np
    return _np.sign(x) * _np.floor(_np.abs(x) + 0.5)


def _find_getitem_consumer(node, index: int):
    """Return the unique operator.getitem consumer of `node` selecting
    `index`, or None. Used by the torch.max/min handlers to find the
    [0] (values) consumer so the parent reduction can write directly
    into that tensor's buffer."""
    op_getitem = _operator.getitem
    for user in node.users:
        if user.op != "call_function" or user.target is not op_getitem:
            continue
        args = user.args
        if len(args) >= 2 and isinstance(args[1], int) and args[1] == index:
            return user
    return None


def _is_const(node, value: float, tol: float = 1e-9) -> bool:
    """Constants in FX show up as literal Python values in node.args
    (not as separate get_attr nodes for these compact benches), so a
    plain float compare suffices."""
    return isinstance(node, (int, float)) and abs(float(node) - value) < tol


def _match_swish(out, x):
    """Pattern: out = mul(x, sigmoid(x)). Order-insensitive."""
    if out.op != "call_function":
        return False
    if out.target not in (_operator.mul, torch.mul):
        return False
    a, b = out.args
    sig_node = None
    other = None
    for cand, alt in ((a, b), (b, a)):
        if (hasattr(cand, "op") and cand.op == "call_function"
                and cand.target in (torch.sigmoid,
                                    torch.nn.functional.sigmoid)):
            sig_node = cand
            other = alt
            break
    if sig_node is None:
        return False
    if other is not x:
        return False
    if len(sig_node.args) != 1 or sig_node.args[0] is not x:
        return False
    return True


def _match_softsign(out, x):
    """Pattern: out = div(x, add(1.0, abs(x))). PyTorch traces
    `x / (1 + |x|)` to operator.truediv with operator.add and
    torch.abs."""
    if out.op != "call_function":
        return False
    if out.target not in (_operator.truediv, torch.div):
        return False
    if len(out.args) != 2 or out.args[0] is not x:
        return False
    add = out.args[1]
    if not (hasattr(add, "op") and add.op == "call_function"
            and add.target in (_operator.add, torch.add)):
        return False
    a, b = add.args
    abs_node, one_val = None, None
    for cand, alt in ((a, b), (b, a)):
        if (hasattr(cand, "op") and cand.op == "call_function"
                and cand.target in (torch.abs, _operator.abs)):
            abs_node = cand
            one_val = alt
            break
    if abs_node is None or not _is_const(one_val, 1.0):
        return False
    if len(abs_node.args) != 1 or abs_node.args[0] is not x:
        return False
    return True


def _match_gelu_exact(out, x):
    """Pattern: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    Tolerant of FX's typical AST flattening: the outer 0.5*x*... can
    show up either as mul(0.5, mul(x, ...)) or mul(mul(0.5, x), ...).
    We walk a small DFS looking for tanh of the right shape."""
    if out.op != "call_function":
        return False
    if out.target not in (_operator.mul, torch.mul):
        return False
    # Find a `tanh(...)` somewhere two levels deep, with a half scalar
    # and an `x` factor in the chain.
    seen_half = [False]
    seen_x = [False]
    tanh_node = [None]

    def _walk(n, depth=0):
        if depth > 6:
            return
        if isinstance(n, (int, float)):
            if abs(float(n) - 0.5) < 1e-6:
                seen_half[0] = True
            return
        if not hasattr(n, "op"):
            return
        if n is x:
            seen_x[0] = True
            return
        if n.op == "call_function":
            if n.target in (torch.tanh, torch.nn.functional.tanh):
                tanh_node[0] = n
            for a in n.args:
                _walk(a, depth + 1)

    _walk(out)
    if not (seen_half[0] and seen_x[0] and tanh_node[0] is not None):
        return False

    # Check the tanh argument is `k * (x + c * pow(x, 3))`-shaped.
    inner = tanh_node[0].args[0]
    has_pow_x3 = [False]
    has_const_k = [False]
    has_const_c = [False]

    def _walk2(n, depth=0):
        if depth > 6:
            return
        if isinstance(n, (int, float)):
            v = float(n)
            if abs(v - 0.7978845608028654) < 1e-3 \
                    or abs(v * v - 2.0 / 3.141592653589793) < 1e-3:
                has_const_k[0] = True
            if abs(v - 0.044715) < 1e-4:
                has_const_c[0] = True
            return
        if not hasattr(n, "op"):
            return
        if n.op == "call_function":
            if n.target in (torch.pow, _operator.pow):
                # pow(x, 3)
                if (n.args[0] is x and isinstance(n.args[1], (int, float))
                        and abs(float(n.args[1]) - 3.0) < 1e-3):
                    has_pow_x3[0] = True
            for a in n.args:
                _walk2(a, depth + 1)

    _walk2(inner)
    return has_pow_x3[0] and has_const_c[0] and has_const_k[0]


def _match_l1_norm(out, x):
    """Pattern: out = div(x, sum(abs(x), dim=K, keepdim=True)). The
    sum is a single torch.sum call; abs may be torch.abs or
    operator.abs."""
    if out.op != "call_function":
        return False
    if out.target not in (_operator.truediv, torch.div):
        return False
    if len(out.args) != 2 or out.args[0] is not x:
        return False
    sum_node = out.args[1]
    if not (hasattr(sum_node, "op") and sum_node.op == "call_function"
            and sum_node.target is torch.sum):
        return False
    abs_node = sum_node.args[0]
    if not (hasattr(abs_node, "op") and abs_node.op == "call_function"
            and abs_node.target in (torch.abs, _operator.abs)):
        return False
    if len(abs_node.args) != 1 or abs_node.args[0] is not x:
        return False
    return True


def _is_norm_target(node) -> bool:
    """torch.norm / torch.linalg.norm / torch.linalg.vector_norm —
    matched by either identity OR the trailing `__name__` since FX
    sometimes records bound CFunction wrappers whose identity comparison
    against the public `torch.norm` reference fails."""
    if not hasattr(node, "op") or node.op != "call_function":
        return False
    t = node.target
    if t in (torch.norm, torch.linalg.norm,
             getattr(torch.linalg, "vector_norm", None)):
        return True
    name = getattr(t, "__name__", "")
    return name in ("norm", "vector_norm")


def _match_l2_norm(out, x):
    """Pattern: out = div(x, torch.norm(x, p=2, dim=K, keepdim=True)).
    FX trace fills in default kwargs even when user-omitted, so we
    treat `dim=None` as "no dim" rather than relying on key presence."""
    if out.op != "call_function":
        return False
    if out.target not in (_operator.truediv, torch.div):
        return False
    if len(out.args) != 2 or out.args[0] is not x:
        return False
    norm_node = out.args[1]
    if not _is_norm_target(norm_node):
        return False
    kw = norm_node.kwargs or {}
    p = kw.get("p", 2)
    dim = kw.get("dim", None)
    if p not in (2, "2") or dim is None:
        return False
    if len(norm_node.args) < 1 or norm_node.args[0] is not x:
        return False
    return True


def _match_frobenius_norm(out, x):
    """Pattern: out = div(x, torch.norm(x, p='fro')). The Frobenius
    norm is a global scalar — `dim` either absent OR explicitly None."""
    if out.op != "call_function":
        return False
    if out.target not in (_operator.truediv, torch.div):
        return False
    if len(out.args) != 2 or out.args[0] is not x:
        return False
    norm_node = out.args[1]
    if not _is_norm_target(norm_node):
        return False
    kw = norm_node.kwargs or {}
    p = kw.get("p", 2)
    dim = kw.get("dim", None)
    if p != "fro" and p not in (2, "2"):
        return False
    if dim is not None:
        return False
    if len(norm_node.args) < 1 or norm_node.args[0] is not x:
        return False
    return True


def _match_rms_norm(out, x):
    """Pattern: out = x / sqrt(mean(x**2, dim=1, keepdim=True) + eps).
    Returns eps (float) on match, else None (KernelBench 36_RMSNorm)."""
    if out.op != "call_function" or out.target not in (_operator.truediv, torch.div):
        return None
    if len(out.args) != 2 or out.args[0] is not x:
        return None
    sqrt_node = out.args[1]
    if not (hasattr(sqrt_node, "op") and sqrt_node.op == "call_function"
            and getattr(sqrt_node.target, "__name__", "") == "sqrt"):
        return None
    add_node = sqrt_node.args[0]
    if not (hasattr(add_node, "op") and add_node.op == "call_function"
            and add_node.target in (_operator.add, torch.add)):
        return None
    mean_node, eps = add_node.args[0], add_node.args[1]
    if not isinstance(eps, (int, float)):
        mean_node, eps = add_node.args[1], add_node.args[0]
    if not isinstance(eps, (int, float)):
        return None
    if not (hasattr(mean_node, "op") and mean_node.op == "call_function"
            and getattr(mean_node.target, "__name__", "") == "mean"):
        return None
    mkw = mean_node.kwargs or {}
    dim = mkw.get("dim", None)
    if dim is None and len(mean_node.args) > 1:
        dim = mean_node.args[1]
    if dim != 1:
        return None
    pow_node = mean_node.args[0]
    if not (hasattr(pow_node, "op") and pow_node.op == "call_function"
            and pow_node.target in (_operator.pow, torch.pow)):
        return None
    if len(pow_node.args) < 2 or pow_node.args[0] is not x or pow_node.args[1] != 2:
        return None
    return float(eps)


def _match_mean_abs_norm(out, x):
    """Pattern: out = x / mean(abs(x), dim=1, keepdim=True) (KernelBench 38,
    L1 normalization). Returns True on match."""
    if out.op != "call_function" or out.target not in (_operator.truediv, torch.div):
        return False
    if len(out.args) != 2 or out.args[0] is not x:
        return False
    mean_node = out.args[1]
    if not (hasattr(mean_node, "op") and mean_node.op == "call_function"
            and getattr(mean_node.target, "__name__", "") == "mean"):
        return False
    mkw = mean_node.kwargs or {}
    dim = mkw.get("dim", None)
    if dim is None and len(mean_node.args) > 1:
        dim = mean_node.args[1]
    if dim != 1:
        return False
    abs_node = mean_node.args[0]
    if not (hasattr(abs_node, "op") and abs_node.op == "call_function"
            and abs_node.target in (torch.abs, _operator.abs)):
        return False
    return len(abs_node.args) >= 1 and abs_node.args[0] is x


def _agents_loss_mse(a, b):
    return a


def _agents_loss_hinge(a, b):
    return a


def _is_mean_all(node) -> bool:
    """torch.mean with no dim → scalar (mean over all elements)."""
    if not (hasattr(node, "op") and node.op == "call_function"
            and getattr(node.target, "__name__", "") == "mean"):
        return False
    kw = node.kwargs or {}
    return kw.get("dim") is None and len(node.args) < 2


def _maybe_fuse_loss(gm) -> None:
    """Fuse whole-graph loss patterns that produce a scalar from 2 inputs into
    a single op: MSE = mean((a-b)^2), Hinge = mean(clamp(1 - a*b, min=0)).
    Single-node losses (cross_entropy, smooth_l1, kl_div, TripletMarginLoss)
    are handled directly in the per-node loop, not here."""
    outputs = [n for n in gm.graph.nodes if n.op == "output"]
    if len(outputs) != 1:
        return
    out_arg = outputs[0].args[0]
    if not (hasattr(out_arg, "op") and _is_mean_all(out_arg)):
        return
    inner = out_arg.args[0]  # the tensor being mean-reduced

    sentinel = None
    # MSE: mean(pow(sub(a, b), 2))
    if (hasattr(inner, "op") and inner.op == "call_function"
            and inner.target in (_operator.pow, torch.pow)
            and len(inner.args) >= 2 and inner.args[1] == 2):
        sub = inner.args[0]
        if (hasattr(sub, "op") and sub.op == "call_function"
                and sub.target in (_operator.sub, torch.sub)
                and all(isinstance(x, torch.fx.Node) for x in sub.args[:2])):
            a, b = sub.args[0], sub.args[1]
            sentinel = _agents_loss_mse
    # Hinge: mean(clamp(sub(1, mul(a, b)), min=0))
    if sentinel is None and (hasattr(inner, "op") and inner.op == "call_function"
            and getattr(inner.target, "__name__", "") == "clamp"):
        rsub = inner.args[0]
        if (hasattr(rsub, "op") and rsub.op == "call_function"
                and rsub.target in (_operator.sub, torch.sub, torch.rsub)):
            mul = rsub.args[1] if rsub.args[0] == 1 else (
                rsub.args[0] if rsub.target is torch.rsub else None)
            if (hasattr(mul, "op") and mul.op == "call_function"
                    and mul.target in (_operator.mul, torch.mul)
                    and all(isinstance(x, torch.fx.Node) for x in mul.args[:2])):
                a, b = mul.args[0], mul.args[1]
                sentinel = _agents_loss_hinge
    if sentinel is None:
        return

    n = int(np.prod(list(inner.meta["tensor_meta"].shape)))
    with gm.graph.inserting_before(outputs[0]):
        new_node = gm.graph.call_function(sentinel, args=(a, b))
    new_node.meta["loss_n"] = n
    if "tensor_meta" in out_arg.meta:
        new_node.meta["tensor_meta"] = out_arg.meta["tensor_meta"]
    outputs[0].args = (new_node,)
    gm.graph.eliminate_dead_code()
    gm.recompile()


def _match_exclusive_cumsum(out, x):
    """Pattern (KernelBench 92): cumsum(cat([zeros_like(...), x])[:-1], dim=1).
    x is 2D [B, N]; output is [B-1, N+1]. Returns True on match."""
    if not (out.op == "call_function"
            and getattr(out.target, "__name__", "") == "cumsum"):
        return False
    gi = out.args[0]
    if not (hasattr(gi, "op") and gi.op == "call_function"
            and getattr(gi.target, "__name__", "") == "getitem"):
        return False
    cat = gi.args[0]
    if not (hasattr(cat, "op") and cat.op == "call_function"
            and getattr(cat.target, "__name__", "") == "cat"):
        return False
    seq = cat.args[0]
    if not (isinstance(seq, (tuple, list)) and len(seq) == 2 and seq[1] is x):
        return False
    zl = seq[0]
    return (hasattr(zl, "op") and zl.op == "call_function"
            and getattr(zl.target, "__name__", "") == "zeros_like")


def _maybe_fuse_compound_activation(gm) -> None:
    """If the entire forward graph matches a known compound activation,
    rewrite the FX graph in place: remove the multi-op subgraph and
    replace it with a single call_function to one of the
    `_agents_compound_*` sentinels. Caller's per-node loop then emits
    a single IR op for it.

    No-op when the graph doesn't match — full networks (DroNet,
    MobileNet, ...) fall through to the standard per-node handling."""
    nodes = list(gm.graph.nodes)
    placeholders = [n for n in nodes if n.op == "placeholder"]
    outputs = [n for n in nodes if n.op == "output"]
    if len(placeholders) != 1 or len(outputs) != 1:
        return
    x = placeholders[0]
    out_node = outputs[0]
    out_arg = out_node.args[0]
    if isinstance(out_arg, (tuple, list)):
        return

    if _match_swish(out_arg, x):
        sentinel = _agents_compound_swish
    elif _match_softsign(out_arg, x):
        sentinel = _agents_compound_softsign
    elif _match_gelu_exact(out_arg, x):
        sentinel = _agents_compound_gelu_exact
    elif _match_l1_norm(out_arg, x):
        sentinel = _agents_compound_l1_norm
    elif _match_l2_norm(out_arg, x):
        sentinel = _agents_compound_l2_norm
    elif _match_frobenius_norm(out_arg, x):
        sentinel = _agents_compound_frobenius_norm
    elif (_rms_eps := _match_rms_norm(out_arg, x)) is not None:
        sentinel = _agents_compound_rms_norm
    elif _match_mean_abs_norm(out_arg, x):
        sentinel = _agents_compound_mean_abs_norm
    elif _match_exclusive_cumsum(out_arg, x):
        sentinel = _agents_compound_excl_cumsum
    else:
        return

    # Rewrite: insert a fused sentinel node, retarget the output, drop
    # everything else via dead-code elimination.
    with gm.graph.inserting_before(out_node):
        new_node = gm.graph.call_function(sentinel, args=(x,))
    # Copy tensor_meta onto the new node. Most compounds are shape-preserving
    # (copy the input's meta); exclusive_cumsum reshapes, so copy the original
    # output node's meta instead.
    if sentinel is _agents_compound_excl_cumsum:
        if "tensor_meta" in out_arg.meta:
            new_node.meta["tensor_meta"] = out_arg.meta["tensor_meta"]
    elif "tensor_meta" in x.meta:
        new_node.meta["tensor_meta"] = x.meta["tensor_meta"]
    if sentinel is _agents_compound_rms_norm:
        new_node.meta["rms_eps"] = _rms_eps
    out_node.args = (new_node,)
    gm.graph.eliminate_dead_code()
    gm.recompile()


# Sentinel call-targets used to mark compound-activation subgraphs that
# we rewrite into a single FX node before the per-node IR-emit loop.
# Each accepts a single tensor and returns it unchanged (the FX
# rewriter never actually invokes them — it only records them as the
# `target` of a fused node so the call_function branch can detect
# them by identity). See _maybe_fuse_compound_activation below.
def _agents_compound_swish(x):
    return x


def _agents_compound_softsign(x):
    return x


def _agents_compound_gelu_exact(x):
    return x


def _agents_compound_l1_norm(x):
    return x


def _agents_compound_l2_norm(x):
    return x


def _agents_compound_frobenius_norm(x):
    return x


def _agents_compound_rms_norm(x):
    return x


def _agents_compound_mean_abs_norm(x):
    return x


def _agents_compound_excl_cumsum(x):
    return x


SUPPORTED_MODULES = (
    torch.nn.Linear,
    torch.nn.ReLU,
    torch.nn.ReLU6,        # MobileNetV2 activations — clamped at 6
    # KernelBench Phase 2 activations (module surfaces).
    torch.nn.LeakyReLU,
    torch.nn.Tanh,
    torch.nn.GELU,
    torch.nn.SELU,
    torch.nn.Hardsigmoid,
    torch.nn.Softplus,
    torch.nn.Softsign,
    torch.nn.Hardtanh,
    torch.nn.ELU,
    torch.nn.Conv2d,
    torch.nn.ConvTranspose2d,  # transposed / fractionally-strided 2D conv
    torch.nn.Conv1d,  # mapped to conv2d with a unit height dim
    torch.nn.ConvTranspose1d,  # mapped to conv_transpose2d with unit height
    torch.nn.Conv3d,  # 3D conv (NCDHW)
    torch.nn.ConvTranspose3d,  # 3D transposed conv
    torch.nn.MaxPool2d,
    torch.nn.MaxPool1d,  # mapped to maxpool2d with a unit height dim
    torch.nn.MaxPool3d,  # 3D max pool
    torch.nn.AvgPool2d,  # fixed-window 2D average pool (KernelBench 45)
    torch.nn.AvgPool1d,  # mapped to avgpool2d with a unit height dim
    torch.nn.AvgPool3d,  # 3D average pool
    # RMSNorm only exists in newer torch; tolerate its absence.
    *((torch.nn.RMSNorm,) if hasattr(torch.nn, "RMSNorm") else ()),
    torch.nn.AdaptiveAvgPool2d,  # global avg pool head used by classifiers
    torch.nn.Dropout,  # eval-mode no-op; we still record a passthrough alias
    torch.nn.BatchNorm2d,  # pre-folded into a per-channel scale + bias
    torch.nn.LayerNorm,  # normalize over trailing normalized_shape dims
    torch.nn.GroupNorm,  # per-(sample,group) normalize; InstanceNorm is G==C
    torch.nn.InstanceNorm2d,  # per-(sample,channel) spatial normalize
    torch.nn.TripletMarginLoss,  # 3-input margin loss → scalar
    torch.nn.Sigmoid,
    # YOLOv8 backbone uses SiLU activation throughout; neck uses Upsample.
    torch.nn.SiLU,
    torch.nn.Upsample,
    torch.nn.LSTM,   # decomposed into per-layer lstm ops (fp32 + int8 paths)
)


def _pair(v) -> tuple[int, int]:
    """Coerce int or 2-tuple to a (h, w) pair."""
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _as_tuple1(v) -> tuple[int]:
    """Coerce int or 1-tuple (the 1D nn.Conv1d/Pool1d param form) to (x,)."""
    if isinstance(v, (tuple, list)):
        return (int(v[0]),)
    return (int(v),)


def _triple(v) -> tuple[int, int, int]:
    """Coerce int or 3-tuple to a (d, h, w) triple (nn.Conv3d/Pool3d params)."""
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1]), int(v[2])
    return int(v), int(v), int(v)


def _tensor_meta(node: torch.fx.Node) -> dict[str, Any]:
    tm = node.meta.get("tensor_meta")
    if tm is None:
        raise RuntimeError(f"missing tensor_meta on node {node.name}; ShapeProp failed")
    shape = list(tm.shape)
    dtype = {torch.float32: "f32", torch.float16: "f16", torch.int8: "i8",
             torch.int32: "i32", torch.int64: "i64"}.get(tm.dtype)
    if dtype is None:
        raise RuntimeError(f"unsupported dtype {tm.dtype} on {node.name}")
    return {"shape": shape, "dtype": dtype, "quant": None}


# ---------------------------------------------------------------------------
# int8 PTQ helpers (per-tensor symmetric)
# ---------------------------------------------------------------------------

# Symmetric int8 has 127 positive levels (we deliberately give up the -128
# slot so multiplier math doesn't need to handle the asymmetric range).
_INT8_RANGE = 127.0


def _scale_from_max_abs(t: torch.Tensor) -> float:
    """Per-tensor symmetric scale: maps [-max_abs, max_abs] onto [-127, 127]."""
    m = float(t.detach().abs().max().item())
    return max(m, 1e-8) / _INT8_RANGE


def _quantize_per_tensor_sym(t: torch.Tensor, scale: float) -> np.ndarray:
    q = torch.round(t.detach() / scale).clamp(-127, 127).to(torch.int8)
    return q.cpu().numpy()


def _requantize_int(acc: np.ndarray, multiplier: int, shift: int) -> np.ndarray:
    """Bit-exact Python mirror of the requantize step in kernel_linear_s8.

    Implements `(acc * multiplier + (1<<30)) >> 31`, then a positive arithmetic
    right shift with rounding (or a left shift if `shift` is negative).
    Operates in int64 to avoid overflow.
    """
    acc64 = acc.astype(np.int64)
    prod = acc64 * np.int64(multiplier)
    # Round-to-nearest (ties to +inf, matching the kernel's `+ (1<<30)` term).
    prod = (prod + (1 << 30)) >> 31
    if shift > 0:
        round_term = 1 << (shift - 1)
        return (prod.astype(np.int32) + round_term) >> shift
    else:
        return prod.astype(np.int32) << (-shift)


# Targets where the NEW extended-fusion ops (conv2d_silu_s8, conv2d_pool_s8)
# are safe to fire today. Neither op has ANY curated/vectorized kernel yet
# on ANY backend (both are new, first-cut -- see their KernelSpecs in
# reference_kernels.py) -- so firing them always costs the acceleration the
# UNFUSED ops already have there (conv2d_s8's curated RVV/gemmini kernel,
# silu_s8's / maxpool2d_s8's own curated kernels), in exchange for one
# fewer dispatch + no intermediate DRAM round trip. On RVV that trade is a
# clear net loss today: dronet's best-validated RVV config (7,496,398
# cycles) uses a tuned conv2d_s8 AND a 3.44x-optimised maxpool2d_s8: fusing
# them into a reference-only conv2d_pool_s8 would silently regress it.
#
# This list USED to hold {gemmini, gemmini_q31, gemmini_q31_rvv}, on the
# stated assumption that "on Gemmini the trade is directionally the
# intended one" because Gemmini's accumulator drains straight into
# activation+pool hardware. That assumption has now been FALSIFIED by
# FPGA measurement on every entry it contained:
#
#   conv2d_pool_s8  dronet   gemmini_q31       14,147,329 ->    110,818,876   7.84x SLOWER
#   conv2d_silu_s8  yolov8n  gemmini_q31      315,521,575 -> 12,819,955,173  40.6x SLOWER
#   conv2d_pool_s8  dronet   gemmini_q31_rvv    2,478,782 ->    105,675,367  42.6x SLOWER
#
# All three are numerically exact (max_abs_err=0) -- they are pure
# performance losses. Two distinct causes, both structural:
#
#   1. conv2d_silu_s8 has NO curated kernel on ANY backend (its only
#      algorithm is `direct`, the scalar reference), so fusing always
#      discards a curated conv2d_s8 and lands on scalar.
#   2. conv2d_pool_s8's gemmini_tiled_conv_pool is affinity-tagged
#      ('gemmini','gemmini_q31') only -- NOT gemmini_q31_rvv -- so on that
#      target it also falls all the way back to the scalar reference,
#      costing 97% of the model in a single op.
#
# Fusion here is a PREREQUISITE for a win, not a win: it only pays once a
# curated fused kernel exists AND is affinity-tagged for the target. So the
# list is now EMPTY and extended fusion fires nowhere. The --enable-fusion
# flag and all the machinery stay in place; re-add a target here only
# alongside an FPGA measurement showing it beats the unfused path.
#
# extract_graph's IR is otherwise target-independent by design (generated/
# holds target-indep IR; generated/<target>/ holds the target-specific
# code) -- this allowlist is a deliberate, narrow exception scoped to ONLY
# the extended-fusion opt-in path, not a general target-coupling of the IR.
_FUSION_SAFE_TARGETS: frozenset = frozenset()  # see above: every prior
# entry was measured as a 7.8x-42.6x regression. Re-add only with an FPGA
# measurement showing the fused path beats the unfused one.


def _fusion_target_is_safe(fusion_target: "str | None") -> bool:
    """True if extended fusion (conv2d_silu_s8 / conv2d_pool_s8) is allowed
    to fire for the given eventual build target. Fails CLOSED: an unknown
    or unspecified target (fusion_target=None, e.g. a caller that hasn't
    been updated to pass --fusion-target) is treated as unsafe, since the
    risk is a SILENT regression on exactly the targets (rvv, scalar) that
    already have good curated kernels for the unfused ops."""
    return fusion_target in _FUSION_SAFE_TARGETS


def _sim_conv2d_int32_acc(in_4d: np.ndarray, w_q: np.ndarray, b_q: np.ndarray,
                          sh: dict, input_offset: int, filter_offset: int
                          ) -> np.ndarray:
    """Direct sliding-window int32 conv accumulate — the pre-requantize
    stage shared by the conv2d_s8/conv2d_s8_pc and conv2d_silu_s8 IR
    simulator branches (extract_int8's numpy golden). Slow but bit-exact
    against kernel_conv2d_s8 / kernel_conv2d_silu_s8's own accumulate loop
    (both reference C impls, and every algorithm candidate, use the same
    acc = bias + sum((in + input_offset) * (w + filter_offset)) order).
    """
    OH, OW = sh["OH"], sh["OW"]
    KH, KW = sh["KH"], sh["KW"]
    SH, SW = sh["SH"], sh["SW"]
    PH, PW = sh["PH"], sh["PW"]
    out = np.zeros((sh["N"], sh["OC"], OH, OW), dtype=np.int32)
    for n in range(sh["N"]):
        for oc in range(sh["OC"]):
            out[n, oc] = b_q[oc]
            for ic in range(sh["IC"]):
                for kh in range(KH):
                    for kw in range(KW):
                        ih_start = -PH + kh
                        iw_start = -PW + kw
                        for oh in range(OH):
                            ih = oh * SH + ih_start
                            if ih < 0 or ih >= sh["IH"]:
                                in_row = np.full(OW, input_offset, dtype=np.int32)
                            else:
                                in_row = np.zeros(OW, dtype=np.int32)
                                for ow in range(OW):
                                    iw = ow * SW + iw_start
                                    if iw < 0 or iw >= sh["IW"]:
                                        in_row[ow] = input_offset
                                    else:
                                        in_row[ow] = in_4d[n, ic, ih, iw] + input_offset
                            w_v = w_q[oc, ic, kh, kw] + filter_offset
                            out[n, oc, oh] += in_row * w_v
    return out


def _requantize_multiplier_shift(real_mult: float) -> tuple[int, int]:
    """Decompose `real_mult` (typically < 1) into (Q0.31 multiplier, shift).

    Convention matches CMSIS-NN / muRISCV-NN: the kernel computes
        acc = (int32_t)(((int64_t)acc * multiplier + (1 << 30)) >> 31);
        acc = (acc + (1 << (shift - 1))) >> shift;   if shift > 0
        acc = acc << -shift;                         if shift < 0
    Multiplier is in [2^30, 2^31), shift adjusts the binary point.
    """
    if real_mult <= 0.0:
        return 0, 0
    # Decompose into mantissa in [0.5, 1.0) and integer exponent.
    mantissa, exp = np.frexp(real_mult)
    multiplier = int(round(mantissa * (1 << 31)))
    if multiplier == (1 << 31):
        multiplier //= 2
        exp += 1
    shift = -exp  # positive shift = right shift after the Q0.31 multiply
    if multiplier > 0x7FFFFFFF:
        multiplier = 0x7FFFFFFF
    return int(multiplier), int(shift)


# ---------------------------------------------------------------------------
# int8 golden-simulator primitives. These are the single source of truth for
# the per-op integer math used both by the standalone op branches and by the
# fused conv2d_batchnorm2d(_silu)_s8 branches — keeping them identical is what
# guarantees a fused op's golden is bit-exact with the unfused conv+bn(+silu)
# it replaces (and hence with the composed C reference kernel).
# ---------------------------------------------------------------------------

def _sim_conv2d_s8(in_arr, sh, q, w_q, b_q):
    """int8 conv2d + Q0.31 requantize (direct sliding window). Weights are
    OIHW ([OC, IC, KH, KW]) — the raw pre-pack layout stored in the blob."""
    w_q = w_q.astype(np.int32)
    b_q = b_q.astype(np.int32)
    in_4d = in_arr.reshape(sh["N"], sh["IC"], sh["IH"], sh["IW"]).astype(np.int32)
    OH, OW = sh["OH"], sh["OW"]
    KH, KW = sh["KH"], sh["KW"]
    SH, SW = sh["SH"], sh["SW"]
    PH, PW = sh["PH"], sh["PW"]
    out = np.zeros((sh["N"], sh["OC"], OH, OW), dtype=np.int32)
    for n in range(sh["N"]):
        for oc in range(sh["OC"]):
            out[n, oc] = b_q[oc]
            for ic in range(sh["IC"]):
                for kh in range(KH):
                    for kw in range(KW):
                        ih_start = -PH + kh
                        iw_start = -PW + kw
                        for oh in range(OH):
                            ih = oh * SH + ih_start
                            if ih < 0 or ih >= sh["IH"]:
                                in_row = np.full(OW, q["input_offset"], dtype=np.int32)
                            else:
                                in_row = np.zeros(OW, dtype=np.int32)
                                for ow in range(OW):
                                    iw = ow * SW + iw_start
                                    if iw < 0 or iw >= sh["IW"]:
                                        in_row[ow] = q["input_offset"]
                                    else:
                                        in_row[ow] = in_4d[n, ic, ih, iw] + q["input_offset"]
                            w_v = w_q[oc, ic, kh, kw] + q["filter_offset"]
                            out[n, oc, oh] += in_row * w_v
    scaled = _requantize_int(out, q["output_multiplier"], q["output_shift"])
    scaled += q["output_offset"]
    scaled = np.clip(scaled, q["activation_min"], q["activation_max"])
    return scaled.astype(np.int8)


def _sim_batchnorm2d_s8(in_arr, sh, q, scale_pc, bias_pc):
    """Per-channel BN affine on int8: dequant → gamma*x+beta → requant+clamp."""
    scale_pc = scale_pc.astype(np.float32)
    bias_pc = bias_pc.astype(np.float32)
    in_4d = in_arr.reshape(sh["N"], sh["C"], sh["H"], sh["W"]).astype(np.float32)
    scale_in = np.float32(q["scale_in"])
    scale_out = np.float32(q["scale_out"])
    fv = in_4d * scale_in
    y = scale_pc[None, :, None, None] * fv + bias_pc[None, :, None, None]
    v = np.round(y / scale_out).astype(np.int32)
    v = np.clip(v, q["activation_min"], q["activation_max"])
    return v.astype(np.int8)


def _sim_silu_s8(in_arr, q):
    """int8 SiLU: dequant → x*sigmoid(x) → requant+clamp."""
    fv = in_arr.astype(np.float32) * np.float32(q["scale_in"])
    silu_out = fv / (np.float32(1.0) + np.exp(-fv).astype(np.float32))
    v = np.round(silu_out.astype(np.float32) / np.float32(q["scale_out"])).astype(np.int32)
    v = np.clip(v, q["activation_min"], q["activation_max"])
    return v.astype(np.int8)


class _CaptureTensors(torch.fx.Interpreter):
    """FX Interpreter that records every tensor produced by every node."""
    def __init__(self, gm):
        super().__init__(gm)
        self.tensors: dict[str, torch.Tensor] = {}

    def run_node(self, n):
        result = super().run_node(n)
        if isinstance(result, torch.Tensor):
            self.tensors[n.name] = result.detach().clone()
        return result


def _promote_ops_to_fp16(ops, fp16_names, tensors_meta, weights_blob,
                         fp32_stash) -> list[str]:
    """Mixed precision: turn every op whose IR name is in `fp16_names` from its
    int8 (`*_s8`) kind into the fp16 (`*_f16`) kind, re-store its weights as
    float16 from the fp32 stash, and mark its output (and, for an LSTM, its
    recurrent state) as dtype f16. Everything else stays int8; the int8<->fp16
    boundaries are materialized afterwards by `_insert_casts_i8_f16`.

    Why this exists: the fused sensor net's compute bulk is its int8 encoders,
    but its accuracy is dominated by the recurrent tail. A per-tensor int8
    3-layer LSTM loses the answer; the same tail in fp16 costs ~5% of the
    dispatch time. So precision is chosen per op, not per model.

    Returns the names of the ops that were actually promoted (so a spec naming
    an op that is not in the graph is reported rather than silently ignored).
    """
    promoted: list[str] = []
    for op in ops:
        if op["name"] not in fp16_names:
            op.setdefault("precision", "int8")
            continue
        op["precision"] = "fp16"
        if op["op"].endswith("_s8"):
            op["op"] = op["op"][:-3] + "_f16"
        promoted.append(op["name"])

        if op["op"] == "lstm_f16":
            # The int8 cell carries separate int32-quantized b_ih / b_hh in the
            # accumulator domain; the fp16 cell takes ONE float bias, so fold
            # them here (PyTorch adds both anyway).
            b_key, bh_key = op.get("bias"), op.get("bias_hh")
            if b_key and b_key in fp32_stash:
                folded = fp32_stash[b_key].astype(np.float32)
                if bh_key and bh_key in fp32_stash:
                    folded = folded + fp32_stash[bh_key].astype(np.float32)
                weights_blob[b_key] = folded.astype(np.float16)
            op.pop("bias_hh", None)
            for wk in ("weight", "weight_hh"):
                key = op.get(wk)
                if key and key in fp32_stash:
                    weights_blob[key] = fp32_stash[key].astype(np.float16)
        else:
            for wk in ("weight", "bias", "weight_hh", "bias_hh"):
                key = op.get(wk)
                if key and key in fp32_stash:
                    weights_blob[key] = fp32_stash[key].astype(np.float16)

        # Outputs and recurrent state become f16 surfaces.
        touched = list(op.get("outputs", []))
        st = op.get("state")
        if isinstance(st, dict):
            touched += [v for v in st.values() if v]
        elif isinstance(st, (list, tuple)):
            touched += [v for v in st if v]
        for t in touched:
            if t in tensors_meta:
                meta = dict(tensors_meta[t])
                meta["dtype"] = "f16"
                meta["quant"] = None
                tensors_meta[t] = meta
    return promoted


def _insert_casts_i8_f16(ops, tensors_meta, scales) -> list[dict]:
    """Materialize `cast_i8_to_f16` / `cast_f16_to_i8` ops at int8<->fp16 tensor
    boundaries. Same pass as `_ExportWalker.insert_casts` in
    extract_graph_export.py, applied to the FX int8 IR -- mixed precision is a
    property of the assembled graph, so each op's emit code stays
    single-precision.
    """
    def consumer_dtype(op) -> str:
        p = op.get("precision")
        if p is None:
            p = "fp16" if op["op"].endswith("_f16") else "int8"
        return "f16" if p == "fp16" else "i8"

    new_ops: list[dict] = []
    cast_intermediates: dict[tuple[str, str], str] = {}
    for op in ops:
        dst = consumer_dtype(op)
        for i, in_name in enumerate(op.get("inputs", [])):
            meta = tensors_meta.get(in_name)
            if meta is None:
                continue
            src = meta.get("dtype", "i8")
            if src == dst or (src, dst) not in (("i8", "f16"), ("f16", "i8")):
                continue
            key = (in_name, dst)
            cast_out = cast_intermediates.get(key)
            if cast_out is None:
                cast_out = f"{in_name}__cast_{dst}"
                scale = scales.get(in_name, 1e-8)
                kind = ("cast_i8_to_f16" if (src == "i8" and dst == "f16")
                        else "cast_f16_to_i8")
                tensors_meta[cast_out] = {
                    "shape": list(meta["shape"]),
                    "dtype": dst,
                    "quant": (None if dst == "f16" else
                              {"scale": float(scale), "zero_point": 0}),
                }
                n = 1
                for d in meta["shape"]:
                    n *= int(d)
                new_ops.append({
                    "name": cast_out,
                    "op": kind,
                    "precision": "fp16" if dst == "f16" else "int8",
                    "inputs": [in_name],
                    "outputs": [cast_out],
                    "shape": {"n": n},
                    "quant": {"scale": float(scale), "zero_point": 0},
                })
                cast_intermediates[key] = cast_out
            op["inputs"][i] = cast_out
        new_ops.append(op)
    return new_ops


def _requantize_int_per_oc(acc, mult_arr, shift_arr, oc_axis):
    """Per-output-channel requantize: apply the scalar _requantize_int with each
    channel's own (multiplier, shift) along `oc_axis`."""
    acc = acc.astype(np.int64)
    out = np.empty(acc.shape, dtype=np.int32)
    for oc in range(acc.shape[oc_axis]):
        sl = [slice(None)] * acc.ndim
        sl[oc_axis] = oc
        sl = tuple(sl)
        out[sl] = _requantize_int(acc[sl], int(mult_arr[oc]), int(shift_arr[oc]))
    return out


def _apply_per_channel(ops, tensors_meta, weights_blob, fp32_stash, scales,
                       skip_names):
    """Re-quantize conv2d_s8 / linear_s8 weights per-OUTPUT-CHANNEL (tighter than
    one per-tensor scale) and switch the op to its _pc kind with per-oc
    multiplier/shift arrays. Ops in `skip_names` (e.g. fp16-promoted) are left
    alone. Mirrors extract_graph_export's per-channel path on the FX IR."""
    for op in ops:
        if op["name"] in skip_names or op["op"] not in ("conv2d_s8", "linear_s8"):
            continue
        wk = op.get("weight")
        w_fp32 = fp32_stash.get(wk)
        if w_fp32 is None:
            continue
        OF = int(w_fp32.shape[0])
        per_oc_max = np.abs(w_fp32).reshape(OF, -1).max(axis=1)
        w_scales = np.maximum(per_oc_max, 1e-8) / 127.0
        w_flat = w_fp32.reshape(OF, -1)
        w_q = np.round(w_flat / w_scales[:, None]).clip(-127, 127).astype(np.int8)
        weights_blob[wk] = w_q.reshape(w_fp32.shape)
        in_scale = float(scales[op["inputs"][0]])
        out_scale = float(scales[op["outputs"][0]])
        bk = op.get("bias")
        if bk and bk in fp32_stash:
            b_fp32 = fp32_stash[bk]
            weights_blob[bk] = np.round(
                b_fp32 / (in_scale * w_scales)).astype(np.int32)
        mult = np.zeros(OF, np.int32); shift = np.zeros(OF, np.int32)
        for oc in range(OF):
            m, s = _requantize_multiplier_shift(
                (in_scale * float(w_scales[oc])) / max(out_scale, 1e-30))
            mult[oc] = m; shift[oc] = s
        mk = f"{op['name']}.output_multiplier_per_oc"
        sk = f"{op['name']}.output_shift_per_oc"
        weights_blob[mk] = mult; weights_blob[sk] = shift
        op["op"] = op["op"] + "_pc"
        q = op["quant"]
        q.pop("output_multiplier", None); q.pop("output_shift", None)
        q["output_multiplier_per_oc_key"] = mk
        q["output_shift_per_oc_key"] = sk


def _get_submodule(root: torch.nn.Module, qualname: str) -> torch.nn.Module:
    obj = root
    for p in qualname.split("."):
        obj = getattr(obj, p)
    return obj


def _set_submodule(root: torch.nn.Module, qualname: str, module: torch.nn.Module) -> None:
    *path, last = qualname.split(".")
    obj = root
    for p in path:
        obj = getattr(obj, p)
    setattr(obj, last, module)


def _fold_conv_bn(gm: "torch.fx.GraphModule") -> list[str]:
    """Graph-level BatchNorm folding: absorb a BatchNorm2d directly
    consuming a Conv2d's output into that conv's weight and bias, in
    float, BEFORE any quantization happens. Mirrors the canonical
    ``torch.nn.utils.fusion.fuse_conv_bn_eval`` (same math the batchnorm2d_s8
    emitter already used to compute its fp32 scale/bias — this just moves
    it upstream of quantization instead of leaving BN as its own op).

    Removing the op is worth doing for every backend: it's a real N×C×H×W
    pass eliminated, and on Gemmini batchnorm2d_s8 can't even be made
    exact (its bias has to land after the multiply; Gemmini's accumulator
    only supports bias before the single mvout scale, so reconciling needs
    a second independent rounding on the bias term — the double-rounding
    failure already proven fatal for conv2d_s8, just relocated).

    Only folds the exact pattern where it is provably lossless:
      * the BN node's sole input is a call_module Conv2d node, AND
      * that conv's output has no OTHER consumer (folding would silently
        change the un-normalized value seen by any other reader), AND
      * the BN is in inference mode with real running stats.
    This deliberately does NOT chase BN-before-conv (pre-activation)
    patterns, or fold across an intervening activation — folding across a
    nonlinearity changes the numerics, not just relocates them. DroNet's
    residual blocks are ``BN -> ReLU -> Conv``, so only the *second* BN in
    each block (which sits directly after a conv, no activation between)
    matches; the first (whose producer is a maxpool/add, and whose output
    feeds a ReLU before the next conv) is correctly left as a standalone
    batchnorm2d_s8 op.
    """
    import torch.nn.utils.fusion as _fusion

    folded: list[str] = []
    for node in list(gm.graph.nodes):
        if node.op != "call_module":
            continue
        bn_mod = _get_submodule(gm, node.target)
        if not isinstance(bn_mod, torch.nn.BatchNorm2d):
            continue
        if bn_mod.running_mean is None or bn_mod.running_var is None:
            continue  # track_running_stats=False: no fixed affine to fold
        if len(node.args) != 1 or not isinstance(node.args[0], torch.fx.Node):
            continue
        prod = node.args[0]
        if prod.op != "call_module":
            continue
        conv_mod = _get_submodule(gm, prod.target)
        if not isinstance(conv_mod, torch.nn.Conv2d):
            continue
        if len(prod.users) != 1:
            # conv output feeds something besides this BN (e.g. a residual
            # branch) — folding would change what that other reader sees.
            continue
        fused_conv = _fusion.fuse_conv_bn_eval(conv_mod, bn_mod)
        _set_submodule(gm, prod.target, fused_conv)
        node.replace_all_uses_with(prod)
        gm.graph.erase_node(node)
        folded.append(node.name)

    if folded:
        gm.graph.lint()
        gm.recompile()
    return folded


def extract_int8(
    model: torch.nn.Module,
    sample_input: "torch.Tensor | list[torch.Tensor] | tuple",
    name: str,
    out_dir: str,
    calibration_samples: "list | None" = None,
    input_dtypes: "list[str] | None" = None,
    fp16_op_names: "set[str] | None" = None,
    per_channel: bool = False,
    enable_fusion: bool = False,
    fusion_target: "str | None" = None,
    fold_conv_bn: bool = True,
) -> dict[str, Any]:
    """int8 PTQ extractor.

    Approach (intentionally minimal first cut):
      * Per-tensor symmetric quant for both weights and activations
        (zero_point = 0 throughout).
      * Activation scales calibrated from a forward pass on
        `sample_input` (for the IR's tensor shapes + io.npz golden).
        When ``calibration_samples`` is provided, per-tensor activation
        max-abs is aggregated across all of them so the int8 scale of
        each tensor reflects the worst-case dynamic range over the
        whole calibration set (not just the io-pinned single sample).
      * Fuses `linear -> relu` into a single `linear_s8` op with
        `activation_min = 0` (the relu becomes a clamp inside the requantize
        tail). Standalone relu nodes get an explicit `relu_s8` op.

    `enable_fusion` (default False) is the opt-in switch for *extended*
    fusion beyond that always-on relu absorption + graph-level bn->conv
    folding. Today it gates one pattern: conv2d -> silu absorption into a
    single `conv2d_silu_s8` op (mirrors the always-on conv2d -> relu
    absorption, but SiLU needs its own op+kernel since it isn't a plain
    clamp). With the flag off, a conv2d immediately followed by nn.SiLU
    still emits as two ops (conv2d_s8 + silu_s8, byte-identical to the
    pre-flag behavior) — both configurations stay reachable so the two can
    be measured against each other.

    `fusion_target` (default None) names the eventual build target this IR
    is headed for (e.g. "rvv", "gemmini_q31") -- extract_graph's IR is
    otherwise target-independent, but the extended-fusion ops have no
    curated kernel on ANY backend yet, so firing them on a target that
    already has a fast curated kernel for the UNFUSED ops (rvv, scalar)
    would be a silent regression. See _FUSION_SAFE_TARGETS. Passing None
    with enable_fusion=True fails closed (no extended fusion fires) rather
    than assuming safety.

    Supported ops in this first cut: nn.Linear, nn.ReLU, torch.relu.
    Any other op kind raises — extend this function as more ops gain int8
    kernels.
    """
    os.makedirs(out_dir, exist_ok=True)
    model = model.eval()

    # Normalise to a list of input tensors. A multi-input model (the fused
    # sensor net: front_grey, tof_cross, lowdim) passes a tuple/list in
    # placeholder order; a single-input model still passes a bare tensor.
    if isinstance(sample_input, (list, tuple)):
        sample_inputs = list(sample_input)
    else:
        sample_inputs = [sample_input]

    gm = torch.fx.symbolic_trace(model)
    # Graph-level BN folding, BEFORE ShapeProp/calibration/quantization —
    # a folded model has no batchnorm2d nodes at all, so everything
    # downstream (activation calibration, op emission) just never sees them.
    # `fold_conv_bn` (default True) is a knob because folding removes a
    # SCHEDULING degree of freedom on a heterogeneous SoC: an unfused
    # batchnorm2d is its own dispatch and can be placed on whichever core is
    # good at it, whereas conv+bn is indivisible and must run wholly on one.
    #
    # That DOF turns out to be worthless in practice. Controlled A/B on
    # yolov8_nano, both arms on ONE AWS F2 bitstream
    # (f2_dual_small_norose_tacit_q31_60mhz) with one curated kernel set
    # (2026-08-28, kernel_opt_log id bnfold-ab-VERDICT):
    #
    #     folding ON   155 dispatches  best-per-dispatch 67.61 ms  floor 65.97 ms
    #     folding OFF  212 dispatches  best-per-dispatch 68.94 ms  floor 67.25 ms
    #     two-core specialisation gain 2.44x -> 2.41x (unchanged)
    #
    # i.e. unfolding is ~2% WORSE, not better. Folding is essentially free
    # (it is absorbed into the conv's weights and requant scale — conv2d_s8
    # best-per-dispatch only moves 57.76 -> 57.89 ms), so unfolding just adds
    # 1.21 ms of batchnorm2d_s8 and buys no placement freedom: the scheduler
    # puts all 57 BN dispatches on the Saturn core, the same core that already
    # takes all 57 silu_s8 and every other elementwise op. The RVV arm also
    # loses bit-exactness (max_abs_err 0 -> 1 LSB) to the curated BN kernel.
    #
    # An earlier note here claimed 204->155 dispatches and 50.43->66.77 ms
    # (32% worse) with specialisation collapsing 6.71x->2.48x. That comparison
    # was confounded — the two profiles came from different bitstreams
    # (firesim_rocket_saturn vs F2 SatGemDualSmall) AND different graphs — and
    # it does not survive the controlled A/B above. Keep folding ON; this flag
    # is for investigation, not a win.
    folded_bn_names = _fold_conv_bn(gm) if fold_conv_bn else []
    ShapeProp(gm).propagate(*sample_inputs)

    # Capture every node's tensor for activation calibration.
    cap = _CaptureTensors(gm)
    final = cap.run(*sample_inputs)
    # Multi-output is fine — `final` may be a tuple. The IR builder picks up
    # the actual output names from the FX `output` node below; we don't need
    # to special-case here.
    _ = final

    # Per-tensor activation max-abs, aggregated across the full
    # calibration set when one is supplied. Each extra sample widens the
    # per-tensor max-abs to its true distribution-wide bound, which is
    # what fixes the cls-logit saturation seen with single-sample
    # calibration on detection models.
    # Every graph input, in placeholder order, paired 1:1 with the samples.
    placeholder_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
    input_node_names = [n.name for n in placeholder_nodes]
    if len(input_node_names) != len(sample_inputs):
        raise RuntimeError(
            f"int8 extract: model has {len(input_node_names)} inputs "
            f"({input_node_names}) but {len(sample_inputs)} sample tensors "
            f"were given")
    input_shapes = {nm: list(si.shape)
                    for nm, si in zip(input_node_names, sample_inputs)}
    # Per-input surface dtype. Default: every input int8. A model declaring
    # get_input_dtypes() may keep one input in float ("f16"/"f32") -- the
    # fused net's lowdim vector, whose optical_flow component is ~1e5x the
    # magnitude of its quaternion components and cannot share one int8 scale.
    in_dtype_list = (list(input_dtypes) if input_dtypes is not None
                     else ["i8"] * len(input_node_names))
    if len(in_dtype_list) != len(input_node_names):
        raise RuntimeError(
            f"int8 extract: input_dtypes has {len(in_dtype_list)} entries for "
            f"{len(input_node_names)} inputs")
    input_dtype_map = dict(zip(input_node_names, in_dtype_list))

    max_abs: dict[str, float] = {}
    for nm, si in zip(input_node_names, sample_inputs):
        max_abs[nm] = float(si.detach().abs().max().item())
    for nname, t in cap.tensors.items():
        max_abs[nname] = float(t.detach().abs().max().item())

    if calibration_samples:
        extra = [s for s in calibration_samples
                 if s is not sample_input]
        for i, s in enumerate(extra):
            s_list = list(s) if isinstance(s, (list, tuple)) else [s]
            cap_i = _CaptureTensors(gm)
            cap_i.run(*s_list)
            for nm, si in zip(input_node_names, s_list):
                cur = float(si.detach().abs().max().item())
                if cur > max_abs.get(nm, 0.0):
                    max_abs[nm] = cur
            for nname, t in cap_i.tensors.items():
                cur = float(t.detach().abs().max().item())
                if cur > max_abs.get(nname, 0.0):
                    max_abs[nname] = cur
        print(f"[extract_int8] calibrated across "
              f"{1 + len(extra)} samples", flush=True)

    scales: dict[str, float] = {
        k: max(v, 1e-8) / _INT8_RANGE for k, v in max_abs.items()
    }

    tensors_meta: dict[str, dict] = {}
    weights_blob: dict[str, np.ndarray] = {}
    # fp32 copies of every quantized weight, kept so that an op promoted to
    # fp16 (mixed precision, `fp16_op_names`) can re-store its weights as
    # float16 rather than re-deriving them from the int8 blob.
    fp32_stash: dict[str, np.ndarray] = {}
    ops: list[dict] = []
    input_names: list[str] = []          # placeholder order
    output_name: str | None = None

    # Helper: register a tensor in the IR with its int8 scale + zero_point.
    def _record(nname: str, dtype: str = "i8") -> None:
        t = cap.tensors.get(nname)
        if t is None:
            # Placeholder (input) -- shape from the paired sample tensor.
            shape = input_shapes.get(nname, [])
        else:
            shape = list(t.shape)
        tensors_meta[nname] = {
            "shape": shape,
            "dtype": dtype,
            "quant": {"scale": scales[nname], "zero_point": 0},
        }

    # Default: single-output. The output handler may overwrite this.
    output_names_multi: Optional[list[str]] = None

    # Two-pass walk: first collect linear→relu, conv2d→relu, add→relu, and
    # batchnorm2d→relu fusions so the relu node is absorbed into the
    # producer's op kind. Split by producer type so the passes_applied.json
    # artifact can credit each fusion pattern separately -- that distinction
    # matters when evaluating which pattern is paying off on which workload.
    nodes = list(gm.graph.nodes)
    fused_linear_relu: set[str] = set()
    fused_conv2d_relu: set[str] = set()
    fused_add_relu: set[str] = set()
    # bn→relu fuse: dronet's pre-activation residual blocks
    # (BN → ReLU → Conv) leave the BN's output feeding a standalone ReLU
    # kernel call. Folding the ReLU into the BN's emit (activation_min=0
    # on the existing batchnorm2d_s8 op) removes one full N×C×H×W pass
    # over the activation tensor per BN — a noticeable win on dronet
    # where 6 such pairs exist.
    fused_bn_relu: set[str] = set()
    # conv→silu fuse for the int8 path (mirror of the fp32 detector).
    # Absorbs the standalone silu_s8 dispatch into the producer conv,
    # emitted as a new op_kind 'conv2d_silu_s8' so the kernel picker
    # can offer a specialized fused variant.
    fused_conv2d_silu: set[str] = set()
    # conv→maxpool fuse (extended fusion, gated by enable_fusion like
    # conv2d_silu). Absorbs a MaxPool2d that is a Conv2d's SOLE consumer
    # (and the Conv2d's ONLY node in between is nothing — i.e. the conv
    # feeds the pool directly with no intervening op) into one
    # `conv2d_pool_s8` op. Unlike SiLU, max-pool never rescales (scale_in
    # == scale_out) and commutes with the conv's own round+clamp (both
    # monotonic non-decreasing), so this is exact given the underlying
    # conv computation is exact — see CONV2D_POOL_S8's docstring in
    # reference_kernels.py. Deliberately does NOT reach through an
    # intervening (already-folded) ReLU — dronet's actual conv->maxpool
    # site has zero intervening ops, so the simple adjacent-node check
    # below covers it; extending to "conv -(folded relu)-> maxpool"
    # chains is a documented follow-up, not attempted here.
    fused_conv2d_pool: set[str] = set()
    # Extended fusion only actually fires when BOTH the opt-in flag is set
    # AND the eventual target is in the safe set (see _fusion_target_is_safe)
    # -- computed once here, used at both gate points below.
    _extended_fusion_ok = enable_fusion and _fusion_target_is_safe(fusion_target)
    for i, node in enumerate(nodes):
        if i + 1 >= len(nodes):
            continue
        nxt = nodes[i + 1]
        is_next_relu = (
            (nxt.op == "call_module"
             and isinstance(gm.get_submodule(nxt.target), torch.nn.ReLU))
            or (nxt.op == "call_function" and nxt.target in (
                torch.relu, torch.nn.functional.relu))
        )
        is_next_silu = (
            (nxt.op == "call_module"
             and isinstance(gm.get_submodule(nxt.target), torch.nn.SiLU))
            or (nxt.op == "call_function" and nxt.target in (
                torch.nn.functional.silu,))
        )
        # conv2d -> maxpool2d: separate check (independent gate, only
        # Conv2d producers considered) so it doesn't disturb the
        # relu/silu detection below at all. Handled and `continue`d here
        # rather than folded into the combined is_next_relu/is_next_silu
        # branch, since a MaxPool2d producer-type restriction (Conv2d
        # only) doesn't fit that branch's Linear/Conv2d/BatchNorm2d
        # dispatch cleanly.
        is_next_maxpool = (
            nxt.op == "call_module"
            and isinstance(gm.get_submodule(nxt.target), torch.nn.MaxPool2d)
        )
        if (_extended_fusion_ok and is_next_maxpool
                and len(nxt.args) == 1 and nxt.args[0] is node
                and node.op == "call_module"
                and isinstance(gm.get_submodule(node.target), torch.nn.Conv2d)):
            fused_conv2d_pool.add(nxt.name)
            continue
        if not ((is_next_relu or is_next_silu)
                and len(nxt.args) == 1 and nxt.args[0] is node):
            continue
        if node.op == "call_module":
            producer_mod = gm.get_submodule(node.target)
            if is_next_silu:
                # Only Conv2d→SiLU is currently considered for absorption
                # (yolov8 backbone pattern). Other (BN→SiLU, etc.) fall
                # through without modifying the relu-fold sets.
                # Gated on enable_fusion (unlike the relu absorptions
                # below, which are always on): SiLU is not a plain clamp,
                # so absorbing it needs its own op+kernel
                # (conv2d_silu_s8) rather than reusing conv2d_s8's
                # activation_min/max fields, and it's new/unproven end to
                # end. Off by default keeps the pre-existing two-op
                # (conv2d_s8 + silu_s8) behavior reachable for comparison.
                if _extended_fusion_ok and isinstance(producer_mod, torch.nn.Conv2d):
                    fused_conv2d_silu.add(nxt.name)
            elif isinstance(producer_mod, torch.nn.Linear):
                fused_linear_relu.add(nxt.name)
            elif isinstance(producer_mod, torch.nn.Conv2d):
                fused_conv2d_relu.add(nxt.name)
            elif isinstance(producer_mod, torch.nn.BatchNorm2d):
                fused_bn_relu.add(nxt.name)
        elif node.op == "call_function":
            t = node.target
            tname = getattr(t, "__name__", "")
            if (tname == "add" or t is torch.add
                    or t is __import__("operator").add):
                fused_add_relu.add(nxt.name)
    fused_relu_after: set[str] = (
        fused_linear_relu | fused_conv2d_relu | fused_add_relu | fused_bn_relu
    )

    # ------------------------------------------------------------------
    # Conv2d → BatchNorm2d (→ ReLU | SiLU) fusion detection.
    #
    # The XPU-RT schedule puts every heavy conv on the gemmini hart and
    # every elementwise glue op (batchnorm2d_s8 / silu_s8 / relu_s8) on
    # rvv; that cross-core ping-pong leaves gemmini stalled. Absorbing the
    # BN (and the trailing activation) into the producing conv collapses
    # the whole block into ONE op that runs on a single hart:
    #   * conv2d_batchnorm2d_s8       (conv→bn, and conv→bn→relu — the relu
    #                                  is folded into the bn sub-op's clamp)
    #   * conv2d_batchnorm2d_silu_s8  (conv→bn→silu)
    # Each fused op carries its constituents under `sub_ops` (the same
    # shape apply_fusion_hint.py produces), so the existing skeleton /
    # kernel-picker plumbing for those registered op-kinds handles it.
    #
    # Detection is by FX adjacency + single-consumer checks: the conv must
    # feed ONLY the immediately-following BN, and (for the activation) the
    # BN must feed ONLY the immediately-following relu/silu. This mirrors
    # the relu/silu absorb above and correctly leaves alone the dronet
    # blocks where a maxpool or residual-add sits between conv and bn.
    conv_bn_fusion: dict[str, dict] = {}
    conv_bn_consumed: set = set()
    for i, node in enumerate(nodes):
        if node.op != "call_module":
            continue
        if not isinstance(gm.get_submodule(node.target), torch.nn.Conv2d):
            continue
        if i + 1 >= len(nodes) or len(list(node.users)) != 1:
            continue
        bn_node = nodes[i + 1]
        if not (bn_node.op == "call_module"
                and isinstance(gm.get_submodule(bn_node.target),
                               torch.nn.BatchNorm2d)
                and len(bn_node.args) >= 1 and bn_node.args[0] is node):
            continue
        act_node = None
        act_kind = None
        if len(list(bn_node.users)) == 1 and i + 2 < len(nodes):
            cand = nodes[i + 2]
            cand_is_relu = (
                (cand.op == "call_module"
                 and isinstance(gm.get_submodule(cand.target), torch.nn.ReLU))
                or (cand.op == "call_function" and cand.target in (
                    torch.relu, torch.nn.functional.relu)))
            cand_is_silu = (
                (cand.op == "call_module"
                 and isinstance(gm.get_submodule(cand.target), torch.nn.SiLU))
                or (cand.op == "call_function" and cand.target in (
                    torch.nn.functional.silu,)))
            if ((cand_is_relu or cand_is_silu)
                    and len(cand.args) == 1 and cand.args[0] is bn_node):
                act_node = cand
                act_kind = "relu" if cand_is_relu else "silu"
        conv_bn_fusion[node.name] = {
            "bn": bn_node, "act": act_node, "act_kind": act_kind}
        conv_bn_consumed.add(bn_node)
        if act_node is not None:
            conv_bn_consumed.add(act_node)

    # Nodes to skip during the main walk (e.g. getitem consumers of chunk,
    # and the BN / activation nodes absorbed by conv_bn_fusion above).
    _skip_nodes: set = set(conv_bn_consumed)

    for node in nodes:
        if node in _skip_nodes:
            continue
        if node.op == "placeholder":
            input_names.append(node.name)
            _record(node.name, dtype=input_dtype_map.get(node.name, "i8"))

        elif node.op == "call_module":
            mod = gm.get_submodule(node.target)
            in_name = node.args[0].name

            if isinstance(mod, torch.nn.Linear):
                _record(node.name, dtype="i8")
                w_fp32 = mod.weight.detach()
                b_fp32 = mod.bias.detach() if mod.bias is not None else None
                w_scale = _scale_from_max_abs(w_fp32)
                w_q = _quantize_per_tensor_sym(w_fp32, w_scale)
                in_scale = scales[in_name]
                out_scale = scales[node.name]
                # bias is in scale s_in * s_w (int32 accumulator domain).
                if b_fp32 is not None:
                    b_q = torch.round(b_fp32 / (in_scale * w_scale)).to(
                        torch.int32).cpu().numpy()
                else:
                    b_q = np.zeros((mod.out_features,), dtype=np.int32)
                w_key = f"{node.target}.weight_q"
                b_key = f"{node.target}.bias_q"
                weights_blob[w_key] = w_q
                weights_blob[b_key] = b_q
                # Keep fp32 copies so promoting this linear to fp16 (mixed
                # precision) can re-store float16 weights, instead of trying to
                # invert the int8 quantization.
                fp32_stash[w_key] = w_fp32.cpu().numpy().astype(np.float32)
                fp32_stash[b_key] = (
                    b_fp32.cpu().numpy().astype(np.float32)
                    if b_fp32 is not None
                    else np.zeros((mod.out_features,), np.float32))
                # Requantize: real_mult = s_in * s_w / s_out
                real_mult = (in_scale * w_scale) / out_scale
                multiplier, shift = _requantize_multiplier_shift(real_mult)
                # If a relu follows that we'll fuse, clamp at 0 (zp=0); else
                # clamp at the int8 range.
                next_node = nodes[nodes.index(node) + 1] if nodes.index(node) + 1 < len(nodes) else None
                fuse_relu = (next_node is not None
                             and next_node.name in fused_relu_after)
                act_min = 0 if fuse_relu else -128
                act_max = 127
                in_shape = tensors_meta[in_name]["shape"]
                out_shape = tensors_meta[node.name]["shape"]
                M = int(np.prod(in_shape[:-1]))
                K = int(in_shape[-1])
                N = int(out_shape[-1])
                # If the relu is fused, the linear's output IS the relu's
                # output — record an alias so subsequent ops reading from the
                # relu name find this linear's buffer.
                ops.append({
                    "name": str(node.target),
                    "op": "linear_s8",
                    "inputs": [in_name],
                    "outputs": [
                        next_node.name if fuse_relu else node.name
                    ],
                    "weight": w_key,
                    "bias": b_key,
                    "shape": {"M": M, "K": K, "N": N},
                    "quant": {
                        "input_offset": 0,    # zp_in
                        "filter_offset": 0,   # zp_w
                        "output_offset": 0,   # zp_out
                        "output_multiplier": multiplier,
                        "output_shift": shift,
                        "activation_min": act_min,
                        "activation_max": act_max,
                    },
                })
                if fuse_relu:
                    # The relu output uses the same scale as the linear output,
                    # which is now the linear+relu output.
                    if next_node.name not in tensors_meta:
                        tensors_meta[next_node.name] = dict(tensors_meta[node.name])

            elif isinstance(mod, torch.nn.ReLU):
                if node.name in fused_relu_after:
                    continue  # absorbed
                _record(node.name, dtype="i8")
                n = int(np.prod(tensors_meta[in_name]["shape"]))
                ops.append({
                    "name": str(node.target),
                    "op": "relu_s8",
                    "inputs": [in_name],
                    "outputs": [node.name],
                    "shape": {"n": n},
                })

            elif isinstance(mod, torch.nn.ReLU6):
                # Standalone ReLU6: clamp real values to [0, 6]. Keep the
                # output at the input's scale (like relu_s8 keeps scale for
                # plain relu) and express the 6.0 ceiling in int8 units:
                # qmax = round(6 / s), saturated to [1, 127]. zp = 0.
                in_scale = scales[in_name]
                scales[node.name] = in_scale       # relu6 preserves scale
                _record(node.name, dtype="i8")
                qmax = max(1, min(127, int(round(6.0 / in_scale))))
                n = int(np.prod(tensors_meta[in_name]["shape"]))
                ops.append({
                    "name": str(node.target),
                    "op": "relu6_s8",
                    "inputs": [in_name],
                    "outputs": [node.name],
                    "shape": {"n": n},
                    "clamp_max": qmax,
                })

            elif isinstance(mod, torch.nn.Conv2d):
                if mod.groups != 1:
                    raise NotImplementedError(
                        f"int8 extract: Conv2d groups={mod.groups} not "
                        f"supported at {node.name}"
                    )
                if mod.dilation != (1, 1):
                    raise NotImplementedError(
                        f"int8 extract: Conv2d dilation={mod.dilation} not "
                        f"supported at {node.name}"
                    )

                # ---- Conv2d → BatchNorm2d (→ ReLU|SiLU) fusion --------------
                fuse = conv_bn_fusion.get(node.name)
                if fuse is not None:
                    bn_node = fuse["bn"]
                    act_node = fuse["act"]
                    act_kind = fuse["act_kind"]
                    bn_mod = gm.get_submodule(bn_node.target)

                    # -- conv sub-op (int8 conv + Q0.31 requantize, full range;
                    #    BN reads the full-range int8 conv output) --
                    w_fp32 = mod.weight.detach()
                    b_fp32 = mod.bias.detach() if mod.bias is not None else None
                    w_scale = _scale_from_max_abs(w_fp32)
                    w_q = _quantize_per_tensor_sym(w_fp32, w_scale)
                    in_scale = scales[in_name]
                    conv_out_scale = scales[node.name]
                    if b_fp32 is not None:
                        b_q = torch.round(
                            b_fp32 / (in_scale * w_scale)).to(
                            torch.int32).cpu().numpy()
                    else:
                        b_q = np.zeros((mod.out_channels,), dtype=np.int32)
                    w_key = f"{node.target}.weight_q"
                    b_key = f"{node.target}.bias_q"
                    weights_blob[w_key] = w_q
                    weights_blob[b_key] = b_q
                    real_mult = (in_scale * w_scale) / conv_out_scale
                    multiplier, shift = _requantize_multiplier_shift(real_mult)
                    in_shape = tensors_meta[in_name]["shape"]
                    conv_out_shape = list(cap.tensors[node.name].shape)
                    N_, IC, IH, IW = (int(s) for s in in_shape)
                    _, OC, OH, OW = (int(s) for s in conv_out_shape)
                    KH, KW = _pair(mod.kernel_size)
                    SH, SW = _pair(mod.stride)
                    PH, PW = _pair(mod.padding)
                    conv_sub = {
                        "name": str(node.target), "op": "conv2d_s8",
                        "inputs": [in_name], "outputs": [node.name],
                        "weight": w_key, "bias": b_key,
                        "shape": {"N": N_, "IC": IC, "IH": IH, "IW": IW,
                                  "OC": OC, "OH": OH, "OW": OW,
                                  "KH": KH, "KW": KW, "SH": SH, "SW": SW,
                                  "PH": PH, "PW": PW},
                        "quant": {
                            "input_offset": 0, "filter_offset": 0,
                            "output_offset": 0,
                            "output_multiplier": multiplier,
                            "output_shift": shift,
                            "activation_min": -128, "activation_max": 127,
                        },
                    }

                    # -- bn sub-op (per-channel affine folded to scale + bias) --
                    gamma = (bn_mod.weight.detach().cpu().numpy().astype(np.float32)
                             if bn_mod.weight is not None else
                             np.ones((bn_mod.num_features,), dtype=np.float32))
                    beta = (bn_mod.bias.detach().cpu().numpy().astype(np.float32)
                            if bn_mod.bias is not None else
                            np.zeros((bn_mod.num_features,), dtype=np.float32))
                    bn_mean = bn_mod.running_mean.detach().cpu().numpy().astype(np.float32)
                    bn_var = bn_mod.running_var.detach().cpu().numpy().astype(np.float32)
                    bn_eps = float(bn_mod.eps)
                    bn_scale_pc = (gamma / np.sqrt(bn_var + bn_eps)).astype(np.float32)
                    bn_bias_pc = (beta - bn_mean * bn_scale_pc).astype(np.float32)
                    bn_s_key = f"{bn_node.target}.scale"
                    bn_b_key = f"{bn_node.target}.bias_fused"
                    weights_blob[bn_s_key] = bn_scale_pc
                    weights_blob[bn_b_key] = bn_bias_pc
                    bn_out_scale = scales[bn_node.name]
                    # relu → clamp at 0 on the bn requantize (the relu is the
                    # bn output stage); silu/none → full int8 range.
                    bn_act_min = 0 if act_kind == "relu" else -128
                    # Where the bn writes: for relu the fused output IS the
                    # relu tensor (bn's clamp does the relu); for silu the bn
                    # feeds the silu sub-op; for none the bn IS the output.
                    if act_kind == "relu":
                        bn_out_name = act_node.name
                    else:
                        bn_out_name = bn_node.name
                    bn_sub = {
                        "name": str(bn_node.target), "op": "batchnorm2d_s8",
                        "inputs": [node.name], "outputs": [bn_out_name],
                        "weight": bn_s_key, "bias": bn_b_key,
                        "shape": {"N": N_, "C": OC, "H": OH, "W": OW},
                        "quant": {
                            "scale_in": conv_out_scale,
                            "scale_out": bn_out_scale,
                            "activation_min": bn_act_min,
                            "activation_max": 127,
                        },
                    }

                    sub_ops = [conv_sub, bn_sub]
                    if act_kind == "silu":
                        silu_out_scale = scales[act_node.name]
                        n_elem = int(N_ * OC * OH * OW)
                        sub_ops.append({
                            "name": str(getattr(act_node, "target", act_node.name)),
                            "op": "silu_s8",
                            "inputs": [bn_node.name],
                            "outputs": [act_node.name],
                            "shape": {"n": n_elem},
                            "quant": {
                                "scale_in": bn_out_scale,
                                "scale_out": silu_out_scale,
                                "activation_min": -128,
                                "activation_max": 127,
                            },
                        })
                        fused_op_kind = "conv2d_batchnorm2d_silu_s8"
                        final_out = act_node.name
                        final_scale = silu_out_scale
                    elif act_kind == "relu":
                        fused_op_kind = "conv2d_batchnorm2d_s8"
                        final_out = act_node.name
                        # relu (bn-clamp) keeps the bn output scale, matching
                        # the standalone bn→relu fusion's aliasing.
                        final_scale = bn_out_scale
                    else:
                        fused_op_kind = "conv2d_batchnorm2d_s8"
                        final_out = bn_node.name
                        final_scale = bn_out_scale

                    # Register only the fused OUTPUT tensor; the conv/bn
                    # intermediates live inside the single kernel and need no
                    # global buffer.
                    tensors_meta[final_out] = {
                        "shape": (list(cap.tensors[final_out].shape)
                                  if final_out in cap.tensors
                                  else [N_, OC, OH, OW]),
                        "dtype": "i8",
                        "quant": {"scale": final_scale, "zero_point": 0},
                    }
                    scales[final_out] = final_scale
                    ops.append({
                        "name": str(node.target),
                        "op": fused_op_kind,
                        "inputs": [in_name],
                        "outputs": [final_out],
                        "sub_ops": sub_ops,
                    })
                    continue
                # ---- end Conv2d → BatchNorm2d fusion -----------------------

                _record(node.name, dtype="i8")
                w_fp32 = mod.weight.detach()
                b_fp32 = mod.bias.detach() if mod.bias is not None else None
                w_scale = _scale_from_max_abs(w_fp32)
                w_q = _quantize_per_tensor_sym(w_fp32, w_scale)
                in_scale = scales[in_name]
                out_scale = scales[node.name]
                if b_fp32 is not None:
                    b_q = torch.round(b_fp32 / (in_scale * w_scale)).to(
                        torch.int32).cpu().numpy()
                else:
                    b_q = np.zeros((mod.out_channels,), dtype=np.int32)
                w_key = f"{node.target}.weight_q"
                b_key = f"{node.target}.bias_q"
                weights_blob[w_key] = w_q
                weights_blob[b_key] = b_q
                # fp32 copies for a possible per-channel re-quant or fp16 promotion.
                fp32_stash[w_key] = w_fp32.cpu().numpy().astype(np.float32)
                fp32_stash[b_key] = (b_fp32.cpu().numpy().astype(np.float32)
                                     if b_fp32 is not None
                                     else np.zeros((mod.out_channels,), np.float32))
                real_mult = (in_scale * w_scale) / out_scale
                multiplier, shift = _requantize_multiplier_shift(real_mult)
                next_node = nodes[nodes.index(node) + 1] if nodes.index(node) + 1 < len(nodes) else None
                fuse_relu = (next_node is not None
                             and next_node.name in fused_relu_after)
                # Conv→SiLU: only considered when fused_conv2d_silu is
                # non-empty for this site, which itself only happens when
                # enable_fusion was passed (see the two-pass scan above).
                # Emits the KernelSpec CONV2D_SILU_S8 op instead of a
                # standalone conv2d_s8, and the SiLU branch below skips its
                # own emission (output aliased to the SiLU node's name,
                # same pattern as the always-on relu absorption).
                fuse_silu = (next_node is not None
                             and next_node.name in fused_conv2d_silu)
                # Conv→MaxPool: same alias pattern as the relu absorption
                # (fused_conv2d_pool is only populated when enable_fusion
                # was passed — see the two-pass scan above). Unlike SiLU,
                # pool never rescales, so no new quant scale is needed —
                # activation_min/max still bound the PRE-pool intermediate
                # exactly as they would for a standalone conv2d_s8 (pool
                # then selects the max of already-clamped values, which
                # commutes with the clamp).
                fuse_pool = (next_node is not None
                             and next_node.name in fused_conv2d_pool)
                act_min = 0 if fuse_relu else -128
                act_max = 127
                in_shape = tensors_meta[in_name]["shape"]
                out_shape = tensors_meta[node.name]["shape"]
                N_, IC, IH, IW = (int(s) for s in in_shape)
                _, OC, OH, OW = (int(s) for s in out_shape)
                KH, KW = _pair(mod.kernel_size)
                SH, SW = _pair(mod.stride)
                PH, PW = _pair(mod.padding)
                quant = {
                    "input_offset": 0,
                    "filter_offset": 0,
                    "output_offset": 0,
                    "output_multiplier": multiplier,
                    "output_shift": shift,
                    "activation_min": act_min,
                    "activation_max": act_max,
                }
                if fuse_silu:
                    # silu_scale_in is this conv's OWN (intermediate,
                    # pre-SiLU) scale — the calibrated scale the conv would
                    # have used as a standalone op (out_scale, above).
                    # silu_scale_out is the SiLU node's own calibrated
                    # scale (a real requantize happens in the LUT, unlike
                    # the relu case where clamping doesn't rescale).
                    quant["silu_scale_in"] = out_scale
                    quant["silu_scale_out"] = scales[next_node.name]
                shape = {
                    "N": N_, "IC": IC, "IH": IH, "IW": IW,
                    "OC": OC, "OH": OH, "OW": OW,
                    "KH": KH, "KW": KW,
                    "SH": SH, "SW": SW,
                    "PH": PH, "PW": PW,
                }
                op_kind = "conv2d_s8"
                if fuse_silu:
                    op_kind = "conv2d_silu_s8"
                elif fuse_pool:
                    op_kind = "conv2d_pool_s8"
                    pool_mod = gm.get_submodule(next_node.target)
                    pKH, pKW = _pair(pool_mod.kernel_size)
                    pSH, pSW = _pair(pool_mod.stride)
                    pPH, pPW = _pair(pool_mod.padding)
                    pDH, pDW = _pair(pool_mod.dilation)
                    shape.update({
                        "pool_KH": pKH, "pool_KW": pKW,
                        "pool_SH": pSH, "pool_SW": pSW,
                        "pool_PH": pPH, "pool_PW": pPW,
                        "pool_DH": pDH, "pool_DW": pDW,
                    })
                ops.append({
                    "name": str(node.target),
                    "op": op_kind,
                    "inputs": [in_name],
                    "outputs": [
                        next_node.name if (fuse_relu or fuse_silu or fuse_pool)
                        else node.name
                    ],
                    "weight": w_key,
                    "bias": b_key,
                    "shape": shape,
                    "quant": quant,
                })
                if fuse_relu and next_node.name not in tensors_meta:
                    # Relu alias: shape is UNCHANGED by relu, so copying the
                    # conv's whole tensors_meta entry (shape + scale) is
                    # exact.
                    tensors_meta[next_node.name] = dict(tensors_meta[node.name])
                elif fuse_pool and next_node.name not in tensors_meta:
                    # Pool alias: UNLIKE relu, pool changes shape (that's
                    # the whole point) — must NOT blindly copy the conv's
                    # tensors_meta (its shape is the PRE-pool [OH,OW], not
                    # the pool's own [OHp,OWp]). _record() pulls the
                    # correct shape from cap.tensors (ShapeProp already
                    # ran maxpool2d's real forward() on the float graph,
                    # so the pool node's true output shape is there);
                    # the scale is fixed up to the conv's out_scale right
                    # below (no rescale happens in a pool, same as the
                    # standalone maxpool2d_s8 emitter's own
                    # `scales[node.name] = scales[in_name]` line).
                    _record(next_node.name, dtype="i8")
                elif fuse_silu and next_node.name not in tensors_meta:
                    # Unlike the relu alias above, SiLU's output genuinely
                    # requantizes to its own (already-calibrated) scale —
                    # record it properly rather than copying the conv's.
                    _record(next_node.name, dtype="i8")
                if fuse_pool:
                    # No rescale happens in a pool -- force the pooled
                    # tensor's scale to match the conv's out_scale exactly
                    # (mirrors the standalone maxpool2d_s8 emitter's own
                    # `scales[node.name] = scales[in_name]` +
                    # `tensors_meta[node.name]["quant"]["scale"] =
                    # scales[in_name]` pair). _record() above seeded
                    # tensors_meta[next_node.name]["quant"]["scale"] from
                    # the pool's OWN independently-calibrated scale, which
                    # should be numerically close but isn't guaranteed
                    # identical -- both must agree with what the C kernel
                    # (which does zero rescaling) actually produces.
                    scales[next_node.name] = out_scale
                    tensors_meta[next_node.name]["quant"]["scale"] = out_scale

            elif isinstance(mod, torch.nn.MaxPool2d):
                if node.name in fused_conv2d_pool:
                    continue  # absorbed into the producer's conv2d_pool_s8
                _record(node.name, dtype="i8")
                in_shape = tensors_meta[in_name]["shape"]
                out_shape = tensors_meta[node.name]["shape"]
                N_, C, IH, IW = (int(s) for s in in_shape)
                _, _, OH, OW = (int(s) for s in out_shape)
                KH, KW = _pair(mod.kernel_size)
                SH, SW = _pair(mod.stride)
                PH, PW = _pair(mod.padding)
                DH, DW = _pair(mod.dilation)
                ops.append({
                    "name": str(node.target),
                    "op": "maxpool2d_s8",
                    "inputs": [in_name],
                    "outputs": [node.name],
                    "shape": {
                        "N": N_, "C": C,
                        "IH": IH, "IW": IW,
                        "OH": OH, "OW": OW,
                        "KH": KH, "KW": KW,
                        "SH": SH, "SW": SW,
                        "PH": PH, "PW": PW,
                        "DH": DH, "DW": DW,
                    },
                })
                # MaxPool keeps the same scale as its input (no requantize) —
                # overwrite the calibrated scale to match for downstream
                # consumers that read this tensor's scale.
                tensors_meta[node.name]["quant"]["scale"] = scales[in_name]
                scales[node.name] = scales[in_name]

            elif isinstance(mod, torch.nn.Dropout):
                # Eval-mode dropout is a view; alias is handled by the
                # skeleton via op="view".
                _record(node.name, dtype="i8")
                # Same scale as input.
                tensors_meta[node.name]["quant"]["scale"] = scales[in_name]
                scales[node.name] = scales[in_name]
                n = int(np.prod(tensors_meta[in_name]["shape"]))
                ops.append({
                    "name": str(node.target),
                    "op": "view",
                    "inputs": [in_name],
                    "outputs": [node.name],
                    "shape": {"n": n},
                })

            elif isinstance(mod, torch.nn.BatchNorm2d):
                _record(node.name, dtype="i8")
                gamma = (mod.weight.detach().cpu().numpy().astype(np.float32)
                         if mod.weight is not None else
                         np.ones((mod.num_features,), dtype=np.float32))
                beta = (mod.bias.detach().cpu().numpy().astype(np.float32)
                        if mod.bias is not None else
                        np.zeros((mod.num_features,), dtype=np.float32))
                mean = mod.running_mean.detach().cpu().numpy().astype(np.float32)
                var = mod.running_var.detach().cpu().numpy().astype(np.float32)
                eps = float(mod.eps)
                bn_scale = (gamma / np.sqrt(var + eps)).astype(np.float32)
                bn_bias = (beta - mean * bn_scale).astype(np.float32)
                s_key = f"{node.target}.scale"
                b_key = f"{node.target}.bias_fused"
                weights_blob[s_key] = bn_scale
                weights_blob[b_key] = bn_bias
                in_shape = tensors_meta[in_name]["shape"]
                N_, C, H, W = (int(s) for s in in_shape)
                # Same fuse-with-following-relu pattern as conv2d_s8: if the
                # next FX node is a ReLU and was captured in the fusion
                # scan, clamp at 0 on the batchnorm op and route the output
                # tensor name to the ReLU's name. The separate ReLU kernel
                # call then becomes a no-op (its emit branch sees the alias
                # already exists and skips).
                next_node = (nodes[nodes.index(node) + 1]
                             if nodes.index(node) + 1 < len(nodes) else None)
                fuse_relu = (next_node is not None
                             and next_node.name in fused_relu_after
                             and next_node.name in fused_bn_relu)
                act_min = 0 if fuse_relu else -128
                ops.append({
                    "name": str(node.target),
                    "op": "batchnorm2d_s8",
                    "inputs": [in_name],
                    "outputs": [
                        next_node.name if fuse_relu else node.name
                    ],
                    "weight": s_key,
                    "bias": b_key,
                    "shape": {"N": N_, "C": C, "H": H, "W": W},
                    "quant": {
                        "scale_in":   scales[in_name],
                        "scale_out":  scales[node.name],
                        "activation_min": act_min,
                        "activation_max": 127,
                    },
                })
                if fuse_relu and next_node.name not in tensors_meta:
                    tensors_meta[next_node.name] = dict(tensors_meta[node.name])

            elif isinstance(mod, torch.nn.Sigmoid):
                _record(node.name, dtype="i8")
                n = int(np.prod(tensors_meta[in_name]["shape"]))
                ops.append({
                    "name": str(node.target),
                    "op": "sigmoid_s8",
                    "inputs": [in_name],
                    "outputs": [node.name],
                    "shape": {"n": n},
                    "quant": {
                        "scale_in":  scales[in_name],
                        "scale_out": scales[node.name],
                        "activation_min": -128,
                        "activation_max": 127,
                    },
                })

            elif isinstance(mod, torch.nn.SiLU):
                if node.name in fused_conv2d_silu:
                    continue  # absorbed into the producing conv2d_silu_s8
                _record(node.name, dtype="i8")
                n = int(np.prod(tensors_meta[in_name]["shape"]))
                ops.append({
                    "name": str(node.target),
                    "op": "silu_s8",
                    "inputs": [in_name],
                    "outputs": [node.name],
                    "shape": {"n": n},
                    "quant": {
                        "scale_in":  scales[in_name],
                        "scale_out": scales[node.name],
                        "activation_min": -128,
                        "activation_max": 127,
                    },
                })

            elif isinstance(mod, torch.nn.ELU):
                _record(node.name, dtype="i8")
                n = int(np.prod(tensors_meta[in_name]["shape"]))
                ops.append({
                    "name": str(node.target),
                    "op": "elu_s8",
                    "inputs": [in_name],
                    "outputs": [node.name],
                    "shape": {"n": n},
                    "quant": {
                        "scale_in":  scales[in_name],
                        "scale_out": scales[node.name],
                        "activation_min": -128,
                        "activation_max": 127,
                        "alpha": float(mod.alpha),
                    },
                })

            elif isinstance(mod, torch.nn.GELU):
                # gelu_s8 existed as a kernel with no extractor branch, so the
                # op inventory counted it as "covered" while nothing could emit
                # it. Kernel presence is not coverage.
                _record(node.name, dtype="i8")
                n = int(np.prod(tensors_meta[in_name]["shape"]))
                ops.append({
                    "name": str(node.target),
                    "op": "gelu_s8",
                    "inputs": [in_name],
                    "outputs": [node.name],
                    "shape": {"n": n},
                    "quant": {
                        "scale_in":  scales[in_name],
                        "scale_out": scales[node.name],
                        "activation_min": -128, "activation_max": 127,
                    },
                })

            elif isinstance(mod, (torch.nn.LayerNorm,
                                  getattr(torch.nn, "RMSNorm", torch.nn.LayerNorm))) \
                    and type(mod).__name__ in ("LayerNorm", "RMSNorm"):
                is_rms = type(mod).__name__ == "RMSNorm"
                _record(node.name, dtype="i8")
                in_shape = tensors_meta[in_name]["shape"]
                K = int(in_shape[-1])
                M = int(np.prod(in_shape[:-1])) if len(in_shape) > 1 else 1
                ns = mod.normalized_shape
                ns = (ns,) if isinstance(ns, int) else tuple(ns)
                if len(ns) != 1 or int(ns[0]) != K:
                    raise NotImplementedError(
                        f"int8 extract: {type(mod).__name__} at {node.name} "
                        f"normalizes over {ns}; only the last dimension is "
                        f"supported")
                base = f"{node.target}"
                g = (mod.weight.detach().cpu().numpy().astype(np.float32)
                     if getattr(mod, "weight", None) is not None
                     else np.ones((K,), np.float32))
                weights_blob[f"{base}.gamma"] = g
                op_rec = {
                    "name": base,
                    "op": "rmsnorm_s8" if is_rms else "layernorm_s8",
                    "inputs": [in_name],
                    "outputs": [node.name],
                    "weight": f"{base}.gamma",
                    "shape": {"M": M, "K": K},
                    "quant": {
                        "scale_in": scales[in_name],
                        "scale_out": scales[node.name],
                        "eps": float(getattr(mod, "eps", 1e-5) or 1e-5),
                        "activation_min": -128, "activation_max": 127,
                    },
                }
                if not is_rms:
                    b = (mod.bias.detach().cpu().numpy().astype(np.float32)
                         if getattr(mod, "bias", None) is not None
                         else np.zeros((K,), np.float32))
                    weights_blob[f"{base}.beta"] = b
                    op_rec["bias"] = f"{base}.beta"
                ops.append(op_rec)

            elif isinstance(mod, torch.nn.LSTM):
                # One `lstm_s8` op per layer. The recurrent state is carried in
                # dedicated tensors that are NOT scratch: they must survive
                # across run_model() calls, because a recurrent model's
                # invocation k depends on k-1. generate_skeleton gives any
                # tensor named `<...>.h_state` / `.c_state` a persistent,
                # zero-initialised buffer.
                if mod.batch_first:
                    raise NotImplementedError(
                        "int8 extract: LSTM batch_first=True is not supported; "
                        "the cell expects [seq, batch, feature]")
                if mod.bidirectional:
                    raise NotImplementedError(
                        "int8 extract: bidirectional LSTM is not supported")
                # nn.LSTM returns (output, (h_n, c_n)), so the FX node for the
                # module produces a TUPLE and calibration never assigned it a
                # scale. The tensor everything downstream actually uses arrives
                # via operator.getitem(node, 0); that is the name our op must
                # write, and node.name itself never becomes a buffer.
                out_node = _find_getitem_consumer(node, 0)
                out_node_name = out_node.name if out_node is not None else None
                if out_node_name is None:
                    raise NotImplementedError(
                        f"int8 extract: LSTM at {node.name} has no "
                        f"getitem(0) consumer; the hidden-state tuple output "
                        f"is not supported, only the sequence output")
                H = int(mod.hidden_size)
                in_scale = scales[in_name]
                # The hidden state is a tanh output, so it lives in [-1, 1] and
                # 1/127 covers it exactly. The cell state is NOT bounded --
                # f*c + i*g can grow -- so it gets a wider, separate scale.
                # Sharing one scale between them is the obvious mistake and it
                # saturates c silently.
                h_scale = 1.0 / 127.0
                c_scale = 8.0 / 127.0
                cur_in = in_name
                cur_scale = in_scale
                cur_size = int(tensors_meta[in_name]["shape"][-1])
                for layer in range(int(mod.num_layers)):
                    w_ih = getattr(mod, f"weight_ih_l{layer}").detach()
                    w_hh = getattr(mod, f"weight_hh_l{layer}").detach()
                    s_ih = _scale_from_max_abs(w_ih)
                    s_hh = _scale_from_max_abs(w_hh)
                    q_ih = _quantize_per_tensor_sym(w_ih, s_ih)
                    q_hh = _quantize_per_tensor_sym(w_hh, s_hh)
                    has_bias = 1 if mod.bias else 0
                    # Both bias vectors share the input-side accumulator scale.
                    b_scale = cur_scale * s_ih
                    if has_bias:
                        b_ih = torch.round(
                            getattr(mod, f"bias_ih_l{layer}").detach()
                            / b_scale).to(torch.int32).cpu().numpy()
                        b_hh = torch.round(
                            getattr(mod, f"bias_hh_l{layer}").detach()
                            / b_scale).to(torch.int32).cpu().numpy()
                    else:
                        b_ih = np.zeros((4 * H,), dtype=np.int32)
                        b_hh = np.zeros((4 * H,), dtype=np.int32)
                    base = f"{node.target}.l{layer}"
                    weights_blob[f"{base}.w_ih_q"] = q_ih
                    weights_blob[f"{base}.w_hh_q"] = q_hh
                    weights_blob[f"{base}.b_ih_q"] = b_ih
                    weights_blob[f"{base}.b_hh_q"] = b_hh
                    # fp32 copies for a possible fp16 promotion of this layer.
                    fp32_stash[f"{base}.w_ih_q"] = (
                        w_ih.cpu().numpy().astype(np.float32))
                    fp32_stash[f"{base}.w_hh_q"] = (
                        w_hh.cpu().numpy().astype(np.float32))
                    if has_bias:
                        fp32_stash[f"{base}.b_ih_q"] = (
                            getattr(mod, f"bias_ih_l{layer}").detach()
                            .cpu().numpy().astype(np.float32))
                        fp32_stash[f"{base}.b_hh_q"] = (
                            getattr(mod, f"bias_hh_l{layer}").detach()
                            .cpu().numpy().astype(np.float32))
                    else:
                        fp32_stash[f"{base}.b_ih_q"] = np.zeros((4 * H,), np.float32)
                        fp32_stash[f"{base}.b_hh_q"] = np.zeros((4 * H,), np.float32)
                    h_name = f"{base}.h_state"
                    c_name = f"{base}.c_state"
                    for nm, sc in ((h_name, h_scale), (c_name, c_scale)):
                        tensors_meta[nm] = {
                            "shape": [1, H], "dtype": "i8",
                            "quant": {"scale": sc, "zero_point": 0},
                            "persistent": True,
                        }
                        scales[nm] = sc
                    out_nm = (out_node_name
                              if layer == int(mod.num_layers) - 1
                              else f"{base}.out")
                    if out_nm != out_node_name:
                        tensors_meta[out_nm] = {
                            "shape": [1, H], "dtype": "i8",
                            "quant": {"scale": h_scale, "zero_point": 0},
                        }
                        scales[out_nm] = h_scale
                    ops.append({
                        "name": base,
                        "op": "lstm_s8",
                        "inputs": [cur_in],
                        "outputs": [out_nm],
                        "state": [h_name, c_name],
                        "weight": f"{base}.w_ih_q",
                        "weight_hh": f"{base}.w_hh_q",
                        "bias": f"{base}.b_ih_q",
                        "bias_hh": f"{base}.b_hh_q",
                        "shape": {"input_size": cur_size, "hidden_size": H},
                        "quant": {
                            "scale_in": cur_scale,
                            "scale_w_ih": float(s_ih),
                            "scale_w_hh": float(s_hh),
                            "scale_b": float(b_scale),
                            "scale_h": h_scale,
                            "scale_c": c_scale,
                            "has_bias": has_bias,
                        },
                    })
                    cur_in = out_nm
                    cur_scale = h_scale
                    cur_size = H
                # The final layer's h IS the module output. Register it under
                # the getitem name with the scale the kernel actually writes.
                tensors_meta[out_node_name] = {
                    "shape": [1, H], "dtype": "i8",
                    "quant": {"scale": h_scale, "zero_point": 0},
                }
                scales[out_node_name] = h_scale
                # The getitem that unpacks (output, (h, c)) is now redundant:
                # our op already wrote its tensor. Skip it in the main walk, the
                # same way chunk's getitem consumers are skipped.
                _skip_nodes.add(out_node)
                # `y, _ = self.lstm(x)` also produces a getitem(1) for the
                # (h_n, c_n) tuple. When it is discarded -- which is how both
                # VitFly and this test model use it -- skip it. If something
                # actually consumes it, say so rather than silently dropping
                # the state the caller asked for: the state lives in the
                # persistent buffers and is not exposed as a tensor.
                state_node = _find_getitem_consumer(node, 1)
                if state_node is not None:
                    if len(state_node.users) > 0:
                        raise NotImplementedError(
                            "int8 extract: LSTM (h_n, c_n) output is consumed "
                            "at {}; the recurrent state is held in persistent "
                            "buffers and is not exposed as a tensor".format(
                                state_node.name))
                    _skip_nodes.add(state_node)

            elif isinstance(mod, torch.nn.LeakyReLU):
                _record(node.name, dtype="i8")
                n = int(np.prod(tensors_meta[in_name]["shape"]))
                ops.append({
                    "name": str(node.target),
                    "op": "leaky_relu_s8",
                    "inputs": [in_name],
                    "outputs": [node.name],
                    "shape": {"n": n},
                    "quant": {
                        "scale_in":  scales[in_name],
                        "scale_out": scales[node.name],
                        "activation_min": -128,
                        "activation_max": 127,
                        "negative_slope": float(mod.negative_slope),
                    },
                })

            elif isinstance(mod, torch.nn.AvgPool2d):
                _record(node.name, dtype="i8")
                in_shape = tensors_meta[in_name]["shape"]
                out_shape = tensors_meta[node.name]["shape"]
                N_, C, IH, IW = (int(s) for s in in_shape)
                _, _, OH, OW = (int(s) for s in out_shape)
                KH, KW = _pair(mod.kernel_size)
                SH, SW = _pair(mod.stride if mod.stride is not None
                               else mod.kernel_size)
                PH, PW = _pair(mod.padding)
                ops.append({
                    "name": str(node.target),
                    "op": "avgpool2d_s8",
                    "inputs": [in_name],
                    "outputs": [node.name],
                    "shape": {
                        "N": N_, "C": C,
                        "IH": IH, "IW": IW,
                        "OH": OH, "OW": OW,
                        "KH": KH, "KW": KW,
                        "SH": SH, "SW": SW,
                        "PH": PH, "PW": PW,
                        "count_include_pad": 1 if mod.count_include_pad else 0,
                    },
                })
                # Averaging values on a scale leaves them on that scale, so
                # like MaxPool this does not requantize. Overwrite the
                # calibrated scale so downstream consumers read the right one --
                # leaving the calibrated value here would silently rescale the
                # tensor.
                tensors_meta[node.name]["quant"]["scale"] = scales[in_name]
                scales[node.name] = scales[in_name]

            elif isinstance(mod, torch.nn.Upsample):
                if mod.mode != "nearest":
                    raise NotImplementedError(
                        f"int8 extract: Upsample mode={mod.mode!r} at "
                        f"{node.name}: only 'nearest' is supported."
                    )
                sf = mod.scale_factor
                if sf is None or float(sf) != int(float(sf)):
                    raise NotImplementedError(
                        f"int8 extract: Upsample scale_factor={sf} at "
                        f"{node.name}: only integer scales are supported."
                    )
                sf = int(float(sf))
                _record(node.name, dtype="i8")
                # Nearest upsample copies pixels without change — no requant.
                tensors_meta[node.name]["quant"]["scale"] = scales[in_name]
                scales[node.name] = scales[in_name]
                in_shape = tensors_meta[in_name]["shape"]
                N_, C, IH, IW = (int(s) for s in in_shape)
                ops.append({
                    "name": str(node.target),
                    "op": "upsample_nearest_s8",
                    "inputs": [in_name],
                    "outputs": [node.name],
                    "shape": {"N": N_, "C": C, "IH": IH, "IW": IW,
                              "scale": sf},
                })
            else:
                raise NotImplementedError(
                    f"int8 extract: unsupported module {type(mod).__name__} "
                    f"at {node.name}"
                )

        elif node.op == "call_function":
            target = node.target
            tname = getattr(target, "__name__", str(target))
            if (tname == "relu" or target is torch.relu
                    or target is torch.nn.functional.relu):
                in_name = node.args[0].name
                if node.name in fused_relu_after:
                    continue
                _record(node.name, dtype="i8")
                n = int(np.prod(tensors_meta[in_name]["shape"]))
                ops.append({
                    "name": node.name,
                    "op": "relu_s8",
                    "inputs": [in_name],
                    "outputs": [node.name],
                    "shape": {"n": n},
                })
            elif tname == "mul" or target is torch.mul \
                    or target is __import__("operator").mul:
                # mul_s8 existed with no call_function branch, the same shape of
                # gap as gelu_s8: a kernel nothing could emit.
                if not all(hasattr(a, "name") for a in node.args[:2]):
                    raise NotImplementedError(
                        f"int8 extract: mul at {node.name} with a scalar "
                        f"operand is not supported; both sides must be tensors")
                a_name = node.args[0].name
                b_name = node.args[1].name
                if tensors_meta[a_name]["shape"] != tensors_meta[b_name]["shape"]:
                    raise NotImplementedError(
                        f"int8 extract: mul at {node.name} broadcasts "
                        f"{tensors_meta[a_name]['shape']} against "
                        f"{tensors_meta[b_name]['shape']}; only equal shapes "
                        f"are supported")
                _record(node.name, dtype="i8")
                n = int(np.prod(tensors_meta[a_name]["shape"]))
                ops.append({
                    "name": node.name, "op": "mul_s8",
                    "inputs": [a_name, b_name], "outputs": [node.name],
                    "shape": {"n": n},
                    "quant": {"scale_a": scales[a_name],
                              "scale_b": scales[b_name],
                              "scale_out": scales[node.name],
                              "activation_min": -128, "activation_max": 127},
                })
            elif tname in ("sin", "cos") or target in (torch.sin, torch.cos):
                kind = "sin_s8" if (tname == "sin" or target is torch.sin) else "cos_s8"
                in_name = node.args[0].name
                _record(node.name, dtype="i8")
                n = int(np.prod(tensors_meta[in_name]["shape"]))
                ops.append({
                    "name": node.name, "op": kind,
                    "inputs": [in_name], "outputs": [node.name],
                    "shape": {"n": n},
                    "quant": {"scale_in": scales[in_name],
                              "scale_out": scales[node.name],
                              "activation_min": -128, "activation_max": 127},
                })
            elif tname == "softmax" or target is torch.softmax \
                    or getattr(target, "__name__", "") == "softmax":
                in_name = node.args[0].name
                _record(node.name, dtype="i8")
                shp = tensors_meta[in_name]["shape"]
                K = int(shp[-1]); M = int(np.prod(shp[:-1])) if len(shp) > 1 else 1
                dim = node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else -1)
                if dim not in (-1, len(shp) - 1):
                    raise NotImplementedError(
                        f"int8 extract: softmax at {node.name} over dim={dim}; "
                        f"only the last dimension is supported")
                ops.append({
                    "name": node.name, "op": "softmax_s8",
                    "inputs": [in_name], "outputs": [node.name],
                    "shape": {"M": M, "K": K},
                    "quant": {"scale_in": scales[in_name],
                              "scale_out": scales[node.name]},
                })
            elif getattr(target, "__name__", "") == "scaled_dot_product_attention":
                # Decompose rather than adding a fused kernel: matmul_s8
                # already carries transpose_b and scale_div, which is exactly
                # QK^T/sqrt(d), and softmax_s8 exists. Three existing kernels
                # beat one new one that would duplicate all three.
                q_n, k_n, v_n = (a.name for a in node.args[:3])
                if len(node.args) > 3 or node.kwargs.get("attn_mask") is not None:
                    raise NotImplementedError(
                        f"int8 extract: sdpa at {node.name} with an attention "
                        f"mask is not supported; only the unmasked form")
                if node.kwargs.get("is_causal"):
                    raise NotImplementedError(
                        f"int8 extract: causal sdpa at {node.name} needs a mask "
                        f"kernel; only the unmasked form is supported")
                _record(node.name, dtype="i8")
                qs = tensors_meta[q_n]["shape"]; ks = tensors_meta[k_n]["shape"]
                M = int(np.prod(qs[:-1])); D = int(qs[-1]); S = int(np.prod(ks[:-1]))
                scores = f"{node.name}__scores"
                probs = f"{node.name}__probs"
                # Scale for the attention scores.
                #
                # score = (q . k) / sqrt(D). In int8 units each operand has
                # sigma ~ 127/3, so the dot over D terms has sigma
                # ~ sqrt(D)*(127/3)^2 and the 1/sqrt(D) cancels it: the score
                # magnitude is ~independent of D and lands near
                # 3*(127/3)^2 ~ 1800 * scale_q * scale_k at 3 sigma. 32*sq*sk
                # puts 127 levels over ~4060*sq*sk, i.e. roughly 2x headroom on
                # that.
                #
                # The obvious-looking sq*sk*sqrt(D) is what was here first and
                # it CLIPS: measured on attn_block it covered 0.022 against an
                # actual score range of 0.095, and cosine at that step was
                # 0.914 instead of ~0.999.
                sc_scale = scales[q_n] * scales[k_n] * 32.0
                for nm, sc in ((scores, float(sc_scale)), (probs, 1.0 / 127.0)):
                    tensors_meta[nm] = {"shape": [M, S], "dtype": "i8",
                                        "quant": {"scale": sc, "zero_point": 0}}
                    scales[nm] = sc
                ops.append({
                    "name": f"{node.name}.qk", "op": "matmul_s8",
                    "inputs": [q_n, k_n], "outputs": [scores],
                    "shape": {"M": M, "K": D, "N": S, "transpose_b": 1},
                    "quant": {"scale_a": scales[q_n], "scale_b": scales[k_n],
                              "scale_out": float(sc_scale),
                              "scale_div": float(np.sqrt(D)),
                              "activation_min": -128, "activation_max": 127},
                })
                ops.append({
                    "name": f"{node.name}.softmax", "op": "softmax_s8",
                    "inputs": [scores], "outputs": [probs],
                    "shape": {"M": M, "K": S},
                    "quant": {"scale_in": float(sc_scale),
                              "scale_out": 1.0 / 127.0},
                })
                ops.append({
                    "name": f"{node.name}.av", "op": "matmul_s8",
                    "inputs": [probs, v_n], "outputs": [node.name],
                    "shape": {"M": M, "K": S, "N": int(tensors_meta[v_n]["shape"][-1]),
                              "transpose_b": 0},
                    "quant": {"scale_a": 1.0 / 127.0, "scale_b": scales[v_n],
                              "scale_out": scales[node.name], "scale_div": 1.0,
                              "activation_min": -128, "activation_max": 127},
                })
            elif tname == "relu6" or target is torch.nn.functional.relu6:
                in_name = node.args[0].name
                in_scale = scales[in_name]
                scales[node.name] = in_scale       # relu6 preserves scale
                _record(node.name, dtype="i8")
                qmax = max(1, min(127, int(round(6.0 / in_scale))))
                n = int(np.prod(tensors_meta[in_name]["shape"]))
                ops.append({
                    "name": node.name,
                    "op": "relu6_s8",
                    "inputs": [in_name],
                    "outputs": [node.name],
                    "shape": {"n": n},
                    "clamp_max": qmax,
                })
            elif tname == "flatten" or target is torch.flatten:
                in_name = node.args[0].name
                _record(node.name, dtype="i8")
                tensors_meta[node.name]["quant"]["scale"] = scales[in_name]
                scales[node.name] = scales[in_name]
                n = int(np.prod(tensors_meta[in_name]["shape"]))
                ops.append({
                    "name": node.name,
                    "op": "view",
                    "inputs": [in_name],
                    "outputs": [node.name],
                    "shape": {"n": n},
                })
            elif tname == "add" or target is torch.add or target is __import__("operator").add:
                a_name = node.args[0].name
                b_name = node.args[1].name
                _record(node.name, dtype="i8")
                n = int(np.prod(tensors_meta[a_name]["shape"]))
                # add→relu fusion: clamp at 0 instead of -128 and alias the
                # relu's output name to this add's buffer (mirrors the
                # linear/conv2d fusion logic above).
                next_node = (nodes[nodes.index(node) + 1]
                             if nodes.index(node) + 1 < len(nodes) else None)
                fuse_relu = (next_node is not None
                             and next_node.name in fused_add_relu)
                act_min = 0 if fuse_relu else -128
                if fuse_relu:
                    _skip_nodes.add(next_node)
                ops.append({
                    "name": node.name,
                    "op": "add_s8",
                    "inputs": [a_name, b_name],
                    "outputs": [
                        next_node.name if fuse_relu else node.name
                    ],
                    "shape": {"n": n},
                    "quant": {
                        "scale_a":   scales[a_name],
                        "scale_b":   scales[b_name],
                        "scale_out": scales[node.name],
                        "activation_min": act_min,
                        "activation_max": 127,
                    },
                })
                if fuse_relu and next_node.name not in tensors_meta:
                    tensors_meta[next_node.name] = dict(tensors_meta[node.name])
                    scales[next_node.name] = scales[node.name]
            elif target is torch.cat or tname == "cat":
                tensors_arg = node.args[0]
                if not isinstance(tensors_arg, (list, tuple)):
                    raise NotImplementedError(
                        f"int8 extract: cat at {node.name}: first arg must "
                        f"be a list/tuple of tensors."
                    )
                dim = int(node.args[1] if len(node.args) > 1 else
                          node.kwargs.get("dim", 0))
                if dim != 1:
                    raise NotImplementedError(
                        f"int8 extract: cat at {node.name}: dim={dim}, "
                        f"only dim=1 (channel concat) is supported."
                    )
                in_names = [t.name for t in tensors_arg]
                first_shape = list(tensors_meta[in_names[0]]["shape"])
                # 4D NCHW channel-concat, or 2D [N, C] feature-concat handled
                # as [N, C, 1, 1]. The cat_c1 kernels are layout-agnostic once
                # H=W=1, and the 2D case is the fused net's
                # [vision | depth | lowdim] fuse feeding the LSTM.
                if len(first_shape) == 4:
                    N_, _, H_, W_ = (int(s) for s in first_shape)
                elif len(first_shape) == 2:
                    N_ = int(first_shape[0])
                    H_ = W_ = 1
                else:
                    raise NotImplementedError(
                        f"int8 extract: cat at {node.name}: only 2D [N,C] or "
                        f"4D NCHW inputs supported (got {first_shape})."
                    )
                c_inputs = [int(tensors_meta[n]["shape"][1]) for n in in_names]
                n_inputs = len(in_names)
                if n_inputs not in (2, 3, 4):
                    raise NotImplementedError(
                        f"int8 extract: cat at {node.name}: {n_inputs} "
                        f"inputs; only 2/3/4-input cat is supported."
                    )
                op_kind = f"cat{n_inputs}_c1_s8"
                _record(node.name, dtype="i8")
                ops.append({
                    "name": node.name, "op": op_kind,
                    "inputs": in_names, "outputs": [node.name],
                    "shape": {"N": N_, "H": H_, "W": W_,
                              "C_inputs": c_inputs,
                              "C_total": sum(c_inputs)},
                    "quant": {
                        "scales_in": [scales[n] for n in in_names],
                        "scale_out": scales[node.name],
                        "activation_min": -128,
                        "activation_max": 127,
                    },
                })
            else:
                raise NotImplementedError(
                    f"int8 extract: unsupported function {tname} at {node.name}"
                )

        elif node.op == "call_method":
            target_name = node.target
            if target_name == "chunk":
                in_name = node.args[0].name
                n_chunks = int(node.args[1])
                dim_arg = int(node.args[2]) if len(node.args) > 2 else \
                    int(node.kwargs.get("dim", 0))
                if n_chunks != 2 or dim_arg != 1:
                    raise NotImplementedError(
                        f"int8 extract: chunk at {node.name}: only "
                        f"chunk(2, dim=1) is supported."
                    )
                in_shape = list(tensors_meta[in_name]["shape"])
                if len(in_shape) != 4:
                    raise NotImplementedError(
                        f"int8 extract: chunk at {node.name}: only 4D "
                        f"NCHW inputs supported."
                    )
                N_, C, H_, W_ = (int(s) for s in in_shape)
                if C % 2 != 0:
                    raise NotImplementedError(
                        f"int8 extract: chunk at {node.name}: C={C} is "
                        f"odd; can't split evenly."
                    )
                c_each = C // 2
                import operator as _op_mod
                gi0 = gi1 = None
                for user in node.users:
                    if (user.op == "call_function"
                            and user.target is _op_mod.getitem
                            and len(user.args) >= 2
                            and isinstance(user.args[1], int)):
                        if user.args[1] == 0:
                            gi0 = user
                        elif user.args[1] == 1:
                            gi1 = user
                if gi0 is None or gi1 is None:
                    raise NotImplementedError(
                        f"int8 extract: chunk at {node.name}: expected "
                        f"both getitem(_, 0) and getitem(_, 1) consumers."
                    )
                for gi in (gi0, gi1):
                    tensors_meta[gi.name] = {
                        "shape": [N_, c_each, H_, W_],
                        "dtype": "i8",
                        "quant": {
                            "scale": scales.get(gi.name, scales[in_name]),
                            "zero_point": 0,
                        },
                    }
                    scales[gi.name] = scales.get(gi.name, scales[in_name])
                    _skip_nodes.add(gi)
                ops.append({
                    "name": node.name, "op": "chunk2_c1",
                    "inputs": [in_name],
                    "outputs": [gi0.name, gi1.name],
                    "shape": {"N": N_, "C": C, "H": H_, "W": W_,
                              "c_each": c_each},
                })
            elif target_name in ("unsqueeze", "squeeze"):
                # Pure rank changes: the element order and count are unchanged,
                # so this is a view like flatten. LSTM inputs need one of these
                # to reach [seq, batch, feature].
                in_name = node.args[0].name
                _record(node.name, dtype="i8")
                tensors_meta[node.name]["quant"]["scale"] = scales[in_name]
                scales[node.name] = scales[in_name]
                n = int(np.prod(tensors_meta[in_name]["shape"]))
                if n != int(np.prod(tensors_meta[node.name]["shape"])):
                    raise NotImplementedError(
                        f"int8 extract: {target_name} at {node.name} changed "
                        f"the element count; only rank changes are views")
                ops.append({
                    "name": node.name,
                    "op": "view",
                    "inputs": [in_name],
                    "outputs": [node.name],
                    "shape": {"n": n},
                })
            else:
                raise NotImplementedError(
                    f"int8 extract: unsupported call_method "
                    f"'{target_name}' at {node.name}"
                )

        elif node.op == "output":
            arg = node.args[0]
            if isinstance(arg, (tuple, list)):
                output_names_local = [a.name for a in arg]
            else:
                output_names_local = [arg.name]
            output_name = output_names_local[0]
            # Stash full multi-output list for the IR builder.
            if len(output_names_local) > 1:
                output_names_multi = output_names_local
            else:
                output_names_multi = None

        elif node.op == "get_attr":
            raise NotImplementedError(f"int8 extract: get_attr {node.name} not supported")

    if not input_names or output_name is None:
        raise RuntimeError("int8 extract: graph missing input/output")

    # Per-channel int8: tighten conv/linear weights to per-output-channel scales
    # (skip ops about to be promoted to fp16). Backwards compatible — off by default.
    if per_channel:
        _apply_per_channel(ops, tensors_meta, weights_blob, fp32_stash, scales,
                           skip_names=(fp16_op_names or set()))

    # Mixed precision: promote the requested ops to fp16 and materialize the
    # int8<->fp16 boundary casts. With no spec this is a no-op and the IR is
    # byte-identical to the all-int8 one.
    if fp16_op_names:
        promoted = _promote_ops_to_fp16(ops, set(fp16_op_names), tensors_meta,
                                        weights_blob, fp32_stash)
        missing = sorted(set(fp16_op_names) - set(promoted))
        if missing:
            raise RuntimeError(
                f"int8 extract: fp16 promotion asked for op(s) {missing} that "
                f"are not in the graph; the graph has "
                f"{sorted(o['name'] for o in ops)}. A precision spec that "
                f"names a nonexistent op would silently leave that op int8.")
        ops = _insert_casts_i8_f16(ops, tensors_meta, scales)
        print(f"[extract_int8] fp16 islands: {promoted}", flush=True)

    output_tensors = (output_names_multi
                      if output_names_multi is not None
                      else [output_name])

    dispatches = _annotate_dispatches(ops)

    # Quantize each input with its own scale and surface dtype. int8 inputs are
    # symmetric-quantized; f16/f32 inputs pass through as floats (they feed an
    # fp16 island directly, so no cast op is needed at the surface).
    inputs_q: list[np.ndarray] = []
    for nm, si in zip(input_node_names, sample_inputs):
        dt = input_dtype_map.get(nm, "i8")
        if dt == "i8":
            q = _quantize_per_tensor_sym(si, scales[nm]).reshape(
                list(si.shape)).astype(np.int8)
        elif dt == "f16":
            q = si.detach().cpu().numpy().astype(np.float16)
        elif dt == "f32":
            q = si.detach().cpu().numpy().astype(np.float32)
        else:
            raise NotImplementedError(
                f"int8 extract: input dtype {dt!r} for {nm} is not supported "
                f"(supported: i8, f16, f32)")
        inputs_q.append(q)

    # Build the `input` IR field. Single-input keeps the legacy `tensor` key
    # alone. Multi-input adds `packed_inputs`: the harness hands run_model ONE
    # flat buffer holding every input back to back, and each entry says where
    # its tensor starts. `offset` is in elements of the buffer's own element
    # type (valid when every input shares a dtype); `byte_offset` and `dtype`
    # are always present so a MIXED-dtype buffer (i8 images + f16 lowdim) can
    # be addressed by byte and cast at the call site.
    _NP_OF = {"i8": np.int8, "f16": np.float16, "f32": np.float32}
    if len(input_names) == 1:
        ir_input: dict = {"tensor": input_names[0],
                          "tensors": input_names}
    else:
        packed: list[dict] = []
        byte_off = 0
        elem_off = 0
        for nm, q in zip(input_names, inputs_q):
            dt = input_dtype_map.get(nm, "i8")
            itemsize = np.dtype(_NP_OF[dt]).itemsize
            # Keep every slice naturally aligned for its own element type.
            if byte_off % itemsize:
                byte_off += itemsize - (byte_off % itemsize)
            sz = int(np.prod(q.shape))
            packed.append({"name": nm, "offset": elem_off, "size": sz,
                           "dtype": dt, "byte_offset": byte_off})
            byte_off += sz * itemsize
            elem_off += sz
        uniform = len({input_dtype_map.get(nm, "i8") for nm in input_names})
        ir_input = {"tensor": input_names[0],
                    "tensors": input_names,
                    "packed_inputs": packed,
                    "packed_dtype": (input_dtype_map.get(input_names[0], "i8")
                                     if uniform == 1 else "mixed"),
                    "packed_bytes": byte_off}
    ir = {
        "name": name,
        "version": 1,
        "quant": "int8",
        "input": ir_input,
        "output": {
            "tensors": output_tensors,
            "tensor": output_tensors[0] if len(output_tensors) == 1 else None,
        },
        "tensors": tensors_meta,
        "ops": ops,
        "dispatches": dispatches,
    }

    # Simulate the integer pipeline in Python so the golden output matches
    # bit-exactly what the C kernel will produce. Walking the IR ops:
    activations: dict[str, np.ndarray] = {
        nm: q for nm, q in zip(input_node_names, inputs_q)
    }
    for op in ops:
        in_name = op["inputs"][0]
        out_name = op["outputs"][0]
        in_arr = activations[in_name]
        if op["op"] == "linear_s8":
            w_q = weights_blob[op["weight"]]              # int8 [N, K]
            b_q = weights_blob[op["bias"]]                # int32 [N]
            sh = op["shape"]
            q = op["quant"]
            in_2d = in_arr.reshape(sh["M"], sh["K"]).astype(np.int32)
            w_2d = w_q.reshape(sh["N"], sh["K"]).astype(np.int32)
            # acc = sum over K of (in + zp_in) * (w + zp_w) + bias
            acc = (in_2d + q["input_offset"]) @ (w_2d + q["filter_offset"]).T
            acc += b_q.astype(np.int32)
            scaled = _requantize_int(acc, q["output_multiplier"],
                                     q["output_shift"])
            scaled += q["output_offset"]
            scaled = np.clip(scaled, q["activation_min"], q["activation_max"])
            activations[out_name] = scaled.astype(np.int8)
        elif op["op"] == "linear_s8_pc":
            w_q = weights_blob[op["weight"]]
            b_q = weights_blob[op["bias"]]
            sh = op["shape"]; q = op["quant"]
            mult = weights_blob[q["output_multiplier_per_oc_key"]]
            shift = weights_blob[q["output_shift_per_oc_key"]]
            in_2d = in_arr.reshape(sh["M"], sh["K"]).astype(np.int32)
            w_2d = w_q.reshape(sh["N"], sh["K"]).astype(np.int32)
            acc = in_2d @ w_2d.T + b_q.astype(np.int32)         # [M, N]
            scaled = _requantize_int_per_oc(acc, mult, shift, oc_axis=1)
            scaled = np.clip(scaled, q["activation_min"], q["activation_max"])
            activations[out_name] = scaled.astype(np.int8)
        elif op["op"] == "relu_s8":
            activations[out_name] = np.maximum(in_arr, 0).astype(np.int8)
        elif op["op"] == "relu6_s8":
            activations[out_name] = np.clip(
                in_arr, 0, op["clamp_max"]).astype(np.int8)
        elif op["op"] == "conv2d_s8":
            activations[out_name] = _sim_conv2d_s8(
                in_arr, op["shape"], op["quant"],
                weights_blob[op["weight"]], weights_blob[op["bias"]])
        elif op["op"] in ("conv2d_batchnorm2d_s8",
                          "conv2d_batchnorm2d_silu_s8"):
            # Fused conv→bn(→silu): run the sub-ops in sequence through the
            # same primitives as the standalone path so the golden is
            # bit-exact with the composed C reference kernel.
            cur = in_arr
            for sub in op["sub_ops"]:
                sk = sub["op"]
                if sk == "conv2d_s8":
                    cur = _sim_conv2d_s8(
                        cur, sub["shape"], sub["quant"],
                        weights_blob[sub["weight"]], weights_blob[sub["bias"]])
                elif sk == "batchnorm2d_s8":
                    cur = _sim_batchnorm2d_s8(
                        cur, sub["shape"], sub["quant"],
                        weights_blob[sub["weight"]], weights_blob[sub["bias"]])
                elif sk == "silu_s8":
                    cur = _sim_silu_s8(cur, sub["quant"])
                else:
                    raise NotImplementedError(
                        f"int8 simulator: fused sub-op {sk!r} unsupported")
            activations[out_name] = cur
        elif op["op"] == "conv2d_s8_pc":
            sh = op["shape"]
            q = op["quant"]
            w_q = weights_blob[op["weight"]].astype(np.int32)  # [OC, IC, KH, KW]
            b_q = weights_blob[op["bias"]].astype(np.int32)    # [OC]
            in_4d = in_arr.reshape(sh["N"], sh["IC"], sh["IH"], sh["IW"]).astype(np.int32)
            # Compute via direct sliding window (slow but correct simulator).
            out = _sim_conv2d_int32_acc(in_4d, w_q, b_q, sh,
                                        q["input_offset"], q["filter_offset"])
            if op["op"] == "conv2d_s8_pc":
                mult = weights_blob[q["output_multiplier_per_oc_key"]]
                shift = weights_blob[q["output_shift_per_oc_key"]]
                scaled = _requantize_int_per_oc(out, mult, shift, oc_axis=1)
            scaled = np.clip(scaled, q["activation_min"], q["activation_max"])
            activations[out_name] = scaled.astype(np.int8)
        elif op["op"] == "conv2d_silu_s8":
            # Fused conv2d + SiLU (extended fusion, gated by --enable-fusion
            # in extract_int8). Same accumulate + requantize as conv2d_s8,
            # but the intermediate is clamped to the PLAIN int8 range
            # (bias/activation_min/max only bound the final SiLU output,
            # per CONV2D_SILU_S8's semantics in reference_kernels.py) —
            # mirrors kernel_conv2d_silu_s8's reference impl exactly,
            # including reusing the intermediate as the SiLU LUT index.
            sh = op["shape"]
            q = op["quant"]
            w_q = weights_blob[op["weight"]].astype(np.int32)
            b_q = weights_blob[op["bias"]].astype(np.int32)
            in_4d = in_arr.reshape(sh["N"], sh["IC"], sh["IH"], sh["IW"]).astype(np.int32)
            out = _sim_conv2d_int32_acc(in_4d, w_q, b_q, sh,
                                        q["input_offset"], q["filter_offset"])
            scaled = _requantize_int(out, q["output_multiplier"], q["output_shift"])
            scaled += q["output_offset"]
            intermediate = np.clip(scaled, -128, 127).astype(np.int8)
            fv = intermediate.astype(np.float32) * np.float32(q["silu_scale_in"])
            silu_out = fv / (np.float32(1.0) + np.exp(-fv).astype(np.float32))
            v = np.round(silu_out.astype(np.float32)
                         / np.float32(q["silu_scale_out"])).astype(np.int32)
            v = np.clip(v, q["activation_min"], q["activation_max"])
            activations[out_name] = v.astype(np.int8)
        elif op["op"] == "conv2d_pool_s8":
            # Fused conv2d + maxpool2d (extended fusion, gated by
            # --enable-fusion). Same accumulate + requantize + clamp as
            # conv2d_s8 (activation_min/max bound the PRE-pool
            # intermediate exactly as a standalone conv2d_s8 would), then
            # the SAME sliding-window max as the standalone maxpool2d_s8
            # branch below, parameterized by this op's pool_* shape keys.
            # No second requantize — pool is scale-preserving.
            sh = op["shape"]
            q = op["quant"]
            w_q = weights_blob[op["weight"]].astype(np.int32)
            b_q = weights_blob[op["bias"]].astype(np.int32)
            in_4d = in_arr.reshape(sh["N"], sh["IC"], sh["IH"], sh["IW"]).astype(np.int32)
            out = _sim_conv2d_int32_acc(in_4d, w_q, b_q, sh,
                                        q["input_offset"], q["filter_offset"])
            scaled = _requantize_int(out, q["output_multiplier"], q["output_shift"])
            scaled += q["output_offset"]
            scaled = np.clip(scaled, q["activation_min"], q["activation_max"])
            conv_out = scaled.astype(np.int8)  # [N, OC, OH, OW]
            pKH, pKW = sh["pool_KH"], sh["pool_KW"]
            pSH, pSW = sh["pool_SH"], sh["pool_SW"]
            pPH, pPW = sh.get("pool_PH", 0), sh.get("pool_PW", 0)
            pDH, pDW = sh.get("pool_DH", 1), sh.get("pool_DW", 1)
            OHp = (sh["OH"] + 2*pPH - pDH*(pKH-1) - 1) // pSH + 1
            OWp = (sh["OW"] + 2*pPW - pDW*(pKW-1) - 1) // pSW + 1
            if pPH or pPW:
                pool_in = np.pad(conv_out,
                                 ((0, 0), (0, 0), (pPH, pPH), (pPW, pPW)),
                                 mode="constant",
                                 constant_values=np.iinfo(np.int8).min)
            else:
                pool_in = conv_out
            pooled = np.zeros((sh["N"], sh["OC"], OHp, OWp), dtype=np.int8)
            for ohp in range(OHp):
                for owp in range(OWp):
                    ih0 = ohp * pSH
                    iw0 = owp * pSW
                    cells = []
                    for kh in range(pKH):
                        for kw in range(pKW):
                            cells.append(pool_in[:, :, ih0 + kh*pDH, iw0 + kw*pDW])
                    pooled[:, :, ohp, owp] = np.stack(cells, axis=-1).max(axis=-1)
            activations[out_name] = pooled
        elif op["op"] == "maxpool2d_s8":
            sh = op["shape"]
            in_4d = in_arr.reshape(sh["N"], sh["C"], sh["IH"], sh["IW"])
            OH, OW = sh["OH"], sh["OW"]
            KH, KW = sh["KH"], sh["KW"]
            SH, SW = sh["SH"], sh["SW"]
            PH, PW = sh.get("PH", 0), sh.get("PW", 0)
            DH, DW = sh.get("DH", 1), sh.get("DW", 1)
            # Pad with int8 minimum so OOB lanes lose every max comparison.
            # Matches torch.nn.MaxPool2d's -inf semantics in the integer domain.
            if PH or PW:
                in_padded = np.pad(in_4d,
                                   ((0, 0), (0, 0), (PH, PH), (PW, PW)),
                                   mode="constant",
                                   constant_values=np.iinfo(np.int8).min)
            else:
                in_padded = in_4d
            out = np.zeros((sh["N"], sh["C"], OH, OW), dtype=np.int8)
            for oh in range(OH):
                for ow in range(OW):
                    ih0 = oh * SH
                    iw0 = ow * SW
                    # Build a (KH, KW, N, C) gather that honors dilation.
                    cells = []
                    for kh in range(KH):
                        for kw in range(KW):
                            cells.append(in_padded[:, :, ih0 + kh*DH, iw0 + kw*DW])
                    out[:, :, oh, ow] = np.stack(cells, axis=-1).max(axis=-1)
            activations[out_name] = out
        elif op["op"] == "view":
            activations[out_name] = in_arr  # alias
        elif op["op"] == "add_s8":
            sh = op["shape"]
            q = op["quant"]
            a = activations[op["inputs"][0]].astype(np.float32) * np.float32(q["scale_a"])
            b = activations[op["inputs"][1]].astype(np.float32) * np.float32(q["scale_b"])
            f = (a + b) / np.float32(q["scale_out"])
            v = np.round(f).astype(np.int32)
            v = np.clip(v, q["activation_min"], q["activation_max"])
            activations[out_name] = v.astype(np.int8)
        elif op["op"] == "batchnorm2d_s8":
            activations[out_name] = _sim_batchnorm2d_s8(
                in_arr, op["shape"], op["quant"],
                weights_blob[op["weight"]], weights_blob[op["bias"]])
        elif op["op"] == "sigmoid_s8":
            q = op["quant"]
            fv = in_arr.astype(np.float32) * np.float32(q["scale_in"])
            sig = 1.0 / (1.0 + np.exp(-fv.astype(np.float32)))
            v = np.round(sig.astype(np.float32) / np.float32(q["scale_out"])).astype(np.int32)
            v = np.clip(v, q["activation_min"], q["activation_max"])
            activations[out_name] = v.astype(np.int8)
        elif op["op"] == "silu_s8":
            activations[out_name] = _sim_silu_s8(in_arr, op["quant"])
        elif op["op"] == "elu_s8":
            q = op["quant"]
            fv = in_arr.astype(np.float32) * np.float32(q["scale_in"])
            alpha = np.float32(q.get("alpha", 1.0))
            elu_out = np.where(fv > 0, fv, alpha * (np.exp(fv) - np.float32(1.0))).astype(np.float32)
            v = np.round(elu_out / np.float32(q["scale_out"])).astype(np.int32)
            v = np.clip(v, q["activation_min"], q["activation_max"])
            activations[out_name] = v.astype(np.int8)
        elif op["op"] == "mul_s8":
            q = op["quant"]
            # Flatten both: activations are stored with whatever rank their
            # producer used, and this op is elementwise over equal shapes.
            a = activations[op["inputs"][0]].reshape(-1).astype(np.float64) \
                * float(q["scale_a"])
            b = activations[op["inputs"][1]].reshape(-1).astype(np.float64) \
                * float(q["scale_b"])
            v = _round_half_away((a * b) / float(q["scale_out"]))
            activations[out_name] = np.clip(
                v, q["activation_min"], q["activation_max"]).astype(np.int8)
        elif op["op"] in ("sin_s8", "cos_s8"):
            q = op["quant"]
            x = in_arr.astype(np.float64) * float(q["scale_in"])
            y = np.sin(x) if op["op"] == "sin_s8" else np.cos(x)
            v = _round_half_away(y / float(q["scale_out"]))
            activations[out_name] = np.clip(
                v, q["activation_min"], q["activation_max"]).astype(np.int8)
        elif op["op"] == "softmax_s8":
            sh = op["shape"]; q = op["quant"]
            x = in_arr.reshape(sh["M"], sh["K"]).astype(np.float64) \
                * float(q["scale_in"])
            x = x - x.max(axis=1, keepdims=True)      # stable, as the kernel does
            e = np.exp(x)
            y = e / e.sum(axis=1, keepdims=True)
            v = _round_half_away(y / float(q["scale_out"]))
            activations[out_name] = np.clip(v, -128, 127).astype(np.int8)
        elif op["op"] == "matmul_s8":
            sh = op["shape"]; q = op["quant"]
            a = activations[op["inputs"][0]].reshape(sh["M"], sh["K"]) \
                .astype(np.float64) * float(q["scale_a"])
            bm = activations[op["inputs"][1]]
            if sh.get("transpose_b"):
                bm = bm.reshape(sh["N"], sh["K"]).astype(np.float64).T
            else:
                bm = bm.reshape(sh["K"], sh["N"]).astype(np.float64)
            y = (a @ (bm * float(q["scale_b"]))) / float(q.get("scale_div", 1.0))
            v = _round_half_away(y / float(q["scale_out"]))
            activations[out_name] = np.clip(
                v, q["activation_min"], q["activation_max"]).astype(np.int8)
        elif op["op"] == "gelu_s8":
            q = op["quant"]
            import math as _math
            # float32 with the kernel's own constant, not float64 with a more
            # accurate 1/sqrt(2): the golden must reproduce what the device
            # computes, including its constant.
            kInvSqrt2 = np.float32(0.70710678118)
            x = in_arr.astype(np.float32) * np.float32(q["scale_in"])
            erf = np.vectorize(_math.erf, otypes=[np.float32])
            y = np.float32(0.5) * x * (np.float32(1.0) + erf(x * kInvSqrt2))
            v = _round_half_away(y.astype(np.float64)
                                 / float(q["scale_out"]))
            activations[out_name] = np.clip(
                v, q["activation_min"], q["activation_max"]).astype(np.int8)
        elif op["op"] in ("layernorm_s8", "rmsnorm_s8"):
            sh = op["shape"]; q = op["quant"]
            M, K = sh["M"], sh["K"]
            x = in_arr.reshape(M, K).astype(np.float64) * float(q["scale_in"])
            g = weights_blob[op["weight"]].astype(np.float64)
            eps = float(q["eps"])
            if op["op"] == "rmsnorm_s8":
                # RMSNorm does NOT subtract the mean; conflating it with a
                # bias-free LayerNorm is silent and wrong for rows with DC.
                inv = 1.0 / np.sqrt((x * x).mean(axis=1, keepdims=True) + eps)
                y = x * inv * g
            else:
                mu = x.mean(axis=1, keepdims=True)
                var = ((x - mu) ** 2).mean(axis=1, keepdims=True)  # biased, as torch
                y = (x - mu) / np.sqrt(var + eps) * g
                if op.get("bias"):
                    y = y + weights_blob[op["bias"]].astype(np.float64)
            v = _round_half_away(y / float(q["scale_out"]))
            activations[out_name] = np.clip(
                v, q["activation_min"], q["activation_max"]).astype(np.int8)
        elif op["op"] == "lstm_s8":
            sh = op["shape"]; q = op["quant"]
            H = sh["hidden_size"]; IS = sh["input_size"]
            # float64 throughout, matching the kernel's double: the op is
            # evaluated in floating point, so golden and device agree only if
            # both land well inside half an int8 LSB. float32 on both sides
            # still disagreed by up to 2 LSB.
            w_ih = weights_blob[op["weight"]].astype(np.float64) * float(q["scale_w_ih"])
            w_hh = weights_blob[op["weight_hh"]].astype(np.float64) * float(q["scale_w_hh"])
            b = np.zeros((4 * H,), dtype=np.float64)
            if q["has_bias"]:
                b = ((weights_blob[op["bias"]] + weights_blob[op["bias_hh"]])
                     .astype(np.float64) * float(q["scale_b"]))
            h_name, c_name = op["state"]
            # Persistent state: zero on the first step, then whatever the
            # previous invocation left. The simulator has to model that or the
            # golden diverges from the device on step 2 onwards.
            h_q = activations.get(h_name)
            c_q = activations.get(c_name)
            h = (np.zeros(H, np.float64) if h_q is None
                 else h_q.reshape(-1).astype(np.float64) * float(q["scale_h"]))
            c = (np.zeros(H, np.float64) if c_q is None
                 else c_q.reshape(-1).astype(np.float64) * float(q["scale_c"]))
            x = in_arr.reshape(-1).astype(np.float64) * float(q["scale_in"])
            g = w_ih @ x[:IS] + w_hh @ h + b
            i_g = 1.0 / (1.0 + np.exp(-g[0:H]))
            f_g = 1.0 / (1.0 + np.exp(-g[H:2*H]))
            g_g = np.tanh(g[2*H:3*H])
            o_g = 1.0 / (1.0 + np.exp(-g[3*H:4*H]))
            c_new = f_g * c + i_g * g_g
            h_new = o_g * np.tanh(c_new)
            cq = np.clip(_round_half_away(c_new / float(q["scale_c"])), -128, 127).astype(np.int8)
            hq = np.clip(_round_half_away(h_new / float(q["scale_h"])), -128, 127).astype(np.int8)
            activations[c_name] = cq.reshape(1, H)
            activations[h_name] = hq.reshape(1, H)
            activations[out_name] = hq.reshape(1, H)
        elif op["op"] == "leaky_relu_s8":
            q = op["quant"]
            fv = in_arr.astype(np.float32) * np.float32(q["scale_in"])
            slope = np.float32(q.get("negative_slope", 0.01))
            y = np.where(fv > 0, fv, slope * fv).astype(np.float32)
            v = np.round(y / np.float32(q["scale_out"])).astype(np.int32)
            v = np.clip(v, q["activation_min"], q["activation_max"])
            activations[out_name] = v.astype(np.int8)
        elif op["op"] == "avgpool2d_s8":
            sh = op["shape"]
            in_4d = in_arr.reshape(sh["N"], sh["C"], sh["IH"], sh["IW"])
            KH, KW = sh["KH"], sh["KW"]
            SH, SW = sh["SH"], sh["SW"]
            PH, PW = sh["PH"], sh["PW"]
            cip = bool(sh.get("count_include_pad", 1))
            OH, OW = sh["OH"], sh["OW"]
            # Pad with the quantized zero, which is the real 0 under symmetric
            # per-tensor quantization -- so padded taps contribute nothing to
            # the sum and only the divisor distinguishes count_include_pad.
            padded = np.pad(in_4d.astype(np.int32),
                            ((0, 0), (0, 0), (PH, PH), (PW, PW)))
            valid = np.pad(np.ones_like(in_4d, dtype=np.int32),
                           ((0, 0), (0, 0), (PH, PH), (PW, PW)))
            out = np.zeros((sh["N"], sh["C"], OH, OW), dtype=np.int8)
            for oh in range(OH):
                for ow in range(OW):
                    win = padded[:, :, oh*SH:oh*SH+KH, ow*SW:ow*SW+KW]
                    tot = win.sum(axis=(2, 3))
                    if cip:
                        div = np.full_like(tot, KH * KW)
                    else:
                        div = valid[:, :, oh*SH:oh*SH+KH,
                                    ow*SW:ow*SW+KW].sum(axis=(2, 3))
                        div = np.maximum(div, 1)
                    # round half away from zero, matching the C kernel
                    v = np.where(tot >= 0,
                                 (tot + div // 2) // div,
                                 -((-tot + div // 2) // div))
                    out[:, :, oh, ow] = np.clip(v, -128, 127).astype(np.int8)
            activations[out_name] = out
        elif op["op"] == "upsample_nearest_s8":
            sh = op["shape"]
            scale = sh["scale"]
            in_4d = in_arr.reshape(sh["N"], sh["C"], sh["IH"], sh["IW"])
            OH, OW = sh["IH"] * scale, sh["IW"] * scale
            out = np.zeros((sh["N"], sh["C"], OH, OW), dtype=np.int8)
            for oh in range(OH):
                for ow in range(OW):
                    out[:, :, oh, ow] = in_4d[:, :, oh // scale, ow // scale]
            activations[out_name] = out
        elif op["op"] in ("cat2_c1_s8", "cat3_c1_s8", "cat4_c1_s8"):
            sh = op["shape"]
            q = op["quant"]
            N_, H_, W_ = sh["N"], sh["H"], sh["W"]
            parts = []
            for inp_name, s_in, c in zip(op["inputs"], q["scales_in"],
                                         sh["C_inputs"]):
                t = activations[inp_name].reshape(N_, c, H_, W_).astype(np.float32)
                fv = t * np.float32(s_in)
                v = np.round(fv / np.float32(q["scale_out"])).astype(np.int32)
                v = np.clip(v, q["activation_min"], q["activation_max"])
                parts.append(v.astype(np.int8))
            activations[out_name] = np.concatenate(parts, axis=1)
        elif op["op"] == "chunk2_c1":
            sh = op["shape"]
            in_4d = in_arr.reshape(sh["N"], sh["C"], sh["H"], sh["W"])
            c_each = sh["c_each"]
            activations[op["outputs"][0]] = in_4d[:, :c_each, :, :].astype(np.int8)
            activations[op["outputs"][1]] = in_4d[:, c_each:, :, :].astype(np.int8)
        # ---- fp16 islands (mixed precision) --------------------------------
        elif op["op"] == "cast_i8_to_f16":
            activations[out_name] = (
                in_arr.astype(np.float32)
                * np.float32(op["quant"]["scale"])).astype(np.float16)
        elif op["op"] == "cast_f16_to_i8":
            inv = np.float32(1.0 / max(float(op["quant"]["scale"]), 1e-30))
            v = _round_half_away(in_arr.astype(np.float32) * inv)
            activations[out_name] = np.clip(v, -128, 127).astype(np.int8)
        elif op["op"] in ("cat2_c1_f16", "cat3_c1_f16", "cat4_c1_f16"):
            # fp16 concat is a pure copy -- no requantization, which is the
            # whole point: the int8 version had to squeeze 512 vision features
            # (max-abs 0.24) onto the optical-flow-dominated output scale.
            sh = op["shape"]
            N_, H_, W_ = sh["N"], sh["H"], sh["W"]
            parts = [activations[n].reshape(N_, c, H_, W_).astype(np.float16)
                     for n, c in zip(op["inputs"], sh["C_inputs"])]
            activations[out_name] = np.concatenate(parts, axis=1)
        elif op["op"] == "linear_f16":
            # fp16 storage, fp32 accumulate -- matches kernel_linear_f16.
            sh = op["shape"]
            w = weights_blob[op["weight"]].astype(np.float32)
            x = in_arr.reshape(sh["M"], sh["K"]).astype(np.float32)
            y = x @ w.reshape(sh["N"], sh["K"]).T
            if op.get("bias"):
                y = y + weights_blob[op["bias"]].astype(np.float32)
            activations[out_name] = y.astype(np.float16)
        elif op["op"] == "lstm_f16":
            # Mirrors kernel_lstm_f16: fp16 storage AND fp16 gate-GEMM
            # accumulate (each product rounded to fp16, then added in fp16),
            # with the sigmoid/tanh/cell update in float32. The fp16 accumulate
            # is modelled step by step rather than as one matmul, because the
            # rounding is what the device does.
            sh = op["shape"]
            H = int(sh["hidden_size"]); IS = int(sh["input_size"])
            w_ih = weights_blob[op["weight"]].reshape(4 * H, IS)
            w_hh = weights_blob[op["weight_hh"]].reshape(4 * H, H)
            b = weights_blob[op["bias"]].astype(np.float32)
            h_name, c_name = (op["state"] if isinstance(op["state"], (list, tuple))
                              else (op["state"]["h"], op["state"]["c"]))
            h_prev = activations.get(h_name)
            c_prev = activations.get(c_name)
            h = (np.zeros(H, np.float16) if h_prev is None
                 else h_prev.reshape(-1).astype(np.float16))
            c = (np.zeros(H, np.float32) if c_prev is None
                 else c_prev.reshape(-1).astype(np.float32))
            x = in_arr.reshape(-1)[:IS].astype(np.float16)
            px = (x.astype(np.float32)
                  * w_ih.astype(np.float32)).astype(np.float16)   # [4H, IS]
            ph = (h.astype(np.float32)
                  * w_hh.astype(np.float32)).astype(np.float16)   # [4H, H]
            ax = np.zeros(4 * H, np.float16)
            for k in range(IS):
                ax = (ax.astype(np.float32)
                      + px[:, k].astype(np.float32)).astype(np.float16)
            ah = np.zeros(4 * H, np.float16)
            for k in range(H):
                ah = (ah.astype(np.float32)
                      + ph[:, k].astype(np.float32)).astype(np.float16)
            pre = (ax.astype(np.float32) + ah.astype(np.float32) + b)
            ig = (1.0 / (1.0 + np.exp(-pre[0:H]))).astype(np.float32)
            fg = (1.0 / (1.0 + np.exp(-pre[H:2*H]))).astype(np.float32)
            cg = np.tanh(pre[2*H:3*H]).astype(np.float32)
            og = (1.0 / (1.0 + np.exp(-pre[3*H:4*H]))).astype(np.float32)
            c_new = (fg * c + ig * cg).astype(np.float32)
            out_f16 = (og * np.tanh(c_new).astype(np.float32)).astype(np.float16)
            activations[c_name] = c_new.astype(np.float16).reshape(1, H)
            activations[h_name] = out_f16.reshape(1, H)
            activations[out_name] = out_f16.reshape(1, H)
        else:
            raise NotImplementedError(
                f"int8 simulator: unsupported op {op['op']}"
            )

    # Concatenate outputs in IR order (matching multi-output goldens
    # elsewhere). The surface dtype follows the IR: an fp16-promoted tail
    # writes f16 outputs, everything else int8.
    _out_np_dtype = (np.float16
                     if tensors_meta.get(output_tensors[0], {}).get("dtype") == "f16"
                     else np.int8)
    out_q = np.concatenate([
        activations[t].reshape(-1).astype(_out_np_dtype) for t in output_tensors
    ])
    # One flat, native-dtype array per input (input0..N), plus the single
    # packed `input` buffer the harness actually bakes in. When every input
    # shares a dtype the packed buffer is that dtype; when they do not (i8
    # images + f16 lowdim) it is a byte buffer whose slices the generated C
    # re-casts, so the golden must be packed byte-wise the same way.
    inputs_flat = {f"input{i}": q.reshape(-1) for i, q in enumerate(inputs_q)}
    if len(inputs_q) == 1:
        inp_q = inputs_q[0].reshape(-1)
    elif ir_input.get("packed_dtype") != "mixed":
        inp_q = np.concatenate([q.reshape(-1) for q in inputs_q])
    else:
        blob = np.zeros(int(ir_input["packed_bytes"]), dtype=np.int8)
        for entry, q in zip(ir_input["packed_inputs"], inputs_q):
            raw = np.ascontiguousarray(q.reshape(-1)).view(np.int8)
            off = int(entry["byte_offset"])
            blob[off:off + raw.size] = raw
        inp_q = blob

    ir_path = os.path.join(out_dir, "graph.json")
    weights_path = os.path.join(out_dir, "weights.npz")
    io_path = os.path.join(out_dir, "io.npz")
    passes_path = os.path.join(out_dir, "passes_applied.json")
    with open(ir_path, "w") as f:
        json.dump(ir, f, indent=2)
    np.savez(weights_path, **weights_blob)
    np.savez(io_path, input=inp_q, output=out_q, **inputs_flat)
    # passes_applied.json records each fusion / fold pass that fired
    # during this IR extraction plus the IR-side op counts that
    # downstream tooling uses to answer "did my new fusion pattern
    # actually run?" without grepping stdout.
    passes_log = {
        "schema_version": 1,
        "extractor": "extract_graph",
        "enable_fusion": enable_fusion,
        "fusion_target": fusion_target,
        "extended_fusion_active": _extended_fusion_ok,
        "n_fx_nodes": len(nodes),
        "n_ir_ops": len(ir["ops"]),
        "bn_folding_enabled": fold_conv_bn,
        "passes": {
            "linear_relu_fuse": {
                "fired": len(fused_linear_relu),
                "sites": sorted(fused_linear_relu),
            },
            "conv2d_relu_fuse": {
                "fired": len(fused_conv2d_relu),
                "sites": sorted(fused_conv2d_relu),
            },
            "add_relu_fuse": {
                "fired": len(fused_add_relu),
                "sites": sorted(fused_add_relu),
            },
            "bn_relu_fuse": {
                "fired": len(fused_bn_relu),
                "sites": sorted(fused_bn_relu),
            },
            "bn_conv_fuse": {
                "fired": len(folded_bn_names),
                "sites": sorted(folded_bn_names),
            },
            # Gated on --enable-fusion / enable_fusion=True; empty set (and
            # zero-op-count effect) whenever the flag is off, so this key
            # is the flag-on/flag-off diff to grep for on this pass.
            "conv2d_silu_fuse": {
                "fired": len(fused_conv2d_silu),
                "sites": sorted(fused_conv2d_silu),
            },
            "conv2d_pool_fuse": {
                "fired": len(fused_conv2d_pool),
                "sites": sorted(fused_conv2d_pool),
            },
        },
    }
    with open(passes_path, "w") as f:
        json.dump(passes_log, f, indent=2)
    print(f"wrote {ir_path}")
    print(f"wrote {weights_path}  ({len(weights_blob)} tensors)")
    print(f"wrote {io_path}  ({len(inputs_q)} input(s) "
          f"{[str(q.dtype) for q in inputs_q]}, packed dtype={inp_q.dtype}, "
          f"output dtype={out_q.dtype})")
    print(f"wrote {passes_path}  ("
          f"enable_fusion={enable_fusion}, "
          f"linear_relu_fuse={len(fused_linear_relu)}, "
          f"conv2d_relu_fuse={len(fused_conv2d_relu)}, "
          f"add_relu_fuse={len(fused_add_relu)}, "
          f"bn_relu_fuse={len(fused_bn_relu)}, "
          f"bn_conv_fuse={len(folded_bn_names)}, "
          f"conv2d_silu_fuse={len(fused_conv2d_silu)}, "
          f"conv2d_pool_fuse={len(fused_conv2d_pool)})")
    return ir


# ---------------------------------------------------------------------------
# fp32 extractor (unchanged path)
# ---------------------------------------------------------------------------

def extract(
    model: torch.nn.Module,
    sample_input: "torch.Tensor | list[torch.Tensor]",
    name: str,
    out_dir: str,
    quant: str = "fp32",
    calibration_samples: "list[torch.Tensor] | None" = None,
    input_dtypes: "list[str] | None" = None,
    fp16_op_names: "set[str] | None" = None,
    per_channel: bool = False,
    enable_fusion: bool = False,
    fusion_target: "str | None" = None,
    fold_conv_bn: bool = True,
) -> dict[str, Any]:
    """Trace `model`, dump IR + weights + I/O into `out_dir`.

    `quant` is recorded in the IR top-level field so downstream stages (and
    cache-key naming) can branch on it. Supported: fp32, fp16, int8.

    fp16 mode: the graph is traced at fp32 (more stable for ShapeProp —
    a few ops error out on half tensors during tracing), but weights,
    inputs, and the golden output are saved as np.float16, and op names
    get a "_f16" suffix so downstream picks the half-precision kernel
    variants. The golden is recomputed via `model.half()` on
    `input.half()` so we're comparing genuine fp16 numerics, not
    fp32-traced numerics down-cast at the boundary.
    """
    if quant == "int8":
        return extract_int8(
            model, sample_input, name, out_dir,
            calibration_samples=calibration_samples,
            input_dtypes=input_dtypes,
            fp16_op_names=fp16_op_names,
            per_channel=per_channel,
            enable_fusion=enable_fusion,
            fusion_target=fusion_target,
            fold_conv_bn=fold_conv_bn,
        )
    if quant not in ("fp32", "fp16"):
        raise NotImplementedError(
            f"quant={quant!r} not supported (have: fp32, fp16, int8)"
        )
    weight_dtype = np.float16 if quant == "fp16" else np.float32
    op_suffix = "_f16" if quant == "fp16" else ""
    os.makedirs(out_dir, exist_ok=True)
    model = model.eval()

    # Normalise sample_input to a list so multi-input models (matmul A+B,
    # bmm) work with the same code path as single-input ones.
    if isinstance(sample_input, torch.Tensor):
        sample_inputs: list[torch.Tensor] = [sample_input]
    else:
        sample_inputs = list(sample_input)

    gm = torch.fx.symbolic_trace(model)
    ShapeProp(gm).propagate(*sample_inputs)

    # KernelBench Phase 2 has a few compound activations that don't
    # appear as a single FX node — they're hand-rolled with primitive
    # ops (mul/add/abs/pow/div/tanh/sigmoid). When the entire forward
    # graph matches one of those known shapes we collapse it into a
    # single fused op via _maybe_fuse_compound_activation, which
    # rewrites the FX graph in place by replacing the multi-node
    # subgraph with a single call to a stub `_agents_compound_<name>`
    # marker function. The per-node iteration below then sees that
    # marker and emits a clean single-op IR.
    _maybe_fuse_compound_activation(gm)
    _maybe_fuse_loss(gm)

    tensors: dict[str, dict] = {}
    ops: list[dict] = []
    weights: dict[str, np.ndarray] = {}

    input_names: list[str] = []
    output_names: list[str] = []

    # Pre-scan: mark transpose nodes that feed directly into matmul/mm as
    # skip — they get fused into matmul_ta/matmul_tb/matmul_tatb op variants
    # and must not emit their own IR op or allocate a buffer.
    # Two FX forms for transpose:
    #   • A.t()  → call_method  target="t"
    #   • A.T    → call_function target=builtins.getattr, args=(A, 'T')
    def _is_transpose_node(n: Any) -> bool:
        if not isinstance(n, torch.fx.Node):
            return False
        if n.op == "call_method" and n.target == "t":
            return True
        if (n.op == "call_function" and
                getattr(n.target, "__name__", "") == "getattr" and
                len(n.args) == 2 and n.args[1] == "T"):
            return True
        return False

    _skip_nodes: set = set()
    for _n in gm.graph.nodes:
        # `@` traces to operator.matmul (name "matmul"), not torch.matmul —
        # match by name too so the diag/transpose fusions fire for both.
        if (_n.op == "call_function" and
                (_n.target in (torch.matmul, torch.mm, _operator.matmul)
                 or getattr(_n.target, "__name__", "") in ("matmul", "mm"))):
            for _arg in _n.args[:2]:
                if _is_transpose_node(_arg):
                    _skip_nodes.add(_arg)
                # diag(A) @ B is fused into diag_matmul; the diag node itself
                # emits no op.
                elif (isinstance(_arg, torch.fx.Node)
                      and _arg.op == "call_function"
                      and getattr(_arg.target, "__name__", "") == "diag"):
                    _skip_nodes.add(_arg)

    for node in gm.graph.nodes:
        if node in _skip_nodes:
            continue
        if node.op == "placeholder":
            input_names.append(node.name)
            tensors[node.name] = _tensor_meta(node)

        elif node.op == "call_module":
            mod = gm.get_submodule(node.target)
            if not isinstance(mod, SUPPORTED_MODULES):
                raise NotImplementedError(
                    f"unsupported module {type(mod).__name__} at node {node.name}"
                )
            in_name = node.args[0].name
            out_name = node.name
            # nn.LSTM returns a tuple (output, (h,c)); _tensor_meta can't read a
            # tuple node's shape. The LSTM branch registers its own per-layer
            # output tensors below, so skip the scalar-meta assignment here.
            if not isinstance(mod, torch.nn.LSTM):
                tensors[out_name] = _tensor_meta(node)

            if isinstance(mod, torch.nn.Linear):
                w_key = f"{node.target}.weight"
                b_key = f"{node.target}.bias" if mod.bias is not None else None
                weights[w_key] = mod.weight.detach().cpu().numpy().astype(weight_dtype)
                if b_key is not None:
                    weights[b_key] = mod.bias.detach().cpu().numpy().astype(weight_dtype)
                in_shape = tensors[in_name]["shape"]
                out_shape = tensors[out_name]["shape"]
                M = int(np.prod(in_shape[:-1]))
                K = int(in_shape[-1])
                N = int(out_shape[-1])
                ops.append({
                    "name": str(node.target),
                    "op": "linear",
                    "inputs": [in_name],
                    "outputs": [out_name],
                    "weight": w_key,
                    "bias": b_key,
                    "shape": {"M": M, "K": K, "N": N},
                })

            elif isinstance(mod, torch.nn.ReLU):
                n = int(np.prod(tensors[in_name]["shape"]))
                ops.append({
                    "name": str(node.target),
                    "op": "relu",
                    "inputs": [in_name],
                    "outputs": [out_name],
                    "shape": {"n": n},
                })

            elif isinstance(mod, torch.nn.ELU):
                n = int(np.prod(tensors[in_name]["shape"]))
                ops.append({
                    "name": str(node.target),
                    "op": "elu",
                    "inputs": [in_name],
                    "outputs": [out_name],
                    "shape": {"n": n},
                    "alpha": float(mod.alpha),
                })

            elif isinstance(mod, torch.nn.Conv2d):
                w_key = f"{node.target}.weight"
                b_key = f"{node.target}.bias" if mod.bias is not None else None
                weights[w_key] = mod.weight.detach().cpu().numpy().astype(weight_dtype)
                if b_key is not None:
                    weights[b_key] = mod.bias.detach().cpu().numpy().astype(weight_dtype)
                in_shape = tensors[in_name]["shape"]
                out_shape = tensors[out_name]["shape"]
                # NCHW.
                N_, IC, IH, IW = (int(s) for s in in_shape)
                _, OC, OH, OW = (int(s) for s in out_shape)
                KH, KW = _pair(mod.kernel_size)
                SH, SW = _pair(mod.stride)
                PH, PW = _pair(mod.padding)
                DH, DW = _pair(mod.dilation)
                # Depthwise conv: each output channel reads from one input
                # channel via its own [1, KH, KW] filter. Detected when
                # groups == in_channels == out_channels. Different memory
                # access pattern (no IC reduction) so it gets its own kernel.
                if mod.groups == 1:
                    ops.append({
                        "name": str(node.target),
                        "op": "conv2d",
                        "inputs": [in_name],
                        "outputs": [out_name],
                        "weight": w_key,
                        "bias": b_key,
                        "shape": {
                            "N": N_, "IC": IC, "IH": IH, "IW": IW,
                            "OC": OC, "OH": OH, "OW": OW,
                            "KH": KH, "KW": KW,
                            "SH": SH, "SW": SW,
                            "PH": PH, "PW": PW,
                            "DH": DH, "DW": DW,
                        },
                    })
                elif mod.groups == IC and IC == OC and DH == 1 and DW == 1:
                    ops.append({
                        "name": str(node.target),
                        "op": "conv2d_dw",
                        "inputs": [in_name],
                        "outputs": [out_name],
                        "weight": w_key,
                        "bias": b_key,
                        "shape": {
                            "N": N_, "C": IC,
                            "IH": IH, "IW": IW,
                            "OH": OH, "OW": OW,
                            "KH": KH, "KW": KW,
                            "SH": SH, "SW": SW,
                            "PH": PH, "PW": PW,
                        },
                    })
                else:
                    raise NotImplementedError(
                        f"Conv2d with groups={mod.groups} (IC={IC}, OC={OC}) "
                        f"not supported — only groups=1 (standard) and "
                        f"groups=IC=OC (depthwise) are wired up at "
                        f"{node.name}"
                    )

            elif isinstance(mod, torch.nn.Conv1d):
                # 1D conv → conv2d with a unit height dim: input [N,C,L] is
                # [N,C,1,L]; weight [OC,IC,K] → [OC,IC,1,K]; KH=1, KW=K.
                # Dilation maps to DW (the height dim is degenerate, DH=1).
                (Dl,) = _as_tuple1(mod.dilation)
                in_shape = [int(s) for s in tensors[in_name]["shape"]]
                out_shape = [int(s) for s in tensors[out_name]["shape"]]
                N_, IC, IW = in_shape
                OC, OW = out_shape[1], out_shape[2]
                (K,) = _as_tuple1(mod.kernel_size)
                (S,) = _as_tuple1(mod.stride)
                (P,) = _as_tuple1(mod.padding)
                w_key = f"{node.target}.weight"
                b_key = f"{node.target}.bias" if mod.bias is not None else None
                w = mod.weight.detach().cpu().numpy().astype(weight_dtype)
                weights[w_key] = w.reshape(w.shape[0], w.shape[1], 1, w.shape[2])
                if b_key is not None:
                    weights[b_key] = mod.bias.detach().cpu().numpy().astype(weight_dtype)
                if mod.groups == 1:
                    ops.append({
                        "name": str(node.target), "op": "conv2d",
                        "inputs": [in_name], "outputs": [out_name],
                        "weight": w_key, "bias": b_key,
                        "shape": {"N": N_, "IC": IC, "IH": 1, "IW": IW,
                                  "OC": OC, "OH": 1, "OW": OW,
                                  "KH": 1, "KW": K, "SH": 1, "SW": S,
                                  "PH": 0, "PW": P, "DH": 1, "DW": Dl},
                    })
                elif mod.groups == IC and IC == OC and Dl == 1:
                    ops.append({
                        "name": str(node.target), "op": "conv2d_dw",
                        "inputs": [in_name], "outputs": [out_name],
                        "weight": w_key, "bias": b_key,
                        "shape": {"N": N_, "C": IC, "IH": 1, "IW": IW,
                                  "OH": 1, "OW": OW, "KH": 1, "KW": K,
                                  "SH": 1, "SW": S, "PH": 0, "PW": P},
                    })
                else:
                    raise NotImplementedError(
                        f"Conv1d groups={mod.groups} (IC={IC},OC={OC}) "
                        f"dilation={Dl} at {node.name}: only standard and "
                        f"non-dilated depthwise wired")

            elif isinstance(mod, torch.nn.ConvTranspose1d):
                # 1D transposed conv → conv_transpose2d, unit height dim.
                in_shape = [int(s) for s in tensors[in_name]["shape"]]
                out_shape = [int(s) for s in tensors[out_name]["shape"]]
                N_, IC, IW = in_shape
                OC, OW = out_shape[1], out_shape[2]
                (K,) = _as_tuple1(mod.kernel_size)
                (S,) = _as_tuple1(mod.stride)
                (P,) = _as_tuple1(mod.padding)
                (Dl,) = _as_tuple1(mod.dilation)
                w_key = f"{node.target}.weight"
                b_key = f"{node.target}.bias" if mod.bias is not None else None
                w = mod.weight.detach().cpu().numpy().astype(weight_dtype)
                weights[w_key] = w.reshape(w.shape[0], w.shape[1], 1, w.shape[2])
                if b_key is not None:
                    weights[b_key] = mod.bias.detach().cpu().numpy().astype(weight_dtype)
                ops.append({
                    "name": str(node.target), "op": "conv_transpose2d",
                    "inputs": [in_name], "outputs": [out_name],
                    "weight": w_key, "bias": b_key,
                    "shape": {"N": N_, "IC": IC, "IH": 1, "IW": IW,
                              "OC": OC, "OH": 1, "OW": OW,
                              "KH": 1, "KW": K, "SH": 1, "SW": S,
                              "PH": 0, "PW": P, "DH": 1, "DW": Dl,
                              "G": int(mod.groups)},
                })

            elif isinstance(mod, torch.nn.ConvTranspose2d):
                # Transposed conv. Weight is [IC, OC/G, KH, KW]. OH/OW come from
                # the traced output shape, so output_padding is already folded
                # in. The rvv backend repacks this 4D weight to IHWO like any
                # conv weight; the kernel honors that via its IHWOC branch.
                w_key = f"{node.target}.weight"
                b_key = f"{node.target}.bias" if mod.bias is not None else None
                weights[w_key] = mod.weight.detach().cpu().numpy().astype(weight_dtype)
                if b_key is not None:
                    weights[b_key] = mod.bias.detach().cpu().numpy().astype(weight_dtype)
                in_shape = tensors[in_name]["shape"]
                out_shape = tensors[out_name]["shape"]
                N_, IC, IH, IW = (int(s) for s in in_shape)
                _, OC, OH, OW = (int(s) for s in out_shape)
                KH, KW = _pair(mod.kernel_size)
                SH, SW = _pair(mod.stride)
                PH, PW = _pair(mod.padding)
                DH, DW = _pair(mod.dilation)
                G = int(mod.groups)
                ops.append({
                    "name": str(node.target),
                    "op": "conv_transpose2d",
                    "inputs": [in_name],
                    "outputs": [out_name],
                    "weight": w_key,
                    "bias": b_key,
                    "shape": {
                        "N": N_, "IC": IC, "IH": IH, "IW": IW,
                        "OC": OC, "OH": OH, "OW": OW,
                        "KH": KH, "KW": KW,
                        "SH": SH, "SW": SW,
                        "PH": PH, "PW": PW,
                        "DH": DH, "DW": DW,
                        "G": G,
                    },
                })

            elif isinstance(mod, torch.nn.Conv3d):
                # 3D conv (NCDHW). Weight [OC, IC/G, KD, KH, KW] is 5D so the
                # backend leaves it in OIDHW (no IHWO repack for 5D).
                w_key = f"{node.target}.weight"
                b_key = f"{node.target}.bias" if mod.bias is not None else None
                weights[w_key] = mod.weight.detach().cpu().numpy().astype(weight_dtype)
                if b_key is not None:
                    weights[b_key] = mod.bias.detach().cpu().numpy().astype(weight_dtype)
                N_, IC, ID, IH, IW = (int(s) for s in tensors[in_name]["shape"])
                _, OC, OD, OH, OW = (int(s) for s in tensors[out_name]["shape"])
                KD, KH, KW = _triple(mod.kernel_size)
                SD, SH, SW = _triple(mod.stride)
                PD, PH, PW = _triple(mod.padding)
                DD, DH, DW = _triple(mod.dilation)
                ops.append({
                    "name": str(node.target), "op": "conv3d",
                    "inputs": [in_name], "outputs": [out_name],
                    "weight": w_key, "bias": b_key,
                    "shape": {"N": N_, "IC": IC, "ID": ID, "IH": IH, "IW": IW,
                              "OC": OC, "OD": OD, "OH": OH, "OW": OW,
                              "KD": KD, "KH": KH, "KW": KW,
                              "SD": SD, "SH": SH, "SW": SW,
                              "PD": PD, "PH": PH, "PW": PW,
                              "DD": DD, "DH": DH, "DW": DW, "G": int(mod.groups)},
                })

            elif isinstance(mod, torch.nn.ConvTranspose3d):
                # 3D transposed conv. Weight [IC, OC/G, KD, KH, KW] (5D). OD/OH/
                # OW come from the traced output shape (output_padding folded in).
                w_key = f"{node.target}.weight"
                b_key = f"{node.target}.bias" if mod.bias is not None else None
                weights[w_key] = mod.weight.detach().cpu().numpy().astype(weight_dtype)
                if b_key is not None:
                    weights[b_key] = mod.bias.detach().cpu().numpy().astype(weight_dtype)
                N_, IC, ID, IH, IW = (int(s) for s in tensors[in_name]["shape"])
                _, OC, OD, OH, OW = (int(s) for s in tensors[out_name]["shape"])
                KD, KH, KW = _triple(mod.kernel_size)
                SD, SH, SW = _triple(mod.stride)
                PD, PH, PW = _triple(mod.padding)
                DD, DH, DW = _triple(mod.dilation)
                ops.append({
                    "name": str(node.target), "op": "conv_transpose3d",
                    "inputs": [in_name], "outputs": [out_name],
                    "weight": w_key, "bias": b_key,
                    "shape": {"N": N_, "IC": IC, "ID": ID, "IH": IH, "IW": IW,
                              "OC": OC, "OD": OD, "OH": OH, "OW": OW,
                              "KD": KD, "KH": KH, "KW": KW,
                              "SD": SD, "SH": SH, "SW": SW,
                              "PD": PD, "PH": PH, "PW": PW,
                              "DD": DD, "DH": DH, "DW": DW, "G": int(mod.groups)},
                })

            elif isinstance(mod, torch.nn.MaxPool3d):
                N_, C, ID, IH, IW = (int(s) for s in tensors[in_name]["shape"])
                _, _, OD, OH, OW = (int(s) for s in tensors[out_name]["shape"])
                KD, KH, KW = _triple(mod.kernel_size)
                SD, SH, SW = _triple(mod.stride if mod.stride is not None
                                     else mod.kernel_size)
                PD, PH, PW = _triple(mod.padding)
                DD, DH, DW = _triple(mod.dilation)
                ops.append({
                    "name": str(node.target), "op": "maxpool3d",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"N": N_, "C": C, "ID": ID, "IH": IH, "IW": IW,
                              "OD": OD, "OH": OH, "OW": OW,
                              "KD": KD, "KH": KH, "KW": KW,
                              "SD": SD, "SH": SH, "SW": SW,
                              "PD": PD, "PH": PH, "PW": PW,
                              "DD": DD, "DH": DH, "DW": DW},
                })

            elif isinstance(mod, torch.nn.AvgPool3d):
                N_, C, ID, IH, IW = (int(s) for s in tensors[in_name]["shape"])
                _, _, OD, OH, OW = (int(s) for s in tensors[out_name]["shape"])
                KD, KH, KW = _triple(mod.kernel_size)
                SD, SH, SW = _triple(mod.stride if mod.stride is not None
                                     else mod.kernel_size)
                PD, PH, PW = _triple(mod.padding)
                if not mod.count_include_pad and (PD or PH or PW):
                    raise NotImplementedError(
                        f"AvgPool3d count_include_pad=False with padding at "
                        f"{node.name}")
                ops.append({
                    "name": str(node.target), "op": "avgpool3d",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"N": N_, "C": C, "ID": ID, "IH": IH, "IW": IW,
                              "OD": OD, "OH": OH, "OW": OW,
                              "KD": KD, "KH": KH, "KW": KW,
                              "SD": SD, "SH": SH, "SW": SW,
                              "PD": PD, "PH": PH, "PW": PW},
                })

            elif isinstance(mod, torch.nn.MaxPool2d):
                in_shape = tensors[in_name]["shape"]
                out_shape = tensors[out_name]["shape"]
                N_, C, IH, IW = (int(s) for s in in_shape)
                _, _, OH, OW = (int(s) for s in out_shape)
                KH, KW = _pair(mod.kernel_size)
                SH, SW = _pair(mod.stride)
                PH, PW = _pair(mod.padding)
                DH, DW = _pair(mod.dilation)
                ops.append({
                    "name": str(node.target),
                    "op": "maxpool2d",
                    "inputs": [in_name],
                    "outputs": [out_name],
                    "shape": {
                        "N": N_, "C": C,
                        "IH": IH, "IW": IW,
                        "OH": OH, "OW": OW,
                        "KH": KH, "KW": KW,
                        "SH": SH, "SW": SW,
                        "PH": PH, "PW": PW,
                        "DH": DH, "DW": DW,
                    },
                })

            elif isinstance(mod, torch.nn.AvgPool2d):
                in_shape = tensors[in_name]["shape"]
                out_shape = tensors[out_name]["shape"]
                N_, C, IH, IW = (int(s) for s in in_shape)
                _, _, OH, OW = (int(s) for s in out_shape)
                KH, KW = _pair(mod.kernel_size)
                # nn.AvgPool2d.stride defaults to None → same as kernel_size.
                SH, SW = _pair(mod.stride if mod.stride is not None
                               else mod.kernel_size)
                PH, PW = _pair(mod.padding)
                if not mod.count_include_pad and (PH or PW):
                    raise NotImplementedError(
                        f"AvgPool2d count_include_pad=False with padding is not "
                        f"supported at {node.name}")
                ops.append({
                    "name": str(node.target),
                    "op": "avgpool2d",
                    "inputs": [in_name],
                    "outputs": [out_name],
                    "shape": {
                        "N": N_, "C": C,
                        "IH": IH, "IW": IW,
                        "OH": OH, "OW": OW,
                        "KH": KH, "KW": KW,
                        "SH": SH, "SW": SW,
                        "PH": PH, "PW": PW,
                    },
                })

            elif isinstance(mod, torch.nn.MaxPool1d):
                # 1D max pool → maxpool2d with a unit height dim.
                in_shape = [int(s) for s in tensors[in_name]["shape"]]
                out_shape = [int(s) for s in tensors[out_name]["shape"]]
                N_, C, IW = in_shape
                OW = out_shape[2]
                (K,) = _as_tuple1(mod.kernel_size)
                (S,) = _as_tuple1(mod.stride if mod.stride is not None
                                  else mod.kernel_size)
                (P,) = _as_tuple1(mod.padding)
                (Dl,) = _as_tuple1(mod.dilation)
                ops.append({
                    "name": str(node.target), "op": "maxpool2d",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"N": N_, "C": C, "IH": 1, "IW": IW,
                              "OH": 1, "OW": OW, "KH": 1, "KW": K,
                              "SH": 1, "SW": S, "PH": 0, "PW": P,
                              "DH": 1, "DW": Dl},
                })

            elif isinstance(mod, torch.nn.AvgPool1d):
                # 1D average pool → avgpool2d with a unit height dim.
                in_shape = [int(s) for s in tensors[in_name]["shape"]]
                out_shape = [int(s) for s in tensors[out_name]["shape"]]
                N_, C, IW = in_shape
                OW = out_shape[2]
                (K,) = _as_tuple1(mod.kernel_size)
                (S,) = _as_tuple1(mod.stride if mod.stride is not None
                                  else mod.kernel_size)
                (P,) = _as_tuple1(mod.padding)
                ops.append({
                    "name": str(node.target), "op": "avgpool2d",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"N": N_, "C": C, "IH": 1, "IW": IW,
                              "OH": 1, "OW": OW, "KH": 1, "KW": K,
                              "SH": 1, "SW": S, "PH": 0, "PW": P},
                })

            elif isinstance(mod, torch.nn.LayerNorm):
                # LayerNorm over the trailing normalized_shape dims. K = product
                # of normalized_shape, M = product of the leading dims. gamma /
                # beta are flattened to length K (ones/zeros if affine is off).
                in_shape = [int(s) for s in tensors[in_name]["shape"]]
                ns = tuple(int(s) for s in mod.normalized_shape)
                K = int(np.prod(ns)) if ns else 1
                total = int(np.prod(in_shape))
                if K == 0 or total % K != 0:
                    raise NotImplementedError(
                        f"LayerNorm at {node.name}: normalized_shape {ns} does "
                        f"not divide input {in_shape}")
                M = total // K
                g_key = f"{node.target}.weight"
                b_key = f"{node.target}.bias"
                if mod.elementwise_affine and mod.weight is not None:
                    weights[g_key] = mod.weight.detach().cpu().numpy(
                        ).astype(weight_dtype).reshape(-1)
                    weights[b_key] = mod.bias.detach().cpu().numpy(
                        ).astype(weight_dtype).reshape(-1)
                else:
                    weights[g_key] = np.ones(K, dtype=weight_dtype)
                    weights[b_key] = np.zeros(K, dtype=weight_dtype)
                ops.append({
                    "name": str(node.target),
                    "op": "layer_norm",
                    "inputs": [in_name],
                    "outputs": [out_name],
                    "gamma_key": g_key,
                    "beta_key": b_key,
                    "eps": float(mod.eps),
                    "shape": {"M": M, "K": K},
                })

            elif isinstance(mod, (torch.nn.GroupNorm, torch.nn.InstanceNorm2d)):
                # Both normalize over (channels-in-group × spatial) per sample;
                # InstanceNorm2d is the num_groups == num_channels case. gamma /
                # beta are per-channel (ones/zeros when affine is off).
                in_shape = [int(s) for s in tensors[in_name]["shape"]]
                N_ = in_shape[0]
                C = in_shape[1]
                HW = int(np.prod(in_shape[2:])) if len(in_shape) > 2 else 1
                if isinstance(mod, torch.nn.GroupNorm):
                    G = int(mod.num_groups)
                    affine = bool(mod.affine)
                    eps = float(mod.eps)
                else:  # InstanceNorm2d: one group per channel
                    G = C
                    affine = bool(mod.affine)
                    eps = float(mod.eps)
                if C % G != 0:
                    raise NotImplementedError(
                        f"group_norm at {node.name}: C={C} not divisible by "
                        f"G={G}")
                g_key = f"{node.target}.weight"
                b_key = f"{node.target}.bias"
                if affine and getattr(mod, "weight", None) is not None:
                    weights[g_key] = mod.weight.detach().cpu().numpy(
                        ).astype(weight_dtype).reshape(-1)
                    weights[b_key] = mod.bias.detach().cpu().numpy(
                        ).astype(weight_dtype).reshape(-1)
                else:
                    weights[g_key] = np.ones(C, dtype=weight_dtype)
                    weights[b_key] = np.zeros(C, dtype=weight_dtype)
                ops.append({
                    "name": str(node.target),
                    "op": "group_norm",
                    "inputs": [in_name],
                    "outputs": [out_name],
                    "gamma_key": g_key,
                    "beta_key": b_key,
                    "eps": eps,
                    "shape": {"N": N_, "C": C, "G": G, "HW": HW},
                })

            elif isinstance(mod, torch.nn.TripletMarginLoss):
                # 3 inputs (anchor, positive, negative), p=2, reduction=mean.
                a_name = node.args[0].name
                p_name = node.args[1].name
                n_name = node.args[2].name
                a_shape = [int(s) for s in tensors[a_name]["shape"]]
                B = a_shape[0]
                Feat = int(np.prod(a_shape[1:])) if len(a_shape) > 1 else 1
                ops.append({
                    "name": str(node.target), "op": "triplet_loss",
                    "inputs": [a_name, p_name, n_name],
                    "outputs": [out_name],
                    "margin": float(mod.margin),
                    "shape": {"B": B, "F": Feat},
                })

            elif isinstance(mod, torch.nn.Dropout):
                # Eval-mode dropout is identity. Record as a view: the output
                # tensor aliases the input.
                ops.append({
                    "name": str(node.target),
                    "op": "view",
                    "inputs": [in_name],
                    "outputs": [out_name],
                    "shape": {"n": int(np.prod(tensors[in_name]["shape"]))},
                })

            elif isinstance(mod, torch.nn.BatchNorm2d):
                # Pre-fold the running statistics + affine into a single
                # per-channel (scale, bias) pair so the runtime kernel only
                # needs to do one multiply-add per element.
                gamma = mod.weight.detach().cpu().numpy().astype(np.float32) \
                    if mod.weight is not None \
                    else np.ones((mod.num_features,), dtype=np.float32)
                beta = mod.bias.detach().cpu().numpy().astype(np.float32) \
                    if mod.bias is not None \
                    else np.zeros((mod.num_features,), dtype=np.float32)
                mean = mod.running_mean.detach().cpu().numpy().astype(np.float32)
                var = mod.running_var.detach().cpu().numpy().astype(np.float32)
                eps = float(mod.eps)
                scale = gamma / np.sqrt(var + eps)
                bias_fused = beta - mean * scale
                s_key = f"{node.target}.scale"
                b_key = f"{node.target}.bias_fused"
                # Fold in fp32 for accuracy, cast to weight_dtype at save.
                weights[s_key] = scale.astype(weight_dtype)
                weights[b_key] = bias_fused.astype(weight_dtype)
                in_shape = tensors[in_name]["shape"]
                N_, C, H, W = (int(s) for s in in_shape)
                ops.append({
                    "name": str(node.target),
                    "op": "batchnorm2d",
                    "inputs": [in_name],
                    "outputs": [out_name],
                    "weight": s_key,
                    "bias": b_key,
                    "shape": {"N": N_, "C": C, "H": H, "W": W},
                })

            elif isinstance(mod, torch.nn.Sigmoid):
                n = int(np.prod(tensors[in_name]["shape"]))
                ops.append({
                    "name": str(node.target),
                    "op": "sigmoid",
                    "inputs": [in_name],
                    "outputs": [out_name],
                    "shape": {"n": n},
                })

            elif isinstance(mod, torch.nn.ReLU6):
                n = int(np.prod(tensors[in_name]["shape"]))
                ops.append({
                    "name": str(node.target),
                    "op": "relu6",
                    "inputs": [in_name],
                    "outputs": [out_name],
                    "shape": {"n": n},
                })

            elif isinstance(mod, torch.nn.AdaptiveAvgPool2d):
                in_shape = tensors[in_name]["shape"]
                N_, C, IH, IW = (int(s) for s in in_shape)
                out_shape = tensors[out_name]["shape"]
                # Only output_size=(1,1) is wired up — that's what classifier
                # heads use. Detect by checking the output spatial dims.
                out_h = int(out_shape[2]) if len(out_shape) >= 4 else 1
                out_w = int(out_shape[3]) if len(out_shape) >= 4 else 1
                if out_h != 1 or out_w != 1:
                    raise NotImplementedError(
                        f"AdaptiveAvgPool2d only supports output_size=(1,1) "
                        f"for now; got {(out_h, out_w)} at {node.name}"
                    )
                ops.append({
                    "name": str(node.target),
                    "op": "adaptive_avg_pool2d",
                    "inputs": [in_name],
                    "outputs": [out_name],
                    "shape": {"N": N_, "C": C, "IH": IH, "IW": IW},
                })

            # KernelBench Phase 2 activation modules. All pointwise — same
            # IR shape as the existing ReLU / Sigmoid / ELU module branches.
            elif isinstance(mod, torch.nn.LeakyReLU):
                n = int(np.prod(tensors[in_name]["shape"]))
                ops.append({
                    "name": str(node.target), "op": "leaky_relu",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"n": n},
                    "negative_slope": float(mod.negative_slope),
                })
            elif isinstance(mod, torch.nn.Tanh):
                n = int(np.prod(tensors[in_name]["shape"]))
                ops.append({
                    "name": str(node.target), "op": "tanh",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"n": n},
                })
            elif isinstance(mod, torch.nn.GELU):
                n = int(np.prod(tensors[in_name]["shape"]))
                ops.append({
                    "name": str(node.target), "op": "gelu",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"n": n},
                })
            elif isinstance(mod, torch.nn.SELU):
                n = int(np.prod(tensors[in_name]["shape"]))
                ops.append({
                    "name": str(node.target), "op": "selu",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"n": n},
                })
            elif isinstance(mod, torch.nn.Hardsigmoid):
                n = int(np.prod(tensors[in_name]["shape"]))
                ops.append({
                    "name": str(node.target), "op": "hardsigmoid",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"n": n},
                })
            elif isinstance(mod, torch.nn.Softplus):
                n = int(np.prod(tensors[in_name]["shape"]))
                ops.append({
                    "name": str(node.target), "op": "softplus",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"n": n},
                })
            elif isinstance(mod, torch.nn.Softsign):
                n = int(np.prod(tensors[in_name]["shape"]))
                ops.append({
                    "name": str(node.target), "op": "softsign",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"n": n},
                })
            elif isinstance(mod, torch.nn.Hardtanh):
                n = int(np.prod(tensors[in_name]["shape"]))
                ops.append({
                    "name": str(node.target), "op": "hardtanh",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"n": n},
                    "min_val": float(mod.min_val),
                    "max_val": float(mod.max_val),
                })

            elif isinstance(mod, torch.nn.SiLU):
                # SiLU = x * sigmoid(x). Pointwise; same IR shape as ReLU.
                # YOLOv8's Conv block ends with this.
                n = int(np.prod(tensors[in_name]["shape"]))
                ops.append({
                    "name": str(node.target), "op": "silu",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"n": n},
                })

            elif isinstance(mod, torch.nn.Upsample):
                # Only nearest-neighbor with integer scale_factor is wired
                # up — that's what YOLOv8's neck uses (×2 nearest). Bilinear
                # / arbitrary scales would need a different kernel.
                if mod.mode != "nearest":
                    raise NotImplementedError(
                        f"Upsample mode={mod.mode!r} at {node.name}: only "
                        f"'nearest' is supported."
                    )
                sf = mod.scale_factor
                if sf is None or float(sf) != int(sf):
                    raise NotImplementedError(
                        f"Upsample scale_factor={sf} at {node.name}: only "
                        f"integer scales are supported."
                    )
                sf = int(sf)
                in_shape = tensors[in_name]["shape"]
                N_, C, IH, IW = (int(s) for s in in_shape)
                ops.append({
                    "name": str(node.target), "op": "upsample_nearest",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"N": N_, "C": C, "IH": IH, "IW": IW,
                              "scale": sf},
                })

            elif isinstance(mod, torch.nn.LSTM):
                # fp32 mirror of the int8 lstm_s8 decomposition: one `lstm` op
                # per layer, chained via its output tensor, with persistent
                # float h/c state (seq-len-1 unroll; state persists between
                # run_model calls, BSS-zeroed => correct step-0 init).
                if mod.bidirectional:
                    raise NotImplementedError("fp32 LSTM: bidirectional unsupported")
                L = int(mod.num_layers)
                H = int(mod.hidden_size)
                in_size0 = int(tensors[in_name]["shape"][-1])
                gi0 = _find_getitem_consumer(node, 0)
                gi1 = _find_getitem_consumer(node, 1)
                final_name = gi0.name if gi0 is not None else node.name
                if gi0 is not None:
                    _skip_nodes.add(gi0)
                if gi1 is not None:
                    _skip_nodes.add(gi1)
                prefix = node.name
                x_name = in_name
                for lyr in range(L):
                    wih = getattr(mod, f"weight_ih_l{lyr}").detach()
                    whh = getattr(mod, f"weight_hh_l{lyr}").detach()
                    bias = torch.zeros(4 * H)
                    bih = getattr(mod, f"bias_ih_l{lyr}", None)
                    bhh = getattr(mod, f"bias_hh_l{lyr}", None)
                    if bih is not None:
                        bias = bias + bih.detach()
                    if bhh is not None:
                        bias = bias + bhh.detach()
                    in_l = in_size0 if lyr == 0 else H
                    wih_key = f"{prefix}_wih_l{lyr}"
                    whh_key = f"{prefix}_whh_l{lyr}"
                    bias_key = f"{prefix}_bias_l{lyr}"
                    weights[wih_key] = wih.cpu().numpy().reshape(
                        4 * H, in_l).astype(weight_dtype)
                    weights[whh_key] = whh.cpu().numpy().reshape(
                        4 * H, H).astype(weight_dtype)
                    weights[bias_key] = bias.cpu().numpy().astype(weight_dtype)
                    h_name = f"{prefix}_h_l{lyr}"
                    c_name = f"{prefix}_c_l{lyr}"
                    out_nm = final_name if lyr == L - 1 else f"{prefix}_out_l{lyr}"
                    dt = "f16" if weight_dtype == np.float16 else "f32"
                    for nm in (h_name, c_name, out_nm):
                        tensors[nm] = {"shape": [1, H], "dtype": dt, "quant": None}
                    ops.append({
                        "name": f"{node.target}_l{lyr}",
                        "op": "lstm",   # _f16 suffix added by the rename pass below
                        "inputs": [x_name],
                        "outputs": [out_nm],
                        "state": {"h": h_name, "c": c_name},
                        "weight_ih": wih_key,
                        "weight_hh": whh_key,
                        "bias": bias_key,
                        "shape": {"in_size": in_l, "H": H},
                    })
                    x_name = out_nm

        elif node.op == "call_function":
            out_name = node.name
            target = node.target
            tname = getattr(target, "__name__", str(target))
            # `_tensor_meta` only works for tensor outputs. torch.max /
            # torch.min with dim return NamedTuples (TensorMetadata is a
            # list, no `.shape`). The branches that handle them populate
            # tensors[out_name] manually; for everything else, run the
            # auto-call up front.
            _named_tuple_targets = (torch.max, torch.min)
            _is_named_tuple_output = (
                target in _named_tuple_targets
                and (len(node.args) > 1 or "dim" in (node.kwargs or {}))
            )
            if not _is_named_tuple_output:
                tensors[out_name] = _tensor_meta(node)
            if tname == "relu" or target is torch.relu or target is torch.nn.functional.relu:
                in_name = node.args[0].name
                n = int(np.prod(tensors[in_name]["shape"]))
                ops.append({
                    "name": node.name,
                    "op": "relu",
                    "inputs": [in_name],
                    "outputs": [out_name],
                    "shape": {"n": n},
                })
            elif tname == "flatten" or target is torch.flatten:
                # Reshape that doesn't move bytes — emitted as a view in C.
                in_name = node.args[0].name
                n = int(np.prod(tensors[in_name]["shape"]))
                ops.append({
                    "name": node.name,
                    "op": "view",
                    "inputs": [in_name],
                    "outputs": [out_name],
                    "shape": {"n": n},
                })
            elif tname == "add" or target is torch.add or target is __import__("operator").add:
                # 2-input elementwise add (residual connection).
                a_name = node.args[0].name
                b_name = node.args[1].name
                a_shape = tensors[a_name]["shape"]
                b_shape = tensors[b_name]["shape"]
                if a_shape != b_shape:
                    raise NotImplementedError(
                        f"add at {node.name}: broadcasting not supported "
                        f"(a={a_shape} b={b_shape})"
                    )
                n = int(np.prod(a_shape))
                ops.append({
                    "name": node.name,
                    "op": "add",
                    "inputs": [a_name, b_name],
                    "outputs": [out_name],
                    "shape": {"n": n},
                })
            elif tname == "adaptive_avg_pool2d" or \
                    target is torch.nn.functional.adaptive_avg_pool2d:
                # Functional global avg pool. Only output_size=(1,1) is wired.
                in_name = node.args[0].name
                in_shape = tensors[in_name]["shape"]
                N_, C, IH, IW = (int(s) for s in in_shape)
                out_shape = tensors[out_name]["shape"]
                out_h = int(out_shape[2]) if len(out_shape) >= 4 else 1
                out_w = int(out_shape[3]) if len(out_shape) >= 4 else 1
                if out_h != 1 or out_w != 1:
                    raise NotImplementedError(
                        f"adaptive_avg_pool2d only supports output_size=(1,1) "
                        f"for now; got {(out_h, out_w)} at {node.name}"
                    )
                ops.append({
                    "name": node.name,
                    "op": "adaptive_avg_pool2d",
                    "inputs": [in_name],
                    "outputs": [out_name],
                    "shape": {"N": N_, "C": C, "IH": IH, "IW": IW},
                })
            elif tname == "relu6" or \
                    target is torch.nn.functional.relu6:
                in_name = node.args[0].name
                n = int(np.prod(tensors[in_name]["shape"]))
                ops.append({
                    "name": node.name,
                    "op": "relu6",
                    "inputs": [in_name],
                    "outputs": [out_name],
                    "shape": {"n": n},
                })
            elif tname == "sigmoid" or target is torch.sigmoid \
                    or target is torch.nn.functional.sigmoid:
                # KernelBench 21_Sigmoid uses torch.sigmoid (functional);
                # mirror nn.Sigmoid module-side handling.
                in_name = node.args[0].name
                n = int(np.prod(tensors[in_name]["shape"]))
                ops.append({
                    "name": node.name,
                    "op": "sigmoid",
                    "inputs": [in_name],
                    "outputs": [out_name],
                    "shape": {"n": n},
                })
            elif tname == "softmax" or target is torch.softmax \
                    or target is torch.nn.functional.softmax:
                # KernelBench 23_Softmax: torch.softmax(x, dim=...). The
                # reference kernel normalizes over contiguous rows, so only a
                # last-axis softmax is supported (M = leading dims, K = last).
                in_name = node.args[0].name
                in_shape = [int(s) for s in tensors[in_name]["shape"]]
                ndim = len(in_shape)
                dim = node.kwargs.get("dim") if node.kwargs else None
                if dim is None and len(node.args) > 1:
                    dim = node.args[1]
                if dim is None:
                    raise NotImplementedError(
                        f"softmax at {node.name}: implicit dim not supported")
                dim = int(dim)
                if dim < 0:
                    dim += ndim
                if dim != ndim - 1:
                    raise NotImplementedError(
                        f"softmax at {node.name}: only last-axis softmax "
                        f"supported (got dim={dim} of {ndim}D)")
                K = in_shape[-1]
                M = int(np.prod(in_shape[:-1])) if ndim > 1 else 1
                ops.append({
                    "name": node.name,
                    "op": "softmax",
                    "inputs": [in_name],
                    "outputs": [out_name],
                    "shape": {"M": M, "K": K},
                })
            elif tname == "log_softmax" or target is torch.log_softmax \
                    or target is torch.nn.functional.log_softmax:
                # KernelBench 24_LogSoftmax. Last-axis only, as with softmax.
                in_name = node.args[0].name
                in_shape = [int(s) for s in tensors[in_name]["shape"]]
                ndim = len(in_shape)
                dim = node.kwargs.get("dim") if node.kwargs else None
                if dim is None and len(node.args) > 1:
                    dim = node.args[1]
                if dim is None:
                    raise NotImplementedError(
                        f"log_softmax at {node.name}: implicit dim not supported")
                dim = int(dim)
                if dim < 0:
                    dim += ndim
                if dim != ndim - 1:
                    raise NotImplementedError(
                        f"log_softmax at {node.name}: only last-axis supported "
                        f"(got dim={dim} of {ndim}D)")
                K = in_shape[-1]
                M = int(np.prod(in_shape[:-1])) if ndim > 1 else 1
                ops.append({
                    "name": node.name,
                    "op": "log_softmax",
                    "inputs": [in_name],
                    "outputs": [out_name],
                    "shape": {"M": M, "K": K},
                })
            elif tname == "log" or target is torch.log:
                # Elementwise natural log (used before F.kl_div, KernelBench 98).
                in_name = node.args[0].name
                n = int(np.prod(tensors[in_name]["shape"]))
                ops.append({
                    "name": node.name, "op": "log",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"n": n},
                })
            elif tname == "smooth_l1_loss" \
                    or target is torch.nn.functional.smooth_l1_loss:
                # Huber / smooth-L1 loss (KernelBench 96). beta defaults to 1.0.
                a_name = node.args[0].name
                b_name = node.args[1].name
                n = int(np.prod(tensors[a_name]["shape"]))
                beta = float((node.kwargs or {}).get("beta", 1.0))
                ops.append({
                    "name": node.name, "op": "huber_loss",
                    "inputs": [a_name, b_name], "outputs": [out_name],
                    "beta": beta, "shape": {"n": n},
                })
            elif tname == "cross_entropy" \
                    or target is torch.nn.functional.cross_entropy:
                # Mean cross-entropy (KernelBench 95). logits [N, C], targets [N]
                # class indices (carried as floats in the flat io buffer).
                a_name = node.args[0].name
                b_name = node.args[1].name
                a_shape = [int(s) for s in tensors[a_name]["shape"]]
                if len(a_shape) != 2:
                    raise NotImplementedError(
                        f"cross_entropy at {node.name}: only [N, C] logits "
                        f"supported (got {a_shape})")
                ops.append({
                    "name": node.name, "op": "cross_entropy_loss",
                    "inputs": [a_name, b_name], "outputs": [out_name],
                    "shape": {"N": a_shape[0], "C": a_shape[1]},
                })
            elif tname == "scaled_dot_product_attention" \
                    or target is torch.nn.functional.scaled_dot_product_attention:
                # SDPA (KernelBench 97). Q/K/V are [B, H, S, D]; flatten B*H.
                q_name = node.args[0].name
                k_name = node.args[1].name
                v_name = node.args[2].name
                q_shape = [int(s) for s in tensors[q_name]["shape"]]
                if len(q_shape) != 4:
                    raise NotImplementedError(
                        f"sdpa at {node.name}: only [B,H,S,D] supported "
                        f"(got {q_shape})")
                B, H, S, D = q_shape
                ops.append({
                    "name": node.name, "op": "sdpa",
                    "inputs": [q_name, k_name, v_name], "outputs": [out_name],
                    "shape": {"BH": B * H, "S": S, "D": D},
                })
            elif tname == "kl_div" or target is torch.nn.functional.kl_div:
                # KL divergence, reduction='batchmean' (KernelBench 98). First
                # arg is already log(pred).
                a_name = node.args[0].name
                b_name = node.args[1].name
                a_shape = [int(s) for s in tensors[a_name]["shape"]]
                red = (node.kwargs or {}).get("reduction", "mean")
                if red != "batchmean":
                    raise NotImplementedError(
                        f"kl_div at {node.name}: only reduction='batchmean' "
                        f"supported (got {red!r})")
                N = a_shape[0]
                C = int(np.prod(a_shape[1:])) if len(a_shape) > 1 else 1
                ops.append({
                    "name": node.name, "op": "kldiv_loss",
                    "inputs": [a_name, b_name], "outputs": [out_name],
                    "shape": {"N": N, "C": C},
                })
            elif tname == "elu" or target is torch.nn.functional.elu:
                # KernelBench 31_ELU may use functional too. nn.ELU's
                # alpha defaults to 1.0; functional takes alpha kwarg.
                in_name = node.args[0].name
                n = int(np.prod(tensors[in_name]["shape"]))
                alpha = float(node.kwargs.get("alpha", 1.0)) if node.kwargs else 1.0
                if alpha != 1.0:
                    raise NotImplementedError(
                        f"elu at {node.name}: alpha={alpha} != 1.0 not yet wired"
                    )
                ops.append({
                    "name": node.name,
                    "op": "elu",
                    "inputs": [in_name],
                    "outputs": [out_name],
                    "shape": {"n": n},
                })
            # KernelBench Phase 2 activations — single-call-function shapes.
            # Multi-op activations (Swish, Softsign, MinGPTNewGelu) are
            # handled by a post-trace recognizer below since their forward
            # is composed of multiple primitive ops in the FX graph.
            elif (tname == "leaky_relu"
                    or target is torch.nn.functional.leaky_relu):
                in_name = node.args[0].name
                n = int(np.prod(tensors[in_name]["shape"]))
                # functional surface uses kwarg "negative_slope" (default 0.01).
                neg_slope = 0.01
                if node.kwargs and "negative_slope" in node.kwargs:
                    neg_slope = float(node.kwargs["negative_slope"])
                elif len(node.args) > 1 and isinstance(node.args[1], (int, float)):
                    neg_slope = float(node.args[1])
                ops.append({
                    "name": node.name,
                    "op": "leaky_relu",
                    "inputs": [in_name],
                    "outputs": [out_name],
                    "shape": {"n": n},
                    "negative_slope": neg_slope,
                })
            elif tname == "tanh" or target is torch.tanh \
                    or target is torch.nn.functional.tanh:
                in_name = node.args[0].name
                n = int(np.prod(tensors[in_name]["shape"]))
                ops.append({
                    "name": node.name, "op": "tanh",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"n": n},
                })
            elif tname == "gelu" or target is torch.nn.functional.gelu:
                # PyTorch's `approximate` kwarg picks between exact (erf)
                # and the BERT / MinGPT tanh approximation. We route to
                # different kernels for the two — they agree to ~5e-4
                # but the choice matters for tight-tolerance verify.
                in_name = node.args[0].name
                n = int(np.prod(tensors[in_name]["shape"]))
                approx = "none"
                if node.kwargs and "approximate" in node.kwargs:
                    approx = str(node.kwargs["approximate"])
                op_kind = "gelu_exact" if approx == "tanh" else "gelu"
                ops.append({
                    "name": node.name, "op": op_kind,
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"n": n},
                })
            elif tname == "selu" or target is torch.selu \
                    or target is torch.nn.functional.selu:
                in_name = node.args[0].name
                n = int(np.prod(tensors[in_name]["shape"]))
                ops.append({
                    "name": node.name, "op": "selu",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"n": n},
                })
            elif tname == "hardsigmoid" \
                    or target is torch.nn.functional.hardsigmoid:
                in_name = node.args[0].name
                n = int(np.prod(tensors[in_name]["shape"]))
                ops.append({
                    "name": node.name, "op": "hardsigmoid",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"n": n},
                })
            elif tname == "softplus" \
                    or target is torch.nn.functional.softplus:
                # PyTorch defaults: beta=1, threshold=20. The reference
                # kernel uses the standard softplus formula and ignores
                # both — matches torch's output to <1e-5 on common
                # inputs since the threshold path only kicks in for
                # extremely large x.
                in_name = node.args[0].name
                n = int(np.prod(tensors[in_name]["shape"]))
                ops.append({
                    "name": node.name, "op": "softplus",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"n": n},
                })
            # KernelBench Phase 2 reductions over a single dim. The
            # 3D logical shape (outer, reduce, inner) is computed from
            # the input shape and the `dim` argument.
            elif tname in ("cumsum", "cumprod") or target in (torch.cumsum,
                                                              torch.cumprod):
                # Cumulative scan along a dim → {outer, axis, inner}.
                in_name = node.args[0].name
                in_shape = list(tensors[in_name]["shape"])
                d = node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else 0)
                d = int(d)
                if d < 0:
                    d += len(in_shape)
                outer = int(np.prod(in_shape[:d])) if d > 0 else 1
                axis = int(in_shape[d])
                inner = int(np.prod(in_shape[d+1:])) if d + 1 < len(in_shape) else 1
                op_name = "cumsum" if (tname == "cumsum" or target is torch.cumsum) \
                    else "cumprod"
                ops.append({
                    "name": node.name, "op": op_name,
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"outer": outer, "axis": axis, "inner": inner},
                })
            elif ((tname == "mul" or target in (torch.mul, _operator.mul))
                  and isinstance(node.args[0], torch.fx.Node)
                  and isinstance(node.args[1], torch.fx.Node)):
                # Elementwise multiply of two tensors (KernelBench 93 x*mask,
                # 100 pred*targets). Scalar mul (one non-Node arg) is handled
                # below. Requires equal shapes (no broadcast).
                a_name = node.args[0].name
                b_name = node.args[1].name
                a_shape = tensors[a_name]["shape"]
                b_shape = tensors[b_name]["shape"]
                if list(a_shape) != list(b_shape):
                    raise NotImplementedError(
                        f"mul at {node.name}: broadcasting not supported "
                        f"(a={a_shape} b={b_shape})")
                n = int(np.prod(a_shape))
                ops.append({
                    "name": node.name, "op": "mul",
                    "inputs": [a_name, b_name], "outputs": [out_name],
                    "shape": {"n": n},
                })
            elif (tname == "mul" or target in (torch.mul, _operator.mul)):
                # Tensor * scalar constant (KernelBench 5, A * s). Exactly one
                # arg is a Node (the tensor), the other a bound Python scalar.
                a, b = node.args[0], node.args[1]
                if isinstance(a, torch.fx.Node) and not isinstance(b, torch.fx.Node):
                    t_node, s = a, b
                elif isinstance(b, torch.fx.Node) and not isinstance(a, torch.fx.Node):
                    t_node, s = b, a
                else:
                    raise NotImplementedError(
                        f"mul at {node.name}: unexpected args {node.args}")
                n = int(np.prod(tensors[t_node.name]["shape"]))
                ops.append({
                    "name": node.name, "op": "mul_scalar",
                    "inputs": [t_node.name], "outputs": [out_name],
                    "scalar": float(s), "shape": {"n": n},
                })
            elif tname == "sum" or target is torch.sum:
                in_name = node.args[0].name
                in_shape = list(tensors[in_name]["shape"])
                dim = int(node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else 0))
                outer = int(np.prod(in_shape[:dim])) if dim > 0 else 1
                reduce = int(in_shape[dim])
                inner = int(np.prod(in_shape[dim+1:])) if dim + 1 < len(in_shape) else 1
                ops.append({
                    "name": node.name, "op": "sum_dim",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"outer": outer, "reduce": reduce, "inner": inner},
                })
            elif tname == "mean" or target is torch.mean:
                in_name = node.args[0].name
                in_shape = list(tensors[in_name]["shape"])
                dim = int(node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else 0))
                outer = int(np.prod(in_shape[:dim])) if dim > 0 else 1
                reduce = int(in_shape[dim])
                inner = int(np.prod(in_shape[dim+1:])) if dim + 1 < len(in_shape) else 1
                ops.append({
                    "name": node.name, "op": "mean_dim",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"outer": outer, "reduce": reduce, "inner": inner},
                })
            elif tname == "prod" or target is torch.prod:
                in_name = node.args[0].name
                in_shape = list(tensors[in_name]["shape"])
                dim = int(node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else 0))
                outer = int(np.prod(in_shape[:dim])) if dim > 0 else 1
                reduce = int(in_shape[dim])
                inner = int(np.prod(in_shape[dim+1:])) if dim + 1 < len(in_shape) else 1
                ops.append({
                    "name": node.name, "op": "prod_dim",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"outer": outer, "reduce": reduce, "inner": inner},
                })
            elif tname == "argmax" or target is torch.argmax:
                in_name = node.args[0].name
                in_shape = list(tensors[in_name]["shape"])
                dim = int(node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else 0))
                outer = int(np.prod(in_shape[:dim])) if dim > 0 else 1
                reduce = int(in_shape[dim])
                inner = int(np.prod(in_shape[dim+1:])) if dim + 1 < len(in_shape) else 1
                ops.append({
                    "name": node.name, "op": "argmax_dim",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"outer": outer, "reduce": reduce, "inner": inner},
                })
            elif tname == "argmin" or target is torch.argmin:
                in_name = node.args[0].name
                in_shape = list(tensors[in_name]["shape"])
                dim = int(node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else 0))
                outer = int(np.prod(in_shape[:dim])) if dim > 0 else 1
                reduce = int(in_shape[dim])
                inner = int(np.prod(in_shape[dim+1:])) if dim + 1 < len(in_shape) else 1
                ops.append({
                    "name": node.name, "op": "argmin_dim",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"outer": outer, "reduce": reduce, "inner": inner},
                })
            # torch.max / torch.min with dim return NamedTuples;
            # the bench then takes [0] for values. We emit a single
            # max_dim/min_dim op whose output is the getitem node's
            # name (skipping the intermediate NamedTuple buffer) so
            # the kernel writes directly into the model output buffer
            # when the result is the bench's final tensor. A side
            # effect is the getitem node has to be skipped when we
            # reach it — tracked via `_skip_nodes`.
            elif (tname == "max" or target is torch.max) \
                    and (len(node.args) > 1 or "dim" in (node.kwargs or {})):
                in_name = node.args[0].name
                in_shape = list(tensors[in_name]["shape"])
                dim = int(node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else 0))
                outer = int(np.prod(in_shape[:dim])) if dim > 0 else 1
                reduce = int(in_shape[dim])
                inner = int(np.prod(in_shape[dim+1:])) if dim + 1 < len(in_shape) else 1
                # Find the getitem(node, 0) consumer (must exist for the
                # values-only pattern; getitem(_, 1) is not supported).
                gi = _find_getitem_consumer(node, 0)
                if gi is None:
                    raise NotImplementedError(
                        f"torch.max with dim at {node.name}: expected a "
                        f"getitem(_, 0) consumer for the values; bare "
                        f"NamedTuple outputs aren't wired up.")
                values_name = gi.name
                _skip_nodes.add(gi)
                # Compute output shape: drop the reduced dim from input.
                out_shape = list(in_shape)
                del out_shape[dim]
                tensors[values_name] = {"shape": out_shape, "dtype": "f32",
                                        "quant": None}
                ops.append({
                    "name": node.name, "op": "max_dim",
                    "inputs": [in_name], "outputs": [values_name],
                    "shape": {"outer": outer, "reduce": reduce, "inner": inner},
                })
            elif (tname == "min" or target is torch.min) \
                    and (len(node.args) > 1 or "dim" in (node.kwargs or {})):
                in_name = node.args[0].name
                in_shape = list(tensors[in_name]["shape"])
                dim = int(node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else 0))
                outer = int(np.prod(in_shape[:dim])) if dim > 0 else 1
                reduce = int(in_shape[dim])
                inner = int(np.prod(in_shape[dim+1:])) if dim + 1 < len(in_shape) else 1
                gi = _find_getitem_consumer(node, 0)
                if gi is None:
                    raise NotImplementedError(
                        f"torch.min with dim at {node.name}: expected a "
                        f"getitem(_, 0) consumer for the values.")
                values_name = gi.name
                _skip_nodes.add(gi)
                out_shape = list(in_shape)
                del out_shape[dim]
                tensors[values_name] = {"shape": out_shape, "dtype": "f32",
                                        "quant": None}
                ops.append({
                    "name": node.name, "op": "min_dim",
                    "inputs": [in_name], "outputs": [values_name],
                    "shape": {"outer": outer, "reduce": reduce, "inner": inner},
                })
            # Compound-activation sentinels emitted by
            # _maybe_fuse_compound_activation. The fused-up subgraph
            # has been rewritten to a single call_function targeting
            # one of these tags; we just emit the matching IR op.
            elif target is _agents_compound_swish:
                in_name = node.args[0].name
                n = int(np.prod(tensors[in_name]["shape"]))
                ops.append({
                    "name": node.name, "op": "swish",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"n": n},
                })
            elif target is _agents_compound_softsign:
                in_name = node.args[0].name
                n = int(np.prod(tensors[in_name]["shape"]))
                ops.append({
                    "name": node.name, "op": "softsign",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"n": n},
                })
            elif target is _agents_compound_gelu_exact:
                in_name = node.args[0].name
                n = int(np.prod(tensors[in_name]["shape"]))
                ops.append({
                    "name": node.name, "op": "gelu_exact",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"n": n},
                })
            elif target is _agents_compound_l1_norm:
                in_name = node.args[0].name
                in_shape = list(tensors[in_name]["shape"])
                # Bench convention: dim=1, keepdim=True. The placeholder
                # shape is preserved (broadcast division), and the
                # reduction collapses along axis 1.
                dim = 1
                outer = int(np.prod(in_shape[:dim])) if dim > 0 else 1
                reduce = int(in_shape[dim])
                inner = int(np.prod(in_shape[dim+1:])) if dim + 1 < len(in_shape) else 1
                ops.append({
                    "name": node.name, "op": "l1_norm",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"outer": outer, "reduce": reduce, "inner": inner},
                })
            elif target is _agents_compound_l2_norm:
                in_name = node.args[0].name
                in_shape = list(tensors[in_name]["shape"])
                dim = 1
                outer = int(np.prod(in_shape[:dim])) if dim > 0 else 1
                reduce = int(in_shape[dim])
                inner = int(np.prod(in_shape[dim+1:])) if dim + 1 < len(in_shape) else 1
                ops.append({
                    "name": node.name, "op": "l2_norm",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"outer": outer, "reduce": reduce, "inner": inner},
                })
            elif target is _agents_compound_frobenius_norm:
                in_name = node.args[0].name
                n = int(np.prod(tensors[in_name]["shape"]))
                ops.append({
                    "name": node.name, "op": "frobenius_norm",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"n": n},
                })
            elif target is _agents_compound_rms_norm:
                in_name = node.args[0].name
                in_shape = list(tensors[in_name]["shape"])
                dim = 1  # RMSNorm reduces over the channel axis, keepdim=True
                outer = int(np.prod(in_shape[:dim])) if dim > 0 else 1
                reduce = int(in_shape[dim])
                inner = int(np.prod(in_shape[dim+1:])) if dim + 1 < len(in_shape) else 1
                ops.append({
                    "name": node.name, "op": "rms_norm",
                    "inputs": [in_name], "outputs": [out_name],
                    "eps": float(node.meta.get("rms_eps", 1e-5)),
                    "shape": {"outer": outer, "reduce": reduce, "inner": inner},
                })
            elif target is _agents_loss_mse or target is _agents_loss_hinge:
                a_name = node.args[0].name
                b_name = node.args[1].name
                n = int(node.meta.get("loss_n",
                                      int(np.prod(tensors[a_name]["shape"]))))
                if target is _agents_loss_mse:
                    ops.append({
                        "name": node.name, "op": "mse_loss",
                        "inputs": [a_name, b_name], "outputs": [out_name],
                        "shape": {"n": n},
                    })
                else:
                    # Hinge: targ may broadcast over pred (targ_len divides n).
                    targ_len = int(np.prod(tensors[b_name]["shape"]))
                    ops.append({
                        "name": node.name, "op": "hinge_loss",
                        "inputs": [a_name, b_name], "outputs": [out_name],
                        "shape": {"n": n, "targ_len": max(1, targ_len)},
                    })
            elif target is _agents_compound_mean_abs_norm:
                in_name = node.args[0].name
                in_shape = list(tensors[in_name]["shape"])
                dim = 1  # L1 norm divides by mean(|x|) over dim=1, keepdim=True
                outer = int(np.prod(in_shape[:dim])) if dim > 0 else 1
                reduce = int(in_shape[dim])
                inner = int(np.prod(in_shape[dim+1:])) if dim + 1 < len(in_shape) else 1
                ops.append({
                    "name": node.name, "op": "mean_abs_norm",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"outer": outer, "reduce": reduce, "inner": inner},
                })
            elif target is _agents_compound_excl_cumsum:
                in_name = node.args[0].name
                in_shape = [int(s) for s in tensors[in_name]["shape"]]
                if len(in_shape) != 2:
                    raise NotImplementedError(
                        f"exclusive_cumsum at {node.name}: only 2D supported "
                        f"(got {in_shape})")
                B, N = in_shape
                ops.append({
                    "name": node.name, "op": "exclusive_cumsum",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"Bout": B - 1, "N": N},
                })
            elif tname == "hardtanh" \
                    or target is torch.nn.functional.hardtanh:
                in_name = node.args[0].name
                n = int(np.prod(tensors[in_name]["shape"]))
                min_val = -1.0
                max_val = 1.0
                if node.kwargs:
                    min_val = float(node.kwargs.get("min_val", min_val))
                    max_val = float(node.kwargs.get("max_val", max_val))
                # Positional args: hardtanh(x, min_val, max_val).
                if len(node.args) > 1 and isinstance(node.args[1], (int, float)):
                    min_val = float(node.args[1])
                if len(node.args) > 2 and isinstance(node.args[2], (int, float)):
                    max_val = float(node.args[2])
                ops.append({
                    "name": node.name, "op": "hardtanh",
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"n": n},
                    "min_val": min_val, "max_val": max_val,
                })
            elif (target is torch.matmul or target is torch.mm or
                  tname in ("matmul", "mm")):
                arg_a_node, arg_b_node = node.args[0], node.args[1]
                # diag(A) @ B — A is 1D, diag makes it NxN, then row-scales B.
                # Emit a fused diag_matmul (out[i,j]=A[i]*B[i,j]) rather than
                # materializing the NxN diagonal matrix (KernelBench 12).
                if (isinstance(arg_a_node, torch.fx.Node)
                        and arg_a_node.op == "call_function"
                        and getattr(arg_a_node.target, "__name__", "") == "diag"):
                    a1_node = arg_a_node.args[0]
                    b_shape = list(tensors[arg_b_node.name]["shape"])
                    ops.append({
                        "name": node.name, "op": "diag_matmul",
                        "inputs": [a1_node.name, arg_b_node.name],
                        "outputs": [out_name],
                        "shape": {"N": int(b_shape[0]), "M": int(b_shape[1])},
                    })
                else:
                    trans_a = _is_transpose_node(arg_a_node)
                    trans_b = _is_transpose_node(arg_b_node)
                    a_node = arg_a_node.args[0] if trans_a else arg_a_node
                    b_node = arg_b_node.args[0] if trans_b else arg_b_node
                    a_shape = list(tensors[a_node.name]["shape"])
                    b_shape = list(tensors[b_node.name]["shape"])
                    # N-D @ 2D: flatten A's leading dims into M (contiguous
                    # row-major, so the matmul kernel sees a plain [M,K]). This
                    # covers torch.matmul(A_3d/4d, B_2d) (KernelBench 10) and the
                    # einsum "...l,lk->...k" reduction. Transposed forms stay 2D.
                    if trans_a:
                        K, M = int(a_shape[0]), int(a_shape[1])
                    else:
                        M = int(np.prod(a_shape[:-1]))
                        K = int(a_shape[-1])
                    N = int(b_shape[0]) if trans_b else int(b_shape[-1])
                    if trans_a and trans_b:
                        op_kind = "matmul_tatb"
                    elif trans_a:
                        op_kind = "matmul_ta"
                    elif trans_b:
                        op_kind = "matmul_tb"
                    else:
                        op_kind = "matmul"
                    ops.append({
                        "name": node.name, "op": op_kind,
                        "inputs": [a_node.name, b_node.name],
                        "outputs": [out_name],
                        "shape": {"M": M, "K": K, "N": N},
                    })
            elif target is torch.bmm or tname == "bmm":
                a_name = node.args[0].name
                b_name = node.args[1].name
                a_shape = list(tensors[a_name]["shape"])
                b_shape = list(tensors[b_name]["shape"])
                ops.append({
                    "name": node.name, "op": "bmm",
                    "inputs": [a_name, b_name],
                    "outputs": [out_name],
                    "shape": {
                        "batch": int(a_shape[0]),
                        "M": int(a_shape[1]),
                        "K": int(a_shape[2]),
                        "N": int(b_shape[2]),
                    },
                })
            elif tname in ("triu", "tril") or target in (torch.triu, torch.tril):
                # Upper/lower triangular mask of a 2D matrix (KernelBench 14/15,
                # typically triu(matmul(A,B))). diagonal from kwargs/args[1].
                in_name = node.args[0].name
                in_shape = [int(s) for s in tensors[in_name]["shape"]]
                if len(in_shape) != 2:
                    raise NotImplementedError(
                        f"{tname} at {node.name}: only 2D supported "
                        f"(got {in_shape})")
                diagonal = 0
                if node.kwargs and "diagonal" in node.kwargs:
                    diagonal = int(node.kwargs["diagonal"])
                elif len(node.args) > 1:
                    diagonal = int(node.args[1])
                op_name = "triu" if (tname == "triu" or target is torch.triu) \
                    else "tril"
                ops.append({
                    "name": node.name, "op": op_name,
                    "inputs": [in_name], "outputs": [out_name],
                    "shape": {"M": in_shape[0], "N": in_shape[1],
                              "diagonal": diagonal},
                })
            elif tname == "einsum" or target is torch.einsum:
                # Only the "<lead>l,lk-><lead>k" reduction (A's last dim
                # contracts with B's first; B is 2D) is wired — it's a plain
                # matmul with A's leading dims flattened into M (KernelBench 11).
                eq = node.args[0]
                operands = node.args[1:]
                if len(operands) == 1 and isinstance(operands[0], (tuple, list)):
                    operands = tuple(operands[0])
                eq_norm = eq.replace(" ", "")
                a_name = operands[0].name
                b_name = operands[1].name
                a_shape = [int(s) for s in tensors[a_name]["shape"]]
                b_shape = [int(s) for s in tensors[b_name]["shape"]]
                lhs, rhs = eq_norm.split("->")
                a_sub, b_sub = lhs.split(",")
                ok = (len(b_sub) == 2 and len(b_shape) == 2
                      and a_sub[-1] == b_sub[0]           # contract A_last · B_0
                      and rhs == a_sub[:-1] + b_sub[1])   # output = lead + B_1
                if not ok:
                    raise NotImplementedError(
                        f"einsum {eq!r} at {node.name}: only "
                        f"'...l,lk->...k' (matmul) is supported")
                M = int(np.prod(a_shape[:-1]))
                K = int(a_shape[-1])
                N = int(b_shape[1])
                ops.append({
                    "name": node.name, "op": "matmul",
                    "inputs": [a_name, b_name], "outputs": [out_name],
                    "shape": {"M": M, "K": K, "N": N},
                })
            elif target is torch.cat or tname == "cat":
                # torch.cat(tensors_list, dim). YOLOv8 uses dim=1 (channel
                # concat) exclusively — restrict to that for now since the
                # other dims need different memory layout in the kernel.
                tensors_arg = node.args[0]
                if not isinstance(tensors_arg, (list, tuple)):
                    raise NotImplementedError(
                        f"cat at {node.name}: first arg must be a list/tuple "
                        f"of tensors, got {type(tensors_arg).__name__}."
                    )
                dim = node.args[1] if len(node.args) > 1 else \
                    node.kwargs.get("dim", 0)
                dim = int(dim)
                if dim != 1:
                    raise NotImplementedError(
                        f"cat at {node.name}: dim={dim}, only dim=1 (channel "
                        f"concat) is supported."
                    )
                in_names = [t.name for t in tensors_arg]
                first_shape = list(tensors[in_names[0]]["shape"])
                # 4D NCHW channel-concat, or 2D [N,C] feature-concat (as
                # [N,C,1,1] for the cat_c1 kernel) — the vision|depth|state fuse.
                if len(first_shape) == 4:
                    N_, _, H_, W_ = (int(s) for s in first_shape)
                elif len(first_shape) == 2:
                    N_ = int(first_shape[0]); H_ = W_ = 1
                else:
                    raise NotImplementedError(
                        f"cat at {node.name}: only 2D [N,C] or 4D NCHW inputs "
                        f"supported (got {first_shape})."
                    )
                c_inputs = [int(tensors[n]["shape"][1]) for n in in_names]
                op_kind = f"cat{len(in_names)}_c1"
                if len(in_names) not in (2, 3, 4):
                    raise NotImplementedError(
                        f"cat at {node.name}: {len(in_names)} inputs; only "
                        f"2/3/4-input cat kernels are wired up."
                    )
                ops.append({
                    "name": node.name, "op": op_kind,
                    "inputs": in_names, "outputs": [out_name],
                    "shape": {"N": N_, "H": H_, "W": W_,
                              "C_inputs": c_inputs,
                              "C_total": sum(c_inputs)},
                })
            else:
                raise NotImplementedError(f"unsupported function {tname} at {node.name}")

        elif node.op == "call_method":
            # Currently only `chunk` is wired in. Tensor.chunk(2, 1) is the
            # split-channel pattern in YOLOv8's C2f block. The chunk node
            # itself doesn't produce a tensor; getitem(_, 0)/(_, 1) do —
            # find them and emit a chunk2_c1 op with both output names.
            target = node.target
            if target == "chunk":
                in_name = node.args[0].name
                n_chunks = int(node.args[1])
                dim = int(node.args[2]) if len(node.args) > 2 \
                    else int(node.kwargs.get("dim", 0))
                if n_chunks != 2 or dim != 1:
                    raise NotImplementedError(
                        f"chunk at {node.name}: only chunk(2, dim=1) is "
                        f"supported; got chunk({n_chunks}, dim={dim})."
                    )
                in_shape = list(tensors[in_name]["shape"])
                if len(in_shape) != 4:
                    raise NotImplementedError(
                        f"chunk at {node.name}: only 4D NCHW inputs supported."
                    )
                N_, C, H_, W_ = (int(s) for s in in_shape)
                if C % 2 != 0:
                    raise NotImplementedError(
                        f"chunk at {node.name}: input C={C} is odd; can't "
                        f"split evenly."
                    )
                c_each = C // 2
                # Find getitem(_, 0) and getitem(_, 1) consumers — both must
                # exist for the IR to be well-defined (we don't emit a
                # tensor for the chunk node itself, only for its halves).
                gi0 = None
                gi1 = None
                op_getitem = __import__("operator").getitem
                for user in node.users:
                    if (user.op == "call_function" and user.target is op_getitem
                            and len(user.args) >= 2 and isinstance(user.args[1], int)):
                        idx = user.args[1]
                        if idx == 0:
                            gi0 = user
                        elif idx == 1:
                            gi1 = user
                if gi0 is None or gi1 is None:
                    raise NotImplementedError(
                        f"chunk at {node.name}: expected both getitem(_, 0) "
                        f"and getitem(_, 1) consumers; got "
                        f"gi0={gi0} gi1={gi1}."
                    )
                # The two output tensors are named after the getitem nodes,
                # not the chunk node. Both halves have shape [N, C/2, H, W].
                tensors[gi0.name] = {"shape": [N_, c_each, H_, W_],
                                     "dtype": "f32", "quant": None}
                tensors[gi1.name] = {"shape": [N_, c_each, H_, W_],
                                     "dtype": "f32", "quant": None}
                _skip_nodes.add(gi0)
                _skip_nodes.add(gi1)
                ops.append({
                    "name": node.name, "op": "chunk2_c1",
                    "inputs": [in_name],
                    "outputs": [gi0.name, gi1.name],
                    "shape": {"N": N_, "C": C, "H": H_, "W": W_,
                              "c_each": c_each},
                })
            elif target == "flip":
                # Tensor.flip(dim) — reverse along one axis. dims may be an int
                # or a 1-elem tuple/list (KernelBench 91 uses a single dim).
                in_name = node.args[0].name
                in_shape = [int(s) for s in tensors[in_name]["shape"]]
                dims = node.args[1] if len(node.args) > 1 \
                    else node.kwargs.get("dims")
                if isinstance(dims, (tuple, list)):
                    if len(dims) != 1:
                        raise NotImplementedError(
                            f"flip at {node.name}: only single-axis supported")
                    d = int(dims[0])
                else:
                    d = int(dims)
                if d < 0:
                    d += len(in_shape)
                outer = int(np.prod(in_shape[:d])) if d > 0 else 1
                axis = int(in_shape[d])
                inner = int(np.prod(in_shape[d+1:])) if d + 1 < len(in_shape) else 1
                tensors[node.name] = _tensor_meta(node)
                ops.append({
                    "name": node.name, "op": "flip",
                    "inputs": [in_name], "outputs": [node.name],
                    "shape": {"outer": outer, "axis": axis, "inner": inner},
                })
            else:
                raise NotImplementedError(
                    f"unhandled call_method '{target}' at {node.name}"
                )

        elif node.op == "output":
            arg = node.args[0]
            if isinstance(arg, (tuple, list)):
                output_names = [a.name for a in arg]
            else:
                output_names = [arg.name]

        elif node.op == "get_attr":
            # Constants — not expected in MLP, fail loud if encountered.
            raise NotImplementedError(f"get_attr nodes not supported yet: {node.name}")

        else:
            raise NotImplementedError(f"unhandled fx op {node.op} at {node.name}")

    if not input_names or not output_names:
        raise RuntimeError("graph missing input/output")

    # In fp16 mode, suffix every op name (and update tensor dtypes) so
    # downstream stages pick the half-precision kernel variants without
    # touching the otherwise-identical graph extraction logic. Done as
    # a post-pass to keep the per-module branches dtype-agnostic.
    if op_suffix:
        for op in ops:
            if op["op"] != "view":
                op["op"] = op["op"] + op_suffix
        for tname, tmeta in tensors.items():
            if tmeta.get("dtype") == "f32":
                tmeta["dtype"] = "f16"

    dispatches = _annotate_dispatches(ops)
    # Build the input IR field. For single-input models the legacy
    # `tensor` key is sufficient. For multi-input (matmul A+B, bmm)
    # we also add `packed_inputs` — a list of {name, offset, size}
    # entries describing how the inputs are concatenated into one flat
    # buffer. generate_skeleton uses this to emit `(input + offset)`.
    if len(input_names) == 1:
        ir_input: dict = {"tensor": input_names[0]}
    else:
        packed: list[dict] = []
        off = 0
        for nm in input_names:
            sz = int(np.prod(tensors[nm]["shape"]))
            packed.append({"name": nm, "offset": off, "size": sz})
            off += sz
        ir_input = {"tensor": input_names[0], "packed_inputs": packed}
    ir = {
        "name": name,
        "version": 1,
        "quant": quant,
        "input": ir_input,
        # `tensors` is the multi-output form; `tensor` retained for back-compat
        # readers but only populated for single-output models.
        "output": {
            "tensors": output_names,
            "tensor": output_names[0] if len(output_names) == 1 else None,
        },
        "tensors": tensors,
        "ops": ops,
        "dispatches": dispatches,
    }

    # Run reference to capture golden I/O. fp16 mode runs `model.half()`
    # on `input.half()` so the golden reflects genuine half-precision
    # numerics — not an fp32 trace down-cast at the boundary.
    with torch.no_grad():
        if quant == "fp16":
            # The reference kernels use fp16 STORAGE but fp32 MATH (load
            # _Float16 -> float, accumulate in float, store _Float16). Compute
            # the golden the same way: round inputs to fp16, run the model in
            # fp32 on those rounded values, then store the output as fp16. This
            # matches the kernels' numerics (a genuine `model.half()` would
            # instead accumulate in fp16 — e.g. cross_entropy's 4096-class
            # logsumexp drifts ~0.1 — and also lacks CPU Half kernels for some
            # ops like avg_pool3d). ALL tensor inputs (including int class-index
            # targets and bool masks) are round-tripped through fp16 so the
            # golden sees exactly what the kernel reads from the fp16 io buffer
            # — e.g. cross_entropy indices > 2048 aren't representable in fp16,
            # so both golden and kernel must use the same fp16-rounded index.
            ref_inputs_exec = [t.to(torch.float16).to(t.dtype)
                               if torch.is_tensor(t) else t
                               for t in sample_inputs]
            torch_dtype = torch.float16
            out = model.float()(*ref_inputs_exec)
        else:
            ref_inputs_exec = list(sample_inputs)
            ref_model = model
            torch_dtype = torch.float32
            out = ref_model(*ref_inputs_exec)

    # Multi-output models return a tuple; flatten in IR-output order so the
    # downstream comparator just needs to do an elementwise compare.
    if isinstance(out, (tuple, list)):
        flat = np.concatenate([
            o.detach().cpu().numpy().astype(weight_dtype).reshape(-1) for o in out
        ])
    else:
        flat = out.detach().cpu().numpy().astype(weight_dtype).reshape(-1)

    # For multi-input models, concatenate all inputs into one flat array
    # (packed layout, matching the offsets in ir["input"]["packed_inputs"]).
    flat_input = np.concatenate([
        t.detach().cpu().numpy().astype(weight_dtype).reshape(-1)
        for t in ref_inputs_exec
    ])

    ir_path = os.path.join(out_dir, "graph.json")
    weights_path = os.path.join(out_dir, "weights.npz")
    io_path = os.path.join(out_dir, "io.npz")

    with open(ir_path, "w") as f:
        json.dump(ir, f, indent=2)
    np.savez(weights_path, **weights)
    np.savez(
        io_path,
        input=flat_input,
        output=flat,
    )

    print(f"wrote {ir_path}")
    print(f"wrote {weights_path}  ({len(weights)} tensors)")
    print(f"wrote {io_path}")
    return ir


import contextlib


@contextlib.contextmanager
def _cpu_fp32_tensor_creation():
    """Patch torch tensor-creation ops so benches that hard-code
    device='cuda' and/or dtype=float16 in get_inputs (e.g. 97 SDPA) still
    materialize on CPU in fp32. Restores the originals on exit."""
    names = ["rand", "randn", "zeros", "ones", "empty", "full", "randint",
             "arange", "tensor", "eye", "linspace"]
    saved = {n: getattr(torch, n) for n in names if hasattr(torch, n)}

    def _wrap(fn):
        def inner(*args, **kwargs):
            if kwargs.get("device") is not None:
                kwargs["device"] = "cpu"
            if kwargs.get("dtype") == torch.float16:
                kwargs["dtype"] = torch.float32
            return fn(*args, **kwargs)
        return inner

    try:
        for n, fn in saved.items():
            setattr(torch, n, _wrap(fn))
        yield
    finally:
        for n, fn in saved.items():
            setattr(torch, n, fn)


# --- utilization-aware sizing -------------------------------------------------
# The legacy `max_elements` policy halves whatever module attr is largest until
# a tiny total element count is met — which collapses conv/matmul channel counts
# to ~16 and leaves XNNPACK/RVV overhead-bound (41 inst/MAC measured). Instead we
# cap the *baked io footprint* (input+golden bytes) and shrink dims by category:
# structural attrs define the op (never touched); batch just replicates work
# (shrunk first); spatial next; channel/feature dims drive vector/GEMM
# utilization so they're protected (shrunk last, with a floor). Categorization is
# by the attr-name vocabulary the level1 benches actually use.
_KB_STRUCT = {"kernel_size", "stride", "padding", "dilation", "groups",
              "num_classes", "num_heads", "num_groups", "stride_w", "stride_h",
              "padding_w", "padding_h", "reduce_dim"}
_KB_BATCH = {"batch_size", "sequence_length"}
_KB_SPATIAL = {"width", "height", "length", "depth", "width_in", "height_in",
               "spatial"}
_KB_CHANNEL = {"in_channels", "out_channels", "channels", "features", "dim",
               "hidden_size", "embed_dim", "k", "m", "l", "n"}
_KB_FLOOR = {"batch": 1, "spatial": 16, "channel": 32}


def _kb_attr_group(name: str) -> "str | None":
    if name in _KB_STRUCT:
        return None            # structural — never shrink
    if name in _KB_BATCH:
        return "batch"
    if name in _KB_SPATIAL:
        return "spatial"
    if name in _KB_CHANNEL:
        return "channel"
    return "spatial"           # unknown dim → treat like spatial (before channels)


def _kb_footprint_bytes(mod) -> int:
    """Baked io footprint (input + golden output) in fp32 bytes at the module's
    current dims. Runs a forward to size the output."""
    ia = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    m = mod.Model(*ia)
    m.eval()
    ins = mod.get_inputs()
    with torch.no_grad():
        o = m(*ins)
    outs = o if isinstance(o, (list, tuple)) else [o]
    inb = sum(int(np.prod(t.shape)) for t in ins if torch.is_tensor(t))
    outb = sum(int(np.prod(x.shape)) for x in outs if torch.is_tensor(x))
    return (inb + outb) * 4


def _kb_flops(mod) -> int:
    """Total forward FLOPs at the module's current dims, via torch's
    FlopCounterMode (counts matmul/conv/bmm/etc.). Model-agnostic — measures the
    real op cost, so it captures the K-contraction of a matmul or the spatial**3
    of a 3D conv that `_kb_footprint_bytes` (io only) misses. Raises if
    uncountable."""
    from torch.utils.flop_counter import FlopCounterMode
    ia = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    m = mod.Model(*ia)
    m.eval()
    ins = mod.get_inputs()
    fc = FlopCounterMode(display=False)
    with fc, torch.no_grad():
        m(*ins)
    return int(fc.get_total_flops())


def _shrink_for_flops(mod, target_flops: int,
                      target_bytes: "int | None" = None) -> None:
    """Shrink module dims until measured forward FLOPs <= `target_flops` (and io
    bytes <= `target_bytes` if given), preferring batch -> spatial -> channel and
    honoring per-group floors so the shrunk problem still exercises the GEMM/vector
    units (channel/K floor keeps a full vector register busy). Unlike
    `_shrink_for_utilization` (io-byte cap only), this bounds the *compute*, so a
    large-K matmul or a 3D conv shrinks to a size that actually runs on-target
    instead of pinning a tiny-io / huge-FLOP shape. Falls back to a byte cap if
    FLOPs can't be counted; reverts any shrink that makes the forward invalid."""
    order = ["batch", "spatial", "channel"]
    for _ in range(400):
        try:
            fl = _kb_flops(mod)
            fp = _kb_footprint_bytes(mod)
        except Exception:
            if target_bytes:
                _shrink_for_utilization(mod, target_bytes)
            return
        if fl <= target_flops and (target_bytes is None or fp <= target_bytes):
            return
        ints = {k: v for k, v in vars(mod).items()
                if isinstance(v, int) and v > 1 and not k.startswith("_")}
        cand = None
        for grp in order:
            pool = {k: v for k, v in ints.items()
                    if _kb_attr_group(k) == grp and v > _KB_FLOOR[grp]}
            if pool:
                cand = max(pool, key=pool.get)
                break
        if cand is None:
            return             # everything at its floor; can't reduce further
        grp = _kb_attr_group(cand)
        oldv = vars(mod)[cand]
        setattr(mod, cand, max(_KB_FLOOR[grp], oldv // 2))
        try:
            _kb_footprint_bytes(mod)   # validate the new dims forward-pass
        except Exception:
            setattr(mod, cand, oldv)
            return


def _shrink_for_utilization(mod, target_bytes: int) -> None:
    """Shrink module dims to fit `target_bytes` of baked io, preferring
    batch → spatial and protecting channel/feature dims (so the shrunk problem
    still exercises the vector/GEMM units). Reverts any shrink that makes the
    forward invalid."""
    order = ["batch", "spatial", "channel"]
    for _ in range(200):
        try:
            fp = _kb_footprint_bytes(mod)
        except Exception:
            return             # can't instantiate at current dims; leave as-is
        if fp <= target_bytes:
            return
        ints = {k: v for k, v in vars(mod).items()
                if isinstance(v, int) and v > 1 and not k.startswith("_")}
        cand = None
        for grp in order:
            pool = {k: v for k, v in ints.items()
                    if _kb_attr_group(k) == grp and v > _KB_FLOOR[grp]}
            if pool:
                cand = max(pool, key=pool.get)
                break
        if cand is None:
            return             # nothing shrinkable without violating a floor
        grp = _kb_attr_group(cand)
        oldv = vars(mod)[cand]
        setattr(mod, cand, max(_KB_FLOOR[grp], oldv // 2))
        try:
            _kb_footprint_bytes(mod)   # validate the new dims forward-pass
        except Exception:
            setattr(mod, cand, oldv)
            return


def _load_kernelbench(bench_path: str,
                      max_elements: "int | None" = None,
                      target_bytes: "int | None" = None,
                      target_flops: "int | None" = None
                      ) -> "tuple[torch.nn.Module, torch.Tensor, str]":
    """Load a KernelBench level1 file (single-input variant only).

    Schema (every level1 file conforms): a `class Model(nn.Module)` with
    `__init__(*init_args)` + `forward(x)`, module-level shape constants, and
    `get_inputs()` / `get_init_inputs()`.

    Returns (model, sample_input, name); `sample_input` is a bare tensor for
    single-input forwards or a list of tensors for multi-input ones (matmul
    A,B; bmm). name is a C-friendly `kb_<basename>`.

    `max_elements` caps the *total* input element count across all forward
    inputs — KernelBench level1 defaults are sized for GPU memory (batch=16,
    256x256 spatial) and overflow Zephyr's 256 MB RAM region when baked into
    rodata. When set, we shrink integer module-level shape attrs by halving the
    largest until the inputs fit, then re-instantiate, so the PyTorch golden
    corresponds to the shrunken shape.

    Multi-input forwards are threaded through extract()'s packed_inputs path.
    Loss-style benches (preds+targets → scalar) load but fail later at the
    unsupported loss op. See modelblaster/notes/kernelbench_rvv_port_plan.md.
    """
    import importlib.util
    if not os.path.isfile(bench_path):
        raise FileNotFoundError(f"--bench-file {bench_path} not found")
    spec = importlib.util.spec_from_file_location("_kernelbench_mod", bench_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {bench_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "Model"):
        raise RuntimeError(f"{bench_path} missing class Model")
    if not hasattr(mod, "get_inputs"):
        raise RuntimeError(f"{bench_path} missing get_inputs()")

    with _cpu_fp32_tensor_creation():
        if max_elements is not None and max_elements > 0:
            SHAPE_ATTRS_BLOCKLIST = {"num_classes"}  # not a spatial dim
            ints = {k: v for k, v in vars(mod).items()
                    if isinstance(v, int) and v > 1
                    and not k.startswith("_")
                    and k not in SHAPE_ATTRS_BLOCKLIST}
            history: "list[tuple[str, int]]" = []
            for _ in range(64):
                # Cap the TOTAL element count across all forward inputs (matmul
                # A+B, bmm, etc.), not just the first — the second operand can
                # dwarf the first.
                total = sum(int(np.prod(t.shape)) for t in mod.get_inputs()
                            if torch.is_tensor(t))
                if total <= max_elements:
                    break
                if not ints:
                    break
                biggest = max(ints, key=ints.get)
                new = max(1, ints[biggest] // 2)
                if new == ints[biggest]:
                    break
                history.append((biggest, ints[biggest]))
                ints[biggest] = new
                setattr(mod, biggest, new)

            # Some ops (dilated conv) become invalid if a spatial dim is shrunk
            # below the (dilated) kernel extent — the model's own forward then
            # raises. Back off the most recent shrink steps until a forward pass
            # succeeds, so the shape stays valid even if it exceeds max_elements.
            _ia = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
            for _ in range(len(history) + 1):
                try:
                    _m = mod.Model(*_ia)
                    _m.eval()
                    with torch.no_grad():
                        _m(*mod.get_inputs())
                    break
                except Exception:
                    if not history:
                        break
                    attr, oldv = history.pop()
                    setattr(mod, attr, oldv)
        elif target_flops is not None and target_flops > 0:
            # FLOP-budget policy: bound the actual compute (so large-K matmuls /
            # 3D convs shrink to something that runs on-target), while still
            # capping io bytes if a target_bytes ceiling is also supplied.
            _shrink_for_flops(mod, target_flops, target_bytes)
        elif target_bytes is not None and target_bytes > 0:
            # Default policy: cap the baked-io footprint while protecting the
            # channel/feature dims that drive vector/GEMM utilization.
            _shrink_for_utilization(mod, target_bytes)

        init_args = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
        model = mod.Model(*init_args)
        model.eval()
        inputs = mod.get_inputs()
        if not isinstance(inputs, list) or not inputs:
            raise RuntimeError(f"{bench_path} get_inputs() must return non-empty list")
    # Bool masks (e.g. masked_cumsum) become float in the model anyway
    # (x * bool == x * float); cast them so the fp32 pipeline can carry them.
    # Integer class-index targets (cross_entropy) are left alone.
    inputs = [t.float() if (torch.is_tensor(t) and t.dtype == torch.bool) else t
              for t in inputs]

    # Bind non-tensor (scalar) forward args as constants so only tensors are
    # graph inputs (KernelBench 5: `A * s` with s a Python float). The trace
    # then sees `A * <const>`, extracted as a mul_scalar op. Only a single
    # tensor input is supported here (fx can't trace a *args wrapper).
    if any(not torch.is_tensor(t) for t in inputs):
        _tensor_pos = [i for i, t in enumerate(inputs) if torch.is_tensor(t)]
        if len(_tensor_pos) != 1:
            raise NotImplementedError(
                f"{bench_path}: scalar forward args are only supported with a "
                f"single tensor input (got {len(_tensor_pos)})")
        _pos0 = _tensor_pos[0]
        _full = list(inputs)  # concrete scalars + placeholder slot for the tensor
        _orig = model

        class _ScalarBind(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = _orig

            def forward(self, x):
                args = list(_full)
                args[_pos0] = x
                return self.m(*args)

        model = _ScalarBind()
        model.eval()
        inputs = [inputs[_pos0]]

    base = os.path.splitext(os.path.basename(bench_path))[0]
    name = "kb_" + "".join(ch if (ch.isalnum() or ch == "_") else "_"
                           for ch in base).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    # Single-input → return the bare tensor (legacy path). Multi-input
    # (matmul A+B, bmm) → return the list; extract() threads it through as
    # packed_inputs / a flat io.npz buffer. Loss-style multi-input benches
    # (preds+targets → scalar) still load here but fail later at the
    # unsupported loss op, with a clear per-op error.
    return model, (inputs[0] if len(inputs) == 1 else inputs), name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="model id -> models/<id>.py (the well-known ids have "
                         "explicit imports; any other resolves by dynamic "
                         "import of modelblaster.models.<id>). Mutually "
                         "exclusive with --bench-file.")
    ap.add_argument("--bench-file", default=None,
                    help="path to a KernelBench level1 .py file. "
                         "Mutually exclusive with --model.")
    ap.add_argument("--bench-target-mb", type=int, default=256,
                    help="DEFAULT kernelbench sizing: cap baked io (input+golden) "
                         "to this many MiB, shrinking batch->spatial and "
                         "protecting channel/feature dims for good HW "
                         "utilization (default 256). 0 = stock dims.")
    ap.add_argument("--bench-max-elements", type=int, default=0,
                    help="LEGACY override: if >0, cap total input ELEMENTS by "
                         "halving the largest int attr (the old tiny-toy policy). "
                         "Overrides --bench-target-mb. 0 = use --bench-target-mb.")
    ap.add_argument("--bench-target-gflops", type=float,
                    default=float(os.environ.get("BENCH_TARGET_GFLOPS", "0")),
                    help="COMPUTE-budget sizing: if >0, size the bench to this "
                         "forward-FLOP budget (measured via FlopCounterMode), "
                         "bounding compute rather than just io — fixes large-K "
                         "matmuls / 3D convs that have tiny io but huge FLOPs. "
                         "--bench-target-mb stays an io ceiling. Overridden by "
                         "--bench-max-elements. Env: BENCH_TARGET_GFLOPS.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--quant", default="fp32", choices=["fp32", "fp16", "int8"],
                    help="quantization mode. fp32 = stock float, fp16 = "
                         "half precision (uses torch.float16 model + "
                         "_Float16 C kernels, validated against half-cast "
                         "torch golden), int8 = symmetric per-tensor PTQ.")
    ap.add_argument("--fp16-ops", default=None,
                    help="comma-separated IR op names to promote to fp16 in an "
                         "int8 extract (mixed precision). Additive to the "
                         "model's get_precision_spec(). No-op for fp32/fp16.")
    ap.add_argument("--core-registry", default=None,
                    help="optional path to an modelblaster/cores/*.json registry. "
                         "When provided, the post-extraction pass validates "
                         "every dispatch's hardware_target against the listed "
                         "cores' capabilities and aborts on mismatch.")
    ap.add_argument("--num-calibration", type=int, default=1,
                    help="number of calibration samples for int8 PTQ. With "
                         ">1, per-tensor activation max-abs is aggregated "
                         "across the model's get_calibration_spec() or "
                         "get_calibration_samples() result so scales reflect "
                         "the worst-case dynamic range over the whole set "
                         "instead of a single frame. Detection / segmentation "
                         "models need ~16 to avoid cls-logit saturation. "
                         "No-op for fp32 / fp16.")
    ap.add_argument("--per-channel", action="store_true",
                    help="per-output-channel int8 weight quant for conv/linear "
                         "(tighter than per-tensor). No-op for fp32 / fp16.")
    ap.add_argument("--no-bn-folding", dest="fold_conv_bn",
                    action="store_false",
                    default=os.environ.get("MB_NO_BN_FOLDING", "0") != "1",
                    help="disable graph-level batchnorm->conv folding (on by "
                         "default). Unfolding hands the scheduler a placement "
                         "choice back (a standalone batchnorm2d is its own "
                         "dispatch; conv+bn must run whole on one core), but a "
                         "controlled same-bitstream A/B on yolov8_nano showed "
                         "that choice is worth nothing: 155 -> 212 dispatches, "
                         "best-per-dispatch 67.61 -> 68.94 ms and greedy floor "
                         "makespan 65.97 -> 67.25 ms (both ~2%% WORSE), "
                         "two-core specialisation gain flat at 2.44x -> 2.41x, "
                         "and the RVV arm drops from bit-exact to 1 LSB. Use "
                         "for investigation, not as an optimization. Also "
                         "settable via MB_NO_BN_FOLDING=1.")
    ap.add_argument("--enable-fusion", action="store_true",
                    default=os.environ.get("MB_ENABLE_FUSION", "0") == "1",
                    help="opt-in extended operator fusion (beyond the "
                         "always-on linear/conv2d/add/bn -> relu absorption "
                         "and graph-level bn->conv folding). Currently "
                         "gates two patterns: (1) conv2d -> silu absorption "
                         "into a single conv2d_silu_s8 op (bit-exact vs the "
                         "unfused conv2d_s8 + silu_s8 pair); (2) conv2d -> "
                         "maxpool2d absorption into a single conv2d_pool_s8 "
                         "op when the pool is the conv's SOLE, DIRECT "
                         "consumer (numeric_drift on gemmini targets, "
                         "bit-exact elsewhere — see CONV2D_POOL_S8 in "
                         "reference_kernels.py for the accuracy bound; the "
                         "fast alternative to gemmini's exact-but-6x-slower "
                         "separate conv2d_s8 + maxpool2d_s8 pair). Off by "
                         "default so the unfused path stays available for "
                         "comparison. Can also be set via "
                         "MB_ENABLE_FUSION=1. No-op for fp32 / fp16 "
                         "(int8-only today).")
    ap.add_argument("--fusion-target", default=os.environ.get("MB_FUSION_TARGET"),
                    help="eventual build target this IR is headed for (e.g. "
                         "'rvv', 'gemmini_q31') -- required (with "
                         "--enable-fusion) for the extended-fusion ops "
                         "(conv2d_silu_s8, conv2d_pool_s8) to actually "
                         "fire. Neither has a curated kernel on ANY "
                         "backend yet, so firing on a target that already "
                         "has good curated kernels for the UNFUSED ops "
                         "(rvv, scalar) would silently regress it -- so "
                         "this is restricted to gemmini/gemmini_q31/"
                         "gemmini_q31_rvv (see _FUSION_SAFE_TARGETS in "
                         "this file) and fails CLOSED (no extended fusion) "
                         "if omitted. Also settable via MB_FUSION_TARGET.")
    args = ap.parse_args()

    if (args.model is None) == (args.bench_file is None):
        ap.error("pass exactly one of --model or --bench-file")

    model_mod = None
    if args.bench_file is not None:
        # Legacy element-cap wins if explicitly set (>0); else if a FLOP budget is
        # given, bound compute (io stays capped by --bench-target-mb as a ceiling);
        # otherwise the default byte-budget / utilization-aware policy applies.
        _tgt = None if args.bench_max_elements else args.bench_target_mb * 2**20
        _tflops = (int(args.bench_target_gflops * 1e9)
                   if args.bench_target_gflops and args.bench_target_gflops > 0
                   else None)
        model, sample, name = _load_kernelbench(
            args.bench_file,
            max_elements=(args.bench_max_elements or None),
            target_bytes=_tgt,
            target_flops=_tflops)
    else:
        if args.model == "mlp_generic":
            from modelblaster.models import mlp_generic as model_mod
        elif args.model == "mlp_control":
            from modelblaster.models import mlp_control as model_mod
        elif args.model == "lenet":
            from modelblaster.models import lenet as model_mod
        elif args.model == "relu6net":
            from modelblaster.models import relu6net as model_mod
        elif args.model == "dronet":
            from modelblaster.models import dronet as model_mod
        elif args.model == "mobilenet_v2":
            from modelblaster.models import mobilenet_v2 as model_mod
        elif args.model == "yolov8_nano":
            from modelblaster.models import yolov8_nano as model_mod
        elif args.model == "yolov8_nano_64":
            from modelblaster.models import yolov8_nano_64 as model_mod
        elif args.model == "vitfly_frontend":
            from modelblaster.models import vitfly_frontend as model_mod
        elif args.model == "lstm_tiny":
            from modelblaster.models import lstm_tiny as model_mod
        elif args.model == "vitfly_lstm":
            from modelblaster.models import vitfly_lstm as model_mod
        elif args.model == "norm_block":
            from modelblaster.models import norm_block as model_mod
        elif args.model == "attn_block":
            from modelblaster.models import attn_block as model_mod
        elif args.model == "fused_vision":
            from modelblaster.models import fused_vision as model_mod
        elif args.model == "fused_depth":
            from modelblaster.models import fused_depth as model_mod
        elif args.model == "fused_full":
            from modelblaster.models import fused_full as model_mod
        else:
            # Dynamic import so a new models/<name>.py needs no edit here.
            import importlib
            try:
                model_mod = importlib.import_module(
                    f"modelblaster.models.{args.model}")
            except ModuleNotFoundError:
                raise SystemExit(f"unknown model {args.model}")
        model = model_mod.get_model()
        sample = model_mod.get_sample_input()
        name = args.model

    calibration_samples = None
    if model_mod is not None and args.quant == "int8" and args.num_calibration > 1:
        if hasattr(model_mod, "get_calibration_spec"):
            from modelblaster.mb_datasets import materialize_calibration_samples  # noqa: PLC0415
            spec = model_mod.get_calibration_spec(args.num_calibration)
            print(f"[extract_graph] resolving calibration spec "
                  f"({args.num_calibration} samples) ...", flush=True)
            materialized = materialize_calibration_samples(spec)
            # FX path is single-input; pull the first declared input tensor
            # out of each sample dict (preserves spec ordering).
            input_keys = list(spec["inputs"].keys())
            primary = input_keys[0]
            calibration_samples = [d[primary] for d in materialized]
            # The first sample becomes the io.npz golden anchor so the
            # in-binary verify continues to match. Order is preserved by
            # get_calibration_spec; the rest just widen activation ranges.
            sample = calibration_samples[0]
        elif hasattr(model_mod, "get_calibration_samples"):
            print(f"[extract_graph] loading {args.num_calibration} "
                  f"calibration samples via {args.model}."
                  f"get_calibration_samples ...", flush=True)
            calibration_samples = list(model_mod.get_calibration_samples(
                args.num_calibration))
            sample = calibration_samples[0]
        else:
            print(f"[extract_graph] WARN: --num-calibration "
                  f"{args.num_calibration} requested but {args.model} "
                  f"defines neither get_calibration_spec nor "
                  f"get_calibration_samples; falling back to single "
                  f"get_sample_input()", flush=True)

    # Optional per-input surface dtype for multi-input / multi-dtype models
    # ("i8" / "f16" / "f32" in placeholder order). Defaults to all-int8.
    input_dtypes = None
    if hasattr(model_mod, "get_input_dtypes"):
        input_dtypes = list(model_mod.get_input_dtypes())
        print(f"[extract_graph] input dtypes: {input_dtypes}", flush=True)

    # Mixed precision (int8 base + fp16 islands): resolve the model's
    # get_precision_spec() to the set of IR op names to run in fp16. Only
    # 'fp16_ops' (explicit names) is honoured on the FX path; --fp16-ops adds
    # to it. With no spec the IR is byte-identical to the all-int8 one.
    fp16_op_names: "set[str] | None" = None
    if args.quant == "int8":
        _names: set[str] = set()
        if hasattr(model_mod, "get_precision_spec"):
            spec = model_mod.get_precision_spec() or {}
            _names |= set(spec.get("fp16_ops", []))
        if args.fp16_ops:
            _names |= {t.strip() for t in args.fp16_ops.split(",") if t.strip()}
        fp16_op_names = _names or None
        if fp16_op_names:
            print(f"[extract_graph] fp16 ops: {sorted(fp16_op_names)}",
                  flush=True)
    extract(model, sample, name=name, out_dir=args.out_dir,
            quant=args.quant,
            calibration_samples=calibration_samples,
            input_dtypes=input_dtypes,
            fp16_op_names=fp16_op_names,
            per_channel=getattr(args, "per_channel", False),
            enable_fusion=getattr(args, "enable_fusion", False),
            fold_conv_bn=getattr(args, "fold_conv_bn", True),
            fusion_target=getattr(args, "fusion_target", None))

    if args.core_registry:
        from modelblaster.pipeline import core_registry
        reg = core_registry.load(args.core_registry)
        ir = json.load(open(os.path.join(args.out_dir, "graph.json")))
        errs = core_registry.validate_dispatch_targets(reg, ir.get("ops", []))
        if errs:
            for e in errs:
                print(f"core_registry: {e}")
            raise SystemExit(
                f"{len(errs)} dispatch(es) cannot run on registry "
                f"{reg.system!r}; refine hardware_target or pick a different "
                f"system descriptor.")
        print(f"core_registry: validated against {reg.system}: "
              f"{len(ir.get('ops', []))} ops match")


if __name__ == "__main__":
    main()
