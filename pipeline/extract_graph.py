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
    else:
        return

    # Rewrite: insert a fused sentinel node, retarget the output, drop
    # everything else via dead-code elimination.
    with gm.graph.inserting_before(out_node):
        new_node = gm.graph.call_function(sentinel, args=(x,))
    # Copy the placeholder's tensor_meta onto the new node — pointwise
    # activations preserve shape/dtype, and downstream ShapeProp won't
    # re-run on this graph.
    if "tensor_meta" in x.meta:
        new_node.meta["tensor_meta"] = x.meta["tensor_meta"]
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
    torch.nn.MaxPool2d,
    torch.nn.AdaptiveAvgPool2d,  # global avg pool head used by classifiers
    torch.nn.Dropout,  # eval-mode no-op; we still record a passthrough alias
    torch.nn.BatchNorm2d,  # pre-folded into a per-channel scale + bias
    torch.nn.Sigmoid,
    # YOLOv8 backbone uses SiLU activation throughout; neck uses Upsample.
    torch.nn.SiLU,
    torch.nn.Upsample,
)


def _pair(v) -> tuple[int, int]:
    """Coerce int or 2-tuple to a (h, w) pair."""
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


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


def extract_int8(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    name: str,
    out_dir: str,
    calibration_samples: "list[torch.Tensor] | None" = None,
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

    Supported ops in this first cut: nn.Linear, nn.ReLU, torch.relu.
    Any other op kind raises — extend this function as more ops gain int8
    kernels.
    """
    os.makedirs(out_dir, exist_ok=True)
    model = model.eval()

    gm = torch.fx.symbolic_trace(model)
    ShapeProp(gm).propagate(sample_input)

    # Capture every node's tensor for activation calibration.
    cap = _CaptureTensors(gm)
    final = cap.run(sample_input)
    # Multi-output is fine — `final` may be a tuple. The IR builder picks up
    # the actual output names from the FX `output` node below; we don't need
    # to special-case here.
    _ = final

    # Per-tensor activation max-abs, aggregated across the full
    # calibration set when one is supplied. Each extra sample widens the
    # per-tensor max-abs to its true distribution-wide bound, which is
    # what fixes the cls-logit saturation seen with single-sample
    # calibration on detection models.
    input_node_name = next(iter(gm.graph.nodes)).name
    max_abs: dict[str, float] = {}
    max_abs[input_node_name] = float(sample_input.detach().abs().max().item())
    for nname, t in cap.tensors.items():
        max_abs[nname] = float(t.detach().abs().max().item())

    if calibration_samples:
        extra = [s for s in calibration_samples
                 if s is not sample_input]
        for i, s in enumerate(extra):
            cap_i = _CaptureTensors(gm)
            cap_i.run(s)
            cur = float(s.detach().abs().max().item())
            if cur > max_abs[input_node_name]:
                max_abs[input_node_name] = cur
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
    ops: list[dict] = []
    input_name: str | None = None
    output_name: str | None = None

    # Helper: register a tensor in the IR with its int8 scale + zero_point.
    def _record(nname: str, dtype: str = "i8") -> None:
        t = cap.tensors.get(nname)
        if t is None:
            # Placeholder (input)
            shape = list(sample_input.shape)
        else:
            shape = list(t.shape)
        tensors_meta[nname] = {
            "shape": shape,
            "dtype": dtype,
            "quant": {"scale": scales[nname], "zero_point": 0},
        }

    # Default: single-output. The output handler may overwrite this.
    output_names_multi: Optional[list[str]] = None

    # Two-pass walk: first collect linear→relu and conv2d→relu fusions so the
    # relu node is absorbed into the producer's op kind.
    nodes = list(gm.graph.nodes)
    fused_relu_after: set[str] = set()  # names of relu nodes that get fused
    for i, node in enumerate(nodes):
        if i + 1 >= len(nodes):
            continue
        nxt = nodes[i + 1]
        is_fusable_producer = node.op == "call_module" and isinstance(
            gm.get_submodule(node.target),
            (torch.nn.Linear, torch.nn.Conv2d),
        )
        is_next_relu = (
            (nxt.op == "call_module"
             and isinstance(gm.get_submodule(nxt.target), torch.nn.ReLU))
            or (nxt.op == "call_function" and nxt.target in (
                torch.relu, torch.nn.functional.relu))
        )
        if is_fusable_producer and is_next_relu and len(nxt.args) == 1 \
                and nxt.args[0] is node:
            fused_relu_after.add(nxt.name)

    # Nodes to skip during the main walk (e.g. getitem consumers of chunk).
    _skip_nodes: set = set()

    for node in nodes:
        if node in _skip_nodes:
            continue
        if node.op == "placeholder":
            input_name = node.name
            _record(node.name, dtype="i8")

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
                real_mult = (in_scale * w_scale) / out_scale
                multiplier, shift = _requantize_multiplier_shift(real_mult)
                next_node = nodes[nodes.index(node) + 1] if nodes.index(node) + 1 < len(nodes) else None
                fuse_relu = (next_node is not None
                             and next_node.name in fused_relu_after)
                act_min = 0 if fuse_relu else -128
                act_max = 127
                in_shape = tensors_meta[in_name]["shape"]
                out_shape = tensors_meta[node.name]["shape"]
                N_, IC, IH, IW = (int(s) for s in in_shape)
                _, OC, OH, OW = (int(s) for s in out_shape)
                KH, KW = _pair(mod.kernel_size)
                SH, SW = _pair(mod.stride)
                PH, PW = _pair(mod.padding)
                ops.append({
                    "name": str(node.target),
                    "op": "conv2d_s8",
                    "inputs": [in_name],
                    "outputs": [
                        next_node.name if fuse_relu else node.name
                    ],
                    "weight": w_key,
                    "bias": b_key,
                    "shape": {
                        "N": N_, "IC": IC, "IH": IH, "IW": IW,
                        "OC": OC, "OH": OH, "OW": OW,
                        "KH": KH, "KW": KW,
                        "SH": SH, "SW": SW,
                        "PH": PH, "PW": PW,
                    },
                    "quant": {
                        "input_offset": 0,
                        "filter_offset": 0,
                        "output_offset": 0,
                        "output_multiplier": multiplier,
                        "output_shift": shift,
                        "activation_min": act_min,
                        "activation_max": act_max,
                    },
                })
                if fuse_relu and next_node.name not in tensors_meta:
                    tensors_meta[next_node.name] = dict(tensors_meta[node.name])

            elif isinstance(mod, torch.nn.MaxPool2d):
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
                ops.append({
                    "name": str(node.target),
                    "op": "batchnorm2d_s8",
                    "inputs": [in_name],
                    "outputs": [node.name],
                    "weight": s_key,
                    "bias": b_key,
                    "shape": {"N": N_, "C": C, "H": H, "W": W},
                    "quant": {
                        "scale_in":   scales[in_name],
                        "scale_out":  scales[node.name],
                        "activation_min": -128,
                        "activation_max": 127,
                    },
                })

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
                ops.append({
                    "name": node.name,
                    "op": "add_s8",
                    "inputs": [a_name, b_name],
                    "outputs": [node.name],
                    "shape": {"n": n},
                    "quant": {
                        "scale_a":   scales[a_name],
                        "scale_b":   scales[b_name],
                        "scale_out": scales[node.name],
                        "activation_min": -128,
                        "activation_max": 127,
                    },
                })
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
                if len(first_shape) != 4:
                    raise NotImplementedError(
                        f"int8 extract: cat at {node.name}: only 4D NCHW "
                        f"inputs supported."
                    )
                N_, _, H_, W_ = (int(s) for s in first_shape)
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

    if input_name is None or output_name is None:
        raise RuntimeError("int8 extract: graph missing input/output")

    output_tensors = (output_names_multi
                      if output_names_multi is not None
                      else [output_name])

    dispatches = _annotate_dispatches(ops)
    ir = {
        "name": name,
        "version": 1,
        "quant": "int8",
        "input": {"tensor": input_name},
        "output": {
            "tensors": output_tensors,
            "tensor": output_tensors[0] if len(output_tensors) == 1 else None,
        },
        "tensors": tensors_meta,
        "ops": ops,
        "dispatches": dispatches,
    }

    # Quantize the input once with the IR's input scale.
    in_scale = scales[input_name]
    inp_q = _quantize_per_tensor_sym(sample_input, in_scale).reshape(
        list(sample_input.shape)
    )

    # Simulate the integer pipeline in Python so the golden output matches
    # bit-exactly what the C kernel will produce. Walking the IR ops:
    activations: dict[str, np.ndarray] = {input_name: inp_q.astype(np.int8)}
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
        elif op["op"] == "relu_s8":
            activations[out_name] = np.maximum(in_arr, 0).astype(np.int8)
        elif op["op"] == "conv2d_s8":
            sh = op["shape"]
            q = op["quant"]
            w_q = weights_blob[op["weight"]].astype(np.int32)  # [OC, IC, KH, KW]
            b_q = weights_blob[op["bias"]].astype(np.int32)    # [OC]
            in_4d = in_arr.reshape(sh["N"], sh["IC"], sh["IH"], sh["IW"]).astype(np.int32)
            # Compute via direct sliding window (slow but correct simulator).
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
                                # Build the input slice for this (kh, kw).
                                ih_start = -PH + kh
                                iw_start = -PW + kw
                                # Compute valid output ranges.
                                for oh in range(OH):
                                    ih = oh * SH + ih_start
                                    if ih < 0 or ih >= sh["IH"]:
                                        # padded: in_v = input_offset
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
            activations[out_name] = scaled.astype(np.int8)
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
            sh = op["shape"]
            q = op["quant"]
            scale_per_ch = weights_blob[op["weight"]].astype(np.float32)
            bias_per_ch = weights_blob[op["bias"]].astype(np.float32)
            in_4d = in_arr.reshape(sh["N"], sh["C"], sh["H"], sh["W"]).astype(np.float32)
            scale_in = np.float32(q["scale_in"])
            scale_out = np.float32(q["scale_out"])
            fv = in_4d * scale_in
            y = scale_per_ch[None, :, None, None] * fv + bias_per_ch[None, :, None, None]
            v = np.round(y / scale_out).astype(np.int32)
            v = np.clip(v, q["activation_min"], q["activation_max"])
            activations[out_name] = v.astype(np.int8)
        elif op["op"] == "sigmoid_s8":
            q = op["quant"]
            fv = in_arr.astype(np.float32) * np.float32(q["scale_in"])
            sig = 1.0 / (1.0 + np.exp(-fv.astype(np.float32)))
            v = np.round(sig.astype(np.float32) / np.float32(q["scale_out"])).astype(np.int32)
            v = np.clip(v, q["activation_min"], q["activation_max"])
            activations[out_name] = v.astype(np.int8)
        elif op["op"] == "silu_s8":
            q = op["quant"]
            fv = in_arr.astype(np.float32) * np.float32(q["scale_in"])
            silu_out = fv / (np.float32(1.0) + np.exp(-fv).astype(np.float32))
            v = np.round(silu_out.astype(np.float32) / np.float32(q["scale_out"])).astype(np.int32)
            v = np.clip(v, q["activation_min"], q["activation_max"])
            activations[out_name] = v.astype(np.int8)
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
        else:
            raise NotImplementedError(
                f"int8 simulator: unsupported op {op['op']}"
            )

    # Concatenate outputs in IR order (matching multi-output goldens elsewhere).
    out_q = np.concatenate([
        activations[t].reshape(-1).astype(np.int8) for t in output_tensors
    ])
    inp_q = inp_q.reshape(-1)

    ir_path = os.path.join(out_dir, "graph.json")
    weights_path = os.path.join(out_dir, "weights.npz")
    io_path = os.path.join(out_dir, "io.npz")
    with open(ir_path, "w") as f:
        json.dump(ir, f, indent=2)
    np.savez(weights_path, **weights_blob)
    np.savez(io_path, input=inp_q, output=out_q)
    print(f"wrote {ir_path}")
    print(f"wrote {weights_path}  ({len(weights_blob)} tensors)")
    print(f"wrote {io_path}  (input dtype={inp_q.dtype}, output dtype={out_q.dtype})")
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
        if (_n.op == "call_function" and
                _n.target in (torch.matmul, torch.mm)):
            for _arg in _n.args[:2]:
                if _is_transpose_node(_arg):
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
                if mod.dilation != (1, 1):
                    raise NotImplementedError(
                        f"Conv2d with dilation={mod.dilation} not supported "
                        f"yet at {node.name}"
                    )
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
                        },
                    })
                elif mod.groups == IC and IC == OC:
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
                trans_a = _is_transpose_node(arg_a_node)
                trans_b = _is_transpose_node(arg_b_node)
                a_node = arg_a_node.args[0] if trans_a else arg_a_node
                b_node = arg_b_node.args[0] if trans_b else arg_b_node
                # Shape before transpose: a_shape is (K,M) if trans_a else (M,K)
                a_shape = list(tensors[a_node.name]["shape"])
                b_shape = list(tensors[b_node.name]["shape"])
                if trans_a:
                    K, M = int(a_shape[0]), int(a_shape[1])
                else:
                    M, K = int(a_shape[0]), int(a_shape[1])
                N = int(b_shape[0]) if trans_b else int(b_shape[1])
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
                if len(first_shape) != 4:
                    raise NotImplementedError(
                        f"cat at {node.name}: only 4D NCHW inputs supported."
                    )
                N_, _, H_, W_ = (int(s) for s in first_shape)
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
            ref_inputs_exec = [t.half() for t in sample_inputs]
            ref_model = model.half()
            torch_dtype = torch.float16
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlp_generic",
                    choices=["mlp_generic", "mlp_control", "lenet", "dronet",
                             "mobilenet_v2", "yolov8_nano"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--quant", default="fp32", choices=["fp32", "fp16", "int8"],
                    help="quantization mode. fp32 = stock float, fp16 = "
                         "half precision (uses torch.float16 model + "
                         "_Float16 C kernels, validated against half-cast "
                         "torch golden), int8 = symmetric per-tensor PTQ.")
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
    args = ap.parse_args()

    if args.model == "mlp_generic":
        from modelblaster.models import mlp_generic as model_mod
    elif args.model == "mlp_control":
        from modelblaster.models import mlp_control as model_mod
    elif args.model == "lenet":
        from modelblaster.models import lenet as model_mod
    elif args.model == "dronet":
        from modelblaster.models import dronet as model_mod
    elif args.model == "mobilenet_v2":
        from modelblaster.models import mobilenet_v2 as model_mod
    elif args.model == "yolov8_nano":
        from modelblaster.models import yolov8_nano as model_mod
    else:
        raise SystemExit(f"unknown model {args.model}")
    model = model_mod.get_model()
    sample = model_mod.get_sample_input()

    calibration_samples = None
    if args.quant == "int8" and args.num_calibration > 1:
        if hasattr(model_mod, "get_calibration_spec"):
            from modelblaster.datasets import materialize_calibration_samples  # noqa: PLC0415
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

    extract(model, sample, name=args.model, out_dir=args.out_dir,
            quant=args.quant,
            calibration_samples=calibration_samples)

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
