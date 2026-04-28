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


SUPPORTED_MODULES = (
    torch.nn.Linear,
    torch.nn.ReLU,
    torch.nn.ELU,
    torch.nn.Conv2d,
    torch.nn.MaxPool2d,
    torch.nn.Dropout,  # eval-mode no-op; we still record a passthrough alias
    torch.nn.BatchNorm2d,  # pre-folded into a per-channel scale + bias
    torch.nn.Sigmoid,
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
             torch.int32: "i32"}.get(tm.dtype)
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
) -> dict[str, Any]:
    """int8 PTQ extractor.

    Approach (intentionally minimal first cut):
      * Per-tensor symmetric quant for both weights and activations
        (zero_point = 0 throughout).
      * Activation scales calibrated from a single forward pass on
        `sample_input`. Crude but reproducible; replace with a real
        calibration set later.
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

    # Per-tensor scales (input + every node output).
    scales: dict[str, float] = {}
    scales[next(iter(gm.graph.nodes)).name] = _scale_from_max_abs(sample_input)
    for nname, t in cap.tensors.items():
        scales[nname] = _scale_from_max_abs(t)

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

    for node in nodes:
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
            else:
                raise NotImplementedError(
                    f"int8 extract: unsupported function {tname} at {node.name}"
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
            out = np.zeros((sh["N"], sh["C"], OH, OW), dtype=np.int8)
            for oh in range(OH):
                for ow in range(OW):
                    patch = in_4d[:, :, oh*SH:oh*SH+KH, ow*SW:ow*SW+KW]
                    out[:, :, oh, ow] = patch.reshape(sh["N"], sh["C"], -1).max(axis=2)
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
    sample_input: torch.Tensor,
    name: str,
    out_dir: str,
    quant: str = "fp32",
) -> dict[str, Any]:
    """Trace `model`, dump IR + weights + I/O into `out_dir`.

    `quant` is recorded in the IR top-level field so downstream stages (and
    cache-key naming) can branch on it.
    """
    if quant == "int8":
        return extract_int8(model, sample_input, name, out_dir)
    if quant != "fp32":
        raise NotImplementedError(
            f"quant={quant!r} not supported (have: fp32, int8)"
        )
    os.makedirs(out_dir, exist_ok=True)
    model = model.eval()

    gm = torch.fx.symbolic_trace(model)
    ShapeProp(gm).propagate(sample_input)

    tensors: dict[str, dict] = {}
    ops: list[dict] = []
    weights: dict[str, np.ndarray] = {}

    input_name: str | None = None
    output_names: list[str] = []

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            input_name = node.name
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
                weights[w_key] = mod.weight.detach().cpu().numpy().astype(np.float32)
                if b_key is not None:
                    weights[b_key] = mod.bias.detach().cpu().numpy().astype(np.float32)
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
                if mod.groups != 1:
                    raise NotImplementedError(
                        f"Conv2d with groups={mod.groups} not supported yet "
                        f"at {node.name}"
                    )
                if mod.dilation != (1, 1):
                    raise NotImplementedError(
                        f"Conv2d with dilation={mod.dilation} not supported "
                        f"yet at {node.name}"
                    )
                w_key = f"{node.target}.weight"
                b_key = f"{node.target}.bias" if mod.bias is not None else None
                weights[w_key] = mod.weight.detach().cpu().numpy().astype(np.float32)
                if b_key is not None:
                    weights[b_key] = mod.bias.detach().cpu().numpy().astype(np.float32)
                in_shape = tensors[in_name]["shape"]
                out_shape = tensors[out_name]["shape"]
                # NCHW.
                N_, IC, IH, IW = (int(s) for s in in_shape)
                _, OC, OH, OW = (int(s) for s in out_shape)
                KH, KW = _pair(mod.kernel_size)
                SH, SW = _pair(mod.stride)
                PH, PW = _pair(mod.padding)
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

            elif isinstance(mod, torch.nn.MaxPool2d):
                in_shape = tensors[in_name]["shape"]
                out_shape = tensors[out_name]["shape"]
                N_, C, IH, IW = (int(s) for s in in_shape)
                _, _, OH, OW = (int(s) for s in out_shape)
                KH, KW = _pair(mod.kernel_size)
                SH, SW = _pair(mod.stride)
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
                weights[s_key] = scale.astype(np.float32)
                weights[b_key] = bias_fused.astype(np.float32)
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

        elif node.op == "call_function":
            out_name = node.name
            tensors[out_name] = _tensor_meta(node)
            target = node.target
            tname = getattr(target, "__name__", str(target))
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
            else:
                raise NotImplementedError(f"unsupported function {tname} at {node.name}")

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

    if input_name is None or not output_names:
        raise RuntimeError("graph missing input/output")

    dispatches = _annotate_dispatches(ops)
    ir = {
        "name": name,
        "version": 1,
        "quant": quant,
        "input": {"tensor": input_name},
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

    # Run reference to capture golden I/O.
    with torch.no_grad():
        out = model(sample_input)

    # Multi-output models return a tuple; flatten in IR-output order so the
    # downstream comparator just needs to do an elementwise compare.
    if isinstance(out, (tuple, list)):
        flat = np.concatenate([
            o.detach().cpu().numpy().astype(np.float32).reshape(-1) for o in out
        ])
    else:
        flat = out.detach().cpu().numpy().astype(np.float32).reshape(-1)

    ir_path = os.path.join(out_dir, "graph.json")
    weights_path = os.path.join(out_dir, "weights.npz")
    io_path = os.path.join(out_dir, "io.npz")

    with open(ir_path, "w") as f:
        json.dump(ir, f, indent=2)
    np.savez(weights_path, **weights)
    np.savez(
        io_path,
        input=sample_input.detach().cpu().numpy().astype(np.float32),
        output=flat,
    )

    print(f"wrote {ir_path}")
    print(f"wrote {weights_path}  ({len(weights)} tensors)")
    print(f"wrote {io_path}")
    return ir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlp_generic",
                    choices=["mlp_generic", "mlp_control", "lenet", "dronet"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--quant", default="fp32",
                    help="quantization mode (fp32 only for now; recorded in "
                         "the IR for downstream stages)")
    ap.add_argument("--core-registry", default=None,
                    help="optional path to an agents/cores/*.json registry. "
                         "When provided, the post-extraction pass validates "
                         "every dispatch's hardware_target against the listed "
                         "cores' capabilities and aborts on mismatch.")
    args = ap.parse_args()

    if args.model == "mlp_generic":
        from agents.models import mlp_generic as model_mod
    elif args.model == "mlp_control":
        from agents.models import mlp_control as model_mod
    elif args.model == "lenet":
        from agents.models import lenet as model_mod
    elif args.model == "dronet":
        from agents.models import dronet as model_mod
    else:
        raise SystemExit(f"unknown model {args.model}")
    model = model_mod.get_model()
    sample = model_mod.get_sample_input()

    extract(model, sample, name=args.model, out_dir=args.out_dir,
            quant=args.quant)

    if args.core_registry:
        from agents.pipeline import core_registry
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
