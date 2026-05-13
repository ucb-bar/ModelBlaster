"""Export-based graph extractor for agents pipeline.

Companion to ``extract_graph.py`` (FX-based). Use this path when
``torch.fx.symbolic_trace`` can't trace the model — e.g. ViNT, which
hits ``len(...)`` calls inside EfficientNet plus in-place index
assignments + nn.TransformerEncoder internals. ``torch.export`` traces
at aten-op granularity instead of fx-module granularity, so models
with dynamic-Python-shape branches lower cleanly.

Two execution modes:

* ``--inventory-only`` (default): trace, classify every aten op, print
  a human-readable inventory + work-list. Used for op-coverage triage.
* No flag: emit ``graph.json`` / ``weights.npz`` / ``io.npz`` triple
  consumable by ``generate_skeleton.py`` / ``generate_kernels.py``,
  same shape as ``extract_graph.py::extract_int8``.

Usage:

    conda activate xpurt   # ViNT/efficientnet deps live here
    PYTHONPATH=. python -m agents.pipeline.extract_graph_export \\
        --model vint --quant int8 --out-dir <out-dir>

Run via the ``xpurt`` env so the vendored vint_train + efficientnet
deps are importable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

# Reuse the scale + capture helpers + IR conventions from the FX-based
# extractor. Keeps the on-disk format identical (downstream codegen +
# verify don't have to know which extractor produced the IR).
from agents.pipeline.extract_graph import (  # noqa: E402
    _CaptureTensors,
    _INT8_RANGE,  # noqa: F401  — used implicitly via _scale_from_max_abs
    _scale_from_max_abs,
    _quantize_per_tensor_sym,
)


# ----------------------------------------------------------------------
# Aten op vocabulary (kept up-to-date as we add walkers).
# ----------------------------------------------------------------------

# Aten ops we already lower to integer kernels in extract_int8 today,
# and which extract_graph_export now emits as full IR records.
_SUPPORTED_COMPUTE = {
    "conv2d.default",           # incl. depthwise via groups arg (kernel branches)
    "linear.default",
    "relu.default",
    "add.Tensor",               # residual add → add_s8 (gemmini_q31 resadd)
    "max_pool2d.default",
    "max_pool2d_with_indices.default",
    "cat.default",              # → cat{2,3,4}_c1_s8
    "concat.default",
}

# New compute ops we identified for ViNT. The extractor emits IR
# records for these so the downstream skeleton/codegen pipeline can
# wire them up incrementally. Kernels are added op-by-op in follow-ups
# (see agents/notes/vint_zephyr_plan.md §B for the order).
_NEW_COMPUTE = {
    "mul.Tensor":                  "mul_s8",
    "sigmoid.default":             "sigmoid_s8",
    "gelu.default":                "gelu_s8",
    "layer_norm.default":          "layer_norm_s8",
    "adaptive_avg_pool2d.default": "adaptive_avg_pool2d_s8",
    # scaled_dot_product_attention is DECOMPOSED in the walker into
    # (matmul_s8, softmax_s8, matmul_s8) so it's not a single op here.
    "scaled_dot_product_attention.default": "_decompose_sdpa",
}

# Folded into the preceding op at extract time. The walker does the
# fusion; no IR record is emitted for these.
_FOLD = {
    "batch_norm.default",       # fold into preceding conv2d
    # `pad.default` is handled specially in the walker: symmetric pads
    # fold into the next conv2d's padding tuple; asymmetric pads
    # (EfficientNet's same-padding-with-stride pattern) become
    # standalone pad_s8 records below.
    "pad.default",
}

# Zero-compute aliases — same pass-through pattern the FX extractor uses.
# The dispatch-id remap in ingest_xpurt_schedule already filters these
# from the codegen table.
_ALIAS = {
    "view.default", "transpose.int", "slice.Tensor", "select.int",
    "unsqueeze.default", "unflatten.int", "squeeze.dim",
    "contiguous.default", "permute.default", "reshape.default",
    "flatten.using_ints", "split.Tensor", "expand_as.default",
    "clone.default",
}

# Eval-time / artifact no-ops. Skipped entirely.
_NOOP = {
    "dropout.default",
    "wrap_with_set_grad_enabled",
    "<built-in function getitem>",
    "copy_.default",
}

# Tail post-process — runs in scalar code in the C harness's main, not
# as a profiled dispatch.
_TAIL = {
    "cumsum.default",
    "linalg_vector_norm.default",
    "clamp_min.default",
    "div.Tensor",
}


# ----------------------------------------------------------------------
# Inventory helper (cheap path: trace, classify, print).
# ----------------------------------------------------------------------

def _classify(name: str) -> str:
    if name in _SUPPORTED_COMPUTE: return "supported"
    if name in _NEW_COMPUTE:        return "new"
    if name in _ALIAS:              return "alias"
    if name in _FOLD:               return "fold"
    if name in _NOOP:               return "noop"
    if name in _TAIL:               return "tail"
    return "UNKNOWN"


def _op_name(node) -> str:
    if node.op == "call_function":
        return str(node.target).split('.OverloadPacket')[0].replace('aten.', '')
    if node.op == "call_method":
        return f"method:{node.target}"
    return node.op


def _print_inventory(graph_module, out_dir: Path):
    by_class: dict[str, Counter] = defaultdict(Counter)
    for n in graph_module.graph.nodes:
        if n.op in ("placeholder", "get_attr", "output"):
            continue
        opname = _op_name(n)
        by_class[_classify(opname)][opname] += 1

    report_path = out_dir / "op_inventory.txt"
    with open(report_path, "w") as f:
        for cls in ("supported", "new", "fold", "alias", "noop", "tail", "UNKNOWN"):
            if not by_class[cls]:
                continue
            f.write(f"=== {cls} ({sum(by_class[cls].values())}) ===\n")
            for op, n in by_class[cls].most_common():
                f.write(f"  {n:5d}  {op}\n")
            f.write("\n")
    print(f"[extract_export] op inventory → {report_path}", flush=True)

    if by_class["UNKNOWN"]:
        print("\n!!! UNKNOWN ops below need classification in extract_graph_export.py:")
        for op, n in by_class["UNKNOWN"].most_common():
            print(f"    {n:5d}  {op}")
        return False
    return True


# ----------------------------------------------------------------------
# Walker — emit IR.
# ----------------------------------------------------------------------

class _ExportWalker:
    """Walk an aten-level fx Graph (from ``torch.export``) and emit the
    agents IR records (one per supported op kind).

    Conventions match ``extract_graph.py::extract_int8``:
    * per-tensor symmetric quant (zero_point = 0)
    * activation scales from max-abs of a single forward pass
    * weights packed via ``_quantize_per_tensor_sym`` and stored under
      a ``<op>.weights`` key in weights.npz
    * folded batchnorm: scale+bias adjustments are merged into the
      preceding conv's weight + bias before quantization
    * folded pad: constant zero pad is collapsed into the conv's
      padding arg; non-zero / non-constant pads raise
    """

    def __init__(self, ep, calibration_tensors: list[dict[str, torch.Tensor]],
                 input_order: list[str]):
        self.ep = ep
        self.gm = ep.graph_module
        # Aten graph runs over ``flat_args`` (placeholders include
        # params, buffers, then user inputs). The ExportedProgram knows
        # how to materialize those for us via ``module()``.
        self.runtime = ep.module()
        self.calib = calibration_tensors  # list of {name: tensor} dicts
        self.input_order = input_order    # positional order for forward()
        # placeholder-name → state-dict-key for PARAMETER + BUFFER
        # inputs. Lets _resolve_input materialize buffer tensors that
        # flow into compute ops (e.g. positional encoding into add).
        self.placeholder_to_sd: dict[str, str] = {}
        for spec in ep.graph_signature.input_specs:
            kind = spec.kind.name if hasattr(spec.kind, "name") else str(spec.kind)
            if kind in ("PARAMETER", "BUFFER", "CONSTANT_TENSOR"):
                self.placeholder_to_sd[spec.arg.name] = spec.target
        # Tensor capture from the first calibration sample — defines
        # shapes throughout the IR.
        self.tensors: dict[str, torch.Tensor] = {}
        # Per-tensor scales, computed from max-abs over the full
        # calibration set.
        self.scales: dict[str, float] = {}
        # Output IR.
        self.weights_blob: dict[str, np.ndarray] = {}
        self.tensors_meta: dict[str, dict] = {}
        self.ops: list[dict] = []
        # Map from node-name → tensor-name. Aliases let multiple
        # node names resolve to the same logical tensor.
        self.name_map: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Phase 1: run the model, capture every tensor + accumulate max-abs.
    # ------------------------------------------------------------------
    def calibrate(self):
        max_abs: dict[str, float] = {}
        captured_first: dict[str, torch.Tensor] = {}
        for i, sample in enumerate(self.calib):
            # ep.module() returns a callable Module; we call it with
            # the sample inputs to drive forward. Use the recorded
            # positional input order (NOT dict-sorted keys — that
            # alphabetizes "goal" before "obs" and the shape check
            # rejects it).
            args = tuple(sample[k] for k in self.input_order)
            # Re-trace this sample to capture intermediates. The
            # exported graph_module is a regular fx GraphModule; we
            # interpret it directly.
            cap = _CaptureTensors(self.gm)
            # Build the placeholder-bound input list. The exported
            # graph's first placeholders are params/buffers; the
            # ``runtime`` Module handles wiring them, so we use
            # ep.module() to actually run forward and capture via the
            # interpreter on the same graph_module.
            try:
                _ = self.runtime(*args)
            except Exception as e:
                raise RuntimeError(
                    f"calibration sample {i} failed forward pass: {e}"
                )
            # Now re-run via the interpreter (cheap — same numerics).
            # The interpreter needs the same flat-args; ep has a helper.
            flat = self._build_flat_args(args)
            cap_out = cap.run(*flat)  # noqa: F841
            for nname, t in cap.tensors.items():
                if not isinstance(t, torch.Tensor):
                    continue
                m = float(t.detach().abs().max().item())
                if nname not in max_abs or m > max_abs[nname]:
                    max_abs[nname] = m
            if i == 0:
                captured_first = dict(cap.tensors)
        self.tensors = captured_first
        for nname, m in max_abs.items():
            self.scales[nname] = max(m, 1e-8) / 127.0
        # Input placeholders aren't in cap.tensors — seed their scales
        # from the first calibration sample directly.
        first = self.calib[0]
        for k in self.input_order:
            t = first[k]
            self.scales[k] = _scale_from_max_abs(t)

    def _build_flat_args(self, user_args):
        """Materialize the full placeholder list (params + buffers +
        user inputs) for the aten graph. ExportedProgram tracks the
        ordering on ``ep.graph_signature``.
        """
        sig = self.ep.graph_signature
        sd = dict(self.ep.state_dict)
        flat: list = []
        for spec in sig.input_specs:
            kind = spec.kind.name if hasattr(spec.kind, "name") else str(spec.kind)
            if kind == "PARAMETER":
                flat.append(sd[spec.target])
            elif kind == "BUFFER":
                flat.append(sd[spec.target])
            elif kind == "USER_INPUT":
                # USER_INPUT order matches user_args order.
                flat.append(user_args[len([f for f in flat if isinstance(f, torch.Tensor) and id(f) in {id(u) for u in user_args}])])
            elif kind == "CONSTANT_TENSOR":
                flat.append(sd.get(spec.target))
            else:
                raise NotImplementedError(f"input kind {kind} not handled")
        return flat

    # ------------------------------------------------------------------
    # Phase 2 — IR emit (this is the bulk; one branch per op kind).
    # ------------------------------------------------------------------
    def emit(self):
        # Track tensors we have already recorded (for shape lookup).
        for n in self.gm.graph.nodes:
            if n.op == "placeholder":
                # User inputs are recorded by name; param/buffer
                # placeholders we skip (they become weight tables on
                # the relevant op).
                pass  # handled lazily below
        # Walk in topological order (already true for fx Graph).
        for n in self.gm.graph.nodes:
            self._visit(n)

    def _resolve_input(self, val) -> str:
        """Map an fx-Node arg to the canonical tensor name used in IR.

        Special-cases placeholders (user inputs) and buffer/parameter
        ``get_attr`` nodes — both reference data the walker needs to
        record as a tensor (with a scale) before downstream ops can
        consume them.
        """
        if isinstance(val, torch.fx.Node):
            name = val.name
            # If a get_attr buffer (e.g. positional encoding) flows
            # into a compute op, surface it as a constant tensor with
            # its own scale + weights entry so the IR is self-contained.
            if val.op == "get_attr" and name not in self.tensors_meta:
                t = _get_attr_value(self.gm, val.target)
                self._record_constant(name, t)
            # Same treatment for PARAMETER / BUFFER / CONSTANT_TENSOR
            # placeholders — those carry data that's referenced by
            # compute ops via their placeholder name.
            elif val.op == "placeholder" and name in self.placeholder_to_sd \
                    and name not in self.tensors_meta:
                sd_key = self.placeholder_to_sd[name]
                t = self.ep.state_dict.get(sd_key)
                if t is None:
                    t = dict(self.ep.named_buffers()).get(sd_key)
                self._record_constant(name, t)
            return self.name_map.get(name, name)
        if isinstance(val, (list, tuple)):
            return [self._resolve_input(v) for v in val]
        return val

    def _record_constant(self, name: str, t):
        if not isinstance(t, torch.Tensor):
            return
        self.tensors[name] = t.detach().clone()
        sc = _scale_from_max_abs(t)
        self.scales[name] = sc
        self.tensors_meta[name] = {
            "shape": list(t.shape),
            "dtype": "i8",
            "quant": {"scale": sc, "zero_point": 0,
                      "kind": "constant_buffer"},
        }
        self.weights_blob[name] = _quantize_per_tensor_sym(t, sc)

    def _record_tensor(self, name: str, dtype: str = "i8"):
        if name in self.tensors_meta:
            return
        t = self.tensors.get(name)
        if t is None:
            # Try input placeholder by inspecting calib sample dict.
            for sample in self.calib:
                if name in sample:
                    t = sample[name]
                    break
        if t is None:
            return  # constant / unused
        self.tensors_meta[name] = {
            "shape": list(t.shape),
            "dtype": dtype,
            "quant": {
                "scale": self.scales.get(name, 1e-8),
                "zero_point": 0,
            },
        }

    def _visit(self, n):
        # Skip structural nodes — those don't represent computation.
        # Placeholders are param/buffer/user_input declarations;
        # get_attr surfaces a parameter; output marks the graph tail.
        if n.op in ("placeholder", "get_attr", "output"):
            return
        # Hand-tagged dispatch on op kind. Keep cases short and
        # delegate complex emit to dedicated helpers.
        op_kind = _op_name(n)
        cls = _classify(op_kind)
        if cls in ("noop", "tail", "alias"):
            # Pass-through: alias this node's name to its first tensor
            # input so downstream consumers resolve to the underlying
            # tensor. Covers getitem (tuple unpack from an op return),
            # dropout (eval-time identity), view/transpose/slice/...,
            # and the scalar-tail ops (no compute, no IR record).
            #
            # Use _resolve_input on each arg — that path records buffer
            # / get_attr placeholders if they haven't been seen, and
            # returns the canonical tensor name. Pick the first arg
            # whose resolved name is now in tensors or tensors_meta.
            self._record_tensor(n.name)
            src_name: Optional[str] = None
            for a in n.args:
                if not isinstance(a, torch.fx.Node):
                    continue
                resolved = self._resolve_input(a)
                if not isinstance(resolved, str):
                    continue
                if resolved in self.tensors or resolved in self.tensors_meta:
                    src_name = resolved
                    break
            if src_name is not None:
                self.name_map[n.name] = src_name
            return
        if cls == "fold":
            # Walker fuses these into the preceding op; nothing emitted.
            self._record_tensor(n.name)
            return
        if cls == "supported":
            self._emit_supported(n)
            return
        if cls == "new":
            self._emit_new(n)
            return
        raise NotImplementedError(
            f"extract_graph_export: don't know how to handle aten op "
            f"{op_kind!r} (node {n.name}). Add a case to _visit()."
        )

    def _emit_supported(self, n):
        op_kind = _op_name(n)
        if op_kind == "conv2d.default":
            self._emit_conv2d(n); return
        if op_kind == "linear.default":
            self._emit_linear(n); return
        if op_kind in ("relu.default",):
            self._emit_relu(n); return
        if op_kind == "add.Tensor":
            self._emit_add(n); return
        if op_kind in ("cat.default", "concat.default"):
            self._emit_cat(n); return
        if op_kind in ("max_pool2d.default", "max_pool2d_with_indices.default"):
            self._emit_maxpool(n); return
        raise NotImplementedError(f"supported but no emit branch: {op_kind}")

    def _emit_new(self, n):
        """Emit IR records for the ViNT-new ops. Kernels for these are
        added op-by-op in follow-up commits — for now the IR is
        well-formed and codegen will fail with a clear missing-kernel
        message until a matching reference impl lands.
        """
        op_kind = _op_name(n)
        if op_kind == "scaled_dot_product_attention.default":
            self._emit_sdpa_decomposed(n); return
        s8_name = _NEW_COMPUTE[op_kind]
        in_name = self._resolve_input(n.args[0])
        out_name = n.name
        self._record_tensor(out_name)
        rec = {
            "name": str(n.name),
            "op": s8_name,
            "inputs": [in_name] if not isinstance(n.args[0], (list, tuple))
                       else self._resolve_input(n.args[0]),
            "outputs": [out_name],
            "shape": dict(self._shape_kv(out_name)),
            "quant": {
                "scale_in":  self.scales.get(in_name, 1e-8),
                "scale_out": self.scales.get(out_name, 1e-8),
            },
            "_pending_kernel": True,
        }
        if op_kind == "mul.Tensor":
            b = self._resolve_input(n.args[1])
            rec["inputs"] = [in_name, b]
            rec["quant"] = {
                "scale_a":   self.scales.get(in_name, 1e-8),
                "scale_b":   self.scales.get(b, 1e-8),
                "scale_out": self.scales.get(out_name, 1e-8),
            }
        self.ops.append(rec)

    def _emit_sdpa_decomposed(self, n):
        """SDPA(Q, K, V) → matmul(Q, Kᵀ)/√d → softmax → matmul(_, V)

        We emit three IR records sharing the SDPA node's name as a
        prefix (so debug printouts make the lineage obvious) and pin
        each one to a separate output tensor.
        """
        q, k, v = (self._resolve_input(a) for a in n.args[:3])
        head_dim = self.tensors[q.split('::')[0] if '::' in q else q].shape[-1] \
            if q in self.tensors else None
        scores_name = f"{n.name}::scores"
        weights_name = f"{n.name}::weights"
        out_name = n.name
        self._record_tensor(scores_name)
        self._record_tensor(weights_name)
        self._record_tensor(out_name)
        self.ops.append({
            "name": f"{n.name}_qk", "op": "matmul_s8",
            "inputs": [q, k], "outputs": [scores_name],
            "shape": dict(self._shape_kv(scores_name)),
            "quant": {
                "scale_a":   self.scales.get(q, 1e-8),
                "scale_b":   self.scales.get(k, 1e-8),
                "scale_out": self.scales.get(scores_name, 1e-8),
                "transpose_b": True,
                "scale_div_sqrt_dk": head_dim or 1,
            },
            "_pending_kernel": True,
        })
        self.ops.append({
            "name": f"{n.name}_softmax", "op": "softmax_s8",
            "inputs": [scores_name], "outputs": [weights_name],
            "shape": dict(self._shape_kv(weights_name)),
            "quant": {
                "scale_in":  self.scales.get(scores_name, 1e-8),
                "scale_out": self.scales.get(weights_name, 1e-8),
            },
            "_pending_kernel": True,
        })
        self.ops.append({
            "name": f"{n.name}_av", "op": "matmul_s8",
            "inputs": [weights_name, v], "outputs": [out_name],
            "shape": dict(self._shape_kv(out_name)),
            "quant": {
                "scale_a":   self.scales.get(weights_name, 1e-8),
                "scale_b":   self.scales.get(v, 1e-8),
                "scale_out": self.scales.get(out_name, 1e-8),
            },
            "_pending_kernel": True,
        })

    # ------------------------------------------------------------------
    # Conv2d emit — incl. batchnorm/pad fold + depthwise branch.
    # ------------------------------------------------------------------
    def _emit_conv2d(self, n):
        # aten::conv2d(input, weight, bias?, stride, padding, dilation, groups)
        in_name = self._resolve_input(n.args[0])
        weight_node = n.args[1]
        bias_node = n.args[2] if len(n.args) > 2 else None
        stride = _pair(n.args[3]) if len(n.args) > 3 else (1, 1)
        padding = _pair(n.args[4]) if len(n.args) > 4 else (0, 0)
        dilation = _pair(n.args[5]) if len(n.args) > 5 else (1, 1)
        groups = int(n.args[6]) if len(n.args) > 6 else 1

        w = self._get_param_tensor(weight_node)
        b = self._get_param_tensor(bias_node) if bias_node is not None else None
        OC, _IC_per_g, KH, KW = w.shape

        # Look ahead: is the next user of this conv's output a batch_norm?
        bn_user = self._find_bn_user(n)
        if bn_user is not None:
            w, b, post_name = self._fold_batchnorm(w, b, bn_user)
        else:
            post_name = n.name

        # Look behind: was a pad node feeding our input? Fold symmetric
        # pads into the conv's padding tuple. Asymmetric pads
        # (EfficientNet same-padding-with-stride pattern) get emitted
        # as a standalone pad_s8 record above us in the IR walk — the
        # pad node will already have been visited and we just consume
        # its output as our input below.
        if isinstance(n.args[0], torch.fx.Node) and \
           _op_name(n.args[0]) == "pad.default":
            pad_node = n.args[0]
            pads = list(pad_node.args[1]) if pad_node.args[1] else []
            # aten pad order: (left, right, top, bottom) for 4-d input.
            symmetric = (len(pads) >= 4 and pads[0] == pads[1]
                         and pads[2] == pads[3])
            all_zero = all(int(x) == 0 for x in pads) if pads else True
            if all_zero:
                in_name = self._resolve_input(pad_node.args[0])
            elif symmetric:
                padding = (padding[0] + int(pads[2]), padding[1] + int(pads[0]))
                in_name = self._resolve_input(pad_node.args[0])
            else:
                # Asymmetric pad: emit it as its own op (pad_s8) and
                # consume that record's output as this conv's input.
                self._emit_pad(pad_node)
                in_name = self._resolve_input(pad_node)

        is_depthwise = (groups == OC and groups != 1)
        op_kind = "depthwise_conv2d_s8" if is_depthwise else "conv2d_s8"

        w_scale = _scale_from_max_abs(w)
        in_scale = self.scales.get(in_name, 1e-8)
        out_scale = self.scales.get(post_name, 1e-8)

        w_q = _quantize_per_tensor_sym(w, w_scale)
        weights_key = f"{n.name}.weights"
        self.weights_blob[weights_key] = w_q
        if b is not None:
            b_q = (b.detach().cpu().numpy() / (in_scale * w_scale)).round().astype(np.int32)
            self.weights_blob[f"{n.name}.bias"] = b_q

        self._record_tensor(post_name)
        rec = {
            "name": str(n.name),
            "op": op_kind,
            "inputs": [in_name],
            "outputs": [post_name],
            "shape": {
                "N": int(self.tensors[post_name].shape[0]),
                "IC": int(w.shape[1] if not is_depthwise else 1),
                "IH": int(self.tensors[in_name].shape[2]),
                "IW": int(self.tensors[in_name].shape[3]),
                "OC": int(OC), "KH": int(KH), "KW": int(KW),
                "SH": int(stride[0]), "SW": int(stride[1]),
                "PH": int(padding[0]), "PW": int(padding[1]),
                "DH": int(dilation[0]), "DW": int(dilation[1]),
                "groups": int(groups),
            },
            "quant": {
                "scale_in":   in_scale,
                "scale_w":    w_scale,
                "scale_out":  out_scale,
                "input_offset":  0,
                "filter_offset": 0,
                "output_offset": 0,
                "weights_key": weights_key,
                "bias_key": f"{n.name}.bias" if b is not None else None,
            },
        }
        self.ops.append(rec)

    def _emit_linear(self, n):
        in_name = self._resolve_input(n.args[0])
        w = self._get_param_tensor(n.args[1])
        b = self._get_param_tensor(n.args[2]) if len(n.args) > 2 and n.args[2] is not None else None
        OF, IF = w.shape
        out_name = n.name
        self._record_tensor(out_name)
        in_scale = self.scales.get(in_name, 1e-8)
        out_scale = self.scales.get(out_name, 1e-8)
        w_scale = _scale_from_max_abs(w)
        self.weights_blob[f"{n.name}.weights"] = _quantize_per_tensor_sym(w, w_scale)
        if b is not None:
            self.weights_blob[f"{n.name}.bias"] = (
                b.detach().cpu().numpy() / (in_scale * w_scale)
            ).round().astype(np.int32)
        self.ops.append({
            "name": str(n.name),
            "op": "linear_s8",
            "inputs": [in_name],
            "outputs": [out_name],
            "shape": {"M": 1, "K": int(IF), "N": int(OF)},
            "quant": {
                "scale_in": in_scale,
                "scale_w":  w_scale,
                "scale_out": out_scale,
                "input_offset": 0, "filter_offset": 0, "output_offset": 0,
                "weights_key": f"{n.name}.weights",
                "bias_key": f"{n.name}.bias" if b is not None else None,
            },
        })

    def _emit_relu(self, n):
        in_name = self._resolve_input(n.args[0])
        out_name = n.name
        self._record_tensor(out_name)
        self.ops.append({
            "name": str(n.name), "op": "relu_s8",
            "inputs": [in_name], "outputs": [out_name],
            "shape": {"n": int(np.prod(self.tensors[out_name].shape))},
            "quant": {
                "scale_in":  self.scales.get(in_name, 1e-8),
                "scale_out": self.scales.get(out_name, 1e-8),
                "activation_min": 0,  # int8: 0 maps to symmetric -128 + 128
                "activation_max": 127,
            },
        })

    def _emit_add(self, n):
        a = self._resolve_input(n.args[0])
        bv = n.args[1]
        if not isinstance(bv, torch.fx.Node):
            # Adding a scalar/constant — uncommon in our paths; skip
            # quantization-aware handling for the first cut.
            return
        b = self._resolve_input(bv)
        out_name = n.name
        self._record_tensor(out_name)
        self.ops.append({
            "name": str(n.name), "op": "add_s8",
            "inputs": [a, b], "outputs": [out_name],
            "shape": {"n": int(np.prod(self.tensors[out_name].shape))},
            "quant": {
                "scale_a":   self.scales.get(a, 1e-8),
                "scale_b":   self.scales.get(b, 1e-8),
                "scale_out": self.scales.get(out_name, 1e-8),
            },
        })

    def _emit_cat(self, n):
        tensor_list = n.args[0]
        names = [self._resolve_input(t) for t in tensor_list]
        dim = int(n.args[1]) if len(n.args) > 1 else 0
        out_name = n.name
        self._record_tensor(out_name)
        n_inputs = len(names)
        op_kind = f"cat{n_inputs}_c{dim}_s8" if n_inputs <= 4 else f"cat_n_c{dim}_s8"
        self.ops.append({
            "name": str(n.name), "op": op_kind,
            "inputs": names, "outputs": [out_name],
            "shape": dict(self._shape_kv(out_name)),
            "quant": {
                "scales_in": [self.scales.get(nm, 1e-8) for nm in names],
                "scale_out": self.scales.get(out_name, 1e-8),
                "dim": dim,
            },
        })

    def _emit_pad(self, n):
        """Asymmetric pad_s8. Pads tuple is aten's
        (left, right, top, bottom) for a 4-d tensor."""
        in_name = self._resolve_input(n.args[0])
        pads = list(n.args[1])
        out_name = n.name
        self._record_tensor(out_name)
        out_shape = self.tensors[out_name].shape if out_name in self.tensors else None
        self.ops.append({
            "name": str(n.name), "op": "pad_s8",
            "inputs": [in_name], "outputs": [out_name],
            "shape": {
                "N": int(out_shape[0]) if out_shape is not None else 1,
                "C": int(out_shape[1]) if out_shape is not None else 1,
                "IH": int(self.tensors[in_name].shape[2]),
                "IW": int(self.tensors[in_name].shape[3]),
                "pad_left":   int(pads[0]) if len(pads) > 0 else 0,
                "pad_right":  int(pads[1]) if len(pads) > 1 else 0,
                "pad_top":    int(pads[2]) if len(pads) > 2 else 0,
                "pad_bottom": int(pads[3]) if len(pads) > 3 else 0,
            },
            "quant": {
                "scale_in":  self.scales.get(in_name, 1e-8),
                "scale_out": self.scales.get(out_name, 1e-8),
                "pad_value": 0,  # zero-pad in int8 domain
            },
            "_pending_kernel": True,
        })

    def _emit_maxpool(self, n):
        in_name = self._resolve_input(n.args[0])
        ks = _pair(n.args[1]) if len(n.args) > 1 else (2, 2)
        st = _pair(n.args[2]) if len(n.args) > 2 else ks
        out_name = n.name
        self._record_tensor(out_name)
        shape = self.tensors[out_name].shape
        in_shape = self.tensors[in_name].shape
        self.ops.append({
            "name": str(n.name), "op": "maxpool2d_s8",
            "inputs": [in_name], "outputs": [out_name],
            "shape": {
                "N": int(shape[0]), "C": int(shape[1]),
                "IH": int(in_shape[2]), "IW": int(in_shape[3]),
                "KH": int(ks[0]), "KW": int(ks[1]),
                "SH": int(st[0]), "SW": int(st[1]),
                "PH": 0, "PW": 0, "DH": 1, "DW": 1,
            },
            "quant": {
                "scale_in":  self.scales.get(in_name, 1e-8),
                "scale_out": self.scales.get(out_name, 1e-8),
            },
        })

    # ------------------------------------------------------------------
    # Fusion helpers
    # ------------------------------------------------------------------
    def _find_bn_user(self, conv_node) -> Optional[torch.fx.Node]:
        for u in conv_node.users:
            if _op_name(u) == "batch_norm.default":
                return u
        return None

    def _fold_batchnorm(self, w: torch.Tensor, b: Optional[torch.Tensor],
                        bn_node) -> tuple[torch.Tensor, torch.Tensor, str]:
        """w_fold = w * (gamma / sqrt(var + eps)) per output channel,
        b_fold = (b - running_mean) * (gamma / sqrt(var + eps)) + beta."""
        weight = self._get_param_tensor(bn_node.args[1])  # gamma
        bias_ = self._get_param_tensor(bn_node.args[2])    # beta
        running_mean = self._get_param_tensor(bn_node.args[3])
        running_var = self._get_param_tensor(bn_node.args[4])
        eps = float(bn_node.kwargs.get("eps", 1e-5))
        scale = weight / torch.sqrt(running_var + eps)
        w_fold = w * scale.reshape(-1, 1, 1, 1)
        b_orig = b if b is not None else torch.zeros(w.shape[0], dtype=w.dtype)
        b_fold = (b_orig - running_mean) * scale + bias_
        return w_fold, b_fold, bn_node.name

    def _get_param_tensor(self, node) -> torch.Tensor:
        if node is None: return None
        if isinstance(node, torch.Tensor): return node
        if isinstance(node, torch.fx.Node):
            if node.op == "get_attr":
                return _get_attr_value(self.gm, node.target)
            if node.name in self.tensors:
                return self.tensors[node.name]
        raise RuntimeError(f"can't materialize parameter for node {node}")

    def _shape_kv(self, name: str) -> dict[str, int]:
        t = self.tensors.get(name)
        if t is None:
            return {"n": 1}
        s = list(t.shape)
        if len(s) == 4:
            return {"N": int(s[0]), "C": int(s[1]),
                    "H": int(s[2]), "W": int(s[3])}
        if len(s) == 3:
            return {"N": int(s[0]), "S": int(s[1]), "D": int(s[2])}
        if len(s) == 2:
            return {"M": int(s[0]), "K": int(s[1])}
        return {"n": int(np.prod(s))}


def _pair(v) -> tuple[int, int]:
    if isinstance(v, (list, tuple)):
        if len(v) == 1: return (int(v[0]), int(v[0]))
        return (int(v[0]), int(v[1]))
    return (int(v), int(v))


def _get_attr_value(gm, qualname: str) -> torch.Tensor:
    obj = gm
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


# ----------------------------------------------------------------------
# Top-level entry point.
# ----------------------------------------------------------------------

def _load_model(name: str):
    if name == "vint":
        from agents.models import vint as model_mod
    else:
        raise SystemExit(
            f"--model {name} doesn't need extract_graph_export; "
            f"use extract_graph.py (FX-based) instead."
        )
    sample = model_mod.get_sample_input()
    if isinstance(sample, torch.Tensor):
        sample = (sample,)
    return model_mod.get_model(), sample


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=["vint"])
    p.add_argument("--quant", default="int8", choices=["fp32", "int8"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--inventory-only", action="store_true",
                   help="Only print the aten-op classification report; "
                        "don't emit IR.")
    p.add_argument("--num-calibration", type=int, default=1,
                   help="Number of calibration samples (currently uses the "
                        "model's get_sample_input() repeatedly with fresh "
                        "RNG; will grow into a real dataset hook).")
    args = p.parse_args()

    if args.quant != "int8":
        raise SystemExit("--quant fp32 not implemented in the export path yet")

    model, sample = _load_model(args.model)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract_export] tracing {args.model} via torch.export ...",
          flush=True)
    ep = torch.export.export(model, sample)
    print(f"[extract_export] traced: {len(list(ep.graph_module.graph.nodes))} "
          f"aten nodes", flush=True)

    if not _print_inventory(ep.graph_module, out_dir):
        sys.exit(1)
    if args.inventory_only:
        return

    # Build calibration dict. Inputs are keyed by the aten graph's
    # user-input placeholder names (NOT our chosen labels) so the
    # walker can resolve `obs_img`/`goal_img` references the same way
    # ViNT.forward declared them. Find those names from the export
    # signature.
    sig = ep.graph_signature
    user_input_names = [s.arg.name for s in sig.input_specs
                         if s.kind.name == "USER_INPUT"]
    if len(user_input_names) != len(sample):
        raise SystemExit(
            f"export user_input count {len(user_input_names)} != "
            f"sample length {len(sample)}")
    input_order = user_input_names
    calib_sample = {name: t for name, t in zip(input_order, sample)}
    walker = _ExportWalker(ep, [calib_sample], input_order)
    print(f"[extract_export] calibrating + capturing intermediates ...",
          flush=True)
    walker.calibrate()
    print(f"[extract_export] emitting IR ...", flush=True)
    walker.emit()

    # Inputs / outputs from the export's user signature.
    input_names = list(calib_sample.keys())
    for k in input_names:
        walker._record_tensor(k)
    sig = ep.graph_signature
    output_names = [s.arg.name for s in sig.output_specs
                    if hasattr(s, "arg") and s.kind.name == "USER_OUTPUT"]

    ir = {
        "name": args.model,
        "input": input_names[0] if len(input_names) == 1 else input_names,
        "output": output_names[0] if len(output_names) == 1 else output_names,
        "tensors": walker.tensors_meta,
        "ops": walker.ops,
    }

    graph_path = out_dir / "graph.json"
    with open(graph_path, "w") as f:
        json.dump(ir, f, indent=2)
    print(f"[extract_export] wrote {graph_path} "
          f"({len(walker.ops)} op records, "
          f"{sum(1 for o in walker.ops if o.get('_pending_kernel'))} pending kernels)",
          flush=True)

    weights_path = out_dir / "weights.npz"
    np.savez(weights_path, **walker.weights_blob)
    print(f"[extract_export] wrote {weights_path} "
          f"({len(walker.weights_blob)} weight tensors)", flush=True)

    # io.npz: store one calibration sample input + the model's output
    # for that input (used by the spike harness to verify).
    with torch.no_grad():
        out = model(*sample)
    io_npz: dict[str, np.ndarray] = {}
    for k, v in calib_sample.items():
        io_npz[f"input_{k}"] = v.detach().cpu().numpy()
    if isinstance(out, (list, tuple)):
        for i, t in enumerate(out):
            io_npz[f"output_{i}"] = t.detach().cpu().numpy()
    else:
        io_npz["output_0"] = out.detach().cpu().numpy()
    io_path = out_dir / "io.npz"
    np.savez(io_path, **io_npz)
    print(f"[extract_export] wrote {io_path} "
          f"({len(io_npz)} tensors)", flush=True)


if __name__ == "__main__":
    main()
