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
    _annotate_dispatches,
    _CaptureTensors,
    _INT8_RANGE,  # noqa: F401  — used implicitly via _scale_from_max_abs
    _scale_from_max_abs,
    _quantize_per_tensor_sym,
    _requantize_multiplier_shift,
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
    "<built-in function getitem>",
    "copy_.default",
}

# torch.export wraps custom-autograd-Function activations (e.g. EfficientNet's
# MemoryEfficientSwish) in a `wrap_with_set_grad_enabled` call to a sub-
# GraphModule. The sub-graph for MemoryEfficientSwish is just
# `mul(sigmoid(x), x)` (= silu). We decompose those wraps back into a
# sigmoid + mul pair so the missing activation isn't silently dropped from
# the IR — that was the cause of ~50% compounding magnitude drift through
# the 16 MBConv blocks in the fp16 ViNT path.
_SWISH = {
    "wrap_with_set_grad_enabled",
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
    if name in _SWISH:              return "swish"
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
                 input_order: list[str], per_channel: bool = False,
                 quant: str = "int8",
                 op_precision: Optional[dict[str, str]] = None):
        self.ep = ep
        self.gm = ep.graph_module
        # When True, conv2d / linear / matmul emit their per-channel
        # weight-scale variants (one Q0.31 multiplier + shift per
        # output channel) instead of the per-tensor form. Activations
        # remain per-tensor symmetric.
        self.per_channel = per_channel
        # Default precision for ops without an explicit override:
        #   "int8" — existing PTQ path
        #   "fp16" — half-precision (weights cast to float16, op records
        #            carry "_f16" suffix, no scale/multiplier/shift)
        # Mixed-precision overrides live in self.op_precision[node.name];
        # ops without an entry use self.default_quant. See
        # agents/notes/mixed_precision_plan.md.
        if quant not in ("int8", "fp16"):
            raise ValueError(f"quant must be 'int8' or 'fp16' (got {quant!r})")
        self.default_quant = quant
        # node_name → "int8" | "fp16" override for this op.
        self.op_precision: dict[str, str] = dict(op_precision or {})
        # Legacy aliases — kept for any code that still reads walker-wide
        # state. New code should call _quant_for() and dispatch on that.
        self.quant = quant
        self.op_suffix = "_s8" if quant == "int8" else "_f16"
        self.tensor_dtype = "i8" if quant == "int8" else "f16"
        self.weight_np_dtype = np.int8 if quant == "int8" else np.float16
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
    # Phase 1: run the model, capture every tensor's value distribution,
    # then derive per-tensor activation scales. Two modes:
    #   * max_abs  — scale = max(|t|) / 127  (default; tight but
    #                outlier-sensitive)
    #   * percentile — scale = quantile(|t|, p) / 127  (clips outliers,
    #                better typical-range precision)
    # The percentile mode is controlled by AGENTS_ACT_PERCENTILE env
    # var (e.g. 99.99 for 99.99th percentile). Defaults to None
    # (max_abs).
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Mixed-precision helpers. Each aten node's emit decision flows
    # through _quant_for(); call sites use this instead of self.quant
    # so a per-op override (from a model's get_precision_spec or a
    # --fp16-ops CLI flag) flips the emit branch for that op alone.
    # ------------------------------------------------------------------
    def _quant_for(self, n) -> str:
        """Return "int8" or "fp16" for this aten node — honoring the
        per-op precision override map, falling back to default_quant."""
        name = n.name if hasattr(n, "name") else str(n)
        return self.op_precision.get(name, self.default_quant)

    def _dtype_for_quant(self, q: str) -> str:
        return "f16" if q == "fp16" else "i8"

    def _op_suffix_for(self, q: str) -> str:
        return "_f16" if q == "fp16" else "_s8"

    def calibrate(self):
        import os as _os
        pct_env = _os.environ.get("AGENTS_ACT_PERCENTILE", "")
        pct = float(pct_env) if pct_env else None
        max_abs: dict[str, float] = {}
        # When percentile mode is active, accumulate sorted samples of
        # the absolute values per tensor (capped at 1000 elements per
        # sample to bound memory) so we can compute the percentile
        # across the whole calibration set.
        pct_samples: dict[str, list] = {}
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
                abs_t = t.detach().abs()
                m = float(abs_t.max().item())
                if nname not in max_abs or m > max_abs[nname]:
                    max_abs[nname] = m
                if pct is not None:
                    flat = abs_t.flatten()
                    # Subsample large tensors to keep memory bounded.
                    if flat.numel() > 2048:
                        idx = torch.randperm(flat.numel())[:2048]
                        flat = flat[idx]
                    pct_samples.setdefault(nname, []).extend(
                        flat.cpu().tolist())
            if i == 0:
                captured_first = dict(cap.tensors)
        self.tensors = captured_first
        # fp16 doesn't need activation scales — skip the derivation.
        # The first-sample shapes captured above are all the fp16 path
        # needs from calibrate().
        if self.quant != "int8":
            return
        for nname, m in max_abs.items():
            if pct is not None and nname in pct_samples and pct_samples[nname]:
                # Percentile-based scale: clip outliers above the
                # configured quantile so the typical-range precision
                # isn't wasted on rare spikes.
                arr = np.asarray(pct_samples[nname], dtype=np.float32)
                clip = float(np.percentile(arr, pct))
                self.scales[nname] = max(clip, 1e-8) / 127.0
            else:
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

    # ------------------------------------------------------------------
    # Auto-cast pass: for each op, if its declared precision (from the
    # op record's "precision" field or — for ops without an explicit
    # precision tag — inferred from the suffix) demands a different
    # tensor dtype than what an input tensor was produced at, splice in
    # a cast_<i8_to_f16|f16_to_i8> op before the consumer. Mixed
    # precision becomes a property of the assembled IR; per-op emit code
    # stays single-precision-per-op.
    # See agents/notes/mixed_precision_plan.md.
    # ------------------------------------------------------------------
    def insert_casts(self):
        def consumer_dtype(op) -> str:
            p = op.get("precision")
            if p is None:
                p = "fp16" if op["op"].endswith("_f16") else "int8"
            return "f16" if p == "fp16" else "i8"

        new_ops: list[dict] = []
        # (src_tensor_name, dst_dtype) → reusable cast-output name. If
        # two consumers want the same cast, share the buffer.
        cast_intermediates: dict[tuple[str, str], str] = {}

        for op in self.ops:
            dst_dtype = consumer_dtype(op)
            for i, in_name in enumerate(op.get("inputs", [])):
                # The IR's input list can include weight refs for some
                # ops; for cast purposes we only care about actual
                # tensor inputs (those that have a tensors_meta entry).
                meta = self.tensors_meta.get(in_name)
                if meta is None:
                    continue
                src_dtype = meta.get("dtype", "i8")
                if src_dtype == dst_dtype:
                    continue
                # Only handle i8 <-> f16 boundaries; other dtypes (i32
                # bias, etc.) aren't real activations and should pass
                # through untouched.
                if (src_dtype, dst_dtype) not in (("i8", "f16"), ("f16", "i8")):
                    continue
                key = (in_name, dst_dtype)
                cast_out = cast_intermediates.get(key)
                if cast_out is None:
                    cast_out = f"{in_name}__cast_{dst_dtype}"
                    self.tensors_meta[cast_out] = {
                        "shape": list(meta["shape"]),
                        "dtype": dst_dtype,
                    }
                    # The scale stored on the cast op record is the
                    # int8 tensor's per-tensor symmetric scale. For
                    # i8 → f16, that's the source's scale (dequant).
                    # For f16 → i8, that's the destination's scale,
                    # which we approximate as the source's scale (same
                    # tensor's calibrated max-abs / 127) since the
                    # cast just reinterprets the same activation.
                    scale = self.scales.get(in_name, 1e-8)
                    if src_dtype == "i8" and dst_dtype == "f16":
                        cast_op_kind = "cast_i8_to_f16"
                    else:
                        cast_op_kind = "cast_f16_to_i8"
                    n_elems = 1
                    for d in meta["shape"]:
                        n_elems *= int(d)
                    new_ops.append({
                        "name": cast_out,
                        "op": cast_op_kind,
                        "precision": "int8" if dst_dtype == "i8" else "fp16",
                        "inputs": [in_name],
                        "outputs": [cast_out],
                        "shape": {"n": n_elems},
                        "quant": {"scale": float(scale), "zero_point": 0},
                    })
                    cast_intermediates[key] = cast_out
                op["inputs"][i] = cast_out
            new_ops.append(op)
        self.ops = new_ops

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
        if self.quant == "int8":
            sc = _scale_from_max_abs(t)
            self.scales[name] = sc
            self.tensors_meta[name] = {
                "shape": list(t.shape),
                "dtype": "i8",
                "quant": {"scale": sc, "zero_point": 0,
                          "kind": "constant_buffer"},
            }
            self.weights_blob[name] = _quantize_per_tensor_sym(t, sc)
        else:  # fp16
            self.tensors_meta[name] = {
                "shape": list(t.shape),
                "dtype": "f16",
                "quant": {"kind": "constant_buffer"},
            }
            self.weights_blob[name] = t.detach().cpu().numpy().astype(np.float16)

    def _record_tensor(self, name: str, dtype: Optional[str] = None):
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
        eff_dtype = dtype if dtype is not None else self.tensor_dtype
        if self.quant == "int8":
            self.tensors_meta[name] = {
                "shape": list(t.shape),
                "dtype": eff_dtype,
                "quant": {
                    "scale": self.scales.get(name, 1e-8),
                    "zero_point": 0,
                },
            }
        else:  # fp16 — no per-tensor quant metadata
            self.tensors_meta[name] = {
                "shape": list(t.shape),
                "dtype": eff_dtype,
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
            # Special case: slice.Tensor along an NCHW channel dim
            # (dim=1) of a 4-D tensor that has a downstream cat
            # consumer. We CAN'T alias-collapse it because doing so
            # would lose the (C_start, C_end) range info downstream,
            # and the cat would end up reading the full source buffer.
            # Emit a slice_c_s8 op instead.
            if op_kind == "slice.Tensor" and self._is_channel_slice_used_by_cat(n):
                self._emit_slice_c(n)
                return
            # Pass-through: alias this node's name to its first tensor
            # input so downstream consumers resolve to the underlying
            # tensor.
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
        if cls == "swish":
            self._emit_swish(n)
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
        """Emit IR records for the ViNT-new ops. Most lower to a single
        s8 kernel call; SDPA gets decomposed into matmul+softmax+matmul
        so the scheduler can place them independently. Layer-norm pulls
        gamma + beta from its aten args (positions 2 and 3) and records
        them as constant_buffer tensors so the skeleton emitter can
        reference them by name."""
        op_kind = _op_name(n)
        if op_kind == "scaled_dot_product_attention.default":
            self._emit_sdpa_decomposed(n); return
        kern_base = _NEW_COMPUTE[op_kind].removesuffix("_s8")
        kern_name = f"{kern_base}{self._op_suffix_for(self._quant_for(n))}"
        in_name = self._resolve_input(n.args[0])
        out_name = n.name
        # Record the output tensor at the per-op precision's dtype so
        # the auto-cast pass sees the right boundary type.
        op_q = self._quant_for(n)
        self._record_tensor(out_name, dtype=self._dtype_for_quant(op_q))
        is_fp16 = (op_q == "fp16")
        # Default shape: flat "n" for elementwise ops. Per-op overrides
        # below set 4-D / 2-D shape dicts when needed.
        rec = {
            "name": str(n.name),
            "op": kern_name,
            "precision": op_q,
            "inputs": [in_name] if not isinstance(n.args[0], (list, tuple))
                       else self._resolve_input(n.args[0]),
            "outputs": [out_name],
            "shape": {"n": int(np.prod(self.tensors[out_name].shape))},
        }
        if not is_fp16:
            rec["quant"] = {
                "scale_in":  self.scales.get(in_name, 1e-8),
                "scale_out": self.scales.get(out_name, 1e-8),
                "activation_min": -128, "activation_max": 127,
            }
        if op_kind == "mul.Tensor":
            b = self._resolve_input(n.args[1])
            rec["inputs"] = [in_name, b]
            # Detect channel-axis broadcast: one input is [B, C, 1, 1] (or
            # equivalently a length-C tensor) while the other is [B, C, H, W].
            # EfficientNet's SE block multiplies a per-channel gate by an
            # NCHW feature map — without the broadcast variant, mul_f16
            # would read past the end of the gate buffer and produce
            # garbage. Switch to mul_c1_{f16,s8} when this shape pattern
            # is detected; otherwise keep the plain elementwise mul.
            in_a_shape = self.tensors[in_name].shape if in_name in self.tensors else None
            in_b_shape = self.tensors[b].shape if b in self.tensors else None
            def _is_channel_gate(sh, nchw_shape):
                if sh is None or nchw_shape is None or len(nchw_shape) != 4:
                    return False
                # Accept gate as either (B, C, 1, 1), (1, C, 1, 1), or just (C,).
                if len(sh) == 4 and sh[0] in (1, nchw_shape[0]) and \
                        sh[1] == nchw_shape[1] and sh[2] == 1 and sh[3] == 1:
                    return True
                if len(sh) == 1 and sh[0] == nchw_shape[1]:
                    return True
                return False
            gate_name = None
            nchw_name = None
            nchw_shape = None
            if in_a_shape is not None and in_b_shape is not None:
                if _is_channel_gate(in_a_shape, in_b_shape):
                    gate_name, nchw_name, nchw_shape = in_name, b, in_b_shape
                elif _is_channel_gate(in_b_shape, in_a_shape):
                    gate_name, nchw_name, nchw_shape = b, in_name, in_a_shape
            if gate_name is not None:
                # Rewrite this record to mul_c1_*.
                N = int(nchw_shape[0])
                C = int(nchw_shape[1])
                HW = int(nchw_shape[2]) * int(nchw_shape[3])
                rec["op"] = "mul_c1_f16" if is_fp16 else "mul_c1_s8"
                rec["inputs"] = [gate_name, nchw_name]
                rec["shape"] = {"N": N, "C": C, "HW": HW}
                if not is_fp16:
                    rec["quant"] = {
                        "scale_gate": self.scales.get(gate_name, 1e-8),
                        "scale_x":    self.scales.get(nchw_name, 1e-8),
                        "scale_out":  self.scales.get(out_name, 1e-8),
                        "activation_min": -128, "activation_max": 127,
                    }
            elif not is_fp16:
                rec["quant"] = {
                    "scale_a":   self.scales.get(in_name, 1e-8),
                    "scale_b":   self.scales.get(b, 1e-8),
                    "scale_out": self.scales.get(out_name, 1e-8),
                    "activation_min": -128, "activation_max": 127,
                }
        elif op_kind == "layer_norm.default":
            # aten signature: layer_norm(input, normalized_shape, weight,
            # bias, eps). Pull gamma + beta refs via _resolve_input so
            # they get recorded in weights_blob (constant_buffer path).
            gamma = self._resolve_input(n.args[2]) if len(n.args) > 2 and n.args[2] is not None else None
            beta = self._resolve_input(n.args[3]) if len(n.args) > 3 and n.args[3] is not None else None
            eps = float(n.args[4]) if len(n.args) > 4 else 1e-5
            in_shape = self.tensors[in_name].shape
            # Flatten leading dims into M; last dim is K (LayerNorm normalizes
            # over the last axis when normalized_shape == [K]).
            K = int(in_shape[-1])
            M = 1
            for d in in_shape[:-1]: M *= int(d)
            rec["shape"] = {"M": M, "K": K}
            if is_fp16:
                rec["gamma_key"] = gamma
                rec["beta_key"] = beta
                rec["eps"] = eps
            else:
                rec["quant"] = {
                    "scale_in":    self.scales.get(in_name, 1e-8),
                    "scale_gamma": self.scales.get(gamma, 1e-8) if gamma else 0.0,
                    "scale_beta":  self.scales.get(beta, 1e-8) if beta else 0.0,
                    "scale_out":   self.scales.get(out_name, 1e-8),
                    "eps": eps,
                    "gamma_key": gamma,
                    "beta_key":  beta,
                    "activation_min": -128, "activation_max": 127,
                }
        elif op_kind == "adaptive_avg_pool2d.default":
            in_shape = self.tensors[in_name].shape
            out_shape = self.tensors[out_name].shape
            rec["shape"] = {
                "N": int(out_shape[0]), "C": int(out_shape[1]),
                "IH": int(in_shape[2]), "IW": int(in_shape[3]),
                "OH": int(out_shape[2]), "OW": int(out_shape[3]),
            }
        elif op_kind in ("sigmoid.default", "gelu.default"):
            rec["shape"] = {"n": int(np.prod(self.tensors[out_name].shape))}
        self.ops.append(rec)

    def _emit_sdpa_decomposed(self, n):
        """SDPA(Q, K, V) → matmul(Q, Kᵀ)/√d → softmax → matmul(_, V)

        Emits three records (matmul, softmax, matmul) sharing the SDPA
        node's name as a prefix and synthesizing the intermediate
        tensor names (scores, weights). Each gets pinned to its own
        output buffer so the runtime can place them on different harts.
        """
        import math
        q = self._resolve_input(n.args[0])
        k = self._resolve_input(n.args[1])
        v = self._resolve_input(n.args[2])
        # Pull Q/K/V shapes directly from the export's meta['val'] so we
        # don't get confused by upstream view/transpose aliases that
        # would route us to a pre-split QKV tensor with the wrong rank.
        def _shape(node):
            if hasattr(node, 'meta') and 'val' in node.meta:
                return tuple(int(s) for s in node.meta['val'].shape)
            return None
        q_shape = _shape(n.args[0])
        k_shape = _shape(n.args[1])
        v_shape = _shape(n.args[2])
        # Q layout: (..., L_q, head_dim). Last two dims are what matters.
        if q_shape and len(q_shape) >= 2:
            L_q = q_shape[-2]
            head_dim = q_shape[-1]
        else:
            L_q, head_dim = 0, 1
        if k_shape and len(k_shape) >= 2:
            L_k = k_shape[-2]
        else:
            L_k = 0
        if v_shape and len(v_shape) >= 2:
            head_dim_v = v_shape[-1]
        else:
            head_dim_v = head_dim
        scale_div = math.sqrt(max(head_dim, 1))
        scores_name = f"{n.name}__scores"
        weights_name = f"{n.name}__weights"
        out_name = n.name
        is_fp16 = (self._quant_for(n) == "fp16")
        if not is_fp16:
            # Synthesize intermediate-tensor scales (no real capture for
            # these): scores use Q's range × √d, weights are [0, 1] mapped
            # to scale 1/127.
            self.scales[scores_name] = self.scales.get(q, 1e-8) * scale_div
            self.scales[weights_name] = 1.0 / 127.0
            self.tensors_meta[scores_name] = {
                "shape": [L_q, L_k], "dtype": "i8",
                "quant": {"scale": self.scales[scores_name], "zero_point": 0},
            }
            self.tensors_meta[weights_name] = {
                "shape": [L_q, L_k], "dtype": "i8",
                "quant": {"scale": self.scales[weights_name], "zero_point": 0},
            }
        else:
            self.tensors_meta[scores_name] = {
                "shape": [L_q, L_k], "dtype": "f16",
            }
            self.tensors_meta[weights_name] = {
                "shape": [L_q, L_k], "dtype": "f16",
            }
        self._record_tensor(out_name)
        if is_fp16:
            # matmul_tb_f16 = matmul with B transposed = Q · Kᵀ.
            # The 1/√d_k pre-softmax scale is folded into softmax_f16's
            # input_scale parameter so we don't need an extra pointwise op.
            self.ops.append({
                "name": f"{n.name}_qk", "op": "matmul_tb_f16",
                "inputs": [q, k], "outputs": [scores_name],
                "shape": {"M": L_q, "K": head_dim, "N": L_k},
            })
            self.ops.append({
                "name": f"{n.name}_softmax", "op": "softmax_f16",
                "inputs": [scores_name], "outputs": [weights_name],
                "shape": {"M": L_q, "K": L_k},
                "input_scale": 1.0 / scale_div,
            })
            self.ops.append({
                "name": f"{n.name}_av", "op": "matmul_f16",
                "inputs": [weights_name, v], "outputs": [out_name],
                "shape": {"M": L_q, "K": L_k, "N": head_dim_v},
            })
            return
        self.ops.append({
            "name": f"{n.name}_qk", "op": "matmul_s8",
            "inputs": [q, k], "outputs": [scores_name],
            "shape": {"M": L_q, "K": head_dim, "N": L_k},
            "quant": {
                "scale_a":   self.scales.get(q, 1e-8),
                "scale_b":   self.scales.get(k, 1e-8),
                "scale_out": self.scales[scores_name],
                "transpose_b": 1,
                "scale_div_sqrt_dk": scale_div,
                "activation_min": -128, "activation_max": 127,
            },
        })
        self.ops.append({
            "name": f"{n.name}_softmax", "op": "softmax_s8",
            "inputs": [scores_name], "outputs": [weights_name],
            "shape": {"M": L_q, "K": L_k},
            "quant": {
                "scale_in":  self.scales[scores_name],
                "scale_out": self.scales[weights_name],
            },
        })
        self.ops.append({
            "name": f"{n.name}_av", "op": "matmul_s8",
            "inputs": [weights_name, v], "outputs": [out_name],
            "shape": {"M": L_q, "K": L_k, "N": head_dim_v},
            "quant": {
                "scale_a":   self.scales[weights_name],
                "scale_b":   self.scales.get(v, 1e-8),
                "scale_out": self.scales.get(out_name, 1e-8),
                "transpose_b": 0,
                "scale_div_sqrt_dk": 1.0,
                "activation_min": -128, "activation_max": 127,
            },
        })

    # ------------------------------------------------------------------
    # Memory-efficient swish (SiLU) emit. EfficientNet wraps its swish
    # activations in a `wrap_with_set_grad_enabled(False, submod_N, x)`
    # call that invokes a 2-node sub-graph (`mul(sigmoid(x), x)`). The
    # walker doesn't recurse into sub-graphs, so we need to detect the
    # wrap nodes and emit the equivalent sigmoid + mul pair directly
    # against the parent graph.
    # ------------------------------------------------------------------
    def _emit_swish(self, n):
        # Args: (no_grad_flag: bool, submod_node: get_attr, x: tensor).
        # The 3rd arg is always the data tensor.
        input_node = n.args[2]
        input_name = self._resolve_input(input_node)
        out_name = n.name

        # Compute fp32 reference so _record_tensor can find shape + value.
        # cap.tensors stores the wrap node's output as a tuple (because the
        # wrap returns whatever submod returns wrapped in single-element
        # tuple) — we'd rather hold a tensor, so recompute silu directly.
        input_t = self.tensors.get(input_name)
        if input_t is None:
            return  # input not captured; nothing to do
        with torch.no_grad():
            silu_t = (input_t * torch.sigmoid(input_t)).detach()
            self.tensors[out_name] = silu_t

        # Always seed the silu output's scale from the synthesised
        # tensor, regardless of this swish's own precision — the
        # auto-cast pass uses this scale when a downstream int8
        # consumer reads from a fp16 silu (or vice-versa). The
        # calibrate() loop skips wrap_with_set_grad_enabled because
        # cap.tensors stores its output as a TUPLE and the scan only
        # accepts Tensors.
        if out_name not in self.scales:
            self.scales[out_name] = _scale_from_max_abs(silu_t)

        op_q = self._quant_for(n)
        self._record_tensor(out_name, dtype=self._dtype_for_quant(op_q))
        sig_name = f"{n.name}__sigmoid"
        # Intermediate sigmoid output: same shape as input, dtype
        # follows the swish op's precision.
        if sig_name not in self.tensors_meta:
            self.tensors[sig_name] = torch.sigmoid(input_t).detach()
            if op_q == "int8":
                # sigmoid output ∈ (0, 1) — scale is 1/127.
                self.scales[sig_name] = 1.0 / 127.0
                self.tensors_meta[sig_name] = {
                    "shape": list(input_t.shape),
                    "dtype": "i8",
                    "quant": {"scale": 1.0 / 127.0, "zero_point": 0},
                }
            else:
                self.tensors_meta[sig_name] = {
                    "shape": list(input_t.shape),
                    "dtype": "f16",
                }
        n_elems = int(np.prod(input_t.shape))
        if op_q == "fp16":
            self.ops.append({
                "name": sig_name,
                "op": "sigmoid_f16",
                "inputs": [input_name],
                "outputs": [sig_name],
                "shape": {"n": n_elems},
            })
            self.ops.append({
                "name": str(n.name),
                "op": "mul_f16",
                "inputs": [sig_name, input_name],
                "outputs": [out_name],
                "shape": {"n": n_elems},
            })
        else:
            in_scale = self.scales.get(input_name, 1e-8)
            out_scale = self.scales.get(out_name, 1e-8)
            self.ops.append({
                "name": sig_name,
                "op": "sigmoid_s8",
                "inputs": [input_name],
                "outputs": [sig_name],
                "shape": {"n": n_elems},
                "quant": {
                    "scale_in": in_scale,
                    "scale_out": 1.0 / 127.0,
                    "activation_min": -128, "activation_max": 127,
                },
            })
            self.ops.append({
                "name": str(n.name),
                "op": "mul_s8",
                "inputs": [sig_name, input_name],
                "outputs": [out_name],
                "shape": {"n": n_elems},
                "quant": {
                    "scale_a": 1.0 / 127.0,
                    "scale_b": in_scale,
                    "scale_out": out_scale,
                    "activation_min": -128, "activation_max": 127,
                },
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

        # Save originals before any BN-fold so the fp16 path can emit a
        # SEPARATE batchnorm2d_f16 op instead of folding gamma/var into
        # the conv weights. Folding-into-conv then casting to fp16 was
        # the previous fp16 strategy; it diverged from torch.float16
        # because PyTorch's `model.half()` rounds *between* conv and BN
        # (two fp16 stores), and that rounding pattern compounds across
        # the 16 MBConv blocks. Emitting BN separately matches PyTorch.
        w_orig = w
        b_orig = b

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

        # fp16 path: emit conv2d_f16 with ORIGINAL (un-folded) weights so
        # the conv output gets stored as fp16 before BN is applied — that
        # intermediate fp16 rounding is what torch.float16 does, and not
        # doing it (the previous "fold BN into conv weights, cast to fp16"
        # strategy) compounded ~2-3x magnitude drift per layer through the
        # 16 MBConv blocks. If a BN follows, emit a SEPARATE
        # batchnorm2d_f16 right after with pre-computed affine
        # (scale=gamma/sqrt(var+eps), bias=beta-mean*scale).
        if self._quant_for(n) == "fp16":
            # Conv writes to the conv's own aten node name (n.name); if a
            # BN follows, batchnorm2d_f16 then writes to bn_user.name.
            # If no BN, conv writes directly to its own n.name (which is
            # what downstream consumers will reference).
            conv_out_name = n.name
            w_key = f"{n.name}.weights"
            self.weights_blob[w_key] = w_orig.detach().cpu().numpy().astype(np.float16)
            b_key = None
            if b_orig is not None:
                b_key = f"{n.name}.bias"
                self.weights_blob[b_key] = b_orig.detach().cpu().numpy().astype(np.float16)
            self._record_tensor(conv_out_name, dtype="f16")
            op_kind = "depthwise_conv2d_f16" if is_depthwise else "conv2d_f16"
            self.ops.append({
                "name": str(n.name),
                "op": op_kind,
                "precision": "fp16",
                "inputs": [in_name],
                "outputs": [conv_out_name],
                "weight": w_key,
                "bias": b_key,
                "shape": {
                    "N": int(self.tensors[conv_out_name].shape[0]),
                    "IC": int(w_orig.shape[1] if not is_depthwise else 1),
                    "IH": int(self.tensors[in_name].shape[2]),
                    "IW": int(self.tensors[in_name].shape[3]),
                    "OC": int(OC), "KH": int(KH), "KW": int(KW),
                    "SH": int(stride[0]), "SW": int(stride[1]),
                    "PH": int(padding[0]), "PW": int(padding[1]),
                    "DH": int(dilation[0]), "DW": int(dilation[1]),
                    "groups": int(groups),
                },
            })

            # If BN follows, emit it as its own batchnorm2d_f16 op. The
            # affine is precomputed in fp32 (gamma, beta, running_mean,
            # running_var live in the bn aten node's args 1..4) then
            # stored as fp16 — kernel just applies `scale[c] * x + bias[c]`.
            if bn_user is not None:
                gamma = self._get_param_tensor(bn_user.args[1])
                beta = self._get_param_tensor(bn_user.args[2])
                running_mean = self._get_param_tensor(bn_user.args[3])
                running_var = self._get_param_tensor(bn_user.args[4])
                eps = float(bn_user.kwargs.get("eps", 1e-5))
                scale_fp32 = gamma / torch.sqrt(running_var + eps)
                bn_bias_fp32 = beta - running_mean * scale_fp32
                scale_key = f"{bn_user.name}.scale"
                bn_bias_key = f"{bn_user.name}.bias"
                self.weights_blob[scale_key] = (
                    scale_fp32.detach().cpu().numpy().astype(np.float16))
                self.weights_blob[bn_bias_key] = (
                    bn_bias_fp32.detach().cpu().numpy().astype(np.float16))
                self._record_tensor(bn_user.name, dtype="f16")
                t_shape = self.tensors[bn_user.name].shape
                self.ops.append({
                    "name": str(bn_user.name),
                    "op": "batchnorm2d_f16",
                    "precision": "fp16",
                    "inputs": [conv_out_name],
                    "outputs": [bn_user.name],
                    "weight": scale_key,
                    "bias": bn_bias_key,
                    "shape": {
                        "N": int(t_shape[0]), "C": int(t_shape[1]),
                        "H": int(t_shape[2]), "W": int(t_shape[3]),
                    },
                })
            return

        in_scale = self.scales.get(in_name, 1e-8)
        out_scale = self.scales.get(post_name, 1e-8)

        # Per-channel weight scale: one scale per output channel.
        # Depthwise stays per-tensor for now — its kernel doesn't yet
        # accept per-channel arrays and it's a smaller accuracy win
        # (per-channel matters most where output dim is large; depthwise
        # outputs are typically small after the channel mix).
        use_pc = self.per_channel and not is_depthwise
        if use_pc:
            # Per-OC max_abs over the (IC, KH, KW) dims that contribute
            # to each OC.
            w_np = w.detach().cpu().numpy()
            per_oc_max = np.abs(w_np).reshape(OC, -1).max(axis=1)
            w_scales_per_oc = np.maximum(per_oc_max, 1e-8) / 127.0
            w_q = np.empty_like(w_np, dtype=np.int8)
            for oc in range(OC):
                w_q[oc] = np.round(w_np[oc] / w_scales_per_oc[oc]).clip(-127, 127).astype(np.int8)
            op_kind = "conv2d_s8_pc"
            w_key = f"{n.name}.weights"
            self.weights_blob[w_key] = w_q
            b_key = None
            if b is not None:
                b_np = b.detach().cpu().numpy()
                b_q = np.zeros(OC, dtype=np.int32)
                for oc in range(OC):
                    b_q[oc] = int(round(b_np[oc] / (in_scale * w_scales_per_oc[oc])))
                b_key = f"{n.name}.bias"
                self.weights_blob[b_key] = b_q
            mult_arr = np.zeros(OC, dtype=np.int32)
            shift_arr = np.zeros(OC, dtype=np.int32)
            for oc in range(OC):
                real_mult = (in_scale * float(w_scales_per_oc[oc])) / max(out_scale, 1e-30)
                m, s = _requantize_multiplier_shift(real_mult)
                mult_arr[oc] = m
                shift_arr[oc] = s
            mult_key = f"{n.name}.output_multiplier_per_oc"
            shift_key = f"{n.name}.output_shift_per_oc"
            self.weights_blob[mult_key] = mult_arr
            self.weights_blob[shift_key] = shift_arr
        else:
            op_kind = "depthwise_conv2d_s8" if is_depthwise else "conv2d_s8"
            w_scale = _scale_from_max_abs(w)
            w_q = _quantize_per_tensor_sym(w, w_scale)
            w_key = f"{n.name}.weights"
            self.weights_blob[w_key] = w_q
            b_key = None
            if b is not None:
                b_q = (b.detach().cpu().numpy() / (in_scale * w_scale)).round().astype(np.int32)
                b_key = f"{n.name}.bias"
                self.weights_blob[b_key] = b_q
            real_mult = (in_scale * w_scale) / max(out_scale, 1e-30)
            multiplier, shift = _requantize_multiplier_shift(real_mult)
            mult_key = shift_key = None

        self._record_tensor(post_name)
        rec = {
            "name": str(n.name),
            "op": op_kind,
            "inputs": [in_name],
            "outputs": [post_name],
            "weight": w_key,
            "bias": b_key,
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
                "input_offset":     0,
                "filter_offset":    0,
                "output_offset":    0,
                "activation_min":   -128,
                "activation_max":    127,
            },
        }
        if use_pc:
            rec["quant"]["output_multiplier_per_oc_key"] = mult_key
            rec["quant"]["output_shift_per_oc_key"] = shift_key
        else:
            rec["quant"]["output_multiplier"] = multiplier
            rec["quant"]["output_shift"] = shift
        self.ops.append(rec)

    def _emit_linear(self, n):
        in_name = self._resolve_input(n.args[0])
        w = self._get_param_tensor(n.args[1])
        b = self._get_param_tensor(n.args[2]) if len(n.args) > 2 and n.args[2] is not None else None
        OF, IF = w.shape
        out_name = n.name
        self._record_tensor(out_name)
        in_shape = self.tensors[in_name].shape
        M = 1
        for d in in_shape[:-1]: M *= int(d)

        w_key = f"{n.name}.weights"
        b_key = f"{n.name}.bias" if b is not None else None

        # fp16 path: cast weights + bias to fp16, emit a single
        # linear_f16 record (no scale/multiplier/shift).
        # Consults the per-op precision override map so a single linear
        # op (e.g. ViNT's goal-encoder output `linear`) can be promoted
        # to fp16 while the rest of the network stays int8.
        if self._quant_for(n) == "fp16":
            self.weights_blob[w_key] = w.detach().cpu().numpy().astype(np.float16)
            if b is not None:
                self.weights_blob[b_key] = b.detach().cpu().numpy().astype(np.float16)
            # Stamp the output tensor's dtype as f16 so the auto-cast
            # pass and downstream codegen see the right buffer type.
            self.tensors_meta[out_name] = {
                "shape": list(self.tensors[out_name].shape),
                "dtype": "f16",
            }
            self.ops.append({
                "name": str(n.name),
                "op": "linear_f16",
                "precision": "fp16",
                "inputs": [in_name],
                "outputs": [out_name],
                "weight": w_key,
                "bias": b_key,
                "shape": {"M": M, "K": int(IF), "N": int(OF)},
            })
            return

        in_scale = self.scales.get(in_name, 1e-8)
        out_scale = self.scales.get(out_name, 1e-8)

        if self.per_channel:
            w_np = w.detach().cpu().numpy()
            per_oc_max = np.abs(w_np).reshape(OF, -1).max(axis=1)
            w_scales_per_oc = np.maximum(per_oc_max, 1e-8) / 127.0
            w_q = np.empty_like(w_np, dtype=np.int8)
            for oc in range(OF):
                w_q[oc] = np.round(w_np[oc] / w_scales_per_oc[oc]).clip(-127, 127).astype(np.int8)
            self.weights_blob[w_key] = w_q
            if b is not None:
                b_np = b.detach().cpu().numpy()
                b_q = np.zeros(OF, dtype=np.int32)
                for oc in range(OF):
                    b_q[oc] = int(round(b_np[oc] / (in_scale * w_scales_per_oc[oc])))
                self.weights_blob[b_key] = b_q
            mult_arr = np.zeros(OF, dtype=np.int32)
            shift_arr = np.zeros(OF, dtype=np.int32)
            for oc in range(OF):
                real_mult = (in_scale * float(w_scales_per_oc[oc])) / max(out_scale, 1e-30)
                m, s = _requantize_multiplier_shift(real_mult)
                mult_arr[oc] = m
                shift_arr[oc] = s
            mult_key = f"{n.name}.output_multiplier_per_oc"
            shift_key = f"{n.name}.output_shift_per_oc"
            self.weights_blob[mult_key] = mult_arr
            self.weights_blob[shift_key] = shift_arr
            quant = {
                "input_offset": 0, "filter_offset": 0, "output_offset": 0,
                "output_multiplier_per_oc_key": mult_key,
                "output_shift_per_oc_key": shift_key,
                "activation_min": -128, "activation_max": 127,
            }
            op_kind = "linear_s8_pc"
        else:
            w_scale = _scale_from_max_abs(w)
            self.weights_blob[w_key] = _quantize_per_tensor_sym(w, w_scale)
            if b is not None:
                self.weights_blob[b_key] = (
                    b.detach().cpu().numpy() / (in_scale * w_scale)
                ).round().astype(np.int32)
            real_mult = (in_scale * w_scale) / max(out_scale, 1e-30)
            multiplier, shift = _requantize_multiplier_shift(real_mult)
            quant = {
                "input_offset": 0, "filter_offset": 0, "output_offset": 0,
                "output_multiplier": multiplier,
                "output_shift": shift,
                "activation_min": -128, "activation_max": 127,
            }
            op_kind = "linear_s8"
        self.ops.append({
            "name": str(n.name),
            "op": op_kind,
            "precision": "int8",
            "inputs": [in_name],
            "outputs": [out_name],
            "weight": w_key,
            "bias": b_key,
            "shape": {"M": M, "K": int(IF), "N": int(OF)},
            "quant": quant,
        })

    def _emit_relu(self, n):
        in_name = self._resolve_input(n.args[0])
        out_name = n.name
        self._record_tensor(out_name)
        rec = {
            "name": str(n.name), "op": f"relu{self.op_suffix}",
            "inputs": [in_name], "outputs": [out_name],
            "shape": {"n": int(np.prod(self.tensors[out_name].shape))},
        }
        if self.quant == "int8":
            rec["quant"] = {
                "scale_in":  self.scales.get(in_name, 1e-8),
                "scale_out": self.scales.get(out_name, 1e-8),
                "activation_min": 0,  # int8: 0 maps to symmetric -128 + 128
                "activation_max": 127,
            }
        self.ops.append(rec)

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
        rec = {
            "name": str(n.name), "op": f"add{self.op_suffix}",
            "inputs": [a, b], "outputs": [out_name],
            "shape": {"n": int(np.prod(self.tensors[out_name].shape))},
        }
        if self.quant == "int8":
            rec["quant"] = {
                "scale_a":   self.scales.get(a, 1e-8),
                "scale_b":   self.scales.get(b, 1e-8),
                "scale_out": self.scales.get(out_name, 1e-8),
                "activation_min": -128, "activation_max": 127,
            }
        self.ops.append(rec)

    def _emit_cat(self, n):
        tensor_list = n.args[0]
        # Pull per-input shapes from the aten meta['val'] (which has the
        # actual shapes at the cat call site) — self._resolve_input may
        # have alias-collapsed slice/select outputs back to their source,
        # which would lose the per-input shape along the cat axis.
        in_meta_shapes = []
        for t in tensor_list:
            ms = None
            if hasattr(t, 'meta') and 'val' in t.meta:
                ms = tuple(int(s) for s in t.meta['val'].shape)
            in_meta_shapes.append(ms)
        names = [self._resolve_input(t) for t in tensor_list]
        dim = int(n.args[1]) if len(n.args) > 1 else 0
        out_name = n.name
        self._record_tensor(out_name)
        n_inputs = len(names)
        # Layout-reinterpret special case: all inputs resolved to the
        # same upstream tensor name (e.g. ViNT's split-then-concat that
        # shuffles obs_img's leading dims). When the underlying memory
        # is byte-identical, emit an alias and skip the kernel call.
        if all(nm == names[0] for nm in names) and names[0] in self.tensors:
            src_size = int(np.prod(self.tensors[names[0]].shape))
            out_size = int(np.prod(self.tensors[out_name].shape))
            if src_size == out_size:
                self.name_map[out_name] = self.name_map.get(names[0], names[0])
                return
        # Out shape (from captured forward pass; rank-correct).
        out_shape = list(self.tensors[out_name].shape)
        # Reinterpret any non-4-D cat as 4-D NCHW (N=outer, C=cat-axis,
        # H=1, W=inner) so the existing cat{2,3,4}_c1_s8 skeleton works
        # without per-rank variants. The "axis" of the reinterpreted
        # form is always the channel dim (index 1).
        if len(out_shape) == 4:
            N = int(out_shape[0])
            H = int(out_shape[2])
            W = int(out_shape[3])
            C_total = int(out_shape[1])
            C_inputs = []
            for nm, ms in zip(names, in_meta_shapes):
                if ms is not None and len(ms) == 4:
                    C_inputs.append(int(ms[dim]))
                elif nm in self.tensors:
                    C_inputs.append(int(self.tensors[nm].shape[dim]))
                else:
                    C_inputs.append(0)
        else:
            # Fold dims-before-cat into N, dims-after-cat into W.
            N = 1
            for d in out_shape[:dim]: N *= int(d)
            W = 1
            for d in out_shape[dim+1:]: W *= int(d)
            H = 1
            C_total = int(out_shape[dim])
            C_inputs = []
            for nm, ms in zip(names, in_meta_shapes):
                src = ms if ms is not None else (
                    tuple(self.tensors[nm].shape) if nm in self.tensors else None)
                if src is not None and len(src) > dim:
                    C_inputs.append(int(src[dim]))
                else:
                    C_inputs.append(0)
        # Use the 4-D channel-dim cat skeleton for both native-4-D and
        # reinterpreted cases — they share the same kernel signature.
        op_q = self._quant_for(n)
        op_suffix = self._op_suffix_for(op_q)
        # Re-stamp output dtype to match op precision.
        self.tensors_meta[out_name] = {
            "shape": list(self.tensors[out_name].shape) if out_name in self.tensors else self.tensors_meta[out_name]["shape"],
            "dtype": self._dtype_for_quant(op_q),
        }
        if n_inputs <= 4:
            op_kind = f"cat{n_inputs}_c1{op_suffix}"
        else:
            op_kind = f"cat_n_c1{op_suffix}"
        rec = {
            "name": str(n.name), "op": op_kind,
            "precision": op_q,
            "inputs": names, "outputs": [out_name],
            "shape": {
                "N": N, "H": H, "W": W,
                "C_total": C_total, "C_inputs": C_inputs,
            },
        }
        if op_q == "int8":
            rec["quant"] = {
                "scales_in": [self.scales.get(nm, 1e-8) for nm in names],
                "scale_out": self.scales.get(out_name, 1e-8),
                "dim": dim,
                "activation_min": -128,
                "activation_max": 127,
            }
        else:
            rec["dim"] = dim
        self.ops.append(rec)

    def _is_channel_slice_used_by_cat(self, n) -> bool:
        """True iff this slice.Tensor is a NCHW dim=1 slice whose output
        flows into a cat/concat (so the alias collapse would lose the
        slice range info the cat needs)."""
        # aten::slice.Tensor(self, dim, start, end, step=1).
        if len(n.args) < 3:
            return False
        dim = int(n.args[1])
        if dim != 1:
            return False
        # Source must be 4-D (NCHW).
        src = n.args[0]
        if not (hasattr(src, 'meta') and 'val' in src.meta
                and len(src.meta['val'].shape) == 4):
            return False
        # Walk users looking for a cat/concat consumer (possibly through
        # one alias hop).
        seen = set()
        stack = list(n.users)
        while stack:
            u = stack.pop()
            if u.name in seen: continue
            seen.add(u.name)
            t = _op_name(u)
            if t in ("cat.default", "concat.default"):
                return True
            if _classify(t) == "alias":
                stack.extend(u.users)
        return False

    def _emit_slice_c(self, n):
        """Channel-axis slice of NCHW int8: aten::slice(input, dim=1,
        start, end). Emits a slice_c_s8 op record consuming
        (C_end - C_start) channels."""
        src = n.args[0]
        src_name = self._resolve_input(src)
        C_start = int(n.args[2]) if len(n.args) > 2 else 0
        C_end = int(n.args[3]) if len(n.args) > 3 else None
        src_shape = tuple(int(s) for s in src.meta['val'].shape)
        IC, H, W = src_shape[1], src_shape[2], src_shape[3]
        if C_end is None or C_end > IC:
            C_end = IC
        out_name = n.name
        self._record_tensor(out_name)
        rec = {
            "name": str(n.name), "op": f"slice_c{self.op_suffix}",
            "inputs": [src_name], "outputs": [out_name],
            "shape": {
                "N": int(src_shape[0]), "IC": int(IC),
                "C_start": int(C_start), "C_end": int(C_end),
                "H": int(H), "W": int(W),
            },
        }
        if self.quant == "int8":
            in_scale = self.scales.get(src_name, 1e-8)
            out_scale = self.scales.get(out_name, in_scale)
            rec["quant"] = {
                "scale_in":  in_scale,
                "scale_out": out_scale,
                "activation_min": -128, "activation_max": 127,
            }
        self.ops.append(rec)

    def _emit_pad(self, n):
        """Asymmetric pad_s8. Pads tuple is aten's
        (left, right, top, bottom) for a 4-d tensor."""
        in_name = self._resolve_input(n.args[0])
        pads = list(n.args[1])
        out_name = n.name
        self._record_tensor(out_name)
        out_shape = self.tensors[out_name].shape if out_name in self.tensors else None
        rec = {
            "name": str(n.name), "op": f"pad{self.op_suffix}",
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
        }
        if self.quant == "int8":
            rec["quant"] = {
                "scale_in":  self.scales.get(in_name, 1e-8),
                "scale_out": self.scales.get(out_name, 1e-8),
                "pad_value": 0,  # zero-pad in int8 domain
            }
        self.ops.append(rec)

    def _emit_maxpool(self, n):
        in_name = self._resolve_input(n.args[0])
        ks = _pair(n.args[1]) if len(n.args) > 1 else (2, 2)
        st = _pair(n.args[2]) if len(n.args) > 2 else ks
        out_name = n.name
        self._record_tensor(out_name)
        shape = self.tensors[out_name].shape
        in_shape = self.tensors[in_name].shape
        rec = {
            "name": str(n.name), "op": f"maxpool2d{self.op_suffix}",
            "inputs": [in_name], "outputs": [out_name],
            "shape": {
                "N": int(shape[0]), "C": int(shape[1]),
                "IH": int(in_shape[2]), "IW": int(in_shape[3]),
                "KH": int(ks[0]), "KW": int(ks[1]),
                "SH": int(st[0]), "SW": int(st[1]),
                "PH": 0, "PW": 0, "DH": 1, "DW": 1,
            },
        }
        if self.quant == "int8":
            rec["quant"] = {
                "scale_in":  self.scales.get(in_name, 1e-8),
                "scale_out": self.scales.get(out_name, 1e-8),
            }
        self.ops.append(rec)

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

def _emit_vint_post_op(walker, out_dir):
    """Emit the ViNT-specific composite post-process op.

    Absorbs BOTH model outputs (the int8 dist_pred and the int8
    delta-predictions tensor that feeds the cumsum + L2-normalize
    tail) into one fp32 buffer the application consumes. Replaces
    the original user-output names (linear_23, reshape_2) with a
    single composite tensor (vint_combined_output).

    Also captures a (inputs, output) tuple from the first
    calibration sample for verify use (saved as
    app_op_vint_action_post.npz alongside io.npz). This is the
    oracle the curated kernel / LLM-gen works against — bit-exact
    comparison against the captured PyTorch tail evaluation.
    """
    if "linear_24" not in walker.tensors or "linear_23" not in walker.tensors:
        print("[extract_export] WARN: missing linear_23/linear_24 captures; "
              "skipping vint_action_post emission.", flush=True)
        return None
    linear_23_t = walker.tensors["linear_23"]   # dist_pred raw fp32
    linear_24_t = walker.tensors["linear_24"]   # action_pred deltas raw fp32
    scale_dist = walker.scales["linear_23"]
    scale_deltas = walker.scales["linear_24"]

    # Reference fp32 tail evaluation — MUST mirror what the C kernel
    # would compute given the *quantized* inputs (the binary feeds the
    # kernel int8 values from the buffers the upstream linear ops
    # wrote). That means: quantize → dequantize → apply tail. This is
    # the "int8 representation ceiling": what the composite outputs
    # when the int8 forward is bit-exact against PyTorch fp32 at
    # linear_23 / linear_24. Using the raw fp32 captured tensors
    # instead would over-estimate the ceiling — the composite never
    # actually sees pre-quant float values.
    in_dist_int8 = _quantize_per_tensor_sym(linear_23_t, scale_dist).ravel()
    in_deltas_int8 = _quantize_per_tensor_sym(
        linear_24_t, scale_deltas).ravel()
    deq_deltas = in_deltas_int8.astype(np.float32).reshape(5, 4) * scale_deltas
    waypts = np.zeros_like(deq_deltas)
    waypts[:, 0] = np.cumsum(deq_deltas[:, 0])
    waypts[:, 1] = np.cumsum(deq_deltas[:, 1])
    norms = np.linalg.norm(deq_deltas[:, 2:], axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    waypts[:, 2:] = deq_deltas[:, 2:] / norms
    deq_dist = float(in_dist_int8[0]) * scale_dist
    out_combined = np.concatenate([
        np.array([deq_dist], dtype=np.float32),
        waypts.ravel().astype(np.float32),
    ])

    out_name = "vint_combined_output"
    walker.tensors_meta[out_name] = {
        "shape": [21],
        "dtype": "f32",
        "quant": {"scale": 1.0, "zero_point": 0,
                  "kind": "app_op_composite"},
    }
    walker.tensors[out_name] = torch.from_numpy(out_combined)
    walker.ops.append({
        "name": "vint_action_post",
        "op": "vint_action_post",
        "inputs": ["linear_23", "linear_24"],
        "outputs": [out_name],
        "shape": {"n_wp": 5, "n_dim": 4},
        "quant": {
            "scale_dist":   float(scale_dist),
            "scale_deltas": float(scale_deltas),
            # No scale_out — output is fp32.
        },
        "semantics": {
            "description": (
                "ViNT action post: dequant dist_pred + cumsum action_pred "
                "(dx, dy) + L2-normalize (sin, cos), pack into one fp32 "
                "buffer."),
            "source_subgraph": [
                "cumsum.default(dim=1)",
                "linalg_vector_norm(dim=-1)",
                "clamp_min(eps=1e-12)",
                "div.Tensor",
                "tuple-pack dist_pred + action_pred",
            ],
            "io_samples": "app_op_vint_action_post.npz",
        },
    })
    # Capture (input1, input2, output) for the verify oracle.
    in_dist = _quantize_per_tensor_sym(linear_23_t, scale_dist).ravel()
    in_deltas = _quantize_per_tensor_sym(linear_24_t, scale_deltas).ravel()
    np.savez(
        Path(out_dir) / "app_op_vint_action_post.npz",
        input_dist=in_dist,
        input_deltas=in_deltas,
        scale_dist=np.float32(scale_dist),
        scale_deltas=np.float32(scale_deltas),
        output=out_combined,
    )
    # The composite op's output IS the new surface output of the
    # model; the caller (main) replaces output_names with [out_name]
    # so downstream codegen / io.npz / emit_test_io see one fp32
    # tensor instead of (int8 dist + fp32 waypoints) mixed.
    # We deliberately do NOT touch linear_23 / linear_24 / reshape_2
    # metadata — those remain int8 intermediate compute buffers the
    # linear_s8_pc kernels write to.
    return out_name


def _resolve_op_precision(ep, model_mod, fp16_ops_cli: str,
                          fp16_patterns_cli: str, default: str) -> dict[str, str]:
    """Resolve per-op precision overrides into a {node_name: 'fp16'|'int8'}
    map. Sources, in order (later wins on conflict):

      1. ``model_mod.get_precision_spec()`` — per-model defaults if the
         module defines the hook. Spec shape:
           {"default": "int8" | "fp16",
            "fp16_ops": [<aten node names>],
            "fp16_patterns": [<fnmatch globs>],
            "int8_ops": [<aten node names>]}
      2. ``--fp16-ops`` CLI list — explicit promotions.
      3. ``--fp16-patterns`` CLI list — glob promotions.

    Only emits entries that DIFFER from ``default`` so the walker's
    per-op lookup is a true override map (and the all-default path
    stays empty).
    """
    import fnmatch as _fn
    nodes = list(ep.graph_module.graph.nodes)
    aten_names = [n.name for n in nodes]
    node_by_name = {n.name: n for n in nodes}
    explicit_fp16: set[str] = set()
    explicit_int8: set[str] = set()
    spec = (model_mod.get_precision_spec()
            if hasattr(model_mod, "get_precision_spec") else None)

    def _upstream(target_name: str) -> set[str]:
        """All ancestor aten node names of `target_name`, traversed
        backwards through n.all_input_nodes. Used to promote a whole
        subgraph (e.g. the goal_encoder) by naming just its output."""
        if target_name not in node_by_name:
            return set()
        seen: set[str] = set()
        stack = [node_by_name[target_name]]
        while stack:
            cur = stack.pop()
            if cur.name in seen:
                continue
            seen.add(cur.name)
            for parent in cur.all_input_nodes:
                if parent.name not in seen:
                    stack.append(parent)
        return seen

    if spec:
        for nm in spec.get("fp16_ops", []) or []:
            explicit_fp16.add(nm)
        for nm in spec.get("int8_ops", []) or []:
            explicit_int8.add(nm)
        for pat in spec.get("fp16_patterns", []) or []:
            for nm in aten_names:
                if _fn.fnmatch(nm, pat):
                    explicit_fp16.add(nm)
        for target in spec.get("fp16_upstream_of", []) or []:
            # Promote `target` + every aten node feeding into it (the
            # entire subgraph). Cheap-and-precise alternative to listing
            # ops by name for big regions like an encoder branch.
            explicit_fp16.update(_upstream(target))
    if fp16_ops_cli.strip():
        for nm in [s.strip() for s in fp16_ops_cli.split(",") if s.strip()]:
            explicit_fp16.add(nm)
    if fp16_patterns_cli.strip():
        for pat in [s.strip() for s in fp16_patterns_cli.split(",") if s.strip()]:
            for nm in aten_names:
                if _fn.fnmatch(nm, pat):
                    explicit_fp16.add(nm)
    overrides: dict[str, str] = {}
    for nm in explicit_fp16:
        if "fp16" != default:
            overrides[nm] = "fp16"
    for nm in explicit_int8:
        if "int8" != default:
            overrides[nm] = "int8"
    return overrides


def _import_model_module(name: str):
    if name == "vint":
        from agents.models import vint as model_mod
    else:
        raise SystemExit(
            f"--model {name} doesn't need extract_graph_export; "
            f"use extract_graph.py (FX-based) instead."
        )
    return model_mod


def _load_model(name: str):
    model_mod = _import_model_module(name)
    sample = model_mod.get_sample_input()
    if isinstance(sample, torch.Tensor):
        sample = (sample,)
    return model_mod.get_model(), sample


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=["vint"])
    p.add_argument("--quant", default="int8", choices=["fp32", "int8", "fp16"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--inventory-only", action="store_true",
                   help="Only print the aten-op classification report; "
                        "don't emit IR.")
    p.add_argument("--per-channel", action="store_true",
                   help="Enable per-channel weight-scale quant for conv2d "
                        "and linear ops (one Q0.31 multiplier+shift per "
                        "output channel). Big accuracy win on trained "
                        "models — recommended for any int8 deployment.")
    p.add_argument("--inspect", default="",
                   help="Comma list of tensor names to dump at runtime "
                       "(AGENTS_INSPECT_BEGIN/END blocks in stdout, one "
                       "ASCII int per line, plus a scale= header so the "
                       "host can dequantize). Useful for staged accuracy "
                       "debugging: see which intermediate first diverges "
                       "from PyTorch fp32.")
    p.add_argument("--num-calibration", type=int, default=1,
                   help="Number of calibration samples (currently uses the "
                        "model's get_sample_input() repeatedly with fresh "
                        "RNG; will grow into a real dataset hook).")
    p.add_argument("--fp16-ops", default="",
                   help="Comma-separated aten node names to promote to "
                        "fp16 (overriding --quant for these ops alone). "
                        "Additive to the model's get_precision_spec() if "
                        "the model defines one. Example: "
                        "--fp16-ops linear,linear_24")
    p.add_argument("--fp16-patterns", default="",
                   help="Comma-separated fnmatch globs applied to aten "
                        "node names; matching ops are promoted to fp16. "
                        "Additive to --fp16-ops and the model's spec. "
                        "Example: --fp16-patterns 'layer_norm_*'")
    args = p.parse_args()

    if args.quant == "fp32":
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

    # Build the calibration dataset. Inputs are keyed by the aten
    # graph's user-input placeholder names (NOT our chosen labels)
    # so the walker can resolve `obs_img`/`goal_img` references the
    # same way ViNT.forward declared them.
    sig = ep.graph_signature
    user_input_names = [s.arg.name for s in sig.input_specs
                         if s.kind.name == "USER_INPUT"]
    if len(user_input_names) != len(sample):
        raise SystemExit(
            f"export user_input count {len(user_input_names)} != "
            f"sample length {len(sample)}")
    input_order = user_input_names

    # Multi-sample calibration. The model module declares a
    # *calibration spec* (the canonical, reproducible source-of-truth
    # for where activation scales come from). The spec gets resolved
    # via agents.datasets.materialize_calibration_samples and
    # serialized into the generated/ dir for later reference. If the
    # model doesn't ship a spec yet, fall back to its
    # get_calibration_samples helper, then finally the single trace
    # sample.
    calib_samples_tuples: list[tuple] = []
    calib_spec: dict | None = None
    # Always import the model module so we can inspect its optional
    # hooks (get_calibration_spec, get_precision_spec, ...). Cheap; the
    # actual model weights have already been loaded by _load_model.
    mod = _import_model_module(args.model)
    if args.num_calibration > 1:
        if hasattr(mod, "get_calibration_spec"):
            from agents.datasets import materialize_calibration_samples  # noqa: PLC0415
            calib_spec = mod.get_calibration_spec(args.num_calibration)
            print(f"[extract_export] resolving calibration spec "
                  f"({args.num_calibration} samples) ...", flush=True)
            for k, v in calib_spec["inputs"].items():
                src = v.get("path") or v.get("loader")
                print(f"    {k}: {v['loader']} from {src}", flush=True)
            materialized = materialize_calibration_samples(calib_spec)
            # Re-tuple in the order of the model's forward.
            calib_samples_tuples = [
                tuple(d[k] for k in d.keys()) for d in materialized
            ]
        elif hasattr(mod, "get_calibration_samples"):
            print(f"[extract_export] loading {args.num_calibration} "
                  f"calibration samples via {args.model}."
                  f"get_calibration_samples ...", flush=True)
            calib_samples_tuples = mod.get_calibration_samples(
                args.num_calibration)
        else:
            print(f"[extract_export] WARN: {args.model} module has no "
                  f"get_calibration_spec / get_calibration_samples; "
                  f"using single sample (scale quality will be poor for "
                  f"trained nets).", flush=True)
            calib_samples_tuples = [sample]
    else:
        calib_samples_tuples = [sample]

    calib_dicts = [
        {name: t for name, t in zip(input_order, s)}
        for s in calib_samples_tuples
    ]
    # Resolve per-op precision overrides — union of (a) the model's
    # get_precision_spec() if it defines one, (b) --fp16-ops explicit
    # node names, (c) --fp16-patterns fnmatch globs against aten node
    # names. Empty spec ⇒ all ops follow --quant (single-mode behavior).
    op_precision = _resolve_op_precision(
        ep, mod, args.fp16_ops, args.fp16_patterns, default=args.quant)
    if op_precision:
        n_fp16 = sum(1 for q in op_precision.values() if q == "fp16")
        n_int8 = sum(1 for q in op_precision.values() if q == "int8")
        print(f"[extract_export] precision overrides: "
              f"{n_fp16} ops → fp16, {n_int8} ops → int8 (rest = "
              f"{args.quant})", flush=True)

    walker = _ExportWalker(ep, calib_dicts, input_order,
                           per_channel=args.per_channel,
                           quant=args.quant,
                           op_precision=op_precision)
    print(f"[extract_export] calibrating + capturing intermediates ...",
          flush=True)
    walker.calibrate()
    print(f"[extract_export] emitting IR ...", flush=True)
    walker.emit()
    # Register user-input placeholders in tensors_meta BEFORE the
    # auto-cast pass — otherwise the cast pass can't see their dtype
    # and won't insert i8→f16 casts where an fp16 op consumes a
    # surface input directly (e.g. ViNT's cat takes obs slice + goal
    # placeholder; if cat is fp16-promoted, goal_img needs casting).
    for k in list(calib_dicts[0].keys()):
        walker._record_tensor(k)
    # Splice in i8↔f16 cast ops at any dtype boundaries the walker
    # produced. When no op was promoted (all-int8 or all-fp16 mode),
    # this pass is a no-op — every tensor's producer and consumer dtype
    # match by construction.
    _before = len(walker.ops)
    walker.insert_casts()
    _added = len(walker.ops) - _before
    if _added:
        print(f"[extract_export] auto-cast pass: inserted {_added} cast ops "
              f"at i8↔f16 boundaries", flush=True)
    # Weight-dtype reconciliation: when an op was promoted to fp16 but
    # its weight tensors (gamma/beta for layer_norm, the cast scale
    # tensors, etc.) were stored as int8 by _record_constant_buffer
    # (which honored the walker default), re-cast them to fp16 storage.
    # Weights are static so we just re-store at extract time — no
    # runtime cast needed. The auto-cast pass intentionally doesn't
    # touch weight refs (they pass through op["inputs"] unchanged).
    _LAYER_NORM_WEIGHT_KEYS = ("gamma_key", "beta_key")
    for _op in walker.ops:
        if _op.get("precision") != "fp16":
            continue
        for _key in _LAYER_NORM_WEIGHT_KEYS:
            _tname = _op.get(_key)
            if not _tname:
                continue
            _meta = walker.tensors_meta.get(_tname, {})
            if _meta.get("dtype") == "f16":
                continue
            # Re-store as fp16 (use original fp32 captured value).
            _t = walker.tensors.get(_tname)
            if _t is None:
                continue
            walker.weights_blob[_tname] = (
                _t.detach().cpu().numpy().astype(np.float16))
            walker.tensors_meta[_tname] = {
                "shape": list(_t.shape), "dtype": "f16",
                "quant": {"kind": "constant_buffer"},
            }

    # Inputs / outputs from the export's user signature. Use the
    # first calibration sample for sizing / golden capture.
    calib_sample = calib_dicts[0]
    input_names = list(calib_sample.keys())
    for k in input_names:
        walker._record_tensor(k)
    sig = ep.graph_signature
    output_names = [s.arg.name for s in sig.output_specs
                    if hasattr(s, "arg") and s.kind.name == "USER_OUTPUT"]
    # Walk the walker's alias chain (e.g. reshape_2 → linear_24) so each
    # surface output names a tensor that an actual kernel writes to —
    # generate_skeleton's ptr_for routes only top-level output names to
    # the surface buffer, and view/reshape aliases would otherwise leave
    # those slots un-written. The aliased name still appears in
    # walker.tensors_meta with the same shape, so downstream wiring works.
    output_names = [walker.name_map.get(n, n) for n in output_names]

    # Match the IR shape generate_skeleton consumes (see
    # extract_graph::extract_int8 for the canonical form):
    # * ir["input"] is {"tensor": <first_name>, "packed_inputs": [...]}
    #   for multi-input; just {"tensor": <name>} for single-input.
    # * ir["output"] is {"tensor": <name>, "tensors": [<names>]}.
    if len(input_names) == 1:
        ir_input: dict = {"tensor": input_names[0]}
    else:
        packed: list[dict] = []
        off = 0
        for nm in input_names:
            sz = int(np.prod(walker.tensors_meta[nm]["shape"]))
            packed.append({"name": nm, "offset": off, "size": sz})
            off += sz
        ir_input = {"tensor": input_names[0], "packed_inputs": packed}
    # Application-specific composite post-process emission. For
    # models that declare model-specific tail behavior (cumsum +
    # normalize + reshape kinds of patterns), emit one or more
    # composite `app_op_*` IR records here. The walker has already
    # captured the relevant intermediate tensors, so we just bundle
    # them into IR records with semantics info the codegen can use to
    # look up a curated kernel (or eventually drive LLM-gen).
    # The vint composite app_op exists to absorb the int8 walker's
    # mixed-dtype tail (int8 dist + fp32 waypoints) into a single fp32
    # surface output. In fp16 mode everything is already fp16 and the
    # model's surface output IS the desired output, so the composite
    # isn't needed (and emitting it would inject int8 metadata into an
    # otherwise fp16-clean IR).
    if args.model == "vint" and args.quant == "int8":
        composite_out = _emit_vint_post_op(walker, out_dir)
        if composite_out is not None:
            # Replace the original user-output names with the composite
            # — the binary now exposes a single fp32 surface output.
            output_names = [composite_out]

    dispatches = _annotate_dispatches(walker.ops)
    inspect_tensors = (
        [s.strip() for s in args.inspect.split(",") if s.strip()]
        if args.inspect else []
    )
    if inspect_tensors:
        # Also save the captured fp32 values at each inspected tensor
        # so the host comparison knows the PyTorch reference.
        ref = {}
        for t in inspect_tensors:
            resolved = walker.name_map.get(t, t)
            captured = walker.tensors.get(resolved)
            if captured is None:
                print(f"[extract_export] WARN: inspect tensor {t!r} "
                      f"(resolved {resolved!r}) not captured; skipping ref",
                      flush=True)
                continue
            ref[t] = captured.detach().cpu().numpy().astype(np.float32)
            ref[f"{t}__scale"] = np.float32(walker.scales.get(resolved, 1e-8))
        if ref:
            np.savez(out_dir / "inspect_ref.npz", **ref)
            print(f"[extract_export] wrote {out_dir / 'inspect_ref.npz'} "
                  f"({len([k for k in ref if not k.endswith('__scale')])} "
                  f"inspected tensors)", flush=True)
    ir = {
        "name": args.model,
        "version": 1,
        "quant": args.quant,
        "input": ir_input,
        "output": {
            "tensors": output_names,
            "tensor": output_names[0] if len(output_names) == 1 else None,
        },
        "tensors": walker.tensors_meta,
        "ops": walker.ops,
        "dispatches": dispatches,
        "inspect_tensors": inspect_tensors,
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

    # io.npz: emit_test_io expects flat "input" and "output" arrays
    # in the model's surface dtype (int8 for QUANT=int8). The walker
    # quantizes via the same per-tensor scales the IR baked in for
    # the user-input + model-output tensors.
    #
    # Output golden: use the WALKER-CAPTURED intermediate tensor at
    # each output name (e.g. linear_24's pre-cumsum delta predictions
    # for ViNT) — NOT the model's surface output. Two reasons:
    # (1) "tail" ops (cumsum / F.normalize / final reshape) are
    #     elided from the IR by design (we expect the C harness to
    #     apply them in scalar post-process), so the binary's last
    #     dispatch writes the pre-tail tensor's int8 values.
    # (2) the per-tensor scale stored in the IR for an output tensor
    #     is the captured tensor's max_abs, not the post-tail tensor's.
    # Using the captured intermediate keeps "what spike writes" and
    # "what we quantize the golden against" pointing at the same op
    # in the aten graph at the same scale.
    in_parts: list[np.ndarray] = []
    for k in input_names:
        t = calib_sample[k]
        if args.quant == "int8":
            in_parts.append(
                _quantize_per_tensor_sym(t, walker.scales[k]).ravel())
        elif args.quant == "fp16":
            in_parts.append(t.detach().cpu().numpy().ravel().astype(np.float16))
        else:
            in_parts.append(t.detach().cpu().numpy().ravel().astype(np.float32))
    if args.quant == "int8":
        in_dtype = np.int8
    elif args.quant == "fp16":
        in_dtype = np.float16
    else:
        in_dtype = np.float32
    inp_flat = np.concatenate(in_parts).astype(in_dtype)

    # Output: resolve each output name through the alias map; collect
    # each as its native fp32/int8/fp16 array. If all parts share one
    # dtype, concatenate as that dtype so emit_test_io maps to a
    # proper C array.
    #
    # fp16 path: re-run the model in half precision on the same input
    # so the golden reflects genuine fp16 numerics rather than
    # fp32-traced numerics down-cast at the boundary. The IR's surface
    # output is the captured intermediate tensor; we recompute it
    # under .half() so spike's fp16 output can be compared bit-by-bit
    # against the PyTorch fp16 reference.
    out_parts: list[np.ndarray] = []
    out_summary: list[str] = []
    fp16_golden: dict[str, np.ndarray] = {}
    if args.quant == "fp16":
        with torch.no_grad():
            half_model = model.half()
            half_inputs = tuple(calib_sample[k].half() for k in input_names)
            half_out = half_model(*half_inputs)
            if isinstance(half_out, (tuple, list)):
                half_outs = list(half_out)
            else:
                half_outs = [half_out]
            for nm, t in zip(output_names, half_outs):
                fp16_golden[nm] = t.detach().cpu().numpy().astype(np.float16).ravel()
    for out_name_i in output_names:
        resolved = walker.name_map.get(out_name_i, out_name_i)
        if args.quant == "fp16":
            if out_name_i in fp16_golden:
                arr = fp16_golden[out_name_i]
            else:
                # Walker captured the intermediate in fp32; the surface
                # output in fp16 mode is the model's actual fp16 forward,
                # but if the IR has a renamed/composite output we fall
                # back to a cast of the captured fp32 tensor.
                captured = walker.tensors.get(resolved)
                if captured is None:
                    raise SystemExit(
                        f"output {out_name_i!r}: no fp16 golden and no "
                        f"captured tensor under {resolved!r}")
                arr = captured.detach().cpu().numpy().ravel().astype(np.float16)
            out_parts.append(arr)
            out_summary.append(f"{resolved}={arr.size}f16")
            continue
        captured = walker.tensors.get(resolved)
        if captured is None:
            raise SystemExit(
                f"output tensor {out_name_i!r} (resolved to {resolved!r}) "
                f"not captured during calibration; cannot emit io.npz golden")
        meta = walker.tensors_meta.get(resolved, {})
        dtype = meta.get("dtype", "i8" if args.quant == "int8" else "f32")
        if dtype == "f32" or args.quant != "int8":
            arr = captured.detach().cpu().numpy().ravel().astype(np.float32)
            out_parts.append(arr)
            out_summary.append(f"{resolved}={arr.size}f32")
        else:
            sc = walker.scales.get(resolved, 1e-8)
            arr = _quantize_per_tensor_sym(captured, sc).ravel()
            out_parts.append(arr)
            out_summary.append(f"{resolved}={arr.size}i8")
    if all(a.dtype == np.float32 for a in out_parts):
        out_flat = np.concatenate(out_parts).astype(np.float32)
    elif all(a.dtype == np.float16 for a in out_parts):
        out_flat = np.concatenate(out_parts).astype(np.float16)
    elif all(a.dtype == np.int8 for a in out_parts):
        out_flat = np.concatenate(out_parts).astype(np.int8)
    else:
        dtypes = sorted({str(a.dtype) for a in out_parts})
        raise NotImplementedError(
            f"mixed-dtype outputs not packed yet: {dtypes}. Absorb them "
            f"into a single composite app_op so the surface is uniform.")
    io_path = out_dir / "io.npz"
    np.savez(io_path, input=inp_flat, output=out_flat)
    print(f"[extract_export] wrote {io_path} "
          f"(input {inp_flat.size} {in_dtype.__name__}, "
          f"output {out_flat.size} bytes [{', '.join(out_summary)}])",
          flush=True)

    # Persist the calibration spec for reproducibility — anyone with
    # this file + the source data can re-derive the same scales.
    if calib_spec is not None:
        spec_path = out_dir / "calibration_spec.json"
        with open(spec_path, "w") as f:
            json.dump(calib_spec, f, indent=2)
        print(f"[extract_export] wrote {spec_path}", flush=True)


if __name__ == "__main__":
    main()
