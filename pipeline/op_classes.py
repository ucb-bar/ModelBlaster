"""Coarse op classes for cross-kernel knowledge sharing.

KernelBlaster shares implementations and lessons between kernels of the same
class (a tiled-GEMM trick learned on `linear` applies to `bmm`), so every op
an exported problem contains is tagged with one class. Unknown ops are
"other"; the int8/fp16 variants take their base op's class, and a fused op
(``fused__<k0>__...``) its first stage's -- the producer whose kernel the
epilogue joins. The dtype casts of a mixed IR are elementwise.

Recurrent class (A2, opt-in): with ``$MB_RECURRENT_CLASS=1`` (``kb_graph_opt --recurrent-class on``) the sequence
recurrences (``gru_seq``, ``lstm_seq``, ``lstm``) are class ``recurrent`` instead of ``gemm``: KernelBlaster then
offers the recurrent technique menu (persistent recurrence, state in registers / smem, input-projection hoisting,
wavefront pipelines, step-sync minimisation) and the latency-per-step hardware notes instead of GEMM tiling,
memory_coalescing-first and split-K. Default off: every class is what it was.
"""
from __future__ import annotations

import os

OP_CLASSES = ("gemm", "conv", "reduction", "elementwise", "data_movement", "attention", "scan", "recurrent", "other")
RECURRENT_ENV = "MB_RECURRENT_CLASS"
RECURRENT_KINDS = ("lstm", "lstm_seq", "gru_seq")


def recurrent_enabled(value=None) -> bool:
    """``value`` (bool / on|off) wins, else ``$MB_RECURRENT_CLASS`` (default off)."""
    v = os.environ.get(RECURRENT_ENV, "0") if value is None else value
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "on", "true", "yes")

_CLASS = {
    "gemm": ["linear", "matmul", "matmul_ta", "matmul_tb", "matmul_tatb", "bmm", "bmm_strided", "diag_matmul",
             "lstm", "lstm_seq", "gru_seq"],
    "conv": ["conv2d", "conv2d_dw", "conv2d_grouped", "conv3d", "convnd", "conv_transpose2d",
             "conv_transpose3d"],
    "reduction": ["softmax", "softmax_dim", "log_softmax", "layer_norm", "group_norm", "rms_norm",
                  "reduce", "sum_dim", "mean_dim", "max_dim", "min_dim", "prod_dim",
                  "argmax_dim", "argmin_dim", "frobenius_norm", "l1_norm", "l2_norm", "mean_abs_norm",
                  "maxpool2d", "maxpool2d_ex", "maxpool3d", "avgpool2d", "avgpool3d",
                  "adaptive_avg_pool2d", "cross_entropy_loss", "hinge_loss", "huber_loss",
                  "kldiv_loss", "mse_loss", "triplet_loss", "topk"],
    # batchnorm2d is folded inference BN: a per-channel scale + shift
    "elementwise": ["batchnorm2d", "relu", "relu6", "sigmoid", "silu", "swish", "elu", "selu", "gelu", "gelu_exact",
                    "tanh", "log", "hardsigmoid", "hardtanh", "leaky_relu", "softplus", "softsign",
                    "add", "mul", "mul_scalar", "ewise1", "ewise2", "ewise_scalar", "tril", "triu"],
    "data_movement": ["strided_copy", "strided_scatter", "gather", "gather_dim", "index_select",
                      "scatter_dim", "cat2_c1", "cat3_c1", "cat4_c1", "catn", "flip",
                      "upsample_nearest", "view", "chunk2_c1"],
    "attention": ["sdpa", "attention"],
    "scan": ["cumsum", "cumprod", "exclusive_cumsum"],
}
_BY_OP = {op: c for c, ops in _CLASS.items() for op in ops}
# the per-tensor dtype casts of a mixed IR (cast_<src>_to_<dst>, docs/tensor_dtypes.md)
_CAST_DTYPES = ("f32", "f16", "bf16", "i64", "i32", "b8", "u8")
_BY_OP.update({f"cast_{a}_to_{b}": "elementwise" for a in _CAST_DTYPES for b in _CAST_DTYPES if a != b})


def op_class(op: str, recurrent=None) -> str:
    """Class of an op kind; ``recurrent`` (None: ``$MB_RECURRENT_CLASS``) moves the sequence recurrences from
    ``gemm`` to ``recurrent``."""
    if op.startswith("region__"):                       # region slots (region_slots.py): by pattern
        from .region_slots import region_class
        return region_class(op, recurrent_enabled(recurrent))
    c = _base_class(op)
    if c == "gemm" and recurrent_enabled(recurrent) and _base_kind(op) in RECURRENT_KINDS:
        return "recurrent"
    return c


def _base_kind(op: str) -> str:
    if op.startswith("fused__"):
        parts = op.split("__")
        op = parts[2] if parts[1] == "par" and len(parts) > 2 else parts[1]
    for suffix in ("_bf16", "_f16", "_s8_pc", "_s8", "_i8"):
        if op.endswith(suffix) and op not in _BY_OP:
            op = op[: -len(suffix)]
    return op


def _base_class(op: str) -> str:
    return _BY_OP.get(_base_kind(op), "other")
