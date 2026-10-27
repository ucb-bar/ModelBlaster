"""Coarse op classes for cross-kernel knowledge sharing.

KernelBlaster shares implementations and lessons between kernels of the same
class (a tiled-GEMM trick learned on `linear` applies to `bmm`), so every op
an exported problem contains is tagged with one class. Unknown ops are
"other"; the int8/fp16 variants take their base op's class, and a fused op
(``fused__<k0>__...``) its first stage's -- the producer whose kernel the
epilogue joins.
"""
from __future__ import annotations

OP_CLASSES = ("gemm", "conv", "reduction", "elementwise", "data_movement", "attention", "scan", "other")

_CLASS = {
    "gemm": ["linear", "matmul", "matmul_ta", "matmul_tb", "matmul_tatb", "bmm", "diag_matmul",
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
    "attention": ["sdpa"],
    "scan": ["cumsum", "cumprod", "exclusive_cumsum"],
}
_BY_OP = {op: c for c, ops in _CLASS.items() for op in ops}


def op_class(op: str) -> str:
    if op.startswith("fused__"):
        op = op.split("__")[1]
    for suffix in ("_bf16", "_f16", "_s8_pc", "_s8", "_i8"):
        if op.endswith(suffix) and op not in _BY_OP:
            op = op[: -len(suffix)]
    return _BY_OP.get(op, "other")
