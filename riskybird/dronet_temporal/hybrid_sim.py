"""Hybrid int8/fp16 DroNet simulator for per-op precision characterization.

Runs the extract_int8 graph, but any op whose name is in `fp16_set` executes in
fp16 (fp16-rounded operands, fp32 accumulate for GEMM/conv -- matches the
kernel_*_f16 'fp16 storage, fp32 accumulate' contract) using the ORIGINAL FLOAT
weights, while every other op runs the bit-exact int8 golden (reusing
int8_sim's primitives). int8<->fp16 boundaries auto-cast via per-tensor scales.

For fused conv2d_batchnorm2d_s8, the conv and bn halves are promoted
independently (bn sub-op name in fp16_set -> conv stays int8 on the array, bn
runs fp16 on the vector unit -- the real deployable split). BN ops here also
fold a following ReLU (quant.activation_min==0), replicated in fp16.
"""
import numpy as np
from modelblaster.pipeline.extract_graph import (
    _sim_conv2d_s8, _sim_batchnorm2d_s8, _requantize_int)


def _conv_i8_fast(in_arr, sh, q, w_q, b_q):
    """Bit-exact fast int8 conv: float64 conv (exact integer accumulate for these
    magnitudes) then the same integer requantize as _sim_conv2d_s8. ~100x faster."""
    import torch
    import torch.nn.functional as F
    N, IC, IH, IW = sh["N"], sh["IC"], sh["IH"], sh["IW"]
    OC, KH, KW = sh["OC"], sh["KH"], sh["KW"]
    x = torch.from_numpy(in_arr.reshape(N, IC, IH, IW).astype(np.float64)
                         + float(q["input_offset"]))
    w = torch.from_numpy(w_q.reshape(OC, IC, KH, KW).astype(np.float64)
                         + float(q["filter_offset"]))
    acc = F.conv2d(x, w, stride=(sh["SH"], sh["SW"]),
                   padding=(sh["PH"], sh["PW"])).numpy()
    acc = np.rint(acc).astype(np.int64) + b_q.astype(np.int64).reshape(1, OC, 1, 1)
    scaled = _requantize_int(acc.astype(np.int32),
                             q["output_multiplier"], q["output_shift"])
    scaled = scaled + q["output_offset"]
    scaled = np.clip(scaled, q["activation_min"], q["activation_max"])
    return scaled.astype(np.int8)


def _f16(x):  # fp16 storage rounding
    return np.asarray(x, dtype=np.float32).astype(np.float16)


def _conv_f16(x_f, W_f, b_f, sh):
    """fp16 conv: fp16 operands, fp32 accumulate, fp16 output. NCHW."""
    import torch
    xt = torch.from_numpy(_f16(x_f).astype(np.float32)).reshape(
        sh["N"], sh["IC"], sh["IH"], sh["IW"])
    wt = torch.from_numpy(_f16(W_f).astype(np.float32))
    bt = torch.from_numpy(_f16(b_f).astype(np.float32)) if b_f is not None else None
    y = torch.nn.functional.conv2d(xt, wt, bt, stride=(sh["SH"], sh["SW"]),
                                   padding=(sh["PH"], sh["PW"]))
    return y.numpy().astype(np.float16)


def _bn_f16(x_f, bn, relu):
    """fp16 batchnorm (affine w/ running stats) + optional folded relu."""
    g, be, rm, rv, eps = bn
    x = _f16(x_f).astype(np.float32)
    scale = (g / np.sqrt(rv + eps)).astype(np.float32)
    shift = (be - rm * scale).astype(np.float32)
    y = x * scale.reshape(1, -1, 1, 1) + shift.reshape(1, -1, 1, 1)
    if relu:
        y = np.maximum(y, 0.0)
    return y.astype(np.float16)


def _linear_f16(x_f, W_f, b_f, M, K, N):
    x = _f16(x_f).astype(np.float32).reshape(M, K)
    W = _f16(W_f).astype(np.float32).reshape(N, K)
    y = x @ W.T
    if b_f is not None:
        y = y + _f16(b_f).astype(np.float32).reshape(1, N)
    return y.astype(np.float16)


def _maxpool(x, sh):
    N, C = sh["N"], sh["C"]; IH, IW = sh["IH"], sh["IW"]
    OH, OW = sh["OH"], sh["OW"]; KH, KW = sh["KH"], sh["KW"]
    SH, SW = sh["SH"], sh["SW"]; PH, PW = sh.get("PH", 0), sh.get("PW", 0)
    x4 = x.reshape(N, C, IH, IW)
    pad_val = np.iinfo(np.int8).min if x.dtype == np.int8 else np.float16(-65504.0)
    xp = np.pad(x4, ((0, 0), (0, 0), (PH, PH), (PW, PW)),
                constant_values=pad_val) if (PH or PW) else x4
    out = np.zeros((N, C, OH, OW), dtype=x.dtype)
    for oh in range(OH):
        for ow in range(OW):
            win = xp[:, :, oh*SH:oh*SH+KH, ow*SW:ow*SW+KW]
            out[:, :, oh, ow] = win.reshape(N, C, -1).max(axis=-1)
    return out


def simulate_hybrid(ops, wb, inputs_q, in_names, scales, fp16_set, fparams):
    """scales: tensor_name -> int8 scale. fparams: op_name -> weights dict.
    Returns activations dict (int8 arrays or float16 arrays)."""
    act = {nm: q for nm, q in zip(in_names, inputs_q)}

    def to_i8(name, scale=None):
        a = act[name]
        if a.dtype == np.int8:
            return a
        s = scale if scale is not None else scales[name]
        v = np.round(a.astype(np.float32) / np.float32(s))
        return np.clip(v, -128, 127).astype(np.int8)

    def to_f16(name, scale=None):
        a = act[name]
        if a.dtype != np.int8:
            return a.astype(np.float16)
        s = scale if scale is not None else scales[name]
        return (a.astype(np.float32) * np.float32(s)).astype(np.float16)

    for op in ops:
        nm = op["name"]; kind = op["op"]
        out = op["outputs"][0]
        inn = op["inputs"][0]
        if kind == "conv2d_s8":
            if nm in fp16_set:
                fp = fparams[nm]
                act[out] = _conv_f16(to_f16(inn), fp["W"], fp["b"], op["shape"])
            else:
                act[out] = _conv_i8_fast(to_i8(inn), op["shape"], op["quant"],
                                          wb[op["weight"]], wb[op["bias"]])
        elif kind == "conv2d_batchnorm2d_s8":
            csub, bsub = op["sub_ops"][0], op["sub_ops"][1]
            cname, bname = csub["name"], bsub["name"]
            si = float(bsub["quant"]["scale_in"])      # conv-out / bn-in scale
            so = float(bsub["quant"]["scale_out"])     # = op output tensor scale
            bn_relu = bsub["quant"].get("activation_min", -128) == 0
            # ---- conv half ----
            if cname in fp16_set:
                fp = fparams[cname]
                conv_out = _conv_f16(to_f16(inn), fp["W"], fp["b"], csub["shape"])
                conv_dtype = "f16"
            else:
                conv_out = _conv_i8_fast(to_i8(inn), csub["shape"], csub["quant"],
                                          wb[csub["weight"]], wb[csub["bias"]])
                conv_dtype = "i8"
            # ---- bn half ----
            if bname in fp16_set:
                cf = (conv_out.astype(np.float32) * np.float32(si)).astype(np.float16) \
                    if conv_dtype == "i8" else conv_out.astype(np.float16)
                act[out] = _bn_f16(cf, fparams[bname]["bn"], bn_relu)
            else:
                ci = conv_out if conv_dtype == "i8" else \
                    np.clip(np.round(conv_out.astype(np.float32)/np.float32(si)),
                            -128, 127).astype(np.int8)
                act[out] = _sim_batchnorm2d_s8(ci, bsub["shape"], bsub["quant"],
                                               wb[bsub["weight"]], wb[bsub["bias"]])
        elif kind == "batchnorm2d_s8":
            bn_relu = op["quant"].get("activation_min", -128) == 0
            if nm in fp16_set:
                act[out] = _bn_f16(to_f16(inn), fparams[nm]["bn"], bn_relu)
            else:
                act[out] = _sim_batchnorm2d_s8(to_i8(inn), op["shape"], op["quant"],
                                               wb[op["weight"]], wb[op["bias"]])
        elif kind == "add_s8":
            q = op["quant"]; i0, i1 = op["inputs"][0], op["inputs"][1]
            if nm in fp16_set:
                act[out] = (to_f16(i0).astype(np.float32) +
                            to_f16(i1).astype(np.float32)).astype(np.float16)
            else:
                a = to_i8(i0).astype(np.float32) * np.float32(q["scale_a"])
                b = to_i8(i1).astype(np.float32) * np.float32(q["scale_b"])
                f = (a + b) / np.float32(q["scale_out"])
                act[out] = np.clip(np.round(f), q["activation_min"],
                                   q["activation_max"]).astype(np.int8)
        elif kind == "relu_s8":
            if nm in fp16_set:
                act[out] = np.maximum(to_f16(inn), np.float16(0.0))
            else:
                act[out] = np.maximum(to_i8(inn), 0).astype(np.int8)
        elif kind == "linear_s8":
            sh = op["shape"]
            if nm in fp16_set:
                fp = fparams[nm]
                act[out] = _linear_f16(to_f16(inn), fp["W"], fp["b"],
                                       sh["M"], sh["K"], sh["N"])
            else:
                q = op["quant"]
                from modelblaster.pipeline.extract_graph import _requantize_int
                in2 = to_i8(inn).reshape(sh["M"], sh["K"]).astype(np.int32)
                w2 = wb[op["weight"]].reshape(sh["N"], sh["K"]).astype(np.int32)
                acc = (in2 + q["input_offset"]) @ (w2 + q["filter_offset"]).T
                acc += wb[op["bias"]].astype(np.int32)
                sc = _requantize_int(acc, q["output_multiplier"], q["output_shift"])
                sc += q["output_offset"]
                act[out] = np.clip(sc, q["activation_min"],
                                   q["activation_max"]).astype(np.int8)
        elif kind == "maxpool2d_s8":
            src = act[inn]
            act[out] = _maxpool(src, op["shape"])
        elif kind == "view":
            act[out] = act[inn]
        elif kind == "sigmoid_s8":
            act[out] = act[inn]  # collision head unused for steering
        else:
            raise NotImplementedError(f"hybrid_sim: op {kind!r}")
    return act
