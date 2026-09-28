"""Faithful int8 forward: verbatim reuse of modelblaster extract_int8's golden
simulator loop. Same _sim_* primitives, same requantize, same scheme as the deploy.
Returns the `activations` dict so the caller can read any output tensor (e.g. steer)."""
import numpy as np
from modelblaster.pipeline.extract_graph import (
    _sim_conv2d_s8, _sim_batchnorm2d_s8, _sim_silu_s8, _sim_conv2d_int32_acc,
    _requantize_int, _requantize_int_per_oc, _round_half_away,
)

def simulate_int8(ops, weights_blob, inputs_q, input_node_names):
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
    return activations
