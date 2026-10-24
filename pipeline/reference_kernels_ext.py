"""Generic tensor ops: the vocabulary the level3 lowering falls back to.

The per-module lowerings in extract_graph emit one IR op per PyTorch
construct (conv2d, cat2_c1, mean_dim, ...). Real models also do a lot of
tensor plumbing — permute/slice/expand/roll, broadcasting arithmetic,
reductions over arbitrary dims, N-way concat — that doesn't deserve one
kernel per spelling. These ops cover it with a handful of shape-generic
kernels; extract_graph only uses them for nodes its specific lowerings
don't handle, so existing IRs are unchanged.

Rank-generic ops take a fixed rank of MB_MAXR = 6: shapes are
right-aligned and padded with leading 1s (stride 0), and the lowering
merges contiguous dims first, so real ranks above 6 are rare and rejected
there.

Opcodes (the `opc` arg) are listed in the semantics strings and in
EWISE1_OPS / EWISE2_OPS / EWISE_SCALAR_OPS / REDUCE_OPS, which extract_graph
and the GPU walkers share.
"""
from __future__ import annotations

import ctypes
import re


MB_MAXR = 6

EWISE1_OPS = {
    "neg": 0, "exp": 1, "log": 2, "sqrt": 3, "rsqrt": 4, "abs": 5, "tanh": 6,
    "sigmoid": 7, "relu": 8, "sin": 9, "cos": 10, "erf": 11, "gelu": 12,
    "gelu_tanh": 13, "silu": 14, "clamp": 15, "reciprocal": 16, "square": 17,
    "softplus": 18, "leaky_relu": 19, "elu": 20, "hardswish": 21,
    "hardsigmoid": 22, "log1p": 23, "expm1": 24, "sign": 25, "floor": 26,
    "ceil": 27, "round": 28, "logical_not": 29, "copy": 30, "mish": 31,
}
EWISE2_OPS = {
    "add": 0, "sub": 1, "mul": 2, "div": 3, "max": 4, "min": 5, "pow": 6,
    "masked_fill": 7, "eq": 8, "ne": 9, "lt": 10, "le": 11, "gt": 12, "ge": 13,
}
EWISE_SCALAR_OPS = {
    "add": 0, "sub": 1, "rsub": 2, "mul": 3, "div": 4, "rdiv": 5, "pow": 6,
    "rpow": 7, "max": 8, "min": 9, "eq": 10, "ne": 11, "lt": 12, "le": 13,
    "gt": 14, "ge": 15,
}
REDUCE_OPS = {"sum": 0, "mean": 1, "max": 2, "min": 3, "prod": 4, "sumsq": 5,
              "l2": 6, "argmax": 7, "argmin": 8, "var": 9, "var_unbiased": 10}


def _dims(prefix: str) -> str:
    return ", ".join(f"int {prefix}{i}" for i in range(MB_MAXR))


def _argtypes_from_signature(sig: str):
    m = re.match(r"\s*void\s+\w+\s*\((.*)\)\s*$", sig, re.S)
    out = []
    for raw in m.group(1).split(","):
        raw = " ".join(raw.split())
        if "*" in raw:
            out.append(ctypes.POINTER(ctypes.c_int) if " int " in f" {raw} "
                       else ctypes.POINTER(ctypes.c_float))
        elif raw.startswith("float"):
            out.append(ctypes.c_float)
        else:
            out.append(ctypes.c_int)
    return lambda: list(out)


_RAW: list[tuple[str, str, str, str]] = []


def _spec(op, signature, semantics, reference_impl):
    _RAW.append((op, signature, semantics, reference_impl))
    return op


# 6-D index walk shared by the strided kernels: `i` is the contiguous output
# index, i0..i5 its coordinates over d0..d5.
_IDX6 = """\
        long r = i;
        long i5 = r % d5; r /= d5;
        long i4 = r % d4; r /= d4;
        long i3 = r % d3; r /= d3;
        long i2 = r % d2; r /= d2;
        long i1 = r % d1; r /= d1;
        long i0 = r;
"""

STRIDED_COPY = _spec(
    "strided_copy",
    f"void kernel_strided_copy(const float *input, float *output, int off, "
    f"{_dims('d')}, {_dims('s')})",
    "Materialize a strided view: output is contiguous with shape "
    "[d0..d5]; output[i0..i5] = input[off + sum_k i_k * s_k]. Strides may be "
    "0 (broadcast/expand) and need not be monotonic (permute/transpose).",
    f"""\
void kernel_strided_copy(const float *input, float *output, int off,
                         {_dims('d')}, {_dims('s')}) {{
    long n = (long)d0 * d1 * d2 * d3 * d4 * d5;
    for (long i = 0; i < n; i++) {{
{_IDX6}        output[i] = input[off + i0*s0 + i1*s1 + i2*s2 + i3*s3 + i4*s4 + i5*s5];
    }}
}}
""")

STRIDED_SCATTER = _spec(
    "strided_scatter",
    f"void kernel_strided_scatter(const float *input, const float *src, float *output, "
    f"int n, int off, {_dims('d')}, {_dims('s')})",
    "Write a strided box into a copy of a tensor (x[idx] = v with basic "
    "indexing, functionalized): output = input (n elements), then "
    "output[off + sum_k i_k * s_k] = src[i] for src contiguous with shape "
    "[d0..d5]. The box must not overlap itself (strides from basic indexing).",
    f"""\
void kernel_strided_scatter(const float *input, const float *src, float *output,
                            int n, int off, {_dims('d')}, {_dims('s')}) {{
    for (long i = 0; i < n; i++) output[i] = input[i];
    long m = (long)d0 * d1 * d2 * d3 * d4 * d5;
    for (long i = 0; i < m; i++) {{
{_IDX6}        output[off + i0*s0 + i1*s1 + i2*s2 + i3*s3 + i4*s4 + i5*s5] = src[i];
    }}
}}
""")

GATHER = _spec(
    "gather",
    "void kernel_gather(const float *input, const int *index, float *output, int n)",
    "Constant-index gather (data movement that is not one affine view, e.g. "
    "torch.roll): output[i] = index[i] >= 0 ? input[index[i]] : 0.",
    """\
void kernel_gather(const float *input, const int *index, float *output, int n) {
    for (int i = 0; i < n; i++)
        output[i] = index[i] >= 0 ? input[index[i]] : 0.0f;
}
""")

CATN = _spec(
    "catn",
    "void kernel_catn(const float *in0, const float *in1, const float *in2, "
    "const float *in3, const float *in4, const float *in5, const float *in6, "
    "const float *in7, float *output, int n_in, int outer, int inner, "
    "int c0, int c1, int c2, int c3, int c4, int c5, int c6, int c7)",
    "Concatenate up to 8 inputs along one axis. Viewing input k as "
    "[outer, c_k, inner] and the output as [outer, sum(c_k), inner], copy "
    "each input to its slab. Unused in_k are NULL with c_k = 0.",
    """\
void kernel_catn(const float *in0, const float *in1, const float *in2,
                 const float *in3, const float *in4, const float *in5,
                 const float *in6, const float *in7, float *output,
                 int n_in, int outer, int inner,
                 int c0, int c1, int c2, int c3, int c4, int c5, int c6, int c7) {
    const float *ins[8] = {in0, in1, in2, in3, in4, in5, in6, in7};
    int cs[8] = {c0, c1, c2, c3, c4, c5, c6, c7};
    long ctot = 0;
    for (int k = 0; k < n_in; k++) ctot += cs[k];
    long cbase = 0;
    for (int k = 0; k < n_in; k++) {
        long slab = (long)cs[k] * inner;
        for (long o = 0; o < outer; o++)
            for (long j = 0; j < slab; j++)
                output[(o * ctot + cbase) * inner + j] = ins[k][o * slab + j];
        cbase += cs[k];
    }
}
""")

EWISE1 = _spec(
    "ewise1",
    "void kernel_ewise1(const float *input, float *output, int n, int opc, "
    "float p0, float p1)",
    "Elementwise unary op selected by opc: " + ", ".join(
        f"{v}={k}" for k, v in EWISE1_OPS.items()) +
    ". clamp: [p0, p1]; leaky_relu/elu: slope/alpha p0; gelu is the erf form.",
    """\
#include <math.h>
void kernel_ewise1(const float *input, float *output, int n, int opc,
                   float p0, float p1) {
    for (int i = 0; i < n; i++) {
        float x = input[i], y;
        switch (opc) {
        case 0: y = -x; break;
        case 1: y = expf(x); break;
        case 2: y = logf(x); break;
        case 3: y = sqrtf(x); break;
        case 4: y = 1.0f / sqrtf(x); break;
        case 5: y = fabsf(x); break;
        case 6: y = tanhf(x); break;
        case 7: y = 1.0f / (1.0f + expf(-x)); break;
        case 8: y = x > 0.0f ? x : 0.0f; break;
        case 9: y = sinf(x); break;
        case 10: y = cosf(x); break;
        case 11: y = erff(x); break;
        case 12: y = 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); break;
        case 13: y = 0.5f * x * (1.0f + tanhf(0.79788456080286536f * (x + 0.044715f * x * x * x))); break;
        case 14: y = x / (1.0f + expf(-x)); break;
        case 15: y = x < p0 ? p0 : (x > p1 ? p1 : x); break;
        case 16: y = 1.0f / x; break;
        case 17: y = x * x; break;
        case 18: y = x > 20.0f ? x : log1pf(expf(x)); break;
        case 19: y = x >= 0.0f ? x : p0 * x; break;
        case 20: y = x > 0.0f ? x : p0 * (expf(x) - 1.0f); break;
        case 21: { float t = x + 3.0f; t = t < 0.0f ? 0.0f : (t > 6.0f ? 6.0f : t); y = x * t / 6.0f; } break;
        case 22: { float t = x / 6.0f + 0.5f; y = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t); } break;
        case 23: y = log1pf(x); break;
        case 24: y = expm1f(x); break;
        case 25: y = (float)((x > 0.0f) - (x < 0.0f)); break;
        case 26: y = floorf(x); break;
        case 27: y = ceilf(x); break;
        case 28: y = nearbyintf(x); break;
        case 29: y = x == 0.0f ? 1.0f : 0.0f; break;
        case 31: y = x * tanhf(x > 20.0f ? x : log1pf(expf(x))); break;
        default: y = x; break;
        }
        output[i] = y;
    }
}
""")

EWISE2 = _spec(
    "ewise2",
    f"void kernel_ewise2(const float *a, const float *b, float *output, int opc, "
    f"float alpha, {_dims('d')}, {_dims('sa')}, {_dims('sb')})",
    "Broadcasting elementwise binary op. output is contiguous [d0..d5]; "
    "a is read at sum_k i_k*sa_k, b at sum_k i_k*sb_k (stride 0 = broadcast). "
    "opc: " + ", ".join(f"{v}={k}" for k, v in EWISE2_OPS.items()) +
    ". add/sub compute a +/- alpha*b; masked_fill gives b != 0 ? alpha : a; "
    "comparisons give 1.0/0.0.",
    f"""\
#include <math.h>
void kernel_ewise2(const float *a, const float *b, float *output, int opc,
                   float alpha, {_dims('d')}, {_dims('sa')}, {_dims('sb')}) {{
    long n = (long)d0 * d1 * d2 * d3 * d4 * d5;
    for (long i = 0; i < n; i++) {{
{_IDX6}        float x = a[i0*sa0 + i1*sa1 + i2*sa2 + i3*sa3 + i4*sa4 + i5*sa5];
        float y = b[i0*sb0 + i1*sb1 + i2*sb2 + i3*sb3 + i4*sb4 + i5*sb5];
        float z;
        switch (opc) {{
        case 0: z = x + alpha * y; break;
        case 1: z = x - alpha * y; break;
        case 2: z = x * y; break;
        case 3: z = x / y; break;
        case 4: z = (x > y || isnan(x)) ? x : y; break;
        case 5: z = (x < y || isnan(x)) ? x : y; break;
        case 6: z = powf(x, y); break;
        case 7: z = y != 0.0f ? alpha : x; break;
        case 8: z = x == y; break;
        case 9: z = x != y; break;
        case 10: z = x < y; break;
        case 11: z = x <= y; break;
        case 12: z = x > y; break;
        case 13: z = x >= y; break;
        default: z = x; break;
        }}
        output[i] = z;
    }}
}}
""")

EWISE_SCALAR = _spec(
    "ewise_scalar",
    "void kernel_ewise_scalar(const float *input, float *output, int n, int opc, float c)",
    "Elementwise op with a scalar constant c: " + ", ".join(
        f"{v}={k}" for k, v in EWISE_SCALAR_OPS.items()) +
    " (rsub = c - x, rdiv = c / x, rpow = c ** x; comparisons give 1.0/0.0).",
    """\
#include <math.h>
void kernel_ewise_scalar(const float *input, float *output, int n, int opc, float c) {
    for (int i = 0; i < n; i++) {
        float x = input[i], z;
        switch (opc) {
        case 0: z = x + c; break;
        case 1: z = x - c; break;
        case 2: z = c - x; break;
        case 3: z = x * c; break;
        case 4: z = x / c; break;
        case 5: z = c / x; break;
        case 6: z = powf(x, c); break;
        case 7: z = powf(c, x); break;
        case 8: z = (x > c || isnan(x)) ? x : c; break;
        case 9: z = (x < c || isnan(x)) ? x : c; break;
        case 10: z = x == c; break;
        case 11: z = x != c; break;
        case 12: z = x < c; break;
        case 13: z = x <= c; break;
        case 14: z = x > c; break;
        case 15: z = x >= c; break;
        default: z = x; break;
        }
        output[i] = z;
    }
}
""")

REDUCE = _spec(
    "reduce",
    "void kernel_reduce(const float *input, float *output, int outer, int R, "
    "int inner, int opc)",
    "Reduce the middle axis of input viewed as [outer, R, inner] into "
    "output [outer, inner]. opc: " + ", ".join(
        f"{v}={k}" for k, v in REDUCE_OPS.items()) +
    " (l2 = sqrt(sum x^2); argmax/argmin return the first index as float; "
    "var is the population variance, var_unbiased divides by R-1).",
    """\
#include <math.h>
void kernel_reduce(const float *input, float *output, int outer, int R,
                   int inner, int opc) {
    for (long o = 0; o < outer; o++) {
        for (long j = 0; j < inner; j++) {
            const float *x = input + o * (long)R * inner + j;
            float acc;
            if (opc == 7 || opc == 8) {
                long best = 0; float bv = x[0];
                for (long r = 1; r < R; r++) {
                    float v = x[r * inner];
                    if (opc == 7 ? v > bv : v < bv) { bv = v; best = r; }
                }
                acc = (float)best;
            } else if (opc == 9 || opc == 10) {
                float m = 0.0f;
                for (long r = 0; r < R; r++) m += x[r * inner];
                m /= (float)R;
                acc = 0.0f;
                for (long r = 0; r < R; r++) { float d = x[r * inner] - m; acc += d * d; }
                acc /= (float)(opc == 10 ? R - 1 : R);
            } else {
                acc = (opc == 2 || opc == 3) ? x[0] : (opc == 4 ? 1.0f : 0.0f);
                for (long r = (opc == 2 || opc == 3) ? 1 : 0; r < R; r++) {
                    float v = x[r * inner];
                    switch (opc) {
                    case 2: if (v > acc || isnan(v)) acc = v; break;
                    case 3: if (v < acc || isnan(v)) acc = v; break;
                    case 4: acc *= v; break;
                    case 5: case 6: acc += v * v; break;
                    default: acc += v; break;
                    }
                }
                if (opc == 1) acc /= (float)R;
                if (opc == 6) acc = sqrtf(acc);
            }
            output[o * inner + j] = acc;
        }
    }
}
""")

SOFTMAX_DIM = _spec(
    "softmax_dim",
    "void kernel_softmax_dim(const float *input, float *output, int outer, int R, "
    "int inner, int log_out)",
    "Softmax over the middle axis of [outer, R, inner] (any dim of the "
    "original tensor). log_out=1 gives log_softmax. Numerically stable "
    "(max-subtracted); a row of all -inf yields NaN like PyTorch.",
    """\
#include <math.h>
void kernel_softmax_dim(const float *input, float *output, int outer, int R,
                        int inner, int log_out) {
    for (long o = 0; o < outer; o++) {
        for (long j = 0; j < inner; j++) {
            const float *x = input + o * (long)R * inner + j;
            float *y = output + o * (long)R * inner + j;
            float m = x[0];
            for (long r = 1; r < R; r++) if (x[r * inner] > m) m = x[r * inner];
            float s = 0.0f;
            for (long r = 0; r < R; r++) s += expf(x[r * inner] - m);
            for (long r = 0; r < R; r++)
                y[r * inner] = log_out ? (x[r * inner] - m) - logf(s)
                                       : expf(x[r * inner] - m) / s;
        }
    }
}
""")

CONV2D_GROUPED = _spec(
    "conv2d_grouped",
    "void kernel_conv2d_grouped(const float *input, const float *weight, "
    "const float *bias, float *output, int N, int IC, int IH, int IW, int OC, "
    "int KH, int KW, int SH, int SW, int PH, int PW, int DH, int DW, int G)",
    "torch.nn.Conv2d with groups=G (1 < G, not depthwise) and dilation. "
    "input [N, IC, IH, IW], weight [OC, IC/G, KH, KW], bias [OC] or NULL, "
    "output [N, OC, OH, OW], OH = (IH + 2*PH - DH*(KH-1) - 1)/SH + 1 (same "
    "for W). Output channel oc reads input channels "
    "[(oc / (OC/G)) * (IC/G), +IC/G).",
    """\
/* weight OIHW ([OC, IC/G, KH, KW]) unless packed by the backend */
#if defined(MODELBLASTER_GEMMINI_HWIO_WEIGHTS)
#define MB_CONVG_W(oc, ic, kh, kw) ((((long)(kh)*KW + (kw))*ICg + (ic))*OC + (oc))
#elif defined(MODELBLASTER_RVV_IHWOC_WEIGHTS)
#define MB_CONVG_W(oc, ic, kh, kw) ((((long)(ic)*KH + (kh))*KW + (kw))*OC + (oc))
#else
#define MB_CONVG_W(oc, ic, kh, kw) ((((long)(oc)*ICg + (ic))*KH + (kh))*KW + (kw))
#endif
void kernel_conv2d_grouped(const float *input, const float *weight,
                           const float *bias, float *output,
                           int N, int IC, int IH, int IW, int OC,
                           int KH, int KW, int SH, int SW, int PH, int PW,
                           int DH, int DW, int G) {
    int OH = (IH + 2*PH - DH*(KH-1) - 1) / SH + 1;
    int OW = (IW + 2*PW - DW*(KW-1) - 1) / SW + 1;
    int ICg = IC / G, OCg = OC / G;
    for (int n = 0; n < N; n++)
    for (int oc = 0; oc < OC; oc++) {
        int ic0 = (oc / OCg) * ICg;
        for (int oh = 0; oh < OH; oh++)
        for (int ow = 0; ow < OW; ow++) {
            float acc = bias ? bias[oc] : 0.0f;
            for (int ic = 0; ic < ICg; ic++)
            for (int kh = 0; kh < KH; kh++) {
                int ih = oh * SH - PH + kh * DH;
                if (ih < 0 || ih >= IH) continue;
                for (int kw = 0; kw < KW; kw++) {
                    int iw = ow * SW - PW + kw * DW;
                    if (iw < 0 || iw >= IW) continue;
                    acc += input[((long)(n*IC + ic0 + ic)*IH + ih)*IW + iw]
                         * weight[MB_CONVG_W(oc, ic, kh, kw)];
                }
            }
            output[((long)(n*OC + oc)*OH + oh)*OW + ow] = acc;
        }
    }
}
#undef MB_CONVG_W
""")

MAXPOOL2D_EX = _spec(
    "maxpool2d_ex",
    "void kernel_maxpool2d_ex(const float *input, float *output, int N, int C, "
    "int IH, int IW, int OH, int OW, int KH, int KW, int SH, int SW, int PH, "
    "int PW, int DH, int DW)",
    "torch.nn.MaxPool2d with the output size given explicitly (OH, OW), so "
    "ceil_mode=True is expressible: output[n,c,oh,ow] = max over kh, kw of "
    "input[n, c, oh*SH - PH + kh*DH, ow*SW - PW + kw*DW], skipping taps "
    "outside the input (implicit -inf padding).",
    """\
#include <math.h>
void kernel_maxpool2d_ex(const float *input, float *output, int N, int C,
                         int IH, int IW, int OH, int OW, int KH, int KW,
                         int SH, int SW, int PH, int PW, int DH, int DW) {
    for (long nc = 0; nc < (long)N * C; nc++)
    for (int oh = 0; oh < OH; oh++)
    for (int ow = 0; ow < OW; ow++) {
        float m = -INFINITY;
        for (int kh = 0; kh < KH; kh++) {
            int ih = oh * SH - PH + kh * DH;
            if (ih < 0 || ih >= IH) continue;
            for (int kw = 0; kw < KW; kw++) {
                int iw = ow * SW - PW + kw * DW;
                if (iw < 0 || iw >= IW) continue;
                float v = input[(nc * IH + ih) * IW + iw];
                if (v > m || isnan(v)) m = v;
            }
        }
        output[(nc * OH + oh) * OW + ow] = m;
    }
}
""")


LSTM_SEQ = _spec(
    "lstm_seq",
    "void kernel_lstm_seq(const float *x, const float *h0, const float *c0, "
    "const float *w_ih, const float *w_hh, const float *b_gates, float *y, "
    "float *hn, float *cn, int B, int T, int I, int H, int reverse)",
    "One direction of one nn.LSTM layer over a whole sequence (batch-first). "
    "x [B, T, I], h0/c0 [B, H], w_ih [4H, I], w_hh [4H, H], b_gates [4H] "
    "(= b_ih + b_hh), gate order i, f, g, o. Writes y [B, T, H] and the final "
    "state hn/cn [B, H]. reverse=1 walks t = T-1 .. 0 (backward direction).",
    """\
#include <math.h>
void kernel_lstm_seq(const float *x, const float *h0, const float *c0,
                     const float *w_ih, const float *w_hh, const float *b_gates,
                     float *y, float *hn, float *cn,
                     int B, int T, int I, int H, int reverse) {
    for (long k = 0; k < (long)B * H; k++) { hn[k] = h0[k]; cn[k] = c0[k]; }
    for (int s = 0; s < T; s++) {
        int t = reverse ? T - 1 - s : s;
        for (int b = 0; b < B; b++) {
            const float *xt = x + ((long)b * T + t) * I;
            const float *hp = hn + (long)b * H;
            float *yt = y + ((long)b * T + t) * H;
            for (int j = 0; j < H; j++) {
                float g4[4];
                for (int q = 0; q < 4; q++) {
                    long row = (long)q * H + j;
                    float acc = b_gates[row];
                    for (int k = 0; k < I; k++) acc += w_ih[row * I + k] * xt[k];
                    for (int k = 0; k < H; k++) acc += w_hh[row * H + k] * hp[k];
                    g4[q] = acc;
                }
                float ig = 1.0f / (1.0f + expf(-g4[0]));
                float fg = 1.0f / (1.0f + expf(-g4[1]));
                float gg = tanhf(g4[2]);
                float og = 1.0f / (1.0f + expf(-g4[3]));
                float c = fg * cn[(long)b * H + j] + ig * gg;
                cn[(long)b * H + j] = c;
                yt[j] = og * tanhf(c);
            }
        }
        /* h_t for every batch row is only consumed at step t+1 */
        for (int b = 0; b < B; b++)
            for (int j = 0; j < H; j++)
                hn[(long)b * H + j] = y[((long)b * T + t) * H + j];
    }
}
""")

GRU_SEQ = _spec(
    "gru_seq",
    "void kernel_gru_seq(const float *x, const float *h0, const float *w_ih, "
    "const float *w_hh, const float *b_ih, const float *b_hh, float *y, "
    "float *hn, int B, int T, int I, int H, int reverse)",
    "One direction of one nn.GRU layer over a whole sequence (batch-first). "
    "x [B, T, I], h0 [B, H], w_ih [3H, I], w_hh [3H, H], b_ih/b_hh [3H], gate "
    "order r, z, n: r = sig(Wir x + bir + Whr h + bhr), z likewise, "
    "n = tanh(Win x + bin + r * (Whn h + bhn)), h' = (1 - z) * n + z * h. "
    "Writes y [B, T, H] and hn [B, H]; reverse=1 walks t = T-1 .. 0.",
    """\
#include <math.h>
void kernel_gru_seq(const float *x, const float *h0, const float *w_ih,
                    const float *w_hh, const float *b_ih, const float *b_hh,
                    float *y, float *hn, int B, int T, int I, int H, int reverse) {
    for (long k = 0; k < (long)B * H; k++) hn[k] = h0[k];
    for (int s = 0; s < T; s++) {
        int t = reverse ? T - 1 - s : s;
        for (int b = 0; b < B; b++) {
            const float *xt = x + ((long)b * T + t) * I;
            const float *hp = hn + (long)b * H;
            float *yt = y + ((long)b * T + t) * H;
            for (int j = 0; j < H; j++) {
                float gi[3], gh[3];
                for (int q = 0; q < 3; q++) {
                    long row = (long)q * H + j;
                    float ai = b_ih[row], ah = b_hh[row];
                    for (int k = 0; k < I; k++) ai += w_ih[row * I + k] * xt[k];
                    for (int k = 0; k < H; k++) ah += w_hh[row * H + k] * hp[k];
                    gi[q] = ai; gh[q] = ah;
                }
                float r = 1.0f / (1.0f + expf(-(gi[0] + gh[0])));
                float z = 1.0f / (1.0f + expf(-(gi[1] + gh[1])));
                float n = tanhf(gi[2] + r * gh[2]);
                yt[j] = (1.0f - z) * n + z * hp[j];
            }
        }
        for (int b = 0; b < B; b++)
            for (int j = 0; j < H; j++)
                hn[(long)b * H + j] = y[((long)b * T + t) * H + j];
    }
}
""")


# -- runtime-index ops ----------------------------------------------------------
# Index tensors are carried as float (exact below 2**24) like every other
# tensor of the fp32 pipeline. Negative indices wrap like PyTorch's.

INDEX_SELECT = _spec(
    "index_select",
    "void kernel_index_select(const float *input, const float *index, float *output, "
    "int outer, int A, int J, int inner)",
    "Gather slabs along one axis by a runtime index (torch.index_select, "
    "x[idx], embedding): input [outer, A, inner], index [J], output "
    "[outer, J, inner]: output[o, j, i] = input[o, index[j], i] (index < 0 "
    "wraps by +A).",
    """\
void kernel_index_select(const float *input, const float *index, float *output,
                         int outer, int A, int J, int inner) {
    for (long o = 0; o < outer; o++)
        for (long j = 0; j < J; j++) {
            long a = (long)index[j];
            if (a < 0) a += A;
            const float *src = input + (o * A + a) * inner;
            float *dst = output + (o * J + j) * inner;
            for (long i = 0; i < inner; i++) dst[i] = src[i];
        }
}
""")

GATHER_DIM = _spec(
    "gather_dim",
    "void kernel_gather_dim(const float *input, const float *index, float *output, "
    "int outer, int A, int J, int inner)",
    "torch.gather along one axis: input [outer, A, inner], index and output "
    "[outer, J, inner]: output[o, j, i] = input[o, index[o, j, i], i] (index "
    "< 0 wraps by +A).",
    """\
void kernel_gather_dim(const float *input, const float *index, float *output,
                       int outer, int A, int J, int inner) {
    for (long o = 0; o < outer; o++)
        for (long j = 0; j < J; j++)
            for (long i = 0; i < inner; i++) {
                long k = (o * J + j) * inner + i;
                long a = (long)index[k];
                if (a < 0) a += A;
                output[k] = input[(o * A + a) * inner + i];
            }
}
""")

SCATTER_DIM = _spec(
    "scatter_dim",
    "void kernel_scatter_dim(const float *input, const float *index, const float *src, "
    "float *output, int outer, int A, int J, int inner, int mode)",
    "torch.scatter / scatter_add / index_add / index_copy / index_put along one "
    "axis: output = input [outer, A, inner], then for every (o, j, i) of index "
    "and src [outer, J, inner]: output[o, index[o, j, i], i] = src[o, j, i] "
    "(mode 0) or += src[o, j, i] (mode 1). Index < 0 wraps by +A. With mode 0 "
    "and duplicate indices the winner is unspecified, as in PyTorch.",
    """\
void kernel_scatter_dim(const float *input, const float *index, const float *src,
                        float *output, int outer, int A, int J, int inner, int mode) {
    long n = (long)outer * A * inner;
    for (long k = 0; k < n; k++) output[k] = input[k];
    for (long o = 0; o < outer; o++)
        for (long j = 0; j < J; j++)
            for (long i = 0; i < inner; i++) {
                long k = (o * J + j) * inner + i;
                long a = (long)index[k];
                if (a < 0) a += A;
                float *d = output + (o * A + a) * inner + i;
                if (mode) *d += src[k]; else *d = src[k];
            }
}
""")

TOPK = _spec(
    "topk",
    "void kernel_topk(const float *input, float *values, float *indices, int outer, "
    "int A, int inner, int K, int largest)",
    "torch.topk(sorted=True) along the middle axis of [outer, A, inner] "
    "(torch.sort / argsort are K = A): values and indices [outer, K, inner], "
    "largest=1 gives descending order, 0 ascending. NaN counts as larger than "
    "any number; ties keep the lower index first.",
    """\
#include <math.h>
static int mb_topk_before(float a, long ia, float b, long ib, int largest) {
    /* does (a, ia) come before (b, ib) in the output order? */
    int an = isnan(a), bn = isnan(b);
    if (an || bn) {
        if (an && bn) return ia < ib;
        return largest ? an : bn;
    }
    if (a != b) return largest ? a > b : a < b;
    return ia < ib;
}
void kernel_topk(const float *input, float *values, float *indices, int outer,
                 int A, int inner, int K, int largest) {
    for (long o = 0; o < outer; o++)
        for (long i = 0; i < inner; i++) {
            const float *x = input + o * (long)A * inner + i;
            float *v = values + o * (long)K * inner + i;
            float *ix = indices + o * (long)K * inner + i;
            /* insertion into a sorted window of the K best so far */
            long n = 0;
            for (long a = 0; a < A; a++) {
                float xa = x[a * inner];
                if (n == K && !mb_topk_before(xa, a, v[(K - 1) * inner],
                                               (long)ix[(K - 1) * inner], largest))
                    continue;
                long p = n < K ? n++ : K - 1;
                while (p > 0 && mb_topk_before(xa, a, v[(p - 1) * inner],
                                               (long)ix[(p - 1) * inner], largest)) {
                    v[p * inner] = v[(p - 1) * inner];
                    ix[p * inner] = ix[(p - 1) * inner];
                    p--;
                }
                v[p * inner] = xa;
                ix[p * inner] = (float)a;
            }
        }
}
""")

CONVND = _spec(
    "convnd",
    "void kernel_convnd(const float *in0, const float *in1, const float *in2, "
    "float *output, int N, int IC, int ID, int IH, int IW, int OC, int OD, int OH, "
    "int OW, int KD, int KH, int KW, int SD, int SH, int SW, int PD, int PH, int PW, "
    "int DD, int DH, int DW, int G)",
    "Grouped, dilated 1-D/2-D/3-D convolution (F.conv1d/2d/3d) whose weight "
    "and bias are tensors (graph inputs or computed), lower ranks as size-1 "
    "D/H dims: in0 input [N, IC, ID, IH, IW], in1 weight [OC, IC/G, KD, KH, KW], "
    "in2 bias [OC] or NULL, output [N, OC, OD, OH, OW] with zero padding PD/PH/PW: "
    "output[n,oc,od,oh,ow] = bias[oc] + sum over ic<IC/G, kd, kh, kw of "
    "in0[n, g*IC/G + ic, od*SD-PD+kd*DD, oh*SH-PH+kh*DH, ow*SW-PW+kw*DW] * "
    "in1[oc, ic, kd, kh, kw], g = oc / (OC/G).",
    """\
void kernel_convnd(const float *in0, const float *in1, const float *in2,
                   float *output, int N, int IC, int ID, int IH, int IW,
                   int OC, int OD, int OH, int OW, int KD, int KH, int KW,
                   int SD, int SH, int SW, int PD, int PH, int PW,
                   int DD, int DH, int DW, int G) {
    int ICg = IC / G, OCg = OC / G;
    for (long n = 0; n < N; n++)
    for (long oc = 0; oc < OC; oc++) {
        long ic0 = (oc / OCg) * ICg;
        for (long od = 0; od < OD; od++)
        for (long oh = 0; oh < OH; oh++)
        for (long ow = 0; ow < OW; ow++) {
            float acc = in2 ? in2[oc] : 0.0f;
            for (long ic = 0; ic < ICg; ic++)
            for (long kd = 0; kd < KD; kd++) {
                long id = od * SD - PD + kd * DD;
                if (id < 0 || id >= ID) continue;
                for (long kh = 0; kh < KH; kh++) {
                    long ih = oh * SH - PH + kh * DH;
                    if (ih < 0 || ih >= IH) continue;
                    for (long kw = 0; kw < KW; kw++) {
                        long iw = ow * SW - PW + kw * DW;
                        if (iw < 0 || iw >= IW) continue;
                        acc += in0[(((n * IC + ic0 + ic) * ID + id) * IH + ih) * IW + iw]
                             * in1[(((oc * ICg + ic) * KD + kd) * KH + kh) * KW + kw];
                    }
                }
            }
            output[(((n * OC + oc) * OD + od) * OH + oh) * OW + ow] = acc;
        }
    }
}
""")


def ext_specs(kernel_spec_cls) -> dict:
    """KernelSpecs for the ops above. Called by reference_kernels while it
    builds KERNEL_SPECS (taking the class avoids a circular import)."""
    return {op: kernel_spec_cls(op=op, signature=sig, semantics=sem,
                                reference_impl=impl,
                                argtypes_factory=_argtypes_from_signature(sig))
            for op, sig, sem, impl in _RAW}
