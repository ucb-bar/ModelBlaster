"""Integer-closed op kinds: every scale is an integer multiplier/shift or a table, fixed at build time.

These kinds carry no float semantics at run time. Their arithmetic is one integer contract, shared by every kind
here and chosen to be implementable bit-exactly on a CPU, a vector unit and a fixed-point engine:

  * Tensors are symmetric (zero point 0). A tensor's width is given in bits: 8, 15 or 16 (15-bit codes are held in
    int16), and 64 for raw accumulators. sat_b clamps to the signed range, [-16384, 16383] for 15 bits.
  * >> is an arithmetic shift (floor). Rounding is half up: (v + 2^(n-1)) >> n.
  * requant(t, M, s): if s < 0, t <<= -s (a pre-shift, for multipliers >= 1); q = (t*M + 2^30) >> 31 computed
    exactly (t*M can need 79 bits); if s > 0, q = (q + 2^(s-1)) >> s. M is in [2^30, 2^31) or 0.
  * add: y = sat_b((a*Ma + b*Mb + 2^(S-1)) >> S), int64.

Build-time derivation (multipliers from real ratios, tables from the float functions) lives in integer_close.py; a
qparams file can replace any constant verbatim (apply_qparams.py). Each kind's reference_impl is the specification.

Kinds:
  lut_s8          y = T[x + 128]                        int8 -> int8, one 256-entry table per site
  gap_q           y = sat(requant(sum_HW x, M, s))      global average pool, int8/int16 in and out
  mul_c1_q        y = sat(requant(x * g[n,c], M, s))    per-channel gate (squeeze-excite scale)
  add_q           y = sat((a*Ma[row] + b*Mb + r) >> S)  residual / positional-encoding add, per-row Ma
"""
from __future__ import annotations

from . import reference_kernels as rk

# Shared C helpers, emitted once per translation unit (each kernel's source carries the guarded block).
HELPERS = r"""
#ifndef MB_INTEGER_HELPERS
#define MB_INTEGER_HELPERS
#include <stdint.h>
static inline int64_t mbi_ld(const void *p, int bits, long i) {
    if (bits == 8) return ((const int8_t *)p)[i];
    if (bits == 15 || bits == 16) return ((const int16_t *)p)[i];
    if (bits == 32) return ((const int32_t *)p)[i];
    return ((const int64_t *)p)[i];
}
static inline int64_t mbi_sat(int64_t v, int bits) {
    if (bits >= 64) return v;
    int64_t hi = bits == 15 ? 16383 : (((int64_t)1 << (bits - 1)) - 1);
    int64_t lo = bits == 15 ? -16384 : -((int64_t)1 << (bits - 1));
    return v < lo ? lo : (v > hi ? hi : v);
}
static inline void mbi_st(void *p, int bits, long i, int64_t v) {
    if (bits == 8) ((int8_t *)p)[i] = (int8_t)v;
    else if (bits == 15 || bits == 16) ((int16_t *)p)[i] = (int16_t)v;
    else if (bits == 32) ((int32_t *)p)[i] = (int32_t)v;
    else ((int64_t *)p)[i] = v;
}
/* requant(t, M, s), exact: t*M is formed in 128 bits. A negative t is pre-shifted by multiplying (a left shift of a
 * negative value is undefined in C). */
static inline int64_t mbi_requant(int64_t t, int32_t M, int32_t s) {
    if (s < 0) t = t * ((int64_t)1 << (-s));
    __int128 p = (__int128)t * (__int128)M + ((__int128)1 << 30);
    int64_t q = (int64_t)(p >> 31);
    if (s > 0) q = (q + ((int64_t)1 << (s - 1))) >> s;
    return q;
}
#endif
"""


def _src(body: str) -> str:
    return HELPERS + body


def _bits(meta_dtype: str) -> int:
    return {"i8": 8, "i16": 16, "i32": 32, "i64": 64}[meta_dtype]


def _tbits(ctx, op, tensor):
    """Width of a tensor: the op's explicit bits (15-bit tensors) or its dtype."""
    q = op.get("quant", {})
    explicit = q.get("bits", {}).get(tensor) if isinstance(q.get("bits"), dict) else None
    if explicit is not None:
        return int(explicit)
    return _bits(ctx.tensors[tensor]["dtype"])


# ----------------------------------------------------------------------------------------------------------- lut_s8
LUT_S8 = rk.KernelSpec(
    op="lut_s8",
    signature="void kernel_lut_s8(const int8_t *input, int8_t *output, int n, const int8_t *table)",
    semantics="output[i] = table[input[i] + 128]. One 256-entry table per site, built at compile time as "
              "T[i] = sat8(rha(f((i - 128) * s_in) / s_out)) for the site's function f (sigmoid, SiLU, GELU, ...).",
    reference_impl=_src(r"""
void kernel_lut_s8(const int8_t *input, int8_t *output, int n, const int8_t *table) {
    for (int i = 0; i < n; i++) output[i] = table[(int)input[i] + 128];
}
"""),
    extra_shapes=[{"n": 256}],
    emit_call=lambda op, ctx: (
        f"kernel_lut_s8({ctx.ptr_for(op['inputs'][0], 'in')}, {ctx.out_ptr}, {op['shape']['n']}, "
        f"{ctx.weight_name(op['quant']['table_key'])})", []),
)


# ------------------------------------------------------------------------------------------------------------ gap_q
def _emit_gap_q(op, ctx):
    sh, q = op["shape"], op["quant"]
    i, o = op["inputs"][0], op["outputs"][0]
    return (f"kernel_gap_q({ctx.ptr_for(i, 'in')}, {_tbits(ctx, op, i)}, {ctx.out_ptr}, {_tbits(ctx, op, o)}, "
            f"{sh['N']}, {sh['C']}, {sh['HW']}, {int(q['M'])}, {int(q['s'])})", [])


GAP_Q = rk.KernelSpec(
    op="gap_q",
    signature="void kernel_gap_q(const void *input, int in_bits, void *output, int out_bits, "
              "int N, int C, int HW, int32_t M, int32_t s)",
    semantics="Global average pool over HW (NCHW): out[n,c] = sat_out(requant(sum_hw in[n,c,hw], M, s)). 1/(H*W) and "
              "the scale ratio are folded into (M, s); s may be negative (pre-shift).",
    reference_impl=_src(r"""
void kernel_gap_q(const void *input, int in_bits, void *output, int out_bits,
                  int N, int C, int HW, int32_t M, int32_t s) {
    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++) {
            int64_t acc = 0;
            long base = ((long)n * C + c) * HW;
            for (int k = 0; k < HW; k++) acc += mbi_ld(input, in_bits, base + k);
            mbi_st(output, out_bits, (long)n * C + c, mbi_sat(mbi_requant(acc, M, s), out_bits));
        }
}
"""),
    emit_call=_emit_gap_q,
)


# --------------------------------------------------------------------------------------------------------- mul_c1_q
def _emit_mul_c1_q(op, ctx):
    sh, q = op["shape"], op["quant"]
    g, x, o = op["inputs"][0], op["inputs"][1], op["outputs"][0]
    return (f"kernel_mul_c1_q({ctx.ptr_for(x, 'in')}, {_tbits(ctx, op, x)}, {ctx.ptr_for(g, 'in')}, "
            f"{_tbits(ctx, op, g)}, {ctx.out_ptr}, {_tbits(ctx, op, o)}, {sh['N']}, {sh['C']}, {sh['HW']}, "
            f"{int(q['M'])}, {int(q['s'])})", [])


MUL_C1_Q = rk.KernelSpec(
    op="mul_c1_q",
    signature="void kernel_mul_c1_q(const void *x, int x_bits, const void *gate, int g_bits, void *output, "
              "int out_bits, int N, int C, int HW, int32_t M, int32_t s)",
    semantics="Per-channel gate (NCHW): out[n,c,hw] = sat_out(requant(x[n,c,hw] * gate[n,c], M, s)). IR inputs are "
              "(gate, x), as mul_c1_s8's.",
    reference_impl=_src(r"""
void kernel_mul_c1_q(const void *x, int x_bits, const void *gate, int g_bits, void *output, int out_bits,
                     int N, int C, int HW, int32_t M, int32_t s) {
    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++) {
            int64_t g = mbi_ld(gate, g_bits, (long)n * C + c);
            long base = ((long)n * C + c) * HW;
            for (int k = 0; k < HW; k++)
                mbi_st(output, out_bits, base + k,
                       mbi_sat(mbi_requant(mbi_ld(x, x_bits, base + k) * g, M, s), out_bits));
        }
}
"""),
    emit_call=_emit_mul_c1_q,
)


# ------------------------------------------------------------------------------------------------------------ add_q
def _emit_add_q(op, ctx):
    sh, q = op["shape"], op["quant"]
    a, b, o = op["inputs"][0], op["inputs"][1], op["outputs"][0]
    return (f"kernel_add_q({ctx.ptr_for(a, 'in')}, {_tbits(ctx, op, a)}, {ctx.ptr_for(b, 'in')}, "
            f"{_tbits(ctx, op, b)}, {ctx.out_ptr}, {_tbits(ctx, op, o)}, {sh['rows']}, {sh['cols']}, "
            f"{ctx.weight_name(q['ma_key'])}, {sh['n_ma']}, {int(q['mb'])}, {sh['b_rows']}, {int(q['S'])})", [])


ADD_Q = rk.KernelSpec(
    op="add_q",
    signature="void kernel_add_q(const void *a, int a_bits, const void *b, int b_bits, void *output, int out_bits, "
              "int rows, int cols, const int32_t *ma, int n_ma, int32_t mb, int b_rows, int S)",
    semantics="out[r,c] = sat_out((a[r,c]*ma[r % n_ma] + b[r % b_rows, c]*mb + 2^(S-1)) >> S), int64. n_ma = 1 is "
              "a plain residual add; n_ma > 1 gives a per-row a multiplier (tokens with different scales); "
              "b_rows < rows broadcasts b (a constant table repeated per batch).",
    reference_impl=_src(r"""
void kernel_add_q(const void *a, int a_bits, const void *b, int b_bits, void *output, int out_bits,
                  int rows, int cols, const int32_t *ma, int n_ma, int32_t mb, int b_rows, int S) {
    for (int r = 0; r < rows; r++) {
        int64_t m = ma[r % n_ma];
        long bb = (long)(r % b_rows) * cols;
        for (int c = 0; c < cols; c++) {
            int64_t v = mbi_ld(a, a_bits, (long)r * cols + c) * m + mbi_ld(b, b_bits, bb + c) * (int64_t)mb
                        + ((int64_t)1 << (S - 1));
            mbi_st(output, out_bits, (long)r * cols + c, mbi_sat(v >> S, out_bits));
        }
    }
}
"""),
    emit_call=_emit_add_q,
)


# ---------------------------------------------------------------------------------------------------------- conv2d_q
def _emit_conv2d_q(op, ctx):
    sh, q = op["shape"], op["quant"]
    i, o = op["inputs"][0], op["outputs"][0]
    b = ctx.weight_name(op["bias"]) if op.get("bias") else "NULL"
    return (f"kernel_conv2d_q({ctx.ptr_for(i, 'in')}, {_tbits(ctx, op, i)}, {ctx.weight_name(op['weight'])}, {b}, "
            f"{ctx.out_ptr}, {_tbits(ctx, op, o)}, {sh['N']}, {sh['IC']}, {sh['IH']}, {sh['IW']}, {sh['OC']}, "
            f"{sh['KH']}, {sh['KW']}, {sh['SH']}, {sh['SW']}, {sh['pad_l']}, {sh['pad_r']}, {sh['pad_t']}, "
            f"{sh['pad_b']}, {sh['groups']}, {ctx.weight_name(q['output_multiplier_per_oc_key'])}, "
            f"{ctx.weight_name(q['output_shift_per_oc_key'])})", [])


CONV2D_Q = rk.KernelSpec(
    op="conv2d_q",
    signature="void kernel_conv2d_q(const void *input, int in_bits, const int8_t *weight, const int32_t *bias, "
              "void *output, int out_bits, int N, int IC, int IH, int IW, int OC, int KH, int KW, int SH, int SW, "
              "int pad_l, int pad_r, int pad_t, int pad_b, int groups, const int32_t *M, const int32_t *s)",
    semantics="NCHW conv (groups = 1) or depthwise (groups = IC = OC), weights OIHW (I = IC/groups). Asymmetric "
              "static pads are zero taps. acc = bias[oc] + sum x*w in int64; out = sat_out(requant(acc, M[oc], "
              "s[oc])). OH = (IH + pad_t + pad_b - KH)/SH + 1, likewise OW.",
    reference_impl=_src(r"""
void kernel_conv2d_q(const void *input, int in_bits, const int8_t *weight, const int32_t *bias,
                     void *output, int out_bits, int N, int IC, int IH, int IW, int OC, int KH, int KW, int SH, int SW,
                     int pad_l, int pad_r, int pad_t, int pad_b, int groups, const int32_t *M, const int32_t *s) {
    int OH = (IH + pad_t + pad_b - KH) / SH + 1;
    int OW = (IW + pad_l + pad_r - KW) / SW + 1;
    int icg = IC / groups, ocg = OC / groups;
    for (int n = 0; n < N; n++)
        for (int oc = 0; oc < OC; oc++) {
            int g = oc / ocg;
            for (int oh = 0; oh < OH; oh++)
                for (int ow = 0; ow < OW; ow++) {
                    int64_t acc = bias ? bias[oc] : 0;
                    for (int ic = 0; ic < icg; ic++) {
                        int c = g * icg + ic;
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - pad_t + kh;
                            if (ih < 0 || ih >= IH) continue;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = ow * SW - pad_l + kw;
                                if (iw < 0 || iw >= IW) continue;
                                acc += mbi_ld(input, in_bits, (((long)n * IC + c) * IH + ih) * IW + iw)
                                       * (int64_t)weight[(((long)oc * icg + ic) * KH + kh) * KW + kw];
                            }
                        }
                    }
                    mbi_st(output, out_bits, (((long)n * OC + oc) * OH + oh) * OW + ow,
                           mbi_sat(mbi_requant(acc, M[oc], s[oc]), out_bits));
                }
        }
}
"""),
    emit_call=_emit_conv2d_q,
)


# ---------------------------------------------------------------------------------------------------------- linear_q
def _emit_linear_q(op, ctx):
    sh, q = op["shape"], op["quant"]
    i, o = op["inputs"][0], op["outputs"][0]
    b = ctx.weight_name(op["bias"]) if op.get("bias") else "NULL"
    ob = _tbits(ctx, op, o)
    M = ctx.weight_name(q["output_multiplier_per_oc_key"]) if ob < 64 else "NULL"
    S = ctx.weight_name(q["output_shift_per_oc_key"]) if ob < 64 else "NULL"
    return (f"kernel_linear_q({ctx.ptr_for(i, 'in')}, {_tbits(ctx, op, i)}, {ctx.weight_name(op['weight'])}, {b}, "
            f"{ctx.out_ptr}, {ob}, {sh['M']}, {sh['K']}, {sh['N']}, {M}, {S}, {int(q.get('relu', 0))})", [])


LINEAR_Q = rk.KernelSpec(
    op="linear_q",
    signature="void kernel_linear_q(const void *input, int in_bits, const int8_t *weight, const int32_t *bias, "
              "void *output, int out_bits, int M, int K, int N, const int32_t *mult, const int32_t *shift, int relu)",
    semantics="out[m,n] = sat_out(requant(bias[n] + sum_k x[m,k]*w[n,k], mult[n], shift[n])) in int64, then "
              "max(., 0) if relu. out_bits = 64 writes the raw accumulator (no requant; mult/shift unused).",
    reference_impl=_src(r"""
void kernel_linear_q(const void *input, int in_bits, const int8_t *weight, const int32_t *bias,
                     void *output, int out_bits, int M, int K, int N, const int32_t *mult, const int32_t *shift,
                     int relu) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int64_t acc = bias ? bias[n] : 0;
            for (int k = 0; k < K; k++)
                acc += mbi_ld(input, in_bits, (long)m * K + k) * (int64_t)weight[(long)n * K + k];
            if (out_bits < 64) {
                acc = mbi_sat(mbi_requant(acc, mult[n], shift[n]), out_bits);
                if (relu && acc < 0) acc = 0;
            }
            mbi_st(output, out_bits, (long)m * N + n, acc);
        }
}
"""),
    emit_call=_emit_linear_q,
)


# ---------------------------------------------------------------------------------------------------------- gate_q16
GATE_KINDS = {"silu": 0, "sigmoid": 1, "gelu": 2}


def _emit_gate_q16(op, ctx):
    q = op["quant"]
    return (f"kernel_gate_q16({ctx.ptr_for(op['inputs'][0], 'in')}, {ctx.out_ptr}, {op['shape']['n']}, "
            f"{GATE_KINDS[q['kind']]}, {int(q['Mx'])}, {int(q['sx'])}, {int(q.get('M2', 0))}, "
            f"{int(q.get('s2', 0))}, {ctx.weight_name(q['table_key'])})", [])


GATE_Q16 = rk.KernelSpec(
    op="gate_q16",
    signature="void kernel_gate_q16(const int16_t *input, int16_t *output, int n, int kind, int32_t Mx, int32_t sx, "
              "int32_t M2, int32_t s2, const int32_t *table)",
    semantics="int16-input activation through a shared 65-entry Q0.15 table. X = requant(x, Mx, sx) is x on the Q.11 "
              "grid; R = 8 (sigmoid table: kind silu or sigmoid) or 4 (Phi table: kind gelu); L = R*2^11, w = 2L/64. "
              "g = 0 for X < -L, 32767 for X >= L, else T[k] + (((T[k+1]-T[k])*r + w/2) >> log2 w) with u = X + L, "
              "k = u >> log2 w, r = u & (w-1). sigmoid: out = g (Q0.15). silu/gelu: out = sat16(requant(x*g, M2, s2)).",
    reference_impl=_src(r"""
void kernel_gate_q16(const int16_t *input, int16_t *output, int n, int kind, int32_t Mx, int32_t sx,
                     int32_t M2, int32_t s2, const int32_t *table) {
    int R = kind == 2 ? 4 : 8;
    int64_t L = (int64_t)R << 11;
    int lw = R == 8 ? 9 : 8;
    int64_t w = (int64_t)1 << lw;
    for (int i = 0; i < n; i++) {
        int64_t x = input[i];
        int64_t X = mbi_requant(x, Mx, sx), g;
        if (X < -L) g = 0;
        else if (X >= L) g = 32767;
        else {
            int64_t u = X + L, k = u >> lw, r = u & (w - 1);
            g = table[k] + (((int64_t)(table[k + 1] - table[k]) * r + (w >> 1)) >> lw);
        }
        output[i] = (int16_t)(kind == 1 ? g : mbi_sat(mbi_requant(x * g, M2, s2), 16));
    }
}
"""),
    emit_call=_emit_gate_q16,
)


# ----------------------------------------------------------------------------------------------------- layer_norm_q16
def _emit_layer_norm_q16(op, ctx):
    sh, q = op["shape"], op["quant"]
    return (f"kernel_layer_norm_q16({ctx.ptr_for(op['inputs'][0], 'in')}, {ctx.out_ptr}, {sh['M']}, {sh['K']}, "
            f"{int(q['E'])}LL, {ctx.weight_name(q['gamma_key'])}, {ctx.weight_name(q['beta_key'])}, {int(q['M'])}, "
            f"{int(q['s'])}, {ctx.weight_name(q['rsqrt_key'])})", [])


LAYER_NORM_Q16 = rk.KernelSpec(
    op="layer_norm_q16",
    signature="void kernel_layer_norm_q16(const int16_t *input, int16_t *output, int rows, int D, int64_t E, "
              "const int32_t *gam, const int64_t *bet, int32_t M, int32_t s, const int32_t *rsqrt)",
    semantics="Per row: S1 = sum x, S2 = sum x^2, V = D*S2 - S1^2 + E (int64). rsqrt from a 49-entry table over "
              "[1, 4] (segment 1/16): e = (bitlen(V)-1)>>1, mq = V << (16-2e) or V >> (2e-16) (floor), u = mq - 2^16, "
              "k = u >> 12, r = u & 4095, rq = T[k] - (((T[k]-T[k+1])*r + 2048) >> 12). z = D*x - S1; "
              "n13 = (z*rq + 2^(e+1)) >> (e+2); acc = n13*gam[c] + bet[c]; out = sat16(requant(acc, M, s)).",
    reference_impl=_src(r"""
void kernel_layer_norm_q16(const int16_t *input, int16_t *output, int rows, int D, int64_t E,
                           const int32_t *gam, const int64_t *bet, int32_t M, int32_t s, const int32_t *rsqrt) {
    for (int r = 0; r < rows; r++) {
        const int16_t *x = input + (long)r * D;
        int64_t S1 = 0, S2 = 0;
        for (int c = 0; c < D; c++) { S1 += x[c]; S2 += (int64_t)x[c] * x[c]; }
        int64_t V = (int64_t)D * S2 - S1 * S1 + E;
        int L = 0;
        for (uint64_t v = (uint64_t)V; v; v >>= 1) L++;
        int e = (L - 1) >> 1, sh = 16 - 2 * e;
        int64_t mq = sh >= 0 ? V * ((int64_t)1 << sh) : V >> (-sh);
        int64_t u = mq - ((int64_t)1 << 16), k = u >> 12, rr = u & 4095;
        int64_t rq = rsqrt[k] - ((((int64_t)rsqrt[k] - rsqrt[k + 1]) * rr + 2048) >> 12);
        for (int c = 0; c < D; c++) {
            int64_t z = (int64_t)D * x[c] - S1;
            int64_t n13 = (z * rq + ((int64_t)1 << (e + 1))) >> (e + 2);
            int64_t acc = n13 * gam[c] + bet[c];
            output[(long)r * D + c] = (int16_t)mbi_sat(mbi_requant(acc, M, s), 16);
        }
    }
}
"""),
    emit_call=_emit_layer_norm_q16,
)


# ------------------------------------------------------------------------------------------------------ matmul_b_q_s8
def _emit_matmul_b_q_s8(op, ctx):
    sh, q = op["shape"], op["quant"]
    return (f"kernel_matmul_b_q_s8({ctx.ptr_for(op['inputs'][0], 'in')}, {ctx.ptr_for(op['inputs'][1], 'in')}, "
            f"{ctx.out_ptr}, {sh['B']}, {sh['M']}, {sh['K']}, {sh['N']}, {int(q.get('transpose_b', 0))}, "
            f"{int(q['M_q'])}, {int(q['s_q'])})", [])


MATMUL_B_Q_S8 = rk.KernelSpec(
    op="matmul_b_q_s8",
    signature="void kernel_matmul_b_q_s8(const int8_t *a, const int8_t *b, int8_t *out, int B, int M, int K, int N, "
              "int transpose_b, int32_t Mq, int32_t sq)",
    semantics="Batched int8 x int8 activation product: out[b,m,n] = sat8(requant(sum_k a[b,m,k] * B[b,k,n], Mq, sq)), "
              "with B[b,k,n] = b[b,n,k] if transpose_b else b[b,k,n]. The sum is exact in int32.",
    reference_impl=_src(r"""
void kernel_matmul_b_q_s8(const int8_t *a, const int8_t *b, int8_t *out, int B, int M, int K, int N,
                          int transpose_b, int32_t Mq, int32_t sq) {
    for (int bb = 0; bb < B; bb++)
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                int64_t acc = 0;
                for (int k = 0; k < K; k++) {
                    int64_t bv = transpose_b ? b[((long)bb * N + n) * K + k] : b[((long)bb * K + k) * N + n];
                    acc += (int64_t)a[((long)bb * M + m) * K + k] * bv;
                }
                out[((long)bb * M + m) * N + n] = (int8_t)mbi_sat(mbi_requant(acc, Mq, sq), 8);
            }
}
"""),
    emit_call=_emit_matmul_b_q_s8,
)


# ------------------------------------------------------------------------------------------------------ softmax_q_s8
SOFTMAX_Q_S8 = rk.KernelSpec(
    op="softmax_q_s8",
    signature="void kernel_softmax_q_s8(const int8_t *input, int8_t *output, int rows, int L, const int32_t *texp)",
    semantics="Per row of L int8 scores: d = max - q (0..255), e = texp[d] (16-bit entries, a per-site table "
              "min(65535, rha(2^16 exp(-d s_in)))), S = sum e, R = floor((127*2^31 + floor(S/2)) / S), "
              "p = clamp((e*R + 2^30) >> 31, 0, 127) at scale 1/127.",
    reference_impl=_src(r"""
void kernel_softmax_q_s8(const int8_t *input, int8_t *output, int rows, int L, const int32_t *texp) {
    for (int r = 0; r < rows; r++) {
        const int8_t *q = input + (long)r * L;
        int mx = q[0];
        for (int i = 1; i < L; i++) if (q[i] > mx) mx = q[i];
        int64_t S = 0;
        for (int i = 0; i < L; i++) S += texp[mx - q[i]];
        int64_t R = (((int64_t)127 << 31) + (S >> 1)) / S;
        for (int i = 0; i < L; i++) {
            int64_t p = ((int64_t)texp[mx - q[i]] * R + ((int64_t)1 << 30)) >> 31;
            output[(long)r * L + i] = (int8_t)(p < 0 ? 0 : (p > 127 ? 127 : p));
        }
    }
}
"""),
    emit_call=lambda op, ctx: (
        f"kernel_softmax_q_s8({ctx.ptr_for(op['inputs'][0], 'in')}, {ctx.out_ptr}, {op['shape']['M']}, "
        f"{op['shape']['K']}, {ctx.weight_name(op['quant']['table_key'])})", []),
)


# ---------------------------------------------------------------------------------------------------------- cat2_c1_q
def _emit_cat2_c1_q(op, ctx):
    sh = op["shape"]
    a, b, o = op["inputs"][0], op["inputs"][1], op["outputs"][0]
    ca, cb = sh["C_inputs"]
    return (f"kernel_cat2_c1_q({ctx.ptr_for(a, 'in')}, {_tbits(ctx, op, a)}, {ca}, {ctx.ptr_for(b, 'in')}, "
            f"{_tbits(ctx, op, b)}, {cb}, {ctx.out_ptr}, {_tbits(ctx, op, o)}, {sh['N']}, {sh['H'] * sh['W']})", [])


CAT2_C1_Q = rk.KernelSpec(
    op="cat2_c1_q",
    signature="void kernel_cat2_c1_q(const void *a, int a_bits, int Ca, const void *b, int b_bits, int Cb, "
              "void *output, int out_bits, int N, int HW)",
    semantics="Channel concat (N, Ca + Cb, HW) of (N, Ca, HW) and (N, Cb, HW) as pure placement: codes are copied, "
              "widened if the output is wider, never requantized (the inputs keep their own scales; the consumer "
              "accounts for them, e.g. a per-row add).",
    reference_impl=_src(r"""
void kernel_cat2_c1_q(const void *a, int a_bits, int Ca, const void *b, int b_bits, int Cb,
                      void *output, int out_bits, int N, int HW) {
    int C = Ca + Cb;
    for (int n = 0; n < N; n++) {
        for (long i = 0; i < (long)Ca * HW; i++)
            mbi_st(output, out_bits, (long)n * C * HW + i, mbi_ld(a, a_bits, (long)n * Ca * HW + i));
        for (long i = 0; i < (long)Cb * HW; i++)
            mbi_st(output, out_bits, ((long)n * C + Ca) * HW + i, mbi_ld(b, b_bits, (long)n * Cb * HW + i));
    }
}
"""),
    emit_call=_emit_cat2_c1_q,
)


# ------------------------------------------------------------------------------------------- vint_action_post_acc
VINT_ACTION_POST_ACC = rk.KernelSpec(
    op="vint_action_post_acc",
    signature="void kernel_vint_action_post_acc(const int64_t *dist_acc, const double *dist_scale, "
              "const int64_t *act_acc, const double *act_scale, float *out)",
    semantics="ViNT's CPU tail on raw accumulators, in double: dist = acc*scale; the 5x(dx, dy, cos, sin) actions = acc*scale "
              "per output; out = [dist, 5 x (cumsum dx, cumsum dy, (cos, sin)/max(norm, 1e-12))]. The only float "
              "op left in an integer-closed ViNT graph, by design (the tail runs on the CPU).",
    reference_impl=r"""
#include <math.h>
void kernel_vint_action_post_acc(const int64_t *dist_acc, const double *dist_scale,
                                 const int64_t *act_acc, const double *act_scale, float *out) {
    out[0] = (float)((double)dist_acc[0] * dist_scale[0]);
    double x = 0.0, y = 0.0;
    for (int i = 0; i < 5; i++) {
        double d[4];
        for (int j = 0; j < 4; j++) d[j] = (double)act_acc[i * 4 + j] * act_scale[i * 4 + j];
        x += d[0]; y += d[1];
        double nrm = sqrt(d[2] * d[2] + d[3] * d[3]);
        if (nrm < 1e-12) nrm = 1e-12;
        out[1 + i * 4 + 0] = (float)x; out[1 + i * 4 + 1] = (float)y;
        out[1 + i * 4 + 2] = (float)(d[2] / nrm); out[1 + i * 4 + 3] = (float)(d[3] / nrm);
    }
}
""",
    emit_call=lambda op, ctx: (
        f"kernel_vint_action_post_acc({ctx.ptr_for(op['inputs'][0], 'in')}, "
        f"{ctx.weight_name(op['quant']['dist_scale_key'])}, {ctx.ptr_for(op['inputs'][1], 'in')}, "
        f"{ctx.weight_name(op['quant']['act_scale_key'])}, (float *){ctx.out_ptr})", []),
)


# ------------------------------------------------------------------------------------------------------------ copy_q
def _emit_copy_q(op, ctx):
    i, o = op["inputs"][0], op["outputs"][0]
    return (f"kernel_copy_q({ctx.ptr_for(i, 'in')}, {ctx.ptr_for(o, 'in')}, {op['shape']['n']}, "
            f"{_tbits(ctx, op, i)})", [])


COPY_Q = rk.KernelSpec(
    op="copy_q",
    signature="void kernel_copy_q(const void *src, void *dst, int n, int bits)",
    semantics="dst[i] = src[i] for n codes of the given width. The state update of a stateful graph: the last op "
              "writes the next value of a state tensor into the state's buffer, after everything that read the old "
              "value has run.",
    reference_impl=_src(r"""
void kernel_copy_q(const void *src, void *dst, int n, int bits) {
    for (long i = 0; i < n; i++) mbi_st(dst, bits, i, mbi_ld(src, bits, i));
}
"""),
    emit_call=_emit_copy_q,
)


SPECS = [COPY_Q, LUT_S8, GAP_Q, MUL_C1_Q, ADD_Q, CONV2D_Q, LINEAR_Q, GATE_Q16, LAYER_NORM_Q16, MATMUL_B_Q_S8,
         SOFTMAX_Q_S8, CAT2_C1_Q, VINT_ACTION_POST_ACC]
for _s in SPECS:
    if _s.op not in rk.KERNEL_SPECS:
        rk.register_kernel_spec(_s)
