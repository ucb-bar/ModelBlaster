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


SPECS = [LUT_S8, GAP_Q, MUL_C1_Q, ADD_Q]
for _s in SPECS:
    if _s.op not in rk.KERNEL_SPECS:
        rk.register_kernel_spec(_s)
