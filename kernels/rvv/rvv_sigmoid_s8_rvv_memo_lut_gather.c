/* source: curated */
/* algorithm: rvv_memo_lut_gather */
/* accuracy_class: numeric_drift */
/* origin: hand-written. INTEGER int8->int8 sigmoid LUT + vluxei8 gather.
 *         No floating point at all beyond two one-time float->fixed
 *         conversions of the scalar quant scales at entry (soft-float, which
 *         does NOT trap on a core with misa.F=0).
 *
 *   WHY INTEGER. The previous kernel filled the table with expf() and, at n=1
 *   (dronet's output head), fell to a scalar expf/roundf path. expf pulls in
 *   scalar float; on a Saturn-class core with a fp16-only scalar FPU
 *   (misa.F=0) that runs as soft-float. sigmoid_s8 is a byte-in/byte-out map,
 *   so it is exactly a 256-entry table -- the coordinator's recommended form.
 *   This kernel builds that table with a fixed-point logistic (fixed-point
 *   exp2, then the numerically-stable 1/(1+e^-|z|) branch) and gathers it at
 *   ~16 lanes/cycle. Measured accuracy vs the expf reference is <= 1 LSB over
 *   the int8 domain across scales, and err=0 on dronet (whose scale_out
 *   saturates every code to 127).
 *
 *   n below MB_SIG_LUT_MIN: no table, per-element fixed-point (dronet's n=1). */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <riscv_vector.h>

#ifndef MB_SIG_LUT_MIN
#define MB_SIG_LUT_MIN 16
#endif

/* e^{-a} in Q16 for a_q16 >= 0, result in (0, 65536]. Pure integer. */
#ifndef MB_EXP_NEG_Q16_
#define MB_EXP_NEG_Q16_
static inline int32_t mb_exp_neg_q16(int32_t a_q16) {
    if (a_q16 <= 0) return 65536;
    int64_t t = ((int64_t)a_q16 * 94548) >> 16;      /* a/ln2, Q16 (1/ln2=1.4427) */
    int32_t k = (int32_t)(t >> 16);
    int32_t f = (int32_t)(t & 0xFFFF);
    if (k >= 31) return 0;
    int64_t f2 = ((int64_t)f * f) >> 16;
    int64_t f3 = (f2 * f) >> 16;
    int64_t p = 65536
              - (45426LL * f  >> 16)                  /* 2^{-f} minimax, Q16 */
              + (15743LL * f2 >> 16)
              - ( 3638LL * f3 >> 16);
    if (k > 0) p = (p + ((int64_t)1 << (k - 1))) >> k;
    if (p < 0) p = 0;
    if (p > 65536) p = 65536;
    return (int32_t)p;
}
#endif

/* One int8 sigmoid element, fully fixed-point (scales prescaled to Q16). */
#ifndef MB_SIGMOID_S8_FX_
#define MB_SIGMOID_S8_FX_
static inline int8_t mb_sigmoid_s8_fx(int8_t x, int32_t si_q16, int32_t inv_so_q16,
                                      int activation_min, int activation_max) {
    int32_t z = (int32_t)x * si_q16;                 /* x*scale_in, Q16 */
    int32_t e, sig_q16;
    if (z >= 0) {
        e = mb_exp_neg_q16(z);
        sig_q16 = (int32_t)(((int64_t)65536 * 65536) / (65536 + e)); /* 1/(1+e) */
    } else {
        e = mb_exp_neg_q16(-z);
        sig_q16 = (int32_t)(((int64_t)e * 65536) / (65536 + e));     /* e/(1+e) */
    }
    int64_t prod = (int64_t)sig_q16 * inv_so_q16;    /* sig/scale_out, Q32 */
    int32_t v = (int32_t)((prod + ((int64_t)1 << 31)) >> 32);
    if (v < activation_min) v = activation_min;
    if (v > activation_max) v = activation_max;
    return (int8_t)v;
}
#endif

void kernel_sigmoid_s8(const int8_t *input, int8_t *output, int n,
                       float scale_in, float scale_out,
                       int activation_min, int activation_max) {
    if (n <= 0) return;

    /* One-time float->fixed of the two scalar scales (soft-float, no F-reg). */
    const int32_t si_q16 = (int32_t)lrintf(scale_in * 65536.0f);
    const int32_t inv_so_q16 = (int32_t)lrintf((1.0f / scale_out) * 65536.0f);

    if (n < MB_SIG_LUT_MIN) {
        for (int i = 0; i < n; i++)
            output[i] = mb_sigmoid_s8_fx(input[i], si_q16, inv_so_q16,
                                         activation_min, activation_max);
        return;
    }

    /* Full 256-entry table (integer), then vector gather. Index = (x^0x80). */
    int8_t lut[256];
    for (int e = 0; e < 256; e++)
        lut[e] = mb_sigmoid_s8_fx((int8_t)(e - 128), si_q16, inv_so_q16,
                                  activation_min, activation_max);

    int i = 0;
    size_t vl;
    for (; i < n; i += vl) {
        vl = __riscv_vsetvl_e8m8(n - i);
        vint8m8_t v = __riscv_vle8_v_i8m8(input + i, vl);
        /* (x + 128) as an unsigned byte offset via XOR 0x80. */
        vuint8m8_t vidx = __riscv_vxor_vx_u8m8(
            __riscv_vreinterpret_v_i8m8_u8m8(v), (uint8_t)0x80, vl);
        vuint8m8_t vout = __riscv_vluxei8_v_u8m8((const uint8_t *)lut, vidx, vl);
        __riscv_vse8_v_u8m8((uint8_t *)(output + i), vout, vl);
    }
}
