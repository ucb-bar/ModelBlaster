/* source: curated */
/* algorithm: soft_f16 */
/* accuracy_class: bit_exact */
/* origin: hand-written. sigmoid_f16 with the conversions inlined; see
 *         gemmini_q31_conv2d_f16_soft_f16.c.
 *
 *   HONEST CEILING. Unlike the int8 activations, this one keeps its expf:
 *   the input is a full fp16 tensor, so the 256-entry memoized table that
 *   makes sigmoid_s8 50x faster has no analogue (a half-precision table is
 *   65536 entries and would cost 65536 expf to fill). What is removed is the
 *   load conversion and the store conversion only -- two libgcc calls out of
 *   two calls plus one expf. Expect a fraction, not a multiple.
 *
 *   Measured baseline, spike, vint, gemmini_q31, 65 dispatches:
 *   117,025,748 cycles, 1.49%. */
/* Inline IEEE-754 binary16 <-> binary32 conversion.
 *
 * WHY THIS EXISTS AND WHY IT IS THE WHOLE KERNEL. The Gemmini targets
 * compile kernels at -march=rv64imafdc (backends.GEMMINI_Q31) -- no `v`, and
 * critically no Zfh. So `(float)x` on a _Float16 is not a machine
 * instruction, it is `call __extendhfsf2`, and `(_Float16)y` is
 * `call __truncsfhf2`: libgcc's generic soft-fp unpack/round/pack routines,
 * reached through a real function call with the register saves that implies.
 * Verified in the emitted assembly for this toolchain (Zephyr SDK
 * 1.0.0-beta1, gcc 14.3.0): a one-line `o[i] = (_Float16)((float)a[i] *
 * (float)b[i])` compiles to three calls and one fmul.
 *
 * That is why ViNT's fp16 half costs what it does on this target: the fp16
 * ops are not slow because fp16 is slow, they are slow because every
 * OPERAND TOUCH is a function call. conv2d_f16 alone measured
 * 5,146,370,253 spike cycles -- 65.7% of the entire 7.83-billion-cycle
 * model -- for two conversion calls per multiply-accumulate.
 *
 * Both directions here are the standard branchless bit-twiddle forms. They
 * are EXACT, not approximations, and were validated exhaustively against the
 * toolchain's own conversion before use:
 *   h2f  all 65536 half bit patterns, 0 mismatches
 *   f2h  17,044,582 float32 bit patterns swept across the whole space,
 *        0 mismatches, plus all 63,488 exact half-step midpoints (the
 *        round-to-nearest-EVEN tie cases, which is where a hand-written
 *        converter actually goes wrong), 0 mismatches
 *   round trip  h2f then f2h is the identity on all 65536 patterns,
 *        signed zero included
 * (harness: experiments/kcov/f16_cvt_selftest.c).
 *
 * THE ONE DOCUMENTED DIFFERENCE: NaN. h2f propagates the payload; f2h maps
 * every NaN to the canonical qNaN 0x7e00 rather than reproducing libgcc's
 * payload, and neither raises the invalid-operation flag libgcc's soft-fp
 * would for a signalling NaN. No NaN occurs in a quantized activation
 * tensor, and one appearing would already be a model bug rather than a
 * rounding question -- but the claim is "bit-exact on all non-NaN inputs",
 * not "bit-exact", and that is the accuracy_class this file declares.
 *
 * f2h's subnormal path adds `denorm_magic` with the FPU, so it assumes the
 * dynamic rounding mode is RNE. Zephyr never changes frm, and the only
 * kernels in this tree that touch it (the rvv frm_rmm ones) restore it
 * per block by construction.
 *
 * Guarded: every curated kernel body is concatenated into one kernels.c, so
 * a second file defining these would be a redefinition error. */
#include <stddef.h>
#include <stdint.h>
#ifndef MB_SOFT_F16_CVT_
#define MB_SOFT_F16_CVT_

static inline float mb_h2f(uint16_t h)
{
    union { uint32_t u; float f; } o, magic;
    magic.u = 113u << 23;
    const uint32_t shifted_exp = 0x7c00u << 13;   /* exponent mask, shifted */
    o.u = (uint32_t)(h & 0x7fffu) << 13;          /* exponent/mantissa bits */
    uint32_t exp = shifted_exp & o.u;
    o.u += (uint32_t)(127 - 15) << 23;            /* rebias exponent */
    if (exp == shifted_exp) {
        o.u += (uint32_t)(128 - 16) << 23;        /* inf/nan: extra rebias */
    } else if (exp == 0) {
        o.u += 1u << 23;                          /* subnormal: renormalize */
        o.f -= magic.f;
    }
    o.u |= (uint32_t)(h & 0x8000u) << 16;         /* sign */
    return o.f;
}

static inline uint16_t mb_f2h(float ff)
{
    union { uint32_t u; float f; } f, f32infty, f16max, denorm_magic;
    f.f = ff;
    f32infty.u     = 255u << 23;
    f16max.u       = (uint32_t)(127 + 16) << 23;
    denorm_magic.u = (uint32_t)((127 - 15) + (23 - 10) + 1) << 23;
    uint32_t sign = f.u & 0x80000000u;
    uint16_t o;
    f.u ^= sign;
    if (f.u >= f16max.u) {                        /* overflow -> inf, or nan */
        o = (f.u > f32infty.u) ? 0x7e00u : 0x7c00u;
    } else if (f.u < (113u << 23)) {              /* subnormal half */
        f.f += denorm_magic.f;
        o = (uint16_t)(f.u - denorm_magic.u);
    } else {                                      /* normal: RNE via odd bit */
        uint32_t mant_odd = (f.u >> 13) & 1u;
        f.u += ((uint32_t)(15 - 127) << 23) + 0xfffu;
        f.u += mant_odd;
        o = (uint16_t)(f.u >> 13);
    }
    o |= (uint16_t)(sign >> 16);
    return o;
}

/* _Float16 is the ABI type in the kernel signatures; these two move between
 * it and the raw bits without a libgcc call. memcpy is the only
 * strict-aliasing-clean spelling and gcc folds it to a plain lh/sh. */
static inline float mb_f16_load(const _Float16 *p)
{
    uint16_t h;
    __builtin_memcpy(&h, p, sizeof h);
    return mb_h2f(h);
}

static inline void mb_f16_store(_Float16 *p, float v)
{
    uint16_t h = mb_f2h(v);
    __builtin_memcpy(p, &h, sizeof h);
}
#endif /* MB_SOFT_F16_CVT_ */

#include <math.h>

void kernel_sigmoid_f16(const _Float16 *input, _Float16 *output, int n) {
    for (int i = 0; i < n; i++) {
        float v = mb_f16_load(&input[i]);
        mb_f16_store(&output[i], 1.0f / (1.0f + expf(-v)));
    }
}
