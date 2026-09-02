/* source: curated */
/* algorithm: soft_f16 */
/* accuracy_class: bit_exact */
/* origin: hand-written. The conv2d_f16 reference nest, loop for loop, with
 *         every _Float16 <-> float conversion inlined instead of called.
 *
 *   WHY THIS FILE EXISTS. conv2d_f16's two AlgorithmCandidates are both
 *   rvv_f16-affined, so on a Gemmini target the curated probe had nothing to
 *   look for and the op ran the scalar reference. That reference is the
 *   single largest line item in ViNT on gemmini_q31: 5,146,370,253 spike
 *   cycles across 65 dispatches, 65.7% of the whole 7.83-billion-cycle
 *   model.
 *
 *   AND WHY IT IS ONLY A CONVERSION CHANGE. Gemmini is an int8 systolic
 *   array; there is no fp16 Gemmini path to route this to, so the honest
 *   ceiling here is "the same arithmetic, without the libgcc calls". The
 *   arithmetic was never the cost: the inner statement is one fmadd
 *   surrounded by TWO calls to __extendhfsf2, because the target has no Zfh
 *   (see the conversion helper's header below).

 *   A SECOND OPTIMIZATION THAT MEASURED WORSE, kept here as the record.
 *   Inlining the conversions took conv2d_f16 from 5,146,370,253 to
 *   3,981,610,335 cycles -- only 1.29x, i.e. 112 cycles per MAC remained.
 *   The obvious next move is to stop bounds-testing every tap: the valid kh
 *   range for an output row is [max(0,-ih0), min(KH, IH-ih0)) and depends
 *   only on oh, the kw range only on ow, so four comparisons per tap become
 *   two clamps per output pixel. Measured on the same model:
 *
 *       conv2d_f16            3,981,610,335 -> 3,988,846,653   (0.2% WORSE)
 *       depthwise_conv2d_f16    234,934,453 ->   246,435,557   (4.9% WORSE)
 *
 *   Two reasons, and both are worth knowing before trying it again here.
 *   (1) kernels.c is compiled -Os, not -O2 (Zephyr's default; confirmed in
 *   this build's compile_commands.json), and at -Os GCC will not keep the
 *   extra loop-invariant range variables in registers -- the disassembly of
 *   the clamped version is a wall of sd/ld stack spills. The hoist paid for
 *   itself in tests removed and lost it again in spills. (2) ViNT's fp16
 *   convolutions are almost all PH=PW=0, so there were no out-of-bounds taps
 *   to skip in the first place; the tests were predictable and nearly free.
 *   Where padding IS heavy the same change wins outright -- see
 *   gemmini_q31_depthwise_conv2d_s8_scalar_tap_ranges.c, whose 5x5-kernel /
 *   PH=PW=2 / 2x2-input layers waste 21 of every 25 taps and which went from
 *   0.93x to 1.15x on exactly this rewrite. The optimization is not wrong,
 *   it is shape-dependent, and these shapes are the wrong ones.
 *
 *   WHAT IS ALSO DELIBERATELY NOT DONE. Hoisting the IC*KH*KW weight patch
 *   into a float scratch once per output channel would remove the remaining
 *   weight conversions, but these kernels run on modelblaster_pool threads
 *   with CONFIG_DYNAMIC_THREAD_STACK_SIZE=32768 and ViNT's widest conv needs
 *   more than that; a per-dispatch heap allocation would be measured as part
 *   of the op.
 *
 *   BIT-EXACT: the accumulation order, the float accumulator, the bias
 *   handling and the bounds tests are the reference's, unchanged. Only the
 *   spelling of the conversions differs, and those are exact (see below). */

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

void kernel_conv2d_f16(const _Float16 *input, const _Float16 *weight,
                       const _Float16 *bias, _Float16 *output,
                       int N, int IC, int IH, int IW, int OC,
                       int KH, int KW, int SH, int SW, int PH, int PW) {
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < OC; oc++) {
            float bias_v = bias ? mb_f16_load(&bias[oc]) : 0.0f;
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    float acc = bias_v;
                    for (int ic = 0; ic < IC; ic++) {
                        const _Float16 *ip = input + ((size_t)(n*IC + ic)*IH)*IW;
                        const _Float16 *wp = weight + ((size_t)(oc*IC + ic)*KH)*KW;
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            if (ih < 0 || ih >= IH) continue;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = ow * SW - PW + kw;
                                if (iw < 0 || iw >= IW) continue;
                                float v = mb_f16_load(&ip[(size_t)ih*IW + iw]);
                                float w = mb_f16_load(&wp[(size_t)kh*KW + kw]);
                                acc += v * w;
                            }
                        }
                    }
                    mb_f16_store(&output[((size_t)(n*OC + oc)*OH + oh)*OW + ow],
                                 acc);
                }
            }
        }
    }
}
