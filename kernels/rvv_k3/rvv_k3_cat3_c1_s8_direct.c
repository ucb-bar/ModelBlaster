/* source: curated */
/* algorithm: direct */
/* origin: K3 cat3_c1_s8: each input as one run; identity runs are memcpy.
 * k3-work TASK-017. Same text in kernels/rvv_k3/ (X100) and kernels/rvv_k3_a100/ (A100); the
 * register group is picked from VLEN at run time. */

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <riscv_vector.h>

/* ---- K3 256-entry int8 LUT: per-thread memo + LMUL-fitted register gather ----
 * (k3-work TASK-017; shared text, guarded, so several LUT ops can sit in one
 * kernels.c). Measured on the board (TASK-017 lmul_probe):
 *   - a vector instruction costs by its register-group size, not by vl, on both
 *     K3 cores, and vrgather.vv grows with LMUL squared: A100 (VLEN 1024) m2 =
 *     142 ns per 256 lanes, m8 = 2281 ns per 1024 lanes (0.56 vs 2.2 ns per byte);
 *     X100 (VLEN 256) m8 = 58 ns per 256 lanes. So the table goes in the SMALLEST
 *     register group that holds 256 bytes: m2 on the A100, m8 on the X100.
 *   - the stock kernels rebuild the table (256 expf + divides) whenever the quant
 *     parameters change, i.e. on every call in a network (yolov8_nano's 57 SiLUs
 *     all differ): 16 us per call on the X100, 26 us on the A100. Here each thread
 *     keeps up to MB_K3_LUT_MEMO_N tables keyed by the exact parameter bits, so a
 *     table is built once per thread and reused.
 * Bit-exact by construction: every table entry is the stock kernel's own scalar
 * expression, and the gather only moves bytes. Each thread has its own memo
 * (thread-local), so pool slices never share a table under construction. A table is
 * filled completely before the call that built it returns. */
#ifndef MB_K3_LUT_MEMO_
#define MB_K3_LUT_MEMO_
#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>

#ifndef MB_K3_LUT_MEMO_N
#define MB_K3_LUT_MEMO_N 128   /* power of two; yolov8_nano needs <= 85 per thread */
#endif
typedef struct { uint32_t used, key[6]; int8_t lut[256]; } mb_k3_lut_ent_t;

#if defined(__linux__) || defined(__APPLE__)
static _Thread_local mb_k3_lut_ent_t mb_k3_lut_memo[MB_K3_LUT_MEMO_N];
#define MB_K3_LUT_HAVE_MEMO 1
#else
#define MB_K3_LUT_HAVE_MEMO 0   /* no thread-local storage: build the table per call */
#endif

static inline uint32_t mb_k3_f32_bits(float f) { uint32_t u; memcpy(&u, &f, sizeof u); return u; }

/* The memoised table for `key`; or NULL and *fill = where to build it (a free slot on
 * the key's probe path, else the key's home slot is reused; `fallback` without a memo). */
static inline const int8_t *mb_k3_lut_get(const uint32_t key[6], int8_t **fill, int8_t *fallback)
{
#if MB_K3_LUT_HAVE_MEMO
    uint32_t h = 2166136261u;
    for (int k = 0; k < 6; k++) h = (h ^ key[k]) * 16777619u;
    h ^= h >> 15;
    for (uint32_t p = 0; p < MB_K3_LUT_MEMO_N; p++) {
        mb_k3_lut_ent_t *e = &mb_k3_lut_memo[(h + p) & (MB_K3_LUT_MEMO_N - 1)];
        if (!e->used) {
            e->used = 1; memcpy(e->key, key, sizeof e->key); *fill = e->lut; return NULL;
        }
        if (!memcmp(e->key, key, sizeof e->key)) return e->lut;
    }
    mb_k3_lut_ent_t *e = &mb_k3_lut_memo[h & (MB_K3_LUT_MEMO_N - 1)];
    memcpy(e->key, key, sizeof e->key);
    *fill = e->lut;
    return NULL;
#else
    (void)key;
    *fill = fallback;
    return NULL;
#endif
}

/* output[i] = lut[(uint8_t)input[i]], table held in the smallest register group
 * with room for 256 bytes (vrgather.vv returns 0 for an index >= VLMAX, so a
 * group of 256+ lanes is required); a memory gather below VLEN 256. */
static inline void mb_k3_lut_apply(const int8_t *input, int8_t *output, int n, const int8_t *lut)
{
    size_t vl;
    int i = 0;
    if (__riscv_vsetvlmax_e8m2() >= 256) {
        vuint8m2_t t = __riscv_vle8_v_u8m2((const uint8_t *)lut, __riscv_vsetvl_e8m2(256));
        for (; i < n; i += (int)vl) {
            vl = __riscv_vsetvl_e8m2((size_t)(n - i));
            vuint8m2_t x = __riscv_vle8_v_u8m2((const uint8_t *)(input + i), vl);
            __riscv_vse8_v_u8m2((uint8_t *)(output + i), __riscv_vrgather_vv_u8m2(t, x, vl), vl);
        }
    } else if (__riscv_vsetvlmax_e8m4() >= 256) {
        vuint8m4_t t = __riscv_vle8_v_u8m4((const uint8_t *)lut, __riscv_vsetvl_e8m4(256));
        for (; i < n; i += (int)vl) {
            vl = __riscv_vsetvl_e8m4((size_t)(n - i));
            vuint8m4_t x = __riscv_vle8_v_u8m4((const uint8_t *)(input + i), vl);
            __riscv_vse8_v_u8m4((uint8_t *)(output + i), __riscv_vrgather_vv_u8m4(t, x, vl), vl);
        }
    } else if (__riscv_vsetvlmax_e8m8() >= 256) {
        vuint8m8_t t = __riscv_vle8_v_u8m8((const uint8_t *)lut, __riscv_vsetvl_e8m8(256));
        for (; i < n; i += (int)vl) {
            vl = __riscv_vsetvl_e8m8((size_t)(n - i));
            vuint8m8_t x = __riscv_vle8_v_u8m8((const uint8_t *)(input + i), vl);
            __riscv_vse8_v_u8m8((uint8_t *)(output + i), __riscv_vrgather_vv_u8m8(t, x, vl), vl);
        }
    } else {
        for (; i < n; i += (int)vl) {
            vl = __riscv_vsetvl_e8m1((size_t)(n - i));
            vuint8m1_t x = __riscv_vle8_v_u8m1((const uint8_t *)(input + i), vl);
            __riscv_vse8_v_u8m1((uint8_t *)(output + i), __riscv_vluxei8_v_u8m1((const uint8_t *)lut, x, vl), vl);
        }
    }
}
#endif /* MB_K3_LUT_MEMO_ */

/* ---- K3 channel concat: one input's block, requantized (TASK-017) ----
 * For N = 1 an input of an NCHW channel concat is ONE contiguous run in the source and
 * in the output, so each input is handled as a single run (N > 1: one run per n).
 * Per run, bit-exact with the stock kernel's per-plane loop (same expression per element):
 *   - ratio == 1 and a full int8 clamp: the map is the identity -> memcpy;
 *   - VLEN 256 (X100): the stock expression tabulated for the 256 inputs (per-thread
 *     memo) and an m8 register gather, 0.23 ns per byte against ~0.4 for the float chain;
 *   - otherwise (A100): the stock float chain (vfwcvt, vfmul by ratio, convert in the
 *     current rounding mode, clamp), where the A100's m2 gather would be slower. */
#ifndef MB_K3_CAT_PART_
#define MB_K3_CAT_PART_
static void mb_k3_cat_part(const int8_t *src, int8_t *dst, size_t n, float ratio,
                           int activation_min, int activation_max)
{
    if (n == 0) return;
    if (ratio == 1.0f && activation_min <= -128 && activation_max >= 127) {
        memcpy(dst, src, n);
        return;
    }
    if (__riscv_vsetvlmax_e8m8() == 256 && n >= 256) {
        const uint32_t key[6] = {0xCA7u, mb_k3_f32_bits(ratio), (uint32_t)activation_min,
                                 (uint32_t)activation_max, 0u, 0u};
        int8_t local[256], *t;
        const int8_t *lut = mb_k3_lut_get(key, &t, local);
        if (lut == NULL) {
            for (int v = 0; v < 256; v++) {
                float f = (float)(int8_t)(uint8_t)v * ratio;   /* vfwcvt (exact), vfmul */
                long q = lrintf(f);                            /* vfcvt.x.f: current mode */
                if (q > 2147483647L) q = 2147483647L;          /* the e32 convert saturates */
                if (q < -2147483647L - 1) q = -2147483647L - 1;
                if (q < activation_min) q = activation_min;
                if (q > activation_max) q = activation_max;
                t[v] = (int8_t)q;
            }
            lut = t;
        }
        mb_k3_lut_apply(src, dst, (int)n, lut);
        return;
    }
    size_t vl;
    for (size_t i = 0; i < n; i += vl) {
        /* the stock kernel's inner loop (kernels/rvv/rvv_cat2_c1_s8_direct.c) */
        const size_t n_elem = n - i;
        vl = __riscv_vsetvl_e8m2(n_elem);
        size_t vl8 = vl;
        size_t vl16 = __riscv_vsetvl_e16m4(n_elem);
        size_t vl32 = __riscv_vsetvl_e32m8(n_elem);
        vint8m2_t v8 = __riscv_vle8_v_i8m2(src + i, vl8);
        vint16m4_t v16 = __riscv_vsext_vf2_i16m4(v8, vl16);
        vfloat32m8_t vf = __riscv_vfwcvt_f_x_v_f32m8(v16, vl32);
        vf = __riscv_vfmul_vf_f32m8(vf, ratio, vl32);
        vint32m8_t vi = __riscv_vfcvt_x_f_v_i32m8(vf, vl32);
        vi = __riscv_vmax_vx_i32m8(vi, activation_min, vl32);
        vi = __riscv_vmin_vx_i32m8(vi, activation_max, vl32);
        vint16m4_t vi16 = __riscv_vncvt_x_x_w_i16m4(vi, vl16);
        vint8m2_t vi8 = __riscv_vncvt_x_x_w_i8m2(vi16, vl8);
        __riscv_vse8_v_i8m2(dst + i, vi8, vl8);
    }
}
#endif /* MB_K3_CAT_PART_ */

void kernel_cat3_c1_s8(const int8_t *in0, int c0, float scale0, const int8_t *in1, int c1, float scale1, const int8_t *in2, int c2, float scale2,
                    int8_t *output, int N, int H, int W, float scale_out,
                    int activation_min, int activation_max)
{
    const size_t hw = (size_t)H * (size_t)W;
    const int8_t *ins[3] = {in0, in1, in2};
    const int cs[3] = {c0, c1, c2};
    /* the stock kernel's ratio, computed the same way */
    const float ratio[3] = {scale0 / scale_out, scale1 / scale_out, scale2 / scale_out};
    const int c_total = c0 + c1 + c2;
    for (int n = 0; n < N; n++) {
        size_t off = 0;
        for (int j = 0; j < 3; j++) {
            mb_k3_cat_part(ins[j] + (size_t)n * cs[j] * hw, output + ((size_t)n * c_total + off) * hw,
                           (size_t)cs[j] * hw, ratio[j], activation_min, activation_max);
            off += (size_t)cs[j];
        }
    }
}
