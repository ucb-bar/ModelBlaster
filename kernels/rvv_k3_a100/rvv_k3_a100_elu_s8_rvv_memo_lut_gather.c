/* source: curated */
/* algorithm: rvv_memo_lut_gather */
/* origin: K3 elu_s8: the stock rvv kernel's per-value expression tabulated for all 256
 * inputs, memoised per thread, applied with an LMUL-fitted register gather.
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

void kernel_elu_s8(const int8_t *input, int8_t *output, int n,
                   float scale_in, float scale_out,
                   int activation_min, int activation_max, float alpha)
{
    if (n <= 0) return;
    const uint32_t key[6] = {0xE1Au, mb_k3_f32_bits(scale_in), mb_k3_f32_bits(scale_out),
                             (uint32_t)activation_min, (uint32_t)activation_max,
                             mb_k3_f32_bits(alpha)};
    int8_t local[256], *t;
    const int8_t *lut = mb_k3_lut_get(key, &t, local);
    if (lut == NULL) {
        /* The stock kernel's (kernels/rvv/rvv_elu_s8_rvv_memo_lut_gather.c)
         * per-value expression, verbatim, for all 256 inputs. */
        for (int v = 0; v < 256; v++) {
            float f = (float)(int8_t)(uint8_t)v * scale_in;
            float y = (f > 0.0f) ? f : alpha * (expf(f) - 1.0f);
            int32_t q = (int32_t)roundf(y / scale_out);
            if (q < activation_min) q = activation_min;
            if (q > activation_max) q = activation_max;
            t[v] = (int8_t)q;
        }
        mb_k3_lut_apply(input, output, n, t);
        return;
    }
    mb_k3_lut_apply(input, output, n, lut);
}
