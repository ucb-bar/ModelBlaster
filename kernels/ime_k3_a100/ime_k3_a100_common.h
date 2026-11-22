/* Shared implementation of the SpaceMiT K3 A100 IME (smt.vmadot int8) conv
 * kernels: conv2d_s8 / conv2d_batchnorm2d_s8 / conv2d_batchnorm2d_silu_s8.
 *
 * ONLY LEGAL ON THE A100 (CPUs 8-15, VLEN=1024). A runtime vlenb check falls
 * back to a plain scalar path if the vector length is not 1024, so a wrong
 * placement is slow rather than silently wrong.
 *
 * ALGORITHM (implicit GEMM, no im2col buffer)
 *   vmadot tile (verified, artifacts/k3_probes/ime/summary.md): C[8][8] (int32,
 *   pair vd,vd+1, row-major) += A[8][16] . B[8][16]^T, int8 operands.
 *   Here A = 8 output CHANNELS x 16 input-channel slots (weights), B = 8 output
 *   PIXELS x 16 channel slots, so C[i][j] = channel i, pixel j: every tile row
 *   is 8 consecutive pixels of one channel plane (8 contiguous output bytes).
 *     * The input is converted once per call to a zero-padded NHWC16 layout
 *       [cg][row][col][16 channels] (cg = channel group of 16). One B tile for
 *       tap (kh,kw) and channel group cg is then 8 consecutive pixels = 128
 *       CONTIGUOUS bytes -> a single vle8, no per-tile im2col. Stride 2 uses
 *       two column-phase planes (even/odd columns) so the 8 pixels stay
 *       contiguous.
 *     * Weights (IHWOC, as for rvv) are repacked into [step][tile][8 oc][16 ch]
 *       once and CACHED (key: weight pointer + dims + 4 sampled 64-byte
 *       chunks compared exactly; weights are const in every real graph).
 *     * K steps = (channel groups) x KH x KW; every step is one vmadot per
 *       (oc tile, pixel group). Blocking is PT x PG (<= 8 accumulator pairs):
 *       6 loads per 8 vmadots (the A100 is load-bound).
 *     * Epilogue per tile: Q0.31 vsmul + vnclip + clamp -> int8, then 8
 *       unaligned 8-byte scalar row stores straight into the NCHW output
 *       plane (exact tails for OW % 8 != 0).
 *     * BN (+SiLU) is a per-channel 256-entry int8 table, built with vector
 *       float math (same op sequence as the scalar reference, roundf via
 *       frm=RMM) and CACHED (key: bn arrays + params, compared exactly). It is
 *       applied in place by a vrgather pass over whole channel planes.
 *     * The NCHW -> NHWC16 transpose is done with scalar 64-bit 8x8 byte
 *       transposes (see mb_il16).
 *   A100 measurements that drove this (scripts/k3_ime_layout_probe.c,
 *   artifacts/k3_probes/ime/layout_cpu8.txt): segment loads/stores and strided
 *   accesses are 10-100x slower than unit-stride ops (vsseg8e8 ~1 us/call,
 *   vlseg8e8 up to 8 us), vluxei is ~15 ns/elem, vfdiv m2 is 173 ns, scalar
 *   ld/sd (also unaligned, via plain 64-bit derefs) are ~1 ns, vrgather e8 m1
 *   is ~55 ns per 128 elems.
 *   Bit-exact with the scalar reference whenever input_offset == filter_offset
 *   == 0 (true for every real graph); otherwise the scalar path is used.
 */
#ifndef MB_IME_K3_A100_COMMON_H
#define MB_IME_K3_A100_COMMON_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <riscv_vector.h>

#pragma GCC push_options
#pragma GCC optimize("no-strict-aliasing,no-tree-vectorize,no-tree-slp-vectorize")

/* The cross GCC 14 rejects the smt.vmadot mnemonic and `.insn` does not take
 * v-register names, so emit the R-type word by hand. The .irp defines the
 * assembler symbols v0..v31 so "%0" (printed as v8) is a number here. */
__asm__(".macro mb_vmadot_s8 vd, vs1, vs2\n"
        " .word (0x2b | (\\vd<<7) | (3<<12) | (\\vs1<<15) | (\\vs2<<20) | (0x71<<25))\n"
        ".endm\n"
        ".irp n,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31\n"
        " .set v\\n, \\n\n"
        ".endr\n");

/* c += A . B^T. vtype must be e8/m1 when this issues (the surrounding vle8 of
 * the operands guarantees it; checked in the disassembly). c is an even-aligned
 * LMUL=2 pair (vd, vd+1), which is exactly where the 64 int32 results land. */
static inline __attribute__((always_inline)) vint32m2_t
mb_ime_vmadot(vint32m2_t c, vint8m1_t a, vint8m1_t b)
{
    __asm__ volatile("mb_vmadot_s8 %0, %1, %2" : "+vr"(c) : "vr"(a), "vr"(b));
    return c;
}

/* ---- thread-local grow-only scratch ------------------------------------ */
enum { MB_SCR_PLANE, MB_SCR_WP, MB_SCR_OFF, MB_SCR_STAGE, MB_SCR_ROW, MB_SCR_LUT, MB_SCR_N };
static __thread void *mb_scr_p[MB_SCR_N];
static __thread size_t mb_scr_cap[MB_SCR_N];
static void *mb_scr_get(int i, size_t sz)
{
    if (mb_scr_cap[i] < sz) {
        free(mb_scr_p[i]);
        size_t r = (sz + 255) & ~(size_t)63;
        mb_scr_p[i] = aligned_alloc(64, r);
        mb_scr_cap[i] = mb_scr_p[i] ? r : 0;
    }
    return mb_scr_p[i];
}

/* ---- scalar 8x8 byte transpose / 16-channel interleave ------------------
 * dst[x*16 + c] = rows[c][x] (0 where rows[c] == NULL), x < n. The A100 has
 * fast unaligned scalar ld/sd (~1 ns) but very slow segment/strided vector
 * memory ops (vsseg8e8 ~1 us per call, vlseg8e8 up to 8 us), so the
 * NCHW -> NHWC16 transpose is done as 64-bit scalar 8x8 byte transposes.
 * Pixel x of the row lands at column colbase+x of the destination; with
 * `phases` the columns are split into even/odd planes (dst1 = odd plane).
 * `safe_end` bounds 8-byte over-reads of the source rows. */
static inline __attribute__((always_inline)) uint64_t mb_ld64u(const int8_t *p)
{
    return *(const uint64_t *)p;
}
static inline __attribute__((always_inline)) void mb_tr8(uint64_t *a)
{
#define MB_SW(R0, R1, S_, M_)                                                  \
    { uint64_t t_ = ((a[R0] >> (8 * (S_))) ^ a[R1]) & (M_);                    \
      a[R1] ^= t_; a[R0] ^= t_ << (8 * (S_)); }
    const uint64_t m1 = 0x00FF00FF00FF00FFull, m2 = 0x0000FFFF0000FFFFull,
                   m4 = 0x00000000FFFFFFFFull;
    MB_SW(0, 1, 1, m1) MB_SW(2, 3, 1, m1) MB_SW(4, 5, 1, m1) MB_SW(6, 7, 1, m1)
    MB_SW(0, 2, 2, m2) MB_SW(1, 3, 2, m2) MB_SW(4, 6, 2, m2) MB_SW(5, 7, 2, m2)
    MB_SW(0, 4, 4, m4) MB_SW(1, 5, 4, m4) MB_SW(2, 6, 4, m4) MB_SW(3, 7, 4, m4)
#undef MB_SW
}

static void mb_il16(const int8_t *const *rows, int n, int8_t *dst0, int8_t *dst1,
                    int colbase, int phases, const int8_t *safe_end)
{
    int nv = 0;                         /* rows[0..nv) are non-NULL (zero-padded channels follow) */
    while (nv < 16 && rows[nv]) nv++;
    for (int x0 = 0; x0 < n; x0 += 8) {
        const int cnt = n - x0 < 8 ? n - x0 : 8;
        uint64_t r[16];
        if (nv && rows[nv - 1] + x0 + 8 <= safe_end) {
            for (int c = 0; c < nv; c++) r[c] = mb_ld64u(rows[c] + x0);
        } else {
            for (int c = 0; c < nv; c++) {
                const int8_t *p = rows[c] + x0;
                uint64_t v = 0;
                for (int k = cnt - 1; k >= 0; k--) v = (v << 8) | (uint8_t)p[k];
                r[c] = v;
            }
        }
        for (int c = nv; c < 16; c++) r[c] = 0;
        mb_tr8(r);
        if (nv > 8) mb_tr8(r + 8); else { for (int c = 8; c < 16; c++) r[c] = 0; }
        if (!phases) {
            int8_t *d = dst0 + (size_t)(colbase + x0) * 16;
            if (cnt >= 8) {
                for (int j = 0; j < 8; j++) {
                    *(uint64_t *)(d + j * 16) = r[j];
                    *(uint64_t *)(d + j * 16 + 8) = r[8 + j];
                }
            } else {
                for (int j = 0; j < cnt; j++) {
                    *(uint64_t *)(d + j * 16) = r[j];
                    *(uint64_t *)(d + j * 16 + 8) = r[8 + j];
                }
            }
        } else {
            for (int j = 0; j < cnt; j++) {
                int col = colbase + x0 + j;
                int8_t *d = ((col & 1) ? dst1 : dst0) + (size_t)(col >> 1) * 16;
                *(uint64_t *)d = r[j];
                *(uint64_t *)(d + 8) = r[8 + j];
            }
        }
    }
}

/* ---- per-channel BN(+SiLU) tables ---------------------------------------- */
static void mb_silu_fill(int8_t *lut, float silu_scale_in, float silu_scale_out,
                         int amin, int amax)
{
    for (int v = 0; v < 256; v++) {
        int8_t bn_int8 = (int8_t)(v - 128);
        float fbv = (float)bn_int8 * silu_scale_in;
        float sy = fbv / (1.0f + expf(-fbv));
        int32_t q = (int32_t)roundf(sy / silu_scale_out);
        if (q < amin) q = amin;
        if (q > amax) q = amax;
        lut[v] = (int8_t)q;
    }
}

/* lut[c][v], v = conv_int8 + 128. mode 1: BN stage; mode 2: BN then SiLU.
 * Same float op sequence as the scalar reference: fv = q*sin (rounded),
 * y = fma(bn_s, fv, bn_b), z = y / sout, bv = roundf(z) (RMM), clamp. */
static void mb_build_lut(int8_t *lut, const float *bn_scale, const float *bn_bias,
                         int OC, float sin_, float sout, int bamin, int bamax,
                         const int8_t *silu /* 256 entries or NULL */)
{
    const size_t vl = 256;
    vint32m8_t qv = __riscv_vsub_vx_i32m8(
        __riscv_vreinterpret_v_u32m8_i32m8(__riscv_vid_v_u32m8(vl)), 128, vl);
    vfloat32m8_t fv = __riscv_vfmul_vf_f32m8(__riscv_vfcvt_f_x_v_f32m8(qv, vl), sin_, vl);
    vuint8m1_t slo, shi;
    if (silu) {
        slo = __riscv_vle8_v_u8m1((const uint8_t *)silu, 128);
        shi = __riscv_vle8_v_u8m1((const uint8_t *)silu + 128, 128);
    }
    for (int c = 0; c < OC; c++) {
        vfloat32m8_t y = __riscv_vfmacc_vf_f32m8(
            __riscv_vfmv_v_f_f32m8(bn_bias[c], vl), bn_scale[c], fv, vl);
        y = __riscv_vfdiv_vf_f32m8(y, sout, vl);
        vint32m8_t bv = __riscv_vfcvt_x_f_v_i32m8_rm(y, __RISCV_FRM_RMM, vl);
        bv = __riscv_vmax_vx_i32m8(bv, bamin, vl);
        bv = __riscv_vmin_vx_i32m8(bv, bamax, vl);
        vint8m2_t b8 = __riscv_vnsra_wx_i8m2(__riscv_vnsra_wx_i16m4(bv, 0, vl), 0, vl);
        if (!silu) {
            __riscv_vse8_v_i8m2(lut + (size_t)c * 256, b8, vl);
        } else {
            vuint8m2_t idx = __riscv_vxor_vx_u8m2(
                __riscv_vreinterpret_v_i8m2_u8m2(b8), 0x80, vl);
#define MB_LUTHALF(H_)                                                         \
            { vuint8m1_t ix = __riscv_vget_v_u8m2_u8m1(idx, H_);                \
              vuint8m1_t lo = __riscv_vrgather_vv_u8m1(slo, ix, 128);          \
              vuint8m1_t hi = __riscv_vrgather_vv_u8m1(                        \
                  shi, __riscv_vsub_vx_u8m1(ix, 128, 128), 128);               \
              __riscv_vse8_v_u8m1((uint8_t *)lut + (size_t)c * 256 + (H_) * 128,\
                                  __riscv_vor_vv_u8m1(lo, hi, 128), 128); }
            MB_LUTHALF(0)
            MB_LUTHALF(1)
#undef MB_LUTHALF
        }
    }
}

typedef struct {
    int OC, mode, bamin, bamax, samin, samax;
    float sin_, sout, ssin, ssout;
    float *bn;            /* OC scale floats then OC bias floats */
    int8_t *lut;          /* OC*256 */
} mb_lut_ent_t;
enum { MB_LUT_ENTS = 96 };
static mb_lut_ent_t mb_lut_cache[MB_LUT_ENTS];
static int mb_lut_n;
static int mb_lut_lock;

static inline void mb_spin_lock(int *l)
{
    while (__atomic_exchange_n(l, 1, __ATOMIC_ACQUIRE)) { }
}
static inline void mb_spin_unlock(int *l) { __atomic_store_n(l, 0, __ATOMIC_RELEASE); }

static const int8_t *mb_get_lut(const float *bn_scale, const float *bn_bias, int OC,
                                int mode, float sin_, float sout, int bamin, int bamax,
                                float ssin, float ssout, int samin, int samax)
{
    mb_spin_lock(&mb_lut_lock);
    for (int i = 0; i < mb_lut_n; i++) {
        mb_lut_ent_t *e = &mb_lut_cache[i];
        if (e->OC == OC && e->mode == mode && e->sin_ == sin_ && e->sout == sout &&
            e->bamin == bamin && e->bamax == bamax &&
            (mode == 1 || (e->ssin == ssin && e->ssout == ssout &&
                           e->samin == samin && e->samax == samax)) &&
            !memcmp(e->bn, bn_scale, (size_t)OC * 4) &&
            !memcmp(e->bn + OC, bn_bias, (size_t)OC * 4)) {
            const int8_t *r = e->lut;
            mb_spin_unlock(&mb_lut_lock);
            return r;
        }
    }
    mb_spin_unlock(&mb_lut_lock);

    int8_t silu[256] __attribute__((aligned(64)));
    if (mode == 2) mb_silu_fill(silu, ssin, ssout, samin, samax);
    int8_t *lut = (int8_t *)aligned_alloc(64, ((size_t)OC * 256 + 63) & ~(size_t)63);
    float *bn = (float *)malloc((size_t)OC * 8);
    if (!lut || !bn) {
        free(lut); free(bn);
        lut = (int8_t *)mb_scr_get(MB_SCR_LUT, (size_t)OC * 256);
        if (lut) mb_build_lut(lut, bn_scale, bn_bias, OC, sin_, sout, bamin, bamax,
                              mode == 2 ? silu : NULL);
        return lut;
    }
    mb_build_lut(lut, bn_scale, bn_bias, OC, sin_, sout, bamin, bamax,
                 mode == 2 ? silu : NULL);
    memcpy(bn, bn_scale, (size_t)OC * 4);
    memcpy(bn + OC, bn_bias, (size_t)OC * 4);
    mb_spin_lock(&mb_lut_lock);
    if (mb_lut_n < MB_LUT_ENTS) {
        mb_lut_ent_t *e = &mb_lut_cache[mb_lut_n];
        e->OC = OC; e->mode = mode; e->sin_ = sin_; e->sout = sout;
        e->bamin = bamin; e->bamax = bamax; e->ssin = ssin; e->ssout = ssout;
        e->samin = samin; e->samax = samax; e->bn = bn; e->lut = lut;
        mb_lut_n++;
        mb_spin_unlock(&mb_lut_lock);
        return lut;
    }
    mb_spin_unlock(&mb_lut_lock);
    /* cache full: hand back a per-thread copy */
    int8_t *t = (int8_t *)mb_scr_get(MB_SCR_LUT, (size_t)OC * 256);
    if (t) memcpy(t, lut, (size_t)OC * 256);
    free(lut); free(bn);
    return t;
}

/* ---- packed weight cache ------------------------------------------------- */
typedef struct {
    const void *w;
    int IC, OC, KH, KW;
    int8_t samp[4][64];
    int8_t *packed;
} mb_wp_ent_t;
enum { MB_WP_ENTS = 160 };
static mb_wp_ent_t mb_wp_cache[MB_WP_ENTS];
static int mb_wp_n;
static int mb_wp_lock;

static void mb_wp_sample(int8_t samp[4][64], const int8_t *w, size_t n)
{
    for (int i = 0; i < 4; i++) {
        size_t o = (n > 64) ? (n - 64) * (size_t)i / 3 : 0;
        size_t m = n < 64 ? n : 64;
        memset(samp[i], 0, 64);
        memcpy(samp[i], w + o, m);
    }
}

static void mb_pack_weights(int8_t *wp, const int8_t *weight, int IC, int OC,
                            int KH, int KW, int ntile)
{
    const int cgs = (IC + 15) / 16;
    const size_t wstride = (size_t)ntile * 128;
    for (int cg = 0; cg < cgs; cg++)
        for (int kh = 0; kh < KH; kh++)
            for (int kw = 0; kw < KW; kw++) {
                int s = (cg * KH + kh) * KW + kw;
                const int8_t *rows[16];
                for (int k = 0; k < 16; k++) {
                    int c = cg * 16 + k;
                    rows[k] = (c < IC)
                        ? weight + (((size_t)c * KH + kh) * KW + kw) * OC : NULL;
                }
                int8_t *d = wp + (size_t)s * wstride;
                mb_il16(rows, OC, d, NULL, 0, 0, weight + (size_t)IC * KH * KW * OC);
                if (ntile * 8 > OC)
                    memset(d + (size_t)OC * 16, 0, (size_t)(ntile * 8 - OC) * 16);
            }
}

static const int8_t *mb_get_wp(const int8_t *weight, int IC, int OC, int KH, int KW,
                               int ntile)
{
    const size_t wbytes = (size_t)IC * KH * KW * OC;
    const size_t pbytes = (size_t)((IC + 15) / 16) * KH * KW * ntile * 128;
    int8_t samp[4][64];
    mb_wp_sample(samp, weight, wbytes);
    mb_spin_lock(&mb_wp_lock);
    for (int i = 0; i < mb_wp_n; i++) {
        mb_wp_ent_t *e = &mb_wp_cache[i];
        if (e->w == weight && e->IC == IC && e->OC == OC && e->KH == KH &&
            e->KW == KW && !memcmp(e->samp, samp, sizeof samp)) {
            const int8_t *r = e->packed;
            mb_spin_unlock(&mb_wp_lock);
            return r;
        }
    }
    mb_spin_unlock(&mb_wp_lock);
    int8_t *p = (int8_t *)aligned_alloc(64, (pbytes + 63) & ~(size_t)63);
    if (!p) {
        p = (int8_t *)mb_scr_get(MB_SCR_WP, pbytes);
        if (p) mb_pack_weights(p, weight, IC, OC, KH, KW, ntile);
        return p;
    }
    mb_pack_weights(p, weight, IC, OC, KH, KW, ntile);
    mb_spin_lock(&mb_wp_lock);
    /* an entry for the same weight pointer is stale (weights changed): reuse it */
    int slot = -1;
    for (int i = 0; i < mb_wp_n; i++)
        if (mb_wp_cache[i].w == weight && mb_wp_cache[i].IC == IC &&
            mb_wp_cache[i].OC == OC && mb_wp_cache[i].KH == KH &&
            mb_wp_cache[i].KW == KW) { slot = i; break; }
    if (slot < 0 && mb_wp_n < MB_WP_ENTS) slot = mb_wp_n++;
    if (slot >= 0) {
        mb_wp_ent_t *e = &mb_wp_cache[slot];
        /* the stale buffer is intentionally leaked: another thread may still be
         * reading it */
        e->w = weight; e->IC = IC; e->OC = OC; e->KH = KH; e->KW = KW;
        memcpy(e->samp, samp, sizeof samp);
        e->packed = p;
        mb_spin_unlock(&mb_wp_lock);
        return p;
    }
    mb_spin_unlock(&mb_wp_lock);
    int8_t *t = (int8_t *)mb_scr_get(MB_SCR_WP, pbytes);
    if (t) memcpy(t, p, pbytes);
    free(p);
    return t;
}

/* ---- epilogue context --------------------------------------------------- */
typedef struct {
    int mult, shift, coff, amin, amax;      /* conv requantize */
} mb_ep_t;

/* One 8x8 int32 tile (channel-major: row i = channel, col j = pixel) ->
 * conv_int8 values, `cnt` (<= 8) bytes per channel row stored straight into the
 * NCHW output plane (unaligned 8-byte scalar stores; exact tails). */
static inline __attribute__((always_inline)) void
mb_epi_tile(vint32m2_t acc, int8_t *dst, size_t cs, int rows, int cnt,
            const mb_ep_t *e)
{
    const size_t vl = 64;
    vint32m2_t s = __riscv_vsmul_vx_i32m2(acc, e->mult, __RISCV_VXRM_RNU, vl);
    vint16m1_t o16;
    if (e->shift < 0) {
        vint32m2_t sh = __riscv_vsll_vx_i32m2(s, (size_t)(-e->shift), vl);
        o16 = __riscv_vnclip_wx_i16m1(sh, 0, __RISCV_VXRM_RNU, vl);
    } else if (e->shift < 32) {
        o16 = __riscv_vnclip_wx_i16m1(s, (size_t)e->shift, __RISCV_VXRM_RNU, vl);
    } else {
        int sa2 = e->shift - 31;
        if (sa2 > 31) sa2 = 31;
        vint32m2_t s2 = __riscv_vsra_vx_i32m2(s, 31, vl);
        o16 = __riscv_vnclip_wx_i16m1(s2, (size_t)sa2, __RISCV_VXRM_RNU, vl);
    }
    o16 = __riscv_vsadd_vx_i16m1(o16, (int16_t)e->coff, vl);
    o16 = __riscv_vmax_vx_i16m1(o16, (int16_t)e->amin, vl);
    o16 = __riscv_vmin_vx_i16m1(o16, (int16_t)e->amax, vl);
    vint8mf2_t q = __riscv_vnsra_wx_i8mf2(o16, 0, vl);
    uint64_t tb[8] __attribute__((aligned(64)));
    __riscv_vse8_v_i8mf2((int8_t *)tb, q, vl);
    if (cnt >= 8 && rows >= 8) {
        *(uint64_t *)(dst + 0 * cs) = tb[0];
        *(uint64_t *)(dst + 1 * cs) = tb[1];
        *(uint64_t *)(dst + 2 * cs) = tb[2];
        *(uint64_t *)(dst + 3 * cs) = tb[3];
        *(uint64_t *)(dst + 4 * cs) = tb[4];
        *(uint64_t *)(dst + 5 * cs) = tb[5];
        *(uint64_t *)(dst + 6 * cs) = tb[6];
        *(uint64_t *)(dst + 7 * cs) = tb[7];
    } else {
        for (int i = 0; i < rows; i++) {
            int8_t *d = dst + (size_t)i * cs;
            uint64_t v = tb[i];
            if (cnt >= 8) { *(uint64_t *)d = v; continue; }
            if (cnt & 4) { *(uint32_t *)d = (uint32_t)v; d += 4; v >>= 32; }
            if (cnt & 2) { *(uint16_t *)d = (uint16_t)v; d += 2; v >>= 16; }
            if (cnt & 1) { *d = (int8_t)v; }
        }
    }
}

/* ---- PT x PG register block (generated; vector types cannot be arrays) -- */
static inline __attribute__((always_inline)) void
mb_blk_4x2(const int8_t *plane, const size_t *ab, const int32_t *off,
    int S, const int8_t *wp, size_t wstride, const int32_t *bexp,
    int8_t *dst0, size_t cs, const size_t *go, const int *gc, int rows_left, const mb_ep_t *e)
{
    vint32m2_t b0_0 = __riscv_vle32_v_i32m2(bexp + 0, 64);
    vint32m2_t b0_1 = __riscv_vle32_v_i32m2(bexp + 64, 64);
    vint32m2_t b0_2 = __riscv_vle32_v_i32m2(bexp + 128, 64);
    vint32m2_t b0_3 = __riscv_vle32_v_i32m2(bexp + 192, 64);
    vint32m2_t c0_0 = b0_0;
    vint32m2_t c0_1 = b0_0;
    vint32m2_t c1_0 = b0_1;
    vint32m2_t c1_1 = b0_1;
    vint32m2_t c2_0 = b0_2;
    vint32m2_t c2_1 = b0_2;
    vint32m2_t c3_0 = b0_3;
    vint32m2_t c3_1 = b0_3;
    const int8_t *bp0 = plane + ab[0];
    const int8_t *bp1 = plane + ab[1];
    for (int s = 0; s < S; s++) {
        const size_t o = (size_t)off[s];
        const int8_t *ap = wp + (size_t)s * wstride;
        vint8m1_t a0 = __riscv_vle8_v_i8m1(ap + 0, 128);
        vint8m1_t a1 = __riscv_vle8_v_i8m1(ap + 128, 128);
        vint8m1_t a2 = __riscv_vle8_v_i8m1(ap + 256, 128);
        vint8m1_t a3 = __riscv_vle8_v_i8m1(ap + 384, 128);
        vint8m1_t b0 = __riscv_vle8_v_i8m1(bp0 + o, 128);
        vint8m1_t b1 = __riscv_vle8_v_i8m1(bp1 + o, 128);
        c0_0 = mb_ime_vmadot(c0_0, a0, b0);
        c0_1 = mb_ime_vmadot(c0_1, a0, b1);
        c1_0 = mb_ime_vmadot(c1_0, a1, b0);
        c1_1 = mb_ime_vmadot(c1_1, a1, b1);
        c2_0 = mb_ime_vmadot(c2_0, a2, b0);
        c2_1 = mb_ime_vmadot(c2_1, a2, b1);
        c3_0 = mb_ime_vmadot(c3_0, a3, b0);
        c3_1 = mb_ime_vmadot(c3_1, a3, b1);
    }
    { int rows = rows_left - 0; if (rows > 8) rows = 8;
      if (rows > 0) mb_epi_tile(c0_0, dst0 + 0*cs + go[0], cs, rows, gc[0], e);
      if (rows > 0) mb_epi_tile(c0_1, dst0 + 0*cs + go[1], cs, rows, gc[1], e); }
    { int rows = rows_left - 8; if (rows > 8) rows = 8;
      if (rows > 0) mb_epi_tile(c1_0, dst0 + 8*cs + go[0], cs, rows, gc[0], e);
      if (rows > 0) mb_epi_tile(c1_1, dst0 + 8*cs + go[1], cs, rows, gc[1], e); }
    { int rows = rows_left - 16; if (rows > 8) rows = 8;
      if (rows > 0) mb_epi_tile(c2_0, dst0 + 16*cs + go[0], cs, rows, gc[0], e);
      if (rows > 0) mb_epi_tile(c2_1, dst0 + 16*cs + go[1], cs, rows, gc[1], e); }
    { int rows = rows_left - 24; if (rows > 8) rows = 8;
      if (rows > 0) mb_epi_tile(c3_0, dst0 + 24*cs + go[0], cs, rows, gc[0], e);
      if (rows > 0) mb_epi_tile(c3_1, dst0 + 24*cs + go[1], cs, rows, gc[1], e); }
}
static inline __attribute__((always_inline)) void
mb_blk_4x1(const int8_t *plane, const size_t *ab, const int32_t *off,
    int S, const int8_t *wp, size_t wstride, const int32_t *bexp,
    int8_t *dst0, size_t cs, const size_t *go, const int *gc, int rows_left, const mb_ep_t *e)
{
    vint32m2_t b0_0 = __riscv_vle32_v_i32m2(bexp + 0, 64);
    vint32m2_t b0_1 = __riscv_vle32_v_i32m2(bexp + 64, 64);
    vint32m2_t b0_2 = __riscv_vle32_v_i32m2(bexp + 128, 64);
    vint32m2_t b0_3 = __riscv_vle32_v_i32m2(bexp + 192, 64);
    vint32m2_t c0_0 = b0_0;
    vint32m2_t c1_0 = b0_1;
    vint32m2_t c2_0 = b0_2;
    vint32m2_t c3_0 = b0_3;
    const int8_t *bp0 = plane + ab[0];
    for (int s = 0; s < S; s++) {
        const size_t o = (size_t)off[s];
        const int8_t *ap = wp + (size_t)s * wstride;
        vint8m1_t a0 = __riscv_vle8_v_i8m1(ap + 0, 128);
        vint8m1_t a1 = __riscv_vle8_v_i8m1(ap + 128, 128);
        vint8m1_t a2 = __riscv_vle8_v_i8m1(ap + 256, 128);
        vint8m1_t a3 = __riscv_vle8_v_i8m1(ap + 384, 128);
        vint8m1_t b0 = __riscv_vle8_v_i8m1(bp0 + o, 128);
        c0_0 = mb_ime_vmadot(c0_0, a0, b0);
        c1_0 = mb_ime_vmadot(c1_0, a1, b0);
        c2_0 = mb_ime_vmadot(c2_0, a2, b0);
        c3_0 = mb_ime_vmadot(c3_0, a3, b0);
    }
    { int rows = rows_left - 0; if (rows > 8) rows = 8;
      if (rows > 0) mb_epi_tile(c0_0, dst0 + 0*cs + go[0], cs, rows, gc[0], e); }
    { int rows = rows_left - 8; if (rows > 8) rows = 8;
      if (rows > 0) mb_epi_tile(c1_0, dst0 + 8*cs + go[0], cs, rows, gc[0], e); }
    { int rows = rows_left - 16; if (rows > 8) rows = 8;
      if (rows > 0) mb_epi_tile(c2_0, dst0 + 16*cs + go[0], cs, rows, gc[0], e); }
    { int rows = rows_left - 24; if (rows > 8) rows = 8;
      if (rows > 0) mb_epi_tile(c3_0, dst0 + 24*cs + go[0], cs, rows, gc[0], e); }
}
static inline __attribute__((always_inline)) void
mb_blk_2x4(const int8_t *plane, const size_t *ab, const int32_t *off,
    int S, const int8_t *wp, size_t wstride, const int32_t *bexp,
    int8_t *dst0, size_t cs, const size_t *go, const int *gc, int rows_left, const mb_ep_t *e)
{
    vint32m2_t b0_0 = __riscv_vle32_v_i32m2(bexp + 0, 64);
    vint32m2_t b0_1 = __riscv_vle32_v_i32m2(bexp + 64, 64);
    vint32m2_t c0_0 = b0_0;
    vint32m2_t c0_1 = b0_0;
    vint32m2_t c0_2 = b0_0;
    vint32m2_t c0_3 = b0_0;
    vint32m2_t c1_0 = b0_1;
    vint32m2_t c1_1 = b0_1;
    vint32m2_t c1_2 = b0_1;
    vint32m2_t c1_3 = b0_1;
    const int8_t *bp0 = plane + ab[0];
    const int8_t *bp1 = plane + ab[1];
    const int8_t *bp2 = plane + ab[2];
    const int8_t *bp3 = plane + ab[3];
    for (int s = 0; s < S; s++) {
        const size_t o = (size_t)off[s];
        const int8_t *ap = wp + (size_t)s * wstride;
        vint8m1_t a0 = __riscv_vle8_v_i8m1(ap + 0, 128);
        vint8m1_t a1 = __riscv_vle8_v_i8m1(ap + 128, 128);
        vint8m1_t b0 = __riscv_vle8_v_i8m1(bp0 + o, 128);
        vint8m1_t b1 = __riscv_vle8_v_i8m1(bp1 + o, 128);
        vint8m1_t b2 = __riscv_vle8_v_i8m1(bp2 + o, 128);
        vint8m1_t b3 = __riscv_vle8_v_i8m1(bp3 + o, 128);
        c0_0 = mb_ime_vmadot(c0_0, a0, b0);
        c0_1 = mb_ime_vmadot(c0_1, a0, b1);
        c0_2 = mb_ime_vmadot(c0_2, a0, b2);
        c0_3 = mb_ime_vmadot(c0_3, a0, b3);
        c1_0 = mb_ime_vmadot(c1_0, a1, b0);
        c1_1 = mb_ime_vmadot(c1_1, a1, b1);
        c1_2 = mb_ime_vmadot(c1_2, a1, b2);
        c1_3 = mb_ime_vmadot(c1_3, a1, b3);
    }
    { int rows = rows_left - 0; if (rows > 8) rows = 8;
      if (rows > 0) mb_epi_tile(c0_0, dst0 + 0*cs + go[0], cs, rows, gc[0], e);
      if (rows > 0) mb_epi_tile(c0_1, dst0 + 0*cs + go[1], cs, rows, gc[1], e);
      if (rows > 0) mb_epi_tile(c0_2, dst0 + 0*cs + go[2], cs, rows, gc[2], e);
      if (rows > 0) mb_epi_tile(c0_3, dst0 + 0*cs + go[3], cs, rows, gc[3], e); }
    { int rows = rows_left - 8; if (rows > 8) rows = 8;
      if (rows > 0) mb_epi_tile(c1_0, dst0 + 8*cs + go[0], cs, rows, gc[0], e);
      if (rows > 0) mb_epi_tile(c1_1, dst0 + 8*cs + go[1], cs, rows, gc[1], e);
      if (rows > 0) mb_epi_tile(c1_2, dst0 + 8*cs + go[2], cs, rows, gc[2], e);
      if (rows > 0) mb_epi_tile(c1_3, dst0 + 8*cs + go[3], cs, rows, gc[3], e); }
}
static inline __attribute__((always_inline)) void
mb_blk_1x4(const int8_t *plane, const size_t *ab, const int32_t *off,
    int S, const int8_t *wp, size_t wstride, const int32_t *bexp,
    int8_t *dst0, size_t cs, const size_t *go, const int *gc, int rows_left, const mb_ep_t *e)
{
    vint32m2_t b0_0 = __riscv_vle32_v_i32m2(bexp + 0, 64);
    vint32m2_t c0_0 = b0_0;
    vint32m2_t c0_1 = b0_0;
    vint32m2_t c0_2 = b0_0;
    vint32m2_t c0_3 = b0_0;
    const int8_t *bp0 = plane + ab[0];
    const int8_t *bp1 = plane + ab[1];
    const int8_t *bp2 = plane + ab[2];
    const int8_t *bp3 = plane + ab[3];
    for (int s = 0; s < S; s++) {
        const size_t o = (size_t)off[s];
        const int8_t *ap = wp + (size_t)s * wstride;
        vint8m1_t a0 = __riscv_vle8_v_i8m1(ap + 0, 128);
        vint8m1_t b0 = __riscv_vle8_v_i8m1(bp0 + o, 128);
        vint8m1_t b1 = __riscv_vle8_v_i8m1(bp1 + o, 128);
        vint8m1_t b2 = __riscv_vle8_v_i8m1(bp2 + o, 128);
        vint8m1_t b3 = __riscv_vle8_v_i8m1(bp3 + o, 128);
        c0_0 = mb_ime_vmadot(c0_0, a0, b0);
        c0_1 = mb_ime_vmadot(c0_1, a0, b1);
        c0_2 = mb_ime_vmadot(c0_2, a0, b2);
        c0_3 = mb_ime_vmadot(c0_3, a0, b3);
    }
    { int rows = rows_left - 0; if (rows > 8) rows = 8;
      if (rows > 0) mb_epi_tile(c0_0, dst0 + 0*cs + go[0], cs, rows, gc[0], e);
      if (rows > 0) mb_epi_tile(c0_1, dst0 + 0*cs + go[1], cs, rows, gc[1], e);
      if (rows > 0) mb_epi_tile(c0_2, dst0 + 0*cs + go[2], cs, rows, gc[2], e);
      if (rows > 0) mb_epi_tile(c0_3, dst0 + 0*cs + go[3], cs, rows, gc[3], e); }
}
static inline __attribute__((always_inline)) void
mb_blk_2x2(const int8_t *plane, const size_t *ab, const int32_t *off,
    int S, const int8_t *wp, size_t wstride, const int32_t *bexp,
    int8_t *dst0, size_t cs, const size_t *go, const int *gc, int rows_left, const mb_ep_t *e)
{
    vint32m2_t b0_0 = __riscv_vle32_v_i32m2(bexp + 0, 64);
    vint32m2_t b0_1 = __riscv_vle32_v_i32m2(bexp + 64, 64);
    vint32m2_t c0_0 = b0_0;
    vint32m2_t c0_1 = b0_0;
    vint32m2_t c1_0 = b0_1;
    vint32m2_t c1_1 = b0_1;
    const int8_t *bp0 = plane + ab[0];
    const int8_t *bp1 = plane + ab[1];
    for (int s = 0; s < S; s++) {
        const size_t o = (size_t)off[s];
        const int8_t *ap = wp + (size_t)s * wstride;
        vint8m1_t a0 = __riscv_vle8_v_i8m1(ap + 0, 128);
        vint8m1_t a1 = __riscv_vle8_v_i8m1(ap + 128, 128);
        vint8m1_t b0 = __riscv_vle8_v_i8m1(bp0 + o, 128);
        vint8m1_t b1 = __riscv_vle8_v_i8m1(bp1 + o, 128);
        c0_0 = mb_ime_vmadot(c0_0, a0, b0);
        c0_1 = mb_ime_vmadot(c0_1, a0, b1);
        c1_0 = mb_ime_vmadot(c1_0, a1, b0);
        c1_1 = mb_ime_vmadot(c1_1, a1, b1);
    }
    { int rows = rows_left - 0; if (rows > 8) rows = 8;
      if (rows > 0) mb_epi_tile(c0_0, dst0 + 0*cs + go[0], cs, rows, gc[0], e);
      if (rows > 0) mb_epi_tile(c0_1, dst0 + 0*cs + go[1], cs, rows, gc[1], e); }
    { int rows = rows_left - 8; if (rows > 8) rows = 8;
      if (rows > 0) mb_epi_tile(c1_0, dst0 + 8*cs + go[0], cs, rows, gc[0], e);
      if (rows > 0) mb_epi_tile(c1_1, dst0 + 8*cs + go[1], cs, rows, gc[1], e); }
}
static inline __attribute__((always_inline)) void
mb_blk_1x2(const int8_t *plane, const size_t *ab, const int32_t *off,
    int S, const int8_t *wp, size_t wstride, const int32_t *bexp,
    int8_t *dst0, size_t cs, const size_t *go, const int *gc, int rows_left, const mb_ep_t *e)
{
    vint32m2_t b0_0 = __riscv_vle32_v_i32m2(bexp + 0, 64);
    vint32m2_t c0_0 = b0_0;
    vint32m2_t c0_1 = b0_0;
    const int8_t *bp0 = plane + ab[0];
    const int8_t *bp1 = plane + ab[1];
    for (int s = 0; s < S; s++) {
        const size_t o = (size_t)off[s];
        const int8_t *ap = wp + (size_t)s * wstride;
        vint8m1_t a0 = __riscv_vle8_v_i8m1(ap + 0, 128);
        vint8m1_t b0 = __riscv_vle8_v_i8m1(bp0 + o, 128);
        vint8m1_t b1 = __riscv_vle8_v_i8m1(bp1 + o, 128);
        c0_0 = mb_ime_vmadot(c0_0, a0, b0);
        c0_1 = mb_ime_vmadot(c0_1, a0, b1);
    }
    { int rows = rows_left - 0; if (rows > 8) rows = 8;
      if (rows > 0) mb_epi_tile(c0_0, dst0 + 0*cs + go[0], cs, rows, gc[0], e);
      if (rows > 0) mb_epi_tile(c0_1, dst0 + 0*cs + go[1], cs, rows, gc[1], e); }
}
static inline __attribute__((always_inline)) void
mb_blk_2x1(const int8_t *plane, const size_t *ab, const int32_t *off,
    int S, const int8_t *wp, size_t wstride, const int32_t *bexp,
    int8_t *dst0, size_t cs, const size_t *go, const int *gc, int rows_left, const mb_ep_t *e)
{
    vint32m2_t b0_0 = __riscv_vle32_v_i32m2(bexp + 0, 64);
    vint32m2_t b0_1 = __riscv_vle32_v_i32m2(bexp + 64, 64);
    vint32m2_t c0_0 = b0_0;
    vint32m2_t c1_0 = b0_1;
    const int8_t *bp0 = plane + ab[0];
    for (int s = 0; s < S; s++) {
        const size_t o = (size_t)off[s];
        const int8_t *ap = wp + (size_t)s * wstride;
        vint8m1_t a0 = __riscv_vle8_v_i8m1(ap + 0, 128);
        vint8m1_t a1 = __riscv_vle8_v_i8m1(ap + 128, 128);
        vint8m1_t b0 = __riscv_vle8_v_i8m1(bp0 + o, 128);
        c0_0 = mb_ime_vmadot(c0_0, a0, b0);
        c1_0 = mb_ime_vmadot(c1_0, a1, b0);
    }
    { int rows = rows_left - 0; if (rows > 8) rows = 8;
      if (rows > 0) mb_epi_tile(c0_0, dst0 + 0*cs + go[0], cs, rows, gc[0], e); }
    { int rows = rows_left - 8; if (rows > 8) rows = 8;
      if (rows > 0) mb_epi_tile(c1_0, dst0 + 8*cs + go[0], cs, rows, gc[0], e); }
}
static inline __attribute__((always_inline)) void
mb_blk_1x1(const int8_t *plane, const size_t *ab, const int32_t *off,
    int S, const int8_t *wp, size_t wstride, const int32_t *bexp,
    int8_t *dst0, size_t cs, const size_t *go, const int *gc, int rows_left, const mb_ep_t *e)
{
    vint32m2_t b0_0 = __riscv_vle32_v_i32m2(bexp + 0, 64);
    vint32m2_t c0_0 = b0_0;
    const int8_t *bp0 = plane + ab[0];
    for (int s = 0; s < S; s++) {
        const size_t o = (size_t)off[s];
        const int8_t *ap = wp + (size_t)s * wstride;
        vint8m1_t a0 = __riscv_vle8_v_i8m1(ap + 0, 128);
        vint8m1_t b0 = __riscv_vle8_v_i8m1(bp0 + o, 128);
        c0_0 = mb_ime_vmadot(c0_0, a0, b0);
    }
    { int rows = rows_left - 0; if (rows > 8) rows = 8;
      if (rows > 0) mb_epi_tile(c0_0, dst0 + 0*cs + go[0], cs, rows, gc[0], e); }
}

/* ---- scalar reference path (used when the fast path does not apply) ----- */
static void mb_ime_conv_scalar(
    const int8_t *input, const int8_t *weight, const int32_t *bias,
    const float *bn_scale, const float *bn_bias, int8_t *output,
    int N, int IC, int IH, int IW, int OC,
    int KH, int KW, int SH, int SW, int PH, int PW,
    int input_offset, int filter_offset, int conv_output_offset,
    int conv_output_multiplier, int conv_output_shift,
    int conv_amin, int conv_amax,
    float bn_scale_in, float bn_scale_out, int bn_amin, int bn_amax,
    const int8_t *silu_lut, int mode)
{
    int OH = (IH + 2 * PH - KH) / SH + 1;
    int OW = (IW + 2 * PW - KW) / SW + 1;
    for (int n = 0; n < N; n++)
    for (int oc = 0; oc < OC; oc++) {
        float bn_s = mode ? bn_scale[oc] : 0.f;
        float bn_b = mode ? bn_bias[oc] : 0.f;
        for (int oh = 0; oh < OH; oh++)
        for (int ow = 0; ow < OW; ow++) {
            int32_t acc = bias ? bias[oc] : 0;
            for (int ic = 0; ic < IC; ic++)
            for (int kh = 0; kh < KH; kh++) {
                int ih = oh * SH - PH + kh;
                for (int kw = 0; kw < KW; kw++) {
                    int iw = ow * SW - PW + kw;
                    int32_t in_v = (ih < 0 || ih >= IH || iw < 0 || iw >= IW)
                        ? input_offset
                        : (int32_t)input[(((size_t)n * IC + ic) * IH + ih) * IW + iw]
                              + input_offset;
                    int32_t w_v = (int32_t)weight[(((size_t)ic * KH + kh) * KW + kw) * OC + oc]
                                  + filter_offset;
                    acc += in_v * w_v;
                }
            }
            int64_t prod = (int64_t)acc * (int64_t)conv_output_multiplier;
            prod = (prod + (1LL << 30)) >> 31;
            int32_t scaled = (int32_t)prod;
            if (conv_output_shift > 0)
                scaled = (int32_t)(((int64_t)scaled + ((int64_t)1 << (conv_output_shift - 1)))
                                   >> conv_output_shift);
            else if (conv_output_shift < 0)
                scaled = scaled << (-conv_output_shift);
            scaled += conv_output_offset;
            if (scaled < conv_amin) scaled = conv_amin;
            if (scaled > conv_amax) scaled = conv_amax;
            int8_t r = (int8_t)scaled;
            if (mode >= 1) {
                float fv = (float)r * bn_scale_in;
                float y = bn_s * fv + bn_b;
                int32_t bv = (int32_t)roundf(y / bn_scale_out);
                if (bv < bn_amin) bv = bn_amin;
                if (bv > bn_amax) bv = bn_amax;
                r = (int8_t)bv;
                if (mode == 2) r = silu_lut[(int)r + 128];
            }
            output[((((size_t)n * OC + oc) * OH + oh) * OW) + ow] = r;
        }
    }
}


#ifdef MB_IME_PROF
#include <stdio.h>
static inline uint64_t mb_rdt(void){uint64_t t;__asm__ volatile("rdtime %0":"=r"(t));return t;}
#define MB_T(i) tt[i] = mb_rdt()
#else
#define MB_T(i) ((void)0)
#endif

/* ---- main driver -------------------------------------------------------- */
static __attribute__((noinline)) void mb_ime_conv_driver(
    const int8_t *input, const int8_t *weight, const int32_t *bias,
    const float *bn_scale, const float *bn_bias, int8_t *output,
    int N, int IC, int IH, int IW, int OC,
    int KH, int KW, int SH, int SW, int PH, int PW,
    int input_offset, int filter_offset, int conv_output_offset,
    int conv_output_multiplier, int conv_output_shift,
    int conv_amin, int conv_amax,
    float bn_scale_in, float bn_scale_out, int bn_amin, int bn_amax,
    float silu_scale_in, float silu_scale_out, int silu_amin, int silu_amax,
    int mode)
{
    uint64_t tt[8] = {0};
    MB_T(0);
    const int OH = (IH + 2 * PH - KH) / SH + 1;
    const int OW = (IW + 2 * PW - KW) / SW + 1;
    const int fast_ok =
        __riscv_vsetvlmax_e8m1() == 128 && __riscv_vsetvlmax_e32m8() == 256 &&
        (SW == 1 || SW == 2) && SH >= 1 &&
        input_offset == 0 && filter_offset == 0 && N >= 1 && IC >= 1 &&
        OC >= 1 && OH >= 1 && OW >= 1 && KH >= 1 && KW >= 1 &&
        (IH + 2 * PH) >= KH && (IW + 2 * PW) >= KW;

    const int gpr = (OW + 7) / 8;                 /* pixel groups per out row */
    const int G = OH * gpr;
    const int cgs = (IC + 15) / 16;
    const int S = cgs * KH * KW;
    const int ntile = (OC + 7) / 8;
    const size_t spatial = (size_t)OH * OW;

    int8_t *plane = NULL;
    int32_t *off = NULL;
    const int8_t *wp = NULL, *lut = NULL;
    const int nph = (SW == 2) ? 2 : 1;
    int Wph = (SW == 2) ? (IW + 2 * PW + 1) / 2 : IW + 2 * PW;
    if (Wph < gpr * 8 + KW) Wph = gpr * 8 + KW;
    const int Hp = IH + 2 * PH;
    const size_t planeBytes = (size_t)cgs * Hp * Wph * 16;

    if (fast_ok) {
        plane = (int8_t *)mb_scr_get(MB_SCR_PLANE, planeBytes * nph + 256);
        off = (int32_t *)mb_scr_get(MB_SCR_OFF, (size_t)S * 4 + 64);
        wp = mb_get_wp(weight, IC, OC, KH, KW, ntile);
        if (mode >= 1)
            lut = mb_get_lut(bn_scale, bn_bias, OC, mode, bn_scale_in, bn_scale_out,
                             bn_amin, bn_amax, silu_scale_in, silu_scale_out,
                             silu_amin, silu_amax);
    }
    if (!fast_ok || !plane || !off || !wp || (mode >= 1 && !lut)) {
        int8_t silu[256];
        if (mode == 2) mb_silu_fill(silu, silu_scale_in, silu_scale_out, silu_amin, silu_amax);
        mb_ime_conv_scalar(input, weight, bias, bn_scale, bn_bias, output, N, IC,
                           IH, IW, OC, KH, KW, SH, SW, PH, PW, input_offset,
                           filter_offset, conv_output_offset,
                           conv_output_multiplier, conv_output_shift, conv_amin,
                           conv_amax, bn_scale_in, bn_scale_out, bn_amin,
                           bn_amax, silu, mode);
        return;
    }
    MB_T(1);

    /* step offset table: s = (cg*KH + kh)*KW + kw */
    for (int cg = 0; cg < cgs; cg++)
        for (int kh = 0; kh < KH; kh++)
            for (int kw = 0; kw < KW; kw++) {
                int s = (cg * KH + kh) * KW + kw;
                size_t idx = (SW == 2) ? (size_t)(kw >> 1) : (size_t)kw;
                size_t o = (((size_t)cg * Hp + kh) * Wph + idx) * 16;
                if (SW == 2 && (kw & 1)) o += planeBytes;
                off[s] = (int32_t)o;
            }

    mb_ep_t ep;
    ep.mult = conv_output_multiplier;
    ep.shift = conv_output_shift;
    ep.coff = conv_output_offset;
    ep.amin = conv_amin;
    ep.amax = conv_amax;
    const size_t wstride = (size_t)ntile * 128;

    for (int n = 0; n < N; n++) {
        const int8_t *in_n = input + (size_t)n * IC * IH * IW;
        int8_t *out_n = output + (size_t)n * OC * spatial;
        const int8_t *in_end = in_n + (size_t)IC * IH * IW;

        /* ---- NHWC16 (zero padded) ---- */
        for (int cg = 0; cg < cgs; cg++) {
            const int nvc = IC - cg * 16 < 16 ? IC - cg * 16 : 16;
            const int8_t *cbase = in_n + (size_t)cg * 16 * IH * IW;
            for (int hp = 0; hp < Hp; hp++) {
                int8_t *d0 = plane + ((size_t)cg * Hp + hp) * Wph * 16;
                int8_t *d1 = d0 + planeBytes;       /* odd-column plane (SW==2) */
                if (hp < PH || hp >= PH + IH) {
                    memset(d0, 0, (size_t)Wph * 16);
                    if (SW == 2) memset(d1, 0, (size_t)Wph * 16);
                    continue;
                }
                /* pad columns: left PW px, right (Wph - PW - IW) px (per phase plane for SW==2) */
                if (SW == 1) {
                    if (PW) memset(d0, 0, (size_t)PW * 16);
                    memset(d0 + (size_t)(PW + IW) * 16, 0, (size_t)(Wph - PW - IW) * 16);
                } else {
                    memset(d0, 0, (size_t)Wph * 16);
                    memset(d1, 0, (size_t)Wph * 16);
                }
                const int8_t *rows[16];
                const int8_t *rp = cbase + (size_t)(hp - PH) * IW;
                for (int k = 0; k < 16; k++, rp += (size_t)IH * IW)
                    rows[k] = (k < nvc) ? rp : NULL;
                mb_il16(rows, IW, d0, d1, PW, SW == 2, in_end);
            }
        }
        MB_T(2);

        /* ---- GEMM over (oc tile blocks) x (pixel group blocks) ---- */
        for (int t0 = 0; t0 < ntile;) {
            int rt = ntile - t0;
            const int PT = rt >= 4 ? 4 : (rt >= 2 ? 2 : 1);

            int32_t bexp[4][64] __attribute__((aligned(64)));
            for (int t = 0; t < PT; t++) {
                int oc0 = (t0 + t) * 8;
                for (int i = 0; i < 8; i++) {
                    int32_t bv = (bias && oc0 + i < OC) ? bias[oc0 + i] : 0;
                    for (int j = 0; j < 8; j++) bexp[t][i * 8 + j] = bv;
                }
            }

            const int8_t *wpt = wp + (size_t)t0 * 128;
            int8_t *dst0 = out_n + (size_t)t0 * 8 * spatial;
            const int rows_left = OC - t0 * 8;

            for (int g = 0; g < G;) {
                int rg = G - g, PG;
                if (PT == 4) PG = rg >= 2 ? 2 : 1;
                else PG = rg >= 4 ? 4 : (rg >= 2 ? 2 : 1);
                size_t ab[4], go[4];
                int gc[4];
                for (int i = 0; i < PG; i++) {
                    int gg = g + i;
                    int oh = gg / gpr, ow0 = (gg - oh * gpr) * 8;
                    ab[i] = ((size_t)oh * SH * Wph + ow0) * 16;
                    go[i] = (size_t)oh * OW + ow0;
                    gc[i] = OW - ow0 < 8 ? OW - ow0 : 8;
                }
#define MB_CALL(T_, G_)                                                        \
    mb_blk_##T_##x##G_(plane, ab, off, S, wpt, wstride, &bexp[0][0], dst0, spatial, \
                       go, gc, rows_left, &ep)
                if (PT == 4) {
                    if (PG == 2) MB_CALL(4, 2); else MB_CALL(4, 1);
                } else if (PT == 2) {
                    if (PG == 4) MB_CALL(2, 4);
                    else if (PG == 2) MB_CALL(2, 2);
                    else MB_CALL(2, 1);
                } else {
                    if (PG == 4) MB_CALL(1, 4);
                    else if (PG == 2) MB_CALL(1, 2);
                    else MB_CALL(1, 1);
                }
#undef MB_CALL
                g += PG;
            }
            t0 += PT;
        }
        MB_T(3);

        /* ---- BN/SiLU table pass, in place over each channel plane ---- */
        if (mode >= 1) {
            for (int c = 0; c < OC; c++) {
                vuint8m1_t tlo = __riscv_vle8_v_u8m1((const uint8_t *)lut + (size_t)c * 256, 128);
                vuint8m1_t thi = __riscv_vle8_v_u8m1((const uint8_t *)lut + (size_t)c * 256 + 128, 128);
                uint8_t *p = (uint8_t *)out_n + (size_t)c * spatial;
                for (size_t x = 0; x < spatial; x += 128) {
                    size_t vl = spatial - x < 128 ? spatial - x : 128;
                    vuint8m1_t v = __riscv_vle8_v_u8m1(p + x, vl);
                    vuint8m1_t ix = __riscv_vxor_vx_u8m1(v, 0x80, vl);
                    vuint8m1_t lo = __riscv_vrgather_vv_u8m1(tlo, ix, vl);
                    vuint8m1_t hi = __riscv_vrgather_vv_u8m1(
                        thi, __riscv_vsub_vx_u8m1(ix, 128, vl), vl);
                    __riscv_vse8_v_u8m1(p + x, __riscv_vor_vv_u8m1(lo, hi, vl), vl);
                }
            }
        }
        MB_T(4);
    }
#ifdef MB_IME_PROF
    fprintf(stderr, "PROF setup %llu nhwc %llu gemm %llu pass %llu (ticks 41.7ns)\n",
            (unsigned long long)(tt[1]-tt[0]), (unsigned long long)(tt[2]-tt[1]),
            (unsigned long long)(tt[3]-tt[2]), (unsigned long long)(tt[4]-tt[3]));
#endif
}

#pragma GCC pop_options
#endif /* MB_IME_K3_A100_COMMON_H */
