/* Per-channel ("per-filter") twin of the A100 IME conv2d_s8 (k3-work TASK-019,
 * 2026-11-17): conv2d_s8_pc, one multiplier and one shift per output channel.
 * ime_k3_a100_common.h stays unchanged and supplies everything that does not touch the
 * scaling step (NHWC16 transpose, weight packing and its cache, scratch, scalar helpers);
 * this file adds copies of its 8 register-block functions and its driver whose only
 * difference is the epilogue: mb_epi_tile_pc reads, per lane, the multiplier and the
 * three shift amounts that mb_epi_tile chooses between per call (mb_pc_t, built per block
 * of <= 4 channel tiles like the bias lanes: lane i*8 + j is channel i of the tile).
 * A100 ONLY (CPUs 8-15, VLEN=1024). */
#ifndef MB_IME_K3_A100_CONV_PC_H
#define MB_IME_K3_A100_CONV_PC_H

/* the per-layer driver and scalar path are unused in a kernels.c that has only this kernel */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "ime_k3_a100_common.h"
#pragma GCC diagnostic pop

#pragma GCC push_options
#pragma GCC optimize("no-strict-aliasing,no-tree-vectorize,no-tree-slp-vectorize")

/* Per lane of a block's <= 4 channel tiles (lane i*8 + j = channel i of tile t): the
 * multiplier, the left shift (shift < 0), the extra right shift 31 (shift >= 32) and the
 * vnclip shift -- the three branches of mb_epi_tile -- so that every lane takes its own
 * channel's branch. */
typedef struct {
    int32_t m[4][64];
    uint32_t l[4][64], a[4][64];
    uint16_t n[4][64];
} mb_pc_t;

/* mb_epi_tile with tile t's per-lane multiplier and shifts from pq. */
static inline __attribute__((always_inline)) void
mb_epi_tile_pc(vint32m2_t acc, int8_t *dst, size_t cs, int rows, int cnt,
               const mb_ep_t *e, const mb_pc_t *pq, int t)
{
    const size_t vl = 64;
    /* per lane, the three branches of mb_epi_tile at once: vsll by the left shift (shift < 0),
     * vsra by 31 (shift >= 32), vnclip by the remaining right shift; the amounts a lane's
     * channel does not use are 0 (no-ops) */
    vint32m2_t s = __riscv_vsmul_vv_i32m2(acc, __riscv_vle32_v_i32m2(pq->m[t], vl),
                                          __RISCV_VXRM_RNU, vl);
    s = __riscv_vsll_vv_i32m2(s, __riscv_vle32_v_u32m2(pq->l[t], vl), vl);
    s = __riscv_vsra_vv_i32m2(s, __riscv_vle32_v_u32m2(pq->a[t], vl), vl);
    vint16m1_t o16 = __riscv_vnclip_wv_i16m1(s, __riscv_vle16_v_u16m1(pq->n[t], vl),
                                             __RISCV_VXRM_RNU, vl);
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

/* ---- the PT x PG register blocks of ime_k3_a100_common.h, with the per-channel epilogue
 * (mb_epi_tile_pc for tile T of the block) ---- */
static inline __attribute__((always_inline)) void
mb_blkpc_4x2(const int8_t *plane, const size_t *ab, const int32_t *off,
    int S, const int8_t *wp, size_t wstride, const int32_t *bexp,
    int8_t *dst0, size_t cs, const size_t *go, const int *gc, int rows_left, const mb_ep_t *e,
    const mb_pc_t *q)
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
      if (rows > 0) mb_epi_tile_pc(c0_0, dst0 + 0*cs + go[0], cs, rows, gc[0], e, q, 0);
      if (rows > 0) mb_epi_tile_pc(c0_1, dst0 + 0*cs + go[1], cs, rows, gc[1], e, q, 0); }
    { int rows = rows_left - 8; if (rows > 8) rows = 8;
      if (rows > 0) mb_epi_tile_pc(c1_0, dst0 + 8*cs + go[0], cs, rows, gc[0], e, q, 1);
      if (rows > 0) mb_epi_tile_pc(c1_1, dst0 + 8*cs + go[1], cs, rows, gc[1], e, q, 1); }
    { int rows = rows_left - 16; if (rows > 8) rows = 8;
      if (rows > 0) mb_epi_tile_pc(c2_0, dst0 + 16*cs + go[0], cs, rows, gc[0], e, q, 2);
      if (rows > 0) mb_epi_tile_pc(c2_1, dst0 + 16*cs + go[1], cs, rows, gc[1], e, q, 2); }
    { int rows = rows_left - 24; if (rows > 8) rows = 8;
      if (rows > 0) mb_epi_tile_pc(c3_0, dst0 + 24*cs + go[0], cs, rows, gc[0], e, q, 3);
      if (rows > 0) mb_epi_tile_pc(c3_1, dst0 + 24*cs + go[1], cs, rows, gc[1], e, q, 3); }
}
static inline __attribute__((always_inline)) void
mb_blkpc_4x1(const int8_t *plane, const size_t *ab, const int32_t *off,
    int S, const int8_t *wp, size_t wstride, const int32_t *bexp,
    int8_t *dst0, size_t cs, const size_t *go, const int *gc, int rows_left, const mb_ep_t *e,
    const mb_pc_t *q)
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
      if (rows > 0) mb_epi_tile_pc(c0_0, dst0 + 0*cs + go[0], cs, rows, gc[0], e, q, 0); }
    { int rows = rows_left - 8; if (rows > 8) rows = 8;
      if (rows > 0) mb_epi_tile_pc(c1_0, dst0 + 8*cs + go[0], cs, rows, gc[0], e, q, 1); }
    { int rows = rows_left - 16; if (rows > 8) rows = 8;
      if (rows > 0) mb_epi_tile_pc(c2_0, dst0 + 16*cs + go[0], cs, rows, gc[0], e, q, 2); }
    { int rows = rows_left - 24; if (rows > 8) rows = 8;
      if (rows > 0) mb_epi_tile_pc(c3_0, dst0 + 24*cs + go[0], cs, rows, gc[0], e, q, 3); }
}
static inline __attribute__((always_inline)) void
mb_blkpc_2x4(const int8_t *plane, const size_t *ab, const int32_t *off,
    int S, const int8_t *wp, size_t wstride, const int32_t *bexp,
    int8_t *dst0, size_t cs, const size_t *go, const int *gc, int rows_left, const mb_ep_t *e,
    const mb_pc_t *q)
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
      if (rows > 0) mb_epi_tile_pc(c0_0, dst0 + 0*cs + go[0], cs, rows, gc[0], e, q, 0);
      if (rows > 0) mb_epi_tile_pc(c0_1, dst0 + 0*cs + go[1], cs, rows, gc[1], e, q, 0);
      if (rows > 0) mb_epi_tile_pc(c0_2, dst0 + 0*cs + go[2], cs, rows, gc[2], e, q, 0);
      if (rows > 0) mb_epi_tile_pc(c0_3, dst0 + 0*cs + go[3], cs, rows, gc[3], e, q, 0); }
    { int rows = rows_left - 8; if (rows > 8) rows = 8;
      if (rows > 0) mb_epi_tile_pc(c1_0, dst0 + 8*cs + go[0], cs, rows, gc[0], e, q, 1);
      if (rows > 0) mb_epi_tile_pc(c1_1, dst0 + 8*cs + go[1], cs, rows, gc[1], e, q, 1);
      if (rows > 0) mb_epi_tile_pc(c1_2, dst0 + 8*cs + go[2], cs, rows, gc[2], e, q, 1);
      if (rows > 0) mb_epi_tile_pc(c1_3, dst0 + 8*cs + go[3], cs, rows, gc[3], e, q, 1); }
}
static inline __attribute__((always_inline)) void
mb_blkpc_1x4(const int8_t *plane, const size_t *ab, const int32_t *off,
    int S, const int8_t *wp, size_t wstride, const int32_t *bexp,
    int8_t *dst0, size_t cs, const size_t *go, const int *gc, int rows_left, const mb_ep_t *e,
    const mb_pc_t *q)
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
      if (rows > 0) mb_epi_tile_pc(c0_0, dst0 + 0*cs + go[0], cs, rows, gc[0], e, q, 0);
      if (rows > 0) mb_epi_tile_pc(c0_1, dst0 + 0*cs + go[1], cs, rows, gc[1], e, q, 0);
      if (rows > 0) mb_epi_tile_pc(c0_2, dst0 + 0*cs + go[2], cs, rows, gc[2], e, q, 0);
      if (rows > 0) mb_epi_tile_pc(c0_3, dst0 + 0*cs + go[3], cs, rows, gc[3], e, q, 0); }
}
static inline __attribute__((always_inline)) void
mb_blkpc_2x2(const int8_t *plane, const size_t *ab, const int32_t *off,
    int S, const int8_t *wp, size_t wstride, const int32_t *bexp,
    int8_t *dst0, size_t cs, const size_t *go, const int *gc, int rows_left, const mb_ep_t *e,
    const mb_pc_t *q)
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
      if (rows > 0) mb_epi_tile_pc(c0_0, dst0 + 0*cs + go[0], cs, rows, gc[0], e, q, 0);
      if (rows > 0) mb_epi_tile_pc(c0_1, dst0 + 0*cs + go[1], cs, rows, gc[1], e, q, 0); }
    { int rows = rows_left - 8; if (rows > 8) rows = 8;
      if (rows > 0) mb_epi_tile_pc(c1_0, dst0 + 8*cs + go[0], cs, rows, gc[0], e, q, 1);
      if (rows > 0) mb_epi_tile_pc(c1_1, dst0 + 8*cs + go[1], cs, rows, gc[1], e, q, 1); }
}
static inline __attribute__((always_inline)) void
mb_blkpc_1x2(const int8_t *plane, const size_t *ab, const int32_t *off,
    int S, const int8_t *wp, size_t wstride, const int32_t *bexp,
    int8_t *dst0, size_t cs, const size_t *go, const int *gc, int rows_left, const mb_ep_t *e,
    const mb_pc_t *q)
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
      if (rows > 0) mb_epi_tile_pc(c0_0, dst0 + 0*cs + go[0], cs, rows, gc[0], e, q, 0);
      if (rows > 0) mb_epi_tile_pc(c0_1, dst0 + 0*cs + go[1], cs, rows, gc[1], e, q, 0); }
}
static inline __attribute__((always_inline)) void
mb_blkpc_2x1(const int8_t *plane, const size_t *ab, const int32_t *off,
    int S, const int8_t *wp, size_t wstride, const int32_t *bexp,
    int8_t *dst0, size_t cs, const size_t *go, const int *gc, int rows_left, const mb_ep_t *e,
    const mb_pc_t *q)
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
      if (rows > 0) mb_epi_tile_pc(c0_0, dst0 + 0*cs + go[0], cs, rows, gc[0], e, q, 0); }
    { int rows = rows_left - 8; if (rows > 8) rows = 8;
      if (rows > 0) mb_epi_tile_pc(c1_0, dst0 + 8*cs + go[0], cs, rows, gc[0], e, q, 1); }
}
static inline __attribute__((always_inline)) void
mb_blkpc_1x1(const int8_t *plane, const size_t *ab, const int32_t *off,
    int S, const int8_t *wp, size_t wstride, const int32_t *bexp,
    int8_t *dst0, size_t cs, const size_t *go, const int *gc, int rows_left, const mb_ep_t *e,
    const mb_pc_t *q)
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
      if (rows > 0) mb_epi_tile_pc(c0_0, dst0 + 0*cs + go[0], cs, rows, gc[0], e, q, 0); }
}

/* The scalar path of conv2d_s8_pc (plain conv, per-channel multiplier and shift). */
static void mb_ime_conv_scalar_pc(
    const int8_t *input, const int8_t *weight, const int32_t *bias, int8_t *output,
    int N, int IC, int IH, int IW, int OC,
    int KH, int KW, int SH, int SW, int PH, int PW,
    int input_offset, int filter_offset, int output_offset,
    const int32_t *output_multiplier, const int32_t *output_shift,
    int amin, int amax)
{
    int OH = (IH + 2 * PH - KH) / SH + 1;
    int OW = (IW + 2 * PW - KW) / SW + 1;
    for (int n = 0; n < N; n++)
    for (int oc = 0; oc < OC; oc++) {
        const int32_t mult = output_multiplier[oc], shift = output_shift[oc];
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
            int64_t prod = (int64_t)acc * (int64_t)mult;
            prod = (prod + (1LL << 30)) >> 31;
            int32_t scaled = (int32_t)prod;
            if (shift > 62)
                scaled = 0;
            else if (shift > 0)
                scaled = (int32_t)(((int64_t)scaled + ((int64_t)1 << (shift - 1))) >> shift);
            else if (shift < 0)
                scaled = scaled << (-shift);
            scaled += output_offset;
            if (scaled < amin) scaled = amin;
            if (scaled > amax) scaled = amax;
            output[((((size_t)n * OC + oc) * OH + oh) * OW) + ow] = (int8_t)scaled;
        }
    }
}

/* ---- mb_ime_conv_driver with per-channel lists (mode 0, plain conv, only) -------- */
static __attribute__((noinline)) void mb_ime_conv_driver_pc(
    const int8_t *input, const int8_t *weight, const int32_t *bias,
    const float *bn_scale, const float *bn_bias, int8_t *output,
    int N, int IC, int IH, int IW, int OC,
    int KH, int KW, int SH, int SW, int PH, int PW,
    int input_offset, int filter_offset, int conv_output_offset,
    const int32_t *pc_mult, const int32_t *pc_shift,
    int conv_amin, int conv_amax,
    float bn_scale_in, float bn_scale_out, int bn_amin, int bn_amax,
    float silu_scale_in, float silu_scale_out, int silu_amin, int silu_amax,
    int mode)
{
    uint64_t tt[8] = {0};
    (void)tt;
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
        mb_ime_conv_scalar_pc(input, weight, bias, output, N, IC, IH, IW, OC, KH, KW,
                              SH, SW, PH, PW, input_offset, filter_offset,
                              conv_output_offset, pc_mult, pc_shift, conv_amin, conv_amax);
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
    ep.mult = 0;     /* per channel: pcq below */
    ep.shift = 0;
    ep.coff = conv_output_offset;
    ep.amin = conv_amin;
    ep.amax = conv_amax;
    const size_t wstride = (size_t)ntile * 128;
    mb_pc_t pcq __attribute__((aligned(64)));

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
            /* the same lanes' multiplier and shift amounts, as mb_epi_tile would choose
             * them from this channel's shift */
            for (int t = 0; t < PT; t++) {
                int oc0 = (t0 + t) * 8;
                for (int i = 0; i < 8; i++) {
                    int ok = oc0 + i < OC;
                    int32_t m = ok ? pc_mult[oc0 + i] : 0, sh = ok ? pc_shift[oc0 + i] : 0;
                    uint32_t l = sh < 0 ? 0u - (uint32_t)sh : 0u;
                    uint32_t a = sh >= 32 ? 31u : 0u;
                    int nc = sh < 0 ? 0 : (sh < 32 ? sh : (sh - 31 > 31 ? 31 : sh - 31));
                    for (int j = 0; j < 8; j++) {
                        pcq.m[t][i * 8 + j] = m;
                        pcq.l[t][i * 8 + j] = l;
                        pcq.a[t][i * 8 + j] = a;
                        pcq.n[t][i * 8 + j] = (uint16_t)nc;
                    }
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
    mb_blkpc_##T_##x##G_(plane, ab, off, S, wpt, wstride, &bexp[0][0], dst0, spatial, \
                       go, gc, rows_left, &ep, &pcq)
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
#endif /* MB_IME_K3_A100_CONV_PC_H */
