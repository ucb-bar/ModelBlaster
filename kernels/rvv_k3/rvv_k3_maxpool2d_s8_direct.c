/* source: curated */
/* algorithm: direct */
/* origin: K3 maxpool2d_s8: the TASK-013 A100 algorithm with LMUL-1 vectors.
 * k3-work TASK-017. Same text in kernels/rvv_k3/ (X100) and kernels/rvv_k3_a100/ (A100); the
 * register group is picked from VLEN at run time. */

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <riscv_vector.h>

void kernel_maxpool2d_s8(const int8_t *input, int8_t *output,
                         int N, int C, int IH, int IW,
                         int KH, int KW, int SH, int SW,
                         int PH, int PW, int DH, int DW)
{
    int OH = (IH + 2*PH - DH*(KH-1) - 1) / SH + 1;
    int OW = (IW + 2*PW - DW*(KW-1) - 1) / SW + 1;
    if (N <= 0 || C <= 0 || OH <= 0 || OW <= 0) return;
    /* The A100 kernel's algorithm (kernels/rvv_k3_a100/, TASK-013): per (n, c, oh),
     * (1) vertical max over the valid kh rows into a -128-padded scratch row,
     * (2) horizontal window max from the scratch row with unit-stride loads only
     * (stride 2: load 2*vl bytes, de-interleave with vnsrl). Changed: every
     * instruction is LMUL 1 (the de-interleave load LMUL 2) instead of 8/4. Both
     * K3 cores charge by register group, not by vl (TASK-017 lmul_probe), and these
     * rows are 5-56 bytes. Max over the in-bounds taps; INT8_MIN for an all-padding
     * window, as the reference. */
    /* Stride 1, no dilation, "same" output (the SPPF pools): several whole planes per
     * LMUL-1 vector. Separable max with the window's edge taps masked off: horizontal
     * from the input (masked loads touch no byte outside the image), vertical from a
     * scratch copy of the chunk. Chunks start at plane boundaries, so one set of masks,
     * built per call, serves every chunk. One store->reload per chunk instead of per row
     * (the A100 pays dearly for reloading a line it just stored). */
    {
        size_t vlm = __riscv_vsetvlmax_e8m1();
        size_t P = (size_t)IH * (size_t)IW;
        if (SH == 1 && SW == 1 && DH == 1 && DW == 1 && OH == IH && OW == IW
                && P <= vlm && vlm <= 1024 && KH <= 16 && KW <= 16
                && PH * IW <= 64 && (KH - 1 - PH) * IW <= 64) {
            size_t G = vlm / P, L = G * P;              /* planes and bytes per chunk */
            uint8_t mw[16][128], mh[16][128], tmp[1024];
            int8_t hs[64 + 1024 + 64];   /* the chunk at hs + 64, room for the row offsets */
            size_t vlL = __riscv_vsetvl_e8m1(L);
            for (int d = 0; d < KW; d++) {
                for (size_t j = 0; j < L; j++) { int w = (int)(j % IW) + d - PW; tmp[j] = (w >= 0 && w < IW); }
                __riscv_vsm_v_b8(mw[d], __riscv_vmsne_vx_u8m1_b8(__riscv_vle8_v_u8m1(tmp, vlL), 0, vlL), vlL);
            }
            for (int d = 0; d < KH; d++) {
                for (size_t j = 0; j < L; j++) { int h = (int)((j / IW) % IH) + d - PH; tmp[j] = (h >= 0 && h < IH); }
                __riscv_vsm_v_b8(mh[d], __riscv_vmsne_vx_u8m1_b8(__riscv_vle8_v_u8m1(tmp, vlL), 0, vlL), vlL);
            }
            size_t total = (size_t)N * (size_t)C * P;
            for (size_t base = 0; base < total; base += L) {
                size_t vl = __riscv_vsetvl_e8m1(total - base < L ? total - base : L);
                vint8m1_t neg = __riscv_vmv_v_x_i8m1((int8_t)-128, vl);
                vint8m1_t acc = neg;
                for (int d = 0; d < KW; d++) {
                    vbool8_t m = __riscv_vlm_v_b8(mw[d], vl);
                    acc = __riscv_vmax_vv_i8m1(acc, __riscv_vle8_v_i8m1_mu(m, neg, input + base + d - PW, vl), vl);
                }
                __riscv_vse8_v_i8m1(hs + 64, acc, vl);
                acc = neg;
                for (int d = 0; d < KH; d++) {
                    vbool8_t m = __riscv_vlm_v_b8(mh[d], vl);
                    acc = __riscv_vmax_vv_i8m1(acc, __riscv_vle8_v_i8m1_mu(m, neg, hs + 64 + (ptrdiff_t)(d - PH) * IW, vl), vl);
                }
                __riscv_vse8_v_i8m1(output + base, acc, vl);
            }
            return;
        }
    }
    int PWID = IW + 2*PW + 2*SW*DW + KW*DW + 8;
    int8_t row[PWID];
    for (int i = 0; i < PWID; i++) row[i] = -128;

    for (int nc = 0; nc < N*C; nc++) {
        const int8_t *in_nc = input + (size_t)nc*IH*IW;
        int8_t *out_nc = output + (size_t)nc*OH*OW;

        for (int oh = 0; oh < OH; oh++) {
            for (int x = 0; x < IW; ) {
                size_t vl = __riscv_vsetvl_e8m1((size_t)(IW - x));
                vint8m1_t acc = __riscv_vmv_v_x_i8m1((int8_t)-128, vl);
                for (int kh = 0; kh < KH; kh++) {
                    int ih = oh*SH - PH + kh*DH;
                    if (ih < 0 || ih >= IH) continue;
                    acc = __riscv_vmax_vv_i8m1(acc, __riscv_vle8_v_i8m1(in_nc + (size_t)ih*IW + x, vl), vl);
                }
                __riscv_vse8_v_i8m1(row + PW + x, acc, vl);
                x += (int)vl;
            }
            int8_t *out_row = out_nc + (size_t)oh*OW;
            for (int ow = 0; ow < OW; ) {
                const size_t n_elem = (size_t)(OW - ow);
                size_t vl = __riscv_vsetvl_e8m1(n_elem);
                vint8m1_t acc = __riscv_vmv_v_x_i8m1((int8_t)-128, vl);
                if (SW == 1) {
                    for (int kw = 0; kw < KW; kw++)
                        acc = __riscv_vmax_vv_i8m1(acc, __riscv_vle8_v_i8m1(row + ow + kw*DW, vl), vl);
                } else if (SW == 2) {
                    for (int kw = 0; kw < KW; kw++) {
                        int pair = (DW == 1) && (kw + 1 < KW);
                        size_t vb = __riscv_vsetvl_e8m2(2 * n_elem);
                        vint8m2_t raw = __riscv_vle8_v_i8m2(row + 2*ow + kw*DW, vb);
                        vuint16m2_t w = __riscv_vreinterpret_v_u8m2_u16m2(__riscv_vreinterpret_v_i8m2_u8m2(raw));
                        size_t vn = __riscv_vsetvl_e8m1(n_elem);
                        vuint8m1_t ev = __riscv_vnsrl_wx_u8m1(w, 0, vn);
                        acc = __riscv_vmax_vv_i8m1(acc, __riscv_vreinterpret_v_u8m1_i8m1(ev), vn);
                        if (pair) {
                            vuint8m1_t od = __riscv_vnsrl_wx_u8m1(w, 8, vn);
                            acc = __riscv_vmax_vv_i8m1(acc, __riscv_vreinterpret_v_u8m1_i8m1(od), vn);
                            kw++;
                        }
                    }
                } else {
                    for (int kw = 0; kw < KW; kw++)
                        acc = __riscv_vmax_vv_i8m1(acc,
                                  __riscv_vlse8_v_i8m1(row + ow*SW + kw*DW, SW, vl), vl);
                }
                __riscv_vse8_v_i8m1(out_row + ow, acc, vl);
                ow += (int)vl;
            }
        }
    }
}
