/* source: curated */
/* algorithm: direct */
/* origin: K3 upsample_nearest_s8: scale 2 as one widening per input row (LMUL 1/2);
 * every other scale is the stock rvv kernel.
 * k3-work TASK-017. Same text in kernels/rvv_k3/ (X100) and kernels/rvv_k3_a100/ (A100); the
 * register group is picked from VLEN at run time. */

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <riscv_vector.h>

void kernel_upsample_nearest_s8(const int8_t *input, int8_t *output,
                                int N, int C, int IH, int IW, int scale)
{
    int OH = IH * scale, OW = IW * scale;
    if (N <= 0 || C <= 0 || IH <= 0 || IW <= 0 || scale <= 0) return;
    if (scale == 2) {
        /* NCHW rows are consecutive across planes: input row r (of N*C*IH) becomes
         * output rows 2r and 2r+1. Each input byte is doubled with one widening:
         * zero-extend to 16 bits and multiply by 257 (v | v << 8, little-endian =
         * the byte twice), then the same register is stored to both rows. Every
         * instruction is LMUL 1 (input) / 2 (output): the stock kernel issued an m8
         * vmv + store per input PIXEL, and both K3 cores charge m8 for that. */
        size_t rows = (size_t)N * (size_t)C * (size_t)IH;
        for (size_t r = 0; r < rows; r++) {
            const uint8_t *src = (const uint8_t *)input + r * (size_t)IW;
            uint8_t *d0 = (uint8_t *)output + 2 * r * (size_t)OW;
            uint8_t *d1 = d0 + OW;
            for (int x = 0; x < IW; ) {
                /* One element count handed to every width (see rvv_add_s8's note on
                 * chaining a vsetvl result into the next vsetvl with this GCC). */
                const size_t n_elem = (size_t)(IW - x);
                size_t vl8 = __riscv_vsetvl_e8m1(n_elem);
                vuint8m1_t v = __riscv_vle8_v_u8m1(src + x, vl8);
                size_t vl16 = __riscv_vsetvl_e16m2(n_elem);
                vuint16m2_t w = __riscv_vmul_vx_u16m2(__riscv_vzext_vf2_u16m2(v, vl16), 257, vl16);
                size_t vlo = __riscv_vsetvl_e8m2(2 * n_elem);
                vuint8m2_t b = __riscv_vreinterpret_v_u16m2_u8m2(w);
                __riscv_vse8_v_u8m2(d0 + 2 * (size_t)x, b, vlo);
                __riscv_vse8_v_u8m2(d1 + 2 * (size_t)x, b, vlo);
                x += (int)vl8;
            }
        }
        return;
    }
    /* Any other scale: the stock kernel (kernels/rvv/rvv_upsample_nearest_s8_direct.c). */
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const int8_t *in_plane  = input  + (n*C + c)*IH*IW;
            int8_t       *out_plane = output + (n*C + c)*OH*OW;
            for (int ih = 0; ih < IH; ih++) {
                const int8_t *in_row = in_plane + ih * IW;
                int8_t *out_row0 = out_plane + ih * scale * OW;
                int ow = 0;
                for (int iw = 0; iw < IW; iw++) {
                    int8_t val = in_row[iw];
                    int rem = scale;
                    size_t vl;
                    for (int s = 0; s < rem; ) {
                        vl = __riscv_vsetvl_e8m8(rem - s);
                        vint8m8_t vval = __riscv_vmv_v_x_i8m8(val, vl);
                        __riscv_vse8_v_i8m8(out_row0 + ow + s, vval, vl);
                        s += vl;
                    }
                    ow += scale;
                }
                for (int sr = 1; sr < scale; sr++) {
                    int8_t *out_rowN = out_row0 + sr * OW;
                    int rem = OW;
                    int off = 0;
                    size_t vl;
                    while (rem > 0) {
                        vl = __riscv_vsetvl_e8m8(rem);
                        vint8m8_t v = __riscv_vle8_v_i8m8(out_row0 + off, vl);
                        __riscv_vse8_v_i8m8(out_rowN + off, v, vl);
                        off += vl;
                        rem -= vl;
                    }
                }
            }
        }
    }
}
