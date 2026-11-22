// Shared int8 GEMM core of linear_s8_dot.comp / matmul_s8_dot.comp (the dot4 path):
//   acc[i][j] = sum_k (x[m, k] + io) * (w[n, k] + fo),  x [M, K], w [N, K] row-major int8
// with packed 4 x int8 dot products (VK_KHR_shader_integer_dot_product: one
// instruction per 4 multiply-adds on RDNA2). The offsets are folded in exactly:
//   sum (x + io)(w + fo) = sum xw + fo * sum x + io * sum w + K * io * fo
// in int32 (two's-complement wraparound, as the reference's int32 accumulator),
// the sums of x and w being dot products with 0x01010101.
// A 256-wide workgroup computes a BM x BN output tile (32 x 64 by default; DOT_RM / DOT_RN /
// DOT_BKQ select others): K moves through LDS 4 * BKQ bytes at a time, loaded coalesced (zero-filled past
// K and clamped past M / N); invocation (ty, tx) owns rows 2ty.. and columns
// 4tx... K % 4 == 0 and 4-byte aligned operands (launcher-checked).
#extension GL_EXT_integer_dot_product : require
#extension GL_EXT_control_flow_attributes : require
#ifndef DOT_RM
#define DOT_RM 2
#endif
#ifndef DOT_RN
#define DOT_RN 4
#endif
#ifndef DOT_BKQ
#define DOT_BKQ 16
#endif
// a 16 x 16 thread grid; invocation (ty, tx) owns rows ty*RM.. and columns tx*RN.. of the tile
const int RM = DOT_RM, RN = DOT_RN, BM = 16 * RM, BN = 16 * RN, BKQ = DOT_BKQ;
layout(local_size_x = 256) in;
shared int Xs[BM][BKQ + 1];
shared int Ws[BN][BKQ + 1];
void dot_s8(I32 x, I32 w, int M, int K, int N, int io, int fo, out int acc[RM][RN], out int m0, out int n0) {
    int tid = int(gl_LocalInvocationID.x), tx = tid & 15, ty = tid >> 4;
    int bm = int(gl_WorkGroupID.y) * BM, bn = int(gl_WorkGroupID.x) * BN;
    m0 = bm + ty * RM;
    n0 = bn + tx * RN;
    int K4 = K >> 2;
    int sx[RM], sw[RN];
    [[unroll]] for (int i = 0; i < RM; i++) { sx[i] = 0; [[unroll]] for (int j = 0; j < RN; j++) acc[i][j] = 0; }
    [[unroll]] for (int j = 0; j < RN; j++) sw[j] = 0;
    const int ones = 0x01010101;
    for (int k0 = 0; k0 < K4; k0 += BKQ) {
        [[unroll]] for (int l = 0; l < BM * BKQ / 256; l++) {
            int e = tid + 256 * l, r = e / BKQ, q = e % BKQ;
            Xs[r][q] = k0 + q < K4 ? x.v[min(bm + r, M - 1) * K4 + k0 + q] : 0;
        }
        [[unroll]] for (int l = 0; l < BN * BKQ / 256; l++) {
            int e = tid + 256 * l, r = e / BKQ, q = e % BKQ;
            Ws[r][q] = k0 + q < K4 ? w.v[min(bn + r, N - 1) * K4 + k0 + q] : 0;
        }
        barrier();
        [[unroll]] for (int q = 0; q < BKQ; q++) {
            int xa[RM], wb[RN];
            [[unroll]] for (int i = 0; i < RM; i++) xa[i] = Xs[ty * RM + i][q];
            [[unroll]] for (int j = 0; j < RN; j++) wb[j] = Ws[tx * RN + j][q];
            [[unroll]] for (int i = 0; i < RM; i++)
                [[unroll]] for (int j = 0; j < RN; j++) acc[i][j] += dotPacked4x8EXT(xa[i], wb[j]);
            if (fo != 0) { [[unroll]] for (int i = 0; i < RM; i++) sx[i] += dotPacked4x8EXT(xa[i], ones); }
            if (io != 0) { [[unroll]] for (int j = 0; j < RN; j++) sw[j] += dotPacked4x8EXT(wb[j], ones); }
        }
        barrier();
    }
    int kio = K * io * fo;
    [[unroll]] for (int i = 0; i < RM; i++)
        [[unroll]] for (int j = 0; j < RN; j++) acc[i][j] += fo * sx[i] + io * sw[j] + kio;
}
