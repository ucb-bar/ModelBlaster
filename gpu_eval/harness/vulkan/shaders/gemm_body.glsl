// Body of the batched strided GEMM (gemm_*.comp define WX, WY, TM, TN first):
//   C[b, m, n] = alpha * sum_k A[b*sAb + m*sAm + k*sAk] * B[b*sBb + k*sBk + n*sBn] (+ bias[n])
// the one kernel behind linear / matmul{,_ta,_tb,_tatb} / bmm / patch conv.
//
// A WX x WY workgroup (256 invocations) computes a BM x BN = (WY*TM) x (WX*TN)
// tile; invocation (tx, ty) owns rows ty*TM.. and columns tx*TN.. (TM, TN
// multiples of 4, so its operands come out of shared memory as vec4s). K moves
// through shared memory BK (32) at a time, the k loop over a tile unrolled by
// KUNR (8) so its LDS reads are issued ahead; fp32 accumulation (fma). Global
// loads take each operand's contiguous axis, as vec4 when the host found it 16-byte
// aligned (vecA / vecB): k for a row-major A or an [N, K] weight, n for a
// row-major B.
// Split-K (nsplit > 1, for few output tiles over a long K): workgroup z =
// batch * nsplit + s sums k in [s*kchunk, (s+1)*kchunk) and writes its raw
// partial at C + z*sCb; splitk_reduce.comp then adds the partials, alpha, bias.
#extension GL_EXT_control_flow_attributes : require
#include "mb_common.glsl"
layout(buffer_reference, std430, buffer_reference_align = 16) buffer V4 { vec4 v[]; };
#ifndef KUNR
#define KUNR 8
#endif
layout(local_size_x = WX, local_size_y = WY) in;
layout(push_constant, scalar) uniform PC {
    F32 A; F32 B; F32 C; F32 bias;
    int M; int N; int K;
    int sAb; int sAm; int sAk; int sBb; int sBk; int sBn; int sCb; int sCm;
    float alpha; int nsplit; int kchunk; int vecA; int vecB;
} pc;
#ifndef BK_
#define BK_ 32
#endif
const int BM = WY * TM, BN = WX * TN, BK = BK_;
// vec4 loads per invocation per K step (the register prefetch of the next tile)
const int NA = (BM * (BK / 4) + 255) / 256, NB = (BN * (BK / 4) + 255) / 256;
shared vec4 As[BK][BM / 4 + 1];
shared vec4 Bs[BK][BN / 4 + 1];
void putA(int k, int m, float v) { As[k][m >> 2][m & 3] = v; }
void putB(int k, int n, float v) { Bs[k][n >> 2][n & 3] = v; }
void main() {
    int tx = int(gl_LocalInvocationID.x), ty = int(gl_LocalInvocationID.y);
    int tid = ty * WX + tx;
    int n0 = int(gl_WorkGroupID.x) * BN, m0 = int(gl_WorkGroupID.y) * BM, z = int(gl_WorkGroupID.z);
    int b = z / pc.nsplit, sp = z % pc.nsplit;
    int kbeg = sp * pc.kchunk, kend = min(pc.K, kbeg + pc.kchunk);
    int aoff = b * pc.sAb, boff = b * pc.sBb;
    bool a_kfast = pc.sAk == 1, b_nfast = pc.sBn == 1;
    V4 A4 = V4(uint64_t(pc.A)), B4 = V4(uint64_t(pc.B));
    float acc[TM][TN];
    for (int i = 0; i < TM; i++) for (int j = 0; j < TN; j++) acc[i][j] = 0.0;
    // vec4 operands (the common case: a row-major activation times an [N, K]
    // weight or a row-major B) are double-buffered through registers: the next
    // tile's global loads are issued before this tile's FMAs.
    bool vec = pc.vecA != 0 && pc.vecB != 0;
    vec4 ra[NA], rb[NB];
    if (vec) {
        for (int l = 0; l < NA; l++) {
            int e = tid + 256 * l, kq = e % (BK / 4), mm = e / (BK / 4), gm = m0 + mm, gk = kbeg + 4 * kq;
            ra[l] = (e < BM * (BK / 4) && gm < pc.M && gk < kend) ? A4.v[(aoff + gm * pc.sAm + gk) >> 2] : vec4(0.0);
        }
        for (int l = 0; l < NB; l++) {
            int e = tid + 256 * l;
            if (pc.vecB == 1) {
                int kq = e % (BK / 4), nn = e / (BK / 4), gn = n0 + nn, gk = kbeg + 4 * kq;
                rb[l] = (e < BN * (BK / 4) && gn < pc.N && gk < kend) ? B4.v[(boff + gn * pc.sBn + gk) >> 2] : vec4(0.0);
            } else {
                int nq = e % (BN / 4), kk = e / (BN / 4), gn = n0 + 4 * nq, gk = kbeg + kk;
                rb[l] = (e < BN * (BK / 4) && gn < pc.N && gk < kend) ? B4.v[(boff + gk * pc.sBk + gn) >> 2] : vec4(0.0);
            }
        }
    }
    for (int k0 = kbeg; k0 < kend; k0 += BK) {
        if (vec) {
            for (int l = 0; l < NA; l++) {
                int e = tid + 256 * l, kq = e % (BK / 4), mm = e / (BK / 4);
                if (e < BM * (BK / 4)) {
                    vec4 v = ra[l];
                    putA(4 * kq, mm, v.x); putA(4 * kq + 1, mm, v.y); putA(4 * kq + 2, mm, v.z); putA(4 * kq + 3, mm, v.w);
                }
            }
            for (int l = 0; l < NB; l++) {
                int e = tid + 256 * l;
                if (e >= BN * (BK / 4)) continue;
                if (pc.vecB == 1) {
                    int kq = e % (BK / 4), nn = e / (BK / 4);
                    vec4 v = rb[l];
                    putB(4 * kq, nn, v.x); putB(4 * kq + 1, nn, v.y); putB(4 * kq + 2, nn, v.z); putB(4 * kq + 3, nn, v.w);
                } else {
                    Bs[e / (BN / 4)][e % (BN / 4)] = rb[l];
                }
            }
            barrier();
            int k1 = k0 + BK;
            if (k1 < kend) {
                for (int l = 0; l < NA; l++) {
                    int e = tid + 256 * l, kq = e % (BK / 4), mm = e / (BK / 4), gm = m0 + mm, gk = k1 + 4 * kq;
                    ra[l] = (e < BM * (BK / 4) && gm < pc.M && gk < kend) ? A4.v[(aoff + gm * pc.sAm + gk) >> 2] : vec4(0.0);
                }
                for (int l = 0; l < NB; l++) {
                    int e = tid + 256 * l;
                    if (pc.vecB == 1) {
                        int kq = e % (BK / 4), nn = e / (BK / 4), gn = n0 + nn, gk = k1 + 4 * kq;
                        rb[l] = (e < BN * (BK / 4) && gn < pc.N && gk < kend) ? B4.v[(boff + gn * pc.sBn + gk) >> 2] : vec4(0.0);
                    } else {
                        int nq = e % (BN / 4), kk = e / (BN / 4), gn = n0 + 4 * nq, gk = k1 + kk;
                        rb[l] = (e < BN * (BK / 4) && gn < pc.N && gk < kend) ? B4.v[(boff + gk * pc.sBk + gn) >> 2] : vec4(0.0);
                    }
                }
            }
        } else {
            for (int e = tid; e < BM * BK; e += 256) {
                int kk, mm;
                if (a_kfast) { kk = e % BK; mm = e / BK; } else { mm = e % BM; kk = e / BM; }
                int gm = m0 + mm, gk = k0 + kk;
                putA(kk, mm, (gm < pc.M && gk < kend) ? pc.A.v[aoff + gm * pc.sAm + gk * pc.sAk] : 0.0);
            }
            for (int e = tid; e < BN * BK; e += 256) {
                int kk, nn;
                if (b_nfast) { nn = e % BN; kk = e / BN; } else { kk = e % BK; nn = e / BK; }
                int gn = n0 + nn, gk = k0 + kk;
                putB(kk, nn, (gn < pc.N && gk < kend) ? pc.B.v[boff + gk * pc.sBk + gn * pc.sBn] : 0.0);
            }
            barrier();
        }
        // unrolled by KUNR so the compiler issues the next k's LDS reads ahead of this k's FMAs
        for (int k1 = 0; k1 < BK; k1 += KUNR)
        [[unroll]] for (int kk = 0; kk < KUNR; kk++) {
            int k = k1 + kk;
            vec4 a[TM / 4], bb[TN / 4];
            for (int i = 0; i < TM / 4; i++) a[i] = As[k][ty * (TM / 4) + i];
            for (int j = 0; j < TN / 4; j++) bb[j] = Bs[k][tx * (TN / 4) + j];
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    acc[i][j] = fma(a[i >> 2][i & 3], bb[j >> 2][j & 3], acc[i][j]);
        }
        barrier();
    }
    bool split = pc.nsplit > 1;
    bool hb = !split && uint64_t(pc.bias) != 0ul;
    for (int i = 0; i < TM; i++) {
        int gm = m0 + ty * TM + i;
        if (gm >= pc.M) continue;
        for (int j = 0; j < TN; j++) {
            int gn = n0 + tx * TN + j;
            if (gn >= pc.N) continue;
            float v = split ? acc[i][j] : pc.alpha * acc[i][j];
            if (hb) v += pc.bias.v[gn];
            pc.C.v[z * pc.sCb + gm * pc.sCm + gn] = v;
        }
    }
}
