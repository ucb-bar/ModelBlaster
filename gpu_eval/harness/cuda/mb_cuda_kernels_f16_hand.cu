// fp16 launchers whose `_f16` KernelSpec is hand-written with a signature
// that differs from the fp32 op's (so they can't be generated from
// mb_cuda_kernels.cu). f16_kernels.py appends these to the generated
// mb_cuda_kernels_f16.cu in place of the generated entries of the same name.
// Same contract: __half storage, fp32 math, one rounding per stored value.
#include <cuda_fp16.h>

// softmax over the last axis with a fused input scale (SDPA's 1/sqrt(d));
// the exponentials are recomputed in the normalising pass rather than stored
// in the fp16 output and rescaled (which would round twice)
__global__ void k_softmax_scaled(const __half *x, __half *y, int M, int K, float scale) {
    long r = MB_TID; if (r >= M) return;
    const __half *xr = x + r * K; __half *yr = y + r * K;
    float mx = -INFINITY; for (int k = 0; k < K; k++) mx = fmaxf(mx, mb_ld(xr[k]) * scale);
    double s = 0.0; for (int k = 0; k < K; k++) s += expf(mb_ld(xr[k]) * scale - mx);
    float inv = (float)(1.0 / s);
    for (int k = 0; k < K; k++) yr[k] = expf(mb_ld(xr[k]) * scale - mx) * inv;
}
extern "C" void kernel_softmax_f16(const __half *input, __half *output, int M, int K, float input_scale) {
    k_softmax_scaled<<<mb_blocks(M), mb_threads(M)>>>(input, output, M, K, input_scale);
}

// torch AdaptiveAvgPool2d windows: [floor(o*I/O), ceil((o+1)*I/O))
__global__ void k_adaptive_avg(const __half *x, __half *y, int N, int C, int IH, int IW, int OH, int OW) {
    long i = MB_TID; long tot = (long)N * C * OH * OW;
    if (i >= tot) return;
    int ow = i % OW, oh = (i / OW) % OH; long nc = i / ((long)OW * OH);
    int h0 = (oh * IH) / OH, h1 = ((oh + 1) * IH + OH - 1) / OH;
    int w0 = (ow * IW) / OW, w1 = ((ow + 1) * IW + OW - 1) / OW;
    float s = 0.0f;
    for (int h = h0; h < h1; h++)
        for (int w = w0; w < w1; w++) s += mb_ld(x[(nc * IH + h) * IW + w]);
    y[i] = s / (float)((h1 - h0) * (w1 - w0));
}
extern "C" void kernel_adaptive_avg_pool2d_f16(const __half *input, __half *output, int N, int C,
                                               int IH, int IW, int OH, int OW) {
    long tot = (long)N * C * OH * OW;
    k_adaptive_avg<<<mb_blocks(tot), mb_threads(tot)>>>(input, output, N, C, IH, IW, OH, OW);
}

__global__ void k_copy_c_h(const __half *src, __half *dst, int N, int c, int C, int c_off, int HW) {
    long i = MB_TID; long tot = (long)N * c * HW; if (i >= tot) return;
    long n = i / ((long)c * HW), r = i % ((long)c * HW);
    dst[(n * C + c_off) * HW + r] = src[i];
}
extern "C" void kernel_cat2_c1_f16(const __half *a, const __half *b, __half *output,
                                   int N, int H, int W, int Ca, int Cb) {
    int C = Ca + Cb, HW = H * W;
    k_copy_c_h<<<mb_blocks((long)N * Ca * HW), mb_threads((long)N * Ca * HW)>>>(a, output, N, Ca, C, 0, HW);
    k_copy_c_h<<<mb_blocks((long)N * Cb * HW), mb_threads((long)N * Cb * HW)>>>(b, output, N, Cb, C, Ca, HW);
}

// The fp32 k_sdpa accumulates P.V in its output row, which in fp16 would be
// an fp16 accumulation: one thread per output element instead, all sums in
// fp32 (scores recomputed per d; a correctness seed, like the fp32 one).
__global__ void k_sdpa_h(const __half *Q, const __half *Kt, const __half *V, __half *O, int BH, int S, int D) {
    long t = MB_TID; if (t >= (long)BH * S * D) return;
    int d0 = t % D; long r = t / D; long bh = r / S; int i = r % S;
    const __half *q = Q + (bh * S + i) * D, *Kb = Kt + bh * S * D, *Vb = V + bh * S * D;
    float scale = rsqrtf((float)D), mx = -INFINITY;
    for (int j = 0; j < S; j++) {
        float s = 0; for (int d = 0; d < D; d++) s += mb_ld(q[d]) * mb_ld(Kb[(long)j * D + d]);
        mx = fmaxf(mx, s * scale);
    }
    float sum = 0, acc = 0;
    for (int j = 0; j < S; j++) {
        float s = 0; for (int d = 0; d < D; d++) s += mb_ld(q[d]) * mb_ld(Kb[(long)j * D + d]);
        float p = expf(s * scale - mx); sum += p;
        acc += p * mb_ld(Vb[(long)j * D + d0]);
    }
    O[t] = acc / sum;
}
extern "C" void kernel_sdpa_f16(const __half *Q, const __half *K, const __half *V, __half *output, int BH, int S, int D) {
    k_sdpa_h<<<mb_blocks((long)BH * S * D), mb_threads((long)BH * S * D)>>>(Q, K, V, output, BH, S, D);
}

// bmm with strided operands (layout_fold): C[b0,b1][m,n] = sum_k A[b0,b1][m,k] * B[b0,b1][k,n], every operand and
// the output addressed through the given int strides (elements), fp16 storage / fp32 accumulate.
// TILED seed: 64x64 C tile per 128-thread block (2x2 warps of 32x32), K step 32, tensor cores (mma.sync
// m16n8k16, fragments by ldmatrix), static shared memory only, one block per (batch, m-tile, n-tile).
// The smem tiles are filled with 8-element chunks taken along whichever operand stride is 1 (A: a_sk == 1 -> rows
// of K, else a_sm == 1 -> columns of M; B: b_sn == 1 -> rows of N, else b_sk == 1 -> columns of K), as 16-byte
// vector loads when the launcher found the offsets/strides/pointer 16-byte aligned (transposed operands are
// transposed on the way into smem); everything else (tails, odd strides or offsets, no unit stride) takes
// the same path with scalar loads. Correct for any strides and any M, K, N (K <= 0 gives zeros).
// The launcher passes `fl`: bit0 A chunks along M, bit1 A vector ok, bit2 B chunks along K, bit3 B vector ok,
// bits4-5 C store (0 vector rows of N, 1 scalar along M, 2 scalar along N). (No __launch_bounds__ on the kernel: CudaLibrary names an item by the first identifier before a '(', which
// would be the attribute.) Below sm_80 a scalar loop is used; so for M < 16 or N < 16 (k_bmm_strided_scalar).
__device__ __forceinline__ uint4 mbs_load8(const __half *p, long us, int cnt, bool vec) {
    uint4 r = make_uint4(0u, 0u, 0u, 0u);
    if (cnt <= 0) return r;
    if (vec && cnt >= 8) return *reinterpret_cast<const uint4 *>(p);
    unsigned short *h = reinterpret_cast<unsigned short *>(&r);
    for (int i = 0; i < 8 && i < cnt; i++) h[i] = __half_as_ushort(p[(long)i * us]);
    return r;
}
// one thread per C element, scalar K loop (used for M < 16 or N < 16 and as the sm < 80 body above)
__global__ void k_bmm_strided_scalar(const __half *A, const __half *B, __half *C, int B0, int B1, int M, int K, int N,
                              int a_off, int a_s0, int a_s1, int a_sm, int a_sk,
                              int b_off, int b_s0, int b_s1, int b_sk, int b_sn,
                              int c_off, int c_s0, int c_s1, int c_sm, int c_sn) {
    long i = MB_TID; if (i >= (long)B0 * B1 * M * N) return;
    long n = i % N, m = (i / N) % M, b1 = (i / ((long)M * N)) % B1, b0 = i / ((long)B1 * M * N);
    const __half *a = A + a_off + b0 * a_s0 + b1 * a_s1 + m * a_sm;
    const __half *bb = B + b_off + b0 * b_s0 + b1 * b_s1 + n * b_sn;
    float acc = 0.0f;
    for (long k = 0; k < K; k++) acc += mb_ld(a[k * a_sk]) * mb_ld(bb[k * b_sk]);
    C[c_off + b0 * c_s0 + b1 * c_s1 + m * c_sm + n * c_sn] = acc;
}
__global__ void k_bmm_strided(const __half *A, const __half *B, __half *C, int B0, int B1, int M, int K, int N,
                              int a_off, int a_s0, int a_s1, int a_sm, int a_sk,
                              int b_off, int b_s0, int b_s1, int b_sk, int b_sn,
                              int c_off, int c_s0, int c_s1, int c_sm, int c_sn, int fl) {
    __shared__ __align__(128) __half As[64 * 40];       // [m][k], row pad 8 (conflict-free ldmatrix)
    __shared__ __align__(128) __half Bs[32 * 72];       // [k][n], row pad 8
    __shared__ __align__(128) float Cs[64 * 68];        // epilogue staging
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int nbn = (N + 63) / 64, nbm = (M + 63) / 64;
    long bid = blockIdx.x;
    const int nt = (int)(bid % nbn); bid /= nbn;
    const int mt = (int)(bid % nbm); bid /= nbm;
    const int b1 = (int)(bid % B1), b0 = (int)(bid / B1);
    const int m0 = mt * 64, n0 = nt * 64;
    const __half *Ab = A + a_off + (long)b0 * a_s0 + (long)b1 * a_s1;
    const __half *Bb = B + b_off + (long)b0 * b_s0 + (long)b1 * b_s1;
    __half *Cb = C + c_off + (long)b0 * c_s0 + (long)b1 * c_s1;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    for (int e = tid; e < 64 * 64; e += 128) {
        int m = m0 + (e >> 6), n = n0 + (e & 63);
        if (m >= M || n >= N) continue;
        float acc = 0.0f;
        for (int k = 0; k < K; k++) acc += mb_ld(Ab[(long)m * a_sm + (long)k * a_sk]) * mb_ld(Bb[(long)k * b_sk + (long)n * b_sn]);
        Cb[(long)m * c_sm + (long)n * c_sn] = __float2half_rn(acc);
    }
#else
    const bool a_alongM = fl & 1, a_vec = (fl >> 1) & 1, b_alongK = (fl >> 2) & 1, b_vec = (fl >> 3) & 1;
    const int cmode = (fl >> 4) & 3;
    const int wm = warp >> 1, wn = warp & 1;
    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) acc[i][j][q] = 0.0f;
    uint4 ra[2], rb[2];
    const int nk = (K + 31) / 32;
    // global -> registers for K tile kt (2 chunks of 8 elements per thread and operand)
    auto gload = [&](int kt) {
        const int k0 = kt * 32;
#pragma unroll
        for (int c = 0; c < 2; c++) {
            int j = tid + c * 128;
            if (!a_alongM) {                       // 64 rows (m) x 4 chunks along k
                int m = m0 + (j >> 2), kk = k0 + (j & 3) * 8;
                ra[c] = mbs_load8(Ab + (long)m * a_sm + (long)kk * a_sk, a_sk, m < M ? K - kk : 0, a_vec);
            } else {                               // 32 k-columns x 8 chunks along m
                int kk = k0 + (j >> 3), m = m0 + (j & 7) * 8;
                ra[c] = mbs_load8(Ab + (long)m * a_sm + (long)kk * a_sk, a_sm, kk < K ? M - m : 0, a_vec);
            }
            if (!b_alongK) {                       // 32 k-rows x 8 chunks along n
                int kk = k0 + (j >> 3), n = n0 + (j & 7) * 8;
                rb[c] = mbs_load8(Bb + (long)kk * b_sk + (long)n * b_sn, b_sn, kk < K ? N - n : 0, b_vec);
            } else {                               // 64 n-columns x 4 chunks along k
                int n = n0 + (j >> 2), kk = k0 + (j & 3) * 8;
                rb[c] = mbs_load8(Bb + (long)kk * b_sk + (long)n * b_sn, b_sk, n < N ? K - kk : 0, b_vec);
            }
        }
    };
    auto sstore = [&]() {
        unsigned short *as = reinterpret_cast<unsigned short *>(As), *bs = reinterpret_cast<unsigned short *>(Bs);
#pragma unroll
        for (int c = 0; c < 2; c++) {
            int j = tid + c * 128;
            if (!a_alongM) {
                *reinterpret_cast<uint4 *>(&As[(j >> 2) * 40 + (j & 3) * 8]) = ra[c];
            } else {
                const unsigned short *h = reinterpret_cast<const unsigned short *>(&ra[c]);
#pragma unroll
                for (int i = 0; i < 8; i++) as[((j & 7) * 8 + i) * 40 + (j >> 3)] = h[i];
            }
            if (!b_alongK) {
                *reinterpret_cast<uint4 *>(&Bs[(j >> 3) * 72 + (j & 7) * 8]) = rb[c];
            } else {
                const unsigned short *h = reinterpret_cast<const unsigned short *>(&rb[c]);
#pragma unroll
                for (int i = 0; i < 8; i++) bs[((j & 3) * 8 + i) * 72 + (j >> 2)] = h[i];
            }
        }
    };
    if (nk > 0) gload(0);
    for (int kt = 0; kt < nk; kt++) {
        sstore();
        __syncthreads();
        if (kt + 1 < nk) gload(kt + 1);            // next tile's loads overlap this tile's mma
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
            unsigned af[2][4], bf[4][2];
#pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                unsigned ad = (unsigned)__cvta_generic_to_shared(&As[(wm * 32 + mi * 16 + (lane & 15)) * 40 + ks * 16 + (lane >> 4) * 8]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(af[mi][0]), "=r"(af[mi][1]), "=r"(af[mi][2]), "=r"(af[mi][3]) : "r"(ad));
            }
#pragma unroll
            for (int nj = 0; nj < 2; nj++) {
                unsigned ad = (unsigned)__cvta_generic_to_shared(&Bs[(ks * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * 72 + wn * 32 + nj * 16 + (lane >> 4) * 8]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(bf[2 * nj][0]), "=r"(bf[2 * nj][1]), "=r"(bf[2 * nj + 1][0]), "=r"(bf[2 * nj + 1][1]) : "r"(ad));
            }
#pragma unroll
            for (int mi = 0; mi < 2; mi++)
#pragma unroll
                for (int ni = 0; ni < 4; ni++)
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                                 : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]), "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
                                 : "r"(af[mi][0]), "r"(af[mi][1]), "r"(af[mi][2]), "r"(af[mi][3]), "r"(bf[ni][0]), "r"(bf[ni][1]));
        }
        __syncthreads();
    }
    // epilogue: fragments -> smem (fp32) -> strided global stores (one rounding to fp16)
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int r = wm * 32 + mi * 16 + (lane >> 2), cc = wn * 32 + ni * 8 + (lane & 3) * 2;
            Cs[r * 68 + cc] = acc[mi][ni][0]; Cs[r * 68 + cc + 1] = acc[mi][ni][1];
            Cs[(r + 8) * 68 + cc] = acc[mi][ni][2]; Cs[(r + 8) * 68 + cc + 1] = acc[mi][ni][3];
        }
    __syncthreads();
    if (cmode == 0) {                               // c_sn == 1: 16-byte stores of 8 columns where whole and aligned
        for (int j = tid; j < 64 * 8; j += 128) {
            int r = j >> 3, cc = (j & 7) * 8, m = m0 + r, n = n0 + cc;
            if (m >= M || n >= N) continue;
            __half *dst = Cb + (long)m * c_sm + n;
            if (n + 8 <= N) {
                uint4 v; unsigned short *h = reinterpret_cast<unsigned short *>(&v);
#pragma unroll
                for (int i = 0; i < 8; i++) h[i] = __half_as_ushort(__float2half_rn(Cs[r * 68 + cc + i]));
                *reinterpret_cast<uint4 *>(dst) = v;
            } else {
                for (int i = 0; n + i < N; i++) dst[i] = __float2half_rn(Cs[r * 68 + cc + i]);
            }
        }
    } else {
        for (int e = tid; e < 64 * 64; e += 128) {
            int r = cmode == 1 ? (e & 63) : (e >> 6), cc = cmode == 1 ? (e >> 6) : (e & 63);
            int m = m0 + r, n = n0 + cc;
            if (m < M && n < N) Cb[(long)m * c_sm + (long)n * c_sn] = __float2half_rn(Cs[r * 68 + cc]);
        }
    }
#endif
}
extern "C" void kernel_bmm_strided_f16(const __half *A, const __half *B, __half *C, int B0, int B1, int M, int K, int N,
                                       int a_off, int a_s0, int a_s1, int a_sm, int a_sk,
                                       int b_off, int b_s0, int b_s1, int b_sk, int b_sn,
                                       int c_off, int c_s0, int c_s1, int c_sm, int c_sn) {
    long nb = (long)B0 * B1;
    if (nb <= 0 || M <= 0 || N <= 0) return;
    if (M < 16 || N < 16) {                      // degenerate GEMM (a 64x64 mma tile would be mostly padding): one thread per C element
        long tot = nb * M * N;
        k_bmm_strided_scalar<<<mb_blocks(tot), mb_threads(tot)>>>(A, B, C, B0, B1, M, K, N, a_off, a_s0, a_s1, a_sm, a_sk,
                                                                  b_off, b_s0, b_s1, b_sk, b_sn, c_off, c_s0, c_s1, c_sm, c_sn);
        return;
    }
    long tiles = nb * ((M + 63) / 64) * ((N + 63) / 64);
    auto al = [](const void *p, int o, int s0, int s1, int s2) {      // 16-byte alignment of every row start
        return (((unsigned long)p) & 15) == 0 && o % 8 == 0 && s0 % 8 == 0 && s1 % 8 == 0 && s2 % 8 == 0;
    };
    int fl = 0;
    bool a_m = a_sk != 1 && a_sm == 1, b_k = b_sn != 1 && b_sk == 1;
    if (a_m) fl |= 1;
    if (a_sk == 1 ? al(A, a_off, a_s0, a_s1, a_sm) : (a_sm == 1 && al(A, a_off, a_s0, a_s1, a_sk))) fl |= 2;
    if (b_k) fl |= 4;
    if (b_sn == 1 ? al(B, b_off, b_s0, b_s1, b_sk) : (b_sk == 1 && al(B, b_off, b_s0, b_s1, b_sn))) fl |= 8;
    int cm = (c_sn == 1 && al(C, c_off, c_s0, c_s1, c_sm)) ? 0 : (c_sm == 1 && c_sn != 1 ? 1 : 2);
    fl |= cm << 4;
    k_bmm_strided<<<(unsigned)tiles, 128>>>(A, B, C, B0, B1, M, K, N, a_off, a_s0, a_s1, a_sm, a_sk,
                                            b_off, b_s0, b_s1, b_sk, b_sn, c_off, c_s0, c_s1, c_sm, c_sn, fl);
}


// fused attention (attention_fuse) seed: one warp per query row, 32 keys per step (one per lane), online
// softmax (running max / rescaled sum) in fp32, so the [M, N] score matrix is never materialised. Correct for any
// strides with unit inner stride, D, DV <= 256; no tensor cores (KernelBlaster's job: mma.sync / ldmatrix / cp.async).
#define MB_ATT_WARPS 4
__global__ void k_attention(const __half *Q, const __half *K, const __half *V, __half *O,
                            int B0, int B1, int M, int N, int D, int DV,
                            int q_off, int q_s0, int q_s1, int q_sm, int k_off, int k_s0, int k_s1, int k_sn,
                            int v_off, int v_s0, int v_s1, int v_sn, int o_off, int o_s0, int o_s1, int o_sm,
                            int causal, int act, float scale) {
    __shared__ float qs[MB_ATT_WARPS][256];
    const unsigned FULL = 0xffffffffu;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    long row = (long)blockIdx.x * MB_ATT_WARPS + warp;
    if (row >= (long)B0 * B1 * M) return;            // whole warp leaves together
    long m = row % M, b1 = (row / M) % B1, b0 = row / ((long)M * B1);
    const __half *q = Q + q_off + b0 * q_s0 + b1 * q_s1 + m * q_sm;
    for (int d = lane; d < D; d += 32) qs[warp][d] = mb_ld(q[d]);
    __syncwarp();
    const __half *kb = K + k_off + b0 * k_s0 + b1 * k_s1;
    const __half *vb = V + v_off + b0 * v_s0 + b1 * v_s1;
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.0f;
    float mx = -INFINITY, l = 0.0f;
    long nend = (causal && m + 1 < N) ? m + 1 : N;
    for (long n0 = 0; n0 < nend; n0 += 32) {
        long n = n0 + lane;
        float s = -INFINITY;
        if (n < nend) {
            const __half *kr = kb + n * k_sn;
            float t = 0.0f;
            for (int d = 0; d < D; d++) t += qs[warp][d] * mb_ld(kr[d]);
            s = t * scale;
        }
        float p;
        if (act == 1) {
            p = s > 0.0f ? s : 0.0f;
        } else {
            float cm = s;
            for (int o = 16; o; o >>= 1) cm = fmaxf(cm, __shfl_xor_sync(FULL, cm, o));
            float nm = fmaxf(mx, cm);
            float alpha = mx == -INFINITY ? 0.0f : expf(mx - nm);
            p = s == -INFINITY ? 0.0f : expf(s - nm);
            float ps = p;
            for (int o = 16; o; o >>= 1) ps += __shfl_xor_sync(FULL, ps, o);
            l = l * alpha + ps;
#pragma unroll
            for (int i = 0; i < 8; i++) acc[i] *= alpha;
            mx = nm;
        }
        int cnt = (int)(nend - n0 < 32 ? nend - n0 : 32);
        for (int j = 0; j < cnt; j++) {
            float pj = __shfl_sync(FULL, p, j);
            const __half *vr = vb + (n0 + j) * v_sn;
#pragma unroll
            for (int i = 0; i < 8; i++) { int e = lane + 32 * i; if (e < DV) acc[i] += pj * mb_ld(vr[e]); }
        }
    }
    float inv = act == 1 ? 1.0f : 1.0f / l;
    __half *o = O + o_off + b0 * o_s0 + b1 * o_s1 + m * o_sm;
#pragma unroll
    for (int i = 0; i < 8; i++) { int e = lane + 32 * i; if (e < DV) o[e] = acc[i] * inv; }
}
extern "C" void kernel_attention_f16(const __half *Q, const __half *K, const __half *V, __half *O,
                                     int B0, int B1, int M, int N, int D, int DV,
                                     int q_off, int q_s0, int q_s1, int q_sm, int k_off, int k_s0, int k_s1, int k_sn,
                                     int v_off, int v_s0, int v_s1, int v_sn, int o_off, int o_s0, int o_s1, int o_sm,
                                     int causal, int act, float scale) {
    long rows = (long)B0 * B1 * M;
    if (rows <= 0) return;
    k_attention<<<(unsigned)((rows + MB_ATT_WARPS - 1) / MB_ATT_WARPS), 32 * MB_ATT_WARPS>>>(
        Q, K, V, O, B0, B1, M, N, D, DV, q_off, q_s0, q_s1, q_sm, k_off, k_s0, k_s1, k_sn,
        v_off, v_s0, v_s1, v_sn, o_off, o_s0, o_s1, o_sm, causal, act, scale);
}
