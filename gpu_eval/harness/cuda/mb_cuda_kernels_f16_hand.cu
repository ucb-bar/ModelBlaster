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

// bmm with strided operands (layout_fold): one thread per C element, any strides
__global__ void k_bmm_strided(const __half *A, const __half *B, __half *C, int B0, int B1, int M, int K, int N,
                              int a_off, int a_s0, int a_s1, int a_sm, int a_sk,
                              int b_off, int b_s0, int b_s1, int b_sk, int b_sn,
                              int c_off, int c_s0, int c_s1, int c_sm, int c_sn) {
    long i = MB_TID; if (i >= (long)B0 * B1 * M * N) return;
    long n = i % N, m = (i / N) % M, b1 = (i / ((long)M * N)) % B1, b0 = i / ((long)B1 * M * N);
    const __half *a = A + a_off + b0 * a_s0 + b1 * a_s1 + m * a_sm;
    const __half *bb = B + b_off + b0 * b_s0 + b1 * b_s1 + n * b_sn;
    double acc = 0.0f;
    for (long k = 0; k < K; k++) acc += mb_ld(a[k * a_sk]) * mb_ld(bb[k * b_sk]);
    C[c_off + b0 * c_s0 + b1 * c_s1 + m * c_sm + n * c_sn] = acc;
}
extern "C" void kernel_bmm_strided_f16(const __half *A, const __half *B, __half *C, int B0, int B1, int M, int K, int N,
                                       int a_off, int a_s0, int a_s1, int a_sm, int a_sk,
                                       int b_off, int b_s0, int b_s1, int b_sk, int b_sn,
                                       int c_off, int c_s0, int c_s1, int c_sm, int c_sn) {
    long tot = (long)B0 * B1 * M * N;
    k_bmm_strided<<<mb_blocks(tot), mb_threads(tot)>>>(A, B, C, B0, B1, M, K, N, a_off, a_s0, a_s1, a_sm, a_sk,
                                                       b_off, b_s0, b_s1, b_sk, b_sn, c_off, c_s0, c_s1, c_sm, c_sn);
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
