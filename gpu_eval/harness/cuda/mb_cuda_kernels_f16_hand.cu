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
    k_softmax_scaled<<<MB_GRID(M), MB_THREADS>>>(input, output, M, K, input_scale);
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
    k_adaptive_avg<<<MB_GRID(tot), MB_THREADS>>>(input, output, N, C, IH, IW, OH, OW);
}

__global__ void k_copy_c_h(const __half *src, __half *dst, int N, int c, int C, int c_off, int HW) {
    long i = MB_TID; long tot = (long)N * c * HW; if (i >= tot) return;
    long n = i / ((long)c * HW), r = i % ((long)c * HW);
    dst[(n * C + c_off) * HW + r] = src[i];
}
extern "C" void kernel_cat2_c1_f16(const __half *a, const __half *b, __half *output,
                                   int N, int H, int W, int Ca, int Cb) {
    int C = Ca + Cb, HW = H * W;
    k_copy_c_h<<<MB_GRID((long)N * Ca * HW), MB_THREADS>>>(a, output, N, Ca, C, 0, HW);
    k_copy_c_h<<<MB_GRID((long)N * Cb * HW), MB_THREADS>>>(b, output, N, Cb, C, Ca, HW);
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
    k_sdpa_h<<<MB_GRID((long)BH * S * D), MB_THREADS>>>(Q, K, V, output, BH, S, D);
}
