// Hand-written CUDA implementations of ModelBlaster's per-op C ABI.
//
// Every extern "C" launcher below has EXACTLY the signature of the matching
// `KernelSpec.signature` in modelblaster/pipeline/reference_kernels.py, but
// all pointers are device pointers. That is the seam the plan proposes: a
// GPU backend is "the same kernels.h, implemented in CUDA", and a
// KernelBlaster CUDA problem is the same signature renamed to
// launch_gpu_implementation.
//
// These are deliberately naive (one thread per output element, no tiling):
// they are correctness seeds for the KernelBlaster RL loop, not tuned code.
#include <cuda_runtime.h>
#include <math.h>

// Each .so links its own (static) cudart, so error state must be read from
// inside the library that launched.
extern "C" int mb_last_error(void) { return (int)cudaGetLastError(); }

#define MB_THREADS 256
// at least one block: a zero-size launch is an invalid configuration (every
// kernel bounds-checks its thread id)
#define MB_GRID(n) (int)(((long)(n) > 0 ? (long)(n) + MB_THREADS - 1 : MB_THREADS) / MB_THREADS)
#define MB_TID ((long)blockIdx.x * blockDim.x + threadIdx.x)
// dot-product kernels (conv, linear, matmul, bmm) accumulate in double: a
// sequential fp32 sum over K ~ 1e3 is several times less accurate than
// cuBLAS/torch blocked sums, which ill-conditioned graphs (attention with
// O(1e3) logits) amplify past the 1e-3 golden bound

// ---------------------------------------------------------------- unary/binary
__global__ void k_relu(const float *x, float *y, int n) { long i = MB_TID; if (i < n) y[i] = fmaxf(x[i], 0.0f); }
__global__ void k_relu6(const float *x, float *y, int n) { long i = MB_TID; if (i < n) y[i] = fminf(fmaxf(x[i], 0.0f), 6.0f); }
__global__ void k_sigmoid(const float *x, float *y, int n) { long i = MB_TID; if (i < n) y[i] = 1.0f / (1.0f + expf(-x[i])); }
__global__ void k_silu(const float *x, float *y, int n) { long i = MB_TID; if (i < n) { float v = x[i]; y[i] = v / (1.0f + expf(-v)); } }
__global__ void k_elu(const float *x, float *y, int n, float a) { long i = MB_TID; if (i < n) { float v = x[i]; y[i] = v > 0.0f ? v : a * (expf(v) - 1.0f); } }
__global__ void k_add(const float *a, const float *b, float *y, int n) { long i = MB_TID; if (i < n) y[i] = a[i] + b[i]; }

extern "C" void kernel_relu(const float *input, float *output, int n) { k_relu<<<MB_GRID(n), MB_THREADS>>>(input, output, n); }
extern "C" void kernel_relu6(const float *input, float *output, int n) { k_relu6<<<MB_GRID(n), MB_THREADS>>>(input, output, n); }
extern "C" void kernel_sigmoid(const float *input, float *output, int n) { k_sigmoid<<<MB_GRID(n), MB_THREADS>>>(input, output, n); }
extern "C" void kernel_silu(const float *input, float *output, int n) { k_silu<<<MB_GRID(n), MB_THREADS>>>(input, output, n); }
extern "C" void kernel_elu(const float *input, float *output, int n, float alpha) { k_elu<<<MB_GRID(n), MB_THREADS>>>(input, output, n, alpha); }
extern "C" void kernel_add(const float *a, const float *b, float *output, int n) { k_add<<<MB_GRID(n), MB_THREADS>>>(a, b, output, n); }

// ---------------------------------------------------------------- batchnorm (folded)
__global__ void k_bn(const float *x, const float *s, const float *b, float *y, int N, int C, int HW) {
    long i = MB_TID; long tot = (long)N * C * HW;
    if (i < tot) { int c = (i / HW) % C; y[i] = x[i] * s[c] + b[c]; }
}
extern "C" void kernel_batchnorm2d(const float *input, const float *scale, const float *bias,
                                   float *output, int N, int C, int H, int W) {
    long tot = (long)N * C * H * W;
    k_bn<<<MB_GRID(tot), MB_THREADS>>>(input, scale, bias, output, N, C, H * W);
}

// ---------------------------------------------------------------- conv2d (direct)
// grouped + dilated direct conv; kernel_conv2d is the G=1 case
__global__ void k_convg(const float *x, const float *w, const float *b, float *y,
                        int N, int IC, int IH, int IW, int OC, int OH, int OW, int KH, int KW,
                        int SH, int SW, int PH, int PW, int DH, int DW, int G) {
    long i = MB_TID; long tot = (long)N * OC * OH * OW;
    if (i >= tot) return;
    int ow = i % OW; int oh = (i / OW) % OH; int oc = (i / ((long)OW * OH)) % OC; int n = i / ((long)OW * OH * OC);
    int ICg = IC / G, ic0 = (oc / (OC / G)) * ICg;
    double acc = b ? b[oc] : 0.0f;
    for (int ic = 0; ic < ICg; ic++)
        for (int kh = 0; kh < KH; kh++) {
            int ih = oh * SH - PH + kh * DH; if (ih < 0 || ih >= IH) continue;
            for (int kw = 0; kw < KW; kw++) {
                int iw = ow * SW - PW + kw * DW; if (iw < 0 || iw >= IW) continue;
                acc += x[(((long)n * IC + ic0 + ic) * IH + ih) * IW + iw] * w[(((long)oc * ICg + ic) * KH + kh) * KW + kw];
            }
        }
    y[i] = acc;
}
extern "C" void kernel_conv2d(const float *input, const float *weight, const float *bias, float *output,
                              int N, int IC, int IH, int IW, int OC, int KH, int KW,
                              int SH, int SW, int PH, int PW, int DH, int DW) {
    int OH = (IH + 2 * PH - DH * (KH - 1) - 1) / SH + 1, OW = (IW + 2 * PW - DW * (KW - 1) - 1) / SW + 1;
    long tot = (long)N * OC * OH * OW;
    k_convg<<<MB_GRID(tot), MB_THREADS>>>(input, weight, bias, output, N, IC, IH, IW, OC, OH, OW, KH, KW,
                                          SH, SW, PH, PW, DH, DW, 1);
}
extern "C" void kernel_conv2d_grouped(const float *input, const float *weight, const float *bias, float *output,
                                      int N, int IC, int IH, int IW, int OC, int KH, int KW,
                                      int SH, int SW, int PH, int PW, int DH, int DW, int G) {
    int OH = (IH + 2 * PH - DH * (KH - 1) - 1) / SH + 1, OW = (IW + 2 * PW - DW * (KW - 1) - 1) / SW + 1;
    long tot = (long)N * OC * OH * OW;
    k_convg<<<MB_GRID(tot), MB_THREADS>>>(input, weight, bias, output, N, IC, IH, IW, OC, OH, OW, KH, KW,
                                          SH, SW, PH, PW, DH, DW, G);
}

__global__ void k_conv2d_dw(const float *x, const float *w, const float *b, float *y,
                            int N, int C, int IH, int IW, int OH, int OW,
                            int KH, int KW, int SH, int SW, int PH, int PW) {
    long i = MB_TID; long tot = (long)N * C * OH * OW;
    if (i >= tot) return;
    int ow = i % OW; int oh = (i / OW) % OH; int c = (i / ((long)OW * OH)) % C; int n = i / ((long)OW * OH * C);
    double acc = b ? b[c] : 0.0f;
    for (int kh = 0; kh < KH; kh++) {
        int ih = oh * SH - PH + kh; if (ih < 0 || ih >= IH) continue;
        for (int kw = 0; kw < KW; kw++) {
            int iw = ow * SW - PW + kw; if (iw < 0 || iw >= IW) continue;
            acc += x[(((long)n * C + c) * IH + ih) * IW + iw] * w[((long)c * KH + kh) * KW + kw];
        }
    }
    y[i] = acc;
}
extern "C" void kernel_conv2d_dw(const float *input, const float *weight, const float *bias, float *output,
                                 int N, int C, int IH, int IW, int KH, int KW, int SH, int SW, int PH, int PW) {
    int OH = (IH + 2 * PH - KH) / SH + 1, OW = (IW + 2 * PW - KW) / SW + 1;
    long tot = (long)N * C * OH * OW;
    k_conv2d_dw<<<MB_GRID(tot), MB_THREADS>>>(input, weight, bias, output, N, C, IH, IW, OH, OW, KH, KW, SH, SW, PH, PW);
}

// ---------------------------------------------------------------- pooling
__global__ void k_maxpool2d(const float *x, float *y, int N, int C, int IH, int IW, int OH, int OW,
                            int KH, int KW, int SH, int SW, int PH, int PW, int DH, int DW) {
    long i = MB_TID; long tot = (long)N * C * OH * OW;
    if (i >= tot) return;
    int ow = i % OW; int oh = (i / OW) % OH; long nc = i / ((long)OW * OH);
    float m = -INFINITY;
    for (int kh = 0; kh < KH; kh++) {
        int ih = oh * SH - PH + kh * DH; if (ih < 0 || ih >= IH) continue;
        for (int kw = 0; kw < KW; kw++) {
            int iw = ow * SW - PW + kw * DW; if (iw < 0 || iw >= IW) continue;
            m = fmaxf(m, x[(nc * IH + ih) * IW + iw]);
        }
    }
    y[i] = m;
}
extern "C" void kernel_maxpool2d(const float *input, float *output, int N, int C, int IH, int IW,
                                 int KH, int KW, int SH, int SW, int PH, int PW, int DH, int DW) {
    int OH = (IH + 2 * PH - DH * (KH - 1) - 1) / SH + 1, OW = (IW + 2 * PW - DW * (KW - 1) - 1) / SW + 1;
    long tot = (long)N * C * OH * OW;
    k_maxpool2d<<<MB_GRID(tot), MB_THREADS>>>(input, output, N, C, IH, IW, OH, OW, KH, KW, SH, SW, PH, PW, DH, DW);
}

__global__ void k_avgpool2d(const float *x, float *y, int N, int C, int IH, int IW, int OH, int OW,
                            int KH, int KW, int SH, int SW, int PH, int PW) {
    long i = MB_TID; long tot = (long)N * C * OH * OW;
    if (i >= tot) return;
    int ow = i % OW; int oh = (i / OW) % OH; long nc = i / ((long)OW * OH);
    double acc = 0.0f;
    for (int kh = 0; kh < KH; kh++) {
        int ih = oh * SH - PH + kh; if (ih < 0 || ih >= IH) continue;
        for (int kw = 0; kw < KW; kw++) {
            int iw = ow * SW - PW + kw; if (iw < 0 || iw >= IW) continue;
            acc += x[(nc * IH + ih) * IW + iw];
        }
    }
    y[i] = acc * (1.0f / (float)(KH * KW));
}
extern "C" void kernel_avgpool2d(const float *input, float *output, int N, int C, int IH, int IW,
                                 int KH, int KW, int SH, int SW, int PH, int PW) {
    int OH = (IH + 2 * PH - KH) / SH + 1, OW = (IW + 2 * PW - KW) / SW + 1;
    long tot = (long)N * C * OH * OW;
    k_avgpool2d<<<MB_GRID(tot), MB_THREADS>>>(input, output, N, C, IH, IW, OH, OW, KH, KW, SH, SW, PH, PW);
}

__global__ void k_gap(const float *x, float *y, int NC, int HW) {
    long i = MB_TID; if (i >= NC) return;
    float s = 0.0f; for (int j = 0; j < HW; j++) s += x[i * HW + j];
    y[i] = s / HW;
}
extern "C" void kernel_adaptive_avg_pool2d(const float *input, float *output, int N, int C, int IH, int IW) {
    k_gap<<<MB_GRID(N * C), MB_THREADS>>>(input, output, N * C, IH * IW);
}

// ---------------------------------------------------------------- dense
__global__ void k_linear(const float *x, const float *w, const float *b, float *y, int M, int K, int N) {
    long i = MB_TID; if (i >= (long)M * N) return;
    int m = i / N, n = i % N;
    double acc = b ? b[n] : 0.0f;
    for (int k = 0; k < K; k++) acc += x[(long)m * K + k] * w[(long)n * K + k];
    y[i] = acc;
}
extern "C" void kernel_linear(const float *input, const float *weight, const float *bias, float *output, int M, int K, int N) {
    k_linear<<<MB_GRID((long)M * N), MB_THREADS>>>(input, weight, bias, output, M, K, N);
}

__global__ void k_matmul(const float *A, const float *B, float *C, int M, int K, int N) {
    long i = MB_TID; if (i >= (long)M * N) return;
    int m = i / N, n = i % N;
    double acc = 0.0f;
    for (int k = 0; k < K; k++) acc += A[(long)m * K + k] * B[(long)k * N + n];
    C[i] = acc;
}
extern "C" void kernel_matmul(const float *A, const float *B, float *C, int M, int K, int N) {
    k_matmul<<<MB_GRID((long)M * N), MB_THREADS>>>(A, B, C, M, K, N);
}
// transposed operands: C[m,n] = sum_k A[m*sam + k*sak] * B[k*sbk + n*sbn]
__global__ void k_matmul_s(const float *A, const float *B, float *C, int M, int K, int N,
                           long sam, long sak, long sbk, long sbn) {
    long i = MB_TID; if (i >= (long)M * N) return;
    long m = i / N, n = i % N;
    double acc = 0.0f;
    for (long k = 0; k < K; k++) acc += A[m * sam + k * sak] * B[k * sbk + n * sbn];
    C[i] = acc;
}
// A is [K, M]
extern "C" void kernel_matmul_ta(const float *A, const float *B, float *C, int M, int K, int N) {
    k_matmul_s<<<MB_GRID((long)M * N), MB_THREADS>>>(A, B, C, M, K, N, 1, M, N, 1);
}
// B is [N, K]
extern "C" void kernel_matmul_tb(const float *A, const float *B, float *C, int M, int K, int N) {
    k_matmul_s<<<MB_GRID((long)M * N), MB_THREADS>>>(A, B, C, M, K, N, K, 1, 1, K);
}

// ---------------------------------------------------------------- row ops (one thread per row)
__global__ void k_softmax(const float *x, float *y, int M, int K) {
    long r = MB_TID; if (r >= M) return;
    const float *xr = x + r * K; float *yr = y + r * K;
    float mx = -INFINITY; for (int k = 0; k < K; k++) mx = fmaxf(mx, xr[k]);
    double s = 0.0; for (int k = 0; k < K; k++) { float e = expf(xr[k] - mx); yr[k] = e; s += e; }
    float inv = (float)(1.0 / s);
    for (int k = 0; k < K; k++) yr[k] *= inv;
}
extern "C" void kernel_softmax(const float *input, float *output, int M, int K) {
    k_softmax<<<MB_GRID(M), MB_THREADS>>>(input, output, M, K);
}

__global__ void k_layer_norm(const float *x, const float *g, const float *b, float *y, int M, int K, float eps) {
    long r = MB_TID; if (r >= M) return;
    const float *xr = x + r * K; float *yr = y + r * K;
    float mu = 0.0f; for (int k = 0; k < K; k++) mu += xr[k]; mu /= K;
    float var = 0.0f; for (int k = 0; k < K; k++) { float d = xr[k] - mu; var += d * d; } var /= K;
    float inv = rsqrtf(var + eps);
    for (int k = 0; k < K; k++) yr[k] = (xr[k] - mu) * inv * g[k] + b[k];
}
extern "C" void kernel_layer_norm(const float *input, const float *gamma, const float *beta, float *output,
                                  int M, int K, float eps) {
    k_layer_norm<<<MB_GRID(M), MB_THREADS>>>(input, gamma, beta, output, M, K, eps);
}

// One thread per (bh, query row); S is small for level1 shapes.
__global__ void k_sdpa(const float *Q, const float *Kt, const float *V, float *O, int BH, int S, int D) {
    long r = MB_TID; if (r >= (long)BH * S) return;
    long bh = r / S; int i = r % S;
    const float *q = Q + (bh * S + i) * D; const float *Kb = Kt + bh * S * D; const float *Vb = V + bh * S * D;
    float *o = O + (bh * S + i) * D;
    float scale = rsqrtf((float)D), mx = -INFINITY;
    for (int j = 0; j < S; j++) { float s = 0; for (int d = 0; d < D; d++) s += q[d] * Kb[(long)j * D + d]; mx = fmaxf(mx, s * scale); }
    float sum = 0; for (int d = 0; d < D; d++) o[d] = 0;
    for (int j = 0; j < S; j++) {
        float s = 0; for (int d = 0; d < D; d++) s += q[d] * Kb[(long)j * D + d];
        float p = expf(s * scale - mx); sum += p;
        for (int d = 0; d < D; d++) o[d] += p * Vb[(long)j * D + d];
    }
    for (int d = 0; d < D; d++) o[d] /= sum;
}
extern "C" void kernel_sdpa(const float *Q, const float *K, const float *V, float *output, int BH, int S, int D) {
    k_sdpa<<<MB_GRID((long)BH * S), MB_THREADS>>>(Q, K, V, output, BH, S, D);
}

// ---------------------------------------------------------------- layout
__global__ void k_copy_c(const float *src, float *dst, int N, int c, int C, int c_off, int HW) {
    long i = MB_TID; long tot = (long)N * c * HW; if (i >= tot) return;
    long n = i / ((long)c * HW), r = i % ((long)c * HW);
    dst[(n * C + c_off) * HW + r] = src[i];
}
static void cat_part(const float *src, float *dst, int N, int c, int C, int off, int HW) {
    long tot = (long)N * c * HW; k_copy_c<<<MB_GRID(tot), MB_THREADS>>>(src, dst, N, c, C, off, HW);
}
extern "C" void kernel_cat2_c1(const float *in0, int c0, const float *in1, int c1, float *out, int N, int H, int W) {
    int C = c0 + c1, HW = H * W;
    cat_part(in0, out, N, c0, C, 0, HW); cat_part(in1, out, N, c1, C, c0, HW);
}
extern "C" void kernel_cat3_c1(const float *in0, int c0, const float *in1, int c1, const float *in2, int c2,
                               float *out, int N, int H, int W) {
    int C = c0 + c1 + c2, HW = H * W;
    cat_part(in0, out, N, c0, C, 0, HW); cat_part(in1, out, N, c1, C, c0, HW); cat_part(in2, out, N, c2, C, c0 + c1, HW);
}
extern "C" void kernel_cat4_c1(const float *in0, int c0, const float *in1, int c1, const float *in2, int c2,
                               const float *in3, int c3, float *out, int N, int H, int W) {
    int C = c0 + c1 + c2 + c3, HW = H * W;
    cat_part(in0, out, N, c0, C, 0, HW); cat_part(in1, out, N, c1, C, c0, HW);
    cat_part(in2, out, N, c2, C, c0 + c1, HW); cat_part(in3, out, N, c3, C, c0 + c1 + c2, HW);
}

__global__ void k_upsample(const float *x, float *y, long NC, int IH, int IW, int s) {
    long i = MB_TID; int OH = IH * s, OW = IW * s; if (i >= NC * OH * OW) return;
    int ow = i % OW, oh = (i / OW) % OH; long nc = i / ((long)OW * OH);
    y[i] = x[(nc * IH + oh / s) * IW + ow / s];
}
extern "C" void kernel_upsample_nearest(const float *input, float *output, int N, int C, int IH, int IW, int scale) {
    long tot = (long)N * C * IH * scale * IW * scale;
    k_upsample<<<MB_GRID(tot), MB_THREADS>>>(input, output, (long)N * C, IH, IW, scale);
}


// ================================================================ level3 / generic ops
// (modelblaster/pipeline/reference_kernels_ext.py; same opcodes)

__device__ __forceinline__ void mb_idx6(long i, int d1, int d2, int d3, int d4, int d5,
                                        long &i0, long &i1, long &i2, long &i3, long &i4, long &i5) {
    long r = i;
    i5 = r % d5; r /= d5; i4 = r % d4; r /= d4; i3 = r % d3; r /= d3;
    i2 = r % d2; r /= d2; i1 = r % d1; r /= d1; i0 = r;
}

__global__ void k_strided_copy(const float *x, float *y, long n, int off, int d1, int d2, int d3, int d4,
                               int d5, int s0, int s1, int s2, int s3, int s4, int s5) {
    long i = MB_TID; if (i >= n) return;
    long i0, i1, i2, i3, i4, i5; mb_idx6(i, d1, d2, d3, d4, d5, i0, i1, i2, i3, i4, i5);
    y[i] = x[off + i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3 + i4 * s4 + i5 * s5];
}
extern "C" void kernel_strided_copy(const float *input, float *output, int off,
                                    int d0, int d1, int d2, int d3, int d4, int d5,
                                    int s0, int s1, int s2, int s3, int s4, int s5) {
    long n = (long)d0 * d1 * d2 * d3 * d4 * d5;
    k_strided_copy<<<MB_GRID(n), MB_THREADS>>>(input, output, n, off, d1, d2, d3, d4, d5, s0, s1, s2, s3, s4, s5);
}

__global__ void k_gather(const float *x, const int *idx, float *y, int n) {
    long i = MB_TID; if (i < n) { int j = idx[i]; y[i] = j >= 0 ? x[j] : 0.0f; }
}
extern "C" void kernel_gather(const float *input, const int *index, float *output, int n) {
    k_gather<<<MB_GRID(n), MB_THREADS>>>(input, index, output, n);
}

__global__ void k_cat_slab(const float *x, float *y, int outer, long slab, long ytot, long ybase) {
    long i = MB_TID; if (i >= (long)outer * slab) return;
    long o = i / slab, j = i % slab;
    y[o * ytot + ybase + j] = x[i];
}
extern "C" void kernel_catn(const float *in0, const float *in1, const float *in2, const float *in3,
                            const float *in4, const float *in5, const float *in6, const float *in7,
                            float *output, int n_in, int outer, int inner,
                            int c0, int c1, int c2, int c3, int c4, int c5, int c6, int c7) {
    const float *ins[8] = {in0, in1, in2, in3, in4, in5, in6, in7};
    int cs[8] = {c0, c1, c2, c3, c4, c5, c6, c7};
    long ctot = 0; for (int k = 0; k < n_in; k++) ctot += cs[k];
    long cbase = 0;
    for (int k = 0; k < n_in; k++) {
        long slab = (long)cs[k] * inner;
        k_cat_slab<<<MB_GRID((long)outer * slab), MB_THREADS>>>(ins[k], output, outer, slab, ctot * inner, cbase * inner);
        cbase += cs[k];
    }
}

__device__ float mb_e1(float x, int opc, float p0, float p1) {
    switch (opc) {
    case 0: return -x;  case 1: return expf(x);  case 2: return logf(x);  case 3: return sqrtf(x);
    case 4: return rsqrtf(x);  case 5: return fabsf(x);  case 6: return tanhf(x);
    case 7: return 1.0f / (1.0f + expf(-x));  case 8: return x > 0.0f ? x : 0.0f;
    case 9: return sinf(x);  case 10: return cosf(x);  case 11: return erff(x);
    case 12: return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
    case 13: return 0.5f * x * (1.0f + tanhf(0.79788456080286536f * (x + 0.044715f * x * x * x)));
    case 14: return x / (1.0f + expf(-x));  case 15: return fminf(fmaxf(x, p0), p1);
    case 16: return 1.0f / x;  case 17: return x * x;
    case 18: return x > 20.0f ? x : log1pf(expf(x));  case 19: return x >= 0.0f ? x : p0 * x;
    case 20: return x > 0.0f ? x : p0 * (expf(x) - 1.0f);
    case 21: return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
    case 22: return fminf(fmaxf(x / 6.0f + 0.5f, 0.0f), 1.0f);
    case 23: return log1pf(x);  case 24: return expm1f(x);
    case 25: return (float)((x > 0.0f) - (x < 0.0f));  case 26: return floorf(x);
    case 27: return ceilf(x);  case 28: return nearbyintf(x);  case 29: return x == 0.0f ? 1.0f : 0.0f;
    case 31: return x * tanhf(x > 20.0f ? x : log1pf(expf(x)));
    default: return x;
    }
}
__global__ void k_ewise1(const float *x, float *y, int n, int opc, float p0, float p1) {
    long i = MB_TID; if (i < n) y[i] = mb_e1(x[i], opc, p0, p1);
}
extern "C" void kernel_ewise1(const float *input, float *output, int n, int opc, float p0, float p1) {
    k_ewise1<<<MB_GRID(n), MB_THREADS>>>(input, output, n, opc, p0, p1);
}

__device__ float mb_e2(float x, float y, int opc, float alpha) {
    switch (opc) {
    case 0: return x + alpha * y;  case 1: return x - alpha * y;  case 2: return x * y;
    case 3: return x / y;  case 4: return (x > y || isnan(x)) ? x : y;
    case 5: return (x < y || isnan(x)) ? x : y;  case 6: return powf(x, y);
    case 7: return y != 0.0f ? alpha : x;  case 8: return x == y;  case 9: return x != y;
    case 10: return x < y;  case 11: return x <= y;  case 12: return x > y;  case 13: return x >= y;
    default: return x;
    }
}
__global__ void k_ewise2(const float *a, const float *b, float *y, long n, int opc, float alpha,
                         int d1, int d2, int d3, int d4, int d5,
                         int sa0, int sa1, int sa2, int sa3, int sa4, int sa5,
                         int sb0, int sb1, int sb2, int sb3, int sb4, int sb5) {
    long i = MB_TID; if (i >= n) return;
    long i0, i1, i2, i3, i4, i5; mb_idx6(i, d1, d2, d3, d4, d5, i0, i1, i2, i3, i4, i5);
    float x = a[i0 * sa0 + i1 * sa1 + i2 * sa2 + i3 * sa3 + i4 * sa4 + i5 * sa5];
    float v = b[i0 * sb0 + i1 * sb1 + i2 * sb2 + i3 * sb3 + i4 * sb4 + i5 * sb5];
    y[i] = mb_e2(x, v, opc, alpha);
}
extern "C" void kernel_ewise2(const float *a, const float *b, float *output, int opc, float alpha,
                              int d0, int d1, int d2, int d3, int d4, int d5,
                              int sa0, int sa1, int sa2, int sa3, int sa4, int sa5,
                              int sb0, int sb1, int sb2, int sb3, int sb4, int sb5) {
    long n = (long)d0 * d1 * d2 * d3 * d4 * d5;
    k_ewise2<<<MB_GRID(n), MB_THREADS>>>(a, b, output, n, opc, alpha, d1, d2, d3, d4, d5,
                                         sa0, sa1, sa2, sa3, sa4, sa5, sb0, sb1, sb2, sb3, sb4, sb5);
}

__global__ void k_ewise_scalar(const float *x, float *y, int n, int opc, float c) {
    long i = MB_TID; if (i >= n) return;
    float v = x[i], z;
    switch (opc) {
    case 0: z = v + c; break;  case 1: z = v - c; break;  case 2: z = c - v; break;
    case 3: z = v * c; break;  case 4: z = v / c; break;  case 5: z = c / v; break;
    case 6: z = powf(v, c); break;  case 7: z = powf(c, v); break;
    case 8: z = (v > c || isnan(v)) ? v : c; break;  case 9: z = (v < c || isnan(v)) ? v : c; break;
    case 10: z = v == c; break;  case 11: z = v != c; break;  case 12: z = v < c; break;
    case 13: z = v <= c; break;  case 14: z = v > c; break;  case 15: z = v >= c; break;
    default: z = v;
    }
    y[i] = z;
}
extern "C" void kernel_ewise_scalar(const float *input, float *output, int n, int opc, float c) {
    k_ewise_scalar<<<MB_GRID(n), MB_THREADS>>>(input, output, n, opc, c);
}

// one thread per (outer, inner) output; serial over R
__global__ void k_reduce(const float *in, float *y, int outer, int R, int inner, int opc) {
    long i = MB_TID; if (i >= (long)outer * inner) return;
    long o = i / inner, j = i % inner;
    const float *x = in + o * (long)R * inner + j;
    float acc;
    if (opc == 7 || opc == 8) {
        long best = 0; float bv = x[0];
        for (long r = 1; r < R; r++) { float v = x[r * inner]; if (opc == 7 ? v > bv : v < bv) { bv = v; best = r; } }
        acc = (float)best;
    } else if (opc == 9 || opc == 10) {
        // double accumulators: sequential fp32 sums over ~1e5+ terms drift ~1e-4
        double m = 0.0; for (long r = 0; r < R; r++) m += x[r * inner]; m /= (double)R;
        double q = 0.0; for (long r = 0; r < R; r++) { double d = x[r * inner] - m; q += d * d; }
        acc = (float)(q / (double)(opc == 10 ? R - 1 : R));
    } else {
        bool mm = (opc == 2 || opc == 3);
        double a = mm ? x[0] : (opc == 4 ? 1.0 : 0.0);
        for (long r = mm ? 1 : 0; r < R; r++) {
            float v = x[r * inner];
            if (opc == 2) { if (v > a || isnan(v)) a = v; }
            else if (opc == 3) { if (v < a || isnan(v)) a = v; }
            else if (opc == 4) a *= v;
            else if (opc == 5 || opc == 6) a += (double)v * v;
            else a += v;
        }
        if (opc == 1) a /= (double)R;
        if (opc == 6) a = sqrt(a);
        acc = (float)a;
    }
    y[i] = acc;
}
extern "C" void kernel_reduce(const float *input, float *output, int outer, int R, int inner, int opc) {
    k_reduce<<<MB_GRID((long)outer * inner), MB_THREADS>>>(input, output, outer, R, inner, opc);
}

__global__ void k_softmax_dim(const float *in, float *out, int outer, int R, int inner, int log_out) {
    long i = MB_TID; if (i >= (long)outer * inner) return;
    long o = i / inner, j = i % inner;
    const float *x = in + o * (long)R * inner + j;
    float *y = out + o * (long)R * inner + j;
    float m = x[0];
    for (long r = 1; r < R; r++) m = fmaxf(m, x[r * inner]);
    double sd = 0.0;  // double: R can be ~1e5 (softmax over a flattened volume)
    for (long r = 0; r < R; r++) sd += expf(x[r * inner] - m);
    float s = (float)sd;
    for (long r = 0; r < R; r++) y[r * inner] = log_out ? (x[r * inner] - m) - logf(s) : expf(x[r * inner] - m) / s;
}
extern "C" void kernel_softmax_dim(const float *input, float *output, int outer, int R, int inner, int log_out) {
    k_softmax_dim<<<MB_GRID((long)outer * inner), MB_THREADS>>>(input, output, outer, R, inner, log_out);
}

__global__ void k_maxpool_ex(const float *x, float *y, int N, int C, int IH, int IW, int OH, int OW,
                             int KH, int KW, int SH, int SW, int PH, int PW, int DH, int DW) {
    long i = MB_TID; long tot = (long)N * C * OH * OW;
    if (i >= tot) return;
    int ow = i % OW; int oh = (i / OW) % OH; long nc = i / ((long)OW * OH);
    float m = -INFINITY;
    for (int kh = 0; kh < KH; kh++) {
        int ih = oh * SH - PH + kh * DH; if (ih < 0 || ih >= IH) continue;
        for (int kw = 0; kw < KW; kw++) {
            int iw = ow * SW - PW + kw * DW; if (iw < 0 || iw >= IW) continue;
            float v = x[(nc * IH + ih) * IW + iw]; if (v > m || isnan(v)) m = v;
        }
    }
    y[i] = m;
}
extern "C" void kernel_maxpool2d_ex(const float *input, float *output, int N, int C, int IH, int IW,
                                    int OH, int OW, int KH, int KW, int SH, int SW, int PH, int PW,
                                    int DH, int DW) {
    long tot = (long)N * C * OH * OW;
    k_maxpool_ex<<<MB_GRID(tot), MB_THREADS>>>(input, output, N, C, IH, IW, OH, OW, KH, KW, SH, SW, PH, PW, DH, DW);
}

__global__ void k_conv_t(const float *x, const float *w, const float *b, float *y,
                         int N, int IC, int IH, int IW, int OC, int OH, int OW, int KH, int KW,
                         int SH, int SW, int PH, int PW, int DH, int DW, int G) {
    long i = MB_TID; long tot = (long)N * OC * OH * OW;
    if (i >= tot) return;
    int ow = i % OW; int oh = (i / OW) % OH; int oc = (i / ((long)OW * OH)) % OC; int n = i / ((long)OW * OH * OC);
    int ICpG = IC / G, OCpG = OC / G, g = oc / OCpG, ocg = oc % OCpG;
    double acc = b ? b[oc] : 0.0f;
    for (int kh = 0; kh < KH; kh++) {
        int ihs = oh + PH - kh * DH; if (ihs < 0 || (ihs % SH) != 0) continue;
        int ih = ihs / SH; if (ih >= IH) continue;
        for (int kw = 0; kw < KW; kw++) {
            int iws = ow + PW - kw * DW; if (iws < 0 || (iws % SW) != 0) continue;
            int iw = iws / SW; if (iw >= IW) continue;
            for (int icg = 0; icg < ICpG; icg++) {
                int ic = g * ICpG + icg;
                acc += x[(((long)n * IC + ic) * IH + ih) * IW + iw] * w[(((long)ic * OCpG + ocg) * KH + kh) * KW + kw];
            }
        }
    }
    y[i] = acc;
}
extern "C" void kernel_conv_transpose2d(const float *input, const float *weight, const float *bias, float *output,
                                        int N, int IC, int IH, int IW, int OC, int OH, int OW,
                                        int KH, int KW, int SH, int SW, int PH, int PW, int DH, int DW, int G) {
    long tot = (long)N * OC * OH * OW;
    k_conv_t<<<MB_GRID(tot), MB_THREADS>>>(input, weight, bias, output, N, IC, IH, IW, OC, OH, OW, KH, KW,
                                           SH, SW, PH, PW, DH, DW, G);
}

__global__ void k_bmm(const float *A, const float *B, float *C, int batch, int M, int K, int N) {
    long i = MB_TID; if (i >= (long)batch * M * N) return;
    long n = i % N, m = (i / N) % M, b = i / ((long)M * N);
    const float *a = A + (b * M + m) * K; const float *bb = B + b * (long)K * N + n;
    double acc = 0.0f;
    for (int k = 0; k < K; k++) acc += a[k] * bb[(long)k * N];
    C[i] = acc;
}
extern "C" void kernel_bmm(const float *A, const float *B, float *C, int batch, int M, int K, int N) {
    k_bmm<<<MB_GRID((long)batch * M * N), MB_THREADS>>>(A, B, C, batch, M, K, N);
}

__global__ void k_mul_scalar(const float *x, float *y, int n, float s) { long i = MB_TID; if (i < n) y[i] = x[i] * s; }
__global__ void k_mul(const float *a, const float *b, float *y, int n) { long i = MB_TID; if (i < n) y[i] = a[i] * b[i]; }
__global__ void k_tanh(const float *x, float *y, int n) { long i = MB_TID; if (i < n) y[i] = tanhf(x[i]); }
extern "C" void kernel_mul_scalar(const float *input, float *output, int n, float s) { k_mul_scalar<<<MB_GRID(n), MB_THREADS>>>(input, output, n, s); }
extern "C" void kernel_mul(const float *a, const float *b, float *output, int n) { k_mul<<<MB_GRID(n), MB_THREADS>>>(a, b, output, n); }
extern "C" void kernel_tanh(const float *input, float *output, int n) { k_tanh<<<MB_GRID(n), MB_THREADS>>>(input, output, n); }

// ---- recurrent: one launch per timestep, one thread per (b, j)
__global__ void k_copy(const float *x, float *y, long n) { long i = MB_TID; if (i < n) y[i] = x[i]; }
__global__ void k_rows_from_seq(const float *y, float *h, int B, int T, int H, int t) {
    long i = MB_TID; if (i >= (long)B * H) return;
    long b = i / H, j = i % H;
    h[i] = y[((long)b * T + t) * H + j];
}
__global__ void k_lstm_step(const float *x, const float *w_ih, const float *w_hh, const float *bg,
                            const float *hn, float *cn, float *y, int B, int T, int I, int H, int t) {
    long i = MB_TID; if (i >= (long)B * H) return;
    long b = i / H, j = i % H;
    const float *xt = x + ((long)b * T + t) * I; const float *hp = hn + b * H;
    float g4[4];
    for (int q = 0; q < 4; q++) {
        long row = (long)q * H + j;
        float acc = bg[row];
        for (int k = 0; k < I; k++) acc += w_ih[row * I + k] * xt[k];
        for (int k = 0; k < H; k++) acc += w_hh[row * H + k] * hp[k];
        g4[q] = acc;
    }
    float ig = 1.0f / (1.0f + expf(-g4[0])), fg = 1.0f / (1.0f + expf(-g4[1]));
    float gg = tanhf(g4[2]), og = 1.0f / (1.0f + expf(-g4[3]));
    float c = fg * cn[i] + ig * gg;
    cn[i] = c;
    y[((long)b * T + t) * H + j] = og * tanhf(c);
}
extern "C" void kernel_lstm_seq(const float *x, const float *h0, const float *c0, const float *w_ih,
                                const float *w_hh, const float *b_gates, float *y, float *hn, float *cn,
                                int B, int T, int I, int H, int reverse) {
    long bh = (long)B * H;
    k_copy<<<MB_GRID(bh), MB_THREADS>>>(h0, hn, bh);
    k_copy<<<MB_GRID(bh), MB_THREADS>>>(c0, cn, bh);
    for (int s = 0; s < T; s++) {
        int t = reverse ? T - 1 - s : s;
        k_lstm_step<<<MB_GRID(bh), MB_THREADS>>>(x, w_ih, w_hh, b_gates, hn, cn, y, B, T, I, H, t);
        k_rows_from_seq<<<MB_GRID(bh), MB_THREADS>>>(y, hn, B, T, H, t);
    }
}
__global__ void k_gru_step(const float *x, const float *w_ih, const float *w_hh, const float *b_ih,
                           const float *b_hh, const float *hn, float *y, int B, int T, int I, int H, int t) {
    long i = MB_TID; if (i >= (long)B * H) return;
    long b = i / H, j = i % H;
    const float *xt = x + ((long)b * T + t) * I; const float *hp = hn + b * H;
    float gi[3], gh[3];
    for (int q = 0; q < 3; q++) {
        long row = (long)q * H + j;
        float ai = b_ih[row], ah = b_hh[row];
        for (int k = 0; k < I; k++) ai += w_ih[row * I + k] * xt[k];
        for (int k = 0; k < H; k++) ah += w_hh[row * H + k] * hp[k];
        gi[q] = ai; gh[q] = ah;
    }
    float r = 1.0f / (1.0f + expf(-(gi[0] + gh[0]))), z = 1.0f / (1.0f + expf(-(gi[1] + gh[1])));
    float nn = tanhf(gi[2] + r * gh[2]);
    y[((long)b * T + t) * H + j] = (1.0f - z) * nn + z * hp[j];
}
extern "C" void kernel_gru_seq(const float *x, const float *h0, const float *w_ih, const float *w_hh,
                               const float *b_ih, const float *b_hh, float *y, float *hn,
                               int B, int T, int I, int H, int reverse) {
    long bh = (long)B * H;
    k_copy<<<MB_GRID(bh), MB_THREADS>>>(h0, hn, bh);
    for (int s = 0; s < T; s++) {
        int t = reverse ? T - 1 - s : s;
        k_gru_step<<<MB_GRID(bh), MB_THREADS>>>(x, w_ih, w_hh, b_ih, b_hh, hn, y, B, T, I, H, t);
        k_rows_from_seq<<<MB_GRID(bh), MB_THREADS>>>(y, hn, B, T, H, t);
    }
}

// ---- remaining fp32 MB ops used by level3 IRs
__global__ void k_gelu(const float *x, float *y, int n) {
    long i = MB_TID; if (i < n) { float v = x[i]; y[i] = 0.5f * v * (1.0f + erff(v * 0.70710678118654752f)); }
}
extern "C" void kernel_gelu(const float *input, float *output, int n) { k_gelu<<<MB_GRID(n), MB_THREADS>>>(input, output, n); }
// MB's gelu_exact is the tanh (MinGPT) form
__global__ void k_gelu_tanh(const float *x, float *y, int n) {
    long i = MB_TID;
    if (i < n) { float v = x[i]; y[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v))); }
}
extern "C" void kernel_gelu_exact(const float *input, float *output, int n) { k_gelu_tanh<<<MB_GRID(n), MB_THREADS>>>(input, output, n); }

// sum/mean/max/min over the middle axis of [outer, reduce, inner] == kernel_reduce
extern "C" void kernel_sum_dim(const float *input, float *output, int outer, int reduce, int inner) {
    kernel_reduce(input, output, outer, reduce, inner, 0);
}
extern "C" void kernel_mean_dim(const float *input, float *output, int outer, int reduce, int inner) {
    kernel_reduce(input, output, outer, reduce, inner, 1);
}
extern "C" void kernel_max_dim(const float *input, float *output, int outer, int reduce, int inner) {
    kernel_reduce(input, output, outer, reduce, inner, 2);
}
extern "C" void kernel_min_dim(const float *input, float *output, int outer, int reduce, int inner) {
    kernel_reduce(input, output, outer, reduce, inner, 3);
}

// ---- level2 ops: activations, 3-D conv/pool, group norm
__global__ void k_leaky_relu(const float *x, float *y, int n, float a) { long i = MB_TID; if (i < n) { float v = x[i]; y[i] = v > 0.0f ? v : a * v; } }
__global__ void k_hardtanh(const float *x, float *y, int n, float lo, float hi) { long i = MB_TID; if (i < n) y[i] = fminf(fmaxf(x[i], lo), hi); }
__global__ void k_softplus(const float *x, float *y, int n) {
    long i = MB_TID; if (i < n) { float v = x[i]; y[i] = v > 0.0f ? v + log1pf(expf(-v)) : log1pf(expf(v)); }
}
extern "C" void kernel_leaky_relu(const float *input, float *output, int n, float negative_slope) {
    k_leaky_relu<<<MB_GRID(n), MB_THREADS>>>(input, output, n, negative_slope);
}
extern "C" void kernel_hardtanh(const float *input, float *output, int n, float min_val, float max_val) {
    k_hardtanh<<<MB_GRID(n), MB_THREADS>>>(input, output, n, min_val, max_val);
}
extern "C" void kernel_softplus(const float *input, float *output, int n) { k_softplus<<<MB_GRID(n), MB_THREADS>>>(input, output, n); }

// output index i -> (n, c, od, oh, ow) for an [N, C, OD, OH, OW] tensor
#define MB_IDX5(i, C, OD, OH, OW) \
    int ow = (i) % (OW); int oh = ((i) / (OW)) % (OH); int od = ((i) / ((long)(OW) * (OH))) % (OD); \
    int c_ = ((i) / ((long)(OW) * (OH) * (OD))) % (C); int n_ = (i) / ((long)(OW) * (OH) * (OD) * (C));

__global__ void k_conv3d(const float *x, const float *w, const float *b, float *y,
                         int N, int IC, int ID, int IH, int IW, int OC, int OD, int OH, int OW,
                         int KD, int KH, int KW, int SD, int SH, int SW, int PD, int PH, int PW,
                         int DD, int DH, int DW, int G) {
    long i = MB_TID; if (i >= (long)N * OC * OD * OH * OW) return;
    MB_IDX5(i, OC, OD, OH, OW)
    int ICpG = IC / G, g = c_ / (OC / G);
    double acc = b ? b[c_] : 0.0f;
    for (int icg = 0; icg < ICpG; icg++) {
        int ic = g * ICpG + icg;
        const float *xp = x + ((long)n_ * IC + ic) * ID * IH * IW;
        const float *wp = w + ((long)c_ * ICpG + icg) * KD * KH * KW;
        for (int kd = 0; kd < KD; kd++) {
            int id = od * SD - PD + kd * DD; if (id < 0 || id >= ID) continue;
            for (int kh = 0; kh < KH; kh++) {
                int ih = oh * SH - PH + kh * DH; if (ih < 0 || ih >= IH) continue;
                for (int kw = 0; kw < KW; kw++) {
                    int iw = ow * SW - PW + kw * DW; if (iw < 0 || iw >= IW) continue;
                    acc += xp[((long)id * IH + ih) * IW + iw] * wp[(kd * KH + kh) * KW + kw];
                }
            }
        }
    }
    y[i] = acc;
}
extern "C" void kernel_conv3d(const float *input, const float *weight, const float *bias, float *output,
                              int N, int IC, int ID, int IH, int IW, int OC, int OD, int OH, int OW,
                              int KD, int KH, int KW, int SD, int SH, int SW, int PD, int PH, int PW,
                              int DD, int DH, int DW, int G) {
    long tot = (long)N * OC * OD * OH * OW;
    k_conv3d<<<MB_GRID(tot), MB_THREADS>>>(input, weight, bias, output, N, IC, ID, IH, IW, OC, OD, OH, OW,
                                           KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW, G);
}

// weight is torch's [IC, OC/G, KD, KH, KW]
__global__ void k_conv_t3d(const float *x, const float *w, const float *b, float *y,
                           int N, int IC, int ID, int IH, int IW, int OC, int OD, int OH, int OW,
                           int KD, int KH, int KW, int SD, int SH, int SW, int PD, int PH, int PW,
                           int DD, int DH, int DW, int G) {
    long i = MB_TID; if (i >= (long)N * OC * OD * OH * OW) return;
    MB_IDX5(i, OC, OD, OH, OW)
    int ICpG = IC / G, OCpG = OC / G, g = c_ / OCpG, ocg = c_ % OCpG;
    double acc = b ? b[c_] : 0.0f;
    for (int kd = 0; kd < KD; kd++) {
        int ids = od + PD - kd * DD; if (ids < 0 || (ids % SD) != 0) continue;
        int id = ids / SD; if (id >= ID) continue;
        for (int kh = 0; kh < KH; kh++) {
            int ihs = oh + PH - kh * DH; if (ihs < 0 || (ihs % SH) != 0) continue;
            int ih = ihs / SH; if (ih >= IH) continue;
            for (int kw = 0; kw < KW; kw++) {
                int iws = ow + PW - kw * DW; if (iws < 0 || (iws % SW) != 0) continue;
                int iw = iws / SW; if (iw >= IW) continue;
                for (int icg = 0; icg < ICpG; icg++) {
                    int ic = g * ICpG + icg;
                    acc += x[((((long)n_ * IC + ic) * ID + id) * IH + ih) * IW + iw] *
                           w[((((long)ic * OCpG + ocg) * KD + kd) * KH + kh) * KW + kw];
                }
            }
        }
    }
    y[i] = acc;
}
extern "C" void kernel_conv_transpose3d(const float *input, const float *weight, const float *bias, float *output,
                                        int N, int IC, int ID, int IH, int IW, int OC, int OD, int OH, int OW,
                                        int KD, int KH, int KW, int SD, int SH, int SW, int PD, int PH, int PW,
                                        int DD, int DH, int DW, int G) {
    long tot = (long)N * OC * OD * OH * OW;
    k_conv_t3d<<<MB_GRID(tot), MB_THREADS>>>(input, weight, bias, output, N, IC, ID, IH, IW, OC, OD, OH, OW,
                                             KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW, G);
}

// avg = 1 -> sum / (KD*KH*KW) (MB's count_include_pad semantics), else max
__global__ void k_pool3d(const float *x, float *y, int N, int C, int ID, int IH, int IW, int OD, int OH, int OW,
                         int KD, int KH, int KW, int SD, int SH, int SW, int PD, int PH, int PW,
                         int DD, int DH, int DW, int avg) {
    long i = MB_TID; if (i >= (long)N * C * OD * OH * OW) return;
    MB_IDX5(i, C, OD, OH, OW)
    const float *xp = x + ((long)n_ * C + c_) * ID * IH * IW;
    float acc = avg ? 0.0f : -INFINITY;
    for (int kd = 0; kd < KD; kd++) {
        int id = od * SD - PD + kd * DD; if (id < 0 || id >= ID) continue;
        for (int kh = 0; kh < KH; kh++) {
            int ih = oh * SH - PH + kh * DH; if (ih < 0 || ih >= IH) continue;
            for (int kw = 0; kw < KW; kw++) {
                int iw = ow * SW - PW + kw * DW; if (iw < 0 || iw >= IW) continue;
                float v = xp[((long)id * IH + ih) * IW + iw];
                acc = avg ? acc + v : fmaxf(acc, v);
            }
        }
    }
    y[i] = avg ? acc / (float)(KD * KH * KW) : acc;
}
extern "C" void kernel_maxpool3d(const float *input, float *output, int N, int C, int ID, int IH, int IW,
                                 int OD, int OH, int OW, int KD, int KH, int KW, int SD, int SH, int SW,
                                 int PD, int PH, int PW, int DD, int DH, int DW) {
    long tot = (long)N * C * OD * OH * OW;
    k_pool3d<<<MB_GRID(tot), MB_THREADS>>>(input, output, N, C, ID, IH, IW, OD, OH, OW, KD, KH, KW,
                                           SD, SH, SW, PD, PH, PW, DD, DH, DW, 0);
}
extern "C" void kernel_avgpool3d(const float *input, float *output, int N, int C, int ID, int IH, int IW,
                                 int OD, int OH, int OW, int KD, int KH, int KW, int SD, int SH, int SW,
                                 int PD, int PH, int PW) {
    long tot = (long)N * C * OD * OH * OW;
    k_pool3d<<<MB_GRID(tot), MB_THREADS>>>(input, output, N, C, ID, IH, IW, OD, OH, OW, KD, KH, KW,
                                           SD, SH, SW, PD, PH, PW, 1, 1, 1, 1);
}

// One block per (n, group): two-pass mean / variance with a shared-memory tree.
__device__ float mb_block_sum(float v, float *sh) {
    sh[threadIdx.x] = v; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float r = sh[0]; __syncthreads();
    return r;
}
__global__ void k_group_norm(const float *x, const float *gm, const float *bt, float *y,
                             int C, int G, int HW, float eps) {
    __shared__ float sh[MB_THREADS];
    int n = blockIdx.x / G, g = blockIdx.x % G, cpg = C / G;
    long cnt = (long)cpg * HW, base = ((long)n * C + (long)g * cpg) * HW;
    const float *xp = x + base; float *yp = y + base;
    float s = 0.0f;
    for (long j = threadIdx.x; j < cnt; j += blockDim.x) s += xp[j];
    float mean = mb_block_sum(s, sh) / (float)cnt;
    float q = 0.0f;
    for (long j = threadIdx.x; j < cnt; j += blockDim.x) { float d = xp[j] - mean; q += d * d; }
    float inv = rsqrtf(mb_block_sum(q, sh) / (float)cnt + eps);
    for (long j = threadIdx.x; j < cnt; j += blockDim.x) {
        int c = g * cpg + (int)(j / HW);
        yp[j] = (xp[j] - mean) * inv * gm[c] + bt[c];
    }
}
extern "C" void kernel_group_norm(const float *input, const float *gamma, const float *beta, float *output,
                                  int N, int C, int G, int HW, float eps) {
    k_group_norm<<<N * G, MB_THREADS>>>(input, gamma, beta, output, C, G, HW, eps);
}

// ---------------------------------------------------------- runtime-index ops
// Index tensors are float (the fp32 pipeline carries them as exact floats).
__device__ __forceinline__ long mb_wrap(float f, long A) { long a = (long)f; return a < 0 ? a + A : a; }

__global__ void k_index_select(const float *x, const float *idx, float *y, long n, int A, int J, int inner) {
    long k = MB_TID; if (k >= n) return;
    long i = k % inner, r = k / inner, j = r % J, o = r / J;
    y[k] = x[(o * A + mb_wrap(idx[j], A)) * inner + i];
}
extern "C" void kernel_index_select(const float *input, const float *index, float *output,
                                    int outer, int A, int J, int inner) {
    long n = (long)outer * J * inner;
    k_index_select<<<MB_GRID(n), MB_THREADS>>>(input, index, output, n, A, J, inner);
}

__global__ void k_gather_dim(const float *x, const float *idx, float *y, long n, int A, int J, int inner) {
    long k = MB_TID; if (k >= n) return;
    long i = k % inner, o = k / ((long)J * inner);
    y[k] = x[(o * A + mb_wrap(idx[k], A)) * inner + i];
}
extern "C" void kernel_gather_dim(const float *input, const float *index, float *output,
                                  int outer, int A, int J, int inner) {
    long n = (long)outer * J * inner;
    k_gather_dim<<<MB_GRID(n), MB_THREADS>>>(input, index, output, n, A, J, inner);
}

__global__ void k_scatter_dim(const float *idx, const float *src, float *y, long n, int A, int J,
                              int inner, int mode) {
    long k = MB_TID; if (k >= n) return;
    long i = k % inner, o = k / ((long)J * inner);
    float *d = y + (o * A + mb_wrap(idx[k], A)) * inner + i;
    if (mode) atomicAdd(d, src[k]); else *d = src[k];
}
extern "C" void kernel_scatter_dim(const float *input, const float *index, const float *src,
                                   float *output, int outer, int A, int J, int inner, int mode) {
    long nb = (long)outer * A * inner, n = (long)outer * J * inner;
    k_copy<<<MB_GRID(nb), MB_THREADS>>>(input, output, nb);
    k_scatter_dim<<<MB_GRID(n), MB_THREADS>>>(index, src, output, n, A, J, inner, mode);
}

__device__ __forceinline__ bool mb_topk_before(float a, long ia, float b, long ib, int largest) {
    bool an = isnan(a), bn = isnan(b);
    if (an || bn) { if (an && bn) return ia < ib; return largest ? an : bn; }
    if (a != b) return largest ? a > b : a < b;
    return ia < ib;
}
// one thread per (outer, inner) row: insertion into a sorted window of K
__global__ void k_topk(const float *in, float *val, float *ind, long rows, int A, int inner, int K, int largest) {
    long r = MB_TID; if (r >= rows) return;
    long o = r / inner, i = r % inner;
    const float *x = in + o * (long)A * inner + i;
    float *v = val + o * (long)K * inner + i, *ix = ind + o * (long)K * inner + i;
    long n = 0;
    for (long a = 0; a < A; a++) {
        float xa = x[a * inner];
        if (n == K && !mb_topk_before(xa, a, v[(K - 1) * (long)inner], (long)ix[(K - 1) * (long)inner], largest))
            continue;
        long p = n < K ? n++ : K - 1;
        while (p > 0 && mb_topk_before(xa, a, v[(p - 1) * inner], (long)ix[(p - 1) * inner], largest)) {
            v[p * inner] = v[(p - 1) * inner]; ix[p * inner] = ix[(p - 1) * inner]; p--;
        }
        v[p * inner] = xa; ix[p * inner] = (float)a;
    }
}
__global__ void k_scan(const float *in, float *out, long rows, int A, int inner, int prod) {
    long t = MB_TID; if (t >= rows) return;
    long o = t / inner, i = t % inner;
    double acc = prod ? 1.0 : 0.0;
    for (long a = 0; a < A; a++) {
        long idx = (o * A + a) * inner + i;
        acc = prod ? acc * in[idx] : acc + in[idx];
        out[idx] = (float)acc;
    }
}
extern "C" void kernel_cumsum(const float *input, float *output, int outer, int axis, int inner) {
    long rows = (long)outer * inner;
    k_scan<<<MB_GRID(rows), MB_THREADS>>>(input, output, rows, axis, inner, 0);
}
extern "C" void kernel_cumprod(const float *input, float *output, int outer, int axis, int inner) {
    long rows = (long)outer * inner;
    k_scan<<<MB_GRID(rows), MB_THREADS>>>(input, output, rows, axis, inner, 1);
}

extern "C" void kernel_topk(const float *input, float *values, float *indices, int outer,
                            int A, int inner, int K, int largest) {
    long rows = (long)outer * inner;
    k_topk<<<MB_GRID(rows), MB_THREADS>>>(input, values, indices, rows, A, inner, K, largest);
}

__global__ void k_convnd(const float *x, const float *w, const float *b, float *y, long tot,
                         int IC, int ID, int IH, int IW, int OC, int OD, int OH, int OW,
                         int KD, int KH, int KW, int SD, int SH, int SW, int PD, int PH, int PW,
                         int DD, int DH, int DW, int G) {
    long t = MB_TID; if (t >= tot) return;
    long ow = t % OW, r = t / OW, oh = r % OH; r /= OH;
    long od = r % OD; r /= OD;
    long oc = r % OC, n = r / OC;
    int ICg = IC / G, OCg = OC / G;
    long ic0 = (oc / OCg) * ICg;
    double acc = b ? b[oc] : 0.0f;
    for (long ic = 0; ic < ICg; ic++)
    for (long kd = 0; kd < KD; kd++) {
        long id = od * SD - PD + kd * DD; if (id < 0 || id >= ID) continue;
        for (long kh = 0; kh < KH; kh++) {
            long ih = oh * SH - PH + kh * DH; if (ih < 0 || ih >= IH) continue;
            for (long kw = 0; kw < KW; kw++) {
                long iw = ow * SW - PW + kw * DW; if (iw < 0 || iw >= IW) continue;
                acc += x[(((n * IC + ic0 + ic) * ID + id) * IH + ih) * IW + iw]
                     * w[(((oc * ICg + ic) * KD + kd) * KH + kh) * KW + kw];
            }
        }
    }
    y[t] = acc;
}
extern "C" void kernel_convnd(const float *in0, const float *in1, const float *in2, float *output,
                              int N, int IC, int ID, int IH, int IW, int OC, int OD, int OH, int OW,
                              int KD, int KH, int KW, int SD, int SH, int SW, int PD, int PH, int PW,
                              int DD, int DH, int DW, int G) {
    long tot = (long)N * OC * OD * OH * OW;
    k_convnd<<<MB_GRID(tot), MB_THREADS>>>(in0, in1, in2, output, tot, IC, ID, IH, IW, OC, OD, OH, OW,
                                          KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW, G);
}

__global__ void k_strided_scatter(const float *src, float *y, long m, int off, int d1, int d2, int d3,
                                  int d4, int d5, int s0, int s1, int s2, int s3, int s4, int s5) {
    long i = MB_TID; if (i >= m) return;
    long i0, i1, i2, i3, i4, i5; mb_idx6(i, d1, d2, d3, d4, d5, i0, i1, i2, i3, i4, i5);
    y[off + i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3 + i4 * s4 + i5 * s5] = src[i];
}
extern "C" void kernel_strided_scatter(const float *input, const float *src, float *output, int n, int off,
                                       int d0, int d1, int d2, int d3, int d4, int d5,
                                       int s0, int s1, int s2, int s3, int s4, int s5) {
    long m = (long)d0 * d1 * d2 * d3 * d4 * d5;
    k_copy<<<MB_GRID(n), MB_THREADS>>>(input, output, n);
    k_strided_scatter<<<MB_GRID(m), MB_THREADS>>>(src, output, m, off, d1, d2, d3, d4, d5, s0, s1, s2, s3, s4, s5);
}

// ---------------------------------------------------------------- Level1: elementwise / index ops
__global__ void k_swish(const float *x, float *y, long n) { long i = MB_TID; if (i < n) { float v = x[i]; y[i] = v / (1.0f + expf(-v)); } }
__global__ void k_selu(const float *x, float *y, long n) {
    long i = MB_TID; if (i >= n) return;
    float v = x[i];
    y[i] = 1.0507009873554805f * (v > 0.0f ? v : 1.6732632423543772f * (expf(v) - 1.0f));
}
__global__ void k_hardsigmoid(const float *x, float *y, long n) { long i = MB_TID; if (i < n) y[i] = fminf(fmaxf(x[i] / 6.0f + 0.5f, 0.0f), 1.0f); }
__global__ void k_softsign(const float *x, float *y, long n) { long i = MB_TID; if (i < n) { float v = x[i]; y[i] = v / (1.0f + fabsf(v)); } }
__global__ void k_log(const float *x, float *y, long n) { long i = MB_TID; if (i < n) y[i] = logf(x[i]); }
__global__ void k_tri(const float *x, float *y, int M, int N, int d, int upper) {
    long i = MB_TID; if (i >= (long)M * N) return;
    long r = i / N, c = i % N;
    y[i] = (upper ? c >= r + d : c <= r + d) ? x[i] : 0.0f;
}
__global__ void k_diag_matmul(const float *a, const float *b, float *y, int N, int M) {
    long i = MB_TID; if (i < (long)N * M) y[i] = a[i / M] * b[i];
}
__global__ void k_flip(const float *x, float *y, long n, int A, int inner) {
    long i = MB_TID; if (i >= n) return;
    long j = i % inner, a = (i / inner) % A, o = i / ((long)inner * A);
    y[i] = x[(o * A + (A - 1 - a)) * inner + j];
}

extern "C" void kernel_swish(const float *input, float *output, int n) { k_swish<<<MB_GRID(n), MB_THREADS>>>(input, output, n); }
extern "C" void kernel_selu(const float *input, float *output, int n) { k_selu<<<MB_GRID(n), MB_THREADS>>>(input, output, n); }
extern "C" void kernel_hardsigmoid(const float *input, float *output, int n) { k_hardsigmoid<<<MB_GRID(n), MB_THREADS>>>(input, output, n); }
extern "C" void kernel_softsign(const float *input, float *output, int n) { k_softsign<<<MB_GRID(n), MB_THREADS>>>(input, output, n); }
extern "C" void kernel_log(const float *input, float *output, int n) { k_log<<<MB_GRID(n), MB_THREADS>>>(input, output, n); }
extern "C" void kernel_triu(const float *input, float *output, int M, int N, int diagonal) {
    k_tri<<<MB_GRID((long)M * N), MB_THREADS>>>(input, output, M, N, diagonal, 1);
}
extern "C" void kernel_tril(const float *input, float *output, int M, int N, int diagonal) {
    k_tri<<<MB_GRID((long)M * N), MB_THREADS>>>(input, output, M, N, diagonal, 0);
}
extern "C" void kernel_diag_matmul(const float *a, const float *b, float *output, int N, int M) {
    k_diag_matmul<<<MB_GRID((long)N * M), MB_THREADS>>>(a, b, output, N, M);
}
// A stored [K, M], B stored [N, K]
extern "C" void kernel_matmul_tatb(const float *A, const float *B, float *C, int M, int K, int N) {
    k_matmul_s<<<MB_GRID((long)M * N), MB_THREADS>>>(A, B, C, M, K, N, 1, M, 1, K);
}
extern "C" void kernel_log_softmax(const float *input, float *output, int M, int K) {
    kernel_softmax_dim(input, output, M, K, 1, 1);
}
extern "C" void kernel_flip(const float *input, float *output, int outer, int axis, int inner) {
    long n = (long)outer * axis * inner;
    k_flip<<<MB_GRID(n), MB_THREADS>>>(input, output, n, axis, inner);
}

// ---------------------------------------------------------------- Level1: per-(outer, inner) reductions
// one thread per (o, i) column of [outer, reduce, inner]; double accumulators
__global__ void k_col_norm(const float *x, float *y, int outer, int R, int inner, int kind, float eps) {
    long i = MB_TID; if (i >= (long)outer * inner) return;
    long o = i / inner, j = i % inner;
    const float *xp = x + o * (long)R * inner + j; float *yp = y + o * (long)R * inner + j;
    double a = 0.0;
    for (long r = 0; r < R; r++) { double v = xp[r * inner]; a += kind == 1 ? fabs(v) : v * v; }
    double d = kind == 0 ? sqrt(a / R + eps) : kind == 1 ? a / R : sqrt(a);   // rms, mean|x|, l2
    for (long r = 0; r < R; r++) yp[r * inner] = xp[r * inner] / d;
}
__global__ void k_arg_dim(const float *x, int64_t *y, int outer, int R, int inner, int is_max) {
    long i = MB_TID; if (i >= (long)outer * inner) return;
    long o = i / inner, j = i % inner;
    const float *xp = x + o * (long)R * inner + j;
    long best = 0; float bv = xp[0];
    for (long r = 1; r < R; r++) { float v = xp[r * inner]; if (is_max ? v > bv : v < bv) { bv = v; best = r; } }
    y[i] = best;                                   // ties: first index, as torch
}
// one thread per kept row: out[b] = [0, x0, x0+x1, ...]
__global__ void k_excl_cumsum(const float *x, float *y, int Bout, int N) {
    long b = MB_TID; if (b >= Bout) return;
    const float *xr = x + b * (long)N; float *yr = y + b * (long)(N + 1);
    double a = 0.0; yr[0] = 0.0f;
    for (long k = 0; k < N; k++) { a += xr[k]; yr[k + 1] = (float)a; }
}

extern "C" void kernel_rms_norm(const float *input, float *output, int outer, int reduce, int inner, float eps) {
    k_col_norm<<<MB_GRID((long)outer * inner), MB_THREADS>>>(input, output, outer, reduce, inner, 0, eps);
}
extern "C" void kernel_mean_abs_norm(const float *input, float *output, int outer, int reduce, int inner) {
    k_col_norm<<<MB_GRID((long)outer * inner), MB_THREADS>>>(input, output, outer, reduce, inner, 1, 0.0f);
}
extern "C" void kernel_l2_norm(const float *input, float *output, int outer, int reduce, int inner) {
    k_col_norm<<<MB_GRID((long)outer * inner), MB_THREADS>>>(input, output, outer, reduce, inner, 2, 0.0f);
}
extern "C" void kernel_argmax_dim(const float *input, int64_t *output, int outer, int reduce, int inner) {
    k_arg_dim<<<MB_GRID((long)outer * inner), MB_THREADS>>>(input, output, outer, reduce, inner, 1);
}
extern "C" void kernel_argmin_dim(const float *input, int64_t *output, int outer, int reduce, int inner) {
    k_arg_dim<<<MB_GRID((long)outer * inner), MB_THREADS>>>(input, output, outer, reduce, inner, 0);
}
extern "C" void kernel_exclusive_cumsum(const float *x, float *output, int Bout, int N) {
    k_excl_cumsum<<<MB_GRID(Bout), MB_THREADS>>>(x, output, Bout, N);
}

// ---------------------------------------------------------------- Level1: whole-tensor reductions (losses, norms)
// two stages: block partial sums (double) into a cached device scratch, then one
// block sums the partials and writes the scaled scalar
#define MB_RBLOCKS 1024
__device__ double mb_block_sum_d(double v, double *shd) {
    shd[threadIdx.x] = v; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shd[threadIdx.x] += shd[threadIdx.x + s];
        __syncthreads();
    }
    double r = shd[0]; __syncthreads();
    return r;
}
static double *mb_partials(long n) {
    static double *p = nullptr; static long cap = 0;
    if (n > cap) { if (p) cudaFree(p); cudaMalloc(&p, n * sizeof(double)); cap = n; }
    return p;
}
static int mb_rblocks(long n) { long b = (n + MB_THREADS - 1) / MB_THREADS; return (int)(b < 1 ? 1 : (b > MB_RBLOCKS ? MB_RBLOCKS : b)); }
__global__ void k_sum_partials(const double *part, int nb, double *dst, float *out, double scale) {
    __shared__ double shd[MB_THREADS];
    double v = 0.0;
    for (int i = threadIdx.x; i < nb; i += blockDim.x) v += part[i];
    v = mb_block_sum_d(v, shd);
    if (threadIdx.x == 0) { if (dst) dst[0] = v; if (out) out[0] = (float)(v * scale); }
}
// kind: 0 hinge(a=pred, b=targ, m=targ_len), 1 mse, 2 huber(p=beta), 3 kldiv, 4 sum of squares(a)
__global__ void k_loss_partial(const float *a, const float *b, long n, long m, float p, int kind, double *part) {
    __shared__ double shd[MB_THREADS];
    double acc = 0.0;
    for (long i = MB_TID; i < n; i += (long)gridDim.x * blockDim.x) {
        double x = a[i], t;
        if (kind == 0) { t = 1.0 - x * (double)b[i % m]; t = t > 0.0 ? t : 0.0; }
        else if (kind == 1) { double d = x - (double)b[i]; t = d * d; }
        else if (kind == 2) { double d = fabs(x - (double)b[i]); t = d < p ? 0.5 * d * d / p : d - 0.5 * p; }
        else if (kind == 3) { double g = b[i]; t = g > 0.0 ? g * (log(g) - x) : 0.0; }
        else t = x * x;
        acc += t;
    }
    acc = mb_block_sum_d(acc, shd);
    if (threadIdx.x == 0) part[blockIdx.x] = acc;
}
static void mb_loss(const float *a, const float *b, long n, long m, float p, int kind, float *out, double scale) {
    int nb = mb_rblocks(n); double *part = mb_partials(nb);
    k_loss_partial<<<nb, MB_THREADS>>>(a, b, n, m, p, kind, part);
    k_sum_partials<<<1, MB_THREADS>>>(part, nb, nullptr, out, scale);
}
// one block per row: loss_i = logsumexp(x_i) - x_i[t_i]
__global__ void k_ce_rows(const float *x, const float *t, int C, double *part) {
    __shared__ double shd[MB_THREADS];
    const float *xr = x + (long)blockIdx.x * C;
    double mx = -INFINITY;
    for (int c = threadIdx.x; c < C; c += blockDim.x) mx = fmax(mx, (double)xr[c]);
    shd[threadIdx.x] = mx; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) shd[threadIdx.x] = fmax(shd[threadIdx.x], shd[threadIdx.x + s]); __syncthreads(); }
    mx = shd[0]; __syncthreads();
    double se = 0.0;
    for (int c = threadIdx.x; c < C; c += blockDim.x) se += exp((double)xr[c] - mx);
    se = mb_block_sum_d(se, shd);
    if (threadIdx.x == 0) part[blockIdx.x] = mx + log(se) - (double)xr[(int)t[blockIdx.x]];
}
// one block per sample: max(0, ||a-p|| - ||a-n|| + margin)
__global__ void k_triplet_rows(const float *a, const float *p, const float *ng, int F, float margin, double *part) {
    __shared__ double shd[MB_THREADS];
    long base = (long)blockIdx.x * F;
    double dp = 0.0, dn = 0.0;
    for (int f = threadIdx.x; f < F; f += blockDim.x) {
        double x = a[base + f], u = x - (double)p[base + f], w = x - (double)ng[base + f];
        dp += u * u; dn += w * w;
    }
    dp = mb_block_sum_d(dp, shd); dn = mb_block_sum_d(dn, shd);
    if (threadIdx.x == 0) { double l = sqrt(dp) - sqrt(dn) + margin; part[blockIdx.x] = l > 0.0 ? l : 0.0; }
}
__global__ void k_div_by_norm(const float *x, float *y, long n, const double *ss) {
    long i = MB_TID; if (i < n) y[i] = (float)(x[i] / sqrt(ss[0]));
}

extern "C" void kernel_hinge_loss(const float *pred, const float *targ, float *output, int n, int targ_len) {
    mb_loss(pred, targ, n, targ_len, 0.0f, 0, output, 1.0 / n);
}
extern "C" void kernel_mse_loss(const float *a, const float *b, float *output, int n) {
    mb_loss(a, b, n, n, 0.0f, 1, output, 1.0 / n);
}
extern "C" void kernel_huber_loss(const float *a, const float *b, float *output, int n, float beta) {
    mb_loss(a, b, n, n, beta, 2, output, 1.0 / n);
}
extern "C" void kernel_kldiv_loss(const float *log_input, const float *target, float *output, int N, int C) {
    mb_loss(log_input, target, (long)N * C, 1, 0.0f, 3, output, 1.0 / N);
}
extern "C" void kernel_cross_entropy_loss(const float *logits, const float *targets, float *output, int N, int C) {
    double *part = mb_partials(N);
    k_ce_rows<<<N, MB_THREADS>>>(logits, targets, C, part);
    k_sum_partials<<<1, MB_THREADS>>>(part, N, nullptr, output, 1.0 / N);
}
extern "C" void kernel_triplet_loss(const float *anchor, const float *pos, const float *neg, float *output, int B, int F, float margin) {
    double *part = mb_partials(B);
    k_triplet_rows<<<B, MB_THREADS>>>(anchor, pos, neg, F, margin, part);
    k_sum_partials<<<1, MB_THREADS>>>(part, B, nullptr, output, 1.0 / B);
}
extern "C" void kernel_frobenius_norm(const float *input, float *output, int n) {
    int nb = mb_rblocks(n); double *part = mb_partials(nb + 1);
    k_loss_partial<<<nb, MB_THREADS>>>(input, nullptr, n, 1, 0.0f, 4, part + 1);
    k_sum_partials<<<1, MB_THREADS>>>(part + 1, nb, part, nullptr, 1.0);
    k_div_by_norm<<<MB_GRID(n), MB_THREADS>>>(input, output, n, part);
}
