// Vulkan launchers for ModelBlaster's per-op C ABI (fp32).
//
// Every extern "C" function has EXACTLY the signature of the matching
// KernelSpec.signature in pipeline/reference_kernels.py (the same set of
// launchers as gpu_eval/harness/cuda/mb_cuda_kernels.cu, whose semantics --
// opcode tables, layouts, ceil/floor conventions -- these follow). Pointers are
// host addresses inside mb_vk_alloc() buffers; mb_pc::p turns them into GPU
// addresses. The shaders live in shaders/*.comp and are embedded as SPIR-V.
//
// Unlike the CUDA seeds (one thread per output, double accumulators), row
// reductions use one workgroup per row with an fp32 tree, and every dense
// product goes through one tiled GEMM (shaders/gemm.comp).
#include "mb_vk_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define PIPE(name)                                   \
    static int _pipe = -1;                           \
    if (_pipe < 0) _pipe = mb_vk_pipeline(name);     \
    int pipe = _pipe

// one workgroup per line, folded into 2-D
static void lines(int pipe, long nlines, const mb_pc &pc) {
    if (nlines <= 0) return;
    uint32_t mx = mb_vk_max_group_x() < 65535 ? mb_vk_max_group_x() : 65535;
    uint32_t gx = nlines < (long)mx ? (uint32_t)nlines : mx;
    uint32_t gy = (uint32_t)((nlines + gx - 1) / gx);
    mb_vk_dispatch(pipe, gx, gy, 1, pc);
}

// ------------------------------------------------------------------ elementwise
// The elementwise / data-movement shaders have a VEC = 4 variant (specialization
// constant 0): one vec4 per invocation, for 4x the bytes in flight per wave
// (a plain one-float-per-invocation stream reaches about half the Steam Deck's
// DRAM bandwidth). Taken when sizes and 16-byte alignment allow it.
static bool al16(const void *p) { return (mb_vk_devaddr(p) & 15) == 0; }
static bool al16(const float *p, long off) { return ((mb_vk_devaddr(p) + 4 * off) & 15) == 0; }
static int vec_pipe(const char *name, int vec, int cache[2]) {
    int &p = cache[vec == 4 ? 1 : 0];
    if (p < 0) {
        int c[1] = {vec};
        p = mb_vk_pipeline_spec(name, c, 1);
    }
    return p;
}
// the ewise shaders also take the opcode as constant 1 (the runtime caches per value)
static int op_pipe(const char *name, int vec, int opc) {
    int c[2] = {vec, opc};
    return mb_vk_pipeline_spec(name, c, 2);
}
static void e1(const float *x, float *y, long n, int opc, float p0 = 0.0f, float p1 = 0.0f) {
    int vec = (n % 4 == 0 && al16(x) && al16(y)) ? 4 : 1;
    mb_pc pc;
    pc.p(x).p(y).i((int)n).i(opc).f(p0).f(p1);
    mb_vk_launch1d(op_pipe("ewise1", vec, opc), n / vec, pc);
}
static void es(const float *x, float *y, long n, int opc, float c) {
    int vec = (n % 4 == 0 && al16(x) && al16(y)) ? 4 : 1;
    mb_pc pc;
    pc.p(x).p(y).i((int)n).i(opc).f(c);
    mb_vk_launch1d(op_pipe("ewise_scalar", vec, opc), n / vec, pc);
}
// Coalesce a 6-d strided iteration space (outermost dim first; the output is
// contiguous) over nops operands' strides: drop size-1 dims and merge a dim
// into its inner neighbour when that is contiguous in every operand. The
// active dims end up innermost (d = 1, strides 0 outside them); returns how
// many there are (>= 1), so a shader divides only by those.
static int coalesce(int d[6], int *st[], int nops) {
    int nd = 0, D[6], S[2][6];
    for (int k = 5; k >= 0; k--) {            // inner to outer
        if (d[k] == 1) continue;
        if (nd > 0) {
            bool merge = true;
            for (int o = 0; o < nops; o++)
                if (st[o][k] != S[o][nd - 1] * D[nd - 1]) merge = false;
            if (merge) {
                D[nd - 1] *= d[k];
                continue;
            }
        }
        D[nd] = d[k];
        for (int o = 0; o < nops; o++) S[o][nd] = st[o][k];
        nd++;
    }
    if (nd == 0) {
        D[0] = 1;
        for (int o = 0; o < nops; o++) S[o][0] = 0;
        nd = 1;
    }
    for (int k = 0; k < 6; k++) {
        int j = 5 - k;
        d[k] = j < nd ? D[j] : 1;
        for (int o = 0; o < nops; o++) st[o][k] = j < nd ? S[o][j] : 0;
    }
    return nd;
}
// vec4 over the innermost dim: d5 % 4 == 0 and every operand's innermost stride
// 0 (one broadcast float) or 1 with 16-byte aligned rows
static bool vec4_ok(const int d[6], int *st[], const float *const *base, const long *off, int nops, const float *y) {
    if (d[5] % 4 || !al16(y)) return false;
    for (int o = 0; o < nops; o++) {
        if (st[o][5] == 0) continue;
        if (st[o][5] != 1 || !al16(base[o], off ? off[o] : 0)) return false;
        for (int k = 0; k < 5; k++)
            if (st[o][k] % 4) return false;
    }
    return true;
}
static void e2(const float *a, const float *b, float *y, int opc, float alpha, const int d_[6], const int sa_[6],
               const int sb_[6]) {
    long n = (long)d_[0] * d_[1] * d_[2] * d_[3] * d_[4] * d_[5];
    int d[6], sa[6], sb[6];
    memcpy(d, d_, sizeof d);
    memcpy(sa, sa_, sizeof sa);
    memcpy(sb, sb_, sizeof sb);
    int *st[2] = {sa, sb};
    int nd = coalesce(d, st, 2);
    const float *base[2] = {a, b};
    int vec = vec4_ok(d, st, base, nullptr, 2, y) ? 4 : 1;
    mb_pc pc;
    pc.p(a).p(b).p(y).i((int)n).i(opc).f(alpha);
    for (int k = 1; k < 6; k++) pc.i(d[k]);
    for (int k = 0; k < 6; k++) pc.i(sa[k]);
    for (int k = 0; k < 6; k++) pc.i(sb[k]);
    pc.i(nd);
    mb_vk_launch1d(op_pipe("ewise2", vec, opc), n / vec, pc);
}
static void e2flat(const float *a, const float *b, float *y, long n, int opc, float alpha = 1.0f) {
    int d[6] = {1, 1, 1, 1, 1, (int)n}, s[6] = {0, 0, 0, 0, 0, 1};
    e2(a, b, y, opc, alpha, d, s, s);
}

extern "C" {
void kernel_relu(const float *input, float *output, int n) { e1(input, output, n, 8); }
void kernel_relu6(const float *input, float *output, int n) { e1(input, output, n, 15, 0.0f, 6.0f); }
void kernel_sigmoid(const float *input, float *output, int n) { e1(input, output, n, 7); }
void kernel_silu(const float *input, float *output, int n) { e1(input, output, n, 14); }
void kernel_swish(const float *input, float *output, int n) { e1(input, output, n, 14); }
void kernel_elu(const float *input, float *output, int n, float alpha) { e1(input, output, n, 20, alpha); }
void kernel_tanh(const float *input, float *output, int n) { e1(input, output, n, 6); }
void kernel_log(const float *input, float *output, int n) { e1(input, output, n, 2); }
void kernel_gelu(const float *input, float *output, int n) { e1(input, output, n, 12); }
// MB's gelu_exact is the tanh (MinGPT) form
void kernel_gelu_exact(const float *input, float *output, int n) { e1(input, output, n, 13); }
void kernel_leaky_relu(const float *input, float *output, int n, float negative_slope) { e1(input, output, n, 35, negative_slope); }
void kernel_hardtanh(const float *input, float *output, int n, float min_val, float max_val) { e1(input, output, n, 15, min_val, max_val); }
void kernel_softplus(const float *input, float *output, int n) { e1(input, output, n, 32); }
void kernel_selu(const float *input, float *output, int n) { e1(input, output, n, 33); }
void kernel_hardsigmoid(const float *input, float *output, int n) { e1(input, output, n, 22); }
void kernel_softsign(const float *input, float *output, int n) { e1(input, output, n, 34); }
void kernel_ewise1(const float *input, float *output, int n, int opc, float p0, float p1) { e1(input, output, n, opc, p0, p1); }

void kernel_add(const float *a, const float *b, float *output, int n) { e2flat(a, b, output, n, 0); }
void kernel_mul(const float *a, const float *b, float *output, int n) { e2flat(a, b, output, n, 2); }
void kernel_mul_scalar(const float *input, float *output, int n, float s) { es(input, output, n, 3, s); }
void kernel_ewise_scalar(const float *input, float *output, int n, int opc, float c) { es(input, output, n, opc, c); }
void kernel_ewise2(const float *a, const float *b, float *output, int opc, float alpha,
                   int d0, int d1, int d2, int d3, int d4, int d5,
                   int sa0, int sa1, int sa2, int sa3, int sa4, int sa5,
                   int sb0, int sb1, int sb2, int sb3, int sb4, int sb5) {
    int d[6] = {d0, d1, d2, d3, d4, d5}, sa[6] = {sa0, sa1, sa2, sa3, sa4, sa5}, sb[6] = {sb0, sb1, sb2, sb3, sb4, sb5};
    e2(a, b, output, opc, alpha, d, sa, sb);
}
void kernel_batchnorm2d(const float *input, const float *scale, const float *bias, float *output, int N, int C, int H, int W) {
    PIPE("chan_affine");
    long n = (long)N * C * H * W;
    mb_pc pc;
    pc.p(input).p(scale).p(bias).p(output).i((int)n).i(C).i(H * W);
    mb_vk_launch1d(pipe, n, pc);
}

// ------------------------------------------------------------------ dense (tiled GEMM)
// Split-K when the output has few 64x64 tiles but K is long (a GEMV-like
// linear, M <= 64, over thousands of inputs): enough workgroups to fill the GPU.
// $MB_VK_SPLITK=0 disables it; MB_VK_SPLITK_WG (default 96) is the target count.
static int splitk_target() {
    static int t = -2;
    if (t == -2) {
        const char *e = getenv("MB_VK_SPLITK");
        const char *w = getenv("MB_VK_SPLITK_WG");
        t = (e && !atoi(e)) ? 0 : (w ? atoi(w) : 96);
    }
    return t;
}
// Tile shape (shaders/gemm_*.comp): $MB_VK_GEMM = 16 | 32 | 64 | 64n | big forces
// one; auto picks by M and N (measured on the Steam Deck's RDNA2, see
// docs/vulkan_amd.md).
struct gemm_variant { const char *name, *short_name; int bm, bn; };
static const gemm_variant GV[5] = {{"gemm_16", "16", 16, 256}, {"gemm_32", "32", 32, 128}, {"gemm_64", "64", 64, 64},
                                   {"gemm_64n", "64n", 64, 128}, {"gemm_big", "big", 128, 128}};
static int gemm_pick(int M, int N, int K) {
    static int forced = -2;
    if (forced == -2) {
        const char *e = getenv("MB_VK_GEMM");
        forced = -1;
        for (int i = 0; e && i < 5; i++)
            if (!strcmp(e, GV[i].short_name) || !strcmp(e, GV[i].name)) forced = i;
    }
    if (forced >= 0) return forced;
    if (M <= 16) return 0;
    if (M <= 32) return K >= 2048 ? 2 : 1;
    if (M >= 128 && N >= 128) return 4;
    return 2;
}
static int env_int(const char *name, int dflt) {
    const char *e = getenv(name);
    return e ? atoi(e) : dflt;
}

// Packed weights (mb_vk_set_packed, mb_vk_runtime.h): a linear of M <= 64 rows
// whose weight has a packed copy runs the skinny GEMM on it.
static const mb_packed *find_packed(const void *w) { return mb_vk_find_packed(w); }

// Skinny GEMM (shaders/skinny_body.glsl) for M <= 64 rows times a packed
// weight: one invocation per output column holding all M rows, the K range of
// a workgroup interleaved over MB_VK_SKINNY_KW (default 2) waves and split over
// workgroups until ~MB_VK_SKINNY_WG (default 64) waves run; MB_VK_SKINNY_KU
// (32, or 16; 16 when K % 32 != 0) k per step. Defaults measured on the Steam
// Deck (docs/vulkan_amd.md). $MB_VK_SKINNY=0 disables it (the tiled GEMM then
// takes these shapes).
static bool skinny(const float *A, const mb_packed &W, float *C, const float *bias, int M, int N, int K, int sAm,
                   float alpha) {
    int ku = env_int("MB_VK_SKINNY_KU", 32);
    if (ku != 16 || K % 32) ku = 16;
    static int kw = env_int("MB_VK_SKINNY_KW", 2);
    int consts[4] = {M, ku, kw, 64 * kw};
    int pipe = mb_vk_pipeline_spec(W.f16 ? "skinny_h" : "skinny", consts, 4);   // cached per (M, KU, KW)
    if (pipe < 0) return false;
    int gx = (N + 63) / 64;
    static int target = env_int("MB_VK_SKINNY_WG", 64);
    int waves = gx * kw;
    int nsplit = waves < target ? (target + waves - 1) / waves : 1;
    int maxs = K / (2 * kw * ku);             // >= 2 k-steps per wave per split
    if (nsplit > maxs) nsplit = maxs;
    if (nsplit > 64) nsplit = 64;
    if (nsplit < 1) nsplit = 1;
    int kchunk = ((K + nsplit - 1) / nsplit + ku - 1) / ku * ku;
    nsplit = (K + kchunk - 1) / kchunk;
    float *dst = C;
    if (nsplit > 1) {
        dst = (float *)mb_vk_scratch(0, (size_t)nsplit * M * N * sizeof(float));
        if (!dst) return true;
    }
    mb_pc pc;
    pc.p(A).p(W.p).p(dst).p(bias).i(M).i(N).i(K).i(sAm).i(N).i(kchunk).i(nsplit > 1 ? 1 : 0).f(alpha);
    mb_vk_dispatch(pipe, (uint32_t)gx, (uint32_t)nsplit, 1, pc);
    if (nsplit > 1) {
        static int red = -1;
        if (red < 0) red = mb_vk_pipeline("splitk_reduce");
        mb_pc pr;
        pr.p(dst).p(C).p(bias).i(M * N).i(M * N).i(N).i(nsplit).i(M * N).f(alpha);
        mb_vk_launch1d(red, (long)M * N, pr);
    }
    return true;
}

static void gemm(const float *A, const float *B, float *C, const float *bias, int batch, int M, int N, int K,
                 int sAb, int sAm, int sAk, int sBb, int sBk, int sBn, float alpha = 1.0f) {
    static int pipes[5] = {-1, -1, -1, -1, -1};
    if (M <= 0 || N <= 0 || batch <= 0) return;
    int v = gemm_pick(M, N, K);
    if (pipes[v] < 0) pipes[v] = mb_vk_pipeline(GV[v].name);
    int pipe = pipes[v];
    int vecA = (sAk == 1 && K % 4 == 0 && sAm % 4 == 0 && sAb % 4 == 0 && al16(A)) ? 1 : 0;
    int vecB = (sBk == 1 && K % 4 == 0 && sBn % 4 == 0 && sBb % 4 == 0 && al16(B)) ? 1
             : (sBn == 1 && N % 4 == 0 && sBk % 4 == 0 && sBb % 4 == 0 && al16(B)) ? 2 : 0;
    if (getenv("MB_VK_NOVEC")) vecA = vecB = 0;
    static int use_skinny = env_int("MB_VK_SKINNY", 1);
    if (use_skinny && batch == 1 && M <= 64 && vecA == 1 && sBk == 1 && sBn == K && K % 16 == 0) {
        const mb_packed *pw = find_packed(B);
        if (pw && pw->f16 != 2 && skinny(A, *pw, C, bias, M, N, K, sAm, alpha)) return;
    }
    uint32_t gx = (uint32_t)((N + GV[v].bn - 1) / GV[v].bn), gy = (uint32_t)((M + GV[v].bm - 1) / GV[v].bm);
    long tiles = (long)gx * gy * batch;
    int nsplit = 1, target = splitk_target();
    if (target > 0 && tiles < target && K >= 512) {
        nsplit = (int)((target + tiles - 1) / tiles);
        int maxs = K / 256;                       // keep >= 256 k per split
        if (nsplit > maxs) nsplit = maxs;
        if (nsplit > 64) nsplit = 64;
    }
    if (nsplit > 1) {
        int kchunk = (((K + nsplit - 1) / nsplit) + 31) / 32 * 32;   // whole K steps
        nsplit = (K + kchunk - 1) / kchunk;
        float *part = (float *)mb_vk_scratch(0, (size_t)batch * nsplit * M * N * sizeof(float));
        if (!part) return;
        mb_pc pc;
        pc.p(A).p(B).p(part).p(nullptr).i(M).i(N).i(K).i(sAb).i(sAm).i(sAk).i(sBb).i(sBk).i(sBn).i(M * N).i(N)
            .f(1.0f).i(nsplit).i(kchunk).i(vecA).i(vecB);
        mb_vk_dispatch(pipe, gx, gy, (uint32_t)(batch * nsplit), pc);
        static int red = -1;
        if (red < 0) red = mb_vk_pipeline("splitk_reduce");
        long n = (long)batch * M * N;
        mb_pc pr;
        pr.p(part).p(C).p(bias).i((int)n).i(M * N).i(N).i(nsplit).i(M * N).f(alpha);
        mb_vk_launch1d(red, n, pr);
        return;
    }
    mb_pc pc;
    pc.p(A).p(B).p(C).p(bias).i(M).i(N).i(K).i(sAb).i(sAm).i(sAk).i(sBb).i(sBk).i(sBn).i(M * N).i(N).f(alpha)
        .i(1).i(K).i(vecA).i(vecB);
    mb_vk_dispatch(pipe, gx, gy, (uint32_t)batch, pc);
}
void kernel_linear(const float *input, const float *weight, const float *bias, float *output, int M, int K, int N) {
    gemm(input, weight, output, bias, 1, M, N, K, 0, K, 1, 0, 1, K);
}
void kernel_matmul(const float *A, const float *B, float *C, int M, int K, int N) {
    gemm(A, B, C, nullptr, 1, M, N, K, 0, K, 1, 0, N, 1);
}
void kernel_matmul_ta(const float *A, const float *B, float *C, int M, int K, int N) {   // A is [K, M]
    gemm(A, B, C, nullptr, 1, M, N, K, 0, 1, M, 0, N, 1);
}
void kernel_matmul_tb(const float *A, const float *B, float *C, int M, int K, int N) {   // B is [N, K]
    gemm(A, B, C, nullptr, 1, M, N, K, 0, K, 1, 0, 1, K);
}
void kernel_matmul_tatb(const float *A, const float *B, float *C, int M, int K, int N) {
    gemm(A, B, C, nullptr, 1, M, N, K, 0, 1, M, 0, 1, K);
}
void kernel_bmm(const float *A, const float *B, float *C, int batch, int M, int K, int N) {
    gemm(A, B, C, nullptr, batch, M, N, K, M * K, K, 1, K * N, N, 1);
}

// ------------------------------------------------------------------ conv / pool
static void conv(const float *x, const float *w, const float *b, float *y, int N, int IC, int IH, int IW, int OC,
                 int OH, int OW, int KH, int KW, int SH, int SW, int PH, int PW, int DH, int DW, int G) {
    PIPE("conv2d");
    long n = (long)N * OC * OH * OW;
    mb_pc pc;
    pc.p(x).p(w).p(b).p(y).i((int)n).i(IC).i(IH).i(IW).i(OC).i(OH).i(OW).i(KH).i(KW).i(SH).i(SW).i(PH).i(PW).i(DH).i(DW).i(G);
    mb_vk_launch1d(pipe, n, pc);
}
// A patchify conv (stride == kernel, no padding / dilation / groups, N == 1:
// a ViT patch embedding) is a GEMM: gather the patches with one strided copy
// (im2col: [OH*OW, IC*KH*KW]) and multiply, out[oc, p] = sum_k W[oc, k] * P[p, k].
static void strided_copy6(const float *x, float *y, int off, const int d[6], const int st[6]);
void kernel_conv2d(const float *input, const float *weight, const float *bias, float *output, int N, int IC, int IH,
                   int IW, int OC, int KH, int KW, int SH, int SW, int PH, int PW, int DH, int DW) {
    int OH = (IH + 2 * PH - DH * (KH - 1) - 1) / SH + 1, OW = (IW + 2 * PW - DW * (KW - 1) - 1) / SW + 1;
    if (N == 1 && SH == KH && SW == KW && PH == 0 && PW == 0 && DH == 1 && DW == 1 && !getenv("MB_VK_DIRECT_CONV")) {
        long K = (long)IC * KH * KW, P = (long)OH * OW;
        float *cols = (float *)mb_vk_scratch(1, (size_t)P * K * sizeof(float));
        if (!cols) return;
        int d[6] = {OH, OW, IC, KH, KW, 1}, st[6] = {KH * IW, KW, IH * IW, IW, 1, 0};
        strided_copy6(input, cols, 0, d, st);
        // C[oc, p]: A = weight [OC, K] (k fastest), B[k, p] = cols[p, k]
        gemm(weight, cols, output, nullptr, 1, OC, (int)P, (int)K, 0, (int)K, 1, 0, 1, (int)K);
        if (bias) {   // + bias[oc] along rows: y[oc, p] += b[oc] as an ewise2 broadcast
            int dd[6] = {1, 1, 1, 1, OC, (int)P}, sy[6] = {0, 0, 0, 0, (int)P, 1}, sb[6] = {0, 0, 0, 0, 1, 0};
            e2(output, bias, output, 0, 1.0f, dd, sy, sb);
        }
        return;
    }
    conv(input, weight, bias, output, N, IC, IH, IW, OC, OH, OW, KH, KW, SH, SW, PH, PW, DH, DW, 1);
}
void kernel_conv2d_grouped(const float *input, const float *weight, const float *bias, float *output, int N, int IC,
                           int IH, int IW, int OC, int KH, int KW, int SH, int SW, int PH, int PW, int DH, int DW, int G) {
    int OH = (IH + 2 * PH - DH * (KH - 1) - 1) / SH + 1, OW = (IW + 2 * PW - DW * (KW - 1) - 1) / SW + 1;
    conv(input, weight, bias, output, N, IC, IH, IW, OC, OH, OW, KH, KW, SH, SW, PH, PW, DH, DW, G);
}
void kernel_conv2d_dw(const float *input, const float *weight, const float *bias, float *output, int N, int C, int IH,
                      int IW, int KH, int KW, int SH, int SW, int PH, int PW) {
    int OH = (IH + 2 * PH - KH) / SH + 1, OW = (IW + 2 * PW - KW) / SW + 1;
    conv(input, weight, bias, output, N, C, IH, IW, C, OH, OW, KH, KW, SH, SW, PH, PW, 1, 1, C);
}
static void pool(const float *x, float *y, int N, int C, int IH, int IW, int OH, int OW, int KH, int KW, int SH,
                 int SW, int PH, int PW, int DH, int DW, int mode) {
    PIPE("pool2d");
    long n = (long)N * C * OH * OW;
    mb_pc pc;
    pc.p(x).p(y).i((int)n).i(IH).i(IW).i(OH).i(OW).i(KH).i(KW).i(SH).i(SW).i(PH).i(PW).i(DH).i(DW).i(mode);
    mb_vk_launch1d(pipe, n, pc);
}
void kernel_maxpool2d(const float *input, float *output, int N, int C, int IH, int IW, int KH, int KW, int SH,
                      int SW, int PH, int PW, int DH, int DW) {
    int OH = (IH + 2 * PH - DH * (KH - 1) - 1) / SH + 1, OW = (IW + 2 * PW - DW * (KW - 1) - 1) / SW + 1;
    pool(input, output, N, C, IH, IW, OH, OW, KH, KW, SH, SW, PH, PW, DH, DW, 0);
}
void kernel_maxpool2d_ex(const float *input, float *output, int N, int C, int IH, int IW, int OH, int OW, int KH,
                         int KW, int SH, int SW, int PH, int PW, int DH, int DW) {
    pool(input, output, N, C, IH, IW, OH, OW, KH, KW, SH, SW, PH, PW, DH, DW, 1);
}
void kernel_avgpool2d(const float *input, float *output, int N, int C, int IH, int IW, int KH, int KW, int SH,
                      int SW, int PH, int PW) {
    int OH = (IH + 2 * PH - KH) / SH + 1, OW = (IW + 2 * PW - KW) / SW + 1;
    pool(input, output, N, C, IH, IW, OH, OW, KH, KW, SH, SW, PH, PW, 1, 1, 2);
}
void kernel_upsample_nearest(const float *input, float *output, int N, int C, int IH, int IW, int scale) {
    PIPE("upsample");
    long n = (long)N * C * IH * scale * IW * scale;
    mb_pc pc;
    pc.p(input).p(output).i((int)n).i(IH).i(IW).i(scale);
    mb_vk_launch1d(pipe, n, pc);
}

// ------------------------------------------------------------------ reductions / row ops
static void reduce(const float *x, float *y, int outer, int R, int inner, int opc) {
    PIPE("reduce");
    mb_pc pc;
    pc.p(x).p(y).i(outer).i(R).i(inner).i(opc);
    lines(pipe, (long)outer * inner, pc);
}
void kernel_reduce(const float *input, float *output, int outer, int R, int inner, int opc) { reduce(input, output, outer, R, inner, opc); }
void kernel_sum_dim(const float *input, float *output, int outer, int reduce_, int inner) { reduce(input, output, outer, reduce_, inner, 0); }
void kernel_mean_dim(const float *input, float *output, int outer, int reduce_, int inner) { reduce(input, output, outer, reduce_, inner, 1); }
void kernel_max_dim(const float *input, float *output, int outer, int reduce_, int inner) { reduce(input, output, outer, reduce_, inner, 2); }
void kernel_min_dim(const float *input, float *output, int outer, int reduce_, int inner) { reduce(input, output, outer, reduce_, inner, 3); }
void kernel_adaptive_avg_pool2d(const float *input, float *output, int N, int C, int IH, int IW) {
    reduce(input, output, N * C, IH * IW, 1, 1);
}
static void softmax(const float *x, float *y, int outer, int R, int inner, int log_out, float in_scale = 1.0f) {
    PIPE("softmax");
    mb_pc pc;
    pc.p(x).p(y).i(outer).i(R).i(inner).i(log_out).f(in_scale);
    lines(pipe, (long)outer * inner, pc);
}
void kernel_softmax(const float *input, float *output, int M, int K) { softmax(input, output, M, K, 1, 0); }
void kernel_log_softmax(const float *input, float *output, int M, int K) { softmax(input, output, M, K, 1, 1); }
void kernel_softmax_dim(const float *input, float *output, int outer, int R, int inner, int log_out) {
    softmax(input, output, outer, R, inner, log_out);
}
void kernel_layer_norm(const float *input, const float *gamma, const float *beta, float *output, int M, int K, float eps) {
    PIPE("layer_norm");
    mb_pc pc;
    pc.p(input).p(gamma).p(beta).p(output).i(M).i(K).f(eps);
    lines(pipe, M, pc);
}
static void col_norm(const float *x, float *y, int outer, int R, int inner, int kind, float eps) {
    PIPE("col_norm");
    mb_pc pc;
    pc.p(x).p(y).i(outer).i(R).i(inner).i(kind).f(eps);
    lines(pipe, (long)outer * inner, pc);
}
void kernel_rms_norm(const float *input, float *output, int outer, int reduce_, int inner, float eps) {
    col_norm(input, output, outer, reduce_, inner, 0, eps);
}
void kernel_mean_abs_norm(const float *input, float *output, int outer, int reduce_, int inner) {
    col_norm(input, output, outer, reduce_, inner, 1, 0.0f);
}
void kernel_l2_norm(const float *input, float *output, int outer, int reduce_, int inner) {
    col_norm(input, output, outer, reduce_, inner, 2, 0.0f);
}
void kernel_group_norm(const float *input, const float *gamma, const float *beta, float *output, int N, int C, int G,
                       int HW, float eps) {
    PIPE("group_norm");
    mb_pc pc;
    pc.p(input).p(gamma).p(beta).p(output).i(C).i(G).i(HW).f(eps);
    lines(pipe, (long)N * G, pc);
}
static void scan(const float *x, float *y, int outer, int axis, int inner, int prod) {
    PIPE("scan");
    long rows = (long)outer * inner;
    mb_pc pc;
    pc.p(x).p(y).i((int)rows).i(axis).i(inner).i(prod);
    mb_vk_launch1d(pipe, rows, pc);
}
void kernel_cumsum(const float *input, float *output, int outer, int axis, int inner) { scan(input, output, outer, axis, inner, 0); }
void kernel_cumprod(const float *input, float *output, int outer, int axis, int inner) { scan(input, output, outer, axis, inner, 1); }
static void index_sel(const float *x, const float *idx, float *y, int outer, int A, int J, int inner, int mode) {
    PIPE("index_select");
    long n = (long)outer * J * inner;
    mb_pc pc;
    pc.p(x).p(idx).p(y).i((int)n).i(A).i(J).i(inner).i(mode);
    mb_vk_launch1d(pipe, n, pc);
}
void kernel_index_select(const float *input, const float *index, float *output, int outer, int A, int J, int inner) {
    index_sel(input, index, output, outer, A, J, inner, 0);
}
void kernel_gather_dim(const float *input, const float *index, float *output, int outer, int A, int J, int inner) {
    index_sel(input, index, output, outer, A, J, inner, 1);
}
void kernel_sdpa(const float *Q, const float *K, const float *V, float *output, int BH, int S, int D) {
    PIPE("sdpa");
    mb_pc pc;
    pc.p(Q).p(K).p(V).p(output).i(BH).i(S).i(D);
    lines(pipe, (long)BH * S, pc);
}

// ------------------------------------------------------------------ layout
void kernel_strided_copy(const float *input, float *output, int off, int d0, int d1, int d2, int d3, int d4, int d5,
                         int s0, int s1, int s2, int s3, int s4, int s5) {
    static int pipes[2] = {-1, -1};
    long n = (long)d0 * d1 * d2 * d3 * d4 * d5;
    int d[6] = {d0, d1, d2, d3, d4, d5}, sx[6] = {s0, s1, s2, s3, s4, s5};
    int *st[1] = {sx};
    int nd = coalesce(d, st, 1);
    const float *base[1] = {input};
    long offs[1] = {off};
    int vec = vec4_ok(d, st, base, offs, 1, output) ? 4 : 1;
    mb_pc pc;
    pc.p(input).p(output).i((int)n).i(off);
    for (int k = 1; k < 6; k++) pc.i(d[k]);
    for (int k = 0; k < 6; k++) pc.i(sx[k]);
    pc.i(nd);
    mb_vk_launch1d(vec_pipe("strided_copy", vec, pipes), n / vec, pc);
}
static void strided_copy6(const float *x, float *y, int off, const int d[6], const int st[6]) {
    kernel_strided_copy(x, y, off, d[0], d[1], d[2], d[3], d[4], d[5], st[0], st[1], st[2], st[3], st[4], st[5]);
}
void kernel_gather(const float *input, const int *index, float *output, int n) {
    PIPE("gather");
    mb_pc pc;
    pc.p(input).p(index).p(output).i(n);
    mb_vk_launch1d(pipe, n, pc);
}
static void cat_slab(const float *x, float *y, int outer, long slab, long ytot, long ybase) {
    static int pipes[2] = {-1, -1};
    long n = (long)outer * slab;
    int vec = (slab % 4 == 0 && ytot % 4 == 0 && ybase % 4 == 0 && al16(x) && al16(y)) ? 4 : 1;
    mb_pc pc;
    pc.p(x).p(y).i((int)n).i((int)slab).i((int)ytot).i((int)ybase);
    mb_vk_launch1d(vec_pipe("cat_slab", vec, pipes), n / vec, pc);
}
void kernel_catn(const float *in0, const float *in1, const float *in2, const float *in3, const float *in4,
                 const float *in5, const float *in6, const float *in7, float *output, int n_in, int outer, int inner,
                 int c0, int c1, int c2, int c3, int c4, int c5, int c6, int c7) {
    const float *ins[8] = {in0, in1, in2, in3, in4, in5, in6, in7};
    int cs[8] = {c0, c1, c2, c3, c4, c5, c6, c7};
    long ctot = 0;
    for (int k = 0; k < n_in; k++) ctot += cs[k];
    long cbase = 0;
    for (int k = 0; k < n_in; k++) {
        cat_slab(ins[k], output, outer, (long)cs[k] * inner, ctot * inner, cbase * inner);
        cbase += cs[k];
    }
}
void kernel_cat2_c1(const float *in0, int c0, const float *in1, int c1, float *out, int N, int H, int W) {
    long HW = (long)H * W, C = c0 + c1;
    cat_slab(in0, out, N, c0 * HW, C * HW, 0);
    cat_slab(in1, out, N, c1 * HW, C * HW, c0 * HW);
}
void kernel_cat3_c1(const float *in0, int c0, const float *in1, int c1, const float *in2, int c2, float *out, int N,
                    int H, int W) {
    long HW = (long)H * W, C = c0 + c1 + c2;
    cat_slab(in0, out, N, c0 * HW, C * HW, 0);
    cat_slab(in1, out, N, c1 * HW, C * HW, c0 * HW);
    cat_slab(in2, out, N, c2 * HW, C * HW, (c0 + c1) * HW);
}
void kernel_cat4_c1(const float *in0, int c0, const float *in1, int c1, const float *in2, int c2, const float *in3,
                    int c3, float *out, int N, int H, int W) {
    long HW = (long)H * W, C = c0 + c1 + c2 + c3;
    cat_slab(in0, out, N, c0 * HW, C * HW, 0);
    cat_slab(in1, out, N, c1 * HW, C * HW, c0 * HW);
    cat_slab(in2, out, N, c2 * HW, C * HW, (c0 + c1) * HW);
    cat_slab(in3, out, N, c3 * HW, C * HW, (c0 + c1 + c2) * HW);
}
static void tri(const float *x, const float *b, float *y, long n, int M, int N, int d, int mode) {
    PIPE("tri");
    mb_pc pc;
    pc.p(x).p(b).p(y).i((int)n).i(M).i(N).i(d).i(mode);
    mb_vk_launch1d(pipe, n, pc);
}
void kernel_triu(const float *input, float *output, int M, int N, int diagonal) { tri(input, nullptr, output, (long)M * N, M, N, diagonal, 1); }
void kernel_tril(const float *input, float *output, int M, int N, int diagonal) { tri(input, nullptr, output, (long)M * N, M, N, diagonal, 0); }
void kernel_diag_matmul(const float *a, const float *b, float *output, int N, int M) { tri(a, b, output, (long)N * M, N, M, 0, 2); }
void kernel_flip(const float *input, float *output, int outer, int axis, int inner) {
    tri(input, nullptr, output, (long)outer * axis * inner, axis, inner, 0, 3);
}
}  // extern "C"

// ================================================================== int8 (`_s8`) ops
// The SmolVLA int8 kernel family (layernorm_s8 / rmsnorm_s8 / gelu_s8 / linear_s8 /
// matmul_s8 / softmax_s8 / mul_s8 / sin_s8 / cos_s8), bit-exact with the KernelSpec
// reference C: integer accumulation is exact; float steps are done in the
// reference's precision and order (fp32 divisions through fp64, exact roundf);
// an op whose output is a function of one int8 input is a 256-entry table the
// launcher fills with the reference's own expression and libm.
#include <math.h>
static int8_t clamp8(int32_t v, int lo, int hi) { return (int8_t)(v < lo ? lo : (v > hi ? hi : v)); }
static void lut_s8(const int8_t *x, int8_t *y, int n, const int8_t *table256) {
    PIPE("lut_s8");
    int8_t *lut = (int8_t *)mb_vk_alloc(256);         // lives in the caller's arena (and in a captured graph)
    if (!lut) return;
    memcpy(lut, table256, 256);
    mb_pc pc;
    pc.p(x).p(lut).p(y).i(n);
    mb_vk_launch1d(pipe, n, pc);
}
extern "C" {
void kernel_gelu_s8(const int8_t *input, int8_t *output, int n, float scale_in, float scale_out, int activation_min,
                    int activation_max) {
    const float kInvSqrt2 = 0.70710678118f;
    int8_t t[256];
    for (int i = -128; i < 128; i++) {
        float f = (float)i * scale_in;
        float y = 0.5f * f * (1.0f + erff(f * kInvSqrt2));
        t[i + 128] = clamp8((int32_t)roundf(y / scale_out), activation_min, activation_max);
    }
    lut_s8(input, output, n, t);
}
void kernel_sin_s8(const int8_t *input, int8_t *output, int n, float scale_in, float scale_out, int activation_min,
                   int activation_max) {
    int8_t t[256];
    for (int i = -128; i < 128; i++)
        t[i + 128] = clamp8((int32_t)round(sin((double)i * (double)scale_in) / (double)scale_out), activation_min, activation_max);
    lut_s8(input, output, n, t);
}
void kernel_cos_s8(const int8_t *input, int8_t *output, int n, float scale_in, float scale_out, int activation_min,
                   int activation_max) {
    int8_t t[256];
    for (int i = -128; i < 128; i++)
        t[i + 128] = clamp8((int32_t)round(cos((double)i * (double)scale_in) / (double)scale_out), activation_min, activation_max);
    lut_s8(input, output, n, t);
}
// The dot4 path (shaders/dot_s8_body.glsl: packed int8 dot products, still
// bit-exact) when the device has VK_KHR_shader_integer_dot_product, K % 4 == 0
// and the operands are 4-byte aligned; $MB_VK_INT_DOT=0 forces the scalar kernels.
static bool use_dot4(const void *a, const void *b, int K) {
    static int on = env_int("MB_VK_INT_DOT", 1) && mb_vk_has_int_dot();
    return on && K % 4 == 0 && (mb_vk_devaddr(a) & 3) == 0 && (mb_vk_devaddr(b) & 3) == 0;
}
static void dot4_dispatch(int pipe, int M, int N, const mb_pc &pc) {   // 32 x 64 output tiles
    mb_vk_dispatch(pipe, (uint32_t)((N + 63) / 64), (uint32_t)((M + 31) / 32), 1, pc);
}

void kernel_linear_s8(const int8_t *input, const int8_t *weight, const int32_t *bias, int8_t *output, int M, int K,
                      int N, int input_offset, int filter_offset, int output_offset, int output_multiplier,
                      int output_shift, int activation_min, int activation_max) {
    if (use_dot4(input, weight, K)) {
        static int dp = -1;
        if (dp < 0) dp = mb_vk_pipeline("linear_s8_dot");
        mb_pc pc;
        pc.p(input).p(weight).p(bias).p(output).i(M).i(K).i(N).i(input_offset).i(filter_offset).i(output_offset)
            .i(output_multiplier).i(output_shift).i(activation_min).i(activation_max);
        dot4_dispatch(dp, M, N, pc);
        return;
    }
    PIPE("linear_s8");
    mb_pc pc;
    pc.p(input).p(weight).p(bias).p(output).i(M).i(K).i(N).i(input_offset).i(filter_offset).i(output_offset)
        .i(output_multiplier).i(output_shift).i(activation_min).i(activation_max);
    mb_vk_launch1d(pipe, (long)M * N, pc);
}
void kernel_matmul_s8(const int8_t *a, const int8_t *b, int8_t *output, int M, int K, int N, float scale_a,
                      float scale_b, float scale_out, int transpose_b, float scale_div, int activation_min,
                      int activation_max) {
    PIPE("matmul_s8");
    float total = (scale_a * scale_b) / (scale_out * scale_div);   // as the reference computes it
    if (transpose_b && use_dot4(a, b, K)) {
        static int dp = -1;
        if (dp < 0) dp = mb_vk_pipeline("matmul_s8_dot");
        mb_pc pd;
        pd.p(a).p(b).p(output).i(M).i(K).i(N).f(total).i(activation_min).i(activation_max);
        dot4_dispatch(dp, M, N, pd);
        return;
    }
    mb_pc pc;
    pc.p(a).p(b).p(output).i(M).i(K).i(N).i(transpose_b).f(total).i(activation_min).i(activation_max);
    mb_vk_launch1d(pipe, (long)M * N, pc);
}
void kernel_softmax_s8(const int8_t *input, int8_t *output, int M, int K, float scale_in, float scale_out) {
    PIPE("softmax_s8");
    float *lut = (float *)mb_vk_alloc(256 * sizeof(float));
    if (!lut) return;
    for (int d = 0; d < 256; d++) lut[d] = expf(((float)(-d)) * scale_in);   // expf((row[k] - shift) * scale_in)
    mb_pc pc;
    pc.p(input).p(output).p(lut).i(M).i(K).f(scale_out);
    mb_vk_launch1d(pipe, M, pc);
}
static void norm_s8(const int8_t *x, const float *g, const float *b, int8_t *y, int M, int K, float si, float so,
                    float eps, int amin, int amax, int rms) {
    PIPE("norm_s8");
    mb_pc pc;
    pc.p(x).p(g).p(b).p(y).i(M).i(K).f(si).f(so).f(eps).i(amin).i(amax).i(rms);
    mb_vk_launch1d(pipe, M, pc);
}
void kernel_layernorm_s8(const int8_t *input, const float *gamma, const float *beta, int8_t *output, int M, int K,
                         float scale_in, float scale_out, float eps, int activation_min, int activation_max) {
    norm_s8(input, gamma, beta, output, M, K, scale_in, scale_out, eps, activation_min, activation_max, 0);
}
void kernel_rmsnorm_s8(const int8_t *input, const float *gamma, int8_t *output, int M, int K, float scale_in,
                       float scale_out, float eps, int activation_min, int activation_max) {
    norm_s8(input, gamma, nullptr, output, M, K, scale_in, scale_out, eps, activation_min, activation_max, 1);
}
// fp32 fusions of the int8_mixed pass: RoPE (two slices, four multiplies, sub, add, cat -> one op)
void kernel_rope_half(const float *input, const float *cos_t, const float *sin_t, float *output, int R, int H,
                      int half) {
    PIPE("rope_half");
    mb_pc pc;
    pc.p(input).p(cos_t).p(sin_t).p(output).i(R).i(H).i(half);
    mb_vk_launch1d(pipe, (long)R * H * half, pc);
}
// int8_mixed IRs (pipeline/int8_mixed.py): dynamic per-row quantization and the int8 linear with an
// fp32 (dequantized) output
void kernel_quantize_rows_s8(const float *input, const float *col_inv, int8_t *output, float *row_scale, int M,
                             int K) {
    mb_pc pc;
    pc.p(input).p(col_inv).p(output).p(row_scale).i(M).i(K);
    if (K % 4 == 0 && al16(input) && al16(col_inv) && (mb_vk_devaddr(output) & 3) == 0) {
        static int v4 = -1;
        if (v4 < 0) v4 = mb_vk_pipeline("quantize_rows_s8_v4");
        lines(v4, M, pc);
        return;
    }
    PIPE("quantize_rows_s8");
    lines(pipe, M, pc);
}
// linear_dq_s8 of M <= 64 rows on a packed int8 weight (mb_vk_set_packed(w, p, 2),
// walk_vulkan.pack_weight_s8): shaders/skinny_s8.comp, K split over workgroups until ~MB_VK_SKINNY_WG
// (default 64) waves run, the int32 partials summed by linear_dq_s8_epi.comp.
static bool skinny_s8(const int8_t *x, const float *rs, const mb_packed &W, const float *bias, const float *ws,
                      float *y, int M, int K, int N) {
    if ((mb_vk_devaddr(x) & 15) || K % 16) return false;
    static int nc = env_int("MB_VK_SKINNY_S8_NC", 1);
    int consts[2] = {M, nc};
    int pipe = mb_vk_pipeline_spec("skinny_s8", consts, 2);
    if (pipe < 0) return false;
    int gx = ((N + 63) / 64 + nc - 1) / nc;
    static int target = env_int("MB_VK_SKINNY_WG", 64);
    int nsplit = gx < target ? (target + gx - 1) / gx : 1;
    int maxs = K / 64;                          // >= 4 16-k steps per split
    if (nsplit > maxs) nsplit = maxs;
    if (nsplit > 64) nsplit = 64;
    if (nsplit < 1) nsplit = 1;
    int kchunk = ((K + nsplit - 1) / nsplit + 15) / 16 * 16;
    nsplit = (K + kchunk - 1) / kchunk;
    int32_t *part = nullptr;
    if (nsplit > 1) {
        part = (int32_t *)mb_vk_scratch(0, (size_t)nsplit * M * N * sizeof(int32_t));
        if (!part) return true;
    }
    mb_pc pc;
    pc.p(x).p(rs).p(W.p).p(bias).p(ws).p(y).p(part).i(M).i(K).i(N).i(kchunk).i(nsplit > 1 ? 1 : 0);
    mb_vk_dispatch(pipe, (uint32_t)gx, (uint32_t)nsplit, 1, pc);
    if (nsplit > 1) {
        static int epi = -1;
        if (epi < 0) epi = mb_vk_pipeline("linear_dq_s8_epi");
        mb_pc pe;
        pe.p(part).p(rs).p(bias).p(ws).p(y).i(M).i(N).i(nsplit);
        mb_vk_launch1d(epi, (long)M * N, pe);
    }
    return true;
}
void kernel_linear_dq_s8(const int8_t *input, const float *row_scale, const int8_t *weight, const float *bias,
                         const float *w_scale, float *output, int M, int K, int N) {
    static int use_skinny = env_int("MB_VK_SKINNY", 1);
    if (use_skinny && M <= 64) {
        const mb_packed *pw = find_packed(weight);
        if (pw && pw->f16 == 2 && skinny_s8(input, row_scale, *pw, bias, w_scale, output, M, K, N)) return;
    }
    mb_pc pc;
    pc.p(input).p(row_scale).p(weight).p(bias).p(w_scale).p(output).i(M).i(K).i(N);
    if (use_dot4(input, weight, K)) {
        // the tile: 128 x 64 for >= 128 rows (the vision linears: 1.2-1.4x the 32 x 64 tile's 0.84-0.91
        // Tops on the Deck), 64 x 64 for >= 48, else 32 x 64. $MB_VK_DQ_TILE = 24 | 44 | 84 forces one.
        // (Tried: 64 x 128, and 128-bit LDS reads with register prefetch: slower.)
        static int forced = getenv("MB_VK_DQ_TILE") ? atoi(getenv("MB_VK_DQ_TILE")) : 0;
        static int pipes[3] = {-1, -1, -1};
        static const char *names[3] = {"linear_dq_s8_dot", "linear_dq_s8_dot44_16", "linear_dq_s8_dot84_16"};
        int t = forced == 84 ? 2 : forced == 44 ? 1 : forced == 24 ? 0 : (M >= 128 ? 2 : M >= 48 ? 1 : 0);
        if (pipes[t] < 0) pipes[t] = mb_vk_pipeline(names[t]);
        int dp = pipes[t], rm = t == 2 ? 8 : t == 1 ? 4 : 2, rn = 4;
        mb_vk_dispatch(dp, (uint32_t)((N + 16 * rn - 1) / (16 * rn)), (uint32_t)((M + 16 * rm - 1) / (16 * rm)), 1, pc);
        return;
    }
    PIPE("linear_dq_s8");
    mb_vk_launch1d(pipe, (long)M * N, pc);
}
void kernel_mul_s8(const int8_t *a, const int8_t *b, int8_t *output, int n, float scale_a, float scale_b,
                   float scale_out, int activation_min, int activation_max) {
    PIPE("mul_s8");
    mb_pc pc;
    pc.p(a).p(b).p(output).i(n).f(scale_a).f(scale_b).f(scale_out).i(activation_min).i(activation_max);
    mb_vk_launch1d(pipe, n, pc);
}
}  // extern "C"
