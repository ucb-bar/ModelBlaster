// Hand-written OpenCL seeds for ModelBlaster's per-op C ABI (multi-slot problems).
//
// Every launcher below has EXACTLY the signature of the matching
// `KernelSpec.signature` (pipeline/reference_kernels.py); pointers are
// coarse-grained SVM pointers. Each `static mb_program cl_* = MB_CL(...)` is an
// OpenCL C program (built on first use with the prelude of
// mb_opencl_runtime.h: MB_THREADS, MB_TID, mb_ksum/mb_kadd, mb_idx6,
// mb_atomic_addf). kb_export seeds a slot with the launcher plus everything it
// references (pipeline/opencl_seeds.py); elementwise, axis-reduction and loss
// ops are generated from templates there instead.
//
// Like the CUDA hand kernels these are correctness seeds (mostly one work-item
// per output), not tuned code. Differences from CUDA: no fp64 (Adreno has
// none) -- dot products and long sums use compensated fp32 (mb_kadd); row ops
// (softmax, layer_norm) use one work-group per row so M=1 rows of 5e5 elements
// are not a single serial thread.
#include "mb_opencl_runtime.h"

// ---------------------------------------------------------------- batchnorm (folded)
static mb_program cl_batchnorm = MB_CL(R"CL(
__kernel void k_bn(__global const float *x, __global const float *s, __global const float *b,
                   __global float *y, int C, int HW, long tot) {
    long i = MB_TID;
    if (i < tot) { int c = (i / HW) % C; y[i] = x[i] * s[c] + b[c]; }
}
)CL");
void kernel_batchnorm2d(const float *input, const float *scale, const float *bias, float *output,
                        int N, int C, int H, int W) {
    long tot = (long)N * C * H * W;
    mb_launch(MB_KERNEL(cl_batchnorm, k_bn), mb_grid(tot), input, scale, bias, output, C, H * W, tot);
}

// ---------------------------------------------------------------- conv2d (direct)
// grouped + dilated direct conv; kernel_conv2d is the G=1 case
static mb_program cl_conv2d = MB_CL(R"CL(
__kernel void k_convg(__global const float *x, __global const float *w, __global const float *b, __global float *y,
                      int N, int IC, int IH, int IW, int OC, int OH, int OW, int KH, int KW,
                      int SH, int SW, int PH, int PW, int DH, int DW, int G) {
    long i = MB_TID; long tot = (long)N * OC * OH * OW;
    if (i >= tot) return;
    int ow = i % OW; int oh = (i / OW) % OH; int oc = (i / ((long)OW * OH)) % OC; int n = i / ((long)OW * OH * OC);
    int ICg = IC / G, ic0 = (oc / (OC / G)) * ICg;
    mb_ksum acc = mb_ksum0();
    for (int ic = 0; ic < ICg; ic++)
        for (int kh = 0; kh < KH; kh++) {
            int ih = oh * SH - PH + kh * DH; if (ih < 0 || ih >= IH) continue;
            for (int kw = 0; kw < KW; kw++) {
                int iw = ow * SW - PW + kw * DW; if (iw < 0 || iw >= IW) continue;
                mb_kadd(&acc, x[(((long)n * IC + ic0 + ic) * IH + ih) * IW + iw] * w[(((long)oc * ICg + ic) * KH + kh) * KW + kw]);
            }
        }
    y[i] = acc.s + (b ? b[oc] : 0.0f);
}
)CL");
void kernel_conv2d(const float *input, const float *weight, const float *bias, float *output,
                   int N, int IC, int IH, int IW, int OC, int KH, int KW,
                   int SH, int SW, int PH, int PW, int DH, int DW) {
    int OH = (IH + 2 * PH - DH * (KH - 1) - 1) / SH + 1, OW = (IW + 2 * PW - DW * (KW - 1) - 1) / SW + 1;
    long tot = (long)N * OC * OH * OW;
    mb_launch(MB_KERNEL(cl_conv2d, k_convg), mb_grid(tot), input, weight, bias, output, N, IC, IH, IW, OC, OH, OW,
              KH, KW, SH, SW, PH, PW, DH, DW, 1);
}
void kernel_conv2d_grouped(const float *input, const float *weight, const float *bias, float *output,
                           int N, int IC, int IH, int IW, int OC, int KH, int KW,
                           int SH, int SW, int PH, int PW, int DH, int DW, int G) {
    int OH = (IH + 2 * PH - DH * (KH - 1) - 1) / SH + 1, OW = (IW + 2 * PW - DW * (KW - 1) - 1) / SW + 1;
    long tot = (long)N * OC * OH * OW;
    mb_launch(MB_KERNEL(cl_conv2d, k_convg), mb_grid(tot), input, weight, bias, output, N, IC, IH, IW, OC, OH, OW,
              KH, KW, SH, SW, PH, PW, DH, DW, G);
}

static mb_program cl_conv2d_dw = MB_CL(R"CL(
__kernel void k_conv2d_dw(__global const float *x, __global const float *w, __global const float *b, __global float *y,
                          int N, int C, int IH, int IW, int OH, int OW, int KH, int KW, int SH, int SW, int PH, int PW) {
    long i = MB_TID; long tot = (long)N * C * OH * OW;
    if (i >= tot) return;
    int ow = i % OW; int oh = (i / OW) % OH; int c = (i / ((long)OW * OH)) % C; int n = i / ((long)OW * OH * C);
    mb_ksum acc = mb_ksum0();
    for (int kh = 0; kh < KH; kh++) {
        int ih = oh * SH - PH + kh; if (ih < 0 || ih >= IH) continue;
        for (int kw = 0; kw < KW; kw++) {
            int iw = ow * SW - PW + kw; if (iw < 0 || iw >= IW) continue;
            mb_kadd(&acc, x[(((long)n * C + c) * IH + ih) * IW + iw] * w[((long)c * KH + kh) * KW + kw]);
        }
    }
    y[i] = acc.s + (b ? b[c] : 0.0f);
}
)CL");
void kernel_conv2d_dw(const float *input, const float *weight, const float *bias, float *output,
                      int N, int C, int IH, int IW, int KH, int KW, int SH, int SW, int PH, int PW) {
    int OH = (IH + 2 * PH - KH) / SH + 1, OW = (IW + 2 * PW - KW) / SW + 1;
    long tot = (long)N * C * OH * OW;
    mb_launch(MB_KERNEL(cl_conv2d_dw, k_conv2d_dw), mb_grid(tot), input, weight, bias, output, N, C, IH, IW, OH, OW,
              KH, KW, SH, SW, PH, PW);
}

// weight is torch's [IC, OC/G, KH, KW]
static mb_program cl_conv_t2d = MB_CL(R"CL(
__kernel void k_conv_t(__global const float *x, __global const float *w, __global const float *b, __global float *y,
                       int N, int IC, int IH, int IW, int OC, int OH, int OW, int KH, int KW,
                       int SH, int SW, int PH, int PW, int DH, int DW, int G) {
    long i = MB_TID; long tot = (long)N * OC * OH * OW;
    if (i >= tot) return;
    int ow = i % OW; int oh = (i / OW) % OH; int oc = (i / ((long)OW * OH)) % OC; int n = i / ((long)OW * OH * OC);
    int ICpG = IC / G, OCpG = OC / G, g = oc / OCpG, ocg = oc % OCpG;
    mb_ksum acc = mb_ksum0();
    for (int kh = 0; kh < KH; kh++) {
        int ihs = oh + PH - kh * DH; if (ihs < 0 || (ihs % SH) != 0) continue;
        int ih = ihs / SH; if (ih >= IH) continue;
        for (int kw = 0; kw < KW; kw++) {
            int iws = ow + PW - kw * DW; if (iws < 0 || (iws % SW) != 0) continue;
            int iw = iws / SW; if (iw >= IW) continue;
            for (int icg = 0; icg < ICpG; icg++) {
                int ic = g * ICpG + icg;
                mb_kadd(&acc, x[(((long)n * IC + ic) * IH + ih) * IW + iw] * w[(((long)ic * OCpG + ocg) * KH + kh) * KW + kw]);
            }
        }
    }
    y[i] = acc.s + (b ? b[oc] : 0.0f);
}
)CL");
void kernel_conv_transpose2d(const float *input, const float *weight, const float *bias, float *output,
                             int N, int IC, int IH, int IW, int OC, int OH, int OW,
                             int KH, int KW, int SH, int SW, int PH, int PW, int DH, int DW, int G) {
    long tot = (long)N * OC * OH * OW;
    mb_launch(MB_KERNEL(cl_conv_t2d, k_conv_t), mb_grid(tot), input, weight, bias, output, N, IC, IH, IW, OC, OH, OW,
              KH, KW, SH, SW, PH, PW, DH, DW, G);
}

static mb_program cl_conv3d = MB_CL(R"CL(
__kernel void k_conv3d(__global const float *x, __global const float *w, __global const float *b, __global float *y,
                       int N, int IC, int ID, int IH, int IW, int OC, int OD, int OH, int OW,
                       int KD, int KH, int KW, int SD, int SH, int SW, int PD, int PH, int PW,
                       int DD, int DH, int DW, int G) {
    long i = MB_TID; if (i >= (long)N * OC * OD * OH * OW) return;
    int ow = i % OW; int oh = (i / OW) % OH; int od = (i / ((long)OW * OH)) % OD;
    int c = (i / ((long)OW * OH * OD)) % OC; int n = i / ((long)OW * OH * OD * OC);
    int ICpG = IC / G, g = c / (OC / G);
    mb_ksum acc = mb_ksum0();
    for (int icg = 0; icg < ICpG; icg++) {
        int ic = g * ICpG + icg;
        __global const float *xp = x + ((long)n * IC + ic) * ID * IH * IW;
        __global const float *wp = w + ((long)c * ICpG + icg) * KD * KH * KW;
        for (int kd = 0; kd < KD; kd++) {
            int id = od * SD - PD + kd * DD; if (id < 0 || id >= ID) continue;
            for (int kh = 0; kh < KH; kh++) {
                int ih = oh * SH - PH + kh * DH; if (ih < 0 || ih >= IH) continue;
                for (int kw = 0; kw < KW; kw++) {
                    int iw = ow * SW - PW + kw * DW; if (iw < 0 || iw >= IW) continue;
                    mb_kadd(&acc, xp[((long)id * IH + ih) * IW + iw] * wp[(kd * KH + kh) * KW + kw]);
                }
            }
        }
    }
    y[i] = acc.s + (b ? b[c] : 0.0f);
}
)CL");
void kernel_conv3d(const float *input, const float *weight, const float *bias, float *output,
                   int N, int IC, int ID, int IH, int IW, int OC, int OD, int OH, int OW,
                   int KD, int KH, int KW, int SD, int SH, int SW, int PD, int PH, int PW,
                   int DD, int DH, int DW, int G) {
    long tot = (long)N * OC * OD * OH * OW;
    mb_launch(MB_KERNEL(cl_conv3d, k_conv3d), mb_grid(tot), input, weight, bias, output, N, IC, ID, IH, IW,
              OC, OD, OH, OW, KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW, G);
}

// weight is torch's [IC, OC/G, KD, KH, KW]
static mb_program cl_conv_t3d = MB_CL(R"CL(
__kernel void k_conv_t3d(__global const float *x, __global const float *w, __global const float *b, __global float *y,
                         int N, int IC, int ID, int IH, int IW, int OC, int OD, int OH, int OW,
                         int KD, int KH, int KW, int SD, int SH, int SW, int PD, int PH, int PW,
                         int DD, int DH, int DW, int G) {
    long i = MB_TID; if (i >= (long)N * OC * OD * OH * OW) return;
    int ow = i % OW; int oh = (i / OW) % OH; int od = (i / ((long)OW * OH)) % OD;
    int c = (i / ((long)OW * OH * OD)) % OC; int n = i / ((long)OW * OH * OD * OC);
    int ICpG = IC / G, OCpG = OC / G, g = c / OCpG, ocg = c % OCpG;
    mb_ksum acc = mb_ksum0();
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
                    mb_kadd(&acc, x[((((long)n * IC + ic) * ID + id) * IH + ih) * IW + iw] *
                                  w[((((long)ic * OCpG + ocg) * KD + kd) * KH + kh) * KW + kw]);
                }
            }
        }
    }
    y[i] = acc.s + (b ? b[c] : 0.0f);
}
)CL");
void kernel_conv_transpose3d(const float *input, const float *weight, const float *bias, float *output,
                             int N, int IC, int ID, int IH, int IW, int OC, int OD, int OH, int OW,
                             int KD, int KH, int KW, int SD, int SH, int SW, int PD, int PH, int PW,
                             int DD, int DH, int DW, int G) {
    long tot = (long)N * OC * OD * OH * OW;
    mb_launch(MB_KERNEL(cl_conv_t3d, k_conv_t3d), mb_grid(tot), input, weight, bias, output, N, IC, ID, IH, IW,
              OC, OD, OH, OW, KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW, G);
}

static mb_program cl_convnd = MB_CL(R"CL(
__kernel void k_convnd(__global const float *x, __global const float *w, __global const float *b, __global float *y,
                       long tot, int IC, int ID, int IH, int IW, int OC, int OD, int OH, int OW,
                       int KD, int KH, int KW, int SD, int SH, int SW, int PD, int PH, int PW,
                       int DD, int DH, int DW, int G) {
    long t = MB_TID; if (t >= tot) return;
    long ow = t % OW, r = t / OW, oh = r % OH; r /= OH;
    long od = r % OD; r /= OD;
    long oc = r % OC, n = r / OC;
    int ICg = IC / G, OCg = OC / G;
    long ic0 = (oc / OCg) * ICg;
    mb_ksum acc = mb_ksum0();
    for (long ic = 0; ic < ICg; ic++)
    for (long kd = 0; kd < KD; kd++) {
        long id = od * SD - PD + kd * DD; if (id < 0 || id >= ID) continue;
        for (long kh = 0; kh < KH; kh++) {
            long ih = oh * SH - PH + kh * DH; if (ih < 0 || ih >= IH) continue;
            for (long kw = 0; kw < KW; kw++) {
                long iw = ow * SW - PW + kw * DW; if (iw < 0 || iw >= IW) continue;
                mb_kadd(&acc, x[(((n * IC + ic0 + ic) * ID + id) * IH + ih) * IW + iw]
                              * w[(((oc * ICg + ic) * KD + kd) * KH + kh) * KW + kw]);
            }
        }
    }
    y[t] = acc.s + (b ? b[oc] : 0.0f);
}
)CL");
void kernel_convnd(const float *in0, const float *in1, const float *in2, float *output,
                   int N, int IC, int ID, int IH, int IW, int OC, int OD, int OH, int OW,
                   int KD, int KH, int KW, int SD, int SH, int SW, int PD, int PH, int PW,
                   int DD, int DH, int DW, int G) {
    long tot = (long)N * OC * OD * OH * OW;
    mb_launch(MB_KERNEL(cl_convnd, k_convnd), mb_grid(tot), in0, in1, in2, output, tot, IC, ID, IH, IW, OC, OD, OH, OW,
              KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW, G);
}

// ---------------------------------------------------------------- pooling
static mb_program cl_maxpool2d = MB_CL(R"CL(
__kernel void k_maxpool2d(__global const float *x, __global float *y, int N, int C, int IH, int IW, int OH, int OW,
                          int KH, int KW, int SH, int SW, int PH, int PW, int DH, int DW) {
    long i = MB_TID; long tot = (long)N * C * OH * OW;
    if (i >= tot) return;
    int ow = i % OW; int oh = (i / OW) % OH; long nc = i / ((long)OW * OH);
    float m = -INFINITY;
    for (int kh = 0; kh < KH; kh++) {
        int ih = oh * SH - PH + kh * DH; if (ih < 0 || ih >= IH) continue;
        for (int kw = 0; kw < KW; kw++) {
            int iw = ow * SW - PW + kw * DW; if (iw < 0 || iw >= IW) continue;
            m = fmax(m, x[(nc * IH + ih) * IW + iw]);
        }
    }
    y[i] = m;
}
)CL");
void kernel_maxpool2d(const float *input, float *output, int N, int C, int IH, int IW,
                      int KH, int KW, int SH, int SW, int PH, int PW, int DH, int DW) {
    int OH = (IH + 2 * PH - DH * (KH - 1) - 1) / SH + 1, OW = (IW + 2 * PW - DW * (KW - 1) - 1) / SW + 1;
    long tot = (long)N * C * OH * OW;
    mb_launch(MB_KERNEL(cl_maxpool2d, k_maxpool2d), mb_grid(tot), input, output, N, C, IH, IW, OH, OW,
              KH, KW, SH, SW, PH, PW, DH, DW);
}

// explicit output size; NaN propagates (torch semantics)
static mb_program cl_maxpool2d_ex = MB_CL(R"CL(
__kernel void k_maxpool_ex(__global const float *x, __global float *y, int N, int C, int IH, int IW, int OH, int OW,
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
)CL");
void kernel_maxpool2d_ex(const float *input, float *output, int N, int C, int IH, int IW,
                         int OH, int OW, int KH, int KW, int SH, int SW, int PH, int PW, int DH, int DW) {
    long tot = (long)N * C * OH * OW;
    mb_launch(MB_KERNEL(cl_maxpool2d_ex, k_maxpool_ex), mb_grid(tot), input, output, N, C, IH, IW, OH, OW,
              KH, KW, SH, SW, PH, PW, DH, DW);
}

static mb_program cl_avgpool2d = MB_CL(R"CL(
__kernel void k_avgpool2d(__global const float *x, __global float *y, int N, int C, int IH, int IW, int OH, int OW,
                          int KH, int KW, int SH, int SW, int PH, int PW) {
    long i = MB_TID; long tot = (long)N * C * OH * OW;
    if (i >= tot) return;
    int ow = i % OW; int oh = (i / OW) % OH; long nc = i / ((long)OW * OH);
    float acc = 0.0f;
    for (int kh = 0; kh < KH; kh++) {
        int ih = oh * SH - PH + kh; if (ih < 0 || ih >= IH) continue;
        for (int kw = 0; kw < KW; kw++) {
            int iw = ow * SW - PW + kw; if (iw < 0 || iw >= IW) continue;
            acc += x[(nc * IH + ih) * IW + iw];
        }
    }
    y[i] = acc * (1.0f / (float)(KH * KW));
}
)CL");
void kernel_avgpool2d(const float *input, float *output, int N, int C, int IH, int IW,
                      int KH, int KW, int SH, int SW, int PH, int PW) {
    int OH = (IH + 2 * PH - KH) / SH + 1, OW = (IW + 2 * PW - KW) / SW + 1;
    long tot = (long)N * C * OH * OW;
    mb_launch(MB_KERNEL(cl_avgpool2d, k_avgpool2d), mb_grid(tot), input, output, N, C, IH, IW, OH, OW,
              KH, KW, SH, SW, PH, PW);
}

static mb_program cl_gap = MB_CL(R"CL(
__kernel void k_gap(__global const float *x, __global float *y, int NC, int HW) {
    long i = MB_TID; if (i >= NC) return;
    mb_ksum s = mb_ksum0();
    for (int j = 0; j < HW; j++) mb_kadd(&s, x[i * HW + j]);
    y[i] = s.s / HW;
}
)CL");
void kernel_adaptive_avg_pool2d(const float *input, float *output, int N, int C, int IH, int IW) {
    mb_launch(MB_KERNEL(cl_gap, k_gap), mb_grid((long)N * C), input, output, N * C, IH * IW);
}

// avg = 1 -> sum / (KD*KH*KW) (MB's count_include_pad semantics), else max
static mb_program cl_pool3d = MB_CL(R"CL(
__kernel void k_pool3d(__global const float *x, __global float *y, int N, int C, int ID, int IH, int IW,
                       int OD, int OH, int OW, int KD, int KH, int KW, int SD, int SH, int SW,
                       int PD, int PH, int PW, int DD, int DH, int DW, int avg) {
    long i = MB_TID; if (i >= (long)N * C * OD * OH * OW) return;
    int ow = i % OW; int oh = (i / OW) % OH; int od = (i / ((long)OW * OH)) % OD;
    long nc = i / ((long)OW * OH * OD);
    __global const float *xp = x + nc * ID * IH * IW;
    float acc = avg ? 0.0f : -INFINITY;
    for (int kd = 0; kd < KD; kd++) {
        int id = od * SD - PD + kd * DD; if (id < 0 || id >= ID) continue;
        for (int kh = 0; kh < KH; kh++) {
            int ih = oh * SH - PH + kh * DH; if (ih < 0 || ih >= IH) continue;
            for (int kw = 0; kw < KW; kw++) {
                int iw = ow * SW - PW + kw * DW; if (iw < 0 || iw >= IW) continue;
                float v = xp[((long)id * IH + ih) * IW + iw];
                acc = avg ? acc + v : fmax(acc, v);
            }
        }
    }
    y[i] = avg ? acc / (float)(KD * KH * KW) : acc;
}
)CL");
void kernel_maxpool3d(const float *input, float *output, int N, int C, int ID, int IH, int IW,
                      int OD, int OH, int OW, int KD, int KH, int KW, int SD, int SH, int SW,
                      int PD, int PH, int PW, int DD, int DH, int DW) {
    long tot = (long)N * C * OD * OH * OW;
    mb_launch(MB_KERNEL(cl_pool3d, k_pool3d), mb_grid(tot), input, output, N, C, ID, IH, IW, OD, OH, OW,
              KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW, 0);
}
void kernel_avgpool3d(const float *input, float *output, int N, int C, int ID, int IH, int IW,
                      int OD, int OH, int OW, int KD, int KH, int KW, int SD, int SH, int SW,
                      int PD, int PH, int PW) {
    long tot = (long)N * C * OD * OH * OW;
    mb_launch(MB_KERNEL(cl_pool3d, k_pool3d), mb_grid(tot), input, output, N, C, ID, IH, IW, OD, OH, OW,
              KD, KH, KW, SD, SH, SW, PD, PH, PW, 1, 1, 1, 1);
}

// ---------------------------------------------------------------- dense
static mb_program cl_linear = MB_CL(R"CL(
__kernel void k_linear(__global const float *x, __global const float *w, __global const float *b, __global float *y,
                       int M, int K, int N) {
    long i = MB_TID; if (i >= (long)M * N) return;
    int m = i / N, n = i % N;
    mb_ksum acc = mb_ksum0();
    for (int k = 0; k < K; k++) mb_kadd(&acc, x[(long)m * K + k] * w[(long)n * K + k]);
    y[i] = acc.s + (b ? b[n] : 0.0f);
}
)CL");
void kernel_linear(const float *input, const float *weight, const float *bias, float *output, int M, int K, int N) {
    mb_launch(MB_KERNEL(cl_linear, k_linear), mb_grid((long)M * N), input, weight, bias, output, M, K, N);
}

// C[m,n] = sum_k A[m*sam + k*sak] * B[k*sbk + n*sbn]  (any operand transposed)
static mb_program cl_matmul = MB_CL(R"CL(
__kernel void k_matmul_s(__global const float *A, __global const float *B, __global float *C, int M, int K, int N,
                         long sam, long sak, long sbk, long sbn) {
    long i = MB_TID; if (i >= (long)M * N) return;
    long m = i / N, n = i % N;
    mb_ksum acc = mb_ksum0();
    for (long k = 0; k < K; k++) mb_kadd(&acc, A[m * sam + k * sak] * B[k * sbk + n * sbn]);
    C[i] = acc.s;
}
)CL");
void kernel_matmul(const float *A, const float *B, float *C, int M, int K, int N) {
    mb_launch(MB_KERNEL(cl_matmul, k_matmul_s), mb_grid((long)M * N), A, B, C, M, K, N, (long)K, 1L, (long)N, 1L);
}
// A is [K, M]
void kernel_matmul_ta(const float *A, const float *B, float *C, int M, int K, int N) {
    mb_launch(MB_KERNEL(cl_matmul, k_matmul_s), mb_grid((long)M * N), A, B, C, M, K, N, 1L, (long)M, (long)N, 1L);
}
// B is [N, K]
void kernel_matmul_tb(const float *A, const float *B, float *C, int M, int K, int N) {
    mb_launch(MB_KERNEL(cl_matmul, k_matmul_s), mb_grid((long)M * N), A, B, C, M, K, N, (long)K, 1L, 1L, (long)K);
}
// A is [K, M], B is [N, K]
void kernel_matmul_tatb(const float *A, const float *B, float *C, int M, int K, int N) {
    mb_launch(MB_KERNEL(cl_matmul, k_matmul_s), mb_grid((long)M * N), A, B, C, M, K, N, 1L, (long)M, 1L, (long)K);
}

static mb_program cl_bmm = MB_CL(R"CL(
__kernel void k_bmm(__global const float *A, __global const float *B, __global float *C, int batch, int M, int K, int N) {
    long i = MB_TID; if (i >= (long)batch * M * N) return;
    long n = i % N, m = (i / N) % M, b = i / ((long)M * N);
    __global const float *a = A + (b * M + m) * K;
    __global const float *bb = B + b * (long)K * N + n;
    mb_ksum acc = mb_ksum0();
    for (int k = 0; k < K; k++) mb_kadd(&acc, a[k] * bb[(long)k * N]);
    C[i] = acc.s;
}
)CL");
void kernel_bmm(const float *A, const float *B, float *C, int batch, int M, int K, int N) {
    mb_launch(MB_KERNEL(cl_bmm, k_bmm), mb_grid((long)batch * M * N), A, B, C, batch, M, K, N);
}

// diag(a) @ b: row-wise scaling of b [N, M]
static mb_program cl_diag_matmul = MB_CL(R"CL(
__kernel void k_diag_matmul(__global const float *a, __global const float *b, __global float *y, int N, int M) {
    long i = MB_TID; if (i < (long)N * M) y[i] = a[i / M] * b[i];
}
)CL");
void kernel_diag_matmul(const float *a, const float *b, float *output, int N, int M) {
    mb_launch(MB_KERNEL(cl_diag_matmul, k_diag_matmul), mb_grid((long)N * M), a, b, output, N, M);
}

// ---------------------------------------------------------------- row ops (one work-group per row)
// Tree reductions over MB_THREADS work-items in __local memory.
static mb_program cl_softmax = MB_CL(R"CL(
float wg_max(float v, __local float *sh) {
    int t = get_local_id(0);
    sh[t] = v; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] = fmax(sh[t], sh[t + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float r = sh[0]; barrier(CLK_LOCAL_MEM_FENCE);
    return r;
}
float wg_sum(mb_ksum v, __local float *sh) {
    int t = get_local_id(0);
    sh[t] = v.s; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float r = sh[0]; barrier(CLK_LOCAL_MEM_FENCE);
    return r;
}
// log_out = 1: log-softmax
__kernel void k_softmax_rows(__global const float *x, __global float *y, int K, int log_out) {
    __local float sh[MB_THREADS];
    long r = get_group_id(0); int t = get_local_id(0), nt = get_local_size(0);
    __global const float *xr = x + r * K; __global float *yr = y + r * K;
    float mx = -INFINITY;
    for (int k = t; k < K; k += nt) mx = fmax(mx, xr[k]);
    mx = wg_max(mx, sh);
    mb_ksum s = mb_ksum0();
    for (int k = t; k < K; k += nt) mb_kadd(&s, exp(xr[k] - mx));
    float tot = wg_sum(s, sh);
    if (log_out) { float lse = mx + log(tot); for (int k = t; k < K; k += nt) yr[k] = xr[k] - lse; }
    else { float inv = 1.0f / tot; for (int k = t; k < K; k += nt) yr[k] = exp(xr[k] - mx) * inv; }
}
)CL");
void kernel_softmax(const float *input, float *output, int M, int K) {
    mb_launch(MB_KERNEL(cl_softmax, k_softmax_rows), mb_groups(M), input, output, K, 0);
}
void kernel_log_softmax(const float *input, float *output, int M, int K) {
    mb_launch(MB_KERNEL(cl_softmax, k_softmax_rows), mb_groups(M), input, output, K, 1);
}

static mb_program cl_layer_norm = MB_CL(R"CL(
float wg_sum(mb_ksum v, __local float *sh) {
    int t = get_local_id(0);
    sh[t] = v.s; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float r = sh[0]; barrier(CLK_LOCAL_MEM_FENCE);
    return r;
}
__kernel void k_layer_norm(__global const float *x, __global const float *g, __global const float *b,
                           __global float *y, int K, float eps) {
    __local float sh[MB_THREADS];
    long r = get_group_id(0); int t = get_local_id(0), nt = get_local_size(0);
    __global const float *xr = x + r * K; __global float *yr = y + r * K;
    mb_ksum s = mb_ksum0();
    for (int k = t; k < K; k += nt) mb_kadd(&s, xr[k]);
    float mu = wg_sum(s, sh) / K;
    mb_ksum q = mb_ksum0();
    for (int k = t; k < K; k += nt) { float d = xr[k] - mu; mb_kadd(&q, d * d); }
    float inv = rsqrt(wg_sum(q, sh) / K + eps);
    for (int k = t; k < K; k += nt) yr[k] = (xr[k] - mu) * inv * g[k] + b[k];
}
)CL");
void kernel_layer_norm(const float *input, const float *gamma, const float *beta, float *output,
                       int M, int K, float eps) {
    mb_launch(MB_KERNEL(cl_layer_norm, k_layer_norm), mb_groups(M), input, gamma, beta, output, K, eps);
}

// One work-group per (n, group): two-pass mean / variance with a __local tree.
static mb_program cl_group_norm = MB_CL(R"CL(
float wg_sum(mb_ksum v, __local float *sh) {
    int t = get_local_id(0);
    sh[t] = v.s; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float r = sh[0]; barrier(CLK_LOCAL_MEM_FENCE);
    return r;
}
__kernel void k_group_norm(__global const float *x, __global const float *gm, __global const float *bt,
                           __global float *y, int C, int G, int HW, float eps) {
    __local float sh[MB_THREADS];
    int n = get_group_id(0) / G, g = get_group_id(0) % G, cpg = C / G;
    int t = get_local_id(0), nt = get_local_size(0);
    long cnt = (long)cpg * HW, base = ((long)n * C + (long)g * cpg) * HW;
    __global const float *xp = x + base; __global float *yp = y + base;
    mb_ksum s = mb_ksum0();
    for (long j = t; j < cnt; j += nt) mb_kadd(&s, xp[j]);
    float mean = wg_sum(s, sh) / (float)cnt;
    mb_ksum q = mb_ksum0();
    for (long j = t; j < cnt; j += nt) { float d = xp[j] - mean; mb_kadd(&q, d * d); }
    float inv = rsqrt(wg_sum(q, sh) / (float)cnt + eps);
    for (long j = t; j < cnt; j += nt) {
        int c = g * cpg + (int)(j / HW);
        yp[j] = (xp[j] - mean) * inv * gm[c] + bt[c];
    }
}
)CL");
void kernel_group_norm(const float *input, const float *gamma, const float *beta, float *output,
                       int N, int C, int G, int HW, float eps) {
    mb_launch(MB_KERNEL(cl_group_norm, k_group_norm), mb_groups((long)N * G), input, gamma, beta, output, C, G, HW, eps);
}

// softmax over the middle axis of [outer, R, inner]: one work-item per (outer, inner)
static mb_program cl_softmax_dim = MB_CL(R"CL(
__kernel void k_softmax_dim(__global const float *in, __global float *out, int outer, int R, int inner, int log_out) {
    long i = MB_TID; if (i >= (long)outer * inner) return;
    long o = i / inner, j = i % inner;
    __global const float *x = in + o * (long)R * inner + j;
    __global float *y = out + o * (long)R * inner + j;
    float m = x[0];
    for (long r = 1; r < R; r++) m = fmax(m, x[r * inner]);
    mb_ksum sd = mb_ksum0();
    for (long r = 0; r < R; r++) mb_kadd(&sd, exp(x[r * inner] - m));
    float s = sd.s;
    for (long r = 0; r < R; r++) y[r * inner] = log_out ? (x[r * inner] - m) - log(s) : exp(x[r * inner] - m) / s;
}
)CL");
void kernel_softmax_dim(const float *input, float *output, int outer, int R, int inner, int log_out) {
    mb_launch(MB_KERNEL(cl_softmax_dim, k_softmax_dim), mb_grid((long)outer * inner), input, output, outer, R, inner, log_out);
}

// Frobenius normalisation of the whole tensor: per-group partial sums of
// squares -> one group adds them -> scale.
static mb_program cl_frobenius = MB_CL(R"CL(
__kernel void k_sumsq_partial(__global const float *x, __global float *part, long n) {
    __local float sh[MB_THREADS];
    int t = get_local_id(0);
    mb_ksum s = mb_ksum0();
    for (long i = get_global_id(0); i < n; i += get_global_size(0)) { float v = x[i]; mb_kadd(&s, v * v); }
    sh[t] = s.s; barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = get_local_size(0) / 2; k > 0; k >>= 1) { if (t < k) sh[t] += sh[t + k]; barrier(CLK_LOCAL_MEM_FENCE); }
    if (t == 0) part[get_group_id(0)] = sh[0];
}
__kernel void k_sum_partials(__global float *part, int np) {
    __local float sh[MB_THREADS];
    int t = get_local_id(0);
    mb_ksum s = mb_ksum0();
    for (int i = t; i < np; i += get_local_size(0)) mb_kadd(&s, part[i]);
    sh[t] = s.s; barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = get_local_size(0) / 2; k > 0; k >>= 1) { if (t < k) sh[t] += sh[t + k]; barrier(CLK_LOCAL_MEM_FENCE); }
    if (t == 0) part[np] = sh[0];
}
__kernel void k_scale_by_rnorm(__global const float *x, __global float *y, __global const float *tot, long n) {
    long i = MB_TID; if (i < n) y[i] = x[i] * (1.0f / sqrt(tot[0]));
}
)CL");
void kernel_frobenius_norm(const float *input, float *output, int n) {
    int np = (int)std::min(1024L, ((long)n + MB_THREADS - 1) / MB_THREADS);
    float *part = (float *)MB_SCRATCH(cl_frobenius, (np + 1) * sizeof(float));
    mb_launch(MB_KERNEL(cl_frobenius, k_sumsq_partial), mb_groups(np), input, part, (long)n);
    mb_launch(MB_KERNEL(cl_frobenius, k_sum_partials), mb_groups(1), part, np);
    mb_launch(MB_KERNEL(cl_frobenius, k_scale_by_rnorm), mb_grid(n), input, output, (const float *)(part + np), (long)n);
}

// ---------------------------------------------------------------- attention
// One work-item per (bh, query row); online softmax over keys.
static mb_program cl_sdpa = MB_CL(R"CL(
__kernel void k_sdpa(__global const float *Q, __global const float *Kt, __global const float *V, __global float *O,
                     int BH, int S, int D) {
    long r = MB_TID; if (r >= (long)BH * S) return;
    long bh = r / S; int i = r % S;
    __global const float *q = Q + (bh * S + i) * D;
    __global const float *Kb = Kt + bh * S * D;
    __global const float *Vb = V + bh * S * D;
    __global float *o = O + (bh * S + i) * D;
    float scale = rsqrt((float)D), mx = -INFINITY;
    for (int j = 0; j < S; j++) {
        mb_ksum s = mb_ksum0();
        for (int d = 0; d < D; d++) mb_kadd(&s, q[d] * Kb[(long)j * D + d]);
        mx = fmax(mx, s.s * scale);
    }
    float sum = 0.0f;
    for (int d = 0; d < D; d++) o[d] = 0.0f;
    for (int j = 0; j < S; j++) {
        mb_ksum s = mb_ksum0();
        for (int d = 0; d < D; d++) mb_kadd(&s, q[d] * Kb[(long)j * D + d]);
        float p = exp(s.s * scale - mx); sum += p;
        for (int d = 0; d < D; d++) o[d] += p * Vb[(long)j * D + d];
    }
    for (int d = 0; d < D; d++) o[d] /= sum;
}
)CL");
void kernel_sdpa(const float *Q, const float *K, const float *V, float *output, int BH, int S, int D) {
    mb_launch(MB_KERNEL(cl_sdpa, k_sdpa), mb_grid((long)BH * S), Q, K, V, output, BH, S, D);
}

// ---------------------------------------------------------------- layout
static mb_program cl_cat_c = MB_CL(R"CL(
__kernel void k_copy_c(__global const float *src, __global float *dst, int N, int c, int C, int c_off, int HW) {
    long i = MB_TID; long tot = (long)N * c * HW; if (i >= tot) return;
    long n = i / ((long)c * HW), r = i % ((long)c * HW);
    dst[(n * C + c_off) * HW + r] = src[i];
}
)CL");
static void cat_part(const float *src, float *dst, int N, int c, int C, int off, int HW) {
    long tot = (long)N * c * HW;
    mb_launch(MB_KERNEL(cl_cat_c, k_copy_c), mb_grid(tot), src, dst, N, c, C, off, HW);
}
void kernel_cat2_c1(const float *in0, int c0, const float *in1, int c1, float *out, int N, int H, int W) {
    int C = c0 + c1, HW = H * W;
    cat_part(in0, out, N, c0, C, 0, HW); cat_part(in1, out, N, c1, C, c0, HW);
}
void kernel_cat3_c1(const float *in0, int c0, const float *in1, int c1, const float *in2, int c2,
                    float *out, int N, int H, int W) {
    int C = c0 + c1 + c2, HW = H * W;
    cat_part(in0, out, N, c0, C, 0, HW); cat_part(in1, out, N, c1, C, c0, HW); cat_part(in2, out, N, c2, C, c0 + c1, HW);
}
void kernel_cat4_c1(const float *in0, int c0, const float *in1, int c1, const float *in2, int c2,
                    const float *in3, int c3, float *out, int N, int H, int W) {
    int C = c0 + c1 + c2 + c3, HW = H * W;
    cat_part(in0, out, N, c0, C, 0, HW); cat_part(in1, out, N, c1, C, c0, HW);
    cat_part(in2, out, N, c2, C, c0 + c1, HW); cat_part(in3, out, N, c3, C, c0 + c1 + c2, HW);
}

static mb_program cl_catn = MB_CL(R"CL(
__kernel void k_cat_slab(__global const float *x, __global float *y, int outer, long slab, long ytot, long ybase) {
    long i = MB_TID; if (i >= (long)outer * slab) return;
    long o = i / slab, j = i % slab;
    y[o * ytot + ybase + j] = x[i];
}
)CL");
void kernel_catn(const float *in0, const float *in1, const float *in2, const float *in3,
                 const float *in4, const float *in5, const float *in6, const float *in7,
                 float *output, int n_in, int outer, int inner,
                 int c0, int c1, int c2, int c3, int c4, int c5, int c6, int c7) {
    const float *ins[8] = {in0, in1, in2, in3, in4, in5, in6, in7};
    int cs[8] = {c0, c1, c2, c3, c4, c5, c6, c7};
    long ctot = 0; for (int k = 0; k < n_in; k++) ctot += cs[k];
    long cbase = 0;
    for (int k = 0; k < n_in; k++) {
        long slab = (long)cs[k] * inner;
        mb_launch(MB_KERNEL(cl_catn, k_cat_slab), mb_grid((long)outer * slab), ins[k], output, outer, slab,
                  ctot * inner, cbase * inner);
        cbase += cs[k];
    }
}

static mb_program cl_upsample = MB_CL(R"CL(
__kernel void k_upsample(__global const float *x, __global float *y, long NC, int IH, int IW, int s) {
    long i = MB_TID; int OH = IH * s, OW = IW * s; if (i >= NC * OH * OW) return;
    int ow = i % OW, oh = (i / OW) % OH; long nc = i / ((long)OW * OH);
    y[i] = x[(nc * IH + oh / s) * IW + ow / s];
}
)CL");
void kernel_upsample_nearest(const float *input, float *output, int N, int C, int IH, int IW, int scale) {
    long tot = (long)N * C * IH * scale * IW * scale;
    mb_launch(MB_KERNEL(cl_upsample, k_upsample), mb_grid(tot), input, output, (long)N * C, IH, IW, scale);
}

static mb_program cl_strided_copy = MB_CL(R"CL(
__kernel void k_strided_copy(__global const float *x, __global float *y, long n, int off, int d1, int d2, int d3, int d4,
                             int d5, int s0, int s1, int s2, int s3, int s4, int s5) {
    long i = MB_TID; if (i >= n) return;
    long c[6]; mb_idx6(i, d1, d2, d3, d4, d5, c);
    y[i] = x[off + c[0] * s0 + c[1] * s1 + c[2] * s2 + c[3] * s3 + c[4] * s4 + c[5] * s5];
}
)CL");
void kernel_strided_copy(const float *input, float *output, int off,
                         int d0, int d1, int d2, int d3, int d4, int d5,
                         int s0, int s1, int s2, int s3, int s4, int s5) {
    long n = (long)d0 * d1 * d2 * d3 * d4 * d5;
    mb_launch(MB_KERNEL(cl_strided_copy, k_strided_copy), mb_grid(n), input, output, n, off, d1, d2, d3, d4, d5,
              s0, s1, s2, s3, s4, s5);
}

static mb_program cl_strided_scatter = MB_CL(R"CL(
__kernel void k_strided_scatter(__global const float *src, __global float *y, long m, int off, int d1, int d2, int d3,
                                int d4, int d5, int s0, int s1, int s2, int s3, int s4, int s5) {
    long i = MB_TID; if (i >= m) return;
    long c[6]; mb_idx6(i, d1, d2, d3, d4, d5, c);
    y[off + c[0] * s0 + c[1] * s1 + c[2] * s2 + c[3] * s3 + c[4] * s4 + c[5] * s5] = src[i];
}
)CL");
void kernel_strided_scatter(const float *input, const float *src, float *output, int n, int off,
                            int d0, int d1, int d2, int d3, int d4, int d5,
                            int s0, int s1, int s2, int s3, int s4, int s5) {
    long m = (long)d0 * d1 * d2 * d3 * d4 * d5;
    mb_copy(output, input, (size_t)n * sizeof(float));
    mb_launch(MB_KERNEL(cl_strided_scatter, k_strided_scatter), mb_grid(m), src, output, m, off, d1, d2, d3, d4, d5,
              s0, s1, s2, s3, s4, s5);
}

static mb_program cl_flip = MB_CL(R"CL(
__kernel void k_flip(__global const float *x, __global float *y, int axis, int inner, long n) {
    long k = MB_TID; if (k >= n) return;
    long i = k % inner, r = k / inner, a = r % axis, o = r / axis;
    y[k] = x[(o * axis + (axis - 1 - a)) * inner + i];
}
)CL");
void kernel_flip(const float *input, float *output, int outer, int axis, int inner) {
    long n = (long)outer * axis * inner;
    mb_launch(MB_KERNEL(cl_flip, k_flip), mb_grid(n), input, output, axis, inner, n);
}

// upper = 0: tril (keep j <= i + diagonal), 1: triu (keep j >= i + diagonal)
static mb_program cl_tri = MB_CL(R"CL(
__kernel void k_tri(__global const float *x, __global float *y, int M, int N, int diagonal, int upper) {
    long k = MB_TID; if (k >= (long)M * N) return;
    int i = k / N, j = k % N;
    int keep = upper ? (j >= i + diagonal) : (j <= i + diagonal);
    y[k] = keep ? x[k] : 0.0f;
}
)CL");
void kernel_tril(const float *input, float *output, int M, int N, int diagonal) {
    mb_launch(MB_KERNEL(cl_tri, k_tri), mb_grid((long)M * N), input, output, M, N, diagonal, 0);
}
void kernel_triu(const float *input, float *output, int M, int N, int diagonal) {
    mb_launch(MB_KERNEL(cl_tri, k_tri), mb_grid((long)M * N), input, output, M, N, diagonal, 1);
}

// ---------------------------------------------------------------- indexing
static mb_program cl_gather = MB_CL(R"CL(
__kernel void k_gather(__global const float *x, __global const int *idx, __global float *y, int n) {
    long i = MB_TID; if (i < n) { int j = idx[i]; y[i] = j >= 0 ? x[j] : 0.0f; }
}
)CL");
void kernel_gather(const float *input, const int *index, float *output, int n) {
    mb_launch(MB_KERNEL(cl_gather, k_gather), mb_grid(n), input, index, output, n);
}

// Index tensors are float (the fp32 pipeline carries them as exact floats); they
// convert through int: Adreno 650's compiler aborts on half -> long (the _f16 seeds).
static mb_program cl_index_select = MB_CL(R"CL(
long mb_wrap(float f, long A) { long a = (long)(int)f; return a < 0 ? a + A : a; }
__kernel void k_index_select(__global const float *x, __global const float *idx, __global float *y, long n,
                             int A, int J, int inner) {
    long k = MB_TID; if (k >= n) return;
    long i = k % inner, r = k / inner, j = r % J, o = r / J;
    y[k] = x[(o * A + mb_wrap(idx[j], A)) * inner + i];
}
)CL");
void kernel_index_select(const float *input, const float *index, float *output, int outer, int A, int J, int inner) {
    long n = (long)outer * J * inner;
    mb_launch(MB_KERNEL(cl_index_select, k_index_select), mb_grid(n), input, index, output, n, A, J, inner);
}

static mb_program cl_gather_dim = MB_CL(R"CL(
long mb_wrap(float f, long A) { long a = (long)(int)f; return a < 0 ? a + A : a; }
__kernel void k_gather_dim(__global const float *x, __global const float *idx, __global float *y, long n,
                           int A, int J, int inner) {
    long k = MB_TID; if (k >= n) return;
    long i = k % inner, o = k / ((long)J * inner);
    y[k] = x[(o * A + mb_wrap(idx[k], A)) * inner + i];
}
)CL");
void kernel_gather_dim(const float *input, const float *index, float *output, int outer, int A, int J, int inner) {
    long n = (long)outer * J * inner;
    mb_launch(MB_KERNEL(cl_gather_dim, k_gather_dim), mb_grid(n), input, index, output, n, A, J, inner);
}

// mode 1: scatter-add (float atomics), else overwrite
static mb_program cl_scatter_dim = MB_CL(R"CL(
long mb_wrap(float f, long A) { long a = (long)(int)f; return a < 0 ? a + A : a; }
__kernel void k_scatter_dim(__global const float *idx, __global const float *src, __global float *y, long n,
                            int A, int J, int inner, int mode) {
    long k = MB_TID; if (k >= n) return;
    long i = k % inner, o = k / ((long)J * inner);
    __global float *d = y + (o * A + mb_wrap(idx[k], A)) * inner + i;
    if (mode) mb_atomic_addf(d, src[k]); else *d = src[k];
}
)CL");
void kernel_scatter_dim(const float *input, const float *index, const float *src, float *output,
                        int outer, int A, int J, int inner, int mode) {
    long nb = (long)outer * A * inner, n = (long)outer * J * inner;
    mb_copy(output, input, (size_t)nb * sizeof(float));
    mb_launch(MB_KERNEL(cl_scatter_dim, k_scatter_dim), mb_grid(n), index, src, output, n, A, J, inner, mode);
}

// one work-item per (outer, inner) row: insertion into a sorted window of K
static mb_program cl_topk = MB_CL(R"CL(
int mb_topk_before(float a, long ia, float b, long ib, int largest) {
    int an = isnan(a), bn = isnan(b);
    if (an || bn) { if (an && bn) return ia < ib; return largest ? an : bn; }
    if (a != b) return largest ? a > b : a < b;
    return ia < ib;
}
__kernel void k_topk(__global const float *in, __global float *val, __global float *ind, long rows, int A, int inner,
                     int K, int largest) {
    long r = MB_TID; if (r >= rows) return;
    long o = r / inner, i = r % inner;
    __global const float *x = in + o * (long)A * inner + i;
    __global float *v = val + o * (long)K * inner + i;
    __global float *ix = ind + o * (long)K * inner + i;
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
)CL");
void kernel_topk(const float *input, float *values, float *indices, int outer, int A, int inner, int K, int largest) {
    long rows = (long)outer * inner;
    mb_launch(MB_KERNEL(cl_topk, k_topk), mb_grid(rows), input, values, indices, rows, A, inner, K, largest);
}

// ---------------------------------------------------------------- generic elementwise (opcode ops)
static mb_program cl_ewise1 = MB_CL(R"CL(
float mb_e1(float x, int opc, float p0, float p1) {
    switch (opc) {
    case 0: return -x;  case 1: return exp(x);  case 2: return log(x);  case 3: return sqrt(x);
    case 4: return rsqrt(x);  case 5: return fabs(x);  case 6: return tanh(x);
    case 7: return 1.0f / (1.0f + exp(-x));  case 8: return x > 0.0f ? x : 0.0f;
    case 9: return sin(x);  case 10: return cos(x);  case 11: return erf(x);
    case 12: return 0.5f * x * (1.0f + erf(x * 0.70710678118654752f));
    case 13: return 0.5f * x * (1.0f + tanh(0.79788456080286536f * (x + 0.044715f * x * x * x)));
    case 14: return x / (1.0f + exp(-x));  case 15: return fmin(fmax(x, p0), p1);
    case 16: return 1.0f / x;  case 17: return x * x;
    case 18: return x > 20.0f ? x : log1p(exp(x));  case 19: return x >= 0.0f ? x : p0 * x;
    case 20: return x > 0.0f ? x : p0 * (exp(x) - 1.0f);
    case 21: return x * fmin(fmax(x + 3.0f, 0.0f), 6.0f) / 6.0f;
    case 22: return fmin(fmax(x / 6.0f + 0.5f, 0.0f), 1.0f);
    case 23: return log1p(x);  case 24: return expm1(x);
    case 25: return (float)((x > 0.0f) - (x < 0.0f));  case 26: return floor(x);
    case 27: return ceil(x);  case 28: return rint(x);  case 29: return x == 0.0f ? 1.0f : 0.0f;
    case 31: return x * tanh(x > 20.0f ? x : log1p(exp(x)));
    default: return x;
    }
}
__kernel void k_ewise1(__global const float *x, __global float *y, int n, int opc, float p0, float p1) {
    long i = MB_TID; if (i < n) y[i] = mb_e1(x[i], opc, p0, p1);
}
)CL");
void kernel_ewise1(const float *input, float *output, int n, int opc, float p0, float p1) {
    mb_launch(MB_KERNEL(cl_ewise1, k_ewise1), mb_grid(n), input, output, n, opc, p0, p1);
}

// broadcasting binary op over a 6-d index space with per-operand strides
static mb_program cl_ewise2 = MB_CL(R"CL(
float mb_e2(float x, float y, int opc, float alpha) {
    switch (opc) {
    case 0: return x + alpha * y;  case 1: return x - alpha * y;  case 2: return x * y;
    case 3: return x / y;  case 4: return (x > y || isnan(x)) ? x : y;
    case 5: return (x < y || isnan(x)) ? x : y;  case 6: return pow(x, y);
    case 7: return y != 0.0f ? alpha : x;  case 8: return x == y;  case 9: return x != y;
    case 10: return x < y;  case 11: return x <= y;  case 12: return x > y;  case 13: return x >= y;
    default: return x;
    }
}
__kernel void k_ewise2(__global const float *a, __global const float *b, __global float *y, long n, int opc, float alpha,
                       int d1, int d2, int d3, int d4, int d5,
                       int sa0, int sa1, int sa2, int sa3, int sa4, int sa5,
                       int sb0, int sb1, int sb2, int sb3, int sb4, int sb5) {
    long i = MB_TID; if (i >= n) return;
    long c[6]; mb_idx6(i, d1, d2, d3, d4, d5, c);
    float x = a[c[0] * sa0 + c[1] * sa1 + c[2] * sa2 + c[3] * sa3 + c[4] * sa4 + c[5] * sa5];
    float v = b[c[0] * sb0 + c[1] * sb1 + c[2] * sb2 + c[3] * sb3 + c[4] * sb4 + c[5] * sb5];
    y[i] = mb_e2(x, v, opc, alpha);
}
)CL");
void kernel_ewise2(const float *a, const float *b, float *output, int opc, float alpha,
                   int d0, int d1, int d2, int d3, int d4, int d5,
                   int sa0, int sa1, int sa2, int sa3, int sa4, int sa5,
                   int sb0, int sb1, int sb2, int sb3, int sb4, int sb5) {
    long n = (long)d0 * d1 * d2 * d3 * d4 * d5;
    mb_launch(MB_KERNEL(cl_ewise2, k_ewise2), mb_grid(n), a, b, output, n, opc, alpha, d1, d2, d3, d4, d5,
              sa0, sa1, sa2, sa3, sa4, sa5, sb0, sb1, sb2, sb3, sb4, sb5);
}

static mb_program cl_ewise_scalar = MB_CL(R"CL(
__kernel void k_ewise_scalar(__global const float *x, __global float *y, int n, int opc, float c) {
    long i = MB_TID; if (i >= n) return;
    float v = x[i], z;
    switch (opc) {
    case 0: z = v + c; break;  case 1: z = v - c; break;  case 2: z = c - v; break;
    case 3: z = v * c; break;  case 4: z = v / c; break;  case 5: z = c / v; break;
    case 6: z = pow(v, c); break;  case 7: z = pow(c, v); break;
    case 8: z = (v > c || isnan(v)) ? v : c; break;  case 9: z = (v < c || isnan(v)) ? v : c; break;
    case 10: z = v == c; break;  case 11: z = v != c; break;  case 12: z = v < c; break;
    case 13: z = v <= c; break;  case 14: z = v > c; break;  case 15: z = v >= c; break;
    default: z = v;
    }
    y[i] = z;
}
)CL");
void kernel_ewise_scalar(const float *input, float *output, int n, int opc, float c) {
    mb_launch(MB_KERNEL(cl_ewise_scalar, k_ewise_scalar), mb_grid(n), input, output, n, opc, c);
}

// generic [outer, R, inner] reduction, opcode as reference_kernels_ext: 0 sum, 1 mean, 2 max, 3 min,
// 4 prod, 5 sumsq, 6 L2 norm, 7 argmax, 8 argmin, 9 var, 10 var (unbiased)
static mb_program cl_reduce = MB_CL(R"CL(
__kernel void k_reduce(__global const float *in, __global float *y, int outer, int R, int inner, int opc) {
    long i = MB_TID; if (i >= (long)outer * inner) return;
    long o = i / inner, j = i % inner;
    __global const float *x = in + o * (long)R * inner + j;
    float acc;
    if (opc == 7 || opc == 8) {
        long best = 0; float bv = x[0];
        for (long r = 1; r < R; r++) { float v = x[r * inner]; if (opc == 7 ? v > bv : v < bv) { bv = v; best = r; } }
        acc = (float)best;
    } else if (opc == 9 || opc == 10) {
        mb_ksum m = mb_ksum0(); for (long r = 0; r < R; r++) mb_kadd(&m, x[r * inner]);
        float mu = m.s / (float)R;
        mb_ksum q = mb_ksum0(); for (long r = 0; r < R; r++) { float d = x[r * inner] - mu; mb_kadd(&q, d * d); }
        acc = q.s / (float)(opc == 10 ? R - 1 : R);
    } else if (opc == 2 || opc == 3 || opc == 4) {
        float a = opc == 4 ? 1.0f : x[0];
        for (long r = opc == 4 ? 0 : 1; r < R; r++) {
            float v = x[r * inner];
            if (opc == 2) { if (v > a || isnan(v)) a = v; }
            else if (opc == 3) { if (v < a || isnan(v)) a = v; }
            else a *= v;
        }
        acc = a;
    } else {
        mb_ksum a = mb_ksum0();
        for (long r = 0; r < R; r++) { float v = x[r * inner]; mb_kadd(&a, (opc == 5 || opc == 6) ? v * v : v); }
        acc = a.s;
        if (opc == 1) acc /= (float)R;
        if (opc == 6) acc = sqrt(acc);
    }
    y[i] = acc;
}
)CL");
void kernel_reduce(const float *input, float *output, int outer, int R, int inner, int opc) {
    mb_launch(MB_KERNEL(cl_reduce, k_reduce), mb_grid((long)outer * inner), input, output, outer, R, inner, opc);
}

// ---------------------------------------------------------------- scans
static mb_program cl_scan = MB_CL(R"CL(
__kernel void k_scan(__global const float *in, __global float *out, long rows, int A, int inner, int prod) {
    long t = MB_TID; if (t >= rows) return;
    long o = t / inner, i = t % inner;
    mb_ksum s = mb_ksum0(); float p = 1.0f;
    for (long a = 0; a < A; a++) {
        long idx = (o * A + a) * inner + i;
        if (prod) { p *= in[idx]; out[idx] = p; }
        else { mb_kadd(&s, in[idx]); out[idx] = s.s; }
    }
}
)CL");
void kernel_cumsum(const float *input, float *output, int outer, int axis, int inner) {
    long rows = (long)outer * inner;
    mb_launch(MB_KERNEL(cl_scan, k_scan), mb_grid(rows), input, output, rows, axis, inner, 0);
}
void kernel_cumprod(const float *input, float *output, int outer, int axis, int inner) {
    long rows = (long)outer * inner;
    mb_launch(MB_KERNEL(cl_scan, k_scan), mb_grid(rows), input, output, rows, axis, inner, 1);
}

// output [Bout, N+1]: out[b, 0] = 0, out[b, k] = sum_{j<k} x[b, j]
static mb_program cl_exclusive_cumsum = MB_CL(R"CL(
__kernel void k_excl_cumsum(__global const float *x, __global float *y, int Bout, int N) {
    long b = MB_TID; if (b >= Bout) return;
    mb_ksum s = mb_ksum0();
    for (int k = 0; k <= N; k++) {
        y[b * (N + 1) + k] = s.s;
        if (k < N) mb_kadd(&s, x[b * N + k]);
    }
}
)CL");
void kernel_exclusive_cumsum(const float *x, float *output, int Bout, int N) {
    mb_launch(MB_KERNEL(cl_exclusive_cumsum, k_excl_cumsum), mb_grid(Bout), x, output, Bout, N);
}

// ---------------------------------------------------------------- recurrent
// one launch per timestep, one work-item per (b, j)
static mb_program cl_lstm_seq = MB_CL(R"CL(
__kernel void k_copy(__global const float *x, __global float *y, long n) { long i = MB_TID; if (i < n) y[i] = x[i]; }
__kernel void k_rows_from_seq(__global const float *y, __global float *h, int B, int T, int H, int t) {
    long i = MB_TID; if (i >= (long)B * H) return;
    long b = i / H, j = i % H;
    h[i] = y[((long)b * T + t) * H + j];
}
__kernel void k_lstm_step(__global const float *x, __global const float *w_ih, __global const float *w_hh,
                          __global const float *bg, __global const float *hn, __global float *cn, __global float *y,
                          int B, int T, int I, int H, int t) {
    long i = MB_TID; if (i >= (long)B * H) return;
    long b = i / H, j = i % H;
    __global const float *xt = x + ((long)b * T + t) * I;
    __global const float *hp = hn + b * H;
    float g4[4];
    for (int q = 0; q < 4; q++) {
        long row = (long)q * H + j;
        mb_ksum acc = mb_ksum0();
        for (int k = 0; k < I; k++) mb_kadd(&acc, w_ih[row * I + k] * xt[k]);
        for (int k = 0; k < H; k++) mb_kadd(&acc, w_hh[row * H + k] * hp[k]);
        g4[q] = acc.s + bg[row];
    }
    float ig = 1.0f / (1.0f + exp(-g4[0])), fg = 1.0f / (1.0f + exp(-g4[1]));
    float gg = tanh(g4[2]), og = 1.0f / (1.0f + exp(-g4[3]));
    float c = fg * cn[i] + ig * gg;
    cn[i] = c;
    y[((long)b * T + t) * H + j] = og * tanh(c);
}
)CL");
void kernel_lstm_seq(const float *x, const float *h0, const float *c0, const float *w_ih,
                     const float *w_hh, const float *b_gates, float *y, float *hn, float *cn,
                     int B, int T, int I, int H, int reverse) {
    long bh = (long)B * H;
    mb_launch(MB_KERNEL(cl_lstm_seq, k_copy), mb_grid(bh), h0, hn, bh);
    mb_launch(MB_KERNEL(cl_lstm_seq, k_copy), mb_grid(bh), c0, cn, bh);
    for (int s = 0; s < T; s++) {
        int t = reverse ? T - 1 - s : s;
        mb_launch(MB_KERNEL(cl_lstm_seq, k_lstm_step), mb_grid(bh), x, w_ih, w_hh, b_gates, (const float *)hn, cn, y,
                  B, T, I, H, t);
        mb_launch(MB_KERNEL(cl_lstm_seq, k_rows_from_seq), mb_grid(bh), (const float *)y, hn, B, T, H, t);
    }
}

static mb_program cl_gru_seq = MB_CL(R"CL(
__kernel void k_copy(__global const float *x, __global float *y, long n) { long i = MB_TID; if (i < n) y[i] = x[i]; }
__kernel void k_rows_from_seq(__global const float *y, __global float *h, int B, int T, int H, int t) {
    long i = MB_TID; if (i >= (long)B * H) return;
    long b = i / H, j = i % H;
    h[i] = y[((long)b * T + t) * H + j];
}
__kernel void k_gru_step(__global const float *x, __global const float *w_ih, __global const float *w_hh,
                         __global const float *b_ih, __global const float *b_hh, __global const float *hn,
                         __global float *y, int B, int T, int I, int H, int t) {
    long i = MB_TID; if (i >= (long)B * H) return;
    long b = i / H, j = i % H;
    __global const float *xt = x + ((long)b * T + t) * I;
    __global const float *hp = hn + b * H;
    float gi[3], gh[3];
    for (int q = 0; q < 3; q++) {
        long row = (long)q * H + j;
        mb_ksum ai = mb_ksum0(), ah = mb_ksum0();
        for (int k = 0; k < I; k++) mb_kadd(&ai, w_ih[row * I + k] * xt[k]);
        for (int k = 0; k < H; k++) mb_kadd(&ah, w_hh[row * H + k] * hp[k]);
        gi[q] = ai.s + b_ih[row]; gh[q] = ah.s + b_hh[row];
    }
    float r = 1.0f / (1.0f + exp(-(gi[0] + gh[0]))), z = 1.0f / (1.0f + exp(-(gi[1] + gh[1])));
    float nn = tanh(gi[2] + r * gh[2]);
    y[((long)b * T + t) * H + j] = (1.0f - z) * nn + z * hp[j];
}
)CL");
void kernel_gru_seq(const float *x, const float *h0, const float *w_ih, const float *w_hh,
                    const float *b_ih, const float *b_hh, float *y, float *hn,
                    int B, int T, int I, int H, int reverse) {
    long bh = (long)B * H;
    mb_launch(MB_KERNEL(cl_gru_seq, k_copy), mb_grid(bh), h0, hn, bh);
    for (int s = 0; s < T; s++) {
        int t = reverse ? T - 1 - s : s;
        mb_launch(MB_KERNEL(cl_gru_seq, k_gru_step), mb_grid(bh), x, w_ih, w_hh, b_ih, b_hh, (const float *)hn, y,
                  B, T, I, H, t);
        mb_launch(MB_KERNEL(cl_gru_seq, k_rows_from_seq), mb_grid(bh), (const float *)y, hn, B, T, H, t);
    }
}

// single-step LSTM cell; h is updated after every unit used it (out must not alias h)
static mb_program cl_lstm = MB_CL(R"CL(
__kernel void k_lstm_cell(__global const float *x, __global const float *w_ih, __global const float *w_hh,
                          __global const float *bias, __global const float *h, __global float *c, __global float *out,
                          int I, int H) {
    long j = MB_TID; if (j >= H) return;
    float pre[4];
    for (int g = 0; g < 4; g++) {
        long row = (long)g * H + j;
        mb_ksum ax = mb_ksum0(), ah = mb_ksum0();
        for (int k = 0; k < I; k++) mb_kadd(&ax, x[k] * w_ih[row * I + k]);
        for (int k = 0; k < H; k++) mb_kadd(&ah, h[k] * w_hh[row * H + k]);
        pre[g] = ax.s + ah.s + bias[row];
    }
    float ig = 1.0f / (1.0f + exp(-pre[0])), fg = 1.0f / (1.0f + exp(-pre[1]));
    float cg = tanh(pre[2]), og = 1.0f / (1.0f + exp(-pre[3]));
    float cn = fg * c[j] + ig * cg;
    c[j] = cn;
    out[j] = og * tanh(cn);
}
)CL");
void kernel_lstm(const float *x, const float *w_ih, const float *w_hh, const float *bias, float *h, float *c,
                 float *out, int in_size, int H) {
    mb_launch(MB_KERNEL(cl_lstm, k_lstm_cell), mb_grid(H), x, w_ih, w_hh, bias, (const float *)h, c, out, in_size, H);
    mb_copy(h, out, (size_t)H * sizeof(float));
}
