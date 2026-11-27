/* Reference (scalar C) kernels of the K3 IR runtime. The int8_mixed kinds are the KernelSpec
 * reference implementations of pipeline/int8_mixed.py verbatim (the bit-exact contract); the fp32
 * kinds follow gpu_eval/harness/walk_torch.py's rules. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k3ir.h"

#define IN(i) k3_ptr(g, op->in[i])
#define OUT(i) k3_ptr(g, op->out[i])
#define P(i) (op->ip[i])

/* ---- int8_mixed kinds (KernelSpec reference_impl, unchanged) ------------- */
void kernel_quantize_rows_s8(const float *input, const float *col_inv, int8_t *output,
                             float *row_scale, int M, int K) {
    for (int m = 0; m < M; m++) {
        const float *x = input + (size_t)m * K;
        float amax = 0.0f;
        for (int k = 0; k < K; k++) {
            float v = fabsf(x[k] * col_inv[k]);
            if (v > amax) amax = v;
        }
        const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
        const float inv = 1.0f / s;
        row_scale[m] = s;
        for (int k = 0; k < K; k++) {
            const float v = x[k] * col_inv[k];
            float q = roundf(v * inv);
            if (!(q >= -127.0f)) q = -127.0f;
            if (q > 127.0f) q = 127.0f;
            output[(size_t)m * K + k] = (int8_t)q;
        }
    }
}

void kernel_linear_dq_s8(const int8_t *input, const float *row_scale, const int8_t *weight,
                         const float *bias, const float *w_scale, float *output, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)input[(size_t)m * K + k] * (int32_t)weight[(size_t)n * K + k];
            const double s = (double)row_scale[m] * (double)w_scale[n];
            float y = (float)((double)acc * s);
            if (bias) y = y + bias[n];
            output[(size_t)m * N + n] = y;
        }
    }
}

void kernel_linear3_dq_s8(const int8_t *input, const float *row_scale, const int8_t *weight,
                          const float *bias, const float *w_scale, float *out0, float *out1, float *out2,
                          int M, int K, int N0, int N1, int N2) {
    const int N = N0 + N1 + N2;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)input[(size_t)m * K + k] * (int32_t)weight[(size_t)n * K + k];
            const double s = (double)row_scale[m] * (double)w_scale[n];
            float y = (float)((double)acc * s);
            if (bias) y = y + bias[n];
            if (n < N0) out0[(size_t)m * N0 + n] = y;
            else if (n < N0 + N1) out1[(size_t)m * N1 + (n - N0)] = y;
            else out2[(size_t)m * N2 + (n - N0 - N1)] = y;
        }
    }
}

void kernel_linear_dq_add_s8(const int8_t *input, const float *row_scale, const int8_t *weight,
                             const float *bias, const float *w_scale, const float *addend, float *output,
                             int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)input[(size_t)m * K + k] * (int32_t)weight[(size_t)n * K + k];
            const double s = (double)row_scale[m] * (double)w_scale[n];
            float y = (float)((double)acc * s);
            if (bias) y = y + bias[n];
            output[(size_t)m * N + n] = addend[(size_t)m * N + n] + y;
        }
    }
}

void kernel_rope_half(const float *input, const float *cos_t, const float *sin_t, float *output,
                      int R, int H, int half) {
    for (int r = 0; r < R; r++)
        for (int h = 0; h < H; h++)
            for (int j = 0; j < half; j++) {
                const float *x = input + ((size_t)r * H + h) * 2 * half;
                float *y = output + ((size_t)r * H + h) * 2 * half;
                const float c = cos_t[(size_t)r * half + j], s = sin_t[(size_t)r * half + j];
                const float a = x[j] * c, b = x[half + j] * s, d = x[half + j] * c, e = x[j] * s;
                y[j] = a - b;
                y[half + j] = d + e;
            }
}

void kernel_silu_mul(const float *a, const float *b, float *output, int n) {
    for (int i = 0; i < n; i++) {
        const float x = a[i];
        const float s = x / (1.0f + expf(-x));
        output[i] = s * b[i];
    }
}

/* ---- fp32 kinds -------------------------------------------------------- */
static float ew1(int opc, float x, float p0, float p1)
{
    switch (opc) {
    case 0: return -x;
    case 1: return expf(x);
    case 2: return logf(x);
    case 3: return sqrtf(x);
    case 4: return 1.0f / sqrtf(x);
    case 5: return fabsf(x);
    case 6: return tanhf(x);
    case 7: return 1.0f / (1.0f + expf(-x));
    case 8: return x > 0 ? x : 0.0f;
    case 9: return sinf(x);
    case 10: return cosf(x);
    case 11: return erff(x);
    case 12: return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
    case 13: return 0.5f * x * (1.0f + tanhf(0.79788456080286536f * (x + 0.044715f * x * x * x)));
    case 14: return x / (1.0f + expf(-x));
    case 15: return x < p0 ? p0 : (x > p1 ? p1 : x);
    case 16: return 1.0f / x;
    case 17: return x * x;
    case 19: return x > 0 ? x : x * p0;
    case 20: return x > 0 ? x : p0 * (expf(x) - 1.0f);
    case 30: return x;
    default:
        fprintf(stderr, "ewise1: opcode %d not implemented\n", opc);
        exit(3);
    }
}

static float ew2(int opc, float a, float b, float al)
{
    switch (opc) {
    case 0: return al == 1.0f ? a + b : a + al * b;
    case 1: return al == 1.0f ? a - b : a - al * b;
    case 2: return a * b;
    case 3: return a / b;
    case 4: return a > b ? a : b;
    case 5: return a < b ? a : b;
    case 6: return powf(a, b);
    default:
        fprintf(stderr, "ewise2: opcode %d not implemented\n", opc);
        exit(3);
    }
}

static float ews(int opc, float x, float c)
{
    switch (opc) {
    case 0: return x + c;
    case 1: return x - c;
    case 2: return c - x;
    case 3: return x * c;
    case 4: return x / c;
    case 5: return c / x;
    case 6: return powf(x, c);
    case 8: return x < c ? c : x;
    case 9: return x > c ? c : x;
    default:
        fprintf(stderr, "ewise_scalar: opcode %d not implemented\n", opc);
        exit(3);
    }
}

static void strided_copy(const uint8_t *in, uint8_t *out, const int64_t *ip, size_t es)
{
    const int64_t off = ip[0];
    const int64_t *d = ip + 1, *s = ip + 7;
    size_t o = 0;
    for (int64_t i0 = 0; i0 < d[0]; i0++)
        for (int64_t i1 = 0; i1 < d[1]; i1++)
            for (int64_t i2 = 0; i2 < d[2]; i2++)
                for (int64_t i3 = 0; i3 < d[3]; i3++)
                    for (int64_t i4 = 0; i4 < d[4]; i4++) {
                        int64_t b = off + i0 * s[0] + i1 * s[1] + i2 * s[2] + i3 * s[3] + i4 * s[4];
                        if (s[5] == 1) {
                            memcpy(out + o * es, in + b * es, d[5] * es);
                            o += d[5];
                        } else {
                            for (int64_t i5 = 0; i5 < d[5]; i5++, o++)
                                memcpy(out + o * es, in + (b + i5 * s[5]) * es, es);
                        }
                    }
}

static void bmm_strided(const float *a, const float *b, float *c, const int64_t *p)
{
    const int64_t B0 = p[0], B1 = p[1], M = p[2], K = p[3], N = p[4];
    const int64_t ao = p[5], as0 = p[6], as1 = p[7], asm_ = p[8], ask = p[9];
    const int64_t bo = p[10], bs0 = p[11], bs1 = p[12], bsk = p[13], bsn = p[14];
    const int64_t co = p[15], cs0 = p[16], cs1 = p[17], csm = p[18], csn = p[19];
    memset(c, 0, (size_t)p[20] * sizeof(float));
    for (int64_t x = 0; x < B0; x++)
        for (int64_t y = 0; y < B1; y++)
            for (int64_t m = 0; m < M; m++)
                for (int64_t n = 0; n < N; n++) {
                    float acc = 0.0f;
                    for (int64_t k = 0; k < K; k++)
                        acc += a[ao + x * as0 + y * as1 + m * asm_ + k * ask] *
                               b[bo + x * bs0 + y * bs1 + k * bsk + n * bsn];
                    c[co + x * cs0 + y * cs1 + m * csm + n * csn] = acc;
                }
}

void k3_ref_run(k3_graph *g, k3_op *op)
{
    switch (op->kind) {
    case K_STRIDED_COPY:
        strided_copy(IN(0), OUT(0), op->ip, k3_esz(g->t[op->in[0]].dtype));
        break;
    case K_CATN: {
        const int64_t n_in = P(0), outer = P(1), inner = P(2);
        int64_t ctot = 0;
        for (int i = 0; i < n_in; i++) ctot += P(3 + i);
        const size_t es = k3_esz(g->t[op->out[0]].dtype);
        uint8_t *o = OUT(0);
        for (int64_t x = 0; x < outer; x++) {
            int64_t cpos = 0;
            for (int i = 0; i < n_in; i++) {
                const int64_t c = P(3 + i);
                memcpy(o + ((x * ctot + cpos) * inner) * es,
                       (const uint8_t *)IN(i) + (x * c * inner) * es, (size_t)(c * inner) * es);
                cpos += c;
            }
        }
        break;
    }
    case K_EWISE1: {
        const float *x = IN(0);
        float *y = OUT(0);
        for (int64_t i = 0; i < P(1); i++) y[i] = ew1((int)P(0), x[i], (float)op->fp[0], (float)op->fp[1]);
        break;
    }
    case K_EWISE2: {
        const float *a = IN(0), *b = IN(1);
        float *y = OUT(0);
        const int64_t *d = op->ip + 1, *sa = op->ip + 7, *sb = op->ip + 13;
        const float al = (float)op->fp[0];
        size_t o = 0;
        for (int64_t i0 = 0; i0 < d[0]; i0++)
            for (int64_t i1 = 0; i1 < d[1]; i1++)
                for (int64_t i2 = 0; i2 < d[2]; i2++)
                    for (int64_t i3 = 0; i3 < d[3]; i3++)
                        for (int64_t i4 = 0; i4 < d[4]; i4++) {
                            int64_t pa = i0 * sa[0] + i1 * sa[1] + i2 * sa[2] + i3 * sa[3] + i4 * sa[4];
                            int64_t pb = i0 * sb[0] + i1 * sb[1] + i2 * sb[2] + i3 * sb[3] + i4 * sb[4];
                            for (int64_t i5 = 0; i5 < d[5]; i5++)
                                y[o++] = ew2((int)P(0), a[pa + i5 * sa[5]], b[pb + i5 * sb[5]], al);
                        }
        break;
    }
    case K_EWISE_SCALAR: {
        const float *x = IN(0);
        float *y = OUT(0);
        for (int64_t i = 0; i < P(1); i++) y[i] = ews((int)P(0), x[i], (float)op->fp[0]);
        break;
    }
    case K_MUL_SCALAR: {
        const float *x = IN(0);
        float *y = OUT(0);
        const float c = (float)op->fp[0];
        for (int64_t i = 0; i < P(0); i++) y[i] = x[i] * c;
        break;
    }
    case K_ADD: {
        const float *a = IN(0), *b = IN(1);
        float *y = OUT(0);
        for (int64_t i = 0; i < P(0); i++) y[i] = a[i] + b[i];
        break;
    }
    case K_SILU: case K_RELU: case K_GELU: case K_GELU_EXACT: {
        const float *x = IN(0);
        float *y = OUT(0);
        const int opc = op->kind == K_SILU ? 14 : op->kind == K_RELU ? 8 : op->kind == K_GELU ? 12 : 13;
        for (int64_t i = 0; i < P(0); i++) y[i] = ew1(opc, x[i], 0, 0);
        break;
    }
    case K_SOFTPLUS: case K_TANH: {
        const float *x = IN(0);
        float *y = OUT(0);
        for (int64_t i = 0; i < P(0); i++)
            y[i] = op->kind == K_TANH ? tanhf(x[i]) : (x[i] > 20.0f ? x[i] : log1pf(expf(x[i])));
        break;
    }
    case K_IM2COL: {                 /* [IC, IH, IW] -> [OH*OW, IC*KH*KW] */
        const int64_t IC = P(0), IH = P(1), IW = P(2), OW = P(4), KH = P(5), KW = P(6), SH = P(7), SW = P(8),
                      PH = P(9), PW = P(10), DH = P(11), DW = P(12), KK = IC * KH * KW;
        const float *x = IN(0);
        float *y = OUT(0);
        for (int64_t r = 0; r < P(3) * P(4); r++) {
            const int64_t oh = r / OW, ow = r % OW;
            float *d = y + r * KK;
            for (int64_t ic = 0; ic < IC; ic++)
                for (int64_t kh = 0; kh < KH; kh++) {
                    const int64_t ih = oh * SH - PH + kh * DH;
                    for (int64_t kw = 0; kw < KW; kw++) {
                        const int64_t iw = ow * SW - PW + kw * DW;
                        *d++ = (ih >= 0 && ih < IH && iw >= 0 && iw < IW) ? x[(ic * IH + ih) * IW + iw] : 0.0f;
                    }
                }
        }
        break;
    }
    case K_SIGMOID: {
        const float *x = IN(0);
        float *y = OUT(0);
        for (int64_t i = 0; i < P(0); i++) y[i] = 1.0f / (1.0f + expf(-x[i]));
        break;
    }
    case K_SDPA: {
        const int64_t BH = P(0), S = P(1), D = P(2);
        const float *q = IN(0), *k = IN(1), *v = IN(2);
        float *y = OUT(0), *sc = malloc(sizeof(float) * S);
        const float scale = 1.0f / sqrtf((float)D);
        for (int64_t b = 0; b < BH; b++)
            for (int64_t i = 0; i < S; i++) {
                const float *qi = q + (b * S + i) * D;
                float mx = -INFINITY, sum = 0.0f;
                for (int64_t j = 0; j < S; j++) {
                    float a = 0.0f;
                    for (int64_t d = 0; d < D; d++) a += qi[d] * k[(b * S + j) * D + d];
                    sc[j] = a * scale;
                    mx = sc[j] > mx ? sc[j] : mx;
                }
                for (int64_t j = 0; j < S; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
                for (int64_t d = 0; d < D; d++) {
                    float a = 0.0f;
                    for (int64_t j = 0; j < S; j++) a += sc[j] * v[(b * S + j) * D + d];
                    y[(b * S + i) * D + d] = a / sum;
                }
            }
        free(sc);
        break;
    }
    case K_CONV_T2D: {                 /* weight [IC, OC/G, KH, KW]; output cropped to OH x OW */
        const int64_t N = P(0), IC = P(1), IH = P(2), IW = P(3), OC = P(4), OH = P(5), OW = P(6), KH = P(7),
                      KW = P(8), SH = P(9), SW = P(10), PH = P(11), PW = P(12), DH = P(13), DW = P(14), G = P(15);
        const float *x = IN(0), *w = op->wp[0], *b = op->wp[1];
        float *y = OUT(0);
        const int64_t icg = IC / G, ocg = OC / G;
        for (int64_t i = 0; i < N * OC * OH * OW; i++) y[i] = b ? b[(i / (OH * OW)) % OC] : 0.0f;
        for (int64_t n = 0; n < N; n++)
            for (int64_t ic = 0; ic < IC; ic++)
                for (int64_t ih = 0; ih < IH; ih++)
                    for (int64_t iw = 0; iw < IW; iw++) {
                        const float xv = x[((n * IC + ic) * IH + ih) * IW + iw];
                        const int64_t g = ic / icg;
                        for (int64_t o = 0; o < ocg; o++)
                            for (int64_t kh = 0; kh < KH; kh++) {
                                const int64_t oh = ih * SH - PH + kh * DH;
                                if (oh < 0 || oh >= OH) continue;
                                for (int64_t kw = 0; kw < KW; kw++) {
                                    const int64_t ow = iw * SW - PW + kw * DW;
                                    if (ow < 0 || ow >= OW) continue;
                                    y[((n * OC + g * ocg + o) * OH + oh) * OW + ow] +=
                                        xv * w[((ic * ocg + o) * KH + kh) * KW + kw];
                                }
                            }
                    }
        break;
    }
    case K_MUL: {
        const float *a = IN(0), *b = IN(1);
        float *y = OUT(0);
        for (int64_t i = 0; i < P(0); i++) y[i] = a[i] * b[i];
        break;
    }
    case K_SOFTMAX: {
        const int64_t M = P(0), K = P(1);
        const float *x = IN(0);
        float *y = OUT(0);
        for (int64_t m = 0; m < M; m++) {
            const float *r = x + m * K;
            float *o = y + m * K;
            float mx = -INFINITY;
            for (int64_t k = 0; k < K; k++) mx = r[k] > mx ? r[k] : mx;
            float s = 0.0f;
            for (int64_t k = 0; k < K; k++) { o[k] = expf(r[k] - mx); s += o[k]; }
            for (int64_t k = 0; k < K; k++) o[k] = o[k] / s;
        }
        break;
    }
    case K_LAYER_NORM: {
        const int64_t M = P(0), K = P(1);
        const float *x = IN(0), *ga = op->wp[0], *be = op->wp[1];
        float *y = OUT(0);
        const float eps = (float)op->fp[0];
        for (int64_t m = 0; m < M; m++) {
            const float *r = x + m * K;
            double s = 0;
            for (int64_t k = 0; k < K; k++) s += r[k];
            const float mean = (float)(s / K);
            double v = 0;
            for (int64_t k = 0; k < K; k++) v += (double)(r[k] - mean) * (r[k] - mean);
            const float rstd = 1.0f / sqrtf((float)(v / K) + eps);
            for (int64_t k = 0; k < K; k++) {
                float t = (r[k] - mean) * rstd;
                y[m * K + k] = (ga ? t * ga[k] : t) + (be ? be[k] : 0.0f);
            }
        }
        break;
    }
    case K_RMS_NORM: {
        const int64_t M = P(0), K = P(1);
        const float *x = IN(0);
        float *y = OUT(0);
        const float eps = (float)op->fp[0];
        for (int64_t m = 0; m < M; m++) {
            const float *r = x + m * K;
            float s = 0.0f;
            for (int64_t k = 0; k < K; k++) s += r[k] * r[k];
            const float d = sqrtf(s / (float)K + eps);
            for (int64_t k = 0; k < K; k++) y[m * K + k] = r[k] / d;
        }
        break;
    }
    case K_ROPE_HALF:
        kernel_rope_half(IN(0), IN(1), IN(2), OUT(0), (int)P(0), (int)P(1), (int)P(2));
        break;
    case K_SILU_MUL:
        kernel_silu_mul(IN(0), IN(1), OUT(0), (int)P(0));
        break;
    case K_LINEAR: {
        const int64_t M = P(0), K = P(1), N = P(2);
        const float *x = IN(0), *w = op->wp[0], *b = op->wp[1];
        float *y = OUT(0);
        for (int64_t m = 0; m < M; m++)
            for (int64_t n = 0; n < N; n++) {
                float acc = 0.0f;
                for (int64_t k = 0; k < K; k++) acc += x[m * K + k] * w[n * K + k];
                y[m * N + n] = b ? acc + b[n] : acc;
            }
        break;
    }
    case K_MATMUL: {
        const int64_t M = P(0), K = P(1), N = P(2);
        const float *a = IN(0), *b = IN(1);
        float *y = OUT(0);
        for (int64_t m = 0; m < M; m++)
            for (int64_t n = 0; n < N; n++) {
                float acc = 0.0f;
                for (int64_t k = 0; k < K; k++) acc += a[m * K + k] * b[k * N + n];
                y[m * N + n] = acc;
            }
        break;
    }
    case K_BMM_STRIDED:
        bmm_strided(IN(0), IN(1), OUT(0), op->ip);
        break;
    case K_CONV2D: {
        const int64_t N = P(0), IC = P(1), IH = P(2), IW = P(3), OC = P(4), OH = P(5), OW = P(6), KH = P(7),
                      KW = P(8), SH = P(9), SW = P(10), PH = P(11), PW = P(12), DH = P(13), DW = P(14);
        const float *x = IN(0), *w = op->wp[0], *b = op->wp[1];
        float *y = OUT(0);
        for (int64_t n = 0; n < N; n++)
            for (int64_t oc = 0; oc < OC; oc++)
                for (int64_t oh = 0; oh < OH; oh++)
                    for (int64_t ow = 0; ow < OW; ow++) {
                        float acc = 0.0f;
                        for (int64_t ic = 0; ic < IC; ic++)
                            for (int64_t kh = 0; kh < KH; kh++) {
                                const int64_t ih = oh * SH - PH + kh * DH;
                                if (ih < 0 || ih >= IH) continue;
                                for (int64_t kw = 0; kw < KW; kw++) {
                                    const int64_t iw = ow * SW - PW + kw * DW;
                                    if (iw < 0 || iw >= IW) continue;
                                    acc += x[((n * IC + ic) * IH + ih) * IW + iw] *
                                           w[((oc * IC + ic) * KH + kh) * KW + kw];
                                }
                            }
                        y[((n * OC + oc) * OH + oh) * OW + ow] = b ? acc + b[oc] : acc;
                    }
        break;
    }
    case K_INDEX_SELECT: {
        const int64_t outer = P(0), A = P(1), J = P(2), inner = P(3);
        const size_t es = k3_esz(g->t[op->in[0]].dtype);
        const uint8_t *x = IN(0);
        const int32_t *idx = IN(1);
        uint8_t *y = OUT(0);
        for (int64_t o = 0; o < outer; o++)
            for (int64_t j = 0; j < J; j++) {
                int64_t a = idx[j] < 0 ? idx[j] + A : idx[j];
                memcpy(y + ((o * J + j) * inner) * es, x + ((o * A + a) * inner) * es, (size_t)inner * es);
            }
        break;
    }
    case K_QUANTIZE_ROWS_S8:
        kernel_quantize_rows_s8(IN(0), IN(1), OUT(0), OUT(1), (int)P(0), (int)P(1));
        break;
    case K_LINEAR_DQ_S8:
        kernel_linear_dq_s8(IN(0), IN(1), op->wp[0], op->wp[1], op->wp[2], OUT(0), (int)P(0), (int)P(1), (int)P(2));
        break;
    case K_LINEAR3_DQ_S8:
        kernel_linear3_dq_s8(IN(0), IN(1), op->wp[0], op->wp[1], op->wp[2], OUT(0), OUT(1),
                             op->nout > 2 ? OUT(2) : NULL, (int)P(0), (int)P(1), (int)P(2), (int)P(3), (int)P(4));
        break;
    case K_LINEAR_DQ_ADD_S8:
        kernel_linear_dq_add_s8(IN(0), IN(1), op->wp[0], op->wp[1], op->wp[2], IN(2), OUT(0), (int)P(0),
                                (int)P(1), (int)P(2));
        break;
    case K_SOFTMAX_DIM: case K_MEAN_DIM: case K_SUM_DIM: {
        const int64_t outer = P(0), R = P(1), inner = P(2);
        const float *x = IN(0);
        float *y = OUT(0);
        for (int64_t o = 0; o < outer; o++)
            for (int64_t i = 0; i < inner; i++) {
                const float *r = x + o * R * inner + i;
                if (op->kind == K_SOFTMAX_DIM) {
                    float mx = -INFINITY, s = 0.0f;
                    for (int64_t k = 0; k < R; k++) mx = r[k * inner] > mx ? r[k * inner] : mx;
                    for (int64_t k = 0; k < R; k++) s += expf(r[k * inner] - mx);
                    for (int64_t k = 0; k < R; k++) y[o * R * inner + k * inner + i] = expf(r[k * inner] - mx) / s;
                } else {
                    float s = 0.0f;
                    for (int64_t k = 0; k < R; k++) s += r[k * inner];
                    y[o * inner + i] = op->kind == K_MEAN_DIM ? s / (float)R : s;
                }
            }
        break;
    }
    default:
        fprintf(stderr, "k3ir: no reference kernel for kind %s\n", k3_kind_names[op->kind]);
        exit(3);
    }
}
