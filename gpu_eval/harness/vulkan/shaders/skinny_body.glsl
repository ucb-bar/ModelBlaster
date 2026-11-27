// Body of the skinny GEMM (skinny.comp; skinny_h.comp defines WF16):
//   C[m, n] = alpha * sum_k A[m*sAm + k] * W[n, k] (+ bias[n]),  m < M <= 64
// the linear of a few dozen rows (SmolVLA's prefix and action expert) against
// a constant weight, where the tiled GEMM wastes most of its tile.
//
// W is PRE-PACKED (mb_vk_set_packed, walk_vulkan.pack_weight): for each block
// of 64 columns nb and each k-quad kq, the 64 columns' 4 values are contiguous,
//     P[((nb * K/4 + kq) * 64 + c) * 4 + i] = W[nb*64 + c, 4*kq + i]   (N zero-padded to 64),
// so a wave's vec4 load of one k-quad is one contiguous 1 KiB (fp16: 512 B):
// the weights stream at close to the full DRAM bandwidth.
//
// One invocation per output column (lane c of a 64-wide wave) and ALL M rows:
// acc[MT] (MT = M, specialization constant 0: one pipeline per row count)
// lives in registers; each weight is loaded once (one k-step ahead) and used M
// times. A's k-slice is the same for the whole wave (uniform address), so it
// is read with scalar loads (RDNA s_load: no VGPRs, no LDS) and enters the
// FMAs as a scalar operand. The loops are branch-free (the launcher makes the
// k range a whole number of KU-steps), so the compiler can issue loads early.
// The KW waves of a workgroup (constant 2) share its 64 columns and take
// alternate k-steps (KU k each, constant 1); their partial sums are added
// through LDS at the end. K is further split over gl_WorkGroupID.y (kchunk)
// when the columns alone do not fill the GPU; split partials go to
// C + s*M*N (raw: no alpha / bias) and splitk_reduce.comp sums them.
// WF16: the packed weights are fp16 (half the bytes), widened to fp32;
// activations and accumulation stay fp32.
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_ballot : require
#include "mb_common.glsl"
layout(constant_id = 0) const int MT = 32;
layout(constant_id = 1) const int KU = 32;
layout(constant_id = 2) const int KW = 2;
layout(local_size_x_id = 3) in;             // 64 * KW
layout(buffer_reference, std430, buffer_reference_align = 16) restrict readonly buffer RA4 { vec4 v[]; };
#ifdef WF16
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(buffer_reference, std430, buffer_reference_align = 8) restrict readonly buffer RW4 { f16vec4 v[]; };
#define WLOAD(p, q) vec4(RW4(p).v[(q) * 64])
const uint ES = 2u;
#else
layout(buffer_reference, std430, buffer_reference_align = 16) restrict readonly buffer RW4 { vec4 v[]; };
#define WLOAD(p, q) RW4(p).v[(q) * 64]
const uint ES = 4u;
#endif
layout(push_constant, scalar) uniform PC {
    F32 A; F32 W; F32 C; F32 bias;
    int M; int N; int K; int sAm; int sCm; int kchunk; int split; float alpha;
} pc;
const int RB = 16;                          // rows per LDS reduction round
shared float red[KW > 1 ? (KW - 1) * RB * 64 : 1];
void main() {
    const int N = pc.N, K = pc.K, split = pc.split;
    int c = int(gl_LocalInvocationID.x) & 63;
    int wv = subgroupBroadcastFirst(int(gl_LocalInvocationID.x) >> 6);   // this wave's k phase (uniform)
    int nb = int(gl_WorkGroupID.x);
    int n = nb * 64 + c;
    int s = int(gl_WorkGroupID.y);
    int kbeg = s * pc.kchunk, kend = min(K, kbeg + pc.kchunk);
    int nsteps = (kend - kbeg) / KU;
    int k0 = kbeg + wv * KU;
    uint64_t ap = uint64_t(pc.A) + uint64_t(k0) * 4ul;
    uint64_t rowb = uint64_t(pc.sAm) * 4ul;
    uint64_t wp = uint64_t(pc.W) + ((uint64_t(nb) * uint64_t(K >> 2) + uint64_t(k0 >> 2)) * 64ul + uint64_t(c)) * 4ul * ES;
    const uint64_t wstep = uint64_t(KW * KU / 4) * 64ul * 4ul * ES, astep = uint64_t(KW * KU) * 4ul;
    float acc[MT];
    [[unroll]] for (int m = 0; m < MT; m++) acc[m] = 0.0;
    vec4 w[KU / 4];
    int mine = nsteps > wv ? (nsteps - wv + KW - 1) / KW : 0;   // steps wv, wv+KW, ...
    if (mine > 0) {
        [[unroll]] for (int q = 0; q < KU / 4; q++) w[q] = WLOAD(wp, q);
    }
    for (int it = 0; it < mine; it++) {
        uint64_t wnp = it + 1 < mine ? wp + wstep : wp;
        vec4 wn[KU / 4];                    // next step's weights, in flight during this step's FMAs
        [[unroll]] for (int q = 0; q < KU / 4; q++) wn[q] = WLOAD(wnp, q);
        uint64_t arow = ap;
        [[unroll]] for (int m = 0; m < MT; m++) {
            [[unroll]] for (int q = 0; q < KU / 4; q++) {
                vec4 a = RA4(arow).v[q];
                acc[m] = fma(a.x, w[q].x, acc[m]);
                acc[m] = fma(a.y, w[q].y, acc[m]);
                acc[m] = fma(a.z, w[q].z, acc[m]);
                acc[m] = fma(a.w, w[q].w, acc[m]);
            }
            arow += rowb;
        }
        [[unroll]] for (int q = 0; q < KU / 4; q++) w[q] = wn[q];
        wp = wnp;
        ap += astep;
    }
    if (KW > 1) {                           // wave 0 adds the other waves' partials, RB rows at a time
        [[unroll]] for (int r0 = 0; r0 < MT; r0 += RB) {
            barrier();
            if (wv > 0) {
                [[unroll]] for (int i = 0; i < RB; i++)
                    if (r0 + i < MT) red[((wv - 1) * RB + i) * 64 + c] = acc[r0 + i];
            }
            barrier();
            if (wv == 0) {
                [[unroll]] for (int j = 0; j < KW - 1; j++)
                    [[unroll]] for (int i = 0; i < RB; i++)
                        if (r0 + i < MT) acc[r0 + i] += red[(j * RB + i) * 64 + c];
            }
        }
        if (wv != 0) return;
    }
    if (n >= N) return;
    if (split != 0) {
        int cb = s * MT * N + n;
        [[unroll]] for (int m = 0; m < MT; m++) pc.C.v[cb + m * N] = acc[m];
    } else {
        float bv = uint64_t(pc.bias) != 0ul ? pc.bias.v[n] : 0.0;
        float alpha = pc.alpha;
        int sCm = pc.sCm;
        [[unroll]] for (int m = 0; m < MT; m++) pc.C.v[m * sCm + n] = alpha * acc[m] + bv;
    }
}
