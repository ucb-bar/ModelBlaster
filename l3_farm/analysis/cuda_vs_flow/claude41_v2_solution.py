import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

POLL_ONCE = 0

cuda_src = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <stdint.h>

constexpr int H = 256;
constexpr int BATCH = 10;
constexpr int NB = 16;          // blocks per direction
constexpr int J = H / NB;       // hidden units per block (16)
constexpr int NW = 8;           // warps per block
constexpr int SN = 10;          // batch entries stored in smem
constexpr int GATE_FL = SN * 16;
constexpr int WARP_FL = 4 * GATE_FL;
constexpr int BUF_FL = NW * WARP_FL;

struct LayerParams {
    const __half* x;      // [T][B][IN]
    __half* y;            // [T][B][2H]
    const __half* h0;     // [2][B][H]
    const __half* wih[2];
    const __half* whh[2];
    const __half* bih[2];
    const __half* bhh[2];
    unsigned long long* hx;  // [2 dir][2 slot][B][H/2]
    int T;
    unsigned tagbase;
};

__device__ __forceinline__ unsigned long long ld_relaxed(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_relaxed(unsigned long long* p, unsigned long long v) {
    asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float sigm(float v) { return __fdividef(1.f, 1.f + __expf(-v)); }
__device__ __forceinline__ float tanh_fast(float v) {
    float r;
    asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(v));
    return r;
}
// swizzled smem index inside a gate block: row n (batch), col m (unit)
__device__ __forceinline__ int sidx(int n, int m) { return n * 16 + (m ^ ((n & 2) << 2)); }

template <int IN>
__global__ void __launch_bounds__(256, 1) gru_layer(LayerParams p) {
    constexpr int KI = IN / NW;
    constexpr int KSI = KI / 16;
    constexpr int KH = H / NW;
    constexpr int KSH = KH / 16;
    extern __shared__ float sm[];
    const int dir = blockIdx.x / NB, blk = blockIdx.x % NB;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, q = lane & 3;
    const int j0 = blk * J;
    const int T = p.T;
    const __half* wih = p.wih[dir];
    const __half* whh = p.whh[dir];

    uint32_t aI[3][KSI][4], aH[3][KSH][4];
#pragma unroll
    for (int m = 0; m < 3; m++) {
        const int row = m * H + j0 + g;
#pragma unroll
        for (int ks = 0; ks < KSI; ks++) {
            const int k = warp * KI + ks * 16 + q * 2;
            aI[m][ks][0] = *(const uint32_t*)(wih + (size_t)row * IN + k);
            aI[m][ks][1] = *(const uint32_t*)(wih + (size_t)(row + 8) * IN + k);
            aI[m][ks][2] = *(const uint32_t*)(wih + (size_t)row * IN + k + 8);
            aI[m][ks][3] = *(const uint32_t*)(wih + (size_t)(row + 8) * IN + k + 8);
        }
#pragma unroll
        for (int ks = 0; ks < KSH; ks++) {
            const int k = warp * KH + ks * 16 + q * 2;
            aH[m][ks][0] = *(const uint32_t*)(whh + (size_t)row * H + k);
            aH[m][ks][1] = *(const uint32_t*)(whh + (size_t)(row + 8) * H + k);
            aH[m][ks][2] = *(const uint32_t*)(whh + (size_t)row * H + k + 8);
            aH[m][ks][3] = *(const uint32_t*)(whh + (size_t)(row + 8) * H + k + 8);
        }
    }

    // elementwise role: one (batch, unit) per thread
    const bool ew = tid < BATCH * J;
    const int eb = tid / J, eu = tid % J;
    float br = 0.f, bz = 0.f, bin = 0.f, bhn = 0.f, hprev = 0.f;
    if (ew) {
        const __half* bi = p.bih[dir];
        const __half* bh = p.bhh[dir];
        const int j = j0 + eu;
        br = __half2float(bi[j]) + __half2float(bh[j]);
        bz = __half2float(bi[H + j]) + __half2float(bh[H + j]);
        bin = __half2float(bi[2 * H + j]);
        bhn = __half2float(bh[2 * H + j]);
        hprev = __half2float(p.h0[((size_t)dir * BATCH + eb) * H + j]);
    }

    uint32_t xc[KSI][2][2];
    auto load_x = [&](int t) {
#pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            const int n = nt * 8 + g;
#pragma unroll
            for (int ks = 0; ks < KSI; ks++) {
                if (n < BATCH) {
                    const __half* xp = p.x + ((size_t)t * BATCH + n) * IN + warp * KI + ks * 16 + q * 2;
                    xc[ks][nt][0] = *(const uint32_t*)xp;
                    xc[ks][nt][1] = *(const uint32_t*)(xp + 8);
                } else {
                    xc[ks][nt][0] = 0; xc[ks][nt][1] = 0;
                }
            }
        }
    };
    load_x(dir ? T - 1 : 0);

    for (int s = 0; s < T; s++) {
        const int t = dir ? T - 1 - s : s;
        float aR[2][4], aZ[2][4], aN[2][4], aHN[2][4];
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) { aR[nt][i] = 0.f; aZ[nt][i] = 0.f; aN[nt][i] = 0.f; aHN[nt][i] = 0.f; }
#pragma unroll
        for (int ks = 0; ks < KSI; ks++)
#pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                mma16816(aR[nt], aI[0][ks], xc[ks][nt][0], xc[ks][nt][1]);
                mma16816(aZ[nt], aI[1][ks], xc[ks][nt][0], xc[ks][nt][1]);
                mma16816(aN[nt], aI[2][ks], xc[ks][nt][0], xc[ks][nt][1]);
            }
        if (s + 1 < T) load_x(dir ? T - 2 - s : s + 1);

        uint32_t hb[KSH][2][2];
        if (s == 0) {
#pragma unroll
            for (int ks = 0; ks < KSH; ks++)
#pragma unroll
                for (int nt = 0; nt < 2; nt++) {
                    const int n = nt * 8 + g;
                    if (n < BATCH) {
                        const __half* hp = p.h0 + ((size_t)dir * BATCH + n) * H + warp * KH + ks * 16 + q * 2;
                        hb[ks][nt][0] = *(const uint32_t*)hp;
                        hb[ks][nt][1] = *(const uint32_t*)(hp + 8);
                    } else { hb[ks][nt][0] = 0; hb[ks][nt][1] = 0; }
                }
        } else {
            const unsigned long long* hp = p.hx + (size_t)((dir * 2 + ((s - 1) & 1)) * BATCH) * (H / 2);
            const unsigned want = p.tagbase + s;
            unsigned long long v[KSH][2][2];
            bool ok;
            do {
                ok = true;
#pragma unroll
                for (int ks = 0; ks < KSH; ks++)
#pragma unroll
                    for (int nt = 0; nt < 2; nt++)
#pragma unroll
                        for (int hh = 0; hh < 2; hh++) {
                            const int n = nt * 8 + g;
                            if (n < BATCH) {
                                const int k = warp * KH + ks * 16 + q * 2 + hh * 8;
                                v[ks][nt][hh] = ld_relaxed(hp + n * (H / 2) + (k >> 1));
                            } else {
                                v[ks][nt][hh] = ((unsigned long long)want) << 32;
                            }
                        }
#pragma unroll
                for (int ks = 0; ks < KSH; ks++)
#pragma unroll
                    for (int nt = 0; nt < 2; nt++)
#pragma unroll
                        for (int hh = 0; hh < 2; hh++)
                            ok &= ((unsigned)(v[ks][nt][hh] >> 32) == want);
            } while (!POLL_ONCE && !ok);
#pragma unroll
            for (int ks = 0; ks < KSH; ks++)
#pragma unroll
                for (int nt = 0; nt < 2; nt++)
#pragma unroll
                    for (int hh = 0; hh < 2; hh++) hb[ks][nt][hh] = (uint32_t)v[ks][nt][hh];
        }
#pragma unroll
        for (int ks = 0; ks < KSH; ks++)
#pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                mma16816(aR[nt], aH[0][ks], hb[ks][nt][0], hb[ks][nt][1]);
                mma16816(aZ[nt], aH[1][ks], hb[ks][nt][0], hb[ks][nt][1]);
                mma16816(aHN[nt], aH[2][ks], hb[ks][nt][0], hb[ks][nt][1]);
            }

        // partials to smem: [warp][gate][n][16 swizzled]
        float* sb = sm + (s & 1) * BUF_FL + warp * WARP_FL;
#pragma unroll
        for (int nt = 0; nt < 2; nt++) {
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int n = nt * 8 + q * 2 + (c & 1);
                const int m = g + (c >> 1) * 8;
                if (n < SN) {
                    const int o = sidx(n, m);
                    sb[0 * GATE_FL + o] = aR[nt][c];
                    sb[1 * GATE_FL + o] = aZ[nt][c];
                    sb[2 * GATE_FL + o] = aN[nt][c];
                    sb[3 * GATE_FL + o] = aHN[nt][c];
                }
            }
        }
        __syncthreads();
        if (ew) {
            const float* rb = sm + (s & 1) * BUF_FL + sidx(eb, eu);
            float gr = 0.f, gz = 0.f, gn = 0.f, ghn = 0.f;
#pragma unroll
            for (int w = 0; w < NW; w++) {
                const float* wb = rb + w * WARP_FL;
                gr += wb[0];
                gz += wb[GATE_FL];
                gn += wb[2 * GATE_FL];
                ghn += wb[3 * GATE_FL];
            }
            const float r = sigm(gr + br);
            const float z = sigm(gz + bz);
            const float nn = tanh_fast(gn + bin + r * (ghn + bhn));
            const float hn = (1.f - z) * nn + z * hprev;
            hprev = hn;
            const float hn1 = __shfl_down_sync(0xffffffffu, hn, 1);
            if ((eu & 1) == 0) {
                __half2 h2 = __floats2half2_rn(hn, hn1);
                const uint32_t bits = *(uint32_t*)&h2;
                const unsigned long long word = (((unsigned long long)(p.tagbase + s + 1)) << 32) | bits;
                st_relaxed(p.hx + (size_t)((dir * 2 + (s & 1)) * BATCH + eb) * (H / 2) + ((j0 + eu) >> 1), word);
                *(__half2*)(p.y + ((size_t)t * BATCH + eb) * (2 * H) + dir * H + j0 + eu) = h2;
            }
        }
    }
}

torch::Tensor gru_forward(torch::Tensor x, torch::Tensor h0, std::vector<torch::Tensor> w) {
    const int T = x.size(0);
    const int L = w.size() / 8;
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto opts = x.options();
    auto out = torch::empty({T, BATCH, 2 * H}, opts);
    torch::Tensor bufs[2] = {torch::empty({T, BATCH, 2 * H}, opts), torch::empty({T, BATCH, 2 * H}, opts)};
    auto hx = torch::empty({2 * 2 * BATCH * (H / 2)}, opts.dtype(torch::kInt64));
    cudaMemsetAsync(hx.data_ptr(), 0, hx.numel() * 8, stream);
    const size_t smem = 2 * BUF_FL * sizeof(float);
    static bool init = false;
    if (!init) {
        cudaFuncSetAttribute(gru_layer<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        cudaFuncSetAttribute(gru_layer<512>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        init = true;
    }
    const __half* in = (const __half*)x.data_ptr();
    const int in_dim = x.size(2);
    for (int l = 0; l < L; l++) {
        LayerParams p;
        p.x = in;
        __half* yo = (l == L - 1) ? (__half*)out.data_ptr() : (__half*)bufs[l & 1].data_ptr();
        p.y = yo;
        p.h0 = (const __half*)h0.data_ptr() + (size_t)l * 2 * BATCH * H;
        for (int d = 0; d < 2; d++) {
            p.wih[d] = (const __half*)w[l * 8 + d * 4 + 0].data_ptr();
            p.whh[d] = (const __half*)w[l * 8 + d * 4 + 1].data_ptr();
            p.bih[d] = (const __half*)w[l * 8 + d * 4 + 2].data_ptr();
            p.bhh[d] = (const __half*)w[l * 8 + d * 4 + 3].data_ptr();
        }
        p.hx = (unsigned long long*)hx.data_ptr();
        p.T = T;
        p.tagbase = (unsigned)l * 1024u;
        const int idim = (l == 0) ? in_dim : 2 * H;
        if (idim == 128)
            gru_layer<128><<<2 * NB, 256, smem, stream>>>(p);
        else
            gru_layer<512><<<2 * NB, 256, smem, stream>>>(p);
        in = yo;
    }
    return out;
}
""".replace("POLL_ONCE", str(POLL_ONCE))

cpp_src = "torch::Tensor gru_forward(torch::Tensor x, torch::Tensor h0, std::vector<torch::Tensor> w);"

_ext = load_inline(
    name=f"gru_bidir_ext_v2_{POLL_ONCE}",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["gru_forward"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=True)
        self.num_layers = num_layers

    def forward(self, x, h0):
        ws = []
        for l in range(self.num_layers):
            for sfx in ("", "_reverse"):
                ws.append(getattr(self.gru, f"weight_ih_l{l}{sfx}"))
                ws.append(getattr(self.gru, f"weight_hh_l{l}{sfx}"))
                ws.append(getattr(self.gru, f"bias_ih_l{l}{sfx}"))
                ws.append(getattr(self.gru, f"bias_hh_l{l}{sfx}"))
        return _ext.gru_forward(x, h0, ws)
