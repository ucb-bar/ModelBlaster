import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

REPL = 8     # replicas of the h exchange buffer
NGW = 24     # GEMM worker blocks (0 = all remaining SMs)

cuda_src = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <stdint.h>

constexpr int H = 256;
constexpr int BATCH = 10;
constexpr int G3 = 3 * H;          // 768 gate rows per direction
constexpr int GN = 2 * G3;         // 1536 gate columns in G
constexpr int J = 8;               // hidden units per recurrence block
constexpr int NB = H / J;          // recurrence blocks per direction
constexpr int NREC = 2 * NB;       // recurrence blocks
constexpr int NWR = 8;             // warps per recurrence block
constexpr int REPL = REPL_VAL;
constexpr int XW = (H / 16) * 96;  // words per h exchange buffer (fragment-major, tagged)
constexpr int HXL = 2 * 2 * REPL * XW;   // words per layer-parity set
constexpr int PB = J * 3 * NWR + 4;  // smem floats per batch row of partials
constexpr int MAXL = 6;
constexpr int BM = 128, BN = 128, BK = 32, STAGES = 3, LDS = BK + 8;
constexpr int NT_PER_DIR = G3 / BN;  // 6
constexpr int SYNC_READY = 8;
constexpr int SYNC_DONE = 1024;

struct Params {
    const __half* X[MAXL];
    int K[MAXL];
    const __half* wih[MAXL][2];
    const __half* whh[MAXL][2];
    const __half* bih[MAXL][2];
    const __half* bhh[MAXL][2];
    __half* Y[MAXL];
    const __half* h0;     // [L*2][B][H]
    __half* G;            // [T*B][GN]
    unsigned long long* hx;
    int* sync;
    int L, T, MT;
};

// ---------------------------------------------------------------- helpers
__device__ __forceinline__ unsigned long long ld_relaxed(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_relaxed(unsigned long long* p, unsigned long long v) {
    asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ int ld_acquire(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint32_t ld_cg_h2(const __half* p) {
    uint32_t v;
    asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ float2 u2f2(uint32_t v) {
    __half2 h;
    asm("mov.b32 %0, %1;" : "=r"(*(uint32_t*)&h) : "r"(v));
    return __half22float2(h);
}
__device__ __forceinline__ void bar_rec() { asm volatile("bar.sync 1, %0;" :: "n"(NWR * 32) : "memory"); }
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float tanh_fast(float v) {
    float r;
    asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(v));
    return r;
}
__device__ __forceinline__ float sigm(float v) { return fmaf(0.5f, tanh_fast(0.5f * v), 0.5f); }
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* smem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

// ---------------------------------------------------------------- input projection tile
// G[m][n] = sum_k X[m][k] * Wcat[n][k] + bias[n];  Wcat = [W_ih_fwd; W_ih_bwd]
__device__ __forceinline__ void gemm_tile(const Params& p, int l, int mt, int dir, int nt, unsigned char* smraw) {
    __half* As = (__half*)smraw;
    __half* Bs = As + STAGES * BM * LDS;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wrow0 = nt * BN;
    const int col0 = dir * G3 + wrow0;
    const int K = p.K[l];
    const __half* Ag = p.X[l] + (size_t)mt * BM * K;
    const __half* Bg = p.wih[l][dir] + (size_t)wrow0 * K;
    const int KT = K / BK;

    auto load_stage = [&](int st, int kt) {
        __half* a = As + st * BM * LDS;
        __half* b = Bs + st * BN * LDS;
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int c = tid + i * 256;
            const int r = c >> 2, kc = (c & 3) * 8;
            cp_async16(a + r * LDS + kc, Ag + (size_t)r * K + kt * BK + kc);
            cp_async16(b + r * LDS + kc, Bg + (size_t)r * K + kt * BK + kc);
        }
    };

    const int wm = warp >> 2, wn = warp & 3;
    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[i][j][c] = 0.f;

#pragma unroll
    for (int st = 0; st < STAGES - 1; st++) {
        if (st < KT) load_stage(st, st);
        cp_commit();
    }
    for (int kt = 0; kt < KT; kt++) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        if (kt + STAGES - 1 < KT) load_stage((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
        cp_commit();
        const __half* a = As + (kt % STAGES) * BM * LDS;
        const __half* b = Bs + (kt % STAGES) * BN * LDS;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t af[4][4], bf[4][2];
#pragma unroll
            for (int mi = 0; mi < 4; mi++)
                ldsm_x4(af[mi], a + (wm * 64 + mi * 16 + (lane & 15)) * LDS + kk + (lane >> 4) * 8);
#pragma unroll
            for (int nj = 0; nj < 2; nj++) {
                uint32_t r[4];
                ldsm_x4(r, b + (wn * 32 + nj * 16 + (lane & 7) + ((lane >> 4) << 3)) * LDS + kk + ((lane >> 3) & 1) * 8);
                bf[nj * 2][0] = r[0]; bf[nj * 2][1] = r[1];
                bf[nj * 2 + 1][0] = r[2]; bf[nj * 2 + 1][1] = r[3];
            }
#pragma unroll
            for (int mi = 0; mi < 4; mi++)
#pragma unroll
                for (int ni = 0; ni < 4; ni++) mma16816(acc[mi][ni], af[mi], bf[ni][0], bf[ni][1]);
        }
    }
    cp_wait<0>();

    const int g = lane >> 2, q = lane & 3;
    const __half* bi = p.bih[l][dir];
    const __half* bh = p.bhh[l][dir];
#pragma unroll
    for (int ni = 0; ni < 4; ni++) {
        const int cl = wn * 32 + ni * 8 + q * 2;      // column inside tile
        const int wr = wrow0 + cl;                    // gate row inside direction
        float b0 = __half2float(bi[wr]), b1 = __half2float(bi[wr + 1]);
        if (wr < 2 * H) { b0 += __half2float(bh[wr]); b1 += __half2float(bh[wr + 1]); }
#pragma unroll
        for (int mi = 0; mi < 4; mi++) {
            const int row = mt * BM + wm * 64 + mi * 16 + g;
            __half* gp = p.G + (size_t)row * GN + col0 + cl;
            *(__half2*)gp = __floats2half2_rn(acc[mi][ni][0] + b0, acc[mi][ni][1] + b1);
            *(__half2*)(gp + (size_t)8 * GN) = __floats2half2_rn(acc[mi][ni][2] + b0, acc[mi][ni][3] + b1);
        }
    }
}

__device__ void gemm_worker(const Params& p, unsigned char* smraw) {
    __shared__ int s_tile;
    const int tid = threadIdx.x;
    const int ntiles = p.MT * 2 * NT_PER_DIR;
    for (int l = 0; l < p.L; l++) {
        if (l > 0) {
            if (tid == 0) { while (ld_acquire(p.sync + SYNC_DONE + l - 1) < NREC) { } }
            __syncthreads();
        }
        while (true) {
            if (tid == 0) s_tile = atomicAdd(p.sync + l, 1);
            __syncthreads();
            const int idx = s_tile;
            __syncthreads();
            if (idx >= ntiles) break;
            const int pair = idx / (2 * NT_PER_DIR), w = idx % (2 * NT_PER_DIR);
            const int dir = w / NT_PER_DIR, nt = w % NT_PER_DIR;
            const int mt = dir ? p.MT - 1 - pair : pair;
            gemm_tile(p, l, mt, dir, nt, smraw);
            __syncthreads();
            if (tid == 0) {
                __threadfence();
                atomicAdd(p.sync + SYNC_READY + (l * 2 + dir) * p.MT + mt, 1);
            }
        }
    }
}

// ---------------------------------------------------------------- recurrence
__device__ void rec_block(const Params& p, float* sm /* [2][BATCH*PB] */) {
    constexpr int KW = H / NWR;
    constexpr int KS = KW / 16;
    const int dir = blockIdx.x / NB, blk = blockIdx.x % NB;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, q = lane & 3;
    const int j0 = blk * J;
    const int T = p.T;
    const bool hi = g < BATCH - 8;   // batch g+8 valid

    // elementwise role: thread handles (batch eb, units eu, eu+1)
    const bool ew = tid < BATCH * (J / 2);
    const int eb = tid / (J / 2), eu = (tid % (J / 2)) * 2;
    const int ej = j0 + eu;
    int dst_off;
    {
        const int w = ej / KW, kk = ej % KW, ks = kk / 16, r = kk % 16;
        const int hi8 = r >= 8, qq = (r & 7) >> 1;
        const int grp = eb < 8 ? hi8 : 2;
        const int li = eb < 8 ? eb * 4 + qq : hi8 * 8 + (eb - 8) * 4 + qq;
        dst_off = (w * KS + ks) * 96 + grp * 32 + li;
    }
    const int* ready = p.sync + SYNC_READY;

    for (int l = 0; l < p.L; l++) {
        const __half* whh = p.whh[l][dir];
        uint32_t bw[3][KS][2];
#pragma unroll
        for (int nt = 0; nt < 3; nt++) {
            const int wrow = nt * H + j0 + g;
#pragma unroll
            for (int ks = 0; ks < KS; ks++) {
                const int k = warp * KW + ks * 16 + q * 2;
                bw[nt][ks][0] = *(const uint32_t*)(whh + (size_t)wrow * H + k);
                bw[nt][ks][1] = *(const uint32_t*)(whh + (size_t)wrow * H + k + 8);
            }
        }
        const unsigned tagbase = (unsigned)l * 1024u;
        unsigned long long* hx = p.hx + (size_t)(l & 1) * HXL;
        const __half* h0 = p.h0 + (size_t)l * 2 * BATCH * H;
        __half* Y = p.Y[l];
        const int* rdy = ready + (l * 2 + dir) * p.MT;
        unsigned long long rmask = 0;
        auto ensure_t = [&](int t) {
            const int m0 = (t * BATCH) / BM, m1 = (t * BATCH + BATCH - 1) / BM;
            if (!((rmask >> m0) & 1ull)) { while (ld_acquire(rdy + m0) < NT_PER_DIR) { } rmask |= 1ull << m0; }
            if (!((rmask >> m1) & 1ull)) { while (ld_acquire(rdy + m1) < NT_PER_DIR) { } rmask |= 1ull << m1; }
        };

        float bhn0 = 0.f, bhn1 = 0.f, hp0 = 0.f, hp1 = 0.f;
        uint32_t gr2 = 0, gz2 = 0, gn2 = 0;
        const __half* Gp = p.G + (size_t)eb * GN + dir * G3 + ej;
        if (ew) {
            const __half* bh = p.bhh[l][dir];
            bhn0 = __half2float(bh[2 * H + ej]);
            bhn1 = __half2float(bh[2 * H + ej + 1]);
            hp0 = __half2float(h0[((size_t)dir * BATCH + eb) * H + ej]);
            hp1 = __half2float(h0[((size_t)dir * BATCH + eb) * H + ej + 1]);
            const int t0 = dir ? T - 1 : 0;
            ensure_t(t0);
            const __half* gp = Gp + (size_t)t0 * BATCH * GN;
            gr2 = ld_cg_h2(gp); gz2 = ld_cg_h2(gp + H); gn2 = ld_cg_h2(gp + 2 * H);
        }

        for (int s = 0; s < T; s++) {
            const int t = dir ? T - 1 - s : s;
            uint32_t ha[KS][4];
            if (s == 0) {
#pragma unroll
                for (int ks = 0; ks < KS; ks++) {
                    const __half* hp = h0 + (size_t)dir * BATCH * H + warp * KW + ks * 16 + q * 2;
                    ha[ks][0] = *(const uint32_t*)(hp + g * H);
                    ha[ks][2] = *(const uint32_t*)(hp + g * H + 8);
                    ha[ks][1] = hi ? *(const uint32_t*)(hp + (g + 8) * H) : 0u;
                    ha[ks][3] = hi ? *(const uint32_t*)(hp + (g + 8) * H + 8) : 0u;
                }
            } else {
                const unsigned long long* hp = hx + (size_t)((dir * 2 + ((s - 1) & 1)) * REPL + (blk % REPL)) * XW
                                             + warp * KS * 96 + lane;
                const unsigned want = tagbase + s;
                const unsigned long long dflt = ((unsigned long long)want) << 32;
                unsigned long long v[KS][4];
                bool ok;
                do {
#pragma unroll
                    for (int ks = 0; ks < KS; ks++) {
                        v[ks][0] = ld_relaxed(hp + ks * 96);
                        v[ks][2] = ld_relaxed(hp + ks * 96 + 32);
                        v[ks][1] = hi ? ld_relaxed(hp + ks * 96 + 64) : dflt;
                        v[ks][3] = hi ? ld_relaxed(hp + ks * 96 + 72) : dflt;
                    }
                    ok = true;
#pragma unroll
                    for (int ks = 0; ks < KS; ks++)
#pragma unroll
                        for (int i = 0; i < 4; i++) ok &= ((unsigned)(v[ks][i] >> 32) == want);
                } while (!ok);
#pragma unroll
                for (int ks = 0; ks < KS; ks++)
#pragma unroll
                    for (int i = 0; i < 4; i++) ha[ks][i] = (uint32_t)v[ks][i];
            }
            float acc[3][4];
#pragma unroll
            for (int nt = 0; nt < 3; nt++)
#pragma unroll
                for (int i = 0; i < 4; i++) acc[nt][i] = 0.f;
#pragma unroll
            for (int ks = 0; ks < KS; ks++)
#pragma unroll
                for (int nt = 0; nt < 3; nt++) mma16816(acc[nt], ha[ks], bw[nt][ks][0], bw[nt][ks][1]);

            // partials: sm[b][u][gate][warp]
            float* sb = sm + (s & 1) * (BATCH * PB);
#pragma unroll
            for (int nt = 0; nt < 3; nt++) {
                float* o = sb + g * PB + (q * 2 * 3 + nt) * NWR + warp;
                o[0] = acc[nt][0];
                o[3 * NWR] = acc[nt][1];
                if (hi) {
                    o[8 * PB] = acc[nt][2];
                    o[8 * PB + 3 * NWR] = acc[nt][3];
                }
            }
            bar_rec();
            if (ew) {
                const float* rb = sb + eb * PB + eu * 3 * NWR;
                float gs[2][3];
#pragma unroll
                for (int u = 0; u < 2; u++)
#pragma unroll
                    for (int gt = 0; gt < 3; gt++) {
                        float a = 0.f;
#pragma unroll
                        for (int w4 = 0; w4 < NWR; w4 += 4) {
                            const float4 f = *(const float4*)(rb + (u * 3 + gt) * NWR + w4);
                            a += (f.x + f.y) + (f.z + f.w);
                        }
                        gs[u][gt] = a;
                    }
                const float2 gr = u2f2(gr2), gz = u2f2(gz2), gn = u2f2(gn2);
                const float r0 = sigm(gr.x + gs[0][0]), r1 = sigm(gr.y + gs[1][0]);
                const float z0 = sigm(gz.x + gs[0][1]), z1 = sigm(gz.y + gs[1][1]);
                const float n0 = tanh_fast(gn.x + r0 * (gs[0][2] + bhn0));
                const float n1 = tanh_fast(gn.y + r1 * (gs[1][2] + bhn1));
                hp0 = n0 + z0 * (hp0 - n0);
                hp1 = n1 + z1 * (hp1 - n1);
                __half2 h2 = __floats2half2_rn(hp0, hp1);
                const uint32_t bits = *(uint32_t*)&h2;
                const unsigned long long word = (((unsigned long long)(tagbase + s + 1)) << 32) | bits;
                unsigned long long* dst = hx + (size_t)((dir * 2 + (s & 1)) * REPL) * XW + dst_off;
#pragma unroll
                for (int r = 0; r < REPL; r++) st_relaxed(dst + (size_t)r * XW, word);
                *(__half2*)(Y + ((size_t)t * BATCH + eb) * (2 * H) + dir * H + ej) = h2;
                if (s + 1 < T) {
                    const int tn = dir ? T - 2 - s : s + 1;
                    ensure_t(tn);
                    const __half* gp = Gp + (size_t)tn * BATCH * GN;
                    gr2 = ld_cg_h2(gp); gz2 = ld_cg_h2(gp + H); gn2 = ld_cg_h2(gp + 2 * H);
                }
            }
        }
        bar_rec();
        if (tid == 0) {
            __threadfence();
            atomicAdd(p.sync + SYNC_DONE + l, 1);
        }
    }
}

__global__ void __launch_bounds__(256, 1) gru_all(Params p) {
    extern __shared__ __align__(16) unsigned char dsm[];
    if (blockIdx.x < NREC) {
        if (threadIdx.x >= NWR * 32) return;
        rec_block(p, (float*)dsm);
    } else {
        gemm_worker(p, dsm);
    }
}

torch::Tensor gru_forward(torch::Tensor x, torch::Tensor h0, std::vector<torch::Tensor> w) {
    const int T = x.size(0);
    const int L = w.size() / 8;
    TORCH_CHECK(L <= MAXL && (T * BATCH) % BM == 0 && (T * BATCH) / BM <= 64);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto opts = x.options();
    auto out = torch::empty({T, BATCH, 2 * H}, opts);
    torch::Tensor bufs[2] = {torch::empty({T, BATCH, 2 * H}, opts), torch::empty({T, BATCH, 2 * H}, opts)};
    auto G = torch::empty({(long)T * BATCH * GN}, opts);
    auto hx = torch::empty({2 * HXL}, opts.dtype(torch::kInt64));
    auto sync = torch::empty({2048}, opts.dtype(torch::kInt32));
    cudaMemsetAsync(hx.data_ptr(), 0, hx.numel() * 8, stream);
    cudaMemsetAsync(sync.data_ptr(), 0, sync.numel() * 4, stream);
    const int smem = STAGES * (BM + BN) * LDS * sizeof(__half);
    static int nsm = 0;
    if (!nsm) {
        int dev; cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        cudaFuncSetAttribute(gru_all, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }
    Params p;
    const __half* in = (const __half*)x.data_ptr();
    for (int l = 0; l < L; l++) {
        p.X[l] = in;
        p.K[l] = (l == 0) ? (int)x.size(2) : 2 * H;
        for (int d = 0; d < 2; d++) {
            p.wih[l][d] = (const __half*)w[l * 8 + d * 4 + 0].data_ptr();
            p.whh[l][d] = (const __half*)w[l * 8 + d * 4 + 1].data_ptr();
            p.bih[l][d] = (const __half*)w[l * 8 + d * 4 + 2].data_ptr();
            p.bhh[l][d] = (const __half*)w[l * 8 + d * 4 + 3].data_ptr();
        }
        __half* yo = (l == L - 1) ? (__half*)out.data_ptr() : (__half*)bufs[l & 1].data_ptr();
        p.Y[l] = yo;
        in = yo;
    }
    p.h0 = (const __half*)h0.data_ptr();
    p.G = (__half*)G.data_ptr();
    p.hx = (unsigned long long*)hx.data_ptr();
    p.sync = (int*)sync.data_ptr();
    p.L = L; p.T = T; p.MT = T * BATCH / BM;
    int nblk = nsm > NREC + 1 ? nsm : NREC + 1;
    if (NGW > 0 && NREC + NGW < nblk) nblk = NREC + NGW;
    gru_all<<<nblk, 256, smem, stream>>>(p);
    return out;
}
""".replace("REPL_VAL", str(REPL)).replace("NGW", str(NGW))

cpp_src = "torch::Tensor gru_forward(torch::Tensor x, torch::Tensor h0, std::vector<torch::Tensor> w);"

_ext = load_inline(
    name=f"gru_bidir_sol5_{REPL}_{NGW}_w8b",
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
