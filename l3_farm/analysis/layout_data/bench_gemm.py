import json, statistics as st, math
import torch, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = False
def timeit(fn, iters=50, warm=10):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(warm): g.replay()
    torch.cuda.synchronize(); ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    return round(st.median(ts), 1)
h = dict(device="cuda", dtype=torch.half); R = {}
# 1. linear: W[N,K] (torch layout, x@W^T) vs pre-transposed Wt[K,N] vs padded
lin = {}
for (M, K, N) in [(1024, 8192, 1024), (1024, 1024, 1024), (1024, 1024, 8192), (65536, 768, 2304), (65536, 768, 768), (65536, 768, 3072), (65536, 3072, 768), (10, 25088, 4096), (204800, 512, 32), (204800, 512, 48)]:
    x = torch.randn(M, K, **h); W = torch.randn(N, K, **h) / 30; Wt = W.t().contiguous()
    lin[f"{M}x{K}x{N}"] = dict(W_NK=timeit(lambda: F.linear(x, W)), Wt_KN=timeit(lambda: x @ Wt),
                               floor_bw_us=round((M*K + K*N + M*N) * 2 / 864e3, 1), floor_tc_us=round(2*M*N*K / 181e6, 1))
R["linear"] = lin
# 2. NetVlad bmm: (B,32,100)@(B,100,512)
B = 2048; a = torch.randn(B, 100, 32, **h); x = torch.randn(B, 100, 512, **h)
aT = a.transpose(1, 2).contiguous()
ap = torch.zeros(B, 104, 32, **h); ap[:, :100] = a; xp = torch.zeros(B, 104, 512, **h); xp[:, :100] = x; apT = ap.transpose(1, 2).contiguous()
R["netvlad_bmm"] = dict(transpose_copy=timeit(lambda: a.transpose(1, 2).contiguous()), bmm_contigA=timeit(lambda: torch.bmm(aT, x)),
    bmm_viewA_transA=timeit(lambda: torch.bmm(a.transpose(1, 2), x)), bmm_padK104=timeit(lambda: torch.bmm(apT, xp)),
    out_transpose_copy=timeit(lambda: torch.bmm(aT, x).transpose(1, 2).contiguous()),
    floor_bw_us=round((B*100*512 + B*100*32 + B*32*512) * 2 / 864e3, 1))
# 3. MinGPT attention core, B=128 T=512 C=768 nh=8 hs=96
Bt, T, C, nh = 128, 512, 768, 8; hs = C // nh
qkv = torch.randn(Bt, T, 3 * C, **h); mask = torch.tril(torch.ones(T, T, device="cuda")).view(1, 1, T, T)
def split():
    q, k, v = qkv.split(C, dim=2)
    return [t.view(Bt, T, nh, hs).transpose(1, 2) for t in (q, k, v)]
def ours_like():  # materialize q,k,v contiguous then bmm (what our flow does, minus the getitem copies)
    q, k, v = [t.contiguous() for t in split()]
    att = (q @ k.transpose(-2, -1).contiguous()) * (1.0 / math.sqrt(hs))
    att = att.masked_fill(mask[:, :, :T, :T] == 0, float("-inf")).softmax(-1)
    return (att @ v).transpose(1, 2).contiguous().view(Bt, T, C)
def views():  # let torch.matmul take strided views
    q, k, v = split()
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
    att = att.masked_fill(mask[:, :, :T, :T] == 0, float("-inf")).softmax(-1)
    return (att @ v).transpose(1, 2).contiguous().view(Bt, T, C)
def sdpa():
    q, k, v = split()
    return F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).contiguous().view(Bt, T, C)
qc, kc, vc = [t.contiguous() for t in split()]; kT = kc.transpose(-2, -1).contiguous()
att = torch.randn(Bt, nh, T, T, **h)
R["attn"] = dict(eager_contig_copies=timeit(ours_like), eager_views=timeit(views), sdpa_causal=timeit(sdpa),
    qkv_copies_only=timeit(lambda: [t.contiguous() for t in split()]), bmm_qk_contig=timeit(lambda: qc @ kT),
    bmm_qk_stridedview=timeit(lambda: qc @ kc.transpose(-2, -1)), bmm_av=timeit(lambda: att @ vc),
    softmax_only=timeit(lambda: att.softmax(-1)), out_transpose=timeit(lambda: att[..., :96].transpose(1, 2).contiguous()))
print("@@R@@" + json.dumps(R))
