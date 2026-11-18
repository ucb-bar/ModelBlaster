import json, statistics as st, math
import torch
def timeit(fn, iters=40, warm=10):
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
# MinGPT: QK^T reading q,k straight out of the qkv buffer (lda=2304) vs from contiguous per-head copies
B, T, C, nh = 128, 512, 768, 8; hs = 96
qkv = torch.randn(B, T, 3 * C, **h) / 4
out = torch.empty(nh, B, T, T, **h); out2 = torch.empty(nh, B, T, hs, **h)
def qk_views():
    for i in range(nh):
        torch.bmm(qkv[:, :, i*hs:(i+1)*hs], qkv[:, :, C + i*hs:C + (i+1)*hs].transpose(1, 2), out=out[i])
qc = torch.randn(B * nh, T, hs, **h); kT = torch.randn(B * nh, hs, T, **h); o3 = torch.empty(B * nh, T, T, **h)
def qk_contig():
    torch.bmm(qc, kT, out=o3)
att = torch.randn(B, T, T, **h)
def av_views():
    for i in range(nh):
        torch.bmm(att, qkv[:, :, 2*C + i*hs:2*C + (i+1)*hs], out=out2[i])
attc = torch.randn(B * nh, T, T, **h); vc = torch.randn(B * nh, T, hs, **h); o4 = torch.empty(B * nh, T, hs, **h)
R["mingpt"] = dict(qk_strided_views_8calls=timeit(qk_views), qk_contig_1call=timeit(qk_contig),
                   av_strided_views_8calls=timeit(av_views), av_contig_1call=timeit(lambda: torch.bmm(attc, vc, out=o4)))
# Mamba states einsum: bclhn,bhcl,bclhp->bchpn ; X (b,c,l,h,p) read in place with transA/lda=512 vs permuted copy
b, c, l, H, p, n = 2048, 2, 64, 8, 64, 16
X = torch.randn(b, c, l, H, p, **h); DB = torch.randn(b, c, l, H, n, **h)
Xv = X.view(b * c, l, H, p); DBv = DB.view(b * c, l, H, n); S = torch.empty(H, b * c, p, n, **h)
def states_views():
    for i in range(H):
        torch.bmm(Xv[:, :, i].transpose(1, 2), DBv[:, :, i], out=S[i])
def states_copy():
    Xp = X.permute(0, 1, 3, 4, 2).reshape(b * c * H, p, l)   # (bch, p, l) materialized
    Dp = DB.permute(0, 1, 3, 2, 4).reshape(b * c * H, l, n)
    return torch.bmm(Xp, Dp)
Xp = X.permute(0, 1, 3, 4, 2).reshape(b * c * H, p, l).contiguous(); Dp = DB.permute(0, 1, 3, 2, 4).reshape(b * c * H, l, n).contiguous()
R["mamba_states"] = dict(views_8calls=timeit(states_views), copy_then_bmm=timeit(states_copy), bmm_only=timeit(lambda: torch.bmm(Xp, Dp)),
                         X_permute_copy=timeit(lambda: X.permute(0, 1, 3, 4, 2).contiguous()), einsum=timeit(lambda: torch.einsum("bclhn,bclhp->bchpn", DB, X)))
print("@@R@@" + json.dumps(R))
