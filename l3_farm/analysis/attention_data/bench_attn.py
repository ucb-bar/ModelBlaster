"""Attention-core benchmark on one L40S (fp16): our unfused chain (eager torch ops) vs SDPA backends vs a
Triton flash-style kernel (softmax / relu variants) vs Inductor on the exact model chain; plus numerics vs the
walker-style reference (fp32 math, fp16 rounding after every op) and an fp32 truth.

    python bench_attn.py CFG [CFG ...]       CFG in 31, 43, 50, 28
"""
import json, math, os, sys, time, statistics as st, collections
import torch, torch.nn.functional as F
import triton, triton.language as tl
from torch.nn.attention import sdpa_kernel, SDPBackend

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
DEV = "cuda"

# name: (B, nh, T, hs, causal, act, layout)  layout: "qkv" = MinGPT (B,T,3C) buffer; "mha" = 3 separate (T,B,E) buffers
CFGS = {
    "31": dict(B=2, nh=4, T=16384, hs=32, causal=False, act="softmax", layout="mha"),
    "43": dict(B=128, nh=8, T=512, hs=96, causal=True, act="softmax", layout="qkv"),   # == 44's attention
    "50": dict(B=16, nh=12, T=1024, hs=64, causal=True, act="relu", layout="qkv"),
    "28": dict(B=2, nh=8, T=197, hs=64, causal=False, act="softmax", layout="mha"),
}


def timeit(fn, iters=30, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    return round(st.median(ts), 1)


def kernels(fn, n=3):
    from torch.profiler import profile, ProfilerActivity
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    ks = collections.Counter()
    for e in p.events():
        if e.device_type.name == "CUDA" and e.device_time_total > 0:
            ks[e.name[:90]] += e.device_time_total / n
    return [(k, round(v, 1)) for k, v in ks.most_common(12)]


# ------------------------------------------------------------------ Triton flash-style kernel
@triton.jit
def _attn_fwd(Q, K, V, O, sqb, sqh, sqm, skb, skh, skn, svb, svh, svn, sob, soh, som,
              H, M, N, scale, D: tl.constexpr, DP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
              CAUSAL: tl.constexpr, RELU: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, DP)
    dmask = offs_d < D
    q = tl.load(Q + b * sqb + h * sqh + offs_m[:, None] * sqm + offs_d[None, :],
                mask=(offs_m[:, None] < M) & dmask[None, :], other=0.0)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, DP], tl.float32)
    hi = N
    if CAUSAL:
        hi = tl.minimum(N, (pid_m + 1) * BM)
    kb = K + b * skb + h * skh
    vb = V + b * svb + h * svh
    for start in range(0, hi, BN):
        offs_n = start + tl.arange(0, BN)
        nmask = offs_n < N
        k = tl.load(kb + offs_n[None, :] * skn + offs_d[:, None], mask=nmask[None, :] & dmask[:, None], other=0.0)
        s = tl.dot(q, k) * scale
        keep = nmask[None, :]
        if CAUSAL:
            keep = keep & (offs_m[:, None] >= offs_n[None, :])
        s = tl.where(keep, s, float("-inf"))
        v = tl.load(vb + offs_n[:, None] * svn + offs_d[None, :], mask=nmask[:, None] & dmask[None, :], other=0.0)
        if RELU:
            p = tl.maximum(s, 0.0)
            acc += tl.dot(p.to(tl.float16), v)
        else:
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
            m_i = m_new
    if not RELU:
        acc = acc / l_i[:, None]
    tl.store(O + b * sob + h * soh + offs_m[:, None] * som + offs_d[None, :], acc.to(tl.float16),
             mask=(offs_m[:, None] < M) & dmask[None, :])


def triton_attn(q, k, v, o, scale, causal, relu, BM=64, BN=64, warps=4, stages=2):
    """q,k,v,o: [B, H, T, D] views with unit last stride (any other strides)."""
    B, H, M, D = q.shape
    N = k.shape[2]
    DP = triton.next_power_of_2(D)
    grid = (triton.cdiv(M, BM), B * H)
    _attn_fwd[grid](q, k, v, o, q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
                    v.stride(0), v.stride(1), v.stride(2), o.stride(0), o.stride(1), o.stride(2),
                    H, M, N, scale, D=D, DP=DP, BM=BM, BN=BN, CAUSAL=causal, RELU=relu,
                    num_warps=warps, num_stages=stages)
    return o


# ------------------------------------------------------------------ references
def chain16(q, k, v, scale, causal, relu):
    """walk_torch semantics: fp32 math per op, every op's output rounded to fp16."""
    r = lambda t: t.half().float()
    s = r(q.float() @ k.float().transpose(-1, -2))
    s = r(s * scale)
    if causal:
        M, N = s.shape[-2:]
        s = s.masked_fill(torch.ones(M, N, device=s.device, dtype=torch.bool).triu(1), float("-inf"))
    p = r(torch.relu(s)) if relu else r(torch.softmax(s, -1))
    return r(p @ v.float())


def ref32(q, k, v, scale, causal, relu):
    s = (q.float() @ k.float().transpose(-1, -2)) * scale
    if causal:
        M, N = s.shape[-2:]
        s = s.masked_fill(torch.ones(M, N, device=s.device, dtype=torch.bool).triu(1), float("-inf"))
    p = torch.relu(s) if relu else torch.softmax(s, -1)
    return p @ v.float()


def rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).abs().max() / b.abs().max())


def make(c):
    B, nh, T, hs = c["B"], c["nh"], c["T"], c["hs"]
    C = nh * hs
    g = torch.Generator(device=DEV).manual_seed(0)
    if c["layout"] == "qkv":     # MinGPT: c_attn(x) -> (B,T,3C); q/k/v = split + view + transpose(1,2)
        buf = (torch.randn(B, T, 3 * C, device=DEV, generator=g) * 1.0).half()
        q, k, v = (buf[..., i * C:(i + 1) * C].view(B, T, nh, hs).transpose(1, 2) for i in range(3))
        out_strided = lambda: torch.empty(B, T, nh, hs, device=DEV, dtype=torch.half).transpose(1, 2)
        bufs = [buf]
    else:                        # nn.MultiheadAttention: q/k/v each (T, B, E) -> view(T, B*nh, hs).transpose(0,1)
        bufs = [torch.randn(T, B, C, device=DEV, generator=g).half() for _ in range(3)]
        q, k, v = (t.view(T, B, nh, hs).permute(1, 2, 0, 3) for t in bufs)   # [B, nh, T, hs] strided
        out_strided = lambda: torch.empty(T, B, nh, hs, device=DEV, dtype=torch.half).permute(1, 2, 0, 3)
    return q, k, v, out_strided, bufs


def bench(name):
    c = CFGS[name]
    B, nh, T, hs, causal, relu = c["B"], c["nh"], c["T"], c["hs"], c["causal"], c["act"] == "relu"
    scale = 1.0 / math.sqrt(hs)
    big = T >= 8192
    it = dict(iters=10, warm=3) if big else dict(iters=30, warm=5)
    r = {"name": name, **c, "scale": scale, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
         "triton": triton.__version__, "host": os.uname().nodename}
    q, k, v, out_strided, bufs = make(c)
    qc, kc, vc = (t.contiguous() for t in (q, k, v))
    flops = 4.0 * B * nh * T * T * hs * (0.5 if causal else 1.0)
    r["useful_tflop"] = flops / 1e12
    mask = torch.ones(T, T, device=DEV, dtype=torch.bool).triu(1) if causal else None
    T_ = {}
    with torch.no_grad():
        # (1) our structure: materialized chain, contiguous operands, cuBLAS bmm + separate elementwise ops
        def chain():
            s = qc @ kc.transpose(-1, -2)
            s = s * scale
            if causal:
                s = s.masked_fill(mask, float("-inf"))
            p = torch.relu(s) if relu else torch.softmax(s, -1)
            return p @ vc
        T_["chain_eager_contig"] = timeit(chain, **it)
        torch.cuda.empty_cache()
        # (2) SDPA backends
        if not relu:
            for nm, be in (("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                           ("cudnn", SDPBackend.CUDNN_ATTENTION)):
                for lay, (a, b_, cc) in (("contig", (qc, kc, vc)), ("strided", (q, k, v))):
                    try:
                        with sdpa_kernel([be]):
                            f = lambda: F.scaled_dot_product_attention(a, b_, cc, is_causal=causal, scale=scale)
                            f()
                            T_[f"sdpa_{nm}_{lay}"] = timeit(f, **it)
                            if lay == "strided":   # + writing y back in the (B,T,C)/(T,B,E) layout the out-proj reads
                                o = out_strided()
                                T_[f"sdpa_{nm}_strided+outcopy"] = timeit(lambda: o.copy_(f()), **it)
                    except Exception as e:
                        T_[f"sdpa_{nm}_{lay}"] = f"ERR {type(e).__name__}: {str(e)[:120]}"
            try:
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                    r["sdpa_flash_kernels"] = kernels(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale))
            except Exception as e:
                r["sdpa_flash_kernels"] = repr(e)[:200]
        # (3) Triton flash-style kernel (straightforward tile kernel: what a generated kernel can reach)
        best = None
        for BM, BN, w, s_ in ((64, 64, 4, 2), (128, 64, 4, 2), (128, 64, 8, 2), (64, 32, 4, 3), (128, 32, 4, 3), (64, 64, 4, 3), (128, 128, 8, 2)):
            try:
                o = out_strided()
                f = lambda: triton_attn(q, k, v, o, scale, causal, relu, BM, BN, w, s_)
                t = timeit(f, **it)
                T_[f"triton_strided_{BM}x{BN}w{w}s{s_}"] = t
                if best is None or t < best[0]:
                    best = (t, (BM, BN, w, s_))
            except Exception as e:
                T_[f"triton_strided_{BM}x{BN}w{w}s{s_}"] = f"ERR {type(e).__name__}: {str(e)[:100]}"
        r["triton_best"] = best
        # (4) Inductor on the exact model chain (from the projection buffers to the layout the out-proj reads)
        if c["layout"] == "qkv":
            bias = torch.tril(torch.ones(1024, 1024, device=DEV)).view(1, 1, 1024, 1024).half()
            def model_chain(buf):
                C = nh * hs
                qq, kk, vv = buf.split(C, dim=2)
                kk = kk.view(B, T, nh, hs).transpose(1, 2)
                qq = qq.view(B, T, nh, hs).transpose(1, 2)
                vv = vv.view(B, T, nh, hs).transpose(1, 2)
                att = (qq @ kk.transpose(-2, -1)) * (1.0 / math.sqrt(kk.size(-1)))
                att = att.masked_fill(bias[:, :, :T, :T] == 0, float("-inf"))
                att = F.relu(att) if relu else F.softmax(att, dim=-1)
                y = att @ vv
                return y.transpose(1, 2).contiguous().view(B, T, C)
            args = (bufs[0],)
        else:
            def model_chain(qb, kb, vb):   # F.multi_head_attention_forward core after the in-projections
                qq, kk, vv = (t.view(T, B * nh, hs).transpose(0, 1) for t in (qb, kb, vb))
                att = torch.bmm(qq, kk.transpose(1, 2)) * scale
                att = torch.softmax(att, dim=-1)
                y = torch.bmm(att, vv)
                return y.transpose(0, 1).contiguous().view(T, B, nh * hs)
            args = tuple(bufs)
        T_["model_chain_eager"] = timeit(lambda: model_chain(*args), **it)
        torch.cuda.empty_cache()
        try:
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/krisd/attn/cache/inductor")
            cm = torch.compile(model_chain)
            for _ in range(2):
                cm(*args)
            torch.cuda.synchronize()
            T_["model_chain_inductor"] = timeit(lambda: cm(*args), **it)
            r["inductor_kernels"] = kernels(lambda: cm(*args))
        except Exception as e:
            T_["model_chain_inductor"] = f"ERR {type(e).__name__}: {str(e)[:150]}"
        torch.cuda.empty_cache()
        # (5) numerics on a slice (rows of a few heads): flash / triton vs walker-style chain16 and fp32 truth
        nb = 1 if big else min(B, 2)
        rows = slice(0, 2048) if big else slice(None)
        qs, ks_, vs = q[:nb], k[:nb], v[:nb]
        c16 = chain16(qs[:, :, rows], ks_, vs, scale, causal, relu) if not causal else chain16(qs, ks_, vs, scale, causal, relu)[:, :, rows]
        t32 = ref32(qs[:, :, rows], ks_, vs, scale, causal, relu) if not causal else ref32(qs, ks_, vs, scale, causal, relu)[:, :, rows]
        num = {"chain16_vs_fp32": rel(c16, t32)}
        if not relu:
            for nm, be in (("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION)):
                try:
                    with sdpa_kernel([be]):
                        y = F.scaled_dot_product_attention(qs, ks_, vs, is_causal=causal, scale=scale)[:, :, rows]
                    num[f"{nm}_vs_chain16"] = rel(y, c16); num[f"{nm}_vs_fp32"] = rel(y, t32)
                except Exception as e:
                    num[nm] = repr(e)[:100]
        o = torch.empty_like(qs.contiguous())
        y = triton_attn(qs, ks_, vs, o, scale, causal, relu, *(best[1] if best else (64, 64, 4, 2)))[:, :, rows]
        num["triton_vs_chain16"] = rel(y, c16); num["triton_vs_fp32"] = rel(y, t32)
        # what the flow's checks would say: slot check err <= 2e-3*max|ref| + 2^-23; output check 1e-2*max|ref| + 1e-5
        num["slot_rel_tol"] = 2e-3; num["output_rel_tol"] = 1e-2
        r["numerics"] = num
    r["times_us"] = T_
    for kk_, vv_ in T_.items():
        if isinstance(vv_, float):
            T_[kk_] = vv_
    if best:
        r["triton_tflops"] = round(flops / (best[0] * 1e-6) / 1e12, 1)
    print("@@R@@" + json.dumps(r), flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        try:
            bench(n)
        except Exception as e:
            import traceback; traceback.print_exc()
            print("@@R@@" + json.dumps({"name": n, "error": repr(e)[:400]}), flush=True)
        torch.cuda.empty_cache()
