import json, sys, importlib.util, statistics as st, collections, re
import torch, torch.nn as nn, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
def load(name):
    spec = importlib.util.spec_from_file_location("p", f"/tmp/krisd/layout/{name}.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    torch.manual_seed(0); init = m.get_init_inputs(); torch.set_default_dtype(torch.float16)
    model = m.Model(*init).cuda().half().eval(); torch.set_default_dtype(torch.float32)
    torch.manual_seed(0); xs = [x.cuda().half() for x in m.get_inputs()]
    return model, xs
def timeit(fn, iters=50, warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    return round(st.median(ts), 1)
def graphed(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    return g.replay
def cat(k):
    k = k.lower()
    if any(t in k for t in ("nchwtonhwc", "nhwctonchw", "transpose", "convert", "permute", "copy")): return "layout/copy"
    if any(t in k for t in ("conv", "xmma", "implicit", "fprop", "dgrad", "winograd", "cutlass", "sm80_", "sm86_", "sm89_", "gemm", "s16816", "h16816", "s1688", "h1688", "cudnn")): return "conv/gemm"
    return "other"
def profile(fn):
    from torch.profiler import profile as P, ProfilerActivity as A
    fn(); torch.cuda.synchronize()
    with P(activities=[A.CUDA]) as p:
        for _ in range(5): fn()
        torch.cuda.synchronize()
    c = collections.Counter(); ks = collections.Counter()
    for e in p.events():
        if e.device_type.name == "CUDA" and e.device_time_total > 0 and "Memcpy" not in e.name and "Memset" not in e.name:
            c[cat(e.name)] += e.device_time_total / 5; ks[e.name[:90]] += e.device_time_total / 5
    return {k: round(v, 1) for k, v in c.items()}, [(k, round(v, 1)) for k, v in ks.most_common(8)]
res = {}
for name in sys.argv[1:]:
    model, xs = load(name); r = {}
    with torch.no_grad():
        for bm in (False, True):
            torch.backends.cudnn.benchmark = bm
            model = model.to(memory_format=torch.contiguous_format)
            f = graphed(lambda: model(*xs)); r[f"nchw_bench{int(bm)}_us"] = timeit(f)
            if bm: r["nchw_prof"] = profile(f)
            model = model.to(memory_format=torch.channels_last)
            xcl = [x.to(memory_format=torch.channels_last) if x.dim() == 4 else x for x in xs]
            f = graphed(lambda: model(*xcl)); r[f"cl_bench{int(bm)}_us"] = timeit(f)
            if bm: r["cl_prof"] = profile(f)
        # per-conv-layer: NCHW vs NHWC (weights prepacked to channels_last once)
        torch.backends.cudnn.benchmark = True
        model = model.to(memory_format=torch.contiguous_format)
        shapes = collections.OrderedDict()
        def hook(mod, inp, out):
            k = (tuple(inp[0].shape), mod.out_channels, mod.kernel_size, mod.stride, mod.padding, mod.dilation, mod.groups)
            shapes.setdefault(k, [mod, 0]); shapes[k][1] += 1
        hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
        model(*xs); [h.remove() for h in hs]
        layers = []; tn = tc = 0
        for k, (mod, cnt) in shapes.items():
            x = torch.randn(k[0], device="cuda", dtype=torch.half); w = mod.weight.detach().contiguous(); b = mod.bias
            a = dict(stride=mod.stride, padding=mod.padding, dilation=mod.dilation, groups=mod.groups)
            t1 = timeit(graphed(lambda: F.conv2d(x, w, b, **a)), 30, 5)
            xc = x.to(memory_format=torch.channels_last); wc = w.to(memory_format=torch.channels_last)
            t2 = timeit(graphed(lambda: F.conv2d(xc, wc, b, **a)), 30, 5)
            t3 = timeit(graphed(lambda: F.relu(F.conv2d(xc, wc, b, **a))), 30, 5)
            layers.append(dict(inp=k[0], oc=k[1], k=k[2], s=k[3], g=k[6], count=cnt, nchw_us=t1, nhwc_us=t2, nhwc_relu_us=t3))
            tn += t1 * cnt; tc += t2 * cnt
        r["layers"] = layers; r["sum_conv_nchw_us"] = round(tn, 1); r["sum_conv_nhwc_us"] = round(tc, 1)
    res[name] = r; print("@@R@@" + json.dumps({name: r}), flush=True)
    del model; torch.cuda.empty_cache()
