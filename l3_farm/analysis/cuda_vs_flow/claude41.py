#!/usr/bin/env python3
"""Time Claude's CUDA-only 41_GRUBidirectional solution like baseline_cc/remote_bench.py (seeded Model/ModelNew,
fp16, state_dict loaded, CUDA-graph replay, median of cudaEvent pairs) and two flow-shaped re-arrangements of
its own kernel:

  as_submitted   one launch per layer, grid 2*NB: both directions concurrently (Claude's structure)
  split_dirs     the same kernel, but one launch per (layer, direction): grid NB, forward then reverse
                 (what a one-slot-per-direction IR forces), sequential on one stream
  split_streams  split_dirs, but the two direction launches on two streams (fork/join)

usage: claude41.py SOLUTION.py PROBLEM.py [modes...]
"""
import importlib.util, json, re, sys, time
import torch

sol_path, prob_path = sys.argv[1], sys.argv[2]
modes = sys.argv[3:] or ["as_submitted", "split_dirs", "split_streams"]
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P = load(prob_path, "prob")
src0 = open(sol_path).read()


def variant(mode):
    if mode == "as_submitted":
        return src0
    s = src0
    s = s.replace("    int T;\n    unsigned tagbase;", "    int T;\n    unsigned tagbase;\n    int dir_base;")
    s = s.replace("const int dir = blockIdx.x / NB, blk = blockIdx.x % NB;",
                  "const int dir = p.dir_base + blockIdx.x / NB, blk = blockIdx.x % NB;")
    assert "p.dir_base" in s
    launch = re.compile(r"(gru_layer<(\d+)>)<<<2 \* NB, 256, smem, stream>>>\(p\);")
    if mode == "split_dirs":
        rep = r"{ p.dir_base = 0; \1<<<NB, 256, smem, stream>>>(p); p.dir_base = 1; \1<<<NB, 256, smem, stream>>>(p); }"
    else:
        rep = (r"{ static cudaStream_t s2 = nullptr; static cudaEvent_t ef, ej; if (!s2) { cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking); "
               r"cudaEventCreateWithFlags(&ef, cudaEventDisableTiming); cudaEventCreateWithFlags(&ej, cudaEventDisableTiming); } "
               r"cudaEventRecord(ef, stream); cudaStreamWaitEvent(s2, ef, 0); "
               r"p.dir_base = 0; \1<<<NB, 256, smem, stream>>>(p); p.dir_base = 1; \1<<<NB, 256, smem, s2>>>(p); "
               r"cudaEventRecord(ej, s2); cudaStreamWaitEvent(stream, ej, 0); }")
    s, n = launch.subn(rep, s)
    assert n >= 2, n
    s = re.sub(r'name=f?"([^"]+)"', lambda m: f'name=f"{m.group(1)}_{mode}"', s)
    return s


def build(mode):
    path = f"/tmp/krisd/cvf/claude41/sol_{mode}.py"
    open(path, "w").write(variant(mode))
    return load(path, "sol_" + mode)


def graph_time(fn, xs, warm=10, iters=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(*xs)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = fn(*xs)
    for _ in range(warm):
        g.replay()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    ts.sort()
    return ts[len(ts) // 2], ts[0], y


torch.manual_seed(0)
init = P.get_init_inputs()
torch.set_default_dtype(torch.float16)
ref = P.Model(*init).cuda().half().eval()
torch.set_default_dtype(torch.float32)
torch.manual_seed(0)
xs = [x.cuda().half() for x in P.get_inputs()]
with torch.no_grad():
    ref32 = P.Model(*init).cuda().float().eval()
    ref32.load_state_dict({k: v.float() for k, v in ref.state_dict().items()})
    yref = ref32(*[x.float() for x in xs])
res = {}
for mode in modes:
    M = build(mode)
    torch.manual_seed(0)
    m = M.ModelNew(*init).cuda().half().eval()
    m.load_state_dict(ref.state_dict())
    with torch.no_grad():
        y = m(*xs)
        torch.cuda.synchronize()
        err = (y.float() - yref).abs().max().item()
        med, mn, _ = graph_time(m, xs)
    res[mode] = dict(med_us=round(med, 1), min_us=round(mn, 1), max_abs_err=err)
    print("RES", mode, json.dumps(res[mode]), flush=True)
print("@@JSON@@" + json.dumps(res))
