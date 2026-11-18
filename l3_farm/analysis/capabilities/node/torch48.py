#!/usr/bin/env python3
"""PyTorch reference timings for 48_Mamba2ReturnY on this GPU: fp32 and fp16 (model.half()), CUDA-graph replay,
median of 50 event-timed replays after warm-up; output finiteness. Usage: torch48.py PROBLEM.py"""
import importlib.util, json, sys
import torch
spec = importlib.util.spec_from_file_location("p", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(0)
res = {}
for dt in (torch.float32, torch.float16):
    torch.manual_seed(0)
    model = m.Model(*m.get_init_inputs()).cuda().to(dt).eval()
    torch.manual_seed(0)
    ins = [x.cuda().to(dt) for x in m.get_inputs()]
    with torch.no_grad():
        for _ in range(3):
            out = model(*ins)
        torch.cuda.synchronize()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                model(*ins)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = model(*ins)
        ts = []
        for _ in range(50):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    ts.sort()
    f = out.float()
    res[str(dt)] = {"graph_us_median": ts[len(ts) // 2], "nonfinite": int((~torch.isfinite(f)).sum()), "n": f.numel(),
                    "max_abs_finite": float(f[torch.isfinite(f)].abs().max())}
    del model, ins, out, g
    torch.cuda.empty_cache()
print("@@TORCH48@@" + json.dumps(res))
