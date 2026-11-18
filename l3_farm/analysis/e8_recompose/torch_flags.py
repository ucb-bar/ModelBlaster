#!/usr/bin/env python3
"""E8 side check: does the torch CUDA-graph time depend on cuDNN flags (the flow's built-in torch baseline,
pipeline/torch_bench.py, uses cudnn.benchmark=True and cudnn TF32 off; torch_posthoc.py / remote_bench.py use the
defaults: benchmark off, cudnn TF32 on)? Runs on a node: python torch_flags.py PROBLEM.py -> @@FLAGS@@{json}"""
import importlib.util, json, statistics as st, sys
import torch
spec = importlib.util.spec_from_file_location("p", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(0); init = m.get_init_inputs(); torch.set_default_dtype(torch.float16)
model = m.Model(*init).cuda().half().eval(); torch.set_default_dtype(torch.float32)
torch.manual_seed(0); xs = [x.cuda().half() if x.is_floating_point() else x.cuda() for x in m.get_inputs()]
out = {}
for rnd in range(3):
    for bench in (False, True):
        for tf32 in (True, False):
            torch.backends.cudnn.benchmark = bench; torch.backends.cudnn.allow_tf32 = tf32
            with torch.inference_mode():
                s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3): model(*xs)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g): model(*xs)
                for _ in range(10): g.replay()
                torch.cuda.synchronize(); ts = []
                for _ in range(200):
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record(); g.replay(); b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1000)
            out.setdefault(f"benchmark={bench},cudnn_tf32={tf32}", []).append(st.median(ts))
            del g
print("@@FLAGS@@" + json.dumps(out))
