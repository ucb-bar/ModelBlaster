
import json, os, sys, time, importlib.util, statistics as st, subprocess
import torch
prob, name, iters, warm, do_compile = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] == "1"
out = {"name": name, "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
       "dtype": "float16", "sizing": "native (get_init_inputs/get_inputs, no scaling)", "iters": iters, "warm": warm,
       "method": "median of per-iteration cudaEvent pairs after warm-up (torch44.sh)", "host": os.uname().nodename}
try:
    out["gpu_uuid"] = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], capture_output=True, text=True).stdout.split()[0]
except Exception:
    pass
def others():
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True).stdout
        return [l.strip() for l in r.splitlines() if l.strip() and not l.startswith(str(os.getpid()) + ",")]
    except Exception:
        return None
spec = importlib.util.spec_from_file_location("kbprob", prob); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(0)
init = m.get_init_inputs()
torch.set_default_dtype(torch.float16)   # models that create state inside forward (torch.randn) must create fp16 state
model = m.Model(*init).cuda().half().eval()
# plain tensor attributes (e.g. an RNN's self.h0) are not parameters/buffers, so .half() skips them
for mod in model.modules():
    for k, v in list(vars(mod).items()):
        if torch.is_tensor(v) and v.is_floating_point() and k not in ("_parameters", "_buffers"):
            setattr(mod, k, v.cuda().half())
torch.manual_seed(0)
xs = [x.cuda().half() if torch.is_tensor(x) and x.is_floating_point() else (x.cuda() if torch.is_tensor(x) else x) for x in m.get_inputs()]
out["inputs"] = [list(x.shape) if torch.is_tensor(x) else repr(x) for x in xs]
out["params"] = sum(p.numel() for p in model.parameters())
def timeit(fn):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    return st.median(ts), min(ts)
def shape(y):
    return list(y.shape) if torch.is_tensor(y) else [shape(t) for t in y] if isinstance(y, (list, tuple)) else repr(y)
out["others_before"] = others()
with torch.no_grad():
    y = model(*xs); torch.cuda.synchronize()
    out["output"] = shape(y)
    out["eager_us"], out["eager_min_us"] = timeit(lambda: model(*xs))
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): model(*xs)
    torch.cuda.current_stream().wait_stream(s)
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): yg = model(*xs)
        out["graph_us"], out["graph_min_us"] = timeit(lambda: g.replay())
    except Exception as e:
        out["graph_us"] = None; out["graph_error"] = repr(e)[:300]
    out["compile_us"] = None
    if do_compile:
        try:
            t0 = time.time(); cm = torch.compile(model)
            for _ in range(3): cm(*xs)
            torch.cuda.synchronize(); out["compile_s"] = round(time.time() - t0, 1)
            out["compile_us"], out["compile_min_us"] = timeit(lambda: cm(*xs)); out["compile_mode"] = "default"
        except Exception as e:
            out["compile_error"] = repr(e)[:300]
    else:
        out["compile_error"] = "disabled (--no-compile)"
out["others_after"] = others()
io = f"/tmp/krisd/farm/{name}/irs/{name}/io.npz"   # read-only cross-check against the extracted IR, when still on this node
if os.path.exists(io):
    try:
        import numpy as np
        z = np.load(io); out["ir_io_shapes"] = {k: list(z[k].shape) for k in z.files}
    except Exception as e:
        out["ir_io_shapes"] = repr(e)[:200]
for k in list(out):
    if k.endswith("_us") and isinstance(out[k], float): out[k] = round(out[k], 2)
print("@@POSTHOC@@" + json.dumps(out))
