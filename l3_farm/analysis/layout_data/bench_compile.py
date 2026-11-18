import json, sys, os, glob, importlib.util, statistics as st, collections, base64, io, tarfile, time
import torch
def load(name):
    spec = importlib.util.spec_from_file_location("p", f"/tmp/krisd/layout/{name}.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    torch.manual_seed(0); init = m.get_init_inputs(); torch.set_default_dtype(torch.float16)
    model = m.Model(*init).cuda().half().eval()
    for mod in model.modules():
        for k, v in list(vars(mod).items()):
            if torch.is_tensor(v) and v.is_floating_point() and k not in ("_parameters", "_buffers"): setattr(mod, k, v.cuda().half())
    torch.manual_seed(0); xs = [x.cuda().half() for x in m.get_inputs()]
    torch.set_default_dtype(torch.float32)
    return model, xs
def timeit(fn, iters=30, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    return round(st.median(ts), 1)
def prof(fn):
    from torch.profiler import profile as P, ProfilerActivity as A
    fn(); torch.cuda.synchronize()
    with P(activities=[A.CUDA]) as p:
        for _ in range(3): fn()
        torch.cuda.synchronize()
    ks = collections.Counter(); n = collections.Counter()
    for e in p.events():
        if e.device_type.name == "CUDA" and e.device_time_total > 0:
            ks[e.name[:110]] += e.device_time_total / 3; n[e.name[:110]] += 1 / 3
    return [(k, round(v, 1), round(n[k], 1)) for k, v in ks.most_common(40)]
name = sys.argv[1]
os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/krisd/layout/cache/ind_{name}_{int(time.time())}"  # fresh, so code is regenerated
model, xs = load(name); r = {"name": name}
with torch.no_grad():
    r["eager_us"] = timeit(lambda: model(*xs)); r["eager_prof"] = prof(lambda: model(*xs))
    cm = torch.compile(model)
    for _ in range(3): cm(*xs)
    torch.cuda.synchronize()
    r["compile_us"] = timeit(lambda: cm(*xs)); r["compile_prof"] = prof(lambda: cm(*xs))
print("@@R@@" + json.dumps(r), flush=True)
files = glob.glob(os.environ["TORCHINDUCTOR_CACHE_DIR"] + "/**/*.py", recursive=True)
code = [f for f in files if "def call(" in open(f).read()]
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as t:
    for i, f in enumerate(code): t.add(f, arcname=f"{name}_{i}.py")
print("@@CODE@@" + base64.b64encode(buf.getvalue()).decode())
