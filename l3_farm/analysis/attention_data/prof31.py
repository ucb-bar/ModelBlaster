import sys, json, importlib.util, torch, collections
sys.argv=[sys.argv[0]]
sys.path.insert(0, "/tmp/krisd/attn"); from bench_attn import timeit, kernels
spec = importlib.util.spec_from_file_location("p", "/tmp/krisd/attn/p31.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(0); model = m.Model(*m.get_init_inputs()).cuda().half().eval()
x = m.get_inputs()[0].cuda().half()
with torch.no_grad():
    cm = torch.compile(model)
    for _ in range(3): cm(x)
    torch.cuda.synchronize()
    r = {"compile_us": timeit(lambda: cm(x), iters=20, warm=3), "kernels": kernels(lambda: cm(x))}
    r["eager_us"] = timeit(lambda: model(x), iters=5, warm=2)
print("@@P31@@" + json.dumps(r))
