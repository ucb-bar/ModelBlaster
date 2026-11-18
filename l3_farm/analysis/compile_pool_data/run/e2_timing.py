#!/usr/bin/env python3
"""X-148 E2: does a low-priority nvcc build load on the same host perturb a timed GPU kernel run?
Kernel: real fold shape (bmm_strided_f16 of 44_MiniGPTBlock QK^T: 128x8, 512x96x512) from layout_stage1_data/naive_f16.cu,
timed (a) as a CUDA graph replay, (b) by cuda events around eager launches. Interleaved blocks: idle / nvcc@nice19 x{3,8,32}."""
import ctypes, json, os, subprocess, sys, time, statistics as st, signal
from pathlib import Path
import torch
sys.path.insert(0, "/tmp/krisd/cpool/e2")
import kernel_lib as KL
W = Path("/tmp/krisd/cpool/e2"); W.mkdir(exist_ok=True, parents=True)
real = [c for c in json.loads((W / "cases_real.json").read_text()) if c["problem"] == "44_MiniGPTBlock"][0]
def mk(B0, B1, M, K, N):
    c = dict(real); c.update(B0=B0, B1=B1, M=M, K=K, N=N, a_off=0, b_off=0, c_off=0,
        a_s0=B1*M*K, a_s1=M*K, a_sm=K, a_sk=1, b_s0=B1*K*N, b_s1=K*N, b_sk=N, b_sn=1, c_s0=B1*M*N, c_s1=M*N, c_sm=N, c_sn=1); return c
CASES = {"real": (real, 2, 40), "mid": (mk(16, 8, 512, 96, 512), 10, 60), "small": (mk(1, 4, 128, 64, 128), 100, 200)}
case, INNER, REPS = CASES[os.environ["E2_CASE"]]
so = W / "naive.so"
if not so.exists():
    KL.build(str(W / "naive_f16.cu"), str(so), extra=["--default-stream", "per-thread"])
lib = KL.load(str(so))
dev = "cuda"
v = KL.views(case)
ext = {k: KL.extent(o, d, s)[1] for k, (d, s, o) in v.items()}
A = (torch.rand(ext["A"], device=dev) * 2 - 1).half(); B = (torch.rand(ext["B"], device=dev) * 2 - 1).half()
C = torch.zeros(ext["C"], device=dev, dtype=torch.float16)
def launch(): KL.call(lib, A, B, C, case)
torch.cuda.synchronize()
PT = torch.cuda.ExternalStream(2)          # cudaStreamPerThread
for _ in range(20): launch()
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
graph_ok = True
try:
    with torch.cuda.stream(PT):
        with torch.cuda.graph(g, stream=PT):
            for _ in range(INNER): launch()
except Exception as e:
    graph_ok = False; print("graph capture failed:", repr(e)[:300])
def t_graph():
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); g.replay(); e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) * 1000 / INNER
def t_event():
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(PT)
    with torch.cuda.stream(PT):
        for _ in range(INNER): launch()
    e.record(PT); torch.cuda.synchronize(); return s.elapsed_time(e) * 1000 / INNER
# nvcc load
units = sorted((W / "units").glob("*.cu"))
ARCH = "-O3 -arch=sm_89 -std=c++17 -shared -Xcompiler -fPIC"
def start_load(n):
    if n == 0: return []
    ps = []
    for w in range(n):
        sh = (f"cd {W}/units; i={w}; U=($(ls *.cu)); while :; do f=${{U[$((i % ${{#U[@]}}))]}}; "
              f"nice -n 19 ionice -c3 nvcc {ARCH} -c $f -o /tmp/krisd/cpool/e2/o_{w}.o >/dev/null 2>&1; i=$((i+7)); done")
        ps.append(subprocess.Popen(["bash", "-c", sh], preexec_fn=os.setsid))
    return ps
def stop_load(ps):
    for p in ps:
        try: os.killpg(p.pid, signal.SIGKILL)
        except Exception: pass
    subprocess.run("pkill -9 -f 'nvcc|cicc|ptxas|cudafe|fatbinary' >/dev/null 2>&1", shell=True)
    time.sleep(1.5)
CYCLES = int(os.environ.get("E2_CYCLES", "3"))
LOADS = [0, 3, 8, 32]
res = {k: {"graph": [], "event": []} for k in LOADS}
for cyc in range(CYCLES):
    for n in LOADS:
        ps = start_load(n); time.sleep(4 if n else 1)
        for _ in range(5): t_event()
        for r in range(REPS):
            if graph_ok: res[n]["graph"].append(t_graph())
            res[n]["event"].append(t_event())
        stop_load(ps)
        print(f"cycle {cyc} load {n}: graph med {st.median(res[n]['graph']) if graph_ok else None} event med {st.median(res[n]['event']):.2f} us", flush=True)
def stats(x):
    x = sorted(x); m = st.median(x)
    return dict(n=len(x), median=round(m, 3), mean=round(st.mean(x), 3), sd=round(st.pstdev(x), 3), cv_pct=round(st.pstdev(x) / st.mean(x) * 100, 3),
                p5=round(x[int(.05 * len(x))], 3), p95=round(x[int(.95 * len(x)) - 1], 3), min=round(x[0], 3), max=round(x[-1], 3))
out = {"graph_ok": graph_ok, "inner": INNER, "reps": REPS, "cycles": CYCLES, "case": case,
       "stats": {f"load{n}": {k: stats(v) for k, v in d.items() if v} for n, d in res.items()}, "raw": res}
(W / f"e2_result_{os.environ['E2_CASE']}.json").write_text(json.dumps(out))
for n in LOADS:
    for k, s in out["stats"][f"load{n}"].items():
        print(n, k, s)
