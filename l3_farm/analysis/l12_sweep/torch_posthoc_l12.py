#!/usr/bin/env python3
"""Post-hoc PyTorch baseline for problems whose flow run could not time torch (torch_status != 'ok', e.g. the
nn.TransformerEncoder / nn.MultiheadAttention family, whose state keys have no IR weights). Does not touch the flow.

The KernelBench module is instantiated at its NATIVE sizes (get_init_inputs / get_inputs, exactly as the farm's
extraction does: worker3.sh runs extract_graph with --bench-target-mb 0, i.e. no scaling), cast to fp16, eval(),
no_grad(), on the node's L40S, and timed as in L1_1_square_matmul_run/l3/torch44.sh: per-iteration cudaEvent
pairs, median after warm-up, for eager, CUDA-graph replay and (if it works) torch.compile (default mode).

  python3 torch_posthoc.py run NAME [NAME ...] [--job JOBID] [--iters 50] [--warm 10] [--no-compile]
        -> results/NAME/torch_posthoc.json
  python3 torch_posthoc.py report [--all]
        -> table of torch_posthoc vs model_e2e_us for problems whose summary has torch_status != 'ok'
           (--all: every problem with a torch_posthoc.json), also written to results/torch_posthoc.md

Nodes are reached only through farm.ssh -> `srun --jobid=J --overlap bash -s` with the script on stdin. Files on the
node go to /tmp/krisd/posthoc/NAME (never /tmp/krisd/farm/NAME, which is only read: the IR's io.npz, if present, is
used to cross-check the output shape). Caches: TRITON_CACHE_DIR / TORCHINDUCTOR_CACHE_DIR under /tmp/krisd/posthoc.
GPUs are shared with the running sweep, so the json records the other compute processes seen on the GPU.
"""
import argparse, base64, json, shlex, sys, time
from pathlib import Path

import farm

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "analysis/l12_sweep/torch_posthoc"   # not results_l12: the dispatcher rewrites those dirs on collect
SUM = ROOT / "results_l12"
P = "/tmp/krisd/posthoc"

BENCH = r'''
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
io = f"/tmp/krisd/farm_l12/{name}/irs/{name}/io.npz"   # read-only cross-check against the extracted IR, when still on this node
if os.path.exists(io):
    try:
        import numpy as np
        z = np.load(io); out["ir_io_shapes"] = {k: list(z[k].shape) for k in z.files}
    except Exception as e:
        out["ir_io_shapes"] = repr(e)[:200]
for k in list(out):
    if k.endswith("_us") and isinstance(out[k], float): out[k] = round(out[k], 2)
print("@@POSTHOC@@" + json.dumps(out))
'''

SH = r'''set -e
D={P}/$1; mkdir -p $D {P}/cache/triton {P}/cache/inductor
export TRITON_CACHE_DIR={P}/cache/triton TORCHINDUCTOR_CACHE_DIR={P}/cache/inductor TMPDIR={P}/cache
export PATH=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04/bin:$PATH
export CUDA_HOME=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04
. /tmp/krisd/venv/bin/activate
base64 -d > $D/bench.py <<'__B__'
{BENCH}
__B__
base64 -d > $D/problem.py <<'__P__'
{PROB}
__P__
U=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1 | tr -d " ")
exec 9>/dev/shm/kernelblaster_gpu_$U.lock; flock 9    # the sweep's per-GPU timing lock: never time while its timed run is on the GPU
cd $D && timeout {TO} python3 bench.py $D/problem.py $1 $2 $3 $4 </dev/null 2>$D/bench.err || {{ echo "rc=$?"; tail -20 $D/bench.err; }}
'''


def b64(b):
    return base64.encodebytes(b).decode()


def pick_job(job):
    if job:
        return int(job)
    live = [j for j in farm.jobs() if j["state"] == "RUNNING"]
    if not live:
        sys.exit("no running L40S allocation")
    return max(live, key=lambda j: j["left"])["job"]


def run_one(job, name, a):
    prob = ROOT / "problems" / f"{name}.py"
    if not prob.exists():
        sys.exit(f"no {prob}")
    sh = SH.format(P=P, BENCH=b64(BENCH.encode()), PROB=b64(prob.read_bytes()), TO=a.timeout)
    args = [name, a.iters, a.warm, int(not a.no_compile)]
    cmd = f"srun --jobid={job} --overlap bash -s -- " + " ".join(shlex.quote(str(x)) for x in args)
    t0 = time.time()
    rc, out, err = farm.ssh(cmd, sh.encode(), timeout=a.timeout + 120)
    txt = out.decode(errors="replace")
    line = next((l for l in txt.splitlines() if l.startswith("@@POSTHOC@@")), None)
    if not line:
        print(name, "FAILED rc", rc, txt[-2000:], err.decode(errors="replace")[-1000:])
        return None
    d = json.loads(line[len("@@POSTHOC@@"):])
    d.update(job=job, wall_s=round(time.time() - t0, 1), when=time.strftime("%Y-%m-%d %H:%M:%S"))
    ex = SUM / name / "extract.log"
    d["extract_log"] = ex.read_text()[-400:] if ex.exists() else None
    (RES / name).mkdir(parents=True, exist_ok=True)
    (RES / name / "torch_posthoc.json").write_text(json.dumps(d, indent=1) + "\n")
    print(f"{name}: eager {d['eager_us']} us | graph {d.get('graph_us')} us | compile {d.get('compile_us')} us "
          f"| {d['gpu']} torch {d['torch']} | out {d['output']} | others on GPU {d.get('others_before')}")
    return d


def summary_row(name):
    sj = SUM / name / "work" / "summary.json"
    if not sj.exists():
        return None
    irs = json.loads(sj.read_text()).get("irs", {})
    return next(iter(irs.values()), None) if irs else None


def cmd_report(a):
    rows = ["| problem | torch_status | model_e2e (us) | torch eager (us) | torch graph (us) | torch.compile (us) "
            "| vs eager | vs graph | vs compile | vs best torch | gpu shared |", "|" + "---|" * 11]
    for d in sorted(RES.iterdir(), key=lambda p: int(p.name.split("_")[0]) if p.name[0].isdigit() else 1e9):
        if not d.is_dir():
            continue
        v, ph = summary_row(d.name), d / "torch_posthoc.json"
        if not v:
            continue
        status = v.get("torch_status")
        if status == "ok" and not a.all:
            continue
        if not ph.exists():
            if status != "ok":
                rows.append(f"| {d.name} | {status} | {v.get('model_e2e_us') or ''} | (not run) |||||||")
            continue
        t = json.loads(ph.read_text())
        me = v.get("model_e2e_us")
        tv = {k: t.get(f"{k}_us") for k in ("eager", "graph", "compile")}
        best = min((x for x in tv.values() if x), default=None)
        sp = lambda x: f"{x / me:.2f}x" if x and me else ""
        rows.append(f"| {d.name} | {status} | {me or ''} | {tv['eager'] or ''} | {tv['graph'] or ''} | {tv['compile'] or ''} "
                    f"| {sp(tv['eager'])} | {sp(tv['graph'])} | {sp(tv['compile'])} | {sp(best)} | {len(t.get('others_before') or [])} |")
    rows.append("\nspeedup = torch_us / model_e2e_us (>1: ours faster). torch: fp16, native sizes, eval/no_grad, median "
                "cudaEvent (torch_posthoc.py); 'gpu shared' = other compute processes on the GPU during the timing.")
    (RES / "torch_posthoc.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--job", help="slurm job id of the L40S allocation to use (default: the live one with most time left)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--timeout", type=int, default=900, help="seconds for the remote python (compile included)")
    ap.add_argument("--all", action="store_true", help="report: include problems whose torch_status is ok")
    a = ap.parse_args()
    if a.cmd == "report":
        return cmd_report(a)
    if a.iters < 20:
        sys.exit("--iters must be >= 20")
    if not a.names:
        sys.exit("run needs problem names")
    job = pick_job(a.job)
    print("using job", job)
    for n in a.names:
        run_one(job, n, a)


if __name__ == "__main__":
    main()
