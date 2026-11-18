#!/usr/bin/env python3
"""E8 interleaved benchmark (runs ON the GPU node, one process per (problem, session)).

    python e8_bench.py CFG.json   -> prints one line @@E8@@{json}

All implementations of one KernelBench problem live in this one process, on the same inputs, and are timed by the
SAME loop, interleaved round by round (impl order shuffled per round):
  torch_eager   the original Model, fp16, eval/no_grad (events around model(*xs))
  torch_graph   the same Model captured in a torch.cuda.CUDAGraph (3 side-stream warm-ups, then capture), replayed
  flow          the rebuilt flow final program (kb_export .so: kept slot sources, per-slot units, PTDS), captured
                with e8_shim.e8_capture exactly as the flow's driver (cudaStreamPerThread, thread-local capture of
                launch_gpu_implementation), replayed on PyTorch's current stream
  <solution>    a Claude Code solution.py (ModelNew, state dict loaded strictly from the Model), captured like
                torch_graph (= baseline_cc/remote_bench.py graph_time)
Model construction and inputs are exactly remote_bench.py / torch_posthoc.py: torch.manual_seed(0) before
get_init_inputs + Model (default dtype fp16 while constructing), .cuda().half().eval(), plain tensor attributes moved;
torch.manual_seed(0) before get_inputs. The flow's buffers get THESE tensors (inputs by order, w:<state key> weights,
<rnn>_l<k>[_reverse]__{w_ih,w_hh,b_ih,b_hh} consts, BatchNorm scale / bias_fused derived from the BN module); any
other buffer (constants like a causal-mask slice) keeps the regenerated data.bin value. Every implementation's output
is checked against the fp32-math reference on the same fp16 weights/inputs (TF32 off), with the flow's fp16
tolerance (max|y - ref| <= 1e-2 * max|ref| + 1e-5).
Timing block (one impl, one round): `warm` launches, then n launches each bracketed by its own cudaEvent pair on the
current stream and synchronized (as the flow's time_multi and remote_bench's timeit); n = clamp(budget_ms / t, nmin,
nmax). Per block: median (the statistic every number in the sweep reports), mean, min, max, stdev.
"""
import ctypes, importlib.util, json, math, os, random, statistics as st, subprocess, sys, time, traceback

cfg = json.load(open(sys.argv[1]))
import numpy as np
import torch

HALF = torch.float16
OUT = {"name": cfg["name"], "host": os.uname().nodename, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
       "cfg": {k: v for k, v in cfg.items() if k != "impls"}, "impls": {}, "blocks": [], "errors": {}}


def smi(q="uuid,serial,pci.bus_id,clocks.sm,clocks.mem,clocks.max.sm,power.draw,temperature.gpu,"
          "clocks_throttle_reasons.active,pstate,utilization.gpu"):
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        return dict(zip(q.split(","), [x.strip() for x in r.stdout.strip().split(",")]))
    except Exception as e:
        return {"error": repr(e)}


def others():
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True)
        return [l.strip() for l in r.stdout.splitlines() if l.strip() and not l.startswith(str(os.getpid()) + ",")]
    except Exception:
        return None


OUT["smi_start"] = smi()
OUT["others_start"] = others()


def load(path, modname):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def to_dev(mod):
    for sub in mod.modules():
        for k, v in list(vars(sub).items()):
            if torch.is_tensor(v) and v.is_floating_point() and k not in ("_parameters", "_buffers"):
                setattr(sub, k, v.cuda().half())
    return mod


def flat(y):
    return [y] if torch.is_tensor(y) else [t for z in y for t in flat(z)] if isinstance(y, (list, tuple)) else []


def compare(ys, rs):
    ys, rs = flat(ys), flat(rs)
    if len(ys) != len(rs):
        return {"ok": False, "error": f"{len(ys)} outputs vs {len(rs)}"}
    allok, worst = True, None
    for o, r in zip(ys, rs):
        if o.numel() != r.numel():
            return {"ok": False, "error": f"numel {o.numel()} vs {r.numel()}"}
        of, rf = o.detach().float().reshape(-1), r.detach().float().reshape(-1)
        fin = torch.isfinite(rf)
        scale = float(rf[fin].abs().max()) if fin.any() else 0.0
        bad = bool((fin & ~torch.isfinite(of)).any())
        d = torch.where(fin, (of - rf).abs(), torch.zeros_like(rf))
        e = float(d.max()) if d.numel() else 0.0
        bound = 1e-2 * scale + 1e-5
        allok &= (e <= bound and not bad)
        q = {"max_abs_err": e, "bound": bound, "err_over_bound": e / bound, "nonfinite": bad}
        if worst is None or q["err_over_bound"] > worst["err_over_bound"]:
            worst = q
    return {"ok": allok, **(worst or {})}


# ---------------------------------------------------------------- model, inputs, reference
m = load(cfg["problem"], "kb_ref_problem")
torch.manual_seed(0)
init = m.get_init_inputs()
torch.set_default_dtype(HALF)
model = to_dev(m.Model(*init).cuda().half().eval())
torch.set_default_dtype(torch.float32)
ref_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
torch.manual_seed(0)
xs = m.get_inputs()
xs = list(xs) if isinstance(xs, (list, tuple)) else [xs]
xs = [x.cuda().half() if torch.is_tensor(x) and x.is_floating_point() else (x.cuda() if torch.is_tensor(x) else x) for x in xs]
import copy
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
with torch.no_grad():
    ref32 = copy.deepcopy(model).float()
    for sub in ref32.modules():
        for k, v in list(vars(sub).items()):
            if torch.is_tensor(v) and v.is_floating_point() and k not in ("_parameters", "_buffers"):
                setattr(sub, k, v.float())
    torch.manual_seed(0)
    REF = [t.detach().clone() for t in flat(ref32(*[x.float() if torch.is_tensor(x) and x.is_floating_point() else x for x in xs]))]
    del ref32
torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = False   # PyTorch defaults (as torch_posthoc.py / remote_bench.py)
torch.backends.cudnn.allow_tf32 = True
OUT["inputs"] = [list(x.shape) if torch.is_tensor(x) else repr(x) for x in xs]

stream = torch.cuda.current_stream()
FNS, KEEP = {}, {}


def torch_capture(fn, args):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(*args)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        yg = fn(*args)
    return g, yg


# ---------------------------------------------------------------- impls
def add_torch():
    with torch.no_grad():
        y = model(*xs); torch.cuda.synchronize()
        OUT["impls"]["torch_eager"] = {"check": compare(y, REF)}
        FNS["torch_eager"] = lambda: model(*xs)
        g, yg = torch_capture(model, xs)
        g.replay(); torch.cuda.synchronize()
        OUT["impls"]["torch_graph"] = {"check": compare(yg, REF)}
        KEEP["torch_graph"] = (g, yg)
        FNS["torch_graph"] = g.replay


def flow_buffers(man, data_path):
    """torch tensors for every manifest buffer: data.bin values, then the seed-0 torch tensors where mappable."""
    import re
    esz = {"f16": 2, "f32": 4, "i32": 4, "i64": 8}
    tdt = {"f16": torch.float16, "f32": torch.float32, "i32": torch.int32, "i64": torch.int64}
    blob = np.memmap(data_path, dtype=np.uint8, mode="r")
    sd = model.state_dict()
    mods = dict(model.named_modules())
    bufs, src = [None] * len(man["buffers"]), {}
    inputs_t = [x for x in xs if torch.is_tensor(x)]
    in_bufs = [i for i, b in enumerate(man["buffers"]) if b["kind"] == "input"]

    def mapped(i, b):
        n, kind = b["name"], b["kind"]
        if kind == "input" and len(in_bufs) == len(inputs_t):
            return inputs_t[in_bufs.index(i)], "input"
        if n.startswith("w:"):
            k = n[2:]
            if k in sd:
                return sd[k], "state_dict"
            mm = re.match(r"(.*)\.(scale|bias_fused)$", k)
            if mm and mm.group(1) in mods and isinstance(mods[mm.group(1)], torch.nn.modules.batchnorm._BatchNorm):
                bn = mods[mm.group(1)]
                scale = bn.weight.float() / torch.sqrt(bn.running_var.float() + bn.eps)
                return (scale if mm.group(2) == "scale" else bn.bias.float() - bn.running_mean.float() * scale), "bn_derived"
        mm = re.match(r"^(\w+?)_l(\d+)(_reverse)?__(w_ih|w_hh|b_ih|b_hh)$", n)
        if mm:
            key = f"{mm.group(1).replace('_', '.')}.{'weight' if mm.group(4)[0] == 'w' else 'bias'}_{mm.group(4)[2:]}_l{mm.group(2)}{mm.group(3) or ''}"
            if key in sd:
                return sd[key], "rnn_state_dict"
        return None, None

    for i, b in enumerate(man["buffers"]):
        if b["alias"]:
            continue
        t = torch.zeros(max(b["numel"], 1), dtype=tdt[b["dtype"]], device="cuda")
        if b.get("ref") and b["kind"] in ("input", "const", "weight"):
            o, nbytes = b["ref"]
            t.copy_(torch.from_numpy(np.array(blob[o:o + nbytes]).view(np.dtype({"f16": "<f2", "f32": "<f4", "i32": "<i4", "i64": "<i8"}[b["dtype"]]))))
            src[b["name"]] = "data.bin"
            v, how = mapped(i, b)
            if v is not None:
                if v.numel() != b["numel"]:
                    src[b["name"]] = f"data.bin (mapping {how} numel {v.numel()} != {b['numel']})"
                else:
                    t.copy_(v.detach().reshape(-1).to(t.dtype))
                    src[b["name"]] = how
        bufs[i] = t
    for i, b in enumerate(man["buffers"]):
        if b["alias"]:
            bufs[i] = bufs[b["alias"]["base"]]   # pointer arithmetic below
    ptrs = []
    for i, b in enumerate(man["buffers"]):
        p = bufs[i].data_ptr()
        if b["alias"]:
            p += b["alias"]["offset"] * esz[b["dtype"]]
        ptrs.append(p)
    del blob
    return bufs, ptrs, src


def add_flow(fc):
    man = json.load(open(fc["manifest"]))
    lib = ctypes.CDLL(fc["so"], mode=ctypes.RTLD_GLOBAL)
    shim = ctypes.CDLL(fc["shim"])
    shim.e8_capture.restype = ctypes.c_void_p
    shim.e8_capture.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    shim.e8_replay.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    bufs, ptrs, src = flow_buffers(man, fc["data"])
    arr = (ctypes.c_void_p * len(ptrs))(*ptrs)
    launch = ctypes.cast(lib.mb_so_launch, ctypes.c_void_p)
    torch.cuda.synchronize()
    lib.mb_so_launch.argtypes = [ctypes.c_void_p]
    for _ in range(2):                         # eager: lazy slot scratch, as the driver's checks before the graph
        lib.mb_so_launch(arr)
    torch.cuda.synchronize()
    outs = [bufs[i] for i in man["outputs"]]
    chk_eager = compare(outs, REF) if len(outs) == len(REF) else compare(outs[:len(REF)], REF)
    nodes = ctypes.c_int(0)
    ge = shim.e8_capture(launch, arr, ctypes.byref(nodes))
    if not ge:
        raise RuntimeError("flow graph capture failed")
    for o in outs:
        o.zero_()
    torch.cuda.synchronize()
    shim.e8_replay(ge, ctypes.c_void_p(stream.cuda_stream)); torch.cuda.synchronize()
    chk = compare(outs, REF) if len(outs) == len(REF) else compare(outs[:len(REF)], REF)
    nsrc = {}
    for v in src.values():
        nsrc[v] = nsrc.get(v, 0) + 1
    OUT["impls"]["flow"] = {"check": chk, "check_eager": chk_eager, "graph_nodes": nodes.value, "buffer_sources": nsrc,
                            "databin_buffers": [k for k, v in src.items() if v.startswith("data.bin")][:20]}
    KEEP["flow"] = (lib, shim, arr, bufs, ge)
    s = ctypes.c_void_p(stream.cuda_stream)
    FNS["flow"] = lambda: shim.e8_replay(ge, s)


def add_solution(name, path, extdir):
    os.environ["TORCH_EXTENSIONS_DIR"] = extdir
    os.makedirs(extdir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(os.path.dirname(path))
    try:
        sol = load(path, "solution_" + name)
    finally:
        os.chdir(cwd)
    torch.manual_seed(0)
    init2 = m.get_init_inputs()
    torch.set_default_dtype(HALF)
    new = sol.ModelNew(*init2)
    torch.set_default_dtype(torch.float32)
    new.load_state_dict(ref_state, strict=True)
    new = to_dev(new.cuda().half().eval())
    with torch.no_grad():
        torch.manual_seed(0)
        y = new(*xs); torch.cuda.synchronize()
        chk_e = compare(y, REF)
        xs0 = [x.clone() if torch.is_tensor(x) else x for x in xs]
        g, yg = torch_capture(new, xs0)
        g.replay(); torch.cuda.synchronize()
        chk = compare(yg, REF)
    OUT["impls"][name] = {"check": chk, "check_eager": chk_e, "path": path}
    KEEP[name] = (sol, new, g, yg, xs0)
    FNS[name] = g.replay


t0 = time.time()
for name, ic in cfg["impls"].items():
    try:
        if ic["kind"] == "torch":
            add_torch()
        elif ic["kind"] == "flow":
            add_flow(ic)
        elif ic["kind"] == "solution":
            add_solution(name, ic["path"], ic["extdir"])
    except Exception:
        OUT["errors"][name] = traceback.format_exc()[-2500:]
OUT["setup_s"] = round(time.time() - t0, 1)
torch.cuda.synchronize()


# ---------------------------------------------------------------- interleaved timing
def block(fn, warm, budget_ms, nmin, nmax):
    with torch.no_grad():
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        est = a.elapsed_time(b)
        n = int(min(nmax, max(nmin, math.ceil(budget_ms / max(est, 1e-3)))))
        ts = []
        for _ in range(n):
            a.record(); fn(); b.record(); b.synchronize()
            ts.append(a.elapsed_time(b) * 1000.0)
    ts.sort()
    return {"med": ts[len(ts) // 2], "median": st.median(ts), "mean": st.fmean(ts), "min": ts[0], "max": ts[-1],
            "std": st.stdev(ts) if len(ts) > 1 else 0.0, "n": n}


names = list(FNS)   # the "torch" impl adds both torch_eager and torch_graph
rng = random.Random(cfg.get("shuffle_seed", 0))
OUT["others_before_timing"] = others()
for r in range(cfg["rounds"]):
    order = names[:]
    rng.shuffle(order)
    for pos, n in enumerate(order):
        ts0 = time.time()
        res = block(FNS[n], cfg.get("warm", 10), cfg.get("budget_ms", 200.0), cfg.get("nmin", 20), cfg.get("nmax", 200))
        res.update(round=r, impl=n, pos=pos, t0=round(ts0, 3), t1=round(time.time(), 3))
        OUT["blocks"].append(res)
    OUT.setdefault("smi_rounds", []).append(smi("clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active"))
OUT["others_end"] = others()
OUT["smi_end"] = smi()
for n in names:
    meds = [b["med"] for b in OUT["blocks"] if b["impl"] == n]
    if meds:
        OUT["impls"].setdefault(n, {})["summary"] = {"n_blocks": len(meds), "mean": st.fmean(meds), "median": st.median(meds),
                                                     "stdev": st.stdev(meds) if len(meds) > 1 else 0.0, "min": min(meds), "max": max(meds)}
print("@@E8@@" + json.dumps(OUT), flush=True)
