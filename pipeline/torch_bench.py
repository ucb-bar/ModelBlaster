#!/usr/bin/env python3
"""The PyTorch side of ModelBlaster's comparison against torch (``torch_baseline``).

Self-contained (torch + numpy only): ``torch_baseline`` ships this file with a
bundle -- the model's KernelBench source, ``recipe.json``, ``state.npz`` (the
module's state dict, mapped from the IR's ``weights.npz``) and the IR's
``io.npz`` -- to the GPU host and runs it there as a long-lived server, one per
math mode (``--tf32 0|1``: TF32 is a process-wide setting and torch.compile
bakes it into its kernels):

    python torch_bench.py BUNDLE --tf32 0 [--variants eager,graph,compile_mna,compile_ro] [--lock FILE]

On start it builds every variant on the device ``CUDA_VISIBLE_DEVICES`` names,
checks each one's outputs against the reference in ``io.npz`` with the
driver's tolerance (``max|out - ref| <= out_rel * max|ref| + out_abs`` per
output, matching non-finite values exact), and prints one ``TB_JSON`` line:
``{"event": "ready", "device": {uuid, name, cc}, "versions": {...},
"variants": {name: {"ok", "check", "error", "setup_s"}}}``.

Then one JSON request per stdin line:

- ``{"cmd": "time", "warm": W, "min_reps": 3, "max_reps": 200, "budget_ms": 200, "thru_n": N}``:
  one timing block over the variants that passed, measured exactly as the
  ModelBlaster driver measures its bodies (``time_multi`` + ``MB_WALL``):
  ``W`` warm passes of every variant; then, round robin, each rep a CUDA event
  pair on the default (legacy) stream around one forward pass, synchronised
  before the next (the GPU idle), until every variant has had ``budget_ms`` or
  ``max_reps`` (at least ``min_reps``); then as many host wall-clock reps
  (``perf_counter`` from issuing the forward pass to the return of
  ``torch.cuda.synchronize()``), round robin; then the steady-state throughput
  (one warm pass, ``N`` passes back to back, one sync, per pass). Holds the
  GPU's flock (``--lock``, KernelBlaster's lock file) for the block.
  -> ``TB_JSON {"event": "timing", "variants": {name: {"gpu": {med, mean, min, max, std, n},
  "wall": {...}, "thru_wall_us", "thru_gpu_us", "thru_n"}}}`` (us; med = upper median, as the driver).
- ``{"cmd": "quit"}`` (or EOF): exit.

Variants: ``eager`` (the module under ``torch.inference_mode``), ``graph`` (the
forward captured once with ``torch.cuda.graph`` on static inputs, replayed),
``compile_mna`` (``torch.compile(mode="max-autotune-no-cudagraphs")``),
``compile_ro`` (``torch.compile(mode="reduce-overhead")``: inductor + CUDA
graphs). An fp16 bundle (``recipe.json`` ``dtype``) runs the module in half. cuDNN autotuning on (``cudnn.benchmark``), deterministic off; a
variant that fails to build or check is reported, not timed.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.util
import json
import math
import os
import sys
import time
import traceback

import numpy as np

VARIANTS = ("eager", "graph", "compile_mna", "compile_ro")


def emit(obj) -> None:
    sys.stdout.write("TB_JSON " + json.dumps(obj) + "\n")
    sys.stdout.flush()


def stats(ms: list) -> dict:
    """us; ``med`` is the upper median (``v[n // 2]``), as the driver's ``dist``."""
    v = sorted(1000.0 * x for x in ms)
    n = len(v)
    if not n:
        return {"n": 0}
    mean = sum(v) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    return {"med": v[n // 2], "mean": mean, "min": v[0], "max": v[-1], "std": std, "n": n}


def max_err(a: np.ndarray, r: np.ndarray) -> float:
    """The driver's ``max_err``: matching values (and NaN vs NaN) exact, any other non-finite inf."""
    a = a.astype(np.float64).ravel()
    r = r.astype(np.float64).ravel()
    same = (a == r) | (np.isnan(a) & np.isnan(r))
    a, r = a[~same], r[~same]
    if a.size == 0:
        return 0.0
    if not (np.isfinite(a).all() and np.isfinite(r).all()):
        return math.inf
    return float(np.abs(a - r).max())


def check_outputs(outs, ref_flat: np.ndarray, spec: list, tol: dict) -> dict:
    """Every output against its slice of io.npz ``output`` (concatenated in order)."""
    res, off, ok = {}, 0, True
    flat = [o.detach().float().cpu().numpy().ravel() for o in outs]
    if len(flat) != len(spec):
        return {"ok": False, "error": f"{len(flat)} outputs, the reference has {len(spec)}"}
    for o, s in zip(flat, spec):
        n = int(s["numel"])
        r = ref_flat[off:off + n]
        off += n
        if o.size != n:
            res[s["name"]] = {"ok": False, "error": f"{o.size} values, the reference {n}"}
            ok = False
            continue
        fin = np.abs(r[np.isfinite(r)])
        scale = float(fin.max()) if fin.size else 0.0
        e, bound = max_err(o, r), tol["out_rel"] * scale + tol["out_abs"]
        res[s["name"]] = {"max_abs_err": e, "bound": bound, "rel_err": e / max(scale, 1e-12), "ok": e <= bound}
        ok &= e <= bound
    return {"ok": ok, "outputs": res}


def as_list(o):
    if isinstance(o, (list, tuple)):
        return [x for y in o for x in as_list(y)]
    return [o]


@contextlib.contextmanager
def gpu_lock(path):
    """KernelBlaster's per-GPU flock (``gpu_lock_wrap``); unlocked when the file cannot be opened."""
    fd = None
    if path:
        try:
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError:
            if fd is not None:
                os.close(fd)
            fd = None
    try:
        yield
    finally:
        if fd is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def load_bundle(root: str):
    import torch

    rec = json.load(open(os.path.join(root, "recipe.json")))
    spec = importlib.util.spec_from_file_location("_tb_model", os.path.join(root, rec["source"]))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for k, v in (rec.get("attrs") or {}).items():
        setattr(mod, k, v)
    init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
    torch.manual_seed(0)
    model = mod.Model(*init)
    st = np.load(os.path.join(root, rec["state"]))
    sd = {k: torch.from_numpy(np.array(st[k])) for k in st.files}
    model.load_state_dict(sd, strict=True)
    if rec.get("dtype") == "float16":             # an fp16 IR: the module in half, like our fp16 storage
        model = model.half()
    io = np.load(os.path.join(root, rec["io"]))
    flat, off, tensors = io["input"], 0, []
    for s in rec["inputs"]:
        n = int(np.prod(s["shape"])) if s["shape"] else 1
        t = torch.from_numpy(np.array(flat[off:off + n], dtype=np.float32).reshape(s["shape"]))
        tensors.append(t.to(getattr(torch, s["dtype"])))
        off += n
    # the forward's arguments: the tensors in order, scalars (bound as constants at extraction) in place
    call = rec.get("call") or [{"t": i} for i in range(len(tensors))]
    args = [tensors[c["t"]] if "t" in c else c["v"] for c in call]
    return rec, model, args, np.asarray(io["output"], dtype=np.float32)


def device_info():
    import torch

    p = torch.cuda.get_device_properties(0)
    u = str(getattr(p, "uuid", "") or "")
    return {"name": p.name, "uuid": ("GPU-" + u) if u and not u.startswith("GPU-") else u,
            "cc": f"{p.major}.{p.minor}", "sms": p.multi_processor_count,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def versions():
    import torch

    v = {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
         "python": sys.version.split()[0]}
    try:
        import triton
        v["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        v["triton"] = None
    try:
        import ctypes
        n = ctypes.c_int()
        ctypes.CDLL("libcuda.so.1").cuDriverGetVersion(ctypes.byref(n))
        v["cuda_driver_api"] = n.value
    except Exception:  # noqa: BLE001
        v["cuda_driver_api"] = None
    try:
        v["driver"] = open("/proc/driver/nvidia/version").readline().split("Kernel Module")[-1].split()[0]
    except Exception:  # noqa: BLE001
        v["driver"] = None
    return v


def build(name: str, model, inputs):
    """-> a callable running one forward pass (outputs left on the device), and its outputs."""
    import torch

    if name == "eager":
        def f():
            with torch.inference_mode():
                return model(*inputs)
        return f, f()
    if name == "graph":
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.inference_mode():
            for _ in range(3):
                model(*inputs)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(g):
            out = model(*inputs)
        g.replay()
        return g.replay, out
    if name in ("compile_mna", "compile_ro"):
        mode = "max-autotune-no-cudagraphs" if name == "compile_mna" else "reduce-overhead"
        c = torch.compile(model, mode=mode)
        mark = getattr(torch.compiler, "cudagraph_mark_step_begin", None) if name == "compile_ro" else None

        def f():
            if mark is not None:
                mark()
            with torch.inference_mode():
                return c(*inputs)
        for _ in range(3):
            out = f()
        # reduce-overhead may hand back graph-owned buffers: a copy for the check
        return f, [t.clone() for t in as_list(out)] if mark is not None else out
    raise ValueError(f"unknown variant {name}")


def time_block(fs: dict, warm: int, min_reps: int, max_reps: int, budget_ms: float, thru_n: int | None) -> dict:
    import torch

    names = list(fs)
    n = len(names)
    st = torch.cuda.current_stream()
    ev = {k: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for k in names}
    with torch.inference_mode():
        for _ in range(warm):
            for k in names:
                fs[k]()
        torch.cuda.synchronize()
        gpu = {k: [] for k in names}
        total = 0.0
        while len(gpu[names[0]]) < min_reps or (total < n * budget_ms and len(gpu[names[0]]) < max_reps):
            for k in names:
                a, b = ev[k]
                a.record(st)
                fs[k]()
                b.record(st)
                b.synchronize()
            for k in names:
                t = ev[k][0].elapsed_time(ev[k][1])
                gpu[k].append(t)
                total += t
        reps = len(gpu[names[0]])
        wall = {k: [] for k in names}
        torch.cuda.synchronize()
        pc = time.perf_counter
        for _ in range(reps):
            for k in names:
                t0 = pc()
                fs[k]()
                torch.cuda.synchronize()
                wall[k].append((pc() - t0) * 1e3)
        thru = int(thru_n or max(10, reps))
        out = {}
        for k in names:
            f = fs[k]
            a, b = ev[k]
            f()
            torch.cuda.synchronize()
            t0 = pc()
            a.record(st)
            for _ in range(thru):
                f()
            b.record(st)
            b.synchronize()
            torch.cuda.synchronize()
            tw = (pc() - t0) * 1e3
            out[k] = {"gpu": stats(gpu[k]), "wall": stats(wall[k]), "thru_wall_us": 1000.0 * tw / thru,
                      "thru_gpu_us": 1000.0 * a.elapsed_time(b) / thru, "thru_n": thru}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle")
    ap.add_argument("--tf32", type=int, choices=[0, 1], default=0)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--lock", default=None, help="the GPU's flock file, held for each timing block")
    a = ap.parse_args(argv)

    import torch

    tf32 = bool(a.tf32)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32           # torch's default is True: set it either way
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    ready = {"event": "ready", "tf32": tf32, "device": None, "versions": None, "variants": {}}
    try:
        ready["device"], ready["versions"] = device_info(), versions()
        rec, model, inputs, ref = load_bundle(a.bundle)
        model = model.cuda().eval()
        inputs = [t.cuda() if torch.is_tensor(t) else t for t in inputs]
        tol = rec["tolerance"]
    except Exception as e:  # noqa: BLE001
        ready["error"] = f"{type(e).__name__}: {e}"
        ready["traceback"] = traceback.format_exc()[-3000:]
        emit(ready)
        return 1
    ready["flags"] = {"allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                      "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
                      "float32_matmul_precision": torch.get_float32_matmul_precision(),
                      "cudnn_benchmark": torch.backends.cudnn.benchmark,
                      "cudnn_deterministic": torch.backends.cudnn.deterministic, "grad": "inference_mode"}
    fs = {}
    with gpu_lock(a.lock):                            # cuDNN autotuning / inductor autotuning time kernels too
        for name in [v for v in a.variants.split(",") if v]:
            t0 = time.time()
            info = {"ok": False}
            try:
                f, out = build(name, model, inputs)
                torch.cuda.synchronize()
                info["check"] = check_outputs(as_list(out), ref, rec["outputs"], tol)
                info["ok"] = bool(info["check"]["ok"])
                if info["ok"]:
                    fs[name] = f
                else:
                    info["error"] = "output check failed"
            except Exception as e:  # noqa: BLE001 -- reported, never hidden
                info["error"] = f"{type(e).__name__}: {str(e)[:1500]}"
                torch.cuda.synchronize()
            info["setup_s"] = round(time.time() - t0, 2)
            ready["variants"][name] = info
    emit(ready)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            emit({"event": "error", "error": f"bad request {line[:200]!r}"})
            continue
        if req.get("cmd") == "quit":
            break
        if req.get("cmd") == "time":
            try:
                with gpu_lock(a.lock):
                    t0 = time.time()
                    res = time_block(fs, int(req.get("warm", 2)), int(req.get("min_reps", 3)),
                                     int(req.get("max_reps", 200)), float(req.get("budget_ms", 200.0)),
                                     req.get("thru_n"))
                emit({"event": "timing", "variants": res, "seconds": round(time.time() - t0, 3),
                      "device": device_info()})
            except Exception as e:  # noqa: BLE001
                emit({"event": "error", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]})
        else:
            emit({"event": "error", "error": f"unknown cmd {req.get('cmd')!r}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
