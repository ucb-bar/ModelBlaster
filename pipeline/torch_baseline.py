"""The model against PyTorch / cuDNN: the same model, data and GPU, measured the same way.

``kb_model_run`` (``--torch-baseline``) calls :func:`compare` after the model's
own timing: ModelBlaster's composed model and the original PyTorch module are
timed alternately on the device the model run used, so a speedup over torch
is a like-for-like number. See MB ``docs/gpu_harness_perf.md`` ("Against
PyTorch") for the method and its remaining asymmetries. In short:

- **Same model and data.** The IR's KernelBench source (:func:`find_source`)
  at the dims the extractor shrank it to (:func:`sized_module`: the
  extractor's own sizing helpers, the sizing found by matching the IR's input
  shapes), the IR's ``weights.npz`` loaded into its state dict
  (:func:`map_state`; batchnorm folded at extraction becomes weight = scale,
  bias = folded bias, mean 0, var 1 - eps), the inputs of ``io.npz``, fp32.
  The outputs are checked against ``io.npz``'s reference with the driver's
  tolerance -- on the CPU when the bundle is built and for every variant on the
  GPU.
- **Same device.** The torch servers run with ``CUDA_VISIBLE_DEVICES=<uuid>``
  of the GPU the model was timed on, our side is pinned to it
  (``KBEval.for_device``), and every block's reported UUID is compared: any
  other device is a :class:`DeviceMismatch`.
- **Same measurement.** Both sides time with the driver's ``time_multi`` rule
  (warm-up, then reps of one CUDA event pair on an idle GPU until a budget,
  upper median) and, with ``MB_WALL=1`` / ``torch_bench``, the host wall clock
  per forward pass (issue -> device synchronize, inputs and outputs resident,
  no allocation or copies) and the steady-state throughput (N back to back, one
  sync). Blocks alternate ABBA over ``rounds`` (ours, torch TF32 off, torch TF32
  on, then the reverse) in one session window, every block holding the GPU's
  flock.
- **Same math.** torch is timed with TF32 off and on (one server per mode);
  :func:`detect_math` reads our slot sources (TF32 / tensor-core / reduced
  precision markers). The headline is against TF32 off when ours is plain fp32,
  else against the faster torch mode.

    python -m modelblaster.pipeline.torch_baseline bundle IR_DIR --out DIR [--source FILE]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

FORMAT = "modelblaster.torch_baseline/v1"
VARIANTS = ("eager", "graph", "compile_mna", "compile_ro")
LABELS = {"eager": "torch eager", "graph": "torch CUDA graph", "compile_mna": "torch.compile max-autotune-no-cudagraphs",
          "compile_ro": "torch.compile reduce-overhead"}
BENCH = Path(__file__).with_name("torch_bench.py")
# the extractor's sizing flags (target io MiB, target weight MiB, target GFLOPs), tried in order:
# the ir16 sets were extracted with 16 / 64 (31_VisionAttention with 20 GFLOPs), older ones 64 / 256
SIZINGS = [(16, 64, 0), (16, 64, 20), (64, 0, 0), (16, 0, 0), (256, 0, 0), (4, 0, 0), (0, 0, 0)]


class DeviceMismatch(RuntimeError):
    """A side of the comparison ran on another GPU than the model run's."""


class Skip(RuntimeError):
    """The comparison cannot run here (no source, no torch, weights that do not map ...)."""


def norm_uuid(u) -> str:
    u = str(u or "").strip().lower()
    return u if not u or u.startswith("gpu-") else "gpu-" + u


# ---------------------------------------------------------------- the torch model

def find_source(ir_dir, roots=()) -> Path | None:
    """The KernelBench / SOL source the IR was extracted from: ``<stem>.py`` under ``roots``,
    ``$MB_TORCH_SRC`` (``os.pathsep``-separated) or the ``adapted`` / ``src`` trees next to
    the IR set (``.../levelN/ir16/<stem>`` -> ``.../levelN/src/**/<stem>.py``)."""
    ir_dir = Path(ir_dir).resolve()
    stem = ir_dir.name
    names = [stem] + ([stem[3:]] if stem.startswith("kb_") else [])
    cand = [Path(r) for r in roots or []]
    cand += [Path(r) for r in (os.environ.get("MB_TORCH_SRC") or "").split(os.pathsep) if r]
    for up in list(ir_dir.parents)[:3]:
        cand += [up / "adapted", up / "src"]
    for r in cand:
        if r.is_file() and r.stem in names:
            return r
        if r.is_dir():
            for n in names:
                hit = sorted(r.rglob(f"{n}.py"))
                if hit:
                    return hit[0]
    return None


def _import(path: Path, name: str = "_mb_torch_src"):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _shape_attrs(mod) -> dict:
    return {k: v for k, v in vars(mod).items() if not k.startswith("_") and
            ((isinstance(v, int) and not isinstance(v, bool)) or
             (isinstance(v, (list, tuple)) and all(isinstance(x, int) and not isinstance(x, bool) for x in v)))}


def sized_module(path: Path, sizing: tuple) -> tuple:
    """The source module at the dims ``extract_graph --bench-file`` gives it with
    ``--bench-target-mb / --bench-target-weights-mb / --bench-target-gflops`` = ``sizing``
    (the extractor's own sizing helpers) -> (module, {attr: value} changed from the source)."""
    from . import extract_graph as eg

    mb, wmb, gfl = sizing
    mod = _import(path)
    base = _shape_attrs(mod)
    with eg._cpu_fp32_tensor_creation():
        tb = mb * 2 ** 20 if mb else None
        if gfl:
            eg._shrink_for_flops(mod, int(gfl * 1e9), tb)
        elif tb:
            eg._shrink_for_utilization(mod, tb)
        if wmb:
            eg._shrink_for_weights(mod, wmb * 2 ** 20)
    now = _shape_attrs(mod)
    return mod, {k: (list(v) if isinstance(v, tuple) else v) for k, v in now.items() if base.get(k) != v}


def map_state(model, weights: dict) -> tuple[dict, dict]:
    """The module's state dict from the IR's ``weights.npz`` -> (state, report). By name
    (``<qualname>``, ``const.<qualname>`` for bare parameters), transposed when only that
    fits; a BatchNorm the extractor folded (``<bn>.scale`` / ``<bn>.bias_fused``) becomes
    weight = scale, bias = folded bias, running mean 0 and var 1 - eps (the same affine map).
    ``report``: mapped / defaulted / missing keys and unused IR weights."""
    import numpy as np
    import torch

    sd = model.state_dict()
    out, used, rep = {}, set(), {"mapped": [], "bn_folded": [], "transposed": [], "kept": [], "missing": []}
    bns = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)}

    def take(key, t):
        w = weights.get(key)
        if w is None:
            return None
        w = np.asarray(w)
        if tuple(w.shape) == tuple(t.shape):
            return w
        if w.size == t.numel() and w.ndim == 2 and t.ndim == 2 and tuple(w.shape[::-1]) == tuple(t.shape):
            rep["transposed"].append(key)
            return w.T
        if w.size == t.numel():
            return w.reshape(tuple(t.shape))
        return None

    for k, t in sd.items():
        prefix, _, leaf = k.rpartition(".")
        w = None
        for key in (k, "const." + k):
            w = take(key, t)
            if w is not None:
                used.add(key)
                rep["mapped"].append(k)
                break
        if w is None and prefix in bns:
            bn = bns[prefix]
            src = {"weight": [prefix + ".scale"], "bias": [prefix + ".bias_fused"]}.get(leaf, [])
            for key in src:
                w = take(key, t)
                if w is not None:
                    used.add(key)
                    break
            if w is None and leaf == "running_mean" and (prefix + ".scale") in weights:
                w = np.zeros(tuple(t.shape), np.float32)
            if w is None and leaf == "running_var" and (prefix + ".scale") in weights:
                w = np.full(tuple(t.shape), 1.0 - bn.eps, np.float32)
            if w is not None:
                rep["bn_folded"].append(k)
        if w is None and not t.is_floating_point():
            out[k] = t.clone()                         # num_batches_tracked and other counters
            rep["kept"].append(k)
            continue
        if w is None:
            rep["missing"].append(k)
            continue
        out[k] = torch.from_numpy(np.ascontiguousarray(w)).to(t.dtype)
    rep["unused_ir_weights"] = sorted(set(weights) - used)
    return out, rep


def max_err(a, r) -> float:
    import numpy as np

    a = np.asarray(a, np.float64).ravel()
    r = np.asarray(r, np.float64).ravel()
    same = (a == r) | (np.isnan(a) & np.isnan(r))
    a, r = a[~same], r[~same]
    if a.size == 0:
        return 0.0
    if not (np.isfinite(a).all() and np.isfinite(r).all()):
        return float("inf")
    return float(np.abs(a - r).max())


def check_flat(outs: list, ref, spec: list, tol: dict) -> dict:
    """``torch_bench.check_outputs`` on numpy arrays (the CPU check of a bundle)."""
    import numpy as np

    res, off, ok = {}, 0, True
    if len(outs) != len(spec):
        return {"ok": False, "error": f"{len(outs)} outputs, the reference has {len(spec)}"}
    for o, s in zip(outs, spec):
        o = np.asarray(o, np.float32).ravel()
        n = int(s["numel"])
        r = np.asarray(ref[off:off + n], np.float32)
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


def tolerance(problem: Path | None, graph: dict) -> dict:
    """The driver's output tolerance (the problem manifest's ``output_rel`` / ``output_abs``)."""
    tol = {}
    if problem is not None and (Path(problem) / "manifest.json").exists():
        tol = json.loads((Path(problem) / "manifest.json").read_text()).get("tolerance") or {}
    if not tol:                                   # no problem: the exporter's fp32 default
        from .kb_export import DTYPE_TOL
        from .cuda_dtypes import FP16, FP32
        tol = DTYPE_TOL[FP16 if graph.get("quant") == "fp16" else FP32]
    return {"out_rel": float(tol.get("output_rel", 1e-3)), "out_abs": float(tol.get("output_abs", 1e-5))}


def _key(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
    return h.hexdigest()[:12]


def build_bundle(ir_dir, out, *, source=None, tol: dict, sizings=None, cpu_check: bool = True) -> dict:
    """The torch side's inputs for ``torch_bench.py`` in ``out``: the source, ``recipe.json``
    (dims, inputs, outputs, tolerance), ``state.npz``, ``io.npz`` -> the recipe (with
    ``key``, ``sizing``, ``mapping``, ``cpu_check``). Reused when ``out`` has the same key.
    :class:`Skip` when no sizing reproduces the IR's inputs or the outputs do not check."""
    import numpy as np
    import torch

    ir_dir, out = Path(ir_dir), Path(out)
    src = Path(source) if source else find_source(ir_dir)
    if src is None or not src.exists():
        raise Skip(f"no PyTorch source for {ir_dir.name} (--torch-source FILE or $MB_TORCH_SRC)")
    g = json.loads((ir_dir / "graph.json").read_text())
    W = np.load(ir_dir / "weights.npz")
    weights = {k: W[k] for k in W.files}
    io = np.load(ir_dir / "io.npz")
    flat_in, flat_out = np.asarray(io["input"], np.float32), np.asarray(io["output"], np.float32)
    key = _key(src.read_bytes(), (ir_dir / "graph.json").read_bytes(), (ir_dir / "weights.npz").stat().st_size,
               (ir_dir / "io.npz").stat().st_size, json.dumps(tol, sort_keys=True), json.dumps(sizings or SIZINGS))
    if (out / "recipe.json").exists():
        rec = json.loads((out / "recipe.json").read_text())
        if rec.get("key") == key and (out / rec["state"]).exists():
            return rec
    outs = [o for o in (g.get("output", {}).get("tensors") or [g["output"]["tensor"]])]
    spec_out = [{"name": o, "shape": g["tensors"][o]["shape"],
                 "numel": int(np.prod(g["tensors"][o]["shape"])) if g["tensors"][o]["shape"] else 1} for o in outs]
    tried = []
    for sz in sizings or SIZINGS:
        try:
            mod, attrs = sized_module(src, tuple(sz))
            from . import extract_graph as eg
            with eg._cpu_fp32_tensor_creation():
                ins = mod.get_inputs()
                init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
                torch.manual_seed(0)
                model = mod.Model(*init)
        except Exception as e:  # noqa: BLE001 -- this sizing does not work: the next
            tried.append({"sizing": list(sz), "error": f"{type(e).__name__}: {e}"[:300]})
            continue
        ins = list(ins) if isinstance(ins, (list, tuple)) else [ins]
        tens = [t for t in ins if torch.is_tensor(t)]
        numel = sum(t.numel() for t in tens)
        if numel != flat_in.size:
            tried.append({"sizing": list(sz), "error": f"{numel} input values, io.npz has {flat_in.size}"})
            continue
        state, rep = map_state(model, weights)
        if rep["missing"]:
            tried.append({"sizing": list(sz), "error": f"state keys without IR weights: {rep['missing'][:8]}"})
            continue
        model.load_state_dict(state, strict=True)
        model.eval()
        call, k = [], 0
        for t in ins:
            if torch.is_tensor(t):
                call.append({"t": k})
                k += 1
            elif t is None or isinstance(t, (bool, int, float, str)):
                call.append({"v": t})
            else:
                raise Skip(f"{src.name}: a forward argument of type {type(t).__name__} is not supported")
        # fp16 IRs (extract_graph --quant fp16, docs/fp16_cuda.md): torch runs the module in half
        # (module.half(), float inputs half) -- fp16 storage like ours, torch's own fp16 math
        half = g.get("quant") == "fp16"
        dtypes = ["float16" if half and t.is_floating_point() else str(t.dtype).replace("torch.", "")
                  for t in tens]
        rec = {"format": FORMAT + ":bundle", "key": key, "source": "model.py", "source_path": str(src),
               "ir": str(ir_dir), "ir_name": g.get("name"), "sizing": list(sz), "attrs": attrs,
               "dtype": "float16" if half else "float32",
               "inputs": [{"shape": list(t.shape), "dtype": d} for t, d in zip(tens, dtypes)], "call": call,
               "outputs": spec_out, "tolerance": tol, "state": "state.npz", "io": "io.npz", "mapping": rep,
               "tried": tried}
        if cpu_check:
            off, args = 0, []
            for t, d in zip(tens, dtypes):
                n = t.numel()
                # the CPU check runs in fp32 on the (fp16-rounded) IR data: the golden's own recipe
                args.append(torch.from_numpy(flat_in[off:off + n].reshape(tuple(t.shape)).copy())
                            .to(torch.float32 if d == "float16" else getattr(torch, d)))
                off += n
            argv = [args[c["t"]] if "t" in c else c["v"] for c in call]
            with torch.inference_mode():
                o = model(*argv)
            o = [x for x in (o if isinstance(o, (list, tuple)) else [o])]
            rec["cpu_check"] = check_flat([x.float().numpy() for x in o], flat_out, spec_out, tol)
            if not rec["cpu_check"]["ok"]:
                tried.append({"sizing": list(sz), "error": "CPU output check failed", "check": rec["cpu_check"]})
                continue
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out / "model.py")
        np.savez(out / "state.npz", **{k: v.numpy() for k, v in model.state_dict().items()})
        shutil.copyfile(ir_dir / "io.npz", out / "io.npz")
        (out / "recipe.json").write_text(json.dumps(rec, indent=1, default=str))
        return rec
    raise Skip(f"{src.name}: no sizing reproduces {ir_dir.name}'s inputs and outputs: "
               + "; ".join(f"{t['sizing']}: {t['error']}" for t in tried)[:1500])


# ---------------------------------------------------------------- math mode of our kernels

_TF32 = re.compile(r"tf32|TF32|cvt\.rna\.tf32|__float_to_tf32|CUBLAS_COMPUTE_32F_FAST_TF32|CUBLAS_TF32_TENSOR_OP_MATH")
_TC = re.compile(r"\bwmma::|nvcuda::wmma|\bmma\.sync|\bmma_sync\b|\bwgmma\b|ldmatrix")
_HALF = re.compile(r"\b__half2?\b|\bhalf2\b|__nv_bfloat16|__float2half|__hfma|\b_Float16\b")
_LIB = re.compile(r"\bcublas\w*\(|\bcudnn\w*\(|cublasLt")


def detect_math(texts: dict) -> dict:
    """What math our slot sources use: ``tf32`` (TF32 tensor cores / cuBLAS TF32 modes),
    ``tensor_cores`` (wmma / mma without a TF32 marker: reduced-precision inputs),
    ``reduced`` (fp16 / bf16 arithmetic), else ``fp32``; ``libraries``: slots calling
    cuBLAS / cuDNN (their math mode is the library's default unless set)."""
    ev = {"tf32": {}, "tensor_cores": {}, "reduced": {}, "libraries": {}}
    for sid, t in sorted((texts or {}).items()):
        t = re.sub(r"//[^\n]*|/\*.*?\*/", "", t, flags=re.S)      # comments do not count
        for k, rx in (("tf32", _TF32), ("tensor_cores", _TC), ("reduced", _HALF), ("libraries", _LIB)):
            hits = sorted(set(m.group(0) for m in rx.finditer(t)))
            if hits:
                ev[k][sid] = hits[:6]
    mode = "tf32" if ev["tf32"] else "tensor_cores" if ev["tensor_cores"] else "reduced" if ev["reduced"] else "fp32"
    return {"mode": mode, "evidence": {k: v for k, v in ev.items() if v}, "slots": sorted(texts or {})}


# ---------------------------------------------------------------- the torch servers

class TorchServer:
    """One ``torch_bench.py`` process (local, or through ssh), JSON lines over stdin / stdout."""

    def __init__(self, argv: list[str], *, env: dict | None = None, log: Path | None = None, label: str = "torch"):
        self.label, self.argv = label, list(argv)
        self.log = open(log, "a") if log else subprocess.DEVNULL
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
                                  bufsize=1, env=env)
        self.q: queue.Queue = queue.Queue()
        self.t = threading.Thread(target=self._read, daemon=True)
        self.t.start()

    def _read(self):
        for line in self.p.stdout:
            if line.startswith("TB_JSON "):
                try:
                    self.q.put(json.loads(line[8:]))
                except ValueError:
                    pass
            elif self.log is not subprocess.DEVNULL:
                self.log.write(line)
                self.log.flush()
        self.q.put(None)

    def wait(self, timeout: float) -> dict:
        try:
            m = self.q.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise TimeoutError(f"{self.label}: no reply in {timeout:.0f} s") from None
        if m is None:
            raise RuntimeError(f"{self.label}: the torch server exited (rc {self.p.wait()}); see its log")
        return m

    def request(self, obj: dict, timeout: float) -> dict:
        self.p.stdin.write(json.dumps(obj) + "\n")
        self.p.stdin.flush()
        return self.wait(timeout)

    def close(self):
        try:
            if self.p.poll() is None:
                self.p.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self.p.stdin.flush()
                self.p.stdin.close()
                self.p.wait(timeout=30)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            self.p.kill()
        if self.log is not subprocess.DEVNULL:
            self.log.close()
            self.log = subprocess.DEVNULL


def lock_file(uuid: str, remote: bool = False) -> str | None:
    """KernelBlaster's per-GPU flock file (``resources.gpu_lock_file``) for ``uuid``."""
    if not uuid or os.getenv("KERNELBLASTER_GPU_FLOCK", "1").lower() in ("0", "false", "no", "off"):
        return None
    d = "/dev/shm" if remote else os.getenv("KERNELBLASTER_GPU_LOCK_DIR", "/dev/shm")
    return f"{d}/kernelblaster_gpu_{re.sub(r'[^A-Za-z0-9_.-]', '_', uuid)}.lock"


def _ssh(host: str, key: str | None) -> list[str]:
    return ["ssh", *(["-i", key] if key else []), "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", host]


def ship(bundle: Path, *, host: str, key: str | None, remote_dir: str) -> str:
    """The bundle and ``torch_bench.py`` into ``remote_dir/bundles/<key>`` -> that path."""
    rec = json.loads((bundle / "recipe.json").read_text())
    rd = f"{remote_dir.rstrip('/')}/bundles/{rec['key']}"
    subprocess.run(_ssh(host, key) + [f"mkdir -p {shlex.quote(rd)} {shlex.quote(remote_dir)}/tmp"], check=True,
                   capture_output=True, timeout=120)
    files = [str(bundle / f) for f in ("model.py", "recipe.json", "state.npz", "io.npz")] + [str(BENCH)]
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", *(["-i", key] if key else []), *files, f"{host}:{rd}/"],
                   check=True, capture_output=True, timeout=1800)
    return rd


def start_servers(bundle: Path, *, uuid: str, modes=(0, 1), variants=VARIANTS, python: str, host: str | None = None,
                  key: str | None = None, remote_dir: str | None = None, work: Path,
                  setup_timeout: float = 3600) -> dict:
    """One torch server per TF32 mode on the GPU ``uuid`` -> {mode: (server, ready message)}."""
    work.mkdir(parents=True, exist_ok=True)
    servers = {}
    rd = ship(bundle, host=host, key=key, remote_dir=remote_dir) if host else None
    lock = lock_file(uuid, remote=bool(host))
    for m in modes:
        args = ["--tf32", str(m), "--variants", ",".join(variants)] + (["--lock", lock] if lock else [])
        if host:
            env = {"CUDA_VISIBLE_DEVICES": uuid, "TMPDIR": f"{remote_dir}/tmp",
                   "TORCHINDUCTOR_CACHE_DIR": f"{remote_dir}/inductor", "PYTHONDONTWRITEBYTECODE": "1"}
            cmd = (f"cd {shlex.quote(rd)} && exec env {' '.join(f'{k}={shlex.quote(v)}' for k, v in env.items())} "
                   f"{shlex.quote(python)} -u torch_bench.py . {' '.join(shlex.quote(a) for a in args)}")
            s = TorchServer(_ssh(host, key) + [cmd], log=work / f"torch_tf32_{m}.log", label=f"torch tf32={m}@{host}")
        else:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": uuid, "TORCHINDUCTOR_CACHE_DIR": str(work / "inductor"),
                   "PYTHONDONTWRITEBYTECODE": "1"}
            s = TorchServer([python, "-u", str(BENCH), str(bundle)] + args, env=env, log=work / f"torch_tf32_{m}.log",
                            label=f"torch tf32={m}")
        servers[m] = s
    out = {}
    try:
        for m, s in servers.items():
            r = s.wait(setup_timeout)
            if r.get("error"):
                raise RuntimeError(f"torch server tf32={m}: {r['error']}")
            got = norm_uuid((r.get("device") or {}).get("uuid"))
            if got != norm_uuid(uuid):
                raise DeviceMismatch(f"the torch server (tf32={m}) runs on {got or '?'}, the model on {uuid}")
            out[m] = (s, r)
    except BaseException:
        for s in servers.values():
            s.close()
        raise
    return out


# ---------------------------------------------------------------- our side

def timing_env(warm: int, budget_ms: float, thru_n: int) -> dict:
    """The driver variables of our comparison runs: wall clock + throughput, the same warm-up,
    budget and throughput count as the torch side; attribution and per-op passes off (they run
    after the timing and are not used)."""
    return {"MB_WALL": "1", "MB_WARM": str(warm), "MB_TIME_MS": str(budget_ms), "MB_THRU_N": str(thru_n),
            "MB_ATTR": "0", "MB_OP_US_PASS": "0"}


def kb_ours(kbeval, problem: Path, *, arch: str, work: Path, gpu_model: str | None, env: dict):
    """Our side through KernelBlaster's evaluator (``time_problems --driver-env``): one
    request per block on the pinned device -> callable(round) -> {ok, timing, device, ...}."""
    def run(k: int) -> dict:
        kr = kbeval.run([{"id": "ours", "problem": str(problem), "sources": None, "reps": 1}], arch=arch,
                        work=work, gpu_model=gpu_model, name=f"torchcmp_r{k:02d}", driver_env=env)
        j = kr["jobs"]["ours"]
        r = (j.get("runs") or [{}])[-1]
        return {"ok": bool(r.get("ok")), "timing": r.get("timing") or {}, "device": r.get("gpu") or j.get("gpu"),
                "e2e_us": r.get("e2e_us"), "basis": r.get("e2e_basis"), "outputs": r.get("outputs"),
                "graph_skip": r.get("graph_skip"), "log": (r.get("log") or "")[-1500:] if not r.get("ok") else ""}
    return run


class BuiltinOurs:
    """Our side in ModelBlaster's own persistent host on this machine's default GPU, one host
    for every block (``--evaluator builtin``)."""

    def __init__(self, problem: Path, *, arch: str, work: Path, env: dict, device: dict | None):
        from .kb_export import DriverHost, build_host, build_so

        host, log = build_host(problem, arch=arch, work=work)
        so, log2 = build_so(problem, arch=arch, work=work, name="cmp.so") if host else (None, "")
        if host is None or so is None:
            raise RuntimeError("builtin host build failed: " + (log if host is None else log2)[-1500:])
        self.h, self.so, self.env, self.device = DriverHost(host), so, env, device

    def __call__(self, k: int) -> dict:
        r = self.h.evaluate(self.so, env={"MB_GRAPH": "1", **self.env})
        return {"ok": bool(r.get("passed")), "timing": r.get("timing") or {}, "device": self.device,
                "e2e_us": r.get("e2e_us"), "basis": r.get("e2e_basis"), "outputs": r.get("outputs"),
                "graph_skip": r.get("graph_skip"), "log": (r.get("log") or "")[-1500:] if not r.get("passed") else ""}

    def close(self):
        self.h.close()


# ---------------------------------------------------------------- the comparison

def _umed(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def _cv(d: dict | None):
    return (d["std"] / d["mean"]) if d and d.get("mean") else None


def aggregate(blocks: list[dict]) -> dict:
    """Per-block numbers of one side and variant -> the reported row: medians over the blocks
    of each block's median (upper medians, as the driver) and of each block's mean, the median
    within-block CV, the spread of the block medians ((max - min) / median), the reps."""
    def agg(side):
        meds = [(b.get(side) or {}).get("med") for b in blocks]
        meds = [m for m in meds if m is not None]
        med = _umed(meds)
        return {"us": med, "mean": _umed([(b.get(side) or {}).get("mean") for b in blocks]),
                "cv": _umed([_cv(b.get(side)) for b in blocks]),
                "spread": ((max(meds) - min(meds)) / med) if meds and med else None,
                "blocks": meds, "reps": sum((b.get(side) or {}).get("n") or 0 for b in blocks)}
    g, w = agg("gpu"), agg("wall")
    # gpu_mean_us / wall_mean_us: the median of the block means -- finer than the medians where
    # the event clock is coarse (1.024 us ticks on the L40S)
    return {"gpu_us": g["us"], "gpu_mean_us": g["mean"], "wall_mean_us": w["mean"], "gpu_cv": g["cv"], "gpu_spread": g["spread"], "gpu_blocks": g["blocks"],
            "wall_us": w["us"], "wall_cv": w["cv"], "wall_spread": w["spread"], "wall_blocks": w["blocks"],
            "thru_wall_us": _umed([b.get("thru_wall_us") for b in blocks]),
            "thru_gpu_us": _umed([b.get("thru_gpu_us") for b in blocks]),
            "reps": g["reps"], "n_blocks": len(blocks)}


def run_rounds(ours, servers: dict, *, rounds: int, uuid: str, params: dict, timeout: float = 3600) -> dict:
    """ABBA blocks: round k runs ours then torch (TF32 off, on), round k+1 the reverse, ...
    -> {"ours": [per-block {body: timing}], "torch": {mode: [per-block {variant: timing}]},
    "order": [...], "devices": [...]}. Every block's device must be ``uuid``."""
    modes = list(servers)
    res = {"ours": [], "ours_runs": [], "torch": {m: [] for m in modes}, "order": [], "devices": []}
    want = norm_uuid(uuid)
    for k in range(rounds):
        seq = ["ours", *modes] if k % 2 == 0 else [*modes[::-1], "ours"]
        for side in seq:
            t0 = time.time()
            if side == "ours":
                r = ours(k)
                got = norm_uuid((r.get("device") or {}).get("uuid"))
                res["ours_runs"].append({k2: v for k2, v in r.items() if k2 != "timing"})
                if got != want:
                    raise DeviceMismatch(f"our block {k} ran on {got or '?'} "
                                         f"({(r.get('device') or {}).get('label')}), not on {uuid}")
                if not r.get("ok"):
                    raise RuntimeError(f"our block {k} failed: {r.get('log') or r.get('outputs')}")
                if not r.get("timing"):
                    raise RuntimeError("our driver printed no MB_TIMING lines (a driver from before MB_WALL, or a "
                                       "KernelBlaster checkout without time_problems --driver-env): re-export")
                res["ours"].append(r["timing"])
            else:
                s = servers[side]
                r = s.request({"cmd": "time", **params}, timeout)
                if r.get("event") != "timing":
                    raise RuntimeError(f"torch tf32={side} block {k}: {r.get('error')}")
                got = norm_uuid((r.get("device") or {}).get("uuid"))
                if got != want:
                    raise DeviceMismatch(f"torch block {k} (tf32={side}) ran on {got or '?'}, not on {uuid}")
                res["torch"][side].append(r["variants"])
            res["order"].append({"round": k, "side": f"tf32_{'on' if side == 1 else 'off'}" if side != "ours" else
                                 "ours", "seconds": round(time.time() - t0, 2)})
            res["devices"].append(got)
    return res


def headline_mode(math: dict, dtype: str = "float32") -> str:
    """Which torch math mode the headline compares with: ``fp16`` for an fp16 IR (torch in
    half, its default tensor-core math; TF32 does not apply), ``tf32_off`` when ours is plain
    fp32 (the same arithmetic), else ``best`` (the faster of TF32 off / on per variant)."""
    if dtype == "float16":
        return "fp16"
    return "tf32_off" if (math or {}).get("mode") == "fp32" else "best"


def build_rows(raw: dict, ready: dict, math: dict, dtype: str = "float32") -> tuple[list[dict], dict]:
    """The comparison table and the headline (see :func:`compare`)."""
    rows = []
    ours = {}
    bodies = sorted({b for blk in raw["ours"] for b in blk}, key=lambda b: ("graph", "eager", "ref").index(b)
                    if b in ("graph", "eager", "ref") else 9)
    for b in bodies:
        if b == "ref":
            continue
        a = aggregate([blk[b] for blk in raw["ours"] if b in blk])
        ours[b] = a
        rows.append({"side": "ours", "variant": b, "label": f"ModelBlaster {'CUDA graph' if b == 'graph' else b}",
                     "tf32": None, "check": True, **a})
    dep = ours.get("graph") or ours.get("eager") or {}
    dep_name = "graph" if "graph" in ours else "eager"
    for m, blocks in raw["torch"].items():
        info = ready[m][1]["variants"]
        for v in VARIANTS:
            if v not in info:
                continue
            row = {"side": "torch", "variant": v, "label": LABELS[v] + (" fp16" if dtype == "float16" else ""),
                   "tf32": None if dtype == "float16" else bool(m),
                   "check": bool(info[v].get("ok")), "error": info[v].get("error"),
                   "check_detail": info[v].get("check"), "setup_s": info[v].get("setup_s")}
            if info[v].get("ok") and blocks and all(v in blk for blk in blocks):
                row.update(aggregate([blk[v] for blk in blocks]))
                for k in ("gpu", "wall", "thru_wall", "thru_gpu"):
                    t, o = row.get(f"{k}_us"), dep.get(f"{k}_us")
                    row[f"speedup_{k}"] = (t / o) if t and o else None
            rows.append(row)
    hm = headline_mode(math, dtype)
    cands = [r for r in rows if r["side"] == "torch" and r.get("gpu_us") and
             (hm in ("best", "fp16") or r["tf32"] is False)]

    def pick(key):
        ok = [r for r in cands if r.get(key)]
        return min(ok, key=lambda r: r[key]) if ok else None

    def by_variant(v, key):
        ok = [r for r in cands if r["variant"] == v and r.get(key)]
        return min(ok, key=lambda r: r[key]) if ok else None
    head = {"mode": hm, "ours_variant": dep_name, "ours_us": dep.get("gpu_us"), "ours_wall_us": dep.get("wall_us"),
            "ours_thru_wall_us": dep.get("thru_wall_us")}
    best, bestw = pick("gpu_us"), pick("wall_us")
    for name, r, key in (("best", best, "gpu_us"), ("best_wall", bestw, "wall_us")):
        head[f"torch_{name}_variant"] = (r["variant"] + ("+tf32" if r["tf32"] else "")) if r else None
        head[f"torch_{name}_us"] = r[key] if r else None
    for v in VARIANTS:
        r = by_variant(v, "gpu_us")
        head[f"torch_{v}_us"] = r["gpu_us"] if r else None
        rw = by_variant(v, "wall_us")
        head[f"torch_{v}_wall_us"] = rw["wall_us"] if rw else None
    o, ow = dep.get("gpu_us"), dep.get("wall_us")
    head["speedup_vs_torch"] = (head["torch_best_us"] / o) if o and head["torch_best_us"] else None
    head["speedup_vs_torch_wall"] = (head["torch_best_wall_us"] / ow) if ow and head["torch_best_wall_us"] else None
    head["speedup_vs_torch_eager"] = (head["torch_eager_us"] / o) if o and head["torch_eager_us"] else None
    head["speedup_vs_torch_eager_wall"] = (head["torch_eager_wall_us"] / ow) if ow and head["torch_eager_wall_us"] \
        else None
    return rows, head


def compare(*, ir_dir: Path, problem: Path, device: dict, work: Path, kbeval=None, arch: str,
            gpu_model: str | None = None, python: str | None = None, source=None, rounds: int = 4,
            variants=VARIANTS, warm: int = 10, budget_ms: float = 200.0, thru_n: int = 100,
            modes=None, setup_timeout: float = 3600, ours=None, servers=None) -> dict:
    """The model (the exported whole-model ``problem``) against the PyTorch module of
    ``ir_dir`` on ``device`` (the model run's GPU: uuid, host, index) ->
    ``modelblaster.torch_baseline/v1``: ``rows`` (ours graph / eager, torch eager / graph /
    compiled with TF32 off and on: GPU and wall us, CV, spread, throughput, speedups of ours
    over each), ``headline``, ``math``, ``versions``, ``device``, ``same_device``, ``bundle``,
    ``method``, ``raw`` (every block). ``ours`` / ``servers``: injected for tests."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    uuid = (device or {}).get("uuid")
    if not uuid:
        raise Skip("the model run recorded no device UUID to pin the comparison to")
    host = (device or {}).get("host")
    g = json.loads((Path(ir_dir) / "graph.json").read_text())
    tol = tolerance(problem, g)
    dtype = "float16" if g.get("quant") == "fp16" else "float32"
    if modes is None:                              # TF32 only changes fp32 math: one mode for fp16
        modes = (0,) if dtype == "float16" else (0, 1)
    man = json.loads((Path(problem) / "manifest.json").read_text())
    texts = {s["id"]: (Path(problem) / s["file"]).read_text() for s in man["slots"]}
    math = detect_math(texts)
    params = {"warm": warm, "min_reps": 3, "max_reps": 200, "budget_ms": budget_ms, "thru_n": thru_n}
    env = timing_env(warm, budget_ms, thru_n)
    close = []
    t0 = time.time()
    try:
        if servers is None:
            bundle = work / "bundle"
            rec = build_bundle(ir_dir, bundle, source=source, tol=tol)
            if host and not python:
                raise Skip(f"no torch python for {host} (--torch-python)")
            remote_dir = (getattr(kbeval, "remote_dir", None) or "").rstrip("/")
            servers = start_servers(bundle, uuid=uuid, modes=modes, variants=variants, python=python or sys.executable,
                                    host=host, key=getattr(kbeval, "ssh_key", None),
                                    remote_dir=f"{remote_dir}/torch_baseline" if host else None,
                                    work=work, setup_timeout=setup_timeout)
            close += [s for s, _ in servers.values()]
        else:
            rec = {"injected": True}
        if ours is None:
            if kbeval is not None:
                ours = kb_ours(kbeval.for_device({"host": host, "ident": device.get("index") if host else uuid}),
                               problem, arch=arch, work=work / "kb_time", gpu_model=gpu_model, env=env)
            else:
                ours = BuiltinOurs(problem, arch=arch, work=work / "build_ours", env=env, device=device)
                close.append(ours)
        t1 = time.time()
        raw = run_rounds(ours, {m: s for m, (s, _) in servers.items()}, rounds=rounds, uuid=uuid, params=params)
        t2 = time.time()
    finally:
        for c in close:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
    rows, head = build_rows(raw, servers, math, dtype)
    ready = {m: r for m, (_, r) in servers.items()}
    r0 = next(iter(ready.values()))
    return {"format": FORMAT, "status": "ok", "ir": g.get("name"), "device": device, "uuid": uuid,
            "same_device": all(d == norm_uuid(uuid) for d in raw["devices"]),
            "torch_device": r0.get("device"), "versions": r0.get("versions"),
            "flags": {("tf32_on" if m else "tf32_off"): r.get("flags") for m, r in ready.items()},
            "math": {**math, "headline_mode": head["mode"],
                     "note": "fp16 IR: torch in half (its default tensor-core math)" if head["mode"] == "fp16" else
                     "ours is plain fp32: headline vs torch with TF32 off" if head["mode"] == "tf32_off" else
                     f"ours uses {math['mode']}: headline vs the faster of torch TF32 off / on"},
            "rows": rows, "headline": head,
            "checks": {"ours": [r.get("ok") for r in raw["ours_runs"]],
                       **{("tf32_on" if m else "tf32_off"): {v: i.get("ok") for v, i in r["variants"].items()}
                          for m, r in ready.items()}},
            "method": {**params, "rounds": rounds, "order": "ABBA (ours, torch tf32 off, on; then reversed)",
                       "ours_env": env, "tolerance": tol, "dtype": dtype, "statistic": "upper median of the "
                       "per-block upper medians", "wall": "perf_counter / steady_clock, issue -> device synchronize, "
                       "inputs and outputs resident", "throughput": f"{thru_n} back to back, one sync"},
            "bundle": {k: rec.get(k) for k in ("source_path", "sizing", "attrs", "mapping", "cpu_check", "key")},
            "seconds": {"setup": round(t1 - t0, 1), "rounds": round(t2 - t1, 1)},
            "raw": raw}


def summary_fields(res: dict | None) -> dict:
    """The model run's summary fields of a comparison (``model_runtime.json`` top level)."""
    h = (res or {}).get("headline") or {}
    out = {"torch_status": (res or {}).get("status"), "torch_reason": (res or {}).get("reason"),
           "torch_same_device": (res or {}).get("same_device"), "torch_uuid": (res or {}).get("uuid"),
           "torch_us": h.get("torch_best_us"), "torch_best_us": h.get("torch_best_us"),
           "torch_best_variant": h.get("torch_best_variant"), "torch_wall_us": h.get("torch_best_wall_us"),
           "torch_mode": h.get("mode"), "speedup_vs_torch": h.get("speedup_vs_torch"),
           "speedup_vs_torch_wall": h.get("speedup_vs_torch_wall"),
           "speedup_vs_torch_eager": h.get("speedup_vs_torch_eager"),
           "ours_cmp_us": h.get("ours_us"), "ours_cmp_wall_us": h.get("ours_wall_us")}
    for v in VARIANTS:
        out[f"torch_{v}_us"] = h.get(f"torch_{v}_us")
        out[f"torch_{v}_wall_us"] = h.get(f"torch_{v}_wall_us")
    return out


def _f(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def table(res: dict) -> str:
    """The comparison as one markdown table (and a header of what was compared)."""
    if not res or res.get("status") != "ok":
        return f"torch baseline: {(res or {}).get('status')} ({(res or {}).get('reason')})"
    d, v, h = res.get("device") or {}, res.get("versions") or {}, res["headline"]
    lines = [f"**{res.get('ir')}** on {d.get('name')} {d.get('label') or ''} ({res.get('uuid')}); torch {v.get('torch')}, "
             f"CUDA {v.get('cuda')}, cuDNN {v.get('cudnn')}, driver {v.get('driver')}; ours: {res['math']['mode']} "
             f"({res['math']['note']}); {res['method']['rounds']} rounds ABBA, same device: {res.get('same_device')}",
             "",
             "| variant | TF32 | check | GPU us (CV %, spread %) | GPU mean us | wall us (CV %) | "
             "throughput wall / GPU us | ours faster (GPU) | ours faster (wall) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        if r.get("gpu_us") is None:
            lines.append(f"| {r['label']} | {'on' if r['tf32'] else 'off' if r['tf32'] is False else '-'} | "
                         f"{'pass' if r['check'] else 'FAIL'} | {r.get('error') or '-'} | | | | | |")
            continue
        lines.append(
            f"| {r['label']} | {'on' if r['tf32'] else 'off' if r['tf32'] is False else '-'} | "
            f"{'pass' if r['check'] else 'FAIL'} | {_f(r['gpu_us'])} ({_f(100 * (r['gpu_cv'] or 0), 2)}, "
            f"{_f(100 * (r['gpu_spread'] or 0), 2)}) | {_f(r.get('gpu_mean_us'))} | {_f(r['wall_us'])} ({_f(100 * (r['wall_cv'] or 0), 2)}) | "
            f"{_f(r['thru_wall_us'])} / {_f(r['thru_gpu_us'])} | "
            f"{(_f(r.get('speedup_gpu'), 2) + 'x') if r.get('speedup_gpu') else '-'} | "
            f"{(_f(r.get('speedup_wall'), 2) + 'x') if r.get('speedup_wall') else '-'} |")
    lines += ["", f"headline ({h['mode']}): ours {h['ours_variant']} {_f(h['ours_us'])} us vs torch best "
              f"{h['torch_best_variant']} {_f(h['torch_best_us'])} us -> {_f(h['speedup_vs_torch'], 2)}x (GPU); wall "
              f"{_f(h['ours_wall_us'])} vs {_f(h['torch_best_wall_us'])} us ({h['torch_best_wall_variant']}) -> "
              f"{_f(h['speedup_vs_torch_wall'], 2)}x"]
    return "\n".join(lines)


def torch_python_ok(python: str) -> bool:
    try:
        return subprocess.run([python, "-c", "import torch, numpy; assert torch.cuda.is_available()"],
                              capture_output=True, timeout=120).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle", help="build (and CPU-check) the torch side's bundle of an IR")
    b.add_argument("ir_dir")
    b.add_argument("--out", required=True)
    b.add_argument("--source")
    b.add_argument("--problem", help="exported problem whose tolerance to use (default: the exporter's)")
    b.add_argument("--no-cpu-check", action="store_true")
    a = ap.parse_args()
    g = json.loads((Path(a.ir_dir) / "graph.json").read_text())
    rec = build_bundle(Path(a.ir_dir), Path(a.out), source=a.source, cpu_check=not a.no_cpu_check,
                       tol=tolerance(Path(a.problem) if a.problem else None, g))
    print(json.dumps({k: rec.get(k) for k in ("source_path", "sizing", "attrs", "inputs", "cpu_check", "mapping")},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
