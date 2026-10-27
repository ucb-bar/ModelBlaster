#!/usr/bin/env python3
"""Gates for the SOL-ExecBench port (gpu_eval/tools/sol_upstream.py, docs/sol_execbench.md).

Per converted workload (a row of ``manifest.<variant>.json``), in one worker process:

1. **extract**: ``extract_graph --bench-file F --bench-target-mb 0 --quant Q`` (stock dims:
   nothing is resized), with the refusal reason classified when it fails;
2. **shapes**: the IR's graph inputs / outputs have the workload's shapes, and io.npz's inputs
   are the bench's own ``get_inputs()`` values;
3. **cpu**: ``walk_torch`` on the CPU, against the MB golden (``mbir.compare``, MB's bound) and
   against the **SOL ground truth**, the original reference run on the native inputs
   (``--ref-device``: the CPU by default; SOL runs it on its GPU), with SOL's per-workload tolerance
   (``|out - ref| <= atol + rtol*|ref|`` on >= ``required_matched_ratio`` of the elements, no
   non-finite values, ``max_error_cap``). The MB golden is checked against it the same way: that
   is the port's floor (the fp32 math against the bf16 reference). Unless ``--no-hi-ref``, also
   against a **float64 reference** (the dtype-rewritten reference on the problem's own inputs in
   float64: ``sol_vs_walk_hi`` / ``sol_ok_hi``), whether SOL's tolerance is attainable at all
   (``sol_floor_hi`` / ``sol_tol_attainable``: that answer, stored at each output's SOL dtype,
   against SOL's reference), the inputs a 16-bit io buffer changes (``io_rounding``: ints past
   bf16's 256, fp32 values) and the fp32 outputs a bf16 / fp16 IR stores at 16 bits
   (``out_dtype_mismatch``); ``summary`` uses them to say why a walk misses SOL's tolerance;
4. **cuda**: ``walk_cuda.py --impl hand`` (whole model on the hand kernels, every intermediate
   isolated);
5. **export**: ``kb_export --strategy whole --build-check --runtime host`` (multi-slot problem in
   the persistent driver host), skipped with a reason when its data.bin would exceed
   ``--export-cap-mb``.

``run`` does 1-3 locally, each workload in its own memory-capped scope (``systemd-run --user
--scope -p MemoryMax``; at most ``--jobs-large`` large workloads at once, and only while the host
has the memory free): a workload the cap kills is recorded as "too large for host memory", not
retried. ``remote`` ships the extracted IRs in batches to an ssh GPU host (only when its GPU is
idle), runs 4-5 there (``gpu``, with the host's own torch python, never modified) and merges the
results back; io.npz is deleted once consumed, unless ``--keep``.

    sol_upstream_gates.py run MANIFEST OUT [--stems REGEX] [--jobs 4 --jobs-large 2] [--ref-device cpu]
    sol_upstream_gates.py remote OUT --host U@H --ssh-key K --remote-dir D --python PY [--gpu 2] [--until-done]
    sol_upstream_gates.py gpu IR_ROOT WORK OUT_JSON --specs S --hand LIB [--arch 89]   # on the GPU host
    sol_upstream_gates.py summary OUT
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import as_completed
from contextlib import contextmanager
from pathlib import Path

MB_ROOT = Path(__file__).resolve().parents[2]
HARNESS = MB_ROOT / "gpu_eval" / "harness"
GATES = ("extract", "shapes", "cpu", "cuda", "export")

# extract failure -> taxonomy class (first match wins)
EXTRACT_CLASSES = [
    (r"nonzero", "value-dependent: nonzero()"),
    (r"\.item\(\)|\bitem\b.*(shape|data-dependent)|aten\.item|_local_scalar_dense", "value-dependent: .item()"),
    (r"tolist", "value-dependent: tolist()"),
    (r"bincount", "value-dependent: bincount()"),
    (r"masked_select|boolean.mask|bool.mask|mask indexing", "value-dependent: boolean-mask indexing"),
    (r"unique", "value-dependent: unique()"),
    (r"single-argument where|where\(cond\)", "value-dependent: where(cond)"),
    (r"no fp8/fp4 storage", "in-graph fp8/fp4 quantisation"),
    (r"complex|fft", "complex / FFT"),
    (r"no tensor inputs", "no tensor inputs"),
    (r"does not depend on any input", "output independent of inputs"),
    (r"out of memory|MemoryError|Killed|std::bad_alloc|CUDA out of memory", "out of memory"),
    (r"timeout", "timeout"),
    (r"NotImplementedError|not supported|unsupported|no lowering|Unsupported", "unsupported op / lowering"),
]


def classify_extract(log: str) -> str:
    tail = "\n".join(log.splitlines()[-60:])
    for rx, cls in EXTRACT_CLASSES:
        if re.search(rx, tail, re.I):
            return cls
    return "other extract error"


def last_error(log: str) -> str:
    lines = [ln for ln in log.splitlines() if re.search(r"Error|error|Exception|refus", ln)]
    return (lines[-1] if lines else (log.splitlines() or [""])[-1])[:300]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _visible_gpu() -> tuple:
    """(uuid, free MiB) of the single GPU in CUDA_VISIBLE_DEVICES, or (None, None)."""
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not dev or "," in dev or not shutil.which("nvidia-smi"):
        return None, None
    p = subprocess.run(["nvidia-smi", "-i", dev, "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    try:
        uuid, free = (x.strip() for x in p.stdout.strip().split(","))
        return uuid, float(free)
    except ValueError:
        return None, None


@contextmanager
def gpu_lock(out: Path):
    """One GPU user at a time: among our workers (``out/.gpu.lock``) and, for the GPU in
    CUDA_VISIBLE_DEVICES, with every KernelBlaster run on the host (its per-GPU flock,
    ``$KERNELBLASTER_GPU_LOCK_DIR`` or /dev/shm ``/kernelblaster_gpu_<uuid>.lock``)."""
    files = [open(out / ".gpu.lock", "w")]
    uuid, _ = _visible_gpu()
    if uuid and os.getenv("KERNELBLASTER_GPU_FLOCK", "1").lower() not in ("0", "false", "no", "off"):
        d = os.getenv("KERNELBLASTER_GPU_LOCK_DIR", "/dev/shm")
        try:
            files.append(open(f"{d}/kernelblaster_gpu_{re.sub(r'[^A-Za-z0-9_.-]', '_', uuid)}.lock", "a"))
        except OSError:                       # another user's file: run unlocked, as KernelBlaster does
            pass
    for f in files:
        fcntl.flock(f, fcntl.LOCK_EX)
    try:
        yield
    finally:
        for f in reversed(files):
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()


def gpu_room(g: dict, wait_s: float) -> bool:
    """Enough free memory on the visible GPU for every tensor of the IR (x1.5), waiting up to
    wait_s for other users to finish; True when it cannot be checked."""
    import math
    need = sum(math.prod(t["shape"]) if t["shape"] else 1 for t in g["tensors"].values()) * \
        (2 if g.get("quant") in ("fp16", "bf16") else 4) * 1.5 / 2 ** 20
    t = time.time()
    while True:
        _, free = _visible_gpu()
        if free is None or free >= need + 1024:
            return True
        if time.time() - t > wait_s:
            return False
        time.sleep(30)


# ── SOL correctness (sol_execbench/core/bench/correctness.py @834c7c1, Apache-2.0) ──

def sol_check(out, ref, tol: dict) -> dict:
    """SOL's compute_error_stats on one output tensor -> {ok, max_abs, max_rel, matched_ratio, why}."""
    import torch
    x, y = out.reshape(-1).to(torch.float32), ref.reshape(-1).to(torch.float32)
    if x.shape != y.shape:
        return {"ok": False, "why": f"numel {x.numel()} != {y.numel()}"}
    rn, sn = ~torch.isfinite(y), ~torch.isfinite(x)
    both = None
    if tol.get("allow_negative_inf"):
        both = (y == float("-inf")) & (x == float("-inf"))
        rn, sn = rn & ~both, sn & ~both
    if rn.any() or sn.any():
        return {"ok": False, "why": "non-finite " + ("in reference" if rn.any() else "in output"),
                "nonfinite_ref": int(rn.sum()), "nonfinite_out": int(sn.sum())}
    # SOL's all-zero test runs on the whole tensors, before the matching -inf are dropped
    if float(torch.linalg.vector_norm(y)) > 0 and float(torch.linalg.vector_norm(x)) == 0:
        return {"ok": False, "why": "all-zero output"}
    if both is not None:
        x, y = x[~both], y[~both]
    if x.numel() == 0:
        return {"ok": True, "max_abs": 0.0, "max_rel": 0.0, "matched_ratio": 1.0}
    err = (x - y).abs()
    bound = tol["max_atol"] + tol["max_rtol"] * y.abs()
    bad = (err > bound) | ~torch.isfinite(err)
    ratio = 1.0 - float(bad.sum()) / err.numel()
    max_abs = float(err.max())
    ok = ratio >= tol.get("required_matched_ratio", 0.99)
    cap = tol.get("max_error_cap")
    if cap is not None and max_abs > cap:
        ok = False
    rel = float((err / y.abs().clamp(min=tol["max_atol"])).max())
    r = {"ok": bool(ok), "max_abs": max_abs, "max_rel": rel, "matched_ratio": ratio,
         "range": float(y.abs().max())}
    if not ok:
        r["why"] = f"matched {ratio:.4f} < {tol.get('required_matched_ratio', 0.99)}"
    return r


def split_outputs(flat, shapes):
    import math
    out, off = [], 0
    for sh in shapes:
        n = math.prod(sh) if sh else 1
        out.append(flat[off:off + n].reshape(sh))
        off += n
    return out


def sol_compare(flat, refs, tol) -> dict:
    """Per-output SOL check of a flat output (IR output order = definition output order)."""
    shapes = [list(r.shape) for r in refs]
    if flat.numel() != sum(r.numel() for r in refs):
        return {"ok": False, "why": f"output numel {flat.numel()} != reference {sum(r.numel() for r in refs)}"}
    per = [sol_check(o, r, tol) for o, r in zip(split_outputs(flat, shapes), refs)]
    return {"ok": all(p["ok"] for p in per), "outputs": per,
            "worst_matched_ratio": min(p.get("matched_ratio", 0.0) for p in per),
            "max_abs": max(p.get("max_abs", float("inf")) for p in per)}


def _sorted_topk():
    """topk(sorted=True), as extract_graph computes the golden (the IR's topk sorts)."""
    import torch

    class SortedTopk(torch.overrides.TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = dict(kwargs or {})
            if func in (torch.topk, torch.Tensor.topk):
                if len(args) > 4:
                    args = tuple(args[:4]) + (True,) + tuple(args[5:])
                else:
                    kwargs["sorted"] = True
            return func(*args, **kwargs)
    return SortedTopk()


def hi_reference(bench, ins):
    """The problem at high precision on its own inputs: the dtype-rewritten reference (bench.Model)
    on ``get_inputs()`` with every float tensor in float64 (float32 when the reference refuses it;
    explicit casts inside run() still apply) -> (list of outputs, dtype name) or (None, error)."""
    import torch
    err = None
    for dt in (torch.float64, torch.float32):
        try:
            x = [t.to(dt) if torch.is_tensor(t) and t.is_floating_point() else t for t in ins]
            torch.manual_seed(0)
            with torch.no_grad(), _sorted_topk():
                o = bench.Model().to(dt)(*x)
            o = [o] if torch.is_tensor(o) else list(o.values()) if isinstance(o, dict) else list(o)
            return [v if torch.is_tensor(v) else torch.tensor(v) for v in o], str(dt).split(".")[-1]
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:200]}"
    return None, err


def io_rounding(bench, ins, quant: str) -> dict:
    """Inputs MB's 16-bit io buffer changes (every tensor input, ints included, is stored in it):
    {name: {dtype, changed, of}} -- ints past 256 (bf16) / 2048 (fp16), fp32 values it cannot hold."""
    import torch
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(quant)
    out = {}
    if dt is None:
        return out
    for (name, spec), t in zip(bench.SOL_INPUTS.items(), ins):
        if torch.is_tensor(t) and t.numel():
            d = t.double()
            n = int((d.to(dt).double() != d).sum())
            if n:
                out[name] = {"dtype": spec["dtype"], "changed": n, "of": t.numel()}
    return out


def hi_gates(bench, ins, flat, refs, tol, quant: str) -> dict:
    """The walk against the high-precision reference, and whether SOL's tolerance can be met at all
    by the correctly rounded answer (the hi reference stored at each output's SOL dtype)."""
    import torch
    r = {"io_rounding": io_rounding(bench, ins, quant),
         "out_dtype_mismatch": [n for n, v in bench.SOL_OUTPUTS.items()
                                if v["dtype"] in ("float32", "float64", "bfloat16", "float16")
                                and v["dtype"] != {"bf16": "bfloat16", "fp16": "float16"}.get(quant, v["dtype"])]}
    hi, dt = hi_reference(bench, ins)
    if hi is None:
        r["hi_error"] = dt
        return r
    r["hi_dtype"] = dt
    r["sol_vs_walk_hi"] = sol_compare(flat.float(), hi, tol)
    if len(hi) == len(refs) and all(h.numel() == f.numel() for h, f in zip(hi, refs)):
        rounded = torch.cat([h.to(f.dtype).float().reshape(-1) for h, f in zip(hi, refs)])
        r["sol_floor_hi"] = sol_compare(rounded, refs, tol)
    return r


# ── one workload ──────────────────────────────────────────────────────────────

def native_quant(row: dict) -> str:
    """The MB storage dtype closest to the problem's own: bf16 if any of its tensors is bfloat16,
    else fp16 if any is float16, else fp32 (an IR has one float storage dtype)."""
    dts = set(row.get("dtype_map") or {})
    return "bf16" if "bfloat16" in dts else "fp16" if "float16" in dts else "fp32"


def run_one(row: dict, cfg: dict) -> dict:
    out = Path(cfg["out"])
    stem = row["stem"]
    quant = native_quant(row) if cfg["quant"] == "native" else cfg["quant"]
    ir = out / "ir" / stem
    rec = {"stem": stem, "set": row["set"], "problem": row["problem"], "pick": row["pick"], "uuid": row["uuid"],
           "variant": row["variant"], "quant": quant, "io_mib": row["io_mib"],
           "ref_dtypes": sorted(row.get("dtype_map", {})), "gates": {}}
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    threads = str(cfg.get("threads", 4))
    env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads)
    gates = cfg["gates"]
    t0 = time.time()
    try:
        if "extract" in gates:
            shutil.rmtree(ir, ignore_errors=True)
            if not wait_for_disk(out, 2 * row["io_mib"] + cfg["reserve_mb"], cfg["disk_wait_s"]):
                rec["gates"]["extract"] = {"ok": None, "skipped": "disk: free space below "
                                           f"{2 * row['io_mib'] + cfg['reserve_mb']:.0f} MiB"}
                return rec
            cmd = [sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file", row["file"],
                   "--bench-target-mb", "0", "--out-dir", str(ir), "--quant", quant]
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg["timeout"], env=env)
                log, rc = p.stdout + p.stderr, p.returncode
            except subprocess.TimeoutExpired as e:
                log, rc = f"{e.stdout or ''}{e.stderr or ''}\ntimeout after {cfg['timeout']} s", -1
            (logs / f"{stem}.extract.log").write_text(log[-200000:] if isinstance(log, str) else str(log))
            ok = rc == 0 and (ir / "graph.json").exists()
            rec["gates"]["extract"] = {"ok": ok, "s": round(time.time() - t0, 1)}
            if not ok:
                cls = classify_extract(log)
                ref = Path(row["file"]).read_text()
                if cls in ("unsupported op / lowering", "other extract error") and re.search(
                        r"float8_e4m3fn|float8_e5m2|float4_e2m1", ref):
                    # the reference quantises inside run() (x.to(torch.float8_e4m3fn) ...): no MB dtype
                    cls = "in-graph fp8/fp4 quantisation"
                rec["gates"]["extract"].update(cls=cls, error=last_error(log))
                return rec
        if not (ir / "graph.json").exists():
            rec["gates"]["extract"] = {"ok": False, "cls": "no IR", "error": "not extracted"}
            return rec
        g = json.loads((ir / "graph.json").read_text())
        rec["n_ops"] = len(g["ops"])
        rec["op_types"] = sorted({o["op"] for o in g["ops"]})
        bench = None
        if {"shapes", "cpu"} & set(gates):
            sys.path.insert(0, str(HARNESS))
            import numpy as np
            import torch
            bench = _load(Path(row["file"]), f"_solbench_{abs(hash(stem))}")
            ins = bench.get_inputs()
        if "shapes" in gates:
            rec["gates"]["shapes"] = check_shapes(g, ir, bench, ins)
        if "cpu" in gates:
            import mbir
            import walk_torch
            mbir.strict_fp32()
            irm = mbir.load_ir(ir)
            t1 = time.time()
            with torch.no_grad():
                flat, _ = walk_torch.walk(irm, "cpu")
            cpu = {"mb_golden": mbir.compare(flat, irm.golden_out, irm.dtype), "walk_s": round(time.time() - t1, 1)}
            native = bench._native_inputs()
            refs, ref_dev = None, cfg["ref_device"]
            for dev in ([ref_dev, "cpu"] if ref_dev != "cpu" else ["cpu"]):
                try:
                    if dev.startswith("cuda"):
                        with gpu_lock(out):
                            refs = bench.sol_reference_outputs(native, dev)
                            torch.cuda.empty_cache()
                    else:
                        refs = bench.sol_reference_outputs(native, dev)
                    ref_dev = dev
                    break
                except Exception as e:  # noqa: BLE001 -- e.g. GPU OOM: fall back to the CPU
                    cpu.setdefault("ref_errors", []).append(f"{dev}: {type(e).__name__}: {str(e)[:200]}")
            if refs is None:
                cpu["ok"] = False
                cpu["why"] = "SOL reference failed"
            else:
                tol = row["tolerance"]
                cpu["ref_device"] = ref_dev
                cast = cfg.get("score_cast")
                if cast:
                    # diagnostic variants (e.g. fp16 of a bf16 problem): score in the reference's dtype
                    cpu["sol_vs_walk_raw"] = sol_compare(flat.float(), refs, tol)
                    cpu["score_cast"] = cast
                    flat = flat.to(getattr(torch, cast)).float()
                cpu["sol_vs_walk"] = sol_compare(flat.float(), refs, tol)
                golden = torch.from_numpy(np.asarray(irm.golden_out, np.float32))
                if cast:
                    golden = golden.to(getattr(torch, cast)).float()
                cpu["sol_vs_golden"] = sol_compare(golden, refs, tol)
                cpu["ok"] = bool(cpu["mb_golden"]["ok"])
                cpu["sol_ok"] = bool(cpu["sol_vs_walk"]["ok"])
                cpu["port_floor_ok"] = bool(cpu["sol_vs_golden"]["ok"])
                cpu["range_risk"] = range_risk(refs)
                if cfg.get("hi_ref", True):
                    cpu.update(hi_gates(bench, ins, flat, refs, tol, quant))
                    cpu["sol_ok_hi"] = bool(cpu.get("sol_vs_walk_hi", {}).get("ok"))
                    cpu["sol_tol_attainable"] = cpu.get("sol_floor_hi", {}).get("ok")
                if not cpu["sol_ok"] and ref_dev != "cpu":
                    # is the reference itself reproducible at SOL's tolerance? (GPU vs CPU run of run())
                    try:
                        refs_cpu = bench.sol_reference_outputs(native, "cpu")
                        flat_cpu = torch.cat([t.float().reshape(-1) for t in refs_cpu])
                        cpu["sol_ref_cpu_vs_gpu"] = sol_compare(flat_cpu, refs, tol)
                        cpu["sol_vs_walk_cpu_ref"] = sol_compare(flat.float(), refs_cpu, tol)
                    except Exception as e:  # noqa: BLE001
                        cpu["sol_ref_cpu_error"] = f"{type(e).__name__}: {str(e)[:200]}"
            rec["gates"]["cpu"] = cpu
        rec["gates"].update(gpu_gates(ir, g, cfg, gates, out, env, stem))
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-3000:]
    finally:
        rec["s"] = round(time.time() - t0, 1)
        # io.npz stays for the GPU gates when they run elsewhere (``remote``), else it goes
        if not cfg.get("keep") and ({"cuda", "export"} & set(gates) or not rec["gates"].get("extract", {}).get("ok")):
            (ir / "io.npz").unlink(missing_ok=True)
        (out / "results").mkdir(exist_ok=True)
        (out / "results" / f"{stem}.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec


def gpu_gates(ir: Path, g: dict, cfg: dict, gates, out: Path, env: dict, stem: str) -> dict:
    """The CUDA walk and the multi-slot export of one IR (on the host this runs on)."""
    res = {}
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    if {"cuda", "export"} & set(gates) and not gpu_room(g, cfg.get("gpu_wait_s", 1800)):
        return {k: {"ok": None, "skipped": "GPU memory (not enough free on the GPU)"}
                for k in ("cuda", "export") if k in gates}
    # the hand kernels of the IR's storage dtype: lib_hand.so -> lib_hand_f16.so / lib_hand_bf16.so
    q = {"fp16": "_f16", "bf16": "_bf16"}.get(g.get("quant"), "")
    if q and cfg.get("hand") and not cfg["hand"].endswith(f"{q}.so"):
        cfg = dict(cfg, hand=cfg["hand"].replace(".so", f"{q}.so"))
    if "cuda" in gates:
        with gpu_lock(out):
            j = out / "tmp" / f"{stem}.wc.json"
            j.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "walk_cuda.py", str(ir), "--specs", cfg["specs"], "--hand", cfg["hand"],
                   "--impl", "hand", "--reps", "1", "--json", str(j)]
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg["timeout"], cwd=HARNESS,
                                   env=env)
                (logs / f"{stem}.cuda.log").write_text((p.stdout + p.stderr)[-100000:])
                r = json.loads(j.read_text())[0] if j.exists() else {"ok": False, "error": last_error(p.stderr)}
            except subprocess.TimeoutExpired:
                r = {"ok": False, "error": "timeout"}
            j.unlink(missing_ok=True)
        keep = ("ok", "max_abs_err", "rel_to_range", "worst_intermediate_rel_err", "first_divergent_op",
                "missing", "error", "model_ms", "torch_eager_ms", "why")
        c = {k: r.get(k) for k in keep if k in r}
        c["ok"] = bool(r.get("ok")) and not r.get("first_divergent_op")
        if not c["ok"] and re.search(r"out of memory|cudaErrorMemoryAllocation", str(r.get("error", "")), re.I):
            c = {"ok": None, "skipped": "GPU memory (out of memory)", "error": str(r.get("error"))[:200]}
        elif not c["ok"]:
            c["cls"] = ("missing kernel" if r.get("missing") else
                        "divergent intermediate" if r.get("first_divergent_op") else
                        "out of memory" if "memory" in str(r.get("error", "")).lower() else
                        "golden bound" if "max_abs_err" in r else "error")
        res["cuda"] = c
    if "export" in gates:
        res["export"] = export_gate(ir, g, out, cfg, env, stem)
    return res


def wait_for_disk(path: Path, need_mb: float, wait_s: int) -> bool:
    """Native shapes are large and the disk is shared: wait (up to wait_s) for need_mb free."""
    t = time.time()
    while shutil.disk_usage(path).free / 2 ** 20 < need_mb:
        if time.time() - t > wait_s:
            return False
        time.sleep(20)
    return True


def range_risk(refs) -> dict:
    """bf16 -> fp16 range: reference values past fp16's max (65504) or in its subnormal range."""
    import torch
    big = tiny = 0
    for r in refs:
        a = r.float().abs()
        big += int((a > 65504).sum())
        tiny += int(((a > 0) & (a < 6.1e-5)).sum())
    return {"over_fp16_max": big, "fp16_subnormal": tiny}


def check_shapes(g: dict, ir: Path, bench, ins) -> dict:
    """IR inputs/outputs vs the workload's shapes; io.npz inputs vs the bench's get_inputs()."""
    import numpy as np
    import torch
    tens = [t for t in ins if torch.is_tensor(t)]
    want_in = [list(t.shape) for t in tens]
    packed = g["input"].get("packed_inputs") or [{"name": g["input"]["tensor"]}]
    got_in = [g["tensors"][p["name"]]["shape"] for p in packed]
    outs = g["output"].get("tensors") or [g["output"]["tensor"]]
    got_out = [g["tensors"][o]["shape"] for o in outs]
    want_out = [v["shape_v"] for v in bench.SOL_OUTPUTS.values()]
    norm = lambda s: [d for d in s] if s else [1]  # noqa: E731 -- 0-D == [1]
    r = {"inputs_match": [norm(s) for s in got_in] == [norm(s) for s in want_in],
         "outputs_match": [norm(s) for s in got_out] == [norm(s) for s in want_out]}
    if not r["inputs_match"]:
        r.update(ir_inputs=got_in, workload_inputs=want_in)
    if not r["outputs_match"]:
        r.update(ir_outputs=got_out, workload_outputs=want_out)
    io = np.load(ir / "io.npz")
    flat = np.asarray(io["input"], np.float32).reshape(-1)
    mine = np.concatenate([t.detach().float().reshape(-1).numpy() for t in tens]) if tens else np.zeros(0)
    if g.get("quant") == "fp16":
        mine = mine.astype(np.float16).astype(np.float32)
    elif g.get("quant") == "bf16":                # --quant bf16 rounds every float input to bfloat16
        mine = torch.from_numpy(mine).bfloat16().float().numpy()
    r["inputs_equal"] = bool(flat.shape == mine.shape and np.array_equal(flat, mine))
    r["ok"] = r["inputs_match"] and r["outputs_match"] and r["inputs_equal"]
    return r


def export_gate(ir: Path, g: dict, out: Path, cfg: dict, env: dict, stem: str) -> dict:
    import math
    if not g["ops"]:
        return {"ok": None, "skipped": "no ops (the workload's output is an input: nothing to optimise)"}
    nbytes = sum(math.prod(t["shape"]) if t["shape"] else 1 for t in g["tensors"].values()) * \
        (2 if g.get("quant") in ("fp16", "bf16") else 4)
    if nbytes > cfg["export_cap_mb"] * 2 ** 20:
        return {"ok": None, "skipped": f"data.bin ~{nbytes / 2 ** 20:.0f} MiB > {cfg['export_cap_mb']} MiB"}
    if not wait_for_disk(out, nbytes / 2 ** 20 + cfg["reserve_mb"], cfg["disk_wait_s"]):
        return {"ok": None, "skipped": "disk: no room for data.bin"}
    d = out / "tmp" / f"{stem}.kbx"
    shutil.rmtree(d, ignore_errors=True)
    # export and nvcc build without the GPU; only the driver host's run holds the GPU lock
    cmd = [sys.executable, "-m", "modelblaster.pipeline.kb_export", str(ir), str(d), "--strategy", "whole"]
    chk = [sys.executable, __file__, "export-check", str(d / "p00"), "--arch", str(cfg["arch"]),
           "--lock-dir", str(out), "--timeout", str(cfg["timeout"])]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg["timeout"], env=env)
        log = p.stdout + p.stderr
        if p.returncode == 0:
            p = subprocess.run(chk, capture_output=True, text=True, timeout=2 * cfg["timeout"], env=env)
            log += p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        log += "timeout"
    (out / "logs" / f"{stem}.export.log").write_text(log[-100000:])
    shutil.rmtree(d, ignore_errors=True)
    res = [json.loads(ln) for ln in log.splitlines() if ln.startswith("{") and '"passed"' in ln]
    if not res:
        return {"ok": False, "cls": "export error", "error": last_error(log)}
    r = res[-1]
    bad = {k: v for k, v in (r.get("slots") or {}).items() if not v.get("ok")}
    o = {"ok": bool(r.get("passed")), "stage": r.get("stage"), "e2e_us": r.get("e2e_us"),
         "n_slots": len(r.get("slots") or {})}
    if not o["ok"]:
        o.update(cls="slot check" if bad else f"stage {r.get('stage')}", failed_slots=list(bad)[:8],
                 errors=(r.get("errors") or [])[:3])
    return o


def cmd_export_check(a):
    """kb_export --build-check --runtime host, with the build outside the GPU lock: build the
    persistent driver host and the kernels' .so, then evaluate once holding the lock; prints
    kb_export's result line."""
    from modelblaster.pipeline import kb_export as kx
    prob = Path(a.problem)
    r = kx.build_and_run(prob, arch=a.arch, runtime="host", runs=0, timeout=a.timeout)
    line = {"problem": str(prob)}
    if r.get("stage") == "built":
        with gpu_lock(Path(a.lock_dir)):
            h = kx.DriverHost(r["exe"], timeout=a.timeout)
            try:
                r = h.evaluate(r["so"], env={"MB_GRAPH": "1"})
            finally:
                h.close()
    line.update({k: r.get(k) for k in ("passed", "stage", "partition_us", "slot_us", "slots", "errors",
                                       "e2e_us", "e2e_basis", "graph_skip")})
    if not r.get("passed"):
        line["log"] = (r.get("log") or "")[-1500:]
    print(json.dumps(line), flush=True)
    return 0


# ── driver ────────────────────────────────────────────────────────────────────

def mem_available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2 ** 20
    return 0.0


def capped(cmd: list, max_gb: float) -> list:
    """``cmd`` in its own memory-capped scope (no swap): the OOM is ours, not the host's."""
    if shutil.which("systemd-run"):
        return ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={max_gb:g}G",
                "-p", "MemorySwapMax=0", "--", *cmd]
    return ["prlimit", f"--as={int(max_gb * 2 ** 30)}", "--", *cmd]


def cmd_one(a):
    row = json.loads(Path(a.row).read_text())
    cfg = json.loads(Path(a.cfg).read_text())
    run_one(row, cfg)
    return 0


def _launch(row: dict, cfg: dict, out: Path, a) -> dict:
    """One workload in a capped subprocess; a kill by the cap is recorded, not retried."""
    large = row["io_mib"] > a.large_mib
    cap = a.mem_large_gb if large else a.mem_small_gb
    while mem_available_gb() < cap + a.mem_reserve_gb:      # the host is shared: wait for room
        time.sleep(15)
    tmp = out / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    rf, cf = tmp / f"{row['stem']}.row.json", tmp / f"{row['stem']}.cfg.json"
    rf.write_text(json.dumps(row))
    cf.write_text(json.dumps(cfg))
    res = out / "results" / f"{row['stem']}.json"
    res.unlink(missing_ok=True)
    p = subprocess.run(capped([sys.executable, __file__, "one", str(rf), str(cf)], cap),
                       capture_output=True, text=True)
    rf.unlink(missing_ok=True)
    cf.unlink(missing_ok=True)
    if res.exists() and res.stat().st_size:
        try:
            return json.loads(res.read_text())
        except ValueError:
            pass
    rec = {"stem": row["stem"], "set": row["set"], "problem": row["problem"], "pick": row["pick"],
           "uuid": row["uuid"], "variant": row["variant"], "io_mib": row["io_mib"],
           "quant": native_quant(row) if cfg["quant"] == "native" else cfg["quant"],
           "ref_dtypes": sorted(row.get("dtype_map", {})),
           "gates": {"extract": {"ok": None, "skipped": f"too large for host memory (MemoryMax={cap:g}G, rc "
                                                        f"{p.returncode})"}},
           "stderr": (p.stderr or "")[-1000:]}
    shutil.rmtree(out / "ir" / row["stem"], ignore_errors=True)
    res.parent.mkdir(parents=True, exist_ok=True)
    res.write_text(json.dumps(rec, indent=1))
    return rec


def cmd_run(a):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    man = json.loads(Path(a.manifest).read_text())
    out = Path(a.out).resolve()
    (out / "results").mkdir(parents=True, exist_ok=True)
    rows = [r for r in man["workloads"] if not a.stems or re.search(a.stems, r["stem"])]
    if a.skip_done:
        rows = [r for r in rows if not (out / "results" / f"{r['stem']}.json").exists()]
    for k in ("specs", "hand"):
        if getattr(a, k):
            setattr(a, k, str(Path(getattr(a, k)).resolve()))
    for r in rows:
        r["file"] = str(Path(r["file"]).resolve())
    cfg = {"out": str(out), "quant": a.quant or ("fp32" if man["variant"] == "fp32" else man["variant"]),
           "gates": a.gates.split(","), "timeout": a.timeout, "ref_device": a.ref_device, "arch": a.arch,
           "specs": a.specs, "hand": a.hand, "keep": a.keep, "export_cap_mb": a.export_cap_mb,
           "threads": a.threads, "reserve_mb": a.reserve_mb, "disk_wait_s": a.disk_wait_s,
           "score_cast": a.score_cast, "hi_ref": not a.no_hi_ref}
    rows.sort(key=lambda r: r["io_mib"])          # the small ones report first
    big = threading.Semaphore(a.jobs_large)       # at most this many large workloads at once

    def job(row):
        if row["io_mib"] > a.large_mib:
            with big:
                return _launch(row, cfg, out, a)
        return _launch(row, cfg, out, a)
    n = 0
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for fut in as_completed([ex.submit(job, r) for r in rows]):
            rec = fut.result()
            n += 1
            gs = " ".join(f"{k}={'ok' if v.get('ok') else ('skip' if v.get('ok') is None else 'FAIL')}"
                          for k, v in rec["gates"].items())
            print(f"[{n}/{len(rows)}] {rec['stem']} {gs} {rec.get('error', '')}", flush=True)
    return 0


def cmd_gpu(a):
    """The GPU gates over IR_ROOT/<stem> (run on the GPU host): -> OUT_JSON {stem: gates}."""
    root, out = Path(a.ir_root).resolve(), Path(a.work).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"specs": str(Path(a.specs).resolve()), "hand": str(Path(a.hand).resolve()), "timeout": a.timeout,
           "arch": a.arch, "export_cap_mb": a.export_cap_mb, "reserve_mb": a.reserve_mb, "disk_wait_s": 600}
    env = dict(os.environ)
    res = {}
    for ir in sorted(d for d in root.iterdir() if (d / "graph.json").exists() and (d / "io.npz").exists()):
        t = time.time()
        try:
            g = json.loads((ir / "graph.json").read_text())
        except (OSError, ValueError) as e:
            res[ir.name] = {"cuda": {"ok": False, "cls": "error", "error": f"{type(e).__name__}: {e}"}}
            continue
        try:
            res[ir.name] = gpu_gates(ir, g, cfg, a.gates.split(","), out, env, ir.name)
        except Exception as e:  # noqa: BLE001
            res[ir.name] = {"cuda": {"ok": False, "cls": "error", "error": f"{type(e).__name__}: {e}"}}
        for v in res[ir.name].values():
            v["host"] = os.uname().nodename
        print(f"{ir.name} {round(time.time() - t, 1)} s "
              + " ".join(f"{k}={v.get('ok')}" for k, v in res[ir.name].items()), flush=True)
        Path(a.out_json).write_text(json.dumps(res, indent=1))
    Path(a.out_json).write_text(json.dumps(res, indent=1))
    return 0


def _ssh(a, cmd: str, **kw):
    return subprocess.run(["ssh", "-i", a.ssh_key, "-o", "ConnectTimeout=30", a.host, cmd],
                          capture_output=True, text=True, **kw)


def _rsync(a, src: list, dst: str, *extra):
    return subprocess.run(["rsync", "-a", *extra, "-e", f"ssh -i {a.ssh_key}", *src, dst],
                          capture_output=True, text=True)


def gpu_idle(a) -> bool:
    p = _ssh(a, f"nvidia-smi -i {a.gpu} --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits")
    try:
        mem, util = (float(x) for x in p.stdout.strip().split(","))
    except ValueError:
        return False
    return util <= a.idle_util and mem <= a.idle_mem_mb


def cmd_remote(a):
    """Ship extracted IRs to the GPU host in batches, run ``gpu`` there, merge the results back.
    Polls until ``run`` has finished (``--until-done``), then cleans its remote IR copies."""
    out = Path(a.out).resolve()
    rd = a.remote_dir.rstrip("/")
    py = a.python
    if not a.no_sync:
        _ssh(a, f"mkdir -p {rd}/mb/modelblaster {rd}/gpu {rd}/ir {rd}/work")
        p = _rsync(a, [str(MB_ROOT) + "/"], f"{a.host}:{rd}/mb/modelblaster/", "--delete",
                   "--exclude=.git", "--exclude=third_party", "--exclude=__pycache__", "--exclude=artifacts",
                   "--exclude=mb_datasets", "--exclude=*.npz", "--exclude=*.bin")
        if p.returncode:
            raise SystemExit(p.stderr)
        setup = (f"cd {rd}/mb/modelblaster/gpu_eval/harness && export PATH=/usr/local/cuda/bin:$PATH && "
                 f"PYTHONPATH={rd}/mb {py} ../tools/dump_specs.py {rd}/gpu/specs.json > /dev/null && "
                 + " && ".join(f"nvcc -O3 -arch=sm_{a.arch} -Xcompiler -fPIC -shared -o {rd}/gpu/lib_hand{q}.so "
                               f"cuda/mb_cuda_kernels{q}.cu" for q in ("", "_f16", "_bf16")))
        p = _ssh(a, setup)
        if p.returncode:
            raise SystemExit(f"remote setup failed: {p.stderr[-2000:]}")
    tries: dict = {}
    while True:
        todo = []
        for f in sorted((out / "results").glob("*.json")):
            try:
                r = json.loads(f.read_text())
            except (OSError, ValueError):        # being written, or truncated by a full disk
                continue
            ir = out / "ir" / r["stem"]
            if "cuda" in r["gates"] or not (ir / "io.npz").exists():
                continue
            if a.stems and not re.search(a.stems, r["stem"]):
                continue
            todo.append((r, f, ir))
        running = subprocess.run(["pgrep", "-f", f"sol_upstream_gates.py run .* {out}"],
                                 capture_output=True).returncode == 0 or \
            subprocess.run(["pgrep", "-f", f"sol_upstream_gates.py run .*{out.name}"],
                           capture_output=True).returncode == 0
        if not todo:
            if a.until_done and running:
                time.sleep(60)
                continue
            break
        batch = todo[:a.batch]
        while not gpu_idle(a):
            print(f"GPU {a.gpu} busy; waiting", flush=True)
            time.sleep(120)
        _ssh(a, f"rm -rf {rd}/ir/* && mkdir -p {rd}/ir")
        p = _rsync(a, [str(ir) for _, _, ir in batch], f"{a.host}:{rd}/ir/")
        if p.returncode:
            print("rsync failed:", p.stderr[-500:], flush=True)
            time.sleep(60)
            continue
        cmd = (f"cd {rd}/mb/modelblaster && export PATH=/usr/local/cuda/bin:$PATH "
               f"CUDA_VISIBLE_DEVICES={a.gpu} PYTHONPATH={rd}/mb OMP_NUM_THREADS=4 && "
               f"{py} gpu_eval/tools/sol_upstream_gates.py gpu {rd}/ir {rd}/work {rd}/work/batch.json "
               f"--specs {rd}/gpu/specs.json --hand {rd}/gpu/lib_hand.so --arch {a.arch} "
               f"--export-cap-mb {a.export_cap_mb} --gates {a.gates} > {rd}/work/batch.log 2>&1")
        p = _ssh(a, cmd)
        got = _ssh(a, f"cat {rd}/work/batch.json")
        try:
            res = json.loads(got.stdout or "{}")
        except json.JSONDecodeError:
            res = {}
        _rsync(a, [f"{a.host}:{rd}/work/logs/"], str(out / "logs") + "/")
        for r, f, ir in batch:
            gg = res.get(ir.name)
            if gg is None:                       # the batch died before this IR: retry it once
                tries[ir.name] = tries.get(ir.name, 0) + 1
                if tries[ir.name] < 2:
                    continue
                gg = {"cuda": {"ok": False, "cls": "remote error", "error": (p.stderr or "")[-300:]}}
            r["gates"].update(gg)
            f.write_text(json.dumps(r, indent=1, default=str))
            if not a.keep:
                (ir / "io.npz").unlink(missing_ok=True)
            print(r["stem"], " ".join(f"{k}={v.get('ok')}" for k, v in gg.items()), flush=True)
    _ssh(a, f"rm -rf {rd}/ir {rd}/work/tmp")
    return 0


# ── summary ───────────────────────────────────────────────────────────────────

def load_results(out: Path) -> list:
    recs = []
    for f in sorted((out / "results").glob("*.json")):
        try:
            recs.append(json.loads(f.read_text()))
        except ValueError:
            print(f"warning: unreadable {f}", file=sys.stderr)
    return recs


def _native_of(r: dict) -> str:
    return native_quant({"dtype_map": {k: k for k in r.get("ref_dtypes") or []}})


def sol_fail_class(r: dict) -> str:
    """Why an IR walk that meets MB's bound misses SOL's tolerance."""
    c = r["gates"]["cpu"]
    dev = c.get("sol_ref_cpu_vs_gpu")
    q = r.get("quant", "fp32")
    io = c.get("io_rounding") or {}
    if q in ("bf16", "fp16") and r.get("quant") != _native_of(r):
        return f"not a {q} problem ({'/'.join(r.get('ref_dtypes') or [])} reference)"
    if any(v["dtype"].startswith(("int", "uint", "bool")) for v in io.values()):
        return f"integer input changed by the {q} io (indices past its exact range)"
    if c.get("out_dtype_mismatch"):
        return f"output dtype: an fp32 output stored as {q}"
    if io:
        return f"fp32 input rounded to {q} by the io"
    if c.get("sol_tol_attainable") is False:
        return "SOL tolerance unattainable: the correctly rounded float64 answer misses SOL's reference too"
    if c.get("sol_ok_hi"):
        return "SOL's reference is the outlier: the walk passes vs the float64 reference"
    if c.get("sol_vs_walk_cpu_ref", {}).get("ok"):
        return "reference device (passes vs the CPU reference run)"
    if dev is not None and not dev.get("ok"):
        return "reference not reproducible at SOL tol (its own CPU vs GPU run fails)"
    if not c.get("port_floor_ok"):
        return f"precision floor: the {q} golden misses too"
    if q in ("bf16", "fp16"):
        return f"{q} intermediate storage (the golden passes; the reference keeps these in fp32)"
    return "walk only (the golden passes)"


def cmd_summary(a):
    """Per-set coverage + failure taxonomy over one or more gate dirs. --native keeps, per stem,
    the record run at the problem's own dtype (bf16 / fp16 / fp32); --by-quant splits the sets."""
    by = {}
    for d in a.out:
        for r in load_results(Path(d)):
            if a.native and r.get("quant") and r["quant"] != _native_of(r):
                continue
            by.setdefault(r["stem"], r)
    recs = list(by.values())
    order = ("L1", "L2", "Quant", "FlashInfer-Bench")
    key = (lambda r: f"{r['set']} {r.get('quant', '?')}") if a.by_quant else (lambda r: r["set"])
    groups = sorted({key(r) for r in recs}, key=lambda k: (order.index(k.split()[0]), k))
    cols = ["extract", "shapes", "cpu (MB bound)", "cpu (SOL tol)", "cpu (SOL tol, fp64 ref)", "golden (SOL tol)",
            "SOL tol attainable", "cuda", "export"]

    def val(r, c):
        g = r["gates"]
        if c == "cpu (MB bound)":
            return g.get("cpu", {}).get("ok")
        if c == "cpu (SOL tol)":
            return g.get("cpu", {}).get("sol_ok")
        if c == "golden (SOL tol)":
            return g.get("cpu", {}).get("port_floor_ok")
        if c == "cpu (SOL tol, fp64 ref)":
            return g.get("cpu", {}).get("sol_ok_hi")
        if c == "SOL tol attainable":
            return g.get("cpu", {}).get("sol_tol_attainable")
        return g.get(c, {}).get("ok")
    lines = ["| set | workloads | problems | " + " | ".join(cols) + " |", "|---" * (len(cols) + 3) + "|"]
    for grp in groups + ["all"]:
        rs = [r for r in recs if grp == "all" or key(r) == grp]
        cells = []
        for c in cols:
            ok = [r for r in rs if val(r, c) is True]
            cells.append(f"{len(ok)} ({len({r['problem'] for r in ok})})")
        lines.append(f"| {grp} | {len(rs)} | {len({r['problem'] for r in rs})} | " + " | ".join(cells) + " |")
    tax = {}
    for r in recs:
        for gname, g in r["gates"].items():
            if g.get("ok") is False:
                cls = g.get("cls") or g.get("why") or (f"{r.get('quant')} golden bound" if gname == "cpu" else "fail")
                tax.setdefault((gname, cls), []).append(r["stem"])
            elif g.get("ok") is None and g.get("skipped"):
                tax.setdefault((gname, "skipped: " + re.sub(r"\d[\d.]*", "N", g["skipped"])[:70]),
                               []).append(r["stem"])
        c = r["gates"].get("cpu", {})
        if c.get("ok") and c.get("sol_ok") is False:
            tax.setdefault(("cpu", "SOL tolerance: " + sol_fail_class(r)), []).append(r["stem"])
        if r.get("error"):
            tax.setdefault(("worker", r["error"][:80]), []).append(r["stem"])
    lines += ["", "| gate | class | workloads | problems | examples |", "|---|---|---|---|---|"]
    for (gname, cls), stems in sorted(tax.items(), key=lambda kv: (kv[0][0], -len(kv[1]))):
        probs = sorted({by[s]["set"] + "/" + by[s]["problem"][:3] for s in stems})
        lines.append(f"| {gname} | {cls} | {len(stems)} | {len(probs)} | {', '.join(probs[:6])} |")
    print("\n".join(lines))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="extract + CPU gates locally (each workload memory-capped)")
    r.add_argument("manifest")
    r.add_argument("out")
    r.add_argument("--stems")
    r.add_argument("--jobs", type=int, default=4, help="workloads at once")
    r.add_argument("--jobs-large", type=int, default=2, help="of which large (io > --large-mib)")
    r.add_argument("--large-mib", type=float, default=32)
    r.add_argument("--mem-large-gb", type=float, default=40)
    r.add_argument("--mem-small-gb", type=float, default=12)
    r.add_argument("--mem-reserve-gb", type=float, default=16, help="host memory left to other users")
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--gates", default="extract,shapes,cpu",
                   help="add cuda,export to run the GPU gates here (default: on the GPU host via `remote`)")
    r.add_argument("--quant", default=None, help="extract_graph --quant: fp32 | fp16 | bf16 | native (per "
                   "problem: bf16 if it has bfloat16 tensors, else fp16 if float16, else fp32); default: the "
                   "manifest's variant")
    r.add_argument("--timeout", type=int, default=1800)
    r.add_argument("--ref-device", default="cpu", help="where the SOL reference run() executes")
    r.add_argument("--score-cast", default=None, help="cast the walk output (and golden) to this torch dtype "
                   "before the SOL check, e.g. bfloat16 for an fp16 diagnostic of a bf16 problem")
    r.add_argument("--arch", default="89")
    r.add_argument("--specs")
    r.add_argument("--hand")
    r.add_argument("--export-cap-mb", type=int, default=1024)
    r.add_argument("--keep", action="store_true", help="keep io.npz")
    r.add_argument("--reserve-mb", type=int, default=1200, help="free disk to leave untouched")
    r.add_argument("--disk-wait-s", type=int, default=1800)
    r.add_argument("--skip-done", action="store_true")
    r.add_argument("--no-hi-ref", action="store_true",
                   help="skip the float64 reference (sol_vs_walk_hi / sol_floor_hi; ~2x the io in memory)")
    o = sub.add_parser("one")
    o.add_argument("row")
    o.add_argument("cfg")
    gp = sub.add_parser("gpu", help="CUDA walk + multi-slot export over IR_ROOT (on the GPU host)")
    gp.add_argument("ir_root")
    gp.add_argument("work")
    gp.add_argument("out_json")
    gp.add_argument("--specs", required=True)
    gp.add_argument("--hand", required=True)
    gp.add_argument("--arch", default="89")
    gp.add_argument("--gates", default="cuda,export")
    gp.add_argument("--timeout", type=int, default=1800)
    gp.add_argument("--export-cap-mb", type=int, default=1024)
    gp.add_argument("--reserve-mb", type=int, default=2000)
    rm = sub.add_parser("remote", help="run `gpu` on an ssh GPU host for the IRs `run` extracted")
    rm.add_argument("out")
    rm.add_argument("--host", required=True)
    rm.add_argument("--ssh-key", required=True)
    rm.add_argument("--remote-dir", required=True)
    rm.add_argument("--python", required=True, help="the remote torch python (used, never modified)")
    rm.add_argument("--gpu", default="2")
    rm.add_argument("--arch", default="89")
    rm.add_argument("--gates", default="cuda,export")
    rm.add_argument("--batch", type=int, default=16)
    rm.add_argument("--stems")
    rm.add_argument("--export-cap-mb", type=int, default=16384)
    # the GPU host's GPUs are shared through KernelBlaster's per-GPU flock (gpu_lock) and checked for
    # free memory per IR (gpu_room); these optional thresholds also wait for an otherwise idle GPU
    rm.add_argument("--idle-util", type=float, default=100)
    rm.add_argument("--idle-mem-mb", type=float, default=1e9)
    rm.add_argument("--until-done", action="store_true", help="keep polling while `run` is still going")
    rm.add_argument("--no-sync", action="store_true", help="the remote code / kernels are current")
    rm.add_argument("--keep", action="store_true", help="keep the local io.npz")
    ec = sub.add_parser("export-check", help="build a kb_export problem, run it holding the GPU lock")
    ec.add_argument("problem")
    ec.add_argument("--arch", default="89")
    ec.add_argument("--lock-dir", required=True)
    ec.add_argument("--timeout", type=int, default=1800)
    s = sub.add_parser("summary")
    s.add_argument("out", nargs="+")
    s.add_argument("--native", action="store_true", help="per stem, only the record at its own dtype")
    s.add_argument("--by-quant", action="store_true")
    a = ap.parse_args(argv)
    return {"run": cmd_run, "one": cmd_one, "gpu": cmd_gpu, "remote": cmd_remote, "summary": cmd_summary,
            "export-check": cmd_export_check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
