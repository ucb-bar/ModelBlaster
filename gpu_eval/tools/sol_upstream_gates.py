#!/usr/bin/env python3
"""Gates for the SOL-ExecBench port (gpu_eval/tools/sol_upstream.py, docs/sol_execbench.md).

Per converted workload (a row of ``manifest.<variant>.json``), in one worker process:

1. **extract**: ``extract_graph --bench-file F --bench-target-mb 0 --quant Q`` (stock dims:
   nothing is resized), with the refusal reason classified when it fails;
2. **shapes**: the IR's graph inputs / outputs have the workload's shapes, and io.npz's inputs
   are the bench's own ``get_inputs()`` values;
3. **cpu**: ``walk_torch`` on the CPU, against the MB golden (``mbir.compare``, MB's bound) and
   against the **SOL ground truth**, the original reference run on the native inputs
   (``--ref-device``, SOL's own GPU by default), with SOL's per-workload tolerance
   (``|out - ref| <= atol + rtol*|ref|`` on >= ``required_matched_ratio`` of the elements, no
   non-finite values, ``max_error_cap``). The MB golden is checked against it the same way: that
   is the port's floor (the fp32 math against the bf16 reference);
4. **cuda**: ``walk_cuda.py --impl hand`` (whole model on the hand kernels, every intermediate
   isolated);
5. **export**: ``kb_export --strategy whole --build-check --runtime host`` (multi-slot problem in
   the persistent driver host), skipped with a reason when its data.bin would exceed
   ``--export-cap-mb``.

io.npz / data.bin are deleted afterwards unless ``--keep``: native shapes are large.

    sol_upstream_gates.py run MANIFEST OUT [--stems REGEX] [--jobs 4] [--gates extract,shapes,cpu,cuda,export]
        [--arch 75] [--ref-device cuda] [--keep]
    sol_upstream_gates.py summary OUT [--md]
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
from concurrent.futures import ProcessPoolExecutor, as_completed
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


@contextmanager
def gpu_lock(out: Path):
    """One GPU user at a time among our workers."""
    with open(out / ".gpu.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ── SOL correctness (sol_execbench/core/bench/correctness.py @834c7c1, Apache-2.0) ──

def sol_check(out, ref, tol: dict) -> dict:
    """SOL's compute_error_stats on one output tensor -> {ok, max_abs, max_rel, matched_ratio, why}."""
    import torch
    x, y = out.reshape(-1).to(torch.float32), ref.reshape(-1).to(torch.float32)
    if x.shape != y.shape:
        return {"ok": False, "why": f"numel {x.numel()} != {y.numel()}"}
    rn, sn = ~torch.isfinite(y), ~torch.isfinite(x)
    if tol.get("allow_negative_inf"):
        both = (y == float("-inf")) & (x == float("-inf"))
        rn, sn = rn & ~both, sn & ~both
        keep = ~both
        x, y = x[keep], y[keep]
    if rn.any() or sn.any():
        return {"ok": False, "why": "non-finite " + ("in reference" if rn.any() else "in output"),
                "nonfinite_ref": int(rn.sum()), "nonfinite_out": int(sn.sum())}
    if float(torch.linalg.vector_norm(y)) > 0 and float(torch.linalg.vector_norm(x)) == 0:
        return {"ok": False, "why": "all-zero output"}
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


# ── one workload ──────────────────────────────────────────────────────────────

def run_one(row: dict, cfg: dict) -> dict:
    out = Path(cfg["out"])
    stem, quant = row["stem"], cfg["quant"]
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
                cpu["sol_vs_walk"] = sol_compare(flat.float(), refs, tol)
                golden = torch.from_numpy(np.asarray(irm.golden_out, np.float32))
                cpu["sol_vs_golden"] = sol_compare(golden, refs, tol)
                cpu["ok"] = bool(cpu["mb_golden"]["ok"])
                cpu["sol_ok"] = bool(cpu["sol_vs_walk"]["ok"])
                cpu["port_floor_ok"] = bool(cpu["sol_vs_golden"]["ok"])
                cpu["range_risk"] = range_risk(refs)
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
            rec["gates"]["cuda"] = {k: r.get(k) for k in keep if k in r}
            rec["gates"]["cuda"]["ok"] = bool(r.get("ok")) and not r.get("first_divergent_op")
            if not rec["gates"]["cuda"]["ok"]:
                rec["gates"]["cuda"]["cls"] = ("missing kernel" if r.get("missing") else
                                               "divergent intermediate" if r.get("first_divergent_op") else
                                               "out of memory" if "memory" in str(r.get("error", "")).lower() else
                                               "golden bound" if "max_abs_err" in r else "error")
        if "export" in gates:
            rec["gates"]["export"] = export_gate(ir, g, out, cfg, env, stem)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-3000:]
    finally:
        rec["s"] = round(time.time() - t0, 1)
        if not cfg.get("keep"):
            for f in ("io.npz",):
                (ir / f).unlink(missing_ok=True)
        (out / "results").mkdir(exist_ok=True)
        (out / "results" / f"{stem}.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec


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
    cmd = [sys.executable, "-m", "modelblaster.pipeline.kb_export", str(ir), str(d), "--strategy", "whole",
           "--build-check", "--runtime", "host", "--arch", str(cfg["arch"])]
    with gpu_lock(out):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg["timeout"], env=env)
            log = p.stdout + p.stderr
        except subprocess.TimeoutExpired:
            log = "timeout"
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


# ── driver / summary ──────────────────────────────────────────────────────────

def _worker(args):
    row, cfg = args
    return run_one(row, cfg)


def cmd_run(a):
    man = json.loads(Path(a.manifest).read_text())
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
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
           "threads": a.threads, "reserve_mb": a.reserve_mb, "disk_wait_s": a.disk_wait_s}
    if a.quant == "fp32" and man["variant"] != "fp32":
        print("warning: fp32 extract of a non-fp32 bench", file=sys.stderr)
    # largest last: the small ones report quickly
    rows.sort(key=lambda r: r["io_mib"])
    n = 0
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(_worker, (r, cfg)) for r in rows]
        for fut in as_completed(futs):
            rec = fut.result()
            n += 1
            gs = " ".join(f"{k}={'ok' if v.get('ok') else ('skip' if v.get('ok') is None else 'FAIL')}"
                          for k, v in rec["gates"].items())
            print(f"[{n}/{len(rows)}] {rec['stem']} {gs} {rec.get('error', '')}", flush=True)
    return 0


def load_results(out: Path) -> list:
    return [json.loads(f.read_text()) for f in sorted((out / "results").glob("*.json"))]


def sol_fail_class(r: dict) -> str:
    """Why an IR walk that meets MB's bound misses SOL's tolerance."""
    c = r["gates"]["cpu"]
    dev = c.get("sol_ref_cpu_vs_gpu")
    if c.get("sol_vs_walk_cpu_ref", {}).get("ok"):
        return "reference device (passes vs the CPU reference run)"
    if dev is not None and not dev.get("ok"):
        return "reference not reproducible at SOL tol (its own CPU vs GPU run fails)"
    if not c.get("port_floor_ok"):
        return ("precision floor (MB golden misses too: the variant's math vs the "
                f"{'/'.join(sorted(set(r.get('ref_dtypes', [])))) or 'native'} reference)")
    return "walk only (MB golden passes)"


def cmd_summary(a):
    recs = load_results(Path(a.out))
    sets = sorted({r["set"] for r in recs}, key=lambda s: ("L1", "L2", "Quant", "FlashInfer-Bench").index(s))
    cols = ["extract", "shapes", "cpu (MB bound)", "cpu (SOL tol)", "golden (SOL tol)", "cuda", "export"]

    def val(r, c):
        g = r["gates"]
        if c == "cpu (MB bound)":
            return g.get("cpu", {}).get("ok")
        if c == "cpu (SOL tol)":
            return g.get("cpu", {}).get("sol_ok")
        if c == "golden (SOL tol)":
            return g.get("cpu", {}).get("port_floor_ok")
        return g.get(c, {}).get("ok")
    lines = ["| set | workloads | problems | " + " | ".join(cols) + " |", "|---" * (len(cols) + 3) + "|"]
    for s in sets + ["all"]:
        rs = [r for r in recs if s == "all" or r["set"] == s]
        probs = {r["problem"] for r in rs}
        cells = []
        for c in cols:
            ok = sum(1 for r in rs if val(r, c) is True)
            pk = len({r["problem"] for r in rs if val(r, c) is True})
            cells.append(f"{ok} ({pk} p)")
        lines.append(f"| {s} | {len(rs)} | {len(probs)} | " + " | ".join(cells) + " |")
    tax = {}
    for r in recs:
        for gname, g in r["gates"].items():
            if g.get("ok") is False:
                k = (gname, g.get("cls") or g.get("why") or "fail")
                tax.setdefault(k, []).append(r["stem"])
        c = r["gates"].get("cpu", {})
        if c.get("ok") and c.get("sol_ok") is False:
            tax.setdefault(("cpu", "SOL tolerance: " + sol_fail_class(r)), []).append(r["stem"])
        if r.get("error"):
            tax.setdefault(("worker", r["error"][:80]), []).append(r["stem"])
    lines += ["", "| gate | class | n | examples |", "|---|---|---|---|"]
    for (gname, cls), stems in sorted(tax.items(), key=lambda kv: (kv[0][0], -len(kv[1]))):
        lines.append(f"| {gname} | {cls} | {len(stems)} | {', '.join(sorted(stems)[:4])} |")
    print("\n".join(lines))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("manifest")
    r.add_argument("out")
    r.add_argument("--stems")
    r.add_argument("--jobs", type=int, default=4)
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--gates", default=",".join(GATES))
    r.add_argument("--quant", default=None, help="extract_graph --quant (default: the manifest's variant)")
    r.add_argument("--timeout", type=int, default=1800)
    r.add_argument("--ref-device", default="cuda")
    r.add_argument("--arch", default="75")
    r.add_argument("--specs")
    r.add_argument("--hand")
    r.add_argument("--export-cap-mb", type=int, default=1024)
    r.add_argument("--keep", action="store_true", help="keep io.npz")
    r.add_argument("--reserve-mb", type=int, default=1200, help="free disk to leave untouched")
    r.add_argument("--disk-wait-s", type=int, default=1800)
    r.add_argument("--skip-done", action="store_true")
    s = sub.add_parser("summary")
    s.add_argument("out")
    s.add_argument("--md", action="store_true")
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_summary(a)


if __name__ == "__main__":
    sys.exit(main())
