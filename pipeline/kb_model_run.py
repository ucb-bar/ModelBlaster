"""The model's result: one whole-model runtime built from every partition's winners.

KernelBlaster optimises the problems ModelBlaster exports -- the whole graph,
or windows / single ops / a cost-ranked selection of it (``kb_export
--strategy window|per_op``, ``--ops``, ``--range``, ``kb_plan``). What counts
is the full model: this step composes every partition's best slot sources
back into the model graph and measures that.

1. **winners** -- the best slots of KernelBlaster runs (``--kb-runs``: run
   directories with ``result.json`` + ``best/slots``), added to a kernel
   cache with each slot's instance shapes as its shape guard, and/or an
   existing cache (``--cache``, e.g. after ``kb_ingest``);
2. **compose** -- the IR with the accepted graph hints applied, every op on
   the cached winner that covers its shape (``kb_ingest.cache_plan``, which
   splits a slot whose instances different partitions tuned), exported as
   ONE whole-graph problem;
3. **check and time** -- the persistent host (``driver.cpp -DMB_HOST``):
   partition outputs against the full reference and every op in isolation,
   then the whole forward pass replayed as a CUDA graph (eager launches as a
   diagnostic, and the eager time where the model is not capturable), the
   hand-seeded original model timed the same way -- in the same process,
   interleaved rep by rep, when the two have the same slots -- for the speedup;
4. **partitions** (``--time-partitions``) -- each run's problem timed the
   same way, so the sum of partition times can be set against the model
   time: the difference is what partition boundaries hide or add (inputs
   that come from the reference instead of the previous window, caches,
   launches across a window edge).

-> ``modelblaster.model_runtime/v1`` (``--out``): ``model_e2e_us`` (graph
replay; the reported result), ``seed_e2e_us``, ``speedup``, eager times,
launch overhead, the composed slots and, with partitions, their sum.

    python -m modelblaster.pipeline.kb_model_run IR_DIR --kb-runs RUN_DIR [...] [--cache DIR] \\
        [--hints graph_hints.json] [--arch 75] [--reps 3] [--time-partitions] --work DIR --out model.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
from pathlib import Path

FORMAT = "modelblaster.model_runtime/v1"


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def find_runs(roots) -> list[Path]:
    """KernelBlaster run directories (result.json + best/slots) under ``roots``."""
    out = []
    for r in roots or []:
        r = Path(r)
        cands = [r] if (r / "result.json").exists() else sorted(p.parent for p in r.rglob("result.json"))
        out += [c for c in cands if (c / "best" / "slots").is_dir()]
    return out


def cache_from_runs(runs: list[Path], cache, specs: dict, arch: str) -> list[dict]:
    """Every verified run's changed best slots into ``cache`` (guard = the slot's instance shapes)."""
    from .kb_ingest import signature_sha
    from .kernel_abi import parse_signature
    from .cuda_dtypes import cuda_signature

    report = []
    for run in runs:
        res = json.loads((run / "result.json").read_text())
        if not res.get("final_ok"):
            report.append({"run": str(run), "status": "final-not-ok"})
            continue
        pdir = Path(res["problem_dir"])
        man = json.loads((pdir / "manifest.json").read_text())
        for s in man["slots"]:
            f = run / "best" / "slots" / Path(s["file"]).name
            spec = specs.get(s["op"])
            if not f.exists() or spec is None:
                report.append({"run": str(run), "slot": s["id"], "status": "no-source" if spec else "unknown-op"})
                continue
            code = f.read_text()
            if code == (pdir / s["file"]).read_text() and s.get("seed_source", "hand") == "hand":
                continue                                        # never improved on the hand seed
            if signature_sha(cuda_signature(spec.signature)) != signature_sha(cuda_signature(s["signature"])):
                report.append({"run": str(run), "slot": s["id"], "status": "stale-signature"})
                continue
            sr = res.get("slots", {}).get(s["id"], {})
            us = sr.get("final_attr_us") or sr.get("final_us") or sr.get("best_us") or 0.0
            base = sr.get("baseline_attr_us") or sr.get("baseline_us")
            entry, _ = parse_signature(spec.signature)
            e, st = cache.add(op=s["op"], entry=entry, signature=spec.signature, arch=f"sm_{arch}",
                              shapes=[i.get("shape", {}) for i in s.get("instances", [])], code=code, slot_us=us,
                              speedup=(base / us) if base and us else None,
                              impl_id=sr.get("impl_id") or f"{res.get('run_id')}:{s['id']}", dtype=man.get("dtype"),
                              source={"run_id": res.get("run_id"), "problem_id": res.get("problem"),
                                      "techniques": sr.get("techniques", [])})
            report.append({"run": str(run), "slot": s["id"], "impl_id": e["impl_id"], "status": st})
    cache.save()
    return report


def _time(prob: Path, *, arch: str, work: Path, sources=None, ref_sources=None, reps: int = 3,
          timeout: int = 3600) -> dict:
    """Check + time a problem in the persistent host (graph replay; eager diagnostic):
    medians over ``reps`` requests; ``ref_sources`` are timed interleaved in the same requests."""
    from .kb_export import DriverHost, build_host, build_so

    host, log = build_host(prob, arch=arch, work=work)
    so, log2 = build_so(prob, arch=arch, sources=sources, work=work, name="model.so") if host else (None, "")
    if host is None or so is None:
        return {"passed": False, "stage": "compile", "log": (log if host is None else log2)[-2000:]}
    ref = None
    if ref_sources is not None:
        ref, _ = build_so(prob, arch=arch, sources=ref_sources, work=work, name="ref.so")
    h = DriverHost(host, timeout=timeout)
    rs = []
    try:
        for k in range(max(1, reps)):
            rs.append(h.evaluate(so, ref=ref, env={"MB_GRAPH": "1"}))
            if not rs[-1]["passed"]:
                break
    finally:
        h.close()
    last = rs[-1]
    out = {"passed": all(r["passed"] for r in rs), "e2e_us": _med([r["e2e_us"] for r in rs]),
           "e2e_basis": last.get("e2e_basis"), "eager_us": _med([r["partition_us"] for r in rs]),
           "graph_us": _med([r["partition_graph_us"] for r in rs]), "graph_skip": last.get("graph_skip"),
           "graph_nodes": last.get("graph_nodes"), "kernel_us": _med([r["attr_sum_us"] for r in rs]),
           "launches": last.get("attr_launches"), "launch_overhead_us": _med([r["launch_overhead_us"] for r in rs]),
           "launch_overhead_frac": _med([r["launch_overhead_frac"] for r in rs]),
           "e2e_runs": [r["e2e_us"] for r in rs], "slots": last.get("slots"), "outputs": last.get("outputs"),
           "attr_slot_us": last.get("attr_slot_us"), "errors": last.get("errors")}
    if ref is not None:
        out.update(ref_e2e_us=_med([r["ref_e2e_us"] for r in rs]),
                   paired_ratio=_med([(r.get("e2e_ratio") or {}).get("med") for r in rs]),
                   paired_basis=(last.get("e2e_ratio") or {}).get("basis"))
    if not out["passed"]:
        out["log"] = last.get("log", "")[-2000:]
    return out


def _slot_texts(prob: Path) -> dict:
    man = json.loads((prob / "manifest.json").read_text())
    return {s["id"]: (prob / s["file"]).read_text() for s in man["slots"]}


def model_runtime(ir_dir: Path, *, work: Path, cache=None, runs=(), hints: Path | None = None, arch: str = "75",
                  reps: int = 3, time_partitions: bool = False, keep: bool = False, specs: dict | None = None) -> dict:
    """Compose the winners into the whole model, check it and time it (see the module doc)."""
    from . import graph_hints as gh
    from .kb_export import export_problem
    from .kb_ingest import CudaCache, cache_plan
    from .kb_proposals import _all_specs, write_variant_ir

    ir_dir, work = Path(ir_dir), Path(work)
    arch = str(arch).removeprefix("sm_")
    work.mkdir(parents=True, exist_ok=True)
    specs = dict(specs or _all_specs())
    g = json.loads((ir_dir / "graph.json").read_text())
    hint_ids = []
    if hints is not None and Path(hints).exists():
        g, added, prov = gh.apply_hints(g, gh.load(Path(hints))["hints"], specs)
        specs.update(added)
        hint_ids = sorted({a["id"] for a in prov["applied"]})
    # a working cache: the given one plus the runs' best slots (the given one is not modified)
    wc = CudaCache(work / "cache")
    wc.entries = [dict(e) for e in cache.entries] if cache is not None else []
    if cache is not None:
        (work / "cache" / "src").mkdir(parents=True, exist_ok=True)
        for e in wc.entries:
            dst = work / "cache" / e["file"]
            if not dst.exists():
                shutil.copyfile(cache.root / e["file"], dst)
    run_dirs = find_runs(runs)
    added = cache_from_runs(run_dirs, wc, specs, arch)
    plan = cache_plan(g, wc, f"sm_{arch}", specs)
    vir = write_variant_ir(ir_dir, plan["graph"], work / "ir")
    comp = export_problem(vir, work / "model", slot_sources=plan["sources"], slot_source_tags=plan["tags"],
                          sub_sources=plan["sub_sources"], specs=specs, strategy="whole")
    seed = export_problem(ir_dir, work / "seed", strategy="whole")
    cm, sm = (json.loads((d / "manifest.json").read_text()) for d in (comp, seed))
    # the same slots and ops: the seed model is the composed problem with the hand kernels,
    # timed in the same requests (paired); otherwise (fused or split slots) on its own
    pair = [s["id"] for s in cm["slots"]] == [s["id"] for s in sm["slots"]] and \
        [o["name"] for o in cm["ops"]] == [o["name"] for o in sm["ops"]]
    model = _time(comp, arch=arch, work=work / "build_model", reps=reps,
                  ref_sources=_slot_texts(seed) if pair else None)
    seed_t = None if pair else _time(seed, arch=arch, work=work / "build_seed", reps=reps)
    seed_us = model.get("ref_e2e_us") if pair else (seed_t or {}).get("e2e_us")
    out = {"format": FORMAT, "ir": cm["ir_name"], "source_ir": str(ir_dir), "arch": f"sm_{arch}",
           "passed": model["passed"], "model_e2e_us": model.get("e2e_us"), "model_basis": model.get("e2e_basis"),
           "model_eager_us": model.get("eager_us"), "graph_skip": model.get("graph_skip"),
           "seed_e2e_us": seed_us, "seed_basis": "graph" if pair and model.get("paired_basis") == "graph"
           else (seed_t or {}).get("e2e_basis") if seed_t else model.get("paired_basis"),
           "speedup": (seed_us / model["e2e_us"]) if seed_us and model.get("e2e_us") else None,
           "paired": pair, "paired_ratio": model.get("paired_ratio"),
           "speedup_paired": (1.0 / model["paired_ratio"]) if model.get("paired_ratio") else None,
           "model": model, "seed": seed_t, "hints": hint_ids, "cache_split": [h["id"] for h in plan["hints"]],
           "composed_slots": {sid: tag for sid, tag in plan["tags"].items()},
           "runs": [str(r) for r in run_dirs], "winners_added": added}
    if time_partitions and run_dirs:
        parts = []
        for k, run in enumerate(run_dirs):
            res = json.loads((run / "result.json").read_text())
            pdir = Path(res["problem_dir"])
            best = {p.stem: p.read_text() for p in (run / "best" / "slots").glob("*.cu")}
            t = _time(pdir, arch=arch, work=work / f"build_part{k:02d}", sources=best, ref_sources=_slot_texts(pdir),
                      reps=reps)
            parts.append({"problem": res.get("problem"), "passed": t["passed"], "e2e_us": t.get("e2e_us"),
                          "seed_e2e_us": t.get("ref_e2e_us"), "basis": t.get("e2e_basis"),
                          "kb_final_e2e_us": res.get("final_e2e_us"), "kb_final_partition_us": res.get("final_partition_us")})
        sp = sum(p["e2e_us"] or 0.0 for p in parts)
        ss = sum(p["seed_e2e_us"] or 0.0 for p in parts)
        out["partitions"] = parts
        out["sum_partitions_e2e_us"] = sp
        out["sum_partitions_seed_e2e_us"] = ss
        if out["model_e2e_us"]:
            out["boundary_effect_us"] = out["model_e2e_us"] - sp     # > 0: the model costs more than its parts
    if not keep:
        for d in work.glob("*"):
            if d.name.startswith("build") or d.name in ("model", "seed", "ir"):
                shutil.rmtree(d, ignore_errors=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ir_dir")
    ap.add_argument("--kb-runs", nargs="*", default=[], help="KernelBlaster run dirs (or roots holding them)")
    ap.add_argument("--cache", help="ModelBlaster CUDA kernel cache to compose from too (read only)")
    ap.add_argument("--hints", help="graph hints file: its accepted hints are applied")
    ap.add_argument("--arch", default="75")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--time-partitions", action="store_true",
                    help="also time every run's partition the same way, for the sum of partition times")
    ap.add_argument("--work", required=True)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--out", help="write the result here (also printed)")
    a = ap.parse_args()
    from .kb_ingest import CudaCache

    res = model_runtime(Path(a.ir_dir), work=Path(a.work), cache=CudaCache(Path(a.cache)) if a.cache else None,
                        runs=a.kb_runs, hints=Path(a.hints) if a.hints else None, arch=a.arch, reps=a.reps,
                        time_partitions=a.time_partitions, keep=a.keep)
    txt = json.dumps(res, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(txt)
    print(json.dumps({k: res.get(k) for k in ("ir", "passed", "model_e2e_us", "model_basis", "seed_e2e_us", "speedup",
                                               "speedup_paired", "sum_partitions_e2e_us", "boundary_effect_us")}))


if __name__ == "__main__":
    main()
