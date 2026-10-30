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
4. **partitions** (``--time-partitions``) -- each run's problem with its best
   slots, paired with the hand seeds, timed the same way, so the sum of
   partition times can be set against the model time: the difference is what
   partition boundaries hide or add (inputs that come from the reference
   instead of the previous window, caches, launches across a window edge).

Steps 3-4 run through KernelBlaster's evaluators (``kb_eval``: its
``scripts/time_problems.py``): ``--evaluator local`` (this host's GPU pool
and locks) or ``ssh`` (``--ssh-host``, remote GPUs; big ``data.bin`` copied
once per host), all on one GPU by default (``--devices``), and on the **GPU
model and arch the KernelBlaster runs recorded** (``result.json`` ``gpu``;
or ``--gpu-model`` / ``--arch``, which must agree): no GPU of that model is an
error (``--allow-gpu-mismatch``: time anyway, with warnings), so the model
number and the component numbers come from the same hardware.
``--evaluator builtin`` keeps ModelBlaster's own host on the default GPU.

5. **against PyTorch** (``--torch-baseline auto|on|off``, default auto): the
   original PyTorch module with the IR's weights and inputs, on the same GPU
   (UUID), eager / CUDA graph / ``torch.compile``, TF32 off and on, timed in ABBA
   blocks with the model the same way (GPU events and host wall clock per
   forward pass, steady-state throughput) -- ``pipeline/torch_baseline.py``.

-> ``modelblaster.model_runtime/v1`` (``--out``): ``model_e2e_us`` (graph
replay; the reported result), ``seed_e2e_us``, ``speedup``, eager times,
launch overhead, the composed slots, the device (``device``, ``gpu_name``,
``gpu_uuid``, ``gpu_host``), ``hardware`` (required vs recorded, ``match``)
and, with partitions, each one's device and their sum; with the baseline
``torch_baseline`` (the table) and ``torch_us`` / ``torch_*_us`` / ``speedup_vs_torch``
(``_wall``, ``_eager``).

    python -m modelblaster.pipeline.kb_model_run IR_DIR --kb-runs RUN_DIR [...] [--cache DIR] \\
        [--hints graph_hints.json] [--arch 89] [--reps 3] [--time-partitions] --work DIR --out model.json \\
        [--evaluator local --gpus auto | --evaluator ssh --ssh-host H [--ssh-host H2] --remote-dir D \\
         [--cuda-visible-devices auto|ids] [--ssh-key K]] [--runtime auto|host|process] [--gpu-model M] \\
        [--torch-baseline auto|on|off] [--torch-python PY] [--torch-source FILE] [--torch-rounds 4]
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from pathlib import Path

from . import kb_telemetry as kbt

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
    """Every verified run's changed best slots into ``cache`` (guard = the slot's instance shapes);
    runs of another slot target than the cache's are skipped."""
    from .kb_ingest import arch_key, signature_sha
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
        if man.get("target", "cuda") != cache.target:
            report.append({"run": str(run), "status": "other-target", "target": man.get("target", "cuda")})
            continue
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
            e, st = cache.add(op=s["op"], entry=entry, signature=spec.signature, arch=arch_key(arch, cache.target),
                              shapes=[i.get("shape", {}) for i in s.get("instances", [])], code=code, slot_us=us,
                              speedup=(base / us) if base and us else None,
                              impl_id=sr.get("impl_id") or f"{res.get('run_id')}:{s['id']}",
                              dtype=s.get("dtype") or man.get("dtype"),   # a mixed problem: the slot's
                              source={"run_id": res.get("run_id"), "problem_id": res.get("problem"),
                                      "techniques": sr.get("techniques", []),
                                      **({"gpu": res["gpu"]} if res.get("gpu") else {})})
            report.append({"run": str(run), "slot": s["id"], "impl_id": e["impl_id"], "status": st})
    cache.save()
    return report


def _time_builtin(prob: Path, *, arch: str, work: Path, sources=None, ref_sources=None, reps: int = 3,
                  timeout: int = 3600) -> dict:
    """``--evaluator builtin``: check + time a problem in ModelBlaster's own persistent host on
    this machine's default GPU (graph replay; eager diagnostic): medians over ``reps``
    requests; ``ref_sources`` are timed interleaved in the same requests. No GPU pool or locks."""
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
           "attr_slot_us": last.get("attr_slot_us"), "errors": last.get("errors"), "runtime": "host",
           "device": _builtin_device(arch)}
    if ref is not None:
        out.update(ref_e2e_us=_med([r["ref_e2e_us"] for r in rs]),
                   paired_ratio=_med([(r.get("e2e_ratio") or {}).get("med") for r in rs]),
                   paired_basis=(last.get("e2e_ratio") or {}).get("basis"))
    if last.get("sol"):                            # a SOL-gated export (sol_tolerance.py)
        out["sol"] = last["sol"]
    if not out["passed"]:
        out["log"] = last.get("log", "")[-2000:]
    return out


def _builtin_device(arch: str) -> dict | None:
    from .kb_eval import builtin_device

    return builtin_device(arch)


def _from_kb(job: dict, *, identical: bool = False) -> dict:
    """A ``time_problems`` job (KernelBlaster EvalResults, one per request) -> the same
    fields as a builtin timing: medians over the requests, the paired seed numbers."""
    rs = job.get("runs") or []
    if not rs:
        return {"passed": False, "stage": "run", "log": "no run", "device": job.get("gpu")}
    last = rs[-1]

    def dep(r):                                    # as deployed: MB_E2E_US, else graph, else eager
        return r.get("e2e_us") or r.get("partition_graph_us") or r.get("partition_us")
    out = {"passed": all(r.get("ok") for r in rs), "e2e_us": _med([dep(r) for r in rs]),
           "e2e_basis": last.get("e2e_basis"), "eager_us": _med([r.get("partition_us") for r in rs]),
           "graph_us": _med([r.get("partition_graph_us") for r in rs]), "graph_skip": last.get("graph_skip"),
           "graph_nodes": last.get("graph_nodes"), "kernel_us": _med([r.get("attr_sum_us") for r in rs]),
           "launches": last.get("attr_launches"), "launch_overhead_us": _med([r.get("launch_overhead_us") for r in rs]),
           "launch_overhead_frac": _med([r.get("launch_overhead_frac") for r in rs]),
           "e2e_runs": [dep(r) for r in rs], "slots": last.get("slot_checks"), "outputs": last.get("outputs"),
           "attr_slot_us": last.get("attr_slot_us"), "errors": last.get("errors"), "runtime": last.get("runtime"),
           "device": job.get("gpu"), "devices": job.get("gpus"), "eval_wall_s": _med([r.get("eval_wall_s") for r in rs])}
    if job.get("paired"):
        if identical:                              # the composed model is the seed model: one timing
            out.update(ref_e2e_us=out["e2e_us"], paired_ratio=1.0, paired_basis=out["e2e_basis"], identical=True)
        elif job.get("pairing") == "host":
            out.update(ref_e2e_us=_med([r.get("ref_e2e_us") for r in rs]),
                       paired_ratio=_med([(r.get("e2e_ratio") or {}).get("med") for r in rs]),
                       paired_basis=(last.get("e2e_ratio") or {}).get("basis"))
        else:                                      # one process per run: the seed right after, same round
            refs = job.get("ref_runs") or []
            ratios = [dep(r) / dep(q) for r, q in zip(rs, refs) if dep(r) and dep(q)]
            out.update(ref_e2e_us=_med([dep(q) for q in refs]), paired_ratio=_med(ratios),
                       paired_basis=refs[-1].get("e2e_basis") if refs else None)
    if last.get("sol"):                            # a SOL-gated export: every run's verdicts, the last one's record
        out["sol"] = last["sol"]
        out["sol_runs_ok"] = [all((d.get(d.get("gate")) or {}).get("ok", True) for d in (r.get("sol") or {}).values())
                              for r in rs]
    if not out["passed"]:
        out["log"] = (last.get("log") or "\n".join(map(str, last.get("errors") or [])))[-2000:]
    return out


def _slot_texts(prob: Path) -> dict:
    man = json.loads((prob / "manifest.json").read_text())
    return {s["id"]: (prob / s["file"]).read_text() for s in man["slots"]}


_AUTO = object()


def _default_eval():
    from .kb_eval import KBEval

    ev = KBEval()
    return ev if ev.available() else None


def _part_jobs(run_dirs: list[Path], reps: int, hand: dict) -> tuple[list[dict], list[dict]]:
    """Each run's problem with its best slots, paired with the hand seeds (``hand``: the
    whole-model seed export's slot sources, by slot id; a slot the seed model has not --
    a fused or split slot -- keeps the problem's own seed)."""
    jobs, meta = [], []
    for k, run in enumerate(run_dirs):
        res = json.loads((run / "result.json").read_text())
        pdir = Path(res["problem_dir"])
        own = _slot_texts(pdir)
        best = {sid: (run / "best" / "slots" / Path(f).name).read_text() for sid, f in _slot_files(pdir).items()
                if (run / "best" / "slots" / Path(f).name).exists()}
        inc = {sid: hand.get(sid, txt) for sid, txt in own.items()}
        jobs.append({"id": f"part{k:02d}", "problem": str(pdir), "sources": best, "incumbent": inc, "reps": reps})
        meta.append({"res": res, "pdir": pdir, "best": best, "identical": {**own, **best} == inc,
                     "seed_kind": "hand" if all(sid in hand for sid in own) else "problem"})
    return jobs, meta


def _slot_files(prob: Path) -> dict:
    return {s["id"]: s["file"] for s in json.loads((prob / "manifest.json").read_text())["slots"]}


def model_runtime(ir_dir: Path, *, work: Path, cache=None, runs=(), hints: Path | None = None, arch: str | None = None,
                  reps: int = 3, time_partitions: bool = False, keep: bool = False, specs: dict | None = None,
                  kbeval=_AUTO, gpu_model: str | None = None, allow_gpu_mismatch: bool = False,
                  recorded: list | None = None, torch_baseline: str = "off", torch_opts: dict | None = None,
                  target: str | None = None) -> dict:
    """Compose the winners into the whole model, check it and time it (see the module doc).

    ``kbeval``: a :class:`kb_eval.KBEval` (KernelBlaster's local or ssh evaluator), None for
    ModelBlaster's builtin host on the default GPU; by default the local evaluator of the
    KernelBlaster checkout (``$MB_KB_ROOT`` / the submodule) when it has one, else builtin.
    ``gpu_model`` / ``arch``: required hardware (default: what the KernelBlaster runs recorded,
    see ``kb_eval.required_hardware``; ``recorded`` adds GPU records of runs not in ``runs``,
    e.g. ``kb_graph_opt``'s rounds). ``allow_gpu_mismatch`` times anyway, with warnings.
    ``torch_baseline`` (off | auto | on) with ``torch_opts`` (:func:`torch_comparison`): the
    model against the PyTorch module on the same GPU (``torch_baseline``, ``torch_*`` fields).

    ``target`` (default: the cache's slot target, else cuda): ``opencl`` composes OpenCL slots and
    times them through ``kbeval`` (the board: one process per run, ``MB_E2E_US basis=replay``, the
    seed model timed right after the composed one in each round); ``arch`` is the device label.
    There is no PyTorch on the board: the reported baseline is the seed model there, and
    ``torch_baseline`` records why not (``status: skipped``)."""
    from . import graph_hints as gh
    from . import kb_eval as ke
    from .kb_export import export_problem
    from .kb_ingest import CudaCache, arch_key, cache_plan
    from .kb_proposals import _all_specs, write_variant_ir

    T = kbt.get()
    target = target or (cache.target if cache is not None else "cuda")
    if kbeval is _AUTO:
        kbeval = _default_eval()
    if target != "cuda" and kbeval is None:
        raise ValueError(f"the {target} model is timed through KernelBlaster's evaluator (kbeval): ModelBlaster's "
                         "builtin host is CUDA-only")
    ir_dir, work = Path(ir_dir), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    run_dirs = find_runs(runs)
    rec = ke.recorded_gpus(run_dirs) + list(recorded or [])
    hw = ke.required_hardware(rec, gpu_model=gpu_model, arch=arch, allow_mismatch=allow_gpu_mismatch)
    arch = hw["arch"] or ("75" if target == "cuda" else None)
    if arch is None:
        raise ValueError(f"{target}: pass the device label as arch (e.g. adreno650)")
    hw["arch"] = arch_key(arch, target)
    with T.span("mb.model_run.compose", ir=ir_dir.name):
        specs = dict(specs or _all_specs())
        g = json.loads((ir_dir / "graph.json").read_text())
        hint_ids = []
        if hints is not None and Path(hints).exists():
            g, added, prov = gh.apply_hints(g, gh.load(Path(hints))["hints"], specs, target=target)
            specs.update(added)
            hint_ids = sorted({a["id"] for a in prov["applied"]})
        # a working cache: the given one plus the runs' best slots (the given one is not modified)
        wc = CudaCache(work / "cache", target=target)
        wc.entries = [dict(e) for e in cache.entries] if cache is not None else []
        if cache is not None:
            (work / "cache" / "src").mkdir(parents=True, exist_ok=True)
            for e in wc.entries:
                dst = work / "cache" / e["file"]
                if not dst.exists():
                    shutil.copyfile(cache.root / e["file"], dst)
        added = cache_from_runs(run_dirs, wc, specs, arch)
        plan = cache_plan(g, wc, arch_key(arch, target), specs)
        # cache entries carry the GPU of the run they came from (cache_from_runs): they must agree too
        used_gpus = [{"run": f"cache:{e.get('impl_id')}", "gpu": (e.get("source") or {}).get("gpu"), "arch": e.get("arch")}
                     for e in plan.get("used", {}).values() if (e.get("source") or {}).get("gpu")]
        if used_gpus:
            hw2 = ke.required_hardware(rec + used_gpus, gpu_model=gpu_model, arch=arch,
                                       allow_mismatch=allow_gpu_mismatch)
            hw.update({k: hw2[k] for k in ("gpu_model", "recorded_models", "verified")})
            hw["warnings"] = list(dict.fromkeys(hw["warnings"] + hw2["warnings"]))
            if hw2["gpu_model"]:
                hw["warnings"] = [w for w in hw["warnings"] if not w.startswith("no KernelBlaster run recorded")]
        vir = write_variant_ir(ir_dir, plan["graph"], work / "ir")
        comp = export_problem(vir, work / "model", slot_sources=plan["sources"], slot_source_tags=plan["tags"],
                              sub_sources=plan["sub_sources"], specs=specs, strategy="whole", target=target)
        seed = export_problem(ir_dir, work / "seed", strategy="whole", target=target)
        cm, sm = (json.loads((d / "manifest.json").read_text()) for d in (comp, seed))
        # the same slots and ops: the seed model is the composed problem with the hand kernels,
        # timed in the same requests (paired); otherwise (fused or split slots) on its own
        pair = [s["id"] for s in cm["slots"]] == [s["id"] for s in sm["slots"]] and \
            [o["name"] for o in cm["ops"]] == [o["name"] for o in sm["ops"]]
        seed_texts = _slot_texts(seed)
        identical = pair and _slot_texts(comp) == seed_texts
        if pair and (seed / sm.get("data", "data.bin")).exists():
            (seed / sm.get("data", "data.bin")).unlink()          # paired: only its kernels are used (big models)
    for w in hw["warnings"]:
        print(f"[kb_model_run] WARNING: {w}", file=sys.stderr, flush=True)
    part_jobs, part_meta = _part_jobs(run_dirs, reps, seed_texts) if time_partitions and run_dirs else ([], [])
    with T.span("mb.model_run.time", ir=ir_dir.name, evaluator=kbeval.evaluator if kbeval else "builtin",
                partitions=len(part_jobs)) as sp:
        if kbeval is not None:
            jobs = [{"id": "model", "problem": str(comp), "sources": None,
                     "incumbent": seed_texts if pair else None, "reps": reps}]
            if not pair:
                jobs.append({"id": "seed", "problem": str(seed), "sources": None, "reps": reps})
            jobs += part_jobs
            try:
                kr = kbeval.run(jobs, arch=arch, work=work / "kb_time", gpu_model=hw["gpu_model"], name="model")
            except ke.HardwareMismatch as e:
                if not allow_gpu_mismatch:
                    raise ke.HardwareMismatch(f"{e} -- the components were optimised on {hw['gpu_model']}; select "
                                              "GPUs of that model (--gpus / --ssh-host / --cuda-visible-devices) or "
                                              "pass --allow-gpu-mismatch") from None
                hw["warnings"].append(f"{e}; timed on another GPU model (--allow-gpu-mismatch)")
                print(f"[kb_model_run] WARNING: {hw['warnings'][-1]}", file=sys.stderr, flush=True)
                kr = kbeval.run(jobs, arch=arch, work=work / "kb_time", gpu_model=None, name="model")
            J = kr["jobs"]
            model = _from_kb(J["model"], identical=identical)
            seed_t = None if pair else _from_kb(J["seed"])
            parts_t = [_from_kb(J[j["id"]], identical=m["identical"]) for j, m in zip(part_jobs, part_meta)]
            ktime = {k: kr.get(k) for k in ("seconds", "pool", "models", "host_stats", "pool_stats")}
            if kr.get("telemetry"):
                ktime["telemetry"] = kbt.compact(kr["telemetry"])
        else:
            model = _time_builtin(comp, arch=arch, work=work / "build_model", reps=reps,
                                  ref_sources=seed_texts if pair else None)
            if identical and model.get("ref_e2e_us") is None and model.get("e2e_us"):
                model.update(ref_e2e_us=model["e2e_us"], paired_ratio=1.0, paired_basis=model.get("e2e_basis"))
            seed_t = None if pair else _time_builtin(seed, arch=arch, work=work / "build_seed", reps=reps)
            parts_t = [_time_builtin(m["pdir"], arch=arch, work=work / f"build_part{k:02d}", sources=m["best"],
                                     ref_sources=j["incumbent"], reps=reps)
                       for k, (j, m) in enumerate(zip(part_jobs, part_meta))]
            ktime = None
        if sp is not None and hasattr(sp, "set"):
            sp.set(device=(model.get("device") or {}).get("label"), gpu=(model.get("device") or {}).get("name"),
                   passed=model.get("passed"))
    dev = model.get("device") or {}
    match = ke.same_gpu_model(dev.get("name"), hw["gpu_model"]) if hw["gpu_model"] else None
    if match is False and not allow_gpu_mismatch:
        raise ke.HardwareMismatch(f"the model was timed on {dev.get('name')!r}, not on {hw['gpu_model']} "
                                  "(the components' GPU model)")
    if hw["gpu_model"] is None and dev.get("name"):
        hw["warnings"].append(f"timed on {dev['name']} with no recorded component GPU to match")
    hw.update(device=dev or None, match=match)
    seed_us = model.get("ref_e2e_us") if pair else (seed_t or {}).get("e2e_us")
    out = {"format": FORMAT, "ir": cm["ir_name"], "source_ir": str(ir_dir), "arch": arch_key(arch, target),
           "passed": model["passed"], "model_e2e_us": model.get("e2e_us"), "model_basis": model.get("e2e_basis"),
           "model_eager_us": model.get("eager_us"), "graph_skip": model.get("graph_skip"),
           "seed_e2e_us": seed_us, "seed_basis": "graph" if pair and model.get("paired_basis") == "graph"
           else (seed_t or {}).get("e2e_basis") if seed_t else model.get("paired_basis"),
           "speedup": (seed_us / model["e2e_us"]) if seed_us and model.get("e2e_us") else None,
           "paired": pair, "paired_ratio": model.get("paired_ratio"),
           "speedup_paired": (1.0 / model["paired_ratio"]) if model.get("paired_ratio") else None,
           "identical_to_seed": identical,
           "evaluator": kbeval.describe() if kbeval is not None else {"evaluator": "builtin", "runtime": "host"},
           "device": dev or None, "gpu_name": dev.get("name"), "gpu_uuid": dev.get("uuid"), "gpu_host": dev.get("host"),
           "hardware": hw,
           "model": model, "seed": seed_t, "hints": hint_ids, "cache_split": [h["id"] for h in plan["hints"]],
           "composed_slots": {sid: tag for sid, tag in plan["tags"].items()},
           "runs": [str(r) for r in run_dirs], "winners_added": added}
    if ktime is not None:
        out["kb_time"] = ktime
    if part_jobs:
        parts = []
        for j, m, t in zip(part_jobs, part_meta, parts_t):
            res = m["res"]
            pd = t.get("device") or {}
            parts.append({"problem": res.get("problem"), "passed": t["passed"], "e2e_us": t.get("e2e_us"),
                          "seed_e2e_us": t.get("ref_e2e_us"), "basis": t.get("e2e_basis"),
                          "eager_us": t.get("eager_us"), "paired_ratio": t.get("paired_ratio"),
                          "seed_kind": m["seed_kind"],
                          "kb_final_e2e_us": res.get("final_e2e_us"), "kb_final_partition_us": res.get("final_partition_us"),
                          "kb_gpu": (res.get("gpu") or {}).get("name"), "device": pd or None,
                          "same_device": bool(pd and dev and pd.get("label") == dev.get("label")
                                              and pd.get("host") == dev.get("host")),
                          "same_gpu_model": ke.same_gpu_model(pd.get("name"), dev.get("name")) if pd and dev else None})
        sp_ = sum(p["e2e_us"] or 0.0 for p in parts)
        ss = sum(p["seed_e2e_us"] or 0.0 for p in parts)
        out["partitions"] = parts
        out["sum_partitions_e2e_us"] = sp_
        out["sum_partitions_seed_e2e_us"] = ss
        if out["model_e2e_us"]:
            out["boundary_effect_us"] = out["model_e2e_us"] - sp_     # > 0: the model costs more than its parts
    if target != "cuda":
        out["target"] = target
    if torch_baseline and torch_baseline != "off" and target != "cuda":
        # the PyTorch baseline is CUDA-only (torch_baseline.compare times the module on the model's
        # CUDA GPU by UUID); the board has no torch: the baseline is the seed model, timed above
        why = (f"no PyTorch baseline on the {target} device (no torch there; the baseline is the seed model, "
               f"seed_e2e_us)")
        print(f"[kb_model_run] torch baseline skipped: {why}", file=sys.stderr, flush=True)
        if torch_baseline == "on":
            raise RuntimeError(f"torch baseline skipped: {why}")
        from .torch_baseline import FORMAT as TB_FORMAT, summary_fields
        tres = {"format": TB_FORMAT, "status": "skipped", "reason": why}
        out["torch_baseline"] = tres
        out.update(summary_fields(tres))
    elif torch_baseline and torch_baseline != "off":
        tres = torch_comparison(ir_dir, comp, model, mode=torch_baseline, work=work / "torch", kbeval=kbeval,
                                arch=arch, gpu_model=hw["gpu_model"], **(torch_opts or {}))
        out["torch_baseline"] = tres
        from .torch_baseline import summary_fields, table
        out.update(summary_fields(tres))
        print(table(tres), file=sys.stderr, flush=True)
    if not keep:
        for d in work.glob("*"):
            if d.name.startswith("build") or d.name in ("model", "seed", "ir"):
                shutil.rmtree(d, ignore_errors=True)
        for d in [*(work / "kb_time").glob("*_kbwork"), *(work / "torch" / "kb_time").glob("*_kbwork"),
                  work / "torch" / "build_ours", work / "torch" / "bundle"]:
            shutil.rmtree(d, ignore_errors=True)
    from . import kb_trace
    kb_trace.record("model_run", mb_stage="model_run", ir=Path(ir_dir).name, work=str(work), result=out)
    return out


def torch_comparison(ir_dir: Path, comp: Path, model: dict, *, mode: str, work: Path, kbeval, arch: str,
                     gpu_model: str | None, python: str | None = None, source: str | None = None, rounds: int = 4,
                     variants=None, warm: int = 10, budget_ms: float = 200.0, thru_n: int = 100) -> dict:
    """The composed model against PyTorch on the model run's GPU (``torch_baseline.compare``).

    ``mode``: ``auto`` runs it when it can -- the model passed, its device has a UUID, the
    IR's PyTorch source is found, and there is a torch python for the timing host
    (``python`` / ``$MB_TORCH_PYTHON``; locally also KernelBlaster's python when it imports
    torch with CUDA) -- else records why not (``status: skipped``), and a failure is
    ``status: failed`` (the model's own result stands); ``on`` raises instead."""
    import os
    from . import torch_baseline as tb

    if mode not in ("auto", "on"):
        raise ValueError(f"torch_baseline must be auto|on|off, not {mode!r}")
    dev = model.get("device") or {}
    python = python or os.environ.get("MB_TORCH_PYTHON") or None

    def no(status, why):
        if mode == "on":
            raise RuntimeError(f"torch baseline {status}: {why}")
        print(f"[kb_model_run] torch baseline {status}: {why}", file=sys.stderr, flush=True)
        return {"format": tb.FORMAT, "status": status, "reason": why}
    if not model.get("passed"):
        return no("skipped", "the model did not pass its checks")
    if not dev.get("uuid"):
        return no("skipped", "the model's device has no UUID")
    if source is None and tb.find_source(ir_dir) is None:
        return no("skipped", f"no PyTorch source for {Path(ir_dir).name} (--torch-source / $MB_TORCH_SRC)")
    if dev.get("host"):
        if not python:
            return no("skipped", f"no torch python for {dev['host']} (--torch-python / $MB_TORCH_PYTHON)")
    elif not python:
        cands = [getattr(kbeval, "python", None), sys.executable]
        python = next((p for p in cands if p and tb.torch_python_ok(p)), None)
        if python is None:
            return no("skipped", "no local python with torch + CUDA (--torch-python)")
    try:
        with kbt.get().span("mb.model_run.torch", ir=Path(ir_dir).name, device=dev.get("label")):
            return tb.compare(ir_dir=ir_dir, problem=comp, device=dev, work=work, kbeval=kbeval, arch=arch,
                              gpu_model=gpu_model, python=python, source=source, rounds=rounds,
                              variants=tuple(variants or tb.VARIANTS), warm=warm, budget_ms=budget_ms, thru_n=thru_n)
    except tb.Skip as e:
        return no("skipped", str(e))
    except Exception as e:  # noqa: BLE001 -- auto: reported next to the model's result
        if mode == "on":
            raise
        return no("failed", f"{type(e).__name__}: {e}")


def add_torch_args(ap) -> None:
    """``--torch-*`` flags (``kb_model_run``; ``kb_graph_opt`` has the same names)."""
    g = ap.add_argument_group("PyTorch baseline (docs/gpu_harness_perf.md, 'Against PyTorch')")
    g.add_argument("--torch-baseline", choices=["auto", "on", "off"], default="auto",
                   help="time the PyTorch module on the model's GPU, interleaved with the model: auto (default) "
                        "when the source and a torch python are there, on (an error otherwise), off")
    g.add_argument("--torch-python", help="python with torch on the timing host (default $MB_TORCH_PYTHON; local: "
                                          "--kb-python / this python when it imports torch)")
    g.add_argument("--torch-source", help="the model's PyTorch source (default: found next to the IR set)")
    g.add_argument("--torch-rounds", type=int, default=4, help="ABBA rounds of ours / torch blocks (default 4)")
    g.add_argument("--torch-variants", default="eager,graph,compile_mna,compile_ro",
                   help="torch variants: eager, graph, compile_mna (max-autotune-no-cudagraphs), compile_ro "
                        "(reduce-overhead)")


def torch_opts(a) -> dict:
    return {"python": a.torch_python, "source": a.torch_source, "rounds": a.torch_rounds,
            "variants": [v for v in a.torch_variants.split(",") if v]}


def main():
    from . import kb_eval as ke

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ir_dir")
    ap.add_argument("--kb-runs", nargs="*", default=[], help="KernelBlaster run dirs (or roots holding them)")
    ap.add_argument("--cache", help="ModelBlaster CUDA kernel cache to compose from too (read only)")
    ap.add_argument("--hints", help="graph hints file: its accepted hints are applied")
    ap.add_argument("--arch", default=None, help="SM number (default: the KernelBlaster runs' arch, else 75); "
                    "OpenCL: the device label (adreno650)")
    ap.add_argument("--target", default="cuda", choices=["cuda", "opencl"],
                    help="slot target of the cache and the composed model (opencl: --evaluator board|local)")
    ap.add_argument("--gpu-model", default=None,
                    help="GPU model to time on, e.g. 'NVIDIA L40S' (default: the one the KernelBlaster runs recorded)")
    ap.add_argument("--allow-gpu-mismatch", action="store_true",
                    help="time even where no GPU of the components' model / arch is available (loud warnings)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--time-partitions", action="store_true",
                    help="also time every run's partition the same way, for the sum of partition times")
    ap.add_argument("--work", required=True)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--out", help="write the result here (also printed)")
    ke.add_args(ap)
    add_torch_args(ap)
    a = ap.parse_args()
    from .kb_ingest import CudaCache

    kbeval = ke.from_args(a)
    tel = kbt.load(kbeval.root if kbeval is not None else ke.default_kb())
    Path(a.work).mkdir(parents=True, exist_ok=True)
    tel.configure(Path(a.work) / "telemetry.jsonl")
    try:
        with tel.span("mb.model_run", collect=True, ir=Path(a.ir_dir).name) as sp:
            res = model_runtime(Path(a.ir_dir), work=Path(a.work),
                                cache=CudaCache(Path(a.cache), target=a.target) if a.cache else None, target=a.target,
                                runs=a.kb_runs, hints=Path(a.hints) if a.hints else None, arch=a.arch, reps=a.reps,
                                time_partitions=a.time_partitions, keep=a.keep, kbeval=kbeval,
                                gpu_model=a.gpu_model, allow_gpu_mismatch=a.allow_gpu_mismatch,
                                torch_baseline=a.torch_baseline, torch_opts=torch_opts(a))
            summ = kbt.compact(tel.full_summary(sp))
        if summ:
            res["telemetry"] = summ
    finally:
        tel.close()
    txt = json.dumps(res, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(txt)
    print(json.dumps({k: res.get(k) for k in ("ir", "passed", "model_e2e_us", "model_basis", "seed_e2e_us", "speedup",
                                               "speedup_paired", "sum_partitions_e2e_us", "boundary_effect_us",
                                               "gpu_name", "gpu_host", "torch_status", "torch_best_us",
                                               "torch_best_variant", "speedup_vs_torch", "speedup_vs_torch_wall")}
                     | {"device": (res.get("device") or {}).get("label"), "match": res["hardware"].get("match")}))


if __name__ == "__main__":
    main()
