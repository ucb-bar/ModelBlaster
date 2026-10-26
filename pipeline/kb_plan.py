"""Cost-ranked partition selection: what ModelBlaster sends to KernelBlaster.

A KernelBlaster budget is best spent where the time is. For one or more IRs:

1. **profile** -- apply the accepted graph hints, take the kernel cache's
   current winners (``kb_ingest.cache_plan``), export the whole graph as one
   problem and run its driver (median of ``runs``): per-op times
   (``MB_OP_US``) with each op's C entry, signature and shape;
2. **rank** -- group op instances by (C entry, signature, shape) across every
   profiled IR: the same layer shape in two models is one item, so it is
   optimised once and its winner (shape-guarded) serves both;
3. **select** -- items in order of total time while they cover less than
   ``coverage`` of the total, at most ``top_k``, each at least ``min_share``,
   skipping ``exclude`` (items already sent in an earlier round);
4. **problems** -- each selected item goes to the IR where it costs most;
   per IR one problem with every instance of its selected items (a partition
   need not be contiguous: inputs come from the reference walk). A winner
   tuned on some of a slot's shapes is used for exactly those instances later
   (``cache_plan`` splits the slot by the cache's shape guards).

The plan (``modelblaster.kb_plan/v1``) records every item, its share and
whether it was selected and why not; :func:`export_plan` (also
``kb_export --plan``) writes its problems.

    python -m modelblaster.pipeline.kb_plan profile IR_DIR [...] --out PROFILE_DIR \\
        [--hints graph_hints.json --cache DIR --arch 75 --runs 3] \\
        [--ssh-host H --remote-dir D [--cuda-visible-devices ids] [--ssh-key K] | --evaluator local] \\
        [--gpu-model M]
    python -m modelblaster.pipeline.kb_plan select PROFILE.json [...] --out PLAN.json \\
        [--coverage 0.8 --top-k 8 --min-share 0.01 --exclude-from OLD_PLAN.json]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from .kernel_abi import parse_signature

FORMAT = "modelblaster.kb_plan/v1"
PROFILE_FORMAT = "mb-kb-profile/1"


def _shape_key(shape: dict) -> str:
    return json.dumps(shape, sort_keys=True)


def item_key(entry: str, sig_sha: str, shape: dict) -> str:
    return f"{entry}|{sig_sha}|{_shape_key(shape)}"


def hinted_ir(ir_dir: Path, hints_doc: dict, work: Path, specs: dict):
    """(IR dir with the accepted hints applied, specs, cache-plan input graph)."""
    from . import graph_hints as gh

    d, specs, _ = gh.apply_to_ir(Path(ir_dir), hints_doc, Path(work) / "ir")
    return d, specs, json.loads((d / "graph.json").read_text())


# profiles redone after rejecting the cached winners that failed in the whole model
PROFILE_FALLBACKS = 3


def failed_cached_slots(res: dict, plan: dict) -> dict:
    """The cache entries (``plan["used"]`` key -> entry) of the slots a failed run blames: a slot check
    with ok=0 (a launch / API error of one of its ops, or a wrong result). {} when the run passed or
    names no slot (a fault ends the run before the slot checks) -- nothing to fall back from."""
    if res.get("passed"):
        return {}
    bad = {sid for sid, sc in (res.get("slots") or {}).items() if not sc.get("ok")}
    return {k: e for k, e in (plan.get("used") or {}).items() if k.split("/")[0] in bad}


def profile_ir(ir_dir: Path, work: Path, *, hints_doc: dict | None = None, cache=None, arch: str = "75",
               specs: dict | None = None, runs: int = 3, keep: bool = False, kbeval=None,
               gpu_model: str | None = None, allow_gpu_mismatch: bool = False) -> dict:
    """Whole-graph per-op profile of an IR in its current state (accepted
    hints applied, cached winners in place).

    ``kbeval`` (a :class:`kb_eval.KBEval`): the ``runs`` go through
    KernelBlaster's evaluator -- this host's GPU pool or the ``--ssh-host``
    GPUs -- on one GPU of ``gpu_model`` (none there: ``HardwareMismatch``
    unless ``allow_gpu_mismatch``); None: built and run here on the default
    GPU (``build_and_run``). Either way the run with the median partition
    time is used and ``device`` records the GPU it ran on."""
    from . import graph_hints as gh
    from . import kb_telemetry as kbt
    from . import kb_trace
    from .kb_eval import builtin_device, run_problem
    from .kb_export import build_and_run, export_problem, slot_id_of
    from .kb_ingest import cache_plan, signature_sha
    from .kb_proposals import _all_specs, write_variant_ir

    work = Path(work)
    specs = dict(specs or _all_specs())
    hdir, specs, g = hinted_ir(Path(ir_dir), hints_doc or gh.new_doc(), work, specs)
    fallback = []
    for attempt in range(PROFILE_FALLBACKS + 1):
        plan = cache_plan(g, cache, f"sm_{arch}", specs) if cache is not None else \
            {"graph": gh.unique_names(g), "sources": {}, "tags": {}, "sub_sources": None, "hints": [], "used": {}}
        pdir = write_variant_ir(Path(ir_dir), plan["graph"], work / "profile_ir") if plan["hints"] else hdir
        prob = export_problem(pdir, work / "problem", slot_sources=plan["sources"], slot_source_tags=plan["tags"],
                              sub_sources=plan["sub_sources"], specs=specs, strategy="whole")
        # as deployed: CUDA graph where capturable; per-op GPU time from the driver's attribution
        with kbt.get().span("mb.profile.run", ir=Path(ir_dir).name, runs=runs,
                            evaluator=kbeval.evaluator if kbeval is not None else "builtin") as sp, \
                kb_trace.get().scope(mb_stage="profile", ir=Path(ir_dir).name):
            if kbeval is not None:
                res = run_problem(kbeval, prob, arch=arch, work=work / "kb_time", reps=runs, gpu_model=gpu_model,
                                  allow_mismatch=allow_gpu_mismatch, name="profile")
            else:
                res = build_and_run(prob, arch=arch, runs=runs, graph=True)
                res["device"] = builtin_device(arch)
            dev = res.get("device") or {}
            if sp is not None and hasattr(sp, "set"):
                sp.set(device=dev.get("label"), gpu=dev.get("name"), host=dev.get("host"), passed=res.get("passed"))
        man = json.loads((prob / "manifest.json").read_text())
        bad = failed_cached_slots(res, plan)
        if res.get("passed") or not bad or attempt == PROFILE_FALLBACKS:
            break
        # cached winners that fail in the whole model (each was validated in its own problem): rejected
        # for this IR, as the verification would, and the profile redone on the next best source (the
        # seed when none) -- one slot's bad kernel must not end a model's run
        for key, e in bad.items():
            if man["ir_name"] not in e["rejected"]:
                e["rejected"].append(man["ir_name"])
        cache.save()
        step = {"attempt": attempt, "rejected": {k: e["impl_id"] for k, e in bad.items()},
                "errors": (res.get("errors") or [])[:20]}
        fallback.append(step)
        kb_trace.record("profile_fallback", ir=man["ir_name"], mb_stage="profile", **step)
        print(json.dumps({"profile_fallback": man["ir_name"], **step}), flush=True)
        for p in (work / "problem", work / "profile_ir", work / "kb_time"):
            shutil.rmtree(p, ignore_errors=True)
    attr = res.get("attr_op_us") or {}
    ops_by_name = {o["name"]: o for o in plan["graph"]["ops"]}
    seed = {s["id"]: s["seed_source"] for s in man["slots"]}
    recs = []
    for r in man["ops"]:
        o = ops_by_name[r["name"]]
        entry, _ = parse_signature(specs[o["op"]].signature)
        recs.append({"name": r["name"], "op": o["op"], "slot": r["slot"], "entry": entry,
                     "signature_sha": signature_sha(specs[o["op"]].signature), "shape": o.get("shape", {}),
                     "us": attr.get(r["name"], res.get("op_us", {}).get(r["name"])),
                     "event_us": res.get("op_us", {}).get(r["name"]), "seed": seed.get(r["slot"]),
                     "slot_base": slot_id_of({"op": o["op"]}, specs)})
    out = {"format": PROFILE_FORMAT, "ir": man["ir_name"], "source_ir": str(ir_dir), "ir_dir": str(pdir),
           "hints_sha": gh.hints_sha(hints_doc or gh.new_doc()), "arch": f"sm_{arch}",
           "passed": res.get("passed"), "partition_us": res.get("partition_us"),
           "partition_us_runs": res.get("partition_us_runs"), "ops": recs,
           "e2e_us": res.get("e2e_us"), "e2e_basis": res.get("e2e_basis"),
           "op_time_basis": "cupti" if attr else "events",
           "launches": res.get("attr_launches"), "launch_overhead_frac": res.get("launch_overhead_frac"),
           # deployed time not spent in kernels, per launch: what a fusion saves besides the kernel
           "launch_us": _launch_us(res),
           "applied_hints": [h["id"] for h in plan["hints"]],
           # cached winners dropped because they failed in this whole model (see failed_cached_slots)
           **({"cache_fallback": fallback} if fallback else {}),
           # where it ran: the GPU (host, label, uuid, name, arch) and the evaluator
           "device": res.get("device"),
           "evaluator": res.get("evaluator") or {"evaluator": "builtin", "runtime": "process"}}
    if res.get("hw_warnings"):
        out["hw_warnings"] = res["hw_warnings"]
    if not res.get("passed"):
        out["log"] = res.get("log", "")[-1500:]
    if not keep:                                   # data.bin and the build are the big parts
        for p in (prob / "data.bin", prob / "build", work / "kb_time" / "profile_kbwork"):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink()
    (work / "profile.json").write_text(json.dumps(out, indent=1))
    kb_trace.record("profile", ir=out["ir"], mb_stage="profile", work=str(work), passed=out["passed"],
                    partition_us=out["partition_us"], e2e_us=out["e2e_us"], device=out["device"],
                    files=kb_trace.files({"profile.json": work / "profile.json",
                                          "manifest.json": prob / "manifest.json"}),
                    log=kb_trace.get().put(res.get("log")) if res.get("log") else None)
    return out


def _launch_us(res: dict) -> float:
    e2e, kern, n = res.get("e2e_us"), res.get("attr_sum_us"), res.get("attr_launches")
    return round(max(0.0, e2e - kern) / n, 3) if e2e and kern is not None and n else 0.0


def rank(profiles: list[dict], weights: dict[str, float] | None = None) -> list[dict]:
    """Op instances grouped by (entry, signature, shape) across IRs, by total
    time (an IR's times scaled by ``weights[ir]``, default 1)."""
    items: dict[str, dict] = {}
    for p in profiles:
        w = (weights or {}).get(p["ir"], 1.0)
        for r in p["ops"]:
            if r["us"] is None:
                continue
            k = item_key(r["entry"], r["signature_sha"], r["shape"])
            it = items.setdefault(k, {"key": k, "entry": r["entry"], "op": r["op"], "signature_sha": r["signature_sha"],
                                      "shape": r["shape"], "total_us": 0.0, "count": 0, "occurrences": {}})
            it["total_us"] += w * r["us"]
            it["count"] += 1
            occ = it["occurrences"].setdefault(p["ir"], {"ir": p["ir"], "ir_dir": p["ir_dir"], "ops": [], "us": 0.0,
                                                         "seed": r["seed"]})
            occ["ops"].append(r["name"])
            occ["us"] += w * r["us"]
    out = sorted(items.values(), key=lambda it: -it["total_us"])
    for it in out:
        it["occurrences"] = sorted(it["occurrences"].values(), key=lambda o: -o["us"])
    return out


def select(items: list[dict], *, coverage: float = 0.8, top_k: int | None = None, min_share: float = 0.01,
           exclude: set | None = None) -> dict:
    """Mark items selected (see module doc) -> plan dict without problems."""
    total = sum(it["total_us"] for it in items) or 1.0
    exclude = exclude or set()
    cum, n, chosen = 0.0, 0, 0.0
    for it in items:
        it["share"] = it["total_us"] / total
        why = None
        if it["key"] in exclude:
            why = "sent-before"
        elif cum >= coverage:
            why = "coverage-reached"
        elif top_k is not None and n >= top_k:
            why = "top-k"
        elif it["share"] < min_share:
            why = "below-min-share"
        it["selected"], it["why_not"] = why is None, why
        if why is None:
            n += 1
            chosen += it["total_us"]
        if why != "sent-before":
            cum += it["share"]
    return {"format": FORMAT, "budget": {"coverage": coverage, "top_k": top_k, "min_share": min_share},
            "total_us": total, "selected_us": chosen, "selected_share": chosen / total, "items": items}


def problems(plan: dict, per_item: bool = False) -> dict:
    """Group the selected items into one problem per representative IR, or (``per_item``) one problem per
    selected item (op kind x shape): each kernel gets its own KernelBlaster run, so the runs go in parallel and
    their winners carry one shape guard each (a model's linears of different shapes are all tuned in one round)."""
    by_ir: dict[str, dict] = {}
    for it in plan["items"]:
        if not it.get("selected"):
            continue
        rep = it["occurrences"][0]
        it["rep_ir"] = rep["ir"]
        if per_item:
            pid = re.sub(r"[^A-Za-z0-9_]+", "_", f"{rep['ir']}__{it['op']}_{len(by_ir)}")
            by_ir[pid] = {"id": pid, "ir": rep["ir"], "ir_dir": rep["ir_dir"], "op_names": list(rep["ops"]),
                          "keys": [it["key"]], "est_us": rep["us"]}
            continue
        p = by_ir.setdefault(rep["ir"], {"ir": rep["ir"], "ir_dir": rep["ir_dir"], "op_names": [], "keys": [],
                                         "est_us": 0.0})
        p["op_names"] += [n for n in rep["ops"] if n not in p["op_names"]]
        p["keys"].append(it["key"])
        p["est_us"] += rep["us"]
    plan["problems"] = sorted(by_ir.values(), key=lambda p: -p["est_us"])
    return plan


def make_plan(profiles: list[dict], **kw) -> dict:
    from . import kb_trace
    excl = len(kw.get("exclude") or ())
    opts = {k: v for k, v in kw.items() if k != "exclude"}
    per_item = kw.pop("per_item", False)
    plan = problems(select(rank(profiles, kw.pop("weights", None)), **kw), per_item=per_item)
    plan["profiles"] = [{"ir": p["ir"], "ir_dir": p["ir_dir"], "partition_us": p["partition_us"]} for p in profiles]
    # the plan's inputs (every profile's per-op times) and its output
    kb_trace.record("plan", mb_stage="plan", options=opts, excluded=excl, selected_share=plan.get("selected_share"),
                    profiles=kb_trace.put_json(profiles), plan=kb_trace.put_json(plan),
                    problems=[(p["ir"], len(p["op_names"])) for p in plan["problems"]])
    return plan


def plan_op_indices(graph: dict, op_names: list[str], pid: str = "") -> list[int]:
    """Indices of a plan problem's ops in ``graph`` (op names already unique). Every name must be there: a
    problem with a missing op would be tuned (and its winner ingested) on the wrong partition or none."""
    want = set(op_names)
    idx = [i for i, o in enumerate(graph["ops"]) if o["name"] in want]
    missing = want - {graph["ops"][i]["name"] for i in idx}
    if missing or not idx:
        raise ValueError(f"plan problem {pid}: ops {sorted(missing) or sorted(want)} not in the IR graph "
                         f"(op names: {[o['name'] for o in graph['ops']][:20]})")
    return idx


def export_plan(plan: dict, out: Path, *, cache=None, arch: str = "75", specs: dict | None = None,
                names: dict | None = None) -> list[Path]:
    """Export each problem of a plan under ``out/<ir>``, seeded from the
    cache's winners for those ops' shapes (``cache_plan`` on the problem's
    IR; the plan's IR dirs already carry the accepted hints)."""
    from .kb_export import export_problem
    from .kb_ingest import cache_plan
    from .kb_proposals import _all_specs, with_fused_specs, write_variant_ir

    from .graph_hints import unique_names

    specs = dict(specs or _all_specs())
    dirs = []
    for p in plan["problems"]:
        ir_dir = Path(p["ir_dir"])
        # the plan's op names are the profile's: unique (`einsum_1#4` for the 5th op named einsum_1), as
        # export_problem numbers them; matched against graph.json's raw names, every repeat of a name missed
        # (an empty partition was sent to KernelBlaster) and the first instance pulled in all of its repeats
        g = unique_names(json.loads((ir_dir / "graph.json").read_text()))
        specs = with_fused_specs(g, specs)
        cp = cache_plan(g, cache, f"sm_{arch}", specs) if cache is not None else None
        if cp is not None and cp["hints"]:
            ir_dir = write_variant_ir(ir_dir, cp["graph"], Path(out) / f"{p['ir']}.ir")
            g = cp["graph"]
        idx = plan_op_indices(g, p["op_names"], p.get("id", p["ir"]))
        pid = p.get("id", p["ir"])                                 # per-item plans: several problems per IR
        name = (names or {}).get(pid, f"{pid}__plan")
        dirs.append(export_problem(ir_dir, Path(out) / pid, op_indices=idx, specs=specs, name=name,
                                   strategy="plan", slot_sources=cp["sources"] if cp else None,
                                   slot_source_tags=cp["tags"] if cp else None,
                                   sub_sources=cp["sub_sources"] if cp else None))
    from . import kb_trace
    kb_trace.record("export", mb_stage="export", out=str(out), problems=[str(d) for d in dirs],
                    manifests={Path(d).name: kb_trace.files({"m": Path(d) / "manifest.json"}).get("m") for d in dirs})
    return dirs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("profile")
    pr.add_argument("irs", nargs="+")
    pr.add_argument("--out", required=True)
    pr.add_argument("--hints")
    pr.add_argument("--cache")
    pr.add_argument("--arch", default="75")
    pr.add_argument("--runs", type=int, default=3)
    pr.add_argument("--gpu-model", help="the GPU model the runs must use (with a KernelBlaster evaluator)")
    pr.add_argument("--allow-gpu-mismatch", action="store_true")
    from . import kb_eval as ke
    ke.add_args(pr, prefix_help="where the profiles run", local_default="builtin")
    se = sub.add_parser("select")
    se.add_argument("profiles", nargs="+")
    se.add_argument("--out", required=True)
    se.add_argument("--coverage", type=float, default=0.8)
    se.add_argument("--top-k", type=int, default=None)
    se.add_argument("--min-share", type=float, default=0.01)
    se.add_argument("--exclude-from", nargs="*", default=[], help="earlier plans: their selected items are skipped")
    a = ap.parse_args()
    if a.cmd == "profile":
        from . import graph_hints as gh
        from .kb_ingest import CudaCache

        doc = gh.load(Path(a.hints)) if a.hints else None
        cache = CudaCache(Path(a.cache)) if a.cache else None
        kbeval = ke.from_args(a, local_default="builtin")
        for d in a.irs:
            p = profile_ir(Path(d), Path(a.out) / Path(d).name, hints_doc=doc, cache=cache, arch=a.arch, runs=a.runs,
                           kbeval=kbeval, gpu_model=a.gpu_model, allow_gpu_mismatch=a.allow_gpu_mismatch)
            top = sorted(p["ops"], key=lambda r: -(r["us"] or 0))[:5]
            print(json.dumps({"ir": p["ir"], "passed": p["passed"], "partition_us": p["partition_us"],
                              "top": [(r["name"], r["us"]) for r in top],
                              "device": (p.get("device") or {}).get("label"),
                              "gpu": (p.get("device") or {}).get("name")}))
    else:
        excl = {it["key"] for f in a.exclude_from for it in json.loads(Path(f).read_text())["items"]
                if it.get("selected")}
        plan = make_plan([json.loads(Path(f).read_text()) for f in a.profiles], coverage=a.coverage,
                         top_k=a.top_k, min_share=a.min_share, exclude=excl)
        Path(a.out).write_text(json.dumps(plan, indent=1))
        print(json.dumps({"selected_share": round(plan["selected_share"], 3),
                          "problems": [(p["ir"], len(p["op_names"])) for p in plan["problems"]]}))


if __name__ == "__main__":
    main()
