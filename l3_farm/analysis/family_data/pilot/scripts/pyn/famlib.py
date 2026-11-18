"""Pilot helpers on the node (shape-family pilot, DenseNet121 conv2d_f16). Uses the prototype tree
(/tmp/krisd/family/ModelBlaster as `modelblaster`)."""
import json, os, shutil
from pathlib import Path

W = Path("/tmp/krisd/family")
IR = W / "irs" / "15_DenseNet121"
SWEEP = W / "in" / "sweep"
HINTS = SWEEP / "graph_hints.json"
ARCH = "89"
OP = "conv2d_f16"
KB = Path(os.environ.get("KB", str(W / "ModelBlaster/third_party/KernelBlaster")))


def specs_and_graph():
    """(specs incl. the accepted fused specs, the IR graph with the accepted hints applied, the hinted IR dir)."""
    from modelblaster.pipeline import graph_hints as gh
    from modelblaster.pipeline.kb_plan import hinted_ir
    from modelblaster.pipeline.kb_proposals import _all_specs
    doc = gh.load(HINTS)
    d, specs, g = hinted_ir(IR, doc, W / "work_hinted", dict(_all_specs()))
    return specs, g, d


def kbeval():
    from modelblaster.pipeline.kb_eval import KBEval
    return KBEval(evaluator="local", kb=KB, python=os.environ.get("VIRTUAL_ENV", "/tmp/krisd/venv") + "/bin/python",
                  gpus="0", runtime="auto")


def export_family(out: Path, cache=None, *, op=OP, split_each=False, sources_override=None):
    """The family partition: every `op` instance of the hinted model graph, one problem.
    cache: seeds from the cache (cache_plan: split by shape guard), None: hand seed, one slot.
    split_each: one slot per instance (per-shape checks and attribution)."""
    from modelblaster.pipeline import graph_hints as gh
    from modelblaster.pipeline.kb_export import export_problem
    from modelblaster.pipeline.kb_ingest import cache_plan
    from modelblaster.pipeline.kb_proposals import write_variant_ir, with_fused_specs
    specs, g, hdir = specs_and_graph()
    g = gh.unique_names(g)
    specs = with_fused_specs(g, specs)
    src = tags = subs = None
    ir_dir = hdir
    if cache is not None:
        cp = cache_plan(g, cache, f"sm_{ARCH}", specs)
        g, src, tags, subs = cp["graph"], cp["sources"], cp["tags"], cp["sub_sources"]
        ir_dir = write_variant_ir(IR, g, Path(out).parent / (Path(out).name + ".ir"))
    elif split_each:
        ops = [o for o in g["ops"] if o["op"] == op]
        hint = {"id": f"split:{op}:each", "kind": "split_slot", "target": "cuda", "slot": op, "by": "pilot",
                "groups": [{"ops": [o["name"]], "cache": None} for o in ops], "status": "accepted",
                "source": [{"by": "pilot"}], "evidence": []}
        g, _, _ = gh.apply_hints(g, [hint], specs)
        ir_dir = write_variant_ir(IR, g, Path(out).parent / (Path(out).name + ".ir"))
    idx = [i for i, o in enumerate(g["ops"]) if o["op"] == op]
    if Path(out).exists():
        shutil.rmtree(out)
    return export_problem(ir_dir, out, op_indices=idx, specs=specs, name=Path(out).name, strategy="plan",
                          slot_sources=sources_override or src, slot_source_tags=tags, sub_sources=subs)


def time_problem(prob: Path, *, sources=None, reps=3, name="t", work=None):
    from modelblaster.pipeline.kb_eval import run_problem
    work = Path(work or (Path(prob).parent / f"{Path(prob).name}_time"))
    return run_problem(kbeval(), Path(prob), arch=ARCH, work=work, reps=reps, sources=sources, name=name)


def manifest(prob):
    return json.loads((Path(prob) / "manifest.json").read_text())


def family_summary(res: dict, prob) -> dict:
    """Per-instance attributed times of a family problem run (op name -> us) and their sum."""
    m = manifest(prob)
    ops = [o["name"] for o in m["ops"]]
    att = res.get("attr_op_us") or {}
    per = {o: att.get(o) for o in ops}
    return {"passed": res.get("passed"), "family_kernel_us": round(sum(v for v in per.values() if v), 3),
            "e2e_us": res.get("e2e_us"), "e2e_basis": res.get("e2e_basis"), "partition_us": res.get("partition_us"),
            "per_op_us": per, "slots_failed": [s for s, c in (res.get("slots") or {}).items() if not c.get("ok")],
            "device": (res.get("device") or {}).get("label")}
