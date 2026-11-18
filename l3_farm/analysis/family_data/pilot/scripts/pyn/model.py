"""Whole-model problem (the sweep's final deployed DenseNet121: hinted graph, cache split, the final slot sources)
and its timing with the conv2d_f16 family slots replaced by a given kernel."""
import json, sys, tarfile
from pathlib import Path
sys.path.insert(0, "/tmp/krisd/family/pyn")
import famlib as F

MP = F.W / "pilot" / "model"
FS = F.W / "in" / "final_slots"


def final_sources():
    if not FS.exists():
        tarfile.open(F.W / "in" / "final_slots.tgz").extractall(F.W / "in")
    return {p.stem: p.read_text() for p in FS.glob("*.cu")}


def build():
    from modelblaster.pipeline.kb_ingest import CudaCache
    from modelblaster.pipeline import graph_hints as gh
    from modelblaster.pipeline.kb_export import export_problem
    from modelblaster.pipeline.kb_ingest import cache_plan
    from modelblaster.pipeline.kb_proposals import write_variant_ir, with_fused_specs
    specs, g, hdir = F.specs_and_graph()
    g = gh.unique_names(g); specs = with_fused_specs(g, specs)
    import shutil
    from modelblaster.pipeline.kernel_abi import parse_signature
    c2 = F.W / "cacheA2"
    shutil.rmtree(c2, ignore_errors=True); shutil.copytree(F.W / "cacheA", c2)
    cache = CudaCache(c2)
    fs0 = final_sources()
    for kind in sorted({o["op"] for o in g["ops"] if o["op"].startswith("fused__")}):
        if kind in fs0:                      # the deployed fused kernel, guarded on all its instances
            cache.add(op=kind, entry=parse_signature(specs[kind].signature)[0], signature=specs[kind].signature,
                      arch=f"sm_{F.ARCH}", shapes=[o.get("shape", {}) for o in g["ops"] if o["op"] == kind],
                      code=fs0[kind], slot_us=0.0, speedup=None, impl_id="final_" + kind[-24:], source={"by": "pilot"})
    cache.save()
    cp = cache_plan(g, cache, f"sm_{F.ARCH}", specs)
    ir = write_variant_ir(F.IR, cp["graph"], F.W / "pilot" / "model.ir")
    fs = final_sources()
    src = dict(cp["sources"]); src.update(fs)
    d = export_problem(ir, MP, specs=specs, name="model", strategy="whole", slot_sources=src,
                       slot_source_tags=cp["tags"], sub_sources=cp["sub_sources"])
    m = F.manifest(d)
    ids = {s["id"] for s in m["slots"]}
    print("model slots", len(ids), "final slots", len(fs), "missing in final", sorted(ids - set(fs))[:8],
          "extra in final", sorted(set(fs) - ids)[:8])
    return d


def conv_slots():
    m = F.manifest(MP)
    return [s["id"] for s in m["slots"] if s["op"] == F.OP]


def variant_sources(family_src=None):
    """A's deployed sources; with family_src every conv2d_f16 slot (one per shape in A) runs that kernel."""
    src = final_sources()
    if family_src is not None:
        for sid in conv_slots():
            src[sid] = family_src
    return src


MP1 = F.W / "pilot" / "model1"


def build_single(family_src: str):
    """The same deployed model, but conv2d_f16 as ONE slot over all 61 instances -- how cache_plan deploys a
    family winner (one cache entry whose guard covers every shape)."""
    import shutil
    from modelblaster.pipeline.kb_ingest import CudaCache, cache_plan
    from modelblaster.pipeline import graph_hints as gh
    from modelblaster.pipeline.kb_export import export_problem
    from modelblaster.pipeline.kb_proposals import write_variant_ir, with_fused_specs
    from modelblaster.pipeline.kernel_abi import parse_signature
    specs, g, hdir = F.specs_and_graph()
    g = gh.unique_names(g); specs = with_fused_specs(g, specs)
    c1 = F.W / "cacheF"
    shutil.rmtree(c1, ignore_errors=True); shutil.copytree(F.W / "cacheA2", c1)
    cache = CudaCache(c1)
    cache.entries = [e for e in cache.entries if e["op"] != F.OP]
    cache.add(op=F.OP, entry=parse_signature(specs[F.OP].signature)[0], signature=specs[F.OP].signature,
              arch=f"sm_{F.ARCH}", shapes=[o.get("shape", {}) for o in g["ops"] if o["op"] == F.OP],
              code=family_src, slot_us=0.0, speedup=None, impl_id="family", source={"by": "pilot"})
    cache.save()
    cp = cache_plan(g, cache, f"sm_{F.ARCH}", specs)
    ir = write_variant_ir(F.IR, cp["graph"], F.W / "pilot" / "model1.ir")
    fs = final_sources()
    src = dict(cp["sources"])
    src.update({k: v for k, v in fs.items() if not k.startswith(F.OP)})
    d = export_problem(ir, MP1, specs=specs, name="model1", strategy="whole", slot_sources=src,
                       slot_source_tags=cp["tags"], sub_sources=cp["sub_sources"])
    m = F.manifest(d)
    print("model1 slots", len(m["slots"]), [(s["id"], len(s["instances"])) for s in m["slots"] if s["op"] == F.OP])
    return d
