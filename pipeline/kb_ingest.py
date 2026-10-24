"""Write KernelBlaster winners back into ModelBlaster's CUDA kernel cache.

KernelBlaster's implementation store exports ``winners.json`` (format
``kb-winners/1``): per (op, signature, arch, shape set) the fastest verified
slot source. Ingest copies them into the cache and re-verifies at model
level before they are used.

Cache layout (``--cache``, default ``kernels/cuda/kb_cache``)::

    index.json        {"format": "mb-cuda-cache/1", "entries": [...]}
    src/<sha>.cu      slot sources (entry function + helpers, MB signature)

An entry carries a **shape guard**: the instance shapes it was verified on.
:func:`lookup` returns an entry for an op only if every instance shape the
caller needs is in the guard (exact match), so a kernel tuned or only
tested at one shape is never silently used at another. Entries are keyed by
the KernelSpec signature hash too: a spec change makes old entries stale.
The hash is of the CUDA spelling of the signature (`_Float16` -> `__half`,
what the slot defines), and every entry carries its storage dtype (fp32 /
fp16): an `_f16` op has its own kind and signature, and lookups also match
the dtype, so fp16 and fp32 kernels never stand in for each other.

Model-level verification exports the whole IR as one partition -- accepted
graph hints applied, the cached winners as slot sources (``cache_plan``,
which also splits a slot whose instances only some winners cover) -- and
runs its driver (golden + every op in isolation). Entries that pass are
marked ``model_verified`` for that IR; a failing slot marks its entry
``rejected`` for that IR.

    python -m modelblaster.pipeline.kb_ingest WINNERS.json [--cache DIR] \\
        [--verify IR_DIR ...] [--hints graph_hints.json] [--arch 75]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from .cuda_dtypes import cuda_signature, op_dtype
from .kernel_abi import parse_signature

MB_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = MB_ROOT / "kernels" / "cuda" / "kb_cache"
FORMAT = "mb-cuda-cache/1"


def signature_sha(sig: str) -> str:
    """Same key KernelBlaster's ImplStore uses."""
    return hashlib.sha1(" ".join(sig.split()).encode()).hexdigest()[:12]


def _shape_key(shape: dict) -> str:
    return json.dumps(shape, sort_keys=True)


class CudaCache:
    def __init__(self, root: Path = DEFAULT_CACHE):
        self.root = Path(root)
        self.index_path = self.root / "index.json"
        self.entries: list[dict] = []
        if self.index_path.exists():
            d = json.loads(self.index_path.read_text())
            assert d.get("format") == FORMAT, f"{self.index_path}: not {FORMAT}"
            self.entries = d["entries"]

    def save(self):
        (self.root / "src").mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps({"format": FORMAT, "entries": self.entries}, indent=1))

    def code(self, e: dict) -> str:
        return (self.root / e["file"]).read_text()

    def add(self, *, op: str, entry: str, signature: str, arch: str, shapes: list, code: str,
            slot_us: float, speedup, impl_id: str, source: dict, dtype: str | None = None) -> tuple[dict, str]:
        sha = hashlib.sha1(code.encode()).hexdigest()
        (self.root / "src").mkdir(parents=True, exist_ok=True)
        (self.root / "src" / f"{sha}.cu").write_text(code)
        guard = sorted({_shape_key(s) for s in shapes})
        new = {"op": op, "entry": entry, "signature_sha": signature_sha(cuda_signature(signature)),
               "dtype": dtype or op_dtype(op), "arch": arch,
               "shape_guard": guard, "slot_us": slot_us, "speedup": speedup, "impl_id": impl_id,
               "file": f"src/{sha}.cu", "source": source, "model_verified": [], "rejected": []}
        for i, e in enumerate(self.entries):
            if (e["op"], e["signature_sha"], e.get("dtype", "fp32"), e["arch"], e["shape_guard"]) == \
                    (new["op"], new["signature_sha"], new["dtype"], new["arch"], guard):
                if slot_us < e["slot_us"]:
                    self.entries[i] = new
                    return new, "replaced"
                return e, "kept-existing"
        self.entries.append(new)
        return new, "added"

    def lookup(self, entry: str, signature: str, arch: str, shapes: list, ir_name: str | None = None):
        """Best entry for a C entry point whose guard covers every shape (and
        that wasn't rejected on ir_name). Keyed by entry, not op kind: op
        kinds sharing an entry (sum_dim / max_dim -> kernel_reduce) share code."""
        need = {_shape_key(s) for s in shapes}
        sig = signature_sha(cuda_signature(signature))
        dtype = op_dtype(entry)                      # kernel_<op>_f16 entries are fp16
        ok = [e for e in self.entries
              if e["entry"] == entry and e["signature_sha"] == sig and e["arch"] == arch
              and e.get("dtype", "fp32") == dtype
              and need <= set(e["shape_guard"]) and (ir_name is None or ir_name not in e["rejected"])]
        return min(ok, key=lambda e: (ir_name not in e["model_verified"], e["slot_us"]), default=None)


def ingest(winners_json: Path, cache: CudaCache, specs: dict, min_speedup: float = 1.05) -> list[dict]:
    w = json.loads(Path(winners_json).read_text())
    assert w.get("format") == "kb-winners/1", "not a KernelBlaster winners file"
    base = Path(winners_json).parent
    report = []
    for r in w["winners"]:
        spec = specs.get(r["op"])
        if spec is None:
            st = "fusion-not-accepted" if r["op"].startswith("fused__") else "unknown-op"
            report.append({"op": r["op"], "impl_id": r["id"], "status": st})
            continue
        if signature_sha(cuda_signature(spec.signature)) != r["signature_sha"]:
            report.append({"op": r["op"], "impl_id": r["id"], "status": "stale-signature"})
            continue
        sp = r.get("speedup_vs_seed")
        if sp is None or sp < min_speedup:
            report.append({"op": r["op"], "impl_id": r["id"], "status": "below-min-speedup", "speedup": sp})
            continue
        entry, _ = parse_signature(spec.signature)
        e, st = cache.add(op=r["op"], entry=entry, signature=spec.signature, arch=r["arch"],
                          shapes=r["shapes"], code=(base / r["file"]).read_text(), slot_us=r["slot_us"],
                          speedup=sp, impl_id=r["id"], dtype=r.get("dtype"),
                          source={"run_id": r.get("run_id"), "problem_id": r.get("problem_id"),
                                  "techniques": r.get("techniques", [])})
        report.append({"op": r["op"], "impl_id": r["id"], "status": st, "speedup": sp})
    cache.save()
    return report


def cache_plan(graph: dict, cache: CudaCache, arch: str, specs: dict, *, split: bool = True) -> dict:
    """Which cached sources an IR can use, per slot.

    A slot (C entry, or a split variant of one) takes the best entry whose
    shape guard covers every instance. When no entry covers them all but some
    cover a part (winners tuned on one partition's shapes), ``split`` makes
    ModelBlaster's own decision to give those instances their own slot: a
    ``split_slot`` hint (source ``mb-cache``) with one group per covering
    entry plus one for the rest (hand kernel), applied to the graph. Fused
    slots without an entry of their own get cached stage kernels guarded on
    the stages' shapes (``sub_sources``) for their seed; a fused slot is also
    split by which first-stage winner covers its instances, so fusing ops
    whose kernels were tuned per shape keeps those kernels.

    -> {"graph", "sources" {slot id: code}, "tags", "used" {slot id: entry},
        "sub_sources" {fused slot id: {stage kind: code}}, "hints" [split hints]}"""
    from . import graph_hints as gh
    from .kb_export import ALIAS_OPS, slot_id_of
    from .kb_proposals import with_fused_specs

    g = gh.unique_names(json.loads(json.dumps(graph)))
    specs = with_fused_specs(g, specs)
    ir_name = g["name"]
    slots: dict[str, list[dict]] = {}
    for o in g["ops"]:
        if o["op"] in specs and o["op"] not in ALIAS_OPS:
            slots.setdefault(slot_id_of(o, specs), []).append(o)

    def entry_sig(kind):
        return parse_signature(specs[kind].signature)[0], specs[kind].signature

    def usable(entry, sig):
        return [e for e in cache.entries if e["entry"] == entry and e["signature_sha"] == signature_sha(sig)
                and e["arch"] == arch and ir_name not in e["rejected"]]

    def own_key(o):
        return _shape_key(o.get("shape", {}))

    def stage0_key(o):
        return _shape_key(o["sub_ops"][0].get("shape", {}))

    src, tags, used, hints = {}, {}, {}, []
    for sid, ops in slots.items():
        entry, sig = entry_sig(ops[0]["op"])
        e = cache.lookup(entry, sig, arch, [o.get("shape", {}) for o in ops], ir_name=ir_name)
        if e is not None:
            src[sid], tags[sid], used[sid] = cache.code(e), f"cache:{e['impl_id']}", e
            continue
        # the same source verified on different shape sets (e.g. one winner taken by
        # several windows of a model) whose guards together cover every instance
        need = {own_key(o) for o in ops}
        by_code: dict[str, list[dict]] = {}
        for c in usable(entry, sig):
            by_code.setdefault(c["file"], []).append(c)
        whole = [cs for cs in by_code.values() if need <= set().union(*(set(c["shape_guard"]) for c in cs))]
        if whole:
            cs = min(whole, key=lambda cs: min(c["slot_us"] for c in cs))
            src[sid], tags[sid], used[sid] = cache.code(cs[0]), "cache:" + "+".join(sorted({c["impl_id"] for c in cs})), cs[0]
            continue
        if not split or any(o.get("slot_variant") for o in ops) or len(ops) < 2:
            continue
        # covering candidates: the slot's own winners; for a fused slot also
        # its first stage's (the dominant kernel), which seed it via sub_sources
        cands = [(c, own_key, True) for c in usable(entry, sig)]
        if "sub_ops" in ops[0]:
            k0 = ops[0]["sub_ops"][0]["op"]
            e0 = cache.lookup(*entry_sig(k0), arch, [o["sub_ops"][0].get("shape", {}) for o in ops], ir_name=ir_name)
            if e0 is None:                                 # else every instance gets it as a stage seed
                cands += [(c, stage0_key, False) for c in usable(*entry_sig(k0))]
        left, parts = list(ops), []
        while left:
            def score(cand):
                c, key, own = cand
                n = sum(1 for o in left if key(o) in c["shape_guard"])
                return (n, own, ir_name in c["model_verified"], -c["slot_us"])
            live = [x for x in cands if score(x)[0]]
            if not live:
                break
            best = max(live, key=score)
            covered = [o for o in left if best[1](o) in best[0]["shape_guard"]]
            parts.append((best, covered))
            left = [o for o in left if o not in covered]
        if not parts or (len(parts) == 1 and not left and not parts[0][0][2]):
            continue                                       # one stage cover for all: plain sub_sources
        groups = [{"ops": [o["name"] for o in cov], "cache": c["impl_id"], "as": "slot" if own else "stage"}
                  for (c, _, own), cov in parts]
        if left:
            groups.append({"ops": [o["name"] for o in left], "cache": None})
        hints.append({"id": f"split:{sid}", "kind": "split_slot", "target": "cuda", "slot": sid,
                      "by": "cache", "groups": groups, "status": "accepted",
                      "source": [{"by": "mb-cache", "ir": ir_name}], "evidence": []})
        for k, ((c, _, own), _) in enumerate(parts):
            if own:
                src[f"{sid}__v{k}"], tags[f"{sid}__v{k}"], used[f"{sid}__v{k}"] = \
                    cache.code(c), f"cache:{c['impl_id']}", c
    if hints:
        g, _, _ = gh.apply_hints(g, hints, specs)
    sub_sources: dict[str, dict[str, str]] = {}
    for o in g["ops"]:
        sid = slot_id_of(o, specs) if o["op"] in specs else None
        if "sub_ops" not in o or sid in src or sid in sub_sources:
            continue
        mine = [x for x in g["ops"] if x["op"] in specs and slot_id_of(x, specs) == sid]
        subs = {}
        for i, st in enumerate(o["sub_ops"]):
            entry, sig = entry_sig(st["op"])
            shapes = [x["sub_ops"][i].get("shape", {}) for x in mine]
            e = cache.lookup(entry, sig, arch, shapes, ir_name=ir_name)
            if e is not None:
                subs[st["op"]] = cache.code(e)
                used[f"{sid}/{st['op']}"] = e
        sub_sources[sid] = subs
    return {"graph": g, "sources": src, "tags": tags, "used": used, "sub_sources": sub_sources, "hints": hints}


def verify_ir(ir_dir: Path, cache: CudaCache, *, arch: str, specs: dict, work: Path | None = None,
              hints: Path | None = None, fusion_registry: Path | None = None) -> dict:
    """Whole-IR export with the accepted graph hints applied and the cached
    winners as sources (``cache_plan``), build and run; mark entries."""
    from . import graph_hints as gh
    from .kb_export import build_and_run, export_problem
    from .kb_proposals import write_variant_ir

    ir_dir = Path(ir_dir)
    sm = arch.removeprefix("sm_")
    g = json.loads((ir_dir / "graph.json").read_text())
    specs = dict(specs)
    hpath = hints or fusion_registry
    applied, hint_ids = [], []
    if hpath is not None:
        g, added, prov = gh.apply_hints(g, gh.load(Path(hpath))["hints"], specs)
        specs.update(added)
        applied = [a["op"] for a in prov["applied"] if a["kind"] == "fuse"]
        hint_ids = sorted({a["id"] for a in prov["applied"]})
    plan = cache_plan(g, cache, f"sm_{sm}", specs)
    if not plan["used"]:
        return {"ir": ir_dir.name, "status": "no-cache-entries"}
    tmp = Path(work or tempfile.mkdtemp(prefix="mb_verify_", dir=str(MB_ROOT / "tmp")
                                       if (MB_ROOT / "tmp").exists() else None))
    try:
        vir = write_variant_ir(ir_dir, plan["graph"], tmp / "ir")
        d = export_problem(vir, tmp / "problem", slot_sources=plan["sources"], slot_source_tags=plan["tags"],
                           sub_sources=plan["sub_sources"], specs=specs, strategy="whole")
        res = build_and_run(d, arch=sm, graph=True)
        man = json.loads((d / "manifest.json").read_text())
        ir_name = man["ir_name"]
        out = {"ir": ir_name, "passed": res.get("passed"), "partition_us": res.get("partition_us"),
               "e2e_us": res.get("e2e_us"), "e2e_basis": res.get("e2e_basis"),
               "fused": applied, "hints": hint_ids, "cache_split": [h["id"] for h in plan["hints"]],
               "slots": {}}
        for key, e in plan["used"].items():
            sid = key.split("/")[0]
            sc = res.get("slots", {}).get(sid, {})
            good = bool(res.get("passed")) or bool(sc.get("ok"))
            lst = e["model_verified"] if good else e["rejected"]
            if ir_name not in lst:
                lst.append(ir_name)
            out["slots"][key] = {"impl_id": e["impl_id"], "ok": good, "rel_err": sc.get("rel_err"),
                                 "slot_us": res.get("slot_us", {}).get(sid)}
        if not res.get("passed"):
            out["log"] = res.get("log", "")[-1500:]
        cache.save()
        return out
    finally:
        if work is None:
            shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("winners", nargs="?", help="KernelBlaster winners.json")
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--verify", nargs="*", default=[], help="IR dirs to verify at model level")
    ap.add_argument("--arch", default="75")
    ap.add_argument("--hints", "--fusion-registry", dest="hints", default=None,
                    help="graph hints file: accepted fusions are applied and known ops "
                         "(default kernels/cuda/graph_hints.json)")
    a = ap.parse_args()
    from .kb_proposals import DEFAULT_REGISTRY, accepted_fused_specs
    from .reference_kernels import KERNEL_SPECS

    specs = {**KERNEL_SPECS, **accepted_fused_specs(Path(a.hints or DEFAULT_REGISTRY), KERNEL_SPECS)}
    cache = CudaCache(Path(a.cache))
    if a.winners:
        for r in ingest(Path(a.winners), cache, specs):
            print(json.dumps(r))
    for d in a.verify:
        print(json.dumps(verify_ir(Path(d), cache, arch=a.arch, specs=specs,
                                   hints=Path(a.hints or DEFAULT_REGISTRY))))


if __name__ == "__main__":
    main()
