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

Model-level verification exports the whole IR as one partition with the
cached winners as slot sources and runs its driver (golden + every op in
isolation). Entries that pass are marked ``model_verified`` for that IR;
a failing slot marks its entry ``rejected`` for that IR.

    python -m modelblaster.pipeline.kb_ingest WINNERS.json [--cache DIR] \\
        [--verify IR_DIR ...] [--arch 75]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

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
            slot_us: float, speedup, impl_id: str, source: dict) -> tuple[dict, str]:
        sha = hashlib.sha1(code.encode()).hexdigest()
        (self.root / "src").mkdir(parents=True, exist_ok=True)
        (self.root / "src" / f"{sha}.cu").write_text(code)
        guard = sorted({_shape_key(s) for s in shapes})
        new = {"op": op, "entry": entry, "signature_sha": signature_sha(signature), "arch": arch,
               "shape_guard": guard, "slot_us": slot_us, "speedup": speedup, "impl_id": impl_id,
               "file": f"src/{sha}.cu", "source": source, "model_verified": [], "rejected": []}
        for i, e in enumerate(self.entries):
            if (e["op"], e["signature_sha"], e["arch"], e["shape_guard"]) == \
                    (new["op"], new["signature_sha"], new["arch"], guard):
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
        sig = signature_sha(signature)
        ok = [e for e in self.entries
              if e["entry"] == entry and e["signature_sha"] == sig and e["arch"] == arch
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
            report.append({"op": r["op"], "impl_id": r["id"], "status": "unknown-op"})
            continue
        if signature_sha(spec.signature) != r["signature_sha"]:
            report.append({"op": r["op"], "impl_id": r["id"], "status": "stale-signature"})
            continue
        sp = r.get("speedup_vs_seed")
        if sp is None or sp < min_speedup:
            report.append({"op": r["op"], "impl_id": r["id"], "status": "below-min-speedup", "speedup": sp})
            continue
        entry, _ = parse_signature(spec.signature)
        e, st = cache.add(op=r["op"], entry=entry, signature=spec.signature, arch=r["arch"],
                          shapes=r["shapes"], code=(base / r["file"]).read_text(), slot_us=r["slot_us"],
                          speedup=sp, impl_id=r["id"],
                          source={"run_id": r.get("run_id"), "problem_id": r.get("problem_id"),
                                  "techniques": r.get("techniques", [])})
        report.append({"op": r["op"], "impl_id": r["id"], "status": st, "speedup": sp})
    cache.save()
    return report


def cache_sources(ir_dir: Path, cache: CudaCache, arch: str, specs: dict) -> tuple[dict, dict, dict]:
    """Slot sources from the cache for an IR: ({op: code}, {op: tag}, {op: entry})."""
    g = json.loads((Path(ir_dir) / "graph.json").read_text())
    shapes: dict[str, list] = {}          # entry -> instance shapes (all op kinds using it)
    sig_of, ops_of = {}, {}
    for o in g["ops"]:
        if o["op"] in specs:
            entry, _ = parse_signature(specs[o["op"]].signature)
            shapes.setdefault(entry, []).append(o.get("shape", {}))
            sig_of[entry] = specs[o["op"]].signature
            ops_of.setdefault(entry, set()).add(o["op"])
    src, tags, used = {}, {}, {}
    for entry, sh in shapes.items():
        e = cache.lookup(entry, sig_of[entry], arch, sh, ir_name=g["name"])
        if e is not None:
            for op in ops_of[entry]:       # export_problem looks sources up by op kind
                src[op], tags[op] = cache.code(e), f"cache:{e['impl_id']}"
            used[entry] = e
    return src, tags, used


def verify_ir(ir_dir: Path, cache: CudaCache, *, arch: str, specs: dict, work: Path | None = None) -> dict:
    """Whole-IR export with cached winners, build and run; mark entries."""
    from .kb_export import build_and_run, export_problem

    ir_dir = Path(ir_dir)
    sm = arch.removeprefix("sm_")
    src, tags, used = cache_sources(ir_dir, cache, f"sm_{sm}", specs)
    if not used:
        return {"ir": ir_dir.name, "status": "no-cache-entries"}
    tmp = Path(work or tempfile.mkdtemp(prefix="mb_verify_", dir=str(MB_ROOT / "tmp")
                                       if (MB_ROOT / "tmp").exists() else None))
    try:
        d = export_problem(ir_dir, tmp / "problem", slot_sources=src, slot_source_tags=tags,
                           specs=specs, strategy="whole")
        res = build_and_run(d, arch=sm)
        man = json.loads((d / "manifest.json").read_text())
        ir_name = man["ir_name"]
        slot_of_entry = {s["entry"]: s["id"] for s in man["slots"]}
        out = {"ir": ir_name, "passed": res.get("passed"), "partition_us": res.get("partition_us"),
               "slots": {}}
        for entry, e in used.items():
            sid = slot_of_entry[entry]
            sc = res.get("slots", {}).get(sid, {})
            good = bool(res.get("passed")) or bool(sc.get("ok"))
            lst = e["model_verified"] if good else e["rejected"]
            if ir_name not in lst:
                lst.append(ir_name)
            out["slots"][sid] = {"impl_id": e["impl_id"], "ok": good, "rel_err": sc.get("rel_err"),
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
    a = ap.parse_args()
    from .reference_kernels import KERNEL_SPECS

    cache = CudaCache(Path(a.cache))
    if a.winners:
        for r in ingest(Path(a.winners), cache, KERNEL_SPECS):
            print(json.dumps(r))
    for d in a.verify:
        print(json.dumps(verify_ir(Path(d), cache, arch=a.arch, specs=KERNEL_SPECS)))


if __name__ == "__main__":
    main()
