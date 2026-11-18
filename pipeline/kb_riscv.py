"""KernelBlaster cache for RISC-V backends: codegen lookup and board re-verify.

The cache is `kb_ingest.CudaCache` with ``target="riscv"`` at its own root,
``kernels/riscv/kb_cache`` (index.json + src/<sha>.c), so the CUDA/OpenCL index
is never read or written here. An entry is a winning kernel source for one
ModelBlaster backend (``arch`` = the backend name, e.g. ``ime_k3_x100``), one
op entry point and one KernelSpec signature, with a **shape guard**: the
instance shapes it was verified on.

Codegen (generate_kernels, before the per-model cache and the curated
library): a cached source is used for an op only when its guard covers EVERY
instance shape of that op in the IR, its ``algorithm`` is the curated
algorithm being probed, and it was not rejected for this IR. Otherwise the
curated kernel is used, as before. The swap goes through the same
cross-compile verify as a curated kernel, and its kernel_picks source is
``kb_cache[riscv]``. MB_KB_RISCV=0 skips the cache.

Model-level re-verify is a board run of the model built with the entry (the
harness compares every output with the golden in-binary): ``mark`` reads the
build's kernel_picks.json and the run log and records the IR under the
entry's ``model_verified`` (bit-exact) or ``rejected`` (anything else).

    python -m modelblaster.pipeline.kb_ingest WINNERS.json --target riscv
    python -m modelblaster.pipeline.kb_riscv mark --gen-dir <build>/generated \\
        --log <run_model_k3.sh log> --ir-name dronet
    python -m modelblaster.pipeline.kb_riscv list
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from .kb_ingest import DEFAULT_RISCV_CACHE, CudaCache, _shape_key, signature_sha
from .cuda_dtypes import cuda_signature
from .kernel_abi import parse_signature

SOURCE_LABEL = "kb_cache[riscv]"


def _cache(root: Optional[Path] = None) -> CudaCache:
    return CudaCache(Path(root or os.environ.get("MB_KB_RISCV_CACHE") or DEFAULT_RISCV_CACHE),
                     target="riscv")


def enabled() -> bool:
    return os.environ.get("MB_KB_RISCV", "1") != "0"


def has_entries(backend: str, root: Optional[Path] = None) -> bool:
    """Any riscv entry for `backend`? (False when the cache is absent or off.)"""
    if not enabled():
        return False
    c = _cache(root)
    return any(c.mine(e) and e.get("arch") == backend for e in c.entries)


def codegen_source(backend: str, spec: Any, algorithm: str, shapes: list[dict],
                   ir_name: Optional[str], root: Optional[Path] = None) -> Optional[str]:
    """Path of the cached source for (backend, spec, algorithm) whose guard covers
    every shape in `shapes` and that was not rejected for `ir_name`, else None.
    Among several, the one model-verified on this IR first, then the fastest."""
    if not enabled() or not shapes:
        return None
    c = _cache(root)
    entry, _ = parse_signature(spec.signature)
    sig = signature_sha(cuda_signature(spec.signature))
    need = {_shape_key(s) for s in shapes}
    ok = [e for e in c.entries
          if c.mine(e) and e.get("arch") == backend and e["entry"] == entry
          and e["signature_sha"] == sig and (e.get("source") or {}).get("algorithm") == algorithm
          and need <= set(e["shape_guard"])
          and (ir_name is None or ir_name not in e.get("rejected", []))]
    if not ok:
        return None
    best = min(ok, key=lambda e: (ir_name not in e.get("model_verified", []), e["slot_us"]))
    return str(c.root / best["file"])


def mark(gen_dir: Path, log_text: str, ir_name: str, root: Optional[Path] = None) -> list[dict]:
    """Record a board run's verdict on every cache entry its build used."""
    c = _cache(root)
    d = json.loads((Path(gen_dir) / "kernel_picks.json").read_text())
    picks = d.get("picks", d) if isinstance(d, dict) else {}   # schema_version 1 nests them
    v = re.findall(r"MODELBLASTER_VERIFY === max_abs_err=([0-9.eE+-]+)", log_text)
    passed = bool(v) and all(float(x) == 0.0 for x in v)
    out = []
    for op, p in sorted(picks.items()):
        if not isinstance(p, dict) or p.get("source") != SOURCE_LABEL:
            continue
        f = Path(p.get("path", ""))
        rel = f"src/{f.name}"
        for e in c.entries:
            if c.mine(e) and e["file"] == rel:
                lst = e["model_verified"] if passed else e["rejected"]
                if ir_name not in lst:
                    lst.append(ir_name)
                out.append({"op": op, "file": rel, "ir": ir_name, "passed": passed, "max_abs_err": v})
    c.save()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mark", help="record a board run's verdict on the entries its build used")
    m.add_argument("--gen-dir", required=True)
    m.add_argument("--log", required=True)
    m.add_argument("--ir-name", required=True)
    m.add_argument("--cache", default=None)
    ls = sub.add_parser("list", help="print the riscv entries")
    ls.add_argument("--cache", default=None)
    a = ap.parse_args()
    if a.cmd == "mark":
        r = mark(Path(a.gen_dir), Path(a.log).read_text(errors="replace"), a.ir_name,
                 Path(a.cache) if a.cache else None)
        print(json.dumps(r, indent=1))
        return 0 if r and all(x["passed"] for x in r) else 1
    c = _cache(Path(a.cache) if a.cache else None)
    for e in c.entries:
        if c.mine(e):
            print(json.dumps({k: e.get(k) for k in ("arch", "op", "entry", "file", "slot_us", "speedup",
                                                    "model_verified", "rejected")}
                             | {"algorithm": (e.get("source") or {}).get("algorithm"),
                                "guard_shapes": len(e["shape_guard"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
