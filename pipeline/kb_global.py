"""Global knowledge for ``kb_graph_opt`` (KernelBlaster ``knowledge/v2/merge.py``; docs in
``l3_farm/analysis/global_knowledge.md``).

``--knowledge-global off`` (default): nothing changes; the work dir's ``knowledge/`` starts from
KernelBlaster's repo seed, as before.

``--knowledge-global read``: a fresh work dir's ``knowledge/`` starts from a seed materialized
from a global snapshot -- ``--knowledge-global-seed DIR`` (a seed dir: ``knowledge.db``,
``impl/``, ``seed.json``, as ``merge seed`` / the farm writes it), or, without one, one
materialized here from ``--knowledge-global-store`` (its current snapshot). The seed is applied
once, when ``knowledge/knowledge.db`` does not exist yet (a resumed work dir keeps its own); what
was applied is recorded in ``state.json`` / ``summary.json`` ``knowledge_global`` and joins the
round key, so the rounds of a seeded run are not mixed with an unseeded one.

``--knowledge-global readwrite``: as ``read``, and at the end the run is merged back into
``--knowledge-global-store`` (a delta against its seed; idempotent by run uid + fingerprint).
For the farm, where the store is on the controller and not on the node, the node runs ``read``
and the controller merges (``l3_farm/farm.py``).

The partition defaults to ``cuda:sm_<arch>|<dtype of the IRs>`` (``--knowledge-global-part``).
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

MODES = ("off", "read", "readwrite")


def ir_dtype(irs) -> str:
    """fp16 when any IR tensor is half, int8 when any is 8-bit integer, else fp32."""
    seen = set()
    for d in irs:
        g = Path(d) / "graph.json"
        if not g.exists():
            continue
        try:
            for t in json.loads(g.read_text()).get("tensors", {}).values():
                seen.add(str(t.get("dtype")))
        except (ValueError, AttributeError):
            continue
    if seen & {"f16", "fp16", "float16"}:
        return "fp16"
    if seen & {"i8", "s8", "int8", "u8"}:
        return "int8"
    return "fp32"


def part_for(cfg, irs) -> str:
    if getattr(cfg, "knowledge_global_part", ""):
        return cfg.knowledge_global_part
    a = str(cfg.arch)
    return f"cuda:sm_{a}|{ir_dtype(irs)}" if a.isdigit() else f"cuda:{a}|{ir_dtype(irs)}"


def _kb_env(cfg) -> dict:
    env = dict(os.environ)
    src = str(Path(cfg.kb) / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _merge_cli(cfg, args: list, timeout: float = 1800) -> dict:
    cmd = [cfg.python, "-m", "kernelblaster.knowledge.v2.merge"] + args
    p = subprocess.run(cmd, capture_output=True, text=True, env=_kb_env(cfg), cwd=str(cfg.kb), timeout=timeout)
    try:
        out = json.loads(p.stdout)
    except ValueError:
        out = {"error": f"exit {p.returncode}: {(p.stderr or p.stdout)[-800:]}"}
    if p.returncode and "error" not in out:
        out["error"] = f"exit {p.returncode}: {p.stderr[-800:]}"
    return out


def seed_info(db: Path) -> dict:
    try:
        c = sqlite3.connect(f"file:{Path(db).resolve()}?mode=ro", uri=True)
        try:
            r = c.execute("SELECT value FROM meta WHERE key='gk_seed'").fetchone()
        finally:
            c.close()
        return json.loads(r[0]) if r else {}
    except sqlite3.Error:
        return {}


def apply_seed(work: Path, cfg, irs, state: dict) -> dict | None:
    """Seed ``work/knowledge`` once (read / readwrite). -> the record kept in state.json, or None (off)."""
    mode = getattr(cfg, "knowledge_global", "off") or "off"
    if mode == "off":
        return None
    if mode not in MODES:
        raise SystemExit(f"--knowledge-global {mode!r}: expected one of {MODES}")
    if state.get("knowledge_global"):
        return state["knowledge_global"]                  # applied when this work dir started
    kdir = Path(work) / "knowledge"
    rec = {"mode": mode, "part": part_for(cfg, irs), "problem": cfg.knowledge_global_problem or Path(work).name}
    if (kdir / "knowledge.db").exists():
        rec.update(applied=False, snapshot=None,
                   reason="knowledge/knowledge.db already exists (a work dir from before); kept as is")
        state["knowledge_global"] = rec
        return rec
    seed = cfg.knowledge_global_seed
    if not seed:
        if not cfg.knowledge_global_store:
            raise SystemExit("--knowledge-global read|readwrite needs --knowledge-global-seed or --knowledge-global-store")
        seed = str(Path(work) / "gk_seed")
        if Path(seed).exists():
            shutil.rmtree(seed)
        res = _merge_cli(cfg, ["--store", cfg.knowledge_global_store, "seed", "--part", rec["part"], "--out", seed,
                               "--problem", rec["problem"]])
        if "error" in res:
            raise SystemExit(f"knowledge-global seed failed: {res['error']}")
    sp = Path(seed)
    db = sp if sp.suffix == ".db" else sp / "knowledge.db"
    if not db.exists():
        raise SystemExit(f"--knowledge-global-seed {seed}: no knowledge.db")
    kdir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(db, kdir / "knowledge.db")
    impl = (sp.parent if sp.suffix == ".db" else sp) / "impl"
    n_impl = 0
    if (impl / "index.jsonl").exists() and not (kdir / "impl" / "index.jsonl").exists():
        shutil.copytree(impl, kdir / "impl", dirs_exist_ok=True)
        n_impl = sum(1 for _ in open(kdir / "impl" / "index.jsonl"))
    meta = seed_info(kdir / "knowledge.db")
    keep = Path(work) / "gk_seed"                          # the base of the merge (the db as it started)
    if sp.resolve() != keep.resolve():
        keep.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(db, keep / "knowledge.db")
        if impl.exists() and not (keep / "impl").exists():
            shutil.copytree(impl, keep / "impl")
    rec.update(applied=True, snapshot=meta.get("snapshot"), part=meta.get("part") or rec["part"],
               run_uid=meta.get("run_uid"), seed_counts=meta.get("counts"), impl_records=n_impl,
               seed=str(keep))
    state["knowledge_global"] = rec
    return rec


def merge_back(work: Path, cfg, rec: dict | None) -> dict | None:
    """readwrite: merge this run into the store (errors are reported, never raised)."""
    if not rec or rec.get("mode") != "readwrite" or not cfg.knowledge_global_store:
        return None
    kdir = Path(work) / "knowledge"
    args = ["--store", cfg.knowledge_global_store, "merge", "--run", str(kdir), "--problem", rec.get("problem") or "",
            "--model", cfg.model]
    if rec.get("applied") and rec.get("seed"):
        args += ["--base", rec["seed"]]
    else:
        args += ["--run-uid", f"workdir:{Path(work).resolve()}", "--part", rec.get("part")]
    try:
        return _merge_cli(cfg, args)
    except Exception as e:  # noqa: BLE001 -- knowledge must never fail a run
        return {"error": repr(e)}
