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


def key_fields(cfg, gk: dict | None) -> dict:
    """What global knowledge adds to the round key: nothing when off (existing work dirs keep their key); the
    seed snapshot when on; with ``--knowledge-global-refresh`` a constant marker (refreshes never change it)."""
    if gk is None:
        return {}
    out = {"knowledge_global": gk.get("snapshot") if gk.get("applied") else "unseeded"}
    if getattr(cfg, "knowledge_global_refresh", ""):
        out["knowledge_global_refresh"] = True
    return out


def refresh_at_round(work: Path, cfg, state: dict, rnd: int, why: str | None = None) -> dict | None:
    """``--knowledge-global-refresh DIR`` at the start of round ``rnd``: import the newest refresh package the
    controller dropped in DIR (``*.tar.gz`` or a dir with ``refresh.json``) into ``work/knowledge`` (KernelBlaster
    ``merge import``: one transaction, other runs' knowledge only, logged in the db for the merge). Every
    package seen is moved to ``DIR/done/``. The result is appended to ``state["knowledge_global"]["refreshes"]``.
    Never raises: a failed import leaves the knowledge dir as it was and the round goes on."""
    d = Path(getattr(cfg, "knowledge_global_refresh", "") or "")
    rec = state.get("knowledge_global")
    if not str(d) or str(d) == "." or rec is None or not rec.get("applied"):
        return None
    kdir = Path(work) / "knowledge"
    if not d.exists() or not (kdir / "knowledge.db").exists():
        return None
    import tarfile
    import time
    cands = [p for p in d.iterdir() if (p.name.endswith(".tar.gz") or (p / "refresh.json").exists())
             and not p.name.startswith(".")]
    if not cands:
        return None
    done = d / "done"
    done.mkdir(exist_ok=True)

    def to_seq(p: Path):
        try:
            if p.is_dir():
                m = json.loads((p / "refresh.json").read_text())
            else:
                with tarfile.open(p) as tf:
                    f = tf.extractfile("./refresh.json") or tf.extractfile("refresh.json")
                    m = json.loads(f.read())
            return int((m.get("to") or {}).get("seq") or 0)
        except Exception:  # noqa: BLE001 -- an unreadable package is moved aside
            return -1

    ranked = sorted(cands, key=lambda p: (to_seq(p), p.stat().st_mtime))
    res = {"status": "none"}
    for p in reversed(ranked):                        # newest first; an overlapping older one adds nothing new
        pkg = p
        tmp = None
        try:
            if not p.is_dir():
                tmp = done / f".x-{p.name[:-7]}"
                shutil.rmtree(tmp, ignore_errors=True)
                tmp.mkdir(parents=True)
                with tarfile.open(p) as tf:
                    tf.extractall(tmp)
                pkg = tmp
            res = _merge_cli(cfg, ["import", "--kdir", str(kdir), "--pkg", str(pkg)])
        except Exception as e:  # noqa: BLE001 -- knowledge must never fail a run
            res = {"status": "error", "error": repr(e)}
        finally:
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)
        if res.get("status") in ("applied", "already_applied", "up_to_date"):
            break
    for p in cands:
        try:
            os.replace(p, done / p.name)
        except OSError:
            pass
    out = {"round": rnd, "ts": time.time(), **{k: res.get(k) for k in ("status", "rid", "counts", "reason", "error",
                                                                          "sync") if k in res}}
    if why:                                           # halfway sync (kb_halfway.py): the import at a pause, not a round start
        out["at"] = why
    rec.setdefault("refreshes", []).append(out)
    return out


# ---------------------------------------------------------------- A5: implementation pools and structural priors
# In-process helpers for other pipeline stages (kb_reuse, graph search). They import KernelBlaster's knowledge
# package from ``cfg.kb`` / the vendored tree; nothing here runs unless called.

def _kb(kb_root=None):
    import importlib
    import sys
    try:
        return importlib.import_module("kernelblaster.knowledge.impl_pool")
    except ImportError:
        root = Path(kb_root) if kb_root else Path(__file__).resolve().parents[1] / "third_party" / "KernelBlaster"
        sys.path.insert(0, str(root / "src"))
        return importlib.import_module("kernelblaster.knowledge.impl_pool")


def impl_pool(impl_dirs, op: str, signature_sha: str, arch: str, dtype: str = "fp32", *, target: str = "cuda",
              exclude_problem: str | None = None, kb_root=None):
    """One KernelBlaster ``Pool`` (kernels + cross-shape matrix) of (op, signature, arch, dtype) over several impl
    stores (``work/knowledge/impl``, a seed's ``impl/``, other runs'): records, seeded global pool kernels and
    observations of every dir, merged (cells keep the minimum time, failures add up). ``arch`` like ``sm_89``.
    Use ``pool.route(instances)`` / ``pool.best_for(shape)`` for zero-LLM routing (predicted choices carry
    ``guard.needs_check``) and ``ImplStore(d).observe(...)`` to record new matrix cells."""
    _kb(kb_root)
    from kernelblaster.knowledge.impl_store import ImplStore
    from kernelblaster.knowledge.v2.pool_store import _merge_into
    out = None
    for d in impl_dirs or []:
        if not (Path(d) / "index.jsonl").exists() and not (Path(d) / "pool").exists():
            continue
        p = ImplStore(Path(d)).pool(op, signature_sha, arch, dtype, target=target, exclude_problem=exclude_problem)
        if out is None:
            out = p
        else:
            _merge_into(out, p)
    if out is None:
        from kernelblaster.knowledge.impl_pool import Pool
        out = Pool(op, signature_sha, arch, dtype, target)
    return out


def struct_priors(*, work=None, store=None, part: str | None = None, kb_root=None):
    """``StructPriors`` from (first found) the work dir's seed (``gk_seed/struct_priors.json``, written by a seed
    materialized with ``Policy.seed_struct``), a global store (``store``: root path) and the work dir's own decisions
    so far (``state.json``)."""
    _kb(kb_root)
    from kernelblaster.knowledge.v2.priors import StructPriors, rows_from_state
    sp = StructPriors([])
    if work is not None:
        for f in (Path(work) / "gk_seed" / "struct_priors.json", Path(work) / "knowledge" / "struct_priors.json"):
            if f.exists():
                sp.add(StructPriors.load(f, part=part).rows)
                break
    if store is not None:
        from kernelblaster.knowledge.v2.merge import GlobalStore
        sp.add(StructPriors.load(GlobalStore(store), part=part).rows)
    if work is not None and (Path(work) / "state.json").exists():
        sp.add(rows_from_state(Path(work) / "state.json", part=part or "", problem=Path(work).name,
                               run_uid=f"workdir:{Path(work).resolve()}"))
    return sp


def struct_prior(kind: str, context, *, work=None, store=None, part: str | None = None, kb_root=None) -> dict:
    """Prior of a structural decision (A4): ``kind`` in fuse / split_slot / fold_layout / fuse_attention /
    fuse_siblings ..., ``context`` = op kinds (["conv2d_f16", "batchnorm2d_f16"]) or a fused kind / hint id.
    -> ``Prior.to_dict()``: n, accepted, rejected, rate (posterior mean, shrunk to the parent level), lo / hi (Wilson
    90%), alpha / beta (Beta posterior: Thompson-sample it), gain_mean / gain_lo / gain_hi, rounds_to_decide,
    level (exact | classes | head | kind | all | none), levels (counts per level)."""
    return struct_priors(work=work, store=store, part=part, kb_root=kb_root).prior(kind, context, part).to_dict()
