"""Before / after claim consolidation (A5): seed contents, the codegen prompt's Lessons section, prompt size.

Materializes a seed for a few problems from two snapshots of a store COPY (never the live store), then for a sample
of real slot contexts (op class, state, technique of stored outcomes) renders what ``KnowledgeV2.retrieve`` puts in
the Lessons section (``ClaimStore.relevant`` k=10 + ``render``) and counts: claims shown, characters, near-duplicate
pairs among the shown claims (same exact scope block, TF-IDF cosine >= 0.35 -- the consolidation rule), distinct
lessons, support covered.

    python3 consolidation_eval.py STORE_COPY BEFORE_VID AFTER_VID [--out impl_db_data/consolidation_eval.json]
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

KB = Path.home() / "Documents/ModelBlaster-clone/ModelBlaster/third_party/KernelBlaster/src"
sys.path.insert(0, str(KB))

from kernelblaster.knowledge.v2.consolidate import TfIdf, _block  # noqa: E402
from kernelblaster.knowledge.v2.merge import GlobalStore, Policy, materialize  # noqa: E402

PART = "cuda:sm_89|fp16"
PROBLEMS = ["10_ResNet101", "1_MLP", "43_MinGPTCausalAttention", "29_SwinMLP"]


def contexts(db: Path, n: int = 300, seed: int = 0) -> list:
    c = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    rows = c.execute("SELECT DISTINCT json_extract(rec,'$.op_class'), json_extract(rec,'$.state_before'), "
                     "json_extract(rec,'$.technique') FROM gk_outcomes").fetchall()
    c.close()
    rows = sorted(r for r in rows if r[0] and r[2])
    random.Random(seed).shuffle(rows)
    return rows[:n]


def lessons(seed_dir: Path, ctxs: list, problem: str) -> dict:
    from kernelblaster.knowledge.v2.knowledge import KnowledgeV2
    work = Path(tempfile.mkdtemp(prefix="a5ce_"))
    shutil.copyfile(seed_dir / "knowledge.db", work / "knowledge.db")
    kv = KnowledgeV2(work)
    try:
        allc = {c.id: c for c in kv.claims.claims.values()} if hasattr(kv.claims, "claims") else {}
        tf = TfIdf({i: c.text for i, c in allc.items()})
        out = {"contexts": 0, "claims": 0, "chars": 0, "dup_pairs": 0, "distinct": 0, "support": 0, "empty": 0}
        for oc, st, tech in ctxs:
            cl = kv.claims.relevant(target="cuda", arch="sm_89", op_class=oc, state=st, technique=tech,
                                    problem_id=None, slot=None, k=10)
            txt = kv.claims.render(cl) if cl else ""
            out["contexts"] += 1
            out["empty"] += not cl
            out["claims"] += len(cl)
            out["chars"] += len(txt)
            out["support"] += sum(int(c.support) for c in cl)
            ids = [c.id for c in cl]
            dup = 0
            reps: list = []
            for i in ids:
                same = [j for j in reps if _block(allc[i].__dict__) == _block(allc[j].__dict__) and tf.cos(i, j) >= 0.35]
                if same:
                    dup += 1
                else:
                    reps.append(i)
            out["dup_pairs"] += dup
            out["distinct"] += len(reps)
        n = max(out["contexts"], 1)
        out.update({f"mean_{k}": round(out[k] / n, 3) for k in ("claims", "chars", "dup_pairs", "distinct", "support")})
        out["n_claims_in_seed"] = len(allc)
        return out
    finally:
        kv.db.conn.close()
        shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("store")
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--out", default=str(Path(__file__).with_name("consolidation_eval.json")))
    a = ap.parse_args()
    st = GlobalStore(a.store)
    ctxs = contexts(st.path(a.before))
    res = {"contexts": len(ctxs)}
    for vid in (a.before, a.after):
        res[vid] = {}
        for p in PROBLEMS:
            with tempfile.TemporaryDirectory(prefix="a5seed_") as td:
                man = materialize(st, PART, Path(td) / "seed", problem=p, snapshot=vid, policy=Policy())
                r = lessons(Path(td) / "seed", ctxs, p)
                r["seed_counts"] = man["counts"]
                res[vid][p] = r
    Path(a.out).write_text(json.dumps(res, indent=1))
    for vid in (a.before, a.after):
        for p, r in res[vid].items():
            print(vid, p, {k: r[k] for k in r if k.startswith("mean_") or k in ("empty", "n_claims_in_seed")},
                  "seed claims", r["seed_counts"].get("claims"))


if __name__ == "__main__":
    main()
