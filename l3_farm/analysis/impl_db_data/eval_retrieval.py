"""Offline retrieval-quality evaluation of the implementation database (A5, zero LLM, CPU only).

Question: for an op/shape, does top-k retrieval contain a kernel within 10% of the best known kernel for that shape,
for the CURRENT ImplStore.retrieve tiers vs the new pool / similarity retrieval?

Data
  (1) pilot cross-evaluation matrix (family_data/pilot/matrix.json): DenseNet121 conv2d_f16, 61 per-shape winners x
      61 shapes, every kernel timed on every shape in the whole-model binary (40 valid everywhere; failing kernels:
      their listed MB_ERROR ops are failures, everything else unknown). Leave-one-shape-out: the query shape's own
      per-shape winner is excluded from the candidates; the query shape's matrix column is hidden from the pool.
  (2) sweep records (results/*/work/knowledge/impl/index.jsonl, 26k records): queries = (op, signature, shape) keys
      timed in >= 2 problems; candidates = other problems' records (leave-one-problem-out); truth = the best time
      any record measured on exactly that shape; a retrieved kernel counts only if it has a measurement on the shape
      (unknown -> miss, reported separately).

    python3 eval_retrieval.py [--out impl_db_data/retrieval_eval.json]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FARM = HERE.parent.parent
KB = Path.home() / "Documents/ModelBlaster-clone/ModelBlaster/third_party/KernelBlaster/src"
sys.path.insert(0, str(KB))

from kernelblaster.knowledge.impl_pool import Pool, SourceIndex, rank_kernels, shape_key  # noqa: E402
from kernelblaster.knowledge.impl_store import ImplRecord, ImplStore, _record, shape_bucket  # noqa: E402

KS = (1, 2, 4, 8)
TOL = 1.10


def comb(n, k):
    return math.comb(n, k) if 0 <= k <= n else 0


def p_random_hit(n, g, k):
    """P(a random k-subset of n candidates contains one of g good ones)."""
    if n <= 0:
        return 0.0
    k = min(k, n)
    return 1.0 - comb(n - g, k) / comb(n, k)


# ------------------------------------------------------------------ (1) pilot matrix

def load_pilot():
    pil = HERE.parent / "family_data/pilot"
    M = json.loads((pil / "matrix.json").read_text())["runs"]
    plan = json.loads((pil / "B_plan.json").read_text())
    shape_of, seed_us = {}, {}
    for it in plan["items"]:
        for oc in it["occurrences"]:
            for o in oc["ops"]:
                shape_of[o] = it["shape"]
                seed_us[o] = oc["us"]                        # the hand seed's time on that shape (plan profile)
    idx = FARM / "results/15_DenseNet121/work/knowledge/impl"
    recs = [json.loads(l) for l in (idx / "index.jsonl").read_text().splitlines() if l.strip()]
    by_code = collections.defaultdict(list)
    for r in recs:
        by_code[r["code_sha"][:12]].append(r)
    kern = {}
    for k12, run in M.items():
        own = run["owner"][0]["op"]
        cands = [r for r in by_code.get(k12, []) if len(r["shapes"]) == 1
                 and shape_key(r["shapes"][0]) == shape_key(shape_of[own])]
        t = {}
        if run["passed"]:
            t = {o: v for o, v in run["per_op_us"].items() if v is not None}
        if cands or by_code.get(k12):
            rec = min(cands or by_code[k12], key=lambda r: r["slot_us"])
        else:
            # 18 of the 61 deployed winners have their source in impl/code but no index line (resume / pack loss):
            # a record as the impl store would have written it (own shape, the matrix time there, seed = plan time)
            sha = next(p.stem for p in (idx / "code").glob(f"{k12}*.cu"))
            us = t.get(own) or 1e9
            rec = {"target": "cuda", "arch": "sm_89", "op": "conv2d_f16", "op_class": "conv", "dtype": "fp16",
                   "signature_sha": recs[0]["signature_sha"], "code_sha": sha, "shapes": [shape_of[own]],
                   "buckets": [shape_bucket("conv", shape_of[own])], "slot_us": us, "seed_us": seed_us.get(own),
                   "speedup_vs_seed": (seed_us.get(own) or us) / us, "techniques": [], "id": f"syn{k12}",
                   "problem_id": f"syn::{own}"}
        fails = set(re.findall(r"MB_ERROR (\S+): ", run.get("err") or "")) if not run["passed"] else set()
        kern[k12] = {"own": own, "rec": rec, "t": t, "fail": fails, "passed": run["passed"],
                     "code": (idx / "code" / f"{rec['code_sha']}.cu").read_text()}
    ops = list(next(iter(M.values()))["per_op_us"])
    return kern, ops, shape_of


def eval_pilot():
    kern, ops, shape_of = load_pilot()
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    route = collections.defaultdict(float)
    route_own = collections.defaultdict(float)
    own_sum = 0.0
    oracle = 0.0
    own_hit = []
    rand = {k: [] for k in KS}
    src_index = SourceIndex({k: v["code"] for k, v in kern.items()})
    detail = []
    for q in ops:
        qs = shape_of[q]
        truth = {k: v["t"][q] for k, v in kern.items() if q in v["t"]}
        best = min(truth.values())
        oracle += best
        own = [k for k, v in kern.items() if v["own"] == q]
        cand = [k for k in kern if k not in own]
        good = {k for k in cand if k in truth and truth[k] <= TOL * best}
        own_hit.append(any(k in truth and truth[k] <= TOL * best for k in own))
        own_t = [truth[k] for k in own if k in truth]
        own_sum += min(own_t) if own_t else best * 3
        for k in KS:
            rand[k].append(p_random_hit(len(cand), len(good), k))

        def record(method, ranked):
            for k in KS:
                top = ranked[:k]
                res[method][k].append(any(c in good for c in top))
            # zero-LLM routing quality: measure the top-4 on the shape and keep the fastest valid one
            t4 = [truth[c] for c in ranked[:4] if c in truth]
            route[method] += min(t4) if t4 else min(truth[c] for c in own if c in truth) if any(c in truth for c in own) else best * 3
            # with the incumbent (the shape's own winner) kept as an option: what a guarded routing pass deploys
            t4o = t4 + [truth[c] for c in own if c in truth]
            route_own[method] += min(t4o) if t4o else best * 3

        # (a) current tiers: ImplStore.retrieve over the candidates' records (top-1 = seed or adapt)
        rows = [_record(kern[c]["rec"]) for c in cand]
        st = _MemStore(rows)
        r = st.retrieve(target="cuda", arch="sm_89", op="conv2d_f16", op_class="conv", signature="x", shapes=[qs],
                        dtype="fp16", sig_override=rows[0].signature_sha)
        top = [x for x in (r["seed"], r["adapt"]) if x is not None]
        record("current_tiers", [x.code_sha[:12] for x in top])
        # (a') current tiers extended to k: exact-bucket records by time per instance, then same-bucket-overlap by
        # speedup over seed (the order the tiers would pick in)
        want = shape_bucket("conv", qs)
        exact = sorted([x for x in rows if want in x.buckets], key=lambda x: x.slot_us)
        rest = sorted([x for x in rows if want not in x.buckets], key=lambda x: -(x.speedup_vs_seed or 0))
        record("bucket_rank", [x.code_sha[:12] for x in exact + rest])
        # (b) similarity over records only (each kernel timed on its own shape: what a per-shape sweep store has)
        p = Pool("conv2d_f16", "s", "sm_89", "fp16")
        for c in cand:
            p.add_kernel(c, record={**kern[c]["rec"], "code_sha": c})
        rb = [x["kernel"] for x in rank_kernels(p, qs)]
        record("similar_records", rb)
        # (b') nearest-shape winner (distance only, no ratio model)
        from kernelblaster.knowledge.impl_pool import shape_distance
        rn = sorted(cand, key=lambda c: (shape_distance(qs, kern[c]["rec"]["shapes"][0]), kern[c]["rec"]["slot_us"]))
        record("nearest_shape", rn)
        # (c) similarity with the cross-shape matrix of the OTHER shapes (pool after cross-evaluation)
        pm = Pool("conv2d_f16", "s", "sm_89", "fp16")
        for c in cand:
            pm.add_kernel(c, record={**kern[c]["rec"], "code_sha": c, "shapes": []})
            for o in ops:
                if o == q:
                    continue
                if o in kern[c]["t"]:
                    pm.observe(c, shape_of[o], us=kern[c]["t"][o], ok=True)
                elif o in kern[c]["fail"]:
                    pm.observe(c, shape_of[o], ok=False)
        rc = [x["kernel"] for x in rank_kernels(pm, qs)]
        record("similar_matrix", rc)
        for m in (1, 2, 8):                               # sensitivity: adaptive width from the m-th neighbour
            record(f"similar_matrix_nb{m}", [x["kernel"] for x in rank_kernels(pm, qs, neighbours=m)])
            record(f"similar_records_nb{m}", [x["kernel"] for x in rank_kernels(p, qs, neighbours=m)])
        for tau in (0.25, 0.75):                          # fixed width (the first version used 0.75)
            record(f"similar_matrix_tau{tau}", [x["kernel"] for x in rank_kernels(pm, qs, tau=tau)])
            record(f"similar_records_tau{tau}", [x["kernel"] for x in rank_kernels(p, qs, tau=tau)])
        # (c') best single kernel on the other shapes (global ranking, no shape similarity)
        tot = {c: sum(kern[c]["t"].get(o, 0) for o in ops if o != q) if kern[c]["passed"] else float("inf") for c in cand}
        record("best_single_other", sorted(cand, key=lambda c: tot[c]))
        # (d) source similarity to the shape's own winner (lineage / diff retrieval: "kernels like mine")
        if own:
            sims = src_index.query(kern[own[0]]["code"])
            record("source_sim_to_own", sorted(cand, key=lambda c: -sims.get(c, 0)))
        # (e) combined: matrix ratio model + small source term
        rcs = [x["kernel"] for x in rank_kernels(pm, qs, query_code=kern[own[0]]["code"] if own else None,
                                                 w_src=0.2, src_index=src_index)]
        record("similar_matrix+src", rcs)
        detail.append({"op": q, "shape": qs, "best_us": best, "n_good": len(good),
                       "current": top[0].code_sha[:12] if top else None, "similar_matrix_top1": rc[0],
                       "similar_records_top1": rb[0]})
    n = len(ops)
    out = {"queries": n, "tolerance": TOL, "own_winner_within_10pct": sum(own_hit) / n,
           "random": {k: round(sum(v) / n, 3) for k, v in rand.items()},
           "hit_at_k": {m: {k: round(sum(v) / n, 3) for k, v in d.items()} for m, d in res.items()},
           "route_top4_family_us": {m: round(v, 1) for m, v in route.items()}, "oracle_family_us": round(oracle, 1),
           "route_top4_plus_own_family_us": {m: round(v, 1) for m, v in route_own.items()},
           "own_winners_family_us": round(own_sum, 1),
           "detail": detail}
    return out


class _MemStore(ImplStore):
    """ImplStore over in-memory records (no files) for the offline evaluation."""

    def __init__(self, rows):
        self._rows = rows

    def records(self):
        return list(self._rows)

    def retrieve(self, *, sig_override=None, **kw):
        # the evaluation's records all carry the real signature sha; route the query to it
        import kernelblaster.knowledge.impl_store as m
        orig = m.signature_sha
        m.signature_sha = lambda s: sig_override
        try:
            return ImplStore.retrieve(self, **kw)
        finally:
            m.signature_sha = orig


# ------------------------------------------------------------------ (2) sweep records

def eval_sweep(universe: str = "records"):
    rows = []
    for f in sorted(glob.glob(str(FARM / "results/*/work/knowledge/impl/index.jsonl"))):
        prob = Path(f).parts[-5]
        for l in open(f):
            r = json.loads(l)
            r["_prob"] = prob
            rows.append(r)
    # truth: best time per (op, sig, exact shape) over all problems; which codes were timed on it
    timed = collections.defaultdict(dict)                  # key -> code_sha -> min us
    probs = collections.defaultdict(set)
    for r in rows:
        if len(r["shapes"]) != 1:
            continue
        k = (r["op"], r["signature_sha"], r["dtype"], shape_key(r["shapes"][0]))
        timed[k][r["code_sha"]] = min(r["slot_us"], timed[k].get(r["code_sha"], float("inf")))
        probs[k].add(r["_prob"])
    queries = [k for k in timed if len(probs[k]) >= 2]
    by_pk = collections.defaultdict(list)
    for r in rows:
        if universe == "winners" and (r.get("speedup_vs_seed") or 0) < 1.02:
            continue
        by_pk[(r["op"], r["signature_sha"], r["dtype"])].append(r)
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    unknown = collections.defaultdict(list)
    cond = collections.defaultdict(list)
    nq = 0
    for k in sorted(queries):
        op, sig, dt, sk = k
        shape = json.loads(sk)
        best = min(timed[k].values())
        good = {c for c, t in timed[k].items() if t <= TOL * best}
        for P in sorted(probs[k]):
            cands = [r for r in by_pk[(op, sig, dt)] if r["_prob"] != P]
            if not cands:
                continue
            nq += 1
            recs = [_record({kk: v for kk, v in r.items() if not kk.startswith("_")}) for r in cands]
            # current tiers
            st = _MemStore(recs)
            oc = cands[0]["op_class"]
            r = st.retrieve(target="cuda", arch=cands[0]["arch"], op=op, op_class=oc, signature="x", shapes=[shape],
                            dtype=dt, sig_override=sig)
            cur = [x.code_sha for x in (r["seed"], r["adapt"]) if x is not None]
            # similarity over the pool of the other problems' records
            p = Pool(op, sig, cands[0]["arch"], dt)
            for x in cands:
                p.add_kernel(x["code_sha"], record=x)
            sim = [p.kernels[x["kernel"]]["code_sha"] for x in rank_kernels(p, shape)]
            for m, ranked in (("current_tiers", cur), ("similar_records", sim)):
                for kk in KS:
                    top = ranked[:kk]
                    res[m][kk].append(any(c in good for c in top))
                    if kk == 1:
                        unknown[m].append(bool(top) and top[0] not in timed[k])
                        if top and top[0] in timed[k]:
                            cond[m].append(top[0] in good)
    return {"universe": universe, "queries": nq, "keys": len(queries),
            "hit_at_k": {m: {k: round(sum(v) / max(len(v), 1), 3) for k, v in d.items()} for m, d in res.items()},
            "top1_unmeasured_on_shape": {m: round(sum(v) / max(len(v), 1), 3) for m, v in unknown.items()},
            "top1_hit_given_measured": {m: [round(sum(v) / max(len(v), 1), 3), len(v)] for m, v in cond.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "retrieval_eval.json"))
    a = ap.parse_args()
    out = {"pilot": eval_pilot(), "sweep_records": eval_sweep("records"), "sweep_winners": eval_sweep("winners")}
    Path(a.out).write_text(json.dumps(out, indent=1))
    show = {k: {kk: vv for kk, vv in v.items() if kk != "detail"} for k, v in out.items()}
    print(json.dumps(show, indent=1))


if __name__ == "__main__":
    main()
