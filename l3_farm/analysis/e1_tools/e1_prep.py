#!/usr/bin/env python3
"""E1 (X-120) local prep: pack the sweep artifacts each problem needs + a per-problem 'global' pool.

  e1_prep.py data [P ...]     -> e1_data/stage/data.tar.gz  (per problem: problem.py, work/rNN/winners, graph_hints.json,
                                 final problem sources, final IR, gimpl/ = per-problem global pool)
  e1_prep.py code             -> e1_data/stage/mb_frozen.tar.gz (+ sha256) frozen snapshot of the live ModelBlaster tree

Global pool (reuse-winners=global): every sweep problem's implementation-store records (results/*/work/knowledge/impl,
speedup_vs_seed >= 1.02, target cuda) from OTHER problems, per op kind of problem P ranked by shape similarity to P's
instance shapes (mean over P's distinct shapes of the closest record shape; log-ratio distance over numeric fields),
top GCODES distinct codes per kind. kb_reuse.pool_candidates then takes <= 6 of them (one source per code).
(Why not the raw store: pool_candidates sorts by slot_us per instance, which across problems just ranks the smallest
shapes first.)
"""
import hashlib, io, json, math, os, shutil, subprocess, sys, tarfile
from pathlib import Path

L3 = Path(__file__).resolve().parents[2]
RES = L3 / "results"
OUT = L3 / "analysis" / "e1_data" / "stage"
GCODES = 8
GKINDS = 12
EXCLUDE = {"16_DenseNet201", "48_Mamba2ReturnY"}


def problems():
    ps = sorted(d.name for d in RES.iterdir() if d.is_dir() and d.name[0].isdigit() and d.name not in EXCLUDE
                and (d / "work" / "final" / d.name / "problem" / "manifest.json").exists())
    return ps


def dist(a, b):
    d = 0.0
    for k in set(a) | set(b):
        x, y = a.get(k), b.get(k)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            d += abs(math.log2((abs(x) + 1) / (abs(y) + 1)))
        elif x != y:
            d += 4.0
    return d


_RECS = None


def all_records():
    global _RECS
    if _RECS is None:
        _RECS = []
        for d in sorted(RES.iterdir()):
            idx = d / "work" / "knowledge" / "impl" / "index.jsonl"
            if not idx.exists():
                continue
            for ln in idx.read_text().splitlines():
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if r.get("target", "cuda") != "cuda" or (r.get("speedup_vs_seed") or 0) < 1.02:
                    continue
                r["_src"] = d.name
                _RECS.append(r)
    return _RECS


def gimpl(p, tw):
    """add problems/<p>-global pool into tar writer tw under data/<p>/gimpl/"""
    g = json.loads((RES / p / "work" / "final" / p / "ir" / "graph.json").read_text())
    kinds = {}
    for o in g["ops"]:
        if o["op"] == "view":
            continue
        kinds.setdefault(o["op"], {})[json.dumps(o.get("shape", {}), sort_keys=True)] = o.get("shape", {})
    order = sorted(kinds, key=lambda k: -len(kinds[k]))[:GKINDS]
    by_kind = {}
    for r in all_records():
        if r["_src"] != p and r["op"] in order:
            by_kind.setdefault(r["op"], []).append(r)
    rows, codes = [], {}
    for k in order:
        shp = list(kinds[k].values())
        best = {}
        for r in by_kind.get(k, []):
            sh = [s for s in (r.get("shapes") or []) if isinstance(s, dict)]
            if not sh:
                continue
            sc = sum(min(dist(a, b) for b in sh) for a in shp) / len(shp)
            key = r["code_sha"]
            if key not in best or (sc, -(r.get("speedup_vs_seed") or 0)) < best[key][0]:
                best[key] = ((sc, -(r.get("speedup_vs_seed") or 0)), r)
        for key, (sc, r) in sorted(best.items(), key=lambda kv: kv[1][0])[:GCODES]:
            c = RES / r["_src"] / "work" / "knowledge" / "impl" / "code" / f"{r['code_sha']}.cu"
            if c.exists():
                rr = {kk: v for kk, v in r.items() if kk != "_src"}
                rr["_from_problem"] = r["_src"]
                rr["_sim"] = round(sc[0], 2)
                rows.append(rr)
                codes[r["code_sha"]] = c
    add(tw, f"data/{p}/gimpl/index.jsonl", ("\n".join(json.dumps(r) for r in rows) + "\n").encode())
    for sha, c in codes.items():
        add(tw, f"data/{p}/gimpl/code/{sha}.cu", c.read_bytes())
    return {"kinds": len(order), "records": len(rows), "codes": len(codes)}


def add(tw, arc, data):
    ti = tarfile.TarInfo(arc)
    ti.size = len(data)
    ti.mtime = 1792900000
    tw.addfile(ti, io.BytesIO(data))


def pack_data(ps):
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {}
    with tarfile.open(OUT / "data.tar.gz", "w:gz") as tw:
        for p in ps:
            R = RES / p
            add(tw, f"data/{p}/problem.py", (L3 / "problems" / f"{p}.py").read_bytes())
            for w in sorted((R / "work").glob("r[0-9][0-9]/winners")):
                for f in sorted(w.rglob("*")):
                    if f.is_file():
                        add(tw, f"data/{p}/work/{w.parent.name}/winners/{f.relative_to(w)}", f.read_bytes())
            gh = R / "work" / "graph_hints.json"
            if gh.exists():
                add(tw, f"data/{p}/work/graph_hints.json", gh.read_bytes())
            fin = R / "work" / "final" / p
            for f in sorted((fin / "problem").rglob("*")):
                if f.is_file() and "/build/" not in str(f) and f.name != "data.bin":
                    add(tw, f"data/{p}/work/final/{p}/problem/{f.relative_to(fin / 'problem')}", f.read_bytes())
            add(tw, f"data/{p}/work/final/{p}/ir/graph.json", (fin / "ir" / "graph.json").read_bytes())
            sj = R / "work" / "summary.json"
            if sj.exists():
                add(tw, f"data/{p}/summary.json", sj.read_bytes())
            stats[p] = gimpl(p, tw)
            print(p, stats[p], flush=True)
    (OUT / "gimpl_stats.json").write_text(json.dumps(stats, indent=1))
    print("size", (OUT / "data.tar.gz").stat().st_size)


def pack_code():
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "mb_frozen.tar.gz"
    root = Path.home() / "Documents" / "ModelBlaster-clone"
    env = dict(os.environ, COPYFILE_DISABLE="1")
    subprocess.run(["tar", "--no-xattrs", "-czf", str(out), "--exclude=.git", "--exclude=__pycache__", "--exclude=*.pyc",
                    "--exclude=ModelBlaster/gpu_eval/runs", "--exclude=ModelBlaster/gpu_eval/ablations",
                    "--exclude=ModelBlaster/experiments", "--exclude=ModelBlaster/third_party/KernelBlaster/runs",
                    "-C", str(root), "ModelBlaster"], check=True, env=env)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    (OUT / "mb_frozen.sha256").write_text(h + "\n")
    print(out, out.stat().st_size, h)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "data":
        pack_data(sys.argv[2:] or problems())
    elif cmd == "code":
        pack_code()
    elif cmd == "list":
        print(len(problems()), problems())
