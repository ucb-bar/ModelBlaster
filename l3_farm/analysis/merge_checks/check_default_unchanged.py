#!/usr/bin/env python3
"""X-135: is the default flow unchanged by the merge? Pre-merge frozen tree vs post-merge frozen tree, same problem, fixed seed,
all flags default, global knowledge off; then again with the default-on switches off.

Arms (names are the run's `arm`; one pair per arm x problem x seed is compared, pre vs post):
  stub_default  zero LLM  : the flow with `--model stub:graphopt` (identity LLM: every slot comes back unchanged), all flags default
  stub_killsw   zero LLM  : same + MB_PRUNE_DEAD=0, --efficiency-rules off, --fallback-repair off, --final-guard off (each only if the
                            tree has the flag); REQUIRED to be identical pre vs post (round keys, IR, manifests, generated slot files,
                            plan, decisions, graph hints)
  live_default  LLM       : the real flow, flags default (the reference numbers; files cannot be byte-identical because the LLM
                            samples, so keys / manifests of round 0 / plan structure are compared and timing within a tolerance)
  live_killsw   LLM       : optional, same with the switches off

The zero-LLM arms are the deterministic part of the check: with an identity LLM the round keys, the IR, the exported manifests, the
slot sources and the plan structure do not depend on sampling. A difference there under the kill switches is a regression of the
default flow; a difference only in the default arms is the intended effect of the default-on features (reported, to be reviewed).

  check_default_unchanged.py run     JOB TARBALL --arms stub_default,stub_killsw [--problem 1_MLP] [--seed 1] [--live-budget "..."]
  check_default_unchanged.py compare --pre pre --post post [--arms ...] [--tol 0.05] [--out PREFIX]
  check_default_unchanged.py diff    DIR_A DIR_B                 two collected run dirs (e.g. the same tree twice: calibration)
  check_default_unchanged.py arms    TARBALL                     which kill switches the tree supports and the resulting arm flags

A run's pack comes from `mc_runner.py collect` (runs/<label>/<arm>/<problem>_s<seed>/). Output: JSON + markdown with tree stamp / sha256,
flags, node job id and memory clock of both sides, and a pass/fail table.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mc_common as C  # noqa: E402
import mc_runner as R  # noqa: E402

KILL_FLAGS = [("efficiency_rules", "--efficiency-rules off"), ("fallback_repair", "--fallback-repair off"),
              ("final_guard", "--final-guard off")]
KILL_ENV = "MB_PRUNE_DEAD=0"


# ------------------------------------------------------------------------------------------------ arms

def supported(tarball) -> dict:
    """Which switches exist in the tree: Config fields of kb_graph_opt.py, MB_PRUNE_DEAD in graph_prune.py (read from the tarball)."""
    txt, prune = "", ""
    with tarfile.open(tarball) as t:
        for m in t:
            if m.name.endswith("pipeline/kb_graph_opt.py"):
                txt = t.extractfile(m).read().decode()
            elif m.name.endswith("pipeline/graph_prune.py"):
                prune = t.extractfile(m).read().decode()
    cfg = txt.split("class Config", 1)[-1].split("\ndef ", 1)[0]
    out = {k: bool(re.search(rf"^\s+{k}\s*:", cfg, re.M)) for k, _ in KILL_FLAGS}
    out["prune_dead_env"] = "MB_PRUNE_DEAD" in prune
    return out


def arm_spec(tarball, arm) -> dict:
    sup = supported(tarball)
    kill_flags = " ".join(f for k, f in KILL_FLAGS if sup.get(k))
    live = arm.startswith("live")
    tb = "" if live else "--torch-baseline off"     # zero-LLM arms skip the (slow, node-dependent) torch.compile comparison; not part of the key
    if "killsw" not in arm:       # stub_default, live_default, stub_default_rep2 (calibration repeats) ...
        return {"stub": not live, "flags": tb, "envs": ""}
    if "killsw" in arm:
        return {"stub": not live, "flags": (kill_flags + " " + tb).strip(), "envs": KILL_ENV if sup.get("prune_dead_env", True) else ""}
    raise SystemExit(f"unknown arm {arm}")


# ------------------------------------------------------------------------------------------------ normalisation

RUN_RE = re.compile(r"/tmp/krisd/mc/runs/[^/\"\s]+/[^/\"\s]+/[^/\"\s]+")
TREE_RE = re.compile(r"/tmp/krisd/mc/[a-z0-9]+/(?=ModelBlaster|modelblaster)")
VOLATILE_KEYS = {"data_sha1", "ts", "timestamp", "t", "started", "finished", "created", "pid", "uuid"}


CACHE_ID_RE = re.compile(r"cache:[0-9a-f]{12}")      # kernel-cache entry ids: not stable between runs of the same tree (seen on 8_ResNetBasicBlock)


def norm_str(s: str) -> str:
    return CACHE_ID_RE.sub("cache:<id>", TREE_RE.sub("<TREE>/", RUN_RE.sub("<RUN>", s)))


KEEP_FLOATS = {"budget"}        # plan.json `budget` holds configuration (min_share, ...), not measurements: compared exactly


def norm_json(x, floats=False):
    """Structural form: run/tree paths replaced, volatile keys dropped; floats -> '<f>' when `floats` (plan-like files: measured shares / us)."""
    if isinstance(x, dict):
        return {k: norm_json(v, floats and k not in KEEP_FLOATS) for k, v in sorted(x.items()) if k not in VOLATILE_KEYS}
    if isinstance(x, list):
        return [norm_json(v, floats) for v in x]
    if isinstance(x, str):
        return norm_str(x)
    if isinstance(x, float) and floats:
        return "<f>"
    return x


def jhash(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()[:16]


def bhash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ one pair

CLASSES = [
    # (class, glob under the run dir, how)
    ("manifest", "work/r*/problems/*/manifest.json", "json"),
    ("manifest", "work/r*/profile/*/problem/manifest.json", "json"),
    ("manifest", "work/final/*/problem/manifest.json", "json"),
    ("slot_source", "work/r*/problems/*/slots/*.cu", "bytes"),
    ("slot_source", "work/r*/problems/*/walker_*.cu", "bytes"),
    ("slot_source", "work/r*/profile/*/problem/slots/*.cu", "bytes"),
    ("slot_source", "work/r*/profile/*/problem/walker_*.cu", "bytes"),
    ("slot_source", "work/final/*/problem/slots/*.cu", "bytes"),
    ("slot_source", "work/final/*/problem/walker_*.cu", "bytes"),
    ("plan", "work/r*/plan.json", "jsonf"),
    ("graph_hints", "work/graph_hints.json", "jsonf"),
    ("csv", "work/**/*.csv", "bytes"),
]


def rel_files(d: Path):
    out = {}
    for cls, pat, how in CLASSES:
        for p in sorted(d.glob(pat)):
            if p.is_file():
                rel = str(p.relative_to(d))
                out[rel] = (cls, how, p)
    return out


def decisions_rows(rd: dict):
    rows = []
    for r in rd["rounds"]:
        for h, s in r["decisions"]:
            rows.append((r["round"], h, s))
    return rows


def diff_runs(a: Path, b: Path) -> dict:
    ra, rb = C.read_run(a), C.read_run(b)
    res = {"a": str(a), "b": str(b), "checks": []}

    def add(name, ok, detail="", info=False):
        # info = does not count for the verdict. Calibration (same tree, same seed, zero LLM, run twice): keys, IR, manifests, slot sources
        # and plan items are identical; graph hints / decisions are NOT (they hang on measured shares: one run proposed linear+relu, the other not)
        res["checks"].append({"check": name, "pass": bool(ok), "detail": detail, "info": info})

    # 1. round keys
    ka, kb = ra["keys"], rb["keys"]
    add("round key per round (path-independent hash)", ka == kb, f"A={ka} B={kb}" if ka != kb else f"{len(ka)} rounds identical: {ka}")
    k0a = (ra["rounds"][0]["key_dict"] if ra["rounds"] else None) or {}
    k0b = (rb["rounds"][0]["key_dict"] if rb["rounds"] else None) or {}
    dd = {k: (k0a.get(k), k0b.get(k)) for k in sorted(set(k0a) | set(k0b)) if k0a.get(k) != k0b.get(k)}
    add("round-0 key entries (IR path reduced to basename)", not dd and bool(k0a), "identical" if not dd else f"differing entries (A, B): {dd}")
    # 2. IR
    ga = sorted(a.glob("irs/*/graph.json"))
    gb = sorted(b.glob("irs/*/graph.json"))
    ha = [bhash(p) for p in ga]
    hb = [bhash(p) for p in gb]
    add("IR graph.json", ha == hb and bool(ha), f"{ha} vs {hb}")
    # 3. generated / exported files
    fa, fb = rel_files(a), rel_files(b)
    only_a, only_b = sorted(set(fa) - set(fb)), sorted(set(fb) - set(fa))
    add("file set (manifests, slot sources, plans, hints)", not only_a and not only_b,
        f"{len(fa)} vs {len(fb)} files; only A: {only_a[:6]}; only B: {only_b[:6]}")
    by = {}
    for rel in sorted(set(fa) & set(fb)):
        cls, how, pa = fa[rel]
        _, _, pb = fb[rel]
        if how == "bytes":
            same = bhash(pa) == bhash(pb)
        else:
            try:
                ja = norm_json(json.loads(pa.read_text()), floats=(how == "jsonf"))
                jb = norm_json(json.loads(pb.read_text()), floats=(how == "jsonf"))
                same = jhash(ja) == jhash(jb)
            except Exception:
                same = bhash(pa) == bhash(pb)
        by.setdefault(cls, []).append((rel, same))
    for cls, items in by.items():
        bad = [r for r, s in items if not s]
        add(f"{cls} files", not bad, f"{len(items)} compared, {len(bad)} differ" + (f": {bad[:6]}" if bad else ""),
            info=(cls == "graph_hints"))
    # 4. decisions / ingest structure
    da, db = decisions_rows(ra), decisions_rows(rb)
    add("graph decisions (round, hint, status)", da == db, f"{len(da)} vs {len(db)} decisions" + ("" if da == db else f"; A={da[:6]} B={db[:6]}"),
        info=True)
    ia = [(r["round"], r["ingest_status"]) for r in ra["rounds"]]
    ib = [(r["round"], r["ingest_status"]) for r in rb["rounds"]]
    add("ingest status counts per round", ia == ib, f"A={ia} B={ib}" if ia != ib else "identical", info=True)
    pa = [(r["round"], r["n_plan_items"]) for r in ra["rounds"]]
    pb = [(r["round"], r["n_plan_items"]) for r in rb["rounds"]]
    add("plan items per round", pa == pb, f"A={pa} B={pb}" if pa != pb else "identical")
    # 5. final timing (informational here; the table applies the tolerance)
    res["timing"] = {"a": {k: ra.get(k) for k in ("model_e2e_us", "quiet_e2e_us", "seed_e2e_us", "torch_us")},
                     "b": {k: rb.get(k) for k in ("model_e2e_us", "quiet_e2e_us", "seed_e2e_us", "torch_us")}}
    res["meta"] = {"a": ra["meta"], "b": rb["meta"]}
    res["flow_status"] = {"a": ra["flow_status"], "b": rb["flow_status"]}
    return res


def ratio(x, y):
    return (y / x) if (x and y) else None


def timing_checks(res, tol):
    ta, tb = res["timing"]["a"], res["timing"]["b"]
    rows = []
    ma, mb = res["meta"]["a"] or {}, res["meta"]["b"] or {}
    same_node = bool(ma.get("job") and ma.get("job") == mb.get("job")) or bool(ma.get("host") and ma.get("host") == mb.get("host"))
    for k in ("model_e2e_us", "quiet_e2e_us"):
        r = ratio(ta.get(k), tb.get(k))
        ok = r is not None and abs(r - 1) <= tol
        rows.append({"check": f"{k} post/pre", "pass": ok, "detail": f"{ta.get(k)} -> {tb.get(k)} (x{r:.3f})" if r else "missing",
                     "info": not same_node})
    slow = [x for x in (ma, mb) if x.get("slow_memory_die")]
    rows.append({"check": "memory clock 9001 MHz on both nodes", "pass": not slow,
                 "detail": f"A {ma.get('mem_clock_mhz')} MHz ({ma.get('host')}), B {mb.get('mem_clock_mhz')} MHz ({mb.get('host')})",
                 "info": False})
    rows.append({"check": "paired on one node", "pass": same_node, "detail": f"A job {ma.get('job')}, B job {mb.get('job')} "
                 "(a cross-node timing difference up to ~10% is a node effect: re-measure both finals on one node)", "info": True})
    return rows


# ------------------------------------------------------------------------------------------------ report

def verdict(arm, checks):
    hard = [c for c in checks if not c["info"]]
    bad = [c for c in hard if not c["pass"]]
    if not bad:
        return "PASS"
    if arm.endswith("killsw"):
        return "FAIL"
    return "REVIEW"      # default arm: a difference is intended only if the kill-switch arm is clean


def compare(pre, post, arms, tol, out):
    pre_d, post_d = C.RUNS / pre, C.RUNS / post
    results = []
    for arm in arms:
        for pa in sorted((pre_d / arm).glob("*_s*")):
            pb = post_d / arm / pa.name
            if not pb.exists():
                results.append({"arm": arm, "run": pa.name, "missing": f"no {pb}"})
                continue
            d = diff_runs(pa, pb)
            d["arm"], d["run"] = arm, pa.name
            tc = timing_checks(d, tol if arm.startswith("live") else max(tol, 0.10))
            if arm.startswith("stub"):          # zero-LLM arms: timing of the seed program is informational only
                for c in tc:
                    c["info"] = True
            d["checks"] += tc
            d["verdict"] = verdict(arm, d["checks"])
            results.append(d)
    rows = []
    for d in results:
        if "missing" in d:
            rows.append([d["arm"], d["run"], "MISSING", d["missing"], "", ""])
            continue
        for c in d["checks"]:
            rows.append([d["arm"], d["run"], ("INFO " if c["info"] and not c["pass"] else "") + ("PASS" if c["pass"] else "FAIL"),
                         c["check"], c["detail"][:160], ""])
    md = C.md_head(f"X-135 default-unchanged check: {pre} vs {post}", {
        "generated": C.now(),
        "tree": f"pre={_tree(pre_d)}; post={_tree(post_d)}", "flags": "see per-run meta.json (cmdline); arms " + ",".join(arms),
        "job": _field(results, "job"), "host": _field(results, "host"), "mem_clock_mhz": _field(results, "mem_clock_mhz")})
    md += "\n## Summary\n\n" + C.md_table([[d.get("arm"), d.get("run"), d.get("verdict", "MISSING")] for d in results],
                                          ["arm", "run", "verdict"])
    md += "\n\nVerdict rules: kill-switch arms must be identical (FAIL otherwise); default arms differing = REVIEW (intended only " \
          "if the kill-switch arm of the same problem is clean); timing rows are INFO when the two runs were on different nodes.\n"
    md += "\n## All checks\n\n" + C.md_table(rows, ["arm", "run", "result", "check", "detail", ""])
    outp = Path(out)
    C.write_json(outp.with_suffix(".json"), {"pre": pre, "post": post, "tol": tol, "generated": C.now(), "results": results})
    outp.with_suffix(".md").write_text(md + "\n")
    print(md)
    print("written", outp.with_suffix(".json"), outp.with_suffix(".md"))
    return results


def reference(label, out):
    """The reference numbers of one tree: one row per collected run (pre-merge: the numbers the post-merge rerun is diffed against)."""
    rows, data = [], []
    for m in sorted((C.RUNS / label).glob("*/*/meta.json")):
        meta = C.load_json(m, {})
        rr = C.read_run(m.parent)
        x = {"arm": meta["arm"], "problem": meta["problem"], "seed": meta["seed"], "budget": meta["budget"], "flags": meta["flags"], "envs": meta["envs"],
             "flow": rr["flow_status"], "model_e2e_us": rr["model_e2e_us"], "quiet_us": rr["quiet_e2e_us"], "seed_us": rr["seed_e2e_us"],
             "torch_us": rr["torch_us"], "calls": meta["llm_calls_ledger"], "usd": meta["llm_usd"], "wall_s": meta["flow_wall_s"],
             "round_key": rr["keys"][:1], "n_rounds": len(rr["rounds"]), "hints": rr["hints"], "job": meta["job"], "host": meta["host"],
             "mem_clock_mhz": meta["mem_clock_mhz"], "decisions": decisions_rows(rr), "ingest": [(r["round"], r["ingest_status"]) for r in rr["rounds"]]}
        data.append(x)
        rows.append([x["arm"], x["problem"], x["seed"], x["flow"], x["model_e2e_us"], x["quiet_us"], x["torch_us"], x["calls"], x["usd"], x["wall_s"],
                     ",".join(x["round_key"]), x["job"], x["mem_clock_mhz"]])
    t = C.tree_info(next(iter(sorted(C.FROZEN.glob(f"mb_{label}_*.tar.gz"))))) if list(C.FROZEN.glob(f"mb_{label}_*.tar.gz")) else {}
    md = C.md_head(f"Reference runs of tree '{label}'", {"generated": C.now(), "tree": f"{t.get('label')} {t.get('stamp')} sha256 {t.get('sha256')}",
                   "flags": "per row (budget, flags, envs in the JSON)", "job": ", ".join(sorted({str(x['job']) for x in data})),
                   "host": ", ".join(sorted({str(x['host']) for x in data})), "mem_clock_mhz": ", ".join(sorted({str(x['mem_clock_mhz']) for x in data}))})
    md += "\n" + C.md_table(rows, ["arm", "problem", "seed", "flow", "model_e2e us", "quiet us", "torch us", "LLM calls", "$", "wall s", "round-0 key", "job", "mem MHz"]) + "\n"
    outp = Path(out)
    C.write_json(outp.with_suffix(".json"), {"label": label, "tree": t, "generated": C.now(), "runs": data})
    outp.with_suffix(".md").write_text(md)
    print(md)


def _tree(d: Path):
    for m in sorted(d.glob("*/*/meta.json")):
        j = C.load_json(m, {})
        t = j.get("tree") or {}
        return f"{t.get('label')} {t.get('stamp')} sha256 {t.get('sha256')}"
    return "?"


def _field(results, k):
    vals = []
    for d in results:
        for side in ("a", "b"):
            v = ((d.get("meta") or {}).get(side) or {}).get(k)
            if v is not None and v not in vals:
                vals.append(v)
    return ", ".join(map(str, vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("job", type=int)
    r.add_argument("tarball")
    r.add_argument("--arms", default="stub_default,stub_killsw")
    r.add_argument("--problem", default="1_MLP")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--budget", default=R.STD_BUDGET)
    r.add_argument("--th", type=float, default=4)
    c = sub.add_parser("compare")
    c.add_argument("--pre", required=True)
    c.add_argument("--post", required=True)
    c.add_argument("--arms", default="stub_default,stub_killsw,live_default,live_killsw")
    c.add_argument("--tol", type=float, default=0.05)
    c.add_argument("--out")
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    q = sub.add_parser("reference")
    q.add_argument("--label", default="pre")
    q.add_argument("--out")
    s = sub.add_parser("arms")
    s.add_argument("tarball")
    x = ap.parse_args()
    if x.cmd == "arms":
        print(json.dumps({"supported": supported(x.tarball), **{a: arm_spec(x.tarball, a) for a in
                          ("stub_default", "stub_killsw", "live_default", "live_killsw")}}, indent=1))
    elif x.cmd == "run":
        for arm in x.arms.split(","):
            spec = arm_spec(x.tarball, arm)
            R.launch(x.job, x.tarball, arm, x.problem, x.seed, stub=spec["stub"], flags=spec["flags"], envs=spec["envs"],
                     budget=x.budget, th=x.th)
    elif x.cmd == "reference":
        reference(x.label, x.out or str(HERE / f"reference_{x.label}"))
    elif x.cmd == "diff":
        res = diff_runs(Path(x.a), Path(x.b))
        for ck in res["checks"]:
            print(("PASS" if ck["pass"] else "FAIL"), ck["check"], "|", ck["detail"][:200])
    else:
        compare(x.pre, x.post, x.arms.split(","), x.tol, x.out or str(HERE / f"default_unchanged_{x.pre}_vs_{x.post}"))


if __name__ == "__main__":
    main()
