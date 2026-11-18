#!/usr/bin/env python3
"""Read-only growth / quality report of the headline global knowledge store (run it every monitoring tick).

    python3 l3_farm/analysis/headline/kb_growth.py [STORE_DIR] [--all] [--top N]

Copies the CURRENT snapshot db (and nothing else is written) to a temp dir, opens the copy read-only, and prints:
  1. per-merge growth table (from gk_ledger: cumulative claims/hyp/states/outcomes, deltas, impl records) with the
     exact table counts from snapshots/vNNNN.json where those snapshots still exist (the farm prunes old ones);
     default shows the last 15 merges + every 10th, --all shows all;
  2. current table counts, db size, impl_code files vs gk_impl rows (orphans), per-problem contribution,
  3. outcome quality: status mix split by phase (char = characterization on the naive seed, r00.. = optimisation rounds),
     speedup outliers (>50x), status/speedup disagreement, error rows, non-positive times,
  4. top technique x op_class success stats (non-char outcomes: share improved, mean partition_effect) and the raw
     gk_stats universal rows,
  5. impl db: ops, speedup_vs_seed range, distinct code_sha, cross-problem identical kernels, reuse_seed count,
  6. RED FLAG lines (empty/regressing growth, stat outliers, stale impl files, ...).
Times are shown in PDT (UTC-7). Python 3 stdlib only.
"""
import collections, datetime, glob, json, os, re, shutil, sqlite3, statistics as S, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
ALL = "--all" in sys.argv
TOP = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 14
if "--top" in sys.argv and args and args[-1] == str(TOP):
    args.pop()
ROOT = args[0] if args else os.path.normpath(os.path.join(HERE, "../../../knowledge_global_headline"))
TABLES = ["gk_claims", "gk_hypotheses", "gk_states", "gk_techniques", "gk_stats", "gk_outcomes", "gk_overrides",
          "gk_proposals", "gk_triggers", "gk_impl", "gk_ledger"]
PDT = datetime.timedelta(hours=-7)


def pdt(ts):
    return (datetime.datetime.fromisoformat(ts) + PDT).strftime("%H:%M:%S")


cur = open(os.path.join(ROOT, "current")).read().strip()
tmp = tempfile.mkdtemp(prefix="kbgrowth-")
try:
    src = os.path.join(ROOT, "snapshots", cur + ".db")
    dst = os.path.join(tmp, cur + ".db")
    shutil.copyfile(src, dst)
    con = sqlite3.connect(f"file:{dst}?mode=ro&immutable=1", uri=True)
    hist = [json.loads(l) for l in open(os.path.join(ROOT, "history.jsonl")) if l.strip()]
    snapjson = {}
    for f in glob.glob(os.path.join(ROOT, "snapshots", "v*.json")):
        try:
            j = json.load(open(f))
            snapjson[j["id"]] = j
        except (OSError, ValueError):
            pass
    flags = []

    # ---- 1. per-merge growth
    led = con.execute("select version,problem,attempt,base,ts,report from gk_ledger order by version").fetchall()
    print(f"store {ROOT}\ncurrent={cur}  merges(history)={len(hist)}  ledger rows={len(led)}  "
          f"first={pdt(hist[0]['ts'])} last={pdt(hist[-1]['ts'])} PDT  problems merged={len({h['run']['problem'] for h in hist})}")
    print("\nversion  time(PDT) problem                         att base  | +cl  ev ret +hyp +st +out +prop implrec | cumCl cumHy cumOut | exact counts (snapshot json: cl/hyp/st/tech/stats/out/ovr/prop/trg/impl)")
    cum = collections.Counter()
    nrows = len(led)
    for i, (v, p, att, base, ts, rep) in enumerate(led):
        d = json.loads(rep)
        cl, ev, ret = d["claims"]["added"], d["claims"]["evidence_updates"], d["claims"]["retired"]
        cum["cl"] += cl; cum["hy"] += d["hypotheses"]["added"]; cum["out"] += d["outcomes"]; cum["st"] += d["states"]["added"]
        if not (ALL or i >= nrows - 15 or (i + 1) % 10 == 0 or i == 0):
            continue
        ex = ""
        if v in snapjson:
            c = snapjson[v]["counts"]
            ex = "/".join(str(c.get(t, "")) for t in TABLES[:-1]) + f" {snapjson[v]['bytes']/1e6:.1f}MB"
        print(f"{v}  {pdt(ts)}  {p[:30]:30} {att:>3} {base[:4]:4} | {cl:3d} {ev:3d} {ret:3d} {d['hypotheses']['added']:3d} {d['states']['added']:3d} "
              f"{d['outcomes']:4d} {d['proposals']:4d} {d['impl']['records']:5d} | {cum['cl']:5d} {cum['hy']:5d} {cum['out']:6d} | {ex}")
    zero = [v for v, p, att, base, ts, rep in led
            if json.loads(rep)["claims"]["added"] == 0 and json.loads(rep)["outcomes"] == 0 and json.loads(rep)["stats"]["rows"] == 0]
    print(f"zero-content merges (0 claims, 0 outcomes, 0 stat rows): {zero}")
    t = [datetime.datetime.fromisoformat(h["ts"]) for h in hist]
    last_h = [x for x in t if (t[-1] - x).total_seconds() <= 3600]
    print(f"merge rate: {len(last_h)} in the last hour before the newest merge; median gap {S.median([(t[i+1]-t[i]).total_seconds() for i in range(len(t)-1)]):.0f}s")

    # ---- 2. current counts
    print("\n== current table counts")
    cnt = {tb: con.execute(f"select count(*) from {tb}").fetchone()[0] for tb in TABLES + ["gk_contrib", "gk_log"]}
    print("  " + "  ".join(f"{k[3:]}={v}" for k, v in cnt.items()), f"| db {os.path.getsize(dst)/1e6:.1f} MB")
    for k in ("gk_claims", "gk_hypotheses", "gk_states", "gk_proposals", "gk_techniques"):
        print(f"  {k} status:", dict(con.execute(f"select status,count(*) from {k} group by 1").fetchall()))
    ob = con.execute("select sum(length(rec))+sum(length(v2)) from gk_outcomes").fetchone()[0] or 0
    print(f"  gk_outcomes payload {ob/1e6:.1f} MB = {100*ob/os.path.getsize(dst):.0f}% of db; {ob/max(1,cnt['gk_outcomes']):.0f} B/outcome")
    ic = os.path.join(ROOT, "impl_code")
    files = {f[:-3] for f in os.listdir(ic) if f.endswith(".cu")} if os.path.isdir(ic) else set()
    I = [(r[0], json.loads(r[1]), json.loads(r[2])) for r in con.execute("select id,data,prov from gk_impl")]
    shas = {d["code_sha"] for _, d, _ in I}
    print(f"  impl_code files={len(files)}  gk_impl rows={len(I)}  distinct code_sha={len(shas)}  files not in this snapshot={len(files-shas)}  rows without file={sum(1 for _,d,_ in I if d['code_sha'] not in files)}")
    if files - shas > set() and len(files - shas) > 40:
        flags.append(f"impl_code has {len(files-shas)} files not referenced by gk_impl (no GC?)")

    # per-problem contribution
    O = [(r[0], r[1], json.loads(r[2]), json.loads(r[3])) for r in con.execute("select id,problem,rec,v2 from gk_outcomes")]
    oc = collections.Counter(o[1] for o in O)
    ic_ = collections.Counter(p["problem"] for _, _, p in I)
    cl_first = collections.Counter(json.loads(r[0])["first"]["problem"] for r in con.execute("select prov from gk_claims"))
    merges = collections.Counter(h["run"]["problem"] for h in hist)
    print("\n== per-problem contribution (merges / claims created / outcomes / impl rows), top %d by outcomes" % TOP)
    for p, n in oc.most_common(TOP):
        print(f"  {p:32} merges={merges[p]} claims={cl_first[p]:3d} outcomes={n:4d} impl={ic_[p]:3d}")
    print(f"  problems with merges={len(merges)}; with only 1 merge={sum(1 for v in merges.values() if v==1)}; "
          f"with >8 outcomes={sum(1 for v in oc.values() if v>8)}; contributing 0 claims={[p for p in merges if cl_first[p]==0][:8]}")

    # ---- 3. outcome quality
    print("\n== outcomes by phase (char=characterization on naive seed; r00..=optimisation rounds; orig/vari=other)")
    ph = collections.defaultdict(list)
    for _, p, r, v in O:
        s = r["problem_id"].split("__")[-1]
        ph["char" if s == "char" else (s[:3] if s[:1] == "r" else s[:4])].append(r)
    for k, l in sorted(ph.items(), key=lambda x: -len(x[1])):
        sp = [x["speedup"] for x in l if x.get("speedup")]
        print(f"  {k:5} n={len(l):5d} improved={100*sum(x['status']=='improved' for x in l)/len(l):3.0f}% median slot speedup={S.median(sp) if sp else 0:.2f} >50x={sum(x>50 for x in sp)}")
    sps = [r["speedup"] for _, _, r, _ in O if r.get("speedup")]
    hi = [r for _, _, r, _ in O if (r.get("speedup") or 0) > 50]
    bad_t = sum(1 for _, _, r, _ in O if any(isinstance(r.get(k), (int, float)) and r[k] <= 0 for k in ("e2e_before", "e2e_after", "metric_before", "metric_after")))
    dis = sum(1 for _, _, r, _ in O if r["status"] == "regressed" and (r.get("speedup") or 0) > 1.05)
    unc = sum(1 for _, _, r, _ in O if (r.get("speedup") or 0) > 1.5 and (r.get("partition_effect") if r.get("partition_effect") is not None else 1) < 0.05)
    errs = collections.Counter((o[1], o[2]["status"]) for o in O if o[2]["status"] in ("compile_error", "runtime_error"))
    print(f"  speedup max={max(sps):.0f}x p99={sorted(sps)[int(.99*len(sps))]:.0f}x  >50x={len(hi)} ({100*len(hi)/len(sps):.1f}%)  non-positive times={bad_t}  "
          f"status=regressed but slot speedup>1.05: {dis}  slot>1.5x but e2e confirm<0.05: {unc}")
    print(f"  error outcomes={sum(errs.values())} by problem/status: {dict(errs.most_common(6))}")
    if bad_t:
        flags.append(f"{bad_t} outcomes with non-positive times")
    if hi and sum(1 for r in hi if '__char' not in r['problem_id']) > 0.3 * len(hi):
        flags.append("many >50x outcomes outside the characterization phase (check baseline/clock)")
    # clock/noise sanity: same parent impl on the SAME shape bucket re-measured with a large spread
    pb = collections.defaultdict(list)
    for _, _, r, _ in O:
        if r.get("parent_impl_id") and r.get("metric_before"):
            pb[(r["parent_impl_id"], r.get("shape_bucket"))].append(r["metric_before"])
    drift = [(k, max(v) / min(v)) for k, v in pb.items() if len(v) >= 3 and max(v) / min(v) > 1.5]
    print(f"  same-parent+same-shape metric_before spread >1.5x (NOT clock: rollouts branch from a naive parent whose id stays while the incumbent improved): {len(drift)} of {sum(1 for v in pb.values() if len(v)>=3)} groups with >=3 samples")
    infra = sum(1 for _, _, r, _ in O if str(r.get("error") or "").startswith("[infra]"))
    print(f"  [infra]-error outcomes stored as runtime_error (no partition_effect, so not in gk_stats): {infra}")
    if infra:
        flags.append(f"{infra} [infra] failures stored as outcomes (scp/node issues), dilute per-technique error/success rates")
    if len(drift) > 5:
        flags.append(f"{len(drift)} (parent,shape) groups re-measured with >1.5x spread")

    # ---- 4. technique x op_class
    print("\n== technique x op_class, NON-char outcomes (n, improved%, mean partition_effect, errors); top by n")
    agg = collections.defaultdict(lambda: [0, 0, 0.0, 0, 0])
    achar = collections.defaultdict(lambda: [0, 0.0])
    for _, p, r, _ in O:
        pe = r.get("partition_effect")
        if "__char" in r["problem_id"]:
            if pe is not None:
                achar[r["technique"]][0] += 1; achar[r["technique"]][1] += pe
            continue
        a = agg[(r["technique"], r["op_class"])]
        a[0] += 1; a[1] += r["status"] == "improved"; a[2] += pe or 0.0; a[3] += r["status"] in ("compile_error", "runtime_error")
    for (t_, oc_), a in sorted(agg.items(), key=lambda x: -x[1][0])[:TOP + 6]:
        print(f"  {t_[:34]:34} {oc_:13} n={a[0]:4d} improved={100*a[1]/a[0]:3.0f}% mean_pe={a[2]/a[0]:+.3f} err={a[3]}")
    print("  characterization-phase mean partition_effect (naive-seed wins; inflates gk_stats):",
          {t_: f"{v[1]/v[0]:.2f}(n={v[0]})" for t_, v in sorted(achar.items(), key=lambda x: -x[1][0])[:5]})
    R = con.execute("select item,op,n,s,ss from gk_stats where level='universal' and state='*'").fetchall()
    print("\n== gk_stats universal op-level rows (n, mean gain, sd) incl. char phase; top by n")
    for it, op, n, s, ss in sorted(R, key=lambda r: -r[2])[:TOP]:
        m = s / n
        print(f"  {it[:34]:34} {op:13} n={int(n):4d} mean={m:+.3f} sd={max(ss/n-m*m,0)**.5:.3f}")
    allst = con.execute("select item,level,op,state,n,s,ss from gk_stats").fetchall()
    insane = [r for r in allst if r[4] <= 0 or r[5] / r[4] > 1.0001 or r[5] / r[4] < -1.0001 or r[6] + 1e-6 < r[5] ** 2 / r[4]]
    print(f"  stat rows={len(allst)} insane (n<=0, |mean|>1, ss<s^2/n)={len(insane)}")
    if insane:
        flags.append(f"{len(insane)} insane gk_stats rows")

    # ---- claims
    C = [(r[0], json.loads(r[1]), json.loads(r[2])) for r in con.execute("select status,data,prov from gk_claims")]
    kinds = collections.Counter(d.get("kind") for _, d, _ in C)
    act = [(s, d, p) for s, d, p in C if s == "active"]
    multi = sum(1 for _, _, p in act if len(p["problems"]) >= 3)
    big = sum(1 for _, d, _ in C if re.search(r"\b(\d{3,})(?:\.\d+)?x\b", d.get("text", "")))
    txt = collections.Counter(d["text"].strip().lower() for _, d, _ in act)
    print(f"\n== claims: {len(C)} total, {len(act)} active, retired={len(C)-len(act)}, kinds={dict(kinds)}")
    print(f"  active claims backed by >=3 problems={multi} ({100*multi/max(1,len(act)):.0f}%); single-problem={sum(1 for _,_,p in act if len(p['problems'])==1)}; "
          f"exact-duplicate texts={sum(v-1 for v in txt.values() if v>1)}; claims quoting >=100x gains={big}; contradicted(contradict>0)={sum(1 for _,d,_ in C if d.get('contradict',0)>0)}")
    if big > 0.1 * len(C):
        flags.append(f"{big} claims quote >=100x speedups (vs naive seed) - risk of misleading priors")

    # ---- 5. impl
    print("\n== impl db")
    sp = sorted(d["speedup_vs_seed"] for _, d, _ in I)
    print(f"  rows={len(I)} ops={len({d['op'] for _,d,_ in I})} speedup_vs_seed min={sp[0]:.2f} med={sp[len(sp)//2]:.2f} max={sp[-1]:.0f} >50x={sum(x>50 for x in sp)} "
          f"max_rel_err max={max((d.get('max_rel_err') or 0) for _,d,_ in I):.3g} reuse_seed-tagged={sum('reuse_seed' in d['techniques'] for _,d,_ in I)}")
    sg = collections.defaultdict(list)
    for _, d, p in I:
        sg[d["code_sha"]].append(p["problem"])
    xp = {k: v for k, v in sg.items() if len(set(v)) > 1}
    print(f"  identical kernels shared by >1 problem (sha reuse across problems)={len(xp)}: {[sorted(set(v)) for v in list(xp.values())[:4]]}")
    byop = collections.defaultdict(set)
    for _, d, p in I:
        byop[d["op"]].add(p["problem"])
    print(f"  ops covered by >=2 problems={sum(1 for v in byop.values() if len(v)>=2)}/{len(byop)}")
    weird = [d["buckets"][0] for _, d, _ in I if d["buckets"] and re.search(r"n\d{10,}", d["buckets"][0])]
    print(f"  degenerate shape buckets (n>=1e10; maxpool/ewise2/strided_copy/catn product-of-dims): {len(weird)} rows")
    if weird:
        flags.append(f"{len(weird)} gk_impl rows have degenerate shape buckets (n>=1e10): weak reuse matching")

    # ---- 6. flags
    if len(hist) >= 3 and cnt["gk_claims"] == 0:
        flags.append("no claims in store")
    if zero:
        flags.append(f"{len(zero)} zero-content merges {zero[:4]}")
    if not any(h["run"]["problem"] for h in hist[-3:]):
        flags.append("history tail malformed")
    gaps = [(t[i + 1] - t[i]).total_seconds() for i in range(len(t) - 1)]
    if (datetime.datetime.now(datetime.timezone.utc) - t[-1]).total_seconds() > 1800:
        flags.append(f"no merge for >30 min (newest {pdt(hist[-1]['ts'])} PDT)")
    if len(led) != len(hist):
        flags.append(f"ledger rows {len(led)} != history rows {len(hist)} (lost/dup merge?)")
    lin = json.loads(con.execute("select value from gk_meta where key='lineage'").fetchone()[0])
    if len(lin) != len(hist) or lin[-1] != cur:
        flags.append("lineage/current mismatch")
    print("\n== RED FLAGS" + ("" if flags else ": none"))
    for f in flags:
        print("  - " + f)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
