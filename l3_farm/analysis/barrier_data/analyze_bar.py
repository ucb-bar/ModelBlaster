#!/usr/bin/env python3
"""X-151 analysis of a barrier test instance: python3 analyze_bar.py i_ind  -> timeline per problem, park / resume overhead, imports."""
import json, sys, time, glob
from pathlib import Path
d = Path(sys.argv[1]); s = json.load(open(d / "farm.json")); res = d / "results"
t0 = min(p.get("meta", {}).get("history", [{}])[0].get("iso") and p.get("started", 1e18) for p in s["problems"].values())
fm = lambda t: "-" if not t else f"{(t - t0) / 60:6.1f}"
b = s["sched"]["barrier"]
print("barrier:", b.get("why"), "| opened at", fm(b.get("opened")), "min after first start")
print(f"{'problem':22s} {'first start':>11s} {'parked':>7s} {'resumed':>8s} {'wait min':>8s}  node-side resume import            halfway/char")
for n, p in sorted(s["problems"].items(), key=lambda kv: kv[1].get("parked_at") or 0):
    st = json.load(open(res / n / "work" / "state.json")) if (res / n / "work" / "state.json").exists() else {}
    bar = st.get("barrier") or {}
    rs = (bar.get("resumes") or [{}])[0]
    kg = st.get("knowledge_global") or {}
    refr = [(r.get("at"), r.get("status"), sum(v for v in (r.get("counts") or {}).values() if isinstance(v, int))) for r in kg.get("refreshes", [])]
    print(f"{n:22s} {fm(p.get('meta', {}).get('history') and p.get('first_started') or p.get('started')):>11s} {fm(p.get('parked_at')):>7s} {fm(p.get('resumed_at')):>8s} "
          f"{(p.get('barrier_wait_s') or 0) / 60:8.1f}  {rs.get('status')} rows={rs.get('imported_rows')} parked_s={rs.get('parked_s')}  {refr}")
    for f in glob.glob(str(res / n / "work" / "parked.json")):
        pk = json.load(open(f)); print("   parked.json:", {k: pk.get(k) for k in ('stage', 'parks', 'round')}, "import", (pk.get("import") or {}).get("status"))
    prog = json.load(open(res / n / "work" / "r00" / "progress.json")) if (res / n / "work" / "r00" / "progress.json").exists() else {}
    att = prog.get("attempts", [])
    print("   r00 attempts (start min):", [round((a["start"] - t0) / 60, 1) for a in att], "resumed_after:", [a["resumed_after"] for a in att][-1:])
    pg = p.get("barrier_pkg"); print("   package:", pg and (pg.get("status"), pg.get("counts")))
