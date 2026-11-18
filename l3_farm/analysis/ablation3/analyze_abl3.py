#!/usr/bin/env python3
"""Ablation-3 analysis. usage (from l3_farm): python3 analysis/ablation3/analyze_abl3.py [--instance abl3|abl3smoke] [--fetch-ledgers] [--out FILE]
Reads state/<inst>/farm.json, results_<inst>/<alias>/work/{summary,model_runtime}.json, analysis/ablation3/abl3_map.json (or smoke_map.json),
torch_fp32/<problem>.json (fp32 reference), and the relay ledgers results_<inst>/_ledgers/<job>.jsonl (--fetch-ledgers pulls /tmp/krisd/farm/llm_ledger.jsonl of every
node in only_jobs through farm.py's ssh; per-problem usd = est_usd of ledger lines inside the problem's [started, finished] window on its node: one problem per node at a time).
Per cell: final model_e2e_us, seed e2e, torch eager/graph/best (fp16, in-flow), fp32 torch graph (reference), accepted hint ids (by kind), llm calls, usd, wall min, rounds,
controller moves (arm A1/A2/A6), region loop commits, windows/merge activity. Tables: per problem x arm final us; paired ratio arm/A0 (A0 us / arm us > 1: arm faster) per problem,
geomean / median / wins over problems finished in both; usd and calls per arm; accepted-hint table. Single runs (~15% noise): the table says n, no significance test."""
import argparse, json, math, os, statistics as st, sys, time
from pathlib import Path
H = Path(__file__).resolve().parent; FARM = H.parent.parent
ap = argparse.ArgumentParser(); ap.add_argument("--instance", default="abl3"); ap.add_argument("--fetch-ledgers", action="store_true"); ap.add_argument("--out", default="")
a = ap.parse_args()
RES = FARM / f"results_{a.instance}"; STATE = json.load(open(FARM / "state" / a.instance / "farm.json"))["problems"]
mp = json.load(open(H / ("abl3_map.json" if a.instance == "abl3" else "smoke_map.json")))
names = mp["names"] if "names" in mp else mp
cfg = json.load(open(H / ("config.json" if a.instance == "abl3" else "config_smoke.json")))
if a.fetch_ledgers:
    os.environ["FARM_CONFIG"] = str(H / ("config.json" if a.instance == "abl3" else "config_smoke.json")); sys.path.insert(0, str(FARM)); import farm
    (RES / "_ledgers").mkdir(parents=True, exist_ok=True)
    for j in cfg["instance"]["only_jobs"]:
        rc, o, e = farm.ssh(f"srun --jobid={j} --overlap bash -c 'cat /tmp/krisd/farm/llm_ledger.jsonl 2>/dev/null'", timeout=300)
        (RES / "_ledgers" / f"{j}.jsonl").write_bytes(o); print("ledger", j, len(o.splitlines()), "lines")
led = {}
for f in (RES / "_ledgers").glob("*.jsonl") if (RES / "_ledgers").exists() else []:
    L = []
    for ln in f.read_text().splitlines():
        try: r = json.loads(ln); L.append((time.mktime(time.strptime(r["ts"][:19], "%Y-%m-%dT%H:%M:%S")), r))
        except Exception: pass
    led[int(f.stem)] = L           # ledger ts are in the nodes' (= controller's, Pacific) local time, farm.json started/finished are epoch: mktime matches
def usd(n, p):
    j, t0, t1 = p.get("node"), p.get("started"), p.get("finished") or time.time()
    if j not in led or not t0: return None
    return round(sum(r.get("est_usd") or 0 for ts, r in led[j] if t0 - 30 <= ts <= t1 + 30), 2)
def cell(alias):
    w = RES / alias / "work"; p = STATE.get(alias, {}); c = {"status": p.get("status")}
    try:
        v = next(iter(json.load(open(w / "model_runtime.json")).values()))
        c.update(final=v.get("model_e2e_us"), seed=v.get("seed_e2e_us"), ok=v.get("passed"), tstatus=v.get("torch_status"), t_eager=v.get("torch_eager_us"), t_graph=v.get("torch_graph_us"),
                 t_best=v.get("torch_best_us"), x_eager=v.get("speedup_vs_torch_eager"), x_graph=v.get("speedup_vs_torch"))
    except Exception: pass
    try:
        s = json.load(open(w / "summary.json")); c.update(calls=s.get("llm_calls"), accepted=s.get("accepted") or [], rejected=s.get("rejected") or [], nrounds=len(s["rounds"]) if isinstance(s.get("rounds"), list) else s.get("rounds"))
        rc = s.get("round_controller") or {}; c["moves"] = "/".join(x.get("move", "?")[0] for x in rc.get("rounds", [])) or None; c["rc_stop"] = rc.get("stop_reason")
        c["keys"] = [k for k in ("round_controller", "merge_sibling_convs", "region_windows", "region_loop") if k in s]
    except Exception: pass
    try:
        e = json.load(open(RES / alias / "run_meta.json"))["events"]; c["wall_min"] = round((e[-1]["t"] - e[0]["t"]) / 60)
    except Exception: pass
    c["usd"] = usd(alias, p)
    return c
C = {n: cell(n) for n in names}
arms = list(mp["arms"]) if "arms" in mp else sorted({v["arm"] for v in names.values()})
probs = sorted({v["problem"] for v in names.values()}, key=lambda x: int(x.split("_")[0]))
at = {(v["problem"], v["arm"]): n for n, v in names.items()}
def gm(x): x = [i for i in x if i]; return math.exp(sum(map(math.log, x)) / len(x)) if x else float("nan")
out = [f"# Ablation 3 results ({a.instance}) generated {time.strftime('%F %T')}", "", "## Final model_e2e_us (fp16, L40S; lower is better; '-' = not finished)", "",
       "| problem | fp32 torch graph | fp16 torch graph | " + " | ".join(arms) + " |", "|---|---|---|" + "---|" * len(arms)]
def fp32(p):
    try: return json.load(open(H / "torch_fp32" / f"{p}.json")).get("graph_us")
    except Exception: return None
for p in probs:
    row = [C.get(at.get((p, A)), {}).get("final") for A in arms]
    tg = next((C[at[(p, A)]].get("t_graph") for A in arms if (p, A) in at and C[at[(p, A)]].get("t_graph")), None)
    out.append(f"| {p} | {fp32(p) or '-'} | {tg and round(tg) or '-'} | " + " | ".join(f"{x:,.0f}" if x else "-" for x in row) + " |")
out += ["", "## Paired vs A0 (A0 final / arm final per problem; >1 = arm faster). Single runs, ~15% run-to-run noise: no significance claimed.", "",
        "| arm | n | geomean | median | wins/n | per problem |", "|---|---|---|---|---|---|"]
for A in arms:
    if A == "A0": continue
    r = []
    for p in probs:
        x0, x1 = C.get(at.get((p, "A0")), {}).get("final"), C.get(at.get((p, A)), {}).get("final")
        if x0 and x1: r.append((p.split("_")[0], x0 / x1))
    if r: out.append(f"| {A} | {len(r)} | {gm([b for _, b in r]):.3f} | {st.median([b for _, b in r]):.3f} | {sum(b > 1 for _, b in r)}/{len(r)} | " + ", ".join(f"{p}:{b:.2f}" for p, b in r) + " |")
out += ["", "## Cost and activity per arm", "", "| arm | cells done | usd (sum, ledger) | usd/problem | llm calls/problem | wall min/problem | accepted hints/problem | summary keys present |", "|---|---|---|---|---|---|---|---|"]
for A in arms:
    cs = [C[at[(p, A)]] for p in probs if (p, A) in at]; d = [c for c in cs if c.get("status") == "done"]
    u = [c["usd"] for c in d if c.get("usd") is not None]; ca = [c["calls"] for c in d if c.get("calls")]; w = [c["wall_min"] for c in d if c.get("wall_min")]
    out.append(f"| {A} | {len(d)}/{len(cs)} | {sum(u):.0f} | {st.mean(u) if u else float('nan'):.1f} | {st.mean(ca) if ca else float('nan'):.0f} | {st.mean(w) if w else float('nan'):.0f} | "
               f"{st.mean([len(c.get('accepted') or []) for c in d]) if d else float('nan'):.1f} | {sorted({k for c in d for k in c.get('keys', [])})} |")
out += ["", "## Per cell", "", "| alias | arm | status | final us | seed us | x torch graph (fp16) | x torch eager | calls | usd | wall min | rounds | moves | accepted |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for n, v in names.items():
    c = C[n]; f_ = lambda x, d=0: "-" if x is None else f"{x:,.{d}f}"
    out.append(f"| {n} | {v['arm']} | {c.get('status')} | {f_(c.get('final'))} | {f_(c.get('seed'))} | {f_(c.get('x_graph'), 2)} | {f_(c.get('x_eager'), 2)} | {c.get('calls', '-')} | {f_(c.get('usd'), 1)} | {c.get('wall_min', '-')} | {c.get('nrounds', '-')} | {c.get('moves') or '-'} | {', '.join(c.get('accepted') or []) or '-'} |")
txt = "\n".join(out); print(txt)
if a.out: Path(a.out).write_text(txt + "\n")
