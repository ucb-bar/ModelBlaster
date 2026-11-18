# Args: <arm>. Graph-search events + round summaries so far.
D=/tmp/krisd/a4/runs/$1/work
python3 - $D <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
f = d / "graph_decisions.jsonl"
if f.exists():
    for l in f.read_text().splitlines():
        e = json.loads(l)
        keep = {k: e.get(k) for k in ("event", "t_run_s", "round", "phase", "hint", "status", "steps", "llm_calls", "usd", "items", "problems", "stopped", "hints", "decisions", "budgets", "steps_saved", "partition_us_after")}
        if e["event"] == "survival":
            keep["problems"] = {k: (v["p_survive"], v["steps"]) for k, v in (e.get("problems") or {}).items() if v["steps"] < e.get("steps", 99)}
        if e.get("evidence"):
            keep["ev"] = {k: e["evidence"].get(k) for k in ("original_us", "variant_us", "why")}
        print(json.dumps({k: v for k, v in keep.items() if v not in (None, [], {})})[:600])
s = d / "state.json"
if s.exists():
    for r in json.loads(s.read_text()).get("rounds", []):
        print("ROUND", r["round"], r.get("partition_us"), r.get("seconds"), r.get("llm_calls_total"), [(x["hint"], x["status"], round(x["gain"], 3) if x.get("gain") is not None else None) for x in r.get("decisions", [])])
PY
