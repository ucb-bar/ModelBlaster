. /tmp/krisd/family/sh/env.sh
WK=/tmp/krisd/family/pilot/smoke; R=$WK/r00/run/kb_problem__conv2d_f16_family__r00
python - <<PY
import json
r=json.load(open("$R/result.json"))
print("calls",r["llm_calls"],"final",r["final_e2e_us"],"base",r["baseline_e2e_us"],"ok",r["final_ok"])
for s in r["steps"]: print(s["status"],s["technique"],s["slot_us_before"],s["slot_us_after"],s["fix_attempts"],s["context"].get("e2e",{}).get("family_guard"))
f=r["family"]["conv2d_f16"]; print("guard rej",f["guard_rejections"], list(f["final_op_us"].items())[:3])
n=r["ncu"]["final"].get("conv2d_f16",{}); print("ncu keys",sorted(n)[:30], len(n.get("kernels",[])), n.get("top_stalls"), n.get("tensor_pipe_pct"))
PY
grep -h "family\|Shape family" $WK/r00/run.log | head -5 | cut -c1-300
ls $WK/r00/run/trace/blobs | head -3
python - <<PY
import json,glob
for l in open("$WK/r00/run/trace/index.jsonl"):
    d=json.loads(l)
    if d.get("kind") in ("llm","prompt") or "prompt" in json.dumps(d)[:200]:
        print(json.dumps(d)[:300]); break
PY
