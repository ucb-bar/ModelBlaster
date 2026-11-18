B=/tmp/krisd/family/pilot/B/r00; TD=$(ls -d $B/family_tune/*/ | head -1)
python3 - $B $TD <<'PY'
import json,sys,glob,os
B,TD=sys.argv[1],sys.argv[2]
r=json.load(open(glob.glob(B+"/run/kb_*/result.json")[0]))
print("llm_calls",r["llm_calls"],"base_e2e",r["baseline_e2e_us"],"final_e2e",r["final_e2e_us"],"family",{k:v for k,v in r["family"]["conv2d_f16"].items() if k in("resyncs","guard_rejections")})
t=r.get("telemetry",{}).get("llm",{}).get("total",{}); print("tel llm", {k:t.get(k) for k in ("calls","output_tokens","cache_write_input_tokens","cache_read_input_tokens","uncached_input_tokens","truncated")})
import collections; print(collections.Counter(s["status"] for s in r["steps"]))
# best-so-far trajectory vs cumulative calls (step order by completion)
best=r["baseline_e2e_us"]; traj=[]
n=0
for s in r["steps"]:
    n+=1+s["fix_attempts"]
    if (s["context"].get("e2e") or {}).get("accept"): best=min(best, s["e2e_after"] or best); traj.append((n,round(best,1)))
print("traj", traj)
tj=json.load(open(TD+"tune.json")); pj=json.load(open(TD+"pool/pool.json"))
print("tune seconds",tj.get("seconds"),"configs ok",tj.get("config_ok"),"heur",tj.get("heuristic_sum"),"tuned",tj.get("tuned_sum"),"oracle",tj.get("oracle_sum"),"rows",tj.get("rows"))
print("pool", {k:pj.get(k) for k in ("valid","sums","chosen","best_single_sum","composite_sum","oracle_sum")})
for f in ("t0","t1","segments"): print(f, open("/tmp/krisd/family/pilot/B/"+f).read().split())
print(os.path.getmtime(TD+"tune.json"), os.path.getmtime(TD+"pool/pool.json"))
PY
