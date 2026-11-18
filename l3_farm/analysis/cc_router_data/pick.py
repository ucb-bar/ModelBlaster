import json, glob, collections, os
E="/Users/krisd/Documents/ModelBlaster-clone/l3_farm/analysis/e0_runs"
by=collections.defaultdict(list)
for idx in glob.glob(f"{E}/*/work/r*/run/trace/index.jsonl"):
    run=idx.split("/e0_runs/")[1].split("/")[0]; rd=idx.split("/work/")[1].split("/")[0]
    for l in open(idx):
        r=json.loads(l)
        if r.get("t")=="llm" and r.get("ok") and "response" in r and r.get("prompt_sha"):
            by[r.get("prompt_sha","?")].append((run,rd,r.get("slot","-"),r.get("technique","-"),r.get("role","-"),r.get("step",-1),r.get("prompt_chars",0),(r["response"].get("n",0) if isinstance(r["response"],dict) else len(r["response"])),(r.get("usage") or {}).get("output_tokens"),idx))
multi=[(k,v) for k,v in by.items() if len({x[0] for x in v})>=2]
print(len(by),"distinct prompts;",len(multi),"seen in >=2 runs")
for k,v in sorted(multi,key=lambda kv:-len(kv[1]))[:25]:
    print(k[:8],len(v),v[0][2],v[0][3],v[0][4],"chars",v[0][6],"resp",[x[7] for x in v][:4],sorted({x[0] for x in v}))
