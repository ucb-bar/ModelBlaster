import asyncio, importlib.util, sys, json, glob, gzip, os, re, difflib, random, collections
E="/Users/krisd/Documents/ModelBlaster-clone/l3_farm/analysis/e0_runs"
sp=importlib.util.spec_from_file_location("agentic_llm","/Users/krisd/Documents/ModelBlaster-clone/ModelBlaster/third_party/KernelBlaster/src/kernelblaster/agents/utils/agentic_llm.py")
m=importlib.util.module_from_spec(sp); sys.modules["agentic_llm"]=m; sp.loader.exec_module(m)
MODEL=sys.argv[1] if len(sys.argv)>1 else "claude-opus-5-5"; NPICK=int(sys.argv[2]) if len(sys.argv)>2 else 8; REP=int(sys.argv[3]) if len(sys.argv)>3 else 2
def blob(idx,sha):
    root=os.path.dirname(idx)
    for p in (f"{root}/blobs/{sha[:2]}/{sha}.gz",):
        if os.path.exists(p): return gzip.open(p).read().decode()
    raise FileNotFoundError(sha)
cands=[]
for idx in glob.glob(f"{E}/*/work/r*/run/trace/index.jsonl"):
    for l in open(idx):
        r=json.loads(l)
        if r.get("t")=="llm" and r.get("ok") and isinstance(r.get("response"),dict) and r.get("prompt_sha") and 2000<r.get("prompt_chars",0)<30000:
            cands.append((idx,r))
random.seed(7)
bucket=collections.defaultdict(list)
for idx,r in cands: bucket[(r.get("role"), "43" in idx)].append((idx,r))
picks=[]
for k,v in sorted(bucket.items(), key=lambda kv: str(kv[0])):
    random.shuffle(v); picks+= v[:max(1,NPICK//len(bucket))]
picks=picks[:NPICK]
FENCE=re.compile(r"```(?:cuda|cpp|c\+\+|cu|c)?\s*\n(.*?)```",re.S)
def code(t):
    b=FENCE.findall(t or ""); return max(b,key=len) if b else None
def sim(a,b):
    if a is None or b is None: return None
    return round(difflib.SequenceMatcher(None,a.splitlines(),b.splitlines(),autojunk=False).ratio(),3)
async def one(idx,r):
    msgs=[]
    for mm in r["messages"]:
        c=mm["content"]; msgs.append({"role":mm["role"],"content":blob(idx,c["sha"]) if isinstance(c,dict) else c})
    api=blob(idx,r["response"]["sha"])
    outs=await asyncio.gather(*[m._claude_cli_once(msgs,MODEL) for _ in range(REP)],return_exceptions=True)
    rows=[]
    ac=code(api)
    for o in outs:
        if isinstance(o,Exception): rows.append({"error":str(o)[:200]}); continue
        t,u=o; cc=code(t)
        rows.append({"chars":len(t),"code":cc is not None,"sim_vs_api":sim(ac,cc),"u":u,"text":t})
    cs=[x for x in rows if x.get("code")]
    return {"run":idx.split("/e0_runs/")[1].split("/")[0],"role":r.get("role"),"slot":r.get("slot"),"tech":r.get("technique"),
            "prompt_chars":r["prompt_chars"],"api_chars":len(api),"api_code":ac is not None,"api_usage":r.get("usage"),"api_seconds":r.get("seconds"),
            "cli":rows,"cli_vs_cli":sim(cs[0]["text"] and code(cs[0]["text"]),code(cs[1]["text"])) if len(cs)>=2 else None,
            "api_text":api}
async def main():
    res=[]
    for i in range(0,len(picks),3):
        res+=await asyncio.gather(*[one(idx,r) for idx,r in picks[i:i+3]])
    json.dump(res,open(f"cmp_{MODEL}.json","w"),indent=1)
    tot=0
    for x in res:
        c=[y for y in x["cli"] if "error" not in y]; cost=sum(y["u"]["cost_usd"] for y in c); tot+=cost
        print(x["run"][:12],x["role"],x["slot"],x["tech"],"prompt",x["prompt_chars"],"| api chars",x["api_chars"],"code",x["api_code"],
              "| cli chars",[y["chars"] for y in c],"code",[y["code"] for y in c],"sim_vs_api",[y["sim_vs_api"] for y in c],"cli_vs_cli",x["cli_vs_cli"],"| $%.3f"%cost,
              "| errs",sum("error" in y for y in x["cli"]))
    print("total cli cost $%.2f"%tot)
asyncio.run(main())
