"""Runs on node: haiku judge on candidate pairs in DESCENDING cosine order; pass1 then pass2 (swapped) on pass1-folds per chunk; wall-time cap."""
import sys, json, time
sys.path.insert(0,'/tmp/krisd/curator')
import llm, judge_lib as J
mini=json.load(open('/tmp/krisd/curator/claims_min.json'))
S=sorted(json.load(open('/tmp/krisd/curator/full_pairs.json')),key=lambda p:-p['cos'])
BUDGET=float(sys.argv[1]); TLIM=float(sys.argv[2])*60; t0=time.time()
def job1(p):
    a,b=mini[p['a']],mini[p['b']]
    t,_=llm.call(llm.H,J.SYS,J.prompt_v1(a,b),120,'full1'); return {'label':J.parse(t,J.M1,'label'),'raw':t[:120]}
def job2(p):
    a,b=mini[p['a']],mini[p['b']]
    t,_=llm.call(llm.H,J.SYS,J.prompt_v2(b,a),120,'full2'); l=J.parse(t,J.M2,'verdict'); return {'label':J.swap(l) if l else None,'raw':t[:120]}
CH=300
for i in range(0,len(S),CH):
    if llm.usd()>BUDGET or time.time()-t0>TLIM: print('stop at',i,'usd',round(llm.usd(),2),'min',round((time.time()-t0)/60,1)); break
    chunk=S[i:i+CH]
    r1=llm.run_many([(f"1|{p['a']}|{p['b']}",p) for p in chunk],job1,12,'/tmp/krisd/curator/jf1.jsonl')
    f1=[p for p in chunk if r1.get(f"1|{p['a']}|{p['b']}",{}).get('label') in J.FOLD]
    llm.run_many([(f"2|{p['a']}|{p['b']}",p) for p in f1],job2,12,'/tmp/krisd/curator/jf2.jsonl')
    print('chunk',i+len(chunk),'/',len(S),'cos>=',chunk[-1]['cos'],'folds1',len(f1),'usd',round(llm.usd(),2),flush=True)
print('done',round(llm.usd(),2))
