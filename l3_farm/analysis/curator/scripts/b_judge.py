"""Runs on the node. usage: b_judge.py sample_file out_cache passes(1|2|O)   O = opus reference on first N"""
import sys, json, random
sys.path.insert(0,'/tmp/krisd/curator')
import llm, judge_lib as J
mini=json.load(open('/tmp/krisd/curator/claims_min.json'))
sample_file, cache, mode = sys.argv[1], sys.argv[2], sys.argv[3]
limit=int(sys.argv[4]) if len(sys.argv)>4 else None
S=json.load(open(sample_file)); S=S['sample'] if isinstance(S,dict) else S
if limit: S=S[:limit]
def job(p):
    a,b=mini[p['a']],mini[p['b']]
    model=llm.O if mode=='O' else llm.H
    if mode=='2':
        t,_=llm.call(model,J.SYS,J.prompt_v2(b,a),120,'judge2')     # swapped order, different wording
        l=J.parse(t,J.M2,'verdict'); l=J.swap(l) if l else None
    else:
        t,_=llm.call(model,J.SYS,J.prompt_v1(a,b),120,'judge'+mode)
        l=J.parse(t,J.M1,'label')
    return {'label':l,'raw':t[:160]}
items=[(f"{mode}|{p['a']}|{p['b']}",p) for p in S]
res=llm.run_many(items,job,12,cache)
import collections
print(mode,collections.Counter(v.get('label') for v in res.values()), 'usd',round(llm.usd(),3))
