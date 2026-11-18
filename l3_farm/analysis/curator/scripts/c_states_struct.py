"""Part A/C (no LLM): structural redundancy of gk_states (predicate intervals + parent + name tokens) and techniques."""
import sys, os, json, re, collections, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
c=con()
S={}
for part,id_,st,d,pv in c.execute('select * from gk_states'):
    S.setdefault(id_,[]).append((st,json.loads(d),json.loads(pv)))
print('states',len(S),'multi-row ids',sum(1 for v in S.values() if len(v)>1))
def conj(p):
    out={}
    def walk(p):
        if not isinstance(p,dict): return
        if 'all' in p:
            for q in p['all']: walk(q)
        elif 'any' in p: pass
        elif 'v' in p or len(p)==2:
            for k,v in p.items():
                if k!='v': out[k]=(v,p['v'])
    walk(p); return out
def interval(op,v):
    if op in ('<','<='): return (-1e9,v)
    if op in ('>','>='): return (v,1e9)
    return (v,v)
def pred_sig(p):
    d={}
    def walk(p):
        if 'all' in p:
            for q in p['all']: walk(q)
        elif 'any' in p: d.setdefault('_any',[]).append(json.dumps(p['any'],sort_keys=True))
        else:
            ks=[k for k in p if k!='v']
            if ks and 'v' in p: d[ks[0]]=(p[ks[0]],p['v'])
    if isinstance(p,dict): walk(p)
    return d
def sim_state(a,b):
    """structural similarity of two predicate dicts: same metric set and each threshold within 35% (abs floor)."""
    A,B=pred_sig(a['predicate']),pred_sig(b['predicate'])
    ma={k for k in A if k!='_any'}; mb={k for k in B if k!='_any'}
    if not ma or not mb: return 0
    inter=ma&mb; jac=len(inter)/len(ma|mb)
    ok=0
    for k in inter:
        (o1,v1),(o2,v2)=A[k],B[k]
        s1='<' if o1 in('<','<=') else '>'; s2='<' if o2 in('<','<=') else '>'
        if s1!=s2 or not all(isinstance(x,(int,float)) for x in (v1,v2)): continue
        if abs(v1-v2)<=max(0.35*max(abs(v1),abs(v2)),0.02): ok+=1
    return jac*(ok/len(inter) if inter else 0)
items={k:v[-1] for k,v in S.items()}
ids=sorted(items)
class UF:
    def __init__(s,n): s.p={i:i for i in n}
    def f(s,i):
        while s.p[i]!=i: s.p[i]=s.p[s.p[i]]; i=s.p[i]
        return i
    def u(s,a,b): s.p[s.f(a)]=s.f(b)
def toks(n): return set(re.split(r'_+',n))
res={}
for thr in (0.75,1.0):
    uf=UF(ids); npair=0
    for a,b in itertools.combinations(ids,2):
        A,B=items[a][1],items[b][1]
        if A.get('parent')!=B.get('parent'): continue
        if sim_state(A,B)>=thr: uf.u(a,b); npair+=1
    cl=collections.Counter(uf.f(i) for i in ids)
    res[f'predicate_thr{thr}']={'clusters':len(cl),'foldable':len(ids)-len(cl),'largest':sorted(cl.values(),reverse=True)[:10],'pairs':npair}
    print(thr,res[f'predicate_thr{thr}'])
# name-token jaccard (same parent) >=0.5
uf=UF(ids)
for a,b in itertools.combinations(ids,2):
    if items[a][1].get('parent')!=items[b][1].get('parent'): continue
    ta,tb=toks(a),toks(b)
    if len(ta&tb)/len(ta|tb)>=0.5: uf.u(a,b)
cl=collections.Counter(uf.f(i) for i in ids)
res['name_jaccard0.5_same_parent']={'clusters':len(cl),'foldable':len(ids)-len(cl),'largest':sorted(cl.values(),reverse=True)[:10]}
print(res['name_jaccard0.5_same_parent'])
# usage in outcomes
use=collections.Counter()
for (r,) in c.execute('select rec from gk_outcomes'):
    r=json.loads(r)
    for k in ('state_before','state_after'):
        if r.get(k): use[r[k]]+=1
res['states_used_in_outcomes']=len([s for s in ids if use[s]]); res['outcome_state_labels_not_in_gk_states']=len([s for s in use if s not in items])
print(res['states_used_in_outcomes'],res['outcome_state_labels_not_in_gk_states'],len(use))
st_status=collections.Counter(items[i][0] for i in ids)
nprob=collections.Counter(len(items[i][2].get('problems',[])) for i in ids)
res['status']=dict(st_status); res['n_problems_hist']=sorted(nprob.items())
print(st_status,sorted(nprob.items()))
# usage concentration
tot=sum(use.values()); top=use.most_common(10); print(tot,top)
json.dump(res,open(BASE+'/results/A_states.json','w'),indent=1)
