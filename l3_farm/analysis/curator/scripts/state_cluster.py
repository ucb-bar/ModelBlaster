import itertools, collections
def _pred_sig(p):
    d={}
    def walk(p):
        if not isinstance(p,dict): return
        if 'all' in p:
            for q in p['all']: walk(q)
        elif 'any' in p: pass
        else:
            ks=[k for k in p if k!='v']
            if ks and 'v' in p: d[ks[0]]=(p[ks[0]],p['v'])
    walk(p); return d
def _sim(a,b):
    A,B=_pred_sig(a['predicate']),_pred_sig(b['predicate'])
    ma,mb=set(A),set(B)
    if not ma or not mb: return 0
    inter=ma&mb; ok=0
    for k in inter:
        (o1,v1),(o2,v2)=A[k],B[k]
        s1='<' if o1 in('<','<=') else '>'; s2='<' if o2 in('<','<=') else '>'
        if s1!=s2 or not all(isinstance(x,(int,float)) for x in (v1,v2)): continue
        if abs(v1-v2)<=max(0.35*max(abs(v1),abs(v2)),0.02): ok+=1
    return len(inter)/len(ma|mb)*(ok/len(inter) if inter else 0)
def state_canon(preds, first, thr=1.0):
    ids=sorted(preds); p={i:i for i in ids}
    def f(i):
        while p[i]!=i: p[i]=p[p[i]]; i=p[i]
        return i
    for a,b in itertools.combinations(ids,2):
        if preds[a].get('parent')==preds[b].get('parent') and _sim(preds[a],preds[b])>=thr: p[f(a)]=f(b)
    groups=collections.defaultdict(list)
    for i in ids: groups[f(i)].append(i)
    canon={}
    for g in groups.values():
        rep=min(g,key=lambda i:(first.get(i,0),i))
        for i in g: canon[i]=rep
    return canon
