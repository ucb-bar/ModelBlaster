"""Assemble the LLM-curated claim set from judged pairs and evaluate it (counts + time-split claim predictions)."""
import sys, os, json, math, random, collections, statistics as S
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d_load import *
import judge_lib as J
random.seed(3)
R=BASE+'/results/'
mini=json.load(open(BASE+'/work/claims_min.json'))
def rd(f,pre):
    d={}
    if not os.path.exists(R+f): return d
    for l in open(R+f):
        x=json.loads(l); k=x['k'].split('|',1)[1]; d[k]=x['v'].get('label')
    return d
j1={**rd('j1.jsonl','1'),**rd('jf1.jsonl','1')}; j2={**rd('j2.jsonl','2'),**rd('jf2.jsonl','2')}
jo=rd('jo.jsonl','O')
def lab(a,b,d):
    if f'{a}|{b}' in d: return d[f'{a}|{b}']
    if f'{b}|{a}' in d: return J.swap(d[f'{b}|{a}'])
    return None
cos={}
P=json.load(open(BASE+'/work/cand_pairs.json'))['pairs']
for a,b,c,k in P: cos[(a,b)]=c
def edge_accept(a,b,mode='strict'):
    l1,l2=lab(a,b,j1),lab(a,b,j2)
    if mode=='strict': return l1 in J.FOLD and l2 in J.FOLD
    if mode=='opus': return lab(a,b,jo) in J.FOLD
    if mode=='verified': return l1 in J.FOLD and l2 in J.FOLD and lab(a,b,jo) in J.FOLD
    return l1 in J.FOLD
def contradict(a,b): return lab(a,b,j1)=='contradict' or lab(a,b,j2)=='contradict'
O=load_outcomes(); byid={o['id']:o for o in O}
cl=[x for x in claims() if x['status']=='active']
cid={x['id']:x for x in cl}
nbr=collections.defaultdict(list)
for (a,b),c in cos.items():
    if a in cid and b in cid: nbr[a].append((b,c)); nbr[b].append((a,c))
def cluster(ids, mode='strict'):
    """leader clustering (no chaining): leaders = highest-support first; a claim joins the leader it has an accepted edge with (max cos)."""
    ids=sorted(ids,key=lambda i:(-cid[i]['support'],i)); leader={}; leaders=set()
    idset=set(ids)
    for i in ids:
        best=None
        for j,c in nbr[i]:
            if j in leaders and j in idset and edge_accept(i,j,mode) and (best is None or c>best[1]): best=(j,c)
        if best: leader[i]=best[0]
        else: leaders.add(i); leader[i]=i
    return leader
def summarize(leader, ids):
    cl_={}
    for i in ids: cl_.setdefault(leader[i],[]).append(i)
    return cl_
def cluster_view(members,N=10**9):
    sup=set(); con=set()
    for i in members:
        sup|=set(cid[i]['support_ids']); con|=set(cid[i]['contradict_ids'])
    ev=[byid[s] for s in sup if s in byid and byid[s]['ver']<=N]
    return {'n_sup':len(ev),'n_con':len([c for c in con if c in byid and byid[c]['ver']<=N]),'probs':len({o['prob'] for o in ev}),'char_only':bool(ev) and all(o['phase']=='cha' for o in ev)}
if __name__=='__main__':
    ids=list(cid)
    out={}
    for mode in ('strict','single_pass','verified'):
        m={'strict':'strict','single_pass':'single','verified':'verified'}[mode]
        leader=cluster(ids,m); cs=summarize(leader,ids)
        bykind=collections.defaultdict(lambda:[0,0])
        for i in ids: bykind[cid[i]['kind']][0]+=1
        for L in cs: bykind[cid[L]['kind']][1]+=1
        views={L:cluster_view(mem) for L,mem in cs.items()}
        raw_views={i:cluster_view([i]) for i in ids}
        o={'claims':len(ids),'clusters':len(cs),'folded':len(ids)-len(cs),'fold_frac':round(1-len(cs)/len(ids),3),
           'by_kind':{k:{'n':v[0],'clusters':v[1]} for k,v in bykind.items()},
           'largest':sorted([len(v) for v in cs.values()],reverse=True)[:10],
           'sizes_hist':dict(collections.Counter(min(len(v),6) for v in cs.values())),
           'raw_ge2_problems':sum(v['probs']>=2 for v in raw_views.values()),'cur_ge2_problems':sum(v['probs']>=2 for v in views.values()),
           'raw_ge3_problems':sum(v['probs']>=3 for v in raw_views.values()),'cur_ge3_problems':sum(v['probs']>=3 for v in views.values()),
           'raw_single_problem':sum(v['probs']==1 for v in raw_views.values()),'cur_single_problem':sum(v['probs']==1 for v in views.values()),
           'cur_promotable(>=2 problems, non-char-only, net>0)':sum(v['probs']>=2 and not v['char_only'] and v['n_sup']>v['n_con'] for v in views.values()),
           'raw_promotable':sum(v['probs']>=2 and not v['char_only'] and v['n_sup']>v['n_con'] for v in raw_views.values()),
           'char_only_clusters':sum(v['char_only'] for v in views.values())}
        out[mode]=o; print(mode,json.dumps(o)[:1800])
        # contradictions among judged pairs
        contra=[(a,b) for (a,b) in cos if a in cid and b in cid and contradict(a,b)]
        out[mode]['contradict_pairs_judged']=len(contra)
        out[mode]['judged_pairs']=len({k for k in list(j1)})
    json.dump(out,open(BASE+'/results/G_curated_counts.json','w'),indent=1)
