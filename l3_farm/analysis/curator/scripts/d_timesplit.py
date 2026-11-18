"""Part D: offline time-split evaluation of raw vs curated recommendation statistics. Pure python."""
import sys, os, json, math, random, collections, itertools, statistics as S
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d_load import *
random.seed(11)
KAPPA=3.0
O=load_outcomes()
c=con()
# ---- maps (built only from info that exists at split N via first_merge)
st_first={}; st_pred={}
for part,id_,st,d,pv in c.execute('select * from gk_states'):
    d=json.loads(d); pv=json.loads(pv); st_first[id_]=pv.get('first_merge',0); st_pred[id_]=d
hyp_base={}
for part,id_,st,d,pv in c.execute('select * from gk_hypotheses'):
    d=json.loads(d); pv=json.loads(pv); hyp_base[id_]=(d.get('based_on') or [None])[0]; 
# state clusters (structural: same parent + same metric set with thresholds within 35%) -- reuse c_states_struct logic
exec(open(os.path.dirname(os.path.abspath(__file__))+'/state_cluster.py').read())
STATE_CANON=state_canon(st_pred, st_first)   # id -> canonical id (lowest first_merge)
TECH_CANON={}   # extra manual/LLM map loaded below if exists
tm=BASE+'/results/tech_canon.json'
if os.path.exists(tm): TECH_CANON=json.load(open(tm))
def mapT(o, level):
    t=o['tech']
    if level>=3:
        if t and t.startswith('h_'): t=hyp_base.get(t) or t
        t=TECH_CANON.get(t,t)
    return t
def mapS(s, N, level):
    if level>=2 and s in STATE_CANON and st_first.get(s,0)<=N and st_first.get(STATE_CANON[s],0)<=N: return STATE_CANON[s]
    return s
ARMS={'R0_raw':dict(drop_char=False,state=0,tech=0,minprob=1,wins=False),
      'C0_raw+clip':dict(drop_char=False,state=0,tech=0,minprob=1,wins=True),
      'C1_filter_char_infra':dict(drop_char=True,state=0,tech=0,minprob=1,wins=True),
      'C2_+state_canon':dict(drop_char=True,state=2,tech=0,minprob=1,wins=True),
      'C3_+tech_canon':dict(drop_char=True,state=2,tech=3,minprob=1,wins=True),
      'C4_+2problem_rule':dict(drop_char=True,state=2,tech=3,minprob=2,wins=True)}
def build(train, N, cfg):
    agg=collections.defaultdict(lambda: collections.defaultdict(lambda:[0,0.0]))  # node -> problem -> [n,s]
    for o in train:
        if o['pe'] is None or o['infra']: continue
        if cfg['drop_char'] and o['phase']=='cha': continue
        t=mapT(o,cfg['tech']); pe=max(-1.0,min(1.0,o['pe'])) if cfg['wins'] else o['pe']
        chain=[mapS(s,N,cfg['state']) for s in (o['chain'] or [o['sb']]) if s]
        nodes=[(t,'*','*'),(t,o['op'],'*')]+[(t,o['op'],s) for s in dict.fromkeys(chain)]
        for nd in nodes:
            a=agg[nd][o['prob']]; a[0]+=1; a[1]+=pe
    return agg
def est(agg, t, op, chain, excl, minprob):
    def node(nd):
        d=agg.get(nd)
        if not d: return 0,0.0
        n=s=0; pr=0
        for p,(a,b) in d.items():
            if p==excl: continue
            n+=a; s+=b; pr+=1
        if pr<minprob: return 0,0.0
        return n,s
    e=0.0
    n,s=node((t,'*','*')); e=(s+KAPPA*e)/(n+KAPPA)
    n,s=node((t,op,'*')); e=(s+KAPPA*e)/(n+KAPPA)
    for st in chain:       # root->leaf
        n,s=node((t,op,st)); e=(s+KAPPA*e)/(n+KAPPA)
    return e
def evaluate(N, name, cfg, cross=True):
    train=[o for o in O if o['ver']<=N]
    test=[o for o in O if o['ver']>N and o['phase']!='cha' and not o['infra'] and o['pe'] is not None]
    agg=build(train,N,cfg)
    # groups
    G=collections.defaultdict(list)
    for o in test: G[(o['pid'],o['parent'],o['ru'])].append(o)
    res={'groups':[], 'rows':[]}
    for (pid,par,ru),rows in G.items():
        r0=rows[0]; prob=r0['prob']
        chain=[mapS(s,N,cfg['state']) for s in (r0['chain'] or [r0['sb']]) if s]
        bytech=collections.defaultdict(list)
        for o in rows: bytech[mapT(o,3 if cfg['tech']>=3 else 0) if False else o['tech']].append(max(-1,min(1,o['pe'])))
        if len(bytech)>=2:
            sc={}
            for t,v in bytech.items():
                tt=t
                if cfg['tech']>=3:
                    tt=hyp_base.get(t) or t if (t and t.startswith('h_')) else t
                    tt=TECH_CANON.get(tt,tt)
                sc[t]=est(agg,tt,r0['op'],list(dict.fromkeys(chain)),prob if cross else None,cfg['minprob'])
            m={t:S.mean(v) for t,v in bytech.items()}
            top=max(sc.values()); tops=[t for t in sc if abs(sc[t]-top)<1e-12]
            gain=S.mean(m[t] for t in tops)-S.mean(m.values())
            best=max(m.values()); bests=[t for t in m if m[t]==best]
            hit=len(set(tops)&set(bests))/len(tops)
            res['groups'].append((prob,gain,hit-len(bests)/len(m)*1.0 if False else hit, 1/len(m) if len(bests)==1 else len(bests)/len(m), S.mean(m.values()), len(m)))
        for o in rows:
            t=o['tech']; tt=t
            if cfg['tech']>=3:
                tt=hyp_base.get(t) or t if (t and t.startswith('h_')) else t
                tt=TECH_CANON.get(tt,tt)
            res['rows'].append((prob, est(agg,tt,o['op'],list(dict.fromkeys([mapS(s,N,cfg['state']) for s in (o['chain'] or [o['sb']]) if s])),prob if cross else None,cfg['minprob']),
                                max(-1,min(1,o['pe'])), 1 if o['pe']>0.02 else 0))
    return res
def auc(pairs):
    pos=[s for s,y in pairs if y]; neg=[s for s,y in pairs if not y]
    if not pos or not neg: return float('nan')
    r=sorted([(s,y) for s,y in pairs]); 
    # rank-sum with ties
    ranks={}; i=0; allv=sorted(s for s,_ in pairs)
    from bisect import bisect_left,bisect_right
    tot=0.0
    for s in pos: tot+=bisect_left(allv,s)+ (bisect_right(allv,s)-bisect_left(allv,s)+1)/2 - 0  # 1-based avg rank
    ranksum=tot
    return (ranksum-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
def spearman(x,y):
    def rk(v):
        o=sorted(range(len(v)),key=lambda i:v[i]); r=[0]*len(v); i=0
        while i<len(o):
            j=i
            while j+1<len(o) and v[o[j+1]]==v[o[i]]: j+=1
            for k in range(i,j+1): r[o[k]]=(i+j)/2
            i=j+1
        return r
    a,b=rk(x),rk(y); ma,mb=S.mean(a),S.mean(b)
    num=sum((p-ma)*(q-mb) for p,q in zip(a,b)); den=math.sqrt(sum((p-ma)**2 for p in a)*sum((q-mb)**2 for q in b))
    return num/den if den else float('nan')
def boot(res, B=1000):
    """cluster bootstrap over test problems: group gain, hit-excess, row AUC."""
    probs=sorted({g[0] for g in res['groups']}|{r[0] for r in res['rows']})
    gb=collections.defaultdict(list); rb=collections.defaultdict(list)
    for g in res['groups']: gb[g[0]].append(g)
    for r in res['rows']: rb[r[0]].append(r)
    def stat(sample):
        G=[g for p in sample for g in gb.get(p,[])]; R=[r for p in sample for r in rb.get(p,[])]
        gain=S.mean(g[1] for g in G) if G else float('nan')
        hit=S.mean(g[2]-g[3] for g in G) if G else float('nan')
        a=auc([(r[1],r[3]) for r in R])
        return gain,hit,a
    return probs,stat
def run(N, cross=True):
    out={}
    results={name:evaluate(N,name,cfg,cross) for name,cfg in ARMS.items()}
    probs,_=boot(results['R0_raw'])
    # shared bootstrap samples across arms for paired differences
    samples=[[random.choice(probs) for _ in probs] for _ in range(1000)]
    stats={}
    for name,res in results.items():
        _,stat=boot(res)
        pt=stat(probs)
        bs=[stat(s) for s in samples]
        stats[name]=(pt,bs)
    table={}
    for name,(pt,bs) in stats.items():
        row={'n_groups':len(results[name]['groups']),'n_rows':len(results[name]['rows'])}
        for i,m in enumerate(('gain_vs_random','hit_excess','auc')):
            vals=sorted(b[i] for b in bs if not math.isnan(b[i]))
            row[m]=(round(pt[i],4),round(vals[int(.025*len(vals))],4),round(vals[int(.975*len(vals))],4))
            if name!='R0_raw':
                d=sorted(b[i]-b0[i] for b,b0 in zip(bs,stats['R0_raw'][1]) if not math.isnan(b[i]) and not math.isnan(b0[i]))
                row[m+'_diff_vs_raw']=(round(pt[i]-stats['R0_raw'][0][i],4),round(d[int(.025*len(d))],4),round(d[int(.975*len(d))],4))
        rr=results[name]['rows']; row['spearman']=round(spearman([r[1] for r in rr],[r[2] for r in rr]),4)
        row['oracle_gain']=None
        table[name]=row
    # oracle & random reference
    g=results['R0_raw']['groups']
    table['_meta']={'N':N,'cross_problem_only':cross,'test_problems':len(probs),'mean_group_best_minus_mean':None}
    return table
if __name__=='__main__':
    allres={}
    for N in (80,120,180):
        for cross in (True,):
            t=run(N,cross); allres[f'N{N}_{"cross" if cross else "all"}']=t
            print('N',N,'cross',cross,t['_meta'])
            for k,v in t.items():
                if k!='_meta': print(' ',k,v['n_groups'],v['n_rows'],'gain',v['gain_vs_random'],v.get('gain_vs_random_diff_vs_raw'),'| hit',v['hit_excess'],'| auc',v['auc'],v.get('auc_diff_vs_raw'),'| rho',v['spearman'])
    json.dump(allres,open(BASE+'/results/D_timesplit.json','w'),indent=1)
