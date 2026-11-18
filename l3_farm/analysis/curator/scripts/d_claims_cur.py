"""D2': claim-level promotion test, raw (unfolded) vs LLM-folded clusters, time-split."""
import sys, os, json, math, random, collections, statistics as S
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g_curate import *
random.seed(9)
app=[x for x in cl if x['kind']=='applicability' and (x['data'] or {}).get('effect') in ('helps','hurts')]
def success(o): return 1 if o['pe']>0.02 else 0
def members_view(members,N):
    v=cluster_view(members,N); 
    xs=[cid[i] for i in members]
    v.update(eff=xs[0]['data']['effect'],tech=xs[0]['technique'],op=xs[0]['op_class'],states={x['state'] for x in xs},
             origin={byid[s]['prob'] for x in xs for s in x['support_ids'] if s in byid and byid[s]['ver']<=N})
    return v
def run(N, mode='verified'):
    ids=[x['id'] for x in app if any(byid[s]['ver']<=N for s in x['support_ids'] if s in byid)]
    leader=cluster(ids,mode); cs=summarize(leader,ids)
    CV=[members_view(m,N) for m in cs.values()]
    RV=[members_view([i],N) for i in ids]
    test=[o for o in O if o['ver']>N and o['phase']!='cha' and not o['infra'] and o['pe'] is not None]
    rules={'RAW all claims':(RV,lambda v:True),
           'RAW promoted (claim >=2 problems, non-char, net>0)':(RV,lambda v:v['probs']>=2 and not v['char_only'] and v['n_sup']>v['n_con']),
           'FOLDED all clusters':(CV,lambda v:True),
           'FOLDED promoted (cluster >=2 problems, non-char, net>0)':(CV,lambda v:v['probs']>=2 and not v['char_only'] and v['n_sup']>v['n_con']),
           'FOLDED promoted >=3 problems':(CV,lambda v:v['probs']>=3 and not v['char_only'] and v['n_sup']>v['n_con'])}
    res={}
    for nm,(V,f) in rules.items():
        sel=[v for v in V if f(v)]
        idx=collections.defaultdict(list)
        for v in sel: idx[(v['tech'],v['op'])].append(v)
        rows=[]
        for o in test:
            ms=[v for v in idx.get((o['tech'],o['op']),[]) if (None in v['states'] or set(o['chain'] or [o['sb']])&v['states']) and not (v['origin']=={o['prob']})]
            h=sum(1 for v in ms if v['eff']=='helps'); u=sum(1 for v in ms if v['eff']=='hurts')
            pred=0 if h==u else (1 if h>u else -1)
            rows.append((o['prob'],pred,success(o),max(-1,min(1,o['pe']))))
        res[nm]=(len(sel),rows)
    return res
def summarize_rows(rows,B=1000):
    probs=sorted({r[0] for r in rows}); by=collections.defaultdict(list)
    for r in rows: by[r[0]].append(r)
    def stat(sample):
        R_=[r for p in sample for r in by[p]]
        hp=[r for r in R_ if r[1]==1]; hu=[r for r in R_ if r[1]==-1]
        base=S.mean(r[2] for r in R_)
        return (S.mean(r[2] for r in hp)-base if hp else float('nan'), S.mean(r[2] for r in hu)-base if hu else float('nan'),len(hp),len(hu),
                S.mean(r[3] for r in hp)-S.mean(r[3] for r in R_) if hp else float('nan'))
    pt=stat(probs); bs=[stat([random.choice(probs) for _ in probs]) for _ in range(B)]
    def ci(i):
        v=sorted(b[i] for b in bs if not math.isnan(b[i])); return [round(v[int(.025*len(v))],3),round(v[int(.975*len(v))],3)] if v else None
    return {'lift_success_helps':[round(pt[0],3),ci(0)],'lift_success_hurts':[round(pt[1],3),ci(1)],'n_helps_rows':pt[2],'n_hurts_rows':pt[3],
            'lift_pe_helps':[round(pt[4],3),ci(4)],'n_test':len(rows)}
allr={}
for N in (80,120,180):
    out=run(N); allr[N]={}
    for nm,(ns,rows) in out.items():
        s=summarize_rows(rows); s['n_units']=ns; allr[N][nm]=s; print(N,nm,s)
json.dump({str(k):v for k,v in allr.items()},open(BASE+'/results/D_claims_folded.json','w'),indent=1)
