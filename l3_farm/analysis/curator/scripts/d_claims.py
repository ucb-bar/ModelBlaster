"""Part D2: do applicability claims (helps/hurts for technique x op_class x state) predict LATER outcomes? raw vs promotion-filtered."""
import sys, os, json, math, random, collections, statistics as S
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d_load import *
random.seed(5)
O=load_outcomes(); byid={o['id']:o for o in O}
cl=[x for x in claims() if x['kind']=='applicability' and (x['data'] or {}).get('effect') in ('helps','hurts')]
print('applicability helps/hurts claims',len(cl))
def success(o): return 1 if o['pe']>0.02 else 0
def view(x,N):
    sup=[byid[i] for i in x['support_ids'] if i in byid and byid[i]['ver']<=N]
    con=[byid[i] for i in x['contradict_ids'] if i in byid and byid[i]['ver']<=N]
    if not sup: return None
    return {'x':x,'eff':x['data']['effect'],'tech':x['technique'],'op':x['op_class'],'state':x['state'],'n_sup':len(sup),'n_con':len(con),
            'probs':len({o['prob'] for o in sup}),'char_only':all(o['phase']=='cha' for o in sup),'infra':any(o['infra'] for o in sup)}
def run(N):
    V=[v for v in (view(x,N) for x in cl) if v]
    test=[o for o in O if o['ver']>N and o['phase']!='cha' and not o['infra'] and o['pe'] is not None]
    sets={'RAW(all claims existing at N)':lambda v:True,
          'CUR(>=2 problems, not char-only, net>0)':lambda v:v['probs']>=2 and not v['char_only'] and v['n_sup']>v['n_con'],
          'CUR3(>=3 problems, not char-only, net>0)':lambda v:v['probs']>=3 and not v['char_only'] and v['n_sup']>v['n_con']}
    out={}
    for nm,f in sets.items():
        sel=[v for v in V if f(v)]
        idx=collections.defaultdict(list)
        for v in sel: idx[(v['tech'],v['op'])].append(v)
        rows=[]   # (prob, predicted_helps(+1)/hurts(-1)/0, success)
        for o in test:
            ms=[v for v in idx.get((o['tech'],o['op']),[]) if v['state'] is None or v['state'] in (o['chain'] or [o['sb']])]
            # exclude claims whose only evidence is the test problem itself (same-problem leakage)
            ms=[v for v in ms if not (v['probs']==1 and byid[v['x']['support_ids'][0]]['prob']==o['prob'])]
            h=sum(1 for v in ms if v['eff']=='helps'); u=sum(1 for v in ms if v['eff']=='hurts')
            pred=0 if (h==0 and u==0) else (1 if h>u else -1 if u>h else 0)
            rows.append((o['prob'],pred,success(o),o['pe']))
        out[nm]=(len(sel),rows)
    return out
def summarize(rows,B=1000):
    probs=sorted({r[0] for r in rows}); by=collections.defaultdict(list)
    for r in rows: by[r[0]].append(r)
    def stat(sample):
        R=[r for p in sample for r in by[p]]
        hp=[r for r in R if r[1]==1]; hu=[r for r in R if r[1]==-1]; n0=[r for r in R if r[1]==0]
        base=S.mean(r[2] for r in R)
        ph=S.mean(r[2] for r in hp) if hp else float('nan'); pu=S.mean(r[2] for r in hu) if hu else float('nan')
        ph_pe=S.mean(r[3] for r in hp) if hp else float('nan')
        return base,ph,pu,len(hp),len(hu),ph_pe,S.mean(r[3] for r in R)
    pt=stat(probs); bs=[stat([random.choice(probs) for _ in probs]) for _ in range(B)]
    def ci(i):
        v=sorted(b[i] for b in bs if not math.isnan(b[i])); return (round(v[int(.025*len(v))],3),round(v[int(.975*len(v))],3)) if v else None
    return {'base_success':round(pt[0],3),'success_when_helps_predicted':(round(pt[1],3),ci(1)),'success_when_hurts_predicted':(round(pt[2],3),ci(2)),'n_helps_pred':pt[3],'n_hurts_pred':pt[4],
            'lift_helps_vs_base':(round(pt[1]-pt[0],3),),'mean_pe_helps_pred':round(pt[5],3),'mean_pe_all':round(pt[6],3),'n_test':len(rows)}
allr={}
for N in (80,120,180):
    out=run(N); allr[N]={}
    for nm,(nsel,rows) in out.items():
        s=summarize(rows); s['n_claims']=nsel; allr[N][nm]=s; print(N,nm,s)
json.dump({str(k):v for k,v in allr.items()},open(BASE+'/results/D_claims.json','w'),indent=1)
