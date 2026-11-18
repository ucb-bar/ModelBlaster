"""Build the curated store artifact (out/curated_store.json) + technique promotion table. Mode 'verified' = haiku pass1 AND pass2 AND opus agree."""
import sys, os, json, math, collections, statistics as S
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g_curate import *
MODE=sys.argv[1] if len(sys.argv)>1 else 'verified'
ids=list(cid); leader=cluster(ids,MODE); cs=summarize(leader,ids)
def wilson(k,n,z=1.96):
    if n==0: return (0,1)
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return ((c-h)/d,(c+h)/d)
def quote50(x):
    import re; m=re.findall(r'(\d+(?:\.\d+)?)\s*x\b',x['text']); return bool(m) and max(float(v) for v in m)>=50
claims_out=[]; cnt=collections.Counter()
for L,mem in cs.items():
    v=cluster_view(mem); x=cid[L]
    if v['char_only'] or (quote50(x) and v['probs']<2): status='quarantined_naive_seed'
    elif v['n_con']>=v['n_sup']: status='retired_contradicted'
    elif v['probs']>=2 and v['n_sup']>v['n_con']: status='promoted'
    else: status='candidate'
    cnt[status]+=1
    claims_out.append({'id':L,'kind':x['kind'],'op_class':x['op_class'],'technique':x['technique'],'status':status,'members':mem if len(mem)>1 else [],
                       'support':v['n_sup'],'contradict':v['n_con'],'problems':v['probs'],'text':x['text']})
# technique x op promotion table (non-char, non-infra, clipped pe), canonical technique names
tc=json.load(open(BASE+'/results/tech_canon.json'))
hyp={}
c=con()
for part,id_,st,d,pv in c.execute('select * from gk_hypotheses'): hyp[id_]=(json.loads(d).get('based_on') or [None])[0]
agg=collections.defaultdict(lambda:{'n':0,'k':0,'s':0.0,'p':set()}); base=collections.defaultdict(lambda:[0,0])
for o in O:
    if o['phase']=='cha' or o['infra'] or o['pe'] is None: continue
    t=o['tech']; t=hyp.get(t) or t if (t and t.startswith('h_')) else t; t=tc.get(t,t)
    a=agg[(t,o['op'])]; a['n']+=1; a['k']+=o['pe']>0.02; a['s']+=max(-1,min(1,o['pe'])); a['p'].add(o['prob'])
    b=base[o['op']]; b[0]+=1; b[1]+=o['pe']>0.02
tbl=[]
for (t,op),a in agg.items():
    lo,hi=wilson(a['k'],a['n']); b=base[op][1]/base[op][0]
    st='promote' if len(a['p'])>=2 and a['n']>=20 and lo>b else 'retire_candidate' if a['n']>=20 and hi<b else 'keep_exploring'
    tbl.append({'technique':t,'op_class':op,'n':a['n'],'problems':len(a['p']),'success':round(a['k']/a['n'],3),'ci':[round(lo,3),round(hi,3)],'op_baseline':round(b,3),'mean_pe':round(a['s']/a['n'],3),'decision':st})
dec=collections.Counter(r['decision'] for r in tbl)
sc=json.load(open(BASE+'/results/state_canon_opus.json'))
out={'mode':MODE,'counts':{'claims_in':len(ids),'claim_clusters':len(cs),'folded':len(ids)-len(cs),'status':dict(cnt),
     'multi_member_clusters':sum(len(m)>1 for m in cs.values()),'states_in':424,'states_after':424-len({k for k,v in sc.items() if k!=v['canonical']}),
     'techniques_in':34,'techniques_after':34-len(tc),'technique_op_decisions':dict(dec)},
     'claims':claims_out,'technique_table':sorted(tbl,key=lambda r:-r['n']),'state_map':{k:v['canonical'] for k,v in sc.items()},'technique_map':tc}
os.makedirs(BASE+'/out',exist_ok=True); json.dump(out,open(BASE+'/out/curated_store.json','w'))
print(json.dumps(out['counts'],indent=1))
print([r for r in out['technique_table'] if r['decision']=='promote'][:8])
