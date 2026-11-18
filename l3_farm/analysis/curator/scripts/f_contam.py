import sys, os, json, re, collections, statistics as S
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d_load import *
O=load_outcomes(); byid={o['id']:o for o in O}
res={}
res['outcomes']=len(O); res['infra']=sum(o['infra'] for o in O); res['char']=sum(o['phase']=='cha' for o in O)
res['char_with_pe']=sum(o['phase']=='cha' and o['pe'] is not None for o in O)
res['pe_null']=sum(o['pe'] is None for o in O); res['pe_lt_-1']=sum(o['pe'] is not None and o['pe']<-1 for o in O)
res['phase_counts']=dict(collections.Counter(o['phase'] for o in O))
cl=[x for x in claims() if x['status']=='active']
def ev(x): return [byid[i] for i in x['support_ids'] if i in byid]
charonly=[x for x in cl if ev(x) and all(o['phase']=='cha' for o in ev(x))]
anychar=[x for x in cl if any(o['phase']=='cha' for o in ev(x))]
infra=[x for x in cl if any(o['infra'] for o in ev(x))]
res['active_claims']=len(cl); res['claims_char_only_evidence']=len(charonly); res['claims_any_char_evidence']=len(anychar); res['claims_any_infra_evidence']=len(infra)
def big(x):
    m=re.findall(r'(\d+(?:\.\d+)?)\s*x\b',x['text']); return bool(m) and max(float(v) for v in m)>=50
res['claims_quote_ge50x']=sum(big(x) for x in cl)
res['claims_by_kind_char_only']=dict(collections.Counter(x['kind'] for x in charonly))
res['claims_single_problem']=sum(len(x['_prov']['problems'])==1 for x in cl)
res['claims_ge2_problems']=sum(len(x['_prov']['problems'])>=2 for x in cl); res['claims_ge3_problems']=sum(len(x['_prov']['problems'])>=3 for x in cl)
# contradicted active: contradict>=support
res['active_contradict_ge_support']=sum(x['contradict']>=x['support'] and x['contradict']>0 for x in cl)
# ranking distortion: top techniques per op_class raw vs filtered
def table(filt):
    agg=collections.defaultdict(lambda:[0,0.0])
    for o in O:
        if o['pe'] is None or not filt(o) : continue
        a=agg[(o['op'],o['tech'])]; a[0]+=1; a[1]+=max(-1,min(1,o['pe']))
    return agg
raw=table(lambda o:True); cur=table(lambda o:o['phase']!='cha' and not o['infra'])
rows=[]
for op in sorted({k[0] for k in raw}):
    r=sorted([(v[1]/v[0],k[1],v[0]) for k,v in raw.items() if k[0]==op and v[0]>=15],reverse=True)[:5]
    c_=sorted([(v[1]/v[0],k[1],v[0]) for k,v in cur.items() if k[0]==op and v[0]>=15],reverse=True)[:5]
    rows.append({'op':op,'raw_top5':[t for _,t,_ in r],'cur_top5':[t for _,t,_ in c_],'overlap':len({t for _,t,_ in r}&{t for _,t,_ in c_})})
res['top5_overlap_by_op']=rows
# mean effect of top tech
print(json.dumps(res,indent=1)[:3500])
json.dump(res,open(BASE+'/results/F_contamination.json','w'),indent=1)
