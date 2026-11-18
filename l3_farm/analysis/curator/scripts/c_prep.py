import sys, json, collections, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d_load import *
exec(open(os.path.dirname(os.path.abspath(__file__))+'/state_cluster.py').read())
O=load_outcomes(); c=con()
tu=collections.defaultdict(lambda:[0,0.0,set(),0]); su=collections.defaultdict(lambda:[0,0.0,set()])
for o in O:
    if o['pe'] is None or o['infra'] or o['phase']=='cha': continue
    a=tu[o['tech']]; a[0]+=1; a[1]+=max(-1,min(1,o['pe'])); a[2].add(o['prob']); a[3]+= o['pe']>0.02
    if o['sb']: b=su[o['sb']]; b[0]+=1; b[1]+=max(-1,min(1,o['pe'])); b[2].add(o['prob'])
P=[]
T=[]
for part,id_,st,d,pv in c.execute('select * from gk_techniques'):
    d=json.loads(d); pv=json.loads(pv)
    t=tu.get(id_,[0,0,set(),0])
    T.append({'id':id_,'status':st,'op_classes':d.get('op_classes'),'mechanism':(d.get('mechanism') or '')[:350],'first_merge':pv.get('first_merge'),
              'origin_problems':[pbase(p) for p in pv.get('problems',[])][:3],'outcomes_nonchar':t[0],'success_rate':round(t[3]/t[0],2) if t[0] else None,'mean_pe':round(t[1]/t[0],3) if t[0] else None,'problems_used':len(t[2])})
tp=("Below are 34 optimization-technique entries discovered by an LLM kernel-optimization system (status = active|retired). Many may be near-duplicates "
    "(same idea, different name). Propose a curation: groups of entries that are the SAME technique (give a canonical id from the members and a one-sentence merged description), "
    "entries that are distinct, and entries to retire (contradicted or no evidence). Be conservative: merge only if the mechanism and applicable op class are the same.\n"
    "Return JSON only: {\"groups\":[{\"canonical\":id,\"members\":[ids],\"description\":\"...\"}],\"retire\":[{\"id\":..,\"why\":\"...\"}]}\n\nENTRIES:\n"+json.dumps(T,indent=0))
prompts=[{'key':'tech','model':'O','prompt':tp}]
# states
Pd={};F={};ST={}
for part,id_,st,d,pv in c.execute('select * from gk_states'):
    d=json.loads(d); pv=json.loads(pv); Pd[id_]=d;F[id_]=pv.get('first_merge',0);ST[id_]=(st,len(pv.get('problems',[])))
canon=state_canon(Pd,F)
g=collections.defaultdict(list)
for i,r in canon.items(): g[r].append(i)
big=sorted([v for v in g.values() if len(v)>=2],key=lambda v:-len(v))
for k,v in enumerate(big):
    ent=[{'id':i,'status':ST[i][0],'parent':Pd[i].get('parent'),'predicate':Pd[i]['predicate'],'mechanism':(Pd[i].get('mechanism') or '')[:200],
          'outcomes_using':su[i][0] if i in su else 0,'mean_pe':round(su[i][1]/su[i][0],3) if i in su and su[i][0] else None} for i in v]
    sp=("These discovered bottleneck-state entries (metric-predicate classes of a kernel profile) were grouped by a structural pass (same parent, same metrics, similar thresholds). "
        "Decide which are the SAME state (a merge would not lose a meaningful distinction) and propose one canonical predicate for each merged group (round thresholds sensibly); "
        "list members that must stay separate.\nReturn JSON only: {\"groups\":[{\"canonical\":id,\"members\":[ids],\"predicate\":{...},\"name\":\"short\"}],\"separate\":[ids]}\n\nENTRIES:\n"+json.dumps(ent))
    prompts.append({'key':f'state{k:03d}','model':'O','prompt':sp,'members':v})
json.dump(prompts,open(BASE+'/work/c_prompts.json','w'))
print(len(prompts),sum(len(p['prompt']) for p in prompts)/4,'tokens approx')
json.dump({'T':T},open(BASE+'/work/c_tech_info.json','w'))
