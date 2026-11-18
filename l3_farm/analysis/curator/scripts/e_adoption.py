"""Part E (no LLM): adoption lag of the 34 techniques and 19 core states (+ all active/probationary states)."""
import sys, os, json, datetime, collections, statistics as S, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
c=con()
iso=lambda t: datetime.datetime.fromisoformat(t).timestamp()
led=c.execute('select version,ts,problem,run_uid from gk_ledger order by version').fetchall()
vt=[(int(v[1:]),iso(t)) for v,t,*_ in led]; vtime=dict(vt)
def seedver_at(t): return max([v for v,tt in vt if tt<=t] or [0])
fj=json.load(open('/Users/krisd/Documents/ModelBlaster-clone/l3_farm/state/headline/farm.json'))['problems']
fj.update(json.load(open('/Users/krisd/Documents/ModelBlaster-clone/l3_farm/state/headline_heavy/farm.json'))['problems'])
fo={ru:iso(ts) for ru,ts in c.execute('select run_uid,min(ts) from gk_outcomes group by run_uid')}
run_problem={}; 
for v,t,p,ru in led: run_problem[ru]=p
runs={}   # run_uid -> dict(problem,start,seedver,auth)
for p,d in fj.items():
    g=d.get('gk') or {}
    if g.get('run_uid') and g.get('seeded'):
        runs[g['run_uid']]={'problem':p,'start':g['seeded'],'seedver':int(g['snapshot'][1:]) if g.get('snapshot') else 0,'auth':True}
for ru,p in run_problem.items():
    if ru not in runs and ru in fo:
        runs[ru]={'problem':p,'start':fo[ru]-150,'seedver':seedver_at(fo[ru]-150),'auth':False}
print('runs',len(runs),'authoritative',sum(r['auth'] for r in runs.values()))
# outcomes
out=[]
for id_,prob,ru,ts,rec in c.execute('select id,problem,run_uid,ts,rec from gk_outcomes'):
    r=json.loads(rec); out.append({'id':id_,'prob':prob,'ru':ru,'ts':iso(ts),'tech':r.get('technique'),'sb':r.get('state_before'),'sa':r.get('state_after'),
       'op':r.get('op_class'),'pe':r.get('partition_effect'),'st':r.get('status'),'pid':r.get('problem_id') or '','err':r.get('error') or ''})
def phase(o): return o['pid'].split('__')[-1][:3]
for o in out: o['phase']=phase(o); o['infra']=o['err'].startswith('[infra]')
clean=[o for o in out if o['phase']!='cha' and not o['infra'] and o['pe'] is not None]
base_all=collections.defaultdict(list)    # op_class -> pe list
for o in clean: base_all[o['op']].append(o['pe'])
def stats(L):
    L=[x for x in L if x is not None]
    return {'n':len(L),'impr':round(sum(1 for x in L if x>0.02)/len(L),3) if L else None,'mean':round(S.mean(L),4) if L else None}
def prob_of(o): return o['prob']
res={'techniques':[],'states':[]}
runlist=list(runs.items())
def adopt(kind,id_,first_merge,status,origin_problems,use_fn,opfilter=None):
    tM=vtime.get(first_merge)
    # runs that started after the item existed in a seed (start >= time of merge first_merge, and seedver >= first_merge)
    before=[(ru,r) for ru,r in runlist if r['seedver']<first_merge]
    after=[(ru,r) for ru,r in runlist if r['seedver']>=first_merge]
    probs_before={r['problem'] for _,r in before}; probs_after={r['problem'] for _,r in after}
    aft_runs={ru for ru,_ in after}
    # later outcomes using it: from runs seeded with it (other than origin problems), and all later outcomes
    used=[o for o in clean if use_fn(o) and o['ts']>=(tM or 0)]
    used_seeded=[o for o in used if o['ru'] in aft_runs and o['prob'] not in origin_problems]
    ops={o['op'] for o in used}
    ref=[o['pe'] for o in clean if o['op'] in ops and o['ts']>=(tM or 0) and not use_fn(o) and o['ru'] in aft_runs and o['prob'] not in origin_problems]
    row={'id':id_,'status':status,'first_merge':first_merge,'first_time':datetime.datetime.utcfromtimestamp(tM).strftime('%m-%d %H:%MZ') if tM else None,
         'origin_problems':sorted(origin_problems),'runs_started_before':len(before),'runs_started_after':len(after),
         'problems_never_saw_it':len(probs_before-probs_after) ,'outcomes_used_any_later':len(used),'outcomes_used_in_seeded_other_problems':len(used_seeded),
         'problems_using_it':len({o['prob'] for o in used_seeded}),'used':stats([o['pe'] for o in used_seeded]),'baseline_same_ops':stats(ref)}
    return row
for part,id_,st,d,pv in c.execute('select * from gk_techniques'):
    pv=json.loads(pv); d=json.loads(d)
    res['techniques'].append(adopt('tech',id_,pv['first_merge'],st,set(pbase(p) for p in pv['problems'][:1]),lambda o,i=id_:o['tech']==i))
for part,id_,st,d,pv in c.execute("select * from gk_states where status='core'"):
    pv=json.loads(pv); d=json.loads(d)
    res['states'].append(adopt('state',id_,pv['first_merge'],st,set(pbase(p) for p in pv['problems'][:1]),lambda o,i=id_:o['sb']==i))
json.dump(res,open(BASE+'/results/E_adoption.json','w'),indent=1)
def show(L):
    print('id | st | first_merge(t) | origin | runs before/after | outcomes any / seeded-other (probs) | used impr/mean | baseline impr/mean')
    for r in L: print(r['id'][:42],r['status'][:4],r['first_merge'],r['first_time'],r['origin_problems'][:1],f"{r['runs_started_before']}/{r['runs_started_after']}",
        f"{r['outcomes_used_any_later']}/{r['outcomes_used_in_seeded_other_problems']}({r['problems_using_it']})",r['used']['impr'],r['used']['mean'],'|',r['baseline_same_ops']['impr'],r['baseline_same_ops']['mean'])
show(res['techniques']); show(res['states'])

# ---- aggregate blindness: items created during a run's life that it could not import
last_out={}
for ru,t in c.execute('select run_uid,max(ts) from gk_outcomes group by run_uid'): last_out[ru]=iso(t)
items_t=[('tech',r['id'],vtime[r['first_merge']]) for r in res['techniques']]+[('state',r['id'],vtime[r['first_merge']]) for r in res['states']]
agg=[]
for ru,r in runs.items():
    end=last_out.get(ru)
    if not end: continue
    inlife=[x for x in items_t if r['start']<x[2]<=end]
    pre=[x for x in items_t if x[2]<=r['start']]
    agg.append({'problem':r['problem'],'run_hours':round((end-r['start'])/3600,2),'items_existing_at_seed':len(pre),'items_created_during_run(never visible)':len(inlife)})
tot_in=sum(a['items_created_during_run(never visible)'] for a in agg); tot_pre=sum(a['items_existing_at_seed'] for a in agg)
summ={'runs':len(agg),'mean_run_hours':round(S.mean(a['run_hours'] for a in agg),2),'items_total':len(items_t),
      'mean_items_visible_at_seed':round(tot_pre/len(agg),1),'mean_items_created_during_run_and_invisible':round(tot_in/len(agg),1),
      'share_of_(item,run)_pairs_invisible_among_items_existing_by_run_end':round(tot_in/(tot_in+tot_pre),3)}
summ['techniques_ever_used_in_a_problem_other_than_origin_after_seed']=sum(1 for r in res['techniques'] if r['problems_using_it']>0)
summ['core_states_ever_used_other_problem']=sum(1 for r in res['states'] if r['problems_using_it']>0)
summ['outcomes_using_named_techniques']=sum(r['outcomes_used_any_later'] for r in res['techniques'])
res['aggregate']=summ; res['per_run']=agg
print(summ)
json.dump(res,open(BASE+'/results/E_adoption.json','w'),indent=1)
