import sys, os, json, datetime, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
iso=lambda t: datetime.datetime.fromisoformat(t).timestamp()
def load_outcomes():
    c=con()
    led=c.execute('select version,ts,run_uid from gk_ledger order by version').fetchall()
    byrun=collections.defaultdict(list)
    for v,ts,ru in led: byrun[ru].append((int(v[1:]),iso(ts)))
    out=[]
    for id_,prob,ru,ts,rec,v2 in c.execute('select id,problem,run_uid,ts,rec,v2 from gk_outcomes'):
        r=json.loads(rec); v=json.loads(v2); t=iso(ts)
        cand=[ver for ver,mt in byrun[ru] if mt>=t]
        ver=min(cand) if cand else max(x[0] for x in byrun[ru])
        pid=r.get('problem_id') or ''
        ph=pid.split('__')[-1][:3]
        err=r.get('error') or ''
        out.append({'id':id_,'prob':prob,'ru':ru,'ts':t,'ver':ver,'tech':r.get('technique'),'sb':r.get('state_before'),'sa':r.get('state_after'),
          'chain':v.get('chain') or [],'op':r.get('op_class'),'pe':r.get('partition_effect'),'st':r.get('status'),'pid':pid,'phase':ph,
          'infra':err.startswith('[infra]'),'err':err[:80],'parent':r.get('parent_impl_id'),'impl':r.get('impl_id'),'slot':r.get('slot'),
          'speedup':r.get('speedup'),'kind':v.get('kind'),'hyp':v.get('hypothesis_id')})
    return out
if __name__=='__main__':
    O=load_outcomes()
    print(len(O), collections.Counter(o['phase'] for o in O))
    print(collections.Counter(o['ver']>120 for o in O), collections.Counter(o['ver']>180 for o in O))
    cl=[o for o in O if o['phase']!='cha' and not o['infra'] and o['pe'] is not None]
    print(len(cl))
    g=collections.defaultdict(list)
    for o in cl: g[(o['pid'],o['parent'])].append(o)
    sz=collections.Counter(len({x['tech'] for x in v}) for v in g.values()); print(sorted(sz.items())[:10], len(g))
    g2=collections.defaultdict(list)
    for o in cl: g2[(o['pid'],o['parent'],o['ru'])].append(o)
    print(collections.Counter(len({x['tech'] for x in v}) for v in g2.values()).most_common(8))
    print(collections.Counter(o['ver']//40*40 for o in cl))
