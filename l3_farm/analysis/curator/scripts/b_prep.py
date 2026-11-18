import sys, os, json, random, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
random.seed(7)
cl=claims(); act={x['id']:x for x in cl if x['status']=='active'}
mini={}
for x in cl:
    d=x['data'] or {}
    mini[x['id']]={'id':x['id'],'kind':x['kind'],'op_class':x['op_class'],'technique':x['technique'],'state':x['state'],
      'problem':pbase(x['problem_id']) if x['problem_id'] else None,'slot':x['slot'],'text':x['text'],'status':x['status'],
      'pol':{k:v for k,v in d.items() if k in ('effect','relation','other','note_type','metric','verdict','proposed_state')},
      'support':x['support'],'contradict':x['contradict'],'probs':x['_prov']['problems'],'first_merge':x['_prov'].get('first_merge'),
      'last_evidence':x['_prov'].get('last_evidence')}
json.dump(mini,open(BASE+'/work/claims_min.json','w'))
P=json.load(open(BASE+'/work/cand_pairs.json')); pairs=P['pairs']; bins=P['bins']
strata=collections.defaultdict(list)
for a,b,c,k in pairs:
    bi=[i for i,(lo,hi) in enumerate(bins) if lo<=c<hi][0]
    strata[(bi,k)].append((a,b,c))
quota={'impl_note':60,'applicability':30,'mechanism':15,'interaction':10,'state_split':10,'pitfall':10}
sample=[]; meta={}
for (bi,k),lst in sorted(strata.items()):
    n=min(len(lst),quota[k]); s=random.sample(lst,n)
    meta[f'{bi}|{k}']={'N':len(lst),'n':n}
    for a,b,c in s:
        if random.random()<.5: a,b=b,a
        sample.append({'a':a,'b':b,'cos':c,'stratum':f'{bi}|{k}'})
print(len(sample)); json.dump({'sample':sample,'strata':meta},open(BASE+'/work/sample_pairs.json','w'))
