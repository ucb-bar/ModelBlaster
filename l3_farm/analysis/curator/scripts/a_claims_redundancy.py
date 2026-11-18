"""Part A: LLM-free redundancy of gk_claims. Run: PYTHONPATH=<KB src> python3 -I a_claims_redundancy.py"""
import sys, os, json, re, collections, itertools, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
sys.path.insert(0,"/Users/krisd/Documents/ModelBlaster-clone/ModelBlaster/third_party/KernelBlaster/src")
from kernelblaster.knowledge.claims import Claim, similar, normalise, _STOP
cl = [x for x in claims() if x['status']=='active']
print('active claims', len(cl))
def toks(text, mask=True):
    n = normalise(text)
    if mask: n = re.sub(r'\d+(?:\.\d+)?', '#', n)
    return set(w for w in n.split() if w not in ('#x','x#') )
def pol(x):
    return Claim.from_dict({k:v for k,v in x.items() if not k.startswith('_')}).dedupe_key()
for x in cl:
    x['_norm']=normalise(x['text']); x['_tok']=toks(x['text']); x['_key']=pol(x)
    d=x['data'] or {}
    k=x['kind']
    polr={'applicability':d.get('effect'),'interaction':(d.get('relation'),d.get('other')),'impl_note':d.get('note_type'),
          'mechanism':(d.get('metric'),d.get('verdict')),'state_split':d.get('proposed_state')}.get(k)
    x['_relkey']=(k,x['op_class'],x['technique'],str(polr))        # drops state/problem/slot/impl
    x['_relkey2']=(k,x['op_class'],x['technique'],str(polr),x['state'])
    x['_probs']=x['_prov']['problems']
def jac(a,b):
    return len(a&b)/len(a|b) if a|b else 0
class UF:
    def __init__(s,n): s.p=list(range(n))
    def f(s,i):
        while s.p[i]!=i: s.p[i]=s.p[s.p[i]]; i=s.p[i]
        return i
    def u(s,a,b): s.p[s.f(a)]=s.f(b)
res={}
# 1. actual mechanical rule: same dedupe_key and similar()
byk=collections.defaultdict(list)
for i,x in enumerate(cl): byk[x['_key']].append(i)
mech_pairs=0; mech_blocked_by_numbers=0; same_key_pairs=0
uf=UF(len(cl))
for k,ix in byk.items():
    for a,b in itertools.combinations(ix,2):
        same_key_pairs+=1
        A,B=cl[a],cl[b]
        if similar(A['_norm'],B['_norm']): mech_pairs+=1; uf.u(a,b)
        elif jac(A['_tok'],B['_tok'])>=0.8: mech_blocked_by_numbers+=1
res['mechanical']={'groups_by_dedupe_key':len(byk),'max_group':max(map(len,byk.values())),'pairs_same_key':same_key_pairs,
    'pairs_would_fold':mech_pairs,'pairs_blocked_only_by_numbers(jac>=.8 masked)':mech_blocked_by_numbers,
    'items_foldable_mech':len(cl)-len({uf.f(i) for i in range(len(cl))})}
print(res['mechanical'])
# 2. relaxed key blocking + masked jaccard thresholds
def cluster(keyf, thr, kinds=None):
    blocks=collections.defaultdict(list)
    for i,x in enumerate(cl):
        if kinds and x['kind'] not in kinds: continue
        blocks[x[keyf]].append(i)
    uf=UF(len(cl)); npairs=0; cand=[]
    for k,ix in blocks.items():
        for a,b in itertools.combinations(ix,2):
            j=jac(cl[a]['_tok'],cl[b]['_tok'])
            if j>=thr: uf.u(a,b); npairs+=1
    roots=collections.Counter(uf.f(i) for i in range(len(cl)) if not kinds or cl[i]['kind'] in kinds)
    return uf,roots,npairs
out={}
for keyf in ('_key','_relkey','_relkey2'):
    for thr in (0.5,0.6,0.7,0.8):
        uf,roots,npairs=cluster(keyf,thr)
        n=len(cl); ncl=len(roots); sizes=sorted(roots.values(),reverse=True)
        bykind=collections.defaultdict(lambda:[0,0])
        for i,x in enumerate(cl): bykind[x['kind']][0]+=1
        for r in roots:
            bykind[cl[r]['kind']][1]+=1
        out[f'{keyf}@{thr}']={'clusters':ncl,'foldable':n-ncl,'fold_frac':round((n-ncl)/n,3),'largest':sizes[:8],'pairs':npairs,
           'by_kind':{k:{'n':v[0],'clusters':v[1]} for k,v in bykind.items()},
           'cross_problem_clusters':sum(1 for r in roots if len({p for i in range(len(cl)) if uf.f(i)==r for p in cl[i]['_probs']})>=2) if keyf=='_relkey' and thr==0.6 else None}
        print(keyf,thr,out[f'{keyf}@{thr}']['clusters'],out[f'{keyf}@{thr}']['foldable'],sizes[:6])
res['relaxed']=out
# 3. support structure
sp=collections.Counter(len(x['_probs']) for x in cl)
res['support_problems_hist']=sorted(sp.items())
res['active_single_problem']=sum(1 for x in cl if len(x['_probs'])==1)
# impl_note with problem scope: share quoting >=50x
big=sum(1 for x in cl if re.search(r'(\d{2,}\.?\d*)\s*x\b',x['text']) and max(float(m) for m in re.findall(r'(\d+\.?\d*)\s*x\b',x['text']))>=50)
res['claims_quoting_ge50x']=big
res['n_active']=len(cl)
json.dump(res,open(BASE+'/results/A_claims.json','w'),indent=1)
