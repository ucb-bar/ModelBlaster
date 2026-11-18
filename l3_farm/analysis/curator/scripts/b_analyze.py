import json, collections, random, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASE
import judge_lib as J
R=BASE+'/results/'
S=json.load(open(BASE+'/work/sample_pairs.json')); samp=S['sample']; strata=S['strata']
def load(f,mode):
    d={}
    for l in open(R+f):
        x=json.loads(l); k=x['k'].split('|',1)[1]; d[k]=x['v']['label']
    return d
j1,j2=load('j1.jsonl','1'),load('j2.jsonl','2')
jo=load('jo.jsonl','O') if os.path.exists(R+'jo.jsonl') else {}
key=lambda p:f"{p['a']}|{p['b']}"
fold=lambda l: l in J.FOLD
n=len(samp)
print('pairs',n,'j1 fold',sum(fold(j1.get(key(p))) for p in samp),'j2 fold',sum(fold(j2.get(key(p))) for p in samp))
agree=sum(fold(j1.get(key(p)))==fold(j2.get(key(p))) for p in samp)/n
exact=sum(j1.get(key(p))==j2.get(key(p)) for p in samp)/n
a=sum(fold(j1.get(key(p))) for p in samp)/n;b=sum(fold(j2.get(key(p))) for p in samp)/n
pe=agree; pc=a*b+(1-a)*(1-b); kappa=(pe-pc)/(1-pc)
print(f'fold-agreement {agree:.3f} exact-label agreement {exact:.3f} cohen kappa(fold) {kappa:.3f}')
# opus vs haiku
if jo:
    sub=[p for p in samp if key(p) in jo]
    for nm,j in (('j1',j1),('j2',j2)):
        tp=sum(fold(j[key(p)]) and fold(jo[key(p)]) for p in sub); fp=sum(fold(j[key(p)]) and not fold(jo[key(p)]) for p in sub); fn=sum((not fold(j[key(p)])) and fold(jo[key(p)]) for p in sub)
        print(nm,'vs opus on',len(sub),'agree',sum(fold(j[key(p)])==fold(jo[key(p)]) for p in sub)/len(sub),'tp',tp,'fp',fp,'fn',fn)
    print('opus fold rate on subset',sum(fold(jo[key(p)]) for p in sub)/len(sub),'haiku j1',sum(fold(j1[key(p)]) for p in sub)/len(sub),'j2',sum(fold(j2[key(p)]) for p in sub)/len(sub))
# stratified estimate (consensus rule: fold if both passes agree fold = 'strict', either = 'loose', j1 only)
def est(rule):
    byst=collections.defaultdict(list)
    for p in samp: byst[p['stratum']].append(1 if rule(p) else 0)
    tot=0; var=0; bybin=collections.defaultdict(lambda:[0,0.0]); bykind=collections.defaultdict(lambda:[0,0.0])
    for s,v in byst.items():
        N=strata[s]['N']; m=len(v); ph=sum(v)/m
        tot+=N*ph; var+=N*N*ph*(1-ph)/max(m-1,1)*(1-m/N)
        bi,k=s.split('|'); bybin[bi][0]+=N; bybin[bi][1]+=N*ph; bykind[k][0]+=N; bykind[k][1]+=N*ph
    return tot,math.sqrt(var),bybin,bykind
for nm,rule in (('j1',lambda p:fold(j1.get(key(p)))),('j2',lambda p:fold(j2.get(key(p)))),('strict(both)',lambda p:fold(j1.get(key(p))) and fold(j2.get(key(p)))),('loose(either)',lambda p:fold(j1.get(key(p))) or fold(j2.get(key(p))))):
    t,se,bb,bk=est(rule)
    print(f'{nm}: est foldable PAIRS among 14,518 candidates = {t:.0f} +- {1.96*se:.0f} (95%)')
    print('  by cos-bin (N, est fold pairs):',{k:(v[0],round(v[1])) for k,v in sorted(bb.items())})
    print('  by kind:',{k:(v[0],round(v[1])) for k,v in bk.items()})
# per-stratum fold rate
byst=collections.defaultdict(list)
for p in samp: byst[p['stratum']].append(fold(j1.get(key(p))) and fold(j2.get(key(p))))
print({s:(round(sum(v)/len(v),2),len(v)) for s,v in sorted(byst.items())})
print(collections.Counter((j1.get(key(p)),j2.get(key(p))) for p in samp).most_common(12))
print('--- opus gold')
t,se,bb,bk=est(lambda p:fold(jo.get(key(p))))
print(f'opus foldable PAIRS of 14,518: {t:.0f} +- {1.96*se:.0f}; by bin',{k:(v[0],round(v[1])) for k,v in sorted(bb.items())})
for nm,r in (('j1',lambda p:fold(j1.get(key(p)))),('strict',lambda p:fold(j1.get(key(p))) and fold(j2.get(key(p))))):
    tp=sum(r(p) and fold(jo[key(p)]) for p in samp); fp=sum(r(p) and not fold(jo[key(p)]) for p in samp); fn=sum((not r(p)) and fold(jo[key(p)]) for p in samp)
    print(nm,'vs opus: precision',round(tp/(tp+fp),3),'recall',round(tp/(tp+fn),3),'(unweighted, stratified sample)')
    # weighted precision
    w=collections.defaultdict(lambda:[0,0,0])
    for p in samp:
        w[p['stratum']][0]+=r(p) and fold(jo[key(p)]); w[p['stratum']][1]+=r(p) and not fold(jo[key(p)]); w[p['stratum']][2]+=(not r(p)) and fold(jo[key(p)])
    W=lambda i:sum(strata[s]['N']/strata[s]['n']*v[i] for s,v in w.items())
    print('   population-weighted precision',round(W(0)/(W(0)+W(1)),3),'recall',round(W(0)/(W(0)+W(2)),3))
