"""Enumerate candidate claim pairs (relaxed block + tf-idf cosine) for the LLM judge; writes work/cand_pairs.json."""
import sys, os, json, collections, itertools, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
sys.path.insert(0,"/Users/krisd/Documents/ModelBlaster-clone/ModelBlaster/third_party/KernelBlaster/src")
from kernelblaster.knowledge.v2.consolidate import TfIdf
cl_all = claims()
cl = [x for x in cl_all if x['status']=='active']
tf = TfIdf({x['id']:x['text'] for x in cl_all})
def polr(x):
    d=x['data'] or {}; k=x['kind']
    return str({'applicability':d.get('effect'),'interaction':(d.get('relation'),d.get('other')),'impl_note':d.get('note_type'),
          'mechanism':(d.get('metric'),d.get('verdict')),'state_split':d.get('proposed_state')}.get(k))
blocks=collections.defaultdict(list)
for x in cl: blocks[(x['kind'],x['op_class'],x['technique'],polr(x))].append(x['id'])
pairs=[]
for k,ids in blocks.items():
    for a,b in itertools.combinations(ids,2):
        c=tf.cos(a,b)
        if c>=0.15: pairs.append((a,b,round(c,3),k[0]))
print('blocks',len(blocks),'pairs>=.15',len(pairs))
bins=[(0.15,0.25),(0.25,0.35),(0.35,0.5),(0.5,0.7),(0.7,1.01)]
cnt=collections.Counter(); 
for a,b,c,k in pairs:
    for i,(lo,hi) in enumerate(bins):
        if lo<=c<hi: cnt[(i,k)]+=1
for i,b in enumerate(bins): print(b,{k:cnt[(i,k)] for k in ['impl_note','applicability','mechanism','interaction','state_split','pitfall'] },sum(v for (j,k),v in cnt.items() if j==i))
json.dump({'pairs':pairs,'bins':bins},open(BASE+'/work/cand_pairs.json','w'))
# per-claim best neighbour coverage
best=collections.defaultdict(float)
for a,b,c,k in pairs: best[a]=max(best[a],c); best[b]=max(best[b],c)
print('claims with >=1 candidate',len(best),'of',len(cl))
