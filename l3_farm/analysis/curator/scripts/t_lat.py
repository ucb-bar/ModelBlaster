import sys,time; sys.path.insert(0,'/tmp/krisd/curator')
import llm
for m in (llm.H,):
    t0=time.time(); llm.call(m,'','Say {"ok":1}',20,'lat'); print(m,time.time()-t0)
import json
L=[json.loads(l) for l in open('/tmp/krisd/curator/ledger.jsonl')]
import collections
print([round(L[-1]['ts']-L[-i]['ts'],1) for i in (10,50,100)])
