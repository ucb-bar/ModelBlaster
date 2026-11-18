import sys, json
sys.path.insert(0,'/tmp/krisd/curator')
import llm
P=json.load(open('/tmp/krisd/curator/c_prompts.json'))
def job(p):
    t,_=llm.call(llm.O,"You are a careful curator of a GPU kernel optimization knowledge base. Answer with JSON only.",p['prompt'],4000 if p['key']=='tech' else 1500,'curate_'+p['key'])
    return {'raw':t}
res=llm.run_many([(p['key'],p) for p in P],job,6,'/tmp/krisd/curator/c_out.jsonl')
print(len(res),'usd',round(llm.usd(),2))
