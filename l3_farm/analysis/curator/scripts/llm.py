"""Tiny relay client (runs ON the driver node). Reads KB_LLM_SERVER_URL / TOKEN_FILE from the environment (never prints them).
Thread pool <= 12, jsonl cache keyed by sha1(model+prompt), usage/cost ledger."""
import os, json, hashlib, time, threading, urllib.request, urllib.error, uuid, socket
from concurrent.futures import ThreadPoolExecutor
URL=os.environ['KB_LLM_SERVER_URL'].rstrip('/')
TOK=open(os.path.expanduser(os.environ['KB_LLM_SERVER_TOKEN_FILE'])).read().strip()
H='azure/anthropic/claude-haiku-5-5'; O='azure/anthropic/claude-opus-5-5'
LEDGER='/tmp/krisd/curator/ledger.jsonl'
_lock=threading.Lock()
def _post(model, messages, max_tokens):
    payload={"messages":messages,"n_tasks":1,"model":model,"role":"curator_offline","client":"curator:"+socket.gethostname(),
             "run_id":"curator","request_id":uuid.uuid4().hex}
    if max_tokens: payload["max_tokens"]=max_tokens
    last=None
    for i in range(6):
        try:
            req=urllib.request.Request(URL+"/v1/generate",json.dumps(payload).encode(),{"Authorization":"Bearer "+TOK,"Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=600) as r: return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body=e.read()[:300].decode('utf8','ignore')
            if e.code in (400,401,402,403,409,413,422): raise RuntimeError(f'HTTP {e.code} {body}')
            last=f'HTTP {e.code} {body}'
        except Exception as e: last=repr(e)
        time.sleep(min(2**i,30))
    raise RuntimeError(last)
def call(model, system, user, max_tokens=400, tag=''):
    msgs=([{"role":"system","content":system}] if system else [])+[{"role":"user","content":user}]
    body=_post(model,msgs,max_tokens)
    u=body.get('usage') or {}
    rec={'ts':time.time(),'model':model,'tag':tag,'in':u.get('input_tokens') or u.get('prompt_tokens') or 0,'out':u.get('output_tokens') or u.get('completion_tokens') or 0,
         'est_usd':u.get('est_usd') or u.get('cost_usd') or (body.get('server') or {}).get('est_usd')}
    with _lock:
        with open(LEDGER,'a') as f: f.write(json.dumps(rec)+'\n')
    g=body.get('generations') or ['']
    return g[0], rec
def run_many(items, fn, workers=12, cache=None):
    """items: list of (key, args). fn(args)->result (json-able). cache: jsonl path; resumes."""
    done={}
    if cache and os.path.exists(cache):
        for l in open(cache):
            try: d=json.loads(l); done[d['k']]=d['v']
            except Exception: pass
    todo=[(k,a) for k,a in items if k not in done]
    f=open(cache,'a') if cache else None
    def w(ka):
        k,a=ka
        try: v=fn(a)
        except Exception as e: v={'error':repr(e)[:200]}
        with _lock:
            if f and 'error' not in v: f.write(json.dumps({'k':k,'v':v})+'\n'); f.flush()
        return k,v
    with ThreadPoolExecutor(workers) as ex:
        for k,v in ex.map(w,todo): done[k]=v
    if f: f.close()
    return done
def spent():
    tot={}
    if os.path.exists(LEDGER):
        for l in open(LEDGER):
            r=json.loads(l); t=tot.setdefault(r['model'],[0,0,0,0.0]); t[0]+=1; t[1]+=r['in']; t[2]+=r['out']; t[3]+=r['est_usd'] or 0
    return tot
PRICE={'haiku':(1.0,5.0),'opus':(5.0,25.0)}
def usd():
    t=spent(); s=0
    for m,(n,i,o,_) in t.items():
        p=PRICE['haiku' if 'haiku' in m else 'opus']; s+=i/1e6*p[0]+o/1e6*p[1]
    return s
