python3 - <<'PY'
import json
P=dict(u=5.0,cr=0.5,cw=6.25,o=25.0)
n=0;usd=0;roles={}
for l in open('/tmp/krisd/farm/llm_ledger.jsonl'):
    d=json.loads(l); n+=d.get('calls',1)
    pt=d.get('prompt_tokens',0); cr=d.get('cache_read_input_tokens',0); cw=d.get('cache_write_input_tokens',0)
    u=max(pt-cr-cw,0)
    c=(u*P['u']+cr*P['cr']+cw*P['cw']+d.get('completion_tokens',0)*P['o'])/1e6
    usd+=c; r=d['what'].split('role=')[1].split()[0]; roles[r]=roles.get(r,0)+1
print("ledger calls",n,"usd %.2f"%usd,roles)
PY
