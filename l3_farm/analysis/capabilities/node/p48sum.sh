cd /tmp/krisd/a2/runs; for p in prec_*; do python3 -c "
import json,sys; r=json.load(open('$p/p48.json'))
for m,v in r.items(): print('$p', m, {k: v.get(k) for k in ('passed','e2e_us','dtype','slots_ok','error')}, (v.get('policy') or {}).get('applied'), (v.get('log_tail') or '')[-600:] if not v.get('passed') else '')
"; done
