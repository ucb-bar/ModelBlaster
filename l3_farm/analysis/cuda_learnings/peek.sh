cd /tmp/krisd/cl/runs
for d in *; do echo "== $d"; python3 -c "
import json,sys
r=json.load(open('$d/reuse/reuse.json'))
print('changed',r.get('changed'),'reverted',r.get('reverted'),'evals',r.get('evals'))
for k in r['kinds']:
    print(' ', k['kind'], k.get('share'), 'pool', len(k.get('pool',[])), k.get('skipped') or k.get('result') or k.get('error'), k.get('before_us'), k.get('after_us'))
    for kk,row in (k.get('table') or {}).items(): print('    ', kk, row['shape'].get('I'), row['shape'].get('reverse'), 'inc', row['incumbent'], {c:(v['ok'],v['us'] and round(v['us'],1)) for c,v in row['cands'].items()})
print(' guard', r.get('guard'))
" 2>&1 | head -40; done
