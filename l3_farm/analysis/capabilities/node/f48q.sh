cd /tmp/krisd/a2/f48/work/r00; python3 -c "
import json,glob
p=json.load(open(glob.glob('profile/*/profile.json')[0])); print('profile passed',p['passed'],'partition_us',p['partition_us'],'e2e',p.get('e2e_us'))
for o in sorted(p['ops'],key=lambda o:-(o['us'] or 0))[:8]: print('  ',o['name'],o['op'],o['us'])
pl=json.load(open('plan.json')); print('plan', [(x.get('id'), len(x['op_names'])) for x in pl['problems']])
"; cat progress.json | head -c 600; ls run variants 2>/dev/null | head
