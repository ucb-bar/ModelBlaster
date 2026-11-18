WK=/tmp/krisd/family/pilot/$1
cat $WK/r00/problems/kb_problem__conv2d_f16_family/family.json | head -12
python3 -c "
import json;m=json.load(open('$WK/r00/problems/kb_problem__conv2d_f16_family/manifest.json'));print([(s['id'],len(s['instances']),s.get('seed_source')) for s in m['slots']])"
ls $WK/r00/run; tail -20 $WK/r00/run/run.log 2>/dev/null | cut -c1-250
