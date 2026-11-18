#!/usr/bin/env python3
"""inspect_node.py JOB MODE NAME : facts of a problem dir on one of my nodes (no secrets)."""
import json, os, sys
os.environ["FARM_CONFIG"] = f"analysis/drill_data/config_drill_{sys.argv[2]}.json"
sys.path.insert(0, "/Users/krisd/Documents/ModelBlaster-clone/l3_farm")
import farm
job, mode, name = int(sys.argv[1]), sys.argv[2], sys.argv[3]
assert str(job) in json.load(open("/Users/krisd/Documents/ModelBlaster-clone/l3_farm/analysis/drill_data/my_jobs.json"))
D = f"/tmp/krisd/fdrill_{mode}/{name}"
sc = f"""cd {D} || exit 1
echo "status=$(cat status)"; echo "job.env: $(tr '\\n' ' ' < job.env)"; echo "job.env sha=$(sha256sum job.env | cut -c1-16)"
echo "cache files: $(find work/cache -type f 2>/dev/null | wc -l)"; ls work 2>/dev/null | tr '\\n' ' '; echo
python3 - <<'PY'
import json
s=json.load(open('work/state.json')); print('rounds', [(r['round'], r['key'], r.get('partition_us')) for r in s['rounds']])
PY
grep -h "dropped partial" run.log worker.out 2>/dev/null | head; tail -n 3 run.log | cut -c1-200
"""
rc, out, err = farm.ssh(f"srun --jobid={job} --overlap bash -s", sc.encode())
print(out.decode(), err.decode()[-300:])
