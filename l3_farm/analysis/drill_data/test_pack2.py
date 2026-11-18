#!/usr/bin/env python3
"""X-128: does pack2.sh carry work/cache (the winners cache)? Builds a fake problem dir on one of my nodes under
/tmp/krisd/fdrill_packtest and lists the pack made by the pre-drill pack2.sh and by the current one."""
import io, json, os, sys, tarfile
os.environ["FARM_CONFIG"] = "analysis/drill_data/config_drill_ind.json"
sys.path.insert(0, "/Users/krisd/Documents/ModelBlaster-clone/l3_farm")
import farm
job = int(sys.argv[1])
mine = json.load(open("/Users/krisd/Documents/ModelBlaster-clone/l3_farm/analysis/drill_data/my_jobs.json"))
assert str(job) in mine
setup = """
set -e
B=/tmp/krisd/fdrill_packtest; rm -rf $B; mkdir -p $B/p1/work/cache/entries $B/p1/work/knowledge $B/p1/work/r00
echo '{}' > $B/p1/work/cache/index.json; echo 'k' > $B/p1/work/cache/entries/k.cu
echo '{"rounds":[1]}' > $B/p1/work/state.json; echo running > $B/p1/status
python3 -c "import sqlite3;c=sqlite3.connect('$B/p1/work/knowledge/knowledge.db');c.execute('create table t(a)');c.commit()"
"""
farm.ssh(f"srun --jobid={job} --overlap bash -s", setup.encode())
for label, script in (("pre-drill pack2.sh", "../state/pack2.sh.pre-drill.bak"), ("current pack2.sh", "pack2.sh")):
    rc, data, err = farm.srun(job, script, ["p1", "/tmp/krisd/fdrill_packtest", "/tmp/krisd/fdrill_packtest/_pack"])
    names = sorted(tarfile.open(fileobj=io.BytesIO(data), mode="r:gz").getnames()) if data else []
    print(f"{label}: rc {rc}, {len(names)} members: {names}")
farm.ssh(f"srun --jobid={job} --overlap bash -c 'rm -rf /tmp/krisd/fdrill_packtest'")
