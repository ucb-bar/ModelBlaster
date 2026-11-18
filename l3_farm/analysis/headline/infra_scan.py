"""Scan problem logs on the driver for infrastructure failures. Run: FARM_CONFIG=analysis/headline/config_headline.json python3 analysis/headline/infra_scan.py
Prints per problem: infra counts (scp closed, [infra], NoDevice, so load, conn reset, final_infra, illegal memory), the rounds hit, and hints rejected unmeasured (not region); flags rerun candidates."""
import sys, os, json
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here + "/../..")
import farm
farm.load()
farm.put("5321587", here + "/infra_scan_remote.py", "/tmp/krisd/infra_scan_remote.py")
rc, o, e = farm.ssh("srun --jobid=5321587 --overlap python3 /tmp/krisd/infra_scan_remote.py")
res = json.loads(o.decode().strip().splitlines()[-1])
INFRA = ("scp_closed", "infra_tag", "nodevice", "so_load", "conn_reset", "final_infra")
rows = []
for n, r in res.items():
    c = r["counts"]; tot = sum(c[k] for k in INFRA)
    rows.append((tot, n, c, r))
rows.sort(key=lambda x: -x[0])
print(f"{'problem':34s} {'infra':>5s} scp  [i] nodev so  rst fin  illmem  rounds  rejected-unmeasured")
for tot, n, c, r in rows:
    if tot == 0 and not r["rejected_unmeasured"] and not c["illegal_mem"]: continue
    print(f"{n:34s} {tot:5d} {c['scp_closed']:3d} {c['infra_tag']:4d} {c['nodevice']:5d} {c['so_load']:3d} {c['conn_reset']:3d} {c['final_infra']:3d} {c['illegal_mem']:6d}  {r['rounds']}  {r['rejected_unmeasured'][:3]}  rounds-hit={sorted(r['per_round'])}")
print("clean problems:", [n for tot, n, c, r in rows if tot == 0 and not r["rejected_unmeasured"]])
