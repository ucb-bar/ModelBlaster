"""scancel only jobs recorded as mine in e1_data/my_jobs.json: e1_release.py JOB [JOB...]"""
import sys
sys.path.insert(0, '/Users/krisd/Documents/ModelBlaster-clone/l3_farm/analysis/e1_tools'); sys.path.insert(0, '/Users/krisd/Documents/ModelBlaster-clone/l3_farm')
import farm, nodes
mine = nodes.mine()
for j in sys.argv[1:]:
    if str(j) not in mine:
        print("not mine, skipped", j); continue
    rc, o, e = farm.ssh(f"scancel {int(j)}; echo rc=$?", b"", timeout=120)
    print(j, o.decode().strip(), e.decode()[-200:])
