"""Usage: release_nodes.py JOB... -- scancel farm-allocated L40S jobs that are idle (refuses any job running a farm problem)."""
import sys, farm
s = farm.load()
busy = {p["node"] for p in s["problems"].values() if p["status"] == "running"}
for j in map(int, sys.argv[1:]):
    if j in busy:
        print(j, "BUSY - skipped"); continue
    rc, out, err = farm.ssh(f"scancel {j}")
    print(j, "scancel rc", rc, err.decode()[:80])
