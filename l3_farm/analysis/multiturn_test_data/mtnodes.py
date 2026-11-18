#!/usr/bin/env python3
"""X-144 node helper. All remote work via farm.ssh (alias computelab, inside this file only).
  mtnodes.py alloc N      N parallel `crun --no-shell` L40S allocations; job ids -> my_jobs.json (own crun output)
  mtnodes.py mine         my jobs + squeue state
  mtnodes.py probe        farm.probe (setup / relay / tree) + GPU memory clock for each of my jobs
  mtnodes.py release      scancel ONLY the jobs in my_jobs.json
"""
import json, re, sys, threading
from pathlib import Path
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parents[1]))
import farm  # noqa: E402
MINE = H / "my_jobs.json"

def mine():
    return json.loads(MINE.read_text()) if MINE.exists() else {}

def squeue():
    rc, out, err = farm.ssh("squeue --me -h -o '%i|%T|%N|%L|%j'")
    return {int(p[0]): p[1:] for p in (l.split("|") for l in out.decode().splitlines()) if len(p) == 5}

def one(k, lock):
    q = farm.CFG["alloc_query"]
    rc, out, err = farm.ssh(f"crun -q '{q}' --no-shell --gpus 1 -t 04:00:00", b"", timeout=1500)
    txt = (out + err).decode(errors="replace")
    m = re.search(r"Job Id\s*:\s*(\d+)", txt)
    with lock:
        if m:
            d = mine(); d[m.group(1)] = {"arm_slot": k}; MINE.write_text(json.dumps(d, indent=1))
            print("job", m.group(1), flush=True)
        else:
            print("alloc", k, "failed:", txt[-300:], flush=True)

def main():
    cmd = sys.argv[1]
    if cmd == "alloc":
        lock = threading.Lock(); ts = [threading.Thread(target=one, args=(k, lock)) for k in range(int(sys.argv[2]))]
        for t in ts: t.start()
        for t in ts: t.join()
    elif cmd == "mine":
        q = squeue()
        for j in mine(): print(j, mine()[j], q.get(int(j)))
    elif cmd == "probe":
        for j in mine():
            j = int(j)
            print(j, farm.probe(j))
            rc, out, err = farm.srun(j, "nodemeta.sh", timeout=60)
            print("  ", [l for l in out.decode().splitlines() if l.startswith(("smi=", "host="))])
    elif cmd == "release":
        ids = " ".join(j for j in (sys.argv[2:] or mine()) if j in mine())
        rc, out, err = farm.ssh(f"scancel {ids}")
        print(rc, out.decode(), err.decode()[-300:])

main()
