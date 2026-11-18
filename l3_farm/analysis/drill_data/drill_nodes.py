#!/usr/bin/env python3
"""X-128 drill node helper (own allocations only; all remote work through the `computelab` alias inside this file).

  drill_nodes.py alloc N     N x crun (L40S, own log drill_data/alloc.log) -> my_jobs.json (max 2 live at a time)
  drill_nodes.py mine        my jobs + squeue state
  drill_nodes.py probe       probe2.sh of my jobs (setup / relay / tree / foreign / farm jobs)
  drill_nodes.py meta        nvidia-smi query of my nodes (remote/nodemeta.sh)
  drill_nodes.py scancel JOB scancel ONE of my own jobs (refuses anything not in my_jobs.json)
  drill_nodes.py release     scancel all my jobs (refuses jobs running a problem of any drill instance state)
"""
import json, os, re, shlex, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("FARM_CONFIG", "analysis/drill_data/config_drill_ind.json")
L3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(L3))
import farm  # noqa: E402

HERE = Path(__file__).resolve().parent
MINE = HERE / "my_jobs.json"
ALOG = HERE / "alloc.log"


def mine():
    return json.loads(MINE.read_text()) if MINE.exists() else {}


def squeue():
    rc, out, err = farm.ssh("squeue --me -h -o '%i|%T|%N|%L|%j'")
    res = {}
    for ln in out.decode().splitlines():
        p = ln.split("|")
        if len(p) == 5:
            res[int(p[0])] = dict(state=p[1], host=p[2], left=p[3], name=p[4])
    return res


def live_mine():
    q = squeue()
    return {j: q[int(j)] for j in mine() if int(j) in q}


def alloc(n, hours="5:00:00"):
    for _ in range(n):
        if len(live_mine()) >= 2:
            print("already 2 live jobs of mine; not allocating"); return
        off = ALOG.stat().st_size if ALOG.exists() else 0
        t0 = time.time()
        lf = open(ALOG, "ab")
        cmd = f"crun -q {shlex.quote(farm.CFG['alloc_query'])} --no-shell --gpus 1 -t {hours}"
        subprocess.Popen(["ssh", "-o", "BatchMode=yes", farm.CFG["host"], cmd], stdout=lf, stderr=lf,
                         stdin=subprocess.DEVNULL, start_new_session=True)
        got = None
        for _ in range(400):                      # up to ~20 min in the queue
            time.sleep(3)
            new = ALOG.read_bytes()[off:].decode(errors="replace")
            ids = re.findall(r"Job Id\s*:\s*(\d+)", new) or re.findall(r"srun --jobid=(\d+)", new)
            if ids:
                got = int(ids[0]); break
        if not got:
            print("no job id seen yet in", ALOG, "(still queued?)"); return
        q = squeue(); d = mine()
        d[str(got)] = dict(t_call=t0, host=(q.get(got) or {}).get("host"), t_alloc=time.time())
        MINE.write_text(json.dumps(d, indent=1))
        print("mine:", got, q.get(got), flush=True)
        time.sleep(5)


def run_script(job, script, args=()):
    assert str(job) in mine(), f"{job} is not mine"
    return farm.srun(int(job), script, list(args), timeout=120)


def main():
    a = sys.argv[1:]
    cmd = a.pop(0)
    if cmd == "alloc":
        alloc(int(a[0]) if a else 1)
    elif cmd == "mine":
        print(json.dumps(live_mine(), indent=1))
    elif cmd == "probe":
        for j in live_mine():
            d = farm.probe(int(j))
            print(j, {k: d.get(k) for k in ("ok", "gpu", "setup", "relay", "tree", "foreign")}, d.get("jobs"))
    elif cmd == "meta":
        for j in live_mine():
            rc, out, err = run_script(j, "nodemeta.sh")
            print(j, out.decode().strip().replace("\n", " | "), err.decode()[-200:])
    elif cmd == "scancel":
        j = a[0]
        assert j in mine(), f"{j} is not one of my jobs {list(mine())}"
        rc, out, err = farm.ssh(f"scancel {int(j)}")
        print("scancel", j, "rc", rc, err.decode()[:100])
    elif cmd == "release":
        busy = set()
        for sf in HERE.glob("i_*/farm.json"):
            st = json.loads(sf.read_text())
            busy |= {str(p["node"]) for p in st["problems"].values() if p["status"] == "running"}
        for j in list(live_mine()):
            if j in busy and "--force" not in a:
                print(j, "BUSY (a drill instance state says running) - skipped"); continue
            rc, out, err = farm.ssh(f"scancel {int(j)}")
            print("scancel", j, "rc", rc, err.decode()[:100])


if __name__ == "__main__":
    main()
