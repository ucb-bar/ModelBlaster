#!/usr/bin/env python3
"""X-148 (compile pool) node helper; all remote work goes through farm.ssh / farm.put -> `srun --jobid=J --overlap`.

  nodes_cp.py alloc [N]            N x farm.alloc (max 4 in all; a dummy state dict is used, the live state/farm.json is never read or written), attributed via state/alloc.log -> my_jobs.json
  nodes_cp.py mine                 my jobs + squeue state
  nodes_cp.py probe                farm.probe of my jobs (setup / relay / foreign / farm jobs)
  nodes_cp.py run JOB SCRIPT.sh [timeout] [--put LOCAL=REMOTE ...] [--detach]
                                   run a bash script on one of my nodes (work dir /tmp/krisd/cpool); --detach: nohup'd srun
                                   step, log in /tmp/krisd/cpool/logs/<script>.log
Only jobs in my_jobs.json are ever touched. Never scancel (tell the user which to release).
"""
import json
import re
import shlex
import sys
import time
from pathlib import Path

L3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(L3))
import farm  # noqa: E402

HERE = Path(__file__).resolve().parent
MINE = HERE / "my_jobs.json"
ALOG = L3 / "state" / "alloc.log"
W = "/tmp/krisd/cpool"


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


def alloc(n):
    have = mine()
    n = min(n, 4 - len(have))
    for _ in range(max(n, 0)):
        off = ALOG.stat().st_size
        t0 = time.time()
        farm.alloc({})
        got = None
        for _ in range(120):
            time.sleep(3)
            new = ALOG.read_bytes()[off:].decode(errors="replace")
            inv = new.count("Invoked command")
            ids = re.findall(r"Job Id\s*:\s*(\d+)", new)
            if inv > 1:
                print("ambiguous: several farm.alloc invocations in the window", ids, "-- not claiming")
                break
            if inv == 1 and ids:
                got = int(ids[0])
                break
        if got:
            q = squeue()
            d = mine()
            d[str(got)] = dict(t_call=t0, host=(q.get(got) or {}).get("host"))
            MINE.write_text(json.dumps(d, indent=1))
            print("mine:", got, q.get(got), flush=True)
        time.sleep(12)


def run(job, script, timeout=900, puts=(), detach=False):
    assert str(job) in mine(), f"{job} is not one of my jobs {list(mine())}"
    for loc, rem in puts:
        assert rem.startswith(W + "/"), rem
        rc, out, err = farm.put(job, Path(loc), rem)
        if rc:
            sys.exit(f"put {loc} failed: {err.decode()[-300:]}")
    pre = f"mkdir -p {W}/logs; cd {W}\n"
    if detach:
        rem = f"{W}/scripts/{Path(script).name}"
        farm.put(job, Path(script), rem)
        log = f"{W}/logs/{Path(script).stem}.log"
        cmd = (f"nohup setsid srun --jobid={job} --overlap bash -c {shlex.quote(pre + f'bash {rem} > {log} 2>&1')} "
               f"</dev/null >/dev/null 2>&1 & sleep 5; echo started")
        rc, out, err = farm.ssh(cmd, b"", timeout=120)
        print(out.decode(), err.decode()[-500:], "rc", rc)
        return
    rc, out, err = farm.ssh(f"srun --jobid={job} --overlap bash -s", (pre + Path(script).read_text() + "\n").encode(),
                            timeout=timeout)
    sys.stdout.write(out.decode(errors="replace"))
    sys.stdout.write(err.decode(errors="replace")[-3000:])
    print("rc", rc)


def main():
    a = sys.argv[1:]
    cmd = a.pop(0)
    if cmd == "alloc":
        alloc(int(a[0]) if a else 1)
    elif cmd == "mine":
        q = squeue()
        for j in mine():
            print(j, q.get(int(j)))
    elif cmd == "probe":
        for j in mine():
            d = farm.probe(int(j))
            print(j, {k: d.get(k) for k in ("ok", "gpu", "setup", "relay", "tree", "foreign")}, d.get("jobs"))
    elif cmd == "run":
        job = int(a.pop(0))
        script = a.pop(0)
        timeout = int(a.pop(0)) if a and a[0].isdigit() else 900
        puts, det = [], False
        while a:
            x = a.pop(0)
            if x == "--put":
                puts.append(tuple(a.pop(0).split("=", 1)))
            elif x == "--detach":
                det = True
        run(job, script, timeout, puts, det)


if __name__ == "__main__":
    main()
