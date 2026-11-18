#!/usr/bin/env python3
"""A2 (capabilities) node helper (copy of e8_recompose/nodes.py). All remote work goes through farm.ssh -> `srun --jobid=J --overlap bash -s` (farm.py unmodified).

  nodes.py alloc [N]        N x farm.alloc(s) ~12 s apart; each new job is attributed to us by matching the crun job
                            name timestamp (crun_MM_DD_YYYY__HH_MM_SS_*) to the time of our call (+-8 s, exactly one
                            candidate), and recorded in analysis/e8_data/my_jobs.json
  nodes.py mine             my E8 jobs + squeue state
  nodes.py probe            farm.probe for each of my jobs (setup / relay / foreign / farm jobs)
  nodes.py tools            ninja into /tmp/krisd/baseline_tools on each of my jobs (baseline_cc/node.py tools)
  nodes.py gpu              hostname, GPU uuid/serial, clocks, power, compute processes per job

Only jobs in my_jobs.json are ever touched.
"""
import json, re, shlex, subprocess, sys, time
from datetime import datetime
from pathlib import Path

L3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(L3))
import farm  # noqa: E402

DATA = L3 / "analysis" / "capabilities"
MINE = DATA / "a2_jobs.json"


def squeue():
    rc, out, err = farm.ssh("squeue --me -h -o '%i|%T|%N|%L|%j'")
    res = {}
    for ln in out.decode().splitlines():
        p = ln.split("|")
        if len(p) == 5:
            res[int(p[0])] = dict(state=p[1], host=p[2], left=p[3], name=p[4])
    return res


def name_time(n):
    m = re.match(r"crun_(\d+)_(\d+)_(\d+)__(\d+)_(\d+)_(\d+)", n)
    if not m:
        return None
    mo, d, y, H, M, S = map(int, m.groups())
    return datetime(y, mo, d, H, M, S).timestamp()


def mine():
    return json.loads(MINE.read_text()) if MINE.exists() else {}


def save_mine(d):
    DATA.mkdir(parents=True, exist_ok=True)
    MINE.write_text(json.dumps(d, indent=1))


ALOG = L3 / "state" / "alloc.log"


def alloc(n):
    """farm.alloc appends its crun output to state/alloc.log (other agents allocating through other means do not).
    A call is attributed when exactly one crun invocation block appears in the bytes appended after our call and it
    reports a Job Id (other farm.alloc callers in the same window -> ambiguous -> not claimed, reported)."""
    for k in range(n):
        off = ALOG.stat().st_size
        t0 = time.time()
        s = farm.load(); farm.alloc(s); farm.save(s)
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
            d = mine(); d[str(got)] = dict(t_call=t0, host=(q.get(got) or {}).get("host"), name=(q.get(got) or {}).get("name"))
            save_mine(d)
            print("mine:", got, q.get(got), flush=True)
        else:
            print("no attributable job for call at", time.strftime("%H:%M:%S", time.localtime(t0)), flush=True)
        time.sleep(12)


def run(job, script, timeout=600):
    if str(job) not in mine():
        raise SystemExit(f"job {job} is not one of mine")
    rc, out, err = farm.ssh(f"srun --jobid={job} --overlap bash -s", script.encode(), timeout=timeout)
    return rc, out.decode(errors="replace"), err.decode(errors="replace")


GPU_SH = r"""hostname
nvidia-smi --query-gpu=name,uuid,serial,pci.bus_id,driver_version,vbios_version,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,clocks_throttle_reasons.active,utilization.gpu,memory.used,pstate,ecc.errors.uncorrected.volatile.total --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
"""


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "mine"
    if cmd == "alloc":
        alloc(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    elif cmd == "mine":
        q = squeue()
        for j, v in mine().items():
            print(j, v, q.get(int(j)))
    elif cmd == "probe":
        for j in mine():
            if int(j) in squeue():
                print(j, farm.probe(int(j)))
    elif cmd == "tools":
        sh = ("set -e; . /tmp/krisd/venv/bin/activate; mkdir -p /tmp/krisd/baseline_tools; "
              "[ -x /tmp/krisd/baseline_tools/py/bin/ninja ] || pip -q install --target /tmp/krisd/baseline_tools/py ninja; "
              "/tmp/krisd/baseline_tools/py/bin/ninja --version\n")
        for j in (sys.argv[2:] or mine()):
            print(j, run(j, sh)[1][-300:])
    elif cmd == "gpu":
        for j in (sys.argv[2:] or mine()):
            rc, o, e = run(j, GPU_SH, 120)
            print("==", j, o, e[-300:])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
