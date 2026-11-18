#!/usr/bin/env python3
"""Node helper of the merge checks. My jobs are recorded in my_jobs.json; nothing else is ever touched.

  mc_nodes.py alloc [N]     N x `crun -q <L40S query> --no-shell --gpus 1` through the computelab alias, ~10 s apart, each
                            into its own log under alloc/ (same command as farm.alloc, but farm.json is not touched);
                            the job id is read from that log
  mc_nodes.py mine          my jobs + squeue state
  mc_nodes.py wait [S]      poll until every one of my jobs is RUNNING and probes setup=done relay=ok (farm's bring-up)
  mc_nodes.py gpu [JOB..]   hostname, GPU uuid, VBIOS, clocks.mem / clocks.max.mem, compute apps (flags non-9001 MHz dies)
  mc_nodes.py release [JOB..]  scancel my jobs (only those in my_jobs.json; default all)

Every ssh happens inside farm.ssh (BatchMode, `computelab` alias).
"""
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mc_common as C  # noqa: E402

ALLOC = HERE / "alloc"

GPU_SH = r"""echo host=$(hostname)
nvidia-smi --query-gpu=name,uuid,driver_version,vbios_version,temperature.gpu,power.draw,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,clocks_throttle_reasons.active,utilization.gpu,memory.used --format=csv,noheader
echo apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')
"""


def alloc(n):
    f = C.farm()
    ALLOC.mkdir(exist_ok=True)
    for i in range(n):
        q = f.CFG["alloc_query"]
        cmd = f"crun -q {shlex.quote(q)} --no-shell --gpus 1 -t {f.CFG['alloc_time']}"
        lf = ALLOC / f"alloc_{int(time.time())}_{i}.log"
        fh = open(lf, "ab")
        subprocess.Popen(["ssh", "-o", "BatchMode=yes", f.CFG["host"], cmd], stdout=fh, stderr=fh,
                         stdin=subprocess.DEVNULL, start_new_session=True)
        print("requested", lf.name, flush=True)
        job = None
        for _ in range(200):
            time.sleep(3)
            m = re.search(r"--jobid=(\d+)", lf.read_text(errors="replace"))
            if m:
                job = int(m.group(1))
                break
        if job:
            d = C.mine()
            d[str(job)] = {"log": lf.name, "t_alloc": time.time(), "released": False}
            C.save_mine(d)
            print("mine:", job, flush=True)
        else:
            print("no job id in", lf.name, "yet (still queued?); re-run `mine` later or check the log", flush=True)
        time.sleep(10)


def adopt_pending():
    """Fill in job ids from logs of requests that were still queued when `alloc` stopped waiting."""
    d = C.mine()
    known = {v.get("log") for v in d.values()}
    for lf in sorted(ALLOC.glob("alloc_*.log")):
        if lf.name in known:
            continue
        m = re.search(r"--jobid=(\d+)", lf.read_text(errors="replace"))
        if m:
            d[m.group(1)] = {"log": lf.name, "t_alloc": lf.stat().st_mtime, "released": False}
            print("adopted", m.group(1), "from", lf.name)
    C.save_mine(d)


def mine():
    adopt_pending()
    q = C.squeue()
    for j, v in C.mine().items():
        print(j, v, q.get(int(j)))


def gpu(jobs):
    out = {}
    for j in jobs or [int(x) for x, v in C.mine().items() if not v.get("released")]:
        rc, o, e = C.rsh(j, GPU_SH, timeout=120)
        mem = None
        m = re.search(r"NVIDIA[^,]*,\s*GPU-[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,\s*(\d+) MHz,[^,]*,\s*(\d+) MHz", o)
        if m:
            mem = int(m.group(2))
        out[j] = {"out": o.strip(), "max_mem_mhz": mem, "slow_die": (mem is not None and mem < C.MEM_CLOCK_OK)}
        print("==", j, o.strip(), e[-200:], "\n   max mem clock:", mem, "SLOW DIE" if out[j]["slow_die"] else "")
    return out


def wait(limit):
    f = C.farm()
    t_end = time.time() + limit
    while True:
        adopt_pending()
        q = C.squeue()
        js = [int(j) for j, v in C.mine().items() if not v.get("released")]
        ready = []
        for j in js:
            if q.get(j, {}).get("state") == "RUNNING":
                d = f.probe(j)
                print(time.strftime("%H:%M:%S"), j, q[j]["host"], d.get("setup"), d.get("relay"), d.get("tree"), d.get("foreign"), flush=True)
                if d.get("setup") == "done" and d.get("relay") == "ok":
                    ready.append(j)
            else:
                print(time.strftime("%H:%M:%S"), j, q.get(j, {}).get("state", "gone"), flush=True)
        if len(ready) == len(js) and js:
            print("READY", ready)
            return
        if time.time() > t_end:
            print("TIMEOUT", ready)
            return
        time.sleep(60)


def release(only=()):
    d = C.mine()
    for j, v in d.items():
        if v.get("released") or (only and int(j) not in only):
            continue
        rc, out, err = C.farm().ssh(f"scancel {int(j)}")
        print("scancel", j, rc, err.decode()[-200:])
        v["released"] = True
        v["t_released"] = time.time()
    C.save_mine(d)


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "mine"
    if c == "alloc":
        alloc(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    elif c == "mine":
        mine()
    elif c == "wait":
        wait(float(sys.argv[2]) if len(sys.argv) > 2 else 3600)
    elif c == "gpu":
        gpu([int(x) for x in sys.argv[2:]])
    elif c == "release":
        release([int(x) for x in sys.argv[2:]])
