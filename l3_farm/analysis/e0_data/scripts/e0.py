#!/usr/bin/env python3
"""E0 noise-floor driver (X-103). Drives only the nodes listed in e0_data/runs.json, through farm.ssh/farm.put
(ssh -> srun --jobid=J --overlap), with this experiment's own remote scripts (scripts/remote/). Never touches
/tmp/krisd/farm/<name> dirs, farm.json or the live trees.

  e0.py probe JOB              farm probe (setup/relay/foreign) + venv check
  e0.py install JOB            upload the frozen tarball, verify sha256, unpack to /tmp/krisd/e0
  e0.py launch JOB PROBLEM SEED
  e0.py status                 all runs
  e0.py collect NAME           ledger slice + pack -> analysis/e0_runs/NAME
"""
import hashlib, json, shlex, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
L3 = HERE.parents[2]
sys.path.insert(0, str(L3))
import farm
import torch_posthoc

DATA = HERE.parent
TB = DATA / "mb_frozen.tar.gz"
SHA = "3bcd788657e9ce13615ab219aa253058e1f87a8fcd0a58c32e9c69faf4b8e0ee"
STAMP = "20261025-123252"
RUNS = DATA / "runs.json"
OUT = L3 / "analysis" / "e0_runs"
E = "/tmp/krisd/e0"


def runs():
    return json.loads(RUNS.read_text()) if RUNS.exists() else {}


def save_runs(r):
    RUNS.write_text(json.dumps(r, indent=1))


def rsh(job, script, args=(), timeout=600):
    body = (HERE / "remote" / script).read_bytes()
    cmd = f"srun --jobid={job} --overlap bash -s -- " + " ".join(shlex.quote(str(a)) for a in args)
    return farm.ssh(cmd, body + b"\n", timeout)


def install(job):
    assert hashlib.sha256(TB.read_bytes()).hexdigest() == SHA
    rc, out, err = farm.put(job, TB, f"{E}/mb.tar.gz")
    print("put", rc, err.decode()[-300:])
    rc, out, err = rsh(job, "e0_install.sh", [SHA, STAMP], timeout=900)
    print(rc, out.decode(), err.decode()[-500:])


def launch(job, prob, seed, th=4):
    name = f"{prob}_s{seed}"
    d = f"{E}/runs/{name}"
    farm.put(job, L3 / "problems" / f"{prob}.py", f"{d}/problem.py")
    farm.put(job, torch_posthoc.BENCH.encode(), f"{d}/bench.py")
    farm.put(job, HERE / "remote" / "e0_worker.sh", f"{E}/e0_worker.sh")
    cmd = (f"nohup setsid srun --jobid={job} --overlap bash {E}/e0_worker.sh {prob} {seed} {th} "
           f"</dev/null >/dev/null 2>&1 & sleep 8; echo started")
    rc, out, err = farm.ssh(cmd, b"", timeout=120)
    ok = rc == 0 and b"started" in out
    r = runs()
    r[name] = dict(job=job, problem=prob, seed=seed, launched=time.time(), ok=ok)
    save_runs(r)
    print(name, "launched" if ok else "FAILED", err.decode()[-300:])


def status():
    by = {}
    for n, v in runs().items():
        by.setdefault(v["job"], []).append(n)
    for job, names in by.items():
        rc, out, err = rsh(job, "e0_status.sh", timeout=60)
        print(f"== job {job} ({','.join(names)}) rc={rc}")
        print(out.decode().strip())


def collect(name):
    v = runs()[name]
    rc, out, err = rsh(v["job"], "e0_ledger.sh", [name, int(v["launched"]) - 5], timeout=300)
    print("ledger lines", out.decode().strip(), err.decode()[-200:])
    body = (HERE / "remote" / "e0_pack.sh").read_bytes()
    rc, data, err = farm.ssh(f"srun --jobid={v['job']} --overlap bash -s -- {shlex.quote(name)}", body + b"\n", 3600)
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "pack.tar.gz").write_bytes(data)
    subprocess.run(["tar", "xzf", str(d / "pack.tar.gz"), "-C", str(d)], check=False)
    print(name, len(data) // 2**20, "MB collected ->", d)


def probe(job):
    print(json.dumps(farm.probe(job)))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "probe":
        probe(int(sys.argv[2]))
    elif c == "install":
        install(int(sys.argv[2]))
    elif c == "launch":
        launch(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), *(map(float, sys.argv[5:6])))
    elif c == "status":
        status()
    elif c == "collect":
        collect(sys.argv[2])


def cost_all():
    tot = 0.0
    for n, v in runs().items():
        rc, out, err = rsh(v["job"], "e0_cost.sh", [int(v["launched"]) - 5], timeout=120)
        s = out.decode().strip()
        print(n, s)
        try:
            tot += float(s.split("usd=")[1])
        except (IndexError, ValueError):
            pass
    print(f"TOTAL usd={tot:.2f}")


if __name__ == "__main__" and sys.argv[1] == "cost":
    cost_all()


if __name__ == "__main__" and sys.argv[1] == "peek":
    for n, v in runs().items():
        if len(sys.argv) > 2 and n not in sys.argv[2:]:
            continue
        rc, out, err = rsh(v["job"], "e0_peek.sh", [n, int(v["launched"]) - 5], timeout=120)
        print("==", n); print(out.decode().strip())


if __name__ == "__main__" and sys.argv[1] == "finish":
    for n in sys.argv[2:]:
        v = runs()[n]
        rc, out, err = rsh(v["job"], "e0_compile.sh", [n, v["problem"]], timeout=2400)
        print(n, "compile bench:", out.decode().strip()[-400:])
        collect(n)
        r = runs(); r[n]["collected"] = time.time(); save_runs(r)
