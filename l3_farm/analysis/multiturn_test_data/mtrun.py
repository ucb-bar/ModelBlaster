#!/usr/bin/env python3
"""X-144 driver (ssh via farm.ssh only).  mtrun.py prep|start|status|fetch [ARM...]
Arms -> jobs in ARMS below (jobs are in my_jobs.json). Frozen tarball = A2's stage 20261025-174638."""
import json, shlex, sys, threading
from pathlib import Path
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parents[1]))
import farm  # noqa: E402
STAMP = "20261025-174638"
TB = H.parent / "capabilities" / "stage" / f"mb_a2_{STAMP}.tar.gz"
MINE = {int(j) for j in json.loads((H / "my_jobs.json").read_text())}
ARMS = json.loads((H / "arms.json").read_text())   # name -> {job, steps, mt, seed, force}

def check(a):
    assert ARMS[a]["job"] in MINE
    return ARMS[a]["job"], f"/tmp/krisd/mt_{a}"

def put(job, local, remote):
    rc, o, e = farm.put(job, local, remote)
    assert rc == 0, (remote, e.decode()[-300:])

def sh(job, script, timeout=900):
    rc, o, e = farm.ssh(f"srun --jobid={job} --overlap bash -s", script.encode(), timeout=timeout)
    return o.decode(errors="replace") + e.decode(errors="replace")[-1500:]

def prep(a):
    job, R = check(a)
    put(job, TB, f"{R}/mb.tar.gz")
    for f in ("b41.tgz", "force_tech_order.patch"):
        put(job, H / f, f"{R}/{f}")
    for f in ("mt_prep.py", "mkprob.py", "mt_prep.sh", "mt_run.sh", "mt_status.sh"):
        put(job, H / "node" / f, f"{R}/{f}")
    out = sh(job, f"mkdir -p {R}; bash {R}/mt_prep.sh {a} {STAMP} 2>&1 | tail -15", 1200)
    print(a, job, out, flush=True)

def start(a):
    job, R = check(a)
    c = ARMS[a]
    args = f"{a} {c['steps']} {c['mt']} {c['seed']} {shlex.quote(c.get('force') or '')}"
    cmd = (f"nohup setsid srun --jobid={job} --overlap bash -c {shlex.quote(f'bash {R}/mt_run.sh {args} > {R}/arm.log 2>&1')} "
           f"</dev/null >/dev/null 2>&1 & sleep 5; echo started")
    rc, o, e = farm.ssh(cmd, b"", timeout=120)
    print(a, job, o.decode(), e.decode()[-300:], flush=True)

def kill(a):
    job, R = check(a)
    put(job, H / "node" / "mt_kill.sh", f"{R}/mt_kill.sh")
    print("==", a, sh(job, f"bash {R}/mt_kill.sh {a}", 120), flush=True)

def status(a):
    job, R = check(a)
    put(job, H / "node" / "mt_status.sh", f"{R}/mt_status.sh")
    print("==", a, job, sh(job, f"bash {R}/mt_status.sh {a}", 120), flush=True)

def fetch(a):
    job, R = check(a)
    dst = H / "arms" / a; dst.mkdir(parents=True, exist_ok=True)
    rc, data, e = farm.ssh(f"srun --jobid={job} --overlap bash -c {shlex.quote(f'cd {R} && tar czf - --exclude=build --exclude=work --exclude=trace out arm.log run.log cost.log ledger.start ledger.rows.jsonl prob/manifest.json 2>/dev/null')}", b"", timeout=900)
    (dst / "arm.tgz").write_bytes(data)
    import subprocess; subprocess.run(["tar", "xzf", str(dst / "arm.tgz"), "-C", str(dst)])
    print(a, len(data), flush=True)

def par(fn, names):
    ts = [threading.Thread(target=fn, args=(a,)) for a in names]
    [t.start() for t in ts]; [t.join() for t in ts]

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] != "remeasure":
    cmd, names = sys.argv[1], (sys.argv[2:] or list(ARMS))
    par({"prep": prep, "start": start, "status": status, "kill": kill, "fetch": fetch}[cmd], names)

def remeasure(names, reps=5, node="M8_s0"):
    """all finals + the seed, one GPU (node's), back to back; best slots pushed from arms/<name>/out/<prob>/best/slots."""
    job, R = check(node)
    put(job, H / "node" / "remeasure.py", f"{R}/remeasure.py")
    args = []
    for n in names:
        d = H / "arms" / n / "out" / "a2_41_gru_rev_l1_5" / "best" / "slots"
        for f in d.glob("*.cu"):
            put(job, f, f"{R}/rm/{n}/{f.name}")
        args.append(f"{n}={R}/rm/{n}")
    out = sh(job, f"cd {R}; . {R}/env.sh; PYTHONPATH=$PYTHONPATH:{R}/code/cur/ModelBlaster/third_party/KernelBlaster/src python {R}/remeasure.py {R}/prob {R}/remeasure.json {reps} {' '.join(args)} 2>&1 | grep -v INFO | tail -80; cat {R}/remeasure.json", 3000)
    print(out)
    js = out[out.index("{"):] if "{" in out else "{}"
    return out

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "remeasure":
    o = remeasure(sys.argv[2:])
    i = o.rfind("\n{")
    (H / "remeasure.json").write_text(o[i + 1:])
