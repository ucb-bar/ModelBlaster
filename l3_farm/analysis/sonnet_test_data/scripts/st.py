#!/usr/bin/env python3
"""Sonnet heterogeneous-routing test on 43_MinGPTCausalAttention (X-139). Drives ONLY the jobs in ../my_jobs.json through
farm.ssh / farm.put (ssh computelab -> srun --jobid=J --overlap). Never touches farm.json, the dispatcher, other nodes or
/tmp/krisd/farm/<name>. Same recipe as analysis/beam_hetero_data (worker4.sh via arm.sh), frozen tree + fix-follow patch.

  st.py alloc N              request N L40S allocations (crun, like farm.alloc), logs in ../alloc/
  st.py jobs                 parse the alloc logs -> my_jobs.json (job id, host)
  st.py meta                 nvidia-smi facts of my nodes (memory clock must be 9001)
  st.py bringup              one bring-up step per node (tree / setup / relay)
  st.py install JOB          frozen tarball (sha-verified) + fix-follow patch -> /tmp/krisd/sonnet_code
  st.py modelcheck JOB       one tiny request per model through the node relay
  st.py launch ARM JOB       S3|S5|S0|H5 (fresh work dir /tmp/krisd/sonnet_<ARM>)
  st.py status               all arms
  st.py collect ARM
  st.py remeasure ARM        quiet re-measure on the arm's own node
"""
import hashlib, json, re, shlex, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
DATA = HERE.parent
L3 = HERE.parents[2]
sys.path.insert(0, str(L3))
import farm

SHA = "3bcd788657e9ce13615ab219aa253058e1f87a8fcd0a58c32e9c69faf4b8e0ee"
TB = L3 / "state" / "mb.tar.gz"
PATCH = DATA / "fix_follow.patch"
MJ = DATA / "my_jobs.json"
NAME = "43_MinGPTCausalAttention"
CR = "/tmp/krisd/sonnet_code"
OPUS = "azure/anthropic/claude-opus-5-5"
SONNET = "azure/anthropic/claude-sonnet-5-5"
HAIKU = "azure/anthropic/claude-haiku-5-5"
ARMS = {
    "S3": (SONNET, "category,stall:3,fix-escalate"),
    "S5": (SONNET, "category,stall:5,fix-follow,fix-escalate"),
    "S0": (SONNET, "category,fix-escalate"),
    "H5": (HAIKU, "category,stall:5,fix-follow,fix-escalate"),
}


def mj():
    return json.loads(MJ.read_text()) if MJ.exists() else {}


def save(d):
    MJ.write_text(json.dumps(d, indent=1))


def mine():
    return [int(j) for j in mj().get("jobs", {})]


def arm_job(arm):
    return int(mj()["arms"][arm])


def run_local(job, path, args=(), timeout=600):
    assert job in mine(), job
    body = Path(path).read_bytes()
    cmd = f"srun --jobid={job} --overlap bash -s -- " + " ".join(shlex.quote(str(a)) for a in args)
    return farm.ssh(cmd, body + b"\n", timeout)


def detach_local(job, path, args=()):
    assert job in mine(), job
    remote = f"/tmp/krisd/sonnet_scripts/{Path(path).name}"
    farm.put(job, path, remote)
    a = " ".join(shlex.quote(str(x)) for x in args)
    cmd = f"nohup setsid srun --jobid={job} --overlap bash {remote} {a} </dev/null >/dev/null 2>&1 & sleep 8; echo started"
    rc, out, err = farm.ssh(cmd, b"", timeout=120)
    return rc == 0 and b"started" in out


def alloc(n):
    for i in range(n):
        q = farm.CFG["alloc_query"]
        cmd = f"crun -q {shlex.quote(q)} --no-shell --gpus 1 -t {farm.CFG['alloc_time']}"
        lf = open(DATA / "alloc" / f"alloc_{int(time.time())}_{i}.log", "ab")
        subprocess.Popen(["ssh", "-o", "BatchMode=yes", farm.CFG["host"], cmd], stdout=lf, stderr=lf,
                         stdin=subprocess.DEVNULL, start_new_session=True)
        print("requested", i, flush=True)
        time.sleep(10)


def jobs_from_logs():
    d = mj()
    d.setdefault("jobs", {})
    for lf in sorted((DATA / "alloc").glob("alloc_*.log")):
        t = lf.read_text(errors="replace")
        m = re.search(r"Job Id\s*:\s*(\d+)", t)
        h = re.search(r"Node\s*:\s*(\S+)", t)
        if m:
            d["jobs"].setdefault(m.group(1), {})
            d["jobs"][m.group(1)].update(host=h.group(1) if h else None, log=lf.name)
    save(d)
    print(json.dumps(d, indent=1))


def meta():
    for j in mine():
        print(j, json.dumps(farm.node_meta(j)))


def bringup():
    s = farm.load()
    for j in mine():
        nd = farm.probe(j)
        if not nd["ok"]:
            print(j, "probe failed", nd); continue
        nd["jobs"]["__running__"] = False
        if farm.node_ready(nd, 600):
            print(j, "READY", nd); continue
        print(j, "step:", farm.bring_up(j, nd, s), {k: nd.get(k) for k in ("setup", "relay", "tree", "gpu")})


def install(job):
    assert hashlib.sha256(TB.read_bytes()).hexdigest() == SHA
    rc, out, err = farm.put(job, TB, "/tmp/krisd/sonnet_in/mb.tar.gz")
    print("put tar", rc, err.decode()[-200:])
    rc, out, err = farm.put(job, PATCH, "/tmp/krisd/sonnet_in/fix_follow.patch")
    print("put patch", rc, err.decode()[-200:])
    rc, out, err = run_local(job, HERE / "install.sh", [SHA, CR], timeout=900)
    print(rc, out.decode(), err.decode()[-500:])


def launch(arm, job):
    assert job in mine() and arm in ARMS
    cheap, pol = ARMS[arm]
    FD = f"/tmp/krisd/sonnet_{arm}"
    D = f"{FD}/{NAME}"
    k = farm.CFG["kb"]
    extra = f"--model-simple {cheap} --model-hard {OPUS} --hetero-policy {pol}"
    env = (f"MODEL={OPUS}\nARCH={k['arch']}\nMIN_GAIN={k['min_gain_pct']}\nROUNDS={k['rounds']}\n"
           f"ROLLOUTS={k['rollouts']}\nSTEPS={k['steps']}\nVSTEPS={k['variant_steps']}\nMAXVAR={k['max_variants']}\n"
           f"TIMEOUT_H=3\nPLAN_TOPK={k['plan_top_k']}\nPLAN_MINSHARE={k['plan_min_share']}\n"
           f"CODE_ROOT={CR}\nKB_EXTRA='{extra}'\n")
    print(env)
    rc, out, err = farm.ssh(f"srun --jobid={job} --overlap bash -c 'test -e {D} && echo EXISTS || echo fresh'")
    assert b"fresh" in out, f"{D} exists already"
    farm.put(job, L3 / "problems" / f"{NAME}.py", f"{D}/problem.py")
    farm.put(job, env.encode(), f"{D}/job.env")
    farm.put(job, L3 / "remote" / "worker4.sh", f"{FD}/worker4.sh")
    print("detached:", detach_local(job, HERE / "arm.sh", [arm]))
    d = mj(); d.setdefault("arms", {})[arm] = job; d.setdefault("launched", {})[arm] = time.time(); save(d)


def status():
    for arm, job in mj().get("arms", {}).items():
        rc, out, err = run_local(job, HERE / "status.sh", [arm], timeout=120)
        print(f"=== {arm} job {job} rc={rc}")
        print(out.decode().strip())


def collect(arm):
    job = arm_job(arm)
    out = DATA / "res" / arm
    out.mkdir(parents=True, exist_ok=True)
    rc, data, err = farm.srun(job, "pack2.sh", [NAME, f"/tmp/krisd/sonnet_{arm}", f"/tmp/krisd/sonnet_pack_{arm}"], timeout=1800)
    (out / "pack.tar.gz").write_bytes(data)
    subprocess.run(["tar", "xzf", str(out / "pack.tar.gz"), "-C", str(out)], check=False)
    D = f"/tmp/krisd/sonnet_{arm}/{NAME}"
    rc, data, err = farm.ssh(f"srun --jobid={job} --overlap bash -c 'D={D}; "
                             f"cat $D/ledger_start $D/ledger_end $D/t_start $D/t_end 2>/dev/null | tr \"\\n\" \" \"; echo; "
                             f"tail -n +$(( $(cut -d\" \" -f1 $D/ledger_start) + 1 )) /tmp/krisd/farm/llm_ledger.jsonl'", timeout=600)
    lines = data.decode().splitlines()
    (out / "markers.txt").write_text(lines[0] + "\n")
    (out / "ledger.jsonl").write_text("\n".join(lines[1:]) + "\n")
    print(arm, "collected", len(lines) - 1, "ledger rows;", lines[0])


def remeasure(arm, tag=None, on=None):
    job = on or arm_job(arm)
    rc, o, e = run_local(job, HERE / "remeasure.sh", [tag or arm, f"{arm}_on_{job}"], timeout=3600)
    line = next((l for l in o.decode().splitlines() if l.startswith("@@RM@@")), None)
    res = json.loads(line[6:]) if line else {"error": o.decode()[-800:] + e.decode()[-800:]}
    (DATA / "res" / f"remeasure_{arm}_on_{job}.json").write_text(json.dumps(res, indent=1))
    print(arm, job, json.dumps(res))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "alloc": alloc(int(sys.argv[2]))
    elif c == "jobs": jobs_from_logs()
    elif c == "meta": meta()
    elif c == "bringup": bringup()
    elif c == "install": install(int(sys.argv[2]))
    elif c == "modelcheck":
        rc, o, e = run_local(int(sys.argv[2]), HERE / "modelcheck.sh", [OPUS, SONNET, HAIKU], timeout=600)
        print(o.decode(), e.decode()[-500:])
    elif c == "release":
        for j in mine():
            print(j, farm.ssh(f"scancel {j}")[0])
    elif c == "launch": launch(sys.argv[2], int(sys.argv[3]))
    elif c == "status": status()
    elif c == "collect": collect(sys.argv[2])
    elif c == "remeasure": remeasure(*sys.argv[2:])
