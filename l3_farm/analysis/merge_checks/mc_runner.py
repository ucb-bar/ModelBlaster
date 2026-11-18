#!/usr/bin/env python3
"""Run driver shared by the three checks: install a frozen tree on one of my nodes, launch a flow run from it, watch it,
collect the pack. Also usable on its own:

  mc_runner.py install JOB TARBALL
  mc_runner.py launch  JOB TARBALL ARM PROBLEM SEED [--stub] [--flags "..."] [--envs "A=1 B=2"] [--budget "..."] [--ncu best|off] [--th 4]
  mc_runner.py status  [JOB ..]
  mc_runner.py collect NAME            NAME = <label>/<arm>/<problem>_s<seed> (see registry.json)
  mc_runner.py cost    JOB T0 [T1]

Registry: runs/registry.json (name -> job, tarball sha256, flags, launch time). Only jobs in my_jobs.json are used.
The worker never receives the API key: it sources the node's relay env (/tmp/krisd/farm/relay.env).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mc_common as C  # noqa: E402

REG = C.RUNS / "registry.json"
STD_BUDGET = "--rounds 3 --rollouts 4 --steps 4"          # the farm's / E0's budget


def registry() -> dict:
    return C.load_json(REG, {})


def save_registry(r):
    C.write_json(REG, r)


def install(job, tarball):
    ti = C.tree_info(tarball)
    rc, o, e = C.put(job, ti["path"], f"{C.NODE_ROOT}/{ti['label']}.tar.gz")
    rc, out, err = C.rfile(job, "mc_install.sh", [ti["label"], ti["sha256"], ti["stamp"]], timeout=900)
    print(out.strip(), err[-300:])
    return ti, "installed" in out


def run_name(ti, arm, problem, seed):
    return f"{ti['label']}/{arm}/{problem}_s{seed}"


def launch(job, tarball, arm, problem, seed, stub=False, flags="", envs="", budget=STD_BUDGET, ncu=None, th=4):
    ti = C.tree_info(tarball)
    name = run_name(ti, arm, problem, seed)
    d = f"{C.NODE_ROOT}/runs/{name}"
    ncu = ncu or ("off" if stub else "best")
    env = (f'LABEL={ti["label"]}\nPROBLEM={problem}\nSEED={seed}\nSTUB={1 if stub else 0}\nNCU={ncu}\n'
           f'BUDGET="{budget}"\nFLAGS="{flags}"\nENVS="{envs}"\nTH={th}\nARMNAME={arm}\nMODEL={C.MODEL}\nCUDA_N={C.CUDA_N}\n')
    C.put(job, (C.L3 / "problems" / f"{problem}.py"), f"{d}/problem.py")
    C.put(job, env.encode(), f"{d}/run.env")
    C.put(job, HERE / "remote" / "mc_worker.sh", f"{C.NODE_ROOT}/mc_worker.sh")
    cmd = (f"nohup setsid srun --jobid={job} --overlap bash {C.NODE_ROOT}/mc_worker.sh {d} "
           f"</dev/null >/dev/null 2>&1 & sleep 8; echo started")
    C.check_mine(job)
    rc, out, err = C.farm().ssh(cmd, b"", timeout=120)
    ok = rc == 0 and b"started" in out
    r = registry()
    r[name] = dict(job=job, tree=ti, arm=arm, problem=problem, seed=seed, stub=stub, flags=flags, envs=envs,
                   budget=budget, ncu=ncu, launched=time.time(), launched_str=C.now(), ok=ok)
    save_registry(r)
    print(name, "launched" if ok else "FAILED", err.decode()[-300:])
    return name


def status(jobs=None):
    by = {}
    for n, v in registry().items():
        if jobs and v["job"] not in jobs:
            continue
        by.setdefault(v["job"], []).append(n)
    for job in by:
        rc, out, err = C.rfile(job, "mc_status.sh", timeout=90)
        print(f"== job {job} rc={rc}\n{out.strip()}")


def node_status(job, name) -> str:
    """The status word of one run (running / remeasuring / done / failed:... / ...:died / missing)."""
    try:
        rc, out, err = C.rfile(job, "mc_status.sh", timeout=150)
    except subprocess.TimeoutExpired:
        return "unreachable"
    for ln in out.splitlines():
        p = ln.split()
        if len(p) > 2 and p[0] == "run" and p[1].rstrip("/") == name.rstrip("/"):
            return p[2]
    return "missing"


def parse_smi(txt: str) -> dict:
    d = {}
    m = re.search(r"job=(\S+) host=(\S+)", txt)
    if m:
        d["job_id"], d["host"] = m.group(1), m.group(2)
    rows = [ln for ln in txt.splitlines() if ln.startswith("NVIDIA")]
    if rows:
        f = [x.strip() for x in rows[0].split(",")]
        keys = ["name", "uuid", "driver", "vbios", "temp_c", "clock_sm", "clock_mem", "max_clock_sm", "max_clock_mem", "throttle", "util"]
        d.update(dict(zip(keys, f)))
        for k in ("clock_mem", "max_clock_mem"):
            m2 = re.match(r"(\d+)", d.get(k, ""))
            d[k + "_mhz"] = int(m2.group(1)) if m2 else None
    return d


def collect(name):
    v = registry()[name]
    job = v["job"]
    d = f"{C.NODE_ROOT}/runs/{name}"
    rc, out, err = C.rfile(job, "mc_ledger.sh", [d], timeout=120)
    ledger_lines = out.strip()
    rc, data, err = C.farm().ssh(f"srun --jobid={job} --overlap bash -s -- {d}",
                                 (HERE / "remote" / "mc_pack.sh").read_bytes() + b"\n", 3600)
    out_dir = C.RUNS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pack.tar.gz").write_bytes(data)
    subprocess.run(["tar", "xzf", str(out_dir / "pack.tar.gz"), "-C", str(out_dir)], check=False)
    smi = parse_smi((out_dir / "smi_start.txt").read_text()) if (out_dir / "smi_start.txt").exists() else {}
    smi_end = parse_smi((out_dir / "smi_end.txt").read_text()) if (out_dir / "smi_end.txt").exists() else {}
    calls, usd = C.cost_usd(out_dir / "ledger.jsonl")

    def ts(f):
        p = out_dir / f
        return int(p.read_text().strip()) if p.exists() and p.read_text().strip().isdigit() else None
    t0, t1, t2 = ts("t0"), ts("t1"), ts("t2")
    meta = {"name": name, "tree": v["tree"], "flags": v["flags"], "envs": v["envs"], "budget": v["budget"], "stub": v["stub"],
            "seed": v["seed"], "problem": v["problem"], "arm": v["arm"], "ncu": v.get("ncu"),
            "cmdline": (out_dir / "cmdline.txt").read_text().strip() if (out_dir / "cmdline.txt").exists() else None,
            "job": job, "host": smi.get("host"), "gpu_uuid": smi.get("uuid"), "vbios": smi.get("vbios"),
            "mem_clock_mhz": smi.get("max_clock_mem_mhz"), "mem_clock_end_mhz": smi_end.get("max_clock_mem_mhz"),
            "slow_memory_die": (smi.get("max_clock_mem_mhz") or 0) < C.MEM_CLOCK_OK,
            "flow_wall_s": (t1 - t0) if (t0 and t1) else None, "llm_calls_ledger": calls, "llm_usd": usd,
            "ledger_lines": ledger_lines, "collected": C.now(), "pack_bytes": len(data)}
    C.write_json(out_dir / "meta.json", meta)
    print(name, len(data) // 1024, "KB ->", out_dir, "| calls", calls, "usd", usd, "| mem clock", meta["mem_clock_mhz"])
    return meta


def cost(job, t0, t1="9999999999"):
    rc, out, err = C.rfile(job, "mc_cost.sh", [t0, t1], timeout=120)
    print(out.strip())
    return out.strip()


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "install":
        install(int(sys.argv[2]), sys.argv[3])
    elif c == "launch":
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("job", type=int)
        ap.add_argument("tarball")
        ap.add_argument("arm")
        ap.add_argument("problem")
        ap.add_argument("seed", type=int)
        ap.add_argument("--stub", action="store_true")
        ap.add_argument("--flags", default="")
        ap.add_argument("--envs", default="")
        ap.add_argument("--budget", default=STD_BUDGET)
        ap.add_argument("--ncu")
        ap.add_argument("--th", type=float, default=4)
        a = ap.parse_args(sys.argv[2:])
        launch(a.job, a.tarball, a.arm, a.problem, a.seed, a.stub, a.flags, a.envs, a.budget, a.ncu, a.th)
    elif c == "status":
        status([int(x) for x in sys.argv[2:]] or None)
    elif c == "collect":
        collect(sys.argv[2])
    elif c == "cost":
        cost(int(sys.argv[2]), *sys.argv[3:5])
