#!/usr/bin/env python3
"""Create / reset a drill farm instance: config_drill_<mode>.json + i_<mode>/ (state, frozen tarball copy, results).
  gen_instance.py MODE [--problems 1_MLP,...] [--jobs J1,J2] [--sync-min 3] [--reset]
MODE: ind | staged | pooled. Own state dir, results dir, remote dir (/tmp/krisd/fdrill_<mode>) and code root."""
import json, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
L3 = HERE.parents[1]
STAMP = "20261025-123252"           # state/mb.tar.gz (frozen)


def main():
    a = sys.argv[1:]
    mode = a.pop(0)
    opt = {"--problems": "1_MLP", "--jobs": "", "--sync-min": "3"}
    reset = "--reset" in a
    while a:
        x = a.pop(0)
        if x in opt:
            opt[x] = a.pop(0)
    base = json.loads((L3 / "smoke_sched" / "config_b.json").read_text())
    base.pop("_notes", None)
    nums = [int(p.split("_")[0]) for p in opt["--problems"].split(",")]
    base.update(include=nums, sync_min=int(opt["--sync-min"]), tick_s=45, max_attempts=3, problem_timeout_h=2)
    base["global_knowledge"] = {"enabled": False, "mode": "off"}
    d = f"analysis/drill_data/i_{mode}"
    base["instance"] = {"name": f"drill_{mode}", "state": f"{d}/farm.json", "results": f"{d}/results",
                        "remote_dir": f"/tmp/krisd/fdrill_{mode}", "code_root": f"/tmp/krisd/drill_code_{mode}",
                        "only_jobs": [int(j) for j in opt["--jobs"].split(",") if j], "alloc": False, "bring_up": False}
    if mode == "staged":
        base["scheduling"] = {"mode": "staged", "seed": opt["--problems"].split(","), "gate_w": 1, "max_wait_min": 5,
                              "idle_max_min": 3}
    elif mode == "pooled":
        base["scheduling"] = {"mode": "pooled", "pool": {"max_problems": 2, "policy": "all"}}
    else:
        base["scheduling"] = {"mode": "independent"}
    cf = HERE / f"config_drill_{mode}.json"
    cf.write_text(json.dumps(base, indent=1))
    sd = L3 / d
    sd.mkdir(parents=True, exist_ok=True)
    if not (sd / "mb.tar.gz").exists():
        shutil.copyfile(L3 / "state" / "mb.tar.gz", sd / "mb.tar.gz")
    sf = sd / "farm.json"
    if reset or not sf.exists():
        sf.write_text(json.dumps({"problems": {n: dict(status="pending", attempts=0, node=None)
                                               for n in (p.rsplit(".", 1)[0] for p in
                                                         [q.name for q in sorted((L3 / "problems").glob("*.py"))])
                                               if int(n.split("_")[0]) in nums},
                                  "nodes": {}, "last_alloc": 0, "tree": STAMP}, indent=1))
    print("wrote", cf, sf)


main()
