#!/usr/bin/env python3
"""Create / reset a halfway-test farm instance (X-150): config_hw_<tag>.json + i_<tag>/ (state, the frozen tarball, results) and
a NEW, empty global-knowledge store store_<tag>/ (the live store is never used).
  gen_hw.py TAG [--halfway on|off] [--mode independent|staged|pooled] [--problems 1,2] [--jobs J1,J2] [--rounds 2]
            [--rollouts 2] [--steps 4] [--reset]
Own state dir, results dir, remote dir /tmp/krisd/hw_<tag>, code root /tmp/krisd/hw_code (frozen tree sha256 in frozen/)."""
import json, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
L3 = HERE.parents[1]


def main():
    a = sys.argv[1:]
    tag = a.pop(0)
    opt = {"--halfway": "on", "--mode": "independent", "--problems": "1,2", "--jobs": "", "--rounds": "2", "--rollouts": "2",
           "--steps": "4", "--poll": "15"}
    reset = "--reset" in a
    while a:
        x = a.pop(0)
        if x in opt:
            opt[x] = a.pop(0)
    base = json.loads((L3 / "smoke_sched" / "config_b.json").read_text())
    base.pop("_notes", None)
    nums = [int(p) for p in opt["--problems"].split(",")]
    base.update(include=nums, sync_min=3, tick_s=45, max_attempts=2, problem_timeout_h=2)
    base["kb"].update(model="stub:hw", rounds=int(opt["--rounds"]), rollouts=int(opt["--rollouts"]), steps=int(opt["--steps"]),
                      variant_steps=1, max_variants=1,
                      extra="--plugin /tmp/krisd/hw/hw_plugin.py --torch-baseline off")
    d = f"analysis/halfway_data/i_{tag}"
    base["global_knowledge"] = {"enabled": True, "path": str(HERE / f"store_{tag}"), "mode": "readwrite", "merge_on": "round",
                                "refresh": "round", "halfway": opt["--halfway"] == "on", "halfway_poll_s": int(opt["--poll"]),
                                "halfway_wait_s": 180, "merge_failed": True, "dtype": "fp16", "snapshot_keep": 0,
                                "python": "python3",
                                "policy": {"share": "strict", "seed_max_claims": 1500, "seed_max_open_hyp": 150,
                                           "seed_max_closed_hyp": 150, "stale_merges": 100, "half_life_merges": 100,
                                           "max_active_claims": 4000, "max_local_claims_per_problem": 400,
                                           "impl_min_speedup": 1.02, "seed_fresh_merges": 10, "seed_fresh_max": 400}}
    base["instance"] = {"name": f"hw_{tag}", "state": f"{d}/farm.json", "results": f"{d}/results",
                        "remote_dir": f"/tmp/krisd/hw_{tag}", "code_root": "/tmp/krisd/hw_code",
                        "only_jobs": [int(j) for j in opt["--jobs"].split(",") if j], "alloc": False, "bring_up": False}
    mode = opt["--mode"]
    if mode == "staged":
        base["scheduling"] = {"mode": "staged", "seed": [f"{n}" for n in nums[:1]], "gate_w": 1, "max_wait_min": 5, "idle_max_min": 3}
    elif mode == "pooled":
        base["scheduling"] = {"mode": "pooled", "pool": {"max_problems": 2, "policy": "all"}}
    else:
        base["scheduling"] = {"mode": "independent"}
    cf = HERE / f"config_hw_{tag}.json"
    cf.write_text(json.dumps(base, indent=1))
    sd = L3 / d
    sd.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / "frozen" / "mb_hw.tar.gz", sd / "mb.tar.gz")
    stamp = "hw" + (HERE / "frozen" / "mb_hw.sha256").read_text()[:12]
    sf = sd / "farm.json"
    if reset or not sf.exists():
        names = [p.stem for p in sorted((L3 / "problems").glob("*.py")) if int(p.name.split("_")[0]) in nums]
        sf.write_text(json.dumps({"problems": {n: dict(status="pending", attempts=0, node=None) for n in names},
                                  "nodes": {}, "last_alloc": 0, "tree": stamp}, indent=1))
        shutil.rmtree(HERE / f"store_{tag}", ignore_errors=True)
        shutil.rmtree(sd / "results", ignore_errors=True)
    print("wrote", cf, sf, "tree", stamp)


main()
