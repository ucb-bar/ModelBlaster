#!/usr/bin/env python3
"""Create / reset a barrier-scheduling test farm instance (X-151): config_bar_<tag>.json + i_<tag>/ (state, the frozen tarball, results)
and a NEW, empty global-knowledge store store_<tag>/ (the live store is never used). Stub LLM (stub:hw), no relay traffic.
  gen_bar.py TAG [--mode independent|pooled] [--problems 1,2,3,4,39,40] [--jobs J1,J2,J3] [--rounds 2] [--rollouts 2] [--steps 4]
             [--min-frac 1.0] [--max-wait 20] [--halfway on|off] [--barrier on|off] [--phase2 longest_first|rank|file] [--max-problems N]
             [--reset]
Own state dir, results dir, remote dir /tmp/krisd/bar_<tag>, code root /tmp/krisd/bar_code (frozen tree sha256 in frozen/)."""
import json, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
L3 = HERE.parents[1]


def main():
    a = sys.argv[1:]
    tag = a.pop(0)
    opt = {"--mode": "independent", "--problems": "1,2,3,4,39,40", "--jobs": "", "--rounds": "2", "--rollouts": "2", "--steps": "4",
           "--min-frac": "1.0", "--max-wait": "20", "--halfway": "on", "--barrier": "on", "--phase2": "longest_first",
           "--max-problems": "3", "--tick": "45", "--charsync": "off"}
    reset = "--reset" in a
    while a:
        x = a.pop(0)
        if x in opt:
            opt[x] = a.pop(0)
    base = json.loads((L3 / "smoke_sched" / "config_b.json").read_text())
    base.pop("_notes", None)
    nums = [int(p) for p in opt["--problems"].split(",")]
    base.update(include=nums, sync_min=3, tick_s=int(opt["--tick"]), max_attempts=2, problem_timeout_h=2)
    base["kb"].update(model="stub:hw", rounds=int(opt["--rounds"]), rollouts=int(opt["--rollouts"]), steps=int(opt["--steps"]),
                      variant_steps=1, max_variants=1,
                      extra="--plugin /tmp/krisd/bar/hw_plugin.py --torch-baseline off --graph-characterize 1 --graph-char-top 8")
    d = f"analysis/barrier_data/i_{tag}"
    base["global_knowledge"] = {"enabled": True, "path": str(HERE / f"store_{tag}"), "mode": "readwrite", "merge_on": "round",
                                "refresh": "round", "halfway": opt["--halfway"] == "on", "char_sync": opt["--charsync"] == "on", "halfway_poll_s": 15,
                                "halfway_wait_s": 180, "merge_failed": True, "dtype": "fp16", "snapshot_keep": 0,
                                "python": "python3",
                                "policy": {"share": "strict", "seed_max_claims": 1500, "seed_max_open_hyp": 150,
                                           "seed_max_closed_hyp": 150, "stale_merges": 100, "half_life_merges": 100,
                                           "max_active_claims": 4000, "max_local_claims_per_problem": 400,
                                           "impl_min_speedup": 1.02, "seed_fresh_merges": 10, "seed_fresh_max": 400}}
    base["instance"] = {"name": f"bar_{tag}", "state": f"{d}/farm.json", "results": f"{d}/results",
                        "remote_dir": f"/tmp/krisd/bar_{tag}", "code_root": "/tmp/krisd/bar_code",
                        "only_jobs": [int(j) for j in opt["--jobs"].split(",") if j], "alloc": False, "bring_up": False}
    sched = {"mode": opt["--mode"]}
    if opt["--mode"] == "pooled":
        sched["pool"] = {"max_problems": int(opt["--max-problems"]), "policy": "all"}
    if opt["--barrier"] == "on":
        sched["barrier"] = {"enabled": True, "after": "char", "min_frac": float(opt["--min-frac"]),
                            "max_wait_min": float(opt["--max-wait"]), "phase1_order": "cheap_first", "phase2_order": opt["--phase2"]}
    base["scheduling"] = sched
    cf = HERE / f"config_bar_{tag}.json"
    cf.write_text(json.dumps(base, indent=1))
    sd = L3 / d
    sd.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / "frozen" / "mb_bar.tar.gz", sd / "mb.tar.gz")
    stamp = "bar" + (HERE / "frozen" / "mb_bar.sha256").read_text()[:12]
    sf = sd / "farm.json"
    if reset or not sf.exists():
        names = [p.stem for p in sorted((L3 / "problems").glob("*.py")) if int(p.name.split("_")[0]) in nums]
        sf.write_text(json.dumps({"problems": {n: dict(status="pending", attempts=0, node=None) for n in names},
                                  "nodes": {}, "last_alloc": 0, "tree": stamp}, indent=1))
        shutil.rmtree(HERE / f"store_{tag}", ignore_errors=True)
        shutil.rmtree(sd / "results", ignore_errors=True)
    print("wrote", cf, sf, "tree", stamp)


main()
