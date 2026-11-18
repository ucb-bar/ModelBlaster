#!/usr/bin/env python3
"""KernelBench L3 on B200, pilot: the 8 problems of the L40S ablation (7 8 14 15 18 34 46 47) x 2 arms (A0 plain flow, A6 combo), fp16 native sizes like the L40S sweep.
Local files only (no ssh, no key). Writes instance/ ; names 97<NNN>_<Name> = A0, 98<NNN>_<Name> = A6. Settings = analysis/ablation3/config.json (Opus 5.5 roles, haiku simple/fix,
rounds 2 x rollouts 3 x steps 4 for A0; A6 = round controller auto + merge-sibling + region windows + region loop, rounds 3), gk off (arms must not share knowledge).
Differences from the L40S ablation (arch): arch 100, no --clock-guard (L40S clock pinning), ssh backend with GPU sharing (3 per GPU: the slot-2 claims of free capacity)."""
import json, shutil, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; FARM = HERE.parents[1]
NUMS = [7, 8, 14, 15, 18, 34, 46, 47]
abl = json.loads((FARM / "analysis/ablation3/config.json").read_text())
base = json.loads((FARM / "pilot_sol/a6large_instance/config.json").read_text())
tar = FARM / "analysis/ablation3/mb_fix3.tar.gz"; info = json.loads(Path(str(tar) + ".info.json").read_text())
root = HERE / "instance"; shutil.rmtree(root, ignore_errors=True)
for d in ("state", "problems", "problems_torch", "results"): (root / d).mkdir(parents=True)
shutil.copyfile(tar, root / "state/mb.tar.gz")
for f in ("farm_sched.py", "farm_defaults.py"): (root / f).symlink_to(FARM / f)
(root / "remote").symlink_to(FARM / "remote")
src = {int(p.name.split("_")[0]): p for p in (FARM / "problems").glob("*.py") if p.name.split("_")[0].isdigit() and int(p.name.split("_")[0]) <= 50}
EXTRA = "--region-slots all"
A6 = "--round-controller auto --round-controller-budget-usd 40 --merge-sibling-convs auto --region-windows auto --region-loop on --region-loop-budget 40,120,4,3600 --rounds 3"
names, flags, env, arm_of = [], {}, {}, {}
seed = "/raid/krisd/mb/seeds/strict_mergedtree"
for arm, pre, fl in ((0, 97, EXTRA), (6, 98, A6 + " " + EXTRA)):
    for n in NUMS:
        s = src[n]; stem = s.stem.split("_", 1)[1]; name = f"{pre}{n:03d}_{stem}"
        shutil.copyfile(s, root / "problems" / (name + ".py"))
        names.append(name); flags[name] = fl; arm_of[name] = f"A{arm}"
        env[name] = {"QUANT": "fp16", "DTYPE_TAG": "fp16", "KB_KNOWLEDGE_SEED_DIR": seed}
cfg = json.loads(json.dumps(base))
k = abl["kb"]
cfg["kb"].update({x: k[x] for x in ("model", "rounds", "rollouts", "steps", "variant_steps", "max_variants", "min_gain_pct", "plan_top_k", "plan_min_share", "model_simple", "fix_model" if "fix_model" in k else "model_simple", "model_hard", "reflect_model", "upkeep_model") if x in k})
cfg["kb"]["arch"] = 100; cfg["kb"]["problem_env"] = env; cfg["kb"]["problem_flags"] = flags
cfg["global_knowledge"] = {**base.get("global_knowledge", {}), "enabled": False, "mode": "off"}
cfg["relay"] = {**base["relay"], "budget_usd": 1500}
cfg["max_nodes"] = len(names); cfg["problem_timeout_h"] = 6
cfg["ssh"] = {**base["ssh"], "share_per_gpu": 3, "share_mem_pct": 60, "wall_limit_h": 10, "max_gpus_per_host": 8, "claim_batch": 4}
cfg["instance"] = {**base["instance"], "name": "l3pilot", "remote_dir": "/raid/krisd/mb/farm/l3pilot"}
(root / "config.json").write_text(json.dumps(cfg, indent=1))
(root / "arms.json").write_text(json.dumps(arm_of, indent=1))
(root / "state/farm.json").write_text(json.dumps({"problems": {n: {"status": "pending", "attempts": 0, "node": None} for n in names}, "nodes": {}, "last_alloc": 0, "tree": info["stamp"]}))
print("wrote", root, len(names), "problems; model", cfg["kb"]["model"], "rounds", cfg["kb"]["rounds"], "rollouts", cfg["kb"]["rollouts"], "steps", cfg["kb"]["steps"], "| A0 flags:", EXTRA, "| A6 flags:", A6, EXTRA)
print("max_nodes", cfg["max_nodes"], "share", cfg["ssh"]["share_per_gpu"])
