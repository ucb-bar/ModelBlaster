#!/usr/bin/env python3
"""Write the ablation-3 farm instance files (local only: no ssh, no key, no spend).

  make_abl3.py full  [--jobs J1 J2 ...]   config.json (56 aliases = 7 arms x 8 problems), abl3_map.json, state/abl3/{farm.json,mb.tar.gz}
  make_abl3.py smoke [--jobs J1 J2]       config_smoke.json (8_ResNetBasicBlock x A0,A1,A3,A4,A5, ONE small round), state/abl3smoke/...

Alias number = 11000 + 1000*arm + problem number (arm 0..6 = A0..A6), name = <number>_<problem name>; problems/<alias>.py is a copy of the
original. Instance (own state / results / remote dir / code root, alloc false, only_jobs = the jobs given): the leftover dispatchers of
earlier experiments never see these nodes. Global knowledge OFF; KB_KNOWLEDGE_SEED_DIR is NOT set (the tree's default seed = the L40S one).
"""
import argparse, json, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
FARM = HERE.parent.parent
TAR = FARM / "pilot_sol" / "mb2.tar.gz"

PROBLEMS = ["34_VanillaRNNHidden", "18_SqueezeNet", "14_DenseNet121DenseBlock", "46_NetVladWithGhostClusters",
            "47_NetVladNoGhostClusters", "8_ResNetBasicBlock", "15_DenseNet121", "7_GoogleNetInceptionV1"]
START_ORDER = ["18", "15", "46", "34", "14", "47", "8", "7"]          # longest first (headline wall: 18 ~4 h, 15 1.4 h, 46 1.4 h, rest ~1 h)
LOOP_BUDGET = "40,120,4,3600"      # --region-loop-budget turns,tools,usd,wall_s: PER LOOP (a loop runs in every KB run that has a region slot)
RC_BUDGET = 40
ARMS = {   # arm -> (label, extra kb_graph_opt flags)
    "A0": ("baseline: every new flag off", ""),
    "A1": ("round controller auto", f"--round-controller auto --round-controller-budget-usd {RC_BUDGET}"),
    "A2": ("round controller graph-first (+1 round)", "--round-controller graph-first --rounds 3"),
    "A3": ("merge sibling convs", "--merge-sibling-convs auto"),
    "A4": ("region windows", "--region-windows auto"),
    "A5": ("region loop", f"--region-loop on --region-loop-budget {LOOP_BUDGET}"),
    "A6": ("combo", f"--round-controller auto --round-controller-budget-usd {RC_BUDGET} --merge-sibling-convs auto --region-windows auto "
                    f"--region-loop on --region-loop-budget {LOOP_BUDGET}"),
}
# flow-side PyTorch baseline (kb_model_run --torch-baseline auto): torch python of the node venv (setup.sh installs torch); the source is the
# worker's MB_TORCH_SRC=$D/src/<alias>.py (the KernelBench reference); torch runs in the IR dtype (fp16 -> torch half)
# TRITON/INDUCTOR caches: the smoke showed torch.compile (compile_mna / compile_ro baseline variants) failing with ENOSPC in $HOME/.triton (home quota full) -> /tmp
TORCH_ENV = {"MB_TORCH_PYTHON": "/tmp/krisd/venv/bin/python", "TRITON_CACHE_DIR": "/tmp/krisd/cache/triton", "TORCHINDUCTOR_CACHE_DIR": "/tmp/krisd/cache/inductor"}
COMMON = "--region-slots all"          # all arms (incl. A0): the production L40S flow's setting; without region ops --region-loop is inert (see README)


def alias(arm_i, prob):
    n, rest = prob.split("_", 1)
    return f"{11000 + 1000 * arm_i + int(n)}_{rest}"


def base_cfg(name, remote, budget, rounds, nodes, jobs, include, ptimeout=6):
    return {
        "host": "computelab", "partition_match": "l40s", "alloc_query": 'gpu.product_name="*L40S*"', "alloc_time": "10:00:00",
        "max_nodes": nodes, "ignore_jobs": [5282436], "min_left_min": 60, "problem_timeout_h": ptimeout, "max_attempts": 2,
        "sync_min": 15, "tick_s": 90, "level": 3, "include": include, "exclude": [], "alloc_gap_s": 60,
        "relay": {"concurrency": 96, "queue": 1024, "budget_usd": budget, "enforce": True, "check_min": 10},
        "kb": {"model": "azure/anthropic/claude-opus-5-5", "rounds": rounds[0], "rollouts": rounds[1], "steps": rounds[2], "variant_steps": rounds[3],
               "max_variants": 3, "min_gain_pct": 0.5, "arch": 89, "plan_top_k": 32, "plan_min_share": 0.002,
               "model_simple": "azure/anthropic/claude-haiku-5-5", "fix_model": "azure/anthropic/claude-haiku-5-5",
               "model_hard": "azure/anthropic/claude-opus-5-5", "reflect_model": "azure/anthropic/claude-opus-5-5",
               "upkeep_model": "azure/anthropic/claude-opus-5-5", "extra": COMMON, "problem_flags": {}, "problem_env": {}},
        "global_knowledge": {"enabled": False, "mode": "off"},
        "scheduling": {"mode": "independent"},
        "instance": {"name": name, "state": f"state/{name}/farm.json", "results": f"results_{name}", "remote_dir": f"/tmp/krisd/{name}",
                     "code_root": f"/tmp/krisd/{name}_code", "alloc": False, "bring_up": True, "only_jobs": jobs},
    }


def write_state(name, names):
    info = json.loads(Path(str(TAR) + ".info.json").read_text())
    sd = FARM / "state" / name
    sd.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TAR, sd / "mb.tar.gz")
    (sd / "farm.json").write_text(json.dumps({"problems": {n: {"status": "pending", "attempts": 0, "node": None} for n in names},
                                              "nodes": {}, "last_alloc": 0, "tree": info["stamp"]}, indent=1))
    return info


def copy_problems(names, orig):
    for n in names:
        shutil.copyfile(FARM / "problems" / (orig[n] + ".py"), FARM / "problems" / (n + ".py"))
        pt = FARM / "problems_torch" / (orig[n] + ".py")
        if pt.exists():
            shutil.copyfile(pt, FARM / "problems_torch" / (n + ".py"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["full", "smoke"])
    ap.add_argument("--jobs", nargs="*", type=int, default=[])
    ap.add_argument("--nodes", type=int, default=12)
    ap.add_argument("--budget-total", type=float, default=2500.0)
    a = ap.parse_args()
    if a.which == "full":
        order = sorted(PROBLEMS, key=lambda p: START_ORDER.index(p.split("_")[0]))
        names, amap = [], {}
        for p in order:
            for i, arm in enumerate(ARMS):
                n = alias(i, p)
                names.append(n)
                amap[n] = {"arm": arm, "problem": p, "flags": (COMMON + " " + ARMS[arm][1]).strip()}
        cfg = base_cfg("abl3", None, round(a.budget_total / a.nodes, 2), (2, 3, 4, 2), a.nodes, a.jobs, sorted(int(n.split("_")[0]) for n in names))
        cfg["kb"]["problem_flags"] = {n: ARMS[amap[n]["arm"]][1] for n in names}
        cfg["kb"]["problem_env"] = {n: dict(TORCH_ENV) for n in names}
        orig = {n: amap[n]["problem"] for n in names}
        copy_problems(names, orig)
        info = write_state("abl3", names)
        cfg["_notes"] = ("ablation 3 (2026-10-28): 7 arms x 8 problems of the NEW flow mechanisms on L40S, tree pilot-sol " + info["stamp"] +
                         "; independent scheduling, global knowledge off, default seed; common flag " + COMMON + "; see analysis/ablation3/README.txt")
        (HERE / "config.json").write_text(json.dumps(cfg, indent=1))
        (HERE / "abl3_map.json").write_text(json.dumps({"arms": {k: {"label": v[0], "flags": (COMMON + " " + v[1]).strip(),
                                                         "problem_flags": v[1]} for k, v in ARMS.items()}, "names": amap}, indent=1))
        print("wrote", HERE / "config.json", len(names), "aliases | per-node relay budget", cfg["relay"]["budget_usd"], "| tree", info["stamp"])
    else:
        prob = "8_ResNetBasicBlock"
        smoke_arms = ["A5", "A4", "A3", "A1", "A0"]
        names, amap, flags = [], {}, {}
        small = "--rounds 1 --rollouts 2 --steps 2 --variant-steps 1"
        for arm in smoke_arms:
            i = list(ARMS).index(arm)
            n = alias(i, prob)
            names.append(n)
            f = ARMS[arm][1].replace(LOOP_BUDGET, "20,60,3,1800").replace("auto --round-controller-budget-usd 40", "auto --round-controller-budget-usd 6")
            flags[n] = (f + " " + small).strip()
            amap[n] = {"arm": arm, "problem": prob, "flags": (COMMON + " " + flags[n]).strip()}
        cfg = base_cfg("abl3smoke", None, 30.0, (1, 2, 2, 1), 2, a.jobs, sorted(int(n.split("_")[0]) for n in names), ptimeout=2)
        cfg["kb"]["problem_flags"] = flags
        cfg["kb"]["problem_env"] = {n: dict(TORCH_ENV) for n in names}
        cfg["instance"]["code_root"] = "/tmp/krisd/abl3_code"
        cfg["alloc_time"] = "3:00:00"
        copy_problems(names, {n: prob for n in names})
        write_state("abl3smoke", names)
        (HERE / "config_smoke.json").write_text(json.dumps(cfg, indent=1))
        (HERE / "smoke_map.json").write_text(json.dumps(amap, indent=1))
        print("wrote", HERE / "config_smoke.json", names)


if __name__ == "__main__":
    main()
