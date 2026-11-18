#!/usr/bin/env python3
"""Prepare (NOT launch) the KernelBench Level 1 + Level 2 sweeps on Computelab L40S with the best validated flow:
  * problems: qrb_a6/problems_l1 / problems_l2 (KernelBench, gh api) staged as l3_farm/problems/61NNN_<name>.py (L1) and 62NNN_<name>.py (L2);
  * flow = A6 (round controller auto, merge-sibling auto, region windows auto, region loop on, rounds 3) + --region-slots all --clock-guard on (what the L40S ablation validated);
  * scheduling = pooled (one driver node runs every controller + one shared LLM relay; a small pool of L40S does the timed runs under the per-GPU flock) so the GPUs are not idle while a problem waits for the LLM;
  * extraction sizes: native KernelBench sizes by default (as the L3 sweep); optional --cap-mb N shrinks the problems whose io exceeds N MiB (worker5.sh reads BENCH_TARGET_MB, default 0 = native);
  * global knowledge: fresh technique-only store (policy of the L3 headline sweep), readwrite, merge/refresh per round.
usage: make_l12.py [--cap-mb 0] [--max-nodes 10] [--max-problems 16] [--only-level 1|2]   -> config_l12.json, l12_map.json (plus the staged problem files)"""
import argparse, json, re, shutil
from pathlib import Path
H = Path(__file__).resolve().parent; F = H.parents[1]; Q = F / "qrb_a6"
ap = argparse.ArgumentParser()
ap.add_argument("--cap-mb", type=int, default=0, help="0 = native KernelBench sizes for every problem (default); N = shrink problems whose io exceeds N MiB to --bench-target-mb N"); ap.add_argument("--max-nodes", type=int, default=14); ap.add_argument("--max-problems", type=int, default=19)
ap.add_argument("--only-level", type=int, default=0)
a = ap.parse_args()
sizes = json.load(open(H / "native_sizes.json"))
base = json.load(open(F / "analysis/headline/config_headline.json"))
abl = json.load(open(F / "analysis/ablation3/config.json"))
names, cap_of, amap = [], {}, {}
for lvl in (1, 2):
    if a.only_level and lvl != a.only_level: continue
    for f in sorted((Q / f"problems_l{lvl}").glob("*.py"), key=lambda p: int(p.name.split("_")[0])):
        n = int(f.name.split("_")[0]); alias = f"6{lvl}{n:03d}"; stem = f.stem.split("_", 1)[1]; name = f"{alias}_{stem}"
        shutil.copyfile(f, F / "problems" / f"{name}.py"); names.append(name)
        io = sizes[str(lvl)][f.stem].get("io_mb", 0); cap_of[name] = a.cap_mb if (a.cap_mb and io > a.cap_mb) else 0
        amap[name] = {"level": lvl, "kernelbench_number": n, "native_io_mb": io, "extract_cap_mb": cap_of[name]}
cfg = json.loads(json.dumps(base))
for k in ("_notes",): cfg.pop(k, None)
cfg["_notes"] = ("KernelBench L1 + L2 on L40S, prepared 2026-10-29, NOT launched. A6 flow (validated on the L40S ablation: round controller auto + merge-sibling auto + region windows auto + region loop on, rounds 3, plus "
                 "--region-slots all --clock-guard on). Pooled scheduling without barrier (the barrier needs the interleaved characterization pass, which --round-controller refuses). See analysis/l12_sweep/README.txt.")
cfg["include"] = sorted(int(n.split("_")[0]) for n in names); cfg["exclude"] = []
cfg["max_nodes"] = a.max_nodes; cfg["problem_timeout_h"] = 6; cfg["max_attempts"] = 2
cfg["ignore_jobs"] = [5282436]
cfg["instance"] = {"name": "l12", "state": "state/l12/farm.json", "results": "results_l12", "remote_dir": "/tmp/krisd/farm_l12", "code_root": "/tmp/krisd/l12_code", "alloc": True, "bring_up": True}
cfg["scheduling"] = {"mode": "pooled", "pool": {"policy": "share", "order": "file", "cuda_visible_devices": "all", "max_problems": a.max_problems}}
kb = cfg["kb"]
for k, v in abl["kb"].items():
    if k not in ("problem_env", "problem_flags", "extra", "arch"): kb[k] = v
kb["arch"] = 89; kb["rounds"] = 3; kb["rollouts"] = 3; kb["steps"] = 4; kb["variant_steps"] = 2; kb["max_variants"] = 3
kb.pop("headline_config", None); kb.pop("hetero_policy", None); kb["extra"] = "--region-slots all --clock-guard on"
cfg["relay"] = {"concurrency": 192, "queue": 2048, "budget_usd": 4000, "enforce": True, "check_min": 10}   # the L3 headline (one shared driver relay at 96) queued 21.6% of its LLM latency; B200 per-host relays ~0%
A6 = "--round-controller auto --round-controller-budget-usd 40 --merge-sibling-convs auto --region-windows auto --region-loop on --region-loop-budget 40,120,4,3600 --rounds 3"
kb["problem_flags"] = {n: A6 for n in names}
kb["problem_env"] = {n: ({"BENCH_TARGET_MB": str(cap_of[n])} if cap_of[n] else {}) for n in names}
gk = cfg["global_knowledge"]; gk["path"] = str(F.parent / "knowledge_global_l12"); gk["enabled"] = True; gk["mode"] = "readwrite"; gk["merge_on"] = "round"; gk["refresh"] = "round"; gk["halfway"] = False
(F.parent / "knowledge_global_l12").mkdir(exist_ok=True)
(H / "config_l12.json").write_text(json.dumps(cfg, indent=1)); (H / "l12_map.json").write_text(json.dumps(amap, indent=1))
capped = sum(1 for v in cap_of.values() if v)
TAR = F / "analysis/ablation3/mb_fix3.tar.gz"; info = json.loads(Path(str(TAR) + ".info.json").read_text())
sd = F / "state/l12"; sd.mkdir(parents=True, exist_ok=True); shutil.copyfile(TAR, sd / "mb.tar.gz")
(sd / "farm.json").write_text(json.dumps({"problems": {n: {"status": "pending", "attempts": 0, "node": None} for n in names}, "nodes": {}, "last_alloc": 0, "tree": info["stamp"]}, indent=1))
print(f"{len(names)} problems staged ({sum(1 for n in names if n.startswith('61'))} L1, {sum(1 for n in names if n.startswith('62'))} L2); {capped} capped at {a.cap_mb} MiB, {len(names) - capped} at native size")
print("pooled: max_nodes", a.max_nodes, "max_problems", a.max_problems, "| flags:", A6, "+", kb["extra"])
