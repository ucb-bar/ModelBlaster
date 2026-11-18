#!/usr/bin/env python3
"""Shows `farm.py sched plan` and `farm.py barrier status` for the headline config (16 and 48 excluded), on a TEMP state (never state/headline)."""
import json, os, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; L3 = HERE.parents[1]
cfg = json.loads((L3 / "analysis/headline/config_headline.json").read_text())
cfg["instance"]["state"] = "analysis/barrier_data/i_headline_plan/farm.json"
cfg["instance"]["results"] = "analysis/barrier_data/i_headline_plan/results"
(HERE / "i_headline_plan").mkdir(exist_ok=True)
names = [p.stem for p in sorted((L3 / "problems").glob("*.py"), key=lambda p: int(p.name.split("_")[0])) if int(p.name.split("_")[0]) not in map(int, cfg["exclude"])]
(HERE / "i_headline_plan/farm.json").write_text(json.dumps({"problems": {n: dict(status="pending", attempts=0, node=None) for n in names}, "nodes": {}, "last_alloc": 0, "tree": "x"}))
(HERE / "config_headline_plan.json").write_text(json.dumps(cfg))
env = dict(os.environ, FARM_CONFIG="analysis/barrier_data/config_headline_plan.json")
for sub in (["sched", "plan"], ["barrier", "status"]):
    print("$ farm.py", *sub); print(subprocess.run([sys.executable, "farm.py", *sub], cwd=L3, env=env, capture_output=True, text=True).stdout)
