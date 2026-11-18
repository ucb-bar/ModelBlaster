"""Eager / CUDA-graph torch baselines for the L1+L2 L40S sweep's finished problems (native sizes, fp16), on ONE idle pool GPU under the sweep's per-GPU flock.
  FARM_CONFIG=analysis/l12_sweep/config_l12.json python3 analysis/l12_sweep/run_posthoc.py JOB [NAME ...]   (no names: every done problem without a result yet)"""
import json, os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, "analysis/l12_sweep")
import torch_posthoc_l12 as T
job = int(sys.argv[1]); names = sys.argv[2:]
if names == ["ALL"] or not names:      # ALL: every problem of the sweep (the baseline needs only the problem source, not the flow's result); SHARD=i/n splits the list
    s = json.load(open("state/l12/farm.json"))
    names = [n for n, p in s["problems"].items() if (names == ["ALL"] or p["status"] == "done") and not (T.RES / n / "torch_posthoc.json").exists()]
    if os.environ.get("SHARD"):
        i, k = map(int, os.environ["SHARD"].split("/")); names = names[i::k]
a = types.SimpleNamespace(iters=50, warm=10, no_compile=True, timeout=900)
print(len(names), "problems on job", job, flush=True)
for n in names:
    try: T.run_one(job, n, a)
    except SystemExit as e: print(n, "skipped:", e, flush=True)
