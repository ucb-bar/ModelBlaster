"""Compile sweep driver (eager + CUDA graph + torch.compile default, same process). Results: analysis/l12_sweep/torch_compile/results/<stem>/torch_posthoc.json
  FARM_CONFIG=analysis/l12_sweep/config_l12.json [SHARD=i/n] python3 analysis/l12_sweep/torch_compile/run_compile.py JOB [STEM ...|ALL]
ALL = every stem of problems.txt without a result yet (resume-safe). Failed runs (srun timeout, rc!=0) leave no file and are retried on the next invocation;
a --retry pass is simply a rerun. Real failures (compile error/OOM) are recorded inside the json (compile_error) or, for a crashed bench, in results/<stem>/failed.txt."""
import os, sys, types
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
import torch_compile_l12 as T
job = int(sys.argv[1]); names = sys.argv[2:]
if names in (["ALL"], []):
    allp = [l.strip() for l in open(os.path.join(HERE, "problems.txt")) if l.strip()]
    names = [n for n in allp if not (T.RES / n / "torch_posthoc.json").exists()]
    if os.environ.get("SHARD"):
        i, k = map(int, os.environ["SHARD"].split("/")); names = allp_sh = [n for j, n in enumerate(allp) if j % k == i and n in names]
a = types.SimpleNamespace(iters=50, warm=10, no_compile=False, timeout=900)
print(len(names), "problems on job", job, flush=True)
for n in names:
    try: T.run_one(job, n, a)
    except SystemExit as e: print(n, "skipped:", e, flush=True)
    except Exception as e: print(n, "ERROR", repr(e)[:300], flush=True)
