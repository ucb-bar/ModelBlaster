import sys, os, re, subprocess
sys.path.insert(0, os.path.dirname(__file__))
import rs, reuse_run
D = reuse_run.HERE / "data" / "runs"
for j in rs.JOBS:
    rc, o, e = rs.run(j, f"cd {rs.R}/runs 2>/dev/null && for d in *; do [ -f $d/torch_3.txt ] && echo done $d; done; tail -2 {rs.R}/logs/worker_*.log | tail -4; date +%T")
    done = re.findall(r"done (\S+)", o)
    print(j, "done:", done, "|", o.strip().splitlines()[-3:])
    new = [p for p in done if not (D / p / "torch_3.txt").exists()]
    if new:
        reuse_run.fetch(j, new)
