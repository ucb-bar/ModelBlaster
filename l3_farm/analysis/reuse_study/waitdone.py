import sys, os, re, time
sys.path.insert(0, os.path.dirname(__file__))
import rs, reuse_run
n = int(sys.argv[1]); 
while True:
    tot = 0
    for j in rs.JOBS:
        rc, o, e = rs.run(j, f"cd {rs.R}/runs 2>/dev/null && for d in *; do [ -f $d/torch_3.txt ] && echo done $d; done; grep -c WORKER_DONE {rs.R}/logs/worker_*.log")
        done = set(re.findall(r"done (\S+)", o)); tot += len(done)
        new = [p for p in done if not (reuse_run.HERE / "data" / "runs" / p / "torch_3.txt").exists()]
        if new: reuse_run.fetch(j, new)
    print(time.strftime("%H:%M:%S"), "done total", tot, flush=True)
    if tot >= n: break
    time.sleep(150)
