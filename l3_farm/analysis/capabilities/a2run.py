#!/usr/bin/env python3
"""Run a bash script on one of MY nodes (A2 capabilities agent: jobs in a2_jobs.json) via farm.ssh -> srun --overlap.

  nrun.py JOB SCRIPT.sh [timeout_s] [--put LOCAL=REMOTE ...] [--detach]

Files given with --put are shipped first (farm.put: streamed over ssh into srun cat). Work dir on the node:
/tmp/krisd/a2 (never the farm's /tmp/krisd/farm/<name> dirs). --detach: the script runs as a nohup'd srun step
(output in /tmp/krisd/a2/logs/<script>.log), and this returns at once.
"""
import shlex, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import farm  # noqa: E402

import json
MINE = {int(j) for j in json.loads((Path(__file__).resolve().parent / "a2_jobs.json").read_text())}
args = sys.argv[1:]
job = int(args.pop(0))
assert job in MINE, f"{job} is not one of my nodes {MINE}"
script = Path(args.pop(0))
timeout = int(args.pop(0)) if args and args[0].isdigit() else 900
puts, detach = [], False
while args:
    a = args.pop(0)
    if a == "--put":
        loc, rem = args.pop(0).split("=", 1)
        puts.append((loc, rem))
    elif a == "--detach":
        detach = True
for loc, rem in puts:
    assert rem.startswith("/tmp/krisd/a2/"), rem
    rc, out, err = farm.put(job, Path(loc), rem)
    if rc:
        sys.exit(f"put {loc} failed: {err.decode()[-300:]}")
pre = "mkdir -p /tmp/krisd/a2/logs; cd /tmp/krisd/a2\n"
if detach:
    rem = f"/tmp/krisd/a2/scripts/{script.name}"
    farm.put(job, script, rem)
    log = f"/tmp/krisd/a2/logs/{script.stem}.log"
    cmd = (f"nohup setsid srun --jobid={job} --overlap bash -c {shlex.quote(pre + f'bash {rem} > {log} 2>&1')} "
           f"</dev/null >/dev/null 2>&1 & sleep 5; echo started")
    rc, out, err = farm.ssh(cmd, b"", timeout=120)
    print(out.decode(), err.decode()[-500:], "rc", rc)
    sys.exit(0)
rc, out, err = farm.ssh(f"srun --jobid={job} --overlap bash -s", (pre + script.read_text() + "\n").encode(),
                        timeout=timeout)
sys.stdout.write(out.decode(errors="replace"))
sys.stdout.write(err.decode(errors="replace")[-3000:])
print("rc", rc)
