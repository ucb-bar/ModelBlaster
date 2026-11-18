#!/usr/bin/env python3
"""Push a staged tarball to my node(s) and install it: a2push.py STAMP JOB [JOB...]"""
import json, sys
from pathlib import Path
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parents[1]))
import farm  # noqa: E402
MINE = {int(j) for j in json.loads((H / "a2_jobs.json").read_text())}
stamp, jobs = sys.argv[1], [int(j) for j in sys.argv[2:]]
for j in jobs:
    assert j in MINE, j
    rc, out, err = farm.put(j, H / "stage" / f"mb_a2_{stamp}.tar.gz", f"/tmp/krisd/a2/mb_{stamp}.tar.gz")
    assert rc == 0, err.decode()[-300:]
    farm.put(j, H / "env.sh", "/tmp/krisd/a2/env.sh")
    sh = f"STAMP={stamp}\n" + (H / "install.sh").read_text()
    rc, out, err = farm.ssh(f"srun --jobid={j} --overlap bash -s", sh.encode(), timeout=600)
    print(j, out.decode()[-400:], err.decode()[-400:])
