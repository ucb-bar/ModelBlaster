#!/usr/bin/env python3
"""ip_run.py install JOB TARBALL | sh JOB 'script' (runs foreground, output printed) | bg JOB NAME 'script' | tail JOB NAME [n] | get JOB remote local
Node files under /tmp/krisd/mc only. my_jobs.json = merge_prep/my_jobs.json."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "merge_checks"))
import mc_common as C
C.MINE = HERE.parents[1] / "my_jobs3.json"
c, job = sys.argv[1], int(sys.argv[2])
if c == "install":
    ti = C.tree_info(sys.argv[3])
    print(C.put(job, ti["path"], f"{C.NODE_ROOT}/{ti['label']}.tar.gz")[0])
    rc, o, e = C.rfile(job, "mc_install.sh", [ti["label"], ti["sha256"], ti["stamp"]], timeout=900)
    print(o, e[-500:])
elif c == "sh":
    rc, o, e = C.rsh(job, sys.argv[3], timeout=1700); print(o); print(e[-2000:], file=sys.stderr)
elif c == "bg":
    n, s = sys.argv[3], sys.argv[4]
    C.put(job, s.encode(), f"/tmp/krisd/mc/ip/{n}.sh")
    C.check_mine(job)
    cmd = (f"nohup setsid srun --jobid={job} --overlap bash -c 'bash /tmp/krisd/mc/ip/{n}.sh > /tmp/krisd/mc/ip/{n}.log 2>&1' "
           f"</dev/null >/dev/null 2>&1 & sleep 5; echo started")
    rc, out, err = C.farm().ssh(cmd, b"", timeout=120); print(out.decode(), err.decode()[-200:])
elif c == "tail":
    rc, o, e = C.rsh(job, f"tail -n {sys.argv[4] if len(sys.argv)>4 else 30} /tmp/krisd/mc/ip/{sys.argv[3]}.log"); print(o)
