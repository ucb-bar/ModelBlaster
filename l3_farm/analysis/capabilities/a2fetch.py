#!/usr/bin/env python3
"""Fetch small result files from my node: a2fetch.py JOB REMOTE_GLOB... -> results/<job>/<path under /tmp/krisd/a2>"""
import io, json, shlex, sys, tarfile
from pathlib import Path
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parents[1]))
import farm  # noqa: E402
MINE = {int(j) for j in json.loads((H / "a2_jobs.json").read_text())}
job, globs = int(sys.argv[1]), sys.argv[2:]
assert job in MINE
cmd = (f"srun --jobid={job} --overlap bash -c " + shlex.quote(
    "cd /tmp/krisd/a2 && tar czf - --ignore-failed-read $(ls -d " + " ".join(globs) + " 2>/dev/null) 2>/dev/null"))
rc, out, err = farm.ssh(cmd, b"", timeout=600)
dst = H / "results" / str(job)
dst.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(out), mode="r:gz") as t:
    for m in t.getmembers():                      # plain files / dirs inside dst only
        if (m.isfile() or m.isdir()) and not m.name.startswith("/") and ".." not in Path(m.name).parts:
            t.extract(m, dst)
print("fetched into", dst, len(out), "bytes")
