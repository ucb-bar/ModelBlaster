#!/usr/bin/env python3
"""Fetch files from one of my nodes: fetch.py JOB LOCAL_DIR REMOTE_PATH... (tar over ssh -> srun; /tmp/krisd/cl only)."""
import io, shlex, sys, tarfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import farm  # noqa: E402
MINE = {5298142, 5298144, 5298547}
job, local, *paths = sys.argv[1:]
assert int(job) in MINE
assert all(p.startswith("/tmp/krisd/cl/") for p in paths)
rel = " ".join(shlex.quote(p[len("/tmp/krisd/cl/"):]) for p in paths)
rc, out, err = farm.ssh(f"srun --jobid={job} --overlap bash -c {shlex.quote(f'cd /tmp/krisd/cl && tar czf - --ignore-failed-read {rel} 2>/dev/null')}",
                        b"", timeout=600)
Path(local).mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(out), mode="r:gz") as t:
    t.extractall(local)
print("fetched", len(out), "bytes", err.decode()[-200:])
