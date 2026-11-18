"""Node helper for the reuse study: ONLY jobs 5325505 and 5325510, only /tmp/krisd/reuse_study. Needs FARM_CONFIG=analysis/headline/config_headline.json."""
import os, sys, shlex
from pathlib import Path
L3 = Path(__file__).resolve().parents[2]
os.environ.setdefault("FARM_CONFIG", "analysis/headline/config_headline.json")
sys.path.insert(0, str(L3))
import farm  # noqa: E402
JOBS = ["5325505", "5325510"]
R = "/tmp/krisd/reuse_study"


def run(job, script, timeout=600):
    assert str(job) in JOBS, "not one of my jobs"
    rc, out, err = farm.ssh(f"srun --jobid={job} --overlap bash -s", script.encode(), timeout=timeout)
    return rc, out.decode(errors="replace"), err.decode(errors="replace")


def put(job, local, remote):
    assert str(job) in JOBS
    rc, out, err = farm.put(job, local, remote)
    if rc:
        raise RuntimeError(err.decode()[-300:])


if __name__ == "__main__":
    for j in JOBS:
        rc, o, e = run(j, "hostname; nvidia-smi --query-gpu=name,clocks.mem,clocks.sm,memory.used,utilization.gpu --format=csv,noheader; nvidia-smi --query-compute-apps=pid --format=csv,noheader; ls -d /tmp/krisd/* | head -20; ls /tmp/krisd/headline_code /tmp/krisd/ModelBlaster 2>&1 | head -5; df -h /tmp | tail -1; nproc; echo $TMPDIR; /tmp/krisd/venv/bin/python -c 'import torch;print(torch.__version__)'")
        print(j, rc, o, e[-300:])
