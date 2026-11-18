#!/usr/bin/env python3
"""fetch node results into layout_stage1_data: fetch.py JOB remote_path... (small text files only, via farm.ssh)"""
import sys, shlex
from pathlib import Path
L3 = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(L3))
import farm
job = sys.argv[1]
for rp in sys.argv[2:]:
    sub = "walker_out" if "/out/" in rp and rp.endswith(".json") and "bench" not in rp and "correct" not in rp else "kernel_out"
    d = Path(__file__).resolve().parent / sub; d.mkdir(exist_ok=True)
    rc, out, err = farm.ssh(f"srun --jobid={job} --overlap cat {shlex.quote(rp)}", b"", timeout=300)
    (d / Path(rp).name).write_bytes(out); print(rp, rc, len(out))
