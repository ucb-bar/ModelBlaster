#!/usr/bin/env python3
"""scancel ONE of my own jobs (must be in my_jobs.json): scancel_one.py JOB"""
import json, sys
from pathlib import Path
L3 = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(L3))
import farm
j = sys.argv[1]
assert j in json.loads(Path(__file__).with_name("my_jobs.json").read_text()), "not my job"
rc, out, err = farm.ssh(f"scancel {int(j)}", b"", timeout=60)
print("scancel", j, rc, out.decode().strip(), err.decode().strip())
