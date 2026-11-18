#!/usr/bin/env python3
"""scancel ONLY the jobs listed in my_jobs.json (farm.ssh helper; inside a python file as required)."""
import json, sys
from pathlib import Path
L3 = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(L3))
import farm
mine = json.loads((Path(__file__).with_name("my_jobs.json")).read_text())
for j in mine:
    rc, out, err = farm.ssh(f"scancel {int(j)}", b"", timeout=60)
    print("scancel", j, rc, out.decode().strip(), err.decode().strip())
rc, out, err = farm.ssh("squeue --me -h -o '%i|%T|%N'", b"", timeout=60)
print(out.decode())
