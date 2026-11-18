#!/usr/bin/env python3
"""E0: request N L40S allocations exactly as farm.alloc does (same crun command), ~10 s apart, but each into its own
log under e0_data/alloc/ so the job ids are attributable to this experiment (farm.json is not touched)."""
import shlex, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import farm
n = int(sys.argv[1])
out = Path(__file__).resolve().parents[1] / "alloc"
for i in range(n):
    q = farm.CFG["alloc_query"]
    cmd = f"crun -q {shlex.quote(q)} --no-shell --gpus 1 -t {farm.CFG['alloc_time']}"
    lf = open(out / f"alloc_{int(time.time())}_{i}.log", "ab")
    subprocess.Popen(["ssh", "-o", "BatchMode=yes", farm.CFG["host"], cmd], stdout=lf, stderr=lf,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    print("requested", i, flush=True)
    time.sleep(10)
