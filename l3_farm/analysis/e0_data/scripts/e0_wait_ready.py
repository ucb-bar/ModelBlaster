#!/usr/bin/env python3
"""Poll farm.probe on the E0 jobs until every node has setup=done and relay=ok (or timeout)."""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import farm
jobs = [int(x) for x in sys.argv[1].split(",")]
t_end = time.time() + float(sys.argv[2] if len(sys.argv) > 2 else 3600)
while True:
    with ThreadPoolExecutor(8) as ex:
        pr = dict(zip(jobs, ex.map(farm.probe, jobs)))
    ready = [j for j, d in pr.items() if d.get("setup") == "done" and d.get("relay") == "ok"]
    print(time.strftime("%H:%M:%S"), {j: (d.get("setup"), d.get("relay"), d.get("tree")) for j, d in pr.items()}, flush=True)
    if len(ready) == len(jobs) or time.time() > t_end:
        print("READY" if len(ready) == len(jobs) else "TIMEOUT", ready, flush=True)
        break
    time.sleep(60)
