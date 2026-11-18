#!/usr/bin/env python3
"""Print one line per E0 run status change (+ rounds); exit when every run is terminal."""
import re, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
last = {}
while True:
    out = subprocess.run([sys.executable, str(HERE / "e0.py"), "status"], capture_output=True, text=True).stdout
    cur = {}
    for m in re.finditer(r"^run (\S+) (\S+) rounds=(\d+)", out, re.M):
        cur[m.group(1)] = (m.group(2), m.group(3))
    for k, v in cur.items():
        if last.get(k) != v:
            print(time.strftime("%H:%M"), k, *v, flush=True)
    last.update(cur)
    if cur and all(not v[0].startswith(("running", "remeasuring")) or v[0].endswith("died") for v in cur.values()):
        print("ALL_TERMINAL", flush=True)
        break
    time.sleep(300)
