#!/usr/bin/env python3
"""cost.py RESULTS_DIR... : LLM calls and $ (assumed Opus prices, sched_report.usd) of the collected work dirs.
Calls/spend of an attempt that died with its node before a pack are not in the telemetry we hold (undercount)."""
import sys
sys.path.insert(0, "/Users/krisd/Documents/ModelBlaster-clone/l3_farm")
import sched_report as R
from pathlib import Path
tot = 0
for d in sys.argv[1:]:
    for w in sorted(Path(d).glob("*/work")):
        t = R.llm(w); u = R.usd(t); tot += u
        print(f"{w.parent.name}: calls {int(t.get('calls') or 0)} usd {u:.2f}")
print(f"total usd {tot:.2f}")
