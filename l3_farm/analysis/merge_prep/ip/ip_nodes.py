#!/usr/bin/env python3
"""Integration-pass node helper: reuses merge_checks/mc_nodes.py with MY OWN job list (merge_prep/my_jobs.json) and alloc dir.
Usage: ip_nodes.py alloc N | mine | wait [S] | gpu | release.  ssh only via farm.ssh (computelab alias)."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
MC = HERE.parents[1] / "merge_checks"
sys.path.insert(0, str(MC))
import mc_common as C
import mc_nodes as N
C.MINE = HERE.parent / "my_jobs.json"
N.ALLOC = HERE / "alloc"
if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "mine"
    if c == "alloc": N.alloc(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    elif c == "mine": N.mine()
    elif c == "wait": N.wait(float(sys.argv[2]) if len(sys.argv) > 2 else 900)
    elif c == "gpu": N.gpu([int(x) for x in sys.argv[3:]] if False else [int(x) for x in sys.argv[2:]])
    elif c == "release": N.release([int(x) for x in sys.argv[2:]])
