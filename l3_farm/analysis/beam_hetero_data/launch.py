#!/usr/bin/env python3
"""Launch one arm: python3 launch.py C|B|H [tag] [extra KB flags...]  (tag default = arm letter)."""
import sys
from pathlib import Path
import nodes
farm = nodes.farm

BT = Path(__file__).resolve().parent
L3 = Path.home() / "Documents/ModelBlaster-clone/l3_farm"
NAME = "43_MinGPTCausalAttention"
JOB = {"C": 5297008, "B": 5297013, "H": 5297016}
OPUS = "azure/anthropic/claude-opus-5-5"
CHEAP = "azure/anthropic/claude-haiku-5-5"
EXTRA = {"C": "",
         "B": "--search beam --beam-width 2 --beam-expand 2",
         "H": f"--model-simple {CHEAP} --model-hard {OPUS} --hetero-policy category,stall:3,fix-escalate"}

arm = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else arm
more = " ".join(sys.argv[3:])
import os
job = int(os.environ.get("JOB_OVERRIDE") or JOB[arm])
FD = f"/tmp/krisd/btest_{tag}"
D = f"{FD}/{NAME}"
k = farm.CFG["kb"]
env = (f"MODEL={OPUS}\nARCH={k['arch']}\nMIN_GAIN={k['min_gain_pct']}\nROUNDS={k['rounds']}\n"
       f"ROLLOUTS={k['rollouts']}\nSTEPS={k['steps']}\nVSTEPS={k['variant_steps']}\nMAXVAR={k['max_variants']}\n"
       f"TIMEOUT_H=3\nPLAN_TOPK={k['plan_top_k']}\nPLAN_MINSHARE={k['plan_min_share']}\n"
       f"CODE_ROOT=/tmp/krisd/btcode_{tag}\nKB_EXTRA='{(EXTRA[arm] + ' ' + more).strip()}'\n")
print(env)
rc, out, err = farm.ssh(f"srun --jobid={job} --overlap bash -c 'test -e {D} && echo EXISTS || echo fresh'")
assert b"fresh" in out, f"{D} exists already: use a fresh tag"
farm.put(job, L3 / "problems" / f"{NAME}.py", f"{D}/problem.py")
farm.put(job, env.encode(), f"{D}/job.env")
farm.put(job, L3 / "remote" / "worker4.sh", f"{FD}/worker4.sh")
if arm == "B":
    farm.put(job, BT / "beam_mb.tar.gz", f"{FD}/beam_mb.tar.gz")
print("detached:", nodes.detach_local(job, BT / "arm.sh", [tag, "beam" if arm == "B" else "live"]))
