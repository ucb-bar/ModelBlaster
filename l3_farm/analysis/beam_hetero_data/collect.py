#!/usr/bin/env python3
"""Collect an arm: python3 collect.py TAG  -> res/TAG/ (pack2.sh stream of the run dir + the ledger rows of the arm)."""
import json, subprocess, sys
from pathlib import Path
import nodes
farm = nodes.farm

BT = Path(__file__).resolve().parent
NAME = "43_MinGPTCausalAttention"
JOBS = {"C": 5297008, "B": 5297013, "H": 5297016}
tag = sys.argv[1]
job = int(sys.argv[2]) if len(sys.argv) > 2 else JOBS[tag[0]]
out = BT / "res" / tag
out.mkdir(parents=True, exist_ok=True)
rc, data, err = farm.srun(job, "pack2.sh", [NAME, f"/tmp/krisd/btest_{tag}", f"/tmp/krisd/bt_pack_{tag}"], timeout=1800)
(out / "pack.tar.gz").write_bytes(data)
subprocess.run(["tar", "xzf", str(out / "pack.tar.gz"), "-C", str(out)], check=False)
# the ledger slice (no secrets in the ledger: model, tokens, timings) + start/end markers
rc, data, err = farm.ssh(f"srun --jobid={job} --overlap bash -c 'D=/tmp/krisd/btest_{tag}/{NAME}; "
                         f"cat $D/ledger_start $D/ledger_end $D/t_start $D/t_end 2>/dev/null | tr \"\\n\" \" \"; echo; "
                         f"tail -n +$(( $(cut -d\" \" -f1 $D/ledger_start) + 1 )) /tmp/krisd/farm/llm_ledger.jsonl'", timeout=600)
lines = data.decode().splitlines()
(out / "markers.txt").write_text(lines[0] + "\n")
(out / "ledger.jsonl").write_text("\n".join(lines[1:]) + "\n")
print(tag, "collected", len(lines) - 1, "ledger rows;", lines[0])
