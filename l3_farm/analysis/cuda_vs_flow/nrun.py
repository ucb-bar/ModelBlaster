#!/usr/bin/env python3
"""Run a bash script on MY profiling node only (job 5297077) via farm.ssh -> srun --overlap.

  nrun.py SCRIPT.sh [timeout_s] [--put LOCAL=REMOTE ...]

Files given with --put are shipped first (base64 heredoc inside the same script). Remote work dir: /tmp/krisd/cvf.
"""
import base64, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import farm  # noqa: E402  (ssh helper only)

JOB = 5297077
args = sys.argv[1:]
script = Path(args.pop(0)).read_text()
timeout = int(args.pop(0)) if args and args[0].isdigit() else 900
puts = []
while args:
    a = args.pop(0)
    if a == "--put":
        loc, rem = args.pop(0).split("=", 1)
        puts.append((loc, rem))
pre = "set -e; mkdir -p /tmp/krisd/cvf\n"
for loc, rem in puts:
    pre += f"mkdir -p $(dirname {rem}); base64 -d > {rem} <<'__F__'\n" + base64.encodebytes(Path(loc).read_bytes()).decode() + "__F__\n"
pre += "set +e\n"
rc, out, err = farm.ssh(f"srun --jobid={JOB} --overlap bash -s", (pre + script + "\n").encode(), timeout=timeout)
sys.stdout.write(out.decode(errors="replace"))
sys.stdout.write(err.decode(errors="replace")[-3000:])
print("rc", rc)
