#!/usr/bin/env python3
"""Run a local bash script (or python bench) on an L40S job via farm.ssh -> srun --overlap (no inline ssh)."""
import sys, shlex, base64
sys.path.insert(0, "/Users/krisd/Documents/ModelBlaster-clone/l3_farm")
import farm

def run_sh(job, body: bytes, args=(), timeout=1800):
    cmd = f"srun --jobid={job} --overlap bash -s -- " + " ".join(shlex.quote(str(a)) for a in args)
    return farm.ssh(cmd, body, timeout)

if __name__ == "__main__":
    job, script = sys.argv[1], sys.argv[2]
    args = sys.argv[3:]
    rc, out, err = run_sh(job, open(script, "rb").read(), args, timeout=3000)
    sys.stdout.write(out.decode(errors="replace")); sys.stderr.write(err.decode(errors="replace")[-3000:])
    sys.exit(rc)
