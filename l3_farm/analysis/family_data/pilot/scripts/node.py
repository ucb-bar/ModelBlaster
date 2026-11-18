#!/usr/bin/env python3
"""Pilot helper for ONE dedicated node (job id in JOB): run a local bash script on it, copy files, detach.
Same scheme as l3_farm/farm.py (ssh computelab -> srun --jobid --overlap), never prints secrets.

  node.py run SCRIPT [args...]        bash SCRIPT on the node (stdin), prints stdout/stderr
  node.py put LOCAL REMOTE            copy a file
  node.py get REMOTE LOCAL            copy a file back
  node.py detach SCRIPT [args...]     long-lived step (nohup setsid srun ... bash REMOTE_COPY)
  node.py probe                       l3_farm farm.probe(JOB)
"""
import shlex, subprocess, sys
from pathlib import Path

JOB = 5295568
HOST = "computelab"
W = "/tmp/krisd/family"


def ssh(cmd, inp=b"", timeout=3600):
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", HOST, cmd], input=inp,
                       capture_output=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def run(script, args=(), timeout=3600):
    body = Path(script).read_bytes()
    cmd = f"srun --jobid={JOB} --overlap bash -s -- " + " ".join(shlex.quote(str(a)) for a in args)
    return ssh(cmd, body, timeout)


def put(local, remote):
    cmd = f"srun --jobid={JOB} --overlap bash -c {shlex.quote(f'mkdir -p $(dirname {remote}) && cat > {remote}')}"
    return ssh(cmd, Path(local).read_bytes(), timeout=1800)


def get(remote, local):
    rc, out, err = ssh(f"srun --jobid={JOB} --overlap cat {shlex.quote(remote)}", b"", timeout=1800)
    if rc == 0:
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        Path(local).write_bytes(out)
    return rc, b"", err


def detach(script, args=()):
    remote = f"{W}/bin/{Path(script).name}"
    put(script, remote)
    a = " ".join(shlex.quote(str(x)) for x in args)
    cmd = f"nohup setsid srun --jobid={JOB} --overlap bash {remote} {a} </dev/null >/dev/null 2>&1 & sleep 8; echo started"
    return ssh(cmd, b"", timeout=120)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "probe":
        sys.path.insert(0, str(Path.home() / "Documents/ModelBlaster-clone/l3_farm"))
        import farm
        print(farm.probe(JOB))
        sys.exit(0)
    rc, out, err = {"run": lambda: run(sys.argv[2], sys.argv[3:]), "put": lambda: put(sys.argv[2], sys.argv[3]),
                    "get": lambda: get(sys.argv[2], sys.argv[3]), "detach": lambda: detach(sys.argv[2], sys.argv[3:])}[c]()
    sys.stdout.write(out.decode(errors="replace"))
    sys.stderr.write(err.decode(errors="replace")[-3000:])
    sys.exit(rc)
