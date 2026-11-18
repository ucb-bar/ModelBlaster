#!/usr/bin/env python3
"""Beam/hetero test helper: drive ONLY my three nodes through farm.py's helpers (no dispatcher side effects).
  nodes.py probe            probe my nodes
  nodes.py bringup          one bring-up step per node (tree / setup / relay), like farm.bring_up
  nodes.py run JOB SCRIPT [args...]   run a local script on the node (srun --overlap), print stdout/stderr
  nodes.py put JOB LOCAL REMOTE
  nodes.py detach JOB SCRIPT [args...]  long-lived step running a local script (copied to the node first)
"""
import sys, json, shlex
from pathlib import Path
sys.path.insert(0, str(Path.home() / "Documents/ModelBlaster-clone/l3_farm"))
import farm

MINE = {5297008: "C", 5297013: "B", 5297016: "H"}


def run_local(job, path, args=(), timeout=600, inp=b""):
    body = Path(path).read_bytes()
    cmd = f"srun --jobid={job} --overlap bash -s -- " + " ".join(shlex.quote(str(a)) for a in args)
    return farm.ssh(cmd, body + b"\n" + inp, timeout)


def detach_local(job, path, args=()):
    remote = f"/tmp/krisd/bt_scripts/{Path(path).name}"
    farm.put(job, path, remote)
    a = " ".join(shlex.quote(str(x)) for x in args)
    cmd = f"nohup setsid srun --jobid={job} --overlap bash {remote} {a} </dev/null >/dev/null 2>&1 & sleep 8; echo started"
    rc, out, err = farm.ssh(cmd, b"", timeout=120)
    return rc == 0 and b"started" in out


def main():
    cmd = sys.argv[1]
    if cmd == "probe":
        for j, arm in MINE.items():
            d = farm.probe(j)
            print(arm, j, json.dumps(d))
    elif cmd == "bringup":
        s = farm.load()
        for j, arm in MINE.items():
            nd = farm.probe(j)
            if not nd["ok"]:
                print(arm, j, "probe failed", nd); continue
            nd["jobs"]["__running__"] = False
            if farm.node_ready(nd, 600):
                print(arm, j, "READY", nd); continue
            print(arm, j, "step:", farm.bring_up(j, nd, s), nd)
    elif cmd == "run":
        j = int(sys.argv[2])
        assert j in MINE, j
        rc, out, err = run_local(j, sys.argv[3], sys.argv[4:], timeout=3600)
        sys.stdout.write(out.decode(errors="replace")); sys.stderr.write(err.decode(errors="replace")[-3000:])
        print("rc", rc)
    elif cmd == "put":
        j = int(sys.argv[2]); assert j in MINE
        rc, out, err = farm.put(j, sys.argv[3], sys.argv[4])
        print("rc", rc, err.decode()[-500:])
    elif cmd == "detach":
        j = int(sys.argv[2]); assert j in MINE
        print(detach_local(j, sys.argv[3], sys.argv[4:]))


if __name__ == "__main__":
    main()
