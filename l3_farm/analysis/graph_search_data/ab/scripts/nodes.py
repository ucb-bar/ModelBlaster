#!/usr/bin/env python3
"""A4 graph-search validation: drive ONLY A4's nodes through farm.py's helpers (no dispatcher side effects).
  nodes.py probe | ready
  nodes.py stage                 tar the live ModelBlaster tree -> a4_mb.tar.gz (+ sha256, stamp)
  nodes.py install JOB           upload + unpack the staged tree to /tmp/krisd/a4/<stamp>
  nodes.py run JOB SCRIPT [args] run a local script on the node (srun --overlap), print output
  nodes.py detach JOB SCRIPT [args]
  nodes.py get JOB REMOTE LOCAL  copy a remote file back (base64 over the srun step)
"""
import base64, hashlib, json, shlex, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path.home() / "Documents/ModelBlaster-clone/l3_farm"))
import farm

HERE = Path(__file__).resolve().parent
MINE = {5299265: "n1", 5299305: "n2"}
MB = Path.home() / "Documents/ModelBlaster-clone/ModelBlaster"


def run_local(job, path, args=(), timeout=600, inp=b""):
    body = Path(path).read_bytes()
    cmd = f"srun --jobid={job} --overlap bash -s -- " + " ".join(shlex.quote(str(a)) for a in args)
    return farm.ssh(cmd, body + b"\n" + inp, timeout)


def detach_local(job, path, args=()):
    remote = f"/tmp/krisd/a4/scripts/{Path(path).name}"
    farm.put(job, path, remote)
    a = " ".join(shlex.quote(str(x)) for x in args)
    cmd = f"nohup setsid srun --jobid={job} --overlap bash {remote} {a} </dev/null >/dev/null 2>&1 & sleep 8; echo started"
    rc, out, err = farm.ssh(cmd, b"", timeout=120)
    return rc == 0 and b"started" in out


def main():
    cmd = sys.argv[1]
    if cmd in ("probe", "ready"):
        for j, n in MINE.items():
            d = farm.probe(j)
            print(n, j, json.dumps({k: d.get(k) for k in ("gpu", "setup", "relay", "tree", "foreign", "ok")}))
    elif cmd == "stage":
        tb = HERE / "a4_mb.tar.gz"
        subprocess.run(["tar", "czf", str(tb), "-C", str(MB.parent), "--exclude=.git", "--exclude=__pycache__",
                        "--exclude=*.pyc", "--exclude=ModelBlaster/gpu_eval/runs", "--exclude=ModelBlaster/gpu_eval/ablations",
                        "--exclude=ModelBlaster/experiments", "ModelBlaster"], check=True,
                       env={"COPYFILE_DISABLE": "1", "PATH": "/usr/bin:/bin"})
        sha = hashlib.sha256(tb.read_bytes()).hexdigest()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        (HERE / "stage.json").write_text(json.dumps({"stamp": stamp, "sha256": sha, "bytes": tb.stat().st_size}))
        print(stamp, sha, tb.stat().st_size // 2**20, "MB")
    elif cmd == "install":
        j = int(sys.argv[2]); assert j in MINE
        st = json.loads((HERE / "stage.json").read_text())
        rc, out, err = farm.put(j, HERE / "a4_mb.tar.gz", f"/tmp/krisd/a4/{st['stamp']}.tar.gz")
        print("put rc", rc, err.decode()[-300:])
        rc, out, err = run_local(j, HERE / "install.sh", [st["stamp"], st["sha256"]], timeout=900)
        print(out.decode()[-2000:], err.decode()[-1000:], "rc", rc)
    elif cmd == "run":
        j = int(sys.argv[2]); assert j in MINE
        rc, out, err = run_local(j, sys.argv[3], sys.argv[4:], timeout=int(__import__("os").environ.get("A4_TIMEOUT", "3600")))
        sys.stdout.write(out.decode(errors="replace")); sys.stderr.write(err.decode(errors="replace")[-3000:])
        print("rc", rc)
    elif cmd == "detach":
        j = int(sys.argv[2]); assert j in MINE
        print(detach_local(j, sys.argv[3], sys.argv[4:]))
    elif cmd == "get":
        j = int(sys.argv[2]); assert j in MINE
        rc, out, err = farm.ssh(f"srun --jobid={j} --overlap bash -c {shlex.quote('base64 -w0 ' + sys.argv[3])}", b"",
                                timeout=1800)
        Path(sys.argv[4]).write_bytes(base64.b64decode(out))
        print("rc", rc, len(out))


if __name__ == "__main__":
    main()
