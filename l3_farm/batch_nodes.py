"""Prototype: acquire and operate L40S nodes through sbatch "holder" jobs instead of `crun --no-shell`.

No side effects on import (farm.py is imported lazily and only for its ssh helper). Nothing here is wired into
farm.py. See analysis/batch_interface/REPORT.md for the measurements and the adaptation sketch.

  submit_holder(...) -> job id     wait_running(job) -> host      attach_run(job, cmd) -> (rc, out, err)
  job_info(job) -> dict            cancel(job)                    holder_script() -> str (remote/holder.sh)
Expiry protocol (see remote/holder.sh): sbatch --signal=B:USR1@<warn_s> -> the holder's trap writes
$MB_WORKDIR/<job>/warn (and runs $MB_HOOK); the farm's probe reads that marker via expiry_warning(job) and starts
collecting the packs of every problem on that node, stops leasing to it and requests a successor holder.
"""
import ast, re, shlex, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "mb-batchproto"
L40S_QUERY = 'gpu.product_name="*L40S*"'            # same selection as config.json alloc_query
# 1-GPU L40S partitions that crun's query resolves to (cdb find ... -c slurm.partitions); sbatch has no query DSL
L40S_PARTITIONS = ["l40s@cr+mp/romed8nl/1gpu-32cpu-128gb", "l40s@qs1/romed8nl/1gpu-32cpu-128gb"]
TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}


def _ssh(cmd, inp=b"", timeout=120):
    import farm                                      # lazy: farm.load()-free, but reads FARM_CONFIG at import
    return farm.ssh(cmd, inp, timeout)


def holder_script() -> str:
    return (ROOT / "remote" / "holder.sh").read_text()


def resolve_partitions(query=L40S_QUERY, ssh=_ssh):
    """Partitions of the nodes matching a crun/cdb query (read-only `cdb find`), 1-GPU ones first; falls back to the constant."""
    rc, out, _ = ssh(f"cdb find {shlex.quote(query)} -c slurm.partitions 2>/dev/null | tail -n +3", b"", 90)
    parts = []
    for ln in out.decode().splitlines():
        m = re.search(r"\[.*\]", ln)
        if m:
            parts += [p for p in ast.literal_eval(m.group(0)) if p not in parts]
    parts = [p for p in parts if "/1gpu-" in p]
    return parts or list(L40S_PARTITIONS)


def hms_to_min(s):
    m = re.match(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)$", s.strip())
    if not m:
        return 0
    d, h, mi, _ = (int(x or 0) for x in m.groups())
    return d * 1440 + h * 60 + mi


def submit_holder(time="10:00:00", gpus=1, partitions=None, name=NAME, account=None, qos=None, warn_s=600,
                  requeue=False, workdir="/tmp/krisd/batchproto", hook=None, life_s=None, dependency=None, begin=None,
                  extra="", ssh=_ssh):
    """sbatch the holder script (remote/holder.sh via stdin). Returns the job id. `time` is validated by Slurm/cli_filter:
    the batch QOS caps at 1-00:00:00 (24 h), a longer request is rejected at submit. dependency e.g. 'after:123+480'
    (start 480 min after job 123 STARTED) or 'afterany:123'; begin e.g. 'now+2hours'."""
    parts = partitions or L40S_PARTITIONS
    a = ["sbatch", "--parsable", "-J", name, "-p", ",".join(parts), f"--gpus={gpus}", "-t", time, "-o", "/dev/null", "-e", "/dev/null",
         "--requeue" if requeue else "--no-requeue"]
    if warn_s:
        a.append(f"--signal=B:USR1@{int(warn_s)}")
    if account:
        a += ["-A", account]
    if qos:
        a += ["--qos", qos]
    if dependency:
        a.append(f"--dependency={dependency}")
    if begin:
        a.append(f"--begin={begin}")
    env = f"ALL,MB_WORKDIR={workdir}" + (f",MB_HOOK={hook}" if hook else "") + (f",MB_LIFE_S={int(life_s)}" if life_s else "")
    a.append(f"--export={env}")
    cmd = " ".join(shlex.quote(x) for x in a) + (" " + extra if extra else "")
    rc, out, err = ssh(cmd, holder_script().encode(), 120)
    m = re.match(r"(\d+)", out.decode().strip())
    if rc or not m:
        raise RuntimeError(f"sbatch failed rc={rc}: {(err or out).decode()[:400]}")
    return int(m.group(1))


def job_info(job, ssh=_ssh):
    """state, time left (min and raw), QOS, partition, host, end time; falls back to sacct once the job left squeue."""
    rc, out, _ = ssh(f"squeue -j {int(job)} -h -o '%T|%N|%L|%P|%q|%a|%j|%r|%S|%e' 2>/dev/null", b"", 60)
    p = out.decode().strip().split("|")
    if rc == 0 and len(p) >= 10:
        return dict(job=job, state=p[0], host=p[1], left_raw=p[2], left=hms_to_min(p[2]), partition=p[3], qos=p[4], account=p[5],
                    name=p[6], reason=p[7], start=p[8], end=p[9])
    rc, out, _ = ssh(f"sacct -j {int(job)} -n -P -X -o State,NodeList,Elapsed,Timelimit,QOS,Partition,Account,JobName,End", b"", 60)
    q = out.decode().strip().split("|")
    if len(q) >= 9:
        return dict(job=job, state=q[0].split()[0], host=q[1], left_raw="0:00", left=0, elapsed=q[2], timelimit=q[3], qos=q[4],
                    partition=q[5], account=q[6], name=q[7], end=q[8], reason="")
    return dict(job=job, state="UNKNOWN", host="", left=0)


def wait_running(job, timeout=1800, poll=10, ssh=_ssh):
    """Block until RUNNING; return the host name. Raises if the job ended first."""
    t0 = time_now()
    while time_now() - t0 < timeout:
        i = job_info(job, ssh)
        if i["state"] == "RUNNING" and i["host"]:
            return i["host"]
        if i["state"] in TERMINAL:
            raise RuntimeError(f"job {job} ended before running: {i}")
        sleep(poll)
    raise TimeoutError(f"job {job} not running after {timeout}s")


def attach_run(job, cmd, timeout=300, pty=False, inp=b"", ssh=_ssh):
    """One srun step bound to the holder job's node (same as farm.srun/put/detach use): srun --jobid=J --overlap bash -c CMD."""
    p = "--pty " if pty else ""
    return ssh(f"srun --jobid={int(job)} --overlap {p}bash -c {shlex.quote(cmd)}", inp, timeout)


def expiry_warning(job, workdir="/tmp/krisd/batchproto", ssh=_ssh):
    """Epoch of the USR1 warning marker on the node, or None (what probe2.sh would report as `expiry_warn=`)."""
    rc, out, _ = attach_run(job, f"cat {workdir}/{int(job)}/warn 2>/dev/null", 60, ssh=ssh)
    t = out.decode().strip()
    return int(t) if rc == 0 and t.isdigit() else None


def cancel(job, name_prefix="mb-", ssh=_ssh):
    """scancel one holder job; refuses jobs whose name does not start with name_prefix (safety against farm allocations)."""
    i = job_info(job, ssh)
    if not str(i.get("name", "")).startswith(name_prefix):
        raise PermissionError(f"refusing to cancel job {job}: name {i.get('name')!r} does not start with {name_prefix!r}")
    rc, out, err = ssh(f"scancel {int(job)}", b"", 60)
    return rc == 0


def time_now():
    import time as _t
    return _t.time()


def sleep(s):
    import time as _t
    _t.sleep(s)
