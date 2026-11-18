"""Test job 1: holder with --signal=B:USR1@240, 6 min limit; attach, steps, signal, limit behaviour."""
import sys, os, time, shlex
sys.path.insert(0, os.path.expanduser("~/Documents/ModelBlaster-clone/l3_farm"))
import farm, batch_nodes as B
def p(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def sh(c, t=90):
    rc, o, e = farm.ssh(c, b"", t); return rc, o.decode(), e.decode()
t0 = time.time()
J = 5324496; t0 = time.mktime(time.strptime("2026-10-26 16:51:45","%Y-%m-%d %H:%M:%S"))
p("submitted", J)
host = B.wait_running(J, timeout=900, poll=5)
p("RUNNING on", host, "after %.0fs" % (time.time() - t0))
p(B.job_info(J))
for c in ["hostname; id -un; echo job=$SLURM_JOB_ID step=$SLURM_STEP_ID; echo CVD=$CUDA_VISIBLE_DEVICES; nvidia-smi -L",
          "cat /tmp/krisd/batchproto/%d/started; ls /tmp/krisd/batchproto/%d; ps -eo pid,ppid,etime,cmd | grep -E 'slurm_script|sleep|holder' | grep -v grep | head" % (J, J),
          "nproc; free -g | head -2"]:
    rc, o, e = B.attach_run(J, c); p("attach rc", rc, "|", o.decode().strip().replace("\n", " ; "), "|", e.decode().strip()[:200])
rc, o, e = B.attach_run(J, "tty; hostname", pty=True, timeout=60); p("pty rc", rc, o.decode().strip()[:200], e.decode().strip()[:300])
rc, o, e = sh(f"scontrol show job {J}"); print(o)
# detached long step (farm.detach pattern) writing a heartbeat
cmd = f"nohup setsid srun --jobid={J} --overlap bash -c 'while true; do date +%s > /tmp/krisd/batchproto/{J}/hb; sleep 5; done' </dev/null >/dev/null 2>&1 & sleep 5; echo started"
rc, o, e = sh(cmd); p("detached step", rc, o.strip())
rc, o, e = sh(f"squeue -s -j {J} -o '%i %j %T %M' 2>&1; sstat -j {J} -n -P -o JobID,NTasks 2>&1 | head -5"); print(o, e)
# test-only forms (nothing is submitted)
P = ",".join(B.L40S_PARTITIONS)
for dep in [f"after:{J}+5", f"afterany:{J}", f"afterok:{J}"]:
    rc, o, e = sh(f"sbatch --test-only -J mb-batchproto -p '{P}' --gpus=1 -A cuopt_fallback -t 00:05:00 --dependency={dep} --wrap='sleep 1' 2>&1 | tail -3"); p("test-only dep", dep, "->", o.strip()[:300])
rc, o, e = sh(f"sbatch --test-only -J mb-batchproto -p '{P}' --gpus=1 -A cuopt_fallback -t 00:05:00 --begin=now+1hour --wrap='sleep 1' 2>&1 | tail -3"); p("test-only begin", o.strip()[:300])
# watch for the warning and the end
seen_warn = None
while True:
    i = B.job_info(J)
    w = B.expiry_warning(J) if i["state"] == "RUNNING" else None
    if w and not seen_warn:
        seen_warn = w; p("WARNING marker seen; epoch", w, "= %.0fs after submit" % (w - t0))
    rc, hb, _ = B.attach_run(J, f"echo hb=$(cat /tmp/krisd/batchproto/{J}/hb) now=$(date +%s) warn=$(cat /tmp/krisd/batchproto/{J}/warn 2>/dev/null) term=$(cat /tmp/krisd/batchproto/{J}/term 2>/dev/null)", 40) if i["state"] == "RUNNING" else (1, b"", b"")
    p("state", i["state"], "left", i.get("left_raw"), hb.decode().strip() if isinstance(hb, bytes) else hb.strip())
    if i["state"] in B.TERMINAL or i["state"] == "UNKNOWN": break
    time.sleep(15)
time.sleep(20)
rc, o, e = sh(f"sacct -j {J} -P -o JobID,JobName,State,Elapsed,Timelimit,Start,End,ExitCode,QOS,Partition,NodeList,Submit"); print(o)
rc, o, e = sh(f"scontrol show job {J} 2>&1 | head -12"); print(o)
rc, o, e = sh(f"squeue --me -h -o '%i %j %T' | grep mb-batchproto"); print("left over:", o)
