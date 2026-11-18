"""Test job 2: script exits by itself after 75 s (life_s); manual USR1 via scancel -b; hook runs; fate of steps when the script exits."""
import sys, os, time
sys.path.insert(0, os.path.expanduser("~/Documents/ModelBlaster-clone/l3_farm"))
import farm, batch_nodes as B
def p(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def sh(c, t=90):
    rc, o, e = farm.ssh(c, b"", t); return rc, o.decode(), e.decode()
HOOK = "/tmp/krisd/batchproto/hook.sh"
t0 = time.time()
J = B.submit_holder(time="00:05:00", warn_s=120, account="cuopt_fallback", life_s=100, hook=HOOK)
p("submitted", J)
host = B.wait_running(J, timeout=900, poll=3); p("RUNNING", host, "after %.0fs" % (time.time() - t0))
rc, o, e = B.attach_run(J, f"mkdir -p /tmp/krisd/batchproto && printf '#!/bin/bash\\necho hook-ran $(date +%s) args=$*\\n' > {HOOK} && chmod +x {HOOK} && echo ok"); p("hook installed", o.decode().strip())
cmd = f"nohup setsid srun --jobid={J} --overlap bash -c 'while true; do date +%s > /tmp/krisd/batchproto/{J}/hb; sleep 3; done' </dev/null >/dev/null 2>&1 & sleep 4; echo started"
p("detached", sh(cmd)[1].strip())
rc, o, e = sh(f"scancel --batch --signal=USR1 {J}; echo scancel-rc=$?"); p("manual USR1:", o.strip(), e.strip())
time.sleep(5)
rc, o, e = B.attach_run(J, f"cd /tmp/krisd/batchproto/{J}; ls; echo warn=$(cat warn); echo hook.rc=$(cat hook.rc); cat hook.log"); p("markers:", o.decode().replace("\n", " ; "))
# foreground step that outlives the script (life 100 s): we run sleep 300 as a step
t1 = time.time()
rc, o, e = B.attach_run(J, "echo step-start; sleep 300; echo step-finished", timeout=200); p("foreground step ended after %.0fs rc=%s out=%r err=%r" % (time.time() - t1, rc, o.decode()[:100], e.decode()[:300]))
for k in range(8):
    i = B.job_info(J); p("info", i["state"], i.get("left_raw"), i.get("elapsed", ""))
    if i["state"] in B.TERMINAL: break
    time.sleep(5)
rc, o, e = sh(f"sacct -j {J} -P -o JobID,JobName,State,Elapsed,Timelimit,Start,End,ExitCode,QOS,NodeList"); print(o)
rc, o, e = B.attach_run(J, "hostname", timeout=60); p("attach after end rc", rc, o.decode().strip(), e.decode().strip()[:200])
rc, o, e = sh(f"squeue --me -h -o '%i %j %T' | grep mb-batchproto"); print("left over:", o)
