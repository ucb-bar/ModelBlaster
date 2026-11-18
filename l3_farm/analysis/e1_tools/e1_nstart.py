"""start a nearest-K worker: [E1_NK=nk2,nk4] e1_nstart.py JOB P [P..]  (runs_nearest/, flags --nearest --nk ..)"""
import os, sys, shlex, time
sys.path.insert(0, '/Users/krisd/Documents/ModelBlaster-clone/l3_farm/analysis/e1_tools')
import e1_run, nodes
R = e1_run.R
job, probs = sys.argv[1], sys.argv[2:]
nk = os.environ.get('E1_NK', 'nk2,nk4')
log = f"{R}/logs/nworker_{int(time.time())}.log"
body = e1_run.ENV + f"export E1_RUNS=runs_nearest E1_FLAGS='--nearest --nk {nk}'\n" + \
    f"flock {R}/gpu.lock bash {R}/bin/worker.sh " + " ".join(shlex.quote(p) for p in probs) + f" > {log} 2>&1\n"
cmd = (f"nohup setsid srun --jobid={job} --overlap bash -c {shlex.quote(f'mkdir -p {R}/logs; cd {R}' + chr(10) + body)} "
       f"</dev/null >/dev/null 2>&1 & sleep 3; echo started {log}")
assert str(job) in nodes.mine()
rc, out, err = e1_run.farm.ssh(cmd, b"", timeout=120)
print(out.decode(), err.decode()[-300:], rc)
