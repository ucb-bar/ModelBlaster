"""Phase-1 node bring-up for the compile sweep (CPU/network only). Usage: FARM_CONFIG=analysis/l12_sweep/config_l12.json python3 node_setup.py status|start JOB...|check JOB...
Uses farm.ssh/farm.detach only; never touches the driver job 5422711."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import farm
DRIVER = "5422711"
def run(job, body, t=90):
    rc, out, err = farm.ssh(f"srun --jobid={job} --overlap bash -c {__import__('shlex').quote(body)}", b"", timeout=t)
    return rc, out.decode(errors="replace").strip(), err.decode(errors="replace").strip()[-200:]
STAT = "hostname; cat /tmp/krisd/farm/setup.status 2>/dev/null || echo nostatus; test -x /tmp/krisd/venv/bin/python && echo venv=yes || echo venv=no; pgrep -f '[s]etup.sh' >/dev/null && echo setup_running || echo setup_idle"
CHK = ". /tmp/krisd/venv/bin/activate && CUDA_VISIBLE_DEVICES= python -c 'import torch,sys;print(\"torch\",torch.__version__,torch.version.cuda)'"
cmd, jobs = sys.argv[1], sys.argv[2:]
if cmd == "status" or not jobs:
    jobs = jobs or [j for j in json.load(open("state/l12/farm.json"))["nodes"] if j != DRIVER]
for j in jobs:
    if j == DRIVER: print(j, "SKIP driver"); continue
    try:
        if cmd == "status": print(j, run(j, STAT))
        elif cmd == "start": print(j, "detach ->", farm.detach(int(j), "setup.sh"))
        elif cmd == "check": print(j, run(j, CHK, 180))
    except Exception as e: print(j, "ERR", repr(e)[:150])
