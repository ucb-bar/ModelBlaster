"""Apply memory.patch's two files to a node's code dir WITHOUT restarting anything (atomic mv; running processes keep
their old modules, every process started afterwards -- farm retries, resumed/new problems -- gets the fix).
Run from l3_farm with FARM_CONFIG set (main instance: driver job 5310063, code dir /tmp/krisd/headline_code;
heavy instance: jobs 5312377/5312379, code dir /tmp/krisd/headline_code_heavy):
  FARM_CONFIG=analysis/headline/config_headline.json python3 analysis/live_debug/apply_memory_fix.py 5310063 /tmp/krisd/headline_code
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")
import farm
job, code = sys.argv[1], sys.argv[2]
R = "ModelBlaster"
for rel in ("pipeline/kb_export.py", "gpu_eval/harness/walk_torch.py"):
    local = f"{os.path.dirname(os.path.abspath(__file__))}/patched/{rel.split('/')[-1]}"
    dst = f"{code}/{R}/{rel}"
    farm.put(job, local, dst + ".new")
    rc, o, e = farm.ssh(f"srun --jobid={job} --overlap bash -c 'python3 -c \"import ast,sys;ast.parse(open(\\\"{dst}.new\\\").read())\" && mv -f {dst}.new {dst} && echo ok {dst}'")
    print(o.decode().strip(), e.decode()[-200:])
